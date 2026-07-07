#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import inspect
import re
import sys

import numpy as np
import torch
import torch.nn.functional as F

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer


NUMERIC_TOKEN_RE = re.compile(
    r"###([+-]?(?:\d+(?:\.\d*)?|\.\d+)|Nan|NaN|nan)###"
)


def is_bfloat16_supported():
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


def collect_numeric_token_ids_and_values(tokenizer):
    rows = []

    vocab = tokenizer.get_vocab()

    for token, token_id in vocab.items():
        m = NUMERIC_TOKEN_RE.fullmatch(str(token))
        if m is None:
            continue

        raw = m.group(1)

        if raw.lower() == "nan":
            continue

        value = float(raw)

        rows.append(
            {
                "token": token,
                "token_id": token_id,
                "value": value,
            }
        )

    rows = sorted(rows, key=lambda x: x["value"])

    if len(rows) == 0:
        raise ValueError(
            "No numeric tokens found. "
            "Call collect_numeric_token_ids_and_values() after tokenizer.add_tokens()."
        )

    numeric_token_ids = torch.tensor(
        [r["token_id"] for r in rows],
        dtype=torch.long,
    )

    numeric_token_values = torch.tensor(
        [r["value"] for r in rows],
        dtype=torch.float32,
    )

    print("numeric token count:", len(rows))
    print(
        "value range:",
        numeric_token_values.min().item(),
        numeric_token_values.max().item(),
    )

    return numeric_token_ids, numeric_token_values


class NumericValueNormRegularizedSFTTrainer(SFTTrainer):
    """
    Mamba 用の value-norm 正則化 Trainer。

    数値トークンの |value| を embedding norm に対応させる。

    目的:
        |value| が小さい token -> embedding norm 小さめ
        |value| が大きい token -> embedding norm 大きめ

    total_loss:
        total_loss = lm_loss + value_norm_reg_weight * value_norm_loss

    value_norm_loss:
        norm_ratio_i = ||e_i|| / mean(||e_numeric||)

        target_ratio_i =
            value_norm_min_ratio
            + (value_norm_max_ratio - value_norm_min_ratio) * |value_i|

        value_norm_loss = mean((norm_ratio_i - target_ratio_i)^2)
    """

    def __init__(
        self,
        *args,
        numeric_token_ids=None,
        numeric_token_values=None,
        value_norm_reg_weight=0.01,
        value_norm_min_ratio=0.8,
        value_norm_max_ratio=1.2,
        regularize_lm_head=False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if numeric_token_ids is None or numeric_token_values is None:
            raise ValueError("numeric_token_ids and numeric_token_values are required.")

        self.numeric_token_ids_cpu = numeric_token_ids.detach().cpu().long()
        self.numeric_token_values_cpu = numeric_token_values.detach().cpu().float()

        self.value_norm_reg_weight = value_norm_reg_weight
        self.value_norm_min_ratio = value_norm_min_ratio
        self.value_norm_max_ratio = value_norm_max_ratio
        self.regularize_lm_head = regularize_lm_head

    def _value_norm_regularization_matrix(self, weight_matrix):
        device = weight_matrix.device

        numeric_ids = self.numeric_token_ids_cpu.to(device)
        values = self.numeric_token_values_cpu.to(device)

        emb = weight_matrix[numeric_ids].float()
        norms = emb.norm(dim=-1)

        center_norm = norms.detach().mean().clamp(min=1e-6)

        abs_values = values.abs().clamp(min=0.0, max=1.0)

        target_ratio = (
            self.value_norm_min_ratio
            + (self.value_norm_max_ratio - self.value_norm_min_ratio) * abs_values
        )

        norm_ratio = norms / center_norm

        value_norm_loss = F.mse_loss(norm_ratio, target_ratio)

        return value_norm_loss

    def numeric_value_norm_regularization_loss(self, model):
        input_emb = model.get_input_embeddings().weight
        loss = self._value_norm_regularization_matrix(input_emb)

        if self.regularize_lm_head:
            output_emb = model.get_output_embeddings()
            if output_emb is not None and hasattr(output_emb, "weight"):
                loss = loss + self._value_norm_regularization_matrix(output_emb.weight)

        return loss

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)

        if isinstance(outputs, dict):
            lm_loss = outputs["loss"]
        else:
            lm_loss = outputs[0]

        if self.value_norm_reg_weight > 0:
            value_norm_loss = self.numeric_value_norm_regularization_loss(model)
            weighted_value_norm_loss = self.value_norm_reg_weight * value_norm_loss
            loss = lm_loss + weighted_value_norm_loss
        else:
            value_norm_loss = torch.tensor(0.0, device=lm_loss.device)
            weighted_value_norm_loss = torch.tensor(0.0, device=lm_loss.device)
            loss = lm_loss

        if self.state.global_step % max(1, self.args.logging_steps) == 0:
            self.log(
                {
                    "lm_loss": lm_loss.detach().float().item(),
                    "value_norm_loss": value_norm_loss.detach().float().item(),
                    "weighted_value_norm_loss": weighted_value_norm_loss.detach().float().item(),
                    "total_loss_with_value_norm_reg": loss.detach().float().item(),
                }
            )

        return (loss, outputs) if return_outputs else loss


@torch.no_grad()
def rescale_numeric_token_embedding_norms(
    model,
    numeric_token_ids,
    numeric_token_values,
    min_ratio=0.8,
    max_ratio=1.2,
    regularize_lm_head=False,
):
    """
    学習開始前に、数値 token の embedding norm を |value| に応じて初期調整する。

    |value| = 0.0 -> min_ratio * center_norm
    |value| = 1.0 -> max_ratio * center_norm
    """

    input_emb = model.get_input_embeddings()
    input_w = input_emb.weight

    device = input_w.device
    ids = numeric_token_ids.to(device)
    values = numeric_token_values.to(device)

    def rescale_matrix(weight):
        emb = weight[ids].float()
        norms = emb.norm(dim=-1).clamp(min=1e-6)

        center_norm = norms.mean().clamp(min=1e-6)

        abs_values = values.abs().clamp(min=0.0, max=1.0)

        target_ratio = min_ratio + (max_ratio - min_ratio) * abs_values
        target_norms = center_norm * target_ratio

        scale = (target_norms / norms).to(dtype=weight.dtype)
        weight[ids] *= scale.unsqueeze(-1)

    rescale_matrix(input_w)

    if regularize_lm_head:
        output_emb = model.get_output_embeddings()
        if output_emb is not None and hasattr(output_emb, "weight"):
            rescale_matrix(output_emb.weight)

    print("Rescaled numeric token embedding norms by |value|.")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--code_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--log_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)

    parser.add_argument("--max_seq_length", type=int, default=2048)

    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--random_seed", type=int, default=3407)

    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=64)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=500)

    parser.add_argument("--learning_rate", type=float, default=2e-4)

    parser.add_argument("--low_limit", type=float, default=-1)
    parser.add_argument("--high_limit", type=float, default=1)
    parser.add_argument("--n_tokens", type=int, default=10002)
    parser.add_argument("--prec", type=int, default=4)
    parser.add_argument("--time_sep", type=str, default=" ")
    parser.add_argument("--time_flag", type=str, default="###")
    parser.add_argument("--nan_flag", type=str, default="Nan")

    parser.add_argument("--value_norm_reg_weight", type=float, default=0.01)
    parser.add_argument("--value_norm_min_ratio", type=float, default=0.8)
    parser.add_argument("--value_norm_max_ratio", type=float, default=1.2)
    parser.add_argument("--regularize_lm_head", action="store_true", default=False)

    parser.add_argument("--disable_initial_norm_rescale", action="store_true", default=False)

    parser.add_argument("--wandb_run_name", type=str, default=None)

    args = parser.parse_args()

    sys.path.append(args.code_path)
    from utils.tools import Discretizer, Serializer

    if args.wandb_run_name is None:
        args.wandb_run_name = (
            f"mamba-value-norm-{args.model_path.split('/')[-1]}"
            f"-bs{args.per_device_train_batch_size}"
            f"-ga{args.gradient_accumulation_steps}"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    discretizer = Discretizer(
        low_limit=args.low_limit,
        high_limit=args.high_limit,
        n_tokens=args.n_tokens,
    )

    serializer = Serializer(
        prec=args.prec,
        time_sep=args.time_sep,
        time_flag=args.time_flag,
        nan_flag=args.nan_flag,
    )

    vocabulary = np.concatenate(
        (discretizer.centers[1:-1], [np.nan])
    ).reshape(-1, 1)

    vocabulary = np.array(
        [serializer.serialize(i) for i in vocabulary]
    )

    print(f"\nVocabulary sample:\n{vocabulary[:10]}\n...\n{vocabulary[-10:]}\n")

    old_vocab_size = len(tokenizer)
    print("Old tokenizer size:", old_vocab_size)

    num_added_tokens = tokenizer.add_tokens(vocabulary.tolist())

    print("Added tokens:", num_added_tokens)
    print("New tokenizer size:", len(tokenizer))

    numeric_token_ids, numeric_token_values = collect_numeric_token_ids_and_values(
        tokenizer
    )

    EOS_TOKEN = tokenizer.eos_token

    print("\nLoading model:", args.model_path)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if is_bfloat16_supported() else torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    model.resize_token_embeddings(len(tokenizer))
    model.config.use_cache = False

    if not args.disable_initial_norm_rescale:
        rescale_numeric_token_embedding_norms(
            model=model,
            numeric_token_ids=numeric_token_ids,
            numeric_token_values=numeric_token_values,
            min_ratio=args.value_norm_min_ratio,
            max_ratio=args.value_norm_max_ratio,
            regularize_lm_head=args.regularize_lm_head,
        )

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "x_proj",
            "in_proj",
            "dt_proj",
        ],
        modules_to_save=[
            "backbone.embeddings",
            "lm_head",
        ],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("Input embedding requires_grad:")
    print(model.get_input_embeddings().weight.requires_grad)

    output_emb = model.get_output_embeddings()
    if output_emb is not None and hasattr(output_emb, "weight"):
        print("Output embedding requires_grad:")
        print(output_emb.weight.requires_grad)

    print(f"\nLoading dataset from {args.dataset_path}")

    dataset = load_dataset(
        "csv",
        data_files=(
            f"https://huggingface.co/datasets/"
            f"{args.dataset_path}/resolve/main/ChatTime-1-Pretrain-1M.csv"
        ),
        split="train",
    )

    print(f"Dataset example:\n{dataset[0]['text']}\n")

    def add_eos(example):
        text = example["text"]
        if EOS_TOKEN is not None and not text.endswith(EOS_TOKEN):
            text = text + EOS_TOKEN
        return {"text": text}

    dataset = dataset.map(add_eos, num_proc=8)

    training_args = TrainingArguments(
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        weight_decay=0.01,
        warmup_ratio=0.05,
        max_grad_norm=1.0,
        learning_rate=args.learning_rate,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        max_steps=args.max_steps,
        save_total_limit=1,
        logging_first_step=True,
        optim="adamw_8bit",
        lr_scheduler_type="cosine",
        seed=args.random_seed,
        output_dir=args.log_path,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        report_to="wandb",
        run_name=args.wandb_run_name,
    )

    trainer_kwargs = dict(
        model=model,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        packing=False,
        numeric_token_ids=numeric_token_ids,
        numeric_token_values=numeric_token_values,
        value_norm_reg_weight=args.value_norm_reg_weight,
        value_norm_min_ratio=args.value_norm_min_ratio,
        value_norm_max_ratio=args.value_norm_max_ratio,
        regularize_lm_head=args.regularize_lm_head,
        args=training_args,
    )

    trainer_init_params = inspect.signature(SFTTrainer.__init__).parameters

    if "processing_class" in trainer_init_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_init_params:
        trainer_kwargs["tokenizer"] = tokenizer
    else:
        print("[Warning] SFTTrainer has neither processing_class nor tokenizer argument.")

    trainer = NumericValueNormRegularizedSFTTrainer(**trainer_kwargs)

    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)

    print(f"\nGPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.\n")

    trainer_stats = trainer.train()

    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_training = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    training_percentage = round(used_memory_for_training / max_memory * 100, 3)

    print(f"\n{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(f"{round(trainer_stats.metrics['train_runtime'] / 60, 2)} minutes used for training.")
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_training} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(f"Peak reserved memory for training % of max memory = {training_percentage} %.\n")

    print(f"Saving LoRA adapter to {args.output_path}")

    model.save_pretrained(args.output_path)
    tokenizer.save_pretrained(args.output_path)

    print("Save completed.")


if __name__ == "__main__":
    main()