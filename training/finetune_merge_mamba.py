#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Instruction fine-tuning of a merged CPT model with a fresh LoRA adapter.

This script expects --model_path to be a FULL merged CPT model directory,
not a PEFT adapter directory. The base model, input embeddings and LM head
are frozen. Only the newly attached SFT LoRA parameters are optimized.

Two-GPU example:
    CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
        sft_mamba.py \
        --model_path /path/to/merged_cpt_model \
        --dataset_file /path/to/ChatTime-1-Finetune-100K.csv \
        --log_path /path/to/logs \
        --output_path /path/to/sft_adapter \
        --gradient_accumulation_steps 16 \
        --report_to none
"""

import argparse
import inspect
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

RESPONSE_MARKERS = (
    "\n\n#### Response:\n\n",
    "#### Response:\n\n",
    "#### Response:",
    "### Response:",
)

NUMERIC_CHECK_TOKENS = (
    "###-0.9999###",
    "###-0.5001###",
    "###0.0001###",
    "###0.5001###",
    "###0.9999###",
)

TARGET_MODULES = ["in_proj"]


def get_distributed_info():
    """Return torchrun/DDP process information."""
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return local_rank, rank, world_size, world_size > 1, rank == 0


def print_main(*args, **kwargs):
    """Print only from the global rank-zero process."""
    if int(os.environ.get("RANK", "0")) == 0:
        print(*args, **kwargs)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_dtype(name):
    if name == "auto":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16

    return {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[name]


def find_response_end(text):
    candidates = []

    for marker in RESPONSE_MARKERS:
        index = text.rfind(marker)
        if index >= 0:
            candidates.append((index + len(marker), marker))

    if not candidates:
        raise ValueError(
            "A response marker was not found in the training example."
        )

    return max(candidates, key=lambda item: item[0])


def truncate_prompt_ids(prompt_ids, budget, head_tokens):
    if budget <= 0:
        return []

    if len(prompt_ids) <= budget:
        return prompt_ids

    head = min(head_tokens, budget)
    tail = budget - head

    if tail <= 0:
        return prompt_ids[:budget]

    return prompt_ids[:head] + prompt_ids[-tail:]


class CompletionOnlyDataset(Dataset):
    def __init__(
        self,
        raw_dataset,
        tokenizer,
        max_length,
        prompt_head_tokens=384,
    ):
        self.rows = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        skipped = 0

        for row_id, example in enumerate(raw_dataset):
            text = str(example["text"])

            if tokenizer.eos_token and text.endswith(tokenizer.eos_token):
                text = text[:-len(tokenizer.eos_token)]

            try:
                response_end, marker = find_response_end(text)
            except ValueError:
                skipped += 1
                continue

            prompt_text = text[:response_end]
            completion_text = text[response_end:].strip()

            if not completion_text:
                skipped += 1
                continue

            prompt_ids = tokenizer.encode(
                prompt_text,
                add_special_tokens=False,
            )
            completion_ids = tokenizer.encode(
                completion_text,
                add_special_tokens=False,
            )

            eos_ids = (
                [tokenizer.eos_token_id]
                if tokenizer.eos_token_id is not None
                else []
            )

            if len(completion_ids) + len(eos_ids) >= max_length:
                completion_budget = max_length - len(eos_ids)

                if completion_budget <= 0:
                    skipped += 1
                    continue

                completion_ids = completion_ids[:completion_budget]
                prompt_ids = []
            else:
                prompt_budget = (
                    max_length
                    - len(completion_ids)
                    - len(eos_ids)
                )
                prompt_ids = truncate_prompt_ids(
                    prompt_ids,
                    prompt_budget,
                    prompt_head_tokens,
                )

            input_ids = prompt_ids + completion_ids + eos_ids
            labels = [-100] * len(prompt_ids) + completion_ids + eos_ids

            if not input_ids or all(label == -100 for label in labels):
                skipped += 1
                continue

            self.rows.append(
                {
                    "input_ids": torch.tensor(
                        input_ids,
                        dtype=torch.long,
                    ),
                    "attention_mask": torch.ones(
                        len(input_ids),
                        dtype=torch.long,
                    ),
                    "labels": torch.tensor(
                        labels,
                        dtype=torch.long,
                    ),
                    "row_id": row_id,
                    "response_marker": marker,
                }
            )

        if not self.rows:
            raise ValueError(
                "No valid completion-only training examples were created."
            )

        print_main(
            f"Prepared examples: {len(self.rows):,}; "
            f"skipped: {skipped:,}"
        )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]

        return {
            key: value
            for key, value in row.items()
            if key in {"input_ids", "attention_mask", "labels"}
        }


class CompletionCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        input_ids = pad_sequence(
            [feature["input_ids"] for feature in features],
            batch_first=True,
            padding_value=self.pad_token_id,
        )

        attention_mask = pad_sequence(
            [feature["attention_mask"] for feature in features],
            batch_first=True,
            padding_value=0,
        )

        labels = pad_sequence(
            [feature["labels"] for feature in features],
            batch_first=True,
            padding_value=-100,
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def verify_merged_model_path(model_path):
    path = Path(model_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Merged model path does not exist: {path}"
        )

    full_model_files = (
        "model.safetensors",
        "pytorch_model.bin",
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    )

    is_adapter_directory = (
        (path / "adapter_config.json").exists()
        and not any((path / name).exists() for name in full_model_files)
    )

    if is_adapter_directory:
        raise ValueError(
            "--model_path appears to be an adapter-only checkpoint. "
            "Supply the full merged CPT model directory."
        )


def verify_tokenizer_and_vocab(model, tokenizer):
    input_rows = model.get_input_embeddings().weight.shape[0]
    output_layer = model.get_output_embeddings()

    output_rows = (
        output_layer.weight.shape[0]
        if output_layer is not None
        and hasattr(output_layer, "weight")
        else None
    )

    if input_rows != len(tokenizer):
        raise ValueError(
            "Input-embedding/tokenizer vocabulary mismatch: "
            f"input={input_rows}, tokenizer={len(tokenizer)}"
        )

    if output_rows is not None and output_rows != len(tokenizer):
        raise ValueError(
            "LM-head/tokenizer vocabulary mismatch: "
            f"output={output_rows}, tokenizer={len(tokenizer)}"
        )

    for token in NUMERIC_CHECK_TOKENS:
        token_id = tokenizer.convert_tokens_to_ids(token)
        encoded = tokenizer.encode(
            token,
            add_special_tokens=False,
        )

        if (
            token_id is None
            or token_id == tokenizer.unk_token_id
            or encoded != [token_id]
        ):
            raise ValueError(
                "Numeric token is not preserved as one token: "
                f"{token} -> id={token_id}, encoded={encoded}"
            )


def validate_target_modules(model, target_modules):
    module_suffixes = {
        name.rsplit(".", 1)[-1]
        for name, _ in model.named_modules()
    }

    missing = [
        name
        for name in target_modules
        if name not in module_suffixes
    ]

    if missing:
        relevant_suffixes = sorted(
            suffix
            for suffix in module_suffixes
            if "proj" in suffix
        )[:100]

        raise ValueError(
            f"LoRA target modules were not found: {missing}. "
            "Available relevant suffixes include: "
            f"{relevant_suffixes}"
        )


def assert_only_lora_trainable(model):
    unexpected = []
    lora_count = 0

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        if "lora_" in name.lower():
            lora_count += parameter.numel()
        else:
            unexpected.append(name)

    if lora_count == 0:
        raise ValueError(
            "No trainable LoRA parameters were found."
        )

    if unexpected:
        raise ValueError(
            "Non-LoRA parameters are unexpectedly trainable:\n"
            + "\n".join(unexpected[:100])
        )

    input_trainable = (
        model.get_input_embeddings().weight.requires_grad
    )

    output_layer = model.get_output_embeddings()
    output_trainable = bool(
        output_layer is not None
        and hasattr(output_layer, "weight")
        and output_layer.weight.requires_grad
    )

    if input_trainable or output_trainable:
        raise ValueError(
            "Embedding/LM head must be frozen: "
            f"input={input_trainable}, output={output_trainable}"
        )

    total = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print_main(
        f"Trainable LoRA parameters: {lora_count:,} / "
        f"{total:,} ({100 * lora_count / total:.6f}%)"
    )


def gradient_smoke_test(model, dataset, collator):
    """Confirm that the LoRA target receives a finite gradient."""
    model.train()
    model.zero_grad(set_to_none=True)

    batch = collator([dataset[0]])
    device = next(model.parameters()).device
    batch = {
        key: value.to(device)
        for key, value in batch.items()
    }

    output = model(**batch)
    loss = output.loss

    if not torch.isfinite(loss):
        raise ValueError(
            f"Non-finite smoke-test loss: {loss.item()}"
        )

    loss.backward()

    grad_by_target = {}

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        if "lora_" not in name.lower():
            continue

        target = next(
            (
                part
                for part in TARGET_MODULES
                if f".{part}." in f".{name}."
            ),
            "unknown",
        )

        norm = (
            0.0
            if parameter.grad is None
            else float(
                parameter.grad.detach().float().norm().cpu()
            )
        )

        grad_by_target[target] = (
            grad_by_target.get(target, 0.0) + norm
        )

    model.zero_grad(set_to_none=True)

    smoke_result = {
        "loss": float(loss.detach().cpu()),
        "gradient_norm_sum_by_target": grad_by_target,
    }

    print_main(
        "Gradient smoke test:",
        json.dumps(smoke_result, indent=2),
    )

    zero_targets = [
        name
        for name in TARGET_MODULES
        if grad_by_target.get(name, 0.0) == 0.0
    ]

    if zero_targets:
        raise ValueError(
            "No LoRA gradient reached these target modules: "
            f"{zero_targets}. This may indicate a fused forward "
            "path bypassing PEFT LoRA."
        )


def load_raw_dataset(args):
    if args.dataset_file:
        dataset = load_dataset(
            "csv",
            data_files=args.dataset_file,
            split="train",
            num_proc=args.num_proc,
        )
    else:
        url = (
            f"https://huggingface.co/datasets/"
            f"{args.dataset_path}/resolve/main/"
            "ChatTime-1-Finetune-100K.csv"
        )

        dataset = load_dataset(
            "csv",
            data_files=url,
            split="train",
            num_proc=args.num_proc,
        )

    if "text" not in dataset.column_names:
        raise ValueError(
            "Dataset must contain a text column; "
            f"columns={dataset.column_names}"
        )

    return dataset


def save_merged_output(
    model,
    tokenizer,
    output_path,
    dtype,
):
    print_main(
        "Merging the SFT LoRA for final export..."
    )

    merged = model.merge_and_unload()
    merged.to(dtype=dtype)

    merged.save_pretrained(
        output_path,
        safe_serialization=True,
        max_shard_size="5GB",
    )

    tokenizer.save_pretrained(output_path)


def build_parser(model_label):
    parser = argparse.ArgumentParser(
        description=(
            "Fresh-LoRA instruction fine-tuning for merged "
            f"{model_label} CPT model"
        )
    )

    parser.add_argument(
        "--model_path",
        required=True,
        help="Full merged CPT model directory",
    )
    parser.add_argument(
        "--dataset_path",
        default="ChengsenWang/ChatTime-1-Finetune-100K",
    )
    parser.add_argument(
        "--dataset_file",
        default=None,
    )
    parser.add_argument(
        "--log_path",
        required=True,
    )
    parser.add_argument(
        "--output_path",
        required=True,
        help="SFT LoRA adapter output directory",
    )
    parser.add_argument(
        "--merged_output_path",
        default=None,
        help="Optional final full merged SFT model directory",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=2048,
    )
    parser.add_argument(
        "--prompt_head_tokens",
        type=int,
        default=384,
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "fp16", "bf16", "fp32"],
        default="auto",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--num_train_epochs",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--eval_ratio",
        type=float,
        default=0.02,
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=3407,
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        default=None,
    )
    parser.add_argument(
        "--report_to",
        default="wandb",
        choices=["wandb", "none"],
    )
    parser.add_argument(
        "--run_name",
        default=None,
    )

    return parser


def validate_args(args):
    if not 0.0 < args.eval_ratio < 1.0:
        raise ValueError(
            "--eval_ratio must be greater than 0 and less than 1."
        )

    if args.max_seq_length <= 0:
        raise ValueError(
            "--max_seq_length must be greater than zero."
        )

    if args.prompt_head_tokens < 0:
        raise ValueError(
            "--prompt_head_tokens must be zero or greater."
        )

    if args.gradient_accumulation_steps <= 0:
        raise ValueError(
            "--gradient_accumulation_steps must be greater than zero."
        )

    if args.save_steps <= 0 or args.eval_steps <= 0:
        raise ValueError(
            "--save_steps and --eval_steps must be greater than zero."
        )

    if (
        args.early_stopping_patience > 0
        and args.save_steps != args.eval_steps
    ):
        raise ValueError(
            "When early stopping and load_best_model_at_end are used, "
            "--save_steps and --eval_steps should be equal."
        )


def run_training(args):
    validate_args(args)
    verify_merged_model_path(args.model_path)

    (
        local_rank,
        rank,
        world_size,
        is_distributed,
        is_main_process,
    ) = get_distributed_info()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is required for this training script."
        )

    visible_gpu_count = torch.cuda.device_count()

    if local_rank >= visible_gpu_count:
        raise RuntimeError(
            f"LOCAL_RANK={local_rank}, but only "
            f"{visible_gpu_count} CUDA devices are visible."
        )

    # Each torchrun process selects one corresponding GPU.
    torch.cuda.set_device(local_rank)

    set_seed(args.random_seed)
    dtype = resolve_dtype(args.dtype)

    # mkdir with exist_ok=True is safe on all ranks.
    Path(args.log_path).mkdir(parents=True, exist_ok=True)
    Path(args.output_path).mkdir(parents=True, exist_ok=True)

    if args.merged_output_path:
        Path(args.merged_output_path).mkdir(
            parents=True,
            exist_ok=True,
        )

    print_main(
        f"Distributed training: {is_distributed}; "
        f"world_size={world_size}; "
        f"visible_gpus={visible_gpu_count}; "
        f"dtype={dtype}"
    )

    if not is_distributed and visible_gpu_count > 1:
        print_main(
            "Warning: Multiple GPUs are visible, but this process was "
            "not started with torchrun. Only one GPU will be used."
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        use_fast=True,
    )

    if tokenizer.eos_token_id is None:
        raise ValueError(
            "Tokenizer must define eos_token_id."
        )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    # Do not set device_map for DDP.
    # Trainer/Accelerate places each process on its LOCAL_RANK device.
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    verify_tokenizer_and_vocab(model, tokenizer)
    validate_target_modules(model, TARGET_MODULES)

    model.config.use_cache = False

    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={
                    "use_reentrant": False
                }
            )
        except TypeError:
            model.gradient_checkpointing_enable()

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=TARGET_MODULES,
        modules_to_save=None,
    )

    model = get_peft_model(model, lora_config)
    assert_only_lora_trainable(model)

    raw_dataset = load_raw_dataset(args)

    split = raw_dataset.train_test_split(
        test_size=args.eval_ratio,
        seed=args.random_seed,
        shuffle=True,
    )

    train_dataset = CompletionOnlyDataset(
        split["train"],
        tokenizer,
        args.max_seq_length,
        args.prompt_head_tokens,
    )

    eval_dataset = CompletionOnlyDataset(
        split["test"],
        tokenizer,
        args.max_seq_length,
        args.prompt_head_tokens,
    )

    collator = CompletionCollator(
        tokenizer.pad_token_id
    )

    callbacks = []

    if args.early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=(
                    args.early_stopping_patience
                ),
                early_stopping_threshold=0.0,
            )
        )

    training_kwargs = {
        "output_dir": args.log_path,
        "per_device_train_batch_size": (
            args.per_device_train_batch_size
        ),
        "per_device_eval_batch_size": (
            args.per_device_eval_batch_size
        ),
        "gradient_accumulation_steps": (
            args.gradient_accumulation_steps
        ),
        "num_train_epochs": args.num_train_epochs,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "max_grad_norm": 1.0,
        "logging_strategy": "steps",
        "logging_steps": args.logging_steps,
        "logging_first_step": True,
        "eval_strategy": "steps",
        "eval_steps": args.eval_steps,
        "save_strategy": "steps",
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "load_best_model_at_end": False,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "lr_scheduler_type": "cosine",
        "seed": args.random_seed,
        "data_seed": args.random_seed,
        "fp16": dtype == torch.float16,
        "bf16": dtype == torch.bfloat16,
        "remove_unused_columns": False,
        "report_to": (
            []
            if args.report_to == "none"
            else [args.report_to]
        ),
        "run_name": args.run_name,
        "ddp_find_unused_parameters": False,
    }

    # Older Transformers versions use evaluation_strategy.
    signature = inspect.signature(
        TrainingArguments.__init__
    ).parameters

    if "eval_strategy" not in signature:
        training_kwargs["evaluation_strategy"] = (
            training_kwargs.pop("eval_strategy")
        )

    training_args = TrainingArguments(
        **training_kwargs
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        callbacks=callbacks,
    )

    # Trainer has now placed the model on the rank-local GPU.
    gradient_smoke_test(
        trainer.model,
        train_dataset,
        collator,
    )

    trainer.accelerator.wait_for_everyone()

    train_result = trainer.train(
        resume_from_checkpoint=args.resume_from_checkpoint
    )

    trainer.accelerator.wait_for_everyone()

    metrics = trainer.evaluate()
    print_main(
        "Final evaluation:",
        json.dumps(metrics, indent=2),
    )

    trainer.accelerator.wait_for_everyone()

    # Only global rank zero writes the final adapter and tokenizer.
    if trainer.is_world_process_zero():
        unwrapped_model = trainer.accelerator.unwrap_model(
            trainer.model
        )

        unwrapped_model.save_pretrained(
            args.output_path,
            safe_serialization=True,
        )

        tokenizer.save_pretrained(
            args.output_path
        )

        summary = {
            "merged_cpt_model": args.model_path,
            "target_modules": TARGET_MODULES,
            "completion_only_loss": True,
            "world_size": world_size,
            "dtype": str(dtype),
            "train_metrics": train_result.metrics,
            "eval_metrics": metrics,
        }

        summary_path = (
            Path(args.output_path)
            / "training_summary.json"
        )

        with open(
            summary_path,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                summary,
                file,
                indent=2,
                ensure_ascii=False,
            )

    trainer.accelerator.wait_for_everyone()

    # Merge and export only from global rank zero.
    if (
        args.merged_output_path
        and trainer.is_world_process_zero()
    ):
        unwrapped_model = trainer.accelerator.unwrap_model(
            trainer.model
        )

        save_merged_output(
            unwrapped_model,
            tokenizer,
            args.merged_output_path,
            dtype,
        )

    trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
    parser = build_parser("Mamba")
    run_training(parser.parse_args())