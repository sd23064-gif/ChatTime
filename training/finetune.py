#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Instruction fine-tuning of a merged CPT model with a fresh LoRA adapter.

This script expects --model_path to be a FULL merged CPT model directory,
not a PEFT adapter directory. The base model, input embeddings and LM head
are frozen. Only the newly attached SFT LoRA parameters are optimized.
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
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback, Trainer, TrainingArguments

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

RESPONSE_MARKERS = ("\n\n#### Response:\n\n", "#### Response:\n\n", "#### Response:", "### Response:")
NUMERIC_CHECK_TOKENS = ("###-0.9999###", "###-0.5001###", "###0.0001###", "###0.5001###", "###0.9999###")


TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_dtype(name):
    if name == "auto":
        return torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[name]


def find_response_end(text):
    candidates = []
    for marker in RESPONSE_MARKERS:
        index = text.rfind(marker)
        if index >= 0:
            candidates.append((index + len(marker), marker))
    if not candidates:
        raise ValueError("A response marker was not found in the training example.")
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
    def __init__(self, raw_dataset, tokenizer, max_length, prompt_head_tokens=384):
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

            prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
            completion_ids = tokenizer.encode(completion_text, add_special_tokens=False)
            eos_ids = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []

            if len(completion_ids) + len(eos_ids) >= max_length:
                completion_ids = completion_ids[:max_length - len(eos_ids)]
                prompt_ids = []
            else:
                prompt_budget = max_length - len(completion_ids) - len(eos_ids)
                prompt_ids = truncate_prompt_ids(prompt_ids, prompt_budget, prompt_head_tokens)

            input_ids = prompt_ids + completion_ids + eos_ids
            labels = [-100] * len(prompt_ids) + completion_ids + eos_ids
            if not input_ids or all(label == -100 for label in labels):
                skipped += 1
                continue

            self.rows.append({
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                "row_id": row_id,
                "response_marker": marker,
            })

        if not self.rows:
            raise ValueError("No valid completion-only training examples were created.")
        print(f"Prepared examples: {len(self.rows):,}; skipped: {skipped:,}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        return {key: value for key, value in row.items() if key in {"input_ids", "attention_mask", "labels"}}


class CompletionCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        input_ids = pad_sequence([f["input_ids"] for f in features], batch_first=True, padding_value=self.pad_token_id)
        attention_mask = pad_sequence([f["attention_mask"] for f in features], batch_first=True, padding_value=0)
        labels = pad_sequence([f["labels"] for f in features], batch_first=True, padding_value=-100)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def verify_merged_model_path(model_path):
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Merged model path does not exist: {path}")
    if (path / "adapter_config.json").exists() and not any((path / name).exists() for name in ("model.safetensors", "pytorch_model.bin", "model.safetensors.index.json", "pytorch_model.bin.index.json")):
        raise ValueError("--model_path appears to be an adapter-only checkpoint. Supply the full merged CPT model directory.")


def verify_tokenizer_and_vocab(model, tokenizer):
    input_rows = model.get_input_embeddings().weight.shape[0]
    output_layer = model.get_output_embeddings()
    output_rows = output_layer.weight.shape[0] if output_layer is not None and hasattr(output_layer, "weight") else None
    if input_rows != len(tokenizer):
        raise ValueError(f"Input-embedding/tokenizer vocabulary mismatch: input={input_rows}, tokenizer={len(tokenizer)}")
    if output_rows is not None and output_rows != len(tokenizer):
        raise ValueError(f"LM-head/tokenizer vocabulary mismatch: output={output_rows}, tokenizer={len(tokenizer)}")

    for token in NUMERIC_CHECK_TOKENS:
        token_id = tokenizer.convert_tokens_to_ids(token)
        encoded = tokenizer.encode(token, add_special_tokens=False)
        if token_id is None or token_id == tokenizer.unk_token_id or encoded != [token_id]:
            raise ValueError(f"Numeric token is not preserved as one token: {token} -> id={token_id}, encoded={encoded}")


def validate_target_modules(model, target_modules):
    module_suffixes = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
    missing = [name for name in target_modules if name not in module_suffixes]
    if missing:
        raise ValueError(f"LoRA target modules were not found: {missing}. Available relevant suffixes include: {sorted(x for x in module_suffixes if 'proj' in x)[:100]}")


def assert_only_lora_trainable(model):
    unexpected = []
    lora_count = 0
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            if "lora_" in name.lower():
                lora_count += parameter.numel()
            else:
                unexpected.append(name)
    if lora_count == 0:
        raise ValueError("No trainable LoRA parameters were found.")
    if unexpected:
        raise ValueError("Non-LoRA parameters are unexpectedly trainable:\n" + "\n".join(unexpected[:100]))

    input_trainable = model.get_input_embeddings().weight.requires_grad
    output_layer = model.get_output_embeddings()
    output_trainable = bool(output_layer is not None and hasattr(output_layer, "weight") and output_layer.weight.requires_grad)
    if input_trainable or output_trainable:
        raise ValueError(f"Embedding/LM head must be frozen: input={input_trainable}, output={output_trainable}")

    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable LoRA parameters: {lora_count:,} / {total:,} ({100*lora_count/total:.6f}%)")


def gradient_smoke_test(model, dataset, collator):
    model.train()
    model.zero_grad(set_to_none=True)
    batch = collator([dataset[0]])
    device = next(model.parameters()).device
    batch = {key: value.to(device) for key, value in batch.items()}
    output = model(**batch)
    loss = output.loss
    if not torch.isfinite(loss):
        raise ValueError(f"Non-finite smoke-test loss: {loss.item()}")
    loss.backward()

    grad_by_target = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and "lora_" in name.lower():
            target = next((part for part in TARGET_MODULES if f".{part}." in f".{name}."), "unknown")
            norm = 0.0 if parameter.grad is None else float(parameter.grad.detach().float().norm().cpu())
            grad_by_target[target] = grad_by_target.get(target, 0.0) + norm

    model.zero_grad(set_to_none=True)
    print("Gradient smoke test:", json.dumps({"loss": float(loss.detach().cpu()), "gradient_norm_sum_by_target": grad_by_target}, indent=2))
    zero_targets = [name for name in TARGET_MODULES if grad_by_target.get(name, 0.0) == 0.0]
    if zero_targets:
        raise ValueError(f"No LoRA gradient reached these target modules: {zero_targets}. This may indicate a fused forward path bypassing PEFT LoRA.")


def load_raw_dataset(args):
    if args.dataset_file:
        dataset = load_dataset("csv", data_files=args.dataset_file, split="train")
    else:
        url = f"https://huggingface.co/datasets/{args.dataset_path}/resolve/main/ChatTime-1-Finetune-100K.csv"
        dataset = load_dataset("csv", data_files=url, split="train")
    if "text" not in dataset.column_names:
        raise ValueError(f"Dataset must contain a text column; columns={dataset.column_names}")
    return dataset


def save_merged_output(model, tokenizer, output_path, dtype):
    print("Merging the SFT LoRA for final export...")
    merged = model.merge_and_unload()
    merged.to(dtype=dtype)
    merged.save_pretrained(output_path, safe_serialization=True, max_shard_size="5GB")
    tokenizer.save_pretrained(output_path)


def build_parser(model_label):
    parser = argparse.ArgumentParser(description=f"Fresh-LoRA instruction fine-tuning for merged {model_label} CPT model")
    parser.add_argument("--model_path", required=True, help="Full merged CPT model directory")
    parser.add_argument("--dataset_path", default="ChengsenWang/ChatTime-1-Finetune-100K")
    parser.add_argument("--dataset_file", default=None)
    parser.add_argument("--log_path", required=True)
    parser.add_argument("--output_path", required=True, help="SFT LoRA adapter output directory")
    parser.add_argument("--merged_output_path", default=None, help="Optional final full merged SFT model directory")
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--prompt_head_tokens", type=int, default=384)
    parser.add_argument("--dtype", choices=["auto", "fp16", "bf16", "fp32"], default="auto")
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=32)
    parser.add_argument("--eval_ratio", type=float, default=0.02)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_total_limit", type=int, default=5)
    parser.add_argument("--random_seed", type=int, default=3407)
    parser.add_argument("--num_proc", type=int, default=64)
    parser.add_argument("--early_stopping_patience", type=int, default=0)
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--report_to", default="wandb", choices=["wandb", "none"])
    parser.add_argument("--run_name", default=None)
    return parser


def run_training(args):
    verify_merged_model_path(args.model_path)
    set_seed(args.random_seed)
    dtype = resolve_dtype(args.dtype)
    Path(args.log_path).mkdir(parents=True, exist_ok=True)
    Path(args.output_path).mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=True)
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must define eos_token_id.")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map={"": 0},
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    verify_tokenizer_and_vocab(model, tokenizer)
    validate_target_modules(model, TARGET_MODULES)

    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
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

    raw = load_raw_dataset(args)
    split = raw.train_test_split(test_size=args.eval_ratio, seed=args.random_seed, shuffle=True)
    train_dataset = CompletionOnlyDataset(split["train"], tokenizer, args.max_seq_length, args.prompt_head_tokens)
    eval_dataset = CompletionOnlyDataset(split["test"], tokenizer, args.max_seq_length, args.prompt_head_tokens)
    collator = CompletionCollator(tokenizer.pad_token_id)

    gradient_smoke_test(model, train_dataset, collator)

    callbacks = []
    if args.early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience, early_stopping_threshold=0.0))

    training_kwargs = dict(
        output_dir=args.log_path,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=1.0,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        logging_first_step=True,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        lr_scheduler_type="cosine",
        seed=args.random_seed,
        data_seed=args.random_seed,
        fp16=dtype == torch.float16,
        bf16=dtype == torch.bfloat16,
        remove_unused_columns=False,
        report_to=[] if args.report_to == "none" else [args.report_to],
        run_name=args.run_name,
    )
    # Transformers versions before eval_strategy used evaluation_strategy.
    signature = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" not in signature:
        training_kwargs["evaluation_strategy"] = training_kwargs.pop("eval_strategy")

    trainer = Trainer(
        model=model,
        args=TrainingArguments(**training_kwargs),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        callbacks=callbacks,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    metrics = trainer.evaluate()
    print("Final evaluation:", json.dumps(metrics, indent=2))

    model.save_pretrained(args.output_path, safe_serialization=True)
    tokenizer.save_pretrained(args.output_path)
    with open(Path(args.output_path) / "training_summary.json", "w", encoding="utf-8") as file:
        json.dump({"merged_cpt_model": args.model_path, "target_modules": TARGET_MODULES, "completion_only_loss": True, "eval_metrics": metrics}, file, indent=2)

    if args.merged_output_path:
        Path(args.merged_output_path).mkdir(parents=True, exist_ok=True)
        save_merged_output(model, tokenizer, args.merged_output_path, dtype)


if __name__ == "__main__":
    parser = build_parser("Llama")
    run_training(parser.parse_args())
