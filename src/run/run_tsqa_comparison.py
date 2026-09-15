#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import argparse
import ast
import gc
import json
import random
import re
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import logging

from model.model import ChatTime
from model.mamba_model import ChatTimeMamba

warnings.filterwarnings("ignore")
logging.set_verbosity_error()

CHOICES = ("(a)", "(b)", "(c)")


def reset_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_series(value):
    if isinstance(value, np.ndarray):
        array = value.astype(np.float64, copy=False)
    elif isinstance(value, (list, tuple)):
        array = np.asarray(value, dtype=np.float64)
    elif isinstance(value, str):
        array = np.asarray(ast.literal_eval(value), dtype=np.float64)
    else:
        array = np.asarray(value, dtype=np.float64)
    return array.reshape(-1)


def extract_choice(value):
    if value is None:
        return None
    text = str(value).strip().lower()

    patterns = (
        r"\(([abc])\)",
        r"(?:answer|choice|option)\s*(?:is|:)?\s*([abc])\b",
        r"^\s*([abc])\s*[\.)]?(?:\s|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return f"({match.group(1).lower()})"
    return None


def resolve_gold_choice(answer, label):
    answer_choice = extract_choice(answer)
    label_choice = extract_choice(label)
    if answer_choice is not None and label_choice is not None and answer_choice != label_choice:
        raise ValueError(
            f"Answer/Label choice mismatch: answer={answer_choice}, label={label_choice}"
        )
    return answer_choice or label_choice


def make_balanced_subset(frame, max_samples, seed, balance_by_size=False):
    frame = frame.copy()
    if max_samples is None or max_samples <= 0 or len(frame) <= max_samples:
        return frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    group_columns = ["Task", "Size"] if balance_by_size else ["Task"]
    groups = list(frame.groupby(group_columns, dropna=False, sort=True))
    if not groups:
        raise ValueError("No evaluation groups were found.")

    base_quota = max_samples // len(groups)
    remainder = max_samples % len(groups)
    sampled_parts = []

    for group_index, (_, group) in enumerate(groups):
        quota = base_quota + int(group_index < remainder)
        if quota <= 0:
            continue
        sampled_parts.append(
            group.sample(n=min(quota, len(group)), random_state=seed + group_index)
        )

    selected = pd.concat(sampled_parts, ignore_index=False)
    if len(selected) < max_samples:
        remaining = frame.drop(index=selected.index, errors="ignore")
        extra_count = min(max_samples - len(selected), len(remaining))
        if extra_count > 0:
            extra = remaining.sample(n=extra_count, random_state=seed + 10000)
            selected = pd.concat([selected, extra], ignore_index=False)

    return selected.sample(frac=1.0, random_state=seed).reset_index(drop=True)


class RandomChoiceModel:
    def __init__(self):
        pass

    def analyze(self, question, series):
        return random.choice(CHOICES)


class FixedChoiceModel:
    def __init__(self, choice="(a)"):
        if choice not in CHOICES:
            raise ValueError(f"Invalid fixed choice: {choice}")
        self.choice = choice

    def analyze(self, question, series):
        return self.choice


def build_model_configs(args):
    configs = []
    if args.include_random:
        configs.append({"name": "random", "type": "random"})
    if args.include_fixed_a:
        configs.append({"name": "fixed_a", "type": "fixed", "choice": "(a)"})
    if args.llama_base_model_path:
        configs.append({
            "name": args.llama_name,
            "type": "llama",
            "base_model_path": args.llama_base_model_path,
            "adapter_path": args.llama_adapter_path,
            "tokenizer_path": args.llama_tokenizer_path,
        })
    if args.mamba_base_model_path:
        configs.append({
            "name": args.mamba_name,
            "type": "mamba",
            "base_model_path": args.mamba_base_model_path,
            "adapter_path": args.mamba_adapter_path,
            "tokenizer_path": args.mamba_tokenizer_path,
        })
    if not configs:
        raise ValueError(
            "No model selected. Specify a Llama/Mamba base path or a baseline option."
        )
    return configs


def load_eval_model(config, args):
    if config["type"] == "random":
        return RandomChoiceModel()
    if config["type"] == "fixed":
        return FixedChoiceModel(config["choice"])

    common = {
        "hist_len": None,
        "pred_len": None,
        "max_pred_len": args.max_new_tokens,
        "num_samples": args.num_samples,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "temperature": args.temperature,
        "debug_generation": args.debug_generation,
        "debug_samples": args.debug_samples,
        "verbose": args.verbose_model,
    }

    if config["type"] == "llama":
        return ChatTime(
            base_model_path=config["base_model_path"],
            adapter_path=config.get("adapter_path"),
            tokenizer_path=config.get("tokenizer_path") or config["base_model_path"],
            merge_adapter=args.merge_adapter,
            local_files_only=args.local_files_only,
            **common,
        )

    if config["type"] == "mamba":
        dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
        return ChatTimeMamba(
            base_model_path=config["base_model_path"],
            adapter_path=config.get("adapter_path"),
            tokenizer_path=config.get("tokenizer_path") or config["base_model_path"],
            merge_lora=args.merge_adapter,
            torch_dtype=dtype,
            local_files_only=args.local_files_only,
            **common,
        )

    raise ValueError(f"Unknown model type: {config['type']}")


def safe_analyze(model, question, series, seed):
    reset_seed(seed)
    started = time.perf_counter()
    try:
        model_output = model.analyze(question, series)
        pred_choice = extract_choice(model_output)
        parse_success = pred_choice is not None
        return {
            "model_output": "" if model_output is None else str(model_output),
            "pred_choice": pred_choice,
            "parse_success": bool(parse_success),
            "error": None,
            "inference_seconds": time.perf_counter() - started,
        }
    except Exception as error:
        return {
            "model_output": "",
            "pred_choice": None,
            "parse_success": False,
            "error": str(error),
            "inference_seconds": time.perf_counter() - started,
        }


def cleanup_model(model):
    if model is not None:
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except RuntimeError:
            pass


def summarize(frame, group_columns):
    return (
        frame.groupby(group_columns, dropna=False, as_index=False)
        .agg(
            accuracy=("correct", "mean"),
            parse_success_rate=("parse_success", "mean"),
            accuracy_on_parsed=(
                "correct_on_parsed",
                lambda values: float(values.dropna().mean()) if values.notna().any() else np.nan,
            ),
            n_samples=("correct", "size"),
            n_correct=("correct", "sum"),
            n_parse_success=("parse_success", "sum"),
            n_errors=("error", lambda values: int(values.notna().sum())),
            inference_seconds_mean=("inference_seconds", "mean"),
        )
        .sort_values(group_columns)
        .reset_index(drop=True)
    )


def build_confusion_matrix(frame):
    rows = []
    pred_levels = ["(a)", "(b)", "(c)", "UNPARSED"]
    for model_name, model_frame in frame.groupby("model", sort=True):
        local = model_frame.copy()
        local["pred_choice_display"] = local["pred_choice"].fillna("UNPARSED")
        table = pd.crosstab(local["gold_choice"], local["pred_choice_display"])
        table = table.reindex(index=CHOICES, columns=pred_levels, fill_value=0)
        for gold in CHOICES:
            for pred in pred_levels:
                rows.append({
                    "model": model_name,
                    "gold_choice": gold,
                    "pred_choice": pred,
                    "count": int(table.loc[gold, pred]),
                })
    return pd.DataFrame(rows)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Llama and Mamba on the ChengsenWang/TSQA multiple-choice dataset."
    )
    parser.add_argument("--dataset_repo", default="ChengsenWang/TSQA")
    parser.add_argument("--dataset_split", default="train")
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--output_dir", default="outputs/tsqa_model_comparison")
    parser.add_argument("--max_eval_samples", type=int, default=100)
    parser.add_argument("--balance_by_size", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--include_random", action="store_true")
    parser.add_argument("--include_fixed_a", action="store_true")
    parser.add_argument("--llama_base_model_path", default=None)
    parser.add_argument("--llama_adapter_path", default=None)
    parser.add_argument("--llama_tokenizer_path", default=None)
    parser.add_argument("--llama_name", default="llama")
    parser.add_argument("--mamba_base_model_path", default=None)
    parser.add_argument("--mamba_adapter_path", default=None)
    parser.add_argument("--mamba_tokenizer_path", default=None)
    parser.add_argument("--mamba_name", default="mamba")
    parser.add_argument("--merge_adapter", action="store_true")
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    parser.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--debug_first_n", type=int, default=0)
    parser.add_argument("--debug_generation", action="store_true")
    parser.add_argument("--debug_samples", type=int, default=2)
    parser.add_argument("--verbose_model", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    reset_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_configs = build_model_configs(args)
    print("Model configurations:")
    for config in model_configs:
        print(config)

    dataset = load_dataset(
        args.dataset_repo,
        split=args.dataset_split,
        cache_dir=args.cache_dir,
    )
    frame = dataset.to_pandas()
    frame.columns = frame.columns.str.replace("\ufeff", "", regex=False).str.strip()

    required_columns = {"Task", "Size", "Question", "Answer", "Label", "Series"}
    missing = required_columns - set(frame.columns)
    if missing:
        raise ValueError(
            f"Required columns are missing: {sorted(missing)}. "
            f"Available columns: {frame.columns.tolist()}"
        )

    frame = frame.dropna(subset=["Task", "Size", "Question", "Series"]).copy()
    frame["Task"] = frame["Task"].astype(str).str.strip()
    frame["Size"] = pd.to_numeric(frame["Size"], errors="raise").astype(int)
    frame["Series"] = frame["Series"].apply(parse_series)
    frame["gold_choice"] = [
        resolve_gold_choice(answer, label)
        for answer, label in zip(frame["Answer"], frame["Label"])
    ]
    invalid_gold = frame["gold_choice"].isna()
    if invalid_gold.any():
        examples = frame.loc[invalid_gold, ["Answer", "Label"]].head(10).to_dict("records")
        raise ValueError(f"Could not resolve gold choices for {int(invalid_gold.sum())} rows. Examples: {examples}")

    eval_frame = make_balanced_subset(
        frame,
        max_samples=args.max_eval_samples,
        seed=args.seed,
        balance_by_size=args.balance_by_size,
    )
    eval_frame = eval_frame.reset_index(drop=False).rename(columns={"index": "source_index"})

    print({
        "dataset_rows": len(frame),
        "evaluation_rows": len(eval_frame),
        "tasks": eval_frame["Task"].value_counts().to_dict(),
        "sizes": eval_frame["Size"].value_counts().sort_index().to_dict(),
    })

    results = []
    for config in model_configs:
        model = None
        try:
            print(f"\nLoading model: {config['name']}")
            model = load_eval_model(config, args)

            for eval_id, row in tqdm(
                eval_frame.iterrows(), total=len(eval_frame), desc=f"TSQA {config['name']}"
            ):
                series = np.asarray(row["Series"], dtype=np.float64).reshape(-1)
                if not np.isfinite(series).all():
                    continue

                sample_seed = args.seed + int(eval_id)
                analysis = safe_analyze(model, str(row["Question"]), series, sample_seed)
                correct = int(analysis["pred_choice"] == row["gold_choice"])
                correct_on_parsed = float(correct) if analysis["parse_success"] else np.nan

                result = {
                    "eval_id": int(eval_id),
                    "source_index": int(row["source_index"]),
                    "model": config["name"],
                    "task": str(row["Task"]),
                    "size": int(row["Size"]),
                    "actual_size": int(len(series)),
                    "question": str(row["Question"]),
                    "answer": str(row["Answer"]),
                    "label": str(row["Label"]),
                    "gold_choice": row["gold_choice"],
                    "model_output": analysis["model_output"],
                    "pred_choice": analysis["pred_choice"],
                    "parse_success": int(analysis["parse_success"]),
                    "correct": correct,
                    "correct_on_parsed": correct_on_parsed,
                    "inference_seconds": analysis["inference_seconds"],
                    "seed": sample_seed,
                    "error": analysis["error"],
                }
                results.append(result)

                if eval_id < args.debug_first_n:
                    print("\n--- sample ---")
                    print("Model:", config["name"])
                    print("Task:", result["task"])
                    print("Size:", result["size"], "actual:", result["actual_size"])
                    print("Question:", result["question"][:600])
                    print("Gold:", result["gold_choice"])
                    print("Output:", result["model_output"])
                    print("Predicted:", result["pred_choice"])
                    print("Parse success:", result["parse_success"])
                    print("Correct:", result["correct"])
                    print("Error:", result["error"])

            pd.DataFrame(results).to_csv(output_dir / "tsqa_details_tmp.csv", index=False)
        finally:
            cleanup_model(model)

    result_frame = pd.DataFrame(results)
    if result_frame.empty:
        raise RuntimeError("No TSQA results were produced.")

    details_path = output_dir / "tsqa_details.csv"
    result_frame.to_csv(details_path, index=False)

    overall = summarize(result_frame, ["model"])
    by_task = summarize(result_frame, ["model", "task"])
    by_size = summarize(result_frame, ["model", "size"])
    by_task_size = summarize(result_frame, ["model", "task", "size"])
    confusion = build_confusion_matrix(result_frame)

    overall_path = output_dir / "tsqa_summary_overall.csv"
    by_task_path = output_dir / "tsqa_summary_by_task.csv"
    by_size_path = output_dir / "tsqa_summary_by_size.csv"
    by_task_size_path = output_dir / "tsqa_summary_by_task_size.csv"
    confusion_path = output_dir / "tsqa_confusion_matrix.csv"

    overall.to_csv(overall_path, index=False)
    by_task.to_csv(by_task_path, index=False)
    by_size.to_csv(by_size_path, index=False)
    by_task_size.to_csv(by_task_size_path, index=False)
    confusion.to_csv(confusion_path, index=False)

    config_path = output_dir / "tsqa_config.json"
    with config_path.open("w", encoding="utf-8") as file:
        json.dump(
            {**vars(args), "models": model_configs},
            file,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    print("\nOverall summary")
    print(overall.to_string(index=False))
    print("\nSaved:")
    for path in (
        details_path, overall_path, by_task_path, by_size_path,
        by_task_size_path, confusion_path, config_path,
    ):
        print(" -", path)


if __name__ == "__main__":
    main()
