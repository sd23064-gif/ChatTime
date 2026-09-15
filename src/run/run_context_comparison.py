#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import argparse
import ast
import gc
import json
import random
import time
import warnings
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from huggingface_hub import list_repo_files
from tqdm import tqdm
from transformers import logging

from model.model import ChatTime
from model.mamba_model import ChatTimeMamba

warnings.filterwarnings("ignore")
logging.set_verbosity_error()

CGTSF_REPO_ID = "ChengsenWang/CGTSF"
SUPPORTED_DATASETS = {"LEU", "MSPG", "PTF"}


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


def normalize_context(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value).strip()


def discover_dataset_csv(dataset_name, dataset_file=None):
    dataset_name = str(dataset_name).strip().upper()
    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Unsupported dataset '{dataset_name}'. "
            f"Choose from {sorted(SUPPORTED_DATASETS)}."
        )

    repo_files = list_repo_files(repo_id=CGTSF_REPO_ID, repo_type="dataset")
    csv_files = sorted(
        name for name in repo_files
        if name.startswith(f"{dataset_name}/") and name.lower().endswith(".csv")
    )
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV file was found under {dataset_name}/ in {CGTSF_REPO_ID}."
        )

    if dataset_file is not None:
        selected = str(PurePosixPath(dataset_file))
        if selected not in repo_files:
            raise FileNotFoundError(
                f"The requested repository file does not exist: {selected}\n"
                f"Available CSV files: {csv_files}"
            )
        if not selected.startswith(f"{dataset_name}/"):
            raise ValueError(
                f"--dataset_file must be inside {dataset_name}/, received: {selected}"
            )
        if not selected.lower().endswith(".csv"):
            raise ValueError("--dataset_file must point to a CSV file.")
        return selected, csv_files

    preferred = f"{dataset_name}/{dataset_name}.csv"
    if preferred in csv_files:
        return preferred, csv_files
    if len(csv_files) == 1:
        return csv_files[0], csv_files

    same_stem = [
        name for name in csv_files
        if PurePosixPath(name).stem.upper() == dataset_name
    ]
    if len(same_stem) == 1:
        return same_stem[0], csv_files

    raise ValueError(
        f"Multiple CSV files were found for {dataset_name}. "
        "Specify one with --dataset_file.\n"
        + "\n".join(csv_files)
    )


def load_cgtsf_frame(dataset_name, dataset_file=None, cache_dir=None):
    selected_file, available = discover_dataset_csv(dataset_name, dataset_file)
    print(f"Available CSV files under {dataset_name}/:")
    for name in available:
        print(" -", name)
    print("Selected CGTSF file:", selected_file)

    dataset = load_dataset(
        CGTSF_REPO_ID,
        data_files={"train": selected_file},
        split="train",
        cache_dir=cache_dir,
    )
    frame = dataset.to_pandas()
    if frame.empty:
        raise ValueError(f"The selected dataset is empty: {selected_file}")
    return frame, selected_file


def chronological_split(frame, train_ratio, val_ratio, group_column="Idx"):
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be in (0, 1).")
    if val_ratio < 0.0 or train_ratio + val_ratio >= 1.0:
        raise ValueError("val_ratio must be non-negative and train_ratio + val_ratio < 1.")

    if group_column not in frame.columns:
        n = len(frame)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        return (
            frame.iloc[:train_end].copy(),
            frame.iloc[train_end:val_end].copy(),
            frame.iloc[val_end:].copy(),
        )

    train_parts, val_parts, test_parts = [], [], []
    for _, group in frame.groupby(group_column, sort=False):
        group = group.sort_values("Date").reset_index(drop=True)
        n = len(group)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        train_parts.append(group.iloc[:train_end])
        val_parts.append(group.iloc[train_end:val_end])
        test_parts.append(group.iloc[val_end:])

    return (
        pd.concat(train_parts, ignore_index=True),
        pd.concat(val_parts, ignore_index=True),
        pd.concat(test_parts, ignore_index=True),
    )


def select_eval_rows(test_frame, max_windows):
    test_frame = test_frame.reset_index(drop=False).rename(columns={"index": "source_row"})
    if max_windows is None or max_windows <= 0 or len(test_frame) <= max_windows:
        return test_frame
    indices = np.linspace(0, len(test_frame) - 1, max_windows, dtype=int)
    return test_frame.iloc[np.unique(indices)].reset_index(drop=True)


def aligned_arrays(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: true={len(y_true)}, pred={len(y_pred)}")
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    return y_true[mask], y_pred[mask]


def mae(y_true, y_pred):
    y_true, y_pred = aligned_arrays(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred))) if len(y_true) else np.nan


def rmse(y_true, y_pred):
    y_true, y_pred = aligned_arrays(y_true, y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2))) if len(y_true) else np.nan


def smape(y_true, y_pred, eps=1e-8):
    y_true, y_pred = aligned_arrays(y_true, y_pred)
    if not len(y_true):
        return np.nan
    return float(np.mean(2.0 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + eps)))


def sign_flip_rate(y_true, y_pred):
    y_true, y_pred = aligned_arrays(y_true, y_pred)
    return float(np.mean(y_true * y_pred < 0)) if len(y_true) else np.nan


def magnitude_mae(y_true, y_pred):
    y_true, y_pred = aligned_arrays(y_true, y_pred)
    return float(np.mean(np.abs(np.abs(y_true) - np.abs(y_pred)))) if len(y_true) else np.nan


def bootstrap_ci(values, seed, n_bootstrap=10000):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(0, len(values), size=(n_bootstrap, len(values)))
    means = values[sample_indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


class NaiveLastModel:
    def __init__(self, hist_len, pred_len):
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.last_prediction_stats = {}

    def predict(self, hist_data, context=None):
        self.last_prediction_stats = {
            "parsed_ratio": 1.0,
            "fallback_ratio": 0.0,
            "parse_errors": 0,
        }
        return np.full(self.pred_len, float(hist_data[-1]), dtype=np.float64)


def build_model_configs(args):
    configs = []
    if args.include_naive:
        configs.append({"name": "naive_last", "type": "naive_last"})
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
        raise ValueError("Specify at least one model path or --include_naive.")
    return configs


def load_eval_model(config, hist_len, pred_len, args):
    common = {
        "hist_len": hist_len,
        "pred_len": pred_len,
        "max_pred_len": args.max_pred_len,
        "num_samples": args.num_samples,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "temperature": args.temperature,
        "debug_generation": args.debug_generation,
        "debug_samples": args.debug_samples,
        "verbose": args.verbose_model,
    }
    if config["type"] == "naive_last":
        return NaiveLastModel(hist_len, pred_len)
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


def safe_predict(model, hist_data, context, pred_len, seed):
    reset_seed(seed)
    started = time.perf_counter()
    try:
        prediction = np.asarray(model.predict(hist_data, context=context), dtype=np.float64).reshape(-1)
        if len(prediction) != pred_len:
            raise ValueError(f"Prediction length mismatch: {len(prediction)} != {pred_len}")
        if not np.isfinite(prediction).all():
            raise ValueError("Prediction contains NaN or Inf.")
        stats = dict(getattr(model, "last_prediction_stats", {}) or {})
        return prediction, time.perf_counter() - started, None, stats
    except Exception as error:
        return np.full(pred_len, np.nan), time.perf_counter() - started, str(error), {}


def is_strict_success(error, prediction, stats, require_complete):
    generation_success = error is None and np.isfinite(prediction).all()
    if not require_complete:
        return generation_success
    parsed_ratio = stats.get("parsed_ratio", np.nan)
    fallback_ratio = stats.get("fallback_ratio", np.nan)
    return (
        generation_success
        and np.isfinite(parsed_ratio)
        and parsed_ratio >= 1.0 - 1e-12
        and np.isfinite(fallback_ratio)
        and fallback_ratio <= 1e-12
    )


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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare context-free and context-guided forecasts on one CGTSF dataset."
    )
    parser.add_argument("--dataset_name", type=str.upper, choices=sorted(SUPPORTED_DATASETS), required=True)
    parser.add_argument("--dataset_file", default=None,
                        help="Optional exact repository path, for example PTF/PTF.csv.")
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--output_dir", default="outputs/context_comparison")
    parser.add_argument("--hist_lengths", default="48,72,96,120")
    parser.add_argument("--pred_len", type=int, default=24)
    parser.add_argument("--max_eval_windows", type=int, default=50)
    parser.add_argument("--train_ratio", type=float, default=0.6)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--max_pred_len", type=int, default=16)
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--llama_base_model_path", default=None)
    parser.add_argument("--llama_adapter_path", default=None)
    parser.add_argument("--llama_tokenizer_path", default=None)
    parser.add_argument("--llama_name", default="llama")
    parser.add_argument("--mamba_base_model_path", default=None)
    parser.add_argument("--mamba_adapter_path", default=None)
    parser.add_argument("--mamba_tokenizer_path", default=None)
    parser.add_argument("--mamba_name", default="mamba")
    parser.add_argument("--include_naive", action="store_true")
    parser.add_argument("--merge_adapter", action="store_true")
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--require_complete_generation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save_predictions", action="store_true")
    parser.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--debug_generation", action="store_true")
    parser.add_argument("--debug_samples", type=int, default=2)
    parser.add_argument("--verbose_model", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    reset_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    hist_lengths = [int(value.strip()) for value in args.hist_lengths.split(",") if value.strip()]
    if not hist_lengths or any(value <= 0 for value in hist_lengths):
        raise ValueError("hist_lengths must contain positive integers.")
    if args.pred_len <= 0:
        raise ValueError("pred_len must be positive.")

    model_configs = build_model_configs(args)
    print("Model configurations:")
    for config in model_configs:
        print(config)

    frame, selected_file = load_cgtsf_frame(
        args.dataset_name,
        dataset_file=args.dataset_file,
        cache_dir=args.cache_dir,
    )
    required_columns = {"Hist", "Pred", "Text"}
    missing = required_columns - set(frame.columns)
    if missing:
        raise ValueError(
            f"Required columns are missing: {sorted(missing)}. "
            f"Available columns: {frame.columns.tolist()}"
        )
    if "Idx" not in frame.columns:
        frame["Idx"] = 0
    if "Date" not in frame.columns:
        frame["Date"] = np.arange(len(frame))
    else:
        parsed_dates = pd.to_datetime(frame["Date"], errors="coerce")
        if parsed_dates.notna().all():
            frame["Date"] = parsed_dates

    frame["Hist"] = frame["Hist"].apply(parse_series)
    frame["Pred"] = frame["Pred"].apply(parse_series)
    frame["Text"] = frame["Text"].apply(normalize_context)

    train_frame, validation_frame, test_frame = chronological_split(
        frame, args.train_ratio, args.val_ratio
    )
    eval_frame = select_eval_rows(test_frame, args.max_eval_windows)
    print({
        "dataset_name": args.dataset_name,
        "selected_file": selected_file,
        "total_rows": len(frame),
        "train_rows": len(train_frame),
        "validation_rows": len(validation_frame),
        "test_rows": len(test_frame),
        "evaluated_rows": len(eval_frame),
        "series_count": int(frame["Idx"].nunique()),
    })

    results = []
    for config in model_configs:
        for hist_len in hist_lengths:
            model = None
            try:
                print(f"\nLoading {config['name']}, hist_len={hist_len}")
                model = load_eval_model(config, hist_len, args.pred_len, args)

                for eval_id, row in tqdm(
                    eval_frame.iterrows(), total=len(eval_frame),
                    desc=f"{config['name']}, hist={hist_len}"
                ):
                    history = row["Hist"][-hist_len:]
                    truth = row["Pred"][:args.pred_len]
                    context = row["Text"]
                    if len(history) != hist_len or len(truth) != args.pred_len:
                        continue
                    if not np.isfinite(history).all() or not np.isfinite(truth).all():
                        continue

                    pair_seed = args.seed + hist_len * 1_000_000 + int(eval_id)
                    pred_zero, zero_seconds, zero_error, zero_stats = safe_predict(
                        model, history, None, args.pred_len, pair_seed
                    )
                    pred_context, context_seconds, context_error, context_stats = safe_predict(
                        model, history, context, args.pred_len, pair_seed
                    )

                    zero_success = is_strict_success(
                        zero_error, pred_zero, zero_stats, args.require_complete_generation
                    )
                    context_success = is_strict_success(
                        context_error, pred_context, context_stats, args.require_complete_generation
                    )

                    row_result = {
                        "dataset": args.dataset_name,
                        "dataset_file": selected_file,
                        "model": config["name"],
                        "hist_len": hist_len,
                        "pred_len": args.pred_len,
                        "eval_id": int(eval_id),
                        "source_row": int(row["source_row"]),
                        "idx": row["Idx"],
                        "date": row["Date"],
                        "context_length": len(context),
                        "seed": pair_seed,
                        "zero_success": zero_success,
                        "context_success": context_success,
                        "paired_success": zero_success and context_success,
                        "mae_zero": mae(truth, pred_zero),
                        "mae_context": mae(truth, pred_context),
                        "rmse_zero": rmse(truth, pred_zero),
                        "rmse_context": rmse(truth, pred_context),
                        "smape_zero": smape(truth, pred_zero),
                        "smape_context": smape(truth, pred_context),
                        "sign_flip_zero": sign_flip_rate(truth, pred_zero),
                        "sign_flip_context": sign_flip_rate(truth, pred_context),
                        "magnitude_mae_zero": magnitude_mae(truth, pred_zero),
                        "magnitude_mae_context": magnitude_mae(truth, pred_context),
                        "zero_seconds": zero_seconds,
                        "context_seconds": context_seconds,
                        "zero_parsed_ratio": zero_stats.get("parsed_ratio", np.nan),
                        "context_parsed_ratio": context_stats.get("parsed_ratio", np.nan),
                        "zero_fallback_ratio": zero_stats.get("fallback_ratio", np.nan),
                        "context_fallback_ratio": context_stats.get("fallback_ratio", np.nan),
                        "zero_parse_errors": zero_stats.get("parse_errors", np.nan),
                        "context_parse_errors": context_stats.get("parse_errors", np.nan),
                        "zero_error": zero_error,
                        "context_error": context_error,
                        "text": context,
                    }
                    row_result["mae_context_improvement"] = (
                        row_result["mae_zero"] - row_result["mae_context"]
                    )
                    row_result["rmse_context_improvement"] = (
                        row_result["rmse_zero"] - row_result["rmse_context"]
                    )
                    if args.save_predictions:
                        row_result.update({
                            "history": json.dumps(history.tolist()),
                            "true": json.dumps(truth.tolist()),
                            "pred_zero": json.dumps(pred_zero.tolist()),
                            "pred_context": json.dumps(pred_context.tolist()),
                        })
                    results.append(row_result)

                pd.DataFrame(results).to_csv(
                    output_dir / "context_comparison_details_tmp.csv", index=False
                )
            finally:
                cleanup_model(model)

    details = pd.DataFrame(results)
    if details.empty:
        raise RuntimeError("No evaluation results were produced.")
    details_path = output_dir / "context_comparison_details.csv"
    details.to_csv(details_path, index=False)

    paired = details[details["paired_success"]].copy()
    if paired.empty:
        raise RuntimeError(
            "No paired successful predictions. Inspect context_comparison_details.csv."
        )

    summary_rows = []
    for keys, group in paired.groupby(["dataset", "model", "hist_len", "pred_len"]):
        improvement = group["mae_context_improvement"].to_numpy(dtype=np.float64)
        ci_low, ci_high = bootstrap_ci(improvement, args.seed + int(keys[2]))
        summary_rows.append({
            "dataset": keys[0],
            "model": keys[1],
            "hist_len": keys[2],
            "pred_len": keys[3],
            "n_paired": int(len(group)),
            "mae_zero_mean": float(group["mae_zero"].mean()),
            "mae_context_mean": float(group["mae_context"].mean()),
            "mae_context_improvement_mean": float(improvement.mean()),
            "mae_context_improvement_median": float(np.median(improvement)),
            "mae_context_improvement_ci95_low": ci_low,
            "mae_context_improvement_ci95_high": ci_high,
            "context_better_ratio": float(np.mean(improvement > 0)),
            "zero_better_ratio": float(np.mean(improvement < 0)),
            "equal_ratio": float(np.mean(np.isclose(improvement, 0.0))),
            "rmse_zero_mean": float(group["rmse_zero"].mean()),
            "rmse_context_mean": float(group["rmse_context"].mean()),
            "smape_zero_mean": float(group["smape_zero"].mean()),
            "smape_context_mean": float(group["smape_context"].mean()),
            "zero_parsed_ratio_mean": float(group["zero_parsed_ratio"].mean()),
            "context_parsed_ratio_mean": float(group["context_parsed_ratio"].mean()),
            "zero_fallback_ratio_mean": float(group["zero_fallback_ratio"].mean()),
            "context_fallback_ratio_mean": float(group["context_fallback_ratio"].mean()),
            "zero_seconds_mean": float(group["zero_seconds"].mean()),
            "context_seconds_mean": float(group["context_seconds"].mean()),
        })

    summary = pd.DataFrame(summary_rows)
    summary_path = output_dir / "context_comparison_summary.csv"
    summary.to_csv(summary_path, index=False)

    config_path = output_dir / "context_comparison_config.json"
    with config_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                **vars(args),
                "selected_dataset_file": selected_file,
                "hist_lengths_parsed": hist_lengths,
                "models": model_configs,
            },
            file,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    print("\nSummary")
    print(summary.to_string(index=False))
    print("\nSaved:")
    print(" -", details_path)
    print(" -", summary_path)
    print(" -", config_path)


if __name__ == "__main__":
    main()
