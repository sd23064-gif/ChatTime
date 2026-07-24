#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import gc
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from model.model import ChatTime
from model.mamba_model import ChatTimeMamba

np.NaN = np.nan


def reset_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def align_finite(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: true={len(y_true)}, pred={len(y_pred)}")
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    return y_true[mask], y_pred[mask]


def mae(y_true, y_pred):
    y_true, y_pred = align_finite(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred))) if len(y_true) else np.nan


def rmse(y_true, y_pred):
    y_true, y_pred = align_finite(y_true, y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2))) if len(y_true) else np.nan


def smape(y_true, y_pred, eps=1e-8):
    y_true, y_pred = align_finite(y_true, y_pred)
    if not len(y_true):
        return np.nan
    denominator = np.abs(y_true) + np.abs(y_pred) + eps
    return float(np.mean(2.0 * np.abs(y_true - y_pred) / denominator))


def validate_nonnegative(name, value):
    if np.isfinite(value) and value < -1e-12:
        raise ValueError(f"{name} must be non-negative, received {value}")


def chronological_split_indices(n_rows, train_ratio, val_ratio):
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be in (0, 1).")
    if not 0.0 <= val_ratio < 1.0 or train_ratio + val_ratio >= 1.0:
        raise ValueError("val_ratio must be non-negative and train_ratio + val_ratio must be below 1.")
    train_end = int(n_rows * train_ratio)
    val_end = int(n_rows * (train_ratio + val_ratio))
    return {"train_start": 0, "train_end": train_end, "val_start": train_end,
            "val_end": val_end, "test_start": val_end, "test_end": n_rows}


def standardize_from_train(value_df, train_end):
    train_df = value_df.iloc[:train_end]
    mean = train_df.mean(axis=0)
    std = train_df.std(axis=0, ddof=1)
    std = std.mask(~np.isfinite(std) | (std.abs() < 1e-12), 1.0)
    return (value_df - mean) / std, mean, std


def generate_prediction_starts(n_rows, test_start, hist_len, pred_len, stride,
                               max_eval_windows, history_in_test_only):
    if stride <= 0:
        raise ValueError("window_stride must be positive.")
    first_start = test_start + hist_len if history_in_test_only else test_start
    last_start = n_rows - pred_len
    if first_start > last_start:
        return []
    starts = list(range(first_start, last_start + 1, stride))
    if max_eval_windows > 0 and len(starts) > max_eval_windows:
        indices = np.linspace(0, len(starts) - 1, max_eval_windows, dtype=int)
        starts = [starts[index] for index in indices]
    return sorted(set(starts))


def build_model_configs(args):
    configs = []
    if args.evaluate_llama:
        if not args.llama_base_model_path:
            raise ValueError("--llama_base_model_path is required when Llama evaluation is enabled.")
        configs.append({"name": "llama", "type": "llama",
                        "base_model_path": args.llama_base_model_path,
                        "adapter_path": args.llama_adapter_path})
    if args.evaluate_mamba:
        if not args.mamba_base_model_path:
            raise ValueError("--mamba_base_model_path is required when Mamba evaluation is enabled.")
        configs.append({"name": "mamba", "type": "mamba",
                        "base_model_path": args.mamba_base_model_path,
                        "adapter_path": args.mamba_adapter_path})
    if not configs:
        raise ValueError("Enable at least one model.")
    return configs


def load_eval_model(config, hist_len, pred_len, args):
    common = dict(hist_len=hist_len, pred_len=pred_len, max_pred_len=args.max_pred_len,
                  num_samples=args.num_samples, top_k=args.top_k,
                  top_p=args.top_p, temperature=args.temperature)
    if config["type"] == "llama":
        tokenizer_path = config["adapter_path"] or config["base_model_path"]
        return ChatTime(base_model_path=config["base_model_path"],
                        adapter_path=config["adapter_path"], tokenizer_path=tokenizer_path,
                        merge_adapter=args.merge_adapter, **common)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    return ChatTimeMamba(base_model_path=config["base_model_path"],
                         adapter_path=config["adapter_path"], torch_dtype=dtype,
                         merge_lora=args.merge_adapter, **common)


def safe_predict(model, hist_data, pred_len, seed):
    reset_seed(seed)
    started = time.perf_counter()
    try:
        prediction = np.asarray(model.predict(hist_data), dtype=np.float64).reshape(-1)
        if len(prediction) != pred_len:
            raise ValueError(f"Prediction length mismatch: actual={len(prediction)}, expected={pred_len}")
        if not np.isfinite(prediction).all():
            raise ValueError("Prediction contains NaN or Inf.")
        stats = dict(getattr(model, "last_prediction_stats", {}) or {})
        return prediction, time.perf_counter() - started, None, stats
    except Exception as error:
        return np.full(pred_len, np.nan), time.perf_counter() - started, str(error), {}


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


def paired_bootstrap_ci(values, seed, n_bootstrap=10000):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(n_bootstrap, len(values)))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def parse_args():
    parser = argparse.ArgumentParser(description="Fair ETTh2 comparison of ChatTime Llama and Mamba.")
    parser.add_argument("--dataset_path", default="./dataset/ETTh2.csv")
    parser.add_argument("--dataset_name", default="ETTh2")
    parser.add_argument("--date_column", default="date")
    parser.add_argument("--columns", nargs="*", default=None)
    parser.add_argument("--output_path", default="outputs/etth2_llama_mamba")
    parser.add_argument("--evaluate_llama", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--evaluate_mamba", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--llama_base_model_path", default=None)
    parser.add_argument("--llama_adapter_path", default=None)
    parser.add_argument("--mamba_base_model_path", default=None)
    parser.add_argument("--mamba_adapter_path", default=None)
    parser.add_argument("--hist_lengths", default="48,72,96,120")
    parser.add_argument("--pred_len", type=int, default=24)
    parser.add_argument("--window_stride", type=int, default=24)
    parser.add_argument("--max_eval_windows", type=int, default=50)
    parser.add_argument("--train_ratio", type=float, default=0.6)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--history_in_test_only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max_pred_len", type=int, default=16)
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    parser.add_argument("--merge_adapter", action="store_true")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--save_predictions", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    reset_seed(args.seed)
    output_dir = Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    hist_lengths = [int(value.strip()) for value in args.hist_lengths.split(",") if value.strip()]
    if not hist_lengths or any(value <= 0 for value in hist_lengths):
        raise ValueError("hist_lengths must contain positive integers.")
    if args.pred_len <= 0:
        raise ValueError("pred_len must be positive.")

    raw_df = pd.read_csv(args.dataset_path)
    if args.date_column in raw_df.columns:
        raw_df[args.date_column] = pd.to_datetime(raw_df[args.date_column], errors="coerce")
        if raw_df[args.date_column].isna().any():
            raise ValueError("The date column contains invalid values.")
        raw_df = raw_df.sort_values(args.date_column).drop_duplicates(args.date_column, keep="last").reset_index(drop=True)

    candidates = [column for column in raw_df.columns if column != args.date_column]
    selected_columns = args.columns or candidates
    missing = [column for column in selected_columns if column not in raw_df.columns]
    if missing:
        raise ValueError(f"Columns not found: {missing}")
    value_df = raw_df[selected_columns].apply(pd.to_numeric, errors="coerce")
    split = chronological_split_indices(len(value_df), args.train_ratio, args.val_ratio)
    value_df_std, train_mean, train_std = standardize_from_train(value_df, split["train_end"])
    model_configs = build_model_configs(args)

    print("Dataset:", args.dataset_path)
    print("Shape:", value_df_std.shape)
    print("Columns:", selected_columns)
    print("Split:", split)
    results = []

    for config in model_configs:
        for hist_len in hist_lengths:
            starts = generate_prediction_starts(len(value_df_std), split["test_start"], hist_len,
                                                args.pred_len, args.window_stride,
                                                args.max_eval_windows, args.history_in_test_only)
            if not starts:
                print(f"Skipping {config['name']}, hist_len={hist_len}: no valid windows.")
                continue
            model = None
            try:
                print(f"\nLoading {config['name']}, hist_len={hist_len}")
                model = load_eval_model(config, hist_len, args.pred_len, args)
                total = len(selected_columns) * len(starts)
                with tqdm(total=total, desc=f"{config['name']}, hist={hist_len}") as progress:
                    for column_index, column in enumerate(selected_columns):
                        series = value_df_std[column].to_numpy(dtype=np.float64)
                        for window_id, pred_start in enumerate(starts):
                            hist_start, pred_end = pred_start - hist_len, pred_start + args.pred_len
                            hist_data = series[hist_start:pred_start]
                            true_data = series[pred_start:pred_end]
                            sample_seed = args.seed + hist_len * 1_000_000 + column_index * 10_000 + window_id
                            if len(hist_data) != hist_len or len(true_data) != args.pred_len:
                                progress.update(1)
                                continue
                            if not np.isfinite(hist_data).all() or not np.isfinite(true_data).all():
                                progress.update(1)
                                continue

                            pred_data, elapsed, error, stats = safe_predict(model, hist_data, args.pred_len, sample_seed)
                            naive_data = np.full(args.pred_len, hist_data[-1], dtype=np.float64)
                            values = {
                                "mae": mae(true_data, pred_data), "rmse": rmse(true_data, pred_data),
                                "smape": smape(true_data, pred_data), "naive_mae": mae(true_data, naive_data),
                                "naive_rmse": rmse(true_data, naive_data), "naive_smape": smape(true_data, naive_data),
                            }
                            for metric_name, metric_value in values.items():
                                validate_nonnegative(metric_name, metric_value)

                            row = {
                                "dataset": args.dataset_name, "model": config["name"], "column": column,
                                "hist_len": hist_len, "pred_len": args.pred_len, "window_id": window_id,
                                "sample_seed": sample_seed, "hist_start": hist_start,
                                "pred_start": pred_start, "pred_end": pred_end,
                                "success": error is None, **values,
                                "mae_improvement_vs_naive": values["naive_mae"] - values["mae"],
                                "mae_change_vs_naive": values["mae"] - values["naive_mae"],
                                "inference_seconds": elapsed,
                                "parsed_ratio": stats.get("parsed_ratio", np.nan),
                                "fallback_ratio": stats.get("fallback_ratio", np.nan),
                                "parse_errors": stats.get("parse_errors", np.nan), "error": error,
                            }
                            if args.date_column in raw_df.columns:
                                row["prediction_start_date"] = raw_df.iloc[pred_start][args.date_column]
                            if args.save_predictions:
                                row.update({"history": json.dumps(hist_data.tolist()),
                                            "true": json.dumps(true_data.tolist()),
                                            "prediction": json.dumps(pred_data.tolist()),
                                            "naive_prediction": json.dumps(naive_data.tolist())})
                            results.append(row)
                            progress.update(1)

                pd.DataFrame(results).to_csv(output_dir / "details_tmp.csv", index=False)
            finally:
                cleanup_model(model)

    result_df = pd.DataFrame(results)
    if result_df.empty:
        raise RuntimeError("No results were produced.")
    detail_path = output_dir / "llama_mamba_etth2_details.csv"
    result_df.to_csv(detail_path, index=False)
    successful = result_df[result_df["success"]].copy()
    if successful.empty:
        raise RuntimeError("All predictions failed. Check the error column in the details CSV.")

    summary = successful.groupby(["dataset", "model", "hist_len", "pred_len"], as_index=False).agg(
        mae_mean=("mae", "mean"), mae_median=("mae", "median"), mae_std=("mae", "std"),
        rmse_mean=("rmse", "mean"), smape_mean=("smape", "mean"),
        naive_mae_mean=("naive_mae", "mean"),
        mae_improvement_vs_naive_mean=("mae_improvement_vs_naive", "mean"),
        model_better_than_naive_ratio=("mae_improvement_vs_naive", lambda x: float(np.mean(x > 0))),
        inference_seconds_mean=("inference_seconds", "mean"),
        parsed_ratio_mean=("parsed_ratio", "mean"), fallback_ratio_mean=("fallback_ratio", "mean"),
        n_samples=("mae", "count"))
    summary_path = output_dir / "llama_mamba_etth2_summary.csv"
    summary.to_csv(summary_path, index=False)

    by_column = successful.groupby(["dataset", "model", "column", "hist_len", "pred_len"], as_index=False).agg(
        mae_mean=("mae", "mean"), mae_median=("mae", "median"), mae_std=("mae", "std"),
        rmse_mean=("rmse", "mean"), smape_mean=("smape", "mean"), n_samples=("mae", "count"))
    by_column_path = output_dir / "llama_mamba_etth2_summary_by_column.csv"
    by_column.to_csv(by_column_path, index=False)

    keys = ["dataset", "column", "hist_len", "pred_len", "window_id", "pred_start"]
    pivot = successful[keys + ["model", "mae"]].pivot_table(index=keys, columns="model", values="mae", aggfunc="first").reset_index()
    paired_summary = pd.DataFrame()
    paired_path = None
    paired_summary_path = None
    if {"llama", "mamba"}.issubset(pivot.columns):
        paired = pivot.dropna(subset=["llama", "mamba"]).rename(columns={"llama": "mae_llama", "mamba": "mae_mamba"})
        paired["mamba_mae_improvement_vs_llama"] = paired["mae_llama"] - paired["mae_mamba"]
        paired_path = output_dir / "llama_mamba_etth2_paired.csv"
        paired.to_csv(paired_path, index=False)
        rows = []
        for group_keys, group in paired.groupby(["dataset", "hist_len", "pred_len"]):
            dataset, hist_len, pred_len = group_keys
            difference = group["mamba_mae_improvement_vs_llama"].to_numpy(dtype=np.float64)
            low, high = paired_bootstrap_ci(difference, args.seed + int(hist_len))
            rows.append({"dataset": dataset, "hist_len": hist_len, "pred_len": pred_len,
                         "llama_mae_mean": float(group["mae_llama"].mean()),
                         "mamba_mae_mean": float(group["mae_mamba"].mean()),
                         "mamba_mae_improvement_vs_llama_mean": float(difference.mean()),
                         "mamba_mae_improvement_ci95_low": low,
                         "mamba_mae_improvement_ci95_high": high,
                         "mamba_better_ratio": float(np.mean(difference > 0)),
                         "llama_better_ratio": float(np.mean(difference < 0)),
                         "equal_ratio": float(np.mean(np.isclose(difference, 0.0))),
                         "n_paired_samples": int(len(group))})
        paired_summary = pd.DataFrame(rows)
        paired_summary_path = output_dir / "llama_mamba_etth2_paired_summary.csv"
        paired_summary.to_csv(paired_summary_path, index=False)

    config = vars(args).copy()
    config.update({"selected_columns": selected_columns, "hist_lengths_parsed": hist_lengths,
                   "split": split, "train_mean": train_mean.to_dict(), "train_std": train_std.to_dict()})
    config_path = output_dir / "llama_mamba_etth2_config.json"
    with config_path.open("w", encoding="utf-8") as file:
        json.dump(config, file, ensure_ascii=False, indent=2, default=str)

    print("\nSummary")
    print(summary.to_string(index=False))
    if not paired_summary.empty:
        print("\nPaired comparison")
        print(paired_summary.to_string(index=False))
    print("\nSaved:")
    for path in [detail_path, summary_path, by_column_path, paired_path, paired_summary_path, config_path]:
        if path is not None:
            print(" -", path)


if __name__ == "__main__":
    main()
