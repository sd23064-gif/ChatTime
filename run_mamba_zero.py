#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from model.mamba_model import ChatTimeMamba


TARGET_COLUMN = "OT"

DEFAULT_EXOGENOUS_COLUMNS = [
    "p (mbar)", "T (degC)", "Tpot (K)", "Tdew (degC)", "rh (%)",
    "VPmax (mbar)", "VPact (mbar)", "VPdef (mbar)", "sh (g/kg)",
    "H2OC (mmol/mol)", "rho (g/m**3)", "wv (m/s)", "max. wv (m/s)",
    "wd (deg)", "rain (mm)", "raining (s)", "Tlog (degC)",
]


def reset_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def mae(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, dtype=np.float64), np.asarray(y_pred, dtype=np.float64)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]))) if mask.any() else np.nan


def rmse(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, dtype=np.float64), np.asarray(y_pred, dtype=np.float64)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2))) if mask.any() else np.nan


def smape(y_true, y_pred, eps=1e-8):
    y_true, y_pred = np.asarray(y_true, dtype=np.float64), np.asarray(y_pred, dtype=np.float64)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not mask.any():
        return np.nan
    denominator = np.abs(y_true[mask]) + np.abs(y_pred[mask]) + eps
    return float(np.mean(2.0 * np.abs(y_pred[mask] - y_true[mask]) / denominator))


def calculate_metrics(y_true, y_pred):
    return {"mae": mae(y_true, y_pred), "rmse": rmse(y_true, y_pred), "smape": smape(y_true, y_pred)}


def linear_slope(values):
    values = np.asarray(values, dtype=np.float64)
    mask = np.isfinite(values)
    if mask.sum() < 2:
        return np.nan

    x, y = np.arange(len(values), dtype=np.float64)[mask], values[mask]
    x = x - np.mean(x)
    denominator = np.sum(x ** 2)
    return 0.0 if denominator < 1e-12 else float(np.sum(x * (y - np.mean(y))) / denominator)


def format_number(value):
    return "unknown" if not np.isfinite(value) else f"{float(value):.6g}"


def chronological_split_indices(row_count, train_ratio=0.6, val_ratio=0.2):
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"train_ratio must be between 0 and 1: {train_ratio}")
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError(f"val_ratio must be between 0 and 1: {val_ratio}")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be less than 1.")

    train_end = int(row_count * train_ratio)
    val_end = int(row_count * (train_ratio + val_ratio))
    return {
        "train_start": 0, "train_end": train_end,
        "val_start": train_end, "val_end": val_end,
        "test_start": val_end, "test_end": row_count,
    }


def generate_test_window_starts(row_count, test_start, hist_len, pred_len, stride, max_eval_windows, history_in_test_only=True):
    if stride <= 0:
        raise ValueError(f"window_stride must be positive: {stride}")

    first_hist_start = test_start if history_in_test_only else max(0, test_start - hist_len)
    last_hist_start = row_count - hist_len - pred_len

    if last_hist_start < first_hist_start:
        raise ValueError(
            "Test split is too short for one evaluation window: "
            f"test_rows={row_count - test_start}, required={hist_len + pred_len}"
        )

    starts = np.arange(first_hist_start, last_hist_start + 1, stride, dtype=int)

    if not history_in_test_only:
        starts = starts[starts + hist_len >= test_start]

    if max_eval_windows is not None and max_eval_windows > 0 and len(starts) > max_eval_windows:
        selected = np.linspace(0, len(starts) - 1, max_eval_windows, dtype=int)
        starts = starts[selected]

    return np.unique(starts)


def build_multivariate_context(history_df, exogenous_columns, date_column=None, context_style="summary"):
    descriptions = []

    if date_column in history_df.columns:
        dates = history_df[date_column].dropna()
        if len(dates) > 0:
            start_date = dates.iloc[0].strftime("%Y-%m-%d %H:%M:%S")
            end_date = dates.iloc[-1].strftime("%Y-%m-%d %H:%M:%S")
            descriptions.append(f"The historical observation window runs from {start_date} to {end_date}.")

    for column in exogenous_columns:
        values = pd.to_numeric(history_df[column], errors="coerce").to_numpy(dtype=np.float64)
        finite = values[np.isfinite(values)]
        if len(finite) == 0:
            continue

        latest, mean = finite[-1], np.mean(finite)
        std, minimum, maximum = np.std(finite), np.min(finite), np.max(finite)
        slope = linear_slope(values)

        if context_style == "compact":
            descriptions.append(
                f"{column}: latest={format_number(latest)}, mean={format_number(mean)}, "
                f"range=[{format_number(minimum)}, {format_number(maximum)}], trend={format_number(slope)}."
            )
        else:
            descriptions.append(
                f"For {column}, the latest value is {format_number(latest)}, the historical mean is "
                f"{format_number(mean)}, the standard deviation is {format_number(std)}, the minimum is "
                f"{format_number(minimum)}, the maximum is {format_number(maximum)}, and the linear "
                f"trend per observation step is {format_number(slope)}."
            )

    if not descriptions:
        return None

    return (
        "This sequence represents the target variable OT from a multivariate weather time series. "
        "Use the observed exogenous-variable information together with the OT history to predict "
        "the future OT sequence. " + " ".join(descriptions)
    )


def safe_predict(model, hist_data, context, pred_len, seed):
    reset_seed(seed)
    start = time.perf_counter()

    try:
        prediction = np.asarray(model.predict(hist_data, context=context), dtype=np.float64).reshape(-1)
        elapsed = time.perf_counter() - start

        if len(prediction) != pred_len:
            raise ValueError(f"Prediction length mismatch: actual={len(prediction)}, expected={pred_len}")

        stats = getattr(model, "last_prediction_stats", {}).copy()
        success = np.isfinite(prediction).all()
        error = None if success else "Prediction contains NaN or Inf."
        return prediction, elapsed, error, stats

    except Exception as error:
        elapsed = time.perf_counter() - start
        return np.full(pred_len, np.nan, dtype=np.float64), elapsed, str(error), {}


def bootstrap_mean_ci(values, n_bootstrap=10000, seed=3407):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(n_bootstrap, len(values)))
    bootstrap_means = values[indices].mean(axis=1)
    return float(np.quantile(bootstrap_means, 0.025)), float(np.quantile(bootstrap_means, 0.975))


def safe_mean(series):
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if len(values) else np.nan


def safe_median(series):
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if len(values) else np.nan


def safe_std(series):
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.std(values, ddof=1)) if len(values) > 1 else np.nan


def make_summary(window_df, split, args, exogenous_columns, stride):
    paired = window_df[window_df["paired_success"]].copy()
    if len(paired) == 0:
        raise RuntimeError("No paired successful windows were obtained.")

    multi_gain = paired["mae_gain_multivariate"].to_numpy(dtype=np.float64)
    univariate_naive_gain = paired["mae_gain_univariate_vs_naive"].to_numpy(dtype=np.float64)
    multivariate_naive_gain = paired["mae_gain_multivariate_vs_naive"].to_numpy(dtype=np.float64)

    multi_ci = bootstrap_mean_ci(multi_gain, seed=args.seed)
    uni_naive_ci = bootstrap_mean_ci(univariate_naive_gain, seed=args.seed)
    multi_naive_ci = bootstrap_mean_ci(multivariate_naive_gain, seed=args.seed)

    return {
        "target_column": TARGET_COLUMN,
        "comparison": "OT-only versus OT plus exogenous-variable text context",
        "train_rows": split["train_end"] - split["train_start"],
        "validation_rows": split["val_end"] - split["val_start"],
        "test_rows": split["test_end"] - split["test_start"],
        "train_ratio": args.train_ratio,
        "validation_ratio": args.val_ratio,
        "test_ratio": 1.0 - args.train_ratio - args.val_ratio,
        "history_in_test_only": args.history_in_test_only,
        "hist_len": args.hist_len,
        "pred_len": args.pred_len,
        "window_stride": stride,
        "exogenous_column_count": len(exogenous_columns),
        "total_windows": int(len(window_df)),
        "paired_successful_windows": int(len(paired)),
        "paired_success_ratio": float(len(paired) / len(window_df)),
        "univariate_mae_mean": safe_mean(paired["univariate_mae"]),
        "univariate_mae_median": safe_median(paired["univariate_mae"]),
        "univariate_mae_std": safe_std(paired["univariate_mae"]),
        "multivariate_mae_mean": safe_mean(paired["multivariate_mae"]),
        "multivariate_mae_median": safe_median(paired["multivariate_mae"]),
        "multivariate_mae_std": safe_std(paired["multivariate_mae"]),
        "naive_mae_mean": safe_mean(paired["naive_mae"]),
        "univariate_rmse_mean": safe_mean(paired["univariate_rmse"]),
        "multivariate_rmse_mean": safe_mean(paired["multivariate_rmse"]),
        "naive_rmse_mean": safe_mean(paired["naive_rmse"]),
        "univariate_smape_mean": safe_mean(paired["univariate_smape"]),
        "multivariate_smape_mean": safe_mean(paired["multivariate_smape"]),
        "naive_smape_mean": safe_mean(paired["naive_smape"]),
        "mae_gain_multivariate_mean": safe_mean(paired["mae_gain_multivariate"]),
        "mae_gain_multivariate_median": safe_median(paired["mae_gain_multivariate"]),
        "mae_gain_multivariate_ci95_low": multi_ci[0],
        "mae_gain_multivariate_ci95_high": multi_ci[1],
        "multivariate_better_ratio": float(np.mean(multi_gain > 0)),
        "univariate_better_ratio": float(np.mean(multi_gain < 0)),
        "equal_ratio": float(np.mean(np.isclose(multi_gain, 0.0))),
        "univariate_gain_vs_naive_mean": safe_mean(paired["mae_gain_univariate_vs_naive"]),
        "univariate_gain_vs_naive_ci95_low": uni_naive_ci[0],
        "univariate_gain_vs_naive_ci95_high": uni_naive_ci[1],
        "multivariate_gain_vs_naive_mean": safe_mean(paired["mae_gain_multivariate_vs_naive"]),
        "multivariate_gain_vs_naive_ci95_low": multi_naive_ci[0],
        "multivariate_gain_vs_naive_ci95_high": multi_naive_ci[1],
        "univariate_better_than_naive_ratio": float(np.mean(univariate_naive_gain > 0)),
        "multivariate_better_than_naive_ratio": float(np.mean(multivariate_naive_gain > 0)),
        "univariate_seconds_mean": safe_mean(paired["univariate_seconds"]),
        "multivariate_seconds_mean": safe_mean(paired["multivariate_seconds"]),
        "univariate_parsed_ratio_mean": safe_mean(paired["univariate_parsed_ratio"]),
        "multivariate_parsed_ratio_mean": safe_mean(paired["multivariate_parsed_ratio"]),
        "univariate_fallback_ratio_mean": safe_mean(paired["univariate_fallback_ratio"]),
        "multivariate_fallback_ratio_mean": safe_mean(paired["multivariate_fallback_ratio"]),
    }


def main():
    parser = argparse.ArgumentParser(description="Compare univariate and multivariate-context OT forecasting.")

    parser.add_argument("--csv_path", required=True)
    parser.add_argument("--base_model_path", required=True)
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--date_column", default="date")
    parser.add_argument("--exogenous_columns", nargs="*", default=None)

    parser.add_argument("--train_ratio", type=float, default=0.6)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--hist_len", type=int, default=48)
    parser.add_argument("--pred_len", type=int, default=24)
    parser.add_argument("--window_stride", type=int, default=None)
    parser.add_argument("--max_eval_windows", type=int, default=50)
    parser.add_argument(
        "--history_in_test_only", action=argparse.BooleanOptionalAction, default=True,
        help="Trueなら履歴と予測対象の両方をtest区間内に限定する。"
    )

    parser.add_argument("--max_pred_len", type=int, default=16)
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    parser.add_argument("--merge_lora", action="store_true")
    parser.add_argument("--context_style", choices=["summary", "compact"], default="compact")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--save_predictions", action="store_true")
    args = parser.parse_args()

    reset_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv_path)
    if args.date_column not in df.columns:
        raise ValueError(f"Date column '{args.date_column}' was not found.")
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' was not found.")

    df[args.date_column] = pd.to_datetime(df[args.date_column], errors="coerce")
    invalid_dates = int(df[args.date_column].isna().sum())
    if invalid_dates > 0:
        raise ValueError(f"The date column contains {invalid_dates} invalid values.")

    df = df.sort_values(args.date_column).drop_duplicates(subset=[args.date_column], keep="last").reset_index(drop=True)

    exogenous_columns = args.exogenous_columns if args.exogenous_columns else DEFAULT_EXOGENOUS_COLUMNS
    missing_exogenous = [column for column in exogenous_columns if column not in df.columns]
    if args.exogenous_columns and missing_exogenous:
        raise ValueError(f"Requested exogenous columns were not found: {missing_exogenous}")

    exogenous_columns = [
        column for column in exogenous_columns
        if column in df.columns and column not in {TARGET_COLUMN, args.date_column}
    ]
    if not exogenous_columns:
        raise ValueError("No valid exogenous columns were selected.")

    for column in [TARGET_COLUMN] + exogenous_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    split = chronological_split_indices(len(df), args.train_ratio, args.val_ratio)
    stride = args.window_stride if args.window_stride is not None else args.pred_len
    window_starts = generate_test_window_starts(
        len(df), split["test_start"], args.hist_len, args.pred_len, stride,
        args.max_eval_windows, args.history_in_test_only
    )

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16

    print("Target column:", TARGET_COLUMN)
    print("Rows:", len(df))
    print("Split:", split)
    print("Evaluation windows:", len(window_starts))
    print("Window stride:", stride)
    print("Exogenous columns:", exogenous_columns)
    print("Loading model:", args.base_model_path)
    print("Loading adapter:", args.adapter_path)

    model = ChatTimeMamba(
        base_model_path=args.base_model_path,
        adapter_path=args.adapter_path,
        hist_len=args.hist_len,
        pred_len=args.pred_len,
        max_pred_len=args.max_pred_len,
        num_samples=args.num_samples,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        torch_dtype=dtype,
        merge_lora=args.merge_lora,
    )

    window_rows, detail_rows = [], []

    for window_id, hist_start in enumerate(tqdm(window_starts, desc="Evaluating windows")):
        hist_end, pred_end = hist_start + args.hist_len, hist_start + args.hist_len + args.pred_len
        history_df, target_df = df.iloc[hist_start:hist_end], df.iloc[hist_end:pred_end]
        hist_data = history_df[TARGET_COLUMN].to_numpy(dtype=np.float64)
        true_data = target_df[TARGET_COLUMN].to_numpy(dtype=np.float64)

        if not np.isfinite(hist_data).all() or not np.isfinite(true_data).all():
            window_rows.append({
                "window_id": window_id, "hist_start": hist_start, "hist_end": hist_end,
                "pred_end": pred_end, "paired_success": False,
                "error_univariate": "OT history or target contains NaN/Inf.",
                "error_multivariate": "OT history or target contains NaN/Inf.",
            })
            continue

        context = build_multivariate_context(
            history_df, exogenous_columns, args.date_column, args.context_style
        )
        window_seed = args.seed + window_id

        pred_uni, time_uni, error_uni, stats_uni = safe_predict(
            model, hist_data, None, args.pred_len, window_seed
        )
        pred_multi, time_multi, error_multi, stats_multi = safe_predict(
            model, hist_data, context, args.pred_len, window_seed
        )

        naive = np.full(args.pred_len, hist_data[-1], dtype=np.float64)
        uni_metrics = calculate_metrics(true_data, pred_uni)
        multi_metrics = calculate_metrics(true_data, pred_multi)
        naive_metrics = calculate_metrics(true_data, naive)

        paired_success = (
            error_uni is None and error_multi is None
            and np.isfinite(pred_uni).all() and np.isfinite(pred_multi).all()
        )

        window_rows.append({
            "window_id": window_id, "window_seed": window_seed,
            "hist_start": hist_start, "hist_end": hist_end, "pred_end": pred_end,
            "history_start_date": history_df.iloc[0][args.date_column],
            "history_end_date": history_df.iloc[-1][args.date_column],
            "target_start_date": target_df.iloc[0][args.date_column],
            "target_end_date": target_df.iloc[-1][args.date_column],
            "paired_success": paired_success,
            "univariate_mae": uni_metrics["mae"],
            "multivariate_mae": multi_metrics["mae"],
            "naive_mae": naive_metrics["mae"],
            "mae_gain_multivariate": uni_metrics["mae"] - multi_metrics["mae"],
            "mae_gain_univariate_vs_naive": naive_metrics["mae"] - uni_metrics["mae"],
            "mae_gain_multivariate_vs_naive": naive_metrics["mae"] - multi_metrics["mae"],
            "univariate_rmse": uni_metrics["rmse"],
            "multivariate_rmse": multi_metrics["rmse"],
            "naive_rmse": naive_metrics["rmse"],
            "univariate_smape": uni_metrics["smape"],
            "multivariate_smape": multi_metrics["smape"],
            "naive_smape": naive_metrics["smape"],
            "univariate_seconds": time_uni,
            "multivariate_seconds": time_multi,
            "univariate_parsed_ratio": stats_uni.get("parsed_ratio", np.nan),
            "multivariate_parsed_ratio": stats_multi.get("parsed_ratio", np.nan),
            "univariate_fallback_ratio": stats_uni.get("fallback_ratio", np.nan),
            "multivariate_fallback_ratio": stats_multi.get("fallback_ratio", np.nan),
            "univariate_parse_errors": stats_uni.get("parse_errors", np.nan),
            "multivariate_parse_errors": stats_multi.get("parse_errors", np.nan),
            "error_univariate": error_uni,
            "error_multivariate": error_multi,
            "context": context,
        })

        for horizon in range(args.pred_len):
            row_index = hist_end + horizon
            detail = {
                "window_id": window_id, "window_seed": window_seed,
                "horizon": horizon + 1, "row_index": row_index,
                "date": df.iloc[row_index][args.date_column],
                "true_ot": true_data[horizon],
                "univariate_prediction": pred_uni[horizon],
                "multivariate_prediction": pred_multi[horizon],
                "naive_prediction": naive[horizon],
                "univariate_absolute_error": abs(true_data[horizon] - pred_uni[horizon]),
                "multivariate_absolute_error": abs(true_data[horizon] - pred_multi[horizon]),
                "naive_absolute_error": abs(true_data[horizon] - naive[horizon]),
            }
            if args.save_predictions:
                detail["history_ot"] = json.dumps(hist_data.tolist())
            detail_rows.append(detail)

    window_df = pd.DataFrame(window_rows)
    detail_df = pd.DataFrame(detail_rows)
    summary = make_summary(window_df, split, args, exogenous_columns, stride)
    summary_df = pd.DataFrame([summary])

    window_path = output_dir / "mamba_ot_univariate_multivariate_windows.csv"
    detail_path = output_dir / "mamba_ot_univariate_multivariate_details.csv"
    summary_path = output_dir / "mamba_ot_univariate_multivariate_summary.csv"
    config_path = output_dir / "mamba_ot_univariate_multivariate_config.json"

    window_df.to_csv(window_path, index=False)
    detail_df.to_csv(detail_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    config = {
        "csv_path": args.csv_path,
        "target_column": TARGET_COLUMN,
        "exogenous_columns": exogenous_columns,
        "base_model_path": args.base_model_path,
        "adapter_path": args.adapter_path,
        "split": split,
        "hist_len": args.hist_len,
        "pred_len": args.pred_len,
        "window_stride": stride,
        "max_eval_windows": args.max_eval_windows,
        "history_in_test_only": args.history_in_test_only,
        "max_pred_len": args.max_pred_len,
        "num_samples": args.num_samples,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "temperature": args.temperature,
        "dtype": args.dtype,
        "merge_lora": args.merge_lora,
        "context_style": args.context_style,
        "seed": args.seed,
    }

    with config_path.open("w", encoding="utf-8") as file:
        json.dump(config, file, ensure_ascii=False, indent=2, default=str)

    print("\nEvaluation summary")
    print(summary_df.to_string(index=False))
    print("\nInterpretation")
    print("mae_gain_multivariate > 0: multivariate context is better.")
    print("mae_gain_multivariate < 0: univariate input is better.")
    print("The confidence interval crossing zero means the difference is inconclusive.")
    print("\nSaved")
    print(window_path)
    print(detail_path)
    print(summary_path)
    print(config_path)


if __name__ == "__main__":
    main()