import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

from model.mamba_model import ChatTimeMamba


def mae(y_true, y_pred):
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]))) if mask.any() else np.nan


def rmse(y_true, y_pred):
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2))) if mask.any() else np.nan


def smape(y_true, y_pred, eps=1e-8):
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not mask.any():
        return np.nan
    denominator = np.abs(y_true[mask]) + np.abs(y_pred[mask]) + eps
    return float(np.mean(2.0 * np.abs(y_pred[mask] - y_true[mask]) / denominator))
def normalize_with_history(values, history_min, history_max, clip=True):
    values = np.asarray(values, dtype=np.float64)
    scale = history_max - history_min

    if not np.isfinite(scale) or scale < 1e-12:
        normalized = np.zeros_like(values, dtype=np.float64)
    else:
        normalized = (values - history_min) / scale

    return np.clip(normalized, 0.0, 1.0) if clip else normalized

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--base_model_path", type=str, required=True)
    parser.add_argument("--adapter_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--date_column", type=str, default="date")
    parser.add_argument("--columns", nargs="*", default=None)
    parser.add_argument("--hist_len", type=int, default=48)
    parser.add_argument("--pred_len", type=int, default=24)
    parser.add_argument("--max_pred_len", type=int, default=16)
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--window_start", type=int, default=-1, help="-1ならCSV末尾を評価")
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.csv_path)
    if args.date_column in df.columns:
        df[args.date_column] = pd.to_datetime(df[args.date_column], errors="coerce")

    candidate_columns = [col for col in df.columns if col != args.date_column]
    value_columns = args.columns if args.columns else candidate_columns

    missing_columns = [col for col in value_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Columns not found: {missing_columns}")

    for col in value_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    required_length = args.hist_len + args.pred_len
    if len(df) < required_length:
        raise ValueError(f"CSV is too short: required={required_length}, actual={len(df)}")

    if args.window_start < 0:
        hist_start = len(df) - required_length
    else:
        hist_start = args.window_start

    hist_end = hist_start + args.hist_len
    pred_end = hist_end + args.pred_len

    if hist_start < 0 or pred_end > len(df):
        raise ValueError(f"Invalid window: hist_start={hist_start}, pred_end={pred_end}, rows={len(df)}")

    print("Loading ChatTimeMamba")
    print("Base model:", args.base_model_path)
    print("Adapter:", args.adapter_path)
    print("Columns:", value_columns)
    print("Window:", hist_start, "to", pred_end - 1)

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
        torch_dtype=torch.float16,
        merge_lora=False,
    )

    detail_rows = []
    summary_rows = []

    for col in value_columns:
        series = df[col].to_numpy(dtype=np.float64)
        hist_data = series[hist_start:hist_end]
        true_data = series[hist_end:pred_end]

        if not np.isfinite(hist_data).all() or not np.isfinite(true_data).all():
            print(f"Skipping {col}: history or target contains NaN/Inf")
            continue

        try:
            pred_data = np.asarray(model.predict(hist_data), dtype=np.float64).reshape(-1)
        except Exception as error:
            print(f"Prediction failed for {col}: {error}")
            summary_rows.append({"column": col, "status": "failed", "error": str(error)})
            continue

        if len(pred_data) != args.pred_len:
            print(f"Length mismatch for {col}: pred={len(pred_data)}, expected={args.pred_len}")
            summary_rows.append({
                "column": col, "status": "length_mismatch",
                "prediction_length": len(pred_data), "expected_length": args.pred_len
            })
            continue

        naive_data = np.full(args.pred_len, hist_data[-1], dtype=np.float64)

        history_min = float(np.min(hist_data))
        history_max = float(np.max(hist_data))

        hist_norm = normalize_with_history(hist_data, history_min, history_max)
        true_norm = normalize_with_history(true_data, history_min, history_max)
        pred_norm = normalize_with_history(pred_data, history_min, history_max)
        naive_norm = normalize_with_history(naive_data, history_min, history_max)

        model_mae = mae(true_norm, pred_norm)
        model_rmse = rmse(true_norm, pred_norm)
        model_smape = smape(true_norm, pred_norm)
        naive_mae = mae(true_norm, naive_norm)

        summary_rows.append({
            "column": col,
            "status": "success",
            "history_min": history_min,
            "history_max": history_max,
            "normalized_mae": model_mae,
            "normalized_rmse": model_rmse,
            "normalized_smape": model_smape,
            "normalized_naive_mae": naive_mae,
            "normalized_mae_improvement_vs_naive_pct": (
                (naive_mae - model_mae) / naive_mae * 100.0
                if naive_mae > 1e-12 else np.nan
            ),
            "raw_mae": mae(true_data, pred_data),
            "raw_rmse": rmse(true_data, pred_data),
            "prediction_nan_count": int(np.isnan(pred_data).sum()),
            "target_outside_history_range_count": int(
                ((true_data < history_min) | (true_data > history_max)).sum()
            ),
            "prediction_outside_history_range_count": int(
                ((pred_data < history_min) | (pred_data > history_max)).sum()
            ),
        })

        for horizon in range(args.pred_len):
            row_index = hist_end + horizon
            detail_rows.append({
                "column": col,
                "horizon": horizon + 1,
                "row_index": row_index,
                "date": df.iloc[row_index][args.date_column] if args.date_column in df.columns else None,
                "history_min": history_min,
                "history_max": history_max,
                "true_raw": true_data[horizon],
                "prediction_raw": pred_data[horizon],
                "naive_prediction_raw": naive_data[horizon],
                "true_normalized": true_norm[horizon],
                "prediction_normalized": pred_norm[horizon],
                "naive_prediction_normalized": naive_norm[horizon],
                "normalized_absolute_error": abs(true_norm[horizon] - pred_norm[horizon]),
                "raw_absolute_error": abs(true_data[horizon] - pred_data[horizon]),
            })

        print(f"\nColumn: {col}")
        print("History tail, raw:", hist_data[-5:])
        print("Target, raw:", true_data)
        print("Prediction, raw:", pred_data)
        print("Target, normalized:", true_norm)
        print("Prediction, normalized:", pred_norm)
        print(f"Normalized MAE={model_mae:.6f}")
        print(f"Normalized RMSE={model_rmse:.6f}")
        print(f"Normalized naive MAE={naive_mae:.6f}")

    detail_df = pd.DataFrame(detail_rows)
    summary_df = pd.DataFrame(summary_rows)

    detail_path = os.path.join(args.output_dir, "mamba_csv_prediction_details.csv")
    summary_path = os.path.join(args.output_dir, "mamba_csv_prediction_summary.csv")
    config_path = os.path.join(args.output_dir, "mamba_csv_prediction_config.json")

    detail_df.to_csv(detail_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    config = {
        "csv_path": args.csv_path,
        "base_model_path": args.base_model_path,
        "adapter_path": args.adapter_path,
        "columns": value_columns,
        "hist_len": args.hist_len,
        "pred_len": args.pred_len,
        "hist_start": hist_start,
        "hist_end": hist_end,
        "pred_end": pred_end,
    }
    with open(config_path, "w", encoding="utf-8") as file:
        json.dump(config, file, ensure_ascii=False, indent=2, default=str)

    print("\nSummary")
    print(summary_df.to_string(index=False))
    print("\nSaved:")
    print(detail_path)
    print(summary_path)
    print(config_path)


if __name__ == "__main__":
    main()