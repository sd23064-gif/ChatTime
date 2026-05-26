import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import gc
import torch

from model.model import ChatTime

np.NaN = np.nan

def mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return np.mean(np.abs(y_true - y_pred))


def chronological_split(df, train_ratio=0.6, val_ratio=0.2):
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    return train_df, val_df, test_df


def main():
    dataset_path = "./dataset/ETTh2.csv"
    model_path = "ChengsenWang/ChatTime-1-7B-Chat"

    dataset_name = "ETTh2"
    pred_len = 24
    hist_lengths = [48, 72, 96, 120]

    os.makedirs("outputs", exist_ok=True)

    # =========================
    # 1. Load dataset
    # =========================
    raw_df = pd.read_csv(dataset_path)

    value_df = raw_df.drop(columns=["date"]).apply(pd.to_numeric, errors="coerce")

    train_df_raw, val_df_raw, test_df_raw = chronological_split(value_df)

    print("Dataset:", dataset_path)
    print("Total shape:", value_df.shape)
    print("Train shape:", train_df_raw.shape)
    print("Val shape:", val_df_raw.shape)
    print("Test shape:", test_df_raw.shape)

    # =========================
    # 2. Standardization
    # train statistics only
    # =========================
    mean = train_df_raw.mean()
    std = train_df_raw.std()

    # avoid division by zero
    std = std.replace(0, 1.0)

    value_df_std = (value_df - mean) / std

    train_df, val_df, test_df = chronological_split(value_df_std)

    test_start = len(train_df) + len(val_df)

    results = []

    selected_columns = value_df_std.columns.tolist()
    max_eval_windows = 50

    # =========================
    # 3. Zero-shot forecasting
    # =========================
    for hist_len in hist_lengths:
        print(f"\nEvaluating hist_len={hist_len}, pred_len={pred_len}")

        model = ChatTime(
            hist_len=hist_len,
            pred_len=pred_len,
            model_path=model_path
        )

        for col in tqdm(selected_columns, desc=f"hist_len={hist_len}"):
            series = value_df_std[col].to_numpy(dtype=np.float64)

            possible_starts = range(
                test_start,
                len(series) - pred_len + 1,
                pred_len
            )   
            if max_eval_windows is not None:
                possible_starts = possible_starts[:max_eval_windows]
            for start in possible_starts:
                hist_start = start - hist_len
                hist_end = start
                pred_start = start
                pred_end = start + pred_len

                if hist_start < 0:
                    continue

                hist_data = series[hist_start:hist_end]
                true_data = series[pred_start:pred_end]

                if np.isnan(hist_data).any() or np.isnan(true_data).any():
                    continue

                pred_data = model.predict(hist_data)

                score = mae(true_data, pred_data)

                results.append({
                    "dataset": dataset_name,
                    "column": col,
                    "hist_len": hist_len,
                    "pred_len": pred_len,
                    "start_index": start,
                    "mae": score
                })

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # =========================
    # 4. Save detailed results
    # =========================
    result_df = pd.DataFrame(results)

    detail_path = "outputs/chattime_etth2_mae_details.csv"
    result_df.to_csv(detail_path, index=False)

    # =========================
    # 5. Summary: paper-like result
    # =========================
    summary_df = (
        result_df
        .groupby(["dataset", "hist_len", "pred_len"], as_index=False)
        .agg(
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            n_samples=("mae", "count")
        )
    )

    summary_path = "outputs/chattime_etth2_mae_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    # =========================
    # 6. Summary by column
    # =========================
    summary_col_df = (
        result_df
        .groupby(["dataset", "column", "hist_len", "pred_len"], as_index=False)
        .agg(
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            n_samples=("mae", "count")
        )
    )

    summary_col_path = "outputs/chattime_etth2_mae_summary_by_column.csv"
    summary_col_df.to_csv(summary_col_path, index=False)

    print("\nPaper-like Summary")
    print(summary_df)

    print("\nSaved:")
    print(" -", detail_path)
    print(" -", summary_path)
    print(" -", summary_col_path)


if __name__ == "__main__":
    main()