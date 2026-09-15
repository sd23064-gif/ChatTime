import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import warnings
warnings.filterwarnings("ignore")

import ast
import gc
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import logging

logging.set_verbosity_error()

from model.model import ChatTime


def parse_series(x):
    """
    Hist / Pred列を numpy配列に変換する関数。
    CSV内では "[1.0, 2.0, ...]" のような文字列なので、
    ast.literal_evalでlistに変換してからnp.array化する。
    """
    if isinstance(x, np.ndarray):
        return x.astype(np.float64)

    if isinstance(x, list):
        return np.asarray(x, dtype=np.float64)

    if isinstance(x, str):
        return np.asarray(ast.literal_eval(x), dtype=np.float64)

    return np.asarray(x, dtype=np.float64)


def chronological_split(df, train_ratio=0.6, val_ratio=0.2):
    """
    論文設定に合わせて時系列順に 6:2:2 で分割する。
    """
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end].reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].reset_index(drop=True)
    test_df = df.iloc[val_end:].reset_index(drop=True)

    return train_df, val_df, test_df


def mae_raw(y_true, y_pred):
    """
    元スケールでのMAE。
    PTFの場合、交通量そのもののスケールなので数十程度になることがある。
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    n = min(len(y_true), len(y_pred))
    y_true = y_true[:n]
    y_pred = y_pred[:n]

    return np.nanmean(np.abs(y_true - y_pred))


def mae_standardized(y_true, y_pred, mean, std):
    """
    標準化スケールでのMAE。
    論文Table 5の値に近づけて比較するために使用。
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    n = min(len(y_true), len(y_pred))
    y_true = y_true[:n]
    y_pred = y_pred[:n]

    y_true_std = (y_true - mean) / std
    y_pred_std = (y_pred - mean) / std

    return np.nanmean(np.abs(y_true_std - y_pred_std))


def get_train_mean_std(train_df):
    """
    train splitのPredから平均・標準偏差を計算。
    test情報を使わないようにする。
    """
    all_pred_values = np.concatenate(train_df["Pred"].values)
    mean = np.mean(all_pred_values)
    std = np.std(all_pred_values)

    if std == 0 or np.isnan(std):
        std = 1.0

    return mean, std


def debug_one_sample(model, row, hist_len, pred_len, train_mean, train_std):
    """
    1サンプルだけ中身を確認するための関数。
    予測値のスケールが壊れていないか確認できる。
    """
    hist_data = row["Hist"][-hist_len:]
    true_data = row["Pred"][:pred_len]
    text = row["Text"]

    pred_zero = model.predict(hist_data, context=None)
    pred_ctx = model.predict(hist_data, context=text)

    pred_zero = np.asarray(pred_zero, dtype=np.float64)[:pred_len]
    pred_ctx = np.asarray(pred_ctx, dtype=np.float64)[:pred_len]

    print("\n========== Debug One Sample ==========")
    print("Text:")
    print(text)

    print("\nLength check")
    print("hist:", len(hist_data))
    print("true:", len(true_data))
    print("pred_zero:", len(pred_zero))
    print("pred_ctx:", len(pred_ctx))

    print("\nValue range")
    print("true min/max:", np.min(true_data), np.max(true_data))
    print("zero min/max:", np.nanmin(pred_zero), np.nanmax(pred_zero))
    print("ctx min/max:", np.nanmin(pred_ctx), np.nanmax(pred_ctx))

    print("\nValues")
    print("true:", true_data)
    print("pred_zero:", pred_zero)
    print("pred_ctx:", pred_ctx)

    print("\nRaw MAE")
    print("zero:", mae_raw(true_data, pred_zero))
    print("context:", mae_raw(true_data, pred_ctx))

    print("\nStandardized MAE")
    print("zero:", mae_standardized(true_data, pred_zero, train_mean, train_std))
    print("context:", mae_standardized(true_data, pred_ctx, train_mean, train_std))
    print("======================================\n")


def main():
    # =========================
    # 0. Settings
    # =========================
    dataset_name = "PTF"
    model_path = "ChengsenWang/ChatTime-1-7B-Chat"

    # 論文Table 5のPTF設定
    hist_lengths = [48, 72, 96, 120]
    pred_len = 24

    # 最初は10で動作確認。
    # 論文再現に近づけるときは 100 → None と増やす。
    #max_eval_windows = 10
    max_eval_windows = 100
    #max_eval_windows = None

    os.makedirs("outputs", exist_ok=True)

    # =========================
    # 1. Load PTF only
    # =========================
    ds = load_dataset(
        "ChengsenWang/CGTSF",
        data_files={"train": f"{dataset_name}/{dataset_name}.csv"}
    )

    df = ds["train"].to_pandas()

    print("Dataset:", dataset_name)
    print("Original shape:", df.shape)
    print("Columns:", df.columns.tolist())

    # =========================
    # 2. Parse Hist / Pred
    # =========================
    df["Hist"] = df["Hist"].apply(parse_series)
    df["Pred"] = df["Pred"].apply(parse_series)

    # =========================
    # 3. Chronological split
    # =========================
    train_df, val_df, test_df = chronological_split(df)

    print("Train shape:", train_df.shape)
    print("Val shape:", val_df.shape)
    print("Test shape:", test_df.shape)

    # =========================
    # 4. Train statistics for standardized MAE
    # =========================
    train_mean, train_std = get_train_mean_std(train_df)

    print("Train Pred mean:", train_mean)
    print("Train Pred std:", train_std)

    results = []

    # =========================
    # 5. Evaluation
    # =========================
    for hist_len in hist_lengths:
        print(f"\nEvaluating {dataset_name}: hist_len={hist_len}, pred_len={pred_len}")

        model = ChatTime(
            hist_len=hist_len,
            pred_len=pred_len,
            model_path=model_path
        )

        eval_df = test_df

        if max_eval_windows is not None and len(test_df) > max_eval_windows:
            indices = np.linspace(0, len(test_df) - 1, max_eval_windows, dtype=int)
            eval_df = test_df.iloc[indices].reset_index(drop=True)
        else:
            eval_df = test_df.reset_index(drop=True)

        if max_eval_windows is not None:
            eval_df = eval_df.iloc[:max_eval_windows].reset_index(drop=True)

        # まず1サンプルだけ中身確認
        if len(eval_df) > 0:
            debug_one_sample(
                model=model,
                row=eval_df.iloc[0],
                hist_len=hist_len,
                pred_len=pred_len,
                train_mean=train_mean,
                train_std=train_std
            )

        for _, row in tqdm(eval_df.iterrows(), total=len(eval_df), desc=f"hist_len={hist_len}"):
            hist_full = row["Hist"]
            true_full = row["Pred"]
            text = row["Text"]
            date = row["Date"]

            # 重要：配列に変換済みのものをスライスする
            hist_data = hist_full[-hist_len:]
            true_data = true_full[:pred_len]

            if len(hist_data) < hist_len or len(true_data) < pred_len:
                continue

            if np.isnan(hist_data).any() or np.isnan(true_data).any():
                continue

            # =========================
            # Contextなし予測
            # =========================
            pred_zero = model.predict(hist_data, context=None)
            pred_zero = np.asarray(pred_zero, dtype=np.float64)[:pred_len]

            # =========================
            # Contextあり予測
            # =========================
            pred_ctx = model.predict(hist_data, context=text)
            pred_ctx = np.asarray(pred_ctx, dtype=np.float64)[:pred_len]

            # =========================
            # Metrics
            # =========================
            raw_mae_zero = mae_raw(true_data, pred_zero)
            raw_mae_ctx = mae_raw(true_data, pred_ctx)

            std_mae_zero = mae_standardized(
                true_data,
                pred_zero,
                train_mean,
                train_std
            )

            std_mae_ctx = mae_standardized(
                true_data,
                pred_ctx,
                train_mean,
                train_std
            )

            # 異常値確認
            if raw_mae_zero > 200 or raw_mae_ctx > 200:
                print("\n[Warning] Very large raw MAE detected")
                print("Date:", date)
                print("true min/max:", np.min(true_data), np.max(true_data))
                print("zero min/max:", np.nanmin(pred_zero), np.nanmax(pred_zero))
                print("ctx min/max:", np.nanmin(pred_ctx), np.nanmax(pred_ctx))
                print("raw_mae_zero:", raw_mae_zero)
                print("raw_mae_context:", raw_mae_ctx)

            results.append({
                "dataset": dataset_name,
                "hist_len": hist_len,
                "pred_len": pred_len,
                "date": date,

                "raw_mae_zero": raw_mae_zero,
                "raw_mae_context": raw_mae_ctx,

                "std_mae_zero": std_mae_zero,
                "std_mae_context": std_mae_ctx,

                "text": text
            })

        del model
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # =========================
    # 6. Save details
    # =========================
    result_df = pd.DataFrame(results)

    detail_path = f"outputs/chattime_{dataset_name.lower()}_cgtsf_details.csv"
    result_df.to_csv(detail_path, index=False)

    # =========================
    # 7. Summary
    # =========================
    summary_df = (
        result_df
        .groupby(["dataset", "hist_len", "pred_len"], as_index=False)
        .agg(
            raw_mae_zero_mean=("raw_mae_zero", "mean"),
            raw_mae_context_mean=("raw_mae_context", "mean"),
            raw_mae_zero_std=("raw_mae_zero", "std"),
            raw_mae_context_std=("raw_mae_context", "std"),

            std_mae_zero_mean=("std_mae_zero", "mean"),
            std_mae_context_mean=("std_mae_context", "mean"),
            std_mae_zero_std=("std_mae_zero", "std"),
            std_mae_context_std=("std_mae_context", "std"),

            n_samples=("raw_mae_zero", "count")
        )
    )

    summary_path = f"outputs/chattime_{dataset_name.lower()}_cgtsf_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\nSummary:")
    print(summary_df)

    print("\nSaved:")
    print(" -", detail_path)
    print(" -", summary_path)


if __name__ == "__main__":
    main()