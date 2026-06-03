import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import ast
import gc
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import logging

logging.set_verbosity_error()

from model.model import ChatTime
from utils.tools import Discretizer


if not hasattr(np, "NaN"):
    np.NaN = np.nan


def parse_series(x):
    if isinstance(x, np.ndarray):
        return x.astype(np.float64)
    if isinstance(x, list):
        return np.asarray(x, dtype=np.float64)
    if isinstance(x, str):
        return np.asarray(ast.literal_eval(x), dtype=np.float64)
    return np.asarray(x, dtype=np.float64)


def mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    n = min(len(y_true), len(y_pred))
    y_true = y_true[:n]
    y_pred = y_pred[:n]

    return np.nanmean(np.abs(y_true - y_pred))


def chronological_split(df, train_ratio=0.6, val_ratio=0.2):
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end].reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].reset_index(drop=True)
    test_df = df.iloc[val_end:].reset_index(drop=True)

    return train_df, val_df, test_df


def make_eval_subset(test_df, max_eval_windows=20):
    if max_eval_windows is None:
        return test_df.reset_index(drop=True)

    if len(test_df) <= max_eval_windows:
        return test_df.reset_index(drop=True)

    indices = np.linspace(
        0,
        len(test_df) - 1,
        max_eval_windows,
        dtype=int
    )

    return test_df.iloc[indices].reset_index(drop=True)


def get_train_std(train_df):
    values = np.concatenate(train_df["Pred"].values)
    std = np.nanstd(values)

    if std == 0 or np.isnan(std):
        std = 1.0

    return std


class VariableBinDiscretizer(Discretizer):
    """
    推論時だけbin数を変えるDiscretizer。

    effective_bins=10000 が元論文設定に近い。
    effective_binsを小さくすると粗い。
    effective_binsを大きくすると細かい。
    """

    def __init__(self, effective_bins=10000, low_limit=-1, high_limit=1):
        n_tokens = effective_bins + 2
        super().__init__(
            low_limit=low_limit,
            high_limit=high_limit,
            n_tokens=n_tokens
        )

        self.effective_bins = effective_bins


def main():
    dataset_name = "PTF"
    model_path = "ChengsenWang/ChatTime-1-7B-Chat"

    hist_lengths = [48, 72, 96, 120]
    pred_len = 24

    # 粗い → 標準 → 細かい
    bin_list = [2000, 5000, 10000, 20000]

    # まずは小さく確認。安定したら100やNoneに変更
    max_eval_windows = 100

    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    print("Loading CGTSF / PTF from Hugging Face...")

    ds = load_dataset(
        "ChengsenWang/CGTSF",
        data_files={"train": "PTF/PTF.csv"}
    )

    df = ds["train"].to_pandas()
    df.columns = df.columns.str.replace("\ufeff", "", regex=False).str.strip()

    df["Hist"] = df["Hist"].apply(parse_series)
    df["Pred"] = df["Pred"].apply(parse_series)

    train_df, val_df, test_df = chronological_split(df)
    train_std = get_train_std(train_df)

    eval_df = make_eval_subset(
        test_df,
        max_eval_windows=max_eval_windows
    )

    print("Dataset shape:", df.shape)
    print("Eval shape:", eval_df.shape)
    print("Train std:", train_std)

    results = []

    for bins in bin_list:
        print("\n" + "=" * 70)
        print(f"bins = {bins}")
        print("=" * 70)

        for hist_len in hist_lengths:
            print(f"\nPTF | hist_len={hist_len} | pred_len={pred_len} | bins={bins}")

            model = ChatTime(
                model_path=model_path,
                hist_len=hist_len,
                pred_len=pred_len,
                max_pred_len=pred_len,
                num_samples=8,
                top_k=100,
                top_p=1.0,
                temperature=1.0,
            )

            # ここで分割数だけ変更
            model.discretizer = VariableBinDiscretizer(
                effective_bins=bins
            )

            for pos, row in tqdm(eval_df.iterrows(), total=len(eval_df)):
                hist = row["Hist"][-hist_len:]
                true = row["Pred"][:pred_len]
                text = row["Text"]

                if len(hist) < hist_len or len(true) < pred_len:
                    continue

                if np.isnan(hist).any() or np.isnan(true).any():
                    continue

                try:
                    pred_zero = model.predict(hist, context=None)
                    pred_zero = np.asarray(pred_zero, dtype=np.float64)[:pred_len]
                    err_zero = ""
                except Exception as e:
                    pred_zero = np.full(pred_len, np.nan)
                    err_zero = str(e)

                try:
                    pred_ctx = model.predict(hist, context=text)
                    pred_ctx = np.asarray(pred_ctx, dtype=np.float64)[:pred_len]
                    err_ctx = ""
                except Exception as e:
                    pred_ctx = np.full(pred_len, np.nan)
                    err_ctx = str(e)

                raw_zero = mae(true, pred_zero)
                raw_ctx = mae(true, pred_ctx)

                std_zero = raw_zero / train_std
                std_ctx = raw_ctx / train_std

                results.append({
                    "dataset": dataset_name,
                    "bins": bins,
                    "hist_len": hist_len,
                    "pred_len": pred_len,
                    "position": pos,
                    "raw_mae_zero": raw_zero,
                    "raw_mae_context": raw_ctx,
                    "std_mae_zero": std_zero,
                    "std_mae_context": std_ctx,
                    "error_zero": err_zero,
                    "error_context": err_ctx,
                })

            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    result_df = pd.DataFrame(results)

    detail_path = os.path.join(
        output_dir,
        "ptf_bins_ablation_details.csv"
    )
    result_df.to_csv(detail_path, index=False)

    summary_df = (
        result_df
        .groupby(["dataset", "bins", "hist_len", "pred_len"], as_index=False)
        .agg(
            raw_zero=("raw_mae_zero", "mean"),
            raw_context=("raw_mae_context", "mean"),
            std_zero=("std_mae_zero", "mean"),
            std_context=("std_mae_context", "mean"),
            n_samples=("std_mae_context", "count"),
            n_error_zero=("error_zero", lambda x: (x != "").sum()),
            n_error_context=("error_context", lambda x: (x != "").sum()),
        )
    )

    summary_path = os.path.join(
        output_dir,
        "ptf_bins_ablation_summary.csv"
    )
    summary_df.to_csv(summary_path, index=False)

    baseline = summary_df[summary_df["bins"] == 10000][
        ["hist_len", "std_context"]
    ].rename(columns={
        "std_context": "baseline_10000"
    })

    compare_df = summary_df.merge(
        baseline,
        on="hist_len",
        how="left"
    )

    compare_df["change_vs_10000"] = (
        compare_df["std_context"] - compare_df["baseline_10000"]
    ) / compare_df["baseline_10000"]

    compare_path = os.path.join(
        output_dir,
        "ptf_bins_ablation_compare.csv"
    )
    compare_df.to_csv(compare_path, index=False)

    print("\n[*Scrapbox summary]")
    print("PTF bins ablation")
    print("metric: std_mae_context")
    print("baseline: bins=10000")
    print("")

    for hist_len in hist_lengths:
        sub = compare_df[compare_df["hist_len"] == hist_len].sort_values("bins")

        print(f"hist_len={hist_len}")
        for _, r in sub.iterrows():
            print(
                f" bins={int(r['bins'])}"
                f" std_ctx={r['std_context']:.4f}"
                f" change={r['change_vs_10000']*100:+.2f}%"
                f" n={int(r['n_samples'])}"
            )
        print("")

    print("Saved:")
    print(detail_path)
    print(summary_path)
    print(compare_path)


if __name__ == "__main__":
    main()