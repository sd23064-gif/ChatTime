import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import warnings
warnings.filterwarnings("ignore")

import ast
import gc
import argparse
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import logging

logging.set_verbosity_error()

# 元の ChatTime
from model.model import ChatTime

# 先ほど作った Mamba 用クラス
from model.mamba_model import ChatTimeMamba


np.NaN = np.nan


def parse_series(x):
    if isinstance(x, np.ndarray):
        return x.astype(np.float64)

    if isinstance(x, list):
        return np.asarray(x, dtype=np.float64)

    if isinstance(x, str):
        return np.asarray(ast.literal_eval(x), dtype=np.float64)

    return np.asarray(x, dtype=np.float64)


def chronological_split(df, train_ratio=0.6, val_ratio=0.2):
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end].reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].reset_index(drop=True)
    test_df = df.iloc[val_end:].reset_index(drop=True)

    return train_df, val_df, test_df


def mae_raw(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    n = min(len(y_true), len(y_pred))
    y_true = y_true[:n]
    y_pred = y_pred[:n]

    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)

    if mask.sum() == 0:
        return np.nan

    return np.mean(np.abs(y_true[mask] - y_pred[mask]))


def mae_standardized(y_true, y_pred, mean, std):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    n = min(len(y_true), len(y_pred))
    y_true = y_true[:n]
    y_pred = y_pred[:n]

    y_true_std = (y_true - mean) / std
    y_pred_std = (y_pred - mean) / std

    mask = ~np.isnan(y_true_std) & ~np.isnan(y_pred_std)

    if mask.sum() == 0:
        return np.nan

    return np.mean(np.abs(y_true_std[mask] - y_pred_std[mask]))


def get_train_mean_std(train_df):
    all_pred_values = np.concatenate(train_df["Pred"].values)
    mean = np.mean(all_pred_values)
    std = np.std(all_pred_values)

    if std == 0 or np.isnan(std):
        std = 1.0

    return mean, std


def sign_flip_rate(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    n = min(len(y_true), len(y_pred))
    y_true = y_true[:n]
    y_pred = y_pred[:n]

    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)

    if mask.sum() == 0:
        return np.nan

    return np.mean(y_true[mask] * y_pred[mask] < 0)


def magnitude_mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    n = min(len(y_true), len(y_pred))
    y_true = y_true[:n]
    y_pred = y_pred[:n]

    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)

    if mask.sum() == 0:
        return np.nan

    return np.mean(np.abs(np.abs(y_true[mask]) - np.abs(y_pred[mask])))


class NaiveLastModel:
    def __init__(self, hist_len=None, pred_len=None):
        self.hist_len = hist_len
        self.pred_len = pred_len

    def predict(self, hist_data, context=None):
        return np.full(self.pred_len, hist_data[-1], dtype=np.float64)


def load_eval_model(config, hist_len, pred_len):
    model_type = config["type"]

    if model_type == "naive_last":
        return NaiveLastModel(
            hist_len=hist_len,
            pred_len=pred_len,
        )

    if model_type == "chattime":
        return ChatTime(
            hist_len=hist_len,
            pred_len=pred_len,
            model_path=config["model_path"],
            max_pred_len=config.get("max_pred_len", 16),
            num_samples=config.get("num_samples", 8),
            top_k=config.get("top_k", 100),
            top_p=config.get("top_p", 1.0),
            temperature=config.get("temperature", 1.0),
        )

    if model_type == "mamba":
        return ChatTimeMamba(
            base_model_path=config["base_model_path"],
            adapter_path=config.get("adapter_path", None),
            hist_len=hist_len,
            pred_len=pred_len,
            max_pred_len=config.get("max_pred_len", 16),
            num_samples=config.get("num_samples", 8),
            top_k=config.get("top_k", 100),
            top_p=config.get("top_p", 1.0),
            temperature=config.get("temperature", 1.0),
            torch_dtype=torch.float16,
            merge_lora=config.get("merge_lora", False),
        )

    raise ValueError(f"Unknown model type: {model_type}")


def cleanup_model(model):
    del model
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def safe_predict(model, hist_data, context, pred_len):
    try:
        pred = model.predict(hist_data, context=context)
        pred = np.asarray(pred, dtype=np.float64)

        if len(pred) < pred_len:
            pred = np.concatenate(
                [
                    pred,
                    np.full(pred_len - len(pred), np.nan),
                ]
            )

        return pred[:pred_len], None

    except Exception as e:
        return np.full(pred_len, np.nan), str(e)


def debug_one_sample(model, row, hist_len, pred_len, train_mean, train_std, model_name):
    hist_data = row["Hist"][-hist_len:]
    true_data = row["Pred"][:pred_len]
    text = row["Text"]

    pred_zero, err_zero = safe_predict(
        model=model,
        hist_data=hist_data,
        context=None,
        pred_len=pred_len,
    )

    pred_ctx, err_ctx = safe_predict(
        model=model,
        hist_data=hist_data,
        context=text,
        pred_len=pred_len,
    )

    print("\n========== Debug One Sample ==========")
    print("Model:", model_name)
    print("Text:")
    print(text)

    print("\nLength check")
    print("hist:", len(hist_data))
    print("true:", len(true_data))
    print("pred_zero:", len(pred_zero))
    print("pred_ctx:", len(pred_ctx))

    print("\nErrors")
    print("zero error:", err_zero)
    print("context error:", err_ctx)

    print("\nValue range")
    print("true min/max:", np.nanmin(true_data), np.nanmax(true_data))
    print("zero min/max:", np.nanmin(pred_zero), np.nanmax(pred_zero))
    print("ctx min/max:", np.nanmin(pred_ctx), np.nanmax(pred_ctx))

    print("\nRaw MAE")
    print("zero:", mae_raw(true_data, pred_zero))
    print("context:", mae_raw(true_data, pred_ctx))

    print("\nStandardized MAE")
    print("zero:", mae_standardized(true_data, pred_zero, train_mean, train_std))
    print("context:", mae_standardized(true_data, pred_ctx, train_mean, train_std))
    print("======================================\n")


def build_model_configs(args):
    configs = []

    if args.include_naive:
        configs.append({
            "name": "naive_last",
            "type": "naive_last",
        })

    if args.include_chattime:
        configs.append({
            "name": "chattime_7b",
            "type": "chattime",
            "model_path": args.chattime_model_path,
            "num_samples": args.num_samples,
            "max_pred_len": args.max_pred_len,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "temperature": args.temperature,
        })

    if args.mamba_base:
        configs.append({
            "name": "mamba_base",
            "type": "mamba",
            "base_model_path": args.mamba_base_model_path,
            "adapter_path": None,
            "num_samples": args.num_samples,
            "max_pred_len": args.max_pred_len,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "temperature": args.temperature,
        })

    if args.mamba_pretrain_adapter is not None:
        configs.append({
            "name": "mamba_pretrain",
            "type": "mamba",
            "base_model_path": args.mamba_base_model_path,
            "adapter_path": args.mamba_pretrain_adapter,
            "num_samples": args.num_samples,
            "max_pred_len": args.max_pred_len,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "temperature": args.temperature,
        })

    if args.mamba_finetune_adapter is not None:
        configs.append({
            "name": "mamba_finetuned",
            "type": "mamba",
            "base_model_path": args.mamba_base_model_path,
            "adapter_path": args.mamba_finetune_adapter,
            "num_samples": args.num_samples,
            "max_pred_len": args.max_pred_len,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "temperature": args.temperature,
        })

    return configs


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_name", type=str, default="PTF")
    parser.add_argument("--output_dir", type=str, default="outputs")

    parser.add_argument("--hist_lengths", type=str, default="48,72,96,120")
    parser.add_argument("--pred_len", type=int, default=24)
    parser.add_argument("--max_eval_windows", type=int, default=100)

    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--max_pred_len", type=int, default=16)
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)

    parser.add_argument("--include_naive", action="store_true")
    parser.add_argument("--include_chattime", action="store_true")
    parser.add_argument("--chattime_model_path", type=str, default="ChengsenWang/ChatTime-1-7B-Chat")

    parser.add_argument("--mamba_base", action="store_true")
    parser.add_argument("--mamba_base_model_path", type=str, default="state-spaces/mamba-370m-hf")
    parser.add_argument("--mamba_pretrain_adapter", type=str, default=None)
    parser.add_argument("--mamba_finetune_adapter", type=str, default=None)

    parser.add_argument("--save_predictions", action="store_true")
    parser.add_argument("--debug_first_sample", action="store_true")

    args = parser.parse_args()

    dataset_name = args.dataset_name
    hist_lengths = [int(x) for x in args.hist_lengths.split(",")]
    pred_len = args.pred_len

    os.makedirs(args.output_dir, exist_ok=True)

    model_configs = build_model_configs(args)

    if len(model_configs) == 0:
        raise ValueError(
            "評価するモデルが指定されていません。"
            "--mamba_finetune_adapter などを指定してください。"
        )

    print("Model configs:")
    for c in model_configs:
        print(c)

    # =========================
    # 1. Load CGTSF dataset
    # =========================
    ds = load_dataset(
        "ChengsenWang/CGTSF",
        data_files={"train": f"{dataset_name}/{dataset_name}.csv"}
    )

    df = ds["train"].to_pandas()

    print("Dataset:", dataset_name)
    print("Original shape:", df.shape)
    print("Columns:", df.columns.tolist())

    required_cols = ["Hist", "Pred", "Text"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column not found: {col}")

    if "Date" not in df.columns:
        df["Date"] = np.arange(len(df))

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
    # 4. Train statistics
    # =========================
    train_mean, train_std = get_train_mean_std(train_df)

    print("Train Pred mean:", train_mean)
    print("Train Pred std:", train_std)

    results = []

    # =========================
    # 5. Evaluation
    # =========================
    for config in model_configs:
        model_name = config["name"]

        print("\n" + "=" * 80)
        print(f"Evaluating model: {model_name}")
        print("=" * 80)

        for hist_len in hist_lengths:
            print(f"\nDataset={dataset_name}, model={model_name}, hist_len={hist_len}, pred_len={pred_len}")

            model = load_eval_model(
                config=config,
                hist_len=hist_len,
                pred_len=pred_len,
            )

            eval_df = test_df

            if args.max_eval_windows is not None and args.max_eval_windows > 0 and len(test_df) > args.max_eval_windows:
                indices = np.linspace(
                    0,
                    len(test_df) - 1,
                    args.max_eval_windows,
                    dtype=int,
                )
                eval_df = test_df.iloc[indices].reset_index(drop=True)
            else:
                eval_df = test_df.reset_index(drop=True)

            if args.debug_first_sample and len(eval_df) > 0:
                debug_one_sample(
                    model=model,
                    row=eval_df.iloc[0],
                    hist_len=hist_len,
                    pred_len=pred_len,
                    train_mean=train_mean,
                    train_std=train_std,
                    model_name=model_name,
                )

            for sample_id, row in tqdm(
                eval_df.iterrows(),
                total=len(eval_df),
                desc=f"{model_name}, hist_len={hist_len}"
            ):
                hist_full = row["Hist"]
                true_full = row["Pred"]
                text = row["Text"]
                date = row["Date"]

                hist_data = hist_full[-hist_len:]
                true_data = true_full[:pred_len]

                if len(hist_data) < hist_len or len(true_data) < pred_len:
                    continue

                if np.isnan(hist_data).any() or np.isnan(true_data).any():
                    continue

                # =========================
                # Contextなし
                # =========================
                pred_zero, err_zero = safe_predict(
                    model=model,
                    hist_data=hist_data,
                    context=None,
                    pred_len=pred_len,
                )

                # =========================
                # Contextあり
                # =========================
                pred_ctx, err_ctx = safe_predict(
                    model=model,
                    hist_data=hist_data,
                    context=text,
                    pred_len=pred_len,
                )

                raw_mae_zero = mae_raw(true_data, pred_zero)
                raw_mae_ctx = mae_raw(true_data, pred_ctx)

                std_mae_zero = mae_standardized(
                    true_data,
                    pred_zero,
                    train_mean,
                    train_std,
                )

                std_mae_ctx = mae_standardized(
                    true_data,
                    pred_ctx,
                    train_mean,
                    train_std,
                )

                sign_flip_zero = sign_flip_rate(true_data, pred_zero)
                sign_flip_ctx = sign_flip_rate(true_data, pred_ctx)

                mag_mae_zero = magnitude_mae(true_data, pred_zero)
                mag_mae_ctx = magnitude_mae(true_data, pred_ctx)

                row_result = {
                    "dataset": dataset_name,
                    "model": model_name,
                    "hist_len": hist_len,
                    "pred_len": pred_len,
                    "sample_id": sample_id,
                    "date": date,

                    "raw_mae_zero": raw_mae_zero,
                    "raw_mae_context": raw_mae_ctx,
                    "raw_context_gain": raw_mae_zero - raw_mae_ctx,

                    "std_mae_zero": std_mae_zero,
                    "std_mae_context": std_mae_ctx,
                    "std_context_gain": std_mae_zero - std_mae_ctx,

                    "sign_flip_zero": sign_flip_zero,
                    "sign_flip_context": sign_flip_ctx,

                    "magnitude_mae_zero": mag_mae_zero,
                    "magnitude_mae_context": mag_mae_ctx,

                    "error_zero": err_zero,
                    "error_context": err_ctx,

                    "text": text,
                }

                if args.save_predictions:
                    row_result["true"] = true_data.tolist()
                    row_result["pred_zero"] = pred_zero.tolist()
                    row_result["pred_context"] = pred_ctx.tolist()
                    row_result["hist"] = hist_data.tolist()

                results.append(row_result)

            cleanup_model(model)

            # 途中保存
            tmp_path = os.path.join(
                args.output_dir,
                f"cgtsf_{dataset_name.lower()}_context_details_tmp.csv"
            )
            pd.DataFrame(results).to_csv(tmp_path, index=False)

    # =========================
    # 6. Save details
    # =========================
    result_df = pd.DataFrame(results)

    detail_path = os.path.join(
        args.output_dir,
        f"cgtsf_{dataset_name.lower()}_context_model_comparison_details.csv"
    )
    result_df.to_csv(detail_path, index=False)

    if len(result_df) == 0:
        print("No valid results.")
        return

    # =========================
    # 7. Summary
    # =========================
    summary_df = (
        result_df
        .groupby(["dataset", "model", "hist_len", "pred_len"], as_index=False)
        .agg(
            raw_mae_zero_mean=("raw_mae_zero", "mean"),
            raw_mae_context_mean=("raw_mae_context", "mean"),
            raw_context_gain_mean=("raw_context_gain", "mean"),

            raw_mae_zero_std=("raw_mae_zero", "std"),
            raw_mae_context_std=("raw_mae_context", "std"),

            std_mae_zero_mean=("std_mae_zero", "mean"),
            std_mae_context_mean=("std_mae_context", "mean"),
            std_context_gain_mean=("std_context_gain", "mean"),

            std_mae_zero_std=("std_mae_zero", "std"),
            std_mae_context_std=("std_mae_context", "std"),

            sign_flip_zero_mean=("sign_flip_zero", "mean"),
            sign_flip_context_mean=("sign_flip_context", "mean"),

            magnitude_mae_zero_mean=("magnitude_mae_zero", "mean"),
            magnitude_mae_context_mean=("magnitude_mae_context", "mean"),

            n_samples=("raw_mae_zero", "count"),
        )
    )

    summary_path = os.path.join(
        args.output_dir,
        f"cgtsf_{dataset_name.lower()}_context_model_comparison_summary.csv"
    )
    summary_df.to_csv(summary_path, index=False)

    # =========================
    # 8. Overall summary
    # =========================
    overall_df = (
        result_df
        .groupby(["dataset", "model"], as_index=False)
        .agg(
            raw_mae_zero_mean=("raw_mae_zero", "mean"),
            raw_mae_context_mean=("raw_mae_context", "mean"),
            raw_context_gain_mean=("raw_context_gain", "mean"),

            std_mae_zero_mean=("std_mae_zero", "mean"),
            std_mae_context_mean=("std_mae_context", "mean"),
            std_context_gain_mean=("std_context_gain", "mean"),

            sign_flip_zero_mean=("sign_flip_zero", "mean"),
            sign_flip_context_mean=("sign_flip_context", "mean"),

            n_samples=("raw_mae_zero", "count"),
        )
        .sort_values("std_mae_context_mean")
    )

    overall_path = os.path.join(
        args.output_dir,
        f"cgtsf_{dataset_name.lower()}_context_model_comparison_overall.csv"
    )
    overall_df.to_csv(overall_path, index=False)

    print("\nSummary:")
    print(summary_df)

    print("\nOverall:")
    print(overall_df)

    print("\nSaved:")
    print(" -", detail_path)
    print(" -", summary_path)
    print(" -", overall_path)


if __name__ == "__main__":
    main()