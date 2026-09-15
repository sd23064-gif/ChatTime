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

# Mamba 用クラス
from model.mamba_model import ChatTimeMamba


# numpy 2.x では np.NaN が削除されているため、依存ライブラリ用に補完する
if not hasattr(np, "NaN"):
    np.NaN = np.nan


# =====================================================================
# データ処理
# =====================================================================
def parse_series(x):
    if x is None:
        return np.asarray([], dtype=np.float64)

    if isinstance(x, np.ndarray):
        return x.astype(np.float64)

    if isinstance(x, list):
        return np.asarray(x, dtype=np.float64)

    if isinstance(x, str):
        return np.asarray(ast.literal_eval(x), dtype=np.float64)

    return np.asarray(x, dtype=np.float64)


def chronological_split(df, train_ratio=0.6, val_ratio=0.2, group_column="Idx"):
    if group_column not in df.columns:
        df = df.sort_values("Date").reset_index(drop=True)
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        return (
            df.iloc[:train_end].reset_index(drop=True),
            df.iloc[train_end:val_end].reset_index(drop=True),
            df.iloc[val_end:].reset_index(drop=True),
        )

    train_parts = []
    val_parts = []
    test_parts = []

    for _, group in df.groupby(group_column, sort=False):
        group = group.sort_values("Date").reset_index(drop=True)
        n = len(group)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_parts.append(group.iloc[:train_end])
        val_parts.append(group.iloc[train_end:val_end])
        test_parts.append(group.iloc[val_end:])

    train_df = pd.concat(train_parts, ignore_index=True)
    val_df = pd.concat(val_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)

    return train_df, val_df, test_df


# =====================================================================
# 評価指標
# =====================================================================
def _align(y_true, y_pred):
    """長さを揃えて NaN マスクを返す共通処理。"""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    n = min(len(y_true), len(y_pred))
    y_true = y_true[:n]
    y_pred = y_pred[:n]

    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)

    return y_true, y_pred, mask


def mae_raw(y_true, y_pred):
    y_true, y_pred, mask = _align(y_true, y_pred)

    if mask.sum() == 0:
        return np.nan

    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))


def mae_standardized(y_true, y_pred, mean, std):
    """標準化スケールでの MAE。

    (y_true - mean)/std と (y_pred - mean)/std の差を取るため mean は相殺され、
    実質 raw_mae / std と等価。引数は互換性のため残している。
    """
    y_true, y_pred, mask = _align(y_true, y_pred)

    if mask.sum() == 0:
        return np.nan

    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])) / std)


def get_train_mean_std(train_df):
    series_list = [s for s in train_df["Pred"].values if len(s) > 0]

    if len(series_list) == 0:
        raise ValueError("train_df に有効な Pred 系列がありません。")

    all_pred_values = np.concatenate(series_list)
    all_pred_values = all_pred_values[~np.isnan(all_pred_values)]

    if len(all_pred_values) == 0:
        raise ValueError("train_df の Pred が全て NaN です。")

    mean = float(np.mean(all_pred_values))
    std = float(np.std(all_pred_values))

    if std == 0 or np.isnan(std):
        print("[warn] train std が 0 または NaN のため 1.0 に置き換えます。")
        std = 1.0

    return mean, std


def sign_flip_rate(y_true, y_pred):
    y_true, y_pred, mask = _align(y_true, y_pred)

    if mask.sum() == 0:
        return np.nan

    return float(np.mean(y_true[mask] * y_pred[mask] < 0))


def magnitude_mae(y_true, y_pred):
    y_true, y_pred, mask = _align(y_true, y_pred)

    if mask.sum() == 0:
        return np.nan

    return float(np.mean(np.abs(np.abs(y_true[mask]) - np.abs(y_pred[mask]))))


def n_valid_points(y_true, y_pred):
    _, _, mask = _align(y_true, y_pred)
    return int(mask.sum())


# =====================================================================
# モデル
# =====================================================================
class NaiveLastModel:
    """直近値をそのまま繰り返すベースライン（context は無視）。"""

    def __init__(self, hist_len=None, pred_len=None):
        self.hist_len = hist_len
        self.pred_len = pred_len

    def predict(self, hist_data, context=None):
        hist_data = np.asarray(hist_data, dtype=np.float64)

        if len(hist_data) == 0:
            return np.full(self.pred_len, np.nan, dtype=np.float64)

        return np.full(self.pred_len, hist_data[-1], dtype=np.float64)


def load_eval_model(config, hist_len, pred_len, torch_dtype=torch.float16):
    model_type = config["type"]

    if model_type == "naive_last":
        return NaiveLastModel(
            hist_len=hist_len,
            pred_len=pred_len,
        )

    if model_type == "llama":
        return ChatTime(
            hist_len=hist_len,
            pred_len=pred_len,
            base_model_path=config["base_model_path"],
            adapter_path=config.get("adapter_path", None),
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
            torch_dtype=torch_dtype,
            merge_lora=config.get("merge_lora", False),
        )

    raise ValueError(f"Unknown model type: {model_type}")


def try_set_horizon(model, hist_len, pred_len):
    """hist_len / pred_len だけを差し替えられるモデルなら再ロードを省略する。"""
    if not (hasattr(model, "hist_len") and hasattr(model, "pred_len")):
        return False

    model.hist_len = hist_len
    model.pred_len = pred_len
    return True


def cleanup_model(model):
    del model
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def reset_generation_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_predict(model, hist_data, context, pred_len, seed=None):
    """seed を指定すると生成前に乱数状態をリセットする。

    context あり / なしを同じ乱数状態から生成することで、
    差分がサンプリングノイズではなく context の効果になるようにする。
    """
    if seed is not None:
        reset_generation_seed(seed)

    try:
        pred = model.predict(hist_data, context=context)
        pred = np.asarray(pred, dtype=np.float64).reshape(-1)

        if len(pred) < pred_len:
            pred = np.concatenate(
                [
                    pred,
                    np.full(pred_len - len(pred), np.nan),
                ]
            )

        return pred[:pred_len], None

    except Exception as e:
        return np.full(pred_len, np.nan), f"{type(e).__name__}: {e}"


def safe_minmax(x):
    x = np.asarray(x, dtype=np.float64)

    if len(x) == 0 or np.all(np.isnan(x)):
        return np.nan, np.nan

    return float(np.nanmin(x)), float(np.nanmax(x))


def debug_one_sample(model, row, hist_len, pred_len, train_mean, train_std, model_name, seed):
    hist_data = row["Hist"][-hist_len:]
    true_data = row["Pred"][:pred_len]
    text = row["Text"]

    pred_zero, err_zero = safe_predict(
        model=model,
        hist_data=hist_data,
        context=None,
        pred_len=pred_len,
        seed=seed,
    )

    pred_ctx, err_ctx = safe_predict(
        model=model,
        hist_data=hist_data,
        context=text,
        pred_len=pred_len,
        seed=seed,
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
    print("true min/max:", *safe_minmax(true_data))
    print("zero min/max:", *safe_minmax(pred_zero))
    print("ctx  min/max:", *safe_minmax(pred_ctx))

    print("\nRaw MAE")
    print("zero:", mae_raw(true_data, pred_zero))
    print("context:", mae_raw(true_data, pred_ctx))

    print("\nStandardized MAE")
    print("zero:", mae_standardized(true_data, pred_zero, train_mean, train_std))
    print("context:", mae_standardized(true_data, pred_ctx, train_mean, train_std))
    print("======================================\n")


# =====================================================================
# 設定
# =====================================================================
def build_model_configs(args):
    configs = []

    common = {
        "num_samples": args.num_samples,
        "max_pred_len": args.max_pred_len,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "temperature": args.temperature,
    }

    if args.include_naive:
        configs.append({
            "name": "naive_last",
            "type": "naive_last",
        })

    if args.llama_base:
        configs.append({
            "name": "llama_base",
            "type": "llama",
            "base_model_path": args.llama_base_model_path,
            "adapter_path": None,
            **common,
        })

    if args.llama_pretrain_adapter is not None:
        configs.append({
            "name": "llama_pretrain",
            "type": "llama",
            "base_model_path": args.llama_base_model_path,
            "adapter_path": args.llama_pretrain_adapter,
            **common,
        })

    if args.llama_finetune_adapter is not None:
        configs.append({
            "name": "llama_finetuned",
            "type": "llama",
            "base_model_path": args.llama_base_model_path,
            "adapter_path": args.llama_finetune_adapter,
            **common,
        })

    if args.mamba_base:
        configs.append({
            "name": "mamba_base",
            "type": "mamba",
            "base_model_path": args.mamba_base_model_path,
            "adapter_path": None,
            "merge_lora": args.merge_lora,
            **common,
        })

    if args.mamba_pretrain_adapter is not None:
        configs.append({
            "name": "mamba_pretrain",
            "type": "mamba",
            "base_model_path": args.mamba_base_model_path,
            "adapter_path": args.mamba_pretrain_adapter,
            "merge_lora": args.merge_lora,
            **common,
        })

    if args.mamba_finetune_adapter is not None:
        configs.append({
            "name": "mamba_finetuned",
            "type": "mamba",
            "base_model_path": args.mamba_base_model_path,
            "adapter_path": args.mamba_finetune_adapter,
            "merge_lora": args.merge_lora,
            **common,
        })

    return configs


def build_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_name", type=str, default="PTF")
    parser.add_argument("--output_dir", type=str, default="outputs")

    parser.add_argument("--hist_lengths", type=str, default="48,72,96,120")
    parser.add_argument("--pred_len", type=int, default=24)
    parser.add_argument("--max_eval_windows", type=int, default=10)

    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--max_pred_len", type=int, default=16)
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)

    parser.add_argument("--include_naive", action="store_true")

    parser.add_argument("--llama_base", action="store_true")
    parser.add_argument("--llama_base_model_path", type=str, default="ChengsenWang/ChatTime-1-7B-Chat")
    parser.add_argument("--llama_pretrain_adapter", type=str, default=None)
    parser.add_argument("--llama_finetune_adapter", type=str, default=None)

    parser.add_argument("--mamba_base", action="store_true")
    parser.add_argument("--mamba_base_model_path", type=str, default="state-spaces/mamba-2.8b-hf")
    parser.add_argument("--mamba_pretrain_adapter", type=str, default=None)
    parser.add_argument("--mamba_finetune_adapter", type=str, default=None)
    parser.add_argument("--merge_lora", action="store_true")
    parser.add_argument("--torch_dtype", type=str, default="float16",
                        choices=["float16", "bfloat16", "float32"])

    parser.add_argument("--random_seed", type=int, default=3407)
    parser.add_argument(
        "--reuse_model_across_hist_len",
        action="store_true",
        help="hist_len ごとにモデルを再ロードせず、属性だけ差し替えて使い回す",
    )

    parser.add_argument("--save_predictions", action="store_true")
    parser.add_argument("--debug_first_sample", action="store_true")

    return parser


DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


# =====================================================================
# main
# =====================================================================
def main():
    args = build_arg_parser().parse_args()

    dataset_name = args.dataset_name
    hist_lengths = [int(x) for x in args.hist_lengths.split(",") if x.strip()]
    pred_len = args.pred_len
    torch_dtype = DTYPE_MAP[args.torch_dtype]

    if len(hist_lengths) == 0:
        raise ValueError("--hist_lengths が空です。")

    os.makedirs(args.output_dir, exist_ok=True)

    reset_generation_seed(args.random_seed)

    model_configs = build_model_configs(args)

    if len(model_configs) == 0:
        raise ValueError(
            "評価するモデルが指定されていません。"
            "--include_naive / --mamba_base / --mamba_finetune_adapter などを指定してください。"
        )

    print("Model configs:")
    for c in model_configs:
        print(c)

    # =========================
    # 1. Load CGTSF dataset
    # =========================
    ds = load_dataset(
        "ChengsenWang/CGTSF",
        data_files={"train": f"{dataset_name}/{dataset_name}.csv"},
        verification_mode="no_checks",
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
        print("[warn] Date 列がないため行番号を Date として使用します。")
        df["Date"] = np.arange(len(df))

    # =========================
    # 2. Parse Hist / Pred
    # =========================
    df["Hist"] = df["Hist"].apply(parse_series)
    df["Pred"] = df["Pred"].apply(parse_series)

    min_hist = int(df["Hist"].apply(len).min())
    min_pred = int(df["Pred"].apply(len).min())
    print("Min Hist length:", min_hist)
    print("Min Pred length:", min_pred)

    too_long = [h for h in hist_lengths if h > min_hist]
    if too_long:
        print(f"[warn] hist_len={too_long} は最短 Hist 長 {min_hist} を超えるため、"
              "多くのサンプルがスキップされます。")

    if pred_len > min_pred:
        print(f"[warn] pred_len={pred_len} は最短 Pred 長 {min_pred} を超えるため、"
              "多くのサンプルがスキップされます。")

    # =========================
    # 3. Chronological split
    # =========================
    train_df, val_df, test_df = chronological_split(df)

    print("Train shape:", train_df.shape)
    print("Val shape:", val_df.shape)
    print("Test shape:", test_df.shape)
    print("Train date range:", train_df["Date"].min(), train_df["Date"].max())
    print("Validation date range:", val_df["Date"].min(), val_df["Date"].max())
    print("Test date range:", test_df["Date"].min(), test_df["Date"].max())

    if "Idx" in df.columns:
        print("All series:", df["Idx"].nunique())
        print("Train series:", train_df["Idx"].nunique())
        print("Validation series:", val_df["Idx"].nunique())
        print("Test series:", test_df["Idx"].nunique())

    if len(test_df) == 0:
        raise ValueError("test_df が空です。分割条件を確認してください。")

    # =========================
    # 4. Train statistics
    # =========================
    train_mean, train_std = get_train_mean_std(train_df)

    print("Train Pred mean:", train_mean)
    print("Train Pred std:", train_std)

    # =========================
    # 5. 評価対象ウィンドウを先に固定（モデル間で同一サンプルを使う）
    # =========================
    test_df = test_df.reset_index(drop=True)

    if args.max_eval_windows is not None and 0 < args.max_eval_windows < len(test_df):
        eval_indices = np.linspace(0, len(test_df) - 1, args.max_eval_windows, dtype=int)
        eval_indices = np.unique(eval_indices)
    else:
        eval_indices = np.arange(len(test_df))

    eval_df = test_df.loc[eval_indices]
    print(f"Eval windows: {len(eval_df)} / {len(test_df)}")

    results = []

    # =========================
    # 6. Evaluation
    # =========================
    for config in model_configs:
        model_name = config["name"]

        print("\n" + "=" * 80)
        print(f"Evaluating model: {model_name}")
        print("=" * 80)

        model = None

        for hist_len in hist_lengths:
            print(f"\nDataset={dataset_name}, model={model_name}, "
                  f"hist_len={hist_len}, pred_len={pred_len}")

            if model is None:
                model = load_eval_model(
                    config=config,
                    hist_len=hist_len,
                    pred_len=pred_len,
                    torch_dtype=torch_dtype,
                )
            elif args.reuse_model_across_hist_len and try_set_horizon(model, hist_len, pred_len):
                pass
            else:
                cleanup_model(model)
                model = load_eval_model(
                    config=config,
                    hist_len=hist_len,
                    pred_len=pred_len,
                    torch_dtype=torch_dtype,
                )

            if args.debug_first_sample and len(eval_df) > 0:
                debug_one_sample(
                    model=model,
                    row=eval_df.iloc[0],
                    hist_len=hist_len,
                    pred_len=pred_len,
                    train_mean=train_mean,
                    train_std=train_std,
                    model_name=model_name,
                    seed=args.random_seed,
                )

            n_skipped = 0

            for sample_id, row in tqdm(
                eval_df.iterrows(),
                total=len(eval_df),
                desc=f"{model_name}, hist_len={hist_len}",
            ):
                hist_full = row["Hist"]
                true_full = row["Pred"]
                text = row["Text"]
                date = row["Date"]

                hist_data = hist_full[-hist_len:]
                true_data = true_full[:pred_len]

                if len(hist_data) < hist_len or len(true_data) < pred_len:
                    n_skipped += 1
                    continue

                if np.isnan(hist_data).any() or np.isnan(true_data).any():
                    n_skipped += 1
                    continue

                # context あり / なしで同じ乱数状態から生成する
                sample_seed = args.random_seed + int(sample_id) * 1_000 + hist_len

                pred_zero, err_zero = safe_predict(
                    model=model,
                    hist_data=hist_data,
                    context=None,
                    pred_len=pred_len,
                    seed=sample_seed,
                )

                pred_ctx, err_ctx = safe_predict(
                    model=model,
                    hist_data=hist_data,
                    context=text,
                    pred_len=pred_len,
                    seed=sample_seed,
                )

                row_result = {
                    "dataset": dataset_name,
                    "model": model_name,
                    "hist_len": hist_len,
                    "pred_len": pred_len,
                    "sample_id": int(sample_id),
                    "date": date,

                    "raw_mae_zero": mae_raw(true_data, pred_zero),
                    "raw_mae_context": mae_raw(true_data, pred_ctx),

                    "std_mae_zero": mae_standardized(true_data, pred_zero, train_mean, train_std),
                    "std_mae_context": mae_standardized(true_data, pred_ctx, train_mean, train_std),

                    "sign_flip_zero": sign_flip_rate(true_data, pred_zero),
                    "sign_flip_context": sign_flip_rate(true_data, pred_ctx),

                    "magnitude_mae_zero": magnitude_mae(true_data, pred_zero),
                    "magnitude_mae_context": magnitude_mae(true_data, pred_ctx),

                    "n_valid_zero": n_valid_points(true_data, pred_zero),
                    "n_valid_context": n_valid_points(true_data, pred_ctx),

                    "failed_zero": int(err_zero is not None),
                    "failed_context": int(err_ctx is not None),

                    "error_zero": err_zero,
                    "error_context": err_ctx,

                    "text": text,
                }

                # gain は MAE 算出後に差分を取る（NaN 伝播をそのまま残す）
                row_result["raw_context_gain"] = (
                    row_result["raw_mae_zero"] - row_result["raw_mae_context"]
                )
                row_result["std_context_gain"] = (
                    row_result["std_mae_zero"] - row_result["std_mae_context"]
                )

                if args.save_predictions:
                    row_result["true"] = true_data.tolist()
                    row_result["pred_zero"] = pred_zero.tolist()
                    row_result["pred_context"] = pred_ctx.tolist()
                    row_result["hist"] = hist_data.tolist()

                results.append(row_result)

            if n_skipped > 0:
                print(f"[warn] hist_len={hist_len}: {n_skipped} 件を長さ不足 / NaN でスキップしました。")

            # 途中保存
            tmp_path = os.path.join(
                args.output_dir,
                f"cgtsf_{dataset_name.lower()}_context_details_tmp.csv",
            )
            pd.DataFrame(results).to_csv(tmp_path, index=False)

        if model is not None:
            cleanup_model(model)
            model = None

    # =========================
    # 7. Save details
    # =========================
    result_df = pd.DataFrame(results)

    if len(result_df) == 0:
        print("No valid results. hist_len / pred_len とデータ長を確認してください。")
        return

    detail_path = os.path.join(
        args.output_dir,
        f"cgtsf_{dataset_name.lower()}_context_model_comparison_details.csv",
    )
    result_df.to_csv(detail_path, index=False)

    # =========================
    # 8. Summary
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

            n_failed_zero=("failed_zero", "sum"),
            n_failed_context=("failed_context", "sum"),

            n_samples=("sample_id", "count"),
            n_scored_zero=("raw_mae_zero", "count"),
            n_scored_context=("raw_mae_context", "count"),
        )
    )

    summary_path = os.path.join(
        args.output_dir,
        f"cgtsf_{dataset_name.lower()}_context_model_comparison_summary.csv",
    )
    summary_df.to_csv(summary_path, index=False)

    # =========================
    # 9. Overall summary
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

            n_failed_zero=("failed_zero", "sum"),
            n_failed_context=("failed_context", "sum"),

            n_samples=("sample_id", "count"),
        )
        .sort_values("std_mae_context_mean")
    )

    overall_path = os.path.join(
        args.output_dir,
        f"cgtsf_{dataset_name.lower()}_context_model_comparison_overall.csv",
    )
    overall_df.to_csv(overall_path, index=False)

    print("\nSummary:")
    print(summary_df.to_string(index=False))

    print("\nOverall:")
    print(overall_df.to_string(index=False))

    print("\nSaved:")
    print(" -", detail_path)
    print(" -", summary_path)
    print(" -", overall_path)


if __name__ == "__main__":
    main()