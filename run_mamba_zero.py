import os
import gc
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import argparse
import json
from model.model import ChatTime
from model.mamba_model import ChatTimeMamba
import time

np.NaN = np.nan

def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)

    if mask.sum() == 0:
        return np.nan

    return np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2))


def smape(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)

    if mask.sum() == 0:
        return np.nan

    denom = np.abs(y_true[mask]) + np.abs(y_pred[mask]) + eps
    return np.mean(2.0 * np.abs(y_pred[mask] - y_true[mask]) / denom)

def mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)

    if mask.sum() == 0:
        return np.nan

    return np.mean(np.abs(y_true[mask] - y_pred[mask]))


def chronological_split(df, train_ratio=0.6, val_ratio=0.2):
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    return train_df, val_df, test_df


class NaiveLastModel:
    def __init__(self, hist_len=None, pred_len=None):
        self.hist_len = hist_len
        self.pred_len = pred_len

    def predict(self, hist_data):
        last_value = hist_data[-1]
        return np.full(self.pred_len, last_value, dtype=np.float64)


def load_eval_model(config, hist_len, pred_len):
    model_type = config["type"]

    if model_type == "naive_last":
        return NaiveLastModel(
            hist_len=hist_len,
            pred_len=pred_len,
        )

    elif model_type == "chattime":
        return ChatTime(
            hist_len=hist_len,
            pred_len=pred_len,
            model_path=config["model_path"],
        )

    elif model_type == "mamba":
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

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def cleanup_model(model):
    del model
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/mamba_2.8b_analyze",
        help="評価結果を保存するディレクトリ",
    )

    parser.add_argument(
        "--dataset_path",
        type=str,
        default="./dataset/ETTh2.csv",
        help="評価に使うCSVデータセット",
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        default="ETTh2",
        help="結果ファイルに記録するデータセット名",
    )

    parser.add_argument(
        "--max_eval_windows",
        type=int,
        default=10,
        help="各系列ごとに評価する最大window数。-1なら全window",
    )

    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    dataset_path = args.dataset_path
    dataset_name = args.dataset_name

    pred_len = 24
    hist_lengths = [48]

    max_eval_windows = args.max_eval_windows
    if max_eval_windows < 0:
        max_eval_windows = None


    # =========================
    # Model configs
    # =========================
    model_configs = [
        {
            "name": "naive_last",
            "type": "naive_last",
        },
        {
            "name": "mamba_pretrain_2.8b",
            "type": "mamba",
            "base_model_path": "state-spaces/mamba-2.8b-hf",
            "adapter_path": "outputs/logs/mamba-2.8b-pretrain-r/checkpoint-4000",
            "max_pred_len": 24,
            "num_samples": 8,
            "merge_lora": False,
        }
    ]
    # =========================
    # 1. Load dataset
    # =========================
    raw_df = pd.read_csv(dataset_path)

    if "date" in raw_df.columns:
        value_df = raw_df.drop(columns=["date"]).apply(pd.to_numeric, errors="coerce")
    else:
        value_df = raw_df.apply(pd.to_numeric, errors="coerce")

    train_df_raw, val_df_raw, test_df_raw = chronological_split(value_df)

    print("Dataset:", dataset_path)
    print("Total shape:", value_df.shape)
    print("Train shape:", train_df_raw.shape)
    print("Val shape:", val_df_raw.shape)
    print("Test shape:", test_df_raw.shape)

    # =========================
    # 2. Standardization
    # =========================
    mean = train_df_raw.mean()
    std = train_df_raw.std()
    std = std.replace(0, 1.0)

    value_df_std = (value_df - mean) / std

    train_df, val_df, test_df = chronological_split(value_df_std)

    test_start = len(train_df) + len(val_df)

    selected_columns = value_df_std.columns.tolist()

    all_results = []

    # =========================
    # 3. Evaluate each model
    # =========================
    for config in model_configs:
        model_name = config["name"]

        print("\n" + "=" * 80)
        print(f"Evaluating model: {model_name}")
        print("=" * 80)

        for hist_len in hist_lengths:
            print(f"\nModel={model_name}, hist_len={hist_len}, pred_len={pred_len}")

            load_start = time.perf_counter()

            model = load_eval_model(
                config=config,
                hist_len=hist_len,
                pred_len=pred_len,
            )

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            load_seconds = time.perf_counter() - load_start
            print(f"Model load time: {load_seconds:.2f} seconds")

            for col in tqdm(selected_columns, desc=f"{model_name}, hist_len={hist_len}"):
                series = value_df_std[col].to_numpy(dtype=np.float64)

                possible_starts = range(
                    test_start,
                    len(series) - pred_len + 1,
                    pred_len,
                )

                possible_starts = list(possible_starts)

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

                    try:
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()

                        predict_start = time.perf_counter()

                        pred_data = model.predict(hist_data)

                        if torch.cuda.is_available():
                            torch.cuda.synchronize()

                        predict_seconds = time.perf_counter() - predict_start
                        pred_data = np.asarray(pred_data, dtype=np.float64)

                        print(
                            f"Prediction time: model={model_name}, col={col}, "
                            f"hist_len={hist_len}, start={start}, "
                            f"seconds={predict_seconds:.3f}"
                        )

                        if len(pred_data) != len(true_data):
                            print(
                                f"Length mismatch: model={model_name}, col={col}, "
                                f"hist_len={hist_len}, start={start}, "
                                f"pred={len(pred_data)}, true={len(true_data)}"
                            )
                            continue

                        mae_std = mae(true_data, pred_data)
                        rmse_std = rmse(true_data, pred_data)
                        smape_std = smape(true_data, pred_data)

                        pred_raw = pred_data * std[col] + mean[col]
                        true_raw = true_data * std[col] + mean[col]

                        mae_raw = mae(true_raw, pred_raw)
                        rmse_raw = rmse(true_raw, pred_raw)
                        smape_raw = smape(true_raw, pred_raw)

                        if np.isnan(mae_std):
                            continue

                        all_results.append({
                            "dataset": dataset_name,
                            "model": model_name,
                            "column": col,
                            "hist_len": hist_len,
                            "pred_len": pred_len,
                            "start_index": start,
                            "num_samples": config.get("num_samples", 1),
                            "inference_seconds": predict_seconds,
                            "seconds_per_predicted_point": predict_seconds / pred_len,
                            "mae_std": mae_std,
                            "rmse_std": rmse_std,
                            "smape_std": smape_std,
                            "mae_raw": mae_raw,
                            "rmse_raw": rmse_raw,
                            "smape_raw": smape_raw,
                        })
                    except Exception as e:
                        print(
                            f"Prediction failed: model={model_name}, col={col}, "
                            f"hist_len={hist_len}, start={start}, error={e}"
                        )
                        continue

            cleanup_model(model)

            # 途中保存
            tmp_df = pd.DataFrame(all_results)
            tmp_path = os.path.join(output_dir, "model_comparison_etth2_mae_details_tmp.csv")
            tmp_df.to_csv(tmp_path, index=False)
    # =========================
    # 4. Save detailed results
    # =========================
    result_df = pd.DataFrame(all_results)

    detail_path = os.path.join(output_dir, "model_comparison_etth2_mae_details.csv")
    result_df.to_csv(detail_path, index=False)

    if len(result_df) == 0:
        print("No valid results.")
        return

    # =========================
    # 5. Summary by model
    # =========================
    summary_df = (
        result_df
        .groupby(["dataset", "model", "hist_len", "pred_len"], as_index=False)
        .agg(
            mae_std_mean=("mae_std", "mean"),
            mae_std_std=("mae_std", "std"),
            rmse_std_mean=("rmse_std", "mean"),
            smape_std_mean=("smape_std", "mean"),

            mae_raw_mean=("mae_raw", "mean"),
            mae_raw_std=("mae_raw", "std"),
            rmse_raw_mean=("rmse_raw", "mean"),
            smape_raw_mean=("smape_raw", "mean"),

            n_samples=("mae_std", "count"),
        )
    )

    summary_path = os.path.join(output_dir, "model_comparison_etth2_mae_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    # =========================
    # 6. Summary by model and column
    # =========================
    summary_col_df = (
        result_df
        .groupby(["dataset", "model", "column", "hist_len", "pred_len"], as_index=False)
        .agg(
            mae_std_mean=("mae_std", "mean"),
            mae_std_std=("mae_std", "std"),
            rmse_std_mean=("rmse_std", "mean"),
            smape_std_mean=("smape_std", "mean"),

            mae_raw_mean=("mae_raw", "mean"),
            mae_raw_std=("mae_raw", "std"),
            rmse_raw_mean=("rmse_raw", "mean"),
            smape_raw_mean=("smape_raw", "mean"),

            n_samples=("mae_std", "count"),
        )
    )

    summary_col_path = os.path.join(output_dir, "model_comparison_etth2_mae_summary_by_column.csv")
    summary_col_df.to_csv(summary_col_path, index=False)

    # =========================
    # 7. Overall ranking
    # =========================
    ranking_df = (
        result_df
        .groupby(["dataset", "model"], as_index=False)
        .agg(
            mae_std_mean=("mae_std", "mean"),
            mae_std_std=("mae_std", "std"),
            rmse_std_mean=("rmse_std", "mean"),
            smape_std_mean=("smape_std", "mean"),

            mae_raw_mean=("mae_raw", "mean"),
            mae_raw_std=("mae_raw", "std"),

            n_samples=("mae_std", "count"),
        )
        .sort_values("mae_std_mean")
    )
    baseline_name = "mamba_normal_finetuned"

    if baseline_name in ranking_df["model"].values:
        baseline_mae = ranking_df.loc[
            ranking_df["model"] == baseline_name,
            "mae_std_mean"
        ].iloc[0]

        ranking_df["mae_std_improvement_vs_normal_mamba_pct"] = (
            (baseline_mae - ranking_df["mae_std_mean"]) / baseline_mae * 100.0
        )
    else:
        ranking_df["mae_std_improvement_vs_normal_mamba_pct"] = np.nan
    ranking_path = os.path.join(output_dir, "model_comparison_etth2_mae_ranking.csv")
    ranking_df.to_csv(ranking_path, index=False)
    import json

    eval_config = {
        "dataset_path": dataset_path,
        "dataset_name": dataset_name,
        "pred_len": pred_len,
        "hist_lengths": hist_lengths,
        "max_eval_windows": max_eval_windows,
        "models": model_configs,
    }

    eval_config_path = os.path.join(output_dir, "model_comparison_etth2_eval_config.json")

    with open(eval_config_path, "w", encoding="utf-8") as f:
        json.dump(eval_config, f, ensure_ascii=False, indent=2)
        print("\nSummary")
    print(summary_df)

    print("\nRanking")
    print(ranking_df)

    print("\nSaved:")
    print(" -", detail_path)
    print(" -", summary_path)
    print(" -", summary_col_path)
    print(" -", ranking_path)
    print(" -", eval_config_path)


if __name__ == "__main__":
    main()