import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model.model import ChatTime


def calc_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    err = y_pred - y_true

    mae = np.mean(np.abs(err))
    rmse = np.sqrt(np.mean(err ** 2))

    amp = np.max(y_true) - np.min(y_true)
    nrmse = rmse / (amp + 1e-8)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1.0 - ss_res / (ss_tot + 1e-8)

    if np.std(y_true) < 1e-8 or np.std(y_pred) < 1e-8:
        corr = np.nan
    else:
        corr = np.corrcoef(y_true, y_pred)[0, 1]

    return {
        "mae": mae,
        "rmse": rmse,
        "nrmse": nrmse,
        "r2": r2,
        "corr": corr,
    }


def read_ecg_record(root, filename_lr):
    record_path = Path(root) / filename_lr
    signal, fields = wfdb.rdsamp(str(record_path))
    return signal, fields["sig_name"]


def get_lead_index(lead_names, target_lead):
    normalized = [x.upper() for x in lead_names]
    target = target_lead.upper()

    if target not in normalized:
        raise ValueError(
            f"Lead {target_lead} が見つかりません。利用可能な誘導: {lead_names}"
        )

    return normalized.index(target)


def make_baseline_predictions(hist, pred_len):
    hist = np.asarray(hist, dtype=np.float32)

    # Baseline 1: 最後の値をそのまま未来に伸ばす
    last_value_pred = np.full(pred_len, hist[-1], dtype=np.float32)

    # Baseline 2: 直前 pred_len サンプルをそのまま繰り返す
    if len(hist) >= pred_len:
        repeat_pred = hist[-pred_len:].astype(np.float32)
    else:
        repeat_pred = np.resize(hist, pred_len).astype(np.float32)

    return {
        "last_value": last_value_pred,
        "repeat": repeat_pred,
    }


def save_window_plot(
    ecg_id,
    lead,
    start,
    hist,
    y_true,
    pred_dict,
    hist_len,
    pred_len,
    out_dir,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    hist_x = np.arange(start, start + hist_len)
    pred_x = np.arange(start + hist_len, start + hist_len + pred_len)

    plt.figure(figsize=(12, 4))

    plt.plot(hist_x, hist, label="history", color="black")
    plt.plot(pred_x, y_true, label="true future", color="blue")

    colors = {
        "chattime": "orange",
        "last_value": "green",
        "repeat": "purple",
    }

    for name, y_pred in pred_dict.items():
        plt.plot(
            pred_x,
            y_pred,
            label=name,
            color=colors.get(name, None),
        )

    plt.axvline(start + hist_len - 1, color="red", linestyle="--", label="prediction start")

    plt.title(f"ECG prediction: ecg_id={ecg_id}, lead={lead}, start={start}")
    plt.xlabel("sample")
    plt.ylabel("amplitude")
    plt.legend()
    plt.tight_layout()

    filename = f"ecg_{ecg_id}_lead_{lead}_start_{start}_full.png"
    plt.savefig(out_dir / filename, dpi=200)
    plt.close()

    # 予測区間だけの拡大図
    plt.figure(figsize=(10, 4))
    future_x = np.arange(pred_len)

    plt.plot(future_x, y_true, label="true future", color="blue")

    for name, y_pred in pred_dict.items():
        plt.plot(
            future_x,
            y_pred,
            label=name,
            color=colors.get(name, None),
        )

    plt.title(f"Future only: ecg_id={ecg_id}, lead={lead}, start={start}")
    plt.xlabel("future sample")
    plt.ylabel("amplitude")
    plt.legend()
    plt.tight_layout()

    filename = f"ecg_{ecg_id}_lead_{lead}_start_{start}_future_only.png"
    plt.savefig(out_dir / filename, dpi=200)
    plt.close()

    # CSV保存
    df_plot = pd.DataFrame({
        "future_index": np.arange(pred_len),
        "y_true": y_true,
    })

    for name, y_pred in pred_dict.items():
        df_plot[f"y_pred_{name}"] = y_pred
        df_plot[f"error_{name}"] = y_pred - y_true

    csv_name = f"ecg_{ecg_id}_lead_{lead}_start_{start}_prediction.csv"
    df_plot.to_csv(out_dir / csv_name, index=False)

def build_ecg_context(hist, lead="II", fs=100):
    hist = np.asarray(hist, dtype=np.float32)

    amp_min = float(np.min(hist))
    amp_max = float(np.max(hist))
    amp_range = amp_max - amp_min
    mean_val = float(np.mean(hist))
    std_val = float(np.std(hist))

    # 簡易ピーク検出：history区間だけを使う
    threshold = mean_val + 0.5 * std_val
    peaks = []
    for i in range(1, len(hist) - 1):
        if hist[i] > hist[i - 1] and hist[i] > hist[i + 1] and hist[i] > threshold:
            peaks.append(i)

    if len(peaks) >= 2:
        rr_intervals = np.diff(peaks)
        mean_rr = float(np.mean(rr_intervals))
        estimated_hr = 60.0 * fs / mean_rr
        rhythm_text = (
            f"The history contains approximately {len(peaks)} prominent positive peaks. "
            f"The average interval between detected peaks is about {mean_rr:.1f} samples, "
            f"corresponding to an approximate heart rate of {estimated_hr:.1f} beats per minute. "
        )
    else:
        rhythm_text = (
            f"The history contains approximately {len(peaks)} prominent positive peaks. "
        )

    context = (
        f"This time series is an ECG waveform sampled at {fs} Hz from lead {lead}. "
        f"The history length is {len(hist)} samples. "
        f"The amplitude range in the history is {amp_range:.4f}, "
        f"with minimum {amp_min:.4f}, maximum {amp_max:.4f}, "
        f"mean {mean_val:.4f}, and standard deviation {std_val:.4f}. "
        f"{rhythm_text}"
        "Abrupt sharp upward and downward deflections are expected ECG morphology, "
        "such as QRS complexes and R peaks. "
        "These sharp changes should not be treated as noise or outliers. "
        "Do not smooth out clinically meaningful sharp peaks. "
        "Predict the next ECG waveform by preserving the recent rhythm, amplitude scale, "
        "and sharp ECG morphology."
    )

    return context

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        type=str,
        default="/workspace/data/ptb-xl/1.0.3",
        help="PTB-XL v1.0.3 のルートディレクトリ",
    )

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="ChatTimeモデル、またはローカルsnapshotパス",
    )

    parser.add_argument("--hist_len", type=int, default=200)
    parser.add_argument("--pred_len", type=int, default=10)
    parser.add_argument("--lead", type=str, default="II")
    parser.add_argument("--fold", type=int, default=10)

    parser.add_argument(
        "--limit_records",
        type=int,
        default=10,
        help="評価するECGレコード数。全件なら -1",
    )

    parser.add_argument(
        "--window_stride",
        type=int,
        default=50,
        help="window開始位置の間隔。100Hzなので50なら0.5秒ごと",
    )

    parser.add_argument(
        "--max_windows_per_record",
        type=int,
        default=5,
        help="1レコードあたり最大何window評価するか。全windowなら -1",
    )

    parser.add_argument(
        "--num_plots",
        type=int,
        default=10,
        help="保存する可視化サンプル数",
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default="/workspace/results/chattime_windows_baseline",
        help="結果保存ディレクトリ",
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="ChatTimeの生成サンプル数。OOM対策なら1推奨",
    )

    parser.add_argument(
        "--max_pred_len",
        type=int,
        default=10,
        help="ChatTime内部の1回あたり最大予測長。pred_len=10なら10推奨",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="The sequence is an ECG signal. It may include periodic cardiac waveform patterns, sharp peaks, and sudden amplitude changes."
    )
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    plot_dir = out_dir / "plots"

    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    db_path = root / "ptbxl_database.csv"
    df = pd.read_csv(db_path)
    
    if "strat_fold" in df.columns:
        df = df[df["strat_fold"] == args.fold].copy()

    if args.limit_records > 0:
        df = df.head(args.limit_records).copy()

    print("=== Evaluation setting ===")
    print(f"records              : {len(df)}")
    print(f"root                 : {root}")
    print(f"model_path           : {args.model_path}")
    print(f"lead                 : {args.lead}")
    print(f"hist_len             : {args.hist_len}")
    print(f"pred_len             : {args.pred_len}")
    print(f"fold                 : {args.fold}")
    print(f"window_stride        : {args.window_stride}")
    print(f"max_windows_per_record: {args.max_windows_per_record}")
    print(f"out_dir              : {out_dir}")
    print("normalize            : False")
    print("==========================")

    # ChatTimeモデル
    # 公式demoに合わせて外側正規化は行わない
    model = ChatTime(
        hist_len=args.hist_len,
        pred_len=args.pred_len,
        model_path=args.model_path,
        num_samples=args.num_samples,
        max_pred_len=args.max_pred_len,
    )

    rows = []
    plot_count = 0

    need_len = args.hist_len + args.pred_len

    for _, row in tqdm(df.iterrows(), total=len(df)):
        ecg_id = row["ecg_id"]
        filename_lr = row["filename_lr"]

        try:
            signal, lead_names = read_ecg_record(root, filename_lr)
            lead_idx = get_lead_index(lead_names, args.lead)

            x = signal[:, lead_idx].astype(np.float32)

            if len(x) < need_len:
                rows.append({
                    "ecg_id": ecg_id,
                    "filename_lr": filename_lr,
                    "lead": args.lead,
                    "start": np.nan,
                    "model": "none",
                    "mae": np.nan,
                    "rmse": np.nan,
                    "nrmse": np.nan,
                    "r2": np.nan,
                    "corr": np.nan,
                    "error": f"signal too short: len={len(x)}, need={need_len}",
                })
                continue

            possible_starts = list(range(0, len(x) - need_len + 1, args.window_stride))

            if args.max_windows_per_record > 0:
                possible_starts = possible_starts[:args.max_windows_per_record]

            for start in possible_starts:
                hist = x[start : start + args.hist_len]
                y_true = x[start + args.hist_len : start + args.hist_len + args.pred_len]

                pred_dict = {}

                # Baselines
                baseline_preds = make_baseline_predictions(hist, args.pred_len)
                #pred_dict.update(baseline_preds)
                context = "This is a 100 Hz lead II ECG waveform.Sharp rapid rises and falls are meaningful ECG components, not outliers.Preserve QRS-like sharp peaks and the recent rhythm when predicting the next waveform."
                # ChatTime prediction
                try:
                    y_pred_chattime = model.predict(hist, context=context)
                    y_pred_chattime = np.asarray(y_pred_chattime, dtype=np.float32).reshape(-1)

                    if len(y_pred_chattime) > args.pred_len:
                        y_pred_chattime = y_pred_chattime[: args.pred_len]
                    elif len(y_pred_chattime) < args.pred_len:
                        pad_len = args.pred_len - len(y_pred_chattime)
                        if len(y_pred_chattime) == 0:
                            y_pred_chattime = np.full(args.pred_len, np.nan, dtype=np.float32)
                        else:
                            y_pred_chattime = np.pad(y_pred_chattime, (0, pad_len), mode="edge")

                    pred_dict["chattime"] = y_pred_chattime

                except Exception as e:
                    pred_dict["chattime"] = np.full(args.pred_len, np.nan, dtype=np.float32)
                    rows.append({
                        "ecg_id": ecg_id,
                        "filename_lr": filename_lr,
                        "lead": args.lead,
                        "start": start,
                        "model": "chattime",
                        "mae": np.nan,
                        "rmse": np.nan,
                        "nrmse": np.nan,
                        "r2": np.nan,
                        "corr": np.nan,
                        "error": str(e),
                    })

                # Metrics for each model
                for model_name, y_pred in pred_dict.items():
                    if np.isnan(y_pred).all():
                        continue

                    m = calc_metrics(y_true, y_pred)

                    rows.append({
                        "ecg_id": ecg_id,
                        "filename_lr": filename_lr,
                        "lead": args.lead,
                        "start": start,
                        "model": model_name,
                        **m,
                        "error": "",
                    })

                # Plot
                if plot_count < args.num_plots:
                    save_window_plot(
                        ecg_id=ecg_id,
                        lead=args.lead,
                        start=start,
                        hist=hist,
                        y_true=y_true,
                        pred_dict=pred_dict,
                        hist_len=args.hist_len,
                        pred_len=args.pred_len,
                        out_dir=plot_dir,
                    )
                    plot_count += 1

        except Exception as e:
            rows.append({
                "ecg_id": ecg_id,
                "filename_lr": filename_lr,
                "lead": args.lead,
                "start": np.nan,
                "model": "none",
                "mae": np.nan,
                "rmse": np.nan,
                "nrmse": np.nan,
                "r2": np.nan,
                "corr": np.nan,
                "error": str(e),
            })

    result_df = pd.DataFrame(rows)

    metric_cols = ["mae", "rmse", "nrmse", "r2", "corr"]
    for col in metric_cols:
        if col not in result_df.columns:
            result_df[col] = np.nan

    result_path = out_dir / "window_metrics.csv"
    result_df.to_csv(result_path, index=False)

    valid_df = result_df.dropna(subset=["mae", "rmse", "nrmse", "r2"], how="any").copy()

    summary_rows = []

    for model_name, g in valid_df.groupby("model"):
        summary_rows.append({
            "model": model_name,
            "num_windows": len(g),
            "mean_mae": g["mae"].mean(),
            "mean_rmse": g["rmse"].mean(),
            "mean_nrmse": g["nrmse"].mean(),
            "mean_r2": g["r2"].mean(),
            "mean_corr": g["corr"].mean(),
            "median_mae": g["mae"].median(),
            "median_rmse": g["rmse"].median(),
            "median_nrmse": g["nrmse"].median(),
            "median_r2": g["r2"].median(),
            "median_corr": g["corr"].median(),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = out_dir / "summary_by_model.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\n=== Summary by model ===")
    if len(summary_df) > 0:
        print(summary_df.sort_values("mean_rmse").to_string(index=False))
    else:
        print("有効な評価結果がありません。")

    if "error" in result_df.columns:
        error_df = result_df[result_df["error"].astype(str).str.len() > 0]
        if len(error_df) > 0:
            print("\n=== Errors ===")
            print(error_df["error"].value_counts().head(10))

    print("\nSaved:")
    print(f"  metrics : {result_path}")
    print(f"  summary : {summary_path}")
    print(f"  plots   : {plot_dir}")


if __name__ == "__main__":
    main()