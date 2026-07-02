import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm

from model.model import ChatTime

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt



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
def save_prediction_plot(
    ecg_id,
    hist,
    y_true,
    y_pred,
    hist_len,
    pred_len,
    lead,
    out_dir,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    hist = np.asarray(hist)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    hist_x = np.arange(hist_len)
    pred_x = np.arange(hist_len, hist_len + pred_len)
    pred_t = np.arange(pred_len)

    # 1. 履歴 + 未来予測の全体図
    plt.figure(figsize=(12, 4))
    plt.plot(hist_x, hist, label="history", color="black")
    plt.plot(pred_x, y_true, label="true future", color="blue")
    plt.plot(pred_x, y_pred, label="predicted future", color="orange")
    plt.axvline(hist_len - 1, color="red", linestyle="--", label="prediction start")
    plt.title(f"ECG prediction: ecg_id={ecg_id}, lead={lead}")
    plt.xlabel("sample")
    plt.ylabel("amplitude")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"ecg_{ecg_id}_lead_{lead}_full.png", dpi=200)
    plt.close()

    # 2. 予測区間だけの拡大図
    plt.figure(figsize=(10, 4))
    plt.plot(pred_t, y_true, label="true future", color="blue")
    plt.plot(pred_t, y_pred, label="predicted future", color="orange")
    plt.title(f"Prediction window only: ecg_id={ecg_id}, lead={lead}")
    plt.xlabel("future sample")
    plt.ylabel("amplitude")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"ecg_{ecg_id}_lead_{lead}_future_only.png", dpi=200)
    plt.close()

    # 3. CSV保存
    max_len = max(hist_len, pred_len)

    df_plot = pd.DataFrame({
        "index": np.arange(max_len),
        "history": np.pad(hist, (0, max_len - hist_len), constant_values=np.nan),
        "y_true_future": np.pad(y_true, (0, max_len - pred_len), constant_values=np.nan),
        "y_pred_future": np.pad(y_pred, (0, max_len - pred_len), constant_values=np.nan),
    })

    df_plot.to_csv(out_dir / f"ecg_{ecg_id}_lead_{lead}_prediction.csv", index=False)

def read_ecg_record(root, filename_lr):
    record_path = Path(root) / filename_lr
    signal, fields = wfdb.rdsamp(str(record_path))
    lead_names = fields["sig_name"]
    return signal, lead_names


def get_lead_index(lead_names, target_lead):
    normalized = [x.upper() for x in lead_names]
    target = target_lead.upper()

    if target not in normalized:
        raise ValueError(
            f"Lead {target_lead} が見つかりません。利用可能な誘導: {lead_names}"
        )

    return normalized.index(target)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        default="data/ptb-xl/1.0.3",
        help="PTB-XL v1.0.3 のルートディレクトリ",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="ChengsenWang/ChatTime-1-7B-Chat",
    )
    parser.add_argument("--hist_len", type=int, default=500)
    parser.add_argument("--pred_len", type=int, default=100)
    parser.add_argument("--lead", type=str, default="II")
    parser.add_argument("--fold", type=int, default=10)
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="まずは小さく試す。全件評価は -1",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="履歴区間の平均・標準偏差で正規化してから予測",
    )
    parser.add_argument(
        "--plot_dir",
        type=str,
        default="results/plots",
        help="予測波形の可視化結果を保存するディレクトリ",
    )

    parser.add_argument(
        "--num_plots",
        type=int,
        default=5,
        help="保存する可視化サンプル数",
    )

    args = parser.parse_args()

    plot_dir = Path(args.plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    plot_count = 0

    root = Path(args.root)
    db_path = root / "ptbxl_database.csv"

    df = pd.read_csv(db_path)

    if "strat_fold" in df.columns:
        df = df[df["strat_fold"] == args.fold].copy()

    if args.limit > 0:
        df = df.head(args.limit).copy()

    print(f"評価対象レコード数: {len(df)}")
    print(f"Lead: {args.lead}")
    print(f"hist_len: {args.hist_len}, pred_len: {args.pred_len}")

    model = ChatTime(
        hist_len=args.hist_len,
        pred_len=args.pred_len,
        model_path=args.model_path,
    )

    rows = []
    all_true = []
    all_pred = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        ecg_id = row["ecg_id"]
        filename_lr = row["filename_lr"]

        try:
            signal, lead_names = read_ecg_record(root, filename_lr)
            lead_idx = get_lead_index(lead_names, args.lead)

            x = signal[:, lead_idx].astype(np.float32)

            need_len = args.hist_len + args.pred_len
            if len(x) < need_len:
                continue

            segment = x[:need_len]
            hist = segment[: args.hist_len]
            y_true = segment[args.hist_len : need_len]

            if args.normalize:
                mu = np.mean(hist)
                sigma = np.std(hist) + 1e-8
                hist_input = (hist - mu) / sigma
            else:
                mu = 0.0
                sigma = 1.0
                hist_input = hist

            y_pred = model.predict(hist_input)
            y_pred = np.asarray(y_pred, dtype=np.float32).reshape(-1)

            if len(y_pred) > args.pred_len:
                y_pred = y_pred[: args.pred_len]
            elif len(y_pred) < args.pred_len:
                pad_len = args.pred_len - len(y_pred)
                y_pred = np.pad(y_pred, (0, pad_len), mode="edge")

            if args.normalize:
                y_pred = y_pred * sigma + mu

            m = calc_metrics(y_true, y_pred)

            rows.append(
                {
                    "ecg_id": ecg_id,
                    "filename_lr": filename_lr,
                    "lead": args.lead,
                    **m,
                }
            )
            if plot_count < args.num_plots:
                save_prediction_plot(
                    ecg_id=ecg_id,
                    hist=hist,
                    y_true=y_true,
                    y_pred=y_pred,
                    hist_len=args.hist_len,
                    pred_len=args.pred_len,
                    lead=args.lead,
                    out_dir=plot_dir,
                )
                plot_count += 1
            all_true.append(y_true)
            all_pred.append(y_pred)

        except Exception as e:
            rows.append(
                {
                    "ecg_id": ecg_id,
                    "filename_lr": filename_lr,
                    "lead": args.lead,
                    "error": str(e),
                }
            )
            
    result_df = pd.DataFrame(rows)
    result_df.to_csv("ptbxl_chattime_record_metrics.csv", index=False)

    valid_df = result_df.dropna(subset=["mae", "rmse", "nrmse", "r2"], how="any")

    if len(all_true) > 0:
        all_true = np.concatenate(all_true)
        all_pred = np.concatenate(all_pred)
        overall = calc_metrics(all_true, all_pred)
    else:
        overall = {}

    summary = {
        "num_records_requested": len(df),
        "num_records_success": len(valid_df),
        "lead": args.lead,
        "hist_len": args.hist_len,
        "pred_len": args.pred_len,
        "fold": args.fold,
        "mean_mae": valid_df["mae"].mean() if len(valid_df) else np.nan,
        "mean_rmse": valid_df["rmse"].mean() if len(valid_df) else np.nan,
        "mean_nrmse": valid_df["nrmse"].mean() if len(valid_df) else np.nan,
        "mean_r2": valid_df["r2"].mean() if len(valid_df) else np.nan,
        "mean_corr": valid_df["corr"].mean() if len(valid_df) else np.nan,
        "overall_mae": overall.get("mae", np.nan),
        "overall_rmse": overall.get("rmse", np.nan),
        "overall_nrmse": overall.get("nrmse", np.nan),
        "overall_r2": overall.get("r2", np.nan),
        "overall_corr": overall.get("corr", np.nan),
    }

    summary_df = pd.DataFrame([summary])
    summary_df.to_csv("ptbxl_chattime_summary.csv", index=False)

    print("\n=== Summary ===")
    print(summary_df.T)


if __name__ == "__main__":
    main()