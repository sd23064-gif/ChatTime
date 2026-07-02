import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

from utils.tools import Discretizer, Serializer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_ecg_record(root, filename_lr):
    record_path = Path(root) / filename_lr
    signal, fields = wfdb.rdsamp(str(record_path))
    return signal, fields["sig_name"]


def get_lead_index(lead_names, target_lead):
    normalized = [x.upper() for x in lead_names]
    target = target_lead.upper()

    if target not in normalized:
        raise ValueError(f"Lead {target_lead} not found. Available: {lead_names}")

    return normalized.index(target)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="/workspace/data/ptb-xl/1.0.3")
    parser.add_argument("--lead", type=str, default="II")
    parser.add_argument("--hist_len", type=int, default=200)
    parser.add_argument("--ecg_index", type=int, default=0)
    parser.add_argument("--fold", type=int, default=10)
    parser.add_argument("--start", type=int, default=0)
    args = parser.parse_args()

    root = Path(args.root)
    df = pd.read_csv(root / "ptbxl_database.csv")

    if "strat_fold" in df.columns:
        df = df[df["strat_fold"] == args.fold].copy()

    
    matched = df[df["ecg_id"] == 38]

    if len(matched) == 0:
        raise ValueError(f"指定された ecg_id={args.ecg_id} が見つかりません。")

    row = matched.iloc[0]


    ecg_id = row["ecg_id"]
    filename_lr = row["filename_lr"]

    signal, lead_names = read_ecg_record(root, filename_lr)
    lead_idx = get_lead_index(lead_names, args.lead)

    x = signal[:, lead_idx].astype(np.float32)
    hist = x[args.start : args.start + args.hist_len]

    discretizer = Discretizer()
    serializer = Serializer()

    dispersed = discretizer.discretize(hist)
    restored = discretizer.inverse_discretize(dispersed)
    serialized = serializer.serialize(dispersed)

    print("=== ECG record ===")
    print(f"ecg_id      : {ecg_id}")
    print(f"filename_lr : {filename_lr}")
    print(f"lead        : {args.lead}")
    print(f"start       : {args.start}")
    print(f"hist_len    : {args.hist_len}")
    print()

    print("=== Original ECG hist ===")
    print(f"shape : {hist.shape}")
    print(f"min   : {np.min(hist)}")
    print(f"max   : {np.max(hist)}")
    print(f"mean  : {np.mean(hist)}")
    print(f"std   : {np.std(hist)}")
    print("first 20 values:")
    print(hist[:20])
    print()

    print("=== After Discretizer.discretize(hist) ===")
    print(f"shape : {np.asarray(dispersed).shape}")
    print(f"min   : {np.nanmin(dispersed)}")
    print(f"max   : {np.nanmax(dispersed)}")
    print(f"mean  : {np.nanmean(dispersed)}")
    print(f"std   : {np.nanstd(dispersed)}")
    print("first 20 values:")
    print(np.asarray(dispersed)[:20])
    print()

    print("=== After inverse_discretize(dispersed) ===")
    print(f"shape : {np.asarray(restored).shape}")
    print(f"min   : {np.nanmin(restored)}")
    print(f"max   : {np.nanmax(restored)}")
    print(f"mean  : {np.nanmean(restored)}")
    print(f"std   : {np.nanstd(restored)}")
    print("first 20 values:")
    print(np.asarray(restored)[:20])
    print()

    diff = np.asarray(restored) - hist
    print("=== Reconstruction error: restored - original ===")
    print(f"MAE  : {np.mean(np.abs(diff))}")
    print(f"RMSE : {np.sqrt(np.mean(diff ** 2))}")
    print(f"max_abs_error : {np.max(np.abs(diff))}")
    print()

    print("=== Serialized text ===")
    print("first 1000 characters:")
    print(serialized[:1000])
    print()

    # Discretizer内部の属性を表示
    print("=== Discretizer internal attributes ===")
    for k, v in discretizer.__dict__.items():
        print(f"{k}: {type(v)}")
        try:
            if isinstance(v, np.ndarray):
                print(f"  shape={v.shape}, min={np.nanmin(v)}, max={np.nanmax(v)}")
            else:
                print(f"  value={v}")
        except Exception:
            print(f"  value={v}")
    out_dir = Path("/workspace/results/discretizer_debug")
    out_dir.mkdir(parents=True, exist_ok=True)

    t = np.arange(len(hist))

    plt.figure(figsize=(12, 4))
    plt.plot(t, hist, label="original ECG", color="blue")
    plt.plot(t, dispersed, label="after discretize", color="orange", linestyle="--")
    plt.title(f"Discretizer reconstruction: ecg_id={ecg_id}, lead={args.lead}, start={args.start}")
    plt.xlabel("sample")
    plt.ylabel("amplitude")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"discretizer_ecg_{ecg_id}_lead_{args.lead}_start_{args.start}.png", dpi=200)
    plt.close()

    pd.DataFrame({
        "sample": t,
        "original": hist,
        "discretized": dispersed,
        "restored": restored,
        "error": restored - hist,
    }).to_csv(out_dir / f"discretizer_ecg_{ecg_id}_lead_{args.lead}_start_{args.start}.csv", index=False)

    print(f"Saved plot and csv to: {out_dir}")


if __name__ == "__main__":
    main()