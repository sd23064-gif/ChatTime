import argparse
import ast
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm

import sys


LEAD_NAMES = [
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6",
]


def robust_scale_to_unit(x, eps=1e-8):
    """
    x: shape (T, C)
    leadごとに99 percentileで[-1, 1]へ正規化
    """
    x = x.astype(np.float32)
    scale = np.percentile(np.abs(x), 99, axis=0, keepdims=True)
    x = x / (scale + eps)
    x = np.clip(x, -1.0, 1.0)
    return x


def serialize_signal(signal, serializer, max_points=None, include_lead_names=True):
    """
    signal: shape (T, 12)
    """
    if max_points is not None:
        signal = signal[:max_points]

    chunks = []

    for lead_idx in range(signal.shape[1]):
        values = signal[:, lead_idx].astype(np.float32)

        serialized_values = serializer.serialize(values)

        if include_lead_names:
            lead_header = f"{serializer.time_flag} Lead_{LEAD_NAMES[lead_idx]}"
        else:
            lead_header = serializer.time_flag

        chunks.append(lead_header + " " + serialized_values)

    return " ".join(chunks)


def load_ptbxl_record(row, ptbxl_root, sampling_rate):
    if sampling_rate == 100:
        rel_path = row["filename_lr"]
    elif sampling_rate == 500:
        rel_path = row["filename_hr"]
    else:
        raise ValueError("sampling_rate must be 100 or 500")

    record_path = ptbxl_root / rel_path
    signal, meta = wfdb.rdsamp(str(record_path))
    return signal, meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ptbxl_root", type=str, required=True)
    parser.add_argument("--code_path", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)

    parser.add_argument("--sampling_rate", type=int, default=100)
    parser.add_argument("--max_records", type=int, default=None)

    parser.add_argument("--low_limit", type=float, default=-1)
    parser.add_argument("--high_limit", type=float, default=1)
    parser.add_argument("--prec", type=int, default=4)
    parser.add_argument("--time_sep", type=str, default=" ")
    parser.add_argument("--time_flag", type=str, default="###")
    parser.add_argument("--nan_flag", type=str, default="Nan")

    # 100Hzで1000点。max_seq_length対策で短くしたい場合に使う
    parser.add_argument("--max_points", type=int, default=1000)

    # label付きにしたい場合
    parser.add_argument("--with_labels", action="store_true")

    args = parser.parse_args()

    sys.path.append(args.code_path)
    from utils.tools import Serializer

    ptbxl_root = Path(args.ptbxl_root)
    db_path = ptbxl_root / "ptbxl_database.csv"

    df = pd.read_csv(db_path)

    if args.max_records is not None:
        df = df.iloc[: args.max_records].copy()

    serializer = Serializer(
        prec=args.prec,
        time_sep=args.time_sep,
        time_flag=args.time_flag,
        nan_flag=args.nan_flag,
    )

    rows = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        signal, meta = load_ptbxl_record(row, ptbxl_root, args.sampling_rate)

        # signal shape: (T, 12)
        signal = robust_scale_to_unit(signal)

        text = serialize_signal(
            signal=signal,
            serializer=serializer,
            max_points=args.max_points,
            include_lead_names=True,
        )

        if args.with_labels:
            # scp_codes は文字列dictとして入っている
            scp_codes = ast.literal_eval(row["scp_codes"])
            label_text = " ".join([f"{k}:{v}" for k, v in scp_codes.items()])
            text = text + f" Diagnosis {label_text}"

        rows.append({"text": text})

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.output_csv, index=False)

    print(f"Saved: {args.output_csv}")
    print(out_df.head())


if __name__ == "__main__":
    main()