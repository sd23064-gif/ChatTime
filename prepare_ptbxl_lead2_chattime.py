import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm


LEAD_NAMES = [
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6",
]


def robust_scale_to_unit(x, eps=1e-8):
    """
    ECG波形をleadごとに[-1, 1]へ正規化する。

    x: np.ndarray, shape (T, C)
    """
    x = x.astype(np.float32)

    scale = np.percentile(np.abs(x), 99, axis=0, keepdims=True)
    x = x / (scale + eps)
    x = np.clip(x, -1.0, 1.0)

    return x


def serialize_lead_only(
    signal,
    serializer,
    lead_idx=1,
    max_points=1000,
    add_lead_header=True,
):
    """
    1つのECG recordから指定leadのみをChatTime用テキストへ変換する。

    signal: np.ndarray, shape (T, 12)
    lead_idx=1 は Lead II
    """
    values = signal[:max_points, lead_idx].astype(np.float32)

    # 重要:
    # Serializer.serialize には 1次元配列を渡す。
    # np.array([[v]]) のような2次元配列を渡すと
    # unsupported format string passed to numpy.ndarray.__format__
    # が出る。
    serialized_values = serializer.serialize(values)

    if add_lead_header:
        return f"{serializer.time_flag} Lead_{LEAD_NAMES[lead_idx]} " + serialized_values
    else:
        return serialized_values


def load_ptbxl_record(row, ptbxl_root, sampling_rate):
    """
    PTB-XLの1 recordを読み込む。
    sampling_rate=100 の場合 filename_lr,
    sampling_rate=500 の場合 filename_hr を使う。
    """
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
    parser.add_argument("--max_points", type=int, default=1000)

    # Lead II
    parser.add_argument("--lead_idx", type=int, default=1)

    # ChatTime Serializer設定
    parser.add_argument("--prec", type=int, default=4)
    parser.add_argument("--time_sep", type=str, default=" ")
    parser.add_argument("--time_flag", type=str, default="###")
    parser.add_argument("--nan_flag", type=str, default="Nan")

    # 正規化設定
    parser.add_argument("--normalize", action="store_true", default=True)
    parser.add_argument("--no_normalize", action="store_false", dest="normalize")

    args = parser.parse_args()

    sys.path.append(args.code_path)
    from utils.tools import Serializer

    ptbxl_root = Path(args.ptbxl_root)
    db_path = ptbxl_root / "ptbxl_database.csv"

    if not db_path.exists():
        raise FileNotFoundError(f"ptbxl_database.csv not found: {db_path}")

    if args.lead_idx < 0 or args.lead_idx >= len(LEAD_NAMES):
        raise ValueError(f"lead_idx must be 0-11, got {args.lead_idx}")

    print("PTB-XL root:", ptbxl_root)
    print("Database:", db_path)
    print("Sampling rate:", args.sampling_rate)
    print("Target lead:", args.lead_idx, LEAD_NAMES[args.lead_idx])
    print("Max points:", args.max_points)
    print("Normalize:", args.normalize)

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
        signal, meta = load_ptbxl_record(
            row=row,
            ptbxl_root=ptbxl_root,
            sampling_rate=args.sampling_rate,
        )

        # signal shape: (T, 12)
        if args.normalize:
            signal = robust_scale_to_unit(signal)

        text = serialize_lead_only(
            signal=signal,
            serializer=serializer,
            lead_idx=args.lead_idx,
            max_points=args.max_points,
            add_lead_header=True,
        )

        rows.append({
            "text": text,
        })

    out_df = pd.DataFrame(rows)

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)

    print("")
    print(f"Saved: {output_csv}")
    print("Num rows:", len(out_df))
    print("")
    print("Example text:")
    print(out_df.iloc[0]["text"][:1000])


if __name__ == "__main__":
    main()