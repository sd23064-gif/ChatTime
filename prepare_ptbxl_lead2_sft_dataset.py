import argparse
import ast
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

DIAGNOSTIC_SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


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


def serialize_lead_only(signal, serializer, lead_idx=1, max_points=1000):
    """
    1つのECG recordから指定leadのみをChatTime用テキストへ変換する。
    lead_idx=1 は Lead II。
    """
    values = signal[:max_points, lead_idx].astype(np.float32)

    # Serializer.serialize には 1次元配列を渡す
    serialized_values = serializer.serialize(values)

    return f"{serializer.time_flag} ECG Lead_{LEAD_NAMES[lead_idx]} " + serialized_values


def load_scp_statements(ptbxl_root):
    """
    scp_statements.csv を読み込み、SCP code -> diagnostic superclass の対応を作る。
    """
    scp_path = ptbxl_root / "scp_statements.csv"

    if not scp_path.exists():
        raise FileNotFoundError(f"scp_statements.csv not found: {scp_path}")

    scp_df = pd.read_csv(scp_path, index_col=0)

    code_to_superclass = {}

    for code, row in scp_df.iterrows():
        diagnostic = row.get("diagnostic", 0)
        diagnostic_class = row.get("diagnostic_class", None)

        if pd.isna(diagnostic_class):
            continue

        try:
            is_diagnostic = bool(float(diagnostic))
        except Exception:
            is_diagnostic = bool(diagnostic)

        if is_diagnostic and diagnostic_class in DIAGNOSTIC_SUPERCLASSES:
            code_to_superclass[str(code)] = str(diagnostic_class)

    return code_to_superclass


def extract_superclasses_from_scp_codes(scp_codes_str, code_to_superclass):
    """
    ptbxl_database.csv の scp_codes から診断superclass集合を取り出す。
    """
    try:
        scp_codes = ast.literal_eval(scp_codes_str)
    except Exception:
        return []

    labels = set()

    for code in scp_codes.keys():
        code = str(code)
        if code in code_to_superclass:
            labels.add(code_to_superclass[code])

    labels = sorted(labels, key=lambda x: DIAGNOSTIC_SUPERCLASSES.index(x))

    return labels


def build_sft_text(
    ecg_text,
    labels,
    task_style="diagnostic_superclass",
):
    """
    ChatTime / SFTTrainer 用の text を作る。
    """
    if len(labels) == 0:
        answer = "UNKNOWN"
    else:
        answer = ", ".join(labels)

    if task_style == "diagnostic_superclass":
        instruction = (
            "Question: What are the diagnostic superclasses of this ECG? "
            "Choose from NORM, MI, STTC, CD, and HYP."
        )
    else:
        instruction = "Question: Analyze this ECG."

    text = (
        f"{ecg_text}\n"
        f"{instruction}\n"
        f"Answer: {answer}"
    )

    return text


def convert_split_to_csv(
    df,
    ptbxl_root,
    output_csv,
    serializer,
    code_to_superclass,
    sampling_rate=100,
    lead_idx=1,
    max_points=1000,
    normalize=True,
    drop_unknown=True,
):
    rows = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=str(output_csv)):
        labels = extract_superclasses_from_scp_codes(
            row["scp_codes"],
            code_to_superclass,
        )

        if drop_unknown and len(labels) == 0:
            continue

        signal, meta = load_ptbxl_record(
            row=row,
            ptbxl_root=ptbxl_root,
            sampling_rate=sampling_rate,
        )

        if normalize:
            signal = robust_scale_to_unit(signal)

        ecg_text = serialize_lead_only(
            signal=signal,
            serializer=serializer,
            lead_idx=lead_idx,
            max_points=max_points,
        )

        text = build_sft_text(
            ecg_text=ecg_text,
            labels=labels,
        )

        rows.append({
            "text": text,
            "ecg_id": row.get("ecg_id", None),
            "patient_id": row.get("patient_id", None),
            "strat_fold": row.get("strat_fold", None),
            "labels": ",".join(labels),
        })

    out_df = pd.DataFrame(rows)

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)

    print("")
    print(f"Saved: {output_csv}")
    print("rows:", len(out_df))

    if len(out_df) > 0:
        print("Example:")
        print(out_df.iloc[0]["text"][:1000])


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ptbxl_root", type=str, required=True)
    parser.add_argument("--code_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--sampling_rate", type=int, default=100)
    parser.add_argument("--lead_idx", type=int, default=1)
    parser.add_argument("--max_points", type=int, default=1000)
    parser.add_argument("--max_records_per_split", type=int, default=None)

    parser.add_argument("--prec", type=int, default=4)
    parser.add_argument("--time_sep", type=str, default=" ")
    parser.add_argument("--time_flag", type=str, default="###")
    parser.add_argument("--nan_flag", type=str, default="Nan")

    parser.add_argument("--normalize", action="store_true", default=True)
    parser.add_argument("--no_normalize", action="store_false", dest="normalize")

    parser.add_argument("--keep_unknown", action="store_true", default=False)

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

    if "strat_fold" not in df.columns:
        raise ValueError("ptbxl_database.csv に strat_fold 列がありません。")

    code_to_superclass = load_scp_statements(ptbxl_root)

    print("Loaded SCP superclass mapping:", len(code_to_superclass))
    print("Superclass candidates:", DIAGNOSTIC_SUPERCLASSES)

    serializer = Serializer(
        prec=args.prec,
        time_sep=args.time_sep,
        time_flag=args.time_flag,
        nan_flag=args.nan_flag,
    )

    train_df = df[df["strat_fold"].isin([1, 2, 3, 4, 5, 6, 7, 8])].copy()
    val_df = df[df["strat_fold"] == 9].copy()
    test_df = df[df["strat_fold"] == 10].copy()

    if args.max_records_per_split is not None:
        train_df = train_df.iloc[: args.max_records_per_split]
        val_df = val_df.iloc[: args.max_records_per_split]
        test_df = test_df.iloc[: args.max_records_per_split]

    print("Raw train rows:", len(train_df))
    print("Raw val rows:", len(val_df))
    print("Raw test rows:", len(test_df))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    drop_unknown = not args.keep_unknown

    convert_split_to_csv(
        df=train_df,
        ptbxl_root=ptbxl_root,
        output_csv=output_dir / "ptbxl_lead2_sft_train.csv",
        serializer=serializer,
        code_to_superclass=code_to_superclass,
        sampling_rate=args.sampling_rate,
        lead_idx=args.lead_idx,
        max_points=args.max_points,
        normalize=args.normalize,
        drop_unknown=drop_unknown,
    )

    convert_split_to_csv(
        df=val_df,
        ptbxl_root=ptbxl_root,
        output_csv=output_dir / "ptbxl_lead2_sft_val.csv",
        serializer=serializer,
        code_to_superclass=code_to_superclass,
        sampling_rate=args.sampling_rate,
        lead_idx=args.lead_idx,
        max_points=args.max_points,
        normalize=args.normalize,
        drop_unknown=drop_unknown,
    )

    convert_split_to_csv(
        df=test_df,
        ptbxl_root=ptbxl_root,
        output_csv=output_dir / "ptbxl_lead2_sft_test.csv",
        serializer=serializer,
        code_to_superclass=code_to_superclass,
        sampling_rate=args.sampling_rate,
        lead_idx=args.lead_idx,
        max_points=args.max_points,
        normalize=args.normalize,
        drop_unknown=drop_unknown,
    )


if __name__ == "__main__":
    main()