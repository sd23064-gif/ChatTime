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


CHAT_TIME_WINDOW_CONFIGS = [
    {"window_size": 576, "hist_len": 512, "pred_len": 64, "stride": 32},
    {"window_size": 288, "hist_len": 256, "pred_len": 32, "stride": 16},
    {"window_size": 144, "hist_len": 128, "pred_len": 16, "stride": 8},
    {"window_size": 72,  "hist_len": 64,  "pred_len": 8,  "stride": 4},
    {"window_size": 36,  "hist_len": 32,  "pred_len": 4,  "stride": 2},
]


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


def robust_scale_1d_to_unit(x, eps=1e-8):
    x = x.astype(np.float32)
    scale = np.percentile(np.abs(x), 99)
    x = x / (scale + eps)
    x = np.clip(x, -1.0, 1.0)
    return x


def make_chattime_text_from_window(
    x,
    start,
    hist_len,
    pred_len,
    discretizer,
    serializer,
):
    total_len = hist_len + pred_len
    window = x[start:start + total_len].astype(np.float32)

    if len(window) != total_len:
        raise ValueError(
            f"Invalid window length: got {len(window)}, expected {total_len}"
        )

    # ChatTime predict と同じ思想:
    # history部分でscalerをfitし、future部分も同じスケールで離散化する
    dispersed = discretizer.discretize(
        window,
        fit_length=hist_len,
    )

    text = serializer.serialize(dispersed)
    return text


def convert_df_to_chattime_windows(
    df,
    ptbxl_root,
    output_csv,
    discretizer,
    serializer,
    sampling_rate=100,
    lead_idx=1,
    window_configs=None,
    max_windows_per_record_per_config=-1,
    normalize=False,
    save_metadata=False,
):
    if window_configs is None:
        window_configs = CHAT_TIME_WINDOW_CONFIGS

    rows = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=str(output_csv)):
        ecg_id = row.get("ecg_id", None)
        patient_id = row.get("patient_id", None)
        strat_fold = row.get("strat_fold", None)

        try:
            signal, meta = load_ptbxl_record(
                row=row,
                ptbxl_root=ptbxl_root,
                sampling_rate=sampling_rate,
            )

            if lead_idx < 0 or lead_idx >= signal.shape[1]:
                            raise ValueError(
                                f"lead_idx={lead_idx} is invalid for signal shape {signal.shape}"
                            )
            x = signal[:, lead_idx].astype(np.float32)

            if normalize:
                x = robust_scale_1d_to_unit(x)

            for cfg in window_configs:
                hist_len = cfg["hist_len"]
                pred_len = cfg["pred_len"]
                stride = cfg["stride"]
                window_size = cfg["window_size"]

                total_len = hist_len + pred_len
                assert total_len == window_size

                if len(x) < total_len:
                    continue

                possible_starts = list(
                    range(
                        0,
                        len(x) - total_len + 1,
                        stride,
                    )
                )

                # ChatTime論文では大きいwindowを優先するため、
                # 必要なら各config内で上限を設ける
                if max_windows_per_record_per_config > 0 and len(possible_starts) > max_windows_per_record_per_config:
                    indices = np.linspace(
                        0,
                        len(possible_starts) - 1,
                        max_windows_per_record_per_config,
                        dtype=int,
                    )
                    possible_starts = [possible_starts[i] for i in indices]

                for start in possible_starts:
                    text = make_chattime_text_from_window(
                        x=x,
                        start=start,
                        hist_len=hist_len,
                        pred_len=pred_len,
                        discretizer=discretizer,
                        serializer=serializer,
                    )

                    if save_metadata:
                        rows.append({
                            "text": text,
                            "ecg_id": ecg_id,
                            "patient_id": patient_id,
                            "strat_fold": strat_fold,
                            "lead": LEAD_NAMES[lead_idx],
                            "start": start,
                            "window_size": window_size,
                            "hist_len": hist_len,
                            "pred_len": pred_len,
                            "stride": stride,
                        })
                    else:
                        rows.append({
                            "text": text,
                        })

        except Exception as e:
            print(
                f"[WARN] failed ecg_id={ecg_id}, "
                f"patient_id={patient_id}, error={e}"
            )

    out_df = pd.DataFrame(rows)

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)

    print("")
    print(f"Saved: {output_csv}")
    print("Rows:", len(out_df))

    if len(out_df) > 0:
        print("")
        print("Example text:")
        print(out_df.iloc[0]["text"][:1000])


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ptbxl_root", type=str, required=True)
    parser.add_argument("--code_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--sampling_rate", type=int, default=100)
    parser.add_argument("--lead_idx", type=int, default=1)

    parser.add_argument(
        "--max_windows_per_record_per_config",
        type=int,
        default=-1,
        help="1 record・1 window設定あたり最大何window作るか。-1なら全window。",
    )

    parser.add_argument(
        "--max_records_per_split",
        type=int,
        default=None,
        help="debug用。各splitで最大何record使うか。Noneなら全件。",
    )

    parser.add_argument("--low_limit", type=float, default=-1)
    parser.add_argument("--high_limit", type=float, default=1)
    parser.add_argument("--n_tokens", type=int, default=10002)

    parser.add_argument("--prec", type=int, default=4)
    parser.add_argument("--time_sep", type=str, default=" ")
    parser.add_argument("--time_flag", type=str, default="###")
    parser.add_argument("--nan_flag", type=str, default="Nan")

    parser.add_argument(
        "--normalize",
        action="store_true",
        default=False,
        help="外側でrobust normalizeする。通常B-1ではFalse推奨。",
    )

    parser.add_argument(
        "--save_metadata",
        action="store_true",
        default=False,
        help="text以外にecg_id, patient_id, strat_fold等も保存する。",
    )

    args = parser.parse_args()

    sys.path.append(args.code_path)
    from utils.tools import Discretizer, Serializer

    ptbxl_root = Path(args.ptbxl_root)
    db_path = ptbxl_root / "ptbxl_database.csv"

    if not db_path.exists():
        raise FileNotFoundError(f"ptbxl_database.csv not found: {db_path}")

    if args.lead_idx < 0 or args.lead_idx >= len(LEAD_NAMES):
        raise ValueError(f"lead_idx must be 0-11, got {args.lead_idx}")

    print("=== PTB-XL to ChatTime B-1 continuous pretraining dataset ===")
    print("PTB-XL root:", ptbxl_root)
    print("Database:", db_path)
    print("Sampling rate:", args.sampling_rate)
    print("Lead:", args.lead_idx, LEAD_NAMES[args.lead_idx])
    print("Window configs:")
    for cfg in CHAT_TIME_WINDOW_CONFIGS:
        print(" ", cfg)
    print("max_windows_per_record_per_config:", args.max_windows_per_record_per_config)
    print("normalize:", args.normalize)
    print("save_metadata:", args.save_metadata)
    print("============================================================")

    df = pd.read_csv(db_path)

    if "strat_fold" not in df.columns:
        raise ValueError("ptbxl_database.csv に strat_fold 列がありません。")

    train_df = df[df["strat_fold"].isin([1, 2, 3, 4, 5, 6, 7, 8])].copy()
    val_df = df[df["strat_fold"] == 9].copy()
    test_df = df[df["strat_fold"] == 10].copy()

    if args.max_records_per_split is not None:
        train_df = train_df.iloc[: args.max_records_per_split].copy()
        val_df = val_df.iloc[: args.max_records_per_split].copy()
        test_df = test_df.iloc[: args.max_records_per_split].copy()

    print("Train records:", len(train_df))
    print("Val records:", len(val_df))
    print("Test records:", len(test_df))

    discretizer = Discretizer(
        low_limit=args.low_limit,
        high_limit=args.high_limit,
        n_tokens=args.n_tokens,
    )

    serializer = Serializer(
        prec=args.prec,
        time_sep=args.time_sep,
        time_flag=args.time_flag,
        nan_flag=args.nan_flag,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    convert_df_to_chattime_windows(
        df=train_df,
        ptbxl_root=ptbxl_root,
        output_csv=output_dir / "ptbxl_lead2_train.csv",
        discretizer=discretizer,
        serializer=serializer,
        sampling_rate=args.sampling_rate,
        lead_idx=args.lead_idx,
        window_configs=CHAT_TIME_WINDOW_CONFIGS,
        max_windows_per_record_per_config=args.max_windows_per_record_per_config,
        normalize=args.normalize,
        save_metadata=args.save_metadata,
    )

    convert_df_to_chattime_windows(
        df=val_df,
        ptbxl_root=ptbxl_root,
        output_csv=output_dir / "ptbxl_lead2_val.csv",
        discretizer=discretizer,
        serializer=serializer,
        sampling_rate=args.sampling_rate,
        lead_idx=args.lead_idx,
        window_configs=CHAT_TIME_WINDOW_CONFIGS,
        max_windows_per_record_per_config=args.max_windows_per_record_per_config,
        normalize=args.normalize,
        save_metadata=args.save_metadata,
    )

    convert_df_to_chattime_windows(
        df=test_df,
        ptbxl_root=ptbxl_root,
        output_csv=output_dir / "ptbxl_lead2_test.csv",
        discretizer=discretizer,
        serializer=serializer,
        sampling_rate=args.sampling_rate,
        lead_idx=args.lead_idx,
        window_configs=CHAT_TIME_WINDOW_CONFIGS,
        max_windows_per_record_per_config=args.max_windows_per_record_per_config,
        normalize=args.normalize,
        save_metadata=args.save_metadata,
    )


if __name__ == "__main__":
    main()