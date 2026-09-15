"""
filter_dataset_by_dominance.py

audit_dataset_dominance.py が出力した dominance_audit_full.csv を使い、
dominant_token_frac が閾値以上の行(単一の値トークンが異常に連続する行)を
元の学習データセットから除外し、そのまま --train_file として使える
新しいCSVを書き出すスクリプト。

トークナイザ・モデルのロードは不要(audit結果と元CSVをrow_indexで突き合わせるだけ)
なので、GPU無し・高速に実行できる。

使い方の例:
    python filter_dataset_by_dominance.py \
        --audit_csv /workspace/outputs/dominance_audit/dominance_audit_full.csv \
        --dataset_path ChengsenWang/ChatTime-1-Pretrain-1M \
        --output_file /workspace/data/ChatTime-1-Pretrain-1M.filtered.csv \
        --dominance_threshold 0.5
"""

import argparse
import os

import pandas as pd
from datasets import load_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="dominance監査結果を使って、異常に偏った行を除外した学習用CSVを作成する"
    )
    parser.add_argument(
        "--audit_csv", type=str, required=True,
        help="audit_dataset_dominance.py が出力した dominance_audit_full.csv",
    )
    parser.add_argument(
        "--train_file", type=str, default=None,
        help="元のローカルCSVパス。未指定の場合は--dataset_pathからURLを組み立てる",
    )
    parser.add_argument(
        "--dataset_path", type=str, default=None,
        help="train_fileを指定しない場合に使う、HuggingFace Hub上のデータセットID",
    )
    parser.add_argument(
        "--output_file", type=str, required=True,
        help="フィルタ後のCSVの出力パス(そのまま--train_fileとして使える)",
    )
    parser.add_argument(
        "--dominance_threshold", type=float, default=0.5,
        help="この値以上のdominant_token_fracを持つ行を除外する(デフォルト: 0.5)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.train_file is not None:
        train_file = args.train_file
    elif args.dataset_path is not None:
        train_file = (
            f"https://huggingface.co/datasets/{args.dataset_path}"
            "/resolve/main/ChatTime-1-Pretrain-1M.csv"
        )
    else:
        raise ValueError("--train_file か --dataset_path のどちらかを指定してください")

    print(f"Loading original dataset from: {train_file}")
    dataset = load_dataset("csv", data_files={"train": train_file})["train"]

    print(f"Loading audit results from: {args.audit_csv}")
    audit_df = pd.read_csv(args.audit_csv)

    if len(audit_df) != len(dataset):
        raise ValueError(
            f"行数が一致しません: audit={len(audit_df)} dataset={len(dataset)}. "
            "audit_dataset_dominance.pyを実行した時と同じCSV/データセットか確認してください "
            "(--sample_sizeを指定して監査した場合、全件ではなく先頭の一部しか監査していません)。"
        )

    keep_mask = audit_df["dominant_token_frac"] < args.dominance_threshold
    keep_indices = audit_df.loc[keep_mask, "row_index"].tolist()

    removed = len(dataset) - len(keep_indices)
    print(f"Total rows: {len(dataset)}")
    print(
        f"Removed (dominant_token_frac >= {args.dominance_threshold}): "
        f"{removed} ({100 * removed / len(dataset):.2f}%)"
    )
    print(
        f"Kept: {len(keep_indices)} "
        f"({100 * len(keep_indices) / len(dataset):.2f}%)"
    )

    filtered = dataset.select(keep_indices)

    out_dir = os.path.dirname(args.output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    filtered.to_csv(args.output_file, index=False)
    print(f"Filtered dataset saved to: {args.output_file}")


if __name__ == "__main__":
    main()