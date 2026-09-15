"""
audit_dataset_dominance.py

学習データ(CSVの"text"列)を、pretrain_falcon_mamba.pyと全く同じ手順で
数値語彙を拡張したトークナイザでスキャンし、各行について
「値トークンのうち、単一のトークンが占める割合(dominant_token_frac)」
を計算するための監査スクリプト。

- 欠損区間がプレースホルダー値(例: -0.5, 0.0, +0.5など)で埋められている行や、
  Nanフラグトークンを含む行を検出するのが目的。
- モデル本体のロードは不要(トークナイザとutils.tools.Discretizer/Serializerのみ使用)なので、
  GPUなしで高速に実行できる。

使い方の例:
    python audit_dataset_dominance.py \
        --code_path /workspace/training \
        --model_path tiiuae/falcon-mamba-7b \
        --train_file /workspace/data/ChatTime-1-Pretrain-1M.csv \
        --output_path /workspace/outputs/dominance_audit \
        --dominance_threshold 0.3 \
        --dataset_num_proc 16

    # まず先頭10万行だけで素早く様子を見たい場合
        --sample_size 100000
"""

import argparse
import os
import sys
from collections import Counter

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "学習データ中で、単一の数値トークンが異常に連続する行"
            "(欠損/プレースホルダー疑い)を検出する監査スクリプト"
        )
    )
    parser.add_argument(
        "--code_path", type=str, required=True,
        help="utils.tools (Discretizer/Serializer) が置かれているコードのパス",
    )
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="トークナイザを読み込むモデルパス(例: tiiuae/falcon-mamba-7b)",
    )

    parser.add_argument(
        "--train_file", type=str, default=None,
        help="ローカルのCSVパス。未指定の場合は--dataset_pathからURLを組み立てる",
    )
    parser.add_argument(
        "--dataset_path", type=str, default=None,
        help=(
            "train_fileを指定しない場合に使う、HuggingFace Hub上のデータセットID"
            "(例: ChengsenWang/ChatTime-1-Pretrain-1M)。"
            "pretrain_falcon_mamba.pyと同じくHF_HOMEのキャッシュを利用する"
        ),
    )
    parser.add_argument(
        "--output_path", type=str, required=True,
        help="監査レポート(CSV)の出力先ディレクトリ",
    )

    # pretrain_falcon_mamba.pyと同じデフォルト値
    parser.add_argument("--low_limit", type=float, default=-1)
    parser.add_argument("--high_limit", type=float, default=1)
    parser.add_argument("--n_tokens", type=int, default=10002)
    parser.add_argument("--prec", type=int, default=4)
    parser.add_argument("--time_sep", type=str, default=" ")
    parser.add_argument("--time_flag", type=str, default="###")
    parser.add_argument("--nan_flag", type=str, default="Nan")

    parser.add_argument(
        "--dominance_threshold", type=float, default=0.3,
        help="値トークン中でこの割合を超えて同一トークンが占める行を「異常」として検出する閾値",
    )
    parser.add_argument(
        "--sample_size", type=int, default=None,
        help="指定した場合、データセットの先頭からこの件数だけ監査する(未指定なら全件)",
    )
    parser.add_argument("--dataset_num_proc", type=int, default=16)
    parser.add_argument(
        "--max_preview_rows", type=int, default=1000,
        help="フラグが立った行のうち、テキストプレビューを付与する最大件数(件数が多いと重くなるための安全策)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    sys.path.append(args.code_path)
    from utils.tools import Discretizer, Serializer  # noqa: E402

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    discretizer = Discretizer(
        low_limit=args.low_limit, high_limit=args.high_limit, n_tokens=args.n_tokens
    )
    serializer = Serializer(
        prec=args.prec, time_sep=args.time_sep, time_flag=args.time_flag, nan_flag=args.nan_flag
    )

    # 【重要】 pretrain_falcon_mamba.pyと全く同じ手順で語彙を拡張する。
    # こうしないと、同じ数値文字列が違うトークンIDにマッピングされてしまい、
    # 監査結果が実際の学習と一致しなくなる。
    vocabulary = np.concatenate((discretizer.centers[1:-1], [np.nan])).reshape(-1, 1)
    vocabulary = np.array([serializer.serialize(value) for value in vocabulary])
    added_tokens = [str(token) for token in vocabulary.reshape(-1).tolist()]

    old_vocab_size = len(tokenizer)
    num_added_tokens = tokenizer.add_tokens(added_tokens)
    print(f"Old tokenizer size: {old_vocab_size}")
    print(f"Added tokens: {num_added_tokens}")
    print(f"New tokenizer size: {len(tokenizer)}")

    # 最後の1個(nan)を除いた、純粋な数値トークンのID集合
    numeric_token_ids = set(tokenizer.convert_tokens_to_ids(t) for t in added_tokens[:-1])
    nan_token_id = tokenizer.convert_tokens_to_ids(added_tokens[-1])

    print(f"numeric value token count: {len(numeric_token_ids)}")
    print(f"nan-flag token id: {nan_token_id}")

    # load dataset (pretrain_falcon_mamba.pyと同じ読み込み方)
    # 【変更前】 --train_file を必須にしていた
    # 【変更後】 --train_fileが無ければ--dataset_pathからURLを組み立てる
    #   (pretrain_falcon_mamba.pyのフォールバックと同じロジック。
    #    HF_HOME配下のキャッシュがあればそれがそのまま使われる)
    if args.train_file is not None:
        train_file = args.train_file
    elif args.dataset_path is not None:
        train_file = (
            f"https://huggingface.co/datasets/{args.dataset_path}"
            "/resolve/main/ChatTime-1-Pretrain-1M.csv"
        )
    else:
        raise ValueError("--train_file か --dataset_path のどちらかを指定してください")

    print(f"Loading dataset from: {train_file}")
    dataset_dict = load_dataset("csv", data_files={"train": train_file})
    dataset = dataset_dict["train"]

    if args.sample_size is not None:
        dataset = dataset.select(range(min(args.sample_size, len(dataset))))

    print(f"Auditing {len(dataset)} rows...")

    def analyze_row(example):
        text = example["text"]
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]

        value_ids = [i for i in ids if i in numeric_token_ids]
        nan_count = sum(1 for i in ids if i == nan_token_id)

        if len(value_ids) == 0:
            dominant_token_id = -1
            dominant_frac = 0.0
        else:
            counts = Counter(value_ids)
            dominant_token_id, dominant_count = counts.most_common(1)[0]
            dominant_frac = dominant_count / len(value_ids)

        return {
            "seq_len": len(ids),
            "num_value_tokens": len(value_ids),
            "nan_token_count": nan_count,
            "dominant_token_id": dominant_token_id,
            "dominant_token_frac": dominant_frac,
        }

    analyzed = dataset.map(analyze_row, num_proc=args.dataset_num_proc)

    df = analyzed.to_pandas()[
        ["seq_len", "num_value_tokens", "nan_token_count", "dominant_token_id", "dominant_token_frac"]
    ].copy()
    df.insert(0, "row_index", range(len(df)))

    os.makedirs(args.output_path, exist_ok=True)

    full_report_path = os.path.join(args.output_path, "dominance_audit_full.csv")
    df.to_csv(full_report_path, index=False)

    flagged = df[df["dominant_token_frac"] >= args.dominance_threshold].copy()

    # 目視確認しやすいよう、フラグが立った行に元テキストの冒頭を付与する
    # (件数が多い場合は max_preview_rows 件までに制限)
    preview_texts = []
    for row_index in flagged["row_index"].head(args.max_preview_rows).tolist():
        preview_texts.append(dataset[int(row_index)]["text"][:300])
    if len(preview_texts) < len(flagged):
        preview_texts += [None] * (len(flagged) - len(preview_texts))
    flagged["text_preview"] = preview_texts

    flagged_path = os.path.join(args.output_path, "dominance_audit_flagged.csv")
    flagged.to_csv(flagged_path, index=False)

    print("=" * 60)
    print(f"Total rows audited: {len(df)}")
    print(
        f"Rows with dominant_token_frac >= {args.dominance_threshold}: "
        f"{len(flagged)} ({100 * len(flagged) / len(df):.2f}%)"
    )
    print(
        f"Rows containing at least one Nan-flag token: "
        f"{(df['nan_token_count'] > 0).sum()} ({100 * (df['nan_token_count'] > 0).mean():.2f}%)"
    )
    print()
    print("dominant_token_frac の分布:")
    print(df["dominant_token_frac"].describe())

    if len(flagged) > 0:
        print()
        print("フラグが立った行で頻出するdominant_token_id 上位10件:")
        print(flagged["dominant_token_id"].value_counts().head(10))

    print()
    print(f"Full report saved to: {full_report_path}")
    print(f"Flagged rows saved to: {flagged_path}")


if __name__ == "__main__":
    main()