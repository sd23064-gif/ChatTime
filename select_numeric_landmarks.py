#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
select_numeric_landmarks.py

最近傍解析CSVから、数値的にも埋め込み的にも信頼できる数値トークンを
landmarkとして抽出する。

選択条件:
  1. rank == target_rank
  2. queryとneighborの数値差 <= max_value_diff
  3. cosine similarityが候補内の指定quantile以上
  4. 必要に応じて符号反転ペアを除外
  5. query valueをbin分割し、各binから最大max_per_bin件を選択

入力CSVに期待する主な列:
  rank
  query_token または token
  query_token_id（なくても可）
  query_value または value
  neighbor_token（任意）
  neighbor_value（任意）
  cosine_similarity
  abs_value_diff（なくてもquery_valueとneighbor_valueから計算）

実行例:
python select_numeric_landmarks.py \
  --input_csv outputs/analysis/normal_mamba_nearest_added_tokens.csv \
  --output_csv outputs/analysis/numeric_landmark_tokens.csv \
  --max_value_diff 0.01 \
  --similarity_quantile 0.90 \
  --value_bin_width 0.05 \
  --max_per_bin 20 \
  --exclude_sign_reversal
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def choose_column(df, candidates, required=True):
    for column in candidates:
        if column in df.columns:
            return column
    if required:
        raise ValueError(
            f"Required column not found. Expected one of {candidates}. "
            f"Actual columns: {df.columns.tolist()}"
        )
    return None


def to_numeric_column(df, column):
    df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def assign_value_bins(values, bin_width):
    if bin_width <= 0:
        raise ValueError("value_bin_width must be greater than 0.")

    minimum = float(np.floor(values.min() / bin_width) * bin_width)
    maximum = float(np.ceil(values.max() / bin_width) * bin_width)

    if np.isclose(minimum, maximum):
        maximum = minimum + bin_width

    edges = np.arange(minimum, maximum + bin_width * 1.0001, bin_width)
    edges = np.unique(np.round(edges, 10))

    if len(edges) < 2:
        edges = np.array([minimum, minimum + bin_width], dtype=float)

    return pd.cut(values, bins=edges, include_lowest=True, duplicates="drop")


def select_landmarks(
    df,
    target_rank,
    max_value_diff,
    similarity_quantile,
    value_bin_width,
    max_per_bin,
    exclude_sign_reversal,
    min_cosine_similarity,
):
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
    )

    rank_column = choose_column(df, ["rank"])
    token_column = choose_column(df, ["query_token", "token"])
    token_id_column = choose_column(
        df,
        ["query_token_id", "token_id"],
        required=False,
    )
    value_column = choose_column(df, ["query_value", "value"])
    similarity_column = choose_column(
        df,
        ["cosine_similarity", "similarity", "cosine_sim"],
    )
    neighbor_value_column = choose_column(
        df,
        ["neighbor_value"],
        required=False,
    )

    for column in [rank_column, value_column, similarity_column]:
        df = to_numeric_column(df, column)

    if neighbor_value_column is not None:
        df = to_numeric_column(df, neighbor_value_column)

    if token_id_column is not None:
        df[token_id_column] = pd.to_numeric(
            df[token_id_column], errors="coerce"
        )

    df = df[df[rank_column] == target_rank].copy()
    df = df.dropna(subset=[token_column, value_column, similarity_column])

    if "abs_value_diff" in df.columns:
        df["abs_value_diff"] = pd.to_numeric(
            df["abs_value_diff"], errors="coerce"
        )
    elif neighbor_value_column is not None:
        df["abs_value_diff"] = (
            df[value_column] - df[neighbor_value_column]
        ).abs()
    else:
        raise ValueError(
            "Cannot calculate abs_value_diff. Input needs abs_value_diff "
            "or neighbor_value."
        )

    df = df.dropna(subset=["abs_value_diff"])

    if "sign_reversed" in df.columns:
        raw = df["sign_reversed"]
        if raw.dtype == bool:
            df["sign_reversed_normalized"] = raw
        else:
            df["sign_reversed_normalized"] = (
                raw.astype(str).str.strip().str.lower()
                .isin(["true", "1", "yes", "y"])
            )
    elif neighbor_value_column is not None:
        df["sign_reversed_normalized"] = (
            df[value_column] * df[neighbor_value_column] < 0
        )
    else:
        df["sign_reversed_normalized"] = False

    rank_count = int(len(df))

    candidates = df[df["abs_value_diff"] <= max_value_diff].copy()

    if exclude_sign_reversal:
        candidates = candidates[
            ~candidates["sign_reversed_normalized"]
        ].copy()

    if len(candidates) == 0:
        raise ValueError(
            "No landmark candidates remain after value-distance/sign filters. "
            "Increase --max_value_diff or allow sign reversal."
        )

    quantile_threshold = float(
        candidates[similarity_column].quantile(similarity_quantile)
    )

    effective_threshold = quantile_threshold
    if min_cosine_similarity is not None:
        effective_threshold = max(
            effective_threshold,
            float(min_cosine_similarity),
        )

    candidates = candidates[
        candidates[similarity_column] >= effective_threshold
    ].copy()

    if len(candidates) == 0:
        raise ValueError(
            "No candidates remain after cosine-similarity filtering. "
            "Lower --similarity_quantile or --min_cosine_similarity."
        )

    candidates["value_bin"] = assign_value_bins(
        candidates[value_column],
        value_bin_width,
    )

    # 各値域で、まず数値差が小さく、その後に類似度が高い候補を優先する。
    candidates = candidates.sort_values(
        ["value_bin", "abs_value_diff", similarity_column],
        ascending=[True, True, False],
    )

    if max_per_bin > 0:
        landmarks = (
            candidates.groupby("value_bin", observed=True, group_keys=False)
            .head(max_per_bin)
            .copy()
        )
    else:
        landmarks = candidates.copy()

    landmarks = landmarks.sort_values(value_column).reset_index(drop=True)

    # 学習側で読み込みやすい標準列を先頭に追加する。
    landmarks.insert(0, "landmark_index", np.arange(len(landmarks)))
    landmarks["token"] = landmarks[token_column].astype(str)
    landmarks["value"] = landmarks[value_column].astype(float)
    landmarks["abs_value"] = landmarks["value"].abs()
    landmarks["landmark_cosine_similarity"] = landmarks[
        similarity_column
    ].astype(float)

    if token_id_column is not None:
        landmarks["token_id"] = landmarks[token_id_column].astype("Int64")

    landmarks["value_bin"] = landmarks["value_bin"].astype(str)

    summary = {
        "input_rank_count": rank_count,
        "candidate_count_after_all_filters": int(len(candidates)),
        "landmark_count": int(len(landmarks)),
        "target_rank": int(target_rank),
        "max_value_diff": float(max_value_diff),
        "similarity_quantile": float(similarity_quantile),
        "cosine_quantile_threshold": quantile_threshold,
        "effective_cosine_threshold": effective_threshold,
        "value_bin_width": float(value_bin_width),
        "max_per_bin": int(max_per_bin),
        "exclude_sign_reversal": bool(exclude_sign_reversal),
        "value_min": float(landmarks["value"].min()),
        "value_max": float(landmarks["value"].max()),
        "abs_value_diff_mean": float(landmarks["abs_value_diff"].mean()),
        "abs_value_diff_max": float(landmarks["abs_value_diff"].max()),
        "cosine_similarity_mean": float(
            landmarks["landmark_cosine_similarity"].mean()
        ),
        "cosine_similarity_min": float(
            landmarks["landmark_cosine_similarity"].min()
        ),
        "occupied_bin_count": int(landmarks["value_bin"].nunique()),
    }

    return landmarks, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--target_rank", type=int, default=1)
    parser.add_argument("--max_value_diff", type=float, default=0.01)
    parser.add_argument(
        "--similarity_quantile",
        type=float,
        default=0.90,
        help="0.90 means keep the top 10 percent by cosine similarity.",
    )
    parser.add_argument(
        "--min_cosine_similarity",
        type=float,
        default=None,
        help="Optional absolute lower bound in addition to quantile threshold.",
    )
    parser.add_argument("--value_bin_width", type=float, default=0.05)
    parser.add_argument("--max_per_bin", type=int, default=20)
    parser.add_argument(
        "--exclude_sign_reversal",
        action="store_true",
        help="Exclude query-neighbor pairs whose signs are reversed.",
    )
    args = parser.parse_args()

    if not 0.0 <= args.similarity_quantile <= 1.0:
        raise ValueError("similarity_quantile must be between 0 and 1.")

    input_path = Path(args.input_csv)
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)

    landmarks, summary = select_landmarks(
        df=df,
        target_rank=args.target_rank,
        max_value_diff=args.max_value_diff,
        similarity_quantile=args.similarity_quantile,
        value_bin_width=args.value_bin_width,
        max_per_bin=args.max_per_bin,
        exclude_sign_reversal=args.exclude_sign_reversal,
        min_cosine_similarity=args.min_cosine_similarity,
    )

    landmarks.to_csv(output_path, index=False)

    summary_path = output_path.with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    bin_summary = (
        landmarks.groupby("value_bin", observed=True)
        .agg(
            landmark_count=("token", "size"),
            value_min=("value", "min"),
            value_max=("value", "max"),
            value_mean=("value", "mean"),
            abs_value_diff_mean=("abs_value_diff", "mean"),
            cosine_similarity_mean=(
                "landmark_cosine_similarity", "mean"
            ),
            cosine_similarity_min=(
                "landmark_cosine_similarity", "min"
            ),
        )
        .reset_index()
    )

    bin_summary_path = output_path.with_name(
        output_path.stem + "_by_bin.csv"
    )
    bin_summary.to_csv(bin_summary_path, index=False)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("Saved landmarks:", output_path)
    print("Saved summary:", summary_path)
    print("Saved bin summary:", bin_summary_path)


if __name__ == "__main__":
    main()
