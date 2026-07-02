#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_rank1_sign_reversal_bins_0p25.py

chattime_nearest_added_tokens.csv から、rank1 のうち符号反転しているペアを抽出し、
query token の数値を 0.25 幅で区切って集計するスクリプト。

出力:
  - rank1_sign_reversal_by_query_bin_0p25.csv
  - rank1_sign_reversal_by_abs_bin_0p25.csv
  - rank1_sign_reversal_query_neighbor_bin_matrix_0p25.csv
  - rank1_sign_reversal_pairs_0p25.csv
  - rank1_sign_reversal_summary_0p25.json
  - png 図いくつか

実行例:
  python analyze_rank1_sign_reversal_bins_0p25.py \
    --input_csv chattime_nearest_added_tokens.csv \
    --output_dir outputs/rank1_sign_reversal_0p25
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_value_from_token_repr(x):
    if pd.isna(x):
        return np.nan
    m = re.search(r"###([+-]?(?:\d+(?:\.\d*)?|\.\d+)|Nan|NaN|nan)###", str(x))
    if m is None:
        return np.nan
    s = m.group(1)
    if s.lower() == "nan":
        return np.nan
    try:
        return float(s)
    except Exception:
        return np.nan


def add_numeric_columns(df):
    # 既に列がある場合はそれを使う。なければ token 表現から parse する。
    if "query_value" in df.columns:
        df["query_value"] = pd.to_numeric(df["query_value"], errors="coerce")
    else:
        # よくある列名に対応
        candidate_cols = ["query_token_repr", "query_token", "query"]
        col = next((c for c in candidate_cols if c in df.columns), None)
        if col is None:
            raise ValueError("query_value も query_token_repr/query_token も見つかりません。")
        df["query_value"] = df[col].map(parse_value_from_token_repr)

    if "neighbor_value" in df.columns:
        df["neighbor_value"] = pd.to_numeric(df["neighbor_value"], errors="coerce")
    else:
        candidate_cols = ["neighbor_token_repr", "neighbor_token", "neighbor"]
        col = next((c for c in candidate_cols if c in df.columns), None)
        if col is None:
            raise ValueError("neighbor_value も neighbor_token_repr/neighbor_token も見つかりません。")
        df["neighbor_value"] = df[col].map(parse_value_from_token_repr)

    if "abs_value_diff" not in df.columns:
        df["abs_value_diff"] = (df["query_value"] - df["neighbor_value"]).abs()
    else:
        df["abs_value_diff"] = pd.to_numeric(df["abs_value_diff"], errors="coerce")

    if "cosine_similarity" in df.columns:
        df["cosine_similarity"] = pd.to_numeric(df["cosine_similarity"], errors="coerce")

    return df


def interval_mid(interval):
    try:
        return interval.mid
    except Exception:
        return np.nan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, default="chattime_nearest_added_tokens.csv")
    parser.add_argument("--output_dir", type=str, default="outputs/rank1_sign_reversal_0p25")
    parser.add_argument("--bin_width", type=float, default=0.25)
    parser.add_argument("--min_value", type=float, default=-1.0)
    parser.add_argument("--max_value", type=float, default=1.0)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_csv)
    df = add_numeric_columns(df)

    if "rank" not in df.columns:
        raise ValueError("rank列が必要です。")

    df["rank"] = pd.to_numeric(df["rank"], errors="coerce")

    # rank1 + 数値ペア
    r1 = df[
        (df["rank"] == 1)
        & df["query_value"].notna()
        & df["neighbor_value"].notna()
    ].copy()

    # 符号反転: 0 は除外。q*n < 0
    r1["sign_reversed"] = r1["query_value"] * r1["neighbor_value"] < 0
    sign_rev = r1[r1["sign_reversed"]].copy()

    sign_rev["abs_query_value"] = sign_rev["query_value"].abs()
    sign_rev["abs_neighbor_value"] = sign_rev["neighbor_value"].abs()
    sign_rev["abs_magnitude_diff"] = (
        sign_rev["abs_query_value"] - sign_rev["abs_neighbor_value"]
    ).abs()

    # 0.25幅 bin
    edges = np.arange(
        args.min_value,
        args.max_value + 1e-9,
        args.bin_width,
    )

        # 念のため最後が max_value でない場合だけ追加
    if edges[-1] < args.max_value:
        edges = np.append(edges, args.max_value)

    edges = np.unique(np.round(edges, 10))
    sign_rev["query_value_bin_0p25"] = pd.cut(
        sign_rev["query_value"],
        bins=edges,
        include_lowest=True,
    )
    sign_rev["neighbor_value_bin_0p25"] = pd.cut(
        sign_rev["neighbor_value"],
        bins=edges,
        include_lowest=True,
    )

    abs_edges = np.arange(
        0.0,
        1.0 + 1e-9,
        args.bin_width,
    )

    if abs_edges[-1] < 1.0:
        abs_edges = np.append(abs_edges, 1.0)

    abs_edges = np.unique(np.round(abs_edges, 10))
    sign_rev["abs_query_value_bin_0p25"] = pd.cut(
        sign_rev["abs_query_value"],
        bins=abs_edges,
        include_lowest=True,
    )

    # 分母: rank1全体の query bin 別件数
    r1["query_value_bin_0p25"] = pd.cut(
        r1["query_value"], bins=edges, include_lowest=True
    )
    denom = r1.groupby("query_value_bin_0p25", observed=True).size().rename("rank1_total_count")

    # query bin別集計
    agg_kwargs = dict(
        sign_reversal_count=("sign_reversed", "size"),
        query_value_mean=("query_value", "mean"),
        neighbor_value_mean=("neighbor_value", "mean"),
        abs_value_diff_mean=("abs_value_diff", "mean"),
        abs_value_diff_median=("abs_value_diff", "median"),
        abs_magnitude_diff_mean=("abs_magnitude_diff", "mean"),
        abs_magnitude_diff_median=("abs_magnitude_diff", "median"),
    )
    if "cosine_similarity" in sign_rev.columns:
        agg_kwargs.update(
            cosine_mean=("cosine_similarity", "mean"),
            cosine_median=("cosine_similarity", "median"),
        )

    by_query_bin = sign_rev.groupby("query_value_bin_0p25", observed=True).agg(**agg_kwargs).reset_index()
    by_query_bin = by_query_bin.merge(denom.reset_index(), on="query_value_bin_0p25", how="left")
    by_query_bin["sign_reversal_ratio_among_rank1"] = (
        by_query_bin["sign_reversal_count"] / by_query_bin["rank1_total_count"]
    )
    by_query_bin["query_value_bin_0p25"] = by_query_bin["query_value_bin_0p25"].astype(str)
    by_query_bin.to_csv(out / "rank1_sign_reversal_by_query_bin_0p25.csv", index=False)

    # 絶対値 bin 別集計
    by_abs_bin = sign_rev.groupby("abs_query_value_bin_0p25", observed=True).agg(**agg_kwargs).reset_index()
    by_abs_bin["abs_query_value_bin_0p25"] = by_abs_bin["abs_query_value_bin_0p25"].astype(str)
    by_abs_bin.to_csv(out / "rank1_sign_reversal_by_abs_bin_0p25.csv", index=False)

    # query bin x neighbor bin matrix
    matrix = pd.crosstab(
        sign_rev["query_value_bin_0p25"].astype(str),
        sign_rev["neighbor_value_bin_0p25"].astype(str),
    )
    matrix.to_csv(out / "rank1_sign_reversal_query_neighbor_bin_matrix_0p25.csv")

    # 個別ペア保存
    save_cols = [
        c for c in [
            "query_token_repr", "query_token_id", "query_value",
            "neighbor_token_repr", "neighbor_token_id", "neighbor_value",
            "cosine_similarity", "abs_value_diff", "abs_magnitude_diff",
            "query_value_bin_0p25", "neighbor_value_bin_0p25", "abs_query_value_bin_0p25",
        ] if c in sign_rev.columns
    ]
    pairs = sign_rev[save_cols].copy()
    for c in ["query_value_bin_0p25", "neighbor_value_bin_0p25", "abs_query_value_bin_0p25"]:
        if c in pairs.columns:
            pairs[c] = pairs[c].astype(str)
    pairs.to_csv(out / "rank1_sign_reversal_pairs_0p25.csv", index=False)

    summary = {
        "input_csv": args.input_csv,
        "rank1_numeric_count": int(len(r1)),
        "rank1_sign_reversal_count": int(len(sign_rev)),
        "rank1_sign_reversal_ratio": float(len(sign_rev) / len(r1)) if len(r1) else None,
        "bin_width": args.bin_width,
        "min_value": args.min_value,
        "max_value": args.max_value,
        "mean_abs_value_diff_sign_reversal": float(sign_rev["abs_value_diff"].mean()) if len(sign_rev) else None,
        "median_abs_value_diff_sign_reversal": float(sign_rev["abs_value_diff"].median()) if len(sign_rev) else None,
        "mean_abs_magnitude_diff_sign_reversal": float(sign_rev["abs_magnitude_diff"].mean()) if len(sign_rev) else None,
        "median_abs_magnitude_diff_sign_reversal": float(sign_rev["abs_magnitude_diff"].median()) if len(sign_rev) else None,
    }
    with open(out / "rank1_sign_reversal_summary_0p25.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 可視化: bin別 count / ratio
    plot_df = by_query_bin.copy()
    # interval文字列から中心値を復元しやすいように、元のgroupから再作成
    tmp = sign_rev.groupby("query_value_bin_0p25", observed=True).size().reset_index(name="dummy")
    tmp["bin_center"] = tmp["query_value_bin_0p25"].map(interval_mid)
    tmp["query_value_bin_0p25"] = tmp["query_value_bin_0p25"].astype(str)
    plot_df = plot_df.merge(tmp[["query_value_bin_0p25", "bin_center"]], on="query_value_bin_0p25", how="left")

    plt.figure(figsize=(10, 5))
    plt.bar(plot_df["query_value_bin_0p25"], plot_df["sign_reversal_count"])
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("query value bin, width=0.25")
    plt.ylabel("count")
    plt.title("Rank1 sign-reversal count by query value bin")
    plt.tight_layout()
    plt.savefig(out / "rank1_sign_reversal_count_by_query_bin_0p25.png", dpi=220)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.bar(plot_df["query_value_bin_0p25"], plot_df["sign_reversal_ratio_among_rank1"])
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("query value bin, width=0.25")
    plt.ylabel("ratio among rank1")
    plt.title("Rank1 sign-reversal ratio by query value bin")
    plt.tight_layout()
    plt.savefig(out / "rank1_sign_reversal_ratio_by_query_bin_0p25.png", dpi=220)
    plt.close()

    plt.figure(figsize=(8, 7))
    plt.imshow(matrix.values, aspect="auto", origin="lower")
    plt.colorbar(label="count")
    plt.xticks(np.arange(len(matrix.columns)), matrix.columns, rotation=90)
    plt.yticks(np.arange(len(matrix.index)), matrix.index)
    plt.xlabel("neighbor value bin")
    plt.ylabel("query value bin")
    plt.title("Rank1 sign-reversal query-neighbor bin matrix")
    plt.tight_layout()
    plt.savefig(out / "rank1_sign_reversal_query_neighbor_bin_matrix_0p25.png", dpi=220)
    plt.close()

    print("Saved to:", out)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nBy query bin:")
    print(by_query_bin.to_string(index=False))


if __name__ == "__main__":
    main()
