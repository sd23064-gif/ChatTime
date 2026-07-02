#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_rank1_value_distance_bins.py

chattime_nearest_added_tokens.csv 形式のCSVから、rank1 neighbor のうち
数値距離 |query_value - neighbor_value| が threshold 以上のものを抽出し、
query_value を bin_width 幅で区切って集計する。

例:
  python analyze_rank1_value_distance_bins.py \
    --input_csv chattime_nearest_added_tokens.csv \
    --output_dir outputs/rank1_value_distance_ge_0p10 \
    --threshold 0.1 \
    --bin_width 0.1
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def normalize_columns(df):
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
    )
    return df


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
    df = normalize_columns(df)

    if "query_value" in df.columns:
        df["query_value"] = pd.to_numeric(df["query_value"], errors="coerce")
    else:
        candidate_cols = ["query_token_repr", "query_token", "query"]
        col = next((c for c in candidate_cols if c in df.columns), None)
        if col is None:
            raise ValueError(f"query_value/query_token列が見つかりません。columns={df.columns.tolist()}")
        df["query_value"] = df[col].map(parse_value_from_token_repr)

    if "neighbor_value" in df.columns:
        df["neighbor_value"] = pd.to_numeric(df["neighbor_value"], errors="coerce")
    else:
        candidate_cols = ["neighbor_token_repr", "neighbor_token", "neighbor"]
        col = next((c for c in candidate_cols if c in df.columns), None)
        if col is None:
            raise ValueError(f"neighbor_value/neighbor_token列が見つかりません。columns={df.columns.tolist()}")
        df["neighbor_value"] = df[col].map(parse_value_from_token_repr)

    if "rank" not in df.columns:
        raise ValueError(f"rank列が見つかりません。columns={df.columns.tolist()}")
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce")

    if "cosine_similarity" in df.columns:
        df["cosine_similarity"] = pd.to_numeric(df["cosine_similarity"], errors="coerce")

    if "abs_value_diff" in df.columns:
        df["abs_value_diff"] = pd.to_numeric(df["abs_value_diff"], errors="coerce")
    else:
        df["abs_value_diff"] = (df["query_value"] - df["neighbor_value"]).abs()

    return df


def make_edges(min_value, max_value, bin_width):
    edges = np.arange(min_value, max_value + 1e-9, bin_width)
    if len(edges) == 0 or edges[0] > min_value:
        edges = np.insert(edges, 0, min_value)
    if edges[-1] < max_value:
        edges = np.append(edges, max_value)
    return np.unique(np.round(edges, 10))


def safe_suffix(x):
    return f"{x:.2f}".replace(".", "p")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/rank1_value_distance_ge_0p10")
    parser.add_argument("--threshold", type=float, default=0.1, help="|query_value - neighbor_value| のしきい値")
    parser.add_argument("--bin_width", type=float, default=0.1, help="query_value のbin幅")
    parser.add_argument("--min_value", type=float, default=-1.0)
    parser.add_argument("--max_value", type=float, default=1.0)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    suffix = safe_suffix(args.bin_width)
    threshold_suffix = safe_suffix(args.threshold)

    df = pd.read_csv(args.input_csv)
    df = add_numeric_columns(df)

    r1 = df[
        (df["rank"] == 1)
        & df["query_value"].notna()
        & df["neighbor_value"].notna()
        & df["abs_value_diff"].notna()
    ].copy()

    r1["is_far_by_value"] = r1["abs_value_diff"] >= args.threshold
    r1["sign_reversed"] = r1["query_value"] * r1["neighbor_value"] < 0
    r1["abs_query_value"] = r1["query_value"].abs()
    r1["abs_neighbor_value"] = r1["neighbor_value"].abs()
    r1["abs_magnitude_diff"] = (r1["abs_query_value"] - r1["abs_neighbor_value"]).abs()

    far = r1[r1["is_far_by_value"]].copy()

    edges = make_edges(args.min_value, args.max_value, args.bin_width)
    abs_edges = make_edges(0.0, max(abs(args.min_value), abs(args.max_value)), args.bin_width)

    query_bin_col = f"query_value_bin_{suffix}"
    neighbor_bin_col = f"neighbor_value_bin_{suffix}"
    abs_bin_col = f"abs_query_value_bin_{suffix}"

    r1[query_bin_col] = pd.cut(r1["query_value"], bins=edges, include_lowest=True, duplicates="drop")
    far[query_bin_col] = pd.cut(far["query_value"], bins=edges, include_lowest=True, duplicates="drop")
    far[neighbor_bin_col] = pd.cut(far["neighbor_value"], bins=edges, include_lowest=True, duplicates="drop")
    far[abs_bin_col] = pd.cut(far["abs_query_value"], bins=abs_edges, include_lowest=True, duplicates="drop")

    denom = r1.groupby(query_bin_col, observed=True).size().rename("rank1_total_count")

    agg = {
        "far_count": ("is_far_by_value", "size"),
        "query_value_mean": ("query_value", "mean"),
        "neighbor_value_mean": ("neighbor_value", "mean"),
        "abs_value_diff_mean": ("abs_value_diff", "mean"),
        "abs_value_diff_median": ("abs_value_diff", "median"),
        "abs_value_diff_min": ("abs_value_diff", "min"),
        "abs_value_diff_max": ("abs_value_diff", "max"),
        "sign_reversal_count_in_far": ("sign_reversed", "sum"),
        "sign_reversal_ratio_in_far": ("sign_reversed", "mean"),
        "abs_magnitude_diff_mean": ("abs_magnitude_diff", "mean"),
        "abs_magnitude_diff_median": ("abs_magnitude_diff", "median"),
    }
    if "cosine_similarity" in far.columns:
        agg.update({
            "cosine_mean": ("cosine_similarity", "mean"),
            "cosine_median": ("cosine_similarity", "median"),
        })

    by_query_bin = far.groupby(query_bin_col, observed=True).agg(**agg).reset_index()
    by_query_bin = by_query_bin.merge(denom.reset_index(), on=query_bin_col, how="left")
    by_query_bin["far_ratio_among_rank1"] = by_query_bin["far_count"] / by_query_bin["rank1_total_count"]
    by_query_bin[query_bin_col] = by_query_bin[query_bin_col].astype(str)
    by_query_bin.to_csv(out / f"rank1_abs_value_diff_ge_{threshold_suffix}_by_query_bin_{suffix}.csv", index=False)

    by_abs_bin = far.groupby(abs_bin_col, observed=True).agg(**agg).reset_index()
    by_abs_bin[abs_bin_col] = by_abs_bin[abs_bin_col].astype(str)
    by_abs_bin.to_csv(out / f"rank1_abs_value_diff_ge_{threshold_suffix}_by_abs_query_bin_{suffix}.csv", index=False)

    matrix = pd.crosstab(
        far[query_bin_col].astype(str),
        far[neighbor_bin_col].astype(str),
    )
    matrix.to_csv(out / f"rank1_abs_value_diff_ge_{threshold_suffix}_query_neighbor_bin_matrix_{suffix}.csv")

    save_cols = [
        c for c in [
            "query_token", "query_token_repr", "query_token_id", "query_value",
            "rank",
            "neighbor_token", "neighbor_token_repr", "neighbor_token_id", "neighbor_value",
            "cosine_similarity", "abs_value_diff", "abs_magnitude_diff", "sign_reversed",
            query_bin_col, neighbor_bin_col, abs_bin_col,
        ] if c in far.columns
    ]
    pairs = far[save_cols].copy()
    for c in [query_bin_col, neighbor_bin_col, abs_bin_col]:
        if c in pairs.columns:
            pairs[c] = pairs[c].astype(str)
    pairs.to_csv(out / f"rank1_abs_value_diff_ge_{threshold_suffix}_pairs.csv", index=False)

    summary = {
        "input_csv": args.input_csv,
        "threshold": args.threshold,
        "bin_width": args.bin_width,
        "rank1_numeric_count": int(len(r1)),
        "rank1_far_count": int(len(far)),
        "rank1_far_ratio": float(len(far) / len(r1)) if len(r1) else None,
        "far_sign_reversal_count": int(far["sign_reversed"].sum()) if len(far) else 0,
        "far_sign_reversal_ratio": float(far["sign_reversed"].mean()) if len(far) else None,
        "mean_abs_value_diff_far": float(far["abs_value_diff"].mean()) if len(far) else None,
        "median_abs_value_diff_far": float(far["abs_value_diff"].median()) if len(far) else None,
    }
    with open(out / f"rank1_abs_value_diff_ge_{threshold_suffix}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # plots
    plot_df = by_query_bin.copy()
    plt.figure(figsize=(12, 5))
    plt.bar(plot_df[query_bin_col], plot_df["far_count"])
    plt.xticks(rotation=45, ha="right")
    plt.xlabel(f"query value bin, width={args.bin_width}")
    plt.ylabel(f"count, abs_value_diff >= {args.threshold}")
    plt.title(f"Rank1 pairs with abs value diff >= {args.threshold} by query bin")
    plt.tight_layout()
    plt.savefig(out / f"rank1_abs_value_diff_ge_{threshold_suffix}_count_by_query_bin_{suffix}.png", dpi=220)
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.bar(plot_df[query_bin_col], plot_df["far_ratio_among_rank1"])
    plt.xticks(rotation=45, ha="right")
    plt.xlabel(f"query value bin, width={args.bin_width}")
    plt.ylabel("ratio among rank1")
    plt.title(f"Ratio of rank1 pairs with abs value diff >= {args.threshold}")
    plt.tight_layout()
    plt.savefig(out / f"rank1_abs_value_diff_ge_{threshold_suffix}_ratio_by_query_bin_{suffix}.png", dpi=220)
    plt.close()

    plt.figure(figsize=(10, 8))
    plt.imshow(matrix.values, aspect="auto", origin="lower")
    plt.colorbar(label="count")
    plt.xticks(np.arange(len(matrix.columns)), matrix.columns, rotation=90)
    plt.yticks(np.arange(len(matrix.index)), matrix.index)
    plt.xlabel("neighbor value bin")
    plt.ylabel("query value bin")
    plt.title(f"Rank1 pairs with abs value diff >= {args.threshold}: query-neighbor bins")
    plt.tight_layout()
    plt.savefig(out / f"rank1_abs_value_diff_ge_{threshold_suffix}_query_neighbor_bin_matrix_{suffix}.png", dpi=220)
    plt.close()

    print("Saved to:", out)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nBy query bin:")
    print(by_query_bin.to_string(index=False))


if __name__ == "__main__":
    main()
