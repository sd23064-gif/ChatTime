#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_nearest_added_tokens.py

通常学習 Mamba の nearest CSV と、数値距離ペナルティ後の nearest CSV を比較する。

例:
python compare_nearest_added_tokens.py \
  --baseline_csv nearest_added_tokens.csv \
  --reg_csv /workspace/outputs/analysis/chattime_nearest_added_tokens_after_numeric_reg.csv \
  --output_dir outputs/compare_nearest_baseline_vs_numeric_reg \
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

NUMERIC_TOKEN_RE = re.compile(r"###([+-]?(?:\d+(?:\.\d*)?|\.\d+)|Nan|NaN|nan)###")


def parse_value(x):
    if pd.isna(x):
        return np.nan
    m = NUMERIC_TOKEN_RE.fullmatch(str(x))
    if m is None:
        return np.nan
    s = m.group(1)
    if s.lower() == "nan":
        return np.nan
    return float(s)


def safe_suffix(x):
    return f"{x:.2f}".replace(".", "p")


def make_edges(min_value, max_value, bin_width):
    edges = np.arange(min_value, max_value + 1e-9, bin_width)
    if edges[-1] < max_value:
        edges = np.append(edges, max_value)
    return np.unique(np.round(edges, 10))


def load_rank1_numeric(csv_path, threshold):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.astype(str).str.replace("\ufeff", "", regex=False).str.strip()

    if "rank" not in df.columns:
        raise ValueError(f"rank column not found in {csv_path}: {df.columns.tolist()}")
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce")

    if "query_value" not in df.columns:
        if "query_token_repr" not in df.columns:
            raise ValueError(f"query_value/query_token_repr not found in {csv_path}")
        df["query_value"] = df["query_token_repr"].map(parse_value)
    else:
        df["query_value"] = pd.to_numeric(df["query_value"], errors="coerce")

    if "neighbor_value" not in df.columns:
        if "neighbor_token_repr" not in df.columns:
            raise ValueError(f"neighbor_value/neighbor_token_repr not found in {csv_path}")
        df["neighbor_value"] = df["neighbor_token_repr"].map(parse_value)
    else:
        df["neighbor_value"] = pd.to_numeric(df["neighbor_value"], errors="coerce")

    if "cosine_similarity" in df.columns:
        df["cosine_similarity"] = pd.to_numeric(df["cosine_similarity"], errors="coerce")

    if "abs_value_diff" not in df.columns:
        df["abs_value_diff"] = (df["query_value"] - df["neighbor_value"]).abs()
    else:
        df["abs_value_diff"] = pd.to_numeric(df["abs_value_diff"], errors="coerce")

    r1 = df[(df["rank"] == 1) & df["query_value"].notna() & df["neighbor_value"].notna()].copy()
    r1["sign_reversed"] = r1["query_value"] * r1["neighbor_value"] < 0
    r1["abs_magnitude_diff"] = (r1["query_value"].abs() - r1["neighbor_value"].abs()).abs()
    r1["far_ge_threshold"] = r1["abs_value_diff"] >= threshold
    return r1


def summarize_r1(name, r1, threshold):
    return {
        "model": name,
        "rank1_numeric_count": int(len(r1)),
        "rank1_abs_value_diff_mean": float(r1["abs_value_diff"].mean()),
        "rank1_abs_value_diff_median": float(r1["abs_value_diff"].median()),
        f"rank1_abs_value_diff_ge_{safe_suffix(threshold)}_count": int((r1["abs_value_diff"] >= threshold).sum()),
        f"rank1_abs_value_diff_ge_{safe_suffix(threshold)}_ratio": float((r1["abs_value_diff"] >= threshold).mean()),
        "rank1_sign_reversal_count": int(r1["sign_reversed"].sum()),
        "rank1_sign_reversal_ratio": float(r1["sign_reversed"].mean()),
        "rank1_abs_magnitude_diff_mean": float(r1["abs_magnitude_diff"].mean()),
        "rank1_abs_magnitude_diff_median": float(r1["abs_magnitude_diff"].median()),
        "rank1_cosine_similarity_mean": float(r1["cosine_similarity"].mean()) if "cosine_similarity" in r1.columns else np.nan,
        "rank1_cosine_similarity_median": float(r1["cosine_similarity"].median()) if "cosine_similarity" in r1.columns else np.nan,
    }


def by_bin_summary(model_name, r1, threshold, bin_width):
    edges = make_edges(-1.0, 1.0, bin_width)
    bin_col = f"query_bin_{safe_suffix(bin_width)}"
    r1 = r1.copy()
    r1[bin_col] = pd.cut(r1["query_value"], bins=edges, include_lowest=True, duplicates="drop")
    out = r1.groupby(bin_col, observed=True).agg(
        rank1_total_count=("query_value", "size"),
        far_count=("far_ge_threshold", "sum"),
        far_ratio=("far_ge_threshold", "mean"),
        sign_reversal_count=("sign_reversed", "sum"),
        sign_reversal_ratio=("sign_reversed", "mean"),
        abs_value_diff_mean=("abs_value_diff", "mean"),
        abs_value_diff_median=("abs_value_diff", "median"),
        cosine_similarity_mean=("cosine_similarity", "mean") if "cosine_similarity" in r1.columns else ("query_value", "mean"),
    ).reset_index()
    out.insert(0, "model", model_name)
    out[bin_col] = out[bin_col].astype(str)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_csv", type=str, required=True)
    parser.add_argument("--reg_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/compare_nearest_baseline_vs_numeric_reg")
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--bin_width", type=float, default=0.1)
    parser.add_argument("--baseline_name", type=str, default="baseline")
    parser.add_argument("--reg_name", type=str, default="numeric_reg")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    baseline = load_rank1_numeric(args.baseline_csv, args.threshold)
    reg = load_rank1_numeric(args.reg_csv, args.threshold)

    summary_df = pd.DataFrame([
        summarize_r1(args.baseline_name, baseline, args.threshold),
        summarize_r1(args.reg_name, reg, args.threshold),
    ])

    # improvement row
    base = summary_df.iloc[0]
    new = summary_df.iloc[1]
    ge_col = f"rank1_abs_value_diff_ge_{safe_suffix(args.threshold)}_ratio"
    imp = {
        "metric": "improvement_numeric_reg_minus_baseline",
        "abs_value_diff_mean_delta": float(new["rank1_abs_value_diff_mean"] - base["rank1_abs_value_diff_mean"]),
        "abs_value_diff_median_delta": float(new["rank1_abs_value_diff_median"] - base["rank1_abs_value_diff_median"]),
        f"ge_{safe_suffix(args.threshold)}_ratio_delta": float(new[ge_col] - base[ge_col]),
        f"ge_{safe_suffix(args.threshold)}_ratio_relative_change": float((new[ge_col] - base[ge_col]) / base[ge_col]) if base[ge_col] != 0 else np.nan,
        "sign_reversal_ratio_delta": float(new["rank1_sign_reversal_ratio"] - base["rank1_sign_reversal_ratio"]),
    }

    summary_df.to_csv(out / "nearest_comparison_summary.csv", index=False)
    pd.Series(imp).to_json(out / "nearest_comparison_improvement.json", force_ascii=False, indent=2)

    bybin = pd.concat([
        by_bin_summary(args.baseline_name, baseline, args.threshold, args.bin_width),
        by_bin_summary(args.reg_name, reg, args.threshold, args.bin_width),
    ], ignore_index=True)
    bybin.to_csv(out / f"nearest_comparison_by_query_bin_{safe_suffix(args.bin_width)}.csv", index=False)

    # plots
    for metric in ["far_ratio", "abs_value_diff_mean", "sign_reversal_ratio"]:
        plt.figure(figsize=(13, 5))
        bin_col = f"query_bin_{safe_suffix(args.bin_width)}"
        for model_name, g in bybin.groupby("model"):
            plt.plot(g[bin_col], g[metric], marker="o", linewidth=1, label=model_name)
        plt.xticks(rotation=45, ha="right")
        plt.xlabel(f"query value bin, width={args.bin_width}")
        plt.ylabel(metric)
        plt.title(f"Baseline vs numeric regularized: {metric}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out / f"compare_{metric}_by_query_bin_{safe_suffix(args.bin_width)}.png", dpi=220)
        plt.close()

    # hist abs diff
    plt.figure(figsize=(10, 5))
    bins = np.linspace(0, 2, 101)
    plt.hist(baseline["abs_value_diff"], bins=bins, alpha=0.5, label=args.baseline_name, density=True)
    plt.hist(reg["abs_value_diff"], bins=bins, alpha=0.5, label=args.reg_name, density=True)
    plt.axvline(args.threshold, linestyle="--", linewidth=1)
    plt.xlabel("rank1 abs_value_diff")
    plt.ylabel("density")
    plt.title("Distribution of rank1 numerical distance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / f"compare_abs_value_diff_hist_threshold_{safe_suffix(args.threshold)}.png", dpi=220)
    plt.close()

    print("Saved to", out)
    print(summary_df.to_string(index=False))
    print(json.dumps(imp, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
