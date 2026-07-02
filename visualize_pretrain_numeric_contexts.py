#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualize_pretrain_numeric_contexts.py

analyze_pretrain_numeric_token_contexts_fast.py が出力した CSV を可視化するスクリプト。

想定入力ディレクトリ:
  outputs/pretrain_numeric_token_contexts_fast_all/
    - pretrain_numeric_token_context_stats_by_token.csv
    - pretrain_numeric_token_context_summary_by_value_bin.csv
    - transition_matrix_current_to_next.csv
    - transition_matrix_current_to_next_counts.csv
    - numeric_token_count_per_sample.csv

出力:
  指標別 line/scatter、2D scatter、相関ヒートマップ、transition heatmap、頻度重み付き図など。

実行例:
  python visualize_pretrain_numeric_contexts.py \
    --input_dir outputs/pretrain_numeric_token_contexts_fast_all \
    --output_dir outputs/pretrain_numeric_token_contexts_visuals \
    --min_count 50
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def safe_read_csv(path):
    path = Path(path)
    if not path.exists():
        print(f"[WARN] missing: {path}")
        return None
    return pd.read_csv(path)


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def add_derived_columns(df):
    df = df.copy()
    df["abs_token_value"] = df["token_value"].abs()
    df["sign"] = np.sign(df["token_value"])
    df["log_count"] = np.log10(df["count"].clip(lower=1))

    # 左右の平均変化量。値が大きいほど、前後どちらかで大きく動く文脈に出やすい。
    if {"mean_abs_prev_delta", "mean_abs_next_delta"}.issubset(df.columns):
        df["mean_abs_neighbor_delta"] = (
            df["mean_abs_prev_delta"] + df["mean_abs_next_delta"]
        ) / 2

    # 上昇寄り / 下降寄り。正なら上昇方向、負なら下降方向。
    if {"mean_prev_delta", "mean_next_delta"}.issubset(df.columns):
        df["mean_direction_delta"] = (
            df["mean_prev_delta"] + df["mean_next_delta"]
        ) / 2

    # peak と valley の差。正なら peak 寄り、負なら valley 寄り。
    if {"peak_ratio", "valley_ratio"}.issubset(df.columns):
        df["peak_minus_valley_ratio"] = df["peak_ratio"] - df["valley_ratio"]

    return df


def filter_df(df, min_count):
    return df[df["count"] >= min_count].copy().sort_values("token_value")


def plot_metric_by_value(df, y, out, title=None, ylabel=None, min_count=50, rolling=1):
    plot_df = filter_df(df, min_count)
    if len(plot_df) == 0 or y not in plot_df.columns:
        return

    x = plot_df["token_value"].to_numpy()
    yy = plot_df[y].to_numpy()

    if rolling and rolling > 1:
        yy = pd.Series(yy).rolling(rolling, center=True, min_periods=1).mean().to_numpy()

    plt.figure(figsize=(12, 5))
    plt.plot(x, yy, linewidth=1)
    plt.axvline(0, linestyle="--", linewidth=1)
    plt.xlabel("token value")
    plt.ylabel(ylabel or y)
    plt.title(title or f"{y} by token value")
    plt.tight_layout()
    plt.savefig(out, dpi=220)
    plt.close()


def plot_metric_scatter(df, y, out, title=None, ylabel=None, min_count=50):
    plot_df = filter_df(df, min_count)
    if len(plot_df) == 0 or y not in plot_df.columns:
        return

    sizes = 8 + 45 * (plot_df["log_count"] - plot_df["log_count"].min()) / max(
        1e-12, plot_df["log_count"].max() - plot_df["log_count"].min()
    )

    plt.figure(figsize=(10, 5))
    plt.scatter(
        plot_df["token_value"],
        plot_df[y],
        s=sizes,
        alpha=0.65,
    )
    plt.axvline(0, linestyle="--", linewidth=1)
    plt.xlabel("token value")
    plt.ylabel(ylabel or y)
    plt.title(title or f"{y} by token value; size=log_count")
    plt.tight_layout()
    plt.savefig(out, dpi=220)
    plt.close()


def plot_count_distribution(df, out, min_count=1):
    plot_df = filter_df(df, min_count)
    if len(plot_df) == 0:
        return

    plt.figure(figsize=(12, 5))
    plt.plot(plot_df["token_value"], plot_df["count"], linewidth=1)
    plt.yscale("log")
    plt.axvline(0, linestyle="--", linewidth=1)
    plt.xlabel("token value")
    plt.ylabel("count, log scale")
    plt.title("Numeric token frequency by value")
    plt.tight_layout()
    plt.savefig(out, dpi=220)
    plt.close()


def plot_context_map(df, x_col, y_col, color_col, out, title, min_count=50):
    plot_df = filter_df(df, min_count)
    required = {x_col, y_col, color_col}
    if len(plot_df) == 0 or not required.issubset(plot_df.columns):
        return

    sizes = 6 + 35 * (plot_df["log_count"] - plot_df["log_count"].min()) / max(
        1e-12, plot_df["log_count"].max() - plot_df["log_count"].min()
    )

    plt.figure(figsize=(8, 6))
    sc = plt.scatter(
        plot_df[x_col],
        plot_df[y_col],
        c=plot_df[color_col],
        s=sizes,
        alpha=0.72,
    )
    plt.colorbar(sc, label=color_col)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out, dpi=220)
    plt.close()


def plot_corr_heatmap(df, out, min_count=50):
    plot_df = filter_df(df, min_count)
    cols = [
        "token_value",
        "abs_token_value",
        "log_count",
        "local_mean_abs_diff_mean",
        "local_std_mean",
        "local_range_mean",
        "flat_ratio",
        "increasing_ratio",
        "decreasing_ratio",
        "peak_ratio",
        "valley_ratio",
        "turning_point_ratio",
        "near_sign_change_ratio",
        "local_slope_mean",
        "peak_minus_valley_ratio",
        "mean_abs_neighbor_delta",
        "mean_direction_delta",
    ]
    cols = [c for c in cols if c in plot_df.columns]
    if len(cols) < 2:
        return

    corr = plot_df[cols].corr(numeric_only=True)
    corr.to_csv(out.with_suffix(".csv"))

    plt.figure(figsize=(12, 10))
    im = plt.imshow(corr.values, vmin=-1, vmax=1, cmap="coolwarm")
    plt.colorbar(im, label="correlation")
    plt.xticks(np.arange(len(cols)), cols, rotation=90)
    plt.yticks(np.arange(len(cols)), cols)
    plt.title("Correlation between token context features")
    plt.tight_layout()
    plt.savefig(out, dpi=220)
    plt.close()


def plot_transition_heatmap(matrix_csv, out_png, title):
    matrix_csv = Path(matrix_csv)
    if not matrix_csv.exists():
        return

    mat = pd.read_csv(matrix_csv, index_col=0)
    values = mat.values.astype(float)

    plt.figure(figsize=(10, 8))
    im = plt.imshow(values, aspect="auto", origin="lower")
    plt.colorbar(im, label="P(next bin | current bin)")
    plt.xlabel("next value bin")
    plt.ylabel("current value bin")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


def plot_sample_token_counts(sample_df, out_dir):
    if sample_df is None or len(sample_df) == 0:
        return

    if "n_numeric_tokens" not in sample_df.columns:
        return

    plt.figure(figsize=(10, 5))
    plt.hist(sample_df["n_numeric_tokens"], bins=50)
    plt.xlabel("numeric tokens per row")
    plt.ylabel("row count")
    plt.title("Distribution of numeric token count per sample")
    plt.tight_layout()
    plt.savefig(out_dir / "sample_numeric_token_count_hist.png", dpi=220)
    plt.close()

    cols = [c for c in ["mean_value", "std_value", "min_value", "max_value"] if c in sample_df.columns]
    for c in cols:
        plt.figure(figsize=(10, 5))
        plt.hist(sample_df[c].dropna(), bins=60)
        plt.xlabel(c)
        plt.ylabel("row count")
        plt.title(f"Distribution of per-row {c}")
        plt.tight_layout()
        plt.savefig(out_dir / f"sample_{c}_hist.png", dpi=220)
        plt.close()


def export_top_tables(df, out_dir, min_count=50, top_n=100):
    plot_df = filter_df(df, min_count)
    targets = [
        ("top_flat_tokens.csv", "flat_ratio", False),
        ("top_volatile_tokens.csv", "local_mean_abs_diff_mean", False),
        ("top_turning_point_tokens.csv", "turning_point_ratio", False),
        ("top_near_sign_change_tokens.csv", "near_sign_change_ratio", False),
        ("top_peak_tokens.csv", "peak_ratio", False),
        ("top_valley_tokens.csv", "valley_ratio", False),
        ("top_frequent_tokens.csv", "count", False),
        ("lowest_flat_tokens.csv", "flat_ratio", True),
    ]

    base_cols = [
        "token",
        "token_value",
        "count",
        "flat_ratio",
        "local_mean_abs_diff_mean",
        "local_std_mean",
        "turning_point_ratio",
        "peak_ratio",
        "valley_ratio",
        "near_sign_change_ratio",
        "most_common_prev_token",
        "most_common_next_token",
    ]
    base_cols = [c for c in base_cols if c in plot_df.columns]

    for filename, col, ascending in targets:
        if col not in plot_df.columns:
            continue
        plot_df.sort_values(col, ascending=ascending)[base_cols].head(top_n).to_csv(
            out_dir / filename,
            index=False,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--min_count", type=int, default=50)
    parser.add_argument("--rolling", type=int, default=1)
    parser.add_argument("--top_n", type=int, default=100)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = ensure_dir(args.output_dir or input_dir / "visuals")

    token_csv = input_dir / "pretrain_numeric_token_context_stats_by_token.csv"
    bin_csv = input_dir / "pretrain_numeric_token_context_summary_by_value_bin.csv"
    sample_csv = input_dir / "numeric_token_count_per_sample.csv"
    transition_csv = input_dir / "transition_matrix_current_to_next.csv"

    token_df = safe_read_csv(token_csv)
    if token_df is None:
        raise FileNotFoundError(f"required file not found: {token_csv}")

    token_df = add_derived_columns(token_df)
    token_df.to_csv(output_dir / "token_stats_with_derived_columns.csv", index=False)

    bin_df = safe_read_csv(bin_csv)
    sample_df = safe_read_csv(sample_csv)

    export_top_tables(token_df, output_dir, min_count=args.min_count, top_n=args.top_n)

    plot_count_distribution(token_df, output_dir / "01_token_frequency_log.png", min_count=1)

    metrics = [
        ("local_mean_abs_diff_mean", "mean local absolute change", "02_local_mean_abs_diff_by_value.png"),
        ("local_std_mean", "mean local std", "03_local_std_by_value.png"),
        ("local_range_mean", "mean local range", "04_local_range_by_value.png"),
        ("flat_ratio", "flat ratio", "05_flat_ratio_by_value.png"),
        ("turning_point_ratio", "turning point ratio", "06_turning_point_ratio_by_value.png"),
        ("peak_ratio", "peak ratio", "07_peak_ratio_by_value.png"),
        ("valley_ratio", "valley ratio", "08_valley_ratio_by_value.png"),
        ("near_sign_change_ratio", "near sign-change ratio", "09_near_sign_change_ratio_by_value.png"),
        ("local_slope_mean", "mean local slope", "10_local_slope_by_value.png"),
        ("peak_minus_valley_ratio", "peak ratio - valley ratio", "11_peak_minus_valley_by_value.png"),
        ("mean_abs_neighbor_delta", "mean abs neighbor delta", "12_mean_abs_neighbor_delta_by_value.png"),
        ("mean_direction_delta", "mean direction delta", "13_mean_direction_delta_by_value.png"),
    ]

    for col, ylabel, filename in metrics:
        if col in token_df.columns:
            plot_metric_by_value(
                token_df,
                col,
                output_dir / filename,
                ylabel=ylabel,
                title=f"{ylabel} by token value",
                min_count=args.min_count,
                rolling=args.rolling,
            )
            plot_metric_scatter(
                token_df,
                col,
                output_dir / filename.replace(".png", "_scatter.png"),
                ylabel=ylabel,
                title=f"{ylabel} by token value; point size=frequency",
                min_count=args.min_count,
            )

    # 2D context maps
    plot_context_map(
        token_df,
        "flat_ratio",
        "local_mean_abs_diff_mean",
        "token_value",
        output_dir / "20_flat_vs_volatility_colored_by_value.png",
        "Flat ratio vs local change; color=token value",
        min_count=args.min_count,
    )

    plot_context_map(
        token_df,
        "turning_point_ratio",
        "near_sign_change_ratio",
        "abs_token_value",
        output_dir / "21_turning_vs_signchange_colored_by_abs_value.png",
        "Turning point ratio vs sign-change ratio; color=|value|",
        min_count=args.min_count,
    )

    plot_context_map(
        token_df,
        "peak_ratio",
        "valley_ratio",
        "token_value",
        output_dir / "22_peak_vs_valley_colored_by_value.png",
        "Peak ratio vs valley ratio; color=token value",
        min_count=args.min_count,
    )

    plot_corr_heatmap(token_df, output_dir / "30_feature_correlation_heatmap.png", min_count=args.min_count)
    plot_transition_heatmap(transition_csv, output_dir / "40_transition_heatmap.png", "Transition heatmap: current token bin -> next token bin")
    plot_sample_token_counts(sample_df, output_dir)

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "n_tokens_total": int(len(token_df)),
        "n_tokens_min_count": int((token_df["count"] >= args.min_count).sum()),
        "min_count": args.min_count,
        "rolling": args.rolling,
        "top_n": args.top_n,
    }
    with open(output_dir / "visualization_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Saved visualizations to:", output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
