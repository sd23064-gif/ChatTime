#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merge_context_with_pca.py

数値トークンの「学習データ内の局所文脈特徴」と「embedding PCA座標」を結合して可視化するスクリプト。

入力例:
  --context_csv outputs/pretrain_numeric_token_contexts_fast_all/pretrain_numeric_token_context_stats_by_token.csv
  --pca_csv outputs/chattime_embedding_vis/chattime_embedding_pca_coordinates.csv

出力:
  - token_context_pca_merged.csv
  - context_pca_correlation.csv
  - pc_context_feature_correlation.csv
  - PCA上で文脈特徴量を色付けした図
  - PC1/PC2 と文脈特徴量の関係図
  - +a/-a ペアの文脈特徴差分とPCA距離の表/図

実行例:
python merge_context_with_pca.py \
  --context_csv outputs/pretrain_numeric_token_contexts_fast_all/pretrain_numeric_token_context_stats_by_token.csv \
  --pca_csv outputs/chattime_embedding_vis/chattime_embedding_pca_coordinates.csv \
  --output_dir outputs/context_pca_merged \
  --min_count 50
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def add_context_derived_columns(df):
    df = df.copy()

    if "token_value" in df.columns:
        df["abs_token_value"] = df["token_value"].abs()
        df["sign"] = np.sign(df["token_value"])

    if "count" in df.columns:
        df["log_count"] = np.log10(df["count"].clip(lower=1))

    if {"mean_abs_prev_delta", "mean_abs_next_delta"}.issubset(df.columns):
        df["mean_abs_neighbor_delta"] = (
            df["mean_abs_prev_delta"] + df["mean_abs_next_delta"]
        ) / 2

    if {"mean_prev_delta", "mean_next_delta"}.issubset(df.columns):
        df["mean_direction_delta"] = (
            df["mean_prev_delta"] + df["mean_next_delta"]
        ) / 2

    if {"peak_ratio", "valley_ratio"}.issubset(df.columns):
        df["peak_minus_valley_ratio"] = df["peak_ratio"] - df["valley_ratio"]

    return df


def normalize_join_value(series, decimals=4):
    return pd.to_numeric(series, errors="coerce").round(decimals)


def merge_context_pca(context_df, pca_df):
    context_df = add_context_derived_columns(context_df)
    pca_df = pca_df.copy()

    if "token_value" not in context_df.columns:
        raise ValueError("context_csv must contain token_value column")
    if "value" not in pca_df.columns:
        raise ValueError("pca_csv must contain value column")

    context_df["join_value"] = normalize_join_value(context_df["token_value"])
    pca_df["join_value"] = normalize_join_value(pca_df["value"])

    merged = context_df.merge(
        pca_df,
        on="join_value",
        how="inner",
        suffixes=("_context", "_pca"),
    )

    # 列名を使いやすく整理
    if "value" in merged.columns and "token_value" in merged.columns:
        pass

    return merged


def weighted_corr(x, y, w):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)

    mask = ~np.isnan(x) & ~np.isnan(y) & ~np.isnan(w) & (w > 0)
    if mask.sum() < 3:
        return np.nan

    x = x[mask]
    y = y[mask]
    w = w[mask]
    w = w / w.sum()

    mx = np.sum(w * x)
    my = np.sum(w * y)
    cov = np.sum(w * (x - mx) * (y - my))
    vx = np.sum(w * (x - mx) ** 2)
    vy = np.sum(w * (y - my) ** 2)

    if vx <= 0 or vy <= 0:
        return np.nan

    return float(cov / np.sqrt(vx * vy))


def save_correlation_tables(merged, out_dir, min_count):
    df = merged[merged["count"] >= min_count].copy() if "count" in merged.columns else merged.copy()

    candidate_cols = [
        "pc1", "pc2", "value", "abs_value", "sign_value",
        "token_value", "abs_token_value", "sign", "log_count",
        "local_mean_abs_diff_mean", "local_std_mean", "local_range_mean",
        "flat_ratio", "increasing_ratio", "decreasing_ratio",
        "peak_ratio", "valley_ratio", "turning_point_ratio",
        "near_sign_change_ratio", "local_slope_mean",
        "peak_minus_valley_ratio", "mean_abs_neighbor_delta", "mean_direction_delta",
        "amp_increasing_ratio", "amp_decreasing_ratio",
    ]
    cols = [c for c in candidate_cols if c in df.columns]

    corr = df[cols].corr(numeric_only=True)
    corr.to_csv(out_dir / "context_pca_correlation.csv")

    pc_cols = [c for c in ["pc1", "pc2"] if c in df.columns]
    context_cols = [c for c in cols if c not in ["pc1", "pc2"]]

    rows = []
    for pc in pc_cols:
        for c in context_cols:
            rows.append({
                "pc": pc,
                "feature": c,
                "pearson_corr": df[pc].corr(df[c]),
                "weighted_corr_by_count": weighted_corr(
                    df[pc], df[c], df["count"] if "count" in df.columns else np.ones(len(df))
                ),
                "n": int(df[[pc, c]].dropna().shape[0]),
            })

    pc_corr = pd.DataFrame(rows).sort_values(["pc", "pearson_corr"], ascending=[True, False])
    pc_corr.to_csv(out_dir / "pc_context_feature_correlation.csv", index=False)

    # heatmap
    if len(cols) > 1:
        plt.figure(figsize=(13, 11))
        im = plt.imshow(corr.values, vmin=-1, vmax=1, cmap="coolwarm")
        plt.colorbar(im, label="correlation")
        plt.xticks(np.arange(len(cols)), cols, rotation=90)
        plt.yticks(np.arange(len(cols)), cols)
        plt.title("Correlation: PCA coordinates and token context features")
        plt.tight_layout()
        plt.savefig(out_dir / "context_pca_correlation_heatmap.png", dpi=220)
        plt.close()

    return corr, pc_corr


def point_sizes(df):
    if "count" not in df.columns:
        return np.full(len(df), 8.0)
    log_count = np.log10(df["count"].clip(lower=1))
    denom = max(1e-12, log_count.max() - log_count.min())
    return 5 + 35 * (log_count - log_count.min()) / denom


def plot_pca_colored(merged, feature, out_dir, min_count):
    if feature not in merged.columns:
        return
    if "pc1" not in merged.columns or "pc2" not in merged.columns:
        return

    df = merged[merged["count"] >= min_count].copy() if "count" in merged.columns else merged.copy()
    df = df.dropna(subset=["pc1", "pc2", feature])
    if len(df) == 0:
        return

    plt.figure(figsize=(8, 6))
    sc = plt.scatter(
        df["pc1"],
        df["pc2"],
        c=df[feature],
        s=point_sizes(df),
        alpha=0.75,
    )
    plt.colorbar(sc, label=feature)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"PCA colored by {feature}; size=frequency")
    plt.tight_layout()
    plt.savefig(out_dir / f"pca_colored_by_{feature}.png", dpi=220)
    plt.close()


def plot_pc_vs_feature(merged, feature, out_dir, min_count):
    if feature not in merged.columns:
        return
    df = merged[merged["count"] >= min_count].copy() if "count" in merged.columns else merged.copy()
    for pc in ["pc1", "pc2"]:
        if pc not in df.columns:
            continue
        plot_df = df.dropna(subset=[pc, feature])
        if len(plot_df) == 0:
            continue
        plt.figure(figsize=(8, 5))
        plt.scatter(
            plot_df[feature],
            plot_df[pc],
            s=point_sizes(plot_df),
            alpha=0.65,
        )
        plt.xlabel(feature)
        plt.ylabel(pc)
        plt.title(f"{pc} vs {feature}; size=frequency")
        plt.tight_layout()
        plt.savefig(out_dir / f"{pc}_vs_{feature}.png", dpi=220)
        plt.close()


def plot_value_vs_feature_with_pc_color(merged, feature, out_dir, min_count, color_col="pc1"):
    if feature not in merged.columns or color_col not in merged.columns:
        return
    value_col = "token_value" if "token_value" in merged.columns else "value"
    if value_col not in merged.columns:
        return

    df = merged[merged["count"] >= min_count].copy() if "count" in merged.columns else merged.copy()
    df = df.dropna(subset=[value_col, feature, color_col])
    if len(df) == 0:
        return

    plt.figure(figsize=(10, 5))
    sc = plt.scatter(
        df[value_col],
        df[feature],
        c=df[color_col],
        s=point_sizes(df),
        alpha=0.70,
    )
    plt.colorbar(sc, label=color_col)
    plt.axvline(0, linestyle="--", linewidth=1)
    plt.xlabel("token value")
    plt.ylabel(feature)
    plt.title(f"{feature} by token value; color={color_col}, size=frequency")
    plt.tight_layout()
    plt.savefig(out_dir / f"value_vs_{feature}_colored_by_{color_col}.png", dpi=220)
    plt.close()


def build_sign_pair_table(merged):
    if "token_value" not in merged.columns:
        return pd.DataFrame()
    if "pc1" not in merged.columns or "pc2" not in merged.columns:
        return pd.DataFrame()

    df = merged.copy()
    df["rounded_value"] = df["token_value"].round(4)
    value_to_row = {row["rounded_value"]: row for _, row in df.iterrows()}

    feature_cols = [
        "count", "local_mean_abs_diff_mean", "local_std_mean", "flat_ratio",
        "turning_point_ratio", "near_sign_change_ratio", "peak_ratio", "valley_ratio",
        "local_slope_mean", "pc1", "pc2",
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]

    rows = []
    for v, pos_row in value_to_row.items():
        if not np.isfinite(v) or v <= 0:
            continue
        neg_v = round(-v, 4)
        if neg_v not in value_to_row:
            continue

        neg_row = value_to_row[neg_v]
        row = {
            "abs_value": abs(v),
            "pos_value": v,
            "neg_value": neg_v,
            "pos_token": pos_row.get("token", None),
            "neg_token": neg_row.get("token", None),
            "pca_distance": float(np.sqrt((pos_row["pc1"] - neg_row["pc1"]) ** 2 + (pos_row["pc2"] - neg_row["pc2"]) ** 2)),
            "pc1_diff_pos_minus_neg": float(pos_row["pc1"] - neg_row["pc1"]),
            "pc2_diff_pos_minus_neg": float(pos_row["pc2"] - neg_row["pc2"]),
        }

        for c in feature_cols:
            row[f"pos_{c}"] = pos_row[c]
            row[f"neg_{c}"] = neg_row[c]
            if c != "count":
                row[f"absdiff_{c}"] = abs(pos_row[c] - neg_row[c])

        rows.append(row)

    return pd.DataFrame(rows).sort_values("abs_value").reset_index(drop=True)


def plot_sign_pair_results(pair_df, out_dir):
    if pair_df is None or len(pair_df) == 0:
        return

    pair_df.to_csv(out_dir / "sign_reversal_pair_context_pca.csv", index=False)

    # +a/-a のPCA距離
    plt.figure(figsize=(10, 5))
    plt.plot(pair_df["abs_value"], pair_df["pca_distance"], linewidth=1)
    plt.xlabel("|token value|")
    plt.ylabel("PCA distance between +a and -a")
    plt.title("PCA distance of sign-reversal token pairs")
    plt.tight_layout()
    plt.savefig(out_dir / "sign_pair_pca_distance_by_abs_value.png", dpi=220)
    plt.close()

    # 文脈特徴差とPCA距離
    target_diffs = [
        "absdiff_local_mean_abs_diff_mean",
        "absdiff_local_std_mean",
        "absdiff_flat_ratio",
        "absdiff_turning_point_ratio",
        "absdiff_near_sign_change_ratio",
    ]
    for col in target_diffs:
        if col not in pair_df.columns:
            continue
        plt.figure(figsize=(7, 5))
        plt.scatter(pair_df[col], pair_df["pca_distance"], s=8, alpha=0.6)
        plt.xlabel(col)
        plt.ylabel("PCA distance between +a and -a")
        plt.title(f"Sign-pair PCA distance vs {col}")
        plt.tight_layout()
        plt.savefig(out_dir / f"sign_pair_pca_distance_vs_{col}.png", dpi=220)
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--context_csv", type=str, required=True)
    parser.add_argument("--pca_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/context_pca_merged")
    parser.add_argument("--min_count", type=int, default=50)
    parser.add_argument("--top_n", type=int, default=100)
    args = parser.parse_args()

    out_dir = ensure_dir(args.output_dir)

    context_df = pd.read_csv(args.context_csv)
    pca_df = pd.read_csv(args.pca_csv)

    merged = merge_context_pca(context_df, pca_df)
    merged_path = out_dir / "token_context_pca_merged.csv"
    merged.to_csv(merged_path, index=False)

    corr, pc_corr = save_correlation_tables(merged, out_dir, args.min_count)

    # PCA上で色付けしたい重要特徴
    color_features = [
        "token_value", "abs_token_value", "log_count",
        "local_mean_abs_diff_mean", "local_std_mean", "local_range_mean",
        "flat_ratio", "turning_point_ratio", "near_sign_change_ratio",
        "peak_ratio", "valley_ratio", "local_slope_mean",
        "peak_minus_valley_ratio", "mean_abs_neighbor_delta", "mean_direction_delta",
    ]
    for f in color_features:
        plot_pca_colored(merged, f, out_dir, args.min_count)
        plot_pc_vs_feature(merged, f, out_dir, args.min_count)

    # 値軸上で文脈特徴をPC色で見る
    for f in [
        "local_mean_abs_diff_mean", "local_std_mean", "flat_ratio",
        "turning_point_ratio", "near_sign_change_ratio", "local_slope_mean",
    ]:
        plot_value_vs_feature_with_pc_color(merged, f, out_dir, args.min_count, color_col="pc1")
        plot_value_vs_feature_with_pc_color(merged, f, out_dir, args.min_count, color_col="pc2")

    # +a/-a ペア解析
    pair_df = build_sign_pair_table(merged[merged["count"] >= args.min_count].copy() if "count" in merged.columns else merged)
    plot_sign_pair_results(pair_df, out_dir)

    # 上位相関だけ別ファイル
    abs_pc_corr = pc_corr.copy()
    abs_pc_corr["abs_pearson_corr"] = abs_pc_corr["pearson_corr"].abs()
    abs_pc_corr.sort_values("abs_pearson_corr", ascending=False).head(args.top_n).to_csv(
        out_dir / "top_pc_context_correlations.csv",
        index=False,
    )

    summary = {
        "context_csv": args.context_csv,
        "pca_csv": args.pca_csv,
        "output_dir": str(out_dir),
        "n_context_rows": int(len(context_df)),
        "n_pca_rows": int(len(pca_df)),
        "n_merged_rows": int(len(merged)),
        "min_count": args.min_count,
        "n_merged_rows_min_count": int((merged["count"] >= args.min_count).sum()) if "count" in merged.columns else int(len(merged)),
    }
    with open(out_dir / "merge_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Saved merged file:", merged_path)
    print("Saved outputs to:", out_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nTop PC-context correlations:")
    print(abs_pc_corr.sort_values("abs_pearson_corr", ascending=False).head(20).to_string(index=False))


if __name__ == "__main__":
    main()
