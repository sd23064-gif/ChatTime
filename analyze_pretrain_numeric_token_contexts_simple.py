#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_pretrain_numeric_token_contexts_fast.py

ChatTime pretrain のような「text列に数値トークン列だけが入っているCSV」を、
巨大な occurrence DataFrame を作らずに高速・省メモリに逐次集計するスクリプトです。

入力例:
  text
  ###-0.4159### ###-0.4731### ###-0.4343### ...

主な出力:
  - pretrain_numeric_token_context_stats_by_token.csv
  - pretrain_numeric_token_context_summary_by_value_bin.csv
  - transition_matrix_current_to_next.csv
  - numeric_token_count_per_sample.csv
  - summary.json
  - 各種 png

実行例:
  python analyze_pretrain_numeric_token_contexts_fast.py \
    --dataset_path ChengsenWang/ChatTime-1-Pretrain-1M \
    --file_name ChatTime-1-Pretrain-1M.csv \
    --text_column text \
    --output_dir outputs/pretrain_numeric_token_contexts_fast \
    --local_window 3 \
    --flat_threshold 0.01 \
    --streaming \
    --max_rows 100000
"""

import argparse
import json
import math
import os
import re
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from datasets import load_dataset


NUMERIC_TOKEN_RE = re.compile(
    r"###([+-]?(?:\d+(?:\.\d*)?|\.\d+)|Nan|NaN|nan)###"
)


def extract_numeric_tokens(text):
    """textから ###0.1234### 形式の数値トークンを順序付きで抽出する。"""
    tokens = []
    values = []

    for m in NUMERIC_TOKEN_RE.finditer(str(text)):
        raw = m.group(1)
        token = f"###{raw}###"

        if raw.lower() == "nan":
            value = np.nan
        else:
            value = float(raw)

        tokens.append(token)
        values.append(value)

    return tokens, np.asarray(values, dtype=np.float64)


def classify_local_pattern(prev_delta, next_delta, flat_threshold):
    prev_flat = abs(prev_delta) <= flat_threshold
    next_flat = abs(next_delta) <= flat_threshold

    if prev_flat and next_flat:
        return "flat"
    if prev_delta > flat_threshold and next_delta > flat_threshold:
        return "increasing"
    if prev_delta < -flat_threshold and next_delta < -flat_threshold:
        return "decreasing"
    if prev_delta > flat_threshold and next_delta < -flat_threshold:
        return "peak"
    if prev_delta < -flat_threshold and next_delta > flat_threshold:
        return "valley"
    return "mixed"


def calc_local_slope(window):
    """window内の単純線形傾き。NaNは除外。"""
    window = np.asarray(window, dtype=np.float64)
    mask = ~np.isnan(window)
    if mask.sum() < 2:
        return np.nan

    y = window[mask]
    x = np.arange(len(window))[mask]

    # polyfitより少し軽い閉形式
    x_mean = x.mean()
    y_mean = y.mean()
    denom = ((x - x_mean) ** 2).sum()
    if denom == 0:
        return np.nan
    return float(((x - x_mean) * (y - y_mean)).sum() / denom)


def init_token_stats():
    return {
        "count": 0,
        "sum_prev_delta": 0.0,
        "sum_next_delta": 0.0,
        "sum_abs_prev_delta": 0.0,
        "sum_abs_next_delta": 0.0,
        "sum_prev_abs_delta": 0.0,
        "sum_next_abs_delta": 0.0,
        "sum_local_std": 0.0,
        "sum_local_range": 0.0,
        "sum_local_mean_abs_diff": 0.0,
        "sum_local_max_abs_diff": 0.0,
        "sum_local_slope": 0.0,
        "n_local_slope": 0,
        "flat_count": 0,
        "increasing_count": 0,
        "decreasing_count": 0,
        "peak_count": 0,
        "valley_count": 0,
        "turning_point_count": 0,
        "near_sign_change_count": 0,
        "amp_increasing_count": 0,
        "amp_decreasing_count": 0,
        "prev_counter": Counter(),
        "next_counter": Counter(),
    }


def init_bin_stats():
    return {
        "count": 0,
        "sum_local_std": 0.0,
        "sum_local_mean_abs_diff": 0.0,
        "sum_local_range": 0.0,
        "flat_count": 0,
        "increasing_count": 0,
        "decreasing_count": 0,
        "peak_count": 0,
        "valley_count": 0,
        "turning_point_count": 0,
        "near_sign_change_count": 0,
        "amp_increasing_count": 0,
        "amp_decreasing_count": 0,
    }


def value_to_bin_idx(value, n_bins=40, vmin=-1.0, vmax=1.0):
    if np.isnan(value):
        return None
    if value < vmin or value > vmax:
        return None
    idx = int((value - vmin) / (vmax - vmin) * n_bins)
    if idx == n_bins:
        idx = n_bins - 1
    if idx < 0 or idx >= n_bins:
        return None
    return idx


def bin_label(idx, n_bins=40, vmin=-1.0, vmax=1.0):
    left = vmin + idx * (vmax - vmin) / n_bins
    right = vmin + (idx + 1) * (vmax - vmin) / n_bins
    if idx == 0:
        return f"[{left:.3f}, {right:.3f}]"
    return f"({left:.3f}, {right:.3f}]"


def update_stats_for_sequence(
    tokens,
    values,
    token_stats,
    bin_stats,
    transition_counts,
    local_window=3,
    flat_threshold=0.01,
    n_bins=40,
):
    """1行分の token sequence から逐次集計する。"""
    if len(values) < 3:
        return 0

    n_occ = 0

    for t in range(1, len(values) - 1):
        x = values[t]
        prev_value = values[t - 1]
        next_value = values[t + 1]

        if np.isnan(x) or np.isnan(prev_value) or np.isnan(next_value):
            continue

        prev_delta = x - prev_value
        next_delta = next_value - x
        abs_prev_delta = abs(prev_delta)
        abs_next_delta = abs(next_delta)
        prev_abs_delta = abs(x) - abs(prev_value)
        next_abs_delta = abs(next_value) - abs(x)

        left = max(0, t - local_window)
        right = min(len(values), t + local_window + 1)
        window = values[left:right]
        window_non_nan = window[~np.isnan(window)]
        if len(window_non_nan) < 2:
            continue

        diffs = np.diff(window_non_nan)
        local_std = float(np.std(window_non_nan))
        local_range = float(np.max(window_non_nan) - np.min(window_non_nan))
        local_mean_abs_diff = float(np.mean(np.abs(diffs))) if len(diffs) > 0 else np.nan
        local_max_abs_diff = float(np.max(np.abs(diffs))) if len(diffs) > 0 else np.nan
        slope = calc_local_slope(window)

        pattern = classify_local_pattern(prev_delta, next_delta, flat_threshold)
        is_flat = int(pattern == "flat")
        is_increasing = int(pattern == "increasing")
        is_decreasing = int(pattern == "decreasing")
        is_peak = int(pattern == "peak")
        is_valley = int(pattern == "valley")
        is_turning = int(pattern in ["peak", "valley"])
        near_sign_change = int((prev_value * x < 0) or (x * next_value < 0))
        amp_increasing = int(prev_abs_delta > flat_threshold and next_abs_delta > flat_threshold)
        amp_decreasing = int(prev_abs_delta < -flat_threshold and next_abs_delta < -flat_threshold)

        token = tokens[t]
        st = token_stats[(token, x)]
        st["count"] += 1
        st["sum_prev_delta"] += prev_delta
        st["sum_next_delta"] += next_delta
        st["sum_abs_prev_delta"] += abs_prev_delta
        st["sum_abs_next_delta"] += abs_next_delta
        st["sum_prev_abs_delta"] += prev_abs_delta
        st["sum_next_abs_delta"] += next_abs_delta
        st["sum_local_std"] += local_std
        st["sum_local_range"] += local_range
        if not np.isnan(local_mean_abs_diff):
            st["sum_local_mean_abs_diff"] += local_mean_abs_diff
        if not np.isnan(local_max_abs_diff):
            st["sum_local_max_abs_diff"] += local_max_abs_diff
        if not np.isnan(slope):
            st["sum_local_slope"] += slope
            st["n_local_slope"] += 1
        st["flat_count"] += is_flat
        st["increasing_count"] += is_increasing
        st["decreasing_count"] += is_decreasing
        st["peak_count"] += is_peak
        st["valley_count"] += is_valley
        st["turning_point_count"] += is_turning
        st["near_sign_change_count"] += near_sign_change
        st["amp_increasing_count"] += amp_increasing
        st["amp_decreasing_count"] += amp_decreasing
        st["prev_counter"][tokens[t - 1]] += 1
        st["next_counter"][tokens[t + 1]] += 1

        bidx = value_to_bin_idx(x, n_bins=n_bins)
        if bidx is not None:
            bs = bin_stats[bidx]
            bs["count"] += 1
            bs["sum_local_std"] += local_std
            if not np.isnan(local_mean_abs_diff):
                bs["sum_local_mean_abs_diff"] += local_mean_abs_diff
            bs["sum_local_range"] += local_range
            bs["flat_count"] += is_flat
            bs["increasing_count"] += is_increasing
            bs["decreasing_count"] += is_decreasing
            bs["peak_count"] += is_peak
            bs["valley_count"] += is_valley
            bs["turning_point_count"] += is_turning
            bs["near_sign_change_count"] += near_sign_change
            bs["amp_increasing_count"] += amp_increasing
            bs["amp_decreasing_count"] += amp_decreasing

        cb = value_to_bin_idx(x, n_bins=n_bins)
        nb = value_to_bin_idx(next_value, n_bins=n_bins)
        if cb is not None and nb is not None:
            transition_counts[cb, nb] += 1

        n_occ += 1

    return n_occ


def mean_or_nan(total, count):
    return float(total / count) if count > 0 else np.nan


def build_token_stats_df(token_stats):
    rows = []
    for (token, value), st in token_stats.items():
        c = st["count"]
        if c == 0:
            continue
        rows.append({
            "token": token,
            "token_value": value,
            "count": c,
            "mean_prev_delta": mean_or_nan(st["sum_prev_delta"], c),
            "mean_next_delta": mean_or_nan(st["sum_next_delta"], c),
            "mean_abs_prev_delta": mean_or_nan(st["sum_abs_prev_delta"], c),
            "mean_abs_next_delta": mean_or_nan(st["sum_abs_next_delta"], c),
            "mean_prev_abs_delta": mean_or_nan(st["sum_prev_abs_delta"], c),
            "mean_next_abs_delta": mean_or_nan(st["sum_next_abs_delta"], c),
            "local_std_mean": mean_or_nan(st["sum_local_std"], c),
            "local_range_mean": mean_or_nan(st["sum_local_range"], c),
            "local_mean_abs_diff_mean": mean_or_nan(st["sum_local_mean_abs_diff"], c),
            "local_max_abs_diff_mean": mean_or_nan(st["sum_local_max_abs_diff"], c),
            "local_slope_mean": mean_or_nan(st["sum_local_slope"], st["n_local_slope"]),
            "flat_ratio": mean_or_nan(st["flat_count"], c),
            "increasing_ratio": mean_or_nan(st["increasing_count"], c),
            "decreasing_ratio": mean_or_nan(st["decreasing_count"], c),
            "peak_ratio": mean_or_nan(st["peak_count"], c),
            "valley_ratio": mean_or_nan(st["valley_count"], c),
            "turning_point_ratio": mean_or_nan(st["turning_point_count"], c),
            "near_sign_change_ratio": mean_or_nan(st["near_sign_change_count"], c),
            "amp_increasing_ratio": mean_or_nan(st["amp_increasing_count"], c),
            "amp_decreasing_ratio": mean_or_nan(st["amp_decreasing_count"], c),
            "most_common_prev_token": st["prev_counter"].most_common(1)[0][0] if len(st["prev_counter"]) else None,
            "most_common_next_token": st["next_counter"].most_common(1)[0][0] if len(st["next_counter"]) else None,
        })
    return pd.DataFrame(rows).sort_values("token_value").reset_index(drop=True)


def build_bin_summary_df(bin_stats, n_bins=40):
    rows = []
    for idx in range(n_bins):
        st = bin_stats[idx]
        c = st["count"]
        if c == 0:
            rows.append({"token_value_bin": bin_label(idx, n_bins=n_bins), "count": 0})
            continue
        rows.append({
            "token_value_bin": bin_label(idx, n_bins=n_bins),
            "bin_center": -1.0 + (idx + 0.5) * 2.0 / n_bins,
            "count": c,
            "local_std_mean": mean_or_nan(st["sum_local_std"], c),
            "local_mean_abs_diff_mean": mean_or_nan(st["sum_local_mean_abs_diff"], c),
            "local_range_mean": mean_or_nan(st["sum_local_range"], c),
            "flat_ratio": mean_or_nan(st["flat_count"], c),
            "increasing_ratio": mean_or_nan(st["increasing_count"], c),
            "decreasing_ratio": mean_or_nan(st["decreasing_count"], c),
            "peak_ratio": mean_or_nan(st["peak_count"], c),
            "valley_ratio": mean_or_nan(st["valley_count"], c),
            "turning_point_ratio": mean_or_nan(st["turning_point_count"], c),
            "near_sign_change_ratio": mean_or_nan(st["near_sign_change_count"], c),
            "amp_increasing_ratio": mean_or_nan(st["amp_increasing_count"], c),
            "amp_decreasing_ratio": mean_or_nan(st["amp_decreasing_count"], c),
        })
    return pd.DataFrame(rows)


def load_rows_iter(args):
    """CSV/HF datasetを1行ずつ返す iterator を返す。"""
    if args.csv_path is not None:
        print("Loading local CSV by chunks:", args.csv_path)
        # pandas chunk iterator
        for chunk in pd.read_csv(args.csv_path, chunksize=args.chunksize):
            chunk.columns = chunk.columns.str.replace("\ufeff", "", regex=False).str.strip()
            for _, row in chunk.iterrows():
                yield row.to_dict()
        return

    if args.dataset_path is None:
        raise ValueError("Either --csv_path or --dataset_path must be specified.")

    data_url = f"https://huggingface.co/datasets/{args.dataset_path}/resolve/main/{args.file_name}"
    print("Loading Hugging Face CSV:")
    print(data_url)

    if args.streaming:
        dataset = load_dataset("csv", data_files=data_url, split="train", streaming=True)
        for row in dataset:
            yield row
    else:
        dataset = load_dataset("csv", data_files=data_url, split="train")
        if args.max_rows is not None and args.max_rows > 0:
            dataset = dataset.select(range(min(args.max_rows, len(dataset))))
        for row in dataset:
            yield row


def save_line_plot(df, y_col, out_path, ylabel, title, min_count=20):
    if "count" in df.columns:
        plot_df = df[df["count"] >= min_count].copy()
    else:
        plot_df = df.copy()
    if len(plot_df) == 0 or y_col not in plot_df.columns:
        return
    plt.figure(figsize=(11, 5))
    plt.plot(plot_df["token_value"], plot_df[y_col], linewidth=1)
    plt.xlabel("token value")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def save_outputs(args, token_stats, bin_stats, transition_counts, sample_rows, n_rows, n_occurrences):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    token_df = build_token_stats_df(token_stats)
    token_df.to_csv(out / "pretrain_numeric_token_context_stats_by_token.csv", index=False)

    bin_df = build_bin_summary_df(bin_stats, n_bins=args.n_bins)
    bin_df.to_csv(out / "pretrain_numeric_token_context_summary_by_value_bin.csv", index=False)

    sample_df = pd.DataFrame(sample_rows)
    sample_df.to_csv(out / "numeric_token_count_per_sample.csv", index=False)

    # transition matrix normalized by row
    trans_raw = pd.DataFrame(
        transition_counts,
        index=[bin_label(i, n_bins=args.n_bins) for i in range(args.n_bins)],
        columns=[bin_label(i, n_bins=args.n_bins) for i in range(args.n_bins)],
    )
    trans_raw.to_csv(out / "transition_matrix_current_to_next_counts.csv")

    row_sums = transition_counts.sum(axis=1, keepdims=True)
    trans_norm = np.divide(
        transition_counts,
        row_sums,
        out=np.zeros_like(transition_counts, dtype=np.float64),
        where=row_sums != 0,
    )
    trans_norm_df = pd.DataFrame(
        trans_norm,
        index=[bin_label(i, n_bins=args.n_bins) for i in range(args.n_bins)],
        columns=[bin_label(i, n_bins=args.n_bins) for i in range(args.n_bins)],
    )
    trans_norm_df.to_csv(out / "transition_matrix_current_to_next.csv")

    # plots
    if len(token_df) > 0:
        save_line_plot(token_df, "count", out / "token_value_vs_count.png", "count", "Token frequency by value", min_count=1)
        save_line_plot(token_df, "local_mean_abs_diff_mean", out / "token_value_vs_local_mean_abs_diff.png", "mean local absolute change", "Mean local absolute change around each token", args.min_count_for_plots)
        save_line_plot(token_df, "local_std_mean", out / "token_value_vs_local_std.png", "mean local std", "Local volatility around each token", args.min_count_for_plots)
        save_line_plot(token_df, "flat_ratio", out / "token_value_vs_flat_ratio.png", "flat ratio", "Flat-context ratio around each token", args.min_count_for_plots)
        save_line_plot(token_df, "turning_point_ratio", out / "token_value_vs_turning_point_ratio.png", "turning point ratio", "Peak/valley-context ratio around each token", args.min_count_for_plots)
        save_line_plot(token_df, "near_sign_change_ratio", out / "token_value_vs_near_sign_change_ratio.png", "near sign-change ratio", "Near sign-change ratio around each token", args.min_count_for_plots)
        save_line_plot(token_df, "local_slope_mean", out / "token_value_vs_local_slope_mean.png", "mean local slope", "Mean local slope around each token", args.min_count_for_plots)

    plt.figure(figsize=(9, 8))
    plt.imshow(trans_norm, aspect="auto", origin="lower")
    plt.colorbar(label="P(next value bin | current value bin)")
    plt.xlabel("next value bin")
    plt.ylabel("current value bin")
    plt.title("Transition heatmap: current token value -> next token value")
    plt.tight_layout()
    plt.savefig(out / "transition_heatmap_current_to_next.png", dpi=220)
    plt.close()

    summary = {
        "csv_path": args.csv_path,
        "dataset_path": args.dataset_path,
        "file_name": args.file_name,
        "text_column": args.text_column,
        "streaming": args.streaming,
        "n_rows_processed": int(n_rows),
        "n_occurrences": int(n_occurrences),
        "n_observed_token_types": int(len(token_stats)),
        "local_window": args.local_window,
        "flat_threshold": args.flat_threshold,
        "n_bins": args.n_bins,
        "mean_numeric_tokens_per_row": float(np.mean([r["n_numeric_tokens"] for r in sample_rows])) if sample_rows else None,
        "median_numeric_tokens_per_row": float(np.median([r["n_numeric_tokens"] for r in sample_rows])) if sample_rows else None,
    }
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nSaved outputs to:", out)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if len(token_df) > 0:
        filtered = token_df[token_df["count"] >= args.min_count_for_plots]
        print("\nTop flat tokens:")
        print(filtered.sort_values("flat_ratio", ascending=False)[["token", "token_value", "count", "flat_ratio", "local_mean_abs_diff_mean"]].head(20).to_string(index=False))
        print("\nTop volatile-context tokens:")
        print(filtered.sort_values("local_mean_abs_diff_mean", ascending=False)[["token", "token_value", "count", "local_mean_abs_diff_mean", "local_std_mean", "near_sign_change_ratio"]].head(20).to_string(index=False))
        print("\nTop turning-point tokens:")
        print(filtered.sort_values("turning_point_ratio", ascending=False)[["token", "token_value", "count", "turning_point_ratio", "peak_ratio", "valley_ratio"]].head(20).to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default=None)
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--file_name", type=str, default="ChatTime-1-Pretrain-1M.csv")
    parser.add_argument("--text_column", type=str, default="text")
    parser.add_argument("--output_dir", type=str, default="outputs/pretrain_numeric_token_contexts_fast")

    parser.add_argument("--streaming", action="store_true", help="HF datasetをstreamingで読む。全件解析では推奨。")
    parser.add_argument("--chunksize", type=int, default=10000, help="local CSV用chunk size")
    parser.add_argument("--max_rows", type=int, default=-1)
    parser.add_argument("--start_row", type=int, default=0)

    parser.add_argument("--local_window", type=int, default=3)
    parser.add_argument("--flat_threshold", type=float, default=0.01)
    parser.add_argument("--min_count_for_plots", type=int, default=20)
    parser.add_argument("--n_bins", type=int, default=40)
    parser.add_argument("--save_every", type=int, default=0, help="途中保存する行数間隔。0なら途中保存しない。")

    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    token_stats = defaultdict(init_token_stats)
    bin_stats = defaultdict(init_bin_stats)
    transition_counts = np.zeros((args.n_bins, args.n_bins), dtype=np.int64)
    sample_rows = []

    n_rows_seen = 0
    n_rows_processed = 0
    n_occurrences = 0

    iterator = load_rows_iter(args)

    pbar_total = args.max_rows if args.max_rows and args.max_rows > 0 else None
    pbar = tqdm(total=pbar_total, desc="Fast aggregate token contexts")

    for row in iterator:
        if n_rows_seen < args.start_row:
            n_rows_seen += 1
            continue

        if args.max_rows is not None and args.max_rows > 0 and n_rows_processed >= args.max_rows:
            break

        text = row.get(args.text_column, None)
        tokens, values = extract_numeric_tokens(text)

        sample_rows.append({
            "row_idx_original": n_rows_seen,
            "row_idx_processed": n_rows_processed,
            "n_numeric_tokens": int(len(tokens)),
            "first_value": float(values[0]) if len(values) > 0 and not np.isnan(values[0]) else np.nan,
            "last_value": float(values[-1]) if len(values) > 0 and not np.isnan(values[-1]) else np.nan,
            "mean_value": float(np.nanmean(values)) if len(values) > 0 else np.nan,
            "std_value": float(np.nanstd(values)) if len(values) > 0 else np.nan,
            "min_value": float(np.nanmin(values)) if len(values) > 0 else np.nan,
            "max_value": float(np.nanmax(values)) if len(values) > 0 else np.nan,
        })

        n_occ = update_stats_for_sequence(
            tokens=tokens,
            values=values,
            token_stats=token_stats,
            bin_stats=bin_stats,
            transition_counts=transition_counts,
            local_window=args.local_window,
            flat_threshold=args.flat_threshold,
            n_bins=args.n_bins,
        )
        n_occurrences += n_occ

        n_rows_seen += 1
        n_rows_processed += 1
        pbar.update(1)

        if args.save_every and args.save_every > 0 and n_rows_processed % args.save_every == 0:
            print(f"\n[checkpoint] saving partial outputs at processed rows={n_rows_processed}")
            save_outputs(args, token_stats, bin_stats, transition_counts, sample_rows, n_rows_processed, n_occurrences)

    pbar.close()

    save_outputs(args, token_stats, bin_stats, transition_counts, sample_rows, n_rows_processed, n_occurrences)


if __name__ == "__main__":
    main()
