#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_LAYER_PAIRS = {
    "depth_25": {"llama": 7, "mamba": 16, "relative_depth": 0.25},
    "depth_50": {"llama": 14, "mamba": 32, "relative_depth": 0.50},
    "depth_100": {"llama": 28, "mamba": 64, "relative_depth": 1.00},
}


def parse_layer_pairs(text):
    """
    例:
    depth_25:7:16:0.25,depth_50:14:32:0.5,depth_100:28:64:1.0
    """
    if text is None:
        return DEFAULT_LAYER_PAIRS

    pairs = {}

    for item in text.split(","):
        name, llama_layer, mamba_layer, relative_depth = item.split(":")

        pairs[name] = {
            "llama": int(llama_layer),
            "mamba": int(mamba_layer),
            "relative_depth": float(relative_depth),
        }

    return pairs


def make_comparison(summary, layer_pairs):
    rows = []

    index_columns = [
        "state",
        "target",
        "metric",
    ]

    for depth_name, pair in layer_pairs.items():
        llama_layer = pair["llama"]
        mamba_layer = pair["mamba"]

        llama = summary[
            (summary["model"] == "llama")
            & (summary["layer"] == llama_layer)
        ].copy()

        mamba = summary[
            (summary["model"] == "mamba")
            & (summary["layer"] == mamba_layer)
        ].copy()

        llama = llama.set_index(index_columns)
        mamba = mamba.set_index(index_columns)

        common_index = llama.index.intersection(mamba.index)

        if len(common_index) == 0:
            raise ValueError(
                f"No common probe rows for Llama layer {llama_layer} "
                f"and Mamba layer {mamba_layer}."
            )

        for state, target, metric in common_index:
            llama_row = llama.loc[(state, target, metric)]
            mamba_row = mamba.loc[(state, target, metric)]

            llama_score = float(llama_row["score_mean"])
            mamba_score = float(mamba_row["score_mean"])
            difference = mamba_score - llama_score

            if difference > 0:
                winner = "mamba"
            elif difference < 0:
                winner = "llama"
            else:
                winner = "equal"

            rows.append({
                "depth_name": depth_name,
                "relative_depth": pair["relative_depth"],
                "llama_layer": llama_layer,
                "mamba_layer": mamba_layer,
                "state": state,
                "target": target,
                "metric": metric,
                "llama_score_mean": llama_score,
                "llama_score_std": float(llama_row["score_std"]),
                "mamba_score_mean": mamba_score,
                "mamba_score_std": float(mamba_row["score_std"]),
                "mamba_minus_llama": difference,
                "absolute_difference": abs(difference),
                "winner": winner,
            })

    return pd.DataFrame(rows)


def make_target_summary(comparison):
    return (
        comparison
        .groupby(["state", "target", "metric"], as_index=False)
        .agg(
            llama_score_mean=("llama_score_mean", "mean"),
            mamba_score_mean=("mamba_score_mean", "mean"),
            mamba_minus_llama_mean=("mamba_minus_llama", "mean"),
            mamba_minus_llama_min=("mamba_minus_llama", "min"),
            mamba_minus_llama_max=("mamba_minus_llama", "max"),
            mamba_win_count=("winner", lambda values: int((values == "mamba").sum())),
            llama_win_count=("winner", lambda values: int((values == "llama").sum())),
            equal_count=("winner", lambda values: int((values == "equal").sum())),
            n_depths=("winner", "size"),
        )
    )


def main():
    parser = argparse.ArgumentParser(
        description="Compare Llama and Mamba probes at corresponding relative depths."
    )
    parser.add_argument("--summary_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--layer_pairs",
        default=None,
        help=(
            "Optional custom mapping such as "
            "depth_25:7:16:0.25,depth_50:14:32:0.5,depth_100:28:64:1.0"
        ),
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(args.summary_csv)

    required_columns = {
        "model",
        "layer",
        "state",
        "target",
        "metric",
        "score_mean",
        "score_std",
    }
    missing = sorted(required_columns - set(summary.columns))

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    layer_pairs = parse_layer_pairs(args.layer_pairs)
    comparison = make_comparison(summary, layer_pairs)
    target_summary = make_target_summary(comparison)

    comparison_path = output_dir / "corresponding_layer_comparison.csv"
    target_summary_path = output_dir / "corresponding_layer_target_summary.csv"

    comparison.to_csv(comparison_path, index=False)
    target_summary.to_csv(target_summary_path, index=False)

    print("\nCorresponding-layer comparison")
    print(
        comparison[
            [
                "depth_name",
                "state",
                "target",
                "llama_score_mean",
                "mamba_score_mean",
                "mamba_minus_llama",
                "winner",
            ]
        ].to_string(index=False)
    )

    print("\nSaved:")
    print(" -", comparison_path)
    print(" -", target_summary_path)


if __name__ == "__main__":
    main()