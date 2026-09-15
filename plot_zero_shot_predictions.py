#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re
import argparse
import ast
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_array(value):
    if isinstance(value, np.ndarray):
        return value.astype(np.float64, copy=False).reshape(-1)

    if isinstance(value, list):
        return np.asarray(value, dtype=np.float64).reshape(-1)

    if pd.isna(value):
        return np.asarray([], dtype=np.float64)

    return np.asarray(
        ast.literal_eval(str(value)),
        dtype=np.float64,
    ).reshape(-1)


def safe_mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)

    if len(y_true) != len(y_pred):
        return np.nan

    mask = np.isfinite(y_true) & np.isfinite(y_pred)

    if not mask.any():
        return np.nan

    return float(
        np.mean(
            np.abs(y_true[mask] - y_pred[mask])
        )
    )


def inverse_standardize(values, mean, std):
    values = np.asarray(values, dtype=np.float64)

    if mean is None or std is None:
        return values

    return values * std + mean


def select_rows(frame, mode, count, seed, row_indices):
    frame = frame.copy()

    if row_indices:
        missing = [
            index for index in row_indices
            if index not in frame.index
        ]

        if missing:
            raise ValueError(
                f"Requested row indices not found: {missing}"
            )

        return frame.loc[row_indices]

    count = min(count, len(frame))

    if mode == "best":
        return frame.nsmallest(count, "plot_mae")

    if mode == "worst":
        return frame.nlargest(count, "plot_mae")

    if mode == "median":
        sorted_frame = frame.sort_values(
            "plot_mae"
        )

        center = len(sorted_frame) // 2
        start = max(0, center - count // 2)

        return sorted_frame.iloc[
            start:start + count
        ]

    if mode == "random":
        return frame.sample(
            n=count,
            random_state=seed,
        )

    if mode == "first":
        return frame.head(count)

    raise ValueError(f"Unknown selection mode: {mode}")


def plot_prediction(
    row,
    output_path,
    history_points=None,
    show_naive=False,
    connect_prediction=True,
    title=None,
    mean=None,
    std=None,
    dpi=200,
):
    history = parse_array(row["history"])
    true_future = parse_array(row["true"])
    prediction = parse_array(row["prediction"])

    if history_points is not None and history_points > 0:
        history = history[-history_points:]

    if len(true_future) != len(prediction):
        raise ValueError(
            "True/prediction length mismatch: "
            f"{len(true_future)} != {len(prediction)}"
        )

    history = inverse_standardize(
        history,
        mean,
        std,
    )
    true_future = inverse_standardize(
        true_future,
        mean,
        std,
    )
    prediction = inverse_standardize(
        prediction,
        mean,
        std,
    )

    hist_len = len(history)
    pred_len = len(prediction)

    history_x = np.arange(hist_len)
    future_x = np.arange(
        hist_len,
        hist_len + pred_len,
    )

    full_true = np.concatenate([
        history,
        true_future,
    ])

    full_x = np.arange(
        len(full_true)
    )

    if connect_prediction and hist_len > 0:
        prediction_x = np.concatenate([
            [hist_len - 1],
            future_x,
        ])

        prediction_y = np.concatenate([
            [history[-1]],
            prediction,
        ])
    else:
        prediction_x = future_x
        prediction_y = prediction

    figure, axis = plt.subplots(
        figsize=(14, 4.5)
    )

    axis.plot(
        full_x,
        full_true,
        color="black",
        linewidth=2.2,
        label="true",
        zorder=2,
    )

    axis.plot(
        prediction_x,
        prediction_y,
        color="tab:orange",
        linewidth=2.2,
        label="pred",
        zorder=3,
    )

    if show_naive and "naive_prediction" in row:
        naive = parse_array(
            row["naive_prediction"]
        )

        naive = inverse_standardize(
            naive,
            mean,
            std,
        )

        if len(naive) == pred_len:
            axis.plot(
                future_x,
                naive,
                color="tab:blue",
                linewidth=1.8,
                linestyle="--",
                label="naive last",
                zorder=1,
            )

    axis.axvline(
        x=hist_len,
        color="red",
        linewidth=2.0,
        label="forecast start",
        zorder=4,
    )

    mae_value = safe_mae(
        true_future,
        prediction,
    )

    if title is None:
        model_name = row.get(
            "model",
            "model",
        )

        column_name = row.get(
            "column",
            "",
        )

        original_hist_len = row.get(
            "hist_len",
            hist_len,
        )

        window_id = row.get(
            "window_id",
            row.name,
        )

        title = (
            f"{model_name} | "
            f"column={column_name} | "
            f"hist={original_hist_len} | "
            f"pred={pred_len} | "
            f"window={window_id} | "
            f"MAE={mae_value:.4f}"
        )

    axis.set_title(
        title,
        fontsize=12,
    )

    axis.set_xlabel(
        "Time step"
    )

    axis.set_ylabel(
        "Value"
    )

    axis.legend(
        loc="upper left",
        frameon=True,
    )

    axis.grid(
        alpha=0.2,
    )

    axis.margins(
        x=0.01,
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
    )

    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Visualize saved zero-shot forecasting results."
        )
    )

    parser.add_argument(
        "--details_csv",
        required=True,
    )

    parser.add_argument(
        "--output_dir",
        default="outputs/forecast_plots",
    )

    parser.add_argument(
        "--model",
        default=None,
    )

    parser.add_argument(
        "--column",
        default=None,
    )

    parser.add_argument(
        "--hist_len",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--selection",
        choices=[
            "best",
            "worst",
            "median",
            "random",
            "first",
        ],
        default="random",
    )

    parser.add_argument(
        "--count",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--row_indices",
        type=int,
        nargs="*",
        default=None,
    )

    parser.add_argument(
        "--history_points",
        type=int,
        default=None,
        help=(
            "Number of recent history points to show. "
            "The full saved history is used when omitted."
        ),
    )

    parser.add_argument(
        "--show_naive",
        action="store_true",
    )

    parser.add_argument(
        "--no_connect_prediction",
        action="store_true",
    )

    parser.add_argument(
        "--only_success",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument(
        "--only_complete",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument(
        "--inverse_standardize",
        action="store_true",
    )

    parser.add_argument(
        "--mean",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--std",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=3407,
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
    )

    args = parser.parse_args()

    frame = pd.read_csv(
        args.details_csv
    )

    required_columns = {
        "history",
        "true",
        "prediction",
    }

    missing_columns = (
        required_columns
        - set(frame.columns)
    )

    if missing_columns:
        raise ValueError(
            "The details CSV does not contain saved "
            "predictions. Missing columns: "
            f"{sorted(missing_columns)}. "
            "Run the evaluation with --save_predictions."
        )

    if args.model is not None:
        if "model" not in frame.columns:
            raise ValueError(
                "The CSV has no model column."
            )

        frame = frame[
            frame["model"] == args.model
        ].copy()

    if args.column is not None:
        if "column" not in frame.columns:
            raise ValueError(
                "The CSV has no column column."
            )

        frame = frame[
            frame["column"] == args.column
        ].copy()

    if args.hist_len is not None:
        if "hist_len" not in frame.columns:
            raise ValueError(
                "The CSV has no hist_len column."
            )

        frame = frame[
            frame["hist_len"] == args.hist_len
        ].copy()

    if (
        args.only_success
        and "success" in frame.columns
    ):
        frame = frame[
            frame["success"].astype(bool)
        ].copy()

    if args.only_complete:
        if "parsed_ratio" in frame.columns:
            frame = frame[
                np.isclose(
                    frame["parsed_ratio"],
                    1.0,
                    equal_nan=False,
                )
            ].copy()

        if "fallback_ratio" in frame.columns:
            frame = frame[
                np.isclose(
                    frame["fallback_ratio"],
                    0.0,
                    equal_nan=False,
                )
            ].copy()

    if frame.empty:
        raise ValueError(
            "No rows remain after filtering."
        )

    frame["plot_mae"] = [
        safe_mae(
            parse_array(true_value),
            parse_array(prediction_value),
        )
        for true_value, prediction_value
        in zip(
            frame["true"],
            frame["prediction"],
        )
    ]

    frame = frame[
        np.isfinite(
            frame["plot_mae"]
        )
    ].copy()

    if frame.empty:
        raise ValueError(
            "No rows have a valid MAE."
        )

    selected = select_rows(
        frame=frame,
        mode=args.selection,
        count=args.count,
        seed=args.seed,
        row_indices=args.row_indices,
    )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    mean = (
        args.mean
        if args.inverse_standardize
        else None
    )

    std = (
        args.std
        if args.inverse_standardize
        else None
    )

    if args.inverse_standardize:
        if mean is None or std is None:
            raise ValueError(
                "--mean and --std are required with "
                "--inverse_standardize."
            )

        if not np.isfinite(std) or std <= 0:
            raise ValueError(
                f"Invalid standard deviation: {std}"
            )

    selection_records = []

    for order, (row_index, row) in enumerate(
        selected.iterrows()
    ):
        model_name = str(
            row.get("model", "model")
        )

        column_name = str(
            row.get("column", "series")
        )

        hist_len = int(
            row.get(
                "hist_len",
                len(parse_array(row["history"])),
            )
        )

        window_id = row.get(
            "window_id",
            row_index,
        )

        file_name = (
            f"{order:02d}_"
            f"{model_name}_"
            f"{column_name}_"
            f"hist{hist_len}_"
            f"window{window_id}.png"
        )

        file_name = re.sub(
            r"[^A-Za-z0-9_.-]+",
            "_",
            file_name,
        )

        output_path = (
            output_dir / file_name
        )

        plot_prediction(
            row=row,
            output_path=output_path,
            history_points=args.history_points,
            show_naive=args.show_naive,
            connect_prediction=(
                not args.no_connect_prediction
            ),
            mean=mean,
            std=std,
            dpi=args.dpi,
        )

        selection_records.append({
            "source_csv_index": int(row_index),
            "output_file": str(output_path),
            "model": model_name,
            "column": column_name,
            "hist_len": hist_len,
            "window_id": window_id,
            "mae": float(row["plot_mae"]),
        })

        print(
            f"Saved {output_path} "
            f"(MAE={row['plot_mae']:.6f})"
        )

    selection_frame = pd.DataFrame(
        selection_records
    )

    selection_path = (
        output_dir
        / "plotted_windows.csv"
    )

    selection_frame.to_csv(
        selection_path,
        index=False,
    )

    print("\nSelected windows:")
    print(
        selection_frame.to_string(
            index=False
        )
    )

    print("\nSaved selection metadata:")
    print(selection_path)


if __name__ == "__main__":
    main()