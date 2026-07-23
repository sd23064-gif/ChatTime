#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm

from model.mamba_model import ChatTimeMamba


NUMERIC_TOKEN_RE = re.compile(
    r"###(?:[+-]?\d+(?:\.\d+)?|Nan|NaN|nan)###"
)


def mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    mask = (
        np.isfinite(y_true)
        & np.isfinite(y_pred)
    )

    if mask.sum() == 0:
        return np.nan

    return float(
        np.mean(
            np.abs(
                y_true[mask] - y_pred[mask]
            )
        )
    )


def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    mask = (
        np.isfinite(y_true)
        & np.isfinite(y_pred)
    )

    if mask.sum() == 0:
        return np.nan

    return float(
        np.sqrt(
            np.mean(
                (
                    y_true[mask]
                    - y_pred[mask]
                ) ** 2
            )
        )
    )


def extract_numeric_token_text(text):
    tokens = NUMERIC_TOKEN_RE.findall(
        str(text)
    )

    return " ".join(tokens)


def decode_numeric_series(
    model,
    text,
):
    numeric_text = extract_numeric_token_text(
        text
    )

    if not numeric_text:
        return np.asarray(
            [],
            dtype=np.float64,
        )

    dispersed = (
        model.serializer.inverse_serialize(
            numeric_text
        )
    )

    values = (
        model.discretizer.inverse_discretize(
            dispersed
        )
    )

    return np.asarray(
        values,
        dtype=np.float64,
    )


def select_window(
    series,
    hist_len,
    pred_len,
    window_mode,
    rng,
):
    required_length = hist_len + pred_len

    if len(series) < required_length:
        return None, None, None

    maximum_start = (
        len(series) - required_length
    )

    if window_mode == "first":
        start = 0

    elif window_mode == "last":
        start = maximum_start

    elif window_mode == "random":
        start = int(
            rng.integers(
                0,
                maximum_start + 1,
            )
        )

    else:
        raise ValueError(
            f"Unknown window mode: "
            f"{window_mode}"
        )

    hist_data = series[
        start:
        start + hist_len
    ]

    true_data = series[
        start + hist_len:
        start + hist_len + pred_len
    ]

    return start, hist_data, true_data


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--code_path",
        type=str,
        default="/workspace",
    )

    parser.add_argument(
        "--base_model_path",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--adapter_path",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--dataset_path",
        type=str,
        default=(
            "ChengsenWang/"
            "ChatTime-1-Pretrain-1M"
        ),
    )

    parser.add_argument(
        "--train_file",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--hist_len",
        type=int,
        default=48,
    )

    parser.add_argument(
        "--pred_len",
        type=int,
        default=24,
    )

    parser.add_argument(
        "--max_pred_len",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--num_eval_examples",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--dataset_scan_count",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=3407,
    )

    parser.add_argument(
        "--window_mode",
        type=str,
        choices=[
            "first",
            "last",
            "random",
        ],
        default="random",
    )

    parser.add_argument(
        "--dtype",
        type=str,
        choices=[
            "fp16",
            "bf16",
            "fp32",
        ],
        default="fp16",
    )

    args = parser.parse_args()

    sys.path.append(args.code_path)

    os.makedirs(
        args.output_dir,
        exist_ok=True,
    )

    dtype_map = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }

    torch_dtype = dtype_map[args.dtype]

    print("Loading ChatTimeMamba")
    print("Base model:", args.base_model_path)
    print("Adapter:", args.adapter_path)

    model = ChatTimeMamba(
        base_model_path=(
            args.base_model_path
        ),
        adapter_path=args.adapter_path,
        hist_len=args.hist_len,
        pred_len=args.pred_len,
        max_pred_len=args.max_pred_len,
        num_samples=args.num_samples,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        torch_dtype=torch_dtype,
        merge_lora=False,
    )

    print(
        "Tokenizer vocabulary size:",
        len(model.tokenizer),
    )

    print(
        "Model vocabulary size:",
        model.model
        .get_input_embeddings()
        .weight.shape[0],
    )

    if args.train_file is not None:
        dataset = load_dataset(
            "csv",
            data_files=args.train_file,
            split="train",
        )

    else:
        dataset = load_dataset(
            "csv",
            data_files=(
                "https://huggingface.co/"
                "datasets/"
                f"{args.dataset_path}/"
                "resolve/main/"
                "ChatTime-1-Pretrain-1M.csv"
            ),
            split="train",
        )

    if "text" not in dataset.column_names:
        raise ValueError(
            "The dataset has no text column. "
            f"Columns: "
            f"{dataset.column_names}"
        )

    scan_count = min(
        args.dataset_scan_count,
        len(dataset),
    )

    subset = (
        dataset
        .shuffle(seed=args.seed)
        .select(range(scan_count))
    )

    rng = np.random.default_rng(
        args.seed
    )

    results = []

    successful_predictions = 0

    for dataset_index, example in tqdm(
        enumerate(subset),
        total=len(subset),
        desc="Evaluate training subset",
    ):
        if (
            successful_predictions
            >= args.num_eval_examples
        ):
            break

        text = str(example["text"])

        result = {
            "dataset_subset_index":
                dataset_index,
            "status": "failed",
            "error": None,
        }

        try:
            series = decode_numeric_series(
                model=model,
                text=text,
            )

            result["extracted_series_length"] = (
                int(len(series))
            )

            (
                window_start,
                hist_data,
                true_data,
            ) = select_window(
                series=series,
                hist_len=args.hist_len,
                pred_len=args.pred_len,
                window_mode=args.window_mode,
                rng=rng,
            )

            if hist_data is None:
                result["status"] = (
                    "skipped_short_series"
                )

                result["error"] = (
                    "Not enough numeric tokens: "
                    f"required="
                    f"{args.hist_len + args.pred_len}, "
                    f"actual={len(series)}"
                )

                results.append(result)
                continue

            if (
                not np.isfinite(hist_data).all()
                or not np.isfinite(true_data).all()
            ):
                result["status"] = (
                    "skipped_non_finite"
                )

                result["error"] = (
                    "History or target contains "
                    "NaN/Inf."
                )

                results.append(result)
                continue

            pred_data = model.predict(
                hist_data
            )

            pred_data = np.asarray(
                pred_data,
                dtype=np.float64,
            )

            result.update(
                {
                    "window_start":
                        int(window_start),
                    "hist_len":
                        args.hist_len,
                    "pred_len":
                        args.pred_len,
                    "prediction_length":
                        int(len(pred_data)),
                    "hist_last_value":
                        float(hist_data[-1]),
                    "target_first_value":
                        float(true_data[0]),
                    "prediction_first_value":
                        (
                            float(pred_data[0])
                            if len(pred_data) > 0
                            else np.nan
                        ),
                    "mae":
                        mae(
                            true_data,
                            pred_data,
                        ),
                    "rmse":
                        rmse(
                            true_data,
                            pred_data,
                        ),
                    "prediction_nan_count":
                        int(
                            np.isnan(
                                pred_data
                            ).sum()
                        ),
                    "history":
                        json.dumps(
                            hist_data.tolist()
                        ),
                    "target":
                        json.dumps(
                            true_data.tolist()
                        ),
                    "prediction":
                        json.dumps(
                            pred_data.tolist()
                        ),
                    "status":
                        "success",
                    "error":
                        None,
                }
            )

            successful_predictions += 1

            print("\n" + "=" * 80)
            print(
                "Successful sample:",
                successful_predictions,
            )
            print(
                "Extracted series length:",
                len(series),
            )
            print(
                "History:",
                hist_data[-10:],
            )
            print(
                "Target:",
                true_data,
            )
            print(
                "Prediction:",
                pred_data,
            )
            print(
                "MAE:",
                result["mae"],
            )
            print(
                "RMSE:",
                result["rmse"],
            )

        except Exception as error:
            result["status"] = "failed"
            result["error"] = str(error)

            print(
                "\nPrediction failed:",
                error,
            )

        results.append(result)

    result_df = pd.DataFrame(results)

    detail_path = os.path.join(
        args.output_dir,
        "train_forecasting_details.csv",
    )

    result_df.to_csv(
        detail_path,
        index=False,
    )

    success_df = result_df[
        result_df["status"] == "success"
    ].copy()

    summary = {
        "base_model_path":
            args.base_model_path,
        "adapter_path":
            args.adapter_path,
        "dataset_path":
            args.dataset_path,
        "hist_len":
            args.hist_len,
        "pred_len":
            args.pred_len,
        "num_scanned_examples":
            int(len(result_df)),
        "num_successful_examples":
            int(len(success_df)),
        "success_rate":
            (
                float(
                    len(success_df)
                    / max(len(result_df), 1)
                )
            ),
        "mae_mean":
            (
                float(
                    success_df["mae"].mean()
                )
                if len(success_df) > 0
                else None
            ),
        "mae_median":
            (
                float(
                    success_df["mae"].median()
                )
                if len(success_df) > 0
                else None
            ),
        "rmse_mean":
            (
                float(
                    success_df["rmse"].mean()
                )
                if len(success_df) > 0
                else None
            ),
        "total_prediction_nan_count":
            (
                int(
                    success_df[
                        "prediction_nan_count"
                    ].sum()
                )
                if len(success_df) > 0
                else 0
            ),
    }

    summary_path = os.path.join(
        args.output_dir,
        "train_forecasting_summary.json",
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\n" + "=" * 80)
    print("Evaluation summary")
    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
    )

    print("Saved details:", detail_path)
    print("Saved summary:", summary_path)

    if len(success_df) == 0:
        raise RuntimeError(
            "No successful predictions. "
            "Review skipped/failed rows in "
            f"{detail_path}"
        )


if __name__ == "__main__":
    main()