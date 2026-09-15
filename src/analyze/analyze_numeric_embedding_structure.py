#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import gc
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer


NUMERIC_TOKEN_RE = re.compile(
    r"###([+-]?(?:\d+(?:\.\d*)?|\.\d+)|Nan|NaN|nan)###"
)


def resolve_dtype(name):
    dtype_map = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    return dtype_map[name]


def collect_numeric_tokens(tokenizer):
    rows = []

    for token, token_id in tokenizer.get_vocab().items():
        match = NUMERIC_TOKEN_RE.fullmatch(str(token))
        if match is None or match.group(1).lower() == "nan":
            continue

        rows.append({
            "token": str(token),
            "token_id": int(token_id),
            "value": float(match.group(1)),
        })

    rows.sort(key=lambda row: row["value"])

    if not rows:
        raise ValueError("No finite numeric tokens were found.")

    return rows


def get_active_weight(model, embedding_source):
    layer = (
        model.get_input_embeddings()
        if embedding_source == "input"
        else model.get_output_embeddings()
    )

    if layer is None:
        raise ValueError(f"Embedding layer not found: {embedding_source}")

    modules_to_save = getattr(layer, "modules_to_save", None)
    active_adapter = getattr(layer, "active_adapter", "default")

    if modules_to_save is not None:
        if isinstance(active_adapter, (list, tuple)):
            active_adapter = active_adapter[0]

        if active_adapter in modules_to_save:
            print(f"Using modules_to_save weight: {active_adapter}")
            return modules_to_save[active_adapter].weight

        if "default" in modules_to_save:
            print("Using modules_to_save weight: default")
            return modules_to_save["default"].weight

    if not hasattr(layer, "weight"):
        raise ValueError(f"Weight not found: {embedding_source}")

    print(f"Using direct layer weight: {type(layer).__name__}")
    return layer.weight


def load_numeric_embeddings(
    base_model_path,
    adapter_path,
    tokenizer_path,
    embedding_source,
    dtype,
):
    print("\n" + "=" * 80)
    print("Loading checkpoint:", adapter_path)

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
    )

    rows = collect_numeric_tokens(tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    base_vocab_size = model.get_input_embeddings().weight.shape[0]

    if base_vocab_size != len(tokenizer):
        print(f"Resizing vocabulary: {base_vocab_size} -> {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))

    model = PeftModel.from_pretrained(
        model,
        adapter_path,
        is_trainable=False,
    )
    model.eval()

    weight = get_active_weight(
        model,
        embedding_source,
    )

    if weight.shape[0] != len(tokenizer):
        raise ValueError(
            f"Vocabulary mismatch: tokenizer={len(tokenizer)}, "
            f"embedding={weight.shape[0]}"
        )

    token_ids = torch.tensor(
        [row["token_id"] for row in rows],
        dtype=torch.long,
        device=weight.device,
    )

    with torch.no_grad():
        embeddings = (
            weight[token_ids]
            .detach()
            .float()
            .cpu()
            .numpy()
        )

    values = np.asarray(
        [row["value"] for row in rows],
        dtype=np.float64,
    )

    tokens = [row["token"] for row in rows]
    token_ids_numpy = token_ids.cpu().numpy()

    print("Numeric token count:", len(rows))
    print("Embedding shape:", embeddings.shape)
    print("Mean norm:", np.linalg.norm(embeddings, axis=1).mean())
    print("Median norm:", np.median(np.linalg.norm(embeddings, axis=1)))

    del model
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "tokens": tokens,
        "token_ids": token_ids_numpy,
        "values": values,
        "embeddings": embeddings,
    }


def check_alignment(first, second):
    if first["tokens"] != second["tokens"]:
        raise ValueError("Numeric token order differs between checkpoints.")

    if not np.array_equal(first["token_ids"], second["token_ids"]):
        raise ValueError("Numeric token IDs differ between checkpoints.")

    if not np.allclose(first["values"], second["values"]):
        raise ValueError("Numeric values differ between checkpoints.")


def analyze_embedding_drift(first, second):
    first_embeddings = first["embeddings"]
    second_embeddings = second["embeddings"]

    first_norms = np.linalg.norm(first_embeddings, axis=1)
    second_norms = np.linalg.norm(second_embeddings, axis=1)

    delta = second_embeddings - first_embeddings
    l2_change = np.linalg.norm(delta, axis=1)
    relative_change = l2_change / np.clip(first_norms, 1e-12, None)

    first_directions = first_embeddings / np.clip(
        first_norms[:, None], 1e-12, None
    )
    second_directions = second_embeddings / np.clip(
        second_norms[:, None], 1e-12, None
    )

    cosine = np.sum(
        first_directions * second_directions,
        axis=1,
    )
    cosine = np.clip(cosine, -1.0, 1.0)

    drift_df = pd.DataFrame({
        "token": first["tokens"],
        "token_id": first["token_ids"],
        "value": first["values"],
        "first_norm": first_norms,
        "second_norm": second_norms,
        "norm_change": second_norms - first_norms,
        "l2_change": l2_change,
        "relative_l2_change": relative_change,
        "cosine_first_second": cosine,
    })

    summary = {
        "mean_l2_change": float(l2_change.mean()),
        "median_l2_change": float(np.median(l2_change)),
        "mean_relative_l2_change": float(relative_change.mean()),
        "median_relative_l2_change": float(np.median(relative_change)),
        "mean_cosine": float(cosine.mean()),
        "median_cosine": float(np.median(cosine)),
        "ratio_relative_change_below_1pct": float(
            np.mean(relative_change < 0.01)
        ),
        "ratio_relative_change_below_5pct": float(
            np.mean(relative_change < 0.05)
        ),
        "ratio_cosine_above_0_999": float(
            np.mean(cosine > 0.999)
        ),
    }

    return drift_df, summary


def evaluate_value_probe(embeddings, values, seed):
    indices = np.arange(len(values))

    train_indices, test_indices = train_test_split(
        indices,
        test_size=0.2,
        random_state=seed,
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(embeddings[train_indices])
    x_test = scaler.transform(embeddings[test_indices])

    model = Ridge(alpha=1.0)
    model.fit(x_train, values[train_indices])

    predictions = model.predict(x_test)

    return {
        "r2": float(r2_score(values[test_indices], predictions)),
        "mae": float(mean_absolute_error(values[test_indices], predictions)),
        "test_count": int(len(test_indices)),
    }


def evaluate_abs_value_probe(embeddings, values, seed):
    targets = np.abs(values)
    indices = np.arange(len(targets))

    train_indices, test_indices = train_test_split(
        indices,
        test_size=0.2,
        random_state=seed,
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(embeddings[train_indices])
    x_test = scaler.transform(embeddings[test_indices])

    model = Ridge(alpha=1.0)
    model.fit(x_train, targets[train_indices])

    predictions = model.predict(x_test)

    return {
        "r2": float(r2_score(targets[test_indices], predictions)),
        "mae": float(mean_absolute_error(targets[test_indices], predictions)),
        "test_count": int(len(test_indices)),
    }


def evaluate_sign_probe(embeddings, values, seed):
    labels = (values > 0).astype(np.int64)
    indices = np.arange(len(labels))

    train_indices, test_indices = train_test_split(
        indices,
        test_size=0.2,
        random_state=seed,
        stratify=labels,
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(embeddings[train_indices])
    x_test = scaler.transform(embeddings[test_indices])

    classifier = LogisticRegression(
        max_iter=3000,
        random_state=seed,
    )
    classifier.fit(x_train, labels[train_indices])

    predictions = classifier.predict(x_test)

    return {
        "accuracy": float(
            accuracy_score(
                labels[test_indices],
                predictions,
            )
        ),
        "test_count": int(len(test_indices)),
    }


def evaluate_pairwise_distance(
    embeddings,
    values,
    pair_count,
    seed,
):
    rng = np.random.default_rng(seed)
    token_count = len(values)

    left = rng.integers(
        0,
        token_count,
        size=pair_count,
    )
    right = rng.integers(
        0,
        token_count,
        size=pair_count,
    )

    valid = left != right
    left = left[valid]
    right = right[valid]

    norms = np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True,
    )
    normalized = embeddings / np.clip(
        norms,
        1e-12,
        None,
    )

    value_distance = np.abs(
        values[left] - values[right]
    )

    cosine_similarity = np.sum(
        normalized[left] * normalized[right],
        axis=1,
    )
    cosine_distance = 1.0 - np.clip(
        cosine_similarity,
        -1.0,
        1.0,
    )

    euclidean_distance = np.linalg.norm(
        embeddings[left] - embeddings[right],
        axis=1,
    )

    cosine_result = spearmanr(
        value_distance,
        cosine_distance,
    )
    euclidean_result = spearmanr(
        value_distance,
        euclidean_distance,
    )

    return {
        "pair_count": int(len(left)),
        "spearman_value_cosine_distance": float(
            cosine_result.statistic
        ),
        "cosine_distance_p_value": float(
            cosine_result.pvalue
        ),
        "spearman_value_euclidean_distance": float(
            euclidean_result.statistic
        ),
        "euclidean_distance_p_value": float(
            euclidean_result.pvalue
        ),
    }


def analyze_checkpoint(data, pair_count, seed):
    embeddings = data["embeddings"]
    values = data["values"]

    return {
        "value_probe": evaluate_value_probe(
            embeddings,
            values,
            seed,
        ),
        "abs_value_probe": evaluate_abs_value_probe(
            embeddings,
            values,
            seed,
        ),
        "sign_probe": evaluate_sign_probe(
            embeddings,
            values,
            seed,
        ),
        "pairwise_distance": evaluate_pairwise_distance(
            embeddings,
            values,
            pair_count,
            seed,
        ),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base_model_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--first_adapter_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--second_adapter_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--embedding_source",
        type=str,
        choices=["input", "output"],
        default="input",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["fp16", "bf16", "fp32"],
        default="fp16",
    )
    parser.add_argument(
        "--pair_count",
        type=int,
        default=100000,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=3407,
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    tokenizer_path = (
        args.tokenizer_path
        if args.tokenizer_path is not None
        else args.second_adapter_path
    )

    dtype = resolve_dtype(args.dtype)

    first = load_numeric_embeddings(
        base_model_path=args.base_model_path,
        adapter_path=args.first_adapter_path,
        tokenizer_path=tokenizer_path,
        embedding_source=args.embedding_source,
        dtype=dtype,
    )

    second = load_numeric_embeddings(
        base_model_path=args.base_model_path,
        adapter_path=args.second_adapter_path,
        tokenizer_path=tokenizer_path,
        embedding_source=args.embedding_source,
        dtype=dtype,
    )

    check_alignment(first, second)

    drift_df, drift_summary = analyze_embedding_drift(
        first,
        second,
    )

    drift_path = output_dir / "embedding_drift.csv"
    drift_df.to_csv(
        drift_path,
        index=False,
    )

    first_analysis = analyze_checkpoint(
        first,
        args.pair_count,
        args.seed,
    )

    second_analysis = analyze_checkpoint(
        second,
        args.pair_count,
        args.seed,
    )

    summary = {
        "base_model_path": args.base_model_path,
        "first_adapter_path": args.first_adapter_path,
        "second_adapter_path": args.second_adapter_path,
        "tokenizer_path": tokenizer_path,
        "embedding_source": args.embedding_source,
        "numeric_token_count": int(
            len(first["values"])
        ),
        "embedding_dimension": int(
            first["embeddings"].shape[1]
        ),
        "embedding_drift": drift_summary,
        "first_checkpoint": first_analysis,
        "second_checkpoint": second_analysis,
    }

    summary_path = output_dir / "embedding_structure_summary.json"

    with summary_path.open(
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
    print("Embedding drift")
    print(
        json.dumps(
            drift_summary,
            ensure_ascii=False,
            indent=2,
        )
    )

    print("\nFirst checkpoint analysis")
    print(
        json.dumps(
            first_analysis,
            ensure_ascii=False,
            indent=2,
        )
    )

    print("\nSecond checkpoint analysis")
    print(
        json.dumps(
            second_analysis,
            ensure_ascii=False,
            indent=2,
        )
    )

    print("\nLargest relative embedding changes")
    print(
        drift_df.nlargest(
            20,
            "relative_l2_change",
        )[
            [
                "token",
                "token_id",
                "value",
                "first_norm",
                "second_norm",
                "l2_change",
                "relative_l2_change",
                "cosine_first_second",
            ]
        ].to_string(index=False)
    )

    print("\nSaved:")
    print(drift_path)
    print(summary_path)


if __name__ == "__main__":
    main()