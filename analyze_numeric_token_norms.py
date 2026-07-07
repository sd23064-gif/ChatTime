#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_numeric_token_norms.py

Mamba / PEFT adapter から追加数値トークンの embedding norm を抽出し、
|value| と embedding_norm の相関を確認するスクリプト。

主な出力:
  - numeric_token_norms_input.csv
  - numeric_token_norm_summary_input.json
  - scatter_abs_value_vs_norm_input.png
  - scatter_value_vs_norm_input.png
  - numeric_token_norm_by_abs_value_bin_input.csv
  - mean_norm_by_abs_value_bin_input.png

例:
python analyze_numeric_token_norms.py \
  --base_model_path state-spaces/mamba-370m-hf \
  --adapter_path /workspace/outputs/model/mamba-370m-value-norm-abs-w001 \
  --output_dir /workspace/outputs/analysis/mamba-370m-value-norm-abs-w001_norm_analysis \
  --embedding_source input
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


NUMERIC_TOKEN_RE = re.compile(
    r"###([+-]?(?:\d+(?:\.\d*)?|\.\d+)|Nan|NaN|nan)###"
)


def parse_numeric_token(token):
    """###-0.1234### のような token から float value を返す。Nan や非数値は None。"""
    match = NUMERIC_TOKEN_RE.fullmatch(str(token))
    if match is None:
        return None

    raw = match.group(1)
    if raw.lower() == "nan":
        return None

    return float(raw)


def collect_numeric_tokens(tokenizer):
    """tokenizer vocab から追加数値 token を集める。"""
    rows = []

    for token, token_id in tokenizer.get_vocab().items():
        value = parse_numeric_token(token)
        if value is None:
            continue

        rows.append(
            {
                "token": token,
                "token_id": int(token_id),
                "value": float(value),
                "abs_value": abs(float(value)),
            }
        )

    rows = sorted(rows, key=lambda x: x["value"])

    if len(rows) == 0:
        raise ValueError(
            "No numeric tokens found in tokenizer. "
            "Make sure tokenizer is loaded from the adapter directory that contains added_tokens.json."
        )

    return rows


def resolve_dtype(dtype_name):
    if dtype_name == "fp16":
        return torch.float16
    if dtype_name == "bf16":
        return torch.bfloat16
    if dtype_name == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def get_embedding_weight(model, embedding_source):
    if embedding_source == "input":
        embedding_layer = model.get_input_embeddings()
        if embedding_layer is None or not hasattr(embedding_layer, "weight"):
            raise ValueError("Input embedding weight not found.")
        return embedding_layer.weight

    if embedding_source == "output":
        output_layer = model.get_output_embeddings()
        if output_layer is None or not hasattr(output_layer, "weight"):
            raise ValueError("Output embedding / lm_head weight not found.")
        return output_layer.weight

    raise ValueError("embedding_source must be 'input' or 'output'.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str, required=True)
    parser.add_argument("--adapter_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--embedding_source",
        type=str,
        default="input",
        choices=["input", "output"],
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp16",
        choices=["fp16", "bf16", "fp32"],
    )
    parser.add_argument("--bin_count", type=int, default=20)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading tokenizer from adapter_path:", args.adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(
        args.adapter_path,
        trust_remote_code=True,
    )

    torch_dtype = resolve_dtype(args.dtype)

    print("Loading base model:", args.base_model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    # adapter 側 tokenizer には追加 token が含まれるため、base model embedding を resize する。
    model.resize_token_embeddings(len(tokenizer))

    print("Loading PEFT adapter:", args.adapter_path)
    model = PeftModel.from_pretrained(
        model,
        args.adapter_path,
        is_trainable=False,
    )
    model.eval()

    rows = collect_numeric_tokens(tokenizer)
    print("numeric token count:", len(rows))
    print("value range:", rows[0]["value"], rows[-1]["value"])

    weight = get_embedding_weight(model, args.embedding_source)
    device = weight.device

    token_ids = torch.tensor(
        [row["token_id"] for row in rows],
        dtype=torch.long,
        device=device,
    )

    with torch.no_grad():
        embeddings = weight[token_ids].float()
        norms = embeddings.norm(dim=-1).detach().cpu().numpy()

    df = pd.DataFrame(rows)
    df["embedding_norm"] = norms

    corr_value_norm = df["value"].corr(df["embedding_norm"])
    corr_abs_value_norm = df["abs_value"].corr(df["embedding_norm"])

    print("corr(value, embedding_norm):", corr_value_norm)
    print("corr(abs_value, embedding_norm):", corr_abs_value_norm)

    summary = {
        "base_model_path": args.base_model_path,
        "adapter_path": args.adapter_path,
        "embedding_source": args.embedding_source,
        "numeric_token_count": int(len(df)),
        "value_min": float(df["value"].min()),
        "value_max": float(df["value"].max()),
        "abs_value_min": float(df["abs_value"].min()),
        "abs_value_max": float(df["abs_value"].max()),
        "embedding_norm_mean": float(df["embedding_norm"].mean()),
        "embedding_norm_std": float(df["embedding_norm"].std()),
        "embedding_norm_min": float(df["embedding_norm"].min()),
        "embedding_norm_max": float(df["embedding_norm"].max()),
        "corr_value_embedding_norm": float(corr_value_norm),
        "corr_abs_value_embedding_norm": float(corr_abs_value_norm),
    }

    csv_path = output_dir / f"numeric_token_norms_{args.embedding_source}.csv"
    summary_path = output_dir / f"numeric_token_norm_summary_{args.embedding_source}.json"

    df.to_csv(csv_path, index=False)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Saved CSV:", csv_path)
    print("Saved summary:", summary_path)

    # Scatter: |value| vs norm
    plt.figure(figsize=(7, 5))
    plt.scatter(df["abs_value"], df["embedding_norm"], s=3, alpha=0.4)
    plt.xlabel("|value|")
    plt.ylabel("embedding norm")
    plt.title(f"|value| vs embedding norm\ncorr={corr_abs_value_norm:.4f}")
    plt.tight_layout()
    plt.savefig(output_dir / f"scatter_abs_value_vs_norm_{args.embedding_source}.png", dpi=220)
    plt.close()

    # Scatter: value vs norm
    plt.figure(figsize=(7, 5))
    plt.scatter(df["value"], df["embedding_norm"], s=3, alpha=0.4)
    plt.xlabel("value")
    plt.ylabel("embedding norm")
    plt.title(f"value vs embedding norm\ncorr={corr_value_norm:.4f}")
    plt.tight_layout()
    plt.savefig(output_dir / f"scatter_value_vs_norm_{args.embedding_source}.png", dpi=220)
    plt.close()

    # Bin average by |value|
    bins = np.linspace(0.0, 1.0, args.bin_count + 1)
    df["abs_value_bin"] = pd.cut(df["abs_value"], bins=bins, include_lowest=True)

    by_bin = (
        df.groupby("abs_value_bin", observed=True)
        .agg(
            count=("embedding_norm", "size"),
            abs_value_mean=("abs_value", "mean"),
            embedding_norm_mean=("embedding_norm", "mean"),
            embedding_norm_std=("embedding_norm", "std"),
        )
        .reset_index()
    )

    by_bin["abs_value_bin"] = by_bin["abs_value_bin"].astype(str)

    by_bin_path = output_dir / f"numeric_token_norm_by_abs_value_bin_{args.embedding_source}.csv"
    by_bin.to_csv(by_bin_path, index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(by_bin["abs_value_mean"], by_bin["embedding_norm_mean"], marker="o")
    plt.xlabel("mean |value| in bin")
    plt.ylabel("mean embedding norm")
    plt.title("Mean embedding norm by |value| bin")
    plt.tight_layout()
    plt.savefig(output_dir / f"mean_norm_by_abs_value_bin_{args.embedding_source}.png", dpi=220)
    plt.close()

    print("Saved by-bin CSV:", by_bin_path)
    print("Done.")


if __name__ == "__main__":
    main()
