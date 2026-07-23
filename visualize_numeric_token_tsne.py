#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Visualize numeric-token embeddings with t-SNE.

Example:
python visualize_numeric_token_tsne.py \
  --base_model_path state-spaces/mamba-370m-hf \
  --adapter_path /workspace/outputs/model/mamba-370m-value-norm-finetuned-w001 \
  --output_dir /workspace/outputs/analysis/mamba-value-norm-tsne \
  --embedding_source input \
  --perplexity 30 \
  --seed 42
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
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer

NUMERIC_TOKEN_RE = re.compile(
    r"###([+-]?(?:\d+(?:\.\d*)?|\.\d+)|Nan|NaN|nan)###"
)


def parse_numeric_token(token):
    match = NUMERIC_TOKEN_RE.fullmatch(str(token))
    if match is None:
        return None
    raw = match.group(1)
    if raw.lower() == "nan":
        return None
    return float(raw)


def collect_numeric_tokens(tokenizer):
    rows = []
    for token, token_id in tokenizer.get_vocab().items():
        value = parse_numeric_token(token)
        if value is None:
            continue
        rows.append(
            {
                "token": str(token),
                "token_id": int(token_id),
                "value": float(value),
                "abs_value": abs(float(value)),
            }
        )
    rows.sort(key=lambda row: row["value"])
    if not rows:
        raise ValueError(
            "No numeric tokens found. Load the tokenizer from the adapter directory."
        )
    return rows


def resolve_dtype(name):
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def select_rows(rows, max_tokens):
    if max_tokens is None or max_tokens <= 0 or len(rows) <= max_tokens:
        return rows
    indices = np.linspace(0, len(rows) - 1, max_tokens, dtype=int)
    return [rows[i] for i in indices]


def get_embedding_weight(model, source):
    if source == "input":
        layer = model.get_input_embeddings()
    else:
        layer = model.get_output_embeddings()
    if layer is None or not hasattr(layer, "weight"):
        raise ValueError(f"Embedding weight not found for source={source}")
    return layer.weight


def save_scatter(df, color_column, cmap, colorbar_label, title, output_path):
    fig, ax = plt.subplots(figsize=(9, 7))
    scatter = ax.scatter(
        df["tsne_x"],
        df["tsne_y"],
        c=df[color_column],
        cmap=cmap,
        s=8,
        alpha=0.75,
        linewidths=0,
    )
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label(colorbar_label)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", required=True)
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--embedding_source", choices=["input", "output"], default="input"
    )
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=10000,
        help="0 or negative means all numeric tokens; otherwise evenly sample by value.",
    )
    parser.add_argument(
        "--pca_components",
        type=int,
        default=50,
        help="PCA dimensions before t-SNE. Use 0 to skip PCA.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.adapter_path,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        torch_dtype=resolve_dtype(args.dtype),
        device_map="auto",
        trust_remote_code=True,
    )
    model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(
        model,
        args.adapter_path,
        is_trainable=False,
    )
    model.eval()

    all_rows = collect_numeric_tokens(tokenizer)
    rows = select_rows(all_rows, args.max_tokens)
    print(f"numeric tokens: all={len(all_rows)}, visualized={len(rows)}")

    weight = get_embedding_weight(model, args.embedding_source)
    token_ids = torch.tensor(
        [row["token_id"] for row in rows],
        dtype=torch.long,
        device=weight.device,
    )

    with torch.no_grad():
        embeddings = weight[token_ids].float().cpu().numpy()

    norms = np.linalg.norm(embeddings, axis=1)

    # Standardization prevents a small number of large-variance dimensions from
    # dominating PCA/t-SNE. Norms are measured before this transformation.
    features = StandardScaler().fit_transform(embeddings)

    pca_components = min(
        args.pca_components,
        features.shape[0] - 1,
        features.shape[1],
    )
    if pca_components > 0 and pca_components < features.shape[1]:
        features_for_tsne = PCA(
            n_components=pca_components,
            random_state=args.seed,
        ).fit_transform(features)
    else:
        features_for_tsne = features

    if args.perplexity >= len(rows):
        raise ValueError(
            f"perplexity ({args.perplexity}) must be smaller than token count ({len(rows)})."
        )

    tsne = TSNE(
        n_components=2,
        perplexity=args.perplexity,
        init="pca",
        learning_rate="auto",
        random_state=args.seed,
        n_iter=1000,
        metric="euclidean",
    )
    coordinates = tsne.fit_transform(features_for_tsne)

    df = pd.DataFrame(rows)
    df["embedding_norm"] = norms
    df["tsne_x"] = coordinates[:, 0]
    df["tsne_y"] = coordinates[:, 1]

    csv_path = output_dir / f"numeric_token_tsne_{args.embedding_source}.csv"
    df.to_csv(csv_path, index=False)

    save_scatter(
        df,
        color_column="value",
        cmap="coolwarm",
        colorbar_label="value",
        title="t-SNE of numeric-token embeddings colored by value",
        output_path=output_dir / f"tsne_colored_by_value_{args.embedding_source}.png",
    )
    save_scatter(
        df,
        color_column="abs_value",
        cmap="viridis",
        colorbar_label="|value|",
        title="t-SNE of numeric-token embeddings colored by |value|",
        output_path=output_dir / f"tsne_colored_by_abs_value_{args.embedding_source}.png",
    )
    save_scatter(
        df,
        color_column="embedding_norm",
        cmap="plasma",
        colorbar_label="embedding norm",
        title="t-SNE of numeric-token embeddings colored by embedding norm",
        output_path=output_dir / f"tsne_colored_by_norm_{args.embedding_source}.png",
    )

    summary = {
        "base_model_path": args.base_model_path,
        "adapter_path": args.adapter_path,
        "embedding_source": args.embedding_source,
        "numeric_token_count_all": len(all_rows),
        "numeric_token_count_visualized": len(rows),
        "perplexity": args.perplexity,
        "seed": args.seed,
        "pca_components": pca_components,
        "corr_value_embedding_norm": float(df["value"].corr(df["embedding_norm"])),
        "corr_abs_value_embedding_norm": float(
            df["abs_value"].corr(df["embedding_norm"])
        ),
        "embedding_norm_mean": float(df["embedding_norm"].mean()),
        "embedding_norm_std": float(df["embedding_norm"].std()),
    }
    with (output_dir / f"numeric_token_tsne_summary_{args.embedding_source}.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("Saved to:", output_dir)


if __name__ == "__main__":
    main()
