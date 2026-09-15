#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
from transformers import AutoModelForCausalLM, AutoTokenizer


NUMERIC_TOKEN_RE = re.compile(
    r"###([+-]?(?:\d+(?:\.\d*)?|\.\d+)|Nan|NaN|nan)###"
)


def parse_numeric_token(token):
    match = NUMERIC_TOKEN_RE.fullmatch(str(token))
    if match is None or match.group(1).lower() == "nan":
        return None
    return float(match.group(1))


def collect_numeric_tokens(tokenizer):
    rows = []

    for token, token_id in tokenizer.get_vocab().items():
        value = parse_numeric_token(token)
        if value is None:
            continue

        rows.append({
            "token": str(token),
            "token_id": int(token_id),
            "value": float(value),
            "abs_value": abs(float(value)),
            "sign": "negative" if value < 0 else ("positive" if value > 0 else "zero"),
        })

    rows.sort(key=lambda row: row["value"])

    if not rows:
        raise ValueError(
            "No numeric tokens were found. Load the tokenizer from the adapter directory."
        )

    return rows


def resolve_dtype(name):
    dtype_map = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }

    if name not in dtype_map:
        raise ValueError(f"Unsupported dtype: {name}")

    return dtype_map[name]


def get_embedding_weight(model, source):
    layer = model.get_input_embeddings() if source == "input" else model.get_output_embeddings()

    if layer is None:
        raise ValueError(f"Embedding layer was not found for source={source}.")

    modules_to_save = getattr(layer, "modules_to_save", None)
    active_adapter = getattr(layer, "active_adapter", "default")

    if modules_to_save is not None:
        if isinstance(active_adapter, (list, tuple)):
            active_adapter = active_adapter[0]

        if active_adapter in modules_to_save:
            print(f"Using modules_to_save weight: adapter={active_adapter}")
            return modules_to_save[active_adapter].weight

        if "default" in modules_to_save:
            print("Using modules_to_save weight: adapter=default")
            return modules_to_save["default"].weight

    if not hasattr(layer, "weight"):
        raise ValueError(f"Embedding weight was not found for source={source}.")

    print("Using layer.weight directly:", type(layer).__name__)
    return layer.weight


def select_rows(rows, max_tokens):
    if max_tokens is None or max_tokens <= 0 or len(rows) <= max_tokens:
        return rows

    indices = np.linspace(0, len(rows) - 1, max_tokens, dtype=int)
    return [rows[index] for index in indices]


def prepare_features(embeddings, mode):
    if mode == "raw":
        return embeddings

    if mode == "direction":
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / np.clip(norms, 1e-12, None)

    if mode == "centered":
        return embeddings - embeddings.mean(axis=0, keepdims=True)

    raise ValueError(f"Unsupported feature mode: {mode}")


def detect_norm_outliers(norms, quantile=0.999, mad_threshold=10.0):
    norms = np.asarray(norms, dtype=np.float64)
    median = np.median(norms)
    mad = np.median(np.abs(norms - median))
    quantile_threshold = np.quantile(norms, quantile)

    if mad < 1e-12:
        robust_z = np.zeros_like(norms)
    else:
        robust_z = 0.6745 * (norms - median) / mad

    outlier_mask = (norms > quantile_threshold) | (robust_z > mad_threshold)

    stats = {
        "mean": float(norms.mean()),
        "std": float(norms.std()),
        "median": float(median),
        "mad": float(mad),
        "q99": float(np.quantile(norms, 0.99)),
        "q999": float(np.quantile(norms, 0.999)),
        "max": float(norms.max()),
        "outlier_count": int(outlier_mask.sum()),
        "outlier_ratio": float(outlier_mask.mean()),
    }

    return outlier_mask, robust_z, stats


def fit_pca(features, n_components):
    component_count = min(n_components, features.shape[0], features.shape[1])

    if component_count < 3:
        raise ValueError("At least 3 PCA components are required.")

    pca = PCA(n_components=component_count, random_state=3407)
    coordinates = pca.fit_transform(features)

    return pca, coordinates


def fit_tsne(features, perplexity, pre_pca_dimensions, max_iter, seed):
    if len(features) <= 3:
        raise ValueError("At least 4 samples are required for t-SNE.")

    effective_perplexity = min(float(perplexity), max(2.0, (len(features) - 1) / 3.0))
    pre_pca_dimensions = min(pre_pca_dimensions, features.shape[0] - 1, features.shape[1])

    if pre_pca_dimensions >= 2 and features.shape[1] > pre_pca_dimensions:
        print(
            f"Reducing t-SNE input dimensions: "
            f"{features.shape[1]} -> {pre_pca_dimensions}"
        )
        reducer = PCA(n_components=pre_pca_dimensions, random_state=seed)
        tsne_input = reducer.fit_transform(features)
        pre_pca_variance = float(reducer.explained_variance_ratio_.sum())
    else:
        tsne_input = features
        pre_pca_variance = 1.0

    print("Running t-SNE")
    print("Token count:", len(tsne_input))
    print("Input dimensions:", tsne_input.shape[1])
    print("Perplexity:", effective_perplexity)
    print("Maximum iterations:", max_iter)

    common_kwargs = {
        "n_components": 2,
        "perplexity": effective_perplexity,
        "learning_rate": "auto",
        "init": "pca",
        "random_state": seed,
        "metric": "euclidean",
        "verbose": 1,
    }

    try:
        tsne = TSNE(max_iter=max_iter, **common_kwargs)
    except TypeError:
        tsne = TSNE(n_iter=max_iter, **common_kwargs)

    coordinates = tsne.fit_transform(tsne_input)

    return coordinates, {
        "perplexity": float(effective_perplexity),
        "pre_pca_dimensions": int(tsne_input.shape[1]),
        "pre_pca_explained_variance": pre_pca_variance,
        "kl_divergence": float(tsne.kl_divergence_),
        "iterations": int(getattr(tsne, "n_iter_", max_iter)),
    }


def save_color_scatter(
    df,
    x_column,
    y_column,
    color_column,
    cmap,
    colorbar_label,
    title,
    output_path,
):
    fig, axis = plt.subplots(figsize=(9, 7))

    scatter = axis.scatter(
        df[x_column],
        df[y_column],
        c=df[color_column],
        cmap=cmap,
        s=8,
        alpha=0.75,
        linewidths=0,
    )

    colorbar = fig.colorbar(scatter, ax=axis)
    colorbar.set_label(colorbar_label)
    axis.set_xlabel(x_column.upper())
    axis.set_ylabel(y_column.upper())
    axis.set_title(title)

    fig.tight_layout()
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def save_sign_scatter(df, x_column, y_column, title, output_path):
    sign_colors = {
        "negative": "tab:blue",
        "zero": "tab:gray",
        "positive": "tab:red",
    }

    fig, axis = plt.subplots(figsize=(9, 7))

    for sign_name in ["negative", "zero", "positive"]:
        subset = df[df["sign"] == sign_name]
        if len(subset) == 0:
            continue

        axis.scatter(
            subset[x_column],
            subset[y_column],
            s=8,
            alpha=0.7,
            linewidths=0,
            color=sign_colors[sign_name],
            label=sign_name,
        )

    axis.set_xlabel(x_column.upper())
    axis.set_ylabel(y_column.upper())
    axis.set_title(title)
    axis.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def save_zoomed_pca(df, output_path):
    x_low, x_high = df["pc1"].quantile([0.005, 0.995])
    y_low, y_high = df["pc2"].quantile([0.005, 0.995])

    fig, axis = plt.subplots(figsize=(9, 7))

    scatter = axis.scatter(
        df["pc1"],
        df["pc2"],
        c=df["value"],
        cmap="coolwarm",
        s=8,
        alpha=0.75,
        linewidths=0,
    )

    axis.set_xlim(x_low, x_high)
    axis.set_ylim(y_low, y_high)
    axis.set_xlabel("PC1")
    axis.set_ylabel("PC2")
    axis.set_title("Numeric-token PCA, central 99% view")

    colorbar = fig.colorbar(scatter, ax=axis)
    colorbar.set_label("value")

    fig.tight_layout()
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def save_pca_outputs(df, pca, output_dir, mode):
    save_color_scatter(
        df,
        "pc1",
        "pc2",
        "value",
        "coolwarm",
        "value",
        f"Numeric-token PCA colored by value ({mode})",
        output_dir / "pca_pc1_pc2_by_value.png",
    )

    save_color_scatter(
        df,
        "pc1",
        "pc2",
        "abs_value",
        "viridis",
        "|value|",
        f"Numeric-token PCA colored by |value| ({mode})",
        output_dir / "pca_pc1_pc2_by_abs_value.png",
    )

    save_color_scatter(
        df,
        "pc1",
        "pc2",
        "embedding_norm",
        "plasma",
        "embedding norm",
        f"Numeric-token PCA colored by norm ({mode})",
        output_dir / "pca_pc1_pc2_by_norm.png",
    )

    save_sign_scatter(
        df,
        "pc1",
        "pc2",
        f"Numeric-token PCA colored by sign ({mode})",
        output_dir / "pca_pc1_pc2_by_sign.png",
    )

    save_zoomed_pca(
        df,
        output_dir / "pca_pc1_pc2_by_value_zoomed.png",
    )

    fig = plt.figure(figsize=(10, 8))
    axis = fig.add_subplot(111, projection="3d")

    scatter = axis.scatter(
        df["pc1"],
        df["pc2"],
        df["pc3"],
        c=df["value"],
        cmap="coolwarm",
        s=6,
        alpha=0.65,
    )

    colorbar = fig.colorbar(scatter, ax=axis, pad=0.1)
    colorbar.set_label("value")
    axis.set_xlabel("PC1")
    axis.set_ylabel("PC2")
    axis.set_zlabel("PC3")
    axis.set_title(f"Numeric-token PCA: PC1-PC3 ({mode})")

    fig.tight_layout()
    fig.savefig(output_dir / "pca_pc1_pc2_pc3_by_value.png", dpi=240)
    plt.close(fig)

    variance_df = pd.DataFrame({
        "component": np.arange(1, len(pca.explained_variance_ratio_) + 1),
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_explained_variance_ratio": np.cumsum(
            pca.explained_variance_ratio_
        ),
    })

    variance_df.to_csv(
        output_dir / "pca_explained_variance.csv",
        index=False,
    )

    fig, axis = plt.subplots(figsize=(8, 5))

    axis.plot(
        variance_df["component"],
        variance_df["cumulative_explained_variance_ratio"],
        marker="o",
    )

    axis.set_xlabel("Number of principal components")
    axis.set_ylabel("Cumulative explained variance ratio")
    axis.set_title("PCA cumulative explained variance")

    fig.tight_layout()
    fig.savefig(output_dir / "pca_explained_variance.png", dpi=240)
    plt.close(fig)

    return variance_df


def save_tsne_outputs(df, output_dir, mode):
    save_color_scatter(
        df,
        "tsne1",
        "tsne2",
        "value",
        "coolwarm",
        "value",
        f"Numeric-token t-SNE colored by value ({mode})",
        output_dir / "tsne_by_value.png",
    )

    save_color_scatter(
        df,
        "tsne1",
        "tsne2",
        "abs_value",
        "viridis",
        "|value|",
        f"Numeric-token t-SNE colored by |value| ({mode})",
        output_dir / "tsne_by_abs_value.png",
    )

    save_color_scatter(
        df,
        "tsne1",
        "tsne2",
        "embedding_norm",
        "plasma",
        "embedding norm",
        f"Numeric-token t-SNE colored by norm ({mode})",
        output_dir / "tsne_by_norm.png",
    )

    save_sign_scatter(
        df,
        "tsne1",
        "tsne2",
        f"Numeric-token t-SNE colored by sign ({mode})",
        output_dir / "tsne_by_sign.png",
    )


def run_trimmed_pca(rows, embeddings, embedding_norms, outlier_mask, mode, n_components, output_dir):
    keep_mask = ~outlier_mask

    if keep_mask.sum() < 3:
        print("Too few non-outlier tokens. Trimmed PCA was skipped.")
        return None

    trimmed_embeddings = embeddings[keep_mask]
    trimmed_rows = [row for row, keep in zip(rows, keep_mask) if keep]
    trimmed_norms = embedding_norms[keep_mask]
    trimmed_features = prepare_features(trimmed_embeddings, mode)
    trimmed_pca, trimmed_coordinates = fit_pca(trimmed_features, n_components)

    trimmed_df = pd.DataFrame(trimmed_rows)
    trimmed_df["embedding_norm"] = trimmed_norms
    trimmed_df["pc1"] = trimmed_coordinates[:, 0]
    trimmed_df["pc2"] = trimmed_coordinates[:, 1]
    trimmed_df["pc3"] = trimmed_coordinates[:, 2]

    trimmed_df.to_csv(
        output_dir / "chattime_embedding_pca_trimmed.csv",
        index=False,
    )

    save_color_scatter(
        trimmed_df,
        "pc1",
        "pc2",
        "value",
        "coolwarm",
        "value",
        f"Numeric-token PCA without norm outliers ({mode})",
        output_dir / "pca_pc1_pc2_by_value_trimmed.png",
    )

    return {
        "token_count": int(len(trimmed_df)),
        "pc1_explained_variance_ratio": float(
            trimmed_pca.explained_variance_ratio_[0]
        ),
        "pc2_explained_variance_ratio": float(
            trimmed_pca.explained_variance_ratio_[1]
        ),
        "pc3_explained_variance_ratio": float(
            trimmed_pca.explained_variance_ratio_[2]
        ),
        "corr_value_pc1": float(
            trimmed_df["value"].corr(trimmed_df["pc1"])
        ),
        "corr_value_pc2": float(
            trimmed_df["value"].corr(trimmed_df["pc2"])
        ),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--base_model_path", type=str, required=True)
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)

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
        "--feature_mode",
        "--pca_mode",
        dest="feature_mode",
        type=str,
        choices=["raw", "direction", "centered"],
        default="raw",
    )

    parser.add_argument("--n_components", type=int, default=20)
    parser.add_argument("--max_tokens", type=int, default=10000)
    parser.add_argument("--outlier_quantile", type=float, default=0.999)
    parser.add_argument("--outlier_mad_threshold", type=float, default=10.0)

    parser.add_argument(
        "--run_tsne",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--tsne_perplexity", type=float, default=30.0)
    parser.add_argument("--tsne_pre_pca_dimensions", type=int, default=50)
    parser.add_argument("--tsne_max_iter", type=int, default=1500)
    parser.add_argument("--random_seed", type=int, default=3407)

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_path = (
        args.tokenizer_path
        or args.adapter_path
        or args.base_model_path
    )

    print("Loading tokenizer from:", tokenizer_path)

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
    )

    print("Loading base model:", args.base_model_path)

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        torch_dtype=resolve_dtype(args.dtype),
        device_map="auto",
        trust_remote_code=True,
    )

    model_vocab_size = model.get_input_embeddings().weight.shape[0]
    tokenizer_vocab_size = len(tokenizer)

    print("Model vocabulary size:", model_vocab_size)
    print("Tokenizer vocabulary size:", tokenizer_vocab_size)

    if args.adapter_path is not None:
        if model_vocab_size != tokenizer_vocab_size:
            print(
                "Resizing base-model embeddings:",
                model_vocab_size,
                "->",
                tokenizer_vocab_size,
            )
            model.resize_token_embeddings(tokenizer_vocab_size)

        print("Loading adapter from:", args.adapter_path)

        model = PeftModel.from_pretrained(
            model,
            args.adapter_path,
            is_trainable=False,
        )
    elif model_vocab_size != tokenizer_vocab_size:
        raise ValueError(
            "Full-model vocabulary size does not match tokenizer size: "
            f"model={model_vocab_size}, tokenizer={tokenizer_vocab_size}."
        )

    model.eval()

    all_rows = collect_numeric_tokens(tokenizer)
    rows = select_rows(all_rows, args.max_tokens)

    print("All numeric tokens:", len(all_rows))
    print("Analyzed numeric tokens:", len(rows))

    weight = get_embedding_weight(model, args.embedding_source)

    if weight.shape[0] != len(tokenizer):
        raise ValueError(
            f"Vocabulary mismatch after model loading: "
            f"tokenizer={len(tokenizer)}, embedding={weight.shape[0]}"
        )

    token_ids = torch.tensor(
        [row["token_id"] for row in rows],
        dtype=torch.long,
        device=weight.device,
    )

    with torch.inference_mode():
        embeddings = weight[token_ids].detach().float().cpu().numpy()

    embedding_norms = np.linalg.norm(embeddings, axis=1)

    outlier_mask, robust_z, norm_stats = detect_norm_outliers(
        embedding_norms,
        quantile=args.outlier_quantile,
        mad_threshold=args.outlier_mad_threshold,
    )

    features = prepare_features(embeddings, args.feature_mode)
    pca, pca_coordinates = fit_pca(features, args.n_components)
    component_count = pca_coordinates.shape[1]

    df = pd.DataFrame(rows)
    df["embedding_norm"] = embedding_norms
    df["norm_robust_z"] = robust_z
    df["is_norm_outlier"] = outlier_mask

    for index in range(component_count):
        df[f"pc{index + 1}"] = pca_coordinates[:, index]

    non_outlier_embeddings = embeddings[~outlier_mask]

    robust_center = (
        np.median(non_outlier_embeddings, axis=0)
        if len(non_outlier_embeddings) > 0
        else np.median(embeddings, axis=0)
    )

    df["distance_from_robust_center"] = np.linalg.norm(
        embeddings - robust_center,
        axis=1,
    )

    tsne_summary = None

    if args.run_tsne:
        tsne_coordinates, tsne_summary = fit_tsne(
            features=features,
            perplexity=args.tsne_perplexity,
            pre_pca_dimensions=args.tsne_pre_pca_dimensions,
            max_iter=args.tsne_max_iter,
            seed=args.random_seed,
        )

        df["tsne1"] = tsne_coordinates[:, 0]
        df["tsne2"] = tsne_coordinates[:, 1]

    csv_path = output_dir / "chattime_embedding_analysis.csv"
    df.to_csv(csv_path, index=False)

    outlier_df = df[df["is_norm_outlier"]].copy()
    outlier_df = outlier_df.sort_values(
        "embedding_norm",
        ascending=False,
    )
    outlier_df.to_csv(
        output_dir / "embedding_norm_outliers.csv",
        index=False,
    )

    print("\nEmbedding norm statistics")
    print(json.dumps(norm_stats, ensure_ascii=False, indent=2))

    if len(outlier_df) > 0:
        print("\nLargest embedding norm outliers")
        print(
            outlier_df[
                [
                    "token",
                    "token_id",
                    "value",
                    "embedding_norm",
                    "norm_robust_z",
                    "pc1",
                    "pc2",
                ]
            ].head(30).to_string(index=False)
        )

    variance_df = save_pca_outputs(
        df,
        pca,
        output_dir,
        args.feature_mode,
    )

    if args.run_tsne:
        save_tsne_outputs(
            df,
            output_dir,
            args.feature_mode,
        )

    trimmed_summary = run_trimmed_pca(
        rows=rows,
        embeddings=embeddings,
        embedding_norms=embedding_norms,
        outlier_mask=outlier_mask,
        mode=args.feature_mode,
        n_components=args.n_components,
        output_dir=output_dir,
    )

    fig, axis = plt.subplots(figsize=(8, 5))
    axis.scatter(
        df["abs_value"],
        df["embedding_norm"],
        s=5,
        alpha=0.5,
        linewidths=0,
    )
    axis.set_xlabel("|value|")
    axis.set_ylabel("Embedding norm")
    axis.set_title("|value| vs embedding norm")
    fig.tight_layout()
    fig.savefig(
        output_dir / "embedding_norm_vs_abs_value.png",
        dpi=240,
    )
    plt.close(fig)

    summary = {
        "base_model_path": args.base_model_path,
        "adapter_path": args.adapter_path,
        "tokenizer_path": tokenizer_path,
        "embedding_source": args.embedding_source,
        "feature_mode": args.feature_mode,
        "numeric_token_count_all": int(len(all_rows)),
        "numeric_token_count_analyzed": int(len(df)),
        "embedding_dimension": int(embeddings.shape[1]),
        "pca_component_count": int(component_count),
        "pc1_explained_variance_ratio": float(
            pca.explained_variance_ratio_[0]
        ),
        "pc2_explained_variance_ratio": float(
            pca.explained_variance_ratio_[1]
        ),
        "pc3_explained_variance_ratio": float(
            pca.explained_variance_ratio_[2]
        ),
        "first_2_pc_cumulative_variance": float(
            pca.explained_variance_ratio_[:2].sum()
        ),
        "first_3_pc_cumulative_variance": float(
            pca.explained_variance_ratio_[:3].sum()
        ),
        "corr_value_pc1": float(df["value"].corr(df["pc1"])),
        "corr_value_pc2": float(df["value"].corr(df["pc2"])),
        "corr_abs_value_pc1": float(
            df["abs_value"].corr(df["pc1"])
        ),
        "corr_abs_value_pc2": float(
            df["abs_value"].corr(df["pc2"])
        ),
        "corr_value_embedding_norm": float(
            df["value"].corr(df["embedding_norm"])
        ),
        "corr_abs_value_embedding_norm": float(
            df["abs_value"].corr(df["embedding_norm"])
        ),
        "embedding_norm_statistics": norm_stats,
        "trimmed_pca": trimmed_summary,
        "tsne": tsne_summary,
    }

    summary_path = output_dir / "summary.json"

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

    print("\nAnalysis summary")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nSaved detailed CSV:", csv_path)
    print("Saved PCA variance CSV:", output_dir / "pca_explained_variance.csv")
    print("Saved all visualizations to:", output_dir)


if __name__ == "__main__":
    main()