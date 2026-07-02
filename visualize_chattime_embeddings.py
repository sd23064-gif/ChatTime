import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def parse_value(token):
    m = re.search(r"###([+-]?(?:\d+(?:\.\d*)?|\.\d+|NaN|Nan|nan))###", str(token))
    if m is None:
        return np.nan
    try:
        return float(m.group(1))
    except Exception:
        return np.nan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="ChengsenWang/ChatTime-1-7B-Chat")
    parser.add_argument("--output_dir", type=str, default="outputs/chattime_embedding_vis")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--sample_step", type=int, default=1)
    parser.add_argument("--draw_sign_pairs", action="store_true")
    parser.add_argument("--max_pair_lines", type=int, default=500)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )

    rows = []
    for token_id, token_obj in tokenizer.added_tokens_decoder.items():
        token = str(token_obj)
        value = parse_value(token)

        if not np.isnan(value):
            rows.append({
                "token": token,
                "token_id": int(token_id),
                "value": value,
                "abs_value": abs(value),
                "sign": "positive" if value > 0 else "negative" if value < 0 else "zero",
            })

    token_df = pd.DataFrame(rows).sort_values("token_id").reset_index(drop=True)

    if args.sample_step > 1:
        token_df = token_df.iloc[::args.sample_step].reset_index(drop=True)

    print("numeric token count:", len(token_df))
    print(token_df.head())

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map=None,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.to(args.device)
    model.eval()

    emb = model.get_input_embeddings().weight.detach().float().cpu()

    token_ids = token_df["token_id"].astype(int).to_numpy()
    X = emb[token_ids].numpy()

    # PCA
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=3407)
    Z = pca.fit_transform(X_scaled)

    token_df["pc1"] = Z[:, 0]
    token_df["pc2"] = Z[:, 1]

    token_df.to_csv(out / "chattime_embedding_pca_coordinates.csv", index=False)

    # 1. value color
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(
        token_df["pc1"],
        token_df["pc2"],
        c=token_df["value"],
        s=5,
        cmap="coolwarm",
        alpha=0.8,
    )
    plt.colorbar(sc, label="token value")
    plt.title("ChatTime numeric token embeddings PCA colored by value")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(out / "pca_colored_by_value.png", dpi=220)
    plt.close()

    # 2. abs value color
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(
        token_df["pc1"],
        token_df["pc2"],
        c=token_df["abs_value"],
        s=5,
        cmap="viridis",
        alpha=0.8,
    )
    plt.colorbar(sc, label="absolute token value")
    plt.title("ChatTime numeric token embeddings PCA colored by |value|")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(out / "pca_colored_by_abs_value.png", dpi=220)
    plt.close()

    # 3. sign color
    plt.figure(figsize=(8, 6))

    neg = token_df[token_df["sign"] == "negative"]
    pos = token_df[token_df["sign"] == "positive"]
    zero = token_df[token_df["sign"] == "zero"]

    plt.scatter(neg["pc1"], neg["pc2"], s=5, alpha=0.7, label="negative")
    plt.scatter(pos["pc1"], pos["pc2"], s=5, alpha=0.7, label="positive")
    if len(zero) > 0:
        plt.scatter(zero["pc1"], zero["pc2"], s=20, alpha=0.9, label="zero")

    plt.title("ChatTime numeric token embeddings PCA colored by sign")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "pca_colored_by_sign.png", dpi=220)
    plt.close()

    # 4. sign reversed pairs
    if args.draw_sign_pairs:
        coord = {
            round(row["value"], 4): (row["pc1"], row["pc2"], row["token"])
            for _, row in token_df.iterrows()
        }

        pair_values = []
        for v in sorted(coord.keys()):
            if v > 0 and round(-v, 4) in coord:
                pair_values.append(v)

        if len(pair_values) > args.max_pair_lines:
            idx = np.linspace(0, len(pair_values) - 1, args.max_pair_lines).astype(int)
            pair_values = [pair_values[i] for i in idx]

        plt.figure(figsize=(8, 6))
        plt.scatter(token_df["pc1"], token_df["pc2"], s=3, alpha=0.25)

        lengths = []

        for v in pair_values:
            x1, y1, _ = coord[v]
            x2, y2, _ = coord[round(-v, 4)]
            plt.plot([x1, x2], [y1, y2], linewidth=0.4, alpha=0.35)
            lengths.append(np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))

        plt.title("PCA sign-reversal pairs: +a connected to -a")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.tight_layout()
        plt.savefig(out / "pca_sign_reversal_pairs.png", dpi=220)
        plt.close()

        pair_summary = pd.DataFrame({
            "pair_value_abs": pair_values,
            "pca_pair_distance": lengths,
        })
        pair_summary.to_csv(out / "pca_sign_pair_distances.csv", index=False)

        print("sign pair median PCA distance:", pair_summary["pca_pair_distance"].median())

    print("Saved to:", out)


if __name__ == "__main__":
    main()