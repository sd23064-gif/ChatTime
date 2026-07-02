#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_added_tokens.py

Mamba + PEFT/LoRA で学習したモデルについて、追加トークンと embedding を調べるためのスクリプトです。

主な出力:
  - added_tokens.csv
  - added_token_embedding_stats.csv
  - added_token_embedding_diff.csv
  - nearest_added_tokens.csv
  - added_token_embedding_norm_hist.png
  - added_token_embedding_norm_by_id.png
  - added_token_embedding_diff_by_id.png

例:
python analyze_added_tokens.py \
  --base_model_path state-spaces/mamba-370m-hf \
  --adapter_path /workspace/outputs/mamba-finetune \
  --output_dir outputs/token_analysis \
  --device cpu
"""

import argparse
import os
import math
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--base_model_path", type=str, required=True,
                        help="例: state-spaces/mamba-370m-hf")
    parser.add_argument("--adapter_path", type=str, required=True,
                        help="学習済み LoRA adapter または tokenizer 保存先")
    parser.add_argument("--output_dir", type=str, default="outputs/token_analysis")

    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                        help="解析用デバイス。メモリ節約のため cpu 推奨。")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--seed", type=int, default=3407,
                        help="base model resize 時の乱数固定用。diff は『fresh resized base』との差分です。")

    parser.add_argument("--nearest_top_k", type=int, default=10,
                        help="各追加トークンについて近傍何件を保存するか。")
    parser.add_argument("--nearest_limit", type=int, default=200,
                        help="近傍検索を行う追加トークン数。多すぎると重いので制限します。-1 で全件。")
    parser.add_argument("--nearest_scope", type=str, default="added", choices=["added", "all"],
                        help="近傍検索対象。added は追加トークン内のみ、all は全語彙。")

    parser.add_argument("--save_full_embeddings", action="store_true",
                        help="追加トークン embedding を .npy で保存します。ファイルサイズに注意。")

    return parser.parse_args()


def get_torch_dtype(dtype_name: str):
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float32":
        return torch.float32
    raise ValueError(dtype_name)


def safe_token_str(token):
    """CSVで見やすいように不可視文字を少しだけ可視化。"""
    return str(token).replace("\n", "\\n").replace("\t", "\\t")


def load_tokenizers(base_model_path, adapter_path):
    base_tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True,
    )

    trained_tokenizer = AutoTokenizer.from_pretrained(
        adapter_path,
        trust_remote_code=True,
    )

    if trained_tokenizer.pad_token is None:
        trained_tokenizer.pad_token = trained_tokenizer.eos_token

    return base_tokenizer, trained_tokenizer


def get_added_tokens(base_tokenizer, trained_tokenizer):
    base_vocab = base_tokenizer.get_vocab()
    trained_vocab = trained_tokenizer.get_vocab()

    added = []
    for tok, tok_id in trained_vocab.items():
        if tok not in base_vocab:
            added.append((tok, tok_id))

    added = sorted(added, key=lambda x: x[1])
    return added


def get_embedding_weight(model):
    """PeftModel / 通常Model の input embedding weight を取得。"""
    try:
        emb = model.get_input_embeddings().weight
        return emb
    except Exception:
        pass

    # fallback
    if hasattr(model, "base_model"):
        try:
            emb = model.base_model.model.get_input_embeddings().weight
            return emb
        except Exception:
            pass

    raise RuntimeError("input embeddings を取得できませんでした。model.get_input_embeddings() を確認してください。")


def load_base_embedding(base_model_path, tokenizer_len, dtype, device, seed):
    """
    base model を読み、trained tokenizer の語彙サイズに resize した embedding を返す。

    注意:
      追加トークンの base embedding は fresh initialization です。
      学習開始時の初期値と完全一致する保証はありません。
    """
    torch.manual_seed(seed)

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=dtype,
        device_map=None,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.resize_token_embeddings(tokenizer_len)
    model.eval()

    emb = get_embedding_weight(model).detach().float().cpu().clone()

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return emb


def load_trained_embedding(base_model_path, adapter_path, tokenizer_len, dtype, device):
    """
    adapter_config.json がある場合:
      base model + LoRA adapter として読み込む。
    ない場合:
      adapter_path を full model path とみなして読み込む。
    """
    adapter_config = Path(adapter_path) / "adapter_config.json"

    if adapter_config.exists():
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=dtype,
            device_map=None,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        model.to(device)
        model.resize_token_embeddings(tokenizer_len)

        model = PeftModel.from_pretrained(
            model,
            adapter_path,
            is_trainable=False,
        )
        model.to(device)
    else:
        print(f"[WARN] adapter_config.json が見つかりません: {adapter_config}")
        print("[WARN] adapter_path を full model path として読み込みます。")
        model = AutoModelForCausalLM.from_pretrained(
            adapter_path,
            torch_dtype=dtype,
            device_map=None,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        model.to(device)
        model.resize_token_embeddings(tokenizer_len)

    model.eval()
    emb = get_embedding_weight(model).detach().float().cpu().clone()

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return emb


def save_added_tokens_csv(added_tokens, output_dir):
    rows = []
    for tok, tok_id in added_tokens:
        rows.append({
            "token": tok,
            "token_repr": safe_token_str(tok),
            "token_id": tok_id,
        })

    df = pd.DataFrame(rows).sort_values("token_id")
    path = Path(output_dir) / "added_tokens.csv"
    df.to_csv(path, index=False)
    print(f"Saved: {path}")
    return df


def save_embedding_stats(added_tokens, trained_emb, output_dir):
    rows = []
    for tok, tok_id in added_tokens:
        vec = trained_emb[tok_id]
        rows.append({
            "token": tok,
            "token_repr": safe_token_str(tok),
            "token_id": tok_id,
            "embedding_norm": vec.norm().item(),
            "embedding_mean": vec.mean().item(),
            "embedding_std": vec.std().item(),
            "embedding_min": vec.min().item(),
            "embedding_max": vec.max().item(),
        })

    df = pd.DataFrame(rows).sort_values("token_id")
    path = Path(output_dir) / "added_token_embedding_stats.csv"
    df.to_csv(path, index=False)
    print(f"Saved: {path}")
    return df


def save_embedding_diff(added_tokens, base_emb, trained_emb, output_dir):
    rows = []
    for tok, tok_id in added_tokens:
        base_vec = base_emb[tok_id]
        trained_vec = trained_emb[tok_id]
        diff = trained_vec - base_vec

        cos = F.cosine_similarity(
            base_vec.unsqueeze(0),
            trained_vec.unsqueeze(0),
            dim=-1,
        ).item()

        rows.append({
            "token": tok,
            "token_repr": safe_token_str(tok),
            "token_id": tok_id,
            "base_norm_fresh_resized": base_vec.norm().item(),
            "trained_norm": trained_vec.norm().item(),
            "diff_norm_vs_fresh_resized_base": diff.norm().item(),
            "cosine_similarity_vs_fresh_resized_base": cos,
            "diff_mean": diff.mean().item(),
            "diff_std": diff.std().item(),
        })

    df = pd.DataFrame(rows).sort_values("diff_norm_vs_fresh_resized_base", ascending=False)
    path = Path(output_dir) / "added_token_embedding_diff.csv"
    df.to_csv(path, index=False)
    print(f"Saved: {path}")
    return df


def save_nearest_tokens(added_tokens, trained_emb, tokenizer, output_dir, top_k=10, limit=200, scope="added"):
    if len(added_tokens) == 0:
        print("[WARN] 追加トークンがないため nearest search をスキップします。")
        return pd.DataFrame()

    emb_norm = F.normalize(trained_emb, dim=-1)

    added_token_ids = [tok_id for _, tok_id in added_tokens]

    if scope == "added":
        candidate_ids = torch.tensor(added_token_ids, dtype=torch.long)
    else:
        candidate_ids = torch.arange(trained_emb.shape[0], dtype=torch.long)

    if limit is not None and limit >= 0:
        query_tokens = added_tokens[:limit]
    else:
        query_tokens = added_tokens

    rows = []
    candidate_matrix = emb_norm[candidate_ids]

    for query_tok, query_id in query_tokens:
        q = emb_norm[query_id]
        sims = candidate_matrix @ q

        k = min(top_k + 1, sims.numel())
        values, indices = torch.topk(sims, k=k)

        rank = 0
        for score, local_idx in zip(values.tolist(), indices.tolist()):
            neighbor_id = int(candidate_ids[local_idx].item())
            if neighbor_id == query_id:
                continue
            rank += 1
            neighbor_tok = tokenizer.convert_ids_to_tokens(neighbor_id)
            rows.append({
                "query_token": query_tok,
                "query_token_repr": safe_token_str(query_tok),
                "query_token_id": query_id,
                "rank": rank,
                "neighbor_token": neighbor_tok,
                "neighbor_token_repr": safe_token_str(neighbor_tok),
                "neighbor_token_id": neighbor_id,
                "cosine_similarity": float(score),
                "scope": scope,
            })
            if rank >= top_k:
                break

    df = pd.DataFrame(rows)
    path = Path(output_dir) / "nearest_added_tokens.csv"
    df.to_csv(path, index=False)
    print(f"Saved: {path}")
    return df


def save_plots(stats_df, diff_df, output_dir):
    output_dir = Path(output_dir)

    if len(stats_df) > 0:
        plt.figure(figsize=(8, 5))
        plt.hist(stats_df["embedding_norm"], bins=50)
        plt.xlabel("Embedding norm")
        plt.ylabel("Count")
        plt.title("Distribution of added token embedding norms")
        plt.tight_layout()
        path = output_dir / "added_token_embedding_norm_hist.png"
        plt.savefig(path, dpi=200)
        plt.close()
        print(f"Saved: {path}")

        plt.figure(figsize=(10, 5))
        stats_sorted = stats_df.sort_values("token_id")
        plt.plot(stats_sorted["token_id"], stats_sorted["embedding_norm"])
        plt.xlabel("Token ID")
        plt.ylabel("Embedding norm")
        plt.title("Embedding norm by added token ID")
        plt.tight_layout()
        path = output_dir / "added_token_embedding_norm_by_id.png"
        plt.savefig(path, dpi=200)
        plt.close()
        print(f"Saved: {path}")

    if len(diff_df) > 0:
        diff_sorted = diff_df.sort_values("token_id")
        plt.figure(figsize=(10, 5))
        plt.plot(diff_sorted["token_id"], diff_sorted["diff_norm_vs_fresh_resized_base"])
        plt.xlabel("Token ID")
        plt.ylabel("Diff norm vs fresh resized base")
        plt.title("Added token embedding difference by token ID")
        plt.tight_layout()
        path = output_dir / "added_token_embedding_diff_by_id.png"
        plt.savefig(path, dpi=200)
        plt.close()
        print(f"Saved: {path}")

        plt.figure(figsize=(8, 5))
        plt.hist(diff_df["diff_norm_vs_fresh_resized_base"], bins=50)
        plt.xlabel("Diff norm vs fresh resized base")
        plt.ylabel("Count")
        plt.title("Distribution of added token embedding differences")
        plt.tight_layout()
        path = output_dir / "added_token_embedding_diff_hist.png"
        plt.savefig(path, dpi=200)
        plt.close()
        print(f"Saved: {path}")


def save_summary(args, base_tokenizer, trained_tokenizer, added_tokens, trained_emb, output_dir):
    summary = {
        "base_model_path": args.base_model_path,
        "adapter_path": args.adapter_path,
        "base_vocab_size": len(base_tokenizer.get_vocab()),
        "trained_vocab_size": len(trained_tokenizer.get_vocab()),
        "added_token_count": len(added_tokens),
        "embedding_shape": list(trained_emb.shape),
        "note": (
            "embedding diff は、base model を trained tokenizer の語彙サイズに resize した fresh initialization との差分です。"
            "学習開始時の追加トークン初期値と完全一致する保証はありません。"
        ),
    }
    path = Path(output_dir) / "summary.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Saved: {path}")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA が使えないため CPU に切り替えます。")
        args.device = "cpu"

    dtype = get_torch_dtype(args.dtype)

    print("=" * 80)
    print("Analyze Added Tokens")
    print("=" * 80)
    print("base_model_path:", args.base_model_path)
    print("adapter_path   :", args.adapter_path)
    print("output_dir     :", str(output_dir))
    print("device         :", args.device)
    print("dtype          :", args.dtype)

    print("\n[1/7] Loading tokenizers...")
    base_tokenizer, trained_tokenizer = load_tokenizers(args.base_model_path, args.adapter_path)

    print("Base vocab size   :", len(base_tokenizer.get_vocab()))
    print("Trained vocab size:", len(trained_tokenizer.get_vocab()))

    print("\n[2/7] Detecting added tokens...")
    added_tokens = get_added_tokens(base_tokenizer, trained_tokenizer)
    print("Added token count:", len(added_tokens))

    if len(added_tokens) > 0:
        print("First 20 added tokens:")
        for tok, tok_id in added_tokens[:20]:
            print(f"  {tok_id}: {safe_token_str(tok)}")
    else:
        print("[WARN] base tokenizer と trained tokenizer の差分として追加トークンが見つかりませんでした。")

    added_df = save_added_tokens_csv(added_tokens, output_dir)

    print("\n[3/7] Loading fresh resized base embedding...")
    base_emb = load_base_embedding(
        args.base_model_path,
        tokenizer_len=len(trained_tokenizer),
        dtype=dtype,
        device=args.device,
        seed=args.seed,
    )
    print("Base embedding shape:", tuple(base_emb.shape))

    print("\n[4/7] Loading trained embedding...")
    trained_emb = load_trained_embedding(
        args.base_model_path,
        args.adapter_path,
        tokenizer_len=len(trained_tokenizer),
        dtype=dtype,
        device=args.device,
    )
    print("Trained embedding shape:", tuple(trained_emb.shape))

    if base_emb.shape != trained_emb.shape:
        raise RuntimeError(f"Embedding shape mismatch: base={base_emb.shape}, trained={trained_emb.shape}")

    print("\n[5/7] Saving embedding statistics...")
    stats_df = save_embedding_stats(added_tokens, trained_emb, output_dir)
    diff_df = save_embedding_diff(added_tokens, base_emb, trained_emb, output_dir)

    if args.save_full_embeddings and len(added_tokens) > 0:
        ids = [tok_id for _, tok_id in added_tokens]
        arr = trained_emb[ids].numpy()
        path = output_dir / "added_token_embeddings.npy"
        np.save(path, arr)
        print(f"Saved: {path}")

    print("\n[6/7] Nearest token search...")
    nearest_df = save_nearest_tokens(
        added_tokens=added_tokens,
        trained_emb=trained_emb,
        tokenizer=trained_tokenizer,
        output_dir=output_dir,
        top_k=args.nearest_top_k,
        limit=args.nearest_limit,
        scope=args.nearest_scope,
    )

    print("\n[7/7] Saving plots and summary...")
    save_plots(stats_df, diff_df, output_dir)
    save_summary(args, base_tokenizer, trained_tokenizer, added_tokens, trained_emb, output_dir)

    print("\nDone.")
    print("主な出力:")
    print(" -", output_dir / "added_tokens.csv")
    print(" -", output_dir / "added_token_embedding_stats.csv")
    print(" -", output_dir / "added_token_embedding_diff.csv")
    print(" -", output_dir / "nearest_added_tokens.csv")

    if len(diff_df) > 0:
        print("\nDiff norm 上位 10 件:")
        cols = ["token_repr", "token_id", "trained_norm", "diff_norm_vs_fresh_resized_base", "cosine_similarity_vs_fresh_resized_base"]
        print(diff_df[cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
