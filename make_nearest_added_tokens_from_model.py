#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_nearest_added_tokens_from_model.py

学習済み Mamba/PEFT adapter から数値トークン embedding を取り出し、
各 query token の最近傍 token を cosine similarity で求めて CSV に保存する。

想定出力:
  chattime_nearest_added_tokens_after_numeric_reg.csv

実行例:
  python make_nearest_added_tokens_from_model.py \
    --base_model_path state-spaces/mamba-370m-hf \
    --adapter_path /workspace/outputs/model/mamba-370m-x-numeric-reg-w001 \
    --output_csv /workspace/outputs/analysis/chattime_nearest_added_tokens_after_numeric_reg.csv \
    --top_k 1 \
    --chunk_size 512
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


NUMERIC_TOKEN_RE = re.compile(
    r"###([+-]?(?:\d+(?:\.\d*)?|\.\d+)|Nan|NaN|nan)###"
)


def is_numeric_token(token: str):
    m = NUMERIC_TOKEN_RE.fullmatch(str(token))
    if m is None:
        return False
    return m.group(1).lower() != "nan"


def token_to_value(token: str):
    m = NUMERIC_TOKEN_RE.fullmatch(str(token))
    if m is None:
        return np.nan
    raw = m.group(1)
    if raw.lower() == "nan":
        return np.nan
    return float(raw)


def collect_numeric_tokens(tokenizer):
    rows = []
    vocab = tokenizer.get_vocab()
    for token, token_id in vocab.items():
        if is_numeric_token(token):
            rows.append({
                "token": token,
                "token_repr": token,
                "token_id": int(token_id),
                "value": token_to_value(token),
            })
    rows = sorted(rows, key=lambda r: r["value"])
    if len(rows) == 0:
        raise ValueError("No numeric tokens found. Load tokenizer from adapter_path where added tokens were saved.")
    return rows


def load_model_and_tokenizer(base_model_path, adapter_path, dtype):
    print("Loading tokenizer from adapter:", adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16 if dtype == "fp16" else torch.float32

    print("Loading base model:", base_model_path)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    # tokenizer に追加済み token があるので resize が必須
    model.resize_token_embeddings(len(tokenizer))

    print("Loading PEFT adapter:", adapter_path)
    model = PeftModel.from_pretrained(
        model,
        adapter_path,
        is_trainable=False,
    )
    model.eval()
    return model, tokenizer


def get_embedding_matrix(model, embedding_source="input"):
    if embedding_source == "input":
        emb_layer = model.get_input_embeddings()
    elif embedding_source == "output":
        emb_layer = model.get_output_embeddings()
        if emb_layer is None:
            raise ValueError("model.get_output_embeddings() returned None")
    else:
        raise ValueError("embedding_source must be input or output")

    return emb_layer.weight.detach()


def compute_nearest(rows, emb_weight, top_k=1, chunk_size=512):
    token_ids = torch.tensor([r["token_id"] for r in rows], dtype=torch.long, device=emb_weight.device)
    values = np.asarray([r["value"] for r in rows], dtype=np.float64)
    tokens = [r["token"] for r in rows]
    token_reprs = [r["token_repr"] for r in rows]

    emb = emb_weight[token_ids]
    emb = F.normalize(emb.float(), dim=-1)

    n = emb.shape[0]
    result_rows = []

    print(f"Computing nearest neighbors: n={n}, top_k={top_k}, chunk_size={chunk_size}")

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        query_emb = emb[start:end]
        sim = query_emb @ emb.T

        # 自分自身を除外
        row_idx = torch.arange(end - start, device=emb.device)
        col_idx = torch.arange(start, end, device=emb.device)
        sim[row_idx, col_idx] = -float("inf")

        vals, idxs = torch.topk(sim, k=top_k, dim=1)

        vals = vals.detach().cpu().numpy()
        idxs = idxs.detach().cpu().numpy()

        for i in range(end - start):
            q = start + i
            for rank in range(top_k):
                nb = int(idxs[i, rank])
                cosine = float(vals[i, rank])
                qv = float(values[q])
                nv = float(values[nb])
                result_rows.append({
                    "query_token": tokens[q],
                    "query_token_repr": token_reprs[q],
                    "query_token_id": int(token_ids[q].detach().cpu().item()),
                    "query_value": qv,
                    "rank": rank + 1,
                    "neighbor_token": tokens[nb],
                    "neighbor_token_repr": token_reprs[nb],
                    "neighbor_token_id": int(token_ids[nb].detach().cpu().item()),
                    "neighbor_value": nv,
                    "cosine_similarity": cosine,
                    "abs_value_diff": abs(qv - nv),
                    "sign_reversed": bool(qv * nv < 0),
                    "abs_magnitude_diff": abs(abs(qv) - abs(nv)),
                })

        print(f"processed {end}/{n}")

    return pd.DataFrame(result_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str, required=True)
    parser.add_argument("--adapter_path", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--chunk_size", type=int, default=512)
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--embedding_source", type=str, default="input", choices=["input", "output"])
    args = parser.parse_args()

    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer(args.base_model_path, args.adapter_path, args.dtype)
    rows = collect_numeric_tokens(tokenizer)

    print("numeric token count:", len(rows))
    print("value range:", rows[0]["value"], rows[-1]["value"])

    emb_weight = get_embedding_matrix(model, embedding_source=args.embedding_source)
    df = compute_nearest(rows, emb_weight, top_k=args.top_k, chunk_size=args.chunk_size)
    df.to_csv(out, index=False)

    summary = {
        "base_model_path": args.base_model_path,
        "adapter_path": args.adapter_path,
        "output_csv": str(out),
        "top_k": args.top_k,
        "embedding_source": args.embedding_source,
        "n_rows": int(len(df)),
        "n_query_tokens": int(len(rows)),
        "rank1_abs_value_diff_mean": float(df[df["rank"] == 1]["abs_value_diff"].mean()),
        "rank1_abs_value_diff_median": float(df[df["rank"] == 1]["abs_value_diff"].median()),
        "rank1_abs_value_diff_ge_0p1_ratio": float((df[df["rank"] == 1]["abs_value_diff"] >= 0.1).mean()),
        "rank1_sign_reversal_ratio": float(df[df["rank"] == 1]["sign_reversed"].mean()),
    }
    pd.Series(summary).to_json(out.with_suffix(".summary.json"), force_ascii=False, indent=2)

    print("Saved:", out)
    print(summary)


if __name__ == "__main__":
    main()
