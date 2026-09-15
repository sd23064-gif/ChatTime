import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM


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
    parser.add_argument("--reference_tokenizer_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs/token_analysis_chattime")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--nearest_limit", type=int, default=200)
    parser.add_argument("--nearest_top_k", type=int, default=10)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available. Use CPU.")
        args.device = "cpu"

    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]

    print("Loading tokenizer:", args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    print("Tokenizer vocab size:", len(tokenizer))
    print("Added tokens decoder size:", len(tokenizer.added_tokens_decoder))

    # 1. tokenizer.added_tokens_decoder 由来の追加トークン
    rows = []
    for token_id, token_obj in tokenizer.added_tokens_decoder.items():
        token = str(token_obj)
        rows.append({
            "token": token,
            "token_repr": token.replace("\n", "\\n").replace("\t", "\\t"),
            "token_id": int(token_id),
            "value": parse_value(token),
            "source": "added_tokens_decoder",
        })

    added_decoder_df = pd.DataFrame(rows).sort_values("token_id")
    added_decoder_df.to_csv(out / "chattime_added_tokens_decoder.csv", index=False)

    # 2. reference tokenizer がある場合は vocab 差分も見る
    if args.reference_tokenizer_path is not None:
        print("Loading reference tokenizer:", args.reference_tokenizer_path)
        ref_tokenizer = AutoTokenizer.from_pretrained(
            args.reference_tokenizer_path,
            trust_remote_code=True,
        )

        ref_vocab = ref_tokenizer.get_vocab()
        cur_vocab = tokenizer.get_vocab()

        diff_rows = []
        for tok, tok_id in cur_vocab.items():
            if tok not in ref_vocab:
                diff_rows.append({
                    "token": tok,
                    "token_repr": str(tok).replace("\n", "\\n").replace("\t", "\\t"),
                    "token_id": int(tok_id),
                    "value": parse_value(tok),
                    "source": "vocab_diff",
                })

        diff_df = pd.DataFrame(diff_rows).sort_values("token_id")
        diff_df.to_csv(out / "chattime_added_tokens_vs_reference.csv", index=False)
        target_tokens_df = diff_df
    else:
        target_tokens_df = added_decoder_df

    if len(target_tokens_df) == 0:
        print("No target tokens found.")
        return

    print("Target token count:", len(target_tokens_df))
    print(target_tokens_df.head(20))

    # 3. embedding 統計
    print("Loading model:", args.model_path)
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

    stat_rows = []
    token_ids = []

    for _, row in target_tokens_df.iterrows():
        token_id = int(row["token_id"])

        if token_id < 0 or token_id >= emb.shape[0]:
            continue

        vec = emb[token_id]
        token_ids.append(token_id)

        stat_rows.append({
            "token": row["token"],
            "token_repr": row["token_repr"],
            "token_id": token_id,
            "value": row["value"],
            "embedding_norm": vec.norm().item(),
            "embedding_mean": vec.mean().item(),
            "embedding_std": vec.std().item(),
            "embedding_min": vec.min().item(),
            "embedding_max": vec.max().item(),
        })

    stats_df = pd.DataFrame(stat_rows).sort_values("token_id")
    stats_df.to_csv(out / "chattime_token_embedding_stats.csv", index=False)

    # 4. 追加トークン同士の近傍
    if args.nearest_limit == -1:
        query_df = stats_df
    else:
        query_df = stats_df.head(args.nearest_limit)

    candidate_ids = stats_df["token_id"].astype(int).tolist()
    candidate_ids_tensor = torch.tensor(candidate_ids, dtype=torch.long)

    emb_norm = F.normalize(emb, dim=-1)
    candidate_matrix = emb_norm[candidate_ids_tensor]

    nearest_rows = []

    for _, row in query_df.iterrows():
        qid = int(row["token_id"])
        qvec = emb_norm[qid]

        sims = candidate_matrix @ qvec
        k = min(args.nearest_top_k + 1, sims.numel())

        values, indices = torch.topk(sims, k=k)

        rank = 0
        for score, local_idx in zip(values.tolist(), indices.tolist()):
            nid = int(candidate_ids_tensor[local_idx].item())
            if nid == qid:
                continue

            ntok = tokenizer.convert_ids_to_tokens(nid)
            rank += 1

            nearest_rows.append({
                "query_token": row["token"],
                "query_token_repr": row["token_repr"],
                "query_token_id": qid,
                "query_value": row["value"],
                "rank": rank,
                "neighbor_token": ntok,
                "neighbor_token_repr": str(ntok).replace("\n", "\\n").replace("\t", "\\t"),
                "neighbor_token_id": nid,
                "neighbor_value": parse_value(ntok),
                "cosine_similarity": float(score),
                "abs_value_diff": abs(row["value"] - parse_value(ntok))
                    if not np.isnan(row["value"]) and not np.isnan(parse_value(ntok))
                    else np.nan,
            })

            if rank >= args.nearest_top_k:
                break

    nearest_df = pd.DataFrame(nearest_rows)
    nearest_df.to_csv(out / "chattime_nearest_added_tokens.csv", index=False)

    # 5. summary
    summary = {
        "model_path": args.model_path,
        "reference_tokenizer_path": args.reference_tokenizer_path,
        "vocab_size": len(tokenizer),
        "added_tokens_decoder_count": len(tokenizer.added_tokens_decoder),
        "target_token_count": len(target_tokens_df),
        "embedding_shape": list(emb.shape),
    }

    pd.Series(summary).to_json(out / "chattime_token_summary.json", force_ascii=False, indent=2)

    print("Saved outputs to:", out)


if __name__ == "__main__":
    main()