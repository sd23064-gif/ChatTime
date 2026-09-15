"""
eval_ptbxl.py

fold 10 (PTB-XL の標準テスト分割) で macro AUROC を出す。

各クラスの yes/no を自己回帰的に埋めていく (直前のクラスにはモデル自身の
予測を入れる) ので、教師強制によるラベル漏れがない。
"""

import os
import sys

# 同じディレクトリの ecg_*.py を最優先で解決する。
# sys.path 上の別の場所に同名ファイルがあると、そちらを掴んで
# "cannot import name ... (most likely due to a circular import)" になる。
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# tokenizers の並列化は fork 前に一度使うと警告を出す。
# DataLoader の worker で fork するので先に無効化しておく。
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import json
import os

import numpy as np
import torch
from transformers import AutoTokenizer

from ecg_data import PTBXLTokenDataset, diag_pieces
from ecg_model import load_model
from ecg_vocab import numeric_vocab


@torch.no_grad()
def predict(model, ds, classes, device, max_records=-1, verbose=200):
    yes_id, no_id = ds.yes_id, ds.no_id
    head = list(ds.text("\n###Diagnosis###\n"))
    probs, golds, ecg_ids = [], [], []

    n = len(ds) if max_records < 0 else min(len(ds), max_records)
    for i in range(n):
        item = ds[i]
        rec = ds.recs[i]
        ids = list(item["input_ids"])
        lead = list(item["lead_ids"])

        # Dataset は診断ブロック込みで返すので、プロンプト部分だけ切り出す
        cut = len(ids)
        for k in range(len(ids) - len(head), -1, -1):
            if ids[k:k + len(head)] == head:
                cut = k
                break
        ids, lead = ids[:cut], lead[:cut]

        p_row = {}
        for kind, val in diag_pieces(classes):
            if kind == "text":
                t = ds.text(val)
                ids += list(t)
                lead += [-1] * len(t)
                continue
            x = torch.tensor(ids, device=device).unsqueeze(0)
            l = torch.tensor(lead, device=device).unsqueeze(0)
            hidden = model(input_ids=x, lead_ids=l, return_hidden=True)
            hw = model.lm_head.weight
            logits = model.lm_head(hidden[:, -1].to(hw.dtype)).float()[0]
            pair = torch.softmax(torch.stack([logits[no_id], logits[yes_id]]), 0)
            p = pair[1].item()
            p_row[val] = p
            ids.append(yes_id if p >= 0.5 else no_id)
            lead.append(-1)

        probs.append([p_row[c] for c in classes])
        golds.append([1 if c in set(rec[ds.label_field]) else 0 for c in classes])
        ecg_ids.append(rec["ecg_id"])
        if verbose and (i + 1) % verbose == 0:
            print(f"  {i+1}/{n}")

    return np.array(probs), np.array(golds), ecg_ids


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True, help="stage-2 の出力")
    p.add_argument("--data_path", required=True)
    p.add_argument("--out_json", default="")
    p.add_argument("--test_folds", default="10")
    p.add_argument("--max_records", type=int, default=-1)
    p.add_argument("--load_dtype", default="bfloat16",
                   choices=["float32", "bfloat16"])
    args = p.parse_args()

    tc = json.load(open(os.path.join(args.model_path, "ecg_train_config.json")))

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    num_start, values, _ = numeric_vocab(tokenizer)

    data_cfg = json.load(open(os.path.join(args.data_path, "config.json")))
    lead_subset = [s for s in tc["leads"].split(",") if s] or None
    n_leads = len(lead_subset or data_cfg["leads"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.model_path, tokenizer,
                       dtype=getattr(torch, args.load_dtype),
                       n_leads=n_leads if tc["lead_pos_emb"] else 0)
    model = model.to(device).eval()
    model.config.use_cache = False

    classes = [c for c in tc["classes"].split(",") if c]
    ds = PTBXLTokenDataset(
        args.data_path, tokenizer, num_start, stage=2,
        folds=[int(x) for x in args.test_folds.split(",")],
        order=tc["order"], seconds=tc["seconds"], random_crop=False,
        lead_subset=lead_subset, per_lead_examples=False,
        classes=classes, label_field=tc["label_field"],
        signal_loss_weight=0.0, seed=0)

    probs, golds, ecg_ids = predict(model, ds, classes, device, args.max_records)

    from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
    aucs, aps = {}, {}
    for k, c in enumerate(classes):
        if golds[:, k].min() == golds[:, k].max():
            aucs[c] = aps[c] = float("nan")
            continue
        aucs[c] = roc_auc_score(golds[:, k], probs[:, k])
        aps[c] = average_precision_score(golds[:, k], probs[:, k])

    macro = float(np.nanmean(list(aucs.values())))
    f1 = f1_score(golds, (probs >= 0.5).astype(int), average="macro", zero_division=0)

    print("\n=== PTB-XL test ===")
    for c in classes:
        print(f"  {c:<8} AUROC={aucs[c]:.4f}  AP={aps[c]:.4f}")
    print(f"  macro AUROC  = {macro:.4f}")
    print(f"  macro F1@0.5 = {f1:.4f}")

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump({"auc": aucs, "ap": aps, "macro_auc": macro, "macro_f1": f1,
                       "ecg_ids": ecg_ids, "probs": probs.tolist(),
                       "golds": golds.tolist()}, f, indent=2)


if __name__ == "__main__":
    main()