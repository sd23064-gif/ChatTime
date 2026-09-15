"""
inspect_data.py

前処理済みデータと、そこから組み立てられるトークン列を目視確認する。

    python training/inspect_data.py \
        --data_path ./ptbxl_tok_100hz \
        --tokenizer_path ./out_stage0 \
        --index 0
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np

from ecg_vocab import numeric_vocab


def show_config(data_path):
    cfg = json.load(open(os.path.join(data_path, "config.json")))
    print("=" * 70)
    print("config.json")
    print("=" * 70)
    for k, v in cfg.items():
        print(f"  {k:16} = {v}")
    return cfg


def show_record(data_path, index):
    print("\n" + "=" * 70)
    print(f"records.jsonl [{index}]")
    print("=" * 70)
    with open(os.path.join(data_path, "records.jsonl")) as f:
        for i, line in enumerate(f):
            if i == index:
                rec = json.loads(line)
                break
    for k, v in rec.items():
        if k == "scp_codes":
            v = ", ".join(f"{a}={b:g}" for a, b in v.items())
        print(f"  {k:12} : {v}")
    return rec


def show_signal(data_path, cfg, values, num_start, index):
    print("\n" + "=" * 70)
    print("signals.npy")
    print("=" * 70)
    arr = np.load(os.path.join(data_path, "signals.npy"), mmap_mode="r")
    print(f"  shape = {arr.shape}  dtype = {arr.dtype}  "
          f"({arr.nbytes/1e9:.2f} GB)")
    print(f"  (記録数, 誘導数, サンプル数) = "
          f"({cfg['n_records']}, {len(cfg['leads'])}, {cfg['T']})")

    sig = np.asarray(arr[index])                    # (L, T)
    mv = values[sig] * cfg["mv_scale"]

    print(f"\n  記録 {index} のビン統計:")
    print(f"    {'誘導':<6} {'最小':>6} {'最大':>6} {'中央':>6} "
          f"{'振幅[mV]':>10} {'p2p[mV]':>9}")
    for j, name in enumerate(cfg["leads"]):
        b = sig[j]
        print(f"    {name:<6} {b.min():>6d} {b.max():>6d} {int(np.median(b)):>6d} "
              f"{np.abs(mv[j]).max():>10.3f} {mv[j].max()-mv[j].min():>9.3f}")

    print(f"\n  誘導 {cfg['leads'][0]} の先頭 12 サンプル:")
    print(f"    bin      : {sig[0, :12].tolist()}")
    print(f"    token_id : {(sig[0, :12].astype(int) + num_start).tolist()}")
    print(f"    mV       : {np.round(mv[0, :12], 4).tolist()}")

    lo, hi = values[0] * cfg["mv_scale"], values[-1] * cfg["mv_scale"]
    at_edge = np.mean((sig == 0) | (sig == len(values) - 1))
    print(f"\n  値域 [{lo:.2f}, {hi:.2f}] mV, 端に張り付いた割合 = {at_edge*100:.4f}%")
    print(f"  中央ビン {len(values)//2} = {values[len(values)//2]*cfg['mv_scale']:.4f} mV")
    return sig


def show_tokens(data_path, tokenizer_path, index, stage, order, seconds, classes):
    from transformers import AutoTokenizer

    from ecg_data import PTBXLTokenDataset

    tok = AutoTokenizer.from_pretrained(tokenizer_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    num_start, values, _ = numeric_vocab(tok)

    ds = PTBXLTokenDataset(data_path, tok, num_start, stage=stage, folds=None,
                           order=order, seconds=seconds, random_crop=False,
                           classes=classes, signal_loss_weight=0.1)
    item = ds[index]
    ids = item["input_ids"]
    lead = item["lead_ids"]
    w = item["loss_weights"]

    print("\n" + "=" * 70)
    print(f"トークン列 (stage {stage}, order={order})")
    print("=" * 70)
    n_num_tok = int(((ids >= num_start) & (ids < num_start + len(values))).sum())
    print(f"  全長 {len(ids)} = テキスト {len(ids)-n_num_tok} + 数値 {n_num_tok}")
    print(f"  損失の重み: 0 が {int((w==0).sum())}, "
          f"1 が {int((w==1).sum())}, その他 {int(((w!=0)&(w!=1)).sum())}")

    print("\n  --- 冒頭 (テキスト部分) ---")
    print("  " + tok.decode(ids[:60]).replace("\n", "\n  "))

    is_num = (ids >= num_start) & (ids < num_start + len(values))
    start = int(np.argmax(is_num))
    print(f"\n  --- 信号の先頭 (位置 {start} から) ---")
    for k in range(start, start + 16):
        v = values[ids[k] - num_start] if is_num[k] else None
        名 = "text" if lead[k] < 0 else f"lead {lead[k]}"
        vs = f"{v:+.4f}" if v is not None else tok.decode([ids[k]])
        print(f"    [{k:5d}] id={ids[k]:6d} {名:<8} {vs:>10}  w={w[k]:.2f}")
    print("    (time_major なら 1 サンプルごとに誘導が一巡する)")

    if stage == 2:
        print("\n  --- 末尾 (診断ブロック) ---")
        print("  " + tok.decode(ids[-45:]).replace("\n", "\n  "))
        sup = int((w[-45:] == 1).sum())
        print(f"\n  診断ブロックで損失がかかるトークン数: {sup}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", required=True)
    p.add_argument("--tokenizer_path", default=None,
                   help="省略時は config.json の tokenizer_path")
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--stage", type=int, default=2, choices=[1, 2])
    p.add_argument("--order", default="time_major",
                   choices=["time_major", "lead_major"])
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--classes", default="NORM,MI,STTC,CD,HYP")
    p.add_argument("--skip_tokens", action="store_true",
                   help="transformers を読まず、配列だけ見る")
    args = p.parse_args()

    cfg = show_config(args.data_path)
    tok_path = args.tokenizer_path or cfg.get("tokenizer_path")

    if args.skip_tokens:
        values = np.linspace(cfg["value_min"], cfg["value_max"], cfg["n_num"])
        num_start = 0
    else:
        from transformers import AutoTokenizer
        t = AutoTokenizer.from_pretrained(tok_path)
        num_start, values, _ = numeric_vocab(t)

    show_record(args.data_path, args.index)
    show_signal(args.data_path, cfg, values, num_start, args.index)

    if not args.skip_tokens:
        show_tokens(args.data_path, tok_path, args.index, args.stage,
                    args.order, args.seconds,
                    [c for c in args.classes.split(",") if c])


if __name__ == "__main__":
    main()
