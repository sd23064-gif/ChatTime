"""
ecg_data.py

前処理済みビン列 -> 学習用トークン列。

数値トークンは `bin_index + num_start` で直接 ID を作る。
tokenizer.add_tokens 済みの文字列を通すと Trie マッチで死ぬほど遅くなるため
(元コードの NOTE の通り)、テキスト部分だけ tokenizer を使う。

系列レイアウト:
  time_major : t0 の全誘導 -> t1 の全誘導 -> ...  (既定)
      同時刻の誘導が隣接するので、SSM の固定サイズ状態でも誘導間比較が局所化する。
      連続誘導の ST 変化のような診断上重要な特徴に効く。
  lead_major : 誘導 I 全体 -> 誘導 II 全体 -> ...
      素直だが、I と V6 を比べるのに数千トークン分の状態保持が必要になる。
"""

import json
import math
import os
from typing import Dict, List, Optional

import numpy as np
import torch

SEX_NAME = {0: "male", 1: "female"}


class TextCache:
    """同じ文字列を何度もトークナイズしないための薄いキャッシュ。"""

    def __init__(self, tokenizer):
        self.tok = tokenizer
        self._c: Dict[str, np.ndarray] = {}

    def __call__(self, s: str) -> np.ndarray:
        v = self._c.get(s)
        if v is None:
            v = np.asarray(self.tok(s, add_special_tokens=False)["input_ids"],
                           dtype=np.int64)
            self._c[s] = v
        return v


# ---------------------------------------------------------------------------
# プロンプト文字列
# ---------------------------------------------------------------------------

def header_text(rec: dict, cfg: dict, seconds: float, order: str) -> str:
    age = rec.get("age")
    age_s = "unknown" if age is None else str(age)
    sex_s = SEX_NAME.get(rec.get("sex"), "unknown")
    h, w = rec.get("height"), rec.get("weight")
    h_s = "unknown" if h is None else f"{int(h)}cm"
    w_s = "unknown" if w is None else f"{int(w)}kg"
    return (
        "###ECG###\n"
        f"fs={cfg['fs']}Hz duration={seconds:g}s scale={cfg['mv_scale']:g}mV "
        f"order={order} leads={','.join(cfg['leads'])}\n"
        f"###Patient### age={age_s} sex={sex_s} height={h_s} weight={w_s}\n"
        "###Signal###\n"
    )


def diag_pieces(classes: List[str]):
    """診断ブロックの構成要素を (種別, 内容) で列挙する。

    学習時と評価時で完全に同じ区切りを使うことで、BPE の食い違いを防ぐ。
    """
    yield ("text", "\n###Diagnosis###\n")
    for c in classes:
        yield ("text", f"{c}:")
        yield ("answer", c)
        yield ("text", "\n")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PTBXLTokenDataset(torch.utils.data.Dataset):
    """stage=1 は信号のみ、stage=2 は信号 + 診断チェックリスト。"""

    def __init__(self, data_path: str, tokenizer, num_start: int, *,
                 stage: int = 1,
                 folds: Optional[List[int]] = None,
                 order: str = "time_major",
                 seconds: Optional[float] = None,
                 random_crop: bool = True,
                 lead_subset: Optional[List[str]] = None,
                 per_lead_examples: bool = False,
                 classes: Optional[List[str]] = None,
                 label_field: str = "superclass",
                 signal_loss_weight: float = 0.0,
                 seed: int = 0):
        self.data_path = data_path
        self.cfg = json.load(open(os.path.join(data_path, "config.json")))
        self.num_start = num_start
        self.stage = stage
        self.order = order
        self.random_crop = random_crop
        self.per_lead_examples = per_lead_examples
        self.classes = classes or ["NORM", "MI", "STTC", "CD", "HYP"]
        self.label_field = label_field
        self.signal_loss_weight = signal_loss_weight
        self.text = TextCache(tokenizer)
        self.seed = seed

        all_leads = self.cfg["leads"]
        use = lead_subset or all_leads
        self.lead_idx = [all_leads.index(l) for l in use]
        self.lead_names = use

        recs = [json.loads(l) for l in open(os.path.join(data_path, "records.jsonl"))]
        self.row_of = {}
        keep = []
        for i, r in enumerate(recs):
            if folds is None or r["strat_fold"] in folds:
                self.row_of[len(keep)] = i
                keep.append(r)
        self.recs = keep

        self.T = self.cfg["T"]
        self.fs = self.cfg["fs"]
        self.crop = self.T if seconds is None else int(round(seconds * self.fs))
        self.crop = min(self.crop, self.T)
        self.seconds = self.crop / self.fs

        self._sig = None
        self.epoch = 0          # クロップ位置をエポックごとに変えるため

        # yes / no は 1 トークンであってほしい
        self.yes_id = int(self.text(" yes")[0])
        self.no_id = int(self.text(" no")[0])
        if len(self.text(" yes")) != 1 or len(self.text(" no")) != 1:
            print("[warn] ' yes'/' no' が単一トークンではありません。"
                  "評価は先頭トークンで行われます。")

    # -- 内部 ---------------------------------------------------------------
    def _signals(self):
        if self._sig is None:                       # worker ごとに開く
            self._sig = np.load(os.path.join(self.data_path, "signals.npy"),
                                mmap_mode="r")
        return self._sig

    def __len__(self):
        n = len(self.recs)
        return n * len(self.lead_idx) if self.per_lead_examples else n

    def _rng(self, i):
        # インデックスとエポックから決定的に作る。再現性を保ったまま
        # エポックごとにクロップ位置が変わる。
        return np.random.default_rng(
            (self.seed * 1_000_003 + i) * 31 + self.epoch * 7_919)

    def _signal_block(self, sig, leads, num_start):
        """(L, crop) のビン列 -> (ids, lead_ids) の並び。"""
        L, C = sig.shape
        if self.order == "time_major":
            ids = sig.T.reshape(-1).astype(np.int64) + num_start      # t 優先
            lead = np.tile(np.arange(L, dtype=np.int64), C)
            return [(ids, lead)], None
        out = []
        for j in range(L):
            t = self.text(f"\n###Lead {leads[j]}###\n")
            out.append((t, np.full(len(t), -1, dtype=np.int64)))
            out.append((sig[j].astype(np.int64) + num_start,
                        np.full(C, j, dtype=np.int64)))
        return out, None

    # -- 本体 ---------------------------------------------------------------
    def __getitem__(self, i):
        if self.per_lead_examples:
            rec_i, lead_sel = divmod(i, len(self.lead_idx))
            leads_i = [self.lead_idx[lead_sel]]
            lead_names = [self.lead_names[lead_sel]]
        else:
            rec_i, leads_i, lead_names = i, self.lead_idx, self.lead_names

        rec = self.recs[rec_i]
        rng = self._rng(i)
        start = rng.integers(0, self.T - self.crop + 1) if (
            self.random_crop and self.T > self.crop) else 0

        sig = np.asarray(self._signals()[self.row_of[rec_i]][leads_i,
                                                            start:start + self.crop])

        cfg = dict(self.cfg)
        cfg["leads"] = lead_names

        pieces: List[tuple] = []
        weights: List[np.ndarray] = []

        def add(ids, lead, w):
            pieces.append((np.asarray(ids, dtype=np.int64),
                           np.asarray(lead, dtype=np.int64)))
            weights.append(np.full(len(ids), w, dtype=np.float32))

        head = self.text(header_text(rec, cfg, self.seconds, self.order))
        sig_w = 1.0 if self.stage == 1 else self.signal_loss_weight
        add(head, np.full(len(head), -1), sig_w)

        blocks, _ = self._signal_block(sig, lead_names, self.num_start)
        for ids, lead in blocks:
            add(ids, lead, sig_w)

        answer_pos = []
        if self.stage == 2:
            pos_set = set(rec[self.label_field])
            for kind, val in diag_pieces(self.classes):
                if kind == "text":
                    t = self.text(val)
                    add(t, np.full(len(t), -1), 1.0)
                else:
                    aid = self.yes_id if val in pos_set else self.no_id
                    answer_pos.append(sum(len(p[0]) for p in pieces))
                    add([aid], [-1], 1.0)

        input_ids = np.concatenate([p[0] for p in pieces])
        lead_ids = np.concatenate([p[1] for p in pieces])
        loss_w = np.concatenate(weights)

        return {
            "input_ids": input_ids,
            "lead_ids": lead_ids,
            "loss_weights": loss_w,
            "ecg_id": rec["ecg_id"],
            "answer_pos": np.asarray(answer_pos, dtype=np.int64),
        }


# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------

class EcgCollator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, feats):
        n = max(len(f["input_ids"]) for f in feats)
        B = len(feats)
        ids = np.full((B, n), self.pad_id, dtype=np.int64)
        lead = np.full((B, n), -1, dtype=np.int64)
        w = np.zeros((B, n), dtype=np.float32)
        for b, f in enumerate(feats):
            m = len(f["input_ids"])
            ids[b, :m] = f["input_ids"]
            lead[b, :m] = f["lead_ids"]
            w[b, :m] = f["loss_weights"]

        input_ids = torch.from_numpy(ids)
        loss_w = torch.from_numpy(w)
        labels = input_ids.clone()
        labels[loss_w == 0] = -100
        return {
            "input_ids": input_ids,
            "lead_ids": torch.from_numpy(lead),
            "labels": labels,
            "loss_weights": loss_w,
        }