"""
ecg_vocab.py

トークナイザから数値トークンの ID と値を復元する。

Discretizer を再構築するのではなく、実際に学習に使われたトークナイザから
値を読み取る。stage-0 の serializer 設定 (prec など) を引数で引き回さなくて済み、
モデルが実際に見た値と量子化が必ず一致する。
"""

import re
from typing import Optional, Tuple

import numpy as np

NUMERIC_TOKEN_RE = re.compile(
    r"###([+-]?(?:\d+(?:\.\d*)?|\.\d+)|Nan|NaN|nan)###"
)


def numeric_vocab(tokenizer) -> Tuple[int, np.ndarray, Optional[int]]:
    """(num_start, values, nan_id) を返す。

    values は昇順で、values[i] のトークン ID は num_start + i。
    """
    rows, nan_ids = [], []
    for token, tid in tokenizer.get_vocab().items():
        m = NUMERIC_TOKEN_RE.fullmatch(str(token))
        if m is None:
            continue
        if m.group(1).lower() == "nan":
            nan_ids.append(int(tid))
            continue
        rows.append((int(tid), float(m.group(1))))

    if not rows:
        raise ValueError(
            "数値トークンが見つかりません。--model_path に stage-0 で "
            "save_pretrained したトークナイザがあるか確認してください。")

    rows.sort(key=lambda x: x[1])
    ids = np.array([r[0] for r in rows], dtype=np.int64)
    vals = np.array([r[1] for r in rows], dtype=np.float64)

    # 値の昇順と ID の昇順が一致し、かつ ID が連続していることが
    # 「数値ブロックが語彙末尾に連続して並ぶ」前提の根拠になる。
    if not np.all(np.diff(vals) > 0):
        raise ValueError("数値トークンの値に重複があります (prec が粗すぎる可能性)")
    if not np.all(np.diff(ids) == 1):
        raise ValueError(
            "数値トークンの ID が連続していません。add_tokens の順序が "
            "値の昇順と一致していない可能性があります。")

    return int(ids[0]), vals, (int(nan_ids[0]) if nan_ids else None)


class Quantizer:
    """値 -> 最近傍の数値トークンのインデックス。"""

    def __init__(self, values: np.ndarray):
        self.values = np.asarray(values, dtype=np.float64)
        self.edges = (self.values[1:] + self.values[:-1]) / 2.0

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.clip(x, self.values[0], self.values[-1])
        return np.searchsorted(self.edges, x).astype(np.uint16)


def describe(num_start: int, values: np.ndarray, nan_id: Optional[int],
             vocab_size: int) -> None:
    print(f"numeric tokens : {len(values)}")
    print(f"num_start      : {num_start}")
    print(f"value range    : [{values[0]:.6f}, {values[-1]:.6f}]")
    print(f"bin width      : {np.diff(values).mean():.6f}")
    print(f"nan token id   : {nan_id}")
    print(f"vocab size     : {vocab_size}")