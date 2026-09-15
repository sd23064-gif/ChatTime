"""
ecg_loss.py

重み付きソフトターゲット CE と平滑化正則化。

DDP のため、これらはモデルの forward の内側から呼ばれる。
lm_head を DDP ラップの外で使うと、その勾配が all-reduce されない
(勾配同期は DDP の forward 出力から辿れる範囲を前提にしている)。
"""

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _chunk_nll(h, W, b, lab, num_start, values, soft_sigma):
    """1 チャンク分の負の対数尤度。"""
    # accelerate は mixed precision のとき forward 出力を fp32 に戻すため、
    # 重みの dtype に明示的に合わせる。
    logits = F.linear(h.to(W.dtype), W, b)
    logp = F.log_softmax(logits.float(), dim=-1)
    nll = -logp.gather(1, lab.unsqueeze(1)).squeeze(1)

    if soft_sigma > 0 and values is not None:
        n_num = values.numel()
        is_num = (lab >= num_start) & (lab < num_start + n_num)
        if is_num.any():
            # 先に数値ブロックへ絞ってから行を選ぶ。logp[is_num] と書くと
            # 語彙全体 (60255 列) の複製が作られる。
            logp_num = logp[:, num_start:num_start + n_num][is_num]
            tv = values[lab[is_num] - num_start]
            d = (tv.unsqueeze(-1) - values) / soft_sigma
            w = torch.softmax(-0.5 * d.pow(2), dim=-1)
            nll = nll.clone()
            nll[is_num] = -(w * logp_num).sum(-1)
    return nll


def chunked_loss(hidden, lm_head, labels, weights, *, num_start, values,
                 soft_sigma=0.0, chunk=512, use_ckpt=True):
    """教師信号のある位置だけ lm_head を通し、チャンクに分けて損失を計算する。

    8,000 トークン x 60,255 語彙のロジットは bf16 で約 1 GB、log_softmax を
    fp32 で取るとその倍。チャンク単位で checkpoint を掛けることで、
    保持するのは隠れ状態だけで済む。
    """
    W, b = lm_head.weight, lm_head.bias
    total = hidden.new_zeros((), dtype=torch.float32)
    for i in range(0, hidden.size(0), chunk):
        h, lab, w = hidden[i:i + chunk], labels[i:i + chunk], weights[i:i + chunk]
        if use_ckpt and torch.is_grad_enabled():
            nll = checkpoint(_chunk_nll, h, W, b, lab, num_start, values,
                             soft_sigma, use_reentrant=False)
        else:
            nll = _chunk_nll(h, W, b, lab, num_start, values, soft_sigma)
        total = total + (nll * w).sum()
    return total / weights.sum().clamp_min(1e-6)


def smoothness(block: torch.Tensor) -> torch.Tensor:
    """隣接する数値トークンの埋め込みの 2 階差分ペナルティ (スケール不変)。

    stage-0 の埋め込みはスクラッチ学習なので値の近さと表現の近さが
    対応していない。ECG は等電位付近に値が集中して出現頻度が極端に偏るため、
    稀なビンを近傍から補間させる意味でこの正則化が効きやすい。
    """
    d2 = block[2:] - 2 * block[1:-1] + block[:-2]
    return d2.pow(2).sum(-1).mean() / block.pow(2).sum(-1).mean().clamp_min(1e-8)