"""
numeric_mamba.py

train_base_mamba.py のモデル部分を、ECG 学習向けに以下の点だけ直したもの。

  * ContiguousNumericEmbedding の **パラメータ構成は元コードと完全に同一**。
    numeric_emb.pt をそのまま load_state_dict できる。
  * forward が正式な出力オブジェクトを返す (元コードは動的生成の匿名クラス)。
  * lead_ids を受け取り、任意で誘導位置埋め込みを加算できる。
  * lm_head=None のまま save_pretrained すると共有テンソルで詰まるので、
    保存/復元のヘルパを用意した。
"""

import os
import sys
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import MambaForCausalLM
from transformers.models.mamba.modeling_mamba import MambaCausalLMOutput


# ---------------------------------------------------------------------------
# 元コードと同一の埋め込み (state_dict 互換)
# ---------------------------------------------------------------------------

class ContiguousNumericEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, num_start, numeric_values,
                 feature, residual_scale=0.15, padding_idx=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_start = num_start
        self.n_num = len(numeric_values)
        self.residual_scale = residual_scale

        vals = torch.as_tensor(numeric_values, dtype=torch.float)
        assert torch.all(vals[1:] >= vals[:-1])
        self.register_buffer("values", vals)

        self.base = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.feature = feature
        self.proj = nn.Linear(feature.out_dim, d_model, bias=False)
        self.norm = nn.RMSNorm(d_model) if hasattr(nn, "RMSNorm") else nn.LayerNorm(d_model)
        self.residual = nn.Parameter(torch.randn(self.n_num, d_model) * 0.02)
        self._cached: Optional[torch.Tensor] = None

    def numeric_table(self) -> torch.Tensor:
        if self._cached is None:
            smooth = self.norm(self.proj(self.feature(self.values)))
            self._cached = smooth + self.residual_scale * self.residual
        return self._cached

    def clear_cache(self) -> None:
        self._cached = None

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hi = self.num_start + self.n_num
        is_num = (input_ids >= self.num_start) & (input_ids < hi)
        text_ids = torch.where(is_num, torch.zeros_like(input_ids), input_ids)
        out = self.base(text_ids)
        if is_num.any():
            num_ids = (input_ids - self.num_start).clamp(0, self.n_num - 1)
            num_emb = F.embedding(num_ids, self.numeric_table())
            out = torch.where(is_num.unsqueeze(-1), num_emb.to(out.dtype), out)
        return out

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        w = self.base.weight
        hi = self.num_start + self.n_num
        parts = [hidden @ w[: self.num_start].to(hidden.dtype).t(),
                 hidden @ self.numeric_table().to(hidden.dtype).t()]
        if hi < self.vocab_size:
            parts.append(hidden @ w[hi:].to(hidden.dtype).t())
        return torch.cat(parts, dim=-1)


def build_feature(code_path, low, high, n_bins, min_period_in_bins,
                  num_freq, n_rbf, rbf_sigma) -> nn.Module:
    if code_path and code_path not in sys.path:
        sys.path.append(code_path)
    from numeric_embedding import ConcatFeature, FourierFeature, RBFFeature
    bin_width = (high - low) / n_bins
    return ConcatFeature(
        FourierFeature(num_freq=num_freq,
                       min_period=min_period_in_bins * bin_width,
                       max_period=4.0 * (high - low)),
        RBFFeature(centers=torch.linspace(low, high, n_rbf), sigma=rbf_sigma),
    )


# ---------------------------------------------------------------------------
# 誘導位置埋め込み (time_major レイアウト用、任意)
# ---------------------------------------------------------------------------

class LeadAwareEmbedding(nn.Module):
    """数値埋め込みに「この点はどの誘導か」を加算する。

    time_major では隣接トークンが別誘導になるので、誘導 ID を明示すると
    モデルが位相 (t mod L) を推定する負担が減る。ゼロ初期化なので
    stage-0 チェックポイントの挙動を最初は一切変えない。
    """

    def __init__(self, inner: ContiguousNumericEmbedding, n_leads: int):
        super().__init__()
        self.inner = inner
        self.n_leads = n_leads
        self.lead = nn.Embedding(n_leads + 1, inner.d_model)   # 最後 = テキスト
        nn.init.zeros_(self.lead.weight)
        self.lead_ids: Optional[torch.Tensor] = None

    def forward(self, input_ids):
        out = self.inner(input_ids)
        li = self.lead_ids
        if li is not None:
            idx = torch.where(li < 0, torch.full_like(li, self.n_leads),
                              li.clamp(0, self.n_leads - 1))
            out = out + self.lead(idx).to(out.dtype)
        return out


# ---------------------------------------------------------------------------
# モデル
# ---------------------------------------------------------------------------

class NumericMambaForCausalLM(MambaForCausalLM):
    def attach_numeric_embedding(self, emb: ContiguousNumericEmbedding,
                                 lead_wrapper: Optional[LeadAwareEmbedding] = None):
        self.numeric_emb = emb
        self.lead_wrapper = lead_wrapper
        self.backbone.embeddings = lead_wrapper if lead_wrapper is not None else emb
        self.lm_head = None

    def forward(self, input_ids=None, inputs_embeds=None, cache_params=None,
                labels=None, use_cache=None, lead_ids=None, **kwargs):
        self.numeric_emb.clear_cache()
        if getattr(self, "lead_wrapper", None) is not None:
            self.lead_wrapper.lead_ids = lead_ids
        out = self.backbone(input_ids=input_ids, inputs_embeds=inputs_embeds,
                            cache_params=cache_params, use_cache=use_cache)
        logits = self.numeric_emb.logits(out[0])
        return MambaCausalLMOutput(loss=None, logits=logits,
                                   cache_params=out.cache_params
                                   if hasattr(out, "cache_params") else None)

    def get_input_embeddings(self):
        return self.backbone.embeddings

    def set_input_embeddings(self, new):
        self.backbone.embeddings = new

    def tie_weights(self):
        return


# ---------------------------------------------------------------------------
# 保存 / 復元
# ---------------------------------------------------------------------------

def save_all(model, tokenizer, path):
    os.makedirs(path, exist_ok=True)
    # safetensors は共有テンソル (numeric_emb と backbone.embeddings が同一) を嫌う
    model.save_pretrained(path, safe_serialization=False)
    tokenizer.save_pretrained(path)
    torch.save(model.numeric_emb.state_dict(), os.path.join(path, "numeric_emb.pt"))
    if getattr(model, "lead_wrapper", None) is not None:
        torch.save(model.lead_wrapper.lead.state_dict(),
                   os.path.join(path, "lead_emb.pt"))


def load_numeric_mamba(model_path, code_path, centers, num_start, vocab_size, *,
                       residual_scale=0.15, min_period_in_bins=16.0, num_freq=48,
                       n_rbf=128, rbf_sigma=0.03, low=-1.0, high=1.0,
                       n_leads=0, dtype=torch.bfloat16,
                       numeric_emb_path=None, lead_emb_path=None):
    """stage-0 / stage-1 の出力ディレクトリからモデルを復元する。

    backbone は from_pretrained で読み、数値埋め込みは numeric_emb.pt から読む
    (save_pretrained のキー名がカスタム埋め込みで崩れるため)。
    """
    model = NumericMambaForCausalLM.from_pretrained(model_path, torch_dtype=dtype)
    d_model = model.config.hidden_size

    feature = build_feature(code_path, low, high, len(centers),
                            min_period_in_bins, num_freq, n_rbf, rbf_sigma)
    emb = ContiguousNumericEmbedding(vocab_size=vocab_size, d_model=d_model,
                                     num_start=num_start, numeric_values=centers,
                                     feature=feature, residual_scale=residual_scale)

    path = numeric_emb_path or os.path.join(model_path, "numeric_emb.pt")
    if os.path.exists(path):
        missing, unexpected = emb.load_state_dict(
            torch.load(path, map_location="cpu"), strict=False)
        if missing or unexpected:
            print(f"[warn] numeric_emb: missing={missing} unexpected={unexpected}")
        print(f"loaded numeric embedding from {path}")
    else:
        raise FileNotFoundError(
            f"{path} が見つかりません。stage-0 の出力ディレクトリを指定してください。")
    emb = emb.to(dtype)

    wrapper = None
    if n_leads > 0:
        wrapper = LeadAwareEmbedding(emb, n_leads).to(dtype)
        lp = lead_emb_path or os.path.join(model_path, "lead_emb.pt")
        if os.path.exists(lp):
            wrapper.lead.load_state_dict(torch.load(lp, map_location="cpu"))
            print(f"loaded lead embedding from {lp}")

    model.attach_numeric_embedding(emb, wrapper)
    model.config.vocab_size = vocab_size
    return model
