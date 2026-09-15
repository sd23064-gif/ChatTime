"""
ecg_model.py

stage-0 (PEFT アダプタ) を密なモデルにマージして読み込み、
ECG 用に lead_ids を受け取れるようにしたもの。

PEFT まわりの注意点:

  1. アダプタを読む前に base を resize_token_embeddings する。
     modules_to_save に埋め込みと lm_head が入っているので、
     アダプタ側は拡張後の語彙サイズを持っている。

  2. lm_head は untie されている。
     Mamba は tie_word_embeddings=True が既定だが、PEFT は modules_to_save の
     各モジュールを個別に deepcopy するため共有が切れる。
     マージ後に config.tie_word_embeddings=False を立てておかないと、
     保存・再読込のときに lm_head が埋め込みに再結合されて学習結果が消える。

  3. merge_and_unload() は LoRA を base に畳み込み、modules_to_save は
     学習済みコピーで置き換える。以降は素の MambaForCausalLM として扱える。
"""

import json
import os
from typing import Optional

import torch
import torch.nn as nn
from transformers import MambaForCausalLM

from ecg_loss import chunked_loss, smoothness


class LeadAwareEmbedding(nn.Module):
    """トークン埋め込みに「この点はどの誘導か」を加算する。

    time_major レイアウトでは隣接トークンが別誘導になるので、誘導 ID を
    明示するとモデルが位相 (t mod L) を推定する負担が減る。
    ゼロ初期化なので、最初は stage-0 の挙動を一切変えない。
    """

    def __init__(self, inner: nn.Embedding, n_leads: int):
        super().__init__()
        self.inner = inner
        self.n_leads = n_leads
        self.lead = nn.Embedding(n_leads + 1, inner.embedding_dim)  # 末尾=テキスト
        nn.init.zeros_(self.lead.weight)
        self.lead_ids: Optional[torch.Tensor] = None

    @property
    def weight(self):                      # get_input_embeddings().weight 用
        return self.inner.weight

    @property
    def num_embeddings(self):
        return self.inner.num_embeddings

    def forward(self, input_ids):
        out = self.inner(input_ids)
        li = self.lead_ids
        if li is not None:
            idx = torch.where(li < 0, torch.full_like(li, self.n_leads),
                              li.clamp(0, self.n_leads - 1))
            out = out + self.lead(idx).to(out.dtype)
        return out


class EcgMambaForCausalLM(MambaForCausalLM):
    """lead_ids を受け取り、損失を forward の内側で計算する MambaForCausalLM。

    損失をここで計算する理由は 2 つ。

      1. DDP の勾配同期は forward 出力から辿れる範囲を前提にしている。
         lm_head をラップの外で使うと、その勾配が all-reduce されない。
      2. 8,000 トークン x 60,255 語彙のロジットは bf16 でも約 1 GB。
         教師信号のある位置だけに絞ってチャンク計算したい。
    """

    def configure_loss(self, *, num_start, n_num, values, soft_sigma=0.0,
                       lambda_smooth=0.0, smooth_target="both", loss_chunk=512):
        self.num_start = num_start
        self.n_num = n_num
        self.soft_sigma = soft_sigma
        self.lambda_smooth = lambda_smooth
        self.smooth_target = smooth_target
        self.loss_chunk = loss_chunk
        self.register_buffer("numeric_values",
                             torch.as_tensor(values, dtype=torch.float32),
                             persistent=False)

    def _loss(self, hidden, labels, weights):
        h = hidden[:, :-1].contiguous()
        lab = labels[:, 1:].contiguous()
        w = weights[:, 1:].contiguous()

        h = h.reshape(-1, h.size(-1))
        lab = lab.reshape(-1)
        w = w.reshape(-1).to(torch.float32)

        keep = (lab != -100) & (w > 0)
        h, lab, w = h[keep], lab[keep], w[keep]

        if lab.numel() == 0:
            loss = h.sum() * 0.0
        else:
            loss = chunked_loss(h, self.lm_head, lab, w,
                                num_start=self.num_start,
                                values=self.numeric_values,
                                soft_sigma=self.soft_sigma,
                                chunk=self.loss_chunk)

        if self.lambda_smooth > 0:
            lo, hi = self.num_start, self.num_start + self.n_num
            blocks = [self.get_input_embeddings().weight[lo:hi]]
            if self.smooth_target == "both":
                blocks.append(self.lm_head.weight[lo:hi])
            for blk in blocks:
                loss = loss + self.lambda_smooth * smoothness(blk.float())
        return loss

    def forward(self, input_ids=None, lead_ids=None, labels=None,
                loss_weights=None, return_hidden=False, **kwargs):
        emb = self.backbone.embeddings
        if isinstance(emb, LeadAwareEmbedding):
            emb.lead_ids = lead_ids

        if labels is not None:
            hidden = self.backbone(input_ids=input_ids, **kwargs)[0]
            return {"loss": self._loss(hidden, labels, loss_weights)}
        if return_hidden:
            return self.backbone(input_ids=input_ids, **kwargs)[0]
        return super().forward(input_ids=input_ids, **kwargs)


# ---------------------------------------------------------------------------
# 読み込み / 保存
# ---------------------------------------------------------------------------

def load_model(model_path: str, tokenizer, *, base_model_path: Optional[str] = None,
               dtype=torch.bfloat16, n_leads: int = 0,
               lead_emb_path: Optional[str] = None):
    adapter_cfg = os.path.join(model_path, "adapter_config.json")

    if os.path.exists(adapter_cfg):
        cfg = json.load(open(adapter_cfg))
        base = base_model_path or cfg.get("base_model_name_or_path")
        print(f"PEFT アダプタを検出。base = {base}")
        model = EcgMambaForCausalLM.from_pretrained(base, torch_dtype=dtype,
                                                    trust_remote_code=True)
        # ここで resize してからでないと modules_to_save の形状が合わない
        model.resize_token_embeddings(len(tokenizer))

        from peft import PeftModel
        model = PeftModel.from_pretrained(model, model_path, is_trainable=False)
        model = model.merge_and_unload()
        print("LoRA をマージしました")
    else:
        model = EcgMambaForCausalLM.from_pretrained(model_path, torch_dtype=dtype,
                                                    trust_remote_code=True)

    # lm_head の再結合を防ぐ (PEFT の modules_to_save で untie 済みのため)
    model.config.tie_word_embeddings = False
    model.config.vocab_size = len(tokenizer)
    if model.lm_head.weight.shape[0] != len(tokenizer):
        raise RuntimeError(
            f"lm_head の行数 {model.lm_head.weight.shape[0]} が "
            f"語彙サイズ {len(tokenizer)} と一致しません")

    if n_leads > 0:
        wrapper = LeadAwareEmbedding(model.backbone.embeddings, n_leads).to(dtype)
        lp = lead_emb_path or os.path.join(model_path, "lead_emb.pt")
        if os.path.exists(lp):
            wrapper.lead.load_state_dict(torch.load(lp, map_location="cpu"))
            print(f"loaded lead embedding from {lp}")
        model.backbone.embeddings = wrapper

    return model


def sanity_check(model, num_start: int, n_num: int) -> None:
    """数値行が実際に学習されているか (アダプタが載ったか) の目安。"""
    w = model.get_input_embeddings().weight.detach().float()
    num = w[num_start:num_start + n_num].norm(dim=-1)
    txt = w[:num_start].norm(dim=-1)
    print(f"embedding norm: text median={txt.median():.4f}  "
          f"numeric median={num.median():.4f}  numeric std={num.std():.4f}")
    if num.std() < 1e-6:
        print("[warn] 数値行のノルムがほぼ一定です。"
              "アダプタが載っていない可能性があります。")


def tie_status(model) -> None:
    emb = model.get_input_embeddings().weight
    print("lm_head tied to embeddings:",
          emb.data_ptr() == model.lm_head.weight.data_ptr())


def save_all(model, tokenizer, path: str) -> None:
    os.makedirs(path, exist_ok=True)
    wrapper = model.backbone.embeddings
    has_wrapper = isinstance(wrapper, LeadAwareEmbedding)
    if has_wrapper:                       # 標準の state_dict キーに戻して保存する
        model.backbone.embeddings = wrapper.inner

    model.config.tie_word_embeddings = False
    model.save_pretrained(path, safe_serialization=False)
    tokenizer.save_pretrained(path)

    if has_wrapper:
        torch.save(wrapper.lead.state_dict(), os.path.join(path, "lead_emb.pt"))
        model.backbone.embeddings = wrapper


# ---------------------------------------------------------------------------
# 勾配マスク
# ---------------------------------------------------------------------------

def freeze_rows_below(weight: torch.Tensor, cutoff: int):
    """行 [0, cutoff) の勾配をゼロにするフックを張る。

    stage-0 と同じ「旧語彙を触らない」設定。ただし stage 2 では
    診断ブロックの NORM / yes / no がすべて旧語彙なので使ってはいけない。
    """
    def hook(grad):
        g = grad.clone()
        g[:cutoff].zero_()
        return g
    return weight.register_hook(hook)