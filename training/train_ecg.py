"""
train_ecg.py

stage 1: ECG 波形のみで数値トークンを継続学習 (次トークン予測)
stage 2: 波形 + 患者情報 + 誘導情報 -> 診断チェックリストのファインチューニング

stage-0 (LoRA + modules_to_save で数値行だけ学習したもの) をマージした
密なモデルを出発点にする。stage 1 以降は ECG のダイナミクス自体を学ぶ必要が
あるので、既定では backbone も含めて密に学習する。メモリが厳しければ
--freeze_backbone で埋め込みと lm_head だけに絞れる。
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
# 断片化の緩和。expandable_segments はドライバや環境によって未対応で
# "not supported on this platform" 警告になるため、既定は max_split_size_mb。
# 使える環境なら PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True を
# 明示的に渡せばそちらが優先される。
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512")

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from transformers import (AutoTokenizer, Trainer, TrainerCallback,
                          TrainingArguments)

from ecg_data import EcgCollator, PTBXLTokenDataset
from ecg_loss import smoothness
from ecg_model import freeze_rows_below, load_model, sanity_check, save_all, tie_status
from ecg_vocab import describe, numeric_vocab


# DDP では torchrun が RANK / LOCAL_RANK を設定する
RANK = int(os.environ.get("RANK", "0"))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))


def p0(*a, **kw):
    """rank 0 だけ出力する。"""
    if RANK == 0:
        print(*a, **kw)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class EcgTrainer(Trainer):
    def __init__(self, *a, num_start=0, n_num=0, **kw):
        super().__init__(*a, **kw)
        self.num_start = num_start
        self.n_num = n_num
        self._init_numeric = None

    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        outputs = model(**inputs)          # 損失はモデル内部で計算される
        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss

    # -- ロギング ------------------------------------------------------------
    def _numeric_block(self):
        base = self.model
        lo, hi = self.num_start, self.num_start + self.n_num
        return base.get_input_embeddings().weight[lo:hi].detach()

    @torch.no_grad()
    def _stats(self):
        """数値埋め込みの状態を wandb 用にまとめる。

        stage-0 で出していた numeric_embedding_updates.csv と同じ観点を、
        学習中に逐次見られるようにしたもの。drift が伸び続けるのに loss が
        下がらない場合は学習率が高すぎる。
        """
        out = {}
        g = 1024 ** 3
        if torch.cuda.is_available():
            out["mem/alloc_gb"] = torch.cuda.max_memory_allocated() / g
            out["mem/reserved_gb"] = torch.cuda.max_memory_reserved() / g
            torch.cuda.reset_peak_memory_stats()

        blk = self._numeric_block().float()
        norm = blk.norm(dim=-1)
        out["emb/numeric_norm_mean"] = norm.mean().item()
        out["emb/numeric_norm_std"] = norm.std().item()
        out["emb/smoothness"] = smoothness(blk).item()

        if self._init_numeric is None:
            self._init_numeric = blk.clone()
        else:
            d = (blk - self._init_numeric).norm(dim=-1)
            rel = d / self._init_numeric.norm(dim=-1).clamp_min(1e-8)
            out["emb/drift_l2_mean"] = d.mean().item()
            out["emb/drift_rel_mean"] = rel.mean().item()
            out["emb/drift_rel_max"] = rel.max().item()
        return out

    def log(self, logs, *args, **kwargs):
        if self.is_world_process_zero() and ("loss" in logs or "eval_loss" in logs):
            try:
                logs.update(self._stats())
            except Exception as e:      # ロギングで学習を止めない
                print(f"[warn] stats failed: {e}")
        return super().log(logs, *args, **kwargs)

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """評価では損失だけ返す。

        既定の prediction_step は compute_loss(return_outputs=True) の第 2 要素を
        ロジット扱いして eval ループ中ずっと蓄積する。ここでは隠れ状態
        (B, 8000, 1536) がそれに当たるので、放っておくとメモリが飛ぶ。
        """
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            loss = self.compute_loss(model, inputs)
        return (loss.detach(), None, None)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------



def report_memory_budget(n_all, n_train, args) -> None:
    """学習前に主要なメモリ内訳を見積もって出す。

    OOM は大半がオプティマイザ側のピークで起きるので、
    実行前に内訳が分かるようにしておく。
    """
    pbytes = 4 if args.load_dtype == "float32" else 2
    params = n_all * pbytes
    grads = n_train * pbytes
    if args.optim == "adamw_bnb_8bit":
        states, extra = n_train * 2, 0
    elif args.optim == "adafactor":
        states, extra = n_train * 0.5, 0
    else:
        # torch の AdamW は状態をパラメータと同じ dtype で持つ
        states = n_train * 2 * pbytes              # exp_avg + exp_avg_sq
        # foreach 実装は状態と同サイズの一時領域を確保する
        extra = 0 if args.optim == "adamw_torch_fused" else n_train * pbytes
    total = params + grads + states + extra
    g = 1024 ** 3
    print(f"memory estimate: params={params/g:.2f} grads={grads/g:.2f} "
          f"optim={states/g:.2f} temp={extra/g:.2f} -> {total/g:.2f} GB "
          f"(+ 活性化)")
    if torch.cuda.is_available():
        free, cap = torch.cuda.mem_get_info()
        print(f"GPU free={free/g:.2f} GB / capacity={cap/g:.2f} GB")
        if free < total * 1.4:
            print("[warn] 余裕がありません。--optim adamw_bnb_8bit / "
                  "--freeze_backbone / --seconds 5 を検討してください。")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", type=int, required=True, choices=[1, 2])
    p.add_argument("--model_path", required=True,
                   help="stage1 なら stage-0 の出力 (PEFT 可), stage2 なら stage-1 の出力")
    p.add_argument("--base_model_path", default=None,
                   help="adapter_config.json の base を上書きしたい場合")
    p.add_argument("--data_path", required=True)
    p.add_argument("--output_path", required=True)
    p.add_argument("--log_path", required=True)

    # 系列レイアウト
    p.add_argument("--order", default="time_major",
                   choices=["time_major", "lead_major"])
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--leads", type=str, default="")
    p.add_argument("--per_lead_examples", action="store_true",
                   help="stage1 前半用: 1 誘導 = 1 サンプルで短い系列を大量に回す")
    p.add_argument("--lead_pos_emb", action="store_true")
    p.add_argument("--no_random_crop", action="store_true")

    # 損失
    p.add_argument("--soft_sigma_in_bins", type=float, default=3.0)
    p.add_argument("--lambda_smooth", type=float, default=0.0)
    p.add_argument("--smooth_target", default="both", choices=["input", "both"])
    p.add_argument("--signal_loss_weight", type=float, default=0.1,
                   help="stage2 で波形の再構成損失を残す重み。0 で完全マスク")
    p.add_argument("--loss_chunk", type=int, default=512,
                   help="大きいほど速いがロジットの一時領域が増える")
    p.add_argument("--label_field", default="superclass",
                   choices=["superclass", "subclass"])
    p.add_argument("--classes", default="NORM,MI,STTC,CD,HYP")

    # 学習範囲
    p.add_argument("--freeze_backbone", action="store_true",
                   help="埋め込みと lm_head だけ学習する")
    p.add_argument("--freeze_text_rows", action="store_true",
                   help="旧語彙の行を凍結 (stage 1 のみ。stage 2 では使用不可)")

    # 分割
    p.add_argument("--train_folds", default="1,2,3,4,5,6,7,8")
    p.add_argument("--val_folds", default="9")

    # 最適化
    p.add_argument("--num_train_epochs", type=float, default=2)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=32)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--emb_lr_mult", type=float, default=1.0)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--eval_steps", type=int, default=500)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--random_seed", type=int, default=3407)
    p.add_argument("--ddp_timeout", type=int, default=900,
                   help="集団通信のタイムアウト秒。既定 1800 だとハングが長引く")
    p.add_argument("--ddp_find_unused_parameters", action="store_true",
                   help="通常は不要。損失は forward 内で計算しており全パラメータが使われる")
    p.add_argument("--dataloader_num_workers", type=int, default=4,
                   help="プロセスあたりの worker 数")

    # ログ
    p.add_argument("--report_to", default="wandb", choices=["wandb", "none"])
    p.add_argument("--wandb_project", default="ptbxl-mamba")
    p.add_argument("--wandb_run_name", default=None)
    p.add_argument("--wandb_mode", default=None,
                   choices=["online", "offline", "disabled"],
                   help="API キーがない環境では offline")
    p.add_argument("--optim", default="adamw_torch_fused",
                   choices=["adamw_torch_fused", "adamw_torch", "adamw_bnb_8bit",
                            "adafactor"],
                   help="fused は foreach 実装の一時領域を使わない。"
                        "bnb_8bit は Adam 状態を 1/4 にする")
    p.add_argument("--load_dtype", default="bfloat16",
                   choices=["float32", "bfloat16"],
                   help="Mamba の融合カーネルは bf16 前提。fp32 だと大幅に遅くメモリも食う")
    args = p.parse_args()

    if args.report_to == "wandb":
        os.environ["WANDB_PROJECT"] = args.wandb_project
        if args.wandb_mode:
            os.environ["WANDB_MODE"] = args.wandb_mode
        if args.wandb_run_name is None:
            tag = "perlead" if args.per_lead_examples else args.order
            args.wandb_run_name = (
                f"s{args.stage}-{tag}-{args.seconds:g}s"
                f"-lr{args.learning_rate:g}-{args.load_dtype}"
                f"-ss{args.soft_sigma_in_bins:g}-sm{args.lambda_smooth:g}")

    if args.stage == 2 and args.freeze_text_rows:
        raise ValueError(
            "stage 2 の診断ブロック (NORM / yes / no) は旧語彙です。"
            "--freeze_text_rows を付けるとそれらの行が一切適応しません。")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"          # Mamba は左パディング厳禁

    num_start, values, nan_id = numeric_vocab(tokenizer)
    if RANK == 0:
        describe(num_start, values, nan_id, len(tokenizer))
    n_num = len(values)

    data_cfg = json.load(open(os.path.join(args.data_path, "config.json")))
    lead_subset = [s for s in args.leads.split(",") if s] or None
    n_leads = len(lead_subset or data_cfg["leads"])

    model = load_model(args.model_path, tokenizer,
                       base_model_path=args.base_model_path,
                       dtype=getattr(torch, args.load_dtype),
                       n_leads=n_leads if args.lead_pos_emb else 0)
    if RANK == 0:
        sanity_check(model, num_start, n_num)
        tie_status(model)
    model.config.use_cache = False

    if args.freeze_backbone:
        for n, prm in model.named_parameters():
            prm.requires_grad = ("embeddings" in n) or n.startswith("lm_head")
    if args.freeze_text_rows:
        freeze_rows_below(model.get_input_embeddings().weight, num_start)
        freeze_rows_below(model.lm_head.weight, num_start)
        p0(f"旧語彙 [0, {num_start}) の勾配をマスクしました")

    model.configure_loss(
        num_start=num_start, n_num=n_num, values=values,
        soft_sigma=args.soft_sigma_in_bins * float(np.diff(values).mean()),
        lambda_smooth=args.lambda_smooth,
        smooth_target=args.smooth_target,
        loss_chunk=args.loss_chunk,
    )

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    p0(f"trainable params: {n_train:,} / {n_all:,}")
    if RANK == 0:
        report_memory_budget(n_all, n_train, args)

    classes = [c for c in args.classes.split(",") if c]
    common = dict(order=args.order, seconds=args.seconds, lead_subset=lead_subset,
                  per_lead_examples=args.per_lead_examples, classes=classes,
                  label_field=args.label_field,
                  signal_loss_weight=args.signal_loss_weight, stage=args.stage)

    train_ds = PTBXLTokenDataset(
        args.data_path, tokenizer, num_start,
        folds=[int(x) for x in args.train_folds.split(",")],
        random_crop=not args.no_random_crop, seed=args.random_seed, **common)
    val_ds = PTBXLTokenDataset(
        args.data_path, tokenizer, num_start,
        folds=[int(x) for x in args.val_folds.split(",")],
        random_crop=False, seed=0, **common)

    ex = train_ds[0]
    p0(f"\ntrain={len(train_ds)} val={len(val_ds)} seq_len={len(ex['input_ids'])} "
       f"world_size={WORLD_SIZE}")
    p0("先頭:", tokenizer.decode(ex["input_ids"][:48]).replace("\n", " | "))
    if args.stage == 2:
        p0("末尾:", tokenizer.decode(ex["input_ids"][-40:]).replace("\n", " | "))


    targs = TrainingArguments(
        output_dir=args.log_path,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.03,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        optim=args.optim,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        logging_first_step=True,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=1,
        evaluation_strategy="steps",   # 4.40 系では eval_strategy は存在しない
        eval_steps=args.eval_steps,
        seed=args.random_seed,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        prediction_loss_only=True,
        remove_unused_columns=False,
        dataloader_num_workers=args.dataloader_num_workers,
        ddp_find_unused_parameters=args.ddp_find_unused_parameters,
        ddp_bucket_cap_mb=100,
        ddp_timeout=args.ddp_timeout,
        report_to=["wandb"] if args.report_to == "wandb" else [],
        run_name=args.wandb_run_name,
    )

    trainer = EcgTrainer(
        model=model, args=targs,
        train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=EcgCollator(tokenizer.pad_token_id),
        num_start=num_start, n_num=n_num,
    )

    class CropEpochCallback(TrainerCallback):
        """エポックごとにクロップ位置を変える。

        Dataset はインデックスから決定的に乱数を作るので、
        epoch を渡さないと毎エポック同じ切り出しになる。
        """

        def on_epoch_begin(self, a, state, control, **kw):
            train_ds.epoch = int(state.epoch or 0)
            return control

    trainer.add_callback(CropEpochCallback())

    gpu = torch.cuda.get_device_properties(0)
    p0(f"GPU = {gpu.name}, {gpu.total_memory/1024**3:.1f} GB x {WORLD_SIZE}")

    stats = trainer.train()
    p0(f"\n{stats.metrics['train_runtime']/60:.1f} min, "
          f"peak {torch.cuda.max_memory_reserved()/1024**3:.1f} GB")

    trainer.accelerator.wait_for_everyone()
    if trainer.is_world_process_zero():
        save_all(model, tokenizer, args.output_path)
        with open(os.path.join(args.output_path, "ecg_train_config.json"), "w") as f:
            json.dump(vars(args), f, indent=2, ensure_ascii=False)
        print(f"saved to {args.output_path}")
    trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()