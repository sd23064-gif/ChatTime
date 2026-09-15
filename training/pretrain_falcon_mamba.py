import argparse
import sys
import os

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
# 【変更前】
# from transformers import TrainingArguments, LlamaTokenizer
# 【変更後】
from transformers import AutoTokenizer, AutoModelForCausalLM, EarlyStoppingCallback
# 【変更後】 TrainingArgumentsは使わず、trl.SFTConfig(TrainingArgumentsのサブクラス)に一本化
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
import re
from collections import Counter, deque
import os
import math

from transformers import TrainerCallback
import random


NUMERIC_TOKEN_RE = re.compile(
    r"###([+-]?(?:\d+(?:\.\d*)?|\.\d+)|Nan|NaN|nan)###"
)


def collect_numeric_token_ids(tokenizer):
    rows = []

    for token, token_id in tokenizer.get_vocab().items():
        match = NUMERIC_TOKEN_RE.fullmatch(str(token))

        if match is None:
            continue

        if match.group(1).lower() == "nan":
            continue

        rows.append((int(token_id), float(match.group(1))))

    rows.sort(key=lambda x: x[1])

    numeric_ids = torch.tensor([x[0] for x in rows], dtype=torch.long)
    numeric_values = torch.tensor([x[1] for x in rows], dtype=torch.float32)

    print("numeric token count:", len(numeric_ids))
    print("value range:", numeric_values.min().item(), numeric_values.max().item())

    return numeric_ids, numeric_values


def count_numeric_occurrences(
    dataset,
    tokenizer,
    numeric_ids,
    sample_size=10000,
    seed=3407,
):
    numeric_set = set(
        int(x)
        for x in numeric_ids.tolist()
    )
    counter = Counter()

    n = min(sample_size, len(dataset))

    sample_dataset = dataset.shuffle(
        seed=seed
    ).select(range(n))

    for example in sample_dataset:
        ids = tokenizer(
            example["text"],
            add_special_tokens=False,
        )["input_ids"]

        for token_id in ids:
            if token_id in numeric_set:
                counter[token_id] += 1

    counts = np.array(
        [
            counter.get(int(token_id), 0)
            for token_id in numeric_ids
        ],
        dtype=np.int64,
    )

    return counts

def is_bfloat16_supported():
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


def verify_numeric_tokenization(tokenizer, added_tokens):
    finite_tokens = [
        token
        for token in added_tokens
        if "nan" not in token.lower()
    ]

    sample_indices = np.linspace(
        0,
        len(finite_tokens) - 1,
        5,
        dtype=int,
    )

    check_tokens = [
        finite_tokens[index]
        for index in sample_indices
    ]

    print("\nNumeric tokenization check")

    for token in check_tokens:
        ids = tokenizer.encode(
            token,
            add_special_tokens=False,
        )
        token_id = tokenizer.convert_tokens_to_ids(token)

        print(
            token,
            "token_id=",
            token_id,
            "ids=",
            ids,
            "len=",
            len(ids),
        )

        if len(ids) != 1 or ids[0] != token_id:
            raise ValueError(
                f"{token} is not encoded as one added token: "
                f"id={token_id}, encoded={ids}"
            )

def inspect_embedding_trainability(model):
    input_emb = model.get_input_embeddings()
    output_emb = model.get_output_embeddings()

    print("\nEmbedding trainability")
    print("Input embedding shape:", tuple(input_emb.weight.shape))
    print("Input embedding requires_grad:", input_emb.weight.requires_grad)

    if output_emb is not None and hasattr(output_emb, "weight"):
        print("Output embedding shape:", tuple(output_emb.weight.shape))
        print("Output embedding requires_grad:", output_emb.weight.requires_grad)
        print(
            "Input/output tied:",
            input_emb.weight.data_ptr() == output_emb.weight.data_ptr(),
        )

    print("\nEmbedding-related parameters:")
    for name, param in model.named_parameters():
        lower = name.lower()
        if "embedding" in lower or "embed" in lower or "lm_head" in lower:
            print(name, tuple(param.shape), param.requires_grad)

def register_added_token_gradient_mask(
    weight,
    old_vocab_size,
):
    def mask_gradient(gradient):
        masked = gradient.clone()
        masked[:old_vocab_size].zero_()
        return masked

    return weight.register_hook(mask_gradient)

class SaveTokenizerAtCheckpointCallback(TrainerCallback):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def on_save(
        self,
        args,
        state,
        control,
        **kwargs,
    ):
        # 【変更前】 全プロセスがon_saveのたびに同じチェックポイントへ書き込んでいた
        # 【変更後】 TrainerStateが持つis_world_process_zeroでメインプロセスのみに限定
        if not state.is_world_process_zero:
            return control

        checkpoint_dir = os.path.join(
            args.output_dir,
            f"checkpoint-{state.global_step}",
        )

        os.makedirs(
            checkpoint_dir,
            exist_ok=True,
        )

        self.tokenizer.save_pretrained(
            checkpoint_dir
        )

        print(
            "Saved tokenizer to:",
            checkpoint_dir,
        )

        return control


class DiagnosticSFTTrainer(SFTTrainer):
    """
    【追加】 grad_norm異常検知のための診断用Trainer。
    学習の挙動自体は通常のSFTTrainerと完全に同じで、compute_lossが
    呼ばれるたびに、そのマイクロバッチのinput_idsを直近履歴として
    保持しておくことだけが異なる(gradient_accumulation_steps分だけ保持)。
    NaNGradientDetectorCallbackが異常なgrad_normを検知した際、
    この直近履歴をディスクに保存して、原因バッチの特定に使う。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._recent_batches = deque(
            maxlen=max(1, self.args.gradient_accumulation_steps)
        )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if "input_ids" in inputs:
            self._recent_batches.append(inputs["input_ids"].detach().cpu().clone())

        return super().compute_loss(
            model,
            inputs,
            return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch,
        )


class NaNGradientDetectorCallback(TrainerCallback):
    """
    【追加】 grad_normがNaN/Infまたは閾値超えになった際に、
    その時点までの直近バッチ(input_ids)とステップ番号をディスクに
    保存する診断用コールバック。
    random_seedは固定されているので、再実行すれば同じ場所で再現するはずで、
    次回はここに保存されたバッチを見れば原因データを直接特定できる。
    """

    def __init__(self, trainer, output_dir, threshold=1e6):
        self.trainer = trainer
        self.output_dir = output_dir
        self.threshold = threshold
        self._saved_steps = set()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or "grad_norm" not in logs:
            return control

        grad_norm = logs["grad_norm"]

        is_bad = (
            grad_norm is None
            or (isinstance(grad_norm, float) and (math.isnan(grad_norm) or math.isinf(grad_norm)))
            or (isinstance(grad_norm, (int, float)) and grad_norm > self.threshold)
        )

        if not is_bad or state.global_step in self._saved_steps:
            return control

        self._saved_steps.add(state.global_step)

        # 【重要】 DDPではrank0とrank1で別々のバッチを処理しているため、
        # 原因データがどちらのGPU側にあるか分からない。all-reduce後の
        # grad_normは全rankで(ほぼ)同じ値になり全rankで同時に検知されるので、
        # rank0だけでなく各rankが自分の直近バッチを個別に保存する
        # (ファイル名にrankを含めて衝突を防ぐ)。
        os.makedirs(self.output_dir, exist_ok=True)
        rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
        save_path = os.path.join(
            self.output_dir, f"step_{state.global_step}_rank{rank}.pt"
        )

        torch.save(
            {
                "global_step": state.global_step,
                "rank": rank,
                "grad_norm": grad_norm,
                "logs": logs,
                "recent_input_ids": list(self.trainer._recent_batches),
            },
            save_path,
        )

        print(
            f"[NaNGradientDetectorCallback] Anomalous grad_norm={grad_norm} "
            f"at step {state.global_step} (rank {rank}). Saved recent batches to {save_path}"
        )

        return control


class EmbeddingNormLoggerCallback(TrainerCallback):
    """
    【追加】 指定した数値トークン(例: 値が-0.5/0.0/+0.5付近のトークン)について、
    embedding(入力側)とlm_head(出力側、tie_word_embeddingsでなければ別重み)の
    該当行のL2ノルムを、通常のloggingと同じ頻度でwandbに記録する診断用コールバック。

    「-0.5/+0.5付近のトークンのembeddingが学習中に肥大化し、それがgrad_norm爆発の
    引き金になっているのでは」という仮説を直接検証するために追加。
    """

    def __init__(self, trainer, track_token_ids):
        self.trainer = trainer
        # track_token_ids: {"表示用ラベル": token_id, ...}
        self.track_token_ids = track_token_ids

    def on_log(self, args, state, control, logs=None, **kwargs):
        # 保存ではなくwandbへのログ送信なので、メインプロセスのみでOK
        # (二重送信を避けるため)
        if not state.is_world_process_zero:
            return control

        if len(self.track_token_ids) == 0:
            return control

        try:
            import wandb
        except ImportError:
            return control

        if wandb.run is None:
            return control

        model = self.trainer.accelerator.unwrap_model(self.trainer.model)
        input_embedding = model.get_input_embeddings()
        output_embedding = model.get_output_embeddings()

        # tie_word_embeddingsの場合、入力と出力が同一の重みを共有しているので
        # 二重にログしないようにチェックする
        tied = (
            output_embedding is not None
            and output_embedding.weight.data_ptr() == input_embedding.weight.data_ptr()
        )

        log_dict = {}
        with torch.no_grad():
            for label, token_id in self.track_token_ids.items():
                in_norm = input_embedding.weight[token_id].detach().float().norm().item()
                log_dict[f"embed_norm/input/{label}"] = in_norm

                if output_embedding is not None and not tied:
                    out_norm = output_embedding.weight[token_id].detach().float().norm().item()
                    log_dict[f"embed_norm/output/{label}"] = out_norm

        wandb.log(log_dict, step=state.global_step)

        return control

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--code_path", type=str, required=True, default=None)
    parser.add_argument("--model_path", type=str, required=True, default=None)
    parser.add_argument("--dataset_path", type=str, required=True, default=None)
    parser.add_argument("--log_path", type=str, required=True, default=None)
    parser.add_argument("--output_path", type=str, required=True, default=None)

    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)

    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.00)
    parser.add_argument("--random_seed", type=int, default=3407)

    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=64)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--logging_steps", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=-1)

    parser.add_argument("--low_limit", type=float, default=-1)
    parser.add_argument("--high_limit", type=float, default=1)
    parser.add_argument("--n_tokens", type=int, default=10002)
    parser.add_argument("--prec", type=int, default=4)
    parser.add_argument("--time_sep", type=str, default=" ")
    parser.add_argument("--time_flag", type=str, default="###")
    parser.add_argument("--nan_flag", type=str, default="Nan")

    parser.add_argument("--train_file", type=str, default=None)
    parser.add_argument("--validation_file", type=str, default=None)
    parser.add_argument("--test_file", type=str, default=None)
    parser.add_argument("--do_eval", action="store_true", default=False)
    parser.add_argument("--do_test", action="store_true", default=False)
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--early_stopping_patience", type=int, default=3)
    parser.add_argument("--early_stopping_threshold", type=float, default=0.0)
    parser.add_argument("--dataset_num_proc", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--grad_norm_threshold", type=float, default=1e6)

    parser.add_argument("--wandb_run_name", type=str, default=None)
    args = parser.parse_args()
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)
    
    sys.path.append(args.code_path)
    from utils.tools import Discretizer, Serializer

    if args.wandb_run_name is None:
        args.wandb_run_name = (
            f"mamba-{args.model_path.split('/')[-1]}"
            f"-bs{args.per_device_train_batch_size}"
            f"-ga{args.gradient_accumulation_steps}"
        )
    # construct vocabulary
    
    # 【変更前】
    # tokenizer = LlamaTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    # tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.padding_side = "right"

    # 【変更後】
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    discretizer = Discretizer(low_limit=args.low_limit, high_limit=args.high_limit, n_tokens=args.n_tokens)
    serializer = Serializer(prec=args.prec, time_sep=args.time_sep, time_flag=args.time_flag, nan_flag=args.nan_flag)

    vocabulary = np.concatenate((discretizer.centers[1:-1], [np.nan])).reshape(-1, 1)
    vocabulary = np.array([serializer.serialize(value) for value in vocabulary])
    added_tokens = [str(token) for token in vocabulary.reshape(-1).tolist()]

    old_vocab_size = len(tokenizer)
    num_added_tokens = tokenizer.add_tokens(added_tokens)

    print(f"Old tokenizer size: {old_vocab_size}")
    print(f"Added tokens: {num_added_tokens}")
    print(f"New tokenizer size: {len(tokenizer)}")

    # 【追加】 embeddingノルムを監視したいトークンのIDを、discretizerの実際の
    # bin中心値(数値トークンに対応する値)から動的に求める。
    # ハードコードした固定IDに頼らず、low_limit/high_limit/n_tokensの設定が
    # 変わっても正しく追従できるようにするため。
    numeric_values = discretizer.centers[1:-1]  # nanトークン分を除いた実際の数値
    embedding_norm_track_ids = {}
    for target_value in (-0.5, 0.0, 0.5):
        idx = int(np.argmin(np.abs(numeric_values - target_value)))
        token_id = old_vocab_size + idx
        label = f"value~{numeric_values[idx]:.4f}"
        embedding_norm_track_ids[label] = token_id
    print(f"Tracking embedding norms for: {embedding_norm_track_ids}")

    verify_numeric_tokenization(tokenizer, added_tokens)

    expected_added_tokens = len(added_tokens)

    if num_added_tokens != expected_added_tokens:
        raise ValueError(
            "Unexpected number of added tokens: "
            f"expected={expected_added_tokens}, "
            f"actual={num_added_tokens}. "
            "Check whether model_path already contains the added tokenizer."
        )
    
    print(f"Old tokenizer size: {old_vocab_size}")
    print(f"Added tokens: {num_added_tokens}")
    print(f"New tokenizer size: {len(tokenizer)}")
    
    print(f"Added tokens: {num_added_tokens}")
    print(f"New tokenizer size: {len(tokenizer)}")

    EOS_TOKEN = tokenizer.eos_token


    # load model
    # 【変更前】 device_map="auto" 固定
    #   → 1つのモデルを複数GPUに"分割配置"するモード。7Bはbf16で1枚(49GB)に
    #     余裕で収まるため、torchrun等でプロセスを2つ起動してデータ並列(DDP)
    #     したい場合はむしろ邪魔になる(各プロセスが全GPUを掴もうとして衝突する)。
    # 【変更後】 LOCAL_RANKが設定されている(=torchrun/accelerate launchで
    #   複数プロセス起動されている)場合は、そのプロセスが担当する1枚のGPUだけに
    #   モデルを丸ごと載せる。単一プロセス実行時は従来通りauto。
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    device_map = {"": local_rank} if local_rank != -1 else "auto"
    print(f"LOCAL_RANK={local_rank} -> device_map={device_map}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if is_bfloat16_supported() else torch.float16,
        device_map=device_map,
        trust_remote_code=True,
    )

    model.resize_token_embeddings(len(tokenizer))
    model.config.use_cache = False
    if hasattr(model, "generation_config"):
        model.generation_config.use_cache = False


    # add lora to llama model
    # 【変更前】 target_modules=["q_proj", "k_proj", ...], modules_to_save=["embed_tokens", "lm_head"]

    # 【変更後】

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "x_proj",
            "in_proj",
            "dt_proj",
            "out_proj",
        ],
        modules_to_save=[
            "backbone.embeddings",
            "lm_head",
        ],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    input_embedding = model.get_input_embeddings()

    input_grad_hook = register_added_token_gradient_mask(
        input_embedding.weight,
        old_vocab_size,
    )

    output_embedding = model.get_output_embeddings()

    output_grad_hook = None

    if (
        output_embedding is not None
        and hasattr(output_embedding, "weight")
        and output_embedding.weight.data_ptr()
        != input_embedding.weight.data_ptr()
    ):
        output_grad_hook = register_added_token_gradient_mask(
            output_embedding.weight,
            old_vocab_size,
    )
    inspect_embedding_trainability(model)

    print("\nLinear-like module names:")
    for name, module in model.named_modules():
        if any(key in name for key in ["in_proj", "x_proj", "dt_proj", "out_proj"]):
            print(name, type(module))

    # load dataset
    data_files = {}

    if args.train_file is not None:
        data_files["train"] = args.train_file
    else:
        data_files["train"] = f"https://huggingface.co/datasets/{args.dataset_path}/resolve/main/ChatTime-1-Pretrain-1M.csv"

    if args.validation_file is not None:
        data_files["validation"] = args.validation_file

    if args.test_file is not None:
        data_files["test"] = args.test_file

    dataset_dict = load_dataset("csv", data_files=data_files)
    def add_eos(example):
        text = example["text"]

        if EOS_TOKEN is not None and not text.endswith(EOS_TOKEN):
            text = text + EOS_TOKEN

        return {"text": text}
    train_dataset = dataset_dict["train"]
    eval_dataset = dataset_dict["validation"] if "validation" in dataset_dict else None
    test_dataset = dataset_dict["test"] if "test" in dataset_dict else None
    train_dataset = train_dataset.map(
        add_eos,
        num_proc=args.dataset_num_proc,
    )

    if eval_dataset is not None:
        eval_dataset = eval_dataset.map(
            add_eos,
            num_proc=args.dataset_num_proc,
        )

    if test_dataset is not None:
        test_dataset = test_dataset.map(
            add_eos,
            num_proc=args.dataset_num_proc,
        )

    print("Train example:")
    print(train_dataset[0]["text"])

    numeric_token_ids, numeric_token_values = collect_numeric_token_ids(tokenizer)

    counts = count_numeric_occurrences(
        dataset=train_dataset,
        tokenizer=tokenizer,
        numeric_ids=numeric_token_ids,
        sample_size=10000,
    )
    callbacks = [
        SaveTokenizerAtCheckpointCallback(
            tokenizer=tokenizer,
        )
    ]
    has_eval = eval_dataset is not None
    if has_eval:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=(
                    args.early_stopping_patience
                ),
                early_stopping_threshold=(
                    args.early_stopping_threshold
                ),
            )
        )
    # 【変更前】 全プロセス(rank0, rank1, ...)が同時に同じパスへ保存していた
    #   → ファイル書き込みの競合や破損の原因になるため、実際の保存処理はメインプロセスのみに限定する
    # 【注意】 step0_pathは後段(566行目付近のnumeric_embeddings_step0.pt保存)でも
    #   使うので、パスの文字列自体は全プロセスで定義しておく(NameError防止)
    step0_path = os.path.join(args.output_path, "step-0")

    if local_rank in (-1, 0):
        os.makedirs(step0_path, exist_ok=True)
        model.save_pretrained(step0_path)
        tokenizer.save_pretrained(step0_path)

        print(f"Saved step-0 adapter to: {step0_path}")
    # train model
    sft_config = SFTConfig(
        # --- データセット関連の引数 ---
        # 【変更前】 これらはSFTTrainerに直接渡していた(trl<0.12系のAPI)
        # 【変更後】 trl>=0.12系ではSFTConfig側に持たせる
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        dataset_num_proc=args.dataset_num_proc,
        packing=False,

        # --- 学習系の引数(値は元のTrainingArgumentsと同じ) ---
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=(
            args.per_device_eval_batch_size
            if args.per_device_eval_batch_size is not None
            else args.per_device_train_batch_size
        ),
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,

        weight_decay=0.01,
        warmup_ratio=0.05,
        max_grad_norm=1.0,
        learning_rate=args.learning_rate,

        logging_strategy="steps",
        logging_steps=args.logging_steps,
        logging_first_step=True,

        # 【変更前】 evaluation_strategy=...  (transformers 4.46で削除された旧名)
        # 【変更後】 eval_strategy=...
        eval_strategy="steps" if has_eval else "no",
        eval_steps=args.eval_steps if has_eval else None,

        save_strategy="steps",
        save_steps=args.save_steps,
        max_steps=args.max_steps,
        save_total_limit=None,

        load_best_model_at_end=has_eval,
        metric_for_best_model="eval_loss" if has_eval else None,
        greater_is_better=False if has_eval else None,

        optim="adamw_8bit",
        lr_scheduler_type="cosine",
        seed=args.random_seed,
        output_dir=args.log_path,

        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),

        report_to="wandb",
        run_name=args.wandb_run_name,
    )

    trainer = DiagnosticSFTTrainer(
        model=model,
        # 【変更前】 tokenizer=tokenizer  (trl 0.16で完全削除された旧名)
        # 【変更後】 processing_class=tokenizer
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=sft_config,
        callbacks=callbacks,
    )

    # 【追加】 grad_normがNaN/Infや閾値超えになった際に、直近バッチを保存する診断用コールバック
    trainer.add_callback(
        NaNGradientDetectorCallback(
            trainer=trainer,
            output_dir=os.path.join(args.output_path, "nan_diagnostics"),
            threshold=args.grad_norm_threshold,
        )
    )

    # 【追加】 -0.5/0.0/+0.5付近のembedding/lm_headのL2ノルムをwandbに記録する診断用コールバック
    trainer.add_callback(
        EmbeddingNormLoggerCallback(
            trainer=trainer,
            track_token_ids=embedding_norm_track_ids,
        )
    )

    # title Show current memory stats
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"\nGPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.\n")

    numeric_token_ids, numeric_token_values = collect_numeric_token_ids(tokenizer)
    numeric_ids_device = numeric_token_ids.to(model.get_input_embeddings().weight.device)

    initial_numeric_embeddings = model.get_input_embeddings().weight[numeric_ids_device].detach().float().cpu().clone()

    # 【変更前】 全プロセスが同時に同じファイルへtorch.saveしていた
    # 【変更後】 メインプロセスのみに限定
    if local_rank in (-1, 0):
        torch.save({
            "token_ids": numeric_token_ids,
            "values": numeric_token_values,
            "embeddings": initial_numeric_embeddings,
        }, os.path.join(step0_path, "numeric_embeddings_step0.pt"))

    trainer_stats = trainer.train()

    trained_numeric_embeddings = (
        model.get_input_embeddings()
        .weight[numeric_ids_device]
        .detach()
        .float()
        .cpu()
    )

    delta = trained_numeric_embeddings - initial_numeric_embeddings
    l2_change = delta.norm(dim=-1)

    initial_dir = torch.nn.functional.normalize(
        initial_numeric_embeddings,
        dim=-1,
    )
    trained_dir = torch.nn.functional.normalize(
        trained_numeric_embeddings,
        dim=-1,
    )

    cos_initial_trained = (initial_dir * trained_dir).sum(dim=-1)

    # ============================================================
    # Analyze relation between token frequency and embedding update
    # ============================================================

    initial_norm = initial_numeric_embeddings.norm(dim=-1)
    trained_norm = trained_numeric_embeddings.norm(dim=-1)

    relative_l2_change = (
        l2_change / initial_norm.clamp(min=1e-8)
    )

    norm_change = trained_norm - initial_norm

    print("\nNumeric embedding norm statistics")
    print("mean initial norm:", initial_norm.mean().item())
    print("median initial norm:", initial_norm.median().item())
    print("mean trained norm:", trained_norm.mean().item())
    print("median trained norm:", trained_norm.median().item())
    print(
        "mean relative L2 change:",
        relative_l2_change.mean().item(),
    )
    print(
        "median relative L2 change:",
        relative_l2_change.median().item(),
    )
    print("mean norm change:", norm_change.mean().item())
    print("median norm change:", norm_change.median().item())


    update_df = pd.DataFrame(
        {
            "token_id": numeric_token_ids.cpu().numpy(),
            "value": numeric_token_values.cpu().numpy(),
            "abs_value": numeric_token_values.abs().cpu().numpy(),
            "occurrence_count_sample": counts,
            "initial_norm": initial_norm.cpu().numpy(),
            "trained_norm": trained_norm.cpu().numpy(),
            "norm_change": norm_change.cpu().numpy(),
            "l2_change": l2_change.cpu().numpy(),
            "relative_l2_change": relative_l2_change.cpu().numpy(),
            "initial_trained_cosine": (
                cos_initial_trained.cpu().numpy()
            ),
        }
    )


    # Spearman correlation is suitable because occurrence counts
    # can be strongly skewed and need not have a linear relation.
    frequency_l2_spearman = update_df[
        "occurrence_count_sample"
    ].corr(
        update_df["l2_change"],
        method="spearman",
    )

    frequency_relative_l2_spearman = update_df[
        "occurrence_count_sample"
    ].corr(
        update_df["relative_l2_change"],
        method="spearman",
    )

    frequency_norm_change_spearman = update_df[
        "occurrence_count_sample"
    ].corr(
        update_df["norm_change"],
        method="spearman",
    )

    frequency_cosine_spearman = update_df[
        "occurrence_count_sample"
    ].corr(
        update_df["initial_trained_cosine"],
        method="spearman",
    )


    print("\nFrequency-update correlations")
    print(
        "Spearman corr(occurrence_count, l2_change):",
        frequency_l2_spearman,
    )
    print(
        "Spearman corr(occurrence_count, relative_l2_change):",
        frequency_relative_l2_spearman,
    )
    print(
        "Spearman corr(occurrence_count, norm_change):",
        frequency_norm_change_spearman,
    )
    print(
        "Spearman corr(occurrence_count, initial_trained_cosine):",
        frequency_cosine_spearman,
    )


    not_observed_mask = (
        update_df["occurrence_count_sample"] == 0
    )

    observed_mask = (
        update_df["occurrence_count_sample"] > 0
    )

    unchanged_mask = (
        update_df["l2_change"] < 1e-4
    )


    print("\nObserved / unobserved token statistics")
    print(
        "No-occurrence token count:",
        int(not_observed_mask.sum()),
    )
    print(
        "Observed token count:",
        int(observed_mask.sum()),
    )

    if not_observed_mask.any():
        print(
            "No-occurrence mean L2 change:",
            update_df.loc[
                not_observed_mask,
                "l2_change",
            ].mean(),
        )
        print(
            "No-occurrence unchanged ratio:",
            (
                update_df.loc[
                    not_observed_mask,
                    "l2_change",
                ] < 1e-4
            ).mean(),
        )

    if observed_mask.any():
        print(
            "Observed mean L2 change:",
            update_df.loc[
                observed_mask,
                "l2_change",
            ].mean(),
        )
        print(
            "Observed unchanged ratio:",
            (
                update_df.loc[
                    observed_mask,
                    "l2_change",
                ] < 1e-4
            ).mean(),
        )

    print(
        "Overall unchanged token count:",
        int(unchanged_mask.sum()),
    )


    # 【変更前】 全プロセスが同時にCSV・アダプタを同じパスへ保存していた
    # 【変更後】 メインプロセスのみに限定
    if local_rank in (-1, 0):
        # Save per-token results
        os.makedirs(args.output_path, exist_ok=True)

        update_csv_path = os.path.join(
            args.output_path,
            "numeric_embedding_updates.csv",
        )

        update_df.to_csv(
            update_csv_path,
            index=False,
        )

        print(
            "Saved numeric embedding update analysis:",
            update_csv_path,
        )
        # save model and tokenizer
        print(f"Saving LoRA adapter to {args.output_path}")

        model.save_pretrained(args.output_path)
        tokenizer.save_pretrained(args.output_path)

        print("Save completed")