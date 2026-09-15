import argparse
import inspect
import os
import sys

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


def is_bfloat16_supported():
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


# torchrunが設定する環境変数
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
RANK = int(os.environ.get("RANK", 0))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
IS_MAIN_PROCESS = RANK == 0


def log0(message):
    """rank 0だけがログを出力する。"""
    if IS_MAIN_PROCESS:
        print(message, flush=True)


def add_eos_to_example(example, eos_token):
    """単一文字列と文字列リストの両方にEOSを追加する。"""

    def add_eos(text):
        if text is None:
            text = ""
        elif not isinstance(text, str):
            text = str(text)

        if eos_token and not text.endswith(eos_token):
            text += eos_token

        return text

    texts = example["text"]

    if isinstance(texts, list):
        return [add_eos(text) for text in texts]

    return add_eos(texts)


def print_trainable_parameters(model):
    """学習対象パラメータ名と総数を表示する。"""
    trainable_params = 0
    total_params = 0

    log0("\n=== Trainable parameters ===")

    for name, parameter in model.named_parameters():
        total_params += parameter.numel()

        if parameter.requires_grad:
            trainable_params += parameter.numel()
            log0(
                f"{name}: shape={tuple(parameter.shape)}, "
                f"numel={parameter.numel():,}, dtype={parameter.dtype}"
            )

    trainable_ratio = (
        100.0 * trainable_params / total_params
        if total_params > 0
        else 0.0
    )

    log0(f"\nTrainable parameters: {trainable_params:,}")
    log0(f"Total parameters:     {total_params:,}")
    log0(f"Trainable ratio:      {trainable_ratio:.6f}%\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # 必須パス
    parser.add_argument("--code_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--log_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)

    # モデルとトークナイズ
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)

    # 再現性
    parser.add_argument("--random_seed", type=int, default=3407)

    # 学習設定
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=16,
    )
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)

    # 時系列トークン設定
    parser.add_argument("--low_limit", type=float, default=-1)
    parser.add_argument("--high_limit", type=float, default=1)
    parser.add_argument("--n_tokens", type=int, default=10002)
    parser.add_argument("--prec", type=int, default=4)
    parser.add_argument("--time_sep", type=str, default=" ")
    parser.add_argument("--time_flag", type=str, default="###")
    parser.add_argument("--nan_flag", type=str, default="Nan")

    # Weights & Biases
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="mamba-continuous-pretraining",
    )
    parser.add_argument("--run_name", type=str, default=None)

    # データセット
    parser.add_argument(
        "--dataset_file",
        type=str,
        default="ChatTime-1-Pretrain-1M.csv",
    )
    parser.add_argument("--train_file", type=str, default=None)
    parser.add_argument("--validation_file", type=str, default=None)
    parser.add_argument("--test_file", type=str, default=None)
    parser.add_argument(
        "--dataset_num_proc",
        type=int,
        default=8,
        help="データセット前処理のプロセス数",
    )

    # 追加トークンだけを更新するオプション
    parser.add_argument(
        "--train_new_tokens_only",
        action="store_true",
        default=False,
        help="EmbeddingとLM Headの追加トークン行だけを更新する",
    )

    args = parser.parse_args()

    # W&B設定
    os.environ.setdefault("WANDB_PROJECT", args.wandb_project)

    # 乱数シード
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)
        torch.cuda.set_device(LOCAL_RANK)

    # 外部ユーティリティ
    sys.path.append(args.code_path)
    from utils.tools import Discretizer, Serializer

    # =========================================================
    # 1. 時系列用語彙の構築
    # =========================================================
    discretizer = Discretizer(
        low_limit=args.low_limit,
        high_limit=args.high_limit,
        n_tokens=args.n_tokens,
    )

    serializer = Serializer(
        prec=args.prec,
        time_sep=args.time_sep,
        time_flag=args.time_flag,
        nan_flag=args.nan_flag,
    )

    vocabulary_values = np.concatenate(
        (discretizer.centers[1:-1], [np.nan])
    ).reshape(-1, 1)

    vocabulary = np.array(
        [serializer.serialize(value) for value in vocabulary_values]
    )

    log0(f"\nVocabulary size to add: {len(vocabulary):,}")
    log0(f"Vocabulary sample:\n{vocabulary[:10]}\n")

    # =========================================================
    # 2. Expanded Tokenizerの構築
    # =========================================================
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise RuntimeError(
                "tokenizerにpad_tokenとeos_tokenの両方がありません。"
            )

        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    original_vocab_size = len(tokenizer)

    log0(f"Original tokenizer size: {original_vocab_size:,}")

    # vocabulary.tolist()は文字列のリストを想定
    added_token_count = tokenizer.add_tokens(vocabulary.tolist())
    expanded_vocab_size = len(tokenizer)

    log0(f"Added token count:      {added_token_count:,}")
    log0(f"Expanded tokenizer size:{expanded_vocab_size:,}\n")

    EOS_TOKEN = tokenizer.eos_token

    if EOS_TOKEN is None:
        raise RuntimeError(
            "EOSトークンが設定されていません。"
            "トークナイザー設定を確認してください。"
        )

    # =========================================================
    # 3. モデルを担当GPUへ読み込む
    # =========================================================
    model_kwargs = {
        "trust_remote_code": True,
        "device_map": {"": LOCAL_RANK},
    }

    if args.load_in_4bit:
        log0(
            "[WARNING] 4bit量子化でEmbeddingとLM Headを直接学習する構成は、"
            "モデル実装によっては正常に動作しない場合があります。"
        )

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=(
                torch.bfloat16
                if is_bfloat16_supported()
                else torch.float16
            ),
        )

        model_kwargs["quantization_config"] = bnb_config
    else:
        model_kwargs["torch_dtype"] = (
            torch.bfloat16
            if is_bfloat16_supported()
            else torch.float16
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        **model_kwargs,
    )

    # Expanded Tokenizerに合わせてEmbeddingとLM Headを拡張
    model.resize_token_embeddings(expanded_vocab_size)

    # 学習時はKV cacheを無効化
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    # 図(b)ではMamba層を凍結するため、
    # Gradient Checkpointingは使用しない
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()

    # =========================================================
    # 4. 図(b)の学習対象を設定
    #
    # Embedding:    学習
    # Mamba Layers: 凍結
    # LM Head:      学習
    # =========================================================

    # モデル全体を凍結
    for parameter in model.parameters():
        parameter.requires_grad = False

    # モジュール名に依存せず、標準APIで取得
    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()

    if input_embeddings is None:
        raise RuntimeError(
            "model.get_input_embeddings()で"
            "Embedding層を取得できませんでした。"
        )

    if output_embeddings is None:
        raise RuntimeError(
            "model.get_output_embeddings()で"
            "LM Headを取得できませんでした。"
        )

    # Expanded Embeddingを学習可能にする
    for parameter in input_embeddings.parameters():
        parameter.requires_grad = True

    # Expanded LM Headを学習可能にする
    for parameter in output_embeddings.parameters():
        parameter.requires_grad = True

    input_weight = input_embeddings.weight
    output_weight = output_embeddings.weight

    # EmbeddingとLM Headが重みを共有しているか確認
    tied_embeddings = (
        input_weight.data_ptr() == output_weight.data_ptr()
    )

    log0(f"Input/output weights tied: {tied_embeddings}")
    log0(f"Input embedding shape:  {tuple(input_weight.shape)}")
    log0(f"Output LM Head shape:   {tuple(output_weight.shape)}")

    if hasattr(model.config, "tie_word_embeddings"):
        log0(
            f"config.tie_word_embeddings: "
            f"{model.config.tie_word_embeddings}"
        )

    # =========================================================
    # 5. 必要に応じて追加トークン行だけを更新
    # =========================================================
    gradient_hook_handles = []

    if args.train_new_tokens_only:
        if added_token_count == 0:
            raise RuntimeError(
                "--train_new_tokens_onlyが指定されましたが、"
                "新しいトークンが追加されていません。"
            )

        def mask_original_token_gradients(gradient):
            """
            既存トークン行の勾配を0にする。
            追加トークン行の勾配だけを残す。
            """
            masked_gradient = gradient.clone()
            masked_gradient[:original_vocab_size].zero_()
            return masked_gradient

        # 入力Embeddingへ勾配マスクを登録
        gradient_hook_handles.append(
            input_weight.register_hook(
                mask_original_token_gradients
            )
        )

        # LM HeadがEmbeddingと独立している場合のみ別途登録
        if not tied_embeddings:
            gradient_hook_handles.append(
                output_weight.register_hook(
                    mask_original_token_gradients
                )
            )

        log0(
            "\nTraining mode: new token rows only\n"
            f"Frozen token rows:    0 to {original_vocab_size - 1}\n"
            f"Trainable token rows: {original_vocab_size} "
            f"to {expanded_vocab_size - 1}\n"
        )
    else:
        log0(
            "\nTraining mode: full Embedding and full LM Head\n"
        )

    print_trainable_parameters(model)

    if not any(
        parameter.requires_grad
        for parameter in model.parameters()
    ):
        raise RuntimeError(
            "学習可能なパラメータがありません。"
            "EmbeddingとLM Headの設定を確認してください。"
        )

    # =========================================================
    # 6. データセット読み込み
    # =========================================================
    def formatting_func(example):
        return add_eos_to_example(example, EOS_TOKEN)

    if args.train_file is not None:
        train_file = args.train_file
    else:
        train_file = (
            f"https://huggingface.co/datasets/"
            f"{args.dataset_path}/resolve/main/"
            f"{args.dataset_file}"
        )

    data_files = {
        "train": train_file,
    }

    if args.validation_file is not None:
        data_files["validation"] = args.validation_file

    if args.test_file is not None:
        data_files["test"] = args.test_file

    log0(f"\nLoading dataset files: {data_files}")

    dataset_dict = load_dataset(
        "csv",
        data_files=data_files,
    )

    dataset = dataset_dict["train"]

    if "text" not in dataset.column_names:
        raise KeyError(
            "データセットに'text'列がありません。"
            f"利用可能な列: {dataset.column_names}"
        )

    if len(dataset) == 0:
        raise RuntimeError("学習データセットが空です。")

    log0(f"Dataset size: {len(dataset):,}")
    log0(f"Dataset columns: {dataset.column_names}")
    log0(f"Dataset features: {dataset.features}")
    log0(f"Text type: {type(dataset[0]['text'])}")
    log0(f"Dataset example:\n{dataset[0]['text']}\n")

    # =========================================================
    # 7. SFTConfigの作成
    # =========================================================
    available_cpu_count = os.cpu_count() or 1

    dataset_num_proc = max(
        1,
        min(args.dataset_num_proc, available_cpu_count),
    )

    sft_config_kwargs = {
        "dataset_text_field": "text",
        "dataset_num_proc": dataset_num_proc,
        "packing": False,
        "per_device_train_batch_size": (
            args.per_device_train_batch_size
        ),
        "gradient_accumulation_steps": (
            args.gradient_accumulation_steps
        ),
        "num_train_epochs": args.num_train_epochs,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "max_grad_norm": 1.0,
        "learning_rate": args.learning_rate,
        "logging_strategy": "steps",
        "logging_steps": args.logging_steps,
        "save_strategy": "steps",
        "save_steps": args.save_steps,
        "max_steps": args.max_steps,
        "save_total_limit": 1,
        "logging_first_step": True,
        "optim": (
            "paged_adamw_8bit"
            if args.load_in_4bit
            else "adamw_torch"
        ),
        "lr_scheduler_type": "cosine",
        "seed": args.random_seed,
        "data_seed": args.random_seed,
        "output_dir": args.log_path,
        "fp16": (
            torch.cuda.is_available()
            and not is_bfloat16_supported()
        ),
        "bf16": is_bfloat16_supported(),
        "gradient_checkpointing": False,
        "report_to": "wandb",
        "run_name": args.run_name,

        # EmbeddingとLM Headは常にforwardで使用されるためFalse
        "ddp_find_unused_parameters": False,
    }

    # TRLのバージョンによる引数名の違いへ対応
    sft_config_parameters = inspect.signature(
        SFTConfig.__init__
    ).parameters

    if "max_length" in sft_config_parameters:
        sft_config_kwargs["max_length"] = args.max_seq_length
    elif "max_seq_length" in sft_config_parameters:
        sft_config_kwargs["max_seq_length"] = args.max_seq_length
    else:
        log0(
            "[WARNING] SFTConfigにmax_lengthまたは"
            "max_seq_lengthがありません。"
        )

    sft_config = SFTConfig(**sft_config_kwargs)

    # =========================================================
    # 8. SFTTrainerの作成
    # =========================================================
    trainer_kwargs = {
        "model": model,
        "train_dataset": dataset,
        "formatting_func": formatting_func,
        "args": sft_config,
    }

    # TRLの新旧バージョンへ対応
    trainer_parameters = inspect.signature(
        SFTTrainer.__init__
    ).parameters

    if "processing_class" in trainer_parameters:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_parameters:
        trainer_kwargs["tokenizer"] = tokenizer
    else:
        raise RuntimeError(
            "SFTTrainerにprocessing_classまたは"
            "tokenizer引数がありません。"
        )

    trainer = SFTTrainer(**trainer_kwargs)

    # =========================================================
    # 9. 学習開始前のGPUメモリ情報
    # =========================================================
    start_gpu_memory = 0.0
    max_gpu_memory = 0.0

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(LOCAL_RANK)

        gpu_stats = torch.cuda.get_device_properties(LOCAL_RANK)

        start_gpu_memory = round(
            torch.cuda.memory_reserved(LOCAL_RANK) / 1024**3,
            3,
        )

        max_gpu_memory = round(
            gpu_stats.total_memory / 1024**3,
            3,
        )

        log0(
            f"\nGPU: {gpu_stats.name}\n"
            f"World size: {WORLD_SIZE}\n"
            f"Total GPU memory: {max_gpu_memory} GB\n"
            f"Initially reserved memory: "
            f"{start_gpu_memory} GB\n"
        )

    # =========================================================
    # 10. 学習
    # =========================================================
    trainer_stats = trainer.train()

    # =========================================================
    # 11. 学習終了後のGPUメモリ情報
    # =========================================================
    if torch.cuda.is_available():
        peak_reserved_memory = round(
            torch.cuda.max_memory_reserved(LOCAL_RANK) / 1024**3,
            3,
        )

        additional_training_memory = round(
            peak_reserved_memory - start_gpu_memory,
            3,
        )

        peak_memory_percentage = (
            round(
                peak_reserved_memory
                / max_gpu_memory
                * 100,
                3,
            )
            if max_gpu_memory > 0
            else 0.0
        )

        additional_memory_percentage = (
            round(
                additional_training_memory
                / max_gpu_memory
                * 100,
                3,
            )
            if max_gpu_memory > 0
            else 0.0
        )

        training_runtime = trainer_stats.metrics.get(
            "train_runtime",
            0.0,
        )

        log0(
            f"\nTraining runtime: "
            f"{training_runtime} seconds\n"
            f"Training runtime: "
            f"{round(training_runtime / 60, 2)} minutes\n"
            f"Peak reserved memory: "
            f"{peak_reserved_memory} GB\n"
            f"Additional reserved memory for training: "
            f"{additional_training_memory} GB\n"
            f"Peak memory usage: "
            f"{peak_memory_percentage}%\n"
            f"Additional training memory usage: "
            f"{additional_memory_percentage}%\n"
        )

    # =========================================================
    # 12. モデルとTokenizerの保存
    # =========================================================
    if (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
    ):
        torch.distributed.barrier()

    if IS_MAIN_PROCESS:
        os.makedirs(args.output_path, exist_ok=True)

        # LoRAアダプターではなく通常モデル全体を保存
        trainer.save_model(args.output_path)
        tokenizer.save_pretrained(args.output_path)

        log0(f"\nModel saved to {args.output_path}")

    if (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
    ):
        torch.distributed.barrier()
