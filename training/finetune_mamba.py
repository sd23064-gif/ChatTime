import argparse
import os
import sys

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer
from peft import LoraConfig, PeftModel


def load_train_dataset(dataset_path):
    """
    dataset_path examples:
      - HF repo:
          ChengsenWang/ChatTime-1-Finetune-100K

      - HF csv:
          ChengsenWang/ChatTime-1-Finetune-100K::ChatTime-1-Finetune-100K.csv

      - local csv:
          /workspace/dataset/finetune.csv

      - local dir:
          /workspace/dataset/finetune/
    """
    from huggingface_hub import list_repo_files

    # local csv
    if dataset_path.endswith(".csv") and os.path.exists(dataset_path):
        return load_dataset("csv", data_files=dataset_path, split="train")

    # local directory
    if os.path.isdir(dataset_path):
        csv_files = []
        for root, dirs, files in os.walk(dataset_path):
            for f in files:
                if f.endswith(".csv"):
                    csv_files.append(os.path.join(root, f))

        if len(csv_files) > 0:
            print(f"Found local csv files: {csv_files[:5]}")
            return load_dataset("csv", data_files=csv_files, split="train")

        return load_dataset(dataset_path, split="train")

    # HF repo with explicit csv file:
    # example: repo_id::file.csv
    if "::" in dataset_path:
        repo_id, csv_path = dataset_path.split("::", 1)
        print(f"Loading HF csv: repo={repo_id}, file={csv_path}")

        return load_dataset(
            "csv",
            data_files=f"hf://datasets/{repo_id}/{csv_path}",
            split="train",
        )

    # Try normal HF dataset first
    try:
        dataset = load_dataset(dataset_path, split="train")
        if "text" in dataset.column_names:
            return dataset
        else:
            print(
                f"Normal load_dataset succeeded, but no text column. "
                f"Columns: {dataset.column_names}"
            )
    except Exception as e:
        print(f"Normal load_dataset failed: {e}")

    # If HF dataset is auto-detected incorrectly, find csv
    print(f"Searching CSV files in HF dataset repo: {dataset_path}")

    files = list_repo_files(
        repo_id=dataset_path,
        repo_type="dataset",
    )

    csv_files = [f for f in files if f.endswith(".csv")]

    if len(csv_files) == 0:
        raise ValueError(
            f"No csv file found in HF repo: {dataset_path}. "
            f"Files found: {files[:20]}"
        )

    csv_path = csv_files[0]
    print(f"Using HF csv file: {csv_path}")

    return load_dataset(
        "csv",
        data_files=f"hf://datasets/{dataset_path}/{csv_path}",
        split="train",
    )


def is_peft_adapter_dir(path):
    """
    PEFT adapterとして保存されているかを判定する。
    """
    return os.path.exists(os.path.join(path, "adapter_config.json"))


def main():
    parser = argparse.ArgumentParser()

    # paths
    parser.add_argument("--code_path", type=str, required=True)
    parser.add_argument("--base_model_path", type=str, default=None)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--log_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)

    # training config
    parser.add_argument("--max_seq_length", type=int, default=2048)

    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.00)

    parser.add_argument(
        "--target_modules",
        type=str,
        default="x_proj,in_proj,out_proj",
        help="Comma-separated LoRA target modules for Mamba.",
    )

    parser.add_argument("--random_seed", type=int, default=3407)

    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--max_steps", type=int, default=-1)

    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)

    # precision
    parser.add_argument("--use_bf16", action="store_true", default=False)

    # saving
    parser.add_argument("--merge_lora", action="store_true", default=False)

    # wandb
    parser.add_argument("--wandb_project", type=str, default="chattime-mamba")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--disable_wandb", action="store_true", default=False)

    args = parser.parse_args()

    sys.path.append(args.code_path)

    os.makedirs(args.log_path, exist_ok=True)
    os.makedirs(args.output_path, exist_ok=True)

    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    # =========================
    # wandb
    # =========================
    if not args.disable_wandb:
        import wandb

        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
        )

    # =========================
    # detect model type
    # =========================
    model_path = args.model_path
    is_adapter = is_peft_adapter_dir(model_path)

    print("\nModel loading mode")
    print(f"model_path: {model_path}")
    print(f"is_peft_adapter: {is_adapter}")

    # =========================
    # tokenizer
    # =========================
    print("\nLoading tokenizer...")

    if is_adapter:
        # adapter側に保存されたtokenizerを使う
        tokenizer_path = model_path
    else:
        # merge済みモデルまたは通常モデル
        tokenizer_path = model_path

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    eos_token = tokenizer.eos_token
    if eos_token is None:
        eos_token = ""

    print(f"Tokenizer vocab size: {len(tokenizer)}")

    # =========================
    # model
    # =========================
    if args.use_bf16:
        dtype = torch.bfloat16
    else:
        # Mamba + PEFTではfloat32が安全
        dtype = torch.float32

    if is_adapter:
        if args.base_model_path is None:
            raise ValueError(
                "PEFT adapterをfinetuneする場合は "
                "--base_model_path を指定してください。"
            )

        print(f"\nLoading base model from: {args.base_model_path}")

        model = AutoModelForCausalLM.from_pretrained(
            args.base_model_path,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map="auto",
        )

        print("Resizing token embeddings to match adapter tokenizer...")
        model.resize_token_embeddings(len(tokenizer))

        if hasattr(model.config, "pad_token_id"):
            model.config.pad_token_id = tokenizer.pad_token_id

        print(f"Loading PEFT adapter from: {model_path}")

        model = PeftModel.from_pretrained(
            model,
            model_path,
            is_trainable=True,
        )

        # 既存adapterを継続学習するのでSFTTrainerにはpeft_configを渡さない
        peft_config = None

    else:
        print(f"\nLoading full model from: {model_path}")

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map="auto",
        )

        # 念のためtokenizerサイズとモデルembeddingを合わせる
        current_vocab = model.get_input_embeddings().weight.shape[0]
        if current_vocab != len(tokenizer):
            print(
                f"Resizing token embeddings: "
                f"{current_vocab} -> {len(tokenizer)}"
            )
            model.resize_token_embeddings(len(tokenizer))

        if hasattr(model.config, "pad_token_id"):
            model.config.pad_token_id = tokenizer.pad_token_id

        # full modelに新しくLoRAを付けてfinetuneする
        target_modules = [
            x.strip()
            for x in args.target_modules.split(",")
            if x.strip()
        ]

        print("\nLoRA target modules:")
        print(target_modules)

        peft_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            task_type="CAUSAL_LM",
            bias="none",
        )

    # =========================
    # dataset
    # =========================
    print(f"\nLoading dataset from: {args.dataset_path}")

    dataset = load_train_dataset(args.dataset_path)

    if "text" not in dataset.column_names:
        raise ValueError(
            f"Dataset must contain a 'text' column. "
            f"Current columns: {dataset.column_names}"
        )

    print(f"Dataset columns: {dataset.column_names}")
    print(f"Dataset size: {len(dataset)}")
    print("\nDataset example:")
    print(dataset[0]["text"][:2000])

    def formatting_func(example):
        return example["text"] + eos_token

    # =========================
    # training args
    # =========================
    training_args = TrainingArguments(
        output_dir=args.log_path,

        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,

        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,

        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=1.0,

        logging_strategy="steps",
        logging_steps=args.logging_steps,
        logging_first_step=True,

        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=1,

        lr_scheduler_type="cosine",
        seed=args.random_seed,

        # fp16は "Attempting to unscale FP16 gradients" が出やすいため無効
        fp16=False,
        bf16=args.use_bf16,

        optim="adamw_torch",

        report_to=[] if args.disable_wandb else ["wandb"],
        run_name=args.wandb_run_name,
    )

    # =========================
    # trainer
    # =========================
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        packing=False,
        formatting_func=formatting_func,
        peft_config=peft_config,
        args=training_args,
    )

    # =========================
    # memory before training
    # =========================
    if torch.cuda.is_available():
        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(
            torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024,
            3,
        )
        max_memory = round(
            gpu_stats.total_memory / 1024 / 1024 / 1024,
            3,
        )

        print(f"\nGPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
        print(f"{start_gpu_memory} GB of memory reserved.\n")

    # =========================
    # train
    # =========================
    trainer_stats = trainer.train()

    # =========================
    # memory after training
    # =========================
    if torch.cuda.is_available():
        used_memory = round(
            torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024,
            3,
        )
        used_memory_for_training = round(
            used_memory - start_gpu_memory,
            3,
        )

        print(f"\n{trainer_stats.metrics['train_runtime']} seconds used for training.")
        print(f"{round(trainer_stats.metrics['train_runtime'] / 60, 2)} minutes used for training.")
        print(f"Peak reserved memory = {used_memory} GB.")
        print(f"Peak reserved memory for training = {used_memory_for_training} GB.\n")

        if not args.disable_wandb:
            import wandb
            wandb.log({
                "gpu/peak_reserved_gb": used_memory,
                "gpu/training_reserved_gb": used_memory_for_training,
            })

    # =========================
    # save
    # =========================
    print(f"\nSaving model to: {args.output_path}")

    if args.merge_lora:
        try:
            merged_model = trainer.model.merge_and_unload()
            merged_model.save_pretrained(args.output_path)
            tokenizer.save_pretrained(args.output_path)
            print("Merged model saved.")
        except Exception as e:
            print(f"Could not merge LoRA. Saving adapter instead. Error: {e}")
            trainer.model.save_pretrained(args.output_path)
            tokenizer.save_pretrained(args.output_path)
    else:
        trainer.model.save_pretrained(args.output_path)
        tokenizer.save_pretrained(args.output_path)
        print("Adapter/tokenizer saved.")

    if not args.disable_wandb:
        import wandb
        wandb.finish()

    print("\nDone.")


if __name__ == "__main__":
    main()
