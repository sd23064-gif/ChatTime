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
from peft import LoraConfig


def load_train_dataset(dataset_path):
    """
    dataset_path が Hugging Face dataset / local dataset / csv のどれでもある程度読めるようにする。
    """
    
    if "::" in dataset_path:
        repo_id, csv_path = dataset_path.split("::", 1)

        print(f"Loading HF csv: repo={repo_id}, file={csv_path}")

        return load_dataset(
            "csv",
            data_files=f"hf://datasets/{repo_id}/{csv_path}",
            split="train"
        )


    if os.path.isdir(dataset_path):
        csv_files = []
        for root, dirs, files in os.walk(dataset_path):
            for f in files:
                if f.endswith(".csv"):
                    csv_files.append(os.path.join(root, f))

        if len(csv_files) > 0:
            print(f"Found csv files: {csv_files[:5]}")
            return load_dataset("csv", data_files=csv_files, split="train")

    return load_dataset(dataset_path, split="train")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--code_path", type=str, required=True, default=None)
    parser.add_argument("--model_path", type=str, required=True, default=None)
    parser.add_argument("--dataset_path", type=str, required=True, default=None)
    parser.add_argument("--log_path", type=str, required=True, default=None)
    parser.add_argument("--output_path", type=str, required=True, default=None)

    parser.add_argument("--max_seq_length", type=int, default=2048)

    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.00)

    parser.add_argument(
        "--target_modules",
        type=str,
        default="x_proj,embeddings,in_proj,out_proj",
        help="Comma-separated LoRA target modules for Mamba."
    )

    parser.add_argument("--random_seed", type=int, default=3407)

    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--max_steps", type=int, default=-1)

    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)

    parser.add_argument("--low_limit", type=float, default=-1)
    parser.add_argument("--high_limit", type=float, default=1)
    parser.add_argument("--n_tokens", type=int, default=10002)
    parser.add_argument("--prec", type=int, default=4)
    parser.add_argument("--time_sep", type=str, default=" ")
    parser.add_argument("--time_flag", type=str, default="###")
    parser.add_argument("--nan_flag", type=str, default="Nan")

    parser.add_argument("--use_bf16", action="store_true", default=False)
    parser.add_argument("--use_fp16", action="store_true", default=False)
    parser.add_argument("--merge_lora", action="store_true", default=False)

    
    parser.add_argument("--wandb_project", type=str, default="chattime-mamba")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--disable_wandb", action="store_true", default=False)

    args = parser.parse_args()

    if not args.disable_wandb:
        import wandb

        wandb_kwargs = {
            "project": args.wandb_project,
            "name": args.wandb_run_name,
            "config": vars(args),
        }

        if args.wandb_entity is not None:
            wandb_kwargs["entity"] = args.wandb_entity

        wandb.init(**wandb_kwargs)

    sys.path.append(args.code_path)
    from utils.tools import Discretizer, Serializer

    os.makedirs(args.log_path, exist_ok=True)
    os.makedirs(args.output_path, exist_ok=True)

    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    # ============================================================
    # 1. Construct ChatTime time-series vocabulary
    # ============================================================
    print("\nConstructing ChatTime time-series vocabulary...")

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

    # np.NaN ではなく np.nan を使う
    vocabulary_values = np.concatenate(
        (discretizer.centers[1:-1], [np.nan])
    ).reshape(-1, 1)

    vocabulary = np.array([
        serializer.serialize(i)
        for i in vocabulary_values
    ])

    print(f"Number of time-series tokens: {len(vocabulary)}")
    print("Vocabulary examples:")
    print(vocabulary[:10])
    print(vocabulary[-5:])

    # ============================================================
    # 2. Load tokenizer and add ChatTime tokens
    # ============================================================
    print(f"\nLoading tokenizer: {args.model_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    old_vocab_size = len(tokenizer)
    print(f"Old tokenizer size: {old_vocab_size}")

    num_added = tokenizer.add_tokens(vocabulary.tolist())
    new_vocab_size = len(tokenizer)

    print(f"Added tokens: {num_added}")
    print(f"New tokenizer size: {new_vocab_size}")

    EOS_TOKEN = tokenizer.eos_token
    if EOS_TOKEN is None:
        EOS_TOKEN = ""

    # ============================================================
    # 3. Load Mamba model
    # ============================================================
    print(f"\nLoading Mamba model: {args.model_path}")

    # MambaのPEFT例ではfloat32推奨なので、初回はfloat32が安全
    if args.use_bf16:
        dtype = torch.bfloat16
    elif args.use_fp16:
        dtype = torch.float16
    else:
        dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto",
    )

    # tokenizerを拡張したのでembeddingも拡張
    model.resize_token_embeddings(new_vocab_size)

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    # pad_token_idを設定
    if hasattr(model.config, "pad_token_id"):
        model.config.pad_token_id = tokenizer.pad_token_id

    # ============================================================
    # 4. PEFT / LoRA config for Mamba
    # ============================================================
    target_modules = [
        x.strip()
        for x in args.target_modules.split(",")
        if x.strip() != ""
    ]

    print("\nLoRA target modules:")
    print(target_modules)

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        task_type="CAUSAL_LM",
        bias="none",
    )

    # ============================================================
    # 5. Load dataset
    # ============================================================
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
        return example["text"] + EOS_TOKEN

    # ============================================================
    # 6. Training
    # ============================================================
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

        # Mambaは最初float32が安全。必要なら --use_bf16 / --use_fp16 を指定。
        fp16=args.use_fp16,
        bf16=args.use_bf16,

        # bitsandbytes不要でまず安定動作を優先
        optim="adamw_torch",
        gradient_checkpointing=True,

        
        report_to=[] if args.disable_wandb else ["wandb"],
        run_name=args.wandb_run_name,

    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        packing=False,
        formatting_func=formatting_func,
        peft_config=lora_config,
        args=training_args,
    )

    # ============================================================
    # 7. Memory info
    # ============================================================
    if torch.cuda.is_available():
        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(
            torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024,
            3
        )
        max_memory = round(
            gpu_stats.total_memory / 1024 / 1024 / 1024,
            3
        )

        print(f"\nGPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
        print(f"{start_gpu_memory} GB of memory reserved.\n")

    # ============================================================
    # 8. Train
    # ============================================================
    trainer_stats = trainer.train()

    if torch.cuda.is_available():
        used_memory = round(
            torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024,
            3
        )
        used_memory_for_training = round(used_memory - start_gpu_memory, 3)
        used_percentage = round(used_memory / max_memory * 100, 3)
        train_percentage = round(
            used_memory_for_training / max_memory * 100,
            3
        )

        print(f"\n{trainer_stats.metrics['train_runtime']} seconds used for training.")
        print(f"{round(trainer_stats.metrics['train_runtime'] / 60, 2)} minutes used for training.")
        print(f"Peak reserved memory = {used_memory} GB.")
        print(f"Peak reserved memory for training = {used_memory_for_training} GB.")
        print(f"Peak reserved memory % of max memory = {used_percentage} %.")
        print(f"Peak reserved memory for training % of max memory = {train_percentage} %.\n")

    # ============================================================
    # 9. Save
    # ============================================================
    print(f"\nSaving model to: {args.output_path}")

    if args.merge_lora:
        try:
            merged_model = trainer.model.merge_and_unload()
            merged_model.save_pretrained(args.output_path)
            tokenizer.save_pretrained(args.output_path)
            print("Merged LoRA model saved.")
        except Exception as e:
            print(f"Could not merge LoRA. Saving PEFT adapter instead. Error: {e}")
            trainer.model.save_pretrained(args.output_path)
            tokenizer.save_pretrained(args.output_path)
    else:
        trainer.model.save_pretrained(args.output_path)
        tokenizer.save_pretrained(args.output_path)
        print("PEFT adapter and tokenizer saved.")

    if not args.disable_wandb:
        import wandb
        wandb.finish()
    print("\nDone.")