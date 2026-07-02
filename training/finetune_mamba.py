
import argparse
import os
import sys

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, EarlyStoppingCallback
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model, PeftModel



def is_bfloat16_supported():
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


def load_tokenizer(tokenizer_path, base_model_path):
    """
    先ほどの事前学習で tokenizer を保存している場合は tokenizer_path から読む。
    なければ base_model_path から読む。
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
        )
        print(f"Loaded tokenizer from: {tokenizer_path}")
    except Exception as e:
        print(f"Could not load tokenizer from {tokenizer_path}: {e}")
        print(f"Fallback: loading tokenizer from base model: {base_model_path}")
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=True,
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"
    return tokenizer


def print_trainable_parameters(model):
    trainable = 0
    total = 0

    for _, param in model.named_parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()

    ratio = 100 * trainable / total if total > 0 else 0
    print(
        f"Trainable params: {trainable:,} || "
        f"Total params: {total:,} || "
        f"Trainable: {ratio:.4f}%"
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--code_path", type=str, required=True, default=None)

    parser.add_argument("--base_model_path", type=str, required=True, default=None)

    # 先ほど保存したモデル/adapterのpath
    parser.add_argument("--model_path", type=str, required=True, default=None)

    parser.add_argument("--dataset_path", type=str, required=True, default=None)
    parser.add_argument("--log_path", type=str, required=True, default=None)
    parser.add_argument("--output_path", type=str, required=True, default=None)

    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)

    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--random_seed", type=int, default=3407)

    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=64)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=64)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--save_steps", type=int, default=2)
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--logging_steps", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=-1)

    
    parser.add_argument("--wandb_run_name", type=str, default=None)

    args = parser.parse_args()

    sys.path.append(args.code_path)

    if args.wandb_run_name is None:
        args.wandb_run_name = (
            f"finetune-mamba-370m"
            f"-bs{args.per_device_train_batch_size}"
            f"-ga{args.gradient_accumulation_steps}"
        )

    # load tokenizer

    tokenizer = load_tokenizer(
        tokenizer_path=args.model_path,
        base_model_path=args.base_model_path,
    )

    print(f"\nVocabulary number: {len(tokenizer.get_vocab())}\n")

    EOS_TOKEN = tokenizer.eos_token

    # load model
    
    
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )


    if hasattr(model, "generation_config"):
        model.generation_config.use_cache = False

    model.resize_token_embeddings(len(tokenizer))

    model.config.use_cache = False

    if hasattr(model, "generation_config"):
        model.generation_config.use_cache = False

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    adapter_config_path = os.path.join(args.model_path, "adapter_config.json")
    # add lora to llama model

    print(f"Loading previous LoRA adapter from: {args.model_path}")
    model = PeftModel.from_pretrained(
        model,
        args.model_path,
        is_trainable=True,
    )
    
    model.config.use_cache = False

    if hasattr(model, "generation_config"):
        model.generation_config.use_cache = False

    if hasattr(model, "base_model") and hasattr(model.base_model, "config"):
        model.base_model.config.use_cache = False

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()


    print_trainable_parameters(model)

    # load dataset
    def formatting_func(example):
        return example["text"] + EOS_TOKEN


    print(f"\nLoading dataset in {args.dataset_path}")
    dataset = load_dataset(
            "csv",
            data_files=f"https://huggingface.co/datasets/{args.dataset_path}/resolve/main/ChatTime-1-Finetune-100K.csv",
            split="train",
        )
    print(f"Dataset example: \n{dataset[0]['text']}\n")
    
    dataset = dataset.train_test_split(
        test_size=0.02,
        seed=args.random_seed,
    )

    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    # train model
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        dataset_num_proc=64,
        packing=False,    
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=3,
                early_stopping_threshold=0.0,
            )
        ],

        args=TrainingArguments(
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            num_train_epochs=args.num_train_epochs,
            weight_decay=0.01,
            warmup_ratio=0.05,
            max_grad_norm=1.0,
            learning_rate=5e-5,
            logging_strategy="steps",
            logging_steps=args.logging_steps,
            save_strategy="steps",
            save_steps=args.save_steps,
            evaluation_strategy="steps",
            eval_steps=args.eval_steps,
            max_steps=args.max_steps,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            load_best_model_at_end=True,
            save_total_limit=3,
            logging_first_step=True,
            optim="adamw_8bit",
            lr_scheduler_type="cosine",
            seed=args.random_seed,
            output_dir=args.log_path,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            
            
            prediction_loss_only=True,
            remove_unused_columns=True,
            
            report_to=["wandb"],
            run_name=args.wandb_run_name,

        ),
    )

    # title Show current memory stats
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"\nGPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.\n")

    trainer_stats = trainer.train()
    
    eval_metrics = trainer.evaluate()
    print("Final eval metrics:", eval_metrics)


    # title Show final memory and time stats
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
    print(f"\n{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(f"{round(trainer_stats.metrics['train_runtime'] / 60, 2)} minutes used for training.")
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.\n")

    # save model and tokenizer
    print(f"Saving model to: {args.output_path}")

    print("Saving LoRA adapter only...")
    model.save_pretrained(args.output_path)
    tokenizer.save_pretrained(args.output_path)

    print("Save completed.")

