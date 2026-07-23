import os

# Unsloth / torch.compile の自動コンパイルを抑制
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
os.environ["UNSLOTH_COMPILE_IGNORE_ERRORS"] = "1"
os.environ["TORCH_COMPILE"] = "0"
os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "0"
os.environ["TORCHINDUCTOR_AUTOGRAD_CACHE"] = "0"

import argparse
import math
import sys

import numpy as np

# NumPy 2.x 対策
if not hasattr(np, "NaN"):
    np.NaN = np.nan

import torch
from datasets import load_dataset
from unsloth import FastLanguageModel, is_bfloat16_supported
from transformers import AutoTokenizer, EarlyStoppingCallback, TrainerCallback
from trl import SFTTrainer, SFTConfig


class LossOnlySFTTrainer(SFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)

        if isinstance(outputs, dict):
            loss = outputs["loss"]
        elif hasattr(outputs, "loss"):
            loss = outputs.loss
        else:
            loss = outputs[0]

        if self.state.global_step % max(1, self.args.logging_steps) == 0:
            self.log({
                "lm_loss_raw": loss.detach().float().item(),
            })

        return (loss, outputs) if return_outputs else loss

class SaveTokenizerAtCheckpointCallback(TrainerCallback):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def on_save(self, args, state, control, **kwargs):
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.tokenizer.save_pretrained(checkpoint_dir)
        print(f"Saved tokenizer to: {checkpoint_dir}")
        return control

def verify_added_tokens(tokenizer, added_tokens):
    indices = np.linspace(0, len(added_tokens) - 1, 5, dtype=int)
    print("\nAdded-token verification")

    for index in indices:
        token = added_tokens[index]
        token_id = tokenizer.convert_tokens_to_ids(token)
        encoded = tokenizer.encode(token, add_special_tokens=False)

        print({"token": token, "token_id": token_id, "encoded": encoded})

        if len(encoded) != 1 or encoded[0] != token_id:
            raise ValueError(f"Added token mismatch: {token} -> {encoded}")

def inspect_trainability(model):
    input_layer = model.get_input_embeddings()
    output_layer = model.get_output_embeddings()

    print("\nEmbedding trainability")
    print("Input shape:", tuple(input_layer.weight.shape))
    print("Input trainable:", input_layer.weight.requires_grad)

    if output_layer is not None and hasattr(output_layer, "weight"):
        print("Output shape:", tuple(output_layer.weight.shape))
        print("Output trainable:", output_layer.weight.requires_grad)
        print("Weights tied:", input_layer.weight.data_ptr() == output_layer.weight.data_ptr())

    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())

    print(f"Trainable parameters: {trainable:,}")
    print(f"Total parameters: {total:,}")
    print(f"Trainable ratio: {100.0 * trainable / total:.4f}%")

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--code_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--log_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--use_fast", action=argparse.BooleanOptionalAction, default=False)

    # B-1用: local CSV split
    parser.add_argument("--train_file", type=str, default=None)
    parser.add_argument("--validation_file", type=str, default=None)
    parser.add_argument("--test_file", type=str, default=None)

    # 旧互換: trainだけ指定したい場合
    parser.add_argument("--dataset_file", type=str, default=None)

    parser.add_argument("--do_eval", action="store_true", default=False)
    parser.add_argument("--do_test", action="store_true", default=False)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--early_stopping_patience", type=int, default=3)

    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)

    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--random_seed", type=int, default=3407)

    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=32)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=-1)

    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=0)

    parser.add_argument("--dataset_num_proc", type=int, default=1)

    # ChatTime numerical token settings
    parser.add_argument("--low_limit", type=float, default=-1)
    parser.add_argument("--high_limit", type=float, default=1)
    parser.add_argument("--n_tokens", type=int, default=10002)
    parser.add_argument("--prec", type=int, default=4)
    parser.add_argument("--time_sep", type=str, default=" ")
    parser.add_argument("--time_flag", type=str, default="###")
    parser.add_argument("--nan_flag", type=str, default="Nan")


    # W&B
    parser.add_argument("--wandb_project", type=str, default="chattime-pretrain")
    parser.add_argument("--wandb_run_name", type=str, default=None)

    # save
    parser.add_argument("--save_merged", action="store_true", default=False)

    args = parser.parse_args()

    os.makedirs(args.log_path, exist_ok=True)
    os.makedirs(args.output_path, exist_ok=True)

    sys.path.append(args.code_path)
    from utils.tools import Discretizer, Serializer

    if args.wandb_run_name is None:
        model_name = args.model_path.rstrip("/").split("/")[-1]
        args.wandb_run_name = (
            f"ptbxl-b1-{model_name}"
            f"-bs{args.per_device_train_batch_size}"
            f"-ga{args.gradient_accumulation_steps}"
            f"-lr{args.learning_rate}"
        )

    os.environ["WANDB_PROJECT"] = args.wandb_project
    os.environ["WANDB_NAME"] = args.wandb_run_name

    # -------------------------
    # Construct ChatTime vocab
    # -------------------------
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

    vocabulary = np.concatenate(
        (
            discretizer.centers[1:-1],
            [np.nan],
        )
    ).reshape(-1, 1)

    vocabulary = np.array([serializer.serialize(i) for i in vocabulary])

    print(f"\nVocabulary sample:\n{vocabulary[:10]}")
    print(f"Vocabulary size to add: {len(vocabulary)}\n")

    # -------------------------
    # Tokenizer
    # -------------------------
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        use_fast=args.use_fast,
    )

    print(f"Requested use_fast: {args.use_fast}")
    print(f"Loaded tokenizer class: {tokenizer.__class__.__name__}")
    print(f"Loaded tokenizer is_fast: {getattr(tokenizer, 'is_fast', False)}")


    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    old_vocab_size = len(tokenizer)
    print(f"Old tokenizer size: {old_vocab_size}")

    num_added_tokens = tokenizer.add_tokens(vocabulary.tolist())

    print(f"Added tokens: {num_added_tokens}")
    print(f"New tokenizer size: {len(tokenizer)}")

    EOS_TOKEN = tokenizer.eos_token

    # -------------------------
    # Model
    # -------------------------
    model, _ = FastLanguageModel.from_pretrained(
        model_name=args.model_path,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=args.load_in_4bit,
        resize_model_vocab=len(tokenizer),
    )

    # -------------------------
    # LoRA
    # -------------------------
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        # ChatTime pretrainingでは数値tokenを追加するため、
        # embed_tokens と lm_head も保存対象にする
        modules_to_save=[
            "embed_tokens",
            "lm_head",
        ],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.random_seed,
        max_seq_length=args.max_seq_length,
        temporary_location="/home/chattime/.cache/unsloth_buffers",
    )
    inspect_trainability(model)
    # -------------------------
    # Dataset
    # -------------------------
    data_files = {}

    if args.train_file is not None:
        data_files["train"] = args.train_file
    elif args.dataset_file is not None:
        data_files["train"] = args.dataset_file
    else:
        if args.dataset_path is None:
            raise ValueError(
                "Either --train_file, --dataset_file, or --dataset_path must be specified."
            )

        if args.dataset_path == "dummy":
            raise ValueError(
                "--dataset_path dummy was provided, but no --train_file or --dataset_file was given."
            )

        data_files["train"] = (
            f"https://huggingface.co/datasets/"
            f"{args.dataset_path}/resolve/main/ChatTime-1-Pretrain-1M.csv"
        )

    if args.validation_file is not None:
        data_files["validation"] = args.validation_file

    if args.test_file is not None:
        data_files["test"] = args.test_file

    print("\nLoading dataset:")
    print(data_files)

    dataset_dict = load_dataset(
        "csv",
        data_files=data_files,
    )

    train_dataset = dataset_dict["train"]
    eval_dataset = dataset_dict["validation"] if "validation" in dataset_dict else None
    test_dataset = dataset_dict["test"] if "test" in dataset_dict else None

    print(f"\nTrain dataset size: {len(train_dataset)}")
    print(f"Train example:\n{train_dataset[0]['text'][:1000]}\n")

    if eval_dataset is not None:
        print(f"Validation dataset size: {len(eval_dataset)}")

    if test_dataset is not None:
        print(f"Test dataset size: {len(test_dataset)}")

    def add_eos(example):
        text = example["text"]
        if not text.endswith(EOS_TOKEN):
            text = text + EOS_TOKEN
        return {"text": text}

    train_dataset = train_dataset.map(add_eos)

    if eval_dataset is not None:
        eval_dataset = eval_dataset.map(add_eos)

    if test_dataset is not None:
        test_dataset = test_dataset.map(add_eos)

    # -------------------------
    # Training config
    # -------------------------
    training_args = SFTConfig(
        output_dir=args.log_path,

        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,

        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_grad_norm=1.0,

        logging_strategy="steps",
        logging_steps=args.logging_steps,
        logging_first_step=True,

        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=None,

        eval_strategy=(
            "steps"
            if args.do_eval and eval_dataset is not None
            else "no"
        ),
        eval_steps=args.eval_steps,
        load_best_model_at_end=(
            True
            if args.do_eval and eval_dataset is not None
            else False
        ),
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        optim="adamw_8bit",
        lr_scheduler_type="cosine",
        seed=args.random_seed,

        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),

        dataset_text_field="text",
        max_length=args.max_seq_length,
        packing=False,
        dataset_num_proc=args.dataset_num_proc,

        torch_compile=False,

        report_to="wandb",
        run_name=args.wandb_run_name,
    )

    trainer = LossOnlySFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )
    trainer.add_callback(SaveTokenizerAtCheckpointCallback(tokenizer))


    if args.do_eval and eval_dataset is not None:
        trainer.add_callback(
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience
            )
        )

    # -------------------------
    # Memory stats
    # -------------------------
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

    # -------------------------
    # Train
    # -------------------------
    trainer_stats = trainer.train()

    # -------------------------
    # Final validation
    # -------------------------
    if args.do_eval and eval_dataset is not None:
        print("\nRunning final validation evaluation...")

        # raw eval_dataset は渡さない
        # trainer内部で前処理済みの eval_dataset を使う
        eval_metrics = trainer.evaluate(
            metric_key_prefix="eval",
        )

        if "eval_loss" in eval_metrics:
            try:
                eval_metrics["eval_perplexity"] = math.exp(eval_metrics["eval_loss"])
            except OverflowError:
                eval_metrics["eval_perplexity"] = float("inf")

        print(eval_metrics)
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)


    # -------------------------
    # Final test
    # -------------------------
    if args.do_test and test_dataset is not None:
        print("\nPreparing test dataset and running final test evaluation...")

        # test_dataset は trainer 初期化時に前処理させる
        test_trainer = LossOnlySFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset.select(range(min(1, len(train_dataset)))),
            eval_dataset=test_dataset,
            processing_class=tokenizer,
        )

        test_metrics = test_trainer.evaluate(
            metric_key_prefix="test",
        )

        if "test_loss" in test_metrics:
            try:
                test_metrics["test_perplexity"] = math.exp(test_metrics["test_loss"])
            except OverflowError:
                test_metrics["test_perplexity"] = float("inf")

        print(test_metrics)
        trainer.log_metrics("test", test_metrics)
        trainer.save_metrics("test", test_metrics)


    # -------------------------
    # Final memory stats
    # -------------------------
    used_memory = round(
        torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024,
        3,
    )
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)

    print(f"\n{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(f"{round(trainer_stats.metrics['train_runtime'] / 60, 2)} minutes used for training.")
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.\n")

    # -------------------------
    # Save
    # -------------------------
    print(f"Saving LoRA adapter to {args.output_path}")
    model.save_pretrained(args.output_path)
    tokenizer.save_pretrained(args.output_path)

    if args.save_merged:
        print(f"Saving merged model to {args.output_path}_merged")
        model.save_pretrained_merged(
            args.output_path + "_merged",
            tokenizer,
        )


if __name__ == "__main__":
    main()