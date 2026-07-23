
import argparse
import os
import sys

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, EarlyStoppingCallback
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model, PeftModel, PeftConfig

import re

NUMERIC_TOKEN_RE = re.compile(
    r"###([+-]?(?:\d+(?:\.\d*)?|\.\d+))###"
)


def collect_numeric_token_ids(tokenizer):
    rows = []

    for token, token_id in tokenizer.get_vocab().items():
        match = NUMERIC_TOKEN_RE.fullmatch(str(token))

        if match is None:
            continue

        rows.append(
            (
                int(token_id),
                float(match.group(1)),
            )
        )

    rows.sort(key=lambda row: row[1])

    token_ids = torch.tensor(
        [row[0] for row in rows],
        dtype=torch.long,
    )

    token_values = torch.tensor(
        [row[1] for row in rows],
        dtype=torch.float32,
    )

    print("Numeric token count:", len(token_ids))

    return token_ids, token_values

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
def inspect_embedding_trainability(model):
    input_layer = model.get_input_embeddings()
    output_layer = model.get_output_embeddings()

    print("\nEmbedding trainability")

    print(
        "Input embedding shape:",
        tuple(input_layer.weight.shape),
    )
    print(
        "Input embedding requires_grad:",
        input_layer.weight.requires_grad,
    )

    if output_layer is not None and hasattr(output_layer, "weight"):
        print(
            "Output embedding shape:",
            tuple(output_layer.weight.shape),
        )
        print(
            "Output embedding requires_grad:",
            output_layer.weight.requires_grad,
        )

        print(
            "Input/output tied:",
            input_layer.weight.data_ptr()
            == output_layer.weight.data_ptr(),
        )

    print("\nTrainable embedding-related parameters")

    for name, parameter in model.named_parameters():
        lower_name = name.lower()

        if (
            "embedding" in lower_name
            or "embed" in lower_name
            or "lm_head" in lower_name
        ):
            print(
                name,
                tuple(parameter.shape),
                "requires_grad=",
                parameter.requires_grad,
            )
def verify_numeric_tokenization(tokenizer):
    check_tokens = [
        "###-0.9999###",
        "###-0.5001###",
        "###0.0001###",
        "###0.5001###",
        "###0.9999###",
    ]

    print("\nNumeric tokenization check")

    for token in check_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)

        encoded = tokenizer.encode(
            token,
            add_special_tokens=False,
        )

        decoded = (
            tokenizer.convert_ids_to_tokens(token_id)
            if token_id is not None
            else None
        )

        print(
            {
                "token": token,
                "token_id": token_id,
                "encoded": encoded,
                "length": len(encoded),
                "decoded": decoded,
            }
        )

        if len(encoded) != 1:
            raise ValueError(
                f"Numeric token is not encoded as one token: "
                f"{token} -> {encoded}"
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

    parser.add_argument("--lora_rank", type=int, default=8)
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

    verify_numeric_tokenization(tokenizer)

    print(f"\nVocabulary number: {len(tokenizer.get_vocab())}\n")

    EOS_TOKEN = tokenizer.eos_token

    # load model
    
    
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    peft_config = PeftConfig.from_pretrained(args.model_path)

    print("PEFT base model:", peft_config.base_model_name_or_path)
    print(
        "PEFT modules_to_save:",
        getattr(peft_config, "modules_to_save", None),
    )
    print(
        "PEFT target_modules:",
        getattr(peft_config, "target_modules", None),
    )
    if hasattr(model, "generation_config"):
        model.generation_config.use_cache = False
    base_vocab_size = model.get_input_embeddings().weight.shape[0]
    tokenizer_vocab_size = len(tokenizer)

    print("Base model vocab size:", base_vocab_size)
    print("Tokenizer vocab size:", tokenizer_vocab_size)

    if tokenizer_vocab_size < base_vocab_size:
        raise ValueError(
            "Tokenizer vocabulary is smaller than the base model vocabulary."
        )

    if tokenizer_vocab_size != base_vocab_size:
        print(
            "Resizing token embeddings:",
            base_vocab_size,
            "->",
            tokenizer_vocab_size,
        )

        model.resize_token_embeddings(tokenizer_vocab_size)

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
    inspect_embedding_trainability(model)
    model.config.use_cache = False

    if hasattr(model, "base_model"):
        if hasattr(model.base_model, "config"):
            model.base_model.config.use_cache = False

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    print("\nEmbedding / LM head module names")

    for name, module in model.named_modules():
        lower_name = name.lower()

        if (
            "embedding" in lower_name
            or "embed" in lower_name
            or "lm_head" in lower_name
        ):
            print(name, type(module))

    print_trainable_parameters(model)

    # load dataset

    print(f"\nLoading dataset in {args.dataset_path}")
    dataset = load_dataset(
            "csv",
            data_files=f"https://huggingface.co/datasets/{args.dataset_path}/resolve/main/ChatTime-1-Finetune-100K.csv",
            split="train",
        )
    print(f"Dataset example: \n{dataset[0]['text']}\n")
    def add_eos(example):
        text = example["text"]

        if EOS_TOKEN is not None and not text.endswith(EOS_TOKEN):
            text = text + EOS_TOKEN

        return {"text": text}


    dataset = dataset.map(
        add_eos,
        num_proc=8,
    )

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
    numeric_token_ids, numeric_token_values = (
        collect_numeric_token_ids(tokenizer)
    )

    embedding_weight = model.get_input_embeddings().weight

    numeric_ids_device = numeric_token_ids.to(
        embedding_weight.device
    )

    initial_numeric_embeddings = (
        embedding_weight[numeric_ids_device]
        .detach()
        .float()
        .cpu()
        .clone()
    )
    trainer_stats = trainer.train()
    
    eval_metrics = trainer.evaluate()
    print("Final eval metrics:", eval_metrics)
    trained_numeric_embeddings = (
        model.get_input_embeddings()
        .weight[numeric_ids_device]
        .detach()
        .float()
        .cpu()
    )

    embedding_delta = (
        trained_numeric_embeddings
        - initial_numeric_embeddings
    )

    l2_change = embedding_delta.norm(dim=-1)

    initial_direction = torch.nn.functional.normalize(
        initial_numeric_embeddings,
        dim=-1,
    )

    trained_direction = torch.nn.functional.normalize(
        trained_numeric_embeddings,
        dim=-1,
    )

    initial_trained_cosine = (
        initial_direction * trained_direction
    ).sum(dim=-1)

    print("\nNumeric embedding update statistics")
    print("Mean L2 change:", l2_change.mean().item())
    print("Median L2 change:", l2_change.median().item())
    print(
        "Mean initial-trained cosine:",
        initial_trained_cosine.mean().item(),
    )
    print(
        "Unchanged ratio, L2 < 1e-4:",
        (l2_change < 1e-4).float().mean().item(),
    )

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

