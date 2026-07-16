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
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, EarlyStoppingCallback
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
import re
from collections import Counter
import os

from transformers import TrainerCallback
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

@torch.no_grad()
def initialize_added_tokens_from_subtokens(
    model,
    tokenizer,
    added_tokens,
    subtoken_ids_by_token,
    old_vocab_size,
):
    input_weight = model.get_input_embeddings().weight

    output_layer = model.get_output_embeddings()
    output_weight = (
        output_layer.weight
        if output_layer is not None and hasattr(output_layer, "weight")
        else None
    )

    initialized = 0
    skipped = 0

    for tok in added_tokens:
        new_id = tokenizer.convert_tokens_to_ids(tok)

        old_ids = [
            i for i in subtoken_ids_by_token.get(tok, [])
            if isinstance(i, int) and 0 <= i < old_vocab_size
        ]

        if new_id is None or new_id < 0 or len(old_ids) == 0:
            skipped += 1
            continue

        old_ids_tensor = torch.tensor(
            old_ids,
            device=input_weight.device,
            dtype=torch.long,
        )

        input_vec = input_weight[old_ids_tensor].mean(dim=0)
        input_weight[new_id].copy_(input_vec.to(input_weight.dtype))

        if output_weight is not None:
            output_vec = output_weight[old_ids_tensor].mean(dim=0)
            output_weight[new_id].copy_(output_vec.to(output_weight.dtype))

        initialized += 1

    print(f"Initialized added token rows: {initialized}")
    print(f"Skipped added token rows: {skipped}")

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
    parser.add_argument("--dataset_num_proc", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-5)

    parser.add_argument("--wandb_run_name", type=str, default=None)
    args = parser.parse_args()

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
    vocabulary = np.array([serializer.serialize(i) for i in vocabulary])
    print(f"\nVocabulary: \n{vocabulary}\n")
    added_tokens = [
        str(token)
        for token in vocabulary.reshape(-1).tolist()
    ]

    old_vocab_size = len(tokenizer)

    subtoken_ids_by_token = {
        token: tokenizer.encode(
            token,
            add_special_tokens=False,
        )
        for token in added_tokens
    }

    num_added_tokens = tokenizer.add_tokens(added_tokens)
    verify_numeric_tokenization(
        tokenizer,
        added_tokens,
    )

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
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    model.config.use_cache = False
    if hasattr(model, "generation_config"):
        model.generation_config.use_cache = False


    # add lora to llama model
    # 【変更前】 target_modules=["q_proj", "k_proj", ...], modules_to_save=["embed_tokens", "lm_head"]
    model.resize_token_embeddings(len(tokenizer))

    initialize_added_tokens_from_subtokens(
        model=model,
        tokenizer=tokenizer,
        added_tokens=added_tokens,
        subtoken_ids_by_token=subtoken_ids_by_token,
        old_vocab_size=old_vocab_size,
    )
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

    # train model
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        dataset_num_proc=args.dataset_num_proc,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=args.per_device_train_batch_size,      
            per_device_eval_batch_size=(
                    args.per_device_eval_batch_size
                    if args.per_device_eval_batch_size is not None
                    else args.per_device_train_batch_size
                ),
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            num_train_epochs=args.num_train_epochs,

            weight_decay=0.00,
            warmup_ratio=0.05,
            max_grad_norm=1.0,
            learning_rate=args.learning_rate,

            logging_strategy="steps",
            logging_steps=args.logging_steps,
            logging_first_step=True,

 
            evaluation_strategy="steps" if has_eval else "no",
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
        ),
        callbacks=callbacks,
    )

    # title Show current memory stats
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"\nGPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.\n")

    numeric_ids_device = numeric_token_ids.to(
        model.get_input_embeddings().weight.device
    )

    initial_numeric_embeddings = (
        model.get_input_embeddings()
        .weight[numeric_ids_device]
        .detach()
        .float()
        .cpu()
        .clone()
    )

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