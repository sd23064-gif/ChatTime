import argparse
import sys
import re
import numpy as np
import torch
import pandas as pd
from datasets import load_dataset
# 【変更前】
# from transformers import TrainingArguments, LlamaTokenizer
# 【変更後】
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

import re
import torch.nn.functional as F
NUMERIC_TOKEN_RE = re.compile(
    r"###([+-]?(?:\d+(?:\.\d*)?|\.\d+)|Nan|NaN|nan)###"
)

class NumericLandmarkRegularizedSFTTrainer(SFTTrainer):
    def __init__(
        self,
        *args,
        numeric_token_ids=None,
        numeric_token_values=None,
        landmark_token_ids=None,
        landmark_token_values=None,
        landmark_reg_weight=0.01,
        positive_max_delta=0.01,
        negative_min_delta=0.1,
        triplet_margin=0.1,
        negative_candidate_count=64,
        landmark_pair_batch_size=1024,
        detach_landmarks=True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        required = [
            numeric_token_ids,
            numeric_token_values,
            landmark_token_ids,
            landmark_token_values,
        ]

        if any(item is None for item in required):
            raise ValueError(
                "Numeric token ids/values and landmark ids/values "
                "are required."
            )

        self.numeric_token_ids_cpu = (
            numeric_token_ids.detach().cpu().long()
        )
        self.numeric_token_values_cpu = (
            numeric_token_values.detach().cpu().float()
        )

        self.landmark_token_ids_cpu = (
            landmark_token_ids.detach().cpu().long()
        )
        self.landmark_token_values_cpu = (
            landmark_token_values.detach().cpu().float()
        )

        self.landmark_reg_weight = landmark_reg_weight
        self.positive_max_delta = positive_max_delta
        self.negative_min_delta = negative_min_delta
        self.triplet_margin = triplet_margin
        self.negative_candidate_count = negative_candidate_count
        self.landmark_pair_batch_size = landmark_pair_batch_size
        self.detach_landmarks = detach_landmarks

        self._last_logged_global_step = -1

    def _sample_landmark_triplets(self, weight_matrix):
        device = weight_matrix.device

        numeric_ids = self.numeric_token_ids_cpu.to(device)
        numeric_values = self.numeric_token_values_cpu.to(device)

        landmark_ids = self.landmark_token_ids_cpu.to(device)
        landmark_values = self.landmark_token_values_cpu.to(device)

        numeric_emb = F.normalize(
            weight_matrix[numeric_ids].float(),
            dim=-1,
        )

        landmark_emb = F.normalize(
            weight_matrix[landmark_ids].float(),
            dim=-1,
        )

        n_numeric = numeric_ids.numel()
        batch_size = min(
            self.landmark_pair_batch_size,
            n_numeric,
        )

        anchor_indices = torch.randint(
            0,
            n_numeric,
            (batch_size,),
            device=device,
        )

        anchor_values = numeric_values[anchor_indices]
        anchor_emb = numeric_emb[anchor_indices]

        # ----------------------------------------------------
        # Positive:
        # 数値的に最も近いlandmarkを選ぶ
        # ----------------------------------------------------
        value_diff = torch.abs(
            anchor_values[:, None] - landmark_values[None, :]
        )

        positive_diff, positive_indices = value_diff.min(dim=1)

        valid_positive = (
            positive_diff <= self.positive_max_delta
        )

        # ----------------------------------------------------
        # Hard-negative candidates:
        # 数値差がnegative_min_delta以上のlandmarkから
        # ランダム候補を抽出し、その中で最も類似するものを選ぶ
        # ----------------------------------------------------
        n_landmarks = landmark_ids.numel()
        candidate_count = min(
            self.negative_candidate_count,
            n_landmarks,
        )

        random_candidates = torch.randint(
            0,
            n_landmarks,
            (batch_size, candidate_count),
            device=device,
        )

        candidate_values = landmark_values[random_candidates]

        far_mask = (
            torch.abs(
                anchor_values[:, None] - candidate_values
            )
            >= self.negative_min_delta
        )

        candidate_emb = landmark_emb[random_candidates]

        similarities = torch.einsum(
            "bd,bkd->bk",
            anchor_emb,
            candidate_emb,
        )

        similarities = similarities.masked_fill(
            ~far_mask,
            -float("inf"),
        )

        negative_similarity, best_candidate_position = (
            similarities.max(dim=1)
        )

        has_valid_negative = torch.isfinite(
            negative_similarity
        )

        row_indices = torch.arange(
            batch_size,
            device=device,
        )

        negative_indices = random_candidates[
            row_indices,
            best_candidate_position,
        ]

        valid = valid_positive & has_valid_negative

        return (
            anchor_emb[valid],
            landmark_emb[positive_indices[valid]],
            landmark_emb[negative_indices[valid]],
            positive_diff[valid],
            negative_similarity[valid],
        )

    def numeric_landmark_loss(self, model):
        weight_matrix = model.get_input_embeddings().weight

        (
            anchor_emb,
            positive_emb,
            negative_emb,
            positive_value_diff,
            negative_similarity,
        ) = self._sample_landmark_triplets(weight_matrix)

        if anchor_emb.shape[0] == 0:
            zero = weight_matrix.sum() * 0.0

            return zero, {
                "valid_triplets": 0,
                "positive_value_diff_mean": 0.0,
                "hard_negative_similarity_mean": 0.0,
            }

        if self.detach_landmarks:
            positive_emb = positive_emb.detach()
            negative_emb = negative_emb.detach()

        positive_distance = 1.0 - (
            anchor_emb * positive_emb
        ).sum(dim=-1)

        negative_distance = 1.0 - (
            anchor_emb * negative_emb
        ).sum(dim=-1)

        triplet_loss = F.relu(
            positive_distance
            - negative_distance
            + self.triplet_margin
        ).mean()

        metrics = {
            "valid_triplets": int(anchor_emb.shape[0]),
            "positive_value_diff_mean": (
                positive_value_diff.detach().mean().item()
            ),
            "hard_negative_similarity_mean": (
                negative_similarity.detach().mean().item()
            ),
            "positive_distance_mean": (
                positive_distance.detach().mean().item()
            ),
            "negative_distance_mean": (
                negative_distance.detach().mean().item()
            ),
        }

        return triplet_loss, metrics

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        **kwargs,
    ):
        outputs = model(**inputs)

        if isinstance(outputs, dict):
            lm_loss = outputs["loss"]
        else:
            lm_loss = outputs[0]

        if self.landmark_reg_weight > 0:
            landmark_loss, landmark_metrics = (
                self.numeric_landmark_loss(model)
            )

            weighted_landmark_loss = (
                self.landmark_reg_weight
                * landmark_loss
            )

            loss = lm_loss + weighted_landmark_loss
        else:
            landmark_loss = torch.zeros(
                (),
                device=lm_loss.device,
            )

            weighted_landmark_loss = torch.zeros(
                (),
                device=lm_loss.device,
            )

            landmark_metrics = {
                "valid_triplets": 0,
                "positive_value_diff_mean": 0.0,
                "hard_negative_similarity_mean": 0.0,
                "positive_distance_mean": 0.0,
                "negative_distance_mean": 0.0,
            }

            loss = lm_loss

        current_step = int(self.state.global_step)

        if (
            current_step > 0
            and current_step
            % max(1, self.args.logging_steps)
            == 0
            and current_step
            != self._last_logged_global_step
        ):
            self.log(
                {
                    "lm_loss": (
                        lm_loss.detach().float().item()
                    ),
                    "landmark_triplet_loss": (
                        landmark_loss
                        .detach()
                        .float()
                        .item()
                    ),
                    "weighted_landmark_loss": (
                        weighted_landmark_loss
                        .detach()
                        .float()
                        .item()
                    ),
                    "total_loss_with_landmark_reg": (
                        loss.detach().float().item()
                    ),
                    **landmark_metrics,
                }
            )

            self._last_logged_global_step = (
                current_step
            )

        if return_outputs:
            return loss, outputs

        return loss

def load_landmark_tensors(landmark_csv,tokenizer):
    landmark_df = pd.read_csv(landmark_csv)

    if "query_token" in landmark_df.columns:
        token_column = "query_token"
    elif "token" in landmark_df.columns:
        token_column = "token"
    else:
        raise ValueError(
            "Landmark CSV needs query_token or token column."
        )

    if "query_value" in landmark_df.columns:
        value_column = "query_value"
    elif "value" in landmark_df.columns:
        value_column = "value"
    else:
        raise ValueError(
            "Landmark CSV needs query_value or value column."
        )

    tokens = landmark_df[token_column].astype(str).tolist()

    token_ids = tokenizer.convert_tokens_to_ids(tokens)

    valid_rows = []

    for token, token_id, value in zip(
        tokens,
        token_ids,
        landmark_df[value_column].tolist(),
    ):
        if token_id is None or token_id < 0:
            continue

        valid_rows.append(
            {
                "token": token,
                "token_id": int(token_id),
                "value": float(value),
            }
        )

    landmark_token_ids = torch.tensor(
        [row["token_id"] for row in valid_rows],
        dtype=torch.long,
    )

    landmark_token_values = torch.tensor(
        [row["value"] for row in valid_rows],
        dtype=torch.float32,
    )

    print("landmark token count:", len(valid_rows))
    print(
        "landmark value range:",
        landmark_token_values.min().item(),
        landmark_token_values.max().item(),
    )

    return landmark_token_ids, landmark_token_values
def collect_numeric_token_ids_and_values(tokenizer):
    rows = []

    vocab = tokenizer.get_vocab()

    for token, token_id in vocab.items():
        m = NUMERIC_TOKEN_RE.fullmatch(str(token))
        if m is None:
            continue

        raw = m.group(1)

        if raw.lower() == "nan":
            continue

        value = float(raw)

        rows.append({
            "token": token,
            "token_id": token_id,
            "value": value,
        })

    rows = sorted(rows, key=lambda x: x["value"])

    numeric_token_ids = torch.tensor(
        [r["token_id"] for r in rows],
        dtype=torch.long,
    )

    numeric_token_values = torch.tensor(
        [r["value"] for r in rows],
        dtype=torch.float32,
    )

    print("numeric token count:", len(rows))
    print("value range:", numeric_token_values.min().item(), numeric_token_values.max().item())

    return numeric_token_ids, numeric_token_values
def is_bfloat16_supported():
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()
@torch.no_grad()
def initialize_added_token_embeddings_from_subtokens(
    model,
    tokenizer,
    added_tokens,
    subtoken_ids_by_token,
    old_vocab_size,
    init_noise_std=0.0,
):
    input_emb = model.get_input_embeddings()
    input_w = input_emb.weight

    output_emb = model.get_output_embeddings()
    output_w = (
        output_emb.weight
        if output_emb is not None and hasattr(output_emb, "weight")
        else None
    )

    device = input_w.device

    old_input_mean = input_w[:old_vocab_size].mean(dim=0)

    if output_w is not None:
        old_output_mean = output_w[:old_vocab_size].mean(dim=0)
    else:
        old_output_mean = None

    initialized = 0
    skipped = 0

    for tok in added_tokens:
        new_id = tokenizer.convert_tokens_to_ids(tok)

        if new_id is None or new_id < 0:
            skipped += 1
            continue

        old_ids = subtoken_ids_by_token.get(tok, [])
        old_ids = [
            i for i in old_ids
            if isinstance(i, int) and 0 <= i < old_vocab_size
        ]

        if len(old_ids) > 0:
            old_ids_tensor = torch.tensor(
                old_ids,
                device=device,
                dtype=torch.long,
            )

            new_input_vec = input_w[old_ids_tensor].mean(dim=0)

            if output_w is not None:
                new_output_vec = output_w[old_ids_tensor].mean(dim=0)
            else:
                new_output_vec = None
        else:
            new_input_vec = old_input_mean

            if output_w is not None:
                new_output_vec = old_output_mean
            else:
                new_output_vec = None

        if init_noise_std > 0:
            new_input_vec = (
                new_input_vec
                + init_noise_std * torch.randn_like(new_input_vec)
            )

            if new_output_vec is not None:
                new_output_vec = (
                    new_output_vec
                    + init_noise_std * torch.randn_like(new_output_vec)
                )

        input_w[new_id].copy_(new_input_vec.to(dtype=input_w.dtype))

        if output_w is not None:
            output_w[new_id].copy_(new_output_vec.to(dtype=output_w.dtype))

        initialized += 1

    print(f"Initialized added token rows: {initialized}")
    print(f"Skipped added token rows: {skipped}")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--code_path", type=str, required=True, default=None)
    parser.add_argument("--model_path", type=str, required=True, default=None)
    parser.add_argument("--dataset_path", type=str, required=True, default=None)
    parser.add_argument("--log_path", type=str, required=True, default=None)
    parser.add_argument("--output_path", type=str, required=True, default=None)

    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)

    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.00)
    parser.add_argument("--random_seed", type=int, default=3407)

    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=64)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--save_steps", type=int, default=2)
    parser.add_argument("--logging_steps", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=-1)

    parser.add_argument("--low_limit", type=float, default=-1)
    parser.add_argument("--high_limit", type=float, default=1)
    parser.add_argument("--n_tokens", type=int, default=10002)
    parser.add_argument("--prec", type=int, default=4)
    parser.add_argument("--time_sep", type=str, default=" ")
    parser.add_argument("--time_flag", type=str, default="###")
    parser.add_argument("--nan_flag", type=str, default="Nan")

    parser.add_argument("--embedding_reg_weight", type=float, default=0.001)

    parser.add_argument("--embedding_reg_pair_batch_size", type=int, default=1024)

    parser.add_argument("--embedding_reg_close_delta", type=float, default=0.002)
    parser.add_argument("--embedding_reg_far_delta", type=float, default=0.5)

    parser.add_argument("--embedding_reg_close_margin", type=float, default=0.05)
    parser.add_argument("--embedding_reg_far_margin", type=float, default=0.5)

    parser.add_argument("--regularize_lm_head", action="store_true", default=False)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument(
        "--landmark_csv",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--landmark_reg_weight",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--positive_max_delta",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--negative_min_delta",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--triplet_margin",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--negative_candidate_count",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--landmark_pair_batch_size",
        type=int,
        default=1024,
    )

    parser.add_argument(
        "--detach_landmarks",
        action="store_true",
    )
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
    old_vocab_size = len(tokenizer)

    subtoken_ids_by_token = {
        tok: tokenizer.encode(tok, add_special_tokens=False)
        for tok in vocabulary.tolist()
    }

    num_added_tokens = tokenizer.add_tokens(vocabulary.tolist())

    
    print(f"Added tokens: {num_added_tokens}")
    print(f"New tokenizer size: {len(tokenizer)}")
    numeric_token_ids, numeric_token_values = collect_numeric_token_ids_and_values(tokenizer)
    EOS_TOKEN = tokenizer.eos_token


    # load model
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )


    # add lora to llama model
    # 【変更前】 target_modules=["q_proj", "k_proj", ...], modules_to_save=["embed_tokens", "lm_head"]
    model.resize_token_embeddings(len(tokenizer))
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
            "out_proj"
        ],
        modules_to_save=[
            "backbone.embeddings",
            "lm_head",
        ],
    )

    model = get_peft_model(model, lora_config)
    print("Input embedding requires_grad:")
    print(model.get_input_embeddings().weight.requires_grad)

    output_emb = model.get_output_embeddings()
    if output_emb is not None and hasattr(output_emb, "weight"):
        print("Output embedding requires_grad:")
        print(output_emb.weight.requires_grad)
    
    model.print_trainable_parameters()


    # load dataset
    def formatting_func(example):
        return example["text"] + EOS_TOKEN


    print(f"\nLoading dataset in {args.dataset_path}")
    dataset = load_dataset(
            "csv",
            data_files=f"https://huggingface.co/datasets/{args.dataset_path}/resolve/main/ChatTime-1-Pretrain-1M.csv",
            split="train",
        )

    print(f"Dataset example: \n{dataset[0]['text']}\n")

    # train model
    trainer = NumericLandmarkRegularizedSFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        packing=False,

        numeric_token_ids=numeric_token_ids,
        numeric_token_values=numeric_token_values,

        landmark_token_ids=landmark_token_ids,
        landmark_token_values=landmark_token_values,

        landmark_reg_weight=args.landmark_reg_weight,
        positive_max_delta=args.positive_max_delta,
        negative_min_delta=args.negative_min_delta,
        triplet_margin=args.triplet_margin,
        negative_candidate_count=args.negative_candidate_count,
        landmark_pair_batch_size=args.landmark_pair_batch_size,
        detach_landmarks=args.detach_landmarks,
        args=TrainingArguments(
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            num_train_epochs=args.num_train_epochs,
            weight_decay=0.01,
            warmup_ratio=0.05,
            max_grad_norm=1.0,
            learning_rate=2e-4,
            logging_strategy="steps",
            logging_steps=args.logging_steps,
            save_strategy="steps",
            save_steps=args.save_steps,
            max_steps=args.max_steps,
            save_total_limit=1,
            logging_first_step=True,
            optim="adamw_8bit",
            lr_scheduler_type="cosine",
            seed=args.random_seed,
            output_dir=args.log_path,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),

            report_to="wandb",
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
    print(f"Saving LoRA adapter to {args.output_path}")

    model.save_pretrained(args.output_path)
    tokenizer.save_pretrained(args.output_path)

    print("Save completed")