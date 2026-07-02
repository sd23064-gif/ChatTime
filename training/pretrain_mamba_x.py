import argparse
import sys
import re
import numpy as np
import torch
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

class NumericEmbeddingRegularizedSFTTrainer(SFTTrainer):
    def __init__(
        self,
        *args,
        numeric_token_ids=None,
        numeric_token_values=None,
        embedding_reg_weight=0.001,
        embedding_reg_pair_batch_size=1024,
        embedding_reg_close_delta=0.002,
        embedding_reg_far_delta=0.5,
        embedding_reg_close_margin=0.05,
        embedding_reg_far_margin=0.5,
        regularize_lm_head=False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if numeric_token_ids is None or numeric_token_values is None:
            raise ValueError("numeric_token_ids and numeric_token_values are required.")

        self.numeric_token_ids_cpu = numeric_token_ids.detach().cpu().long()
        self.numeric_token_values_cpu = numeric_token_values.detach().cpu().float()

        self.embedding_reg_weight = embedding_reg_weight
        self.embedding_reg_pair_batch_size = embedding_reg_pair_batch_size

        self.embedding_reg_close_delta = embedding_reg_close_delta
        self.embedding_reg_far_delta = embedding_reg_far_delta

        self.embedding_reg_close_margin = embedding_reg_close_margin
        self.embedding_reg_far_margin = embedding_reg_far_margin

        self.regularize_lm_head = regularize_lm_head

    def _sample_pairs(self, device):
        values = self.numeric_token_values_cpu.to(device)
        n = values.numel()
        m = self.embedding_reg_pair_batch_size

        anchors = torch.randint(0, n, (m,), device=device)

        # 数値tokenはほぼ等間隔なので、値差を index step に変換する
        step = torch.median(torch.diff(values)).abs().clamp(min=1e-8)
        close_steps = max(1, int(round(self.embedding_reg_close_delta / step.item())))

        offsets = torch.randint(1, close_steps + 1, (m,), device=device)
        signs = torch.randint(0, 2, (m,), device=device) * 2 - 1

        positives = anchors + signs * offsets
        positives = torch.clamp(positives, 0, n - 1)

        same = positives == anchors
        positives = torch.where(
            same & (anchors < n - 1),
            anchors + 1,
            positives,
        )
        positives = torch.where(
            same & (anchors >= n - 1),
            anchors - 1,
            positives,
        )

        # far pair
        negatives = torch.randint(0, n, (m,), device=device)

        for _ in range(10):
            bad = (values[anchors] - values[negatives]).abs() < self.embedding_reg_far_delta
            if not bad.any():
                break
            negatives[bad] = torch.randint(0, n, (bad.sum().item(),), device=device)

        bad = (values[anchors] - values[negatives]).abs() < self.embedding_reg_far_delta
        if bad.any():
            farthest = torch.where(
                anchors < n // 2,
                torch.full_like(anchors, n - 1),
                torch.zeros_like(anchors),
            )
            negatives = torch.where(bad, farthest, negatives)

        return anchors, positives, negatives

    def _regularize_matrix(self, weight_matrix):
        device = weight_matrix.device

        numeric_ids = self.numeric_token_ids_cpu.to(device)

        emb = weight_matrix[numeric_ids]
        emb = F.normalize(emb.float(), dim=-1)

        anchors, positives, negatives = self._sample_pairs(device)

        anchor_emb = emb[anchors]
        pos_emb = emb[positives]
        neg_emb = emb[negatives]

        pos_sim = (anchor_emb * pos_emb).sum(dim=-1)
        neg_sim = (anchor_emb * neg_emb).sum(dim=-1)

        pos_dist = 1.0 - pos_sim
        neg_dist = 1.0 - neg_sim

        close_loss = F.relu(
            pos_dist - self.embedding_reg_close_margin
        ).pow(2).mean()

        far_loss = F.relu(
            self.embedding_reg_far_margin - neg_dist
        ).pow(2).mean()

        return close_loss + far_loss

    def numeric_embedding_regularization_loss(self, model):
        input_emb = model.get_input_embeddings().weight
        reg_loss = self._regularize_matrix(input_emb)

        if self.regularize_lm_head:
            output_emb = model.get_output_embeddings()
            if output_emb is not None and hasattr(output_emb, "weight"):
                reg_loss = reg_loss + self._regularize_matrix(output_emb.weight)

        return reg_loss

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)

        if isinstance(outputs, dict):
            lm_loss = outputs["loss"]
        else:
            lm_loss = outputs[0]

        if self.embedding_reg_weight > 0:
            emb_reg_loss = self.numeric_embedding_regularization_loss(model)
            loss = lm_loss + self.embedding_reg_weight * emb_reg_loss
        else:
            emb_reg_loss = torch.tensor(0.0, device=lm_loss.device)
            loss = lm_loss

        # log
        if self.state.global_step % max(1, self.args.logging_steps) == 0:
            self.log({
                "lm_loss": lm_loss.detach().float().item(),
                "numeric_embedding_reg_loss": emb_reg_loss.detach().float().item(),
                "total_loss_with_numeric_reg": loss.detach().float().item(),
            })

        return (loss, outputs) if return_outputs else loss
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
        ],
        modules_to_save=[
            "backbone.embeddings",
            "lm_head",
        ],
    )

    model = get_peft_model(model, lora_config)
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
    trainer = NumericEmbeddingRegularizedSFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        #eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        packing=False,

        numeric_token_ids=numeric_token_ids,
        numeric_token_values=numeric_token_values,

        embedding_reg_weight=args.embedding_reg_weight,
        embedding_reg_pair_batch_size=args.embedding_reg_pair_batch_size,
        embedding_reg_close_delta=args.embedding_reg_close_delta,
        embedding_reg_far_delta=args.embedding_reg_far_delta,
        embedding_reg_close_margin=args.embedding_reg_close_margin,
        embedding_reg_far_margin=args.embedding_reg_far_margin,
        regularize_lm_head=args.regularize_lm_head,
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