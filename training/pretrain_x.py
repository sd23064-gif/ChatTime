import argparse
import sys

from unsloth import FastLanguageModel, is_bfloat16_supported
import numpy as np
import torch
from datasets import load_dataset
from transformers import TrainingArguments, LlamaTokenizer
from trl import SFTTrainer


import torch.nn.functional as F

class NumericEmbeddingRegularizedSFTTrainer(SFTTrainer):
    def __init__(
        self,
        *args,
        numeric_token_ids=None,
        numeric_token_values=None,
        embedding_reg_weight=0.01,
        embedding_reg_pair_batch_size=2048,
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

        # close pair: 数値距離 close_delta 以下を狙う
        # ChatTimeの数値tokenはほぼ等間隔なので、index近傍を使う
        step = torch.median(torch.diff(values)).abs().clamp(min=1e-8)
        close_steps = max(1, int(round(self.embedding_reg_close_delta / step.item())))

        offsets = torch.randint(1, close_steps + 1, (m,), device=device)
        signs = torch.randint(0, 2, (m,), device=device) * 2 - 1

        positives = anchors + signs * offsets
        positives = torch.clamp(positives, 0, n - 1)

        # clampで同一tokenになった場合の補正
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

        # far pair: 数値距離 far_delta 以上になるように rejection sampling
        negatives = torch.randint(0, n, (m,), device=device)

        for _ in range(10):
            bad = (values[anchors] - values[negatives]).abs() < self.embedding_reg_far_delta
            if not bad.any():
                break
            negatives[bad] = torch.randint(0, n, (bad.sum().item(),), device=device)

        # まだbadなら、強制的に端を選ぶ
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

        # cosine distance = 1 - cosine similarity
        pos_sim = (anchor_emb * pos_emb).sum(dim=-1)
        neg_sim = (anchor_emb * neg_emb).sum(dim=-1)

        pos_dist = 1.0 - pos_sim
        neg_dist = 1.0 - neg_sim

        # 近い数値なのに embedding が離れすぎる場合
        close_loss = F.relu(pos_dist - self.embedding_reg_close_margin).pow(2).mean()

        # 遠い数値なのに embedding が近すぎる場合
        far_loss = F.relu(self.embedding_reg_far_margin - neg_dist).pow(2).mean()

        return close_loss + far_loss

    def numeric_embedding_regularization_loss(self, model):
        input_emb = model.get_input_embeddings().weight
        reg_loss = self._regularize_matrix(input_emb)

        if self.regularize_lm_head:
            output_emb_layer = model.get_output_embeddings()
            if output_emb_layer is not None and hasattr(output_emb_layer, "weight"):
                reg_loss = reg_loss + self._regularize_matrix(output_emb_layer.weight)

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

        # logging用
        if self.state.global_step % max(1, self.args.logging_steps) == 0:
            self.log({
                "lm_loss": lm_loss.detach().float().item(),
                "numeric_embedding_reg_loss": emb_reg_loss.detach().float().item(),
                "total_loss_with_numeric_reg": loss.detach().float().item(),
            })

        return (loss, outputs) if return_outputs else loss
    
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

    parser.add_argument("--embedding_reg_weight", type=float, default=0.01)
    parser.add_argument("--embedding_reg_pair_batch_size", type=int, default=2048)

    # 「近い」とみなす数値距離
    parser.add_argument("--embedding_reg_close_delta", type=float, default=0.002)

    # 「遠い」とみなす数値距離
    parser.add_argument("--embedding_reg_far_delta", type=float, default=0.5)

    # 近いペアに許す cosine distance の上限
    parser.add_argument("--embedding_reg_close_margin", type=float, default=0.05)

    # 遠いペアに要求する cosine distance の下限
    parser.add_argument("--embedding_reg_far_margin", type=float, default=0.5)

    # lm_head側にも同じ正則化をかけるか
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
    discretizer = Discretizer(low_limit=args.low_limit, high_limit=args.high_limit, n_tokens=args.n_tokens)
    serializer = Serializer(prec=args.prec, time_sep=args.time_sep, time_flag=args.time_flag, nan_flag=args.nan_flag)

    vocabulary = np.concatenate((discretizer.centers[1:-1], [np.NaN])).reshape(-1, 1)
    vocabulary = np.array([serializer.serialize(i) for i in vocabulary])
    print(f"\nVocabulary: \n{vocabulary}\n")

    
    # add token to llama tokenizer
    tokenizer = LlamaTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    print(f"Old model pieces: {len(tokenizer.get_vocab())}")
    tokenizer.add_tokens(vocabulary.tolist())
    # finite numeric token only, excluding Nan
    finite_values = discretizer.centers[1:-1].astype(np.float32)
    finite_tokens = vocabulary[:-1].tolist()

    numeric_token_ids = tokenizer.convert_tokens_to_ids(finite_tokens)

    if any(tid is None or tid < 0 for tid in numeric_token_ids):
        bad = [
            (tok, tid)
            for tok, tid in zip(finite_tokens, numeric_token_ids)
            if tid is None or tid < 0
        ][:10]
        raise ValueError(f"Some numeric tokens were not found in tokenizer: {bad}")

    numeric_token_ids = torch.tensor(numeric_token_ids, dtype=torch.long)
    numeric_token_values = torch.tensor(finite_values, dtype=torch.float32)

    print("Numeric token ids:", numeric_token_ids.shape)
    print("Numeric token values:", numeric_token_values.shape)
    print("value range:", numeric_token_values.min().item(), numeric_token_values.max().item())
    print(f"New model pieces: {len(tokenizer.get_vocab())}")

    EOS_TOKEN = tokenizer.eos_token

    # load model
    model, _ = FastLanguageModel.from_pretrained(
        model_name=args.model_path,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=args.load_in_4bit,
        resize_model_vocab=len(tokenizer.get_vocab()),
    )

    # add lora to llama model
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", ],
        modules_to_save=["embed_tokens", "lm_head", ],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.random_seed,
        max_seq_length=args.max_seq_length,
    )


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
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        dataset_num_proc=64,
        packing=False,
        formatting_func=formatting_func,

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
    model.save_pretrained_merged(args.output_path, tokenizer)
