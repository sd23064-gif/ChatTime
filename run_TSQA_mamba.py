import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import re
import ast
import gc
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel

from utils.tools import Discretizer, Serializer
from utils.prompt import getPrompt


def parse_series(x):
    if isinstance(x, np.ndarray):
        return x.astype(np.float64)
    if isinstance(x, list):
        return np.asarray(x, dtype=np.float64)
    if isinstance(x, str):
        return np.asarray(ast.literal_eval(x), dtype=np.float64)
    return np.asarray(x, dtype=np.float64)


def extract_choice(text):
    if text is None:
        return None

    text = str(text).strip().lower()

    m = re.search(r"\(([abc])\)", text)
    if m:
        return f"({m.group(1)})"

    m = re.search(r"\b([abc])\b", text)
    if m:
        return f"({m.group(1)})"

    return None


def load_mamba_chattime_model(base_model_path, adapter_path, use_bf16=False):
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_path,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    dtype = torch.bfloat16 if use_bf16 else torch.float32
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
    )

    model.resize_token_embeddings(len(tokenizer))

    if hasattr(model.config, "pad_token_id"):
        model.config.pad_token_id = tokenizer.pad_token_id

    model = PeftModel.from_pretrained(
        model,
        adapter_path,
        is_trainable=False,
    )

    model.to(device)
    model.eval()

    print("Model device:", next(model.parameters()).device)
    print("CUDA device count:", torch.cuda.device_count())

    return model, tokenizer



def analyze_mamba(model, tokenizer, question, series, num_samples=8, max_new_tokens=64):
    discretizer = Discretizer()
    serializer = Serializer()

    dispersed_series = discretizer.discretize(series)
    serialized_series = serializer.serialize(dispersed_series)

    # 回答形式をさらに強く指定
    question_strict = (
        str(question).strip()
        + "\n\nImportant: You must answer with exactly one of (a), (b), or (c)."
        + "\nDo not explain. Only output one choice."
    )

    prompt = getPrompt(
        flag="analysis",
        instruction=question_strict,
        input=serialized_series,
    )

    device = next(model.parameters()).device

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=2048,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    all_generated_texts = []
    responses = []

    for _ in range(num_samples):
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_k=100,
                top_p=1.0,
                temperature=0.7,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

        generated_text = tokenizer.decode(
            output_ids[0],
            skip_special_tokens=False,
        )

        all_generated_texts.append(generated_text)

        # Response以降を優先して見る
        text = generated_text.lower()
        if "### response:" in text:
            text = text.split("### response:", 1)[1]

        matches = re.findall(r"\([abc]\)", text)
        if len(matches) > 0:
            responses.append(matches[0])
            continue

        # 念のため a/b/c 単独も拾う
        matches = re.findall(r"\b([abc])\b", text)
        if len(matches) > 0:
            responses.append(f"({matches[0]})")

    if len(responses) == 0:
        # 重要: Noneだけでなく、生成全文も返す
        return None, "\n\n--- SAMPLE ---\n\n".join(all_generated_texts)

    pred = max(set(responses), key=responses.count)

    return pred, "\n\n--- SAMPLE ---\n\n".join(all_generated_texts)

def make_eval_subset(df, max_eval_samples=100, seed=42):
    """
    Task × Size × Label でなるべく均等にサンプリングする。
    """
    if max_eval_samples is None:
        return df.reset_index(drop=True)

    group_cols = ["Task", "Size", "Label"]
    n_groups = df.groupby(group_cols).ngroups
    per_group = max(1, max_eval_samples // n_groups)

    eval_df = (
        df.groupby(group_cols, group_keys=False)
          .apply(lambda x: x.sample(n=min(per_group, len(x)), random_state=seed))
          .reset_index(drop=True)
    )

    if len(eval_df) > max_eval_samples:
        eval_df = eval_df.sample(n=max_eval_samples, random_state=seed).reset_index(drop=True)

    return eval_df


def summarize(result_df):
    summary_task = (
        result_df
        .groupby("task", as_index=False)
        .agg(
            accuracy=("correct", "mean"),
            n_samples=("correct", "count"),
            n_correct=("correct", "sum"),
            n_error=("error", lambda x: (x != "").sum()),
        )
    )

    overall = pd.DataFrame([{
        "task": "ALL",
        "accuracy": result_df["correct"].mean(),
        "n_samples": len(result_df),
        "n_correct": result_df["correct"].sum(),
        "n_error": (result_df["error"] != "").sum(),
    }])

    return pd.concat([summary_task, overall], ignore_index=True)


def main():
    base_model_path = "state-spaces/mamba-370m-hf"
    adapter_path = "/workspace/outputs/models/ChatTime-Mamba-Chat"

    output_dir = "/workspace/outputs/mamba_tsqa_eval"
    os.makedirs(output_dir, exist_ok=True)

    max_eval_samples = 100
    use_bf16 = False

    print("Loading TSQA dataset...")
    ds = load_dataset("ChengsenWang/TSQA")
    df = ds["train"].to_pandas()
    df.columns = df.columns.str.replace("\ufeff", "", regex=False).str.strip()

    print("Dataset shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print(df["Task"].value_counts())

    eval_df = make_eval_subset(df, max_eval_samples=max_eval_samples, seed=42)

    print("Eval shape:", eval_df.shape)
    print(eval_df["Task"].value_counts())

    print("Loading Mamba ChatTime model...")
    model, tokenizer = load_mamba_chattime_model(
        base_model_path=base_model_path,
        adapter_path=adapter_path,
        use_bf16=use_bf16,
    )

    results = []

    for pos, row in enumerate(tqdm(eval_df.itertuples(index=False), total=len(eval_df))):
        task = str(row.Task)
        size = int(row.Size)
        label = str(row.Label)
        question = str(row.Question)
        answer = str(row.Answer)
        series = parse_series(row.Series)

        error = ""

        try:
            model_output, raw_generation = analyze_mamba(
                model=model,
                tokenizer=tokenizer,
                question=question,
                series=series,
                num_samples=8,
                max_new_tokens=64,
            )
        except Exception as e:
            model_output = None
            raw_generation = ""
            error = str(e)

        gold_choice = extract_choice(answer)
        pred_choice = extract_choice(model_output)

        correct = int(pred_choice == gold_choice)

        results.append({
            "position": pos,
            "task": task,
            "size": size,
            "label": label,
            "question": question,
            "answer": answer,
            "model_output": model_output,
            "raw_generation": raw_generation,
            "gold_choice": gold_choice,
            "pred_choice": pred_choice,
            "correct": correct,
            "error": error,
        })

        if pos < 3:
            print("\n--- sample ---")
            print("Task:", task)
            print("Size:", size)
            print("Label:", label)
            print("Gold:", answer, "=>", gold_choice)
            print("Pred:", model_output, "=>", pred_choice)
            print("Correct:", correct)
            print("Error:", error)

    result_df = pd.DataFrame(results)

    detail_path = os.path.join(output_dir, "mamba_tsqa_details.csv")
    result_df.to_csv(detail_path, index=False)

    summary_df = summarize(result_df)

    summary_path = os.path.join(output_dir, "mamba_tsqa_summary_by_task.csv")
    summary_df.to_csv(summary_path, index=False)

    summary_size = (
        result_df
        .groupby("size", as_index=False)
        .agg(
            accuracy=("correct", "mean"),
            n_samples=("correct", "count"),
            n_correct=("correct", "sum"),
        )
        .sort_values("size")
    )

    size_path = os.path.join(output_dir, "mamba_tsqa_summary_by_size.csv")
    summary_size.to_csv(size_path, index=False)

    summary_task_size = (
        result_df
        .groupby(["task", "size"], as_index=False)
        .agg(
            accuracy=("correct", "mean"),
            n_samples=("correct", "count"),
            n_correct=("correct", "sum"),
        )
        .sort_values(["task", "size"])
    )

    task_size_path = os.path.join(output_dir, "mamba_tsqa_summary_by_task_size.csv")
    summary_task_size.to_csv(task_size_path, index=False)

    print("\nSummary by task:")
    print(summary_df)

    print("\nSummary by size:")
    print(summary_size)

    print("\nSaved:")
    print(detail_path)
    print(summary_path)
    print(size_path)
    print(task_size_path)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
