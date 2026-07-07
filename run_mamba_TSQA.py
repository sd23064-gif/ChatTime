import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import re
import ast
import gc
import argparse
import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import logging

logging.set_verbosity_error()

from model.model import ChatTime
from model.mamba_model import ChatTimeMamba


# ============================================================
# Utility functions
# ============================================================

def parse_series(x):
    """
    Convert Series column to numpy array.

    Hugging Face dataset stores Series as a string like:
    "[0.1, 0.2, ...]"
    """
    if isinstance(x, np.ndarray):
        return x.astype(np.float64)

    if isinstance(x, list):
        return np.asarray(x, dtype=np.float64)

    if isinstance(x, str):
        return np.asarray(ast.literal_eval(x), dtype=np.float64)

    return np.asarray(x, dtype=np.float64)


def extract_choice(text):
    """
    Extract answer choice: (a), (b), or (c).
    """
    if text is None:
        return None

    text = str(text).strip().lower()

    match = re.search(r"\(([abc])\)", text)
    if match:
        return f"({match.group(1)})"

    match = re.search(r"\b([abc])\b", text)
    if match:
        return f"({match.group(1)})"

    return None


def make_eval_subset(df, max_eval_samples=100, seed=42):
    """
    Sample evaluation data evenly from each Task.
    If max_eval_samples is None, use all data.
    """
    if max_eval_samples is None:
        return df.reset_index(drop=True)

    tasks = sorted(df["Task"].dropna().unique().tolist())

    if len(tasks) == 0:
        raise ValueError("No valid Task values found in df['Task'].")

    per_task = max(1, max_eval_samples // len(tasks))

    sampled_list = []

    for task in tasks:
        task_df = df[df["Task"] == task]
        n = min(per_task, len(task_df))

        sampled = task_df.sample(
            n=n,
            random_state=seed,
        )

        sampled_list.append(sampled)

    eval_df = pd.concat(sampled_list, axis=0).reset_index(drop=True)

    if len(eval_df) > max_eval_samples:
        eval_df = eval_df.sample(
            n=max_eval_samples,
            random_state=seed,
        ).reset_index(drop=True)

    return eval_df


def cleanup_model(model):
    del model
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


# ============================================================
# Baseline models
# ============================================================

class RandomChoiceModel:
    def __init__(self, seed=42):
        self.rng = random.Random(seed)

    def analyze(self, question, series):
        return self.rng.choice(["(a)", "(b)", "(c)"])


class FixedChoiceModel:
    def __init__(self, choice="(a)"):
        self.choice = choice

    def analyze(self, question, series):
        return self.choice


# ============================================================
# Model loading
# ============================================================

def load_eval_model(config):
    model_type = config["type"]

    if model_type == "random":
        return RandomChoiceModel(seed=config.get("seed", 42))

    if model_type == "fixed":
        return FixedChoiceModel(choice=config.get("choice", "(a)"))

    if model_type == "chattime":
        return ChatTime(
            model_path=config["model_path"],
            max_pred_len=config.get("max_pred_len", 16),
            num_samples=config.get("num_samples", 8),
            top_k=config.get("top_k", 100),
            top_p=config.get("top_p", 1.0),
            temperature=config.get("temperature", 1.0),
        )

    if model_type == "mamba":
        return ChatTimeMamba(
            base_model_path=config["base_model_path"],
            adapter_path=config.get("adapter_path", None),
            hist_len=None,
            pred_len=None,
            max_pred_len=config.get("max_pred_len", 16),
            num_samples=config.get("num_samples", 8),
            top_k=config.get("top_k", 100),
            top_p=config.get("top_p", 1.0),
            temperature=config.get("temperature", 1.0),
            torch_dtype=torch.float16,
            merge_lora=config.get("merge_lora", False),
        )

    raise ValueError(f"Unknown model type: {model_type}")


def safe_analyze(model, question, series):
    """
    Run model.analyze safely.
    Returns:
        model_output: raw model output
        pred_choice: extracted (a)/(b)/(c)
        parse_success: 1 if valid choice extracted else 0
        error: error message or None
    """
    try:
        model_output = model.analyze(question, series)
        pred_choice = extract_choice(model_output)
        parse_success = int(pred_choice is not None)
        return model_output, pred_choice, parse_success, None

    except Exception as e:
        return "", None, 0, str(e)


# ============================================================
# Summary functions
# ============================================================

def calculate_summary_by_task(result_df):
    return (
        result_df
        .groupby(["model", "task"], as_index=False)
        .agg(
            accuracy=("correct", "mean"),
            parse_success_rate=("parse_success", "mean"),
            n_samples=("correct", "count"),
            n_correct=("correct", "sum"),
            n_parse_success=("parse_success", "sum"),
        )
        .sort_values(["model", "task"])
        .reset_index(drop=True)
    )


def calculate_summary_by_size(result_df):
    return (
        result_df
        .groupby(["model", "size"], as_index=False)
        .agg(
            accuracy=("correct", "mean"),
            parse_success_rate=("parse_success", "mean"),
            n_samples=("correct", "count"),
            n_correct=("correct", "sum"),
            n_parse_success=("parse_success", "sum"),
        )
        .sort_values(["model", "size"])
        .reset_index(drop=True)
    )


def calculate_summary_by_task_size(result_df):
    return (
        result_df
        .groupby(["model", "task", "size"], as_index=False)
        .agg(
            accuracy=("correct", "mean"),
            parse_success_rate=("parse_success", "mean"),
            n_samples=("correct", "count"),
            n_correct=("correct", "sum"),
            n_parse_success=("parse_success", "sum"),
        )
        .sort_values(["model", "task", "size"])
        .reset_index(drop=True)
    )


def calculate_overall_summary(result_df):
    return (
        result_df
        .groupby(["model"], as_index=False)
        .agg(
            accuracy=("correct", "mean"),
            parse_success_rate=("parse_success", "mean"),
            n_samples=("correct", "count"),
            n_correct=("correct", "sum"),
            n_parse_success=("parse_success", "sum"),
            n_errors=("error", lambda s: s.notna().sum()),
        )
        .sort_values("accuracy", ascending=False)
        .reset_index(drop=True)
    )


def calculate_confusion_matrix(result_df):
    rows = []

    for model_name, model_df in result_df.groupby("model"):
        cm = pd.crosstab(
            model_df["gold_choice"],
            model_df["pred_choice"],
            dropna=False,
        )

        cm = cm.reindex(
            index=["(a)", "(b)", "(c)"],
            columns=["(a)", "(b)", "(c)", None],
            fill_value=0,
        )

        for gold in cm.index:
            for pred in cm.columns:
                rows.append({
                    "model": model_name,
                    "gold_choice": gold,
                    "pred_choice": str(pred),
                    "count": int(cm.loc[gold, pred]),
                })

    return pd.DataFrame(rows)

def calculate_accuracy_on_parsed(result_df):
    parsed_df = result_df[result_df["parse_success"] == 1].copy()

    if len(parsed_df) == 0:
        return pd.DataFrame(columns=[
            "model",
            "accuracy_on_parsed",
            "n_parsed",
        ])

    return (
        parsed_df
        .groupby(["model"], as_index=False)
        .agg(
            accuracy_on_parsed=("correct", "mean"),
            n_parsed=("correct", "count"),
        )
        .sort_values("accuracy_on_parsed", ascending=False)
        .reset_index(drop=True)
    )

# ============================================================
# Build model configs
# ============================================================

def build_model_configs(args):
    configs = []

    if args.include_random:
        configs.append({
            "name": "random",
            "type": "random",
            "seed": args.seed,
        })

    if args.include_fixed_a:
        configs.append({
            "name": "fixed_a",
            "type": "fixed",
            "choice": "(a)",
        })

    if args.include_chattime:
        configs.append({
            "name": "chattime_7b",
            "type": "chattime",
            "model_path": args.chattime_model_path,
            "num_samples": args.num_samples,
            "max_pred_len": args.max_pred_len,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "temperature": args.temperature,
        })

    if args.include_mamba_base:
        configs.append({
            "name": "mamba_base",
            "type": "mamba",
            "base_model_path": args.mamba_base_model_path,
            "adapter_path": None,
            "num_samples": args.num_samples,
            "max_pred_len": args.max_pred_len,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "temperature": args.temperature,
        })

    if args.mamba_pretrain_adapter is not None:
        configs.append({
            "name": "mamba_pretrain",
            "type": "mamba",
            "base_model_path": args.mamba_base_model_path,
            "adapter_path": args.mamba_pretrain_adapter,
            "num_samples": args.num_samples,
            "max_pred_len": args.max_pred_len,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "temperature": args.temperature,
        })

    if args.mamba_finetune_adapter is not None:
        configs.append({
            "name": "mamba_finetuned",
            "type": "mamba",
            "base_model_path": args.mamba_base_model_path,
            "adapter_path": args.mamba_finetune_adapter,
            "num_samples": args.num_samples,
            "max_pred_len": args.max_pred_len,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "temperature": args.temperature,
        })
    if args.normal_mamba_adapter is not None:
        configs.append({
            "name": "mamba_normal_finetuned",
            "type": "mamba",
            "base_model_path": args.mamba_base_model_path,
            "adapter_path": args.normal_mamba_adapter,
            "num_samples": args.num_samples,
            "max_pred_len": args.max_pred_len,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "temperature": args.temperature,
        })

    if args.value_norm_pretrain_adapter is not None:
        configs.append({
            "name": "mamba_value_norm_pretrain",
            "type": "mamba",
            "base_model_path": args.mamba_base_model_path,
            "adapter_path": args.value_norm_pretrain_adapter,
            "num_samples": args.num_samples,
            "max_pred_len": args.max_pred_len,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "temperature": args.temperature,
        })

    if args.value_norm_finetune_adapter is not None:
        configs.append({
            "name": "mamba_value_norm_finetuned",
            "type": "mamba",
            "base_model_path": args.mamba_base_model_path,
            "adapter_path": args.value_norm_finetune_adapter,
            "num_samples": args.num_samples,
            "max_pred_len": args.max_pred_len,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "temperature": args.temperature,
        })
    return configs


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", type=str, default="outputs/tsqa_model_comparison")
    parser.add_argument("--max_eval_samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--max_pred_len", type=int, default=16)
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)

    parser.add_argument("--include_random", action="store_true")
    parser.add_argument("--include_fixed_a", action="store_true")

    parser.add_argument("--include_chattime", action="store_true")
    parser.add_argument(
        "--chattime_model_path",
        type=str,
        default="ChengsenWang/ChatTime-1-7B-Chat",
    )

    parser.add_argument("--include_mamba_base", action="store_true")
    parser.add_argument(
        "--mamba_base_model_path",
        type=str,
        default="state-spaces/mamba-370m-hf",
    )
    parser.add_argument("--mamba_pretrain_adapter", type=str, default=None)
    parser.add_argument("--mamba_finetune_adapter", type=str, default=None)

    parser.add_argument("--debug_first_n", type=int, default=3)
    parser.add_argument("--normal_mamba_adapter", type=str, default=None)
    parser.add_argument("--value_norm_pretrain_adapter", type=str, default=None)
    parser.add_argument("--value_norm_finetune_adapter", type=str, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    max_eval_samples = args.max_eval_samples
    if max_eval_samples is not None and max_eval_samples < 0:
        max_eval_samples = None

    model_configs = build_model_configs(args)

    if len(model_configs) == 0:
        raise ValueError(
            "No model specified. "
            "Use --include_random, --include_chattime, "
            "--mamba_pretrain_adapter, or --mamba_finetune_adapter."
        )

    print("\nModel configs:")
    for config in model_configs:
        print(config)

    # --------------------------------------------------------
    # 1. Load Hugging Face TSQA dataset
    # --------------------------------------------------------
    print("\nLoading dataset from Hugging Face: ChengsenWang/TSQA")

    ds = load_dataset("ChengsenWang/TSQA")
    df = ds["train"].to_pandas()

    # Clean column names
    df.columns = df.columns.str.replace("\ufeff", "", regex=False).str.strip()

    print("\nDataset information")
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())

    # --------------------------------------------------------
    # 2. Required column check
    # --------------------------------------------------------
    required_cols = ["Task", "Size", "Question", "Answer", "Label", "Series"]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    print("\nTask distribution in full dataset:")
    print(df["Task"].value_counts(dropna=False))

    print("\nSize distribution in full dataset:")
    print(df["Size"].value_counts(dropna=False).sort_index())

    df = df.dropna(subset=["Task", "Size"]).reset_index(drop=True)

    df["Task"] = df["Task"].astype(str)
    df["Size"] = df["Size"].astype(int)

    # --------------------------------------------------------
    # 3. Parse Series
    # --------------------------------------------------------
    df["Series"] = df["Series"].apply(parse_series)

    # --------------------------------------------------------
    # 4. Create evaluation subset
    # --------------------------------------------------------
    eval_df = make_eval_subset(
        df=df,
        max_eval_samples=max_eval_samples,
        seed=args.seed,
    )

    print("\nEvaluation subset information")
    print("Eval shape:", eval_df.shape)

    print("\nEval task distribution:")
    print(eval_df["Task"].value_counts(dropna=False))

    print("\nEval size distribution:")
    print(eval_df["Size"].value_counts(dropna=False).sort_index())

    results = []

    # --------------------------------------------------------
    # 5. Evaluate each model
    # --------------------------------------------------------
    for config in model_configs:
        model_name = config["name"]

        print("\n" + "=" * 80)
        print(f"Evaluating model: {model_name}")
        print("=" * 80)

        model = load_eval_model(config)

        for i, row in tqdm(
            eval_df.iterrows(),
            total=len(eval_df),
            desc=f"TSQA {model_name}",
        ):
            task = str(row["Task"])
            size = int(row["Size"])
            label = str(row["Label"])
            question = str(row["Question"])
            answer = str(row["Answer"])
            series = row["Series"]

            actual_size = len(series)

            model_output, pred_choice, parse_success, error = safe_analyze(
                model=model,
                question=question,
                series=series,
            )

            gold_choice = extract_choice(answer)

            # fallback: Answerから取れない場合はLabelを見る
            if gold_choice is None:
                gold_choice = extract_choice(label)

            correct = int(pred_choice == gold_choice)

            results.append({
                "index": i,
                "model": model_name,
                "task": task,
                "size": size,
                "actual_size": actual_size,
                "label": label,
                "question": question,
                "answer": answer,
                "model_output": model_output,
                "gold_choice": gold_choice,
                "pred_choice": pred_choice,
                "parse_success": parse_success,
                "correct": correct,
                "error": error,
            })

            if i < args.debug_first_n:
                print("\n--- sample ---")
                print("Model:", model_name)
                print("Task:", task)
                print("Size:", size)
                print("Actual series length:", actual_size)
                print("Label:", label)
                print("Question:", question[:500])
                print("Gold answer:", answer)
                print("Gold choice:", gold_choice)
                print("Model output:", model_output)
                print("Pred choice:", pred_choice)
                print("Parse success:", parse_success)
                print("Correct:", correct)
                print("Error:", error)

        cleanup_model(model)

        tmp_path = os.path.join(
            args.output_dir,
            "tsqa_model_comparison_details_tmp.csv",
        )
        pd.DataFrame(results).to_csv(tmp_path, index=False)

    # --------------------------------------------------------
    # 6. Save detailed results
    # --------------------------------------------------------
    result_df = pd.DataFrame(results)

    detail_path = os.path.join(
        args.output_dir,
        "tsqa_model_comparison_details.csv",
    )
    result_df.to_csv(detail_path, index=False)

    # --------------------------------------------------------
    # 7. Accuracy summaries
    # --------------------------------------------------------
    summary_task_df = calculate_summary_by_task(result_df)
    summary_size_df = calculate_summary_by_size(result_df)
    summary_task_size_df = calculate_summary_by_task_size(result_df)
    overall_df = calculate_overall_summary(result_df)
    baseline_name = "mamba_normal_finetuned"

    if baseline_name in overall_df["model"].values:
        baseline_acc = overall_df.loc[
            overall_df["model"] == baseline_name,
            "accuracy"
        ].iloc[0]

        overall_df["accuracy_delta_vs_normal_mamba"] = (
            overall_df["accuracy"] - baseline_acc
        )

        overall_df["accuracy_improvement_vs_normal_mamba_pct"] = (
            (overall_df["accuracy"] - baseline_acc) / baseline_acc * 100.0
            if baseline_acc != 0
            else np.nan
        )
    else:
        overall_df["accuracy_delta_vs_normal_mamba"] = np.nan
        overall_df["accuracy_improvement_vs_normal_mamba_pct"] = np.nan
    confusion_df = calculate_confusion_matrix(result_df)

    summary_task_path = os.path.join(
        args.output_dir,
        "tsqa_model_comparison_summary_by_task.csv",
    )
    summary_size_path = os.path.join(
        args.output_dir,
        "tsqa_model_comparison_summary_by_size.csv",
    )
    summary_task_size_path = os.path.join(
        args.output_dir,
        "tsqa_model_comparison_summary_by_task_size.csv",
    )
    overall_path = os.path.join(
        args.output_dir,
        "tsqa_model_comparison_overall.csv",
    )
    confusion_path = os.path.join(
        args.output_dir,
        "tsqa_model_comparison_confusion_matrix.csv",
    )

    summary_task_df.to_csv(summary_task_path, index=False)
    summary_size_df.to_csv(summary_size_path, index=False)
    summary_task_size_df.to_csv(summary_task_size_path, index=False)
    overall_df.to_csv(overall_path, index=False)
    confusion_df.to_csv(confusion_path, index=False)
    accuracy_on_parsed_df = calculate_accuracy_on_parsed(result_df)

    accuracy_on_parsed_path = os.path.join(
        args.output_dir,
        "tsqa_model_comparison_accuracy_on_parsed.csv",
    )

    accuracy_on_parsed_df.to_csv(accuracy_on_parsed_path, index=False)

    print(" -", accuracy_on_parsed_path)
    print("\nOverall summary:")
    print(overall_df)

    print("\nSummary by task:")
    print(summary_task_df)

    print("\nSummary by size:")
    print(summary_size_df)

    print("\nSaved:")
    print(" -", detail_path)
    print(" -", overall_path)
    print(" -", summary_task_path)
    print(" -", summary_size_path)
    print(" -", summary_task_size_path)
    print(" -", confusion_path)


if __name__ == "__main__":
    main()