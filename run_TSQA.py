import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import re
import ast
import gc
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from transformers import logging

logging.set_verbosity_error()

from model.model import ChatTime


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
            random_state=seed
        )

        sampled_list.append(sampled)

    eval_df = pd.concat(sampled_list, axis=0).reset_index(drop=True)

    if len(eval_df) > max_eval_samples:
        eval_df = eval_df.sample(
            n=max_eval_samples,
            random_state=seed
        ).reset_index(drop=True)

    return eval_df


def calculate_summary_by_task(result_df):
    """
    Accuracy by Task.
    """
    summary_by_task = (
        result_df
        .groupby("task", as_index=False)
        .agg(
            accuracy=("correct", "mean"),
            n_samples=("correct", "count"),
            n_correct=("correct", "sum")
        )
    )

    overall = pd.DataFrame([{
        "task": "ALL",
        "accuracy": result_df["correct"].mean(),
        "n_samples": result_df["correct"].count(),
        "n_correct": result_df["correct"].sum()
    }])

    return pd.concat([summary_by_task, overall], ignore_index=True)


def calculate_summary_by_size(result_df):
    """
    Accuracy by input series length Size.
    """
    summary_by_size = (
        result_df
        .groupby("size", as_index=False)
        .agg(
            accuracy=("correct", "mean"),
            n_samples=("correct", "count"),
            n_correct=("correct", "sum")
        )
        .sort_values("size")
        .reset_index(drop=True)
    )

    overall = pd.DataFrame([{
        "size": "ALL",
        "accuracy": result_df["correct"].mean(),
        "n_samples": result_df["correct"].count(),
        "n_correct": result_df["correct"].sum()
    }])

    return pd.concat([summary_by_size, overall], ignore_index=True)


def calculate_summary_by_task_size(result_df):
    """
    Accuracy by Task and Size.
    """
    summary_by_task_size = (
        result_df
        .groupby(["task", "size"], as_index=False)
        .agg(
            accuracy=("correct", "mean"),
            n_samples=("correct", "count"),
            n_correct=("correct", "sum")
        )
        .sort_values(["task", "size"])
        .reset_index(drop=True)
    )

    return summary_by_task_size


# ============================================================
# Main
# ============================================================

def main():
    model_path = "ChengsenWang/ChatTime-1-7B-Chat"
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    # --------------------------------------------------------
    # Evaluation setting
    # --------------------------------------------------------
    # 最初は100程度で確認。
    # 全件評価したい場合は None にする。
    max_eval_samples = 500

    # --------------------------------------------------------
    # 1. Load Hugging Face TSQA dataset
    # --------------------------------------------------------
    print("Loading dataset from Hugging Face: ChengsenWang/TSQA")

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

    if df["Task"].isna().any():
        print("\n[Warning] NaN found in Task column. Dropping those rows.")
        df = df.dropna(subset=["Task"]).reset_index(drop=True)

    if df["Size"].isna().any():
        print("\n[Warning] NaN found in Size column. Dropping those rows.")
        df = df.dropna(subset=["Size"]).reset_index(drop=True)

    df["Task"] = df["Task"].astype(str)
    df["Size"] = df["Size"].astype(int)

    # --------------------------------------------------------
    # 3. Create evaluation subset
    # --------------------------------------------------------
    eval_df = make_eval_subset(
        df=df,
        max_eval_samples=max_eval_samples,
        seed=42
    )

    print("\nEvaluation subset information")
    print("Eval shape:", eval_df.shape)

    print("\nEval task distribution:")
    print(eval_df["Task"].value_counts(dropna=False))

    print("\nEval size distribution:")
    print(eval_df["Size"].value_counts(dropna=False).sort_index())

    if eval_df["Task"].isna().any():
        raise ValueError("Task contains NaN in eval_df.")

    if (eval_df["Task"].astype(str) == "None").any():
        raise ValueError("Task contains string 'None' in eval_df.")

    # --------------------------------------------------------
    # 4. Load ChatTime model
    # --------------------------------------------------------
    print("\nLoading ChatTime model...")
    model = ChatTime(model_path=model_path)

    results = []

    # --------------------------------------------------------
    # 5. Evaluation loop
    # --------------------------------------------------------
    print("\nStart TSQA evaluation...")

    for i, row in tqdm(eval_df.iterrows(), total=len(eval_df)):
        task = str(row["Task"])
        size = int(row["Size"])
        label = str(row["Label"])
        question = str(row["Question"])
        answer = str(row["Answer"])
        series = parse_series(row["Series"])

        # 念のため、Size列と実際のSeries長を比較する
        actual_size = len(series)

        try:
            model_output = model.analyze(question, series)
        except Exception as e:
            model_output = ""
            print(f"\n[Error] index={i}, task={task}, size={size}, error={e}")

        gold_choice = extract_choice(answer)
        pred_choice = extract_choice(model_output)

        correct = int(pred_choice == gold_choice)

        results.append({
            "index": i,
            "task": task,
            "size": size,
            "actual_size": actual_size,
            "label": label,
            "question": question,
            "answer": answer,
            "model_output": model_output,
            "gold_choice": gold_choice,
            "pred_choice": pred_choice,
            "correct": correct
        })

        if i < 3:
            print("\n--- sample ---")
            print("Task:", task)
            print("Size:", size)
            print("Actual series length:", actual_size)
            print("Label:", label)
            print("Question:", question[:500])
            print("Gold answer:", answer)
            print("Gold choice:", gold_choice)
            print("Model output:", model_output)
            print("Pred choice:", pred_choice)
            print("Correct:", correct)

    # --------------------------------------------------------
    # 6. Save detailed results
    # --------------------------------------------------------
    result_df = pd.DataFrame(results)

    print("\nResult task distribution:")
    print(result_df["task"].value_counts(dropna=False))

    print("\nResult size distribution:")
    print(result_df["size"].value_counts(dropna=False).sort_index())

    if result_df["task"].isna().any():
        print("\n[Warning] NaN exists in result_df['task'].")

    if (result_df["task"].astype(str) == "None").any():
        print("\n[Warning] String 'None' exists in result_df['task'].")

    detail_path = os.path.join(output_dir, "chattime_tsqa_hf_details.csv")
    result_df.to_csv(detail_path, index=False)

    # --------------------------------------------------------
    # 7. Accuracy summaries
    # --------------------------------------------------------
    summary_task_df = calculate_summary_by_task(result_df)
    summary_size_df = calculate_summary_by_size(result_df)
    summary_task_size_df = calculate_summary_by_task_size(result_df)

    summary_task_path = os.path.join(
        output_dir,
        "chattime_tsqa_hf_summary_by_task.csv"
    )
    summary_size_path = os.path.join(
        output_dir,
        "chattime_tsqa_hf_summary_by_size.csv"
    )
    summary_task_size_path = os.path.join(
        output_dir,
        "chattime_tsqa_hf_summary_by_task_size.csv"
    )

    summary_task_df.to_csv(summary_task_path, index=False)
    summary_size_df.to_csv(summary_size_path, index=False)
    summary_task_size_df.to_csv(summary_task_size_path, index=False)

    print("\nSummary by task:")
    print(summary_task_df)

    print("\nSummary by size:")
    print(summary_size_df)

    print("\nSummary by task and size:")
    print(summary_task_size_df)

    print("\nSaved:")
    print(" -", detail_path)
    print(" -", summary_task_path)
    print(" -", summary_size_path)
    print(" -", summary_task_size_path)

    # --------------------------------------------------------
    # 8. Clean up
    # --------------------------------------------------------
    del model
    gc.collect()


if __name__ == "__main__":
    main()