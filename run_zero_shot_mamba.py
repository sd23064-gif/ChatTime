import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import ast
import re
import gc
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
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


def safe_inverse_serialize(serializer, text):
    pattern = rf"{re.escape(serializer.time_flag)}(.*?){re.escape(serializer.time_flag)}"
    matches = re.findall(pattern, text)

    values = []
    for m in matches:
        m = str(m).strip()
        if m == "":
            values.append(np.nan)
            continue
        if m.lower() == serializer.nan_flag.lower():
            values.append(np.nan)
            continue
        try:
            values.append(float(m))
        except Exception:
            values.append(np.nan)

    return np.asarray(values, dtype=np.float64)


def mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    n = min(len(y_true), len(y_pred))
    y_true = y_true[:n]
    y_pred = y_pred[:n]

    return np.nanmean(np.abs(y_true - y_pred))


def chronological_split(df, train_ratio=0.6, val_ratio=0.2):
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end].reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].reset_index(drop=True)
    test_df = df.iloc[val_end:].reset_index(drop=True)

    return train_df, val_df, test_df


def get_train_std(train_df):
    values = np.concatenate(train_df["Pred"].values)
    std = np.nanstd(values)
    if std == 0 or np.isnan(std):
        std = 1.0
    return std


def make_eval_subset(test_df, max_eval_windows=100):
    if max_eval_windows is None:
        return test_df.reset_index(drop=True)

    if len(test_df) <= max_eval_windows:
        return test_df.reset_index(drop=True)

    idx = np.linspace(0, len(test_df) - 1, max_eval_windows, dtype=int)
    return test_df.iloc[idx].reset_index(drop=True)


def load_dataset_any(dataset_path):
    """
    local csv or HF dataset/csv.
    """
    if dataset_path.endswith(".csv") and os.path.exists(dataset_path):
        return load_dataset("csv", data_files=dataset_path, split="train").to_pandas()

    if "::" in dataset_path:
        repo_id, csv_path = dataset_path.split("::", 1)
        ds = load_dataset(
            "csv",
            data_files=f"hf://datasets/{repo_id}/{csv_path}",
            split="train",
        )
        return ds.to_pandas()

    ds = load_dataset(dataset_path, split="train")
    return ds.to_pandas()


def load_mamba_model(base_model_path, adapter_path, use_bf16=False):
    """
    PEFT adapter形式のChatTime-Mambaを読み込む。
    """
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
    print("Tokenizer vocab size:", len(tokenizer))

    return model, tokenizer


def predict_mamba(
    model,
    tokenizer,
    hist_data,
    pred_len,
    context=None,
    max_pred_len=24,
    num_samples=8,
    max_new_tokens_per_point=2,
):
    """
    ChatTimeのpredict()相当。
    複数生成して時刻ごとの中央値を取る。
    """
    discretizer = Discretizer()
    serializer = Serializer()

    series = np.asarray(hist_data, dtype=np.float64)
    prediction_list = []
    remaining = pred_len

    device = next(model.parameters()).device

    while remaining > 0:
        target_len = min(remaining, max_pred_len)

        # 重要: 初期履歴長だけでfitする方が再現実験として安全
        fit_length = len(hist_data)

        dispersed_series = discretizer.discretize(series, fit_length=fit_length)
        serialized_series = serializer.serialize(dispersed_series)

        prompt = getPrompt(
            flag="prediction",
            context=context,
            input=serialized_series,
        )

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=2048,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        pred_list = []

        for _ in range(num_samples):
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens_per_point * target_len + 16,
                    do_sample=True,
                    top_k=100,
                    top_p=1.0,
                    temperature=1.0,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )

            generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)

            if "### Response:\n" in generated_text:
                response_text = generated_text.split("### Response:\n", 1)[1]
            else:
                response_text = generated_text

            dispersed_pred = safe_inverse_serialize(serializer, response_text)

            if len(dispersed_pred) == 0:
                continue

            pred = discretizer.inverse_discretize(dispersed_pred)

            if len(pred) < target_len:
                pred = np.concatenate([
                    pred,
                    np.full(target_len - len(pred), np.nan),
                ])

            pred_list.append(pred[:target_len])

        if len(pred_list) == 0:
            prediction = np.full(target_len, np.nan)
        else:
            prediction = np.nanmedian(pred_list, axis=0)

        prediction_list.append(prediction)

        remaining -= len(prediction)

        if remaining <= 0:
            break

        series = np.concatenate([series, prediction], axis=-1)

    return np.concatenate(prediction_list, axis=-1)[:pred_len]


def main():
    # =========================
    # Settings
    # =========================
    dataset_name = "PTF"

    # local CSVなら: /workspace/dataset/PTF.csv
    # HF CGTSFなら: ChengsenWang/CGTSF::PTF/PTF.csv
    dataset_path = "ChengsenWang/CGTSF::PTF/PTF.csv"

    base_model_path = "state-spaces/mamba-370m-hf"
    adapter_path = "/workspace/outputs/models/ChatTime-Mamba-Chat"

    output_dir = "/workspace/outputs/mamba_forecasting_eval"
    os.makedirs(output_dir, exist_ok=True)

    hist_lengths = [48, 72, 96, 120]
    pred_len = 24

    max_eval_windows = 100
    use_bf16 = False

    # =========================
    # Load dataset
    # =========================
    print("Loading dataset:", dataset_path)
    df = load_dataset_any(dataset_path)
    df.columns = df.columns.str.replace("\ufeff", "", regex=False).str.strip()

    required_cols = ["Hist", "Pred"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["Hist"] = df["Hist"].apply(parse_series)
    df["Pred"] = df["Pred"].apply(parse_series)

    has_text = "Text" in df.columns

    print("Dataset shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("Has Text context:", has_text)

    train_df, val_df, test_df = chronological_split(df)
    train_std = get_train_std(train_df)

    eval_df = make_eval_subset(test_df, max_eval_windows=max_eval_windows)

    print("Eval shape:", eval_df.shape)
    print("Train std:", train_std)

    # =========================
    # Load model
    # =========================
    model, tokenizer = load_mamba_model(
        base_model_path=base_model_path,
        adapter_path=adapter_path,
        use_bf16=use_bf16,
    )

    results = []

    # =========================
    # Evaluation
    # =========================
    for hist_len in hist_lengths:
        print(f"\nEvaluating hist_len={hist_len}, pred_len={pred_len}")

        for pos, row in tqdm(eval_df.iterrows(), total=len(eval_df)):
            hist_full = row["Hist"]
            true_full = row["Pred"]

            hist_data = hist_full[-hist_len:]
            true_data = true_full[:pred_len]

            if len(hist_data) < hist_len or len(true_data) < pred_len:
                continue

            if np.isnan(hist_data).any() or np.isnan(true_data).any():
                continue

            context = row["Text"] if has_text else None

            error_zero = ""
            error_context = ""

            try:
                pred_zero = predict_mamba(
                    model=model,
                    tokenizer=tokenizer,
                    hist_data=hist_data,
                    pred_len=pred_len,
                    context=None,
                    max_pred_len=pred_len,
                    num_samples=8,
                )
            except Exception as e:
                pred_zero = np.full(pred_len, np.nan)
                error_zero = str(e)

            if has_text:
                try:
                    pred_context = predict_mamba(
                        model=model,
                        tokenizer=tokenizer,
                        hist_data=hist_data,
                        pred_len=pred_len,
                        context=context,
                        max_pred_len=pred_len,
                        num_samples=8,
                    )
                except Exception as e:
                    pred_context = np.full(pred_len, np.nan)
                    error_context = str(e)
            else:
                pred_context = np.full(pred_len, np.nan)
                error_context = "No Text column"

            raw_mae_zero = mae(true_data, pred_zero)
            std_mae_zero = raw_mae_zero / train_std

            raw_mae_context = mae(true_data, pred_context)
            std_mae_context = raw_mae_context / train_std

            results.append({
                "dataset": dataset_name,
                "hist_len": hist_len,
                "pred_len": pred_len,
                "position": pos,
                "raw_mae_zero": raw_mae_zero,
                "std_mae_zero": std_mae_zero,
                "raw_mae_context": raw_mae_context,
                "std_mae_context": std_mae_context,
                "nan_rate_zero": np.isnan(pred_zero).mean(),
                "nan_rate_context": np.isnan(pred_context).mean(),
                "error_zero": error_zero,
                "error_context": error_context,
            })

    result_df = pd.DataFrame(results)

    detail_path = os.path.join(output_dir, "mamba_forecasting_details.csv")
    result_df.to_csv(detail_path, index=False)

    summary_df = (
        result_df
        .groupby(["dataset", "hist_len", "pred_len"], as_index=False)
        .agg(
            raw_mae_zero=("raw_mae_zero", "mean"),
            std_mae_zero=("std_mae_zero", "mean"),
            raw_mae_context=("raw_mae_context", "mean"),
            std_mae_context=("std_mae_context", "mean"),
            nan_rate_zero=("nan_rate_zero", "mean"),
            nan_rate_context=("nan_rate_context", "mean"),
            n_samples=("raw_mae_zero", "count"),
            n_error_zero=("error_zero", lambda x: (x != "").sum()),
            n_error_context=("error_context", lambda x: (x != "").sum()),
        )
    )

    summary_path = os.path.join(output_dir, "mamba_forecasting_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print("\nSummary:")
    print(summary_df)

    print("\nSaved:")
    print(detail_path)
    print(summary_path)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()