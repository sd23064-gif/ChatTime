import os
import re
import gc
import argparse
from statistics import mode

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from utils.prompt import getPrompt
from utils.tools import Discretizer, Serializer

import matplotlib.pyplot as plt
np.NaN = np.nan

def plot_prediction_window(
    hist_data,
    true_data,
    pred_data,
    dataset_name,
    column,
    hist_len,
    pred_len,
    start_index,
    save_path,
):
    hist_data = np.asarray(hist_data, dtype=np.float64)
    true_data = np.asarray(true_data, dtype=np.float64)
    pred_data = np.asarray(pred_data, dtype=np.float64)

    x_hist = np.arange(hist_len)
    x_future = np.arange(hist_len, hist_len + pred_len)

    plt.figure(figsize=(14, 4))

    plt.plot(
        x_hist,
        hist_data,
        color="black",
        linewidth=1.5,
        label="history",
    )

    plt.plot(
        x_future,
        true_data,
        color="blue",
        linewidth=1.5,
        label="true future",
    )

    plt.plot(
        x_future,
        pred_data,
        color="orange",
        linewidth=1.5,
        label="predicted future",
    )

    plt.axvline(
        x=hist_len,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label="prediction start",
    )

    plt.title(
        f"{dataset_name} prediction: column={column}, "
        f"hist_len={hist_len}, pred_len={pred_len}, start={start_index}"
    )
    plt.xlabel("sample")
    plt.ylabel("value")
    plt.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()

def mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return np.mean(np.abs(y_true - y_pred))


def mse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return np.mean((y_true - y_pred) ** 2)


def rmse(y_true, y_pred):
    return np.sqrt(mse(y_true, y_pred))


def chronological_split(df, train_ratio=0.6, val_ratio=0.2):
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    return train_df, val_df, test_df


class MambaChatTime:
    def __init__(
        self,
        base_model_path,
        adapter_path=None,
        hist_len=None,
        pred_len=None,
        max_pred_len=16,
        num_samples=8,
        top_k=100,
        top_p=1.0,
        temperature=1.0,
        torch_dtype=torch.float16,
    ):
        self.base_model_path = base_model_path
        self.adapter_path = adapter_path

        self.hist_len = hist_len
        self.pred_len = pred_len

        self.max_pred_len = max_pred_len
        self.num_samples = num_samples
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature

        self.discretizer = Discretizer()
        self.serializer = Serializer()

        if adapter_path is not None and adapter_path != "":
            tokenizer_path = adapter_path
        else:
            tokenizer_path = base_model_path

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.tokenizer.padding_side = "right"
        self.eos_token_id = self.tokenizer.eos_token_id

        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch_dtype,
            device_map={"": 0},
            trust_remote_code=True,
        )

        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.use_cache = True

        if adapter_path is not None and adapter_path != "":
            print(f"Loading adapter: {adapter_path}")
            self.model = PeftModel.from_pretrained(
                self.model,
                adapter_path,
                is_trainable=False,
            )

        self.model.eval()

        print("\nLoaded MambaChatTime")
        print("Base model:", base_model_path)
        print("Adapter:", adapter_path)
        print("Tokenizer size:", len(self.tokenizer))
        print("Embedding size:", self.model.get_input_embeddings().weight.shape[0])

    @torch.no_grad()
    def predict(self, hist_data, context=None):
        if self.hist_len is None or self.pred_len is None:
            raise ValueError("hist_len and pred_len must be specified before prediction")

        series = np.asarray(hist_data, dtype=np.float64)

        if len(series) == 0:
            return np.full(self.pred_len, np.NaN, dtype=np.float64)

        prediction_list = []
        remaining = self.pred_len

        while remaining > 0:
            current_pred_len = min(remaining, self.max_pred_len)

            if len(series) == 0:
                prediction = np.full(current_pred_len, np.NaN, dtype=np.float64)
                prediction_list.append(prediction)
                remaining -= current_pred_len
                continue

            dispersed_series = self.discretizer.discretize(series)
            serialized_series = self.serializer.serialize(dispersed_series)

            prompt = getPrompt(
                flag="prediction",
                context=context,
                input=serialized_series,
            )

            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                add_special_tokens=False,
            ).to(self.model.device)

            min_new_tokens = 2 * current_pred_len + 8
            max_new_tokens = 2 * current_pred_len + 8

            outputs = self.model.generate(
                **inputs,
                min_new_tokens=min_new_tokens,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                num_return_sequences=self.num_samples,
                top_k=self.top_k,
                top_p=self.top_p,
                temperature=self.temperature,
                eos_token_id=self.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )

            decoded_outputs = self.tokenizer.batch_decode(
                outputs,
                skip_special_tokens=False,
            )

            # =========================
            # Debug: prompt確認
            # =========================
            print("\n========== DEBUG PROMPT ==========")
            print(prompt)
            print("========== END DEBUG PROMPT ==========\n")

            pred_list = []

            for i, generated_text in enumerate(decoded_outputs):
                if i == 0:
                    print("\n========== DEBUG PROMPT ==========")
                    print(prompt)
                    print("========== END DEBUG PROMPT ==========\n")

                    print(f"\n========== DEBUG GENERATED TEXT sample {i} ==========")
                    print(generated_text)
                    print(f"========== END DEBUG GENERATED TEXT sample {i} ==========\n")

                if "### Response:\n" in generated_text:
                    serialized_prediction = generated_text.split("### Response:\n", 1)[1]
                elif "### Response:" in generated_text:
                    serialized_prediction = generated_text.split("### Response:", 1)[1]
                else:
                    input_len = inputs["input_ids"].shape[1]
                    generated_ids = outputs[i][input_len:]
                    serialized_prediction = self.tokenizer.decode(
                        generated_ids,
                        skip_special_tokens=False,
                    )

                if i == 0:
                    print(f"\n========== DEBUG SERIALIZED PREDICTION sample {i} ==========")
                    print(serialized_prediction)
                    print(f"========== END DEBUG SERIALIZED PREDICTION sample {i} ==========\n")

                dispersed_prediction = self.serializer.inverse_serialize(
                    serialized_prediction
                )

                if i == 0:
                    print(f"DEBUG parsed length sample {i}: {len(dispersed_prediction)}")
                    print(f"DEBUG parsed values sample {i}: {dispersed_prediction[:min(10, len(dispersed_prediction))]}")

                if dispersed_prediction is None or len(dispersed_prediction) == 0:
                    pred = np.full(current_pred_len, np.NaN, dtype=np.float64)
                else:
                    pred = self.discretizer.inverse_discretize(
                        dispersed_prediction
                    )

                    if len(pred) < current_pred_len:
                        pred = np.concatenate([
                            pred,
                            np.full(current_pred_len - len(pred), np.NaN),
                        ])

                pred_list.append(pred[:current_pred_len])

            prediction = np.nanmedian(pred_list, axis=0)

            if np.isnan(prediction).all():
                prediction = np.full(current_pred_len, np.NaN, dtype=np.float64)

            prediction_list.append(prediction)
            remaining -= prediction.shape[-1]

            if remaining <= 0:
                break

            # NaNだけの予測を series に追加すると次の scaler が壊れやすいので停止
            if np.isnan(prediction).all():
                break

            series = np.concatenate([series, prediction], axis=-1)

        if len(prediction_list) == 0:
            return np.full(self.pred_len, np.NaN, dtype=np.float64)

        prediction = np.concatenate(prediction_list, axis=-1)

        if len(prediction) < self.pred_len:
            prediction = np.concatenate([
                prediction,
                np.full(self.pred_len - len(prediction), np.NaN),
            ])

        return prediction[:self.pred_len]

    def close(self):
        del self.model
        del self.tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_path", type=str, default="/workspace/dataset/ETTh2.csv")
    parser.add_argument("--dataset_name", type=str, default="ETTh2")

    parser.add_argument(
        "--base_model_path",
        type=str,
        default="/workspace/outputs/model/ChatTime-Mamba-2.8B-fast-b4-merged",
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        default="/workspace/outputs/model/finetune-no-kernel-alltarget-lr1e-6",
    )

    parser.add_argument("--pred_len", type=int, default=24)
    parser.add_argument("--hist_lengths", type=int, nargs="+", default=[48, 72, 96, 120])
    parser.add_argument("--max_eval_windows", type=int, default=50)

    parser.add_argument("--max_pred_len", type=int, default=16)
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)

    parser.add_argument("--output_dir", type=str, default="/workspace/outputs/eval/mamba_chattime_etth2")
    parser.add_argument("--fp32", action="store_true", default=False)

    parser.add_argument("--save_plots", action="store_true", default=False)
    parser.add_argument("--max_plots_per_hist", type=int, default=5)    
    args = parser.parse_args()

    if args.adapter_path == "":
        args.adapter_path = None

    os.makedirs(args.output_dir, exist_ok=True)

    dtype = torch.float32 if args.fp32 else torch.float16

    # =========================
    # 1. Load dataset
    # =========================
    raw_df = pd.read_csv(args.dataset_path)

    if "date" in raw_df.columns:
        value_df = raw_df.drop(columns=["date"]).apply(pd.to_numeric, errors="coerce")
    else:
        value_df = raw_df.apply(pd.to_numeric, errors="coerce")

    train_df_raw, val_df_raw, test_df_raw = chronological_split(value_df)

    print("Dataset:", args.dataset_path)
    print("Total shape:", value_df.shape)
    print("Train shape:", train_df_raw.shape)
    print("Val shape:", val_df_raw.shape)
    print("Test shape:", test_df_raw.shape)

    # =========================
    # 2. Standardization
    # train statistics only
    # =========================
    mean = train_df_raw.mean()
    std = train_df_raw.std()

    std = std.replace(0, 1.0)

    value_df_std = (value_df - mean) / std

    train_df, val_df, test_df = chronological_split(value_df_std)
    test_start = len(train_df) + len(val_df)

    results = []

    selected_columns = value_df_std.columns.tolist()

    # =========================
    # 3. Evaluation only
    # =========================
    for hist_len in args.hist_lengths:
        
        print(f"\nEvaluating hist_len={hist_len}, pred_len={args.pred_len}")
        plot_count = 0

        print(f"\nEvaluating hist_len={hist_len}, pred_len={args.pred_len}")

        model = MambaChatTime(
            hist_len=hist_len,
            pred_len=args.pred_len,
            base_model_path=args.base_model_path,
            adapter_path=args.adapter_path,
            max_pred_len=args.max_pred_len,
            num_samples=args.num_samples,
            top_k=args.top_k,
            top_p=args.top_p,
            temperature=args.temperature,
            torch_dtype=dtype,
        )

        for col in tqdm(selected_columns, desc=f"hist_len={hist_len}"):
            series = value_df_std[col].to_numpy(dtype=np.float64)

            possible_starts = list(
                range(
                    test_start,
                    len(series) - args.pred_len + 1,
                    args.pred_len,
                )
            )

            if args.max_eval_windows is not None and args.max_eval_windows > 0:
                possible_starts = possible_starts[:args.max_eval_windows]

            for start in possible_starts:
                hist_start = start - hist_len
                hist_end = start
                pred_start = start
                pred_end = start + args.pred_len

                if hist_start < 0:
                    continue

                hist_data = series[hist_start:hist_end]
                true_data = series[pred_start:pred_end]

                if np.isnan(hist_data).any() or np.isnan(true_data).any():
                    continue

                pred_data = model.predict(hist_data)

                valid_prediction = (
                    pred_data is not None
                    and len(pred_data) == args.pred_len
                    and not np.isnan(pred_data).any()
                )

                if valid_prediction:
                    score_mae = mae(true_data, pred_data)
                    score_mse = mse(true_data, pred_data)
                    score_rmse = rmse(true_data, pred_data)
                else:
                    score_mae = np.nan
                    score_mse = np.nan
                    score_rmse = np.nan
                
                if (
                    args.save_plots
                    and valid_prediction
                    and plot_count < args.max_plots_per_hist
                ):
                    plot_dir = os.path.join(args.output_dir, "plots")

                    safe_col = str(col).replace("/", "_").replace("\\", "_").replace(" ", "_")

                    plot_path = os.path.join(
                        plot_dir,
                        f"{args.dataset_name}_col-{safe_col}_hist-{hist_len}_"
                        f"pred-{args.pred_len}_start-{start}.png"
                    )

                    plot_prediction_window(
                        hist_data=hist_data,
                        true_data=true_data,
                        pred_data=pred_data,
                        dataset_name=args.dataset_name,
                        column=col,
                        hist_len=hist_len,
                        pred_len=args.pred_len,
                        start_index=start,
                        save_path=plot_path,
                    )

                    plot_count += 1

                results.append({
                    "dataset": args.dataset_name,
                    "column": col,
                    "hist_len": hist_len,
                    "pred_len": args.pred_len,
                    "start_index": start,
                    "mae": score_mae,
                    "mse": score_mse,
                    "rmse": score_rmse,
                    "valid_prediction": valid_prediction,
                })

        model.close()

    # =========================
    # 4. Save detailed results
    # =========================
    result_df = pd.DataFrame(results)

    detail_path = os.path.join(
        args.output_dir,
        "mamba_chattime_etth2_eval_details.csv",
    )
    result_df.to_csv(detail_path, index=False)

    valid_df = result_df[result_df["valid_prediction"] == True].copy()

    # =========================
    # 5. Summary
    # =========================
    summary_df = (
        valid_df
        .groupby(["dataset", "hist_len", "pred_len"], as_index=False)
        .agg(
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            mse_mean=("mse", "mean"),
            mse_std=("mse", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            n_samples=("mae", "count"),
        )
    )

    summary_path = os.path.join(
        args.output_dir,
        "mamba_chattime_etth2_eval_summary.csv",
    )
    summary_df.to_csv(summary_path, index=False)

    # =========================
    # 6. Summary by column
    # =========================
    summary_col_df = (
        valid_df
        .groupby(["dataset", "column", "hist_len", "pred_len"], as_index=False)
        .agg(
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            mse_mean=("mse", "mean"),
            mse_std=("mse", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            n_samples=("mae", "count"),
        )
    )

    summary_col_path = os.path.join(
        args.output_dir,
        "mamba_chattime_etth2_eval_summary_by_column.csv",
    )
    summary_col_df.to_csv(summary_col_path, index=False)

    invalid_ratio = 1.0 - result_df["valid_prediction"].mean()

    print("\nMambaChatTime Evaluation Summary")
    print(summary_df)

    print("\nInvalid prediction ratio:", invalid_ratio)

    print("\nSaved:")
    print(" -", detail_path)
    print(" -", summary_path)
    print(" -", summary_col_path)


if __name__ == "__main__":
    main()