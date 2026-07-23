import argparse
from pathlib import Path
import re

import numpy as np

if not hasattr(np, "NaN"):
    np.NaN = np.nan

import pandas as pd
import torch
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transformers import AutoModelForCausalLM, AutoTokenizer


TOKEN_PATTERN = r"###(Nan|[-+]?\d+(?:\.\d+)?)###"


WINDOW_CONFIGS = {
    576: {"hist_len": 512, "pred_len": 64},
    288: {"hist_len": 256, "pred_len": 32},
    144: {"hist_len": 128, "pred_len": 16},
    72:  {"hist_len": 64,  "pred_len": 8},
    36:  {"hist_len": 32,  "pred_len": 4},
}


def parse_chattime_values(text):
    """
    ###-0.4159### ###0.1234### のような文字列から数値列を抽出する。
    """
    matches = re.findall(TOKEN_PATTERN, str(text))

    values = []
    for m in matches:
        if m == "Nan":
            values.append(np.nan)
        else:
            try:
                values.append(float(m))
            except ValueError:
                values.append(np.nan)

    return np.asarray(values, dtype=np.float32)


def values_to_chattime_text(values, prec=4, time_flag="###", time_sep=" "):
    tokens = []
    for v in values:
        if np.isnan(v):
            tokens.append(f"{time_flag}Nan{time_flag}")
        else:
            tokens.append(f"{time_flag}{v:.{prec}f}{time_flag}")
    return time_sep.join(tokens)


def calc_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    valid = np.isfinite(y_true) & np.isfinite(y_pred)

    if valid.sum() == 0:
        return {
            "mae": np.nan,
            "rmse": np.nan,
            "nrmse": np.nan,
            "r2": np.nan,
            "corr": np.nan,
        }

    y_true = y_true[valid]
    y_pred = y_pred[valid]

    err = y_pred - y_true

    mae = np.mean(np.abs(err))
    rmse = np.sqrt(np.mean(err ** 2))

    amp = np.max(y_true) - np.min(y_true)
    nrmse = rmse / (amp + 1e-8)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1.0 - ss_res / (ss_tot + 1e-8)

    if np.std(y_true) < 1e-8 or np.std(y_pred) < 1e-8:
        corr = np.nan
    else:
        corr = np.corrcoef(y_true, y_pred)[0, 1]

    return {
        "mae": mae,
        "rmse": rmse,
        "nrmse": nrmse,
        "r2": r2,
        "corr": corr,
    }


def make_baseline_predictions(hist, pred_len):
    hist = np.asarray(hist, dtype=np.float32)

    last_value = np.full(pred_len, hist[-1], dtype=np.float32)

    if len(hist) >= pred_len:
        repeat = hist[-pred_len:].astype(np.float32)
    else:
        repeat = np.resize(hist, pred_len).astype(np.float32)

    return {
        "last_value": last_value,
        "repeat": repeat,
    }


class B1CSVForecaster:
    def __init__(
        self,
        model_path,
        num_samples=1,
        top_k=50,
        top_p=0.9,
        temperature=0.7,
        max_new_ratio=4,
        prec=4,
        debug=False,
    ):
        self.model_path = model_path
        self.num_samples = num_samples
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature
        self.max_new_ratio = max_new_ratio
        self.prec = prec
        self.debug = debug

        use_cuda = torch.cuda.is_available()

        if use_cuda and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        elif use_cuda:
            dtype = torch.float16
        else:
            dtype = torch.float32

        print("Loading model:", model_path)
        print("dtype:", dtype)
        print("cuda:", use_cuda)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=True,
            local_files_only=True,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map="auto" if use_cuda else None,
            trust_remote_code=True,
            local_files_only=True,
        )

        self.model.eval()

        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id

    def _device(self):
        return next(self.model.parameters()).device

    def generate_future_values(self, hist_values, pred_len):
        prompt_text = values_to_chattime_text(
            hist_values,
            prec=self.prec,
        )

        inputs = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            add_special_tokens=False,
        )
        inputs = {k: v.to(self._device()) for k, v in inputs.items()}

        input_len = inputs["input_ids"].shape[-1]

        do_sample = self.num_samples > 1 and self.temperature > 0

        generation_kwargs = dict(
            **inputs,
            max_new_tokens=self.max_new_ratio * pred_len + 16,
            min_new_tokens=pred_len,
            do_sample=do_sample,
            num_return_sequences=self.num_samples,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
        )

        if do_sample:
            generation_kwargs.update(
                top_k=self.top_k,
                top_p=self.top_p,
                temperature=self.temperature,
            )

        with torch.no_grad():
            output_ids = self.model.generate(**generation_kwargs)

        pred_list = []

        for ids in output_ids:
            new_ids = ids[input_len:]
            generated_text = self.tokenizer.decode(
                new_ids,
                skip_special_tokens=True,
            )

            pred_values = parse_chattime_values(generated_text)

            if self.debug:
                print("\n=== prompt tail ===")
                print(prompt_text[-500:])
                print("\n=== generated_text ===")
                print(repr(generated_text[:1000]))
                print("parsed:", pred_values[:20])
                print("num parsed:", len(pred_values))

            if len(pred_values) == 0:
                pred = np.full(pred_len, np.nan, dtype=np.float32)
            else:
                pred = pred_values[:pred_len]

                if len(pred) < pred_len:
                    pad_len = pred_len - len(pred)
                    pred = np.concatenate(
                        [
                            pred,
                            np.full(pad_len, np.nan, dtype=np.float32),
                        ]
                    )

            pred_list.append(pred.astype(np.float32))

        pred_arr = np.asarray(pred_list, dtype=np.float32)

        if np.isfinite(pred_arr).sum() == 0:
            return np.full(pred_len, np.nan, dtype=np.float32)

        return np.nanmedian(pred_arr, axis=0).astype(np.float32)


def save_plot(row_id, hist, y_true, pred_dict, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    hist_x = np.arange(len(hist))
    pred_x = np.arange(len(hist), len(hist) + len(y_true))

    colors = {
        "chattime": "orange",
        "last_value": "green",
        "repeat": "purple",
    }

    plt.figure(figsize=(12, 4))
    plt.plot(hist_x, hist, label="history", color="black")
    plt.plot(pred_x, y_true, label="true future", color="blue")

    for name, pred in pred_dict.items():
        plt.plot(
            pred_x,
            pred,
            label=name,
            color=colors.get(name, None),
        )

    plt.axvline(len(hist) - 1, color="red", linestyle="--")
    plt.title(f"B-1 test CSV prediction row={row_id}")
    plt.xlabel("token index")
    plt.ylabel("discretized value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"row_{row_id}_full.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 4))
    future_x = np.arange(len(y_true))
    plt.plot(future_x, y_true, label="true future", color="blue")

    for name, pred in pred_dict.items():
        plt.plot(
            future_x,
            pred,
            label=name,
            color=colors.get(name, None),
        )

    plt.title(f"B-1 future only row={row_id}")
    plt.xlabel("future token index")
    plt.ylabel("discretized value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"row_{row_id}_future_only.png", dpi=200)
    plt.close()


def infer_split_lengths(values_len):
    if values_len in WINDOW_CONFIGS:
        cfg = WINDOW_CONFIGS[values_len]
        return cfg["hist_len"], cfg["pred_len"]

    raise ValueError(
        f"Unknown sequence length={values_len}. "
        f"Expected one of {list(WINDOW_CONFIGS.keys())}."
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--test_file",
        type=str,
        required=True,
        help="Converted test CSV, e.g. dataset/ptbxl_lead2_b1_cap2/ptbxl_lead2_test.csv",
    )

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Merged B-1 model path",
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default="/workspace/ChatTime/results/b1_test_csv_eval",
    )

    parser.add_argument("--limit_rows", type=int, default=100)
    parser.add_argument("--num_plots", type=int, default=10)

    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_new_ratio", type=int, default=4)

    parser.add_argument("--debug_generation", action="store_true", default=False)

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    plot_dir = out_dir / "plots"

    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.test_file)

    if args.limit_rows > 0:
        df = df.head(args.limit_rows).copy()

    print("=== B-1 test CSV evaluation ===")
    print("test_file:", args.test_file)
    print("rows:", len(df))
    print("model_path:", args.model_path)
    print("out_dir:", out_dir)
    print("================================")

    forecaster = B1CSVForecaster(
        model_path=args.model_path,
        num_samples=args.num_samples,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        max_new_ratio=args.max_new_ratio,
        debug=args.debug_generation,
    )

    rows = []
    plot_count = 0

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            values = parse_chattime_values(row["text"])
            hist_len, pred_len = infer_split_lengths(len(values))

            hist = values[:hist_len]
            y_true = values[hist_len:hist_len + pred_len]

            pred_dict = {}
            pred_dict.update(make_baseline_predictions(hist, pred_len))

            y_pred = forecaster.generate_future_values(hist, pred_len)
            pred_dict["chattime"] = y_pred

            for model_name, pred in pred_dict.items():
                if np.isnan(pred).all():
                    metric = {
                        "mae": np.nan,
                        "rmse": np.nan,
                        "nrmse": np.nan,
                        "r2": np.nan,
                        "corr": np.nan,
                    }
                    error = "prediction is all NaN"
                else:
                    metric = calc_metrics(y_true, pred)
                    error = ""

                rows.append({
                    "row_id": idx,
                    "seq_len": len(values),
                    "hist_len": hist_len,
                    "pred_len": pred_len,
                    "model": model_name,
                    **metric,
                    "error": error,
                })

            if plot_count < args.num_plots:
                save_plot(
                    row_id=idx,
                    hist=hist,
                    y_true=y_true,
                    pred_dict=pred_dict,
                    out_dir=plot_dir,
                )
                plot_count += 1

        except Exception as e:
            rows.append({
                "row_id": idx,
                "seq_len": np.nan,
                "hist_len": np.nan,
                "pred_len": np.nan,
                "model": "none",
                "mae": np.nan,
                "rmse": np.nan,
                "nrmse": np.nan,
                "r2": np.nan,
                "corr": np.nan,
                "error": str(e),
            })

    result_df = pd.DataFrame(rows)

    result_path = out_dir / "window_metrics.csv"
    result_df.to_csv(result_path, index=False)

    valid_df = result_df.dropna(
        subset=["mae", "rmse", "nrmse", "r2"],
        how="any",
    )

    summary_rows = []

    for model_name, g in valid_df.groupby("model"):
        summary_rows.append({
            "model": model_name,
            "num_windows": len(g),
            "mean_mae": g["mae"].mean(),
            "mean_rmse": g["rmse"].mean(),
            "mean_nrmse": g["nrmse"].mean(),
            "mean_r2": g["r2"].mean(),
            "mean_corr": g["corr"].mean(),
            "median_mae": g["mae"].median(),
            "median_rmse": g["rmse"].median(),
            "median_nrmse": g["nrmse"].median(),
            "median_r2": g["r2"].median(),
            "median_corr": g["corr"].median(),
        })

    summary_df = pd.DataFrame(summary_rows)

    summary_path = out_dir / "summary_by_model.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\n=== Summary by model ===")
    if len(summary_df) > 0:
        print(summary_df.sort_values("mean_rmse").to_string(index=False))
    else:
        print("有効な評価結果がありません。")

    error_df = result_df[result_df["error"].astype(str).str.len() > 0]
    if len(error_df) > 0:
        print("\n=== Errors ===")
        print(error_df["error"].value_counts().head(20))

    print("\nSaved:")
    print(" metrics:", result_path)
    print(" summary:", summary_path)
    print(" plots:", plot_dir)


if __name__ == "__main__":
    main()