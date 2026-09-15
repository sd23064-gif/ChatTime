import argparse
from pathlib import Path

import numpy as np

# NumPy 2.x 対策
if not hasattr(np, "NaN"):
    np.NaN = np.nan

import pandas as pd
import torch
import wfdb
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.tools import Discretizer, Serializer


class ChatTimeB1Forecaster:
    """
    B-1 継続事前学習済みLlama/ChatTime形式モデル用のECG未来予測クラス。

    B-1の学習データ形式:
        ###-0.4159### ###-0.4731### ###-0.4343### ...

    したがって推論時も:
        履歴token列 -> その続きを生成

    を行う。
    """

    def __init__(
        self,
        model_path,
        hist_len,
        pred_len,
        max_pred_len=20,
        num_samples=1,
        top_k=100,
        top_p=1.0,
        temperature=1.0,
        low_limit=-1,
        high_limit=1,
        n_tokens=10002,
        prec=4,
        time_sep=" ",
        time_flag="###",
        nan_flag="Nan",
        use_bf16_if_available=True,
        debug=False,
    ):
        self.model_path = model_path
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.max_pred_len = max_pred_len
        self.num_samples = num_samples
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature
        self.debug = debug

        self.discretizer = Discretizer(
            low_limit=low_limit,
            high_limit=high_limit,
            n_tokens=n_tokens,
        )
        self.serializer = Serializer(
            prec=prec,
            time_sep=time_sep,
            time_flag=time_flag,
            nan_flag=nan_flag,
        )

        use_cuda = torch.cuda.is_available()
        use_bf16 = (
            use_cuda
            and use_bf16_if_available
            and torch.cuda.is_bf16_supported()
        )

        if use_bf16:
            torch_dtype = torch.bfloat16
        elif use_cuda:
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32

        print("=== Loading B-1 ChatTime model ===")
        print("model_path:", model_path)
        print("torch_dtype:", torch_dtype)
        print("cuda available:", torch.cuda.is_available())

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,
            return_dict=True,
            torch_dtype=torch_dtype,
            device_map="auto" if use_cuda else None,
            trust_remote_code=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=True,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.tokenizer.padding_side = "right"
        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id

        self.model.eval()

    def _get_model_device(self):
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def _serialize_history(self, series):
        """
        ChatTime本家predictと同じ思想で、履歴系列を離散化してserializeする。
        ここでは予測時なので、履歴series全体でscaler fitする。
        """
        dispersed_series = self.discretizer.discretize(series)
        serialized_series = self.serializer.serialize(dispersed_series)
        return serialized_series

    def _generate_new_texts(self, prompt_text, target_len):
        """
        continuation生成。
        prompt_text自体を除いた新規生成部分のみdecodeして返す。
        """
        device = self._get_model_device()

        inputs = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            add_special_tokens=False,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        input_len = inputs["input_ids"].shape[-1]

        do_sample = self.num_samples > 1 or self.temperature > 0

        # 数値tokenは基本1 token想定だが、念のため多めに生成する
        max_new_tokens = 2 * target_len + 8
        min_new_tokens = target_len

        generation_kwargs = dict(
            **inputs,
            min_new_tokens=min_new_tokens,
            max_new_tokens=max_new_tokens,
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

        new_texts = []
        for ids in output_ids:
            new_ids = ids[input_len:]
            text = self.tokenizer.decode(
                new_ids,
                skip_special_tokens=True,
            )
            new_texts.append(text)

        return new_texts

    def _parse_prediction_text(self, generated_text, target_len):
        """
        生成された ###数値### token列から未来値を復元する。
        """
        dispersed_prediction = self.serializer.inverse_serialize(generated_text)

        if len(dispersed_prediction) == 0:
            return np.full(target_len, np.nan, dtype=np.float32)

        pred = self.discretizer.inverse_discretize(dispersed_prediction)
        pred = np.asarray(pred, dtype=np.float32).reshape(-1)

        if len(pred) < target_len:
            pad_len = target_len - len(pred)
            pred = np.concatenate(
                [
                    pred,
                    np.full(pad_len, np.nan, dtype=np.float32),
                ]
            )

        pred = pred[:target_len]
        return pred

    def predict(self, hist_data):
        if self.hist_len is None or self.pred_len is None:
            raise ValueError("hist_len and pred_len must be specified.")

        series = np.asarray(hist_data, dtype=np.float32).reshape(-1)

        if len(series) != self.hist_len:
            raise ValueError(
                f"hist_data length must be {self.hist_len}, got {len(series)}"
            )

        prediction_chunks = []
        remaining = self.pred_len

        while remaining > 0:
            current_pred_len = min(remaining, self.max_pred_len)

            prompt_text = self._serialize_history(series)

            generated_texts = self._generate_new_texts(
                prompt_text=prompt_text,
                target_len=current_pred_len,
            )

            pred_list = []

            for generated_text in generated_texts:
                if self.debug:
                    print("\n=== prompt tail ===")
                    print(prompt_text[-500:])
                    print("\n=== generated_text ===")
                    print(generated_text[:1000])

                pred = self._parse_prediction_text(
                    generated_text=generated_text,
                    target_len=current_pred_len,
                )
                pred_list.append(pred)

            pred_arr = np.asarray(pred_list, dtype=np.float32)

            if self.debug:
                print("pred_arr shape:", pred_arr.shape)
                print("finite count:", np.isfinite(pred_arr).sum())

            if np.isfinite(pred_arr).sum() == 0:
                prediction = np.full(
                    current_pred_len,
                    np.nan,
                    dtype=np.float32,
                )
            else:
                prediction = np.nanmedian(pred_arr, axis=0).astype(np.float32)

            prediction_chunks.append(prediction)

            remaining -= len(prediction)

            if remaining <= 0:
                break

            # autoregressive forecasting
            series = np.concatenate([series, prediction], axis=-1)

        prediction = np.concatenate(prediction_chunks, axis=-1)
        return prediction[:self.pred_len]


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


def read_ecg_record(root, filename_lr):
    record_path = Path(root) / filename_lr
    signal, fields = wfdb.rdsamp(str(record_path))
    return signal, fields["sig_name"]


def get_lead_index(lead_names, target_lead):
    normalized = [x.upper() for x in lead_names]
    target = target_lead.upper()

    if target not in normalized:
        raise ValueError(
            f"Lead {target_lead} が見つかりません。利用可能な誘導: {lead_names}"
        )

    return normalized.index(target)


def make_baseline_predictions(hist, pred_len):
    hist = np.asarray(hist, dtype=np.float32)

    last_value_pred = np.full(pred_len, hist[-1], dtype=np.float32)

    if len(hist) >= pred_len:
        repeat_pred = hist[-pred_len:].astype(np.float32)
    else:
        repeat_pred = np.resize(hist, pred_len).astype(np.float32)

    return {
        "last_value": last_value_pred,
        "repeat": repeat_pred,
    }


def save_window_plot(
    ecg_id,
    lead,
    start,
    hist,
    y_true,
    pred_dict,
    hist_len,
    pred_len,
    out_dir,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    hist_x = np.arange(start, start + hist_len)
    pred_x = np.arange(start + hist_len, start + hist_len + pred_len)

    colors = {
        "chattime": "orange",
        "last_value": "green",
        "repeat": "purple",
    }

    plt.figure(figsize=(12, 4))
    plt.plot(hist_x, hist, label="history", color="black")
    plt.plot(pred_x, y_true, label="true future", color="blue")

    for name, y_pred in pred_dict.items():
        plt.plot(
            pred_x,
            y_pred,
            label=name,
            color=colors.get(name, None),
        )

    plt.axvline(
        start + hist_len - 1,
        color="red",
        linestyle="--",
        label="prediction start",
    )

    plt.title(f"ECG prediction: ecg_id={ecg_id}, lead={lead}, start={start}")
    plt.xlabel("sample")
    plt.ylabel("amplitude")
    plt.legend()
    plt.tight_layout()

    filename = f"ecg_{ecg_id}_lead_{lead}_start_{start}_full.png"
    plt.savefig(out_dir / filename, dpi=200)
    plt.close()

    plt.figure(figsize=(10, 4))
    future_x = np.arange(pred_len)

    plt.plot(future_x, y_true, label="true future", color="blue")

    for name, y_pred in pred_dict.items():
        plt.plot(
            future_x,
            y_pred,
            label=name,
            color=colors.get(name, None),
        )

    plt.title(f"Future only: ecg_id={ecg_id}, lead={lead}, start={start}")
    plt.xlabel("future sample")
    plt.ylabel("amplitude")
    plt.legend()
    plt.tight_layout()

    filename = f"ecg_{ecg_id}_lead_{lead}_start_{start}_future_only.png"
    plt.savefig(out_dir / filename, dpi=200)
    plt.close()

    df_plot = pd.DataFrame({
        "future_index": np.arange(pred_len),
        "y_true": y_true,
    })

    for name, y_pred in pred_dict.items():
        df_plot[f"y_pred_{name}"] = y_pred
        df_plot[f"error_{name}"] = y_pred - y_true

    csv_name = f"ecg_{ecg_id}_lead_{lead}_start_{start}_prediction.csv"
    df_plot.to_csv(out_dir / csv_name, index=False)


def normalize_1d_by_percentile(x, eps=1e-8):
    x = np.asarray(x, dtype=np.float32)
    scale = np.percentile(np.abs(x), 99)
    x = x / (scale + eps)
    x = np.clip(x, -1.0, 1.0)
    return x


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        type=str,
        default="/workspace/ChatTime/data/ptb-xl/1.0.3",
        help="PTB-XL root directory",
    )

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="B-1 trained merged model path",
    )

    parser.add_argument("--hist_len", type=int, default=180)
    parser.add_argument("--pred_len", type=int, default=20)
    parser.add_argument("--lead", type=str, default="II")
    parser.add_argument("--fold", type=int, default=10)

    parser.add_argument("--limit_records", type=int, default=10)
    parser.add_argument("--window_stride", type=int, default=50)
    parser.add_argument("--max_windows_per_record", type=int, default=5)
    parser.add_argument("--num_plots", type=int, default=10)

    parser.add_argument(
        "--out_dir",
        type=str,
        default="/workspace/ChatTime/results/ptbxl_b1_eval",
    )

    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--max_pred_len", type=int, default=20)
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)

    parser.add_argument(
        "--normalize",
        action="store_true",
        default=False,
        help="Apply external robust normalization. Usually false for B-1.",
    )

    parser.add_argument(
        "--debug_generation",
        action="store_true",
        default=False,
        help="Print generated token text.",
    )

    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    plot_dir = out_dir / "plots"

    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    db_path = root / "ptbxl_database.csv"

    if not db_path.exists():
        raise FileNotFoundError(f"ptbxl_database.csv not found: {db_path}")

    df = pd.read_csv(db_path)

    if "strat_fold" in df.columns:
        df = df[df["strat_fold"] == args.fold].copy()

    if args.limit_records > 0:
        df = df.head(args.limit_records).copy()

    print("=== Evaluation setting ===")
    print("records:", len(df))
    print("root:", root)
    print("model_path:", args.model_path)
    print("lead:", args.lead)
    print("hist_len:", args.hist_len)
    print("pred_len:", args.pred_len)
    print("fold:", args.fold)
    print("window_stride:", args.window_stride)
    print("max_windows_per_record:", args.max_windows_per_record)
    print("normalize:", args.normalize)
    print("out_dir:", out_dir)
    print("==========================")

    model = ChatTimeB1Forecaster(
        hist_len=args.hist_len,
        pred_len=args.pred_len,
        model_path=args.model_path,
        num_samples=args.num_samples,
        max_pred_len=args.max_pred_len,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        debug=args.debug_generation,
    )

    rows = []
    plot_count = 0

    need_len = args.hist_len + args.pred_len

    for _, row in tqdm(df.iterrows(), total=len(df)):
        ecg_id = row["ecg_id"]
        filename_lr = row["filename_lr"]

        try:
            signal, lead_names = read_ecg_record(root, filename_lr)
            lead_idx = get_lead_index(lead_names, args.lead)

            x = signal[:, lead_idx].astype(np.float32)

            if args.normalize:
                x = normalize_1d_by_percentile(x)

            if len(x) < need_len:
                rows.append({
                    "ecg_id": ecg_id,
                    "filename_lr": filename_lr,
                    "lead": args.lead,
                    "start": np.nan,
                    "model": "none",
                    "mae": np.nan,
                    "rmse": np.nan,
                    "nrmse": np.nan,
                    "r2": np.nan,
                    "corr": np.nan,
                    "error": f"signal too short: len={len(x)}, need={need_len}",
                })
                continue

            possible_starts = list(
                range(
                    0,
                    len(x) - need_len + 1,
                    args.window_stride,
                )
            )

            if (
                args.max_windows_per_record > 0
                and len(possible_starts) > args.max_windows_per_record
            ):
                indices = np.linspace(
                    0,
                    len(possible_starts) - 1,
                    args.max_windows_per_record,
                    dtype=int,
                )
                possible_starts = [possible_starts[i] for i in indices]

            for start in possible_starts:
                hist = x[start:start + args.hist_len]
                y_true = x[
                    start + args.hist_len:
                    start + args.hist_len + args.pred_len
                ]

                pred_dict = {}

                baseline_preds = make_baseline_predictions(hist, args.pred_len)
                pred_dict.update(baseline_preds)

                try:
                    y_pred_chattime = model.predict(hist)
                    y_pred_chattime = np.asarray(
                        y_pred_chattime,
                        dtype=np.float32,
                    ).reshape(-1)

                    if len(y_pred_chattime) > args.pred_len:
                        y_pred_chattime = y_pred_chattime[:args.pred_len]
                    elif len(y_pred_chattime) < args.pred_len:
                        pad_len = args.pred_len - len(y_pred_chattime)
                        if len(y_pred_chattime) == 0:
                            y_pred_chattime = np.full(
                                args.pred_len,
                                np.nan,
                                dtype=np.float32,
                            )
                        else:
                            y_pred_chattime = np.pad(
                                y_pred_chattime,
                                (0, pad_len),
                                mode="edge",
                            )

                    pred_dict["chattime"] = y_pred_chattime

                except Exception as e:
                    pred_dict["chattime"] = np.full(
                        args.pred_len,
                        np.nan,
                        dtype=np.float32,
                    )
                    rows.append({
                        "ecg_id": ecg_id,
                        "filename_lr": filename_lr,
                        "lead": args.lead,
                        "start": start,
                        "model": "chattime",
                        "mae": np.nan,
                        "rmse": np.nan,
                        "nrmse": np.nan,
                        "r2": np.nan,
                        "corr": np.nan,
                        "error": str(e),
                    })

                for model_name, y_pred in pred_dict.items():
                    y_pred = np.asarray(y_pred, dtype=np.float32)

                    if np.isnan(y_pred).all():
                        rows.append({
                            "ecg_id": ecg_id,
                            "filename_lr": filename_lr,
                            "lead": args.lead,
                            "start": start,
                            "model": model_name,
                            "mae": np.nan,
                            "rmse": np.nan,
                            "nrmse": np.nan,
                            "r2": np.nan,
                            "corr": np.nan,
                            "error": "prediction is all NaN",
                        })
                        continue

                    m = calc_metrics(y_true, y_pred)

                    rows.append({
                        "ecg_id": ecg_id,
                        "filename_lr": filename_lr,
                        "lead": args.lead,
                        "start": start,
                        "model": model_name,
                        **m,
                        "error": "",
                    })

                if plot_count < args.num_plots:
                    save_window_plot(
                        ecg_id=ecg_id,
                        lead=args.lead,
                        start=start,
                        hist=hist,
                        y_true=y_true,
                        pred_dict=pred_dict,
                        hist_len=args.hist_len,
                        pred_len=args.pred_len,
                        out_dir=plot_dir,
                    )
                    plot_count += 1

        except Exception as e:
            rows.append({
                "ecg_id": ecg_id,
                "filename_lr": filename_lr,
                "lead": args.lead,
                "start": np.nan,
                "model": "none",
                "mae": np.nan,
                "rmse": np.nan,
                "nrmse": np.nan,
                "r2": np.nan,
                "corr": np.nan,
                "error": str(e),
            })

    result_df = pd.DataFrame(rows)

    metric_cols = ["mae", "rmse", "nrmse", "r2", "corr"]
    for col in metric_cols:
        if col not in result_df.columns:
            result_df[col] = np.nan

    result_path = out_dir / "window_metrics.csv"
    result_df.to_csv(result_path, index=False)

    valid_df = result_df.dropna(
        subset=["mae", "rmse", "nrmse", "r2"],
        how="any",
    ).copy()

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

    if "error" in result_df.columns:
        error_df = result_df[result_df["error"].astype(str).str.len() > 0]
        if len(error_df) > 0:
            print("\n=== Errors ===")
            print(error_df["error"].value_counts().head(20))

    print("\nSaved:")
    print(f"  metrics : {result_path}")
    print(f"  summary : {summary_path}")
    print(f"  plots   : {plot_dir}")


if __name__ == "__main__":
    main()