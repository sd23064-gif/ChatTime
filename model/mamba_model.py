import os
import re
from collections import Counter

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from utils.prompt import getPrompt
from utils.tools import Discretizer, Serializer
import time



class ChatTimeMamba:
    def __init__(
        self,
        base_model_path,
        adapter_path=None,
        hist_len=None,
        pred_len=None,
        max_pred_len=16,
        num_samples=8,
        top_k=100,
        top_p=1,
        temperature=1,
        torch_dtype=torch.float16,
        merge_lora=False,
    ):
        """
        base_model_path:
            例: "state-spaces/mamba-370m-hf"

        adapter_path:
            先ほど保存した LoRA adapter のパス。
            例: "/workspace/outputs/mamba-finetune"

        merge_lora:
            True にすると LoRA を base model に merge して推論する。
            まずは False 推奨。
        """

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

        tokenizer_path = adapter_path if adapter_path is not None else base_model_path

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
            low_cpu_mem_usage=True,
            return_dict=True,
            torch_dtype=torch_dtype,
            device_map={"": 0},
            trust_remote_code=True,
        )

        # 追加語彙に合わせて embedding サイズを変更
        self.model.resize_token_embeddings(len(self.tokenizer))

        if adapter_path is not None:
            adapter_config_path = os.path.join(adapter_path, "adapter_config.json")

            if not os.path.exists(adapter_config_path):
                raise FileNotFoundError(
                    f"adapter_config.json が見つかりません: {adapter_config_path}\n"
                    f"adapter_path には LoRA adapter の保存先を指定してください。"
                )

            self.model = PeftModel.from_pretrained(
                self.model,
                adapter_path,
                is_trainable=False,
            )

            if merge_lora:
                print("Merging LoRA adapter into base model...")
                self.model = self.model.merge_and_unload()

        self.model.eval()

        # generate 時は cache を使ってよい。
        # ただし環境によって MambaCache 関連で問題が出る場合は False に変更。
        if hasattr(self.model, "config"):
            self.model.config.use_cache = True

        if hasattr(self.model, "generation_config"):
            self.model.generation_config.use_cache = True

    def _device(self):
        return next(self.model.parameters()).device

    def _generate_texts(self, prompt, min_new_tokens=None, max_new_tokens=64):
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=False,
            truncation=False,
        )
        inputs = {key: value.to(self._device()) for key, value in inputs.items()}
        input_len = inputs["input_ids"].shape[-1]

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start = time.perf_counter()

        with torch.inference_mode():
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
                use_cache=True,
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start

        generated_lengths = []
        generated_texts = []

        for output in outputs:
            generated_ids = output[input_len:]
            generated_lengths.append(int(generated_ids.numel()))
            generated_texts.append(
                self.tokenizer.decode(
                    generated_ids,
                    skip_special_tokens=True,
                )
            )

        total_generated_tokens = sum(generated_lengths)

        print({
            "input_tokens": input_len,
            "samples": self.num_samples,
            "generated_lengths": generated_lengths,
            "total_generated_tokens": total_generated_tokens,
            "generate_seconds": elapsed,
            "tokens_per_second": total_generated_tokens / max(elapsed, 1e-9),
        })

        return generated_texts


    def _extract_response(self, text):
        """
        モデルが '### Response:' を再出力した場合にも対応。
        """
        if "### Response:\n" in text:
            return text.split("### Response:\n", 1)[1]
        if "### Response:" in text:
            return text.split("### Response:", 1)[1]
        return text

    def _extract_numeric_tokens(self, text):
        tokens = re.findall(
            r"###(?:[+-]?\d+(?:\.\d+)?|Nan|NaN|nan)###",
            text,
        )

        return " ".join(tokens)

    def predict(self, hist_data, context=None):
        if self.hist_len is None or self.pred_len is None:
            raise ValueError("hist_len and pred_len must be specified before prediction")

        series = np.asarray(hist_data, dtype=np.float64).copy()
        prediction_list = []
        remaining = self.pred_len

        while remaining > 0:
            current_pred_len = min(remaining, self.max_pred_len)

            dispersed_series = self.discretizer.discretize(series)
            serialized_series = self.serializer.serialize(dispersed_series)

            prompt = getPrompt(
                flag="prediction",
                context=context,
                input=serialized_series,
            )

            min_new_tokens = current_pred_len
            max_new_tokens = current_pred_len + 8

            samples = self._generate_texts(
                prompt,
                min_new_tokens=min_new_tokens,
                max_new_tokens=max_new_tokens,
            )

            pred_list = []
            parse_errors = 0

            for sample in samples:
                try:
                    serialized_prediction = self._extract_response(sample)
                    serialized_prediction = self._extract_numeric_tokens(
                        serialized_prediction
                    )

                    dispersed_prediction = self.serializer.inverse_serialize(
                        serialized_prediction
                    )
                    pred = self.discretizer.inverse_discretize(
                        dispersed_prediction
                    )
                    pred = np.asarray(pred, dtype=np.float64).reshape(-1)

                    if len(pred) == 0:
                        raise ValueError(
                            f"Parsed prediction is empty. raw_sample={repr(sample[:300])}"
                        )

                    if len(pred) < current_pred_len:
                        pred = np.concatenate([
                            pred,
                            np.full(
                                current_pred_len - len(pred),
                                np.nan,
                                dtype=np.float64,
                            ),
                        ])

                    pred_list.append(pred[:current_pred_len])

                except Exception as error:
                    parse_errors += 1
                    print(f"Failed to parse prediction sample: {error}")
                    print("Raw sample head:", repr(sample[:300]))

            if len(pred_list) == 0:
                print(
                    "[Warning] All prediction samples failed to parse. "
                    f"Using last-value fallback. current_pred_len={current_pred_len}"
                )
                prediction = np.full(
                    current_pred_len,
                    series[-1],
                    dtype=np.float64,
                )
            else:
                pred_arr = np.asarray(pred_list, dtype=np.float64)

                if pred_arr.ndim != 2 or pred_arr.shape[1] != current_pred_len:
                    raise ValueError(
                        f"Unexpected prediction array shape: {pred_arr.shape}, "
                        f"expected=({len(pred_list)}, {current_pred_len})"
                    )

                if np.isnan(pred_arr).all():
                    print(
                        "[Warning] All parsed predictions are NaN. "
                        f"Using last-value fallback. current_pred_len={current_pred_len}"
                    )
                    prediction = np.full(
                        current_pred_len,
                        series[-1],
                        dtype=np.float64,
                    )
                else:
                    with np.errstate(all="ignore"):
                        prediction = np.nanmedian(pred_arr, axis=0)

                    prediction = np.asarray(
                        prediction,
                        dtype=np.float64,
                    ).reshape(-1)

                    if np.isnan(prediction).any():
                        prediction = np.where(
                            np.isnan(prediction),
                            series[-1],
                            prediction,
                        )

            prediction = np.asarray(
                prediction,
                dtype=np.float64,
            ).reshape(-1)

            if len(prediction) != current_pred_len:
                raise ValueError(
                    "Final chunk length mismatch: "
                    f"actual={len(prediction)}, expected={current_pred_len}"
                )

            if not np.isfinite(prediction).all():
                print(
                    "[Warning] Non-finite predictions remained after aggregation. "
                    "Replacing them with the last observed value."
                )
                prediction = np.where(
                    np.isfinite(prediction),
                    prediction,
                    series[-1],
                )
            print({
                "generated_samples": len(samples),
                "parsed_samples": len(pred_list),
                "parse_errors": parse_errors,
                "prediction_length": len(prediction),
                "prediction_nan_count": int(np.isnan(prediction).sum()),
            })
            prediction_list.append(prediction)
            remaining -= current_pred_len

            if remaining <= 0:
                break

            series = np.concatenate([series, prediction], axis=-1)

        final_prediction = np.concatenate(prediction_list, axis=-1)
        final_prediction = final_prediction[:self.pred_len]

        if len(final_prediction) != self.pred_len:
            raise ValueError(
                "Final prediction length mismatch: "
                f"actual={len(final_prediction)}, expected={self.pred_len}"
            )

        return final_prediction

    def analyze(self, question, series):
        dispersed_series = self.discretizer.discretize(series)
        serialized_series = self.serializer.serialize(dispersed_series)

        prompt = getPrompt(
            flag="analysis",
            instruction=question,
            input=serialized_series,
        )

        samples = self._generate_texts(
            prompt,
            min_new_tokens=None,
            max_new_tokens=self.max_pred_len,
        )

        response_list = []

        for sample in samples:
            try:
                response = self._extract_response(sample)
                response = response.split(".")[0] + "."

                match = re.findall(r"\([abc]\)", response)

                if len(match) > 0:
                    response_list.append(match[0])

            except Exception as e:
                print(f"Failed to parse analysis sample: {e}")

        if len(response_list) == 0:
            return None

        response = Counter(response_list).most_common(1)[0][0]

        return response
    