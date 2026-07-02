import os
import re
from collections import Counter

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from utils.prompt import getPrompt
from utils.tools import Discretizer, Serializer


class ChatTimeMamba:
    def __init__(
        self,
        base_model_path,
        adapter_path=None,
        hist_len=None,
        pred_len=None,
        max_pred_len=16,
        num_samples=8,
        top_k=50,
        top_p=0.9,
        temperature=0.7,
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
            device_map="auto",
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

        inputs = {k: v.to(self._device()) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[-1]

        with torch.no_grad():
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

        generated_texts = []

        for output in outputs:
            # prompt 部分を除いて、新規生成部分のみ decode
            generated_ids = output[input_len:]
            text = self.tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
            )
            generated_texts.append(text)

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
        import re
        tokens = re.findall(
            r"###(?:[+-]?\d+(?:\.\d+)?|Nan|NaN|nan)###",
            text,
        )
        return " ".join(tokens)

    def predict(self, hist_data, context=None):
        if self.hist_len is None or self.pred_len is None:
            raise ValueError("hist_len and pred_len must be specified before prediction")

        series = hist_data
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

            max_new_tokens = 6 * current_pred_len + 32
            min_new_tokens = current_pred_len 

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
                    serialized_prediction = self._extract_numeric_tokens(serialized_prediction)

                    dispersed_prediction = self.serializer.inverse_serialize(
                        serialized_prediction
                    )

                    pred = self.discretizer.inverse_discretize(
                        dispersed_prediction
                    )

                    pred = np.asarray(pred, dtype=np.float64)

                    if len(pred) == 0:
                        raise ValueError(
                            f"Parsed prediction is empty. raw_sample={repr(sample[:300])}"
                        )

                    if len(pred) < current_pred_len:
                        pred = np.concatenate(
                            [
                                pred,
                                np.full(current_pred_len - len(pred), np.nan),
                            ]
                        )

                    pred_list.append(pred[:current_pred_len])

                except Exception as e:
                    parse_errors += 1
                    print(f"Failed to parse prediction sample: {e}")
                    print("Raw sample head:", repr(sample[:300]))
                    continue

            if len(pred_list) == 0:
                print(
                    f"[Warning] All prediction samples failed to parse. "
                    f"Using last-value fallback. current_pred_len={current_pred_len}"
                )

                prediction = np.full(
                    current_pred_len,
                    series[-1],
                    dtype=np.float64,
                )
            else:
                pred_arr = np.asarray(pred_list, dtype=np.float64)

                if np.isnan(pred_arr).all():
                    print(
                        f"[Warning] All parsed predictions are NaN. "
                        f"Using last-value fallback. current_pred_len={current_pred_len}"
                    )
                    prediction = np.full(
                        current_pred_len,
                        series[-1],
                        dtype=np.float64,
                    )
                else:
                    prediction = np.nanmedian(pred_arr, axis=0)

                    # 部分的に NaN が残る場合は last value で埋める
                    if np.isnan(prediction).any():
                        prediction = np.where(
                            np.isnan(prediction),
                            series[-1],
                            prediction,
                        )

            prediction = np.nanmedian(pred_list, axis=0)

            prediction_list.append(prediction)
            remaining -= prediction.shape[-1]

            if remaining <= 0:
                break

            series = np.concatenate([series, prediction], axis=-1)

        prediction = np.concatenate(prediction_list, axis=-1)

        return prediction

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