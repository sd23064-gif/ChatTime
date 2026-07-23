import re
from statistics import mode

import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from utils.prompt import getPrompt
from utils.tools import Discretizer, Serializer


class ChatTimeECG:
    """
    ECG forecasting用 ChatTime class.

    prompt_mode:
        "continuation":
            B-1 継続事前学習モデル向け。
            入力の数値token列の続きをそのまま生成する。

        "chat_prompt":
            公式 ChatTime-1-7B-Chat のような instruction model向け。
            getPrompt(flag="prediction", ...) を使い、### Response: 以降を読む。
    """

    def __init__(
        self,
        model_path,
        hist_len=None,
        pred_len=None,
        max_pred_len=16,
        num_samples=8,
        top_k=100,
        top_p=1.0,
        temperature=1.0,
        prompt_mode="continuation",
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
        self.prompt_mode = prompt_mode
        self.debug = debug

        self.discretizer = Discretizer()
        self.serializer = Serializer()

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

        print("Loading model:", model_path)
        print("torch_dtype:", torch_dtype)
        print("prompt_mode:", prompt_mode)

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            low_cpu_mem_usage=True,
            return_dict=True,
            torch_dtype=torch_dtype,
            device_map="auto" if use_cuda else None,
            trust_remote_code=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
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

    def _build_prompt(self, serialized_series, context=None):
        if self.prompt_mode == "chat_prompt":
            return getPrompt(
                flag="prediction",
                context=context,
                input=serialized_series,
            )

        if self.prompt_mode == "continuation":
            # B-1用:
            # 学習データと同じく数値token列だけを与え、その続きを生成させる。
            return serialized_series

        raise ValueError(
            f"Unknown prompt_mode={self.prompt_mode}. "
            "Use 'continuation' or 'chat_prompt'."
        )

    def _extract_generated_part(self, prompt_text, generated_text):
        """
        生成テキストから、予測部分だけを取り出す。
        """
        if self.prompt_mode == "chat_prompt":
            marker = "### Response:\n"
            if marker in generated_text:
                return generated_text.split(marker, 1)[1]
            return generated_text

        if self.prompt_mode == "continuation":
            # pipelineではなくgenerateを使うが、decode結果にはpromptも含まれる。
            # prompt_textで始まっていれば、その後ろだけを取る。
            if generated_text.startswith(prompt_text):
                return generated_text[len(prompt_text):]
            return generated_text

        return generated_text

    def _generate_text_samples(self, prompt_text, target_len):
        """
        model.generateで複数サンプル生成する。
        """
        device = self._get_model_device()

        inputs = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            add_special_tokens=False,
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        # 数値tokenは基本1 tokenとして追加されている想定。
        # ただし余分な文字やEOSを考慮して少し多めに生成する。
        max_new_tokens = 2 * target_len + 8
        min_new_tokens = target_len

        do_sample = self.num_samples > 1 or self.temperature > 0

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                min_new_tokens=min_new_tokens,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                num_return_sequences=self.num_samples,
                top_k=self.top_k,
                top_p=self.top_p,
                temperature=self.temperature if do_sample else None,
                eos_token_id=self.eos_token_id,
                pad_token_id=self.pad_token_id,
            )

        generated_texts = self.tokenizer.batch_decode(
            outputs,
            skip_special_tokens=True,
        )

        return generated_texts

    def _parse_prediction(self, serialized_prediction, target_len):
        dispersed_prediction = self.serializer.inverse_serialize(
            serialized_prediction
        )

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

    def predict(self, hist_data, context=None):
        if self.hist_len is None or self.pred_len is None:
            raise ValueError(
                "hist_len and pred_len must be specified before prediction"
            )

        series = np.asarray(hist_data, dtype=np.float32).reshape(-1)

        prediction_list = []
        remaining = self.pred_len

        while remaining > 0:
            current_pred_len = min(remaining, self.max_pred_len)

            dispersed_series = self.discretizer.discretize(series)
            serialized_series = self.serializer.serialize(dispersed_series)

            prompt_text = self._build_prompt(
                serialized_series=serialized_series,
                context=context,
            )

            generated_texts = self._generate_text_samples(
                prompt_text=prompt_text,
                target_len=current_pred_len,
            )

            pred_list = []

            for generated_text in generated_texts:
                serialized_prediction = self._extract_generated_part(
                    prompt_text=prompt_text,
                    generated_text=generated_text,
                )

                if self.debug:
                    print("\n=== prompt ===")
                    print(prompt_text[-500:])
                    print("\n=== generated_text ===")
                    print(generated_text[-1000:])
                    print("\n=== serialized_prediction ===")
                    print(serialized_prediction[:1000])

                pred = self._parse_prediction(
                    serialized_prediction=serialized_prediction,
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

            prediction_list.append(prediction)

            remaining -= len(prediction)

            if remaining <= 0:
                break

            # autoregressive forecasting:
            # 予測をseriesに追加して次chunkを予測する。
            series = np.concatenate([series, prediction], axis=-1)

        prediction = np.concatenate(prediction_list, axis=-1)
        prediction = prediction[:self.pred_len]

        return prediction

    def analyze(self, question, series):
        """
        公式ChatTime風のanalysis用。
        B-1波形予測では基本使わない。
        """
        dispersed_series = self.discretizer.discretize(series)
        serialized_series = self.serializer.serialize(dispersed_series)
        serialized_series = getPrompt(
            flag="analysis",
            instruction=question,
            input=serialized_series,
        )

        device = self._get_model_device()

        inputs = self.tokenizer(
            serialized_series,
            return_tensors="pt",
            add_special_tokens=False,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_pred_len,
                do_sample=True,
                num_return_sequences=self.num_samples,
                top_k=self.top_k,
                top_p=self.top_p,
                temperature=self.temperature,
                eos_token_id=self.eos_token_id,
                pad_token_id=self.pad_token_id,
            )

        generated_texts = self.tokenizer.batch_decode(
            outputs,
            skip_special_tokens=True,
        )

        response_list = []

        for generated_text in generated_texts:
            if "### Response:\n" in generated_text:
                response = generated_text.split("### Response:\n", 1)[1]
            else:
                response = generated_text

            response = response.split(".")[0] + "."

            matches = re.findall(r"\([abc]\)", response)
            if len(matches) > 0:
                response_list.append(matches[0])

        if len(response_list) == 0:
            return None

        response = mode(response_list)
        return response