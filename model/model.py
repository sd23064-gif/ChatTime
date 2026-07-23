import re
from statistics import mode

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoTokenizer, LlamaForCausalLM, pipeline

from utils.prompt import getPrompt
from utils.tools import Discretizer, Serializer


class ChatTime:
    def __init__(
        self,
        base_model_path,
        adapter_path=None,
        tokenizer_path=None,
        merge_adapter=False,
        hist_len=None,
        pred_len=None,
        max_pred_len=16,
        num_samples=8,
        top_k=100,
        top_p=1.0,
        temperature=1.0
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

        # GPUではfloat16、CPUではfloat32を使用
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        # =========================
        # 1. Tokenizer loading
        # =========================
        # ファインチューニング時にtokenizerも保存した場合は、
        # tokenizer_pathまたはadapter_pathを指定する
        tokenizer_source = tokenizer_path or adapter_path or base_model_path

        print(f"Loading tokenizer from: {tokenizer_source}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source,
            use_fast=True,
            trust_remote_code=True
        )

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.tokenizer.padding_side = "right"
        self.eos_token_id = self.tokenizer.eos_token_id

        # =========================
        # 2. Base model loading
        # =========================
        base_model = LlamaForCausalLM.from_pretrained(
            base_model_path,
            low_cpu_mem_usage=True,
            return_dict=True,
            torch_dtype=self.torch_dtype,
            device_map="auto"
        )

        # tokenizerに特殊トークンを追加して学習した場合に必要
        tokenizer_size = len(self.tokenizer)
        embedding_size = base_model.get_input_embeddings().weight.shape[0]

        if tokenizer_size != embedding_size:
            print(
                f"Resize token embeddings: "
                f"{embedding_size} -> {tokenizer_size}"
            )
            base_model.resize_token_embeddings(tokenizer_size)

        # =========================
        # 3. Adapter loading
        # =========================
        if adapter_path is not None:
            print(f"Loading adapter: {adapter_path}")

            self.model = PeftModel.from_pretrained(
                base_model,
                adapter_path,
                is_trainable=False
            )

            if merge_adapter:
                print("Merging adapter into base model")
                self.model = self.model.merge_and_unload()
        else:
            print("Adapter is not specified. Using the base model only.")
            self.model = base_model

        self.model.eval()

        # pipelineは予測ループの外で一度だけ生成
        self.pipe = pipeline(
            task="text-generation",
            model=self.model,
            tokenizer=self.tokenizer
        )

    def predict(self, hist_data, context=None):
        if self.hist_len is None or self.pred_len is None:
            raise ValueError(
                "hist_len and pred_len must be specified before prediction"
            )

        series = np.asarray(hist_data, dtype=np.float64)
        prediction_list = []
        remaining = self.pred_len

        while remaining > 0:
            current_pred_len = min(remaining, self.max_pred_len)

            dispersed_series = self.discretizer.discretize(series)
            serialized_series = self.serializer.serialize(dispersed_series)
            serialized_series = getPrompt(
                flag="prediction",
                context=context,
                input=serialized_series
            )

            encoded = self.tokenizer(
                serialized_series,
                add_special_tokens=True,
                return_tensors="pt"
            )

            prompt_token_length = encoded["input_ids"].shape[-1]

            print("系列点数:", len(series))
            print("プロンプト文字数:", len(serialized_series))
            print("プロンプトトークン数:", prompt_token_length)

            samples = self.pipe(
                serialized_series,
                min_new_tokens=2 * current_pred_len + 8,
                max_new_tokens=2 * current_pred_len + 8,
                do_sample=True,
                num_return_sequences=self.num_samples,
                top_k=self.top_k,
                top_p=self.top_p,
                temperature=self.temperature,
                eos_token_id=self.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id
            )

            pred_list = []

            for sample in samples:
                generated_text = sample["generated_text"]

                if "### Response:\n" not in generated_text:
                    pred = np.full(current_pred_len, np.nan)
                    pred_list.append(pred)
                    continue

                serialized_prediction = generated_text.split(
                    "### Response:\n",
                    maxsplit=1
                )[1]

                try:
                    dispersed_prediction = (
                        self.serializer.inverse_serialize(
                            serialized_prediction
                        )
                    )

                    pred = self.discretizer.inverse_discretize(
                        dispersed_prediction
                    )

                    pred = np.asarray(
                        pred,
                        dtype=np.float64
                    ).reshape(-1)

                except (ValueError, TypeError, IndexError):
                    pred = np.array([], dtype=np.float64)

                if len(pred) < current_pred_len:
                    padding = np.full(
                        current_pred_len - len(pred),
                        np.nan
                    )
                    pred = np.concatenate([pred, padding])

                pred_list.append(pred[:current_pred_len])

            pred_array = np.asarray(pred_list, dtype=np.float64)

            with np.errstate(all="ignore"):
                prediction = np.nanmedian(pred_array, axis=0)

            # 全サンプルがNaNだった位置を直前値で補完
            if np.isnan(prediction).any():
                fallback = series[-1]

                for i in range(len(prediction)):
                    if np.isnan(prediction[i]):
                        prediction[i] = (
                            prediction[i - 1]
                            if i > 0
                            else fallback
                        )

            prediction_list.append(prediction)
            remaining -= len(prediction)

            if remaining <= 0:
                break

            series = np.concatenate(
                [series, prediction],
                axis=-1
            )

        prediction = np.concatenate(
            prediction_list,
            axis=-1
        )

        return prediction[:self.pred_len]

    def analyze(self, question, series):
        dispersed_series = self.discretizer.discretize(series)
        serialized_series = self.serializer.serialize(dispersed_series)
        serialized_series = getPrompt(
            flag="analysis",
            instruction=question,
            input=serialized_series
        )

        samples = self.pipe(
            serialized_series,
            max_new_tokens=self.max_pred_len,
            do_sample=True,
            num_return_sequences=self.num_samples,
            top_k=self.top_k,
            top_p=self.top_p,
            temperature=self.temperature,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id
        )

        response_list = []

        for sample in samples:
            generated_text = sample["generated_text"]

            if "### Response:\n" not in generated_text:
                continue

            generated_response = generated_text.split(
                "### Response:\n",
                maxsplit=1
            )[1]

            matches = re.findall(r"\([abc]\)", generated_response)

            if matches:
                response_list.append(matches[0])

        if not response_list:
            raise ValueError(
                "有効な選択肢 (a), (b), (c) を生成結果から取得できませんでした"
            )

        return mode(response_list)