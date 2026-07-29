import os
import re
import time
import warnings
from collections import Counter

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.prompt import getPrompt
from utils.tools import Discretizer, Serializer


class ChatTimeMamba:
    def __init__(
        self,
        base_model_path,
        adapter_path=None,
        tokenizer_path=None,
        hist_len=None,
        pred_len=None,
        max_pred_len=16,
        num_samples=8,
        top_k=100,
        top_p=1.0,
        temperature=1.0,
        torch_dtype=torch.float16,
        merge_lora=False,
        local_files_only=True,
        debug_generation=False,
        debug_samples=2,
        verbose=False,
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
        self.debug_generation = bool(debug_generation)
        self.debug_samples = max(0, int(debug_samples))
        self.verbose = bool(verbose)
        self.last_prediction_stats = {}

        self.discretizer = Discretizer()
        self.serializer = Serializer()

        resolved_tokenizer_path = tokenizer_path or base_model_path
        if self.verbose:
            print("Loading Mamba tokenizer from:", resolved_tokenizer_path)

        self.tokenizer = AutoTokenizer.from_pretrained(
            resolved_tokenizer_path,
            use_fast=True,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is None:
                raise ValueError("Tokenizer has neither pad_token_id nor eos_token_id.")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        self.eos_token_id = self.tokenizer.eos_token_id

        if self.verbose:
            print("Loading Mamba model from:", base_model_path)

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            low_cpu_mem_usage=True,
            return_dict=True,
            torch_dtype=torch_dtype,
            device_map={"": 0} if torch.cuda.is_available() else None,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )

        model_type = getattr(base_model.config, "model_type", None)
        if model_type not in {"mamba", "mamba2"}:
            raise ValueError(
                "ChatTimeMamba expected a Mamba model, but loaded "
                f"model_type={model_type}, class={type(base_model).__name__}"
            )

        tokenizer_size = len(self.tokenizer)
        embedding_size = base_model.get_input_embeddings().weight.shape[0]
        output_layer = base_model.get_output_embeddings()
        output_size = output_layer.weight.shape[0] if output_layer is not None else None

        if adapter_path is None:
            if embedding_size != tokenizer_size:
                raise ValueError(
                    "Merged Mamba model/tokenizer vocabulary mismatch: "
                    f"model={embedding_size}, tokenizer={tokenizer_size}."
                )
            if output_size is not None and output_size != tokenizer_size:
                raise ValueError(
                    "Merged Mamba LM Head/tokenizer vocabulary mismatch: "
                    f"lm_head={output_size}, tokenizer={tokenizer_size}."
                )
        else:
            adapter_config_path = os.path.join(adapter_path, "adapter_config.json")
            if not os.path.isfile(adapter_config_path):
                raise FileNotFoundError(f"adapter_config.json was not found: {adapter_config_path}")
            if embedding_size != tokenizer_size:
                if self.verbose:
                    print("Resizing base vocabulary before adapter load:", embedding_size, "->", tokenizer_size)
                base_model.resize_token_embeddings(tokenizer_size)

            base_model = PeftModel.from_pretrained(
                base_model,
                adapter_path,
                is_trainable=False,
                local_files_only=local_files_only,
            )
            if merge_lora:
                if self.verbose:
                    print("Merging Mamba LoRA adapter")
                base_model = base_model.merge_and_unload(safe_merge=True)

        self.model = base_model
        self.model.eval()
        if hasattr(self.model, "config"):
            self.model.config.use_cache = True
        if hasattr(self.model, "generation_config"):
            self.model.generation_config.use_cache = True
            self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
            self.model.generation_config.eos_token_id = self.tokenizer.eos_token_id

    def _device(self):
        return next(self.model.parameters()).device

    def _generate_texts(self, prompt, min_new_tokens, max_new_tokens):
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=False, truncation=False)
        inputs = {key: value.to(self._device()) for key, value in inputs.items()}
        input_length = inputs["input_ids"].shape[-1]

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter()
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
        elapsed = time.perf_counter() - started

        texts, lengths = [], []
        for sample_index, output in enumerate(outputs):
            generated_ids = output[input_length:]
            decoded = self.tokenizer.decode(generated_ids, skip_special_tokens=False)
            texts.append(decoded)
            lengths.append(int(generated_ids.numel()))
            if self.debug_generation and sample_index < self.debug_samples:
                print(f"\nMamba generated sample {sample_index}")
                print("IDs:", generated_ids.tolist())
                print("Tokens:", self.tokenizer.convert_ids_to_tokens(generated_ids.tolist()))
                print("Decoded:", repr(decoded))

        total = int(sum(lengths))
        stats = {
            "input_tokens": int(input_length),
            "samples": int(len(outputs)),
            "generated_lengths": lengths,
            "total_generated_tokens": total,
            "generate_seconds": float(elapsed),
            "tokens_per_second": float(total / max(elapsed, 1e-9)),
        }
        if self.verbose:
            print("Mamba generation:", stats)
        return texts, stats

    @staticmethod
    def _extract_response(text):
        markers = [
            "##### Response:\n", "##### Response:",
            "#### Response:\n", "#### Response:",
            "### Response:\n", "### Response:",
        ]
        for marker in markers:
            if marker in text:
                return text.split(marker, 1)[1]
        return text

    @staticmethod
    def _extract_numeric_tokens(text):
        return re.findall(
            r"###(?:[+-]?(?:\d+(?:\.\d*)?|\.\d+)|Nan|NaN|nan)###",
            text,
        )

    @staticmethod
    def _finite_summary(values, operation):
        values = np.asarray(values, dtype=np.float64)
        finite = values[np.isfinite(values)]
        if len(finite) == 0:
            return np.nan
        if operation == "mean":
            return float(np.mean(finite))
        if operation == "max":
            return float(np.max(finite))
        raise ValueError(f"Unsupported summary operation: {operation}")

    def _parse_prediction(self, sample, current_pred_len, sample_index):
        response = self._extract_response(sample)
        numeric_tokens = self._extract_numeric_tokens(response)
        if self.debug_generation and sample_index < self.debug_samples:
            print("Extracted numeric tokens:", numeric_tokens)
        if not numeric_tokens:
            raise ValueError(f"No numeric tokens were generated. sample={sample[:300]!r}")

        dispersed = np.asarray(
            self.serializer.inverse_serialize(" ".join(numeric_tokens)),
            dtype=np.float64,
        ).reshape(-1)
        if len(dispersed) == 0 or not np.isfinite(dispersed).any():
            raise ValueError("Serializer returned no finite values.")

        prediction = np.asarray(
            self.discretizer.inverse_discretize(dispersed),
            dtype=np.float64,
        ).reshape(-1)
        raw_length = len(prediction)
        if raw_length == 0:
            raise ValueError("Inverse-discretized prediction is empty.")
        if raw_length < current_pred_len:
            prediction = np.concatenate([
                prediction,
                np.full(current_pred_len - raw_length, np.nan, dtype=np.float64),
            ])
        prediction = prediction[:current_pred_len]
        finite_count = int(np.isfinite(prediction).sum())
        if finite_count == 0:
            raise ValueError("Parsed prediction contains no finite values.")

        return prediction, {
            "sample_index": int(sample_index),
            "numeric_token_count": int(len(numeric_tokens)),
            "raw_prediction_length": int(raw_length),
            "finite_count": finite_count,
            "status": "complete" if finite_count == current_pred_len else "partial",
        }

    def predict(self, hist_data, context=None):
        if self.hist_len is None or self.pred_len is None:
            raise ValueError("hist_len and pred_len must be specified.")

        series = np.asarray(hist_data, dtype=np.float64).reshape(-1)
        if len(series) != self.hist_len:
            raise ValueError(f"History length mismatch: actual={len(series)}, expected={self.hist_len}")
        if not np.isfinite(series).all():
            raise ValueError("History contains NaN or Inf.")

        predictions, all_chunk_stats = [], []
        remaining = self.pred_len

        while remaining > 0:
            current_pred_len = min(remaining, self.max_pred_len)
            serialized_series = self.serializer.serialize(self.discretizer.discretize(series))
            prompt = getPrompt(flag="prediction", context=context, input=serialized_series)

            samples, generation_stats = self._generate_texts(
                prompt,
                min_new_tokens=2 * current_pred_len,
                max_new_tokens=2 * current_pred_len + 8,
            )

            parsed_predictions, sample_stats = [], []
            parse_errors = 0
            for sample_index, sample in enumerate(samples):
                try:
                    parsed, info = self._parse_prediction(sample, current_pred_len, sample_index)
                    parsed_predictions.append(parsed)
                    sample_stats.append(info)
                except Exception as error:
                    parse_errors += 1
                    sample_stats.append({
                        "sample_index": int(sample_index),
                        "status": "parse_failed",
                        "numeric_token_count": 0,
                        "raw_prediction_length": 0,
                        "finite_count": 0,
                        "error": str(error),
                    })
                    if self.debug_generation:
                        print(f"Failed to parse Mamba sample {sample_index}: {error}")

            if parsed_predictions:
                prediction_array = np.asarray(parsed_predictions, dtype=np.float64)
                expected_shape = (len(parsed_predictions), current_pred_len)
                if prediction_array.shape != expected_shape:
                    raise ValueError(
                        f"Unexpected prediction array shape: actual={prediction_array.shape}, "
                        f"expected={expected_shape}"
                    )

                valid_mask = np.isfinite(prediction_array)
                valid_counts = valid_mask.sum(axis=0)

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    prediction_before_fallback = np.nanmedian(prediction_array, axis=0)
                    sample_std_per_position = np.nanstd(prediction_array, axis=0)
                    sample_range_per_position = (
                        np.nanmax(prediction_array, axis=0)
                        - np.nanmin(prediction_array, axis=0)
                    )
                    prediction_q10 = np.nanquantile(prediction_array, 0.10, axis=0)
                    prediction_q25 = np.nanquantile(prediction_array, 0.25, axis=0)
                    prediction_q50 = np.nanquantile(prediction_array, 0.50, axis=0)
                    prediction_q75 = np.nanquantile(prediction_array, 0.75, axis=0)
                    prediction_q90 = np.nanquantile(prediction_array, 0.90, axis=0)
            else:
                prediction_array = np.empty((0, current_pred_len), dtype=np.float64)
                valid_mask = np.zeros((0, current_pred_len), dtype=bool)
                valid_counts = np.zeros(current_pred_len, dtype=np.int64)
                prediction_before_fallback = np.full(current_pred_len, np.nan, dtype=np.float64)
                sample_std_per_position = np.full(current_pred_len, np.nan, dtype=np.float64)
                sample_range_per_position = np.full(current_pred_len, np.nan, dtype=np.float64)
                prediction_q10 = np.full(current_pred_len, np.nan, dtype=np.float64)
                prediction_q25 = prediction_q10.copy()
                prediction_q50 = prediction_q10.copy()
                prediction_q75 = prediction_q10.copy()
                prediction_q90 = prediction_q10.copy()

            interval_width_50_per_position = prediction_q75 - prediction_q25
            interval_width_80_per_position = prediction_q90 - prediction_q10

            fallback_mask = ~np.isfinite(prediction_before_fallback)
            prediction = np.asarray(prediction_before_fallback, dtype=np.float64).copy()

            for position in range(len(prediction)):
                if fallback_mask[position]:
                    prediction[position] = prediction[position - 1] if position > 0 else series[-1]

            if not np.isfinite(prediction).all():
                raise ValueError("Non-finite values remain after fallback.")

            complete_samples = sum(item["status"] == "complete" for item in sample_stats)
            partial_samples = sum(item["status"] == "partial" for item in sample_stats)
            required_values = len(samples) * current_pred_len
            parsed_values = int(np.isfinite(prediction_array).sum())

            chunk_stats = {
                **generation_stats,
                "chunk_length": int(current_pred_len),
                "parsed_samples": int(len(parsed_predictions)),
                "complete_samples": int(complete_samples),
                "partial_samples": int(partial_samples),
                "parse_errors": int(parse_errors),
                "required_value_count": int(required_values),
                "parsed_value_count": int(parsed_values),
                "parsed_ratio": float(parsed_values / required_values) if required_values else np.nan,
                "fallback_count": int(fallback_mask.sum()),
                "fallback_ratio": float(fallback_mask.mean()),
                "valid_counts_per_position": valid_counts.tolist(),
                "sample_std_mean": self._finite_summary(sample_std_per_position, "mean"),
                "sample_std_max": self._finite_summary(sample_std_per_position, "max"),
                "sample_range_mean": self._finite_summary(sample_range_per_position, "mean"),
                "sample_range_max": self._finite_summary(sample_range_per_position, "max"),
                "interval_width_50_mean": self._finite_summary(interval_width_50_per_position, "mean"),
                "interval_width_80_mean": self._finite_summary(interval_width_80_per_position, "mean"),
                "interval_width_80_max": self._finite_summary(interval_width_80_per_position, "max"),
                "sample_std_per_position": sample_std_per_position.tolist(),
                "sample_range_per_position": sample_range_per_position.tolist(),
                "prediction_q10": prediction_q10.tolist(),
                "prediction_q25": prediction_q25.tolist(),
                "prediction_q50": prediction_q50.tolist(),
                "prediction_q75": prediction_q75.tolist(),
                "prediction_q90": prediction_q90.tolist(),
                "sample_stats": sample_stats,
            }
            all_chunk_stats.append(chunk_stats)

            if self.verbose:
                print("Mamba chunk:", {
                    "chunk_length": current_pred_len,
                    "complete_samples": complete_samples,
                    "partial_samples": partial_samples,
                    "parse_errors": parse_errors,
                    "parsed_ratio": chunk_stats["parsed_ratio"],
                    "fallback_ratio": chunk_stats["fallback_ratio"],
                })

            predictions.append(prediction)
            remaining -= current_pred_len
            if remaining > 0:
                series = np.concatenate([series, prediction])

        final_prediction = np.concatenate(predictions)[:self.pred_len]
        total_required = sum(chunk["required_value_count"] for chunk in all_chunk_stats)
        total_parsed = sum(chunk["parsed_value_count"] for chunk in all_chunk_stats)
        total_fallback = sum(chunk["fallback_count"] for chunk in all_chunk_stats)
        total_parse_errors = sum(chunk["parse_errors"] for chunk in all_chunk_stats)

        all_sample_std = np.concatenate([
            np.asarray(chunk["sample_std_per_position"], dtype=np.float64)
            for chunk in all_chunk_stats
        ])[:self.pred_len]
        all_sample_range = np.concatenate([
            np.asarray(chunk["sample_range_per_position"], dtype=np.float64)
            for chunk in all_chunk_stats
        ])[:self.pred_len]
        all_prediction_q10 = np.concatenate([
            np.asarray(chunk["prediction_q10"], dtype=np.float64)
            for chunk in all_chunk_stats
        ])[:self.pred_len]
        all_prediction_q25 = np.concatenate([
            np.asarray(chunk["prediction_q25"], dtype=np.float64)
            for chunk in all_chunk_stats
        ])[:self.pred_len]
        all_prediction_q50 = np.concatenate([
            np.asarray(chunk["prediction_q50"], dtype=np.float64)
            for chunk in all_chunk_stats
        ])[:self.pred_len]
        all_prediction_q75 = np.concatenate([
            np.asarray(chunk["prediction_q75"], dtype=np.float64)
            for chunk in all_chunk_stats
        ])[:self.pred_len]
        all_prediction_q90 = np.concatenate([
            np.asarray(chunk["prediction_q90"], dtype=np.float64)
            for chunk in all_chunk_stats
        ])[:self.pred_len]
        all_interval_width_50 = all_prediction_q75 - all_prediction_q25
        all_interval_width_80 = all_prediction_q90 - all_prediction_q10

        self.last_prediction_stats = {
            "chunks": all_chunk_stats,
            "prediction_length": int(len(final_prediction)),
            "parsed_ratio": float(total_parsed / total_required) if total_required else np.nan,
            "fallback_count": int(total_fallback),
            "fallback_ratio": float(total_fallback / self.pred_len) if self.pred_len else np.nan,
            "parse_errors": int(total_parse_errors),
            "prediction_nan_count": int(np.isnan(final_prediction).sum()),
            "sample_std_mean": self._finite_summary(all_sample_std, "mean"),
            "sample_std_max": self._finite_summary(all_sample_std, "max"),
            "sample_range_mean": self._finite_summary(all_sample_range, "mean"),
            "sample_range_max": self._finite_summary(all_sample_range, "max"),
            "interval_width_50_mean": self._finite_summary(all_interval_width_50, "mean"),
            "interval_width_80_mean": self._finite_summary(all_interval_width_80, "mean"),
            "interval_width_80_max": self._finite_summary(all_interval_width_80, "max"),
            "sample_std_per_position": all_sample_std.tolist(),
            "sample_range_per_position": all_sample_range.tolist(),
            "prediction_q10": all_prediction_q10.tolist(),
            "prediction_q25": all_prediction_q25.tolist(),
            "prediction_q50": all_prediction_q50.tolist(),
            "prediction_q75": all_prediction_q75.tolist(),
            "prediction_q90": all_prediction_q90.tolist(),
        }
        return final_prediction

    def analyze(self, question, series):
        series = np.asarray(series, dtype=np.float64)
        serialized_series = self.serializer.serialize(self.discretizer.discretize(series))
        prompt = getPrompt(flag="analysis", instruction=question, input=serialized_series)
        samples, _ = self._generate_texts(prompt, min_new_tokens=None, max_new_tokens=self.max_pred_len)

        responses = []
        for sample in samples:
            response = self._extract_response(sample)
            match = re.findall(r"\([abc]\)", response)
            if match:
                responses.append(match[0])
        return Counter(responses).most_common(1)[0][0] if responses else None
