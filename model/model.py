#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import time
import warnings
from collections import Counter

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoTokenizer, LlamaForCausalLM

from utils.prompt import getPrompt
from utils.tools import Discretizer, Serializer


class ChatTime:
    NUMERIC_PATTERN = re.compile(r"###(?:[+-]?(?:\d+(?:\.\d*)?|\.\d+)|Nan|NaN|nan)###")

    def __init__(self, base_model_path, adapter_path=None, tokenizer_path=None, merge_adapter=False,
                 hist_len=None, pred_len=None, max_pred_len=16, num_samples=8, top_k=100,
                 top_p=1.0, temperature=1.0, torch_dtype=None, device_map="auto",
                 local_files_only=True, debug_generation=False, debug_samples=2, verbose=False,):
        self.base_model_path = base_model_path
        self.adapter_path = adapter_path
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.max_pred_len = max_pred_len
        self.num_samples = num_samples
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature
        self.debug_generation = debug_generation
        self.debug_samples = max(0, int(debug_samples))
        self.discretizer = Discretizer()
        self.serializer = Serializer()
        self.last_prediction_stats = {}
        self.torch_dtype = torch_dtype or (torch.float16 if torch.cuda.is_available() else torch.float32)
        self.verbose = bool(verbose)

        tokenizer_source = tokenizer_path or base_model_path
        print("Loading tokenizer from:", tokenizer_source)
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source, use_fast=True, trust_remote_code=True,
            local_files_only=local_files_only,
        )
        if self.tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer does not define eos_token_id.")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        self.eos_token_id = self.tokenizer.eos_token_id

        print("Loading Llama model from:", base_model_path)
        base_model = LlamaForCausalLM.from_pretrained(
            base_model_path, low_cpu_mem_usage=True, return_dict=True,
            torch_dtype=self.torch_dtype, device_map=device_map,
            local_files_only=local_files_only,
        )
        tokenizer_size = len(self.tokenizer)
        input_size = base_model.get_input_embeddings().weight.shape[0]
        output_layer = base_model.get_output_embeddings()
        output_size = output_layer.weight.shape[0] if output_layer is not None else None
        print({"tokenizer_vocab_size": tokenizer_size, "input_embedding_size": input_size,
               "lm_head_size": output_size})
        if input_size != tokenizer_size or (output_size is not None and output_size != tokenizer_size):
            raise ValueError(
                "Model/tokenizer vocabulary mismatch. Use the tokenizer saved with the CPT-merged model: "
                f"tokenizer={tokenizer_size}, input={input_size}, output={output_size}"
            )

        if adapter_path is not None:
            print("Loading adapter:", adapter_path)
            self.model = PeftModel.from_pretrained(
                base_model, adapter_path, is_trainable=False,
                local_files_only=local_files_only,
            )
            if merge_adapter:
                print("Merging adapter into base model")
                self.model = self.model.merge_and_unload(safe_merge=True)
        else:
            print("Adapter is not specified. Using the base model only.")
            self.model = base_model

        self.model.eval()
        if hasattr(self.model, "config"):
            self.model.config.use_cache = True
        if hasattr(self.model, "generation_config"):
            self.model.generation_config.use_cache = True
            self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
            self.model.generation_config.eos_token_id = self.eos_token_id

    def _device(self):
        return next(self.model.parameters()).device

    def _extract_numeric_tokens(self, text):
        return self.NUMERIC_PATTERN.findall(text)

    def _generate(self, prompt, min_new_tokens, max_new_tokens):
        encoded = self.tokenizer(prompt, add_special_tokens=True, return_tensors="pt", padding=False, truncation=False)
        encoded = {key: value.to(self._device()) for key, value in encoded.items()}
        input_len = encoded["input_ids"].shape[-1]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            outputs = self.model.generate(
                **encoded, min_new_tokens=min_new_tokens, max_new_tokens=max_new_tokens,
                do_sample=True, num_return_sequences=self.num_samples, top_k=self.top_k,
                top_p=self.top_p, temperature=self.temperature,
                eos_token_id=self.eos_token_id, pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started

        samples, generated_lengths = [], []
        for sample_index, output_ids in enumerate(outputs):
            generated_ids = output_ids[input_len:]
            generated_lengths.append(int(generated_ids.numel()))
            decoded = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            samples.append(decoded)
            if self.debug_generation and sample_index < self.debug_samples:
                print("\n" + "=" * 80)
                print(f"Llama generated sample {sample_index}")
                print("IDs:", generated_ids.tolist())
                print("Tokens:", self.tokenizer.convert_ids_to_tokens(generated_ids.tolist()))
                print("Decoded:", repr(decoded))

        total_tokens = sum(generated_lengths)
        generation_stats = {
            "input_tokens": int(input_len), "samples": int(len(samples)),
            "generated_lengths": generated_lengths, "total_generated_tokens": int(total_tokens),
            "generate_seconds": float(elapsed),
            "tokens_per_second": float(total_tokens / max(elapsed, 1e-9)),
        }
        if self.verbose:
            print("Llama generation:", generation_stats)    
        return samples, generation_stats

    def _parse_prediction_sample(self, sample, current_pred_len, sample_index):
        numeric_tokens = self._extract_numeric_tokens(sample)

        if self.debug_generation and sample_index < self.debug_samples:
            print("Extracted numeric tokens:", numeric_tokens)
            print("Numeric token count:", len(numeric_tokens))

        if not numeric_tokens:
            raise ValueError(
                "No numeric tokens matching ###number### were generated. "
                f"sample={sample[:300]!r}"
            )

        serialized = " ".join(numeric_tokens)
        dispersed = self.serializer.inverse_serialize(serialized)
        dispersed = np.asarray(dispersed, dtype=np.float64).reshape(-1)

        if len(dispersed) == 0:
            raise ValueError("Serializer returned an empty array.")

        if not np.isfinite(dispersed).any():
            raise ValueError("Serializer returned no finite values.")

        pred = self.discretizer.inverse_discretize(dispersed)
        pred = np.asarray(pred, dtype=np.float64).reshape(-1)

        if len(pred) == 0:
            raise ValueError("Inverse-discretized prediction is empty.")

        raw_length = len(pred)

        if raw_length < current_pred_len:
            pred = np.concatenate([
                pred,
                np.full(
                    current_pred_len - raw_length,
                    np.nan,
                    dtype=np.float64,
                ),
            ])

        pred = pred[:current_pred_len]
        finite_count = int(np.isfinite(pred).sum())

        if finite_count == 0:
            raise ValueError(
                "Inverse-discretized prediction contains no finite values."
            )

        if self.debug_generation and sample_index < self.debug_samples:
            print("Dispersed prediction:", dispersed)
            print("Inverse-discretized prediction:", pred)
            print(
                "Finite values:",
                finite_count,
                "/",
                current_pred_len,
            )

        return pred, numeric_tokens, raw_length

    def predict(self, hist_data, context=None):
        if self.hist_len is None or self.pred_len is None:
            raise ValueError("hist_len and pred_len must be specified before prediction.")
        series = np.asarray(hist_data, dtype=np.float64).reshape(-1).copy()
        if len(series) != self.hist_len:
            raise ValueError(f"History length mismatch: actual={len(series)}, expected={self.hist_len}")
        if not np.isfinite(series).all():
            raise ValueError("History contains NaN or Inf.")

        prediction_chunks, all_chunk_stats = [], []
        remaining = self.pred_len
        while remaining > 0:
            current_pred_len = min(remaining, self.max_pred_len)
            dispersed_series = self.discretizer.discretize(series)
            serialized_series = self.serializer.serialize(dispersed_series)
            prompt = getPrompt(flag="prediction", context=context, input=serialized_series)
            if serialized_series[:100] not in prompt:
                raise ValueError("Serialized history is not included in the generated prompt. Check utils/prompt.py.")

            min_new_tokens = 2 * current_pred_len
            max_new_tokens = 2 * current_pred_len + 8
            samples, generation_stats = self._generate(prompt, min_new_tokens, max_new_tokens)

            pred_list, sample_stats = [], []
            parse_errors = complete_samples = partial_samples = 0
            for sample_index, sample in enumerate(samples):
                try:
                    pred, numeric_tokens, raw_length = self._parse_prediction_sample(
                        sample, current_pred_len, sample_index
                    )
                    finite_count = int(np.isfinite(pred).sum())
                    status = "complete" if finite_count == current_pred_len else "partial"
                    complete_samples += status == "complete"
                    partial_samples += status == "partial"
                    pred_list.append(pred)
                    sample_stats.append({
                        "sample_index": sample_index, "status": status,
                        "numeric_token_count": len(numeric_tokens), "raw_prediction_length": raw_length,
                        "finite_count": finite_count,
                    })
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
                        print(
                            f"Failed to parse Llama sample "
                            f"{sample_index}: {error}"
                        )
                        print(
                            "Raw generated sample:",
                            repr(sample[:1000]),
                        )

            if pred_list:
                pred_array = np.asarray(pred_list, dtype=np.float64)
                expected = (len(pred_list), current_pred_len)
                if pred_array.shape != expected:
                    raise ValueError(f"Unexpected prediction shape: actual={pred_array.shape}, expected={expected}")
                valid_mask = np.isfinite(pred_array)
                valid_counts = valid_mask.sum(axis=0)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    prediction_before_fallback = np.nanmedian(pred_array, axis=0)
            else:
                pred_array = np.empty((0, current_pred_len), dtype=np.float64)
                valid_mask = np.zeros((0, current_pred_len), dtype=bool)
                valid_counts = np.zeros(current_pred_len, dtype=np.int64)
                prediction_before_fallback = np.full(current_pred_len, np.nan)

            fallback_mask = ~np.isfinite(prediction_before_fallback)
            prediction = np.asarray(prediction_before_fallback, dtype=np.float64).copy()
            for index in range(current_pred_len):
                if fallback_mask[index]:
                    prediction[index] = prediction[index - 1] if index > 0 else series[-1]

            required_values = len(samples) * current_pred_len
            parsed_values = int(valid_mask.sum())
            chunk_stats = {
                "chunk_length": int(current_pred_len), **generation_stats,
                "parsed_samples": int(len(pred_list)), "complete_samples": int(complete_samples),
                "partial_samples": int(partial_samples), "parse_errors": int(parse_errors),
                "required_value_count": int(required_values), "parsed_value_count": parsed_values,
                "parsed_ratio": float(parsed_values / required_values) if required_values else np.nan,
                "fallback_count": int(fallback_mask.sum()), "fallback_ratio": float(fallback_mask.mean()),
                "valid_counts_per_position": valid_counts.tolist(), "sample_stats": sample_stats,
            }
            all_chunk_stats.append(chunk_stats)
            if self.verbose:
                print("Llama chunk:", {
                    "chunk_length": int(current_pred_len),
                    "generated_samples": int(len(samples)),
                    "complete_samples": int(complete_samples),
                    "partial_samples": int(partial_samples),
                    "parse_errors": int(parse_errors),
                    "parsed_ratio": chunk_stats["parsed_ratio"],
                    "fallback_ratio": chunk_stats["fallback_ratio"],
                    "generate_seconds": generation_stats[
                        "generate_seconds"
                    ],
                })

            if self.debug_generation:
                print("Prediction array:\n", pred_array)
                print("Prediction before fallback:", prediction_before_fallback)
                print("Fallback positions:", np.where(fallback_mask)[0].tolist())
                print("Final chunk prediction:", prediction)

            prediction_chunks.append(prediction)
            remaining -= current_pred_len
            if remaining > 0:
                series = np.concatenate([series, prediction])

        final_prediction = np.concatenate(prediction_chunks)[:self.pred_len]
        if len(final_prediction) != self.pred_len or not np.isfinite(final_prediction).all():
            raise ValueError("Final prediction is invalid.")
        total_required = sum(chunk["required_value_count"] for chunk in all_chunk_stats)
        total_parsed = sum(chunk["parsed_value_count"] for chunk in all_chunk_stats)
        total_fallback = sum(chunk["fallback_count"] for chunk in all_chunk_stats)
        total_parse_errors = sum(
            chunk["parse_errors"]
            for chunk in all_chunk_stats
        )
        self.last_prediction_stats = {
            "chunks": all_chunk_stats,
            "parsed_ratio": (
                float(total_parsed / total_required)
                if total_required
                else np.nan
            ),
            "fallback_count": int(total_fallback),
            "fallback_ratio": (
                float(total_fallback / self.pred_len)
                if self.pred_len
                else np.nan
            ),
            "parse_errors": int(total_parse_errors),
            "prediction_length": int(len(final_prediction)),
            "prediction_nan_count": int(
                np.isnan(final_prediction).sum()
            ),
        }
        if self.debug_generation:
            print("\nFinal Llama prediction:", final_prediction)
            print("Final prediction statistics:", self.last_prediction_stats)
        return final_prediction

    def analyze(self, question, series):
        series = np.asarray(series, dtype=np.float64).reshape(-1)
        dispersed_series = self.discretizer.discretize(series)
        serialized_series = self.serializer.serialize(dispersed_series)
        prompt = getPrompt(flag="analysis", instruction=question, input=serialized_series)
        samples, _ = self._generate(prompt, min_new_tokens=1, max_new_tokens=self.max_pred_len)
        responses = []
        for sample in samples:
            matches = re.findall(r"\([abc]\)", sample)
            if matches:
                responses.append(matches[0])
        return Counter(responses).most_common(1)[0][0] if responses else None
