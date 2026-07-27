import os
import re
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.prompt import getPrompt
from utils.tools import Discretizer, Serializer


NUMERIC_TOKEN_RE = re.compile(
    r"###([+-]?(?:\d+(?:\.\d*)?|\.\d+)|Nan|NaN|nan)###"
)


def parse_periods(text):
    if isinstance(text, (list, tuple)):
        periods = [float(value) for value in text]
    else:
        periods = [float(value.strip()) for value in str(text).split(",") if value.strip()]
    if not periods or any(period <= 0 for period in periods):
        raise ValueError("fone_periods must contain positive values.")
    return periods


def collect_numeric_token_ids(tokenizer):
    rows = []
    for token, token_id in tokenizer.get_vocab().items():
        match = NUMERIC_TOKEN_RE.fullmatch(str(token))
        if match is None or match.group(1).lower() == "nan":
            continue
        rows.append((int(token_id), float(match.group(1))))
    rows.sort(key=lambda item: item[1])
    if not rows:
        raise ValueError(
            "No numeric tokens were found in the tokenizer. Load the tokenizer saved with the adapter."
        )
    return (
        torch.tensor([item[0] for item in rows], dtype=torch.long),
        torch.tensor([item[1] for item in rows], dtype=torch.float32),
    )


def make_fone_features(values, periods):
    values = torch.as_tensor(values, dtype=torch.float64).reshape(-1, 1)
    periods_tensor = torch.tensor(periods, dtype=torch.float64, device=values.device).reshape(1, -1)
    angles = 2.0 * torch.pi * values / periods_tensor
    return torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1).flatten(1).float()


class FoNEFeatureAdapter(nn.Module):
    """Exact inference counterpart of the adapter used by pretrain_mamba_fone_advanced.py."""

    def __init__(self, vocab_size, model_dim, numeric_ids, numeric_values, periods,
                 mode="none", scale=1.0, projection_trainable=False):
        super().__init__()
        self.mode = mode
        self.scale = float(scale)
        self.periods = list(periods)
        self.feature_dim = 2 * len(periods)

        value_table = torch.zeros(vocab_size, dtype=torch.float32)
        numeric_mask = torch.zeros(vocab_size, dtype=torch.bool)
        value_table[numeric_ids] = numeric_values.float()
        numeric_mask[numeric_ids] = True
        self.register_buffer("value_table", value_table)
        self.register_buffer("numeric_mask", numeric_mask)

        feature_table = torch.zeros(vocab_size, self.feature_dim, dtype=torch.float32)
        feature_table[numeric_ids] = make_fone_features(numeric_values, periods)
        self.register_buffer("absolute_feature_table", feature_table)

        self.absolute_projection = None
        self.delta_projection = None
        if mode in {"additive_projection", "absolute_delta"}:
            self.absolute_projection = nn.Linear(self.feature_dim, model_dim, bias=False)
            self.absolute_projection.weight.requires_grad_(projection_trainable)
        if mode == "absolute_delta":
            self.delta_projection = nn.Linear(self.feature_dim, model_dim, bias=False)
            self.delta_projection.weight.requires_grad_(projection_trainable)

    def forward(self, input_ids, base_embeddings):
        if self.mode == "none":
            return base_embeddings

        dtype = base_embeddings.dtype
        absolute_features = self.absolute_feature_table[input_ids].to(dtype)
        numeric = self.numeric_mask[input_ids]

        if self.mode == "residual":
            model_dim = base_embeddings.shape[-1]
            if self.feature_dim >= model_dim:
                fixed = absolute_features[..., :model_dim]
            else:
                fixed = F.pad(absolute_features, (0, model_dim - self.feature_dim))
            output = base_embeddings + self.scale * fixed
        else:
            output = base_embeddings + self.scale * self.absolute_projection(absolute_features)

        if self.mode == "absolute_delta":
            values = self.value_table[input_ids]
            previous_values = torch.roll(values, shifts=1, dims=1)
            previous_numeric = torch.roll(numeric, shifts=1, dims=1)
            previous_numeric[:, 0] = False
            valid_delta = numeric & previous_numeric
            delta_values = torch.where(valid_delta, values - previous_values, torch.zeros_like(values))
            delta_features = make_fone_features(delta_values.reshape(-1), self.periods).to(
                device=input_ids.device, dtype=dtype
            ).reshape(*input_ids.shape, self.feature_dim)
            delta_features = delta_features * valid_delta.unsqueeze(-1)
            output = output + self.scale * self.delta_projection(delta_features)

        return output


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
        dtype=None,
        merge_lora=False,
        merge_adapter=None,
        debug_generation=False,
        debug_samples=2,
        fone_mode="none",
        fone_periods="0.001,0.01,0.1,1,10",
        fone_scale=1.0,
        fone_adapter_path=None,
        freeze_fone_projection=False,
        smoothness_lambda=0.0,
        residual_lambda=0.0,
        **kwargs,
    ):
        if dtype is not None:
            torch_dtype = dtype
        if merge_adapter is not None:
            merge_lora = merge_adapter

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
        self.debug_samples = debug_samples
        self.last_prediction_stats = {}

        self.fone_mode = str(fone_mode)
        self.fone_periods = parse_periods(fone_periods)
        self.fone_scale = float(fone_scale)
        self.fone_adapter_path = fone_adapter_path
        self.numeric_fone_adapter = None

        self.discretizer = Discretizer()
        self.serializer = Serializer()

        tokenizer_path = tokenizer_path or adapter_path or base_model_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
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
        self.model.resize_token_embeddings(len(self.tokenizer))

        if adapter_path is not None:
            adapter_config_path = os.path.join(adapter_path, "adapter_config.json")
            if not os.path.exists(adapter_config_path):
                raise FileNotFoundError(f"adapter_config.json not found: {adapter_config_path}")
            self.model = PeftModel.from_pretrained(self.model, adapter_path, is_trainable=False)
            if merge_lora:
                print("Merging LoRA adapter into base model...")
                self.model = self.model.merge_and_unload()

        self.model.eval()
        if hasattr(self.model, "config"):
            self.model.config.use_cache = self.fone_mode == "none"
        if hasattr(self.model, "generation_config"):
            self.model.generation_config.use_cache = self.fone_mode == "none"

        if self.fone_mode != "none":
            self._load_fone_adapter()

    def _device(self):
        return next(self.model.parameters()).device

    def _load_fone_adapter(self):
        if self.fone_mode not in {"additive_projection", "residual", "absolute_delta"}:
            raise ValueError(f"Unsupported fone_mode: {self.fone_mode}")
        if self.fone_adapter_path is None:
            raise ValueError("fone_adapter_path is required when fone_mode is enabled.")
        if not os.path.exists(self.fone_adapter_path):
            raise FileNotFoundError(f"FoNE adapter not found: {self.fone_adapter_path}")

        numeric_ids, numeric_values = collect_numeric_token_ids(self.tokenizer)
        input_embedding = self.model.get_input_embeddings()
        self.numeric_fone_adapter = FoNEFeatureAdapter(
            vocab_size=len(self.tokenizer),
            model_dim=input_embedding.weight.shape[1],
            numeric_ids=numeric_ids,
            numeric_values=numeric_values,
            periods=self.fone_periods,
            mode=self.fone_mode,
            scale=self.fone_scale,
            projection_trainable=False,
        )
        state_dict = torch.load(self.fone_adapter_path, map_location="cpu")
        self.numeric_fone_adapter.load_state_dict(state_dict, strict=True)
        self.numeric_fone_adapter.to(device=self._device(), dtype=input_embedding.weight.dtype)
        self.numeric_fone_adapter.eval()
        for parameter in self.numeric_fone_adapter.parameters():
            parameter.requires_grad_(False)

        print({
            "fone_mode": self.fone_mode,
            "fone_periods": self.fone_periods,
            "fone_scale": self.fone_scale,
            "fone_adapter_path": self.fone_adapter_path,
            "numeric_tokens": int(len(numeric_ids)),
        })

    def _build_input_embeddings(self, input_ids):
        base_embeddings = self.model.get_input_embeddings()(input_ids)
        if self.numeric_fone_adapter is None:
            return base_embeddings
        return self.numeric_fone_adapter(input_ids, base_embeddings)

    def _sample_next_token(self, logits):
        temperature = max(float(self.temperature), 1e-6)
        logits = logits / temperature

        if self.top_k is not None and self.top_k > 0:
            top_k = min(int(self.top_k), logits.shape[-1])
            threshold = torch.topk(logits, top_k, dim=-1).values[:, -1:].clone()
            logits = logits.masked_fill(logits < threshold, float("-inf"))

        probabilities = torch.softmax(logits, dim=-1)
        if self.top_p is not None and self.top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probabilities, descending=True, dim=-1)
            cumulative = sorted_probs.cumsum(dim=-1)
            remove = cumulative > float(self.top_p)
            remove[:, 1:] = remove[:, :-1].clone()
            remove[:, 0] = False
            sorted_probs = sorted_probs.masked_fill(remove, 0.0)
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            sampled_rank = torch.multinomial(sorted_probs, num_samples=1)
            return sorted_indices.gather(dim=-1, index=sampled_rank)
        return torch.multinomial(probabilities, num_samples=1)

    @torch.inference_mode()
    def _generate_with_fone(self, input_ids, min_new_tokens=None, max_new_tokens=64):
        # Correctness-first implementation: recompute the complete prefix each step.
        # This is required for absolute_delta because the current token depends on its predecessor.
        generated = input_ids.repeat_interleave(self.num_samples, dim=0)
        prompt_length = generated.shape[1]
        finished = torch.zeros(generated.shape[0], dtype=torch.bool, device=generated.device)
        min_new_tokens = 0 if min_new_tokens is None else int(min_new_tokens)

        for step in range(int(max_new_tokens)):
            embeddings = self._build_input_embeddings(generated)
            attention_mask = torch.ones(generated.shape, dtype=torch.long, device=generated.device)
            outputs = self.model(inputs_embeds=embeddings, attention_mask=attention_mask, use_cache=False)
            next_token = self._sample_next_token(outputs.logits[:, -1, :])

            if step + 1 >= min_new_tokens and self.eos_token_id is not None:
                next_token = torch.where(
                    finished.unsqueeze(-1),
                    torch.full_like(next_token, self.tokenizer.pad_token_id),
                    next_token,
                )
                finished |= next_token.squeeze(-1).eq(self.eos_token_id)

            generated = torch.cat((generated, next_token), dim=1)
            if step + 1 >= min_new_tokens and finished.all():
                break

        return generated, prompt_length

    def _generate_texts(self, prompt, min_new_tokens=None, max_new_tokens=64):
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=False, truncation=False)
        inputs = {key: value.to(self._device()) for key, value in inputs.items()}
        input_len = inputs["input_ids"].shape[-1]

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()

        if self.numeric_fone_adapter is None:
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
        else:
            outputs, input_len = self._generate_with_fone(
                inputs["input_ids"], min_new_tokens=min_new_tokens, max_new_tokens=max_new_tokens
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        generated_lengths, generated_texts = [], []
        for sample_index, output in enumerate(outputs):
            generated_ids = output[input_len:]
            generated_tokens = self.tokenizer.convert_ids_to_tokens(generated_ids.tolist())
            decoded_text = self.tokenizer.decode(generated_ids, skip_special_tokens=False)
            generated_lengths.append(int(generated_ids.numel()))
            generated_texts.append(decoded_text)
            if self.debug_generation and sample_index < self.debug_samples:
                print(f"\nGenerated sample {sample_index + 1}")
                print("IDs:", generated_ids.tolist())
                print("Tokens:", generated_tokens)
                print("Decoded:", repr(decoded_text))

        total_generated_tokens = sum(generated_lengths)
        print({
            "input_tokens": input_len,
            "samples": self.num_samples,
            "generated_lengths": generated_lengths,
            "total_generated_tokens": total_generated_tokens,
            "generate_seconds": elapsed,
            "tokens_per_second": total_generated_tokens / max(elapsed, 1e-9),
            "fone_mode": self.fone_mode,
        })
        return generated_texts

    def _extract_response(self, text):
        if "### Response:\n" in text:
            return text.split("### Response:\n", 1)[1]
        if "### Response:" in text:
            return text.split("### Response:", 1)[1]
        return text

    def _extract_numeric_tokens(self, text):
        tokens = re.findall(r"###(?:[+-]?\d+(?:\.\d+)?|Nan|NaN|nan)###", text)
        return " ".join(tokens)

    def predict(self, hist_data, context=None):
        if self.hist_len is None or self.pred_len is None:
            raise ValueError("hist_len and pred_len must be specified before prediction")

        series = np.asarray(hist_data, dtype=np.float64).copy()
        prediction_list = []
        remaining = self.pred_len
        total_samples = total_parsed = total_parse_errors = 0

        while remaining > 0:
            current_pred_len = min(remaining, self.max_pred_len)
            dispersed_series = self.discretizer.discretize(series)
            serialized_series = self.serializer.serialize(dispersed_series)
            prompt = getPrompt(flag="prediction", context=context, input=serialized_series)

            tokens_per_value = 2
            min_new_tokens = current_pred_len * tokens_per_value
            max_new_tokens = current_pred_len * tokens_per_value + 16
            samples = self._generate_texts(prompt, min_new_tokens=min_new_tokens, max_new_tokens=max_new_tokens)

            pred_list = []
            parse_errors = 0
            for sample in samples:
                try:
                    serialized_prediction = self._extract_numeric_tokens(self._extract_response(sample))
                    dispersed_prediction = self.serializer.inverse_serialize(serialized_prediction)
                    pred = np.asarray(
                        self.discretizer.inverse_discretize(dispersed_prediction), dtype=np.float64
                    ).reshape(-1)
                    if len(pred) == 0:
                        raise ValueError(f"Parsed prediction is empty: {sample[:300]!r}")
                    if len(pred) < current_pred_len:
                        pred = np.concatenate((pred, np.full(current_pred_len - len(pred), np.nan)))
                    pred_list.append(pred[:current_pred_len])
                except Exception as error:
                    parse_errors += 1
                    print(f"Failed to parse prediction sample: {error}")

            if not pred_list or np.isnan(np.asarray(pred_list)).all():
                prediction = np.full(current_pred_len, series[-1], dtype=np.float64)
            else:
                with np.errstate(all="ignore"):
                    prediction = np.nanmedian(np.asarray(pred_list, dtype=np.float64), axis=0)
                prediction = np.where(np.isfinite(prediction), prediction, series[-1])

            prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
            if len(prediction) != current_pred_len:
                raise ValueError(
                    f"Final chunk length mismatch: actual={len(prediction)}, expected={current_pred_len}"
                )

            total_samples += len(samples)
            total_parsed += len(pred_list)
            total_parse_errors += parse_errors
            prediction_list.append(prediction)
            remaining -= current_pred_len
            if remaining > 0:
                series = np.concatenate((series, prediction), axis=-1)

        final_prediction = np.concatenate(prediction_list, axis=-1)[:self.pred_len]
        self.last_prediction_stats = {
            "generated_samples": total_samples,
            "parsed_samples": total_parsed,
            "parse_errors": total_parse_errors,
            "parsed_ratio": total_parsed / max(total_samples, 1),
            "fallback_ratio": 1.0 if total_parsed == 0 else 0.0,
            "prediction_length": len(final_prediction),
            "prediction_nan_count": int(np.isnan(final_prediction).sum()),
        }
        print(self.last_prediction_stats)
        return final_prediction

    def analyze(self, question, series):
        dispersed_series = self.discretizer.discretize(series)
        serialized_series = self.serializer.serialize(dispersed_series)
        prompt = getPrompt(flag="analysis", instruction=question, input=serialized_series)
        samples = self._generate_texts(prompt, min_new_tokens=None, max_new_tokens=self.max_pred_len)
        response_list = []
        for sample in samples:
            try:
                response = self._extract_response(sample).split(".")[0] + "."
                match = re.findall(r"\([abc]\)", response)
                if match:
                    response_list.append(match[0])
            except Exception as error:
                print(f"Failed to parse analysis sample: {error}")
        return Counter(response_list).most_common(1)[0][0] if response_list else None
