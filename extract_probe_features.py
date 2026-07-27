#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

NUMERIC_TOKEN_RE = re.compile(r"^###([+-]?(?:\d+(?:\.\d*)?|\.\d+))###$")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract aligned hidden states for numeric-token probing."
    )
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--tokenizer_path", default=None)
    parser.add_argument("--adapter_path", default=None)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model_label", required=True)
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--layers", default="0,0.25,0.5,0.75,1.0")
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--max_series", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--dtype_argument",
        choices=["auto", "dtype", "torch_dtype"],
        default="auto",
        help="Use torch_dtype for older Mamba environments.",
    )
    parser.add_argument(
        "--local_files_only",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--add_bos",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def load_model(path, dtype, device_map, local_files_only, dtype_argument):
    common = {
        "device_map": device_map,
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
        "local_files_only": local_files_only,
    }
    if dtype_argument == "dtype":
        return AutoModelForCausalLM.from_pretrained(path, dtype=dtype, **common)
    if dtype_argument == "torch_dtype":
        return AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype, **common)
    try:
        return AutoModelForCausalLM.from_pretrained(path, dtype=dtype, **common)
    except TypeError as error:
        if "unexpected keyword argument 'dtype'" not in str(error):
            raise
        print("Retrying model load with torch_dtype= for this Transformers version.")
        return AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype, **common)


def build_numeric_vocabulary(tokenizer):
    entries = []
    for token, token_id in tokenizer.get_vocab().items():
        match = NUMERIC_TOKEN_RE.fullmatch(token)
        if match:
            entries.append((float(match.group(1)), int(token_id), token))
    if not entries:
        raise ValueError("No numeric tokens matching ###number### were found.")
    entries.sort(key=lambda item: item[0])
    values = np.asarray([item[0] for item in entries], dtype=np.float64)
    ids = np.asarray([item[1] for item in entries], dtype=np.int64)
    tokens = [item[2] for item in entries]
    if np.any(np.diff(values) <= 0):
        raise ValueError("Numeric-token values are not strictly increasing.")
    return values, ids, tokens


def nearest_numeric_indices(raw_values, numeric_values):
    raw_values = np.asarray(raw_values, dtype=np.float64)
    right = np.searchsorted(numeric_values, raw_values, side="left")
    right = np.clip(right, 0, len(numeric_values) - 1)
    left = np.clip(right - 1, 0, len(numeric_values) - 1)
    choose_right = np.abs(numeric_values[right] - raw_values) < np.abs(numeric_values[left] - raw_values)
    return np.where(choose_right, right, left)


def resolve_separator_ids(tokenizer):
    encoded = tokenizer.encode(" ", add_special_tokens=False)
    if not encoded:
        raise ValueError("Tokenizer produced no token ID for a space separator.")
    return [int(value) for value in encoded]


def build_numeric_input_ids(raw_values, tokenizer, numeric_values, numeric_ids, add_bos):
    nearest = nearest_numeric_indices(raw_values, numeric_values)
    quantized_values = numeric_values[nearest]
    selected_ids = numeric_ids[nearest]
    separator_ids = resolve_separator_ids(tokenizer)
    ids = []
    numeric_positions = []
    if add_bos and tokenizer.bos_token_id is not None:
        ids.append(int(tokenizer.bos_token_id))
    for index, token_id in enumerate(selected_ids.tolist()):
        numeric_positions.append(len(ids))
        ids.append(int(token_id))
        if index + 1 < len(selected_ids):
            ids.extend(separator_ids)
    return ids, numeric_positions, quantized_values


def local_slope(values):
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2:
        return np.nan
    time = np.arange(len(values), dtype=np.float64)
    time -= time.mean()
    denominator = float(np.dot(time, time))
    if denominator <= 0:
        return np.nan
    return float(np.dot(time, values - values.mean()) / denominator)


def make_targets(values, index, window):
    local = values[max(0, index - window + 1):index + 1]
    return {
        "value": float(values[index]),
        "abs_value": float(abs(values[index])),
        "sign": int(np.sign(values[index]) + 1),
        "current_delta": float(values[index] - values[index - 1]) if index > 0 else np.nan,
        "next_value": float(values[index + 1]) if index + 1 < len(values) else np.nan,
        "next_delta": float(values[index + 1] - values[index]) if index + 1 < len(values) else np.nan,
        "second_diff": float(values[index] - 2 * values[index - 1] + values[index - 2]) if index > 1 else np.nan,
        "local_slope": local_slope(local),
        "local_volatility": float(np.std(local, ddof=0)) if len(local) > 1 else np.nan,
    }


def resolve_layers(hidden_state_count, fractions):
    last = hidden_state_count - 1
    layers = sorted(set(int(round(fraction * last)) for fraction in fractions))
    if any(layer < 0 or layer > last for layer in layers):
        raise ValueError(f"Resolved layer outside [0, {last}]: {layers}")
    return layers


def main():
    args = parse_args()
    if args.batch_size != 1:
        raise ValueError("This validated version currently supports --batch_size 1 only.")
    if args.window < 2:
        raise ValueError("--window must be at least 2.")

    dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[args.dtype]

    tokenizer_source = args.tokenizer_path or args.model_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        use_fast=True,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    model = load_model(
        args.model_path,
        dtype,
        args.device_map,
        args.local_files_only,
        args.dtype_argument,
    )

    model_vocab_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) != model_vocab_size:
        raise ValueError(
            f"Tokenizer/model vocabulary mismatch: tokenizer={len(tokenizer)}, model={model_vocab_size}"
        )
    if args.adapter_path:
        model = PeftModel.from_pretrained(
            model,
            args.adapter_path,
            is_trainable=False,
            local_files_only=True,
        )
    model.eval()
    if hasattr(model, "config"):
        model.config.output_hidden_states = True
        model.config.use_cache = False

    numeric_values, numeric_ids, numeric_tokens = build_numeric_vocabulary(tokenizer)
    print({
        "model_class": type(model).__name__,
        "parameter_dtype": str(next(model.parameters()).dtype),
        "tokenizer_size": len(tokenizer),
        "numeric_token_count": len(numeric_values),
        "numeric_min": float(numeric_values[0]),
        "numeric_max": float(numeric_values[-1]),
    })

    frame = pd.read_csv(args.dataset)
    required_columns = {"series_id", "kind", "values"}
    missing = required_columns - set(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    if args.max_series > 0:
        frame = frame.iloc[:args.max_series].copy()

    fractions = [float(value.strip()) for value in args.layers.split(",") if value.strip()]
    if not fractions or any(value < 0 or value > 1 for value in fractions):
        raise ValueError("--layers must contain fractions in [0, 1].")

    metadata_rows = []
    feature_rows = None
    selected_layers = None
    device = next(model.parameters()).device

    for row_number, record in frame.iterrows():
        raw_values = np.fromstring(str(record["values"]), sep=" ", dtype=np.float64)
        if len(raw_values) < 3 or not np.isfinite(raw_values).all():
            raise ValueError(f"Invalid values in series {record['series_id']}")

        input_ids, numeric_positions, quantized_values = build_numeric_input_ids(
            raw_values,
            tokenizer,
            numeric_values,
            numeric_ids,
            args.add_bos,
        )
        if len(numeric_positions) != len(raw_values):
            raise RuntimeError("Internal numeric-position alignment failure.")

        attention_mask = [1] * len(input_ids)
        inputs = {
            "input_ids": torch.tensor([input_ids], dtype=torch.long, device=device),
            "attention_mask": torch.tensor([attention_mask], dtype=torch.long, device=device),
        }
        with torch.inference_mode():
            output = model(
                **inputs,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        hidden_states = getattr(output, "hidden_states", None)
        if hidden_states is None:
            raise ValueError(
                f"{type(model).__name__} returned hidden_states=None. "
                "This model/Transformers version may not support output_hidden_states."
            )

        if selected_layers is None:
            selected_layers = resolve_layers(len(hidden_states), fractions)
            feature_rows = {
                (layer, state): []
                for layer in selected_layers
                for state in ("token_state", "pre_token_state")
            }
            print({
                "hidden_state_count": len(hidden_states),
                "hidden_state_shapes": [tuple(value.shape) for value in hidden_states],
                "selected_layers": selected_layers,
                "input_length": len(input_ids),
                "numeric_positions": len(numeric_positions),
                "first_quantized_values": quantized_values[:5].tolist(),
                "first_numeric_tokens": [numeric_tokens[index] for index in nearest_numeric_indices(raw_values[:5], numeric_values)],
            })

        for position_index, token_position in enumerate(numeric_positions):
            if token_position == 0:
                continue
            metadata_rows.append({
                "series_id": int(record["series_id"]),
                "kind": str(record["kind"]),
                "position": int(position_index),
                "raw_value": float(raw_values[position_index]),
                "quantization_error": float(quantized_values[position_index] - raw_values[position_index]),
                **make_targets(quantized_values, position_index, args.window),
            })
            for layer in selected_layers:
                feature_rows[(layer, "token_state")].append(
                    hidden_states[layer][0, token_position].detach().float().cpu().numpy()
                )
                feature_rows[(layer, "pre_token_state")].append(
                    hidden_states[layer][0, token_position - 1].detach().float().cpu().numpy()
                )

        if (row_number + 1) % 100 == 0:
            print(f"Processed {row_number + 1}/{len(frame)} series")

    if not metadata_rows:
        raise RuntimeError("No probe rows were extracted.")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = pd.DataFrame(metadata_rows)
    metadata.to_csv(str(output_path) + ".metadata.csv", index=False)

    arrays = {
        f"L{layer}_{state}": np.asarray(values, dtype=np.float32)
        for (layer, state), values in feature_rows.items()
    }
    expected_rows = len(metadata)
    for name, array in arrays.items():
        if array.shape[0] != expected_rows:
            raise RuntimeError(f"Feature/metadata mismatch for {name}: {array.shape[0]} != {expected_rows}")
    np.savez_compressed(output_path, **arrays)

    config = {
        "model_label": args.model_label,
        "model_path": args.model_path,
        "tokenizer_path": tokenizer_source,
        "adapter_path": args.adapter_path,
        "dtype": args.dtype,
        "selected_layers": selected_layers,
        "hidden_state_count": len(hidden_states),
        "numeric_token_count": len(numeric_values),
        "series_count": len(frame),
        "row_count": expected_rows,
        "target_source": "nearest tokenizer numeric token value",
    }
    with open(str(output_path) + ".config.json", "w", encoding="utf-8") as file:
        json.dump(config, file, ensure_ascii=False, indent=2)

    print({
        "saved_features": str(output_path),
        "saved_metadata": str(output_path) + ".metadata.csv",
        "rows": expected_rows,
        "feature_arrays": {name: list(array.shape) for name, array in arrays.items()},
        "max_abs_quantization_error": float(metadata["quantization_error"].abs().max()),
    })


if __name__ == "__main__":
    main()
