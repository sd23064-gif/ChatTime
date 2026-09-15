#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import gc
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def dtype_name(dtype):
    return str(dtype).replace("torch.", "")


def json_safe(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.dtype):
        return dtype_name(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [json_safe(item) for item in value]
    if callable(value):
        return f"<callable {getattr(value, '__qualname__', type(value).__name__)}>"
    return str(value)


def tensor_inventory(model):
    parameter_counts = Counter()
    parameter_elements = Counter()
    parameter_examples = defaultdict(list)
    buffer_counts = Counter()
    buffer_elements = Counter()
    buffer_examples = defaultdict(list)

    for name, parameter in model.named_parameters():
        key = dtype_name(parameter.dtype)
        parameter_counts[key] += 1
        parameter_elements[key] += parameter.numel()
        if len(parameter_examples[key]) < 8:
            parameter_examples[key].append({
                "name": name,
                "shape": list(parameter.shape),
                "device": str(parameter.device),
                "requires_grad": bool(parameter.requires_grad),
            })

    for name, buffer in model.named_buffers():
        key = dtype_name(buffer.dtype)
        buffer_counts[key] += 1
        buffer_elements[key] += buffer.numel()
        if len(buffer_examples[key]) < 8:
            buffer_examples[key].append({
                "name": name,
                "shape": list(buffer.shape),
                "device": str(buffer.device),
            })

    return {
        "parameter_tensor_counts": dict(parameter_counts),
        "parameter_element_counts": dict(parameter_elements),
        "parameter_examples": dict(parameter_examples),
        "buffer_tensor_counts": dict(buffer_counts),
        "buffer_element_counts": dict(buffer_elements),
        "buffer_examples": dict(buffer_examples),
    }


def module_dtype_report(model):
    input_layer = model.get_input_embeddings()
    output_layer = model.get_output_embeddings()
    report = {
        "input_embedding": {
            "class": type(input_layer).__name__,
            "dtype": dtype_name(input_layer.weight.dtype),
            "shape": list(input_layer.weight.shape),
            "device": str(input_layer.weight.device),
        },
        "output_embedding": None,
    }
    if output_layer is not None:
        report["output_embedding"] = {
            "class": type(output_layer).__name__,
            "dtype": dtype_name(output_layer.weight.dtype),
            "shape": list(output_layer.weight.shape),
            "device": str(output_layer.weight.device),
            "bias_dtype": (
                dtype_name(output_layer.bias.dtype)
                if getattr(output_layer, "bias", None) is not None
                else None
            ),
        }
    return report


def modules_to_save_report(model):
    """Report both PEFT's top-level name set and ModulesToSaveWrapper contents.

    PEFT uses the same attribute name for two different objects:
    - PeftModel.modules_to_save: a set of module names
    - ModulesToSaveWrapper.modules_to_save: a ModuleDict of saved modules
    """
    rows = []
    for module_name, module in model.named_modules():
        saved = getattr(module, "modules_to_save", None)
        if saved is None:
            continue

        # Do not call active_adapter when it is a bound method. Some PEFT
        # objects raise "No adapter loaded" from that method during inspection,
        # even though child ModulesToSaveWrapper objects contain the adapter.
        active = getattr(module, "active_adapter", None)
        if callable(active):
            active = None

        if active is None:
            active_adapters = getattr(module, "active_adapters", None)
            if not callable(active_adapters):
                active = active_adapters

        if isinstance(active, (set, frozenset, tuple)):
            active = list(active)
        if active is not None and not isinstance(active, (str, int, float, bool, list, dict)):
            active = str(active)

        row = {
            "module_name": module_name,
            "module_class": type(module).__name__,
            "active_adapter": active,
            "attribute_type": type(saved).__name__,
            "saved_module_names": [],
            "saved_modules": {},
        }

        # PeftModel exposes a set such as {'lm_head', 'backbone.embeddings'}.
        if isinstance(saved, (set, frozenset, list, tuple)):
            row["saved_module_names"] = sorted(str(item) for item in saved)
            rows.append(row)
            continue

        # ModulesToSaveWrapper normally exposes torch.nn.ModuleDict.
        if hasattr(saved, "items"):
            for key, saved_module in saved.items():
                weight = getattr(saved_module, "weight", None)
                bias = getattr(saved_module, "bias", None)
                row["saved_modules"][str(key)] = {
                    "class": type(saved_module).__name__,
                    "weight_dtype": dtype_name(weight.dtype) if weight is not None else None,
                    "weight_shape": list(weight.shape) if weight is not None else None,
                    "weight_device": str(weight.device) if weight is not None else None,
                    "bias_dtype": dtype_name(bias.dtype) if bias is not None else None,
                }
            rows.append(row)
            continue

        row["inspection_warning"] = (
            "Unsupported modules_to_save container: " + type(saved).__name__
        )
        rows.append(row)

    return rows


def runtime_probe(model, tokenizer, text):
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=True)
    device = next(model.parameters()).device
    encoded = {key: value.to(device) for key, value in encoded.items()}

    captured = {}
    hooks = []

    def register(name, module):
        if module is None:
            return

        def hook(_, inputs, output):
            input_dtypes = []
            for item in inputs:
                if torch.is_tensor(item):
                    input_dtypes.append(dtype_name(item.dtype))
            if torch.is_tensor(output):
                output_dtype = dtype_name(output.dtype)
                output_shape = list(output.shape)
            elif isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
                output_dtype = dtype_name(output[0].dtype)
                output_shape = list(output[0].shape)
            else:
                output_dtype = None
                output_shape = None
            captured[name] = {
                "input_dtypes": input_dtypes,
                "output_dtype": output_dtype,
                "output_shape": output_shape,
            }

        hooks.append(module.register_forward_hook(hook))

    register("input_embedding", model.get_input_embeddings())
    register("output_embedding", model.get_output_embeddings())

    try:
        with torch.inference_mode():
            outputs = model(**encoded, use_cache=False, return_dict=True)
        logits = outputs.logits
        result = {
            "success": True,
            "input_ids_dtype": dtype_name(encoded["input_ids"].dtype),
            "logits_dtype": dtype_name(logits.dtype),
            "logits_shape": list(logits.shape),
            "logits_finite": bool(torch.isfinite(logits).all().item()),
            "hooked_modules": captured,
        }
    except Exception as error:
        result = {
            "success": False,
            "error_type": type(error).__name__,
            "error": str(error),
            "hooked_modules": captured,
        }
    finally:
        for hook in hooks:
            hook.remove()

    return result


def gpu_report():
    if not torch.cuda.is_available():
        return {"cuda_available": False}
    major, minor = torch.cuda.get_device_capability(0)
    return {
        "cuda_available": True,
        "gpu_name": torch.cuda.get_device_name(0),
        "compute_capability": [major, minor],
        "bf16_supported": bool(torch.cuda.is_bf16_supported()),
        "cuda_runtime": torch.version.cuda,
    }


def resolve_load_dtype(name):
    return {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[name]


def load_model(args, tokenizer):
    dtype = resolve_load_dtype(args.load_dtype)
    common = {
        "torch_dtype": dtype,
        "device_map": args.device_map,
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
        "local_files_only": args.local_files_only,
        "return_dict": True,
    }

    if args.mode == "merged":
        model = AutoModelForCausalLM.from_pretrained(args.model_path, **common)
    else:
        base = AutoModelForCausalLM.from_pretrained(args.base_model_path, **common)
        current_vocab = base.get_input_embeddings().weight.shape[0]
        if current_vocab != len(tokenizer):
            print(f"Resizing vocabulary before adapter load: {current_vocab} -> {len(tokenizer)}")
            base.resize_token_embeddings(len(tokenizer))
        model = PeftModel.from_pretrained(
            base,
            args.adapter_path,
            is_trainable=False,
            local_files_only=args.local_files_only,
        )

    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Inspect Mamba checkpoint dtypes and runtime activations.")
    parser.add_argument("--mode", choices=["merged", "unmerged"], required=True)
    parser.add_argument("--model_path", default=None, help="Required for --mode merged.")
    parser.add_argument("--base_model_path", default=None, help="Required for --mode unmerged.")
    parser.add_argument("--adapter_path", default=None, help="Required for --mode unmerged.")
    parser.add_argument("--tokenizer_path", required=True)
    parser.add_argument("--load_dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--probe_text", default="###0.0001### ")
    parser.add_argument("--output", default="dtype_report.json")
    parser.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if args.mode == "merged" and not args.model_path:
        parser.error("--model_path is required for --mode merged.")
    if args.mode == "unmerged" and (not args.base_model_path or not args.adapter_path):
        parser.error("--base_model_path and --adapter_path are required for --mode unmerged.")

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path,
        use_fast=True,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )

    config_path = args.model_path if args.mode == "merged" else args.base_model_path
    config = AutoConfig.from_pretrained(
        config_path,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )

    report = {
        "mode": args.mode,
        "requested_load_dtype": args.load_dtype,
        "paths": {
            "model_path": args.model_path,
            "base_model_path": args.base_model_path,
            "adapter_path": args.adapter_path,
            "tokenizer_path": args.tokenizer_path,
        },
        "environment": gpu_report(),
        "config_before_load": {
            "model_type": getattr(config, "model_type", None),
            "torch_dtype": str(getattr(config, "torch_dtype", None)),
            "tie_word_embeddings": getattr(config, "tie_word_embeddings", None),
            "vocab_size": getattr(config, "vocab_size", None),
        },
        "tokenizer_size": len(tokenizer),
    }

    print("Loading model...")
    model = load_model(args, tokenizer)

    report["loaded_model"] = {
        "class": type(model).__name__,
        "config_model_type": getattr(model.config, "model_type", None),
        "config_torch_dtype": str(getattr(model.config, "torch_dtype", None)),
        "config_tie_word_embeddings": getattr(model.config, "tie_word_embeddings", None),
        "first_parameter_dtype": dtype_name(next(model.parameters()).dtype),
        "first_parameter_device": str(next(model.parameters()).device),
        "module_dtypes": module_dtype_report(model),
        "inventory": tensor_inventory(model),
        "modules_to_save": modules_to_save_report(model),
        "runtime_probe": runtime_probe(model, tokenizer, args.probe_text),
    }

    expected = dtype_name(resolve_load_dtype(args.load_dtype))
    actual_parameter_dtypes = set(
        report["loaded_model"]["inventory"]["parameter_tensor_counts"].keys()
    )
    floating_parameter_dtypes = {
        name for name in actual_parameter_dtypes
        if name in {"float16", "bfloat16", "float32", "float64"}
    }
    report["verdict"] = {
        "expected_parameter_dtype": expected,
        "floating_parameter_dtypes": sorted(floating_parameter_dtypes),
        "all_floating_parameters_match_requested": floating_parameter_dtypes == {expected},
        "runtime_success": report["loaded_model"]["runtime_probe"]["success"],
        "runtime_logits_dtype": report["loaded_model"]["runtime_probe"].get("logits_dtype"),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(json_safe(report), file, ensure_ascii=False, indent=2)

    print(json.dumps(json_safe(report), ensure_ascii=False, indent=2))
    print("\nSaved:", output_path)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
