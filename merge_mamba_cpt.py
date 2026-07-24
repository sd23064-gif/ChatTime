#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import shutil
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

CHECK_TOKENS = ["###-0.9999###", "###-0.5001###", "###0.0001###", "###0.5001###", "###0.9999###"]


def resolve_dtype(name):
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[name]


def validate_adapter_dir(adapter_path):
    path = Path(adapter_path)
    if not path.is_dir():
        raise FileNotFoundError(f"Adapter directory not found: {path}")
    if not (path / "adapter_config.json").is_file():
        raise FileNotFoundError(f"adapter_config.json not found: {path}")
    weights = [path / "adapter_model.safetensors", path / "adapter_model.bin"]
    if not any(item.is_file() for item in weights):
        raise FileNotFoundError(f"Adapter weights not found under: {path}")


def prepare_output_dir(output_path, overwrite):
    path = Path(output_path)
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory is not empty: {path}. Use --overwrite to replace it.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def verify_numeric_tokens(tokenizer, strict):
    results = []
    for token in CHECK_TOKENS:
        token_id = tokenizer.convert_tokens_to_ids(token)
        encoded = tokenizer.encode(token, add_special_tokens=False)
        decoded = tokenizer.convert_ids_to_tokens(token_id) if token_id is not None else None
        item = {"token": token, "token_id": token_id, "encoded": encoded, "decoded": decoded}
        results.append(item)
        print(item)
        valid = token_id is not None and token_id != tokenizer.unk_token_id and encoded == [token_id] and decoded == token
        if strict and not valid:
            raise ValueError(f"Numeric token is not preserved as one token: {item}")
    return results


def verify_vocab(model, tokenizer, stage):
    input_rows = model.get_input_embeddings().weight.shape[0]
    output_layer = model.get_output_embeddings()
    output_rows = output_layer.weight.shape[0] if output_layer is not None and hasattr(output_layer, "weight") else None
    if input_rows != len(tokenizer):
        raise ValueError(f"Input vocabulary mismatch at {stage}: model={input_rows}, tokenizer={len(tokenizer)}")
    if output_rows is not None and output_rows != len(tokenizer):
        raise ValueError(f"Output vocabulary mismatch at {stage}: model={output_rows}, tokenizer={len(tokenizer)}")
    print({"stage": stage, "tokenizer": len(tokenizer), "input_embedding": input_rows, "lm_head": output_rows})


def active_adapter_name(layer):
    active = getattr(layer, "active_adapter", "default")
    if isinstance(active, (list, tuple)):
        active = active[0]
    return active


def effective_weight(model, source):
    layer = model.get_input_embeddings() if source == "input" else model.get_output_embeddings()
    if layer is None:
        raise ValueError(f"Layer not found: {source}")
    modules_to_save = getattr(layer, "modules_to_save", None)
    if modules_to_save is not None:
        active = active_adapter_name(layer)
        if active in modules_to_save:
            return modules_to_save[active].weight
        if "default" in modules_to_save:
            return modules_to_save["default"].weight
    if not hasattr(layer, "weight"):
        raise ValueError(f"Weight not found: {source}")
    return layer.weight


def capture_selected_weights(model, tokenizer):
    input_weight = effective_weight(model, "input")
    output_weight = effective_weight(model, "output")
    captured = {}
    for token in CHECK_TOKENS:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is None or token_id < 0 or token_id >= input_weight.shape[0]:
            continue
        captured[token] = {
            "token_id": int(token_id),
            "input": input_weight[token_id].detach().float().cpu().clone(),
            "output": output_weight[token_id].detach().float().cpu().clone(),
        }
    return captured


def compare_selected_weights(before, after, tolerance):
    report = {}
    for token in before:
        if token not in after:
            raise ValueError(f"Token missing after merge: {token}")
        input_diff = float((before[token]["input"] - after[token]["input"]).abs().max())
        output_diff = float((before[token]["output"] - after[token]["output"]).abs().max())
        report[token] = {
            "token_id": before[token]["token_id"],
            "input_max_abs_diff": input_diff,
            "output_max_abs_diff": output_diff,
        }
        if input_diff > tolerance or output_diff > tolerance:
            raise ValueError(
                f"Embedding/LM-head mismatch after merge for {token}: input={input_diff}, output={output_diff}, tolerance={tolerance}"
            )
    return report


def verify_no_lora(model):
    names = [name for name, _ in model.named_modules() if "lora" in name.lower()]
    if names:
        raise ValueError("LoRA modules remain after merge:\n" + "\n".join(names[:50]))


def verify_saved_model(path):
    if not (path / "config.json").is_file():
        raise FileNotFoundError(f"config.json was not saved: {path}")
    model_files = [
        path / "model.safetensors", path / "model.safetensors.index.json",
        path / "pytorch_model.bin", path / "pytorch_model.bin.index.json",
    ]
    if not any(item.is_file() for item in model_files):
        raise FileNotFoundError(f"Merged model weights were not saved: {path}")
    if not (path / "tokenizer_config.json").is_file():
        raise FileNotFoundError(f"Tokenizer was not saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Merge a ChatTime Mamba CPT PEFT adapter into its base model.")
    parser.add_argument("--base_model_path", required=True)
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_shard_size", default="5GB")
    parser.add_argument("--weight_tolerance", type=float, default=1e-5)
    parser.add_argument("--strict_numeric_tokens", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    validate_adapter_dir(args.adapter_path)
    output_path = prepare_output_dir(args.output_path, args.overwrite)
    dtype = resolve_dtype(args.dtype)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is unavailable.")

    print("Loading tokenizer from adapter:", args.adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(args.adapter_path, trust_remote_code=True, use_fast=True)
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer has no EOS token.")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    token_report = verify_numeric_tokens(tokenizer, args.strict_numeric_tokens)

    print("Loading base model:", args.base_model_path)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path, torch_dtype=dtype, low_cpu_mem_usage=True,
        trust_remote_code=True, device_map=None,
    )
    original_vocab = base_model.get_input_embeddings().weight.shape[0]
    if original_vocab != len(tokenizer):
        print(f"Resizing vocabulary: {original_vocab} -> {len(tokenizer)}")
        base_model.resize_token_embeddings(len(tokenizer))
    verify_vocab(base_model, tokenizer, "base_after_resize")
    base_model.to(args.device)

    print("Loading CPT adapter:", args.adapter_path)
    peft_model = PeftModel.from_pretrained(base_model, args.adapter_path, is_trainable=False)
    peft_model.to(device=args.device, dtype=dtype)
    peft_model.eval()
    verify_vocab(peft_model, tokenizer, "adapter_loaded")

    for name, config in peft_model.peft_config.items():
        print({
            "adapter": name,
            "peft_type": str(config.peft_type),
            "target_modules": sorted(config.target_modules) if config.target_modules else None,
            "modules_to_save": config.modules_to_save,
            "rank": getattr(config, "r", None),
            "lora_alpha": getattr(config, "lora_alpha", None),
        })

    weights_before = capture_selected_weights(peft_model, tokenizer)
    print("Merging adapter with safe_merge=True")
    merged_model = peft_model.merge_and_unload(progressbar=True, safe_merge=True)
    merged_model.to(device=args.device, dtype=dtype)
    merged_model.eval()
    verify_vocab(merged_model, tokenizer, "after_merge")
    verify_no_lora(merged_model)

    weights_after = capture_selected_weights(merged_model, tokenizer)
    weight_report = compare_selected_weights(weights_before, weights_after, args.weight_tolerance)
    print("Weight validation:", json.dumps(weight_report, indent=2))

    merged_model.config.use_cache = True
    merged_model.config.pad_token_id = tokenizer.pad_token_id
    merged_model.config.eos_token_id = tokenizer.eos_token_id
    if hasattr(merged_model, "generation_config"):
        merged_model.generation_config.use_cache = True
        merged_model.generation_config.pad_token_id = tokenizer.pad_token_id
        merged_model.generation_config.eos_token_id = tokenizer.eos_token_id

    print("Saving merged model:", output_path)
    merged_model.save_pretrained(output_path, safe_serialization=True, max_shard_size=args.max_shard_size)
    tokenizer.save_pretrained(output_path)
    with (output_path / "merge_information.json").open("w", encoding="utf-8") as file:
        json.dump({
            "base_model_path": args.base_model_path,
            "adapter_path": args.adapter_path,
            "dtype": args.dtype,
            "original_vocab_size": original_vocab,
            "merged_vocab_size": len(tokenizer),
            "numeric_token_report": token_report,
            "weight_report": weight_report,
        }, file, ensure_ascii=False, indent=2)
    verify_saved_model(output_path)

    print("Reloading saved model for final validation")
    del merged_model, peft_model, base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    check_tokenizer = AutoTokenizer.from_pretrained(output_path, trust_remote_code=True, use_fast=True)
    check_model = AutoModelForCausalLM.from_pretrained(
        output_path, torch_dtype=dtype, low_cpu_mem_usage=True,
        trust_remote_code=True, device_map=None,
    )
    verify_vocab(check_model, check_tokenizer, "reloaded")
    verify_numeric_tokens(check_tokenizer, args.strict_numeric_tokens)
    verify_no_lora(check_model)
    print("Merge completed successfully:", output_path)


if __name__ == "__main__":
    main()
