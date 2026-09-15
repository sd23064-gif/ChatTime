#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import gc
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

CHECK_TOKENS = [
    "###-0.9999###",
    "###-0.5001###",
    "###0.0001###",
    "###0.5001###",
    "###0.9999###",
]


def resolve_dtype(name):
    return {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[name]


def dtype_config_name(dtype):
    return {
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
        torch.float32: "float32",
    }[dtype]


def validate_adapter_dir(adapter_path):
    path = Path(adapter_path)
    if not path.is_dir():
        raise FileNotFoundError(f"Adapter directory not found: {path}")
    if not (path / "adapter_config.json").is_file():
        raise FileNotFoundError(f"adapter_config.json not found: {path}")
    candidates = [path / "adapter_model.safetensors", path / "adapter_model.bin"]
    if not any(candidate.is_file() for candidate in candidates):
        raise FileNotFoundError(f"Adapter weights not found under: {path}")


def prepare_output_dir(output_path, overwrite):
    path = Path(output_path)
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {path}. Use --overwrite to replace it."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def verify_numeric_tokens(tokenizer, strict=True):
    report = []
    for token in CHECK_TOKENS:
        token_id = tokenizer.convert_tokens_to_ids(token)
        encoded = tokenizer.encode(token, add_special_tokens=False)
        decoded = tokenizer.convert_ids_to_tokens(token_id) if token_id is not None else None
        valid = (
            token_id is not None
            and token_id != tokenizer.unk_token_id
            and encoded == [token_id]
            and decoded == token
        )
        item = {
            "token": token,
            "token_id": token_id,
            "encoded": encoded,
            "decoded": decoded,
            "valid": valid,
        }
        report.append(item)
        print(item)
        if strict and not valid:
            raise ValueError(f"Numeric token is not preserved as one token: {item}")
    return report


def verify_vocab(model, tokenizer, stage):
    input_rows = model.get_input_embeddings().weight.shape[0]
    output_layer = model.get_output_embeddings()
    output_rows = output_layer.weight.shape[0] if output_layer is not None else None
    expected = len(tokenizer)
    if input_rows != expected or output_rows != expected:
        raise ValueError(
            f"Vocabulary mismatch at {stage}: tokenizer={expected}, "
            f"input={input_rows}, output={output_rows}"
        )
    print({
        "stage": stage,
        "tokenizer_vocab": expected,
        "input_embedding_rows": input_rows,
        "lm_head_rows": output_rows,
    })


def active_adapter_name(wrapper):
    active = getattr(wrapper, "active_adapter", None)
    if callable(active):
        active = None
    if active is None:
        active = getattr(wrapper, "active_adapters", None)
        if callable(active):
            active = None
    if isinstance(active, (list, tuple)):
        active = active[0] if active else None
    return active or "default"


def effective_module(layer):
    modules_to_save = getattr(layer, "modules_to_save", None)
    if modules_to_save is None or not hasattr(modules_to_save, "items"):
        return layer

    active = active_adapter_name(layer)
    if active in modules_to_save:
        return modules_to_save[active]
    if "default" in modules_to_save:
        return modules_to_save["default"]
    raise ValueError(
        f"Could not resolve ModulesToSaveWrapper. active={active}, "
        f"available={list(modules_to_save.keys())}"
    )


def capture_full_io_state(peft_model):
    input_module = effective_module(peft_model.get_input_embeddings())
    output_module = effective_module(peft_model.get_output_embeddings())
    if not hasattr(input_module, "weight") or not hasattr(output_module, "weight"):
        raise ValueError("Effective input/output module has no weight.")

    input_weight = input_module.weight.detach().cpu().clone()
    output_weight = output_module.weight.detach().cpu().clone()
    output_bias = None
    if getattr(output_module, "bias", None) is not None:
        output_bias = output_module.bias.detach().cpu().clone()

    print({
        "captured_input_shape": list(input_weight.shape),
        "captured_input_dtype": str(input_weight.dtype),
        "captured_output_shape": list(output_weight.shape),
        "captured_output_dtype": str(output_weight.dtype),
        "input_output_same_pointer_before_merge": (
            input_module.weight.data_ptr() == output_module.weight.data_ptr()
        ),
        "input_output_max_diff_before_merge": float(
            (input_weight.float() - output_weight.float()).abs().max()
        ),
        "input_output_mean_diff_before_merge": float(
            (input_weight.float() - output_weight.float()).abs().mean()
        ),
    })
    return input_weight, output_weight, output_bias


def build_probe_ids(tokenizer, device):
    valid_ids = []
    for token in CHECK_TOKENS:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is not None and token_id != tokenizer.unk_token_id:
            valid_ids.append(int(token_id))
    if not valid_ids:
        valid_ids = [tokenizer.eos_token_id]
    return torch.tensor([valid_ids], dtype=torch.long, device=device)


def capture_probe_logits(model, tokenizer):
    device = next(model.parameters()).device
    input_ids = build_probe_ids(tokenizer, device)
    model.eval()
    with torch.inference_mode():
        output = model(input_ids=input_ids, use_cache=False, return_dict=True)
    logits = output.logits[:, -1].detach().float().cpu()
    if not torch.isfinite(logits).all():
        raise ValueError("Probe logits contain NaN or Inf.")
    return logits


def replace_untied_io_layers(model, input_weight_cpu, output_weight_cpu, output_bias_cpu):
    model.config.tie_word_embeddings = False

    input_layer = model.get_input_embeddings()
    output_layer = model.get_output_embeddings()
    if input_layer is None or output_layer is None:
        raise ValueError("Merged model has no input embedding or LM head.")
    if tuple(input_layer.weight.shape) != tuple(input_weight_cpu.shape):
        raise ValueError(
            f"Input shape mismatch: model={tuple(input_layer.weight.shape)}, "
            f"saved={tuple(input_weight_cpu.shape)}"
        )
    if tuple(output_layer.weight.shape) != tuple(output_weight_cpu.shape):
        raise ValueError(
            f"Output shape mismatch: model={tuple(output_layer.weight.shape)}, "
            f"saved={tuple(output_weight_cpu.shape)}"
        )

    with torch.no_grad():
        input_layer.weight.copy_(
            input_weight_cpu.to(device=input_layer.weight.device, dtype=input_layer.weight.dtype)
        )

    new_output = torch.nn.Linear(
        in_features=output_weight_cpu.shape[1],
        out_features=output_weight_cpu.shape[0],
        bias=output_bias_cpu is not None,
        device=output_layer.weight.device,
        dtype=output_layer.weight.dtype,
    )
    with torch.no_grad():
        new_output.weight.copy_(
            output_weight_cpu.to(device=new_output.weight.device, dtype=new_output.weight.dtype)
        )
        if output_bias_cpu is not None:
            new_output.bias.copy_(
                output_bias_cpu.to(device=new_output.bias.device, dtype=new_output.bias.dtype)
            )

    if hasattr(model, "set_output_embeddings"):
        model.set_output_embeddings(new_output)
    else:
        model.lm_head = new_output

    # Some implementations call tie_weights internally. Keep the config false and
    # verify that the replacement remains independent.
    model.config.tie_word_embeddings = False
    return model


def verify_untied(model, stage, expected_difference=True):
    input_weight = model.get_input_embeddings().weight
    output_weight = model.get_output_embeddings().weight
    same_object = input_weight is output_weight
    same_pointer = input_weight.data_ptr() == output_weight.data_ptr()
    difference = (input_weight.detach().float() - output_weight.detach().float()).abs()
    report = {
        "stage": stage,
        "tie_word_embeddings": bool(getattr(model.config, "tie_word_embeddings", True)),
        "same_python_object": same_object,
        "same_data_pointer": same_pointer,
        "input_output_max_abs_diff": float(difference.max()),
        "input_output_mean_abs_diff": float(difference.mean()),
    }
    print(report)
    if report["tie_word_embeddings"]:
        raise ValueError(f"tie_word_embeddings is still True at {stage}.")
    if same_object or same_pointer:
        raise ValueError(f"Input embedding and LM Head are still tied at {stage}.")
    if expected_difference and report["input_output_max_abs_diff"] == 0.0:
        raise ValueError(f"Input embedding and LM Head unexpectedly contain identical weights at {stage}.")
    return report


def sampled_row_indices(vocab_size, tokenizer):
    indices = {0, vocab_size - 1}
    for token in CHECK_TOKENS:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is not None and 0 <= token_id < vocab_size:
            indices.add(int(token_id))
    for fraction in np.linspace(0.0, 1.0, 17):
        indices.add(min(vocab_size - 1, int(round(fraction * (vocab_size - 1)))))
    return sorted(indices)


def compare_saved_rows(model, input_reference, output_reference, tokenizer, tolerance, stage):
    input_weight = model.get_input_embeddings().weight
    output_weight = model.get_output_embeddings().weight
    rows = sampled_row_indices(input_reference.shape[0], tokenizer)
    input_actual = input_weight[rows].detach().float().cpu()
    output_actual = output_weight[rows].detach().float().cpu()
    input_expected = input_reference[rows].float()
    output_expected = output_reference[rows].float()
    input_diff = (input_actual - input_expected).abs()
    output_diff = (output_actual - output_expected).abs()
    report = {
        "stage": stage,
        "sampled_rows": rows,
        "input_max_abs_diff": float(input_diff.max()),
        "input_mean_abs_diff": float(input_diff.mean()),
        "output_max_abs_diff": float(output_diff.max()),
        "output_mean_abs_diff": float(output_diff.mean()),
    }
    print(report)
    if report["input_max_abs_diff"] > tolerance or report["output_max_abs_diff"] > tolerance:
        raise ValueError(
            f"Saved input/output weights differ at {stage}: {report}, tolerance={tolerance}"
        )
    return report


def compare_logits(reference, actual, tolerance, stage):
    if reference.shape != actual.shape:
        raise ValueError(
            f"Probe logits shape mismatch at {stage}: {reference.shape} != {actual.shape}"
        )
    difference = (reference - actual).abs()
    report = {
        "stage": stage,
        "max_abs_diff": float(difference.max()),
        "mean_abs_diff": float(difference.mean()),
        "argmax_equal": bool(reference.argmax(-1).item() == actual.argmax(-1).item()),
    }
    print(report)
    if report["max_abs_diff"] > tolerance:
        raise ValueError(
            f"Probe logits exceed tolerance at {stage}: {report}, tolerance={tolerance}"
        )
    return report


def verify_no_peft_modules(model):
    bad = []
    for name, module in model.named_modules():
        lowered = name.lower()
        class_name = type(module).__name__.lower()
        if "lora" in lowered or "lora" in class_name or "modulestosavewrapper" in class_name:
            bad.append(f"{name}: {type(module).__name__}")
    if bad:
        raise ValueError("PEFT modules remain after merge:\n" + "\n".join(bad[:50]))


def patch_saved_config(output_path, dtype):
    config_path = output_path / "config.json"
    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)
    config["tie_word_embeddings"] = False
    config["torch_dtype"] = dtype_config_name(dtype)
    with config_path.open("w", encoding="utf-8") as file:
        json.dump(config, file, ensure_ascii=False, indent=2)


def validate_saved_files(output_path):
    required = [output_path / "config.json", output_path / "tokenizer_config.json"]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"Required output file was not saved: {path}")
    model_files = [
        output_path / "model.safetensors",
        output_path / "model.safetensors.index.json",
        output_path / "pytorch_model.bin",
        output_path / "pytorch_model.bin.index.json",
    ]
    if not any(path.is_file() for path in model_files):
        raise FileNotFoundError(f"No model weights found under {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Safely merge a Mamba CPT adapter while preserving independent embedding and LM Head weights."
    )
    parser.add_argument("--base_model_path", required=True)
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--tokenizer_path", default=None)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_shard_size", default="5GB")
    parser.add_argument("--weight_tolerance", type=float, default=0.0)
    parser.add_argument("--logit_tolerance", type=float, default=1.0)
    parser.add_argument("--strict_numeric_tokens", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    validate_adapter_dir(args.adapter_path)
    output_path = prepare_output_dir(args.output_path, args.overwrite)
    dtype = resolve_dtype(args.dtype)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is unavailable.")
    if dtype == torch.bfloat16 and args.device.startswith("cuda") and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 was requested, but the selected GPU does not support BF16.")

    tokenizer_path = args.tokenizer_path or args.adapter_path
    print("Loading tokenizer:", tokenizer_path)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
        use_fast=True,
        local_files_only=args.local_files_only,
    )
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer has no EOS token.")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    token_report = verify_numeric_tokens(tokenizer, args.strict_numeric_tokens)

    print("Loading base model:", args.base_model_path)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
        device_map=None,
        return_dict=True,
    )
    model_type = getattr(base_model.config, "model_type", None)
    if model_type not in {"mamba", "mamba2"}:
        raise ValueError(f"Expected a Mamba model, loaded model_type={model_type}.")

    original_vocab = base_model.get_input_embeddings().weight.shape[0]
    if original_vocab != len(tokenizer):
        print(f"Resizing vocabulary: {original_vocab} -> {len(tokenizer)}")
        base_model.resize_token_embeddings(len(tokenizer))
    base_model.to(args.device)
    verify_vocab(base_model, tokenizer, "base_after_resize")

    print("Loading CPT adapter:", args.adapter_path)
    peft_model = PeftModel.from_pretrained(
        base_model,
        args.adapter_path,
        is_trainable=False,
        local_files_only=args.local_files_only,
    )
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

    print("Capturing independent modules_to_save weights")
    input_reference, output_reference, output_bias_reference = capture_full_io_state(peft_model)
    reference_logits = capture_probe_logits(peft_model, tokenizer)

    print("Merging LoRA weights with safe_merge=True")
    merged_model = peft_model.merge_and_unload(progressbar=True, safe_merge=True)
    merged_model.to(device=args.device, dtype=dtype)
    merged_model.eval()
    verify_vocab(merged_model, tokenizer, "immediately_after_merge")
    verify_no_peft_modules(merged_model)

    print("Restoring independent input embedding and LM Head")
    merged_model = replace_untied_io_layers(
        merged_model,
        input_reference,
        output_reference,
        output_bias_reference,
    )
    merged_model.eval()
    verify_vocab(merged_model, tokenizer, "after_io_restore")

    untied_before_save = verify_untied(merged_model, "before_save")
    rows_before_save = compare_saved_rows(
        merged_model,
        input_reference,
        output_reference,
        tokenizer,
        args.weight_tolerance,
        "before_save",
    )
    merged_logits = capture_probe_logits(merged_model, tokenizer)
    logits_before_save = compare_logits(
        reference_logits,
        merged_logits,
        args.logit_tolerance,
        "before_save",
    )

    merged_model.config.use_cache = True
    merged_model.config.pad_token_id = tokenizer.pad_token_id
    merged_model.config.eos_token_id = tokenizer.eos_token_id
    merged_model.config.tie_word_embeddings = False
    merged_model.config.torch_dtype = dtype
    if hasattr(merged_model, "generation_config"):
        merged_model.generation_config.use_cache = True
        merged_model.generation_config.pad_token_id = tokenizer.pad_token_id
        merged_model.generation_config.eos_token_id = tokenizer.eos_token_id

    for parameter in merged_model.parameters():
        parameter.requires_grad = False

    print("Saving fixed merged model:", output_path)
    merged_model.save_pretrained(
        output_path,
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )
    tokenizer.save_pretrained(output_path)
    patch_saved_config(output_path, dtype)
    validate_saved_files(output_path)

    del merged_model, peft_model, base_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Reloading fixed merged model for final validation")
    check_tokenizer = AutoTokenizer.from_pretrained(
        output_path,
        trust_remote_code=True,
        use_fast=True,
        local_files_only=True,
    )
    check_model = AutoModelForCausalLM.from_pretrained(
        output_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        local_files_only=True,
        device_map=None,
        return_dict=True,
    ).to(args.device)
    check_model.eval()

    verify_vocab(check_model, check_tokenizer, "reloaded")
    verify_numeric_tokens(check_tokenizer, args.strict_numeric_tokens)
    verify_no_peft_modules(check_model)
    untied_after_reload = verify_untied(check_model, "after_reload")
    rows_after_reload = compare_saved_rows(
        check_model,
        input_reference,
        output_reference,
        check_tokenizer,
        args.weight_tolerance,
        "after_reload",
    )
    reloaded_logits = capture_probe_logits(check_model, check_tokenizer)
    logits_after_reload = compare_logits(
        reference_logits,
        reloaded_logits,
        args.logit_tolerance,
        "after_reload",
    )

    merge_information = {
        "base_model_path": args.base_model_path,
        "adapter_path": args.adapter_path,
        "tokenizer_path": tokenizer_path,
        "output_path": str(output_path),
        "dtype": args.dtype,
        "original_vocab_size": original_vocab,
        "merged_vocab_size": len(tokenizer),
        "numeric_token_report": token_report,
        "untied_before_save": untied_before_save,
        "untied_after_reload": untied_after_reload,
        "sampled_weight_validation_before_save": rows_before_save,
        "sampled_weight_validation_after_reload": rows_after_reload,
        "probe_logits_before_save": logits_before_save,
        "probe_logits_after_reload": logits_after_reload,
    }
    with (output_path / "merge_information.json").open("w", encoding="utf-8") as file:
        json.dump(merge_information, file, ensure_ascii=False, indent=2)

    print("\nFixed merge completed successfully:", output_path)
    print(json.dumps(merge_information, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
