#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import ast
import gc
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.prompt import getPrompt
from utils.tools import Discretizer, Serializer

NUMERIC_PATTERN = re.compile(r"###(?:[+-]?(?:\d+(?:\.\d*)?|\.\d+)|Nan|NaN|nan)###")


def parse_array(value):
    if isinstance(value, np.ndarray):
        return value.astype(np.float64, copy=False).reshape(-1)
    if isinstance(value, list):
        return np.asarray(value, dtype=np.float64).reshape(-1)
    return np.asarray(ast.literal_eval(str(value)), dtype=np.float64).reshape(-1)


def resolve_dtype(name):
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise ValueError("BF16 is not supported on this GPU. Use --dtype fp16.")
        return torch.bfloat16
    return torch.float32


def model_device(model):
    return next(model.parameters()).device


def resolve_effective_weight(layer):
    modules_to_save = getattr(layer, "modules_to_save", None)
    if modules_to_save is None:
        return layer.weight

    active_adapter = getattr(layer, "active_adapter", "default")
    if isinstance(active_adapter, (list, tuple)):
        active_adapter = active_adapter[0]
    if active_adapter in modules_to_save:
        return modules_to_save[active_adapter].weight
    if "default" in modules_to_save:
        return modules_to_save["default"].weight
    raise ValueError(
        f"Could not resolve modules_to_save weight. active={active_adapter}, "
        f"available={list(modules_to_save.keys())}"
    )


def numeric_vocab(tokenizer):
    items = []
    for token, token_id in tokenizer.get_vocab().items():
        if NUMERIC_PATTERN.fullmatch(token):
            inner = token[3:-3]
            try:
                value = float(inner)
            except ValueError:
                continue
            if np.isfinite(value):
                items.append((int(token_id), token, value))
    items.sort(key=lambda item: item[0])
    if not items:
        raise ValueError("No numeric tokens were found in the tokenizer vocabulary.")
    return items


def snapshot_numeric_weights(model, numeric_items, max_rows=256):
    input_weight = resolve_effective_weight(model.get_input_embeddings())
    output_layer = model.get_output_embeddings()
    if output_layer is None:
        raise ValueError("Model has no output embedding / LM head.")
    output_weight = resolve_effective_weight(output_layer)

    step = max(1, len(numeric_items) // max_rows)
    selected = numeric_items[::step][:max_rows]
    token_ids = [item[0] for item in selected]
    input_ids = torch.tensor(token_ids, device=input_weight.device, dtype=torch.long)
    output_ids = input_ids.to(output_weight.device)
    return {
        "token_ids": token_ids,
        "tokens": [item[1] for item in selected],
        "input_rows": input_weight.index_select(0, input_ids).detach().float().cpu().numpy(),
        "output_rows": output_weight.index_select(0, output_ids).detach().float().cpu().numpy(),
    }


def load_merged(path, tokenizer, dtype, device_map, local_files_only):
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=dtype,
        device_map=device_map,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        local_files_only=local_files_only,
        return_dict=True,
    )
    model_type = getattr(model.config, "model_type", None)
    if model_type not in {"mamba", "mamba2"}:
        raise ValueError(f"Expected Mamba model, loaded model_type={model_type}.")
    if model.get_input_embeddings().weight.shape[0] != len(tokenizer):
        raise ValueError(
            "Merged model/tokenizer vocabulary mismatch: "
            f"model={model.get_input_embeddings().weight.shape[0]}, tokenizer={len(tokenizer)}"
        )
    model.eval()
    return model


def load_unmerged(base_path, adapter_path, tokenizer, dtype, device_map, local_files_only):
    base = AutoModelForCausalLM.from_pretrained(
        base_path,
        torch_dtype=dtype,
        device_map=device_map,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        local_files_only=local_files_only,
        return_dict=True,
    )
    model_type = getattr(base.config, "model_type", None)
    if model_type not in {"mamba", "mamba2"}:
        raise ValueError(f"Expected Mamba base model, loaded model_type={model_type}.")

    current_vocab = base.get_input_embeddings().weight.shape[0]
    if current_vocab != len(tokenizer):
        print(f"Resizing unmerged base vocabulary: {current_vocab} -> {len(tokenizer)}")
        base.resize_token_embeddings(len(tokenizer))

    model = PeftModel.from_pretrained(
        base,
        adapter_path,
        is_trainable=False,
        local_files_only=local_files_only,
    )
    model.eval()
    return model


def top_k_records(logits, tokenizer, k):
    probabilities = torch.softmax(logits.float(), dim=-1)
    values, indices = torch.topk(probabilities, k=min(k, probabilities.numel()))
    records = []
    for probability, token_id in zip(values.tolist(), indices.tolist()):
        token = tokenizer.convert_ids_to_tokens(int(token_id))
        records.append({
            "token_id": int(token_id),
            "token": token,
            "probability": float(probability),
            "is_numeric": bool(NUMERIC_PATTERN.fullmatch(token or "")),
        })
    return records


def greedy_trace(model, tokenizer, prompt, max_new_tokens, top_k):
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = encoded["input_ids"].to(model_device(model))
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(model_device(model))

    generated_ids = []
    trace = []
    with torch.inference_mode():
        for step in range(max_new_tokens):
            kwargs = {"input_ids": input_ids, "use_cache": False, "return_dict": True}
            if attention_mask is not None:
                kwargs["attention_mask"] = attention_mask
            outputs = model(**kwargs)
            logits = outputs.logits[0, -1].float()
            next_id = int(torch.argmax(logits).item())
            next_token = tokenizer.convert_ids_to_tokens(next_id)
            trace.append({
                "step": step,
                "argmax_id": next_id,
                "argmax_token": next_token,
                "argmax_is_numeric": bool(NUMERIC_PATTERN.fullmatch(next_token or "")),
                "top_k": top_k_records(logits, tokenizer, top_k),
            })
            generated_ids.append(next_id)
            next_tensor = torch.tensor([[next_id]], device=input_ids.device, dtype=input_ids.dtype)
            input_ids = torch.cat([input_ids, next_tensor], dim=-1)
            if attention_mask is not None:
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones((attention_mask.shape[0], 1), device=attention_mask.device,
                               dtype=attention_mask.dtype),
                ], dim=-1)
            if next_id == tokenizer.eos_token_id:
                break

    tokens = tokenizer.convert_ids_to_tokens(generated_ids)
    decoded = tokenizer.decode(generated_ids, skip_special_tokens=False)
    numeric_tokens = [token for token in tokens if NUMERIC_PATTERN.fullmatch(token or "")]
    return {
        "input_token_count": int(encoded["input_ids"].shape[-1]),
        "generated_ids": generated_ids,
        "generated_tokens": tokens,
        "decoded": decoded,
        "numeric_tokens": numeric_tokens,
        "numeric_token_count": len(numeric_tokens),
        "trace": trace,
    }


def teacher_forced_logits(model, tokenizer, prompt, continuation_ids):
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    prefix = encoded["input_ids"].to(model_device(model))
    logits_rows = []
    with torch.inference_mode():
        for token_id in continuation_ids:
            outputs = model(input_ids=prefix, use_cache=False, return_dict=True)
            logits_rows.append(outputs.logits[0, -1].detach().float().cpu().numpy())
            next_tensor = torch.tensor([[int(token_id)]], device=prefix.device, dtype=prefix.dtype)
            prefix = torch.cat([prefix, next_tensor], dim=-1)
    return np.stack(logits_rows) if logits_rows else np.empty((0, len(tokenizer)), dtype=np.float32)


def compare_arrays(reference, candidate):
    if reference.shape != candidate.shape:
        return {"shape_equal": False, "reference_shape": list(reference.shape), "candidate_shape": list(candidate.shape)}
    difference = np.abs(reference - candidate)
    return {
        "shape_equal": True,
        "max_abs_difference": float(difference.max()) if difference.size else 0.0,
        "mean_abs_difference": float(difference.mean()) if difference.size else 0.0,
    }


def cleanup(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except RuntimeError:
            pass


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose whether Mamba repetition is caused by prompting, merging, or CPT training."
    )
    parser.add_argument("--details_csv", required=True)
    parser.add_argument("--hist_len", type=int, required=True)
    parser.add_argument("--window_id", type=int, required=True)
    parser.add_argument("--column", default="OT")
    parser.add_argument("--merged_model_path", required=True)
    parser.add_argument("--base_model_path", required=True)
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--tokenizer_path", default=None)
    parser.add_argument("--output_dir", default="outputs/mamba_merge_diagnosis")
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--weight_rows", type=int, default=256)
    parser.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dtype = resolve_dtype(args.dtype)

    frame = pd.read_csv(args.details_csv)
    required = {"hist_len", "window_id", "column", "history"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Details CSV is missing columns: {sorted(missing)}")
    selected = frame[
        (frame["hist_len"] == args.hist_len)
        & (frame["window_id"] == args.window_id)
        & (frame["column"] == args.column)
    ]
    if len(selected) != 1:
        raise ValueError(f"Expected exactly one matching row, found {len(selected)}.")
    row = selected.iloc[0]
    history = parse_array(row["history"])

    tokenizer_path = args.tokenizer_path or args.adapter_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        use_fast=True,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    discretizer = Discretizer()
    serializer = Serializer()
    serialized = serializer.serialize(discretizer.discretize(history))
    prompts = {
        "instruction": getPrompt(flag="prediction", context=None, input=serialized),
        "numeric_only": serialized.rstrip() + " ",
    }
    numeric_items = numeric_vocab(tokenizer)

    report = {
        "selection": {
            "details_csv": args.details_csv,
            "hist_len": args.hist_len,
            "window_id": args.window_id,
            "column": args.column,
            "history_length": len(history),
            "pred_len": int(row.get("pred_len", -1)),
            "sample_seed": int(row.get("sample_seed", -1)),
        },
        "tokenizer": {
            "path": tokenizer_path,
            "size": len(tokenizer),
            "numeric_token_count": len(numeric_items),
        },
        "conditions": {},
    }

    # Phase 1: unmerged adapter. Its numeric-only greedy continuation becomes the teacher-forced reference.
    print("\n[1/2] Loading unmerged base + adapter")
    unmerged = load_unmerged(
        args.base_model_path, args.adapter_path, tokenizer, dtype,
        args.device_map, args.local_files_only,
    )
    unmerged_snapshot = snapshot_numeric_weights(unmerged, numeric_items, args.weight_rows)
    for prompt_name, prompt in prompts.items():
        print(f"Unmerged / {prompt_name}")
        report["conditions"][f"unmerged_{prompt_name}"] = greedy_trace(
            unmerged, tokenizer, prompt, args.max_new_tokens, args.top_k
        )

    reference_ids = report["conditions"]["unmerged_numeric_only"]["generated_ids"]
    unmerged_tf = teacher_forced_logits(unmerged, tokenizer, prompts["numeric_only"], reference_ids)
    np.save(output_dir / "unmerged_teacher_forced_logits.npy", unmerged_tf)
    np.savez_compressed(
        output_dir / "unmerged_numeric_weights.npz",
        token_ids=np.asarray(unmerged_snapshot["token_ids"], dtype=np.int64),
        input_rows=unmerged_snapshot["input_rows"],
        output_rows=unmerged_snapshot["output_rows"],
    )
    cleanup(unmerged)

    # Phase 2: merged model.
    print("\n[2/2] Loading merged model")
    merged = load_merged(
        args.merged_model_path, tokenizer, dtype,
        args.device_map, args.local_files_only,
    )
    merged_snapshot = snapshot_numeric_weights(merged, numeric_items, args.weight_rows)
    for prompt_name, prompt in prompts.items():
        print(f"Merged / {prompt_name}")
        report["conditions"][f"merged_{prompt_name}"] = greedy_trace(
            merged, tokenizer, prompt, args.max_new_tokens, args.top_k
        )

    merged_tf = teacher_forced_logits(merged, tokenizer, prompts["numeric_only"], reference_ids)
    cleanup(merged)

    report["comparisons"] = {
        "numeric_only_greedy_ids_equal": (
            report["conditions"]["unmerged_numeric_only"]["generated_ids"]
            == report["conditions"]["merged_numeric_only"]["generated_ids"]
        ),
        "instruction_greedy_ids_equal": (
            report["conditions"]["unmerged_instruction"]["generated_ids"]
            == report["conditions"]["merged_instruction"]["generated_ids"]
        ),
        "teacher_forced_logits": compare_arrays(unmerged_tf, merged_tf),
        "numeric_input_weights": compare_arrays(
            unmerged_snapshot["input_rows"], merged_snapshot["input_rows"]
        ),
        "numeric_output_weights": compare_arrays(
            unmerged_snapshot["output_rows"], merged_snapshot["output_rows"]
        ),
    }

    unmerged_numeric = report["conditions"]["unmerged_numeric_only"]["numeric_tokens"]
    merged_numeric = report["conditions"]["merged_numeric_only"]["numeric_tokens"]
    unmerged_instruction = report["conditions"]["unmerged_instruction"]["numeric_tokens"]
    merged_instruction = report["conditions"]["merged_instruction"]["numeric_tokens"]

    def repeated(tokens):
        return len(tokens) >= 2 and len(set(tokens)) == 1

    if not repeated(unmerged_numeric) and repeated(merged_numeric):
        diagnosis = "merge_likely"
        explanation = "Unmerged numeric-only generation varies, but merged numeric-only generation repeats."
    elif repeated(unmerged_numeric) and repeated(merged_numeric):
        diagnosis = "cpt_or_training_likely"
        explanation = "Both unmerged and merged numeric-only generations repeat. Merge is unlikely to be the primary cause."
    elif repeated(merged_instruction) and not repeated(merged_numeric):
        diagnosis = "prompt_mismatch_likely"
        explanation = "Merged instruction prompt repeats, but merged numeric-only prompt varies."
    elif not repeated(unmerged_instruction) and repeated(merged_instruction):
        diagnosis = "merge_or_prompt_interaction"
        explanation = "Instruction generation changes from varying to repeated after merge."
    else:
        diagnosis = "inconclusive"
        explanation = "The four conditions do not match a single simple failure pattern. Inspect traces and logits."

    report["diagnosis"] = {"label": diagnosis, "explanation": explanation}
    report_path = output_dir / "diagnosis_report.json"
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    summary = {
        "diagnosis": report["diagnosis"],
        "comparisons": report["comparisons"],
        "generated_tokens": {
            key: value["generated_tokens"]
            for key, value in report["conditions"].items()
        },
        "numeric_tokens": {
            key: value["numeric_tokens"]
            for key, value in report["conditions"].items()
        },
    }
    summary_path = output_dir / "diagnosis_summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print("DIAGNOSIS")
    print("=" * 80)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nSaved:")
    print(" -", report_path)
    print(" -", summary_path)
    print(" -", output_dir / "unmerged_teacher_forced_logits.npy")
    print(" -", output_dir / "unmerged_numeric_weights.npz")


if __name__ == "__main__":
    main()
