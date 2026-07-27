#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import shutil
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def resolve_effective_weight(layer):
    """
    PEFTのModulesToSaveWrapperなら、現在有効なmodules_to_saveの重みを返す。
    通常のEmbedding/Linearならlayer.weightを返す。
    """
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
        "Could not resolve an active modules_to_save weight. "
        f"active_adapter={active_adapter}, "
        f"available={list(modules_to_save.keys())}"
    )


def resolve_local_or_hub_path(path_value):
    """
    ローカルディレクトリなら絶対パスへ変換し、
    そうでなければHubのrepo IDとしてそのまま返す。
    """
    path = Path(path_value).expanduser()

    if path.exists():
        if not path.is_dir():
            raise NotADirectoryError(path)

        return str(path.resolve())

    return path_value


def validate_adapter_directory(adapter_path):
    required_files = [
        "adapter_config.json",
        "adapter_model.safetensors",
    ]

    for file_name in required_files:
        file_path = adapter_path / file_name

        if not file_path.is_file():
            raise FileNotFoundError(
                f"Required adapter file was not found: {file_path}"
            )


def get_numeric_token_ids(tokenizer):
    numeric_token_ids = []

    for token, token_id in tokenizer.get_vocab().items():
        if (
            token.startswith("###")
            and token.endswith("###")
        ):
            inner = token[3:-3]

            try:
                float(inner)
            except ValueError:
                continue

            numeric_token_ids.append(int(token_id))

    numeric_token_ids.sort()

    if not numeric_token_ids:
        raise ValueError(
            "No numeric tokens were found in the tokenizer."
        )

    return numeric_token_ids


def print_model_structure(model, tokenizer, label):
    input_layer = model.get_input_embeddings()
    output_layer = model.get_output_embeddings()

    input_weight = resolve_effective_weight(input_layer)
    output_weight = resolve_effective_weight(output_layer)

    print("\n" + "=" * 80)
    print(label)
    print("=" * 80)

    print("Input layer type:", type(input_layer).__name__)
    print("Output layer type:", type(output_layer).__name__)
    print("Tokenizer size:", len(tokenizer))
    print("Input rows:", input_weight.shape[0])
    print("Output rows:", output_weight.shape[0])
    print(
        "Config tie_word_embeddings:",
        getattr(
            model.config,
            "tie_word_embeddings",
            None,
        ),
    )


def snapshot_numeric_weights(model, tokenizer):
    """
    数値トークン行だけを検証用に保存する。
    全重みのコピーとは別に、マージ前後比較に使用する。
    """
    numeric_ids = get_numeric_token_ids(tokenizer)

    input_weight = resolve_effective_weight(
        model.get_input_embeddings()
    )

    output_weight = resolve_effective_weight(
        model.get_output_embeddings()
    )

    selected_ids = numeric_ids[
        ::max(1, len(numeric_ids) // 100)
    ]

    selected_ids = selected_ids[:100]

    index_tensor = torch.tensor(
        selected_ids,
        device=input_weight.device,
        dtype=torch.long,
    )

    input_rows = (
        input_weight
        .index_select(0, index_tensor)
        .detach()
        .float()
        .cpu()
        .clone()
    )

    output_index_tensor = index_tensor.to(
        output_weight.device
    )

    output_rows = (
        output_weight
        .index_select(0, output_index_tensor)
        .detach()
        .float()
        .cpu()
        .clone()
    )

    return {
        "token_ids": selected_ids,
        "input_rows": input_rows,
        "output_rows": output_rows,
    }


def compare_snapshot(model, snapshot, label):
    input_weight = model.get_input_embeddings().weight
    output_weight = model.get_output_embeddings().weight

    index_tensor = torch.tensor(
        snapshot["token_ids"],
        device=input_weight.device,
        dtype=torch.long,
    )

    current_input = (
        input_weight
        .index_select(0, index_tensor)
        .detach()
        .float()
        .cpu()
    )

    current_output = (
        output_weight
        .index_select(
            0,
            index_tensor.to(output_weight.device),
        )
        .detach()
        .float()
        .cpu()
    )

    input_difference = (
        current_input - snapshot["input_rows"]
    ).abs()

    output_difference = (
        current_output - snapshot["output_rows"]
    ).abs()

    result = {
        "label": label,
        "input_max_abs_difference": float(
            input_difference.max()
        ),
        "input_mean_abs_difference": float(
            input_difference.mean()
        ),
        "output_max_abs_difference": float(
            output_difference.max()
        ),
        "output_mean_abs_difference": float(
            output_difference.mean()
        ),
    }

    print("\nWeight comparison:")
    print(result)

    return result


def copy_effective_full_weights(peft_model):
    """
    modules_to_saveで有効になっているEmbeddingとLM Headを
    マージ前に完全コピーする。
    """
    input_weight = resolve_effective_weight(
        peft_model.get_input_embeddings()
    )

    output_weight = resolve_effective_weight(
        peft_model.get_output_embeddings()
    )

    print("\nCopying effective Embedding and LM Head weights")
    print("Input shape:", tuple(input_weight.shape))
    print("Output shape:", tuple(output_weight.shape))

    saved_input_weight = (
        input_weight
        .detach()
        .cpu()
        .clone()
    )

    saved_output_weight = (
        output_weight
        .detach()
        .cpu()
        .clone()
    )

    return saved_input_weight, saved_output_weight


def restore_independent_weights(
    merged_model,
    saved_input_weight,
    saved_output_weight,
):
    """
    EmbeddingとLM Headを独立Parameterとして復元する。

    tie_word_embeddings=Trueのままだと再ロード時に片方が
    上書きされる可能性があるため、Falseへ変更する。
    """
    input_layer = merged_model.get_input_embeddings()
    output_layer = merged_model.get_output_embeddings()

    if input_layer.weight.shape != saved_input_weight.shape:
        raise ValueError(
            "Input embedding shape mismatch: "
            f"model={tuple(input_layer.weight.shape)}, "
            f"saved={tuple(saved_input_weight.shape)}"
        )

    if output_layer.weight.shape != saved_output_weight.shape:
        raise ValueError(
            "LM Head shape mismatch: "
            f"model={tuple(output_layer.weight.shape)}, "
            f"saved={tuple(saved_output_weight.shape)}"
        )

    input_device = input_layer.weight.device
    input_dtype = input_layer.weight.dtype
    output_device = output_layer.weight.device
    output_dtype = output_layer.weight.dtype

    # 同じstorageを共有しない独立Parameterとして設定する。
    input_layer.weight = torch.nn.Parameter(
        saved_input_weight.to(
            device=input_device,
            dtype=input_dtype,
        ),
        requires_grad=False,
    )

    output_layer.weight = torch.nn.Parameter(
        saved_output_weight.to(
            device=output_device,
            dtype=output_dtype,
        ),
        requires_grad=False,
    )

    merged_model.config.tie_word_embeddings = False

    if hasattr(merged_model, "generation_config"):
        merged_model.generation_config.pad_token_id = (
            merged_model.config.pad_token_id
        )
        merged_model.generation_config.eos_token_id = (
            merged_model.config.eos_token_id
        )

    same_storage = (
        input_layer.weight.data_ptr()
        == output_layer.weight.data_ptr()
    )

    print("\nRestored independent weights")
    print("tie_word_embeddings:", merged_model.config.tie_word_embeddings)
    print("Input and output share storage:", same_storage)

    if same_storage:
        raise RuntimeError(
            "Embedding and LM Head still share storage "
            "after restoring independent weights."
        )


def get_next_token_logits(model, tokenizer, prompt):
    model.eval()

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=True,
    )

    device = next(model.parameters()).device

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        outputs = model(**inputs)

    return (
        outputs.logits[0, -1]
        .detach()
        .float()
        .cpu()
    )


def compare_logits(reference_logits, candidate_logits):
    difference = (
        reference_logits
        - candidate_logits
    ).abs()

    return {
        "logits_max_abs_difference": float(
            difference.max()
        ),
        "logits_mean_abs_difference": float(
            difference.mean()
        ),
        "reference_argmax": int(
            reference_logits.argmax()
        ),
        "candidate_argmax": int(
            candidate_logits.argmax()
        ),
        "argmax_equal": bool(
            reference_logits.argmax()
            == candidate_logits.argmax()
        ),
    }


def generate_numeric_continuation(
    model,
    tokenizer,
    prompt,
    label,
    max_new_tokens=40,
):
    model.eval()

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=True,
    )

    device = next(model.parameters()).device

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    input_length = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )

    generated_ids = outputs[0, input_length:]

    generated_tokens = (
        tokenizer.convert_ids_to_tokens(
            generated_ids.tolist()
        )
    )

    generated_text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    )

    numeric_tokens = [
        token
        for token in generated_tokens
        if (
            token.startswith("###")
            and token.endswith("###")
        )
    ]

    print("\n" + "=" * 80)
    print(label)
    print("=" * 80)

    print("Generated IDs:")
    print(generated_ids.tolist())

    print("Generated tokens:")
    print(generated_tokens)

    print("Decoded:")
    print(repr(generated_text))

    print("Numeric token count:")
    print(len(numeric_tokens))

    return {
        "generated_ids": generated_ids.tolist(),
        "generated_tokens": generated_tokens,
        "generated_text": generated_text,
        "numeric_token_count": len(numeric_tokens),
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Safely merge a CPT LoRA adapter while preserving "
            "modules_to_save Embedding and LM Head weights."
        )
    )

    parser.add_argument(
        "--adapter_path",
        required=True,
    )

    parser.add_argument(
        "--output_path",
        required=True,
    )

    parser.add_argument(
        "--base_model_path",
        default=None,
    )

    parser.add_argument(
        "--dtype",
        choices=["fp16", "bf16", "fp32"],
        default="fp16",
    )

    parser.add_argument(
        "--device_map",
        default="auto",
    )

    parser.add_argument(
        "--local_files_only",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    parser.add_argument(
        "--max_shard_size",
        default="5GB",
    )

    args = parser.parse_args()

    adapter_path = Path(
        args.adapter_path
    ).expanduser().resolve()

    output_path = Path(
        args.output_path
    ).expanduser().resolve()

    validate_adapter_directory(adapter_path)

    if output_path.exists():
        if any(output_path.iterdir()):
            raise FileExistsError(
                "Output directory already exists and is not empty: "
                f"{output_path}"
            )
    else:
        output_path.mkdir(
            parents=True,
            exist_ok=False,
        )

    with (
        adapter_path / "adapter_config.json"
    ).open("r", encoding="utf-8") as file:
        adapter_config = json.load(file)

    configured_base_path = adapter_config.get(
        "base_model_name_or_path"
    )

    base_model_path = (
        args.base_model_path
        or configured_base_path
    )

    if not base_model_path:
        raise ValueError(
            "Base model path was not provided and was not found "
            "in adapter_config.json."
        )

    base_model_path = resolve_local_or_hub_path(
        base_model_path
    )

    dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[args.dtype]

    print("Adapter:", adapter_path)
    print("Base model:", base_model_path)
    print("Output:", output_path)
    print("modules_to_save:", adapter_config.get("modules_to_save"))
    print("Target modules:", adapter_config.get("target_modules"))

    # Tokenizerは、数値語彙を保存したCPT checkpoint側から読む。
    tokenizer = AutoTokenizer.from_pretrained(
        str(adapter_path),
        use_fast=True,
        trust_remote_code=True,
        local_files_only=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Tokenizer vocabulary size:", len(tokenizer))

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        dtype=dtype,
        device_map=args.device_map,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )

    original_vocab_size = (
        base_model.get_input_embeddings()
        .weight.shape[0]
    )

    if original_vocab_size != len(tokenizer):
        print(
            "Resizing base vocabulary:",
            original_vocab_size,
            "->",
            len(tokenizer),
        )

        base_model.resize_token_embeddings(
            len(tokenizer)
        )

    # アダプターをロードするとmodules_to_saveの完全重みが有効になる。
    peft_model = PeftModel.from_pretrained(
        base_model,
        str(adapter_path),
        is_trainable=False,
        local_files_only=True,
    )

    peft_model.eval()

    print_model_structure(
        peft_model,
        tokenizer,
        "UNMERGED PEFT MODEL",
    )

    test_prompt = (
        "###-0.5001### "
        "###-0.4787### "
        "###-0.4575### "
        "###-0.4361### "
        "###-0.4149### "
        "###-0.3937### "
        "###-0.3723### "
        "###-0.3511### "
    )

    # 数値同士の区切り文字を末尾に追加する。
    test_prompt = test_prompt.rstrip() + " "

    reference_logits = get_next_token_logits(
        peft_model,
        tokenizer,
        test_prompt,
    )

    pre_merge_generation = generate_numeric_continuation(
        peft_model,
        tokenizer,
        test_prompt,
        "UNMERGED PEFT GENERATION",
    )

    if pre_merge_generation["numeric_token_count"] == 0:
        raise RuntimeError(
            "The unmerged PEFT model did not generate numeric tokens. "
            "The adapter itself should be investigated before merging."
        )

    numeric_snapshot = snapshot_numeric_weights(
        peft_model,
        tokenizer,
    )

    saved_input_weight, saved_output_weight = (
        copy_effective_full_weights(
            peft_model
        )
    )

    print("\nMerging LoRA layers")

    merged_model = peft_model.merge_and_unload(
        safe_merge=True,
    )

    # modules_to_saveの完全重みを明示的に戻す。
    restore_independent_weights(
        merged_model,
        saved_input_weight,
        saved_output_weight,
    )

    merged_model.config.vocab_size = len(
        tokenizer
    )

    merged_model.config.pad_token_id = (
        tokenizer.pad_token_id
    )
    merged_model.config.eos_token_id = (
        tokenizer.eos_token_id
    )
    merged_model.config.bos_token_id = (
        tokenizer.bos_token_id
    )

    compare_snapshot(
        merged_model,
        numeric_snapshot,
        "IN-MEMORY MERGED MODEL",
    )

    in_memory_logits = get_next_token_logits(
        merged_model,
        tokenizer,
        test_prompt,
    )

    in_memory_logit_comparison = compare_logits(
        reference_logits,
        in_memory_logits,
    )

    print("\nIn-memory logit comparison:")
    print(in_memory_logit_comparison)

    in_memory_generation = generate_numeric_continuation(
        merged_model,
        tokenizer,
        test_prompt,
        "IN-MEMORY MERGED GENERATION",
    )

    if in_memory_generation["numeric_token_count"] == 0:
        raise RuntimeError(
            "The in-memory merged model lost numeric generation ability."
        )

    print("\nSaving merged model")

    merged_model.save_pretrained(
        str(output_path),
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )

    tokenizer.save_pretrained(
        str(output_path)
    )

    # マージに関する情報を保存する。
    merge_info = {
        "base_model_path": str(base_model_path),
        "adapter_path": str(adapter_path),
        "output_path": str(output_path),
        "dtype": args.dtype,
        "tokenizer_vocab_size": len(tokenizer),
        "tie_word_embeddings": False,
        "pre_merge_generation": pre_merge_generation,
        "in_memory_generation": in_memory_generation,
        "in_memory_logit_comparison": (
            in_memory_logit_comparison
        ),
    }

    with (
        output_path / "merge_validation.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(
            merge_info,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\nReloading saved model")

    del merged_model
    del peft_model
    del base_model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    reloaded_model = (
        AutoModelForCausalLM.from_pretrained(
            str(output_path),
            dtype=dtype,
            device_map=args.device_map,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            local_files_only=True,
        )
    )

    reloaded_tokenizer = (
        AutoTokenizer.from_pretrained(
            str(output_path),
            use_fast=True,
            trust_remote_code=True,
            local_files_only=True,
        )
    )

    reloaded_model.eval()

    if len(reloaded_tokenizer) != len(tokenizer):
        raise ValueError(
            "Reloaded tokenizer vocabulary mismatch: "
            f"saved={len(tokenizer)}, "
            f"reloaded={len(reloaded_tokenizer)}"
        )

    print_model_structure(
        reloaded_model,
        reloaded_tokenizer,
        "RELOADED MERGED MODEL",
    )

    reload_weight_comparison = compare_snapshot(
        reloaded_model,
        numeric_snapshot,
        "RELOADED MERGED MODEL",
    )

    reloaded_logits = get_next_token_logits(
        reloaded_model,
        reloaded_tokenizer,
        test_prompt,
    )

    reload_logit_comparison = compare_logits(
        reference_logits,
        reloaded_logits,
    )

    print("\nReloaded logit comparison:")
    print(reload_logit_comparison)

    reloaded_generation = generate_numeric_continuation(
        reloaded_model,
        reloaded_tokenizer,
        test_prompt,
        "RELOADED MERGED GENERATION",
    )

    if reloaded_generation["numeric_token_count"] == 0:
        raise RuntimeError(
            "The reloaded merged model lost numeric generation ability."
        )

    merge_info.update({
        "reload_weight_comparison": (
            reload_weight_comparison
        ),
        "reload_logit_comparison": (
            reload_logit_comparison
        ),
        "reloaded_generation": (
            reloaded_generation
        ),
    })

    with (
        output_path / "merge_validation.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(
            merge_info,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\n" + "=" * 80)
    print("MERGE COMPLETED AND VALIDATED")
    print("=" * 80)
    print("Saved model:", output_path)
    print(
        "Reloaded numeric tokens:",
        reloaded_generation[
            "numeric_token_count"
        ],
    )
    print(
        "Reloaded argmax equal:",
        reload_logit_comparison[
            "argmax_equal"
        ],
    )


if __name__ == "__main__":
    main()