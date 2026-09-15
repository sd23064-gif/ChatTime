#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import re

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


NUMERIC_PATTERN = re.compile(
    r"###[+-]?(?:\d+(?:\.\d*)?|\.\d+)###"
)


def print_top_tokens(model, tokenizer, prompt, label, top_k=30):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=True,
    )

    inputs = {
        key: value.to(model.device)
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        outputs = model(**inputs)

    logits = outputs.logits[0, -1].float()

    probabilities = torch.softmax(
        logits,
        dim=-1,
    )

    top_probabilities, top_ids = torch.topk(
        probabilities,
        k=top_k,
    )

    print("\n" + "=" * 80)
    print(label)
    print("=" * 80)
    print("Prompt ending:", repr(prompt[-100:]))

    numeric_probability = probabilities[
        128256:138257
    ].sum()

    print(
        "Total probability assigned to numeric-token block:",
        float(numeric_probability),
    )

    numeric_count = 0

    for rank, (token_id, probability) in enumerate(
        zip(
            top_ids.tolist(),
            top_probabilities.tolist(),
        ),
        start=1,
    ):
        token = tokenizer.convert_ids_to_tokens(
            token_id
        )

        is_numeric = bool(
            NUMERIC_PATTERN.fullmatch(token)
        )

        if is_numeric:
            numeric_count += 1

        print({
            "rank": rank,
            "token_id": token_id,
            "token": token,
            "probability": probability,
            "is_numeric": is_numeric,
            "is_eos": (
                token_id
                == tokenizer.eos_token_id
            ),
            "is_pad": (
                token_id
                == tokenizer.pad_token_id
            ),
        })

    print(
        "Numeric tokens in top candidates:",
        numeric_count,
    )


def greedy_generate(model, tokenizer, prompt, label):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=True,
    )

    inputs = {
        key: value.to(model.device)
        for key, value in inputs.items()
    }

    input_length = inputs[
        "input_ids"
    ].shape[-1]

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=40,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

    generated_ids = outputs[
        0,
        input_length:,
    ]

    tokens = tokenizer.convert_ids_to_tokens(
        generated_ids.tolist()
    )

    text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    )

    numeric_tokens = [
        token
        for token in tokens
        if NUMERIC_PATTERN.fullmatch(token)
    ]

    print("\n" + "=" * 80)
    print(label)
    print("=" * 80)
    print("Generated IDs:", generated_ids.tolist())
    print("Generated tokens:", tokens)
    print("Decoded:", repr(text))
    print("Numeric tokens:", numeric_tokens)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base_model_path",
        required=True,
    )

    parser.add_argument(
        "--adapter_path",
        required=True,
    )

    parser.add_argument(
        "--merged_model_path",
        required=True,
    )

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.adapter_path,
        use_fast=True,
        local_files_only=True,
        trust_remote_code=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    if (
        base_model.get_input_embeddings()
        .weight.shape[0]
        != len(tokenizer)
    ):
        print(
            "Resizing base vocabulary:",
            base_model.get_input_embeddings()
            .weight.shape[0],
            "->",
            len(tokenizer),
        )

        base_model.resize_token_embeddings(
            len(tokenizer)
        )

    adapter_model = PeftModel.from_pretrained(
        base_model,
        args.adapter_path,
        is_trainable=False,
        local_files_only=True,
    )

    adapter_model.eval()

    merged_model = AutoModelForCausalLM.from_pretrained(
        args.merged_model_path,
        dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
        local_files_only=True,
        trust_remote_code=True,
    )

    merged_model.eval()

    numeric_sequence = (
        "###-0.5001### "
        "###-0.4787### "
        "###-0.4575### "
        "###-0.4361### "
        "###-0.4149### "
        "###-0.3937### "
        "###-0.3723### "
        "###-0.3511### "
    )

    # 条件1: 最後が数値トークン
    prompt_without_space = (
        numeric_sequence.rstrip()
    )

    # 条件2: 最後が区切り空白
    prompt_with_space = (
        numeric_sequence.rstrip() + " "
    )

    for prompt_name, prompt in [
        (
            "without trailing separator",
            prompt_without_space,
        ),
        (
            "with trailing separator",
            prompt_with_space,
        ),
    ]:
        print_top_tokens(
            adapter_model,
            tokenizer,
            prompt,
            label=(
                "UNMERGED ADAPTER: "
                + prompt_name
            ),
        )

        print_top_tokens(
            merged_model,
            tokenizer,
            prompt,
            label=(
                "MERGED MODEL: "
                + prompt_name
            ),
        )

    greedy_generate(
        adapter_model,
        tokenizer,
        prompt_with_space,
        label="UNMERGED ADAPTER GENERATION",
    )

    greedy_generate(
        merged_model,
        tokenizer,
        prompt_with_space,
        label="MERGED MODEL GENERATION",
    )


if __name__ == "__main__":
    main()