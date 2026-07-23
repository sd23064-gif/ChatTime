import argparse
import json
from pathlib import Path

import torch
from unsloth import FastLanguageModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    args = parser.parse_args()

    adapter_path = Path(args.adapter_path).expanduser().resolve()
    output_path = Path(args.output_path).expanduser().resolve()

    if not adapter_path.is_dir():
        raise FileNotFoundError(
            f"Adapter directory not found: {adapter_path}"
        )

    required_files = [
        "adapter_config.json",
        "adapter_model.safetensors",
    ]

    for file_name in required_files:
        file_path = adapter_path / file_name

        if not file_path.is_file():
            raise FileNotFoundError(
                f"Required adapter file not found: {file_path}"
            )

    with (adapter_path / "adapter_config.json").open(
        "r",
        encoding="utf-8"
    ) as file:
        adapter_config = json.load(file)

    base_model_path = adapter_config.get(
        "base_model_name_or_path"
    )

    if not base_model_path:
        raise ValueError(
            "base_model_name_or_path is not present in "
            "adapter_config.json"
        )

    print("CPT adapter:", adapter_path)
    print("Base model:", base_model_path)
    print("Output:", output_path)

    output_path.mkdir(parents=True, exist_ok=True)

    # adapter_pathを指定すると、Unslothがadapter_config.jsonから
    # ベースモデルを特定してLoRA adapterをロードする
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter_path),
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=False,
    )

    print("CPT adapter loaded successfully")
    print("Saving merged 16-bit model...")

    model.save_pretrained_merged(
        str(output_path),
        tokenizer,
        save_method="merged_16bit",
    )

    print("Merge completed")
    print("Merged model:", output_path)


if __name__ == "__main__":
    main()