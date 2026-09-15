import argparse
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def parse_args():
    parser = argparse.ArgumentParser(
        description="falcon-mamba-7bに学習したLoRAアダプタをマージしてフルモデルとして保存する"
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="tiiuae/falcon-mamba-7b",
        help="ベースモデルのパス、またはHugging Face Hub上のID",
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        required=True,
        help="学習スクリプトが保存したLoRAアダプタのディレクトリ(output_pathまたはstep-N)。"
        "追加した数値トークン入りのtokenizerもここに保存されている",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="マージ後のフルモデルを保存する先",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="マージ・保存に使うdtype",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="マージ処理を行うデバイス。7Bモデルの場合、GPUメモリに余裕があれば"
        "'cuda:0'の方が高速。CPUの場合は十分なRAM(目安28GB以上)が必要",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        default=False,
        help="指定した場合、マージ後のモデルをHugging Face Hubにpushする",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="push_to_hub時のリポジトリID (例: username/falcon-mamba-7b-merged)",
    )
    return parser.parse_args()


def resolve_dtype(name):
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def verify_numeric_tokens_merged(tokenizer, model, old_vocab_size, n_samples=3):
    """
    マージ後、新規追加した数値トークンのembeddingが単なる初期値(ゼロ等)の
    ままになっていないか(=アダプタが正しく反映されているか)を軽く確認する。
    """
    new_vocab_size = len(tokenizer)

    if new_vocab_size <= old_vocab_size:
        print("追加トークンが見つからないため、embedding検証をスキップします")
        return

    sample_ids = torch.linspace(
        old_vocab_size, new_vocab_size - 1, steps=n_samples
    ).long()

    input_embedding = model.get_input_embeddings()

    print("\n追加トークンembeddingの検証(マージ後)")
    for token_id in sample_ids.tolist():
        token_str = tokenizer.convert_ids_to_tokens(token_id)
        norm = input_embedding.weight[token_id].detach().float().norm().item()
        print(f"  token_id={token_id} token={token_str} embedding_norm={norm:.4f}")


def main():
    args = parse_args()
    dtype = resolve_dtype(args.dtype)

    # 【注意】 tokenizerは必ずadapter_path側から読み込む。
    # 学習時にadd_tokens()した数値トークン(###0.1234###等)がここに
    # 含まれているため、base_model_path側のtokenizerを使うと語彙サイズが
    # ずれてembeddingのshapeが一致しなくなる。
    print(f"Loading tokenizer from: {args.adapter_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.adapter_path,
        trust_remote_code=True,
    )
    print(f"Tokenizer vocab size (追加トークン込み): {len(tokenizer)}")

    print(f"Loading base model from: {args.base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        torch_dtype=dtype,
        device_map=args.device,
        trust_remote_code=True,
    )

    # 【重要】 学習時(resize_token_embeddings→アダプタ学習)と同じ順序で
    # embeddingを拡張してからアダプタをロードする。これを先にしないと、
    # adapter_model.safetensors内のbackbone.embeddings/lm_head
    # (modules_to_saveでフル保存されている)のshapeとbase_modelの
    # shapeが一致せず、PeftModel.from_pretrained時にエラーになる。
    old_vocab_size = base_model.get_input_embeddings().weight.shape[0]

    if old_vocab_size != len(tokenizer):
        print(f"Resizing token embeddings: {old_vocab_size} -> {len(tokenizer)}")
        base_model.resize_token_embeddings(len(tokenizer))
    else:
        print("Token embeddings are already the correct size; resize skipped")

    print(f"Loading LoRA adapter from: {args.adapter_path}")
    peft_model = PeftModel.from_pretrained(
        base_model,
        args.adapter_path,
        torch_dtype=dtype,
    )

    print("Merging LoRA weights into the base model...")
    merged_model = peft_model.merge_and_unload()
    merged_model = merged_model.to(dtype)

    verify_numeric_tokens_merged(tokenizer, merged_model, old_vocab_size)

    print(f"\nSaving merged model to: {args.output_path}")
    os.makedirs(args.output_path, exist_ok=True)
    merged_model.save_pretrained(
        args.output_path,
        safe_serialization=True,
    )
    tokenizer.save_pretrained(args.output_path)

    print("Merge complete.")
    print(f"Final vocab size: {len(tokenizer)}")
    print(
        "Final input embedding shape: "
        f"{tuple(merged_model.get_input_embeddings().weight.shape)}"
    )

    if args.push_to_hub:
        if args.hub_model_id is None:
            raise ValueError("--push_to_hub指定時は--hub_model_idが必須です")
        print(f"Pushing merged model to hub: {args.hub_model_id}")
        merged_model.push_to_hub(args.hub_model_id)
        tokenizer.push_to_hub(args.hub_model_id)


if __name__ == "__main__":
    main()
