import argparse
import inspect
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


def is_bfloat16_supported():
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
RANK = int(os.environ.get("RANK", 0))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
IS_MAIN_PROCESS = RANK == 0


def log0(message):
    """rank 0だけがログを出力する。"""
    if IS_MAIN_PROCESS:
        print(message, flush=True)


def add_eos_to_example(example, eos_token):
    """単一文字列と文字列リストの両方にEOSを追加する。"""

    def add_eos(text):
        if text is None:
            text = ""
        elif not isinstance(text, str):
            text = str(text)

        if eos_token and not text.endswith(eos_token):
            text += eos_token

        return text

    texts = example["text"]

    if isinstance(texts, list):
        return [add_eos(text) for text in texts]

    return add_eos(texts)


def print_trainable_parameters(model):
    """学習対象パラメータ名と総数を表示する。"""
    trainable_params = 0
    total_params = 0

    log0("\n=== Trainable parameters ===")

    for name, parameter in model.named_parameters():
        total_params += parameter.numel()

        if parameter.requires_grad:
            trainable_params += parameter.numel()
            log0(
                f"{name}: shape={tuple(parameter.shape)}, "
                f"numel={parameter.numel():,}, dtype={parameter.dtype}"
            )

    trainable_ratio = (
        100.0 * trainable_params / total_params
        if total_params > 0
        else 0.0
    )

    log0(f"\nTrainable parameters: {trainable_params:,}")
    log0(f"Total parameters:     {total_params:,}")
    log0(f"Trainable ratio:      {trainable_ratio:.6f}%\n")


class NumericLossSFTTrainer(SFTTrainer):
    """
    通常のCausal LM Cross-Entropyに加えて、数値トークンに対する
    Wasserstein-1、Ordinal CE、Gaussian soft-label CE、
    期待値回帰、一次difference lossを計算するTrainer。
    """

    def __init__(
        self,
        *trainer_args,
        numeric_token_ids,
        numeric_bin_centers,
        wasserstein_weight=0.0,
        ordinal_ce_weight=0.0,
        soft_ce_weight=0.0,
        regression_weight=0.0,
        difference_weight=0.0,
        soft_label_sigma_bins=1.5,
        soft_label_radius=4,
        numeric_loss_type="smooth_l1",
        ordinal_chunk_size=1024,
        **trainer_kwargs,
    ):
        super().__init__(*trainer_args, **trainer_kwargs)

        self.numeric_token_ids_cpu = torch.as_tensor(
            numeric_token_ids,
            dtype=torch.long,
        )
        self.numeric_bin_centers_cpu = torch.as_tensor(
            numeric_bin_centers,
            dtype=torch.float32,
        )

        if self.numeric_token_ids_cpu.numel() != self.numeric_bin_centers_cpu.numel():
            raise ValueError(
                "numeric_token_idsとnumeric_bin_centersの長さが一致しません。"
            )

        if self.numeric_bin_centers_cpu.numel() < 2:
            raise ValueError("数値ビンが2個未満です。")

        if not torch.all(
            self.numeric_bin_centers_cpu[1:]
            > self.numeric_bin_centers_cpu[:-1]
        ):
            raise ValueError(
                "numeric_bin_centersは厳密な昇順である必要があります。"
            )

        if soft_label_sigma_bins <= 0:
            raise ValueError(
                "soft_label_sigma_binsは0より大きい必要があります。"
            )

        if soft_label_radius < 0:
            raise ValueError(
                "soft_label_radiusは0以上である必要があります。"
            )

        if ordinal_chunk_size <= 0:
            raise ValueError(
                "ordinal_chunk_sizeは0より大きい必要があります。"
            )

        self.wasserstein_weight = wasserstein_weight
        self.ordinal_ce_weight = ordinal_ce_weight
        self.soft_ce_weight = soft_ce_weight
        self.regression_weight = regression_weight
        self.difference_weight = difference_weight

        self.soft_label_sigma_bins = soft_label_sigma_bins
        self.soft_label_radius = soft_label_radius
        self.numeric_loss_type = numeric_loss_type
        self.ordinal_chunk_size = ordinal_chunk_size

        self._numeric_token_ids_cache = {}
        self._bin_centers_cache = {}
        self._vocab_to_bin_cache = {}

    def _get_numeric_tensors(self, device, vocab_size):
        """device上の数値トークン情報と語彙ID→ビンID対応表を返す。"""
        device_key = str(device)

        if device_key not in self._numeric_token_ids_cache:
            self._numeric_token_ids_cache[device_key] = (
                self.numeric_token_ids_cpu.to(device)
            )
            self._bin_centers_cache[device_key] = (
                self.numeric_bin_centers_cpu.to(device)
            )

        cache_key = (device_key, vocab_size)

        if cache_key not in self._vocab_to_bin_cache:
            numeric_token_ids = self._numeric_token_ids_cache[device_key]

            if numeric_token_ids.min().item() < 0:
                raise RuntimeError("数値token IDに負の値が含まれています。")

            if numeric_token_ids.max().item() >= vocab_size:
                raise RuntimeError(
                    "数値token IDがモデルの語彙サイズを超えています。"
                )

            vocab_to_bin = torch.full(
                (vocab_size,),
                fill_value=-1,
                dtype=torch.long,
                device=device,
            )

            vocab_to_bin[numeric_token_ids] = torch.arange(
                numeric_token_ids.numel(),
                dtype=torch.long,
                device=device,
            )

            self._vocab_to_bin_cache[cache_key] = vocab_to_bin

        return (
            self._numeric_token_ids_cache[device_key],
            self._bin_centers_cache[device_key],
            self._vocab_to_bin_cache[cache_key],
        )

    def _pointwise_loss(self, prediction, target):
        """回帰損失とdifference lossの各要素を計算する。"""
        if self.numeric_loss_type == "smooth_l1":
            return F.smooth_l1_loss(
                prediction,
                target,
                reduction="none",
            )

        if self.numeric_loss_type == "mae":
            return (prediction - target).abs()

        if self.numeric_loss_type == "mse":
            return (prediction - target).square()

        raise RuntimeError(
            f"未対応のnumeric_loss_type: {self.numeric_loss_type}"
        )

    @staticmethod
    def _masked_mean(values, mask):
        """mask=Trueの要素だけで平均する。"""
        float_mask = mask.to(values.dtype)
        denominator = float_mask.sum()

        if denominator.detach().item() == 0:
            return values.sum() * 0.0

        return (values * float_mask).sum() / denominator

    def _wasserstein_loss(
        self,
        numeric_probs,
        target_bin_ids,
        bin_centers,
        numeric_mask,
    ):
        """
        one-hot教師に対するWasserstein-1損失。

        W1(p, delta_y) = sum_k p_k * |c_k - c_y|
        """
        safe_target_bin_ids = target_bin_ids.clamp_min(0)
        target_values = bin_centers[safe_target_bin_ids]

        distances = (
            bin_centers.view(1, 1, -1)
            - target_values.unsqueeze(-1)
        ).abs()

        element_loss = (
            numeric_probs * distances
        ).sum(dim=-1)

        return self._masked_mean(
            element_loss,
            numeric_mask,
        )

    def _ordinal_ce_loss(
        self,
        numeric_probs,
        target_bin_ids,
        numeric_mask,
    ):
        """
        予測CDFとone-hot教師CDFの間でBinary CEを計算する。

        予測CDF:
            P_k = P(Y <= k)

        教師CDF:
            Q_k = 1[y <= k]
        """
        num_bins = numeric_probs.size(-1)

        if num_bins < 2:
            return numeric_probs.sum() * 0.0

        pred_cdf = numeric_probs.cumsum(dim=-1)[..., :-1]
        pred_cdf = pred_cdf.clamp(1e-7, 1.0 - 1e-7)

        safe_target_bin_ids = target_bin_ids.clamp_min(0)
        element_loss = pred_cdf.new_zeros(
            pred_cdf.shape[:-1]
        )

        for start in range(
            0,
            num_bins - 1,
            self.ordinal_chunk_size,
        ):
            end = min(
                start + self.ordinal_chunk_size,
                num_bins - 1,
            )

            boundary_ids = torch.arange(
                start,
                end,
                device=pred_cdf.device,
                dtype=torch.long,
            )

            target_cdf = (
                boundary_ids.view(1, 1, -1)
                >= safe_target_bin_ids.unsqueeze(-1)
            ).to(pred_cdf.dtype)

            current_pred_cdf = pred_cdf[..., start:end]

            chunk_loss = -(
                target_cdf * torch.log(current_pred_cdf)
                + (1.0 - target_cdf)
                * torch.log1p(-current_pred_cdf)
            )

            element_loss = element_loss + chunk_loss.sum(dim=-1)

        element_loss = element_loss / float(num_bins - 1)

        return self._masked_mean(
            element_loss,
            numeric_mask,
        )

    def _local_gaussian_soft_ce(
        self,
        numeric_log_probs,
        target_bin_ids,
        numeric_mask,
    ):
        """
        正解ビン周辺だけにGaussian教師確率を与えるsoft-label CE。
        """
        num_bins = numeric_log_probs.size(-1)
        safe_target_bin_ids = target_bin_ids.clamp_min(0)

        offsets = torch.arange(
            -self.soft_label_radius,
            self.soft_label_radius + 1,
            dtype=torch.long,
            device=numeric_log_probs.device,
        )

        local_bin_ids = (
            safe_target_bin_ids.unsqueeze(-1)
            + offsets.view(1, 1, -1)
        )

        valid_local_ids = (
            (local_bin_ids >= 0)
            & (local_bin_ids < num_bins)
        )

        local_bin_ids = local_bin_ids.clamp(
            min=0,
            max=num_bins - 1,
        )

        local_log_probs = torch.gather(
            numeric_log_probs,
            dim=-1,
            index=local_bin_ids,
        )

        gaussian_weights = torch.exp(
            -offsets.float().square()
            / (2.0 * self.soft_label_sigma_bins**2)
        ).view(1, 1, -1)

        gaussian_weights = (
            gaussian_weights
            * valid_local_ids.to(gaussian_weights.dtype)
        )

        gaussian_weights = (
            gaussian_weights
            / gaussian_weights.sum(
                dim=-1,
                keepdim=True,
            ).clamp_min(1e-8)
        )

        element_loss = -(
            gaussian_weights * local_log_probs
        ).sum(dim=-1)

        return self._masked_mean(
            element_loss,
            numeric_mask,
        )

    def _expected_regression_loss(
        self,
        predicted_values,
        target_values,
        numeric_mask,
    ):
        """softmax期待値と正解ビン中心との回帰損失。"""
        element_loss = self._pointwise_loss(
            predicted_values,
            target_values,
        )

        return self._masked_mean(
            element_loss,
            numeric_mask,
        )

    def _difference_loss(
        self,
        predicted_values,
        target_values,
        numeric_mask,
    ):
        """
        連続する2位置が両方とも数値の場合に一次差分を比較する。
        """
        if predicted_values.size(1) < 2:
            return predicted_values.sum() * 0.0

        predicted_difference = (
            predicted_values[:, 1:]
            - predicted_values[:, :-1]
        )
        target_difference = (
            target_values[:, 1:]
            - target_values[:, :-1]
        )

        pair_mask = (
            numeric_mask[:, 1:]
            & numeric_mask[:, :-1]
        )

        element_loss = self._pointwise_loss(
            predicted_difference,
            target_difference,
        )

        return self._masked_mean(
            element_loss,
            pair_mask,
        )

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        """
        Causal LMの1位置シフトを考慮して損失を計算する。

        logits[:, t]はlabels[:, t + 1]を予測するため、
        logits[:, :-1]とlabels[:, 1:]を比較する。
        """
        labels = inputs.get("labels")

        if labels is None:
            raise RuntimeError(
                "inputsにlabelsがありません。"
                "SFTTrainerまたはdata collatorの設定を確認してください。"
            )

        outputs = model(**inputs)
        logits = outputs.logits

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        vocab_size = shift_logits.size(-1)

        hard_ce_loss = F.cross_entropy(
            shift_logits.float().view(-1, vocab_size),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        (
            numeric_token_ids,
            bin_centers,
            vocab_to_bin,
        ) = self._get_numeric_tensors(
            device=shift_logits.device,
            vocab_size=vocab_size,
        )

        valid_label_mask = shift_labels != -100

        safe_labels = shift_labels.masked_fill(
            ~valid_label_mask,
            0,
        )

        target_bin_ids = vocab_to_bin[safe_labels]

        numeric_mask = (
            valid_label_mask
            & (target_bin_ids >= 0)
        )

        # 数値正解トークンが含まれないバッチは通常CEだけを使用
        if not numeric_mask.any():
            if return_outputs:
                return hard_ce_loss, outputs

            return hard_ce_loss

        numeric_logits = shift_logits.index_select(
            dim=-1,
            index=numeric_token_ids,
        ).float()

        numeric_log_probs = F.log_softmax(
            numeric_logits,
            dim=-1,
        )
        numeric_probs = numeric_log_probs.exp()

        safe_target_bin_ids = target_bin_ids.clamp_min(0)
        target_values = bin_centers[safe_target_bin_ids]

        predicted_values = torch.sum(
            numeric_probs
            * bin_centers.view(1, 1, -1),
            dim=-1,
        )

        loss = hard_ce_loss

        wasserstein_loss = hard_ce_loss.new_zeros(())
        ordinal_ce_loss = hard_ce_loss.new_zeros(())
        soft_ce_loss = hard_ce_loss.new_zeros(())
        regression_loss = hard_ce_loss.new_zeros(())
        difference_loss = hard_ce_loss.new_zeros(())

        if self.wasserstein_weight > 0:
            wasserstein_loss = self._wasserstein_loss(
                numeric_probs=numeric_probs,
                target_bin_ids=target_bin_ids,
                bin_centers=bin_centers,
                numeric_mask=numeric_mask,
            )
            loss = (
                loss
                + self.wasserstein_weight
                * wasserstein_loss
            )

        if self.ordinal_ce_weight > 0:
            ordinal_ce_loss = self._ordinal_ce_loss(
                numeric_probs=numeric_probs,
                target_bin_ids=target_bin_ids,
                numeric_mask=numeric_mask,
            )
            loss = (
                loss
                + self.ordinal_ce_weight
                * ordinal_ce_loss
            )

        if self.soft_ce_weight > 0:
            soft_ce_loss = self._local_gaussian_soft_ce(
                numeric_log_probs=numeric_log_probs,
                target_bin_ids=target_bin_ids,
                numeric_mask=numeric_mask,
            )
            loss = (
                loss
                + self.soft_ce_weight
                * soft_ce_loss
            )

        if self.regression_weight > 0:
            regression_loss = self._expected_regression_loss(
                predicted_values=predicted_values,
                target_values=target_values,
                numeric_mask=numeric_mask,
            )
            loss = (
                loss
                + self.regression_weight
                * regression_loss
            )

        if self.difference_weight > 0:
            difference_loss = self._difference_loss(
                predicted_values=predicted_values,
                target_values=target_values,
                numeric_mask=numeric_mask,
            )
            loss = (
                loss
                + self.difference_weight
                * difference_loss
            )

        # W&BとTrainerログへ個別損失を記録
        if model.training:
            self.log(
                {
                    "numeric/hard_ce": hard_ce_loss.detach().float().item(),
                    "numeric/wasserstein": (
                        wasserstein_loss.detach().float().item()
                    ),
                    "numeric/ordinal_ce": (
                        ordinal_ce_loss.detach().float().item()
                    ),
                    "numeric/soft_ce": soft_ce_loss.detach().float().item(),
                    "numeric/regression": (
                        regression_loss.detach().float().item()
                    ),
                    "numeric/difference": (
                        difference_loss.detach().float().item()
                    ),
                    "numeric/target_ratio": (
                        numeric_mask.float().mean().item()
                    ),
                }
            )

        if return_outputs:
            return loss, outputs

        return loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # 必須パス
    parser.add_argument("--code_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--log_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)

    # モデルとトークナイズ
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)

    # 再現性
    parser.add_argument("--random_seed", type=int, default=3407)

    # 学習設定
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=16,
    )
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)

    # 時系列トークン設定
    parser.add_argument("--low_limit", type=float, default=-1)
    parser.add_argument("--high_limit", type=float, default=1)
    parser.add_argument("--n_tokens", type=int, default=10002)
    parser.add_argument("--prec", type=int, default=4)
    parser.add_argument("--time_sep", type=str, default=" ")
    parser.add_argument("--time_flag", type=str, default="###")
    parser.add_argument("--nan_flag", type=str, default="Nan")

    # 数値補助損失
    parser.add_argument(
        "--wasserstein_weight",
        type=float,
        default=0.0,
        help="Wasserstein-1損失の重み。0で無効",
    )
    parser.add_argument(
        "--ordinal_ce_weight",
        type=float,
        default=0.0,
        help="Ordinal Cross-Entropyの重み。0で無効",
    )
    parser.add_argument(
        "--soft_ce_weight",
        type=float,
        default=0.0,
        help="Gaussian soft-label CEの重み。0で無効",
    )
    parser.add_argument(
        "--regression_weight",
        type=float,
        default=0.0,
        help="期待値回帰損失の重み。0で無効",
    )
    parser.add_argument(
        "--difference_weight",
        type=float,
        default=0.0,
        help="一次difference lossの重み。0で無効",
    )
    parser.add_argument(
        "--soft_label_sigma_bins",
        type=float,
        default=1.5,
        help="Gaussian soft labelの標準偏差。ビン単位",
    )
    parser.add_argument(
        "--soft_label_radius",
        type=int,
        default=4,
        help="Gaussian soft labelを与える左右のビン数",
    )
    parser.add_argument(
        "--numeric_loss_type",
        type=str,
        default="smooth_l1",
        choices=["smooth_l1", "mae", "mse"],
        help="回帰損失とdifference lossの形式",
    )
    parser.add_argument(
        "--ordinal_chunk_size",
        type=int,
        default=1024,
        help="Ordinal CEのビン方向チャンクサイズ",
    )

    # Weights & Biases
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="mamba-continuous-pretraining",
    )
    parser.add_argument("--run_name", type=str, default=None)

    # データセット
    parser.add_argument(
        "--dataset_file",
        type=str,
        default="ChatTime-1-Pretrain-1M.csv",
    )
    parser.add_argument("--train_file", type=str, default=None)
    parser.add_argument("--validation_file", type=str, default=None)
    parser.add_argument("--test_file", type=str, default=None)
    parser.add_argument(
        "--dataset_num_proc",
        type=int,
        default=8,
        help="データセット前処理のプロセス数",
    )

    # 追加トークンだけを更新するオプション
    parser.add_argument(
        "--train_new_tokens_only",
        action="store_true",
        default=False,
        help="EmbeddingとLM Headの追加トークン行だけを更新する",
    )

    args = parser.parse_args()

    # 引数検証
    loss_weights = {
        "wasserstein_weight": args.wasserstein_weight,
        "ordinal_ce_weight": args.ordinal_ce_weight,
        "soft_ce_weight": args.soft_ce_weight,
        "regression_weight": args.regression_weight,
        "difference_weight": args.difference_weight,
    }

    for name, value in loss_weights.items():
        if value < 0:
            raise ValueError(f"{name}は0以上で指定してください。")

    os.environ.setdefault("WANDB_PROJECT", args.wandb_project)

    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)
        torch.cuda.set_device(LOCAL_RANK)

    sys.path.append(args.code_path)
    from utils.tools import Discretizer, Serializer

    # =========================================================
    # 1. 時系列用語彙の構築
    # =========================================================
    discretizer = Discretizer(
        low_limit=args.low_limit,
        high_limit=args.high_limit,
        n_tokens=args.n_tokens,
    )

    serializer = Serializer(
        prec=args.prec,
        time_sep=args.time_sep,
        time_flag=args.time_flag,
        nan_flag=args.nan_flag,
    )

    vocabulary_values = np.concatenate(
        (discretizer.centers[1:-1], [np.nan])
    ).reshape(-1, 1)

    vocabulary = np.array(
        [
            serializer.serialize(value)
            for value in vocabulary_values
        ]
    )

    log0(f"\nVocabulary size to add: {len(vocabulary):,}")
    log0(f"Vocabulary sample:\n{vocabulary[:10]}\n")

    # =========================================================
    # 2. Expanded Tokenizerの構築
    # =========================================================
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise RuntimeError(
                "tokenizerにpad_tokenとeos_tokenの両方がありません。"
            )

        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    original_vocab_size = len(tokenizer)

    log0(f"Original tokenizer size: {original_vocab_size:,}")

    numeric_token_strings = vocabulary.tolist()
    added_token_count = tokenizer.add_tokens(
        numeric_token_strings
    )
    expanded_vocab_size = len(tokenizer)

    log0(f"Added token count:       {added_token_count:,}")
    log0(f"Expanded tokenizer size: {expanded_vocab_size:,}\n")

    EOS_TOKEN = tokenizer.eos_token

    if EOS_TOKEN is None:
        raise RuntimeError("EOSトークンが設定されていません。")

    # =========================================================
    # 3. 数値トークンIDとビン中心の対応を構築
    # =========================================================
    numeric_token_ids = tokenizer.convert_tokens_to_ids(
        numeric_token_strings
    )

    if len(numeric_token_ids) != len(numeric_token_strings):
        raise RuntimeError(
            "数値トークン文字列とtoken IDの個数が一致しません。"
        )

    unknown_id = tokenizer.unk_token_id

    if unknown_id is not None:
        unknown_tokens = [
            token
            for token, token_id in zip(
                numeric_token_strings,
                numeric_token_ids,
            )
            if token_id == unknown_id
        ]

        if unknown_tokens:
            raise RuntimeError(
                "数値トークンの一部がUNKへ変換されました。"
                f"例: {unknown_tokens[:5]}"
            )

    numeric_values = vocabulary_values.reshape(-1)
    finite_numeric_mask = np.isfinite(numeric_values)

    finite_numeric_token_ids = np.asarray(
        numeric_token_ids,
        dtype=np.int64,
    )[finite_numeric_mask]

    finite_bin_centers = numeric_values[
        finite_numeric_mask
    ].astype(np.float32)

    nan_numeric_token_ids = np.asarray(
        numeric_token_ids,
        dtype=np.int64,
    )[~finite_numeric_mask]

    sort_indices = np.argsort(finite_bin_centers)
    finite_bin_centers = finite_bin_centers[sort_indices]
    finite_numeric_token_ids = finite_numeric_token_ids[sort_indices]

    if len(np.unique(finite_numeric_token_ids)) != len(
        finite_numeric_token_ids
    ):
        raise RuntimeError(
            "複数の数値文字列が同じtoken IDへ割り当てられています。"
            "precまたはn_tokensを確認してください。"
        )

    if not np.all(np.diff(finite_bin_centers) > 0):
        raise RuntimeError(
            "有限数値ビン中心が厳密な昇順ではありません。"
        )

    log0(
        f"Finite numeric tokens: {len(finite_numeric_token_ids):,}\n"
        f"NaN numeric tokens:    {len(nan_numeric_token_ids):,}\n"
        f"Minimum bin center:    {finite_bin_centers[0]}\n"
        f"Maximum bin center:    {finite_bin_centers[-1]}\n"
    )

    # =========================================================
    # 4. モデルを担当GPUへ読み込む
    # =========================================================
    model_kwargs = {
        "trust_remote_code": True,
        "device_map": {"": LOCAL_RANK},
    }

    if args.load_in_4bit:
        log0(
            "[WARNING] 4bit量子化でEmbeddingとLM Headを直接学習する構成は、"
            "モデル実装によって動作しない場合があります。"
        )

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=(
                torch.bfloat16
                if is_bfloat16_supported()
                else torch.float16
            ),
        )

        model_kwargs["quantization_config"] = bnb_config
    else:
        model_kwargs["torch_dtype"] = (
            torch.bfloat16
            if is_bfloat16_supported()
            else torch.float16
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        **model_kwargs,
    )

    model.resize_token_embeddings(expanded_vocab_size)

    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()

    # =========================================================
    # 5. 学習対象の設定
    # =========================================================
    for parameter in model.parameters():
        parameter.requires_grad = False

    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()

    if input_embeddings is None:
        raise RuntimeError(
            "model.get_input_embeddings()でEmbedding層を取得できません。"
        )

    if output_embeddings is None:
        raise RuntimeError(
            "model.get_output_embeddings()でLM Headを取得できません。"
        )

    for parameter in input_embeddings.parameters():
        parameter.requires_grad = True

    for parameter in output_embeddings.parameters():
        parameter.requires_grad = True

    input_weight = input_embeddings.weight
    output_weight = output_embeddings.weight

    tied_embeddings = (
        input_weight.data_ptr()
        == output_weight.data_ptr()
    )

    log0(f"Input/output weights tied: {tied_embeddings}")
    log0(f"Input embedding shape:  {tuple(input_weight.shape)}")
    log0(f"Output LM Head shape:   {tuple(output_weight.shape)}")

    if hasattr(model.config, "tie_word_embeddings"):
        log0(
            "config.tie_word_embeddings: "
            f"{model.config.tie_word_embeddings}"
        )

    # =========================================================
    # 6. 必要に応じて追加トークン行だけを更新
    # =========================================================
    gradient_hook_handles = []

    if args.train_new_tokens_only:
        if added_token_count == 0:
            raise RuntimeError(
                "--train_new_tokens_onlyが指定されましたが、"
                "新しいトークンが追加されていません。"
            )

        if args.weight_decay != 0:
            log0(
                "[WARNING] train_new_tokens_only使用時もAdamWのweight decayは"
                "既存行へ作用する可能性があります。完全固定には"
                "--weight_decay 0を推奨します。"
            )

        def mask_original_token_gradients(gradient):
            masked_gradient = gradient.clone()
            masked_gradient[:original_vocab_size].zero_()
            return masked_gradient

        gradient_hook_handles.append(
            input_weight.register_hook(
                mask_original_token_gradients
            )
        )

        if not tied_embeddings:
            gradient_hook_handles.append(
                output_weight.register_hook(
                    mask_original_token_gradients
                )
            )

        log0(
            "\nTraining mode: new token rows only\n"
            f"Frozen token rows:    0 to {original_vocab_size - 1}\n"
            f"Trainable token rows: {original_vocab_size} "
            f"to {expanded_vocab_size - 1}\n"
        )
    else:
        log0(
            "\nTraining mode: full Embedding and full LM Head\n"
        )

    print_trainable_parameters(model)

    if not any(
        parameter.requires_grad
        for parameter in model.parameters()
    ):
        raise RuntimeError(
            "学習可能なパラメータがありません。"
        )

    # =========================================================
    # 7. データセット読み込み
    # =========================================================
    def formatting_func(example):
        return add_eos_to_example(example, EOS_TOKEN)

    if args.train_file is not None:
        train_file = args.train_file
    else:
        train_file = (
            "https://huggingface.co/datasets/"
            f"{args.dataset_path}/resolve/main/"
            f"{args.dataset_file}"
        )

    data_files = {"train": train_file}

    if args.validation_file is not None:
        data_files["validation"] = args.validation_file

    if args.test_file is not None:
        data_files["test"] = args.test_file

    log0(f"\nLoading dataset files: {data_files}")

    dataset_dict = load_dataset(
        "csv",
        data_files=data_files,
    )

    dataset = dataset_dict["train"]

    if "text" not in dataset.column_names:
        raise KeyError(
            "データセットに'text'列がありません。"
            f"利用可能な列: {dataset.column_names}"
        )

    if len(dataset) == 0:
        raise RuntimeError("学習データセットが空です。")

    log0(f"Dataset size: {len(dataset):,}")
    log0(f"Dataset columns: {dataset.column_names}")
    log0(f"Dataset features: {dataset.features}")
    log0(f"Text type: {type(dataset[0]['text'])}")
    log0(f"Dataset example:\n{dataset[0]['text']}\n")

    # =========================================================
    # 8. SFTConfig
    # =========================================================
    available_cpu_count = os.cpu_count() or 1

    dataset_num_proc = max(
        1,
        min(args.dataset_num_proc, available_cpu_count),
    )

    sft_config_kwargs = {
        "dataset_text_field": "text",
        "dataset_num_proc": dataset_num_proc,
        "packing": False,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_train_epochs": args.num_train_epochs,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "max_grad_norm": 1.0,
        "learning_rate": args.learning_rate,
        "logging_strategy": "steps",
        "logging_steps": args.logging_steps,
        "save_strategy": "steps",
        "save_steps": args.save_steps,
        "max_steps": args.max_steps,
        "save_total_limit": 1,
        "logging_first_step": True,
        "optim": (
            "paged_adamw_8bit"
            if args.load_in_4bit
            else "adamw_torch"
        ),
        "lr_scheduler_type": "cosine",
        "seed": args.random_seed,
        "data_seed": args.random_seed,
        "output_dir": args.log_path,
        "fp16": (
            torch.cuda.is_available()
            and not is_bfloat16_supported()
        ),
        "bf16": is_bfloat16_supported(),
        "gradient_checkpointing": False,
        "report_to": "wandb",
        "run_name": args.run_name,
        "ddp_find_unused_parameters": False,
    }

    sft_config_parameters = inspect.signature(
        SFTConfig.__init__
    ).parameters

    if "max_length" in sft_config_parameters:
        sft_config_kwargs["max_length"] = args.max_seq_length
    elif "max_seq_length" in sft_config_parameters:
        sft_config_kwargs["max_seq_length"] = args.max_seq_length
    else:
        log0(
            "[WARNING] SFTConfigにmax_lengthまたは"
            "max_seq_lengthがありません。"
        )

    sft_config = SFTConfig(**sft_config_kwargs)

    # =========================================================
    # 9. カスタムSFTTrainer
    # =========================================================
    trainer_kwargs = {
        "model": model,
        "train_dataset": dataset,
        "formatting_func": formatting_func,
        "args": sft_config,
    }

    trainer_parameters = inspect.signature(
        SFTTrainer.__init__
    ).parameters

    if "processing_class" in trainer_parameters:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_parameters:
        trainer_kwargs["tokenizer"] = tokenizer
    else:
        raise RuntimeError(
            "SFTTrainerにprocessing_classまたはtokenizer引数がありません。"
        )

    trainer = NumericLossSFTTrainer(
        **trainer_kwargs,
        numeric_token_ids=finite_numeric_token_ids,
        numeric_bin_centers=finite_bin_centers,
        wasserstein_weight=args.wasserstein_weight,
        ordinal_ce_weight=args.ordinal_ce_weight,
        soft_ce_weight=args.soft_ce_weight,
        regression_weight=args.regression_weight,
        difference_weight=args.difference_weight,
        soft_label_sigma_bins=args.soft_label_sigma_bins,
        soft_label_radius=args.soft_label_radius,
        numeric_loss_type=args.numeric_loss_type,
        ordinal_chunk_size=args.ordinal_chunk_size,
    )

    log0(
        "\n=== Numeric loss configuration ===\n"
        f"Hard CE weight:       1.0\n"
        f"Wasserstein weight:  {args.wasserstein_weight}\n"
        f"Ordinal CE weight:   {args.ordinal_ce_weight}\n"
        f"Soft CE weight:      {args.soft_ce_weight}\n"
        f"Regression weight:   {args.regression_weight}\n"
        f"Difference weight:   {args.difference_weight}\n"
        f"Soft-label sigma:    {args.soft_label_sigma_bins} bins\n"
        f"Soft-label radius:   {args.soft_label_radius} bins\n"
        f"Numeric loss type:   {args.numeric_loss_type}\n"
    )

    # =========================================================
    # 10. 学習開始前のGPUメモリ情報
    # =========================================================
    start_gpu_memory = 0.0
    max_gpu_memory = 0.0

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(LOCAL_RANK)
        gpu_stats = torch.cuda.get_device_properties(LOCAL_RANK)

        start_gpu_memory = round(
            torch.cuda.memory_reserved(LOCAL_RANK) / 1024**3,
            3,
        )
        max_gpu_memory = round(
            gpu_stats.total_memory / 1024**3,
            3,
        )

        log0(
            f"\nGPU: {gpu_stats.name}\n"
            f"World size: {WORLD_SIZE}\n"
            f"Total GPU memory: {max_gpu_memory} GB\n"
            f"Initially reserved memory: {start_gpu_memory} GB\n"
        )

    # =========================================================
    # 11. 学習
    # =========================================================
    trainer_stats = trainer.train()

    # =========================================================
    # 12. 学習終了後のGPUメモリ情報
    # =========================================================
    if torch.cuda.is_available():
        peak_reserved_memory = round(
            torch.cuda.max_memory_reserved(LOCAL_RANK) / 1024**3,
            3,
        )

        additional_training_memory = round(
            peak_reserved_memory - start_gpu_memory,
            3,
        )

        peak_memory_percentage = (
            round(
                peak_reserved_memory / max_gpu_memory * 100,
                3,
            )
            if max_gpu_memory > 0
            else 0.0
        )

        additional_memory_percentage = (
            round(
                additional_training_memory / max_gpu_memory * 100,
                3,
            )
            if max_gpu_memory > 0
            else 0.0
        )

        training_runtime = trainer_stats.metrics.get(
            "train_runtime",
            0.0,
        )

        log0(
            f"\nTraining runtime: {training_runtime} seconds\n"
            f"Training runtime: {round(training_runtime / 60, 2)} minutes\n"
            f"Peak reserved memory: {peak_reserved_memory} GB\n"
            f"Additional reserved memory: {additional_training_memory} GB\n"
            f"Peak memory usage: {peak_memory_percentage}%\n"
            f"Additional memory usage: {additional_memory_percentage}%\n"
        )

    # =========================================================
    # 13. モデルとTokenizerの保存
    # =========================================================
    if (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
    ):
        torch.distributed.barrier()

    if IS_MAIN_PROCESS:
        os.makedirs(args.output_path, exist_ok=True)

        trainer.save_model(args.output_path)
        tokenizer.save_pretrained(args.output_path)

        log0(f"\nModel saved to {args.output_path}")

    if (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
    ):
        torch.distributed.barrier()