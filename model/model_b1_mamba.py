import re
from pathlib import Path

import numpy as np

# NumPy 2.x 対策
if not hasattr(np, "NaN"):
    np.NaN = np.nan

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel
except ImportError:
    PeftModel = None

from utils.tools import Discretizer, Serializer


class ChatTimeB1Mamba:
    """
    B-1 継続事前学習済み Mamba / CausalLM 用の時系列予測クラス。

    このクラスは、ChatTime B-1 の学習形式に合わせて、
    履歴時系列を

        raw series
        -> Discretizer.discretize()
        -> Serializer.serialize()
        -> ###数値### token列

    に変換し、その続きを Mamba 系 CausalLM に生成させる。

    対応する読み込み方式:
        1. merged/full model:
            model_path を指定

        2. LoRA adapter:
            base_model_path と adapter_path を指定
    """

    def __init__(
        self,
        model_path=None,
        base_model_path=None,
        adapter_path=None,
        hist_len=None,
        pred_len=None,
        max_pred_len=16,
        num_samples=1,
        top_k=50,
        top_p=0.9,
        temperature=0.7,
        low_limit=-1,
        high_limit=1,
        n_tokens=10002,
        prec=4,
        time_sep=" ",
        time_flag="###",
        nan_flag="Nan",
        local_files_only=True,
        use_bf16_if_available=True,
        debug=False,
    ):
        self.model_path = model_path
        self.base_model_path = base_model_path
        self.adapter_path = adapter_path

        self.hist_len = hist_len
        self.pred_len = pred_len
        self.max_pred_len = max_pred_len

        self.num_samples = num_samples
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature

        self.local_files_only = local_files_only
        self.debug = debug

        self.discretizer = Discretizer(
            low_limit=low_limit,
            high_limit=high_limit,
            n_tokens=n_tokens,
        )

        self.serializer = Serializer(
            prec=prec,
            time_sep=time_sep,
            time_flag=time_flag,
            nan_flag=nan_flag,
        )

        self.device_available = torch.cuda.is_available()

        self.dtype = self._decide_dtype(
            use_bf16_if_available=use_bf16_if_available
        )

        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.tokenizer.padding_side = "right"

        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id

        self.model.eval()

        print("=== ChatTimeB1Mamba loaded ===")
        print("model_path      :", self.model_path)
        print("base_model_path :", self.base_model_path)
        print("adapter_path    :", self.adapter_path)
        print("dtype           :", self.dtype)
        print("cuda available  :", torch.cuda.is_available())
        print("tokenizer size  :", len(self.tokenizer))
        print("==============================")

    # ------------------------------------------------------------
    # 基本ユーティリティ
    # ------------------------------------------------------------
    def _resolve_if_local(self, path):
        """
        ローカルパスが存在する場合は絶対パスへ変換する。
        存在しない場合は Hugging Face repo id として扱えるようそのまま返す。
        """
        if path is None:
            return None

        p = Path(path).expanduser()

        if p.exists():
            return str(p.resolve())

        return path

    def _decide_dtype(self, use_bf16_if_available=True):
        """
        GPU環境に応じて dtype を決める。
        """
        use_cuda = torch.cuda.is_available()
        use_bf16 = (
            use_cuda
            and use_bf16_if_available
            and torch.cuda.is_bf16_supported()
        )

        if use_bf16:
            return torch.bfloat16

        if use_cuda:
            return torch.float16

        return torch.float32

    def _get_model_device(self):
        """
        model.generate 用に、modelが載っているdeviceを取得する。
        """
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------
    # モデル・tokenizer読み込み
    # ------------------------------------------------------------
    def _get_tokenizer_path(self):
        """
        tokenizerは、追加した ###数値### token を含む保存先から読む必要がある。

        adapter_path がある場合:
            adapter_path に tokenizer.save_pretrained() されている想定。

        merged model の場合:
            model_path に tokenizer がある想定。
        """
        if self.adapter_path is not None:
            return self._resolve_if_local(self.adapter_path)

        if self.model_path is not None:
            return self._resolve_if_local(self.model_path)

        if self.base_model_path is not None:
            return self._resolve_if_local(self.base_model_path)

        raise ValueError(
            "Tokenizer path could not be determined. "
            "Specify model_path or adapter_path/base_model_path."
        )

    def _load_tokenizer(self):
        tokenizer_path = self._get_tokenizer_path()

        print("Loading tokenizer from:", tokenizer_path)

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
            use_fast=True,
            local_files_only=self.local_files_only,
        )

        return tokenizer

    def _load_model(self):
        """
        2種類の読み込みに対応する。

        1. merged/full model:
            model_path を AutoModelForCausalLM.from_pretrained() で読む。

        2. LoRA adapter:
            base_model_path でbase modelを読み、
            adapter_path を PeftModel.from_pretrained() で重ねる。
        """
        use_cuda = torch.cuda.is_available()

        if self.adapter_path is not None:
            if PeftModel is None:
                raise ImportError(
                    "peft is required to load adapter_path. "
                    "Please install peft."
                )

            if self.base_model_path is None:
                raise ValueError(
                    "adapter_path を指定する場合は base_model_path も必要です。"
                )

            base_model_path = self._resolve_if_local(self.base_model_path)
            adapter_path = self._resolve_if_local(self.adapter_path)

            print("Loading base Mamba/CausalLM model:", base_model_path)

            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype=self.dtype,
                device_map="auto" if use_cuda else None,
                trust_remote_code=True,
                local_files_only=self.local_files_only,
            )

            print("Loading LoRA adapter:", adapter_path)

            model = PeftModel.from_pretrained(
                base_model,
                adapter_path,
                local_files_only=self.local_files_only,
            )

            return model

        if self.model_path is None:
            raise ValueError(
                "Either model_path or adapter_path must be specified."
            )

        model_path = self._resolve_if_local(self.model_path)

        print("Loading merged/full Mamba/CausalLM model:", model_path)

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=self.dtype,
            device_map="auto" if use_cuda else None,
            trust_remote_code=True,
            local_files_only=self.local_files_only,
        )

        return model

    # ------------------------------------------------------------
    # ChatTime形式への変換
    # ------------------------------------------------------------
    def serialize_history(self, hist_data):
        """
        rawの時系列を ChatTime token列へ変換する。

        既存ChatTimeと同じく:
            Discretizer.discretize()
            -> Serializer.serialize()
        を使う。
        """
        hist_data = np.asarray(hist_data, dtype=np.float32).reshape(-1)

        dispersed = self.discretizer.discretize(hist_data)
        serialized = self.serializer.serialize(dispersed)

        return serialized

    def robust_parse_chattime_values(self, text):
        """
        生成されたテキストから ###数値### tokenを堅牢に抽出する。

        通常:
            ###-0.1234### ###0.1111###

        スペースなし:
            ###-0.1234######0.1111###

        の両方に対応する。
        """
        pattern = r"###(Nan|[-+]?\d+(?:\.\d+)?)###"
        matches = re.findall(pattern, str(text))

        values = []

        for m in matches:
            if m == "Nan":
                values.append(np.nan)
            else:
                try:
                    values.append(float(m))
                except ValueError:
                    values.append(np.nan)

        return np.asarray(values, dtype=np.float32)

    # ------------------------------------------------------------
    # テキスト生成
    # ------------------------------------------------------------
    def generate_continuation_texts(self, prompt_text, target_len):
        """
        prompt_text の続きを生成する。

        B-1では getPrompt() や ### Response: は使わない。
        数値token列の続きを直接生成する。
        """
        device = self._get_model_device()

        inputs = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            add_special_tokens=False,
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[-1]

        do_sample = self.num_samples > 1 and self.temperature > 0

        max_new_tokens = 4 * target_len + 16
        min_new_tokens = target_len

        generation_kwargs = dict(
            **inputs,
            min_new_tokens=min_new_tokens,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            num_return_sequences=self.num_samples,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
        )

        if do_sample:
            generation_kwargs.update(
                top_k=self.top_k,
                top_p=self.top_p,
                temperature=self.temperature,
            )

        with torch.no_grad():
            output_ids = self.model.generate(**generation_kwargs)

        generated_texts = []

        for ids in output_ids:
            new_ids = ids[input_len:]

            text = self.tokenizer.decode(
                new_ids,
                skip_special_tokens=True,
            )

            generated_texts.append(text)

        return generated_texts

    # ------------------------------------------------------------
    # raw時系列予測
    # ------------------------------------------------------------
    def predict(self, hist_data):
        """
        raw ECGなどの実数時系列から未来値を予測する。

        戻り値:
            shape = (pred_len,)
            raw scaleに戻した予測値
        """
        if self.hist_len is None or self.pred_len is None:
            raise ValueError(
                "hist_len and pred_len must be specified for predict()."
            )

        series = np.asarray(hist_data, dtype=np.float32).reshape(-1)

        if len(series) != self.hist_len:
            raise ValueError(
                f"hist_data length must be {self.hist_len}, got {len(series)}"
            )

        prediction_chunks = []
        remaining = self.pred_len

        while remaining > 0:
            current_pred_len = min(remaining, self.max_pred_len)

            prompt_text = self.serialize_history(series)

            generated_texts = self.generate_continuation_texts(
                prompt_text=prompt_text,
                target_len=current_pred_len,
            )

            pred_list = []

            for generated_text in generated_texts:
                dispersed_pred = self.robust_parse_chattime_values(
                    generated_text
                )

                if self.debug:
                    print("\n=== prompt tail ===")
                    print(prompt_text[-500:])
                    print("\n=== generated text ===")
                    print(repr(generated_text[:1000]))
                    print("parsed count:", len(dispersed_pred))
                    print("parsed head :", dispersed_pred[:20])

                if len(dispersed_pred) == 0:
                    pred = np.full(
                        current_pred_len,
                        np.nan,
                        dtype=np.float32,
                    )
                else:
                    pred = self.discretizer.inverse_discretize(
                        dispersed_pred
                    )
                    pred = np.asarray(pred, dtype=np.float32).reshape(-1)
                    pred = pred[:current_pred_len]

                    if len(pred) < current_pred_len:
                        pad_len = current_pred_len - len(pred)
                        pred = np.concatenate(
                            [
                                pred,
                                np.full(
                                    pad_len,
                                    np.nan,
                                    dtype=np.float32,
                                ),
                            ]
                        )

                pred_list.append(pred)

            pred_arr = np.asarray(pred_list, dtype=np.float32)

            finite_count = np.isfinite(pred_arr).sum()

            if self.debug:
                print("pred_arr shape:", pred_arr.shape)
                print("finite count:", finite_count)

            if finite_count == 0:
                prediction = np.full(
                    current_pred_len,
                    np.nan,
                    dtype=np.float32,
                )
            else:
                prediction = np.nanmedian(
                    pred_arr,
                    axis=0,
                ).astype(np.float32)

            prediction_chunks.append(prediction)

            remaining -= len(prediction)

            if remaining <= 0:
                break

            series = np.concatenate([series, prediction], axis=-1)

        prediction = np.concatenate(prediction_chunks, axis=-1)

        return prediction[:self.pred_len]

    # ------------------------------------------------------------
    # 変換済みCSV用: 離散値空間での予測
    # ------------------------------------------------------------
    def predict_discretized(self, hist_values, pred_len):
        """
        すでに ###数値### から取り出した離散値列を入力し、
        離散値空間で未来を予測する。

        変換済みCSV評価用。
        inverse_discretize は使わない。
        """
        hist_values = np.asarray(hist_values, dtype=np.float32).reshape(-1)

        prompt_text = self.serializer.serialize(hist_values)

        generated_texts = self.generate_continuation_texts(
            prompt_text=prompt_text,
            target_len=pred_len,
        )

        pred_list = []

        for generated_text in generated_texts:
            pred = self.robust_parse_chattime_values(generated_text)

            if self.debug:
                print("\n=== discretized prompt tail ===")
                print(prompt_text[-500:])
                print("\n=== generated text ===")
                print(repr(generated_text[:1000]))
                print("parsed count:", len(pred))
                print("parsed head :", pred[:20])

            if len(pred) == 0:
                pred = np.full(pred_len, np.nan, dtype=np.float32)
            else:
                pred = pred[:pred_len]

                if len(pred) < pred_len:
                    pad_len = pred_len - len(pred)
                    pred = np.concatenate(
                        [
                            pred,
                            np.full(
                                pad_len,
                                np.nan,
                                dtype=np.float32,
                            ),
                        ]
                    )

            pred_list.append(pred.astype(np.float32))

        pred_arr = np.asarray(pred_list, dtype=np.float32)

        if np.isfinite(pred_arr).sum() == 0:
            return np.full(pred_len, np.nan, dtype=np.float32)

        return np.nanmedian(pred_arr, axis=0).astype(np.float32)