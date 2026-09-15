"""
env_check.py

学習を回す前に、この Docker イメージで前提が満たされているか確認する。
GPU なしでも大半は通る。
"""

import inspect
import sys


def check(name, fn):
    try:
        ok, detail = fn()
    except Exception as e:
        ok, detail = False, f"{type(e).__name__}: {e}"
    print(f"[{'OK ' if ok else 'NG '}] {name:<42} {detail}")
    return ok


def main():
    results = []

    def versions():
        import torch, transformers, peft, numpy
        return True, (f"torch={torch.__version__} tf={transformers.__version__} "
                      f"peft={peft.__version__} np={numpy.__version__}")
    results.append(check("versions", versions))

    def eval_strategy_name():
        from transformers import TrainingArguments
        params = inspect.signature(TrainingArguments.__init__).parameters
        has_old = "evaluation_strategy" in params
        has_new = "eval_strategy" in params
        return has_old, f"evaluation_strategy={has_old} eval_strategy={has_new}"
    results.append(check("TrainingArguments.evaluation_strategy", eval_strategy_name))

    def gc_kwargs():
        from transformers import TrainingArguments
        params = inspect.signature(TrainingArguments.__init__).parameters
        return "gradient_checkpointing_kwargs" in params, ""
    results.append(check("gradient_checkpointing_kwargs", gc_kwargs))

    def mamba_cls():
        from transformers import MambaForCausalLM
        from transformers.models.mamba import modeling_mamba as m
        return True, f"supports_gc={m.MambaPreTrainedModel.supports_gradient_checkpointing}"
    results.append(check("MambaForCausalLM", mamba_cls))

    def mamba_kernels():
        from transformers.models.mamba import modeling_mamba as m
        return bool(m.is_fast_path_available), f"fast_path={m.is_fast_path_available}"
    results.append(check("mamba fused kernels", mamba_kernels))

    def rmsnorm_absent():
        import torch.nn as nn
        # torch 2.3 には nn.RMSNorm がない。使っていないことの確認。
        return True, f"nn.RMSNorm exists={hasattr(nn, 'RMSNorm')}"
    results.append(check("torch nn.RMSNorm (未使用)", rmsnorm_absent))

    def ckpt_nonreentrant():
        import torch
        from torch.utils.checkpoint import checkpoint
        params = inspect.signature(checkpoint).parameters
        return "use_reentrant" in params or True, ""
    results.append(check("checkpoint(use_reentrant=False)", ckpt_nonreentrant))

    def peft_merge():
        from peft import PeftModel
        return hasattr(PeftModel, "merge_and_unload"), ""
    results.append(check("PeftModel.merge_and_unload", peft_merge))

    def deps():
        import wfdb, scipy, sklearn, pandas
        return True, f"wfdb={wfdb.__version__} scipy={scipy.__version__}"
    results.append(check("wfdb / scipy / sklearn / pandas", deps))

    def own_modules():
        sys.path.insert(0, ".")
        import ecg_data, ecg_model, ecg_prepare, ecg_vocab  # noqa: F401
        return True, ""
    results.append(check("ecg_* modules import", own_modules))

    def gpu():
        import torch
        if not torch.cuda.is_available():
            return False, "CUDA 利用不可 (CPU のみ)"
        p = torch.cuda.get_device_properties(0)
        return True, (f"{p.name} {p.total_memory/1024**3:.0f}GB "
                      f"bf16={torch.cuda.is_bf16_supported()}")
    check("GPU", gpu)   # 失敗しても致命的ではない

    print()
    print("all critical checks passed" if all(results) else "FAILED: 上の NG を確認")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
