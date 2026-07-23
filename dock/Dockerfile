# syntax=docker/dockerfile:1

FROM pytorch/pytorch:2.7.1-cuda11.8-cudnn9-devel

ARG USERNAME=chattime
ARG USER_UID=1000
ARG USER_GID=1000

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/home/chattime \
    HF_HOME=/home/chattime/.cache/huggingface \
    HF_HUB_CACHE=/home/chattime/.cache/huggingface/hub \
    XDG_CACHE_HOME=/home/chattime/.cache \
    TORCH_HOME=/home/chattime/.cache/torch \
    TORCH_EXTENSIONS_DIR=/home/chattime/.cache/torch_extensions \
    TRITON_CACHE_DIR=/home/chattime/.cache/triton \
    NUMBA_CACHE_DIR=/home/chattime/.cache/numba \
    TOKENIZERS_PARALLELISM=false

# ------------------------------------------------------------
# OSパッケージ
# ------------------------------------------------------------
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git \
        wget \
        curl \
        build-essential \
        ninja-build \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------
# 非rootユーザー
# ------------------------------------------------------------
RUN groupadd --gid "${USER_GID}" "${USERNAME}" \
    && useradd \
        --uid "${USER_UID}" \
        --gid "${USER_GID}" \
        --create-home \
        --shell /bin/bash \
        "${USERNAME}"

# ------------------------------------------------------------
# Python依存関係
# ------------------------------------------------------------
COPY requirements.docker.txt /tmp/requirements.docker.txt
COPY constraints.docker.txt /tmp/constraints.docker.txt

RUN python -m pip install --upgrade pip setuptools wheel

RUN set -eux; \
    test -f /tmp/requirements.docker.txt; \
    test -f /tmp/constraints.docker.txt; \
    echo "===== requirements.docker.txt ====="; \
    cat /tmp/requirements.docker.txt; \
    echo "===== constraints.docker.txt ====="; \
    cat /tmp/constraints.docker.txt; \
    python -m pip install \
        --no-cache-dir \
        --constraint /tmp/constraints.docker.txt \
        --extra-index-url https://download.pytorch.org/whl/cu118 \
        -r /tmp/requirements.docker.txt; \
    python -m pip check

# ------------------------------------------------------------
# パッケージバージョンの検査
# ------------------------------------------------------------
RUN python - <<'PY'
from importlib.metadata import version
import torch

expected = {
    "torchvision": "0.22.1+cu118",
    "torchaudio": "2.7.1+cu118",
    "triton": "3.3.1",
    "xformers": "0.0.31",
    "datasets": "3.6.0",
    "fsspec": "2025.3.0",
    "transformers": "4.56.1",
    "peft": "0.18.0",
    "trl": "0.23.1",
    "unsloth": "2026.6.9",
    "unsloth-zoo": "2026.6.7",
}

print("torch:", torch.__version__)
print("torch CUDA build:", torch.version.cuda)

assert torch.__version__ == "2.7.1+cu118", torch.__version__
assert torch.version.cuda == "11.8", torch.version.cuda

for package, expected_version in expected.items():
    actual_version = version(package)
    print(f"{package}: {actual_version}")
    assert actual_version == expected_version, (
        f"{package}: expected {expected_version}, "
        f"found {actual_version}"
    )

print("Dependency verification: OK")
PY

# ------------------------------------------------------------
# 書き込み対象ディレクトリ
# ------------------------------------------------------------
RUN mkdir -p \
        /workspace/ChatTime \
        /workspace/output \
        /workspace/checkpoints \
        /workspace/models \
        /workspace/logs \
        /home/${USERNAME}/.cache/huggingface/hub \
        /home/${USERNAME}/.cache/torch \
        /home/${USERNAME}/.cache/torch_extensions \
        /home/${USERNAME}/.cache/triton \
        /home/${USERNAME}/.cache/numba \
        /home/${USERNAME}/.config \
        /home/${USERNAME}/.nv \
    && chown -R "${USER_UID}:${USER_GID}" \
        /workspace/ChatTime \
        /workspace/output \
        /workspace/checkpoints \
        /workspace/models \
        /workspace/logs \
        /home/${USERNAME}

WORKDIR /workspace/ChatTime

# ------------------------------------------------------------
# ChatTimeコード
# ------------------------------------------------------------
COPY --chown=${USER_UID}:${USER_GID} . /workspace/ChatTime

# ------------------------------------------------------------
# 最終実行ユーザー
# ------------------------------------------------------------
USER ${USERNAME}:${USERNAME}

# ------------------------------------------------------------
# 非root・書き込み検査
# ------------------------------------------------------------
RUN python - <<'PY'
import os
from pathlib import Path

print("UID:", os.getuid())
print("GID:", os.getgid())
print("HOME:", os.environ["HOME"])

assert os.getuid() != 0

paths = [
    Path.home(),
    Path("/workspace/ChatTime"),
    Path(os.environ["HF_HOME"]),
    Path(os.environ["TORCH_EXTENSIONS_DIR"]),
    Path(os.environ["TRITON_CACHE_DIR"]),
]

for path in paths:
    path.mkdir(parents=True, exist_ok=True)

    test_file = path / ".write_test"
    test_file.write_text("ok", encoding="utf-8")
    test_file.unlink()

    print("Writable:", path)

print("Non-root write verification: OK")
PY

CMD ["/bin/bash"]