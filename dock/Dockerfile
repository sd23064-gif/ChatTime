FROM pytorch/pytorch:2.7.1-cuda11.8-cudnn9-devel

WORKDIR /workspace/ChatTime

ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV HF_HOME=/workspace/.cache/huggingface
ENV TRANSFORMERS_CACHE=/workspace/.cache/huggingface
ENV TOKENIZERS_PARALLELISM=false

RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    build-essential \
    ninja-build \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt

RUN python -m pip install --upgrade pip setuptools wheel

RUN set -eux; \
    grep -v -E '^(torch|torchvision|torchaudio)==' /tmp/requirements.txt > /tmp/requirements_chattime_no_torch.txt; \
    echo "===== filtered requirements ====="; \
    cat /tmp/requirements_chattime_no_torch.txt; \
    echo "===== pip install start ====="; \
    python -m pip install -r /tmp/requirements_chattime_no_torch.txt

# 念のため、torch が cu128 などに置き換わった場合に cu118 へ戻す
RUN python -m pip install --force-reinstall --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu118 \
    torch==2.7.1+cu118 \
    torchvision==0.22.1+cu118 \
    torchaudio==2.7.1+cu118

RUN python - <<'PY'
import torch
print("torch version:", torch.__version__)
print("cuda version:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
PY

COPY . /workspace/ChatTime

CMD ["/bin/bash"]