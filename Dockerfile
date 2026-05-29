FROM pytorch/pytorch:2.7.1-cuda11.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

ENV HF_HOME=/workspace/.cache/huggingface
ENV TRANSFORMERS_CACHE=/workspace/.cache/huggingface
ENV TORCHINDUCTOR_CACHE_DIR=/workspace/.cache/torchinductor
ENV WANDB_DIR=/workspace/.cache/wandb

WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    build-essential \
    ninja-build \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /workspace/requirements.txt

RUN pip install --upgrade pip setuptools wheel

RUN pip install --no-cache-dir -r requirements.txt

COPY . /workspace

CMD ["/bin/bash"]