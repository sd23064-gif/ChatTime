FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

ENV HF_HOME=/workspace/.cache/huggingface
ENV TORCHINDUCTOR_CACHE_DIR=/workspace/.cache/torchinductor
ENV WANDB_DIR=/workspace/.cache/wandb

ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=/opt/conda/lib/python3.10/site-packages/torch/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

ENV MAX_JOBS=4

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

RUN pip install --upgrade pip setuptools wheel packaging ninja

RUN python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.version.cuda)"

RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    pandas==2.2.2 \
    scikit-learn==1.4.2 \
    scipy==1.13.1 \
    tqdm==4.66.4 \
    matplotlib==3.8.4 \
    einops==0.8.0 \
    wandb==0.17.0 \
    wfdb \
    rich==13.7.1 \
    transformers==4.40.2 \
    datasets==2.19.1 \
    huggingface_hub==0.23.2 \
    accelerate==0.30.1 \
    peft==0.11.1 \
    trl==0.8.6 \
    safetensors==0.4.3 \
    sentencepiece==0.2.0 \
    protobuf==4.25.3 \
    bitsandbytes==0.43.1

# 重要: 最新版を入れない
RUN pip uninstall -y causal-conv1d mamba-ssm || true

RUN pip install --no-cache-dir --no-build-isolation --no-deps \
    "causal-conv1d==1.2.2.post1"

RUN pip install --no-cache-dir --no-build-isolation --no-deps \
    "mamba-ssm==2.0.4"

RUN python -c "import torch; import transformers; import trl; import peft; print('basic import OK')"
RUN python -c "import causal_conv1d; print('causal_conv1d OK')"
RUN python -c "import mamba_ssm; print('mamba_ssm OK')"

COPY . /workspace

CMD ["/bin/bash"]