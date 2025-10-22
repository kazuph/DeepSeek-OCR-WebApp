# syntax=docker/dockerfile:1.7
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

ARG USER_NAME=ocr
ARG USER_UID=1000
ARG USER_GID=1000

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    TORCH_CUDA_ARCH_LIST=8.6 \
    HF_HOME=/workspace/.cache/huggingface \
    TOKENIZERS_PARALLELISM=false

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        git-lfs \
        curl \
        wget \
        libgl1 \
        libglib2.0-0 \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        pkg-config \
        poppler-utils \
        python3-dev \
        ninja-build \
        cmake \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel

COPY requirements.txt /tmp/requirements.txt

RUN pip install --extra-index-url https://download.pytorch.org/whl/cu124 \
        torch==2.6.0 \
        torchvision==0.21.0 \
        torchaudio==2.6.0 \
    && pip install --no-cache-dir -r /tmp/requirements.txt \
    && pip install --no-cache-dir flash-attn==2.7.3 --no-build-isolation \
    && pip install --no-cache-dir tqdm accelerate

RUN groupadd --gid ${USER_GID} ${USER_NAME} \
    && useradd --uid ${USER_UID} --gid ${USER_GID} --create-home ${USER_NAME}

RUN mkdir -p /workspace/.cache/huggingface \
    && chown -R ${USER_UID}:${USER_GID} /workspace

USER ${USER_NAME}

WORKDIR /workspace

ENV PYTHONPATH=/workspace/DeepSeek-OCR-master/DeepSeek-OCR-vllm:${PYTHONPATH}

CMD ["bash"]
