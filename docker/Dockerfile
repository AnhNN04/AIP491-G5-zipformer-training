FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel
# Python 3.11

ENV LC_ALL=C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive

# fix segmentation fault with protobuf
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# ── Version pins (CUDA 12.4 + Torch 2.5.1) ────────────────────────────────────
# Wheels verified at:
#   https://k2-fsa.github.io/k2/cuda.html
#   https://csukuangfj.github.io/kaldifeat/cuda.html
ARG K2_VERSION="1.24.4.dev20250715+cuda12.4.torch2.5.1"
ARG KALDIFEAT_VERSION="1.25.5.dev20260626+cuda12.4.torch2.5.1"
ARG TORCHAUDIO_VERSION="2.5.1+cu124"

LABEL authors="VietASR Team"
LABEL k2_version=${K2_VERSION}
LABEL kaldifeat_version=${KALDIFEAT_VERSION}
LABEL github_repo="https://github.com/k2-fsa/icefall"

# ── Layer 1: System packages ───────────────────────────────────────────────────
# Cached independently of Python packages — changes here only rebuild from this layer.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        vim \
        tmux \
        htop \
        libssl-dev \
        autoconf \
        automake \
        bzip2 \
        ca-certificates \
        cmake \
        ffmpeg \
        g++ \
        gfortran \
        git \
        libtool \
        make \
        patch \
        sox \
        subversion \
        unzip \
        valgrind \
        wget \
        zsh \
        zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# ── Layer 2: Core Python dependencies ─────────────────────────────────────────
# Heavy packages (k2, kaldifeat, torchaudio) are installed first so they are
# cached even if lighter packages change.
RUN pip install --no-cache-dir \
      torchaudio==${TORCHAUDIO_VERSION} -f https://download.pytorch.org/whl/torchaudio/ \
      k2==${K2_VERSION} -f https://k2-fsa.github.io/k2/cuda.html \
      git+https://github.com/lhotse-speech/lhotse \
      kaldifeat==${KALDIFEAT_VERSION} -f https://csukuangfj.github.io/kaldifeat/cuda.html \
      kaldi_native_io \
      kaldialign \
      kaldifst \
      kaldilm \
      "sentencepiece>=0.1.96" \
      tensorboard \
      typeguard \
      dill \
      onnx \
      onnxruntime \
      onnxmltools \
      onnxoptimizer \
      onnxsim \
      multi_quantization \
      numpy \
      pytest \
      wandb \
      transformers \
      graphviz

# ── Layer 3: icefall framework ─────────────────────────────────────────────────
# Cloned to /opt/icefall so the project repo can mount cleanly at /workspace.
RUN git clone https://github.com/k2-fsa/icefall /opt/icefall && \
    cd /opt/icefall && \
    pip install --no-cache-dir -r requirements.txt

# ── Layer 4: funasr VAD model ──────────────────────────────────────────────────
# Downloaded at build time to avoid slow cold-start on first run.
RUN pip3 install --no-cache-dir -U funasr modelscope && \
    python -c 'from funasr import AutoModel; AutoModel(model="fsmn-vad", model_revision="v2.0.4")'

# ── Runtime environment ────────────────────────────────────────────────────────
# /opt/icefall            = upstream icefall framework (read-only at runtime)
# /workspace              = project repo (mounted read-write at runtime)
# /workspace/external/icefall = local icefall overrides (takes precedence)
ENV PYTHONPATH=/opt/icefall:/workspace:/workspace/external/icefall:$PYTHONPATH

# Project repo is mounted here at docker run time
WORKDIR /workspace