# Capstone ASR Docker Setup & Environment Guide

This document describes how to build, run, and configure the Docker environment for training the Zipformer ASR system using GPU acceleration inside Windows Subsystem for Linux (WSL2).

---

## Prerequisites

To run GPU-accelerated training inside Docker on WSL2, your Windows host machine must meet the following requirements:

1. **Nvidia Drivers**: Ensure the latest Nvidia GeForce Game Ready or Studio drivers are installed on Windows.
2. **Docker Desktop**:
   * Install Docker Desktop for Windows.
   * Go to **Settings > General** and ensure **"Use the WSL 2 based engine"** is checked.
3. **NVIDIA Container Toolkit**: 
   * Inside WSL instance, test GPU access in Docker:
     ```bash
     docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
     ```
   * If this prints your GPU details, runtime is correctly configured. If it fails, install the NVIDIA Container Toolkit inside WSL.

---

## Step 1: Building the Docker Image

The repository contains a custom Dockerfile that packages all complex C++ and Python dependencies (`k2`, `kaldifeat`, `lhotse`, etc.) so don't need to compile manually.

Run the build command from the repository root:
```bash
docker build -t capstone-asr:latest -f docker/Dockerfile .
```

### What is packaged in this image?
* **Base OS**: PyTorch CUDA Devel Image (`pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel`) running Python 3.11/3.12.
* **Icefall Prerequisites**: Source-compiled libraries for `k2` and `kaldifeat` built specifically to support Blackwell architectures (`sm_120`), along with `kaldialign`, `kaldifst`, and `kaldilm`.
* **VAD Classifier**: Pre-cached FunASR `fsmn-vad` model (saves cold-start download times).
* **Project Requirements**: All dependencies from `pyproject.toml` (`rich`, `matplotlib`, `pypinyin`, `huggingface-hub`).

---

## NVIDIA Blackwell Compatibility (RTX 5080)

If you are using next-generation GPUs based on the **NVIDIA Blackwell** architecture (compute compatibility `sm_120`):

* **The Problem**: Standard PyTorch builds compiled with CUDA 12.4 only support up to Hopper (`sm_90`). Running them on Blackwell causes `RuntimeError: CUDA error: no kernel image is available for execution on the device`.
* **The Resolution**: Upgraded the container base image to `pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel` and defined the compilation environment variables:
  ```dockerfile
  ENV TORCH_CUDA_ARCH_LIST="12.0"
  ```
  During image creation, the `k2` and `kaldifeat` packages are automatically built from source using your GPU toolchain, ensuring native binary support for `sm_120`.

* **Batch Size Tuning (`--max-duration 300`)**:
  To prevent Out-Of-Memory (OOM) errors on 16GB VRAM card, the maximum batch duration inside all training and decoding scripts (`run_ssl.sh`, `finetune.sh`, etc.) is pre-configured to `300` seconds of audio frames. Avoid setting this back to the default `1000` unless you have multiple GPUs.

---

## Step 2: Running the Container

Instead of running manually with long options, use the pre-configured launcher script located in the repository root:

```bash
./docker_run.sh
```

### Launcher Script Details (`docker_run.sh`)
The launcher runs the container with the following crucial parameters:
* **`--gpus all`**: Exposes your Windows GPU to the container.
* **`--ipc=host`**: Shares host memory. **This is critical** for PyTorch training to prevent container crashes due to shared memory constraints in multi-process dataloading.
* **`-v <repo_path>:/workspace`**: Mounts this repository inside the container. Any code changes made inside Windows or WSL are immediately reflected inside the container.
* **`-v <dataset_path>:/dataset`**: Mounts your datasets folder read-only/read-write for training access.

---

## Environment Configurations & Variables

The container is pre-configured with environment variables to optimize training speeds and avoid WSL bottleneck issues:

* **`PYTHONPATH=/opt/icefall:/workspace:/workspace/external/icefall`**: Sets the icefall frameworks in python paths. Priority is given to local workspace overrides under `external/icefall`.
* **`HF_HOME=/root/.cache/huggingface`**: Saves downloaded checkpoints and tokenizers inside the fast native container storage.
* **`LHOTSE_CACHE=/root/.cache/lhotse`**: Prevents writing temporary cache/manifest verification databases to slow Windows/host mounted folders.

---

## Maintenance and Cleanup

To free up disk space when updating or rebuilding:
* **Remove old image dangling layers**:
  ```bash
  docker image prune -f
  ```
* **Rebuild from scratch (bypassing cache)**:
  ```bash
  docker build --no-cache -t capstone-asr:latest -f docker/Dockerfile .
  ```
