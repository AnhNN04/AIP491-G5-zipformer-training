# ASR Module Training Capstone

This repository contains the training pipeline and model definitions for the Automatic Speech Recognition (ASR) module capstone project. The system utilizes a Zipformer-Small (~30M parameters) architecture with an RNN-Transducer (RNN-T) head, built on top of the `lhotse` and `icefall` frameworks.

## Overview

This project implements a self-supervised and semi-supervised ASR training pipeline:
1. **Initial ASR Training**: Training a base ASR model using a labeled training dataset.
2. **K-Means Training & Label Extraction**: Training a k-means model on acoustic features of unlabeled speech data and extracting discrete token representation labels.
3. **Self-Supervised Pretraining (SSL)**: Training a strong speech representation encoder using the extracted discrete labels.
4. **Finetuning & Decoding**: Downstream finetuning of the pretrained encoder on target labeled datasets and decoding evaluation.

---

## Documentation

Detailed documentation of system design, architecture, and step-by-step pipeline execution is available in the [`docs/`](docs/) folder:

| Document | Description |
|----------|-------------|
| [Repository Guide](docs/repository-guide.md) | Complete directory and structure reference for the codebase. |
| [ASR System Guide (EN)](docs/asr-system-guide.md) | Detailed English manual on the ASR architecture, running modes, and parameters. |
| [ASR System Guide (VI)](docs/asr-system-guide-vi.md) | Vietnamese translation of the system architecture and running mode guide. |
| [Data Preparation Guide](docs/data-preparation-guide.md) | How to format and process labeled (ASR) and unlabeled (SSL) audio datasets. |
| [Docker Setup Guide](docs/docker-setup-guide.md) | Custom Docker container instructions, GPU configurations, memory tuning, and path bindings. |
| [Training Pipeline Guide](docs/training-pipeline-guide.md) | Steps to run SSL pretraining, downstream finetuning, and decoding evaluation. |
| [Zipformer Architecture](docs/zipformer-architecture.md) | Explanation of the U-Net styled multi-rate downsampling Zipformer2 encoder. |

---

## Environment Setup

The pipeline relies on `lhotse` for data preprocessing and `icefall` as the core model framework. The recommended runtime is inside the custom Docker container to handle all CUDA/C++ dependency compilations:

1. **Build the container**:
   ```bash
   docker build -t asr-training:latest -f docker/Dockerfile .
   ```
2. **Run the container**:
   ```bash
   ./docker_run.sh
   ```
3. **Environment Paths**:
   Ensure `icefall` is in your `PYTHONPATH` before executing any script:
   ```bash
   export PYTHONPATH=/workspace:/workspace/external/icefall:$PYTHONPATH
   ```

---

## Data Preparation

For complete instructions, refer to the [Data Preparation Guide](docs/data-preparation-guide.md).

### 1. Unlabeled Data (SSL)
Use Voice Activity Detection (VAD) to segment long audio files, place segmented `.wav` files under `SSL/download/ssl_${subset_name}`, and run:
```bash
cd SSL
./prepare_ssl.sh
```
This generates unsupervised manifests under `SSL/data/ssl_${subset_name}/`.

### 2. Labeled Data (ASR)
Place labeled data under `download/supervised` (structured with `.wav` files and corresponding `*.trans.txt` transcription files) and run:
```bash
cd ASR
./prepare.sh
```

---

## Training Pipeline Execution

For detailed commands and options, refer to the [Training Pipeline Guide](docs/training-pipeline-guide.md).

### 1. Initial ASR Model Training
Train a base model on the labeled subset to establish the initial acoustic encoder:
```bash
cd ASR
./scripts/train.sh
```

### 2. Train K-Means Model
Train the k-means model on a subset of unsupervised acoustic features (approx. 100 hours):
```bash
cd SSL
./scripts/learn_vietASR_kmeans.sh
```

### 3. Extract Labels
Generate discrete k-means targets for the unlabeled audio cuts:
```bash
cd SSL
./scripts/extract_vietASR_kmeans.sh
```

### 4. Self-Supervised Pre-Training (SSL)
Pre-train the speech encoder using the extracted discrete k-means labels:
```bash
cd SSL
./scripts/run_ssl.sh
```

### 5. Downstream Fine-Tuning
Finetune the pre-trained encoder on your target labeled dataset:
```bash
cd SSL
./scripts/finetune.sh
```

### 6. Decoding & Evaluation
Evaluate model performance by running:
```bash
cd SSL
./scripts/decode.sh $epoch $avg $gpu_id
```

---

## Model Architecture

The default encoder architecture is configured as a standard **Zipformer (~68M parameters)** with a stateless RNN-T head:

| Component | Configuration |
|-----------|---------------|
| Encoder Stacks | 6 (U-Net shaped) |
| Encoder Dimensions | 192, 256, 384, 512, 384, 256 |
| Layers per Stack | 2, 2, 3, 4, 3, 2 |
| Feedforward Dimensions | 512, 768, 1024, 1536, 1024, 768 |
| Attention Heads | 4, 4, 4, 8, 4, 4 |
| Total Parameters | ~68M |

---

## License

This project is built on the [icefall](https://github.com/k2-fsa/icefall) framework and is licensed under the Apache License 2.0. Detailed license information can be found in the [LICENSE](LICENSE) file.
