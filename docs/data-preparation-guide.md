# Capstone ASR Data Preparation Guide

This guide describes the complete workflow to prepare both **labeled ASR data** and **unlabeled SSL pretraining data** for training the Capstone Zipformer model inside the Docker environment.

---

## Dataset Layout

All raw datasets are located at the same level as the repository, mounted inside the Docker container under `/dataset/`:

```text
/dataset/
├── 100h-labeled-data/        <-- Labeled dataset (100 hours)
│   ├── audio/                <-- Contains 56,427 `.wav` files (16kHz, Mono)
│   └── transcripts/          <-- Contains 56,427 `.txt` files (with matching names)
└── viVoice-unlabeled-data/   <-- Unlabeled dataset (~1,000 hours of raw `.wav` files)
```

---

## Step 1: Entering the Environment

Always run the data preparation scripts inside the optimized Docker container to avoid version conflicts and ensure compatibility with CUDA/GPU acceleration.

Launch the container:
```bash
./docker_run.sh
```

---

## Step 2: Labeled Data Preparation (ASR)

The labeled data preparation processes raw audio-transcript pairs and splits them into training partitions.

### Orchestration Script: `ASR/prepare.sh`
Run Stages 1 to 3 automatically:
```bash
bash prepare.sh --stage 1 --stop-stage 3
```

### Manual Stage Breakdown

#### **Stage 1: Generate Lhotse Manifests**
* **Command**:
  ```bash
  python3 ASR/local/prepare_manifest.py \
    --num-jobs 8 \
    --corpus-dir /dataset/100h-labeled-data \
    --output-dir ASR/data/manifests \
    --language vietnamese
  ```
* **What it does**: Scans the `/dataset/100h-labeled-data/audio` folder and matches `.wav` files with their transcripts in `transcripts/`. It automatically shuffles and splits the dataset into **90% train**, **5% dev**, and **5% test** splits.
* **Output verification**: Check `ASR/data/manifests/` for:
  - `capstone_recordings_train.jsonl.gz` & `capstone_supervisions_train.jsonl.gz`
  - `capstone_recordings_dev.jsonl.gz` & `capstone_supervisions_dev.jsonl.gz`
  - `capstone_recordings_test.jsonl.gz` & `capstone_supervisions_test.jsonl.gz`

#### **Stage 2: Feature Extraction (Filterbanks)**
* **Command**:
  ```bash
  python3 ASR/local/compute_fbank.py
  ```
* **What it does**: Computes 80-dimensional log-mel filterbanks from the audio and maps them to the manifests.
* **Output verification**: Check `ASR/data/fbank/` for:
  - Manifest cuts: `capstone_cuts_train.jsonl.gz`, `capstone_cuts_dev.jsonl.gz`, `capstone_cuts_test.jsonl.gz`
  - Binary feature arrays: `capstone_feats_train/`, `capstone_feats_dev/`, `capstone_feats_test/`

#### **Stage 3: Tokenizer Training (BPE Lang)**
* **Command**:
  ```bash
  mkdir -p ASR/data/lang_bpe_2000
  
  find -L "/dataset/100h-labeled-data/transcripts" -name "*.txt" -exec awk '1' {} + > ASR/data/lang_bpe_2000/transcript_words.txt
  
  # Train SentencePiece BPE model
  python3 ASR/local/train_bpe_model.py \
    --lang-dir ASR/data/lang_bpe_2000 \
    --vocab-size 2000 \
    --transcript ASR/data/lang_bpe_2000/transcript_words.txt
  ```
* **What it does**: Aggregates all words from the transcripts and trains a SentencePiece tokenizer with a vocabulary size of **2,000 subwords**.
* **Output verification**: Check `ASR/data/lang_bpe_2000/` for:
  - `bpe.model` (the binary model used by PyTorch for text tokenization)
  - `tokens.txt` and `words.txt`

---

## Step 3: Unlabeled Data Preparation (SSL)

Unlabeled data preparation requires chunking files using VAD (Voice Activity Detection), sharding the manifest to avoid memory limits, and computing features.

### Orchestration Script: `SSL/prepare_ssl.sh`
Run the complete pipeline:
```bash
cd SSL
bash prepare_ssl.sh --stage 1 --stop-stage 5
```

### Manual Stage Breakdown

#### **Stage 1: Scan & Generate Raw manifests**
* **Command**:
  ```bash
  python3 SSL/local/vietASR_ssl.py \
    --lang Vietnam \
    -j 8 \
    /dataset/viVoice-unlabeled-data \
    data \
    SSL/data/manifest_data
  ```
* **What it does**: Scans the 1000-hour unlabeled audio directory and writes raw recording manifests.

#### **Stage 2: Filter and Normalization**
* **Command**:
  ```bash
  python3 SSL/local/preprocess_vietASR_ssl.py \
    --lang Vietnam \
    --dataset "data" \
    --src-dir SSL/data/manifest_data \
    --tgt-dir SSL/data/ssl_data
  ```
* **What it does**: Filters out cuts that are too short/long and verifies the sampling rates.
* **Output verification**: Generates `SSL/data/ssl_data/capstone-ssl_cuts_data_raw.jsonl.gz`.

#### **Stage 4: Manifest Sharding**
* **Command**:
  ```bash
  lhotse split-lazy SSL/data/ssl_data/capstone-ssl_cuts_data_raw.jsonl.gz SSL/data/ssl_data/data_split 200000
  ```
* **What it does**: Splits the massive manifest into shards containing **200,000 cuts** each. This allows parallel processing and prevents Python memory crashes.
* **Output verification**: Check `SSL/data/ssl_data/data_split/` for files named `capstone-ssl_cuts_data_raw.00000000.jsonl.gz`, `capstone-ssl_cuts_data_raw.00000001.jsonl.gz`, etc.

#### **Stage 5: Parallel Feature Extraction**
* **Command**:
  ```bash
  python3 SSL/local/compute_fbank_vietASR_ssl_splits.py parallel --src-dir SSL/data/ssl_data --dataset data
  ```
* **What it does**: Leverages multiple CPU threads in parallel to compute fbank features for all shards.
* **Output verification**: Generates `capstone-ssl_cuts_data.00000000.jsonl.gz` inside `SSL/data/ssl_data/data_split/` and binary features under `SSL/data/ssl_data/`.
