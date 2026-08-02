# Training Pipeline Execution Log

This document confirms the output format and completion of each step in the training pipeline.

## Phase 1: Labeled Data Preparation (ASR)

### Step 1.2: Generate Manifests
- **Command Executed**: `python3 ASR/local/prepare_manifest.py --num-jobs 8 --corpus-dir /dataset/100h-labeled-data --output-dir ASR/data/manifests --language vietnamese`
- **Output Format Confirmed**: Yes. Manifests were generated in `ASR/data/manifests/`.

### Step 1.3: Compute Fbanks
- **Command Executed**: `python3 local/compute_fbank.py`
- **Output Format Confirmed**: Yes. The script reported that `dev`, `test`, and `train` already existed (skipped because they were created prior to the PC reboot).

### Step 1.4: Train BPE Tokenizer
- **Command Executed**: Merged transcripts and ran `python3 local/train_bpe_model.py`
- **Output Format Confirmed**: Yes. The script reported `data/lang_bpe_2000/unigram_2000.model exists - skipping` as it was already computed.

---

## Phase 2: Unlabeled Data Preparation (SSL)

### Step 2.1: Generate & Preprocess Manifests
- **Command Executed**: `python3 local/vietASR_ssl.py` and `python3 local/preprocess_vietASR_ssl.py`
- **Output Format Confirmed**: Yes. Generated `data/ssl_data/capstone-ssl_cuts_data_raw.jsonl.gz`.

### Step 2.2: Lazy Sharding
- **Command Executed**: `lhotse split-lazy data/ssl_data/capstone-ssl_cuts_data_raw.jsonl.gz data/ssl_data/data_split 200000`
- **Output Format Confirmed**: Yes. Output sharded data without errors.

### Step 2.3: Parallel Feature Extraction
- **Command Executed**: `python3 local/compute_fbank_vietASR_ssl_splits.py parallel --src-dir data/ssl_data --dataset data`
- **Output Format Confirmed**: Yes.

---

## Phase 3: K-Means Pseudo-Labeling (SSL)

### Step 3.1: Train K-Means
- **Command Executed**: Subsetting using `lhotse subset` and running `learn_kmeans.py`
- **Output Format Confirmed**: Yes. Generated `kmeans.pt`.

### Step 3.2: Extract Frame Labels
- **Command Executed**: Generating `task_list.txt` and running `extract_kmeans.py`
- **Output Format Confirmed**: Yes. Generated `_with_km.jsonl.gz` for all splits.

---

## Phase 4: Training & Fine-Tuning

### Step 4.1: SSL Pretraining
- **Command Executed**: `docker run --gpus all --rm --ipc=host -v /home/trant/capstone/AIP491-G5-zipformer-training:/workspace -v /home/trant/capstone/dataset:/dataset -w /workspace/SSL capstone-asr:latest bash scripts/run_ssl.sh`
- **Output Format Confirmed**: Yes. Training process completed all 12 Epochs successfully (Finished at 23:51 on Jul 18). `epoch-12.pt` was generated.
- **Note**: The script `run_ssl.sh` successfully trained up to Epoch 12. Proceeding to Phase 4.2.

