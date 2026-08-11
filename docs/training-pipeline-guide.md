# Capstone ASR Training Pipeline Guide

This guide describes how to run and validate each stage of the ASR training process, detailing input/output formats, verification steps, and expected file structures.

---

## Pipeline Overview

The pipeline is split into four distinct phases:
1. **ASR Labeled Data Preparation**: Prepare manifests, extract fbanks, and train BPE tokenizer.
2. **SSL Unlabeled Data Preparation**: Preprocess raw audio, shard lists, and extract fbanks.
3. **K-Means Pseudo-Labeling**: Train k-means clusters and extract frame-level target labels.
4. **Pretraining & Fine-Tuning**: Train the encoder on unlabeled data, then fine-tune on labeled data.

---

## Phase 1: Labeled Data Preparation (ASR)

### Step 1.1: Input Validation
Before running any code, verify that the raw dataset is formatted correctly:
* **Audio Check**: Wav files must be **16kHz, single-channel (mono), 16-bit PCM** format.
* **Pairs Check**: Every `.wav` file in `/dataset/100h-labeled-data/audio` must have a matching `.txt` file with the exact same name in `/dataset/100h-labeled-data/transcripts`.
* **Transcript Content**: The `.txt` file should contain only the transcript string (e.g. `chúc các bạn học tập tốt`).

---

### Step 1.2: Generate Manifests
* **Execution**:
  ```bash
  python3 ASR/local/prepare_manifest.py
  ```
* **Output Validation**:
  Ensure the following files are created in `ASR/data/manifests/`:
  - `capstone_recordings_train.jsonl.gz`
  - `capstone_supervisions_train.jsonl.gz` (and dev/test equivalents).
* **Format Structure**:
  Unzip and inspect `capstone_recordings_train.jsonl.gz`:
  ```json
  {"id": "utt_001", "sources": [{"type": "file", "channels": [0], "source": "/dataset/.../audio/utt_001.wav"}], "sampling_rate": 16000, "num_samples": 80000, "duration": 5.0}
  ```
  Unzip and inspect `capstone_supervisions_train.jsonl.gz`:
  ```json
  {"id": "utt_001", "recording_id": "utt_001", "start": 0.0, "duration": 5.0, "channel": 0, "text": "chúc các bạn học tập tốt", "language": "vi", "speaker": "unknown"}
  ```

---

### Step 1.3: Compute Fbanks
* **Execution**:
  ```bash
  cd ASR
  python3 local/compute_fbank.py
  ```
* **Output Validation**:
  Ensure `ASR/data/fbank/` contains:
  - `capstone_cuts_train.jsonl.gz` (and dev/test equivalents).
  - Directory `capstone_feats_train/` containing `.ark` and `.scp` files.
* **Format Structure**:
  A cut integrates the recording and supervision metadata with the feature coordinates:
  ```json
  {"id": "utt_001", "type": "MonoCut", "start": 0.0, "duration": 5.0, "features": {"storage_type": "lilcom_chunky", "path": "data/fbank/capstone_feats_train/...", "num_frames": 500, "num_features": 80}}
  ```

---

### Step 1.4: Train BPE Tokenizer
* **Execution**:
  ```bash
  mkdir -p data/lang_bpe_2000
  find -L "/dataset/100h-labeled-data/transcripts" -name "*.txt" -exec awk '1' {} + > data/lang_bpe_2000/transcript_words.txt
  python3 local/train_bpe_model.py
  ```
* **Output Validation**:
  Ensure `ASR/data/lang_bpe_2000/` contains:
  - `bpe.model` (SentencePiece tokenizer model).
  - `tokens.txt` (mapping of token subwords to integers).

---

## 🎧 Phase 2: Unlabeled Data Preparation (SSL)

### Step 2.1: Generate & Preprocess Manifests
* **Execution**:
  ```bash
  cd SSL
  python3 local/asr_ssl.py --lang Vietnam -j 8 /dataset/viVoice-unlabeled-data data data/manifest_data
  python3 local/preprocess_asr_ssl.py --lang Vietnam --dataset "data" --src-dir data/manifest_data --tgt-dir data/ssl_data
  ```
* **Output Validation**:
  Ensure `SSL/data/ssl_data/capstone-ssl_cuts_data_raw.jsonl.gz` is created.

---

### Step 2.2: Lazy Sharding
* **Execution**:
  ```bash
  lhotse split-lazy data/ssl_data/capstone-ssl_cuts_data_raw.jsonl.gz data/ssl_data/data_split 200000
  ```
* **Output Validation**:
  Ensure `SSL/data/ssl_data/data_split/` contains sharded files:
  - `capstone-ssl_cuts_data_raw.00000000.jsonl.gz`
  - `capstone-ssl_cuts_data_raw.00000001.jsonl.gz`

---

### Step 2.3: Parallel Feature Extraction
* **Execution**:
  ```bash
  python3 local/compute_fbank_asr_ssl_splits.py parallel --src-dir data/ssl_data --dataset data
  ```
* **Output Validation**:
  Ensure `SSL/data/ssl_data/data_split/` now contains completed cuts:
  - `capstone-ssl_cuts_data.00000000.jsonl.gz`

---

## Phase 3: Iterative Self-Training (ASR Pseudo-Labeling)

This phase trains the model iteratively using pseudo-labels generated from unlabeled data using ASR model checkpoints.

---

### **Iteration 1 (9 Epochs per Step)**

#### **Step 1: Train ASR Zipformer with 100h Labeled Data**
* **Execution**:
  Train the initial ASR model on the 100h labeled corpus for 9 epochs:
  ```bash
  cd ASR
  python zipformer/train.py \
      --world-size 1 \
      --num-epochs 9 \
      --start-epoch 1 \
      --use-fp16 1 \
      --train-cuts 100h \
      --manifest-dir data/fbank \
      --bpe-model data/lang_bpe_2000/bpe.model \
      --max-duration 300 \
      --enable-musan 0 \
      --exp-dir zipformer/exp \
      --enable-spec-aug 1 \
      --seed 1332 \
      --master-port 12356
  ```
* **Output Validation**:
  Verify ASR checkpoint `ASR/zipformer/exp/epoch-9.pt` is generated.

#### **Step 2: Train K-Means and Extract Frame-Level Pseudo-Labels**
* **Execution**:
  1. Train a K-Means clustering model (500 clusters) on a 50,000 cut subset of the unlabeled data:
     ```bash
     cd SSL
     lhotse subset --first 50000 data/ssl_data/data_split/capstone-ssl_cuts_data.00000000.jsonl.gz data/ssl_data/data_split/kmeans_subset.jsonl.gz
     PYTHONPATH=zipformer_fbank:$PYTHONPATH python3 zipformer_fbank/extract_kmeans_scripts/learn_kmeans.py \
         --km-path data/kmeans.pt \
         --n-clusters 500 \
         --max-iter 100 \
         --bpe-model ../ASR/data/lang_bpe_2000/bpe.model \
         --files data/ssl_data/data_split/kmeans_subset.jsonl.gz \
         --checkpoint-type ASR \
         --pretrained-dir ../ASR/zipformer/exp \
         --epoch 9 \
         --avg 1 \
         --use-averaged-model 0 \
         --do-training
     ```
  2. Generate task list for extraction and extract the frame cluster labels:
     ```bash
     ls data/ssl_data/data_split/capstone-ssl_cuts_data.0000000*.jsonl.gz | while read f; do split_num=$(echo $f | grep -o "[0-9]\{8\}"); echo "$f data/ssl_data/data_split/capstone-ssl_cuts_data_kmeans.${split_num}.jsonl.gz"; done > data/ssl_data/task_list.txt
     
     PYTHONPATH=zipformer_fbank:$PYTHONPATH python3 zipformer_fbank/extract_kmeans_scripts/extract_kmeans.py \
         --task-list data/ssl_data/task_list.txt \
         --model-path data/kmeans.pt \
         --checkpoint-type ASR \
         --pretrained-dir ../ASR/zipformer/exp \
         --epoch 9 \
         --avg 1 \
         --use-averaged-model 0 \
         --bpe-model ../ASR/data/lang_bpe_2000/bpe.model
     ```
* **Output Validation**:
  Ensure annotated cuts `capstone-ssl_cuts_data_kmeans.0000000*.jsonl.gz` are written in `SSL/data/ssl_data/data_split/`.

#### **Step 3: Pre-train Zipformer with Pseudo-Labels and Masking**
* **Execution**:
  Pretrain the model on the unlabeled data using the generated K-Means targets and time/channel masking. *(Note: The dataloader automatically reserves the last split `cut004` as the validation set, and exclusively uses splits 0-3 for training).*
  ```bash
  cd SSL
  python3 zipformer_fbank/pretrain.py \
      --world-size 1 \
      --num-epochs 9 \
      --start-epoch 1 \
      --use-fp16 1 \
      --label-type kmeans \
      --manifest-prefix ssl_ \
      --label-rate 50 \
      --sample-rate 100 \
      --exp-dir zipformer_fbank/exp \
      --max-duration 300 \
      --train-cut large \
      --accum-grad 1 \
      --min-keep-size 200 \
      --mask-before-cnn 1 \
      --max-sample-size 1562 \
      --mask-prob 0.80 \
      --dropout-input 0.1 \
      --dropout-features 0.1 \
      --base-lr 0.045 \
      --save-every-n 15000 \
      --master-port 12356 \
      --manifest-dir data
  ```
* **Output Validation**:
  Verify the pre-trained weights file `SSL/zipformer_fbank/exp/epoch-9.pt` is generated.

#### **Step 4: Fine-tune Zipformer on Labeled Data**
* **Execution**:
  Fine-tune the encoder model using the 100h labeled dataset:
  ```bash
  cd SSL
  python zipformer_fbank/finetune.py \
      --world-size 1 \
      --num-epochs 9 \
      --start-epoch 1 \
      --use-fp16 1 \
      --sample-rate 100 \
      --manifest-dir ../ASR/data/fbank \
      --bpe-model ../ASR/data/lang_bpe_2000/bpe.model \
      --exp-dir zipformer_fbank/exp_ft \
      --max-duration 300 \
      --enable-musan 0 \
      --enable-spec-aug 0 \
      --mask-before-cnn 1 \
      --mask-prob 0.65 \
      --mask-channel-prob 0.5 \
      --mask-channel-length 20 \
      --accum-grad 1 \
      --seed 1556 \
      --base-lr 0.002 \
      --max-lr-update 80000 \
      --phase-ratio "(0.1, 0.4, 0.5)" \
      --pretrained-checkpoint-path zipformer_fbank/exp/epoch-9.pt \
      --pretrained-checkpoint-type SSL \
      --init-encoder-only 1 \
      --use-layer-norm 0 \
      --final-downsample 1 \
      --causal 0 \
      --master-port 12356
  ```
* **Output Validation**:
  Verify the fine-tuned checkpoint `SSL/zipformer_fbank/exp_ft/epoch-9.pt` is successfully created.

---

### **Iteration 2+ (18 Epochs per Step)**

For Iteration 2 (and future iterations), repeat Steps 2 to 4 using the previous iteration's Step 4 checkpoint.

#### **Step 2: Train K-Means and Extract Frame-Level Pseudo-Labels (Iteration 2)**
* **Execution (Learn K-Means)**:
  ```bash
  cd SSL
  PYTHONPATH=zipformer_fbank:$PYTHONPATH python3 zipformer_fbank/extract_kmeans_scripts/learn_kmeans.py \
      --km-path data/kmeans_iter2.pt \
      --n-clusters 500 \
      --max-iter 100 \
      --bpe-model ../ASR/data/lang_bpe_2000/bpe.model \
      --files data/ssl_data/data_split/kmeans_subset.jsonl.gz \
      --checkpoint-type ASR \
      --pretrained-dir zipformer_fbank/exp_ft \
      --epoch 9 \
      --avg 1 \
      --use-averaged-model 0 \
      --do-training
  ```
* **Execution (Extract Pseudo-Labels)**:
  ```bash
  ls data/ssl_data/data_split/capstone-ssl_cuts_data.0000000*.jsonl.gz | while read f; do split_num=$(echo $f | grep -o "[0-9]\{8\}"); echo "$f data/ssl_data/data_split/capstone-ssl_cuts_data_kmeans_iter2.${split_num}.jsonl.gz"; done > data/ssl_data/task_list_iter2.txt
  
  PYTHONPATH=zipformer_fbank:$PYTHONPATH python3 zipformer_fbank/extract_kmeans_scripts/extract_kmeans.py \
      --task-list data/ssl_data/task_list_iter2.txt \
      --model-path data/kmeans_iter2.pt \
      --checkpoint-type ASR \
      --pretrained-dir zipformer_fbank/exp_ft \
      --epoch 9 \
      --avg 1 \
      --use-averaged-model 0 \
      --bpe-model ../ASR/data/lang_bpe_2000/bpe.model
  ```

#### **Step 3: Pre-train Zipformer on Pseudo-Labels with Input Masking (Iteration 2)**
* **Execution**:
  ```bash
  cd SSL
  python3 zipformer_fbank/pretrain.py \
      --world-size 1 \
      --num-epochs 18 \
      --start-epoch 1 \
      --use-fp16 1 \
      --label-type kmeans_iter2 \
      --manifest-prefix ssl_ \
      --label-rate 50 \
      --sample-rate 100 \
      --exp-dir zipformer_fbank/exp_iter2 \
      --max-duration 300 \
      --train-cut large \
      --accum-grad 1 \
      --min-keep-size 200 \
      --mask-before-cnn 1 \
      --max-sample-size 1562 \
      --mask-prob 0.80 \
      --dropout-input 0.1 \
      --dropout-features 0.1 \
      --base-lr 0.045 \
      --save-every-n 15000 \
      --master-port 12356 \
      --manifest-dir data
  ```

#### **Step 4: Fine-tune Zipformer on Labeled Data (Iteration 2)**
* **Execution**:
  ```bash
  cd SSL
  python zipformer_fbank/finetune.py \
      --world-size 1 \
      --num-epochs 18 \
      --start-epoch 1 \
      --use-fp16 1 \
      --sample-rate 100 \
      --manifest-dir ../ASR/data/fbank \
      --bpe-model ../ASR/data/lang_bpe_2000/bpe.model \
      --exp-dir zipformer_fbank/exp_ft_iter2 \
      --max-duration 300 \
      --enable-musan 0 \
      --enable-spec-aug 0 \
      --mask-before-cnn 1 \
      --mask-prob 0.65 \
      --mask-channel-prob 0.5 \
      --mask-channel-length 20 \
      --accum-grad 1 \
      --seed 1556 \
      --base-lr 0.002 \
      --max-lr-update 80000 \
      --phase-ratio "(0.1, 0.4, 0.5)" \
      --pretrained-checkpoint-path zipformer_fbank/exp_iter2/epoch-18.pt \
      --pretrained-checkpoint-type SSL \
      --init-encoder-only 1 \
      --use-layer-norm 0 \
      --final-downsample 1 \
      --causal 0 \
      --master-port 12356
  ```

---

### **Iteration 3 (18 Epochs per Step)**

For Iteration 3, repeat the process using Iteration 2's fine-tuned checkpoint (`zipformer_fbank/exp_ft_iter2/epoch-18.pt`).

#### **Step 2: Train K-Means and Extract Frame-Level Pseudo-Labels (Iteration 3)**
* **Execution (Learn K-Means)**:
  ```bash
  cd SSL
  PYTHONPATH=zipformer_fbank:$PYTHONPATH python3 zipformer_fbank/extract_kmeans_scripts/learn_kmeans.py \
      --km-path data/kmeans_iter3.pt \
      --n-clusters 500 \
      --max-iter 100 \
      --bpe-model ../ASR/data/lang_bpe_2000/bpe.model \
      --files data/ssl_data/data_split/kmeans_subset.jsonl.gz \
      --checkpoint-type ASR \
      --pretrained-dir zipformer_fbank/exp_ft_iter2 \
      --epoch 18 \
      --avg 1 \
      --use-averaged-model 0 \
      --do-training
  ```
* **Execution (Extract Pseudo-Labels)**:
  ```bash
  ls data/ssl_data/data_split/capstone-ssl_cuts_data.0000000*.jsonl.gz | while read f; do split_num=$(echo $f | grep -o "[0-9]\{8\}"); echo "$f data/ssl_data/data_split/capstone-ssl_cuts_data_kmeans_iter3.${split_num}.jsonl.gz"; done > data/ssl_data/task_list_iter3.txt
  
  PYTHONPATH=zipformer_fbank:$PYTHONPATH python3 zipformer_fbank/extract_kmeans_scripts/extract_kmeans.py \
      --task-list data/ssl_data/task_list_iter3.txt \
      --model-path data/kmeans_iter3.pt \
      --checkpoint-type ASR \
      --pretrained-dir zipformer_fbank/exp_ft_iter2 \
      --epoch 18 \
      --avg 1 \
      --use-averaged-model 0 \
      --bpe-model ../ASR/data/lang_bpe_2000/bpe.model
  ```

#### **Step 3: Pre-train Zipformer on Pseudo-Labels with Input Masking (Iteration 3)**
* **Execution**:
  ```bash
  cd SSL
  python3 zipformer_fbank/pretrain.py \
      --world-size 1 \
      --num-epochs 18 \
      --start-epoch 1 \
      --use-fp16 1 \
      --label-type kmeans_iter3 \
      --manifest-prefix ssl_ \
      --label-rate 50 \
      --sample-rate 100 \
      --exp-dir zipformer_fbank/exp_iter3 \
      --max-duration 300 \
      --train-cut large \
      --accum-grad 1 \
      --min-keep-size 200 \
      --mask-before-cnn 1 \
      --max-sample-size 1562 \
      --mask-prob 0.80 \
      --dropout-input 0.1 \
      --dropout-features 0.1 \
      --base-lr 0.02 \
      --save-every-n 15000 \
      --master-port 12356 \
      --manifest-dir data
  ```

#### **Step 4: Fine-tune Zipformer on Labeled Data (Iteration 3)**
* **Execution**:
  ```bash
  cd SSL
  python3 zipformer_fbank/finetune.py \
      --world-size 1 \
      --num-epochs 18 \
      --start-epoch 1 \
      --use-fp16 1 \
      --sample-rate 100 \
      --manifest-dir ../ASR/data/fbank \
      --bpe-model ../ASR/data/lang_bpe_2000/bpe.model \
      --exp-dir zipformer_fbank/exp_ft_iter3 \
      --max-duration 300 \
      --enable-musan 0 \
      --enable-spec-aug 0 \
      --mask-before-cnn 1 \
      --mask-prob 0.65 \
      --mask-channel-prob 0.5 \
      --mask-channel-length 20 \
      --accum-grad 1 \
      --seed 1556 \
      --base-lr 0.002 \
      --max-lr-update 80000 \
      --phase-ratio "(0.1, 0.4, 0.5)" \
      --pretrained-checkpoint-path zipformer_fbank/exp_iter3/epoch-18.pt \
      --pretrained-checkpoint-type SSL \
      --init-encoder-only 1 \
      --use-layer-norm 0 \
      --final-downsample 1 \
      --causal 0 \
      --master-port 12356
  ```
