# Running and Testing Guide

This document describes how to perform audio preprocessing, download pretrained checkpoints, and run Automatic Speech Recognition (ASR) decoding.

---

## 1. Extract Audio from Video (Optional)

If your input is a video file (e.g., `.mp4`), you must convert it to a single-channel wav file with a 16kHz sampling rate using `ffmpeg`:

```bash
ffmpeg -i input-test/video-test-001.mp4 -ar 16000 -ac 1 -c:a pcm_s16le input-test/video-test-001.wav
```

---

## 2. Download Pretrained Checkpoint

To perform inference or fine-tuning, download the pre-trained Iteration 3 VietASR model checkpoint from Hugging Face:

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='zzasdf/viet_iter3_pseudo_label', local_dir='viet_iter3_pseudo_label')"
```

Once downloaded, the folder structure will be:
```
viet_iter3_pseudo_label/
├── exp/
│   └── epoch-12.pt (Model checkpoint)
└── data/
    └── Vietnam_bpe_2000_new/
        └── tokens.txt (Tokenizer tokens)
```

---

## 3. Run Inference / Decoding

Once your audio file and model checkpoints are ready, run the inference command to transcribe the speech:

### macOS / Linux
```bash
# 1. Setup paths
source setup.sh

# 2. Run inference using Zipformer decoding script
python3 ./ASR/zipformer/pretrained.py \
  --checkpoint viet_iter3_pseudo_label/exp/epoch-12.pt \
  --tokens ./viet_iter3_pseudo_label/data/Vietnam_bpe_2000_new/tokens.txt \
  --method modified_beam_search \
  input-test/video-test-001-small.wav
```

### Windows (PowerShell)
```powershell
# 1. Setup paths
$env:PYTHONPATH = "$(Get-Location)/external/icefall;$(Get-Location)/ASR/zipformer;$env:PYTHONPATH"
$env:PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION = "python"

# 2. Run inference
python ./ASR/zipformer/pretrained.py `
  --checkpoint viet_iter3_pseudo_label/exp/epoch-12.pt `
  --tokens ./viet_iter3_pseudo_label/data/Vietnam_bpe_2000_new/tokens.txt `
  --method modified_beam_search `
  input-test/video-test-001-small.wav
```

---

## 4. Run Decoding on Subsets

To evaluate a model checkpoint on validation or test sets, configure the decoding script:

```bash
cd SSL
# Usage: ./scripts/decode.sh <epoch> <average_epochs> <gpu_id>
# Example: Evaluate epoch 12, averaging last 1 epoch, on GPU 0
./scripts/decode.sh 12 1 0
```
*Note: Make sure to update the datasets path inside [SSL/scripts/decode.sh](../SSL/scripts/decode.sh) before running.*
