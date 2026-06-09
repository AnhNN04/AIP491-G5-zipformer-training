# Plan for Running and Testing VietASR (Mac M1)

This document details the completed and remaining steps to set up the environment, run unit tests, and perform ASR inference on a Mac M1.

## Completed Steps

1. **Virtual Environment Setup**: Activated `.venv` (Python 3.11).
2. **Core Dependencies**: Installed `torch`, `torchaudio`, `lhotse`, and `matplotlib`.
3. **k2 Source Compilation**: Cloned and successfully built/installed CPU-only `k2` from source.
4. **icefall Integration**: Cloned `icefall` into `external/icefall`.
5. **kaldifeat Source Compilation**: Successfully built and installed `kaldifeat` with C++17 support:
   ```bash
   export KALDIFEAT_CMAKE_ARGS="-DCMAKE_CXX_STANDARD=17"
   pip install --no-build-isolation kaldifeat
   ```
6. **Unit Tests Verification**:
   - `python ASR/zipformer/test_scaling.py` -> PASSED
   - `python ASR/zipformer/test_subsampling.py` -> PASSED

---

## Remaining Steps

### 1. Extract WAV from MP4 Dialogue
Use `ffmpeg` (install it via `brew install ffmpeg` first if not already available) to convert your podcast video:
```bash
ffmpeg -i input-test/videot-test-001.mp4 -ar 16000 -ac 1 -c:a pcm_s16le input-test/videot-test-001.wav
```

### 2. Download Pretrained Checkpoint
Download the checkpoint `viet_iter3_pseudo_label` from Hugging Face using the corrected Python command (separated with a semicolon `;`):
```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='zzasdf/viet_iter3_pseudo_label', local_dir='viet_iter3_pseudo_label')"
```

### 3. Run Inference Test
Once the model is downloaded and the WAV is generated, run the ASR transcription script:
```bash
# Ensure PYTHONPATH includes icefall
export PYTHONPATH=$(pwd)/external/icefall:$PYTHONPATH

# Run inference
python3 ./ASR/zipformer/pretrained.py \
  --checkpoint viet_iter3_pseudo_label/exp/epoch-12.pt \
  --tokens ./viet_iter3_pseudo_label/data/Vietnam_bpe_2000_new/tokens.txt \
  --method modified_beam_search \
  input-test/videot-test-001.wav
```
