# Modification & Bug Fix History

This document tracks fixes and modifications made to the repository to adapt it for the Capstone project environment.

## 2026-07-18

### 1. Docker Environment Compatibility
* **Issue**: The original Docker configuration compiled Python packages for generic architectures and threw PyTorch CUDA compatibility errors on RTX 5080 (Blackwell `sm_120`).
* **Fix**: Updated `docker/Dockerfile` base image to `pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel` and compiled `k2` and `kaldifeat` entirely from source by setting `TORCH_CUDA_ARCH_LIST="12.0"`.

### 2. Missing `lilcom` Dependency
* **Issue**: Running Stage 2 data preparation failed with `ImportError: To read and write lilcom-compressed arrays, please 'pip install lilcom'`.
* **Fix**: Appended `lilcom` to the dependencies block inside `docker/Dockerfile`.

### 3. Volume Path Mismatches
* **Issue**: `dl_dir` in `ASR/prepare.sh` and `SSL/prepare_ssl.sh` were hardcoded to the Windows WSL host path (`/home/trant/...`), causing "No such directory" errors inside Docker.
* **Fix**: Updated `dl_dir` in both scripts to use the Docker mounted volume paths (`/dataset/100h-labeled-data` and `/dataset/viVoice-unlabeled-data`).

### 4. SentencePiece BPE Empty Sentence Crash
* **Issue**: `ASR/prepare.sh` Stage 3 crashed with `!sentences_.empty()` because the raw `.txt` transcript files lacked trailing newlines. The `cat` command merged 56,427 transcripts into a single 8.1MB long line, which exceeded SentencePiece's memory limits.
* **Fix**: Modified the transcript merging loop in `ASR/prepare.sh` to use `find ... -exec awk '1' {} +`, which automatically guarantees a newline is injected between every text file.

### 5. SSL Preprocess Script Folder Creation Bug
* **Issue**: `SSL/local/preprocess_vietASR_ssl.py` crashed with a `FileNotFoundError` when trying to create the output directory because the parent directory `SSL/data` didn't exist yet, and `mkdir()` lacked the `parents=True` argument.
* **Fix**: Modified the script to use `output_dir.mkdir(parents=True, exist_ok=True)`.

### 6. Phase 3 K-Means Path Inconsistencies
* **Issue**: The commands in `training-pipeline-guide.md` for Phase 3 (Steps 3.1 & 3.2) were missing a `cd SSL` instruction and mixed up root-relative and SSL-relative paths, which would cause `FileNotFoundError` when trying to find the splits or save `kmeans.pt`.
* **Fix**: Updated `training-pipeline-guide.md` to prepend `cd SSL` and corrected all paths to be relative to the `SSL` directory (e.g. `data/ssl_data/data_split` and `../ASR/data/lang_bpe_2000/bpe.model`).

### 7. Phase 3 K-Means Execution Crashes (Missing PYTHONPATH and BPE Model)
* **Issue**: Running Phase 3 K-Means scripts resulted in `ModuleNotFoundError: No module named 'finetune'` because the scripts depend on modules inside `zipformer_fbank` which wasn't in the Python path. Next, it crashed with `NOT_FOUND: "data/lang_bpe_500/bpe.model"` because `learn_kmeans.py` defaults to that path, but our BPE model is in `ASR/data/lang_bpe_2000`.
* **Fix**: Prepended `PYTHONPATH=zipformer_fbank:$PYTHONPATH` to both commands in `training-pipeline-guide.md` and explicitly added `--bpe-model ../ASR/data/lang_bpe_2000/bpe.model` to Step 3.1.

### 8. Phase 3 K-Means Host OOM Crash
* **Issue**: `learn_kmeans.py` loaded the entire `capstone-ssl_cuts_data.00000000.jsonl.gz` file (200,000 cuts) into memory simultaneously for feature extraction before clustering. This exhausted the 60GB host RAM and caused the OS to OOM kill the process (exit code 137).
* **Fix**: Added a step in `training-pipeline-guide.md` to first subset the data using `lhotse subset --first 50000` to create a `kmeans_subset.jsonl.gz` which is fed into k-means to avoid OOM while still providing enough data (50,000 cuts) for learning 500 clusters.

### 9. Missing `einops` Dependency
* **Issue**: `extract_kmeans.py` crashed with `ModuleNotFoundError: No module named 'einops'`.
* **Fix**: Added `einops` to the dependencies block inside `docker/Dockerfile`.

### 10. `extract_kmeans.py` Task List Directory Error
* **Issue**: The command for Step 3.2 passed a directory (`data/ssl_data/data_split`) to `--task-list`, which expects a text file mapping source to target files. This caused an `IsADirectoryError`.
* **Fix**: Added a shell loop in `training-pipeline-guide.md` to generate `data/ssl_data/task_list.txt` before running the extraction script, and updated the script argument to point to this file.

### 11. `run_ssl.sh` Label Mismatch and Missing PYTHONPATH
* **Issue**: `SSL/scripts/run_ssl.sh` had `--label-type kmeans_ASR_100h`, but Phase 3 writes the labels under the key `"kmeans"`. Also, `run_ssl.sh` lacked the `PYTHONPATH` for `zipformer_fbank`, causing it to fail to import modules.
* **Fix**: Updated `SSL/scripts/run_ssl.sh` to use `--label-type kmeans` and export `PYTHONPATH=zipformer_fbank:$PYTHONPATH`.

### 12. SSL Training Manifest Prefix Mismatch and Git Ownership
* **Issue**: `pretrain.py` defaults to `manifest_prefix="ssl_train"`, but the data directory inside `data/` is named `ssl_data`. This caused `lhotse.combine` to fail with `TypeError: reduce() of empty iterable` because it found no splits. Additionally, git complained about dubious ownership because the workspace is owned by a different user than the docker container root.
* **Fix**: Added `--manifest-prefix ssl_data` and `git config --global --add safe.directory /workspace` to `SSL/scripts/run_ssl.sh`.

### 13. Target Filename Format Mismatch in K-Means Extraction
* **Issue**: The labels extraction script expects output filenames to strictly match the pattern `"capstone-ssl_cuts_data_{label_type}.([0-9]+).jsonl.gz"`. Our previous shell loop generated `*_with_km.jsonl.gz`, which caused `ssl_datamodule.py` in Phase 4 to silently skip the files and crash with `TypeError: reduce() of empty iterable`.
* **Fix**: Updated the bash loop in Step 3.2 to correctly output filenames matching the `_kmeans` suffix pattern: `capstone-ssl_cuts_data_kmeans.([0-9]+).jsonl.gz`. Renamed existing files to avoid having to re-extract.

### 14. Missing SSL Dev Dataset
* **Issue**: The `ssl_datamodule.py` script hardcodes a requirement for a `ssl_dev/dev_split` directory for validation during training, which was never prepared in the pipeline guide. This caused a `FileNotFoundError`.
* **Fix**: Modified `ssl_datamodule.py` to point to the existing `ssl_data/data_split` folder instead. Specifically, `dev_cuts_vi_ssl` now grabs the last split (`cut004`, ~6% of the data) for validation, and `train_cuts_vi_ssl` explicitly excludes `cut004` from the training set to prevent overlap.
### 15. Fine-Tuning Script Missing BPE and Fbank Paths
* **Issue**: `SSL/scripts/finetune_ASR_checkpoint.sh` assumes the script is run from `ASR` and points to `data/lang_bpe_2000/bpe.model` and `data/fbank`. Since the Docker container sets the working directory to `/workspace/SSL`, these paths result in `FileNotFoundError`.
* **Fix**: Updated `SSL/scripts/finetune_ASR_checkpoint.sh` to correctly reference `../ASR/data/lang_bpe_2000/bpe.model` and `../ASR/data/fbank`.

### 16. PyTorch 2.6 `torch.load()` `weights_only` Restriction
* **Issue**: Loading `epoch-12.pt` failed with `_pickle.UnpicklingError: Weights only load failed` because PyTorch 2.6 defaults `torch.load` to `weights_only=True`, which blocks loading custom objects like `pathlib.PosixPath` contained within the checkpoint.
* **Fix**: Updated `SSL/zipformer_fbank/finetune.py` (lines 866 and 937) to explicitly set `weights_only=False` in `torch.load()` calls.

### 17. Fine-Tuning Script Full Model Initialization Error
* **Issue**: Fine-tuning crashed with `Exception: Not supported checkpoint type SSL for full model initialization`.
* **Fix**: Updated `SSL/scripts/finetune_ASR_checkpoint.sh` to change `--init-encoder-only` from `0` to `1`, since an SSL pretrained checkpoint only contains encoder weights and cannot be used to initialize the full ASR model (which includes the decoder and joiner).

### 18. Missing `lilcom` Dependency in Docker
* **Issue**: The fine-tuning script crashed with `ImportError: To read and write lilcom-compressed arrays, please 'pip install lilcom'`. The Docker image lacks this library which is necessary to read ASR fbank features.
* **Fix**: Added `pip install lilcom` to the top of `SSL/scripts/finetune_ASR_checkpoint.sh` so it dynamically installs the library before training starts.

### 19. Relative Path Resolution for `.lca` Audio Features
* **Issue**: Lhotse crashed with `FileNotFoundError: No such file or directory: 'data/fbank/capstone_feats_train/feats-0.lca'`. The cut manifests generated in Phase 1 store relative paths to features. Because Docker's working directory is `/workspace/SSL`, it looks for `SSL/data/fbank`, but the actual features are in `ASR/data/fbank`.
* **Fix**: Added `ln -sfn ../../ASR/data/fbank data/fbank` inside `SSL/scripts/finetune_ASR_checkpoint.sh` to create a symlink, allowing Lhotse to resolve the relative paths correctly during data loading.
