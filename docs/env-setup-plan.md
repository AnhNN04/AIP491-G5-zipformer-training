# Environment Installation Guide

This document details the step-by-step instructions to set up the runtime environment for VietASR on Linux, macOS (both Apple Silicon and Intel), and Windows.

---

## Prerequisites

Ensure you have the following packages installed on your system before proceeding:
- **Python >= 3.11** (recommended: Python 3.11.9)
- **CMake** (required to compile C++ extensions like `k2`)
- **FFmpeg** (required for audio preprocessing and VAD)
- **uv** (a fast Python package installer and resolver)

### OS-Specific Prerequisite Commands

#### macOS (using Homebrew)
```bash
brew install cmake ffmpeg uv
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install -y cmake ffmpeg
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### Windows (using winget)
```powershell
# Run in Administrator PowerShell
winget install Kitware.CMake
winget install Gyan.FFmpeg
winget install astral-sh.uv
```

---

## 1. Setup Virtual Environment

Initialize a clean virtual environment using the `uv` package manager:

```bash
# Create a virtual environment using Python 3.11
uv venv --python 3.11

# Activate the virtual environment
# macOS / Linux:
source .venv/bin/activate
# Windows (cmd):
.venv\Scripts\activate.bat
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
```

---

## 2. Install PyTorch

Choose the appropriate command based on your OS and hardware accelerator configuration:

### Linux / Windows (with Nvidia GPU)
```bash
uv add torch torchaudio
```

### macOS (Apple Silicon / Intel)
```bash
uv add torch torchaudio
```

### Linux / Windows (CPU Only)
```bash
uv add torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

---

## 3. Install Core Project Dependencies

Install all the top-level dependencies declared in `pyproject.toml` in editable mode:
```bash
uv pip install -e .
```

---

## 4. Install `k2` (FSA Autograd Library)

`k2` contains C++ extensions and requires specific compilation steps depending on your OS:

### macOS (Apple Silicon M1/M2/M3) - Recommended
To avoid C++ compiler linker ABI conflicts on macOS, install the prebuilt CPU wheel directly:
```bash
# For Python 3.11 + PyTorch 2.12.0
uv pip install https://huggingface.co/csukuangfj2/k2/resolve/main/macos/k2-1.24.4.dev20260520+cpu.torch2.12.0-cp311-cp311-macosx_11_0_arm64.whl
```
*Note: For other Python/PyTorch versions on macOS, find the matching wheel URL from [k2 CPU Wheels](https://k2-fsa.github.io/k2/cpu.html).*

### Linux / Windows (with GPU/CUDA)
Download and install the pre-compiled GPU wheels from the official `k2-fsa` release page:
```bash
# Make sure CUDA is available in path, then download matching nightly wheel
uv pip install k2 -f https://k2-fsa.org/nightly/
```

### Building `k2` from Local Source (`external/k2`)
If you prefer to compile `k2` directly from the local source directory:
```bash
# Disable build isolation so k2 can find torch in the active venv
uv pip install -e external/k2 --no-build-isolation
```
*Note for macOS source compilation: After compilation, copy the compiled dynamic libraries to the source folder to prevent `@rpath/libk2context.dylib` loading issues:*
```bash
mkdir -p external/k2/k2/lib
cp external/k2/build/lib.macosx-*/k2/lib/*.dylib external/k2/k2/lib/
```

---

## 5. Configure Paths

Before running any script or test in this repo, configure your python path.

### macOS / Linux
```bash
source setup.sh
```

### Windows (PowerShell)
```powershell
$env:PYTHONPATH = "$(Get-Location)/external/icefall;$(Get-Location)/ASR/zipformer;$env:PYTHONPATH"
$env:PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION = "python"
```

---

## 6. Verification

Run the built-in unit tests to verify that your environment is fully operational:

```bash
python ASR/zipformer/test_scaling.py
python ASR/zipformer/test_subsampling.py
```

If both tests complete without errors, your environment is successfully set up!
