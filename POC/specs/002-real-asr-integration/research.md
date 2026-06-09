# Technical Research: Real ASR Serving Integration

This document details the architectural decisions, design patterns, and choices made for implementing the real model inference and streaming serving stack for the ASR Model Server (`asr-server`).

## 1. Audio Feature Extraction (Fbank)

* **Decision**: Use `torchaudio.compliance.kaldi.fbank` for feature extraction.
* **Rationale**: 
  * The core `ASR/zipformer` pipeline is based on Kaldi-compliant Log-Mel filterbank features (80 channels, 16kHz sampling frequency).
  * While `kaldifeat` is the Xiaomi/icefall default feature computer, it relies on compiled C++ extensions. Compiling `kaldifeat` is highly non-portable and frequently fails on CPU-only target environments without specific build tools and library pathways.
  * `torchaudio.compliance.kaldi.fbank` implements the identical Kaldi Fbank computation algorithm, is written in C++/Python bindings shipped directly with `torchaudio` binaries, requires no custom compilation, and produces mathematically equivalent output features.
* **Alternatives Considered**:
  * **`kaldifeat`**: Rejected due to high deployment overhead and compile-time dependency chain.
  * **`librosa.feature.melspectrogram`**: Rejected because its Mel-filterbank bank layout and windowing logic differ from Kaldi compliance, leading to mismatch with features used to train the Zipformer model and degraded accuracy.

### Configuration Mapping Table

| Parameter (Kaldi / kaldifeat) | Parameter (`torchaudio.compliance.kaldi.fbank`) | Serving Value |
| --- | --- | --- |
| `dither` | `dither` | `0.0` |
| `snip_edges` | `snip_edges` | `False` |
| `samp_freq` | `sample_frequency` | `16000.0` |
| `num_bins` | `num_mel_bins` | `80` |
| `high_freq` | `high_freq` | `7600.0` (Nyquist 8000Hz minus 400Hz) |
| `low_freq` | `low_freq` | `20.0` |

---

## 2. Model Checkpoint Strategy

* **Decision**: Use `viet_iter3_pseudo_label/exp/jit_script.pt` as the serving checkpoint.
* **Rationale**:
  * `epoch-12.pt` is the raw training state dict (requires instantiating the full `AsrModel` class from `ASR/zipformer/train.py`, with transitive dependencies on `lhotse`, `icefall`, `optim`, `kaldifeat`, etc.).
  * `jit_script.pt` is the **TorchScript-exported** version of the same `epoch-12.pt` weights. It is loaded with the simple `torch.jit.load()` call and is self-contained — no training dependencies needed.
  * Verified empirically: `jit_script.pt` correctly transcribes real Vietnamese audio from `input-test/video-test-001-small.wav` using greedy search.
* **Alternatives Considered**:
  * **Loading `epoch-12.pt` with `get_model()` from `train.py`**: Rejected because it requires the entire training dependency chain (`lhotse`, `icefall`, `optim`, `kaldifeat`) to be installed on the serving host, making deployment fragile and non-portable.

---

## 3. Session-Based Streaming Execution

* **Decision**: Implement a dual-mode `DecodeStream` class in the serving model wrapper:
  1. **Non-Causal Mode (Accumulated Batch Decoding)**: Buffers raw incoming float32 WAV samples in the session state. On each chunk arrival, it runs the entire accumulated audio through the offline model.
  2. **Causal Mode (Incremental Chunk Decoding)**: If the loaded JIT model is causal (defines `get_init_states` and chunk parameters), it computes features incrementally and maintains state caches (`states`, `hyp`, `decoder_out`) across chunk frames.
* **Rationale**:
  * The default pre-trained model checkpoint provided (`viet_iter3_pseudo_label/exp/jit_script.pt`) is non-causal (offline/batch). It does not expose `get_init_states` or streaming variables on its encoder.
  * Attempting true causal frame-by-frame streaming with a non-causal model is mathematically impossible and raises PyTorch attributes errors.
  * Dual-mode `DecodeStream` ensures that the server can load and stream *any* checkpoint, falling back to accumulated offline forward passes for batch models, which provides perfect transcript accuracy and low latency for standard recording sessions.
* **Alternatives Considered**:
  * **Strict Causal-only serving**: Rejected because the required causal JIT checkpoint (`jit_script_chunk_16_left_128.pt`) is not present in the pre-trained exp directory.
  * **Pure HTTP REST with no WebSocket fallback**: Rejected as it violates the P2 user story requirement for real-time speech feedback.

---

## 3. Asynchronous Non-Blocking Model Serving

* **Decision**: Offload model forward passes and search decoders to standard Python thread executors (`asyncio.get_running_loop().run_in_executor(None, ...)`).
* **Rationale**:
  * PyTorch model forward passes and k2/BPE search decoders are CPU-bound and highly intensive operations.
  * Running them directly in the FastAPI route handlers would block the single-threaded Python asynchronous event loop, preventing the server from handling incoming network frames, WebSocket packets, or concurrent requests.
  * Using `run_in_executor` shifts CPU-heavy calculations to thread pools, keeping the FastAPI event loop unblocked and compliant with Constitution Principle II (Asynchronous I/O & Non-Blocking Design).
* **Alternatives Considered**:
  * **Direct Synchronous Calls**: Rejected due to blocking the main event loop, causing severe latency spikes under concurrent loads.
  * **Celery / Process Queues**: Rejected for the proof-of-concept phase as it adds unnecessary infrastructure complexity (Redis/RabbitMQ) when a thread executor handles process separation sufficiently.
