# Feature Specification: Real ASR Serving Integration

**Feature Branch**: `002-real-asr-integration`

**Created**: 2026-06-09

**Status**: Draft

**Input**: User description: "Extract and integrate the real model core from the ASR/ and SSL/ folders into the asr-server to execute real inference (no mocks) for both HTTP REST and WebSocket Streaming."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Batch Speech Recognition with Real Model (Priority: P1)

Users upload an audio file (e.g. MP3, WAV) from the web interface, and the system automatically computes features and performs speech recognition using the real Zipformer model checkpoints, returning the correct transcript and word alignments.

**Why this priority**: This is the core functionality of the ASR system. It validates model loading, feature extraction, and correct database logging.

**Independent Test**: Can be tested independently by issuing a POST upload request with a standard WAV file to the `/api/upload` endpoint, verifying that the returned transcript matches the output of `jit_pretrained.py`.

**Acceptance Scenarios**:

1. **Given** the user has configured "Greedy Search", **When** they upload a WAV file with Vietnamese speech, **Then** the system returns the correct transcript, a "NORTHERN" inferred dialect, and the corresponding confidence.
2. **Given** the user has configured "Modified Beam Search" with a beam size of 4, **When** they upload a WAV file, **Then** the Zipformer model executes beam search decoding and returns the optimal text transcription.

---

### User Story 2 - Real-Time Speech Recognition with Real Model (Priority: P2)

Users speak into their microphone on the web interface, and the client streams raw audio buffers to the server via WebSockets, receiving and rendering incremental transcript updates on the screen in real-time.

**Why this priority**: Provides low-latency feedback and verifies asynchronous stream processing and session state persistence.

**Independent Test**: Can be verified by establishing a WebSocket connection to the `/api/stream` endpoint, completing the handshake configuration, and sending binary PCM chunks to receive live transcript frames.

**Acceptance Scenarios**:

1. **Given** the client has successfully completed the streaming handshake, **When** the client streams raw int16 PCM chunks at 16kHz, **Then** the serving system maintains the session state and returns partial transcript updates.
2. **Given** the client sends a "stop" event frame, **When** the connection is open, **Then** the serving system decodes the final audio frames, returns the complete transcript summary ("finished" event), and closes the connection.

---

### Edge Cases

- **Sudden Disconnection**: If the WebSocket client disconnects abruptly without sending the "stop" frame, the serving server MUST immediately clean up the corresponding `DecodeStream` session object from memory to prevent memory leaks.
- **Handshake Timeout**: If the client connects but fails to transmit a valid "handshake" configuration text frame within 5.0 seconds, the server MUST close the connection with status code `1008` (Policy Violation).
- **Out of Bounds Parameters**: If the client sends invalid decoding parameters (e.g., `beam_size = 25` or an unsupported method), the server MUST immediately reject the request with HTTP 400 or close the WebSocket connection.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The ASR server MUST load the JIT TorchScript model checkpoint from the path configured in `ASR_CHECKPOINT_PATH` (defaulting to `viet_iter3_pseudo_label/exp/jit_script.pt`) at server startup.
- **FR-002**: The ASR server MUST load the BPE symbol table from `viet_iter3_pseudo_label/data/Vietnam_bpe_2000_new/tokens.txt` at server startup to map token IDs to text.
- **FR-003**: The ASR server MUST extract 80-dimensional log-Mel Fbank features using `torchaudio.compliance.kaldi.fbank` to ensure portability and avoid compiled C++ dependencies like `kaldifeat` on the host machine.
- **FR-004**: The ASR server MUST support feature extraction from both WAV file byte buffers (from REST uploads) and raw int16 PCM bytes (from WebSocket stream chunks).
- **FR-005**: For WebSocket streaming, the ASR server MUST manage dynamic session states by instantiating and caching a `DecodeStream` object for each active connection.
- **FR-006**: The serving model wrapper MUST execute the actual greedy search and modified beam search decoding algorithms ported from `ASR/zipformer`.

### Key Entities

- **ServingModel**: Represents the loaded Zipformer TorchScript JIT model.
- **SymbolTable**: Represents the vocabulary mapping table translating BPE token IDs to Vietnamese characters.
- **DecodeStream**: Represents the session-based streaming state tracker, caching hidden layers states and feature padding across chunks.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Feature extraction and model forward pass for a 10-second audio file MUST complete in under 1.0 second on CPU.
- **SC-002**: The real-time streaming endpoint MUST achieve an incremental update latency (Time-to-First-Token) of less than 350 milliseconds.
- **SC-003**: The system MUST dispose of the `DecodeStream` instance and free the memory in under 100 milliseconds after WebSocket closure.

## Assumptions

- The target server environment has PyTorch, Torchaudio, and k2 libraries pre-installed.
- Serving runs on CPU by default (with CUDA automatic fallback enabled).
- Checkpoint models and token configurations are present under the `viet_iter3_pseudo_label/` path.
