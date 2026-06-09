# Technical Research: VietASR Core System

This document captures the key architectural and technological decisions made for the VietASR Proof of Concept (POC) system, outlining the rationale and alternatives evaluated.

---

## 1. Process Separation: Web Backend vs. Model Server

- **Decision**: The machine learning model serving layer (loading PyTorch and `k2` checkpoints) MUST run as an independent FastAPI process completely decoupled from the Web Backend process. The two services communicate via asynchronous REST APIs and WebSockets.
- **Rationale**: Python's Global Interpreter Lock (GIL) and asyncio event loop can be easily blocked by CPU-bound or GPU-bound workloads like deep learning inference. Separating the two components ensures that the Web Backend remains responsive, handles concurrent uploads, serves UI requests, and performs database actions without experiencing event loop starvation.
- **Alternatives considered**: 
  - *Single Process with ThreadPoolExecutor*: Running inference in a background thread within the same FastAPI process. Rejected because the GIL would still restrict CPU concurrency, and high memory utilization of the models could lead to application crashes impacting the API gateway.

---

## 2. Asynchronous Audio Transcoding Pipeline via FFmpeg

- **Decision**: Integrate FFmpeg by spawning asynchronous subprocesses (`asyncio.create_subprocess_exec`) in the Web Backend. Incoming files are transcoded on-the-fly to the target ASR format: 16kHz, single-channel (mono), 16-bit PCM WAV.
- **Rationale**: FFmpeg is highly optimized, widely compatible with video and audio container formats (mp3, mp4, flac, etc.), and supports streaming pipelines. Spawning it asynchronously avoids blocking the main Python process while maintaining low CPU overhead.
- **Alternatives considered**:
  - *Python Libraries (Librosa, Soundfile)*: Relying on native Python sound decoding libraries. Rejected because they lack native support for video containers like `.mp4` and have high memory overhead when loading entire files.

---

## 3. High-Performance Asynchronous Data Storage (PostgreSQL & MinIO)

- **Decision**: Use PostgreSQL for relational metadata (users, histories, dialect stats) queried via SQLAlchemy using the asynchronous `asyncpg` driver. Use MinIO (S3-compatible object storage) running inside Docker Compose for storing both raw user uploads and standardized audio.
- **Rationale**: Relational data requires transactional integrity, while audio binaries require scalable, non-blocking file storage. Using SQLAlchemy's `async` engine combined with MinIO's async S3 client ensures all I/O is non-blocking.
- **Alternatives considered**:
  - *MongoDB*: Using a single document database for both metadata and binary files (GridFS). Rejected because relational structures are superior for tracking routing statistics across dialects and user history, and S3-compatible storage is the industry standard for production-grade media assets.

---

## 4. Dynamic ASR Configuration Protocol

- **Decision**: Support dynamic configuration parameters (greedy vs. beam search, beam size, causal flag, chunk size) passed directly within the JSON request body for REST API calls and inside a handshake frame for WebSocket sessions.
- **Rationale**: This allows client-side flexibility to tune accuracy-vs-speed trade-offs on-the-fly per request or stream, without restarting the server or reloading model checkpoints.
- **Alternatives considered**:
  - *Stateful API Endpoints*: Updating decoding settings via a configuration endpoint and persisting them in session state. Rejected because it introduces race conditions when multiple clients execute concurrent decodings under different parameter expectations.

---

## 5. WebSocket-Based Real-Time Streaming ASR

- **Decision**: Use WebSockets for continuous, bidirectional, low-latency audio chunk streaming and transcript delivery. The React frontend streams PCM audio packages, the Web Backend relays them, and the Model Server returns incremental transcript tokens.
- **Rationale**: WebSockets maintain a single TCP connection, reducing protocol overhead and achieving the sub-500ms latency goal for real-time speech feedback.
- **Alternatives considered**:
  - *HTTP Long Polling / SSE (Server-Sent Events)*: SSE is unidirectional (server to client) and lacks a standard mechanism for uploading binary streams concurrently, making it unsuitable for real-time bidirectional speech interfaces.
