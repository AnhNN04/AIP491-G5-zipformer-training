# Quickstart Validation: VietASR Core System

This guide outlines the validation scenarios and commands required to prove the core system works end-to-end.

---

## 1. Prerequisites

- **Docker & Docker Compose** installed.
- **FFmpeg** installed locally.
- **Python >= 3.11** with dependencies installed (or virtual environment activated).

---

## 2. Infrastructure Startup

Provision the local PostgreSQL and MinIO servers:

```bash
# From the repository root, start the local docker containers
docker-compose -f docker/docker-compose.yml up -d
```

Verify that local storage buckets are created automatically or configure them manually inside the MinIO console (`http://localhost:9001`).

---

## 3. Starting App Components

Start the processes in separate terminals to ensure process separation:

### 3.1. Terminal 1: ASR Model Server
```bash
# Start the machine learning inference server
cd apps/asr-server
python -m uvicorn src.main:app --host 127.0.0.1 --port 8001 --reload
```

### 3.2. Terminal 2: Web Backend
```bash
# Start the Web API Backend
cd apps/web-backend
python -m uvicorn src.main:app --host 127.0.0.1 --port 8000 --reload
```

### 3.3. Terminal 3: React Frontend
```bash
# Start the React client development server
cd apps/web-frontend
npm run dev
```

---

## 4. End-to-End Validation Scenarios

Refer to the database schema in [data-model.md](data-model.md) and API contracts in [rest-api.md](contracts/rest-api.md) for data schemas.

### 4.1. Scenario 1: Batch Upload and Audio Transcoding (P1 Verification)
Upload an audio file in a non-standard format (e.g. an MP3 file) and verify that the system transcodes it, stores the assets in MinIO, logs metadata in Postgres, and returns the transcript.

```bash
# Send a POST upload request with a 1-minute MP3 file
curl -X POST http://127.0.0.1:8000/api/upload \
  -F "file=@/path/to/test-audio.mp3" \
  -F 'config={"method":"greedy_search"}'
```

**Expected Results**:
- An HTTP status `200 OK` is returned in under 2.0 seconds.
- The response JSON structure matches the contract in [rest-api.md](contracts/rest-api.md#12-success-response-http-200-ok).
- A raw file (`test-audio.mp3`) and a standardized file (`standardized-*.wav`) are uploaded to MinIO.

---

### 4.2. Scenario 2: Dynamic Decoding Parameter Verification (P2 Verification)
Send a batch request specifying beam search with a customized beam size to verify parameter propagation.

```bash
# Request transcription using modified beam search with beam size of 4
curl -X POST http://127.0.0.1:8000/api/upload \
  -F "file=@/path/to/test-audio.wav" \
  -F 'config={"method":"modified_beam_search", "beam_size": 4}'
```

**Expected Results**:
- The ASR Model Server logs indicate that the `modified_beam_search` algorithm was executed using `beam_size = 4` for the request.

---

### 4.3. Scenario 3: Real-Time WebSocket Streaming (P3 Verification)
Verify connection establishment and incremental transcript returns. Use a WebSocket testing CLI (e.g., `websocat`).

```bash
# Establish a WebSocket connection to the streaming endpoint
websocat ws://127.0.0.1:8000/api/stream
```

1. **Send Handshake Config Frame (Text)**:
   ```json
   {"event": "handshake", "config": {"method": "greedy_search", "causal": true}}
   ```
   *Expected Response*: `{"event": "handshake_ok", "session_id": "..."}`

2. **Stream Binary Data**:
   Send raw PCM audio bytes.
   *Expected Response*: Incremental `transcript_update` event frames containing recognized text segments.

3. **Send Stop Frame (Text)**:
   ```json
   {"event": "stop"}
   ```
   *Expected Response*: `{"event": "finished", "full_transcript": "..."}` and the connection is closed.
