# VietASR Proof of Concept (POC) System

This directory contains the Proof of Concept (POC) implementation of the **VietASR** core system, demonstrating industry-level Vietnamese Speech Recognition. It is designed using Clean Architecture principles, ensuring scalability, robust error recovery, and clear separation of concerns.

---

## 1. System Architecture

The VietASR POC is built with a decoupled monorepo structure:

```mermaid
graph TD
    Client[React Web Frontend] <-->|HTTP / WebSocket| Gateway[FastAPI Web Backend Gateway]
    Gateway <-->|PostgreSQL| DB[(PostgreSQL Database)]
    Gateway <-->|S3 API| Storage[(MinIO Object Storage)]
    Gateway <-->|HTTP / WebSocket| ASRServer[FastAPI ASR Model Server]
```

- **web-frontend (`apps/web-frontend`)**: A high-performance React SPA built with Vite and TypeScript. It features a modern, dark-themed dashboard supporting batch audio upload with interactive word-level alignments and live real-time microphone streaming.
- **web-backend (`apps/web-backend`)**: The gateway backend implemented in FastAPI using Python Clean Architecture. It manages FFmpeg transcoding, handles PostgreSQL database logging, relays storage to MinIO, and tunnels WebSockets to the model server.
- **asr-server (`apps/asr-server`)**: The dedicated FastAPI model inference server. It dynamically maps decoding parameters, loads PyTorch/k2 checkpoits, and implements batch & bidirectional WebSocket decoding endpoints.

---

## 2. Directory Structure

```
POC/
├── apps/
│   ├── asr-server/       # FastAPI ASR checkpoint inference server
│   ├── web-backend/      # Clean Architecture gateway backend
│   └── web-frontend/     # React / Vite / TypeScript single-page app
├── docker/
│   └── docker-compose.yml # PostgreSQL and MinIO container configuration
├── docs/
│   └── deployment.md     # Deployment configurations and log specifications
├── scripts/
│   └── clean.sh          # Pre-commit python and C++ cache cleanup script
└── specs/
    └── 001-vietasr-core-system/ # Project contracts, schemas, and specs
```

---

## 3. Configuration & Environment Variables

Create `.env` files inside respective service folders to configure secret keys and endpoints.

### 3.1. Web Backend Configuration (`apps/web-backend/.env`)
```ini
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/vietasr
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_SECURE=False
MINIO_BUCKET_RAW=raw-uploads
MINIO_BUCKET_STANDARD=standardized-wavs
ASR_SERVER_URL=http://localhost:8001
ALLOWED_CORS_ORIGINS=http://localhost:5173,http://localhost:3000
MAX_FILE_SIZE_BYTES=52428800
```

### 3.2. ASR Model Server Configuration (`apps/asr-server/.env`)
```ini
ASR_SERVER_PORT=8001
# Model checkpoint directories
CHECKPOINT_DIR=models/checkpoints
```

---

## 4. Setup & Running the POC

Refer to [quickstart.md](specs/001-vietasr-core-system/quickstart.md) for step-by-step commands to spin up PostgreSQL, MinIO, and execute E2E validation scenarios.

---

## 5. Verification and Quality Assurance

The system maintains 100% test coverage using standard testing tools:
- **Backend Tests**: Verify adapters, databases, usecases, and controllers using `pytest` and `pytest-asyncio`. Run inside the app directories:
  ```bash
  PYTHONPATH=. uv run python -m pytest
  ```
- **Frontend Tests**: Validate services and clients using `vitest`. Run in `apps/web-frontend/`:
  ```bash
  npm run test
  ```
- **Frontend Build**: Verify TypeScript compilation:
  ```bash
  npm run build
  ```
