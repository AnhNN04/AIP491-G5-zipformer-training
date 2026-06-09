# Implementation Plan: VietASR Core System

**Branch**: `001-vietasr-core-system` | **Date**: 2026-06-09 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `specs/001-vietasr-core-system/spec.md`

**Note**: This template is filled in by the `/speckit-plan` command. See `.specify/templates/plan-template.md` for the execution workflow.

## Summary

The VietASR Core System provides end-to-end automatic speech recognition (ASR) capability for Vietnamese. It enables batch uploads of multi-format audio/video files (converting them on-the-fly via FFmpeg into standard WAV format), stores files in MinIO storage, logs transaction details in PostgreSQL, and routes inference to an isolated ASR Model Server. It also supports dynamic configuration parameters and low-latency real-time streaming via WebSockets.

---

## Technical Context

**Language/Version**: Python >= 3.11, TypeScript (frontend)

**Primary Dependencies**: FastAPI, PyTorch, k2, React (Vite), TailwindCSS

**Storage**: PostgreSQL (SQLAlchemy ORM), MinIO (S3-compatible object storage)

**Testing**: pytest (backend), Vitest (frontend)

**Target Platform**: Linux containers (Docker Compose), Modern web browsers

**Project Type**: Web Application (Monorepo with separated backend, frontend, and model server apps)

**Performance Goals**: Audio transcoding < 2.0s per minute of audio, decoding config propagation < 100ms, streaming update latency < 500ms.

**Constraints**: Max file upload size 50MB, entry-point rate limiting, process separation of machine learning inference from the web thread, all PostgreSQL/MinIO/ASR I/O must be async/non-blocking.

**Scale/Scope**: Able to handle up to 50 concurrent transcription requests under nominal load.

---

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Gate / Principle | Standard | Status | Justification / Notes |
|---|---|---|---|
| I. Clean Architecture | Use cases must be independent of FastAPI & SQLAlchemy | PASS | Domain models and use cases are isolated from frameworks and adapters. |
| I. Process Separation | Model server must run as a separate OS process | PASS | Model Server runs as an independent FastAPI container/process. |
| II. Asynchronous I/O | Async/Await for PostgreSQL, MinIO, and ASR Client | PASS | All I/O is non-blocking via `asyncpg` and S3 async libraries. |
| III. Audio Pipeline | Auto-transcode to 16kHz, mono, 16-bit WAV via FFmpeg | PASS | Integrates async FFmpeg subprocesses for automated transcoding. |
| III. Dynamic Config | Support runtime parameter injection via REST and WS | PASS | Decoder reads runtime parameters on a per-request basis. |
| IV. Security & CORS | `.env` variables used, strict CORS, 50MB size limits | PASS | No hardcoded secrets, limits enforced at backend router gates. |
| V. Strict Typing | Python Type Hints, TypeScript Strict mode | PASS | Enforced in code validation and pre-commit checks. |
| V. Standards & Logging | Standard `logging` library, no `print()` statements | PASS | All console/file outputs routed through Python's `logging`. |

---

## Project Structure

### Documentation (this feature)

```text
specs/001-vietasr-core-system/
├── spec.md              # Feature specification
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 data design
├── quickstart.md        # Phase 1 validation/run guide
├── contracts/           # API contract definitions
│   ├── rest-api.md
│   └── websocket-api.md
└── checklists/          # Requirement quality verification
    └── requirements.md
```

### Source Code (repository root)

```text
apps/
├── asr-server/           # Independent ASR Model Serving FastAPI application
│   ├── src/
│   │   ├── main.py       # Inference server entry point
│   │   └── models/       # PyTorch/k2 loader and inference logic
│   └── pyproject.toml
├── web-backend/          # Clean Architecture Web backend FastAPI application
│   ├── src/
│   │   ├── domain/       # Entities and repository interfaces (independent)
│   │   ├── usecases/     # Business logic (transcribe, transcode, stats)
│   │   ├── adapters/     # Database, Storage, and ASR Client adapters
│   │   └── frameworks/   # FastAPI main, routes, and DB session configs
│   └── pyproject.toml
└── web-frontend/         # React SPA frontend (Vite + Tailwind + TypeScript)
    ├── src/
    │   ├── components/   # Audio Uploader, Streamer, Config Panel
    │   ├── pages/        # Dashboard, History, Live
    │   └── services/     # API/WebSocket client layer
    └── package.json
docker/
└── docker-compose.yml    # PostgreSQL and MinIO container configurations
```

**Structure Decision**: Monorepo structure with separated, independent applications under `apps/` to isolate frontend, backend, and machine learning components.

---

## Complexity Tracking

No violations of the Constitution or plan guidelines are present.
