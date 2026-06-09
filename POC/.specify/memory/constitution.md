<!--
CONSTITUTION SYNC IMPACT REPORT
- Version change: None -> 1.0.0
- List of modified principles:
  - None -> I. Clean Architecture & Process Separation
  - None -> II. Asynchronous I/O & Non-Blocking Design
  - None -> III. Standardized Audio Pipeline & Dynamic Protocol
  - None -> IV. Robust Security & Environment Isolation
  - None -> V. Strict Typing & Standardized Logging
- Added sections:
  - Core Principles
  - Technology Standards & Core Components
  - Development Workflow & Compliance Gates
  - Governance
- Removed sections: None
- Templates requiring updates:
  - POC/.specify/templates/plan-template.md (✅ updated)
  - POC/.specify/templates/spec-template.md (✅ updated)
  - POC/.specify/templates/tasks-template.md (✅ updated)
- Follow-up TODOs: None
-->

# VietASR POC Constitution

## Core Principles

### I. Clean Architecture & Process Separation
The Web Backend MUST be developed using FastAPI (Python >= 3.11) and adhere strictly to Clean Architecture principles. Core business logic (entities and use cases) MUST remain entirely independent of external frameworks, libraries (such as FastAPI), and database ORMs (such as SQLAlchemy). Furthermore, the Machine Learning Model Server (Python/FastAPI loading PyTorch and `k2` checkpoints) MUST run as a separate, isolated process from the Web Backend to ensure CPU-heavy ASR inference does not block the web application's asynchronous event loop.

### II. Asynchronous I/O & Non-Blocking Design
All network and storage I/O operations within the system MUST be asynchronous. Developers MUST use Python's `async`/`await` patterns for all database queries via SQLAlchemy, file operations on MinIO, and network communications with the ASR Model Server. This prevents blocking calls from degrading concurrent request throughput.

### III. Standardized Audio Pipeline & Dynamic Protocol
The Web Backend MUST automatically transcode all uploaded audio files (supporting formats including `.wav`, `.mp3`, `.flac`, `.m4a`, `.ogg`, `.mp4`) to the target ASR format: 16kHz sample rate, single-channel (mono), 16-bit PCM WAV. Transcoding MUST be done on-the-fly using FFmpeg. To accommodate dynamic configuration, the ASR system MUST support runtime decoding parameters (e.g., method, beam size, causal configuration) passed dynamically via REST API body or WebSocket Handshake frames without requiring server reboots.

### IV. Robust Security & Environment Isolation
Sensitive configuration parameters, including database credentials and MinIO access keys, MUST be managed exclusively via `.env` files and environment variables, and MUST NEVER be hardcoded in the codebase or committed to Git. A strict CORS configuration MUST be enforced between the React frontend, FastAPI backend, and ASR Model Server. Additionally, file uploads MUST be validated, rate-limited, and size-capped at the gateway/backend level to mitigate potential Denial of Service (DoS) attacks.

### V. Strict Typing & Standardized Logging
The codebase MUST maintain high code quality through explicit type annotations: Python code MUST use Type Hints, and TypeScript frontend code MUST enforce Strict Types. To ensure system observability, all debugging, warning, and error outputs MUST use the standard Python `logging` library; the use of raw `print()` statements is strictly forbidden in production-bound code.

## Technology Standards & Core Components
- **Web Backend**: FastAPI (Python >= 3.11) implementing Clean Architecture.
- **Model Server**: Independent Python (FastAPI) instance for loading PyTorch/k2 checkpoints and executing inference.
- **Frontend**: React SPA (Vite) styled with TailwindCSS and implemented in TypeScript.
- **Database**: PostgreSQL relational database mapped via SQLAlchemy ORM.
- **Object Storage**: MinIO (S3-compatible) deployed via Docker Compose to manage original input audio files.
- **Audio Processing**: FFmpeg integration for automatic format detection and conversion.
- **ASR Configuration**: Real-time dynamic decoding parameters (greedy/beam search, beam size, etc.) passed via REST POST requests or WebSocket handshake frames.

## Development Workflow & Compliance Gates
- **Version Control**: Git repository management. Pre-commit hooks should check for raw print statements and missing type annotations.
- **Build Cleanliness**: Temporary build folders, C++ k2 caches, and Python bytecode (`__pycache__`) MUST be cleaned periodically and excluded from commits via `.gitignore`.
- **Documentation Sync**: When architectural or API protocol changes occur, system design documents (e.g., in `docs/`) MUST be updated concurrently before changes are merged.

## Governance
- This Constitution is the authoritative standard for the VietASR POC project.
- Modifications to this Constitution require updating the version according to Semantic Versioning (SemVer 2.0.0):
  - **MAJOR** bump: removal or backward-incompatible redefinition of a core principle.
  - **MINOR** bump: addition of a new principle or significant expansion of existing standards.
  - **PATCH** bump: minor clarifications, spelling corrections, or structural formatting.
- All code reviews and architectural plans must verify compliance with this Constitution. Any deviations or added complexities must be justified.

**Version**: 1.0.0 | **Ratified**: 2026-06-09 | **Last Amended**: 2026-06-09
