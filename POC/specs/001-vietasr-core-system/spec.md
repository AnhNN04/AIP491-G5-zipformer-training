# Feature Specification: VietASR Core System

**Feature Branch**: `001-vietasr-core-system`

**Created**: 2026-06-09

**Status**: Draft

**Input**: User description: "Create spec.md for VietASR POC project in directory POC/. Analyze functional and architectural requirements from docs/poc-implementation-plan.md, and strictly adhere to POC/.specify/memory/constitution.md. Focus on Multi-format Batch Audio Upload & Transcription (P1), Dynamic Decoding Configuration (P2), Real-Time Streaming Speech Recognition (P3), Edge Cases (corrupted audio, DoS limits, disconnects), Functional Requirements (transcoding, MinIO, Postgres, process isolation), Key Entities, and Measurable Success Criteria."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Multi-format Batch Audio Upload & Transcription (Priority: P1)
As a user, I want to upload audio files in various formats (such as WAV, MP3, FLAC, M4A, OGG, or MP4 video) so that the system can automatically standardize the audio format, store it securely, and return the transcribed Vietnamese text.

**Why this priority**: This represents the foundational MVP (Minimum Viable Product) capability. Users must be able to perform speech-to-text on pre-recorded audio files before any other features (like configuration tuning or real-time streaming) can be useful.

**Independent Test**: Can be fully tested by selecting a non-WAV audio/video file (e.g., a `.mp3` or `.mp4`), submitting it, and verifying that:
1. The system accepts the file.
2. The transcribed text is returned correctly.
3. Both the original file and the transcoded standard WAV file are accessible in object storage.

**Acceptance Scenarios**:

1. **Given** a user is logged in and views the batch transcription page, **When** they upload a valid 5MB `.mp3` file, **Then** the system transcodes the file to 16kHz mono 16-bit PCM WAV, saves both original and standard WAV files to object storage, logs the transaction in the metadata database, and returns the transcribed Vietnamese text.
2. **Given** a user is logged in and views the batch transcription page, **When** they upload a valid `.mp4` video file, **Then** the system extracts the audio track, transcodes it to 16kHz mono 16-bit PCM WAV, saves the files to object storage, and returns the transcribed Vietnamese text.

---

### User Story 2 - Dynamic Decoding Configuration (Priority: P2)
As a developer or advanced user, I want to adjust ASR decoding parameters (such as search method, beam size, causal decoding, chunk size, and left context frames) on the user interface so that I can optimize the balance between transcription accuracy and decoding speed dynamically.

**Why this priority**: Essential for experimenting with different models and deployment configurations. Dynamic runtime application avoids server downtime and enables multi-tenant optimization.

**Independent Test**: Can be fully tested by initiating two transcription requests with different search methods (e.g., greedy search vs. beam search with beam size 4) and verifying that the backend applies the parameters instantly to the inference model at runtime.

**Acceptance Scenarios**:

1. **Given** the user selects the 'beam search' method with a beam size of 4 on the configuration interface, **When** they upload an audio file for batch transcription, **Then** the backend includes the parameters in the REST API request, and the ASR Model Server applies them to the decoding algorithm at runtime without restarting.
2. **Given** a user opens a real-time streaming session, **When** they connect via WebSocket and transmit the configuration handshake frame containing causal: true, chunk_size: 16, and left_context_frames: 128, **Then** the backend applies these settings to all subsequent audio chunks processed in that session.

---

### User Story 3 - Real-Time Streaming Speech Recognition (Priority: P3)
As a user, I want to speak directly into my microphone so that the system streams my audio in real-time and displays transcription updates incrementally as I talk.

**Why this priority**: Highly valuable for interactive applications (e.g., dictation, live captioning), but depends on the foundational ASR pipeline (P1) and real-time/causal parameters (P2).

**Independent Test**: Can be fully tested by opening the live streaming page, speaking into the microphone, and verifying that transcription segments appear on screen within sub-second latency from speaking.

**Acceptance Scenarios**:

1. **Given** the user has granted microphone access, **When** they click "Start Streaming" and speak, **Then** the system establishes a WebSocket connection, transmits audio chunks continuously, and displays incremental transcribed text updates on screen.

---

### Edge Cases
- **Corrupted or Unsupported Audio File**: If a user uploads an audio file that is corrupted, has unreadable headers, or is in an unsupported format, the system MUST reject the file immediately, abort transcoding, and return a clear, user-friendly error message indicating that the file is invalid.
- **Denial of Service (DoS) Attempts**: If a user attempts to upload a file exceeding 50MB, or exceeds 60 API requests/minute, the system MUST block the request at the entry gateway and return a client-side warning (HTTP 413 or 429) without processing the request.
- **ASR Model Server Interruption**: If the connection between the Web Backend and the ASR Model Server is lost during a request, the system MUST retry the connection twice and, if still unsuccessful, return a temporary service unavailable notification (HTTP 503) without crashing the Web Backend or database.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST support user uploads of audio/video files in `.wav`, `.mp3`, `.flac`, `.m4a`, `.ogg`, and `.mp4` formats.
- **FR-002**: The system MUST automatically detect file headers and MIME types to validate upload integrity before processing.
- **FR-003**: The Web Backend MUST spawn a non-blocking process (using FFmpeg) to transcode all uploaded audio files on-the-fly to a standard format: 16kHz, single-channel (mono), 16-bit PCM WAV.
- **FR-004**: The system MUST store both the original uploaded file and the standardized WAV file in separate buckets in object storage.
- **FR-005**: The system MUST record user details, audio file metadata, and transcription texts in the database.
- **FR-006**: The system MUST run the machine learning ASR Model Server in a separate OS process from the Web Backend to prevent heavy inference workloads from blocking the asynchronous web request loop.
- **FR-007**: The system MUST support dynamic, runtime decoding parameters (`method`, `beam_size`, `causal`, `chunk_size`, `left_context_frames`) passed via REST payload or WebSocket handshake, applying them to k2 decoding algorithms at runtime without restarting the server.
- **FR-008**: The system MUST support real-time audio chunk streaming and incremental transcript delivery via a WebSocket endpoint.
- **FR-009**: The Web Backend and Model Server MUST manage database credentials and keys via environment variables (e.g., `.env`), with CORS restricted specifically to authorized frontend, backend, and model server domains.
- **FR-010**: The system MUST enforce a 50MB file size ceiling and a rate-limiting policy at the Web Backend entry point.
- **FR-011**: All backend database queries, object storage access, and cross-server requests MUST run asynchronously using non-blocking I/O.
- **FR-012**: The system MUST output all messages via a standard logging framework; raw print statements are prohibited.

### Key Entities *(include if feature involves data)*

- **User**: Represents system users, containing authentication credentials, registration metadata, and historical records.
- **AudioFile**: Represents uploaded audio files, containing raw storage reference, standardized storage reference, original format, duration, size, and upload timestamp.
- **Transcription**: Represents speech recognition outputs, containing references to the target AudioFile, transcribed Vietnamese text, confidence scores, word-level alignment arrays, and configuration options used.
- **DecodingConfig**: Represents the parameter set used for decoding, including search method (greedy/beam search), beam size, causal flags, chunk size, and left context frames.
- **DialectStats**: Tracks metadata and performance metrics (e.g., WER, duration) aggregated by Vietnamese dialect/accent (Northern, Central, Southern) to optimize routing.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: The system MUST transcode a 1-minute audio file into the standard WAV format in under 2.0 seconds.
- **SC-002**: The system MUST propagate dynamic decoding parameters from the user interface to the ASR Model Server at runtime in under 100 milliseconds.
- **SC-003**: The real-time streaming endpoint MUST achieve an incremental transcription update latency (Time-to-First-Token) of less than 500 milliseconds.
- **SC-004**: The system MUST enforce security limits at the gateway, rejecting files larger than 50MB in under 500 milliseconds with an HTTP 413 error code.
- **SC-005**: The system MUST handle at least 50 concurrent transcription requests under nominal load without database connection exhaustion or server event-loop blocking.

## Assumptions

- Users have client devices equipped with working microphones and modern browsers supporting WebSockets and Web Audio API for streaming.
- The original audio uploads contain audible speech in Vietnamese.
- Database access credentials and S3 bucket secrets are safely provisioned in the execution environments.
