# Deployment Guidelines & Logging Specifications

This document contains instructions for deploying the VietASR POC services and outlines the logging conventions enforced across backend components.

---

## 1. Local Deployment with Docker Compose

To run the supporting infrastructure (PostgreSQL & MinIO) in the background:

```bash
# Start MinIO and Postgres containers
docker-compose -f docker/docker-compose.yml up -d
```

### 1.1. Health Checks
- **MinIO Console**: Navigate to `http://localhost:9001` (Credentials: `minioadmin` / `minioadmin`).
- **PostgreSQL Connection**: Verify connectivity on port `5432` with username `postgres` and password `postgres`.

---

## 2. Production Build and Run

### 2.1. Web Backend (Gateway API)
Use Gunicorn with Uvicorn workers for high-concurrency production deployments:

```bash
cd apps/web-backend
uv run gunicorn src.main:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000
```

### 2.2. ASR Inference Server
Run the ASR server with a production-ready server engine:

```bash
cd apps/asr-server
uv run uvicorn src.main:app --host 0.0.0.0 --port 8001
```

### 2.3. React Frontend (Web Frontend)
Build static HTML/JS/CSS assets and serve them using Nginx:

```bash
cd apps/web-frontend
npm run build
```

---

## 3. Logging Specifications

Both `web-backend` and `asr-server` implement logging matching standard principles (Constitution Principle V):
- **No stdout print statements**: Use `logging.getLogger(__name__)`.
- **Log level structure**:
  - `DEBUG`: Internal processing milestones (e.g., audio transcoding chunks, WS frame counts).
  - `INFO`: Business events (e.g., successful upload, connection accepted, handshake succeeded).
  - `WARNING`: Recoverable errors (e.g., bad client configs, connection retry attempt, minor exceptions).
  - `ERROR`: Unrecoverable errors (e.g., ASR model server down, DB connection failure).

### 3.1. Standard Log Layout
The default format is:
`[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s`

---

## 4. Troubleshooting and Connection Recovery

- **Invalid Audio Formats**: Handled gracefully by returning an HTTP 400 response with supported format options (`.wav`, `.mp3`, etc.).
- **ASR Client Disconnections**: The HTTP client retries requests on connection drops twice using an exponential backoff formula (`0.5 * 2^attempt`), logging a `WARNING` during retries and raising `ASRServerConnectionError` if recovery fails.
- **WebSocket Timeout**: Tunnels close with status code `1008` (Policy Violation) if the client does not send a valid handshake text frame within 5.0 seconds.
