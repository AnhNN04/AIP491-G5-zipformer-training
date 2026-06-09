# REST API Contract: VietASR Core System

This document defines the HTTP API endpoint contract for batch audio upload and transcription.

---

## 1. Batch Upload & Transcribe

Initiates a synchronous transcode, storage, and ASR inference pipeline for an uploaded audio or video file.

- **URL**: `/api/upload`
- **Method**: `POST`
- **Content-Type**: `multipart/form-data`

### 1.1. Request Parameters

| Field | Type | Required | Description |
|---|---|---|---|
| `file` | Binary (File) | Yes | The audio/video file. Supported formats: `.wav`, `.mp3`, `.flac`, `.m4a`, `.ogg`, `.mp4`. Maximum size: 50MB. |
| `config` | JSON String | No | A JSON string specifying the decoding parameters. (Default: greedy search). |

#### `config` JSON Schema:
```json
{
  "type": "object",
  "properties": {
    "method": {
      "type": "string",
      "enum": ["greedy_search", "modified_beam_search"],
      "default": "greedy_search"
    },
    "beam_size": {
      "type": "integer",
      "minimum": 1,
      "maximum": 20,
      "default": 4
    },
    "causal": {
      "type": "boolean",
      "default": false
    }
  },
  "required": ["method"]
}
```

### 1.2. Success Response (HTTP 200 OK)

Returns standard JSON containing storage references, transcription text, word alignment tokens, and dialect classification.

- **Content-Type**: `application/json`

```json
{
  "audio_id": "7a3556d1-1311-4f35-94f7-873d63b65ef3",
  "duration_seconds": 2.45,
  "size_bytes": 1048576,
  "status": "SUCCESS",
  "transcription": {
    "text": "chào mừng bạn đến với hệ thống nhận dạng giọng nói",
    "confidence": 0.965,
    "word_alignments": [
      { "word": "chào", "start": 0.12, "end": 0.35, "conf": 0.99 },
      { "word": "mừng", "start": 0.35, "end": 0.62, "conf": 0.98 },
      { "word": "bạn", "start": 0.62, "end": 0.85, "conf": 0.97 },
      { "word": "đến", "start": 0.85, "end": 1.10, "conf": 0.96 },
      { "word": "với", "start": 1.10, "end": 1.35, "conf": 0.95 },
      { "word": "hệ", "start": 1.35, "end": 1.60, "conf": 0.96 },
      { "word": "thống", "start": 1.60, "end": 1.95, "conf": 0.95 }
    ]
  },
  "dialect": {
    "inferred": "NORTHERN",
    "probability": 0.92
  }
}
```

### 1.3. Error Responses

#### HTTP 400 Bad Request (Invalid Format or Parameters)
Returned if the file format is unsupported, headers are corrupted, or ASR config parameters are out of range.
```json
{
  "error": "BAD_REQUEST",
  "message": "Unsupported file format. Supported formats: .wav, .mp3, .flac, .m4a, .ogg, .mp4"
}
```

#### HTTP 413 Payload Too Large
Returned if the uploaded file exceeds 50MB.
```json
{
  "error": "PAYLOAD_TOO_LARGE",
  "message": "File size exceeds the 50MB maximum limit."
}
```

#### HTTP 429 Too Many Requests
Returned if the client exceeds the rate limit of 60 requests per minute.
```json
{
  "error": "RATE_LIMIT_EXCEEDED",
  "message": "Rate limit exceeded. Please wait before submitting another request."
}
```

#### HTTP 503 Service Unavailable
Returned if the backend cannot connect to the ASR Model Server.
```json
{
  "error": "SERVICE_UNAVAILABLE",
  "message": "ASR Inference Server is temporarily unreachable. Please try again."
}
```
