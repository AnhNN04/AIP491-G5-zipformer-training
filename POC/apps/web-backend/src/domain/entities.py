from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID
from typing import Optional, List, Dict, Any

@dataclass
class User:
    username: str
    password_hash: str
    id: Optional[UUID] = None
    created_at: Optional[datetime] = None

@dataclass
class AudioFile:
    raw_storage_path: str
    standard_storage_path: str
    original_format: str
    duration_seconds: float
    size_bytes: int
    user_id: Optional[UUID] = None
    id: Optional[UUID] = None
    created_at: Optional[datetime] = None

@dataclass
class DecodingConfig:
    method: str
    causal: bool = False
    beam_size: Optional[int] = None
    chunk_size: Optional[int] = None
    left_context_frames: Optional[int] = None
    id: Optional[UUID] = None
    created_at: Optional[datetime] = None

@dataclass
class Transcription:
    audio_file_id: UUID
    decoding_config_id: UUID
    text: str
    confidence: float
    word_alignments: Optional[List[Dict[str, Any]]] = None
    id: Optional[UUID] = None
    created_at: Optional[datetime] = None

@dataclass
class DialectStats:
    audio_file_id: UUID
    inferred_dialect: str
    dialect_probability: float
    wer_estimate: Optional[float] = None
    id: Optional[UUID] = None
    created_at: Optional[datetime] = None
