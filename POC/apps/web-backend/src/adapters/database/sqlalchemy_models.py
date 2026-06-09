import uuid
from datetime import datetime
from typing import Optional, List
from sqlalchemy import (
    Column,
    String,
    Integer,
    Float,
    Boolean,
    ForeignKey,
    DateTime,
    BigInteger,
    JSON,
    UUID as SQLUUID
)
from sqlalchemy.orm import relationship, Mapped, mapped_column
from src.frameworks.database import Base
from src.domain.entities import User, AudioFile, DecodingConfig, Transcription, DialectStats

class UserModel(Base):
    __tablename__ = "users"

    id = Column(SQLUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String(50), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    audio_files = relationship("AudioFileModel", back_populates="user", cascade="all, delete-orphan")

    def to_domain(self) -> User:
        return User(
            id=self.id,
            username=self.username,
            password_hash=self.password_hash,
            created_at=self.created_at
        )

    @classmethod
    def from_domain(cls, user: User) -> "UserModel":
        return cls(
            id=user.id or uuid.uuid4(),
            username=user.username,
            password_hash=user.password_hash,
            created_at=user.created_at or datetime.utcnow()
        )


class AudioFileModel(Base):
    __tablename__ = "audio_files"

    id = Column(SQLUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(SQLUUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    raw_storage_path = Column(String(512), nullable=False)
    standard_storage_path = Column(String(512), nullable=False)
    original_format = Column(String(10), nullable=False)
    duration_seconds = Column(Float, nullable=False)
    size_bytes = Column(BigInteger, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    user = relationship("UserModel", back_populates="audio_files")
    transcription = relationship("TranscriptionModel", back_populates="audio_file", uselist=False, cascade="all, delete-orphan")
    dialect_stats = relationship("DialectStatsModel", back_populates="audio_file", uselist=False, cascade="all, delete-orphan")

    def to_domain(self) -> AudioFile:
        return AudioFile(
            id=self.id,
            user_id=self.user_id,
            raw_storage_path=self.raw_storage_path,
            standard_storage_path=self.standard_storage_path,
            original_format=self.original_format,
            duration_seconds=self.duration_seconds,
            size_bytes=self.size_bytes,
            created_at=self.created_at
        )

    @classmethod
    def from_domain(cls, audio_file: AudioFile) -> "AudioFileModel":
        return cls(
            id=audio_file.id or uuid.uuid4(),
            user_id=audio_file.user_id,
            raw_storage_path=audio_file.raw_storage_path,
            standard_storage_path=audio_file.standard_storage_path,
            original_format=audio_file.original_format,
            duration_seconds=audio_file.duration_seconds,
            size_bytes=audio_file.size_bytes,
            created_at=audio_file.created_at or datetime.utcnow()
        )


class DecodingConfigModel(Base):
    __tablename__ = "decoding_configs"

    id = Column(SQLUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    method = Column(String(30), nullable=False)
    beam_size = Column(Integer, nullable=True)
    causal = Column(Boolean, default=False, nullable=False)
    chunk_size = Column(Integer, nullable=True)
    left_context_frames = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    transcriptions = relationship("TranscriptionModel", back_populates="decoding_config")

    def to_domain(self) -> DecodingConfig:
        return DecodingConfig(
            id=self.id,
            method=self.method,
            beam_size=self.beam_size,
            causal=self.causal,
            chunk_size=self.chunk_size,
            left_context_frames=self.left_context_frames,
            created_at=self.created_at
        )

    @classmethod
    def from_domain(cls, config: DecodingConfig) -> "DecodingConfigModel":
        return cls(
            id=config.id or uuid.uuid4(),
            method=config.method,
            beam_size=config.beam_size,
            causal=config.causal,
            chunk_size=config.chunk_size,
            left_context_frames=config.left_context_frames,
            created_at=config.created_at or datetime.utcnow()
        )


class TranscriptionModel(Base):
    __tablename__ = "transcriptions"

    id = Column(SQLUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    audio_file_id = Column(SQLUUID(as_uuid=True), ForeignKey("audio_files.id", ondelete="CASCADE"), unique=True, nullable=False)
    decoding_config_id = Column(SQLUUID(as_uuid=True), ForeignKey("decoding_configs.id", ondelete="RESTRICT"), nullable=False)
    text = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    word_alignments = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    audio_file = relationship("AudioFileModel", back_populates="transcription")
    decoding_config = relationship("DecodingConfigModel", back_populates="transcriptions")

    def to_domain(self) -> Transcription:
        return Transcription(
            id=self.id,
            audio_file_id=self.audio_file_id,
            decoding_config_id=self.decoding_config_id,
            text=self.text,
            confidence=self.confidence,
            word_alignments=self.word_alignments,
            created_at=self.created_at
        )

    @classmethod
    def from_domain(cls, transcription: Transcription) -> "TranscriptionModel":
        return cls(
            id=transcription.id or uuid.uuid4(),
            audio_file_id=transcription.audio_file_id,
            decoding_config_id=transcription.decoding_config_id,
            text=transcription.text,
            confidence=transcription.confidence,
            word_alignments=transcription.word_alignments,
            created_at=transcription.created_at or datetime.utcnow()
        )


class DialectStatsModel(Base):
    __tablename__ = "dialect_stats"

    id = Column(SQLUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    audio_file_id = Column(SQLUUID(as_uuid=True), ForeignKey("audio_files.id", ondelete="CASCADE"), unique=True, nullable=False)
    inferred_dialect = Column(String(15), nullable=False)
    dialect_probability = Column(Float, nullable=False)
    wer_estimate = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    audio_file = relationship("AudioFileModel", back_populates="dialect_stats")

    def to_domain(self) -> DialectStats:
        return DialectStats(
            id=self.id,
            audio_file_id=self.audio_file_id,
            inferred_dialect=self.inferred_dialect,
            dialect_probability=self.dialect_probability,
            wer_estimate=self.wer_estimate,
            created_at=self.created_at
        )

    @classmethod
    def from_domain(cls, stats: DialectStats) -> "DialectStatsModel":
        return cls(
            id=stats.id or uuid.uuid4(),
            audio_file_id=stats.audio_file_id,
            inferred_dialect=stats.inferred_dialect,
            dialect_probability=stats.dialect_probability,
            wer_estimate=stats.wer_estimate,
            created_at=stats.created_at or datetime.utcnow()
        )
