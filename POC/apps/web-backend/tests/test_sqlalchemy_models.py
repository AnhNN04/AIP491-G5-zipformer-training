import uuid
from datetime import datetime
from src.domain.entities import User, AudioFile, DecodingConfig, Transcription, DialectStats
from src.adapters.database.sqlalchemy_models import (
    UserModel,
    AudioFileModel,
    DecodingConfigModel,
    TranscriptionModel,
    DialectStatsModel
)

def test_user_mapping():
    domain_user = User(
        id=uuid.uuid4(),
        username="john_doe",
        password_hash="pbkdf2:sha256:...",
        created_at=datetime.utcnow()
    )
    orm_model = UserModel.from_domain(domain_user)
    assert orm_model.id == domain_user.id
    assert orm_model.username == "john_doe"
    assert orm_model.password_hash == "pbkdf2:sha256:..."

    mapped_back = orm_model.to_domain()
    assert mapped_back.id == domain_user.id
    assert mapped_back.username == domain_user.username
    assert mapped_back.password_hash == domain_user.password_hash


def test_audio_file_mapping():
    domain_audio = AudioFile(
        id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        raw_storage_path="s3://raw/file.mp3",
        standard_storage_path="s3://std/file.wav",
        original_format="mp3",
        duration_seconds=120.5,
        size_bytes=1024000,
        created_at=datetime.utcnow()
    )
    orm_model = AudioFileModel.from_domain(domain_audio)
    assert orm_model.id == domain_audio.id
    assert orm_model.user_id == domain_audio.user_id
    assert orm_model.raw_storage_path == "s3://raw/file.mp3"
    assert orm_model.duration_seconds == 120.5

    mapped_back = orm_model.to_domain()
    assert mapped_back.id == domain_audio.id
    assert mapped_back.user_id == domain_audio.user_id
    assert mapped_back.raw_storage_path == domain_audio.raw_storage_path


def test_decoding_config_mapping():
    domain_config = DecodingConfig(
        id=uuid.uuid4(),
        method="modified_beam_search",
        beam_size=4,
        causal=True,
        chunk_size=16,
        left_context_frames=64,
        created_at=datetime.utcnow()
    )
    orm_model = DecodingConfigModel.from_domain(domain_config)
    assert orm_model.id == domain_config.id
    assert orm_model.method == "modified_beam_search"
    assert orm_model.causal is True

    mapped_back = orm_model.to_domain()
    assert mapped_back.id == domain_config.id
    assert mapped_back.method == domain_config.method
    assert mapped_back.causal is True


def test_transcription_mapping():
    domain_trans = Transcription(
        id=uuid.uuid4(),
        audio_file_id=uuid.uuid4(),
        decoding_config_id=uuid.uuid4(),
        text="Xin chào Việt Nam",
        confidence=0.95,
        word_alignments=[{"word": "Xin", "start": 0.0, "end": 0.2, "conf": 0.96}],
        created_at=datetime.utcnow()
    )
    orm_model = TranscriptionModel.from_domain(domain_trans)
    assert orm_model.id == domain_trans.id
    assert orm_model.audio_file_id == domain_trans.audio_file_id
    assert orm_model.text == "Xin chào Việt Nam"
    assert orm_model.word_alignments == [{"word": "Xin", "start": 0.0, "end": 0.2, "conf": 0.96}]

    mapped_back = orm_model.to_domain()
    assert mapped_back.id == domain_trans.id
    assert mapped_back.audio_file_id == domain_trans.audio_file_id
    assert mapped_back.text == domain_trans.text
    assert mapped_back.word_alignments == domain_trans.word_alignments


def test_dialect_stats_mapping():
    domain_stats = DialectStats(
        id=uuid.uuid4(),
        audio_file_id=uuid.uuid4(),
        inferred_dialect="SOUTHERN",
        dialect_probability=0.88,
        wer_estimate=0.12,
        created_at=datetime.utcnow()
    )
    orm_model = DialectStatsModel.from_domain(domain_stats)
    assert orm_model.id == domain_stats.id
    assert orm_model.audio_file_id == domain_stats.audio_file_id
    assert orm_model.inferred_dialect == "SOUTHERN"
    assert orm_model.wer_estimate == 0.12

    mapped_back = orm_model.to_domain()
    assert mapped_back.id == domain_stats.id
    assert mapped_back.audio_file_id == domain_stats.audio_file_id
    assert mapped_back.inferred_dialect == domain_stats.inferred_dialect


def test_metadata_registration():
    from src.frameworks.database import Base
    tables = Base.metadata.tables
    assert "users" in tables
    assert "audio_files" in tables
    assert "decoding_configs" in tables
    assert "transcriptions" in tables
    assert "dialect_stats" in tables

