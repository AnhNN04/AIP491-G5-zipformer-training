import pytest
import uuid
from unittest.mock import AsyncMock, MagicMock
from src.domain.entities import User, AudioFile, DecodingConfig, Transcription, DialectStats
from src.adapters.database.sqlalchemy_repositories import (
    SQLAlchemyUserRepository,
    SQLAlchemyAudioFileRepository,
    SQLAlchemyDecodingConfigRepository,
    SQLAlchemyTranscriptionRepository,
    SQLAlchemyDialectStatsRepository
)

@pytest.fixture
def mock_session():
    session = AsyncMock()
    session.add = MagicMock()
    # Mock return values for async sessions
    session.execute.return_value = MagicMock()
    return session

@pytest.mark.anyio
async def test_user_repository_get(mock_session):
    repo = SQLAlchemyUserRepository(mock_session)
    user_id = uuid.uuid4()
    
    # Mock session.get return value
    mock_orm_user = MagicMock()
    mock_orm_user.id = user_id
    mock_orm_user.username = "test_user"
    mock_orm_user.password_hash = "hash"
    mock_orm_user.to_domain.return_value = User(id=user_id, username="test_user", password_hash="hash")
    
    mock_session.get.return_value = mock_orm_user
    
    result = await repo.get_by_id(user_id)
    assert result is not None
    assert result.username == "test_user"
    mock_session.get.assert_called_once()


@pytest.mark.anyio
async def test_audio_file_repository_save(mock_session):
    repo = SQLAlchemyAudioFileRepository(mock_session)
    audio = AudioFile(
        raw_storage_path="raw_path",
        standard_storage_path="std_path",
        original_format="mp3",
        duration_seconds=12.5,
        size_bytes=5000
    )
    
    result = await repo.save(audio)
    assert result is not None
    mock_session.add.assert_called_once()
    mock_session.commit.assert_called_once()
