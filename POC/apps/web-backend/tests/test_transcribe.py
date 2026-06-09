import pytest
import uuid
from unittest.mock import AsyncMock, MagicMock, ANY
from src.domain.entities import User, AudioFile, DecodingConfig, Transcription, DialectStats
from src.usecases.transcribe import TranscribeAudioUseCase

@pytest.fixture
def mock_repos():
    return {
        "audio_file_repo": AsyncMock(),
        "transcription_repo": AsyncMock(),
        "decoding_config_repo": AsyncMock(),
        "dialect_stats_repo": AsyncMock()
    }

@pytest.fixture
def mock_storage():
    return AsyncMock()

@pytest.fixture
def mock_asr_client():
    client = AsyncMock()
    client.transcribe.return_value = {
        "text": "chào mừng hệ thống",
        "confidence": 0.97,
        "word_alignments": [{"word": "chào", "start": 0.0, "end": 0.3, "conf": 0.99}]
    }
    return client

@pytest.fixture
def mock_processor():
    processor = MagicMock()
    processor.transcode = AsyncMock(return_value=(b"transcoded pcm wav data", 15.4))
    return processor

@pytest.mark.anyio
async def test_transcribe_usecase_success(
    mock_repos, mock_storage, mock_asr_client, mock_processor
):
    # Set up mock repository saves to return inputs with generated IDs
    def save_audio_mock(audio):
        audio.id = audio.id or uuid.uuid4()
        return audio
    mock_repos["audio_file_repo"].save.side_effect = save_audio_mock
    
    def save_config_mock(config):
        config.id = config.id or uuid.uuid4()
        return config
    mock_repos["decoding_config_repo"].save.side_effect = save_config_mock

    def save_transcription_mock(trans):
        trans.id = trans.id or uuid.uuid4()
        return trans
    mock_repos["transcription_repo"].save.side_effect = save_transcription_mock

    def save_dialect_mock(dialect):
        dialect.id = dialect.id or uuid.uuid4()
        return dialect
    mock_repos["dialect_stats_repo"].save.side_effect = save_dialect_mock

    mock_storage.upload_file.side_effect = lambda bucket_name, object_name, data, content_type: f"s3://{bucket_name}/{object_name}"

    usecase = TranscribeAudioUseCase(
        audio_file_repo=mock_repos["audio_file_repo"],
        transcription_repo=mock_repos["transcription_repo"],
        decoding_config_repo=mock_repos["decoding_config_repo"],
        dialect_stats_repo=mock_repos["dialect_stats_repo"],
        storage_service=mock_storage,
        asr_client=mock_asr_client,
        audio_processor=mock_processor
    )

    user_uuid = uuid.uuid4()
    custom_config = DecodingConfig(method="modified_beam_search", beam_size=6)

    response = await usecase.execute(
        file_data=b"raw mp3 contents",
        filename="input_speech.mp3",
        original_format="mp3",
        user_id=user_uuid,
        decoding_config_input=custom_config
    )

    # Assertions on return value structure
    assert response["status"] == "SUCCESS"
    assert response["duration_seconds"] == 15.4
    assert response["size_bytes"] == len(b"raw mp3 contents")
    assert response["transcription"]["text"] == "chào mừng hệ thống"
    assert response["dialect"]["inferred"] == "NORTHERN"
    assert response["dialect"]["probability"] == 0.70

    # Verify coordinator calls
    mock_processor.validate_format.assert_called_with("mp3")
    mock_processor.transcode.assert_called_with(b"raw mp3 contents", "mp3")
    
    # 2 uploads expected
    assert mock_storage.upload_file.call_count == 2
    mock_storage.upload_file.assert_any_call(
        bucket_name="raw-uploads",
        object_name=ANY,
        data=b"raw mp3 contents",
        content_type="audio/mp3"
    )
    mock_storage.upload_file.assert_any_call(
        bucket_name="standardized-wavs",
        object_name=ANY,
        data=b"transcoded pcm wav data",
        content_type="audio/wav"
    )

    # Verify repo saves
    mock_repos["decoding_config_repo"].save.assert_called_once_with(custom_config)
    mock_repos["audio_file_repo"].save.assert_called_once()
    mock_repos["transcription_repo"].save.assert_called_once()
    mock_repos["dialect_stats_repo"].save.assert_called_once()

    # Verify ASR client call
    mock_asr_client.transcribe.assert_called_once()
    args, kwargs = mock_asr_client.transcribe.call_args
    assert kwargs["audio_data"] == b"transcoded pcm wav data"
    assert kwargs["decoding_config"] == custom_config
