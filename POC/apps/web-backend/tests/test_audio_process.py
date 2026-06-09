import pytest
import os
from unittest.mock import MagicMock, patch, AsyncMock
from src.usecases.audio_process import (
    AudioProcessor,
    InvalidAudioFormatException,
    TranscodingError
)

@pytest.fixture
def processor():
    return AudioProcessor()

def test_validate_format(processor):
    # Valid formats
    for fmt in ['wav', 'mp3', 'flac', 'm4a', 'ogg', 'mp4', '.mp3']:
        processor.validate_format(fmt)
        
    # Invalid formats
    for fmt in ['txt', 'png', 'pdf', 'avi']:
        with pytest.raises(InvalidAudioFormatException):
            processor.validate_format(fmt)


@pytest.mark.anyio
async def test_get_duration_success(processor):
    mock_process = AsyncMock()
    mock_process.returncode = 0
    mock_process.communicate.return_value = (b"124.56\n", b"")
    
    with patch("asyncio.create_subprocess_exec", return_value=mock_process) as mock_exec:
        duration = await processor.get_duration("fake_path.wav")
        assert duration == 124.56
        mock_exec.assert_called_once()
        assert "ffprobe" in mock_exec.call_args[0]


@pytest.mark.anyio
async def test_get_duration_failure(processor):
    mock_process = AsyncMock()
    mock_process.returncode = 1
    mock_process.communicate.return_value = (b"", b"Invalid file header")
    
    with patch("asyncio.create_subprocess_exec", return_value=mock_process):
        with pytest.raises(TranscodingError) as exc_info:
            await processor.get_duration("corrupt_path.wav")
        assert "ffprobe failed" in str(exc_info.value)


@pytest.mark.anyio
async def test_transcode_success(processor):
    mock_process = AsyncMock()
    mock_process.returncode = 0
    mock_process.communicate.return_value = (b"", b"")
    
    # We patch open to return fake wav bytes when reading out_path
    original_open = open
    def mock_open_impl(file, mode="r", *args, **kwargs):
        if "wav" in str(file) and "rb" in mode:
            # Return a mock file handle with fake wav content
            mock_file = MagicMock()
            mock_file.__enter__.return_value = mock_file
            mock_file.read.return_value = b"transcoded wav data"
            return mock_file
        return original_open(file, mode, *args, **kwargs)

    with patch("asyncio.create_subprocess_exec", return_value=mock_process) as mock_exec:
        with patch("src.usecases.audio_process.open", side_effect=mock_open_impl):
            with patch.object(processor, "get_duration", return_value=30.0) as mock_dur:
                wav_bytes, duration = await processor.transcode(
                    audio_data=b"original mp3 bytes",
                    original_format="mp3"
                )
                assert wav_bytes == b"transcoded wav data"
                assert duration == 30.0
                mock_exec.assert_called_once()
                assert "ffmpeg" in mock_exec.call_args[0]


@pytest.mark.anyio
async def test_transcode_ffmpeg_failure(processor):
    mock_process = AsyncMock()
    mock_process.returncode = 1
    mock_process.communicate.return_value = (b"", b"FFmpeg syntax error or corrupt input")
    
    with patch("asyncio.create_subprocess_exec", return_value=mock_process):
        with pytest.raises(TranscodingError) as exc_info:
            await processor.transcode(
                audio_data=b"corrupt mp3 bytes",
                original_format="mp3"
            )
        assert "Transcoding failed" in str(exc_info.value)
