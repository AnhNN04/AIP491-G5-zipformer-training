import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import UploadFile, HTTPException
from src.adapters.controllers.upload_controller import UploadController
from src.adapters.asr_client.http_client import ASRServerConnectionError

@pytest.fixture
def mock_usecase():
    return AsyncMock()

@pytest.fixture
def controller(mock_usecase):
    return UploadController(mock_usecase)

@pytest.mark.anyio
async def test_controller_success(controller, mock_usecase):
    # Mock UploadFile
    mock_file = AsyncMock(spec=UploadFile)
    mock_file.filename = "test.wav"
    mock_file.read.return_value = b"wav audio data"
    
    mock_usecase.execute.return_value = {"audio_id": "123", "status": "SUCCESS"}
    
    config_str = json.dumps({"method": "greedy_search", "causal": False})
    
    result = await controller.upload_and_transcribe(mock_file, config_str)
    assert result["status"] == "SUCCESS"
    mock_usecase.execute.assert_called_once()
    args, kwargs = mock_usecase.execute.call_args
    assert kwargs["original_format"] == "wav"
    assert kwargs["decoding_config_input"].method == "greedy_search"


@pytest.mark.anyio
async def test_controller_invalid_format(controller):
    mock_file = AsyncMock(spec=UploadFile)
    mock_file.filename = "image.png"
    
    with pytest.raises(HTTPException) as exc_info:
        await controller.upload_and_transcribe(mock_file)
    assert exc_info.value.status_code == 400
    assert "Unsupported file format" in exc_info.value.detail


@pytest.mark.anyio
async def test_controller_payload_too_large(controller):
    mock_file = AsyncMock(spec=UploadFile)
    mock_file.filename = "speech.wav"
    
    # 51MB of data (limit is 50MB by default)
    mock_file.read.return_value = b"x" * (51 * 1024 * 1024)
    
    with pytest.raises(HTTPException) as exc_info:
        await controller.upload_and_transcribe(mock_file)
    assert exc_info.value.status_code == 413
    assert "File size exceeds" in exc_info.value.detail


@pytest.mark.anyio
async def test_controller_invalid_config_json(controller):
    mock_file = AsyncMock(spec=UploadFile)
    mock_file.filename = "speech.wav"
    mock_file.read.return_value = b"wav audio data"
    
    with pytest.raises(HTTPException) as exc_info:
        await controller.upload_and_transcribe(mock_file, "{invalid_json}")
    assert exc_info.value.status_code == 400
    assert "Invalid JSON format" in exc_info.value.detail


@pytest.mark.anyio
async def test_controller_invalid_decoding_method(controller):
    mock_file = AsyncMock(spec=UploadFile)
    mock_file.filename = "speech.wav"
    mock_file.read.return_value = b"wav audio data"
    
    config_str = json.dumps({"method": "random_search"})
    
    with pytest.raises(HTTPException) as exc_info:
        await controller.upload_and_transcribe(mock_file, config_str)
    assert exc_info.value.status_code == 400
    assert "Invalid decoding method" in exc_info.value.detail


@pytest.mark.anyio
async def test_controller_invalid_beam_size_range(controller):
    mock_file = AsyncMock(spec=UploadFile)
    mock_file.filename = "speech.wav"
    mock_file.read.return_value = b"wav audio data"
    
    # beam size must be 1 to 20
    config_str = json.dumps({"method": "modified_beam_search", "beam_size": 25})
    
    with pytest.raises(HTTPException) as exc_info:
        await controller.upload_and_transcribe(mock_file, config_str)
    assert exc_info.value.status_code == 400
    assert "Invalid beam_size" in exc_info.value.detail


@pytest.mark.anyio
async def test_controller_unreachable_asr_server(controller, mock_usecase):
    mock_file = AsyncMock(spec=UploadFile)
    mock_file.filename = "speech.wav"
    mock_file.read.return_value = b"wav audio"
    
    mock_usecase.execute.side_effect = ASRServerConnectionError("ASR Server down")
    
    with pytest.raises(HTTPException) as exc_info:
        await controller.upload_and_transcribe(mock_file)
    assert exc_info.value.status_code == 503
    assert "ASR Inference Server is temporarily unreachable" in exc_info.value.detail


@pytest.mark.anyio
async def test_controller_invalid_chunk_size(controller):
    mock_file = AsyncMock(spec=UploadFile)
    mock_file.filename = "speech.wav"
    mock_file.read.return_value = b"wav audio data"
    
    config_str = json.dumps({"method": "greedy_search", "chunk_size": 0})
    with pytest.raises(HTTPException) as exc_info:
        await controller.upload_and_transcribe(mock_file, config_str)
    assert exc_info.value.status_code == 400
    assert "Invalid chunk_size" in exc_info.value.detail


@pytest.mark.anyio
async def test_controller_invalid_left_context(controller):
    mock_file = AsyncMock(spec=UploadFile)
    mock_file.filename = "speech.wav"
    mock_file.read.return_value = b"wav audio data"
    
    config_str = json.dumps({"method": "greedy_search", "left_context_frames": -1})
    with pytest.raises(HTTPException) as exc_info:
        await controller.upload_and_transcribe(mock_file, config_str)
    assert exc_info.value.status_code == 400
    assert "Invalid left_context_frames" in exc_info.value.detail

