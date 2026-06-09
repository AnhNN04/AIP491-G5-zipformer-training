import pytest
import json
import httpx
from unittest.mock import MagicMock, patch
from src.domain.entities import DecodingConfig
from src.adapters.asr_client.http_client import ASRHTTPClient, ASRServerConnectionError

@pytest.mark.anyio
async def test_transcribe_success():
    client = ASRHTTPClient(server_url="http://localhost:8002")
    config = DecodingConfig(method="greedy_search", causal=False)
    
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"text": "chào mừng", "confidence": 0.99}
    
    with patch("httpx.AsyncClient.post") as mock_post:
        mock_post.return_value = mock_response
        
        result = await client.transcribe(
            audio_data=b"dummy wav content",
            filename="audio.wav",
            decoding_config=config
        )
        
        assert result["text"] == "chào mừng"
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert args[0] == "http://localhost:8002/transcribe"
        assert "file" in kwargs["files"]
        assert kwargs["data"]["config"] == json.dumps({"method": "greedy_search", "causal": False})


@pytest.mark.anyio
async def test_transcribe_retry_success():
    client = ASRHTTPClient(server_url="http://localhost:8002")
    
    mock_success = MagicMock()
    mock_success.status_code = 200
    mock_success.json.return_value = {"text": "retried success"}
    
    call_count = 0
    
    async def mock_post_impl(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise httpx.RequestError("Connection failed")
        return mock_success

    with patch("httpx.AsyncClient.post", side_effect=mock_post_impl):
        # We patch asyncio.sleep to make the test run instantly
        with patch("asyncio.sleep") as mock_sleep:
            result = await client.transcribe(
                audio_data=b"wav",
                filename="audio.wav"
            )
            assert result["text"] == "retried success"
            assert call_count == 3
            assert mock_sleep.call_count == 2


@pytest.mark.anyio
async def test_transcribe_failure_after_retries():
    client = ASRHTTPClient(server_url="http://localhost:8002")
    
    async def mock_post_impl(*args, **kwargs):
        raise httpx.RequestError("Permanent connection error")

    with patch("httpx.AsyncClient.post", side_effect=mock_post_impl):
        with patch("asyncio.sleep") as mock_sleep:
            with pytest.raises(ASRServerConnectionError) as exc_info:
                await client.transcribe(
                    audio_data=b"wav",
                    filename="audio.wav"
                )
            assert "temporarily unreachable" in str(exc_info.value)
            assert mock_sleep.call_count == 2
