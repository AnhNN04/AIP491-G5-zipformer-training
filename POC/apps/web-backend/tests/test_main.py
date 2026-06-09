import pytest
import json
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient
from src.frameworks.main import app
from src.frameworks.database import get_db_session

client = TestClient(app)

@pytest.fixture(autouse=True)
def override_db_dependency():
    mock_db = AsyncMock()
    app.dependency_overrides[get_db_session] = lambda: mock_db
    yield mock_db
    app.dependency_overrides.clear()


def test_health_check():
    # Remove override for health check as it doesn't use DB
    app.dependency_overrides.clear()
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_upload_route_success(override_db_dependency):
    mock_response = {
        "audio_id": "7a3556d1-1311-4f35-94f7-873d63b65ef3",
        "duration_seconds": 2.45,
        "size_bytes": 1048576,
        "status": "SUCCESS",
        "transcription": {
            "text": "chào mừng",
            "confidence": 0.98,
            "word_alignments": []
        },
        "dialect": {
            "inferred": "NORTHERN",
            "probability": 0.95
        }
    }

    # Patch the use case execute method to avoid hitting MinIO/ASR server
    with patch("src.usecases.transcribe.TranscribeAudioUseCase.execute", new_callable=AsyncMock) as mock_exec:
        mock_exec.return_value = mock_response
        
        files = {"file": ("test.wav", b"fake audio data", "audio/wav")}
        config_data = {"method": "greedy_search"}
        data = {"config": json.dumps(config_data)}
        
        response = client.post("/api/upload", files=files, data=data)
        
        assert response.status_code == 200
        resp_json = response.json()
        assert resp_json["audio_id"] == "7a3556d1-1311-4f35-94f7-873d63b65ef3"
        assert resp_json["transcription"]["text"] == "chào mừng"
        mock_exec.assert_called_once()


def test_upload_route_invalid_format():
    # Attempt upload with an invalid format (png)
    files = {"file": ("test.png", b"fake image data", "image/png")}
    response = client.post("/api/upload", files=files)
    
    assert response.status_code == 400
    assert "Unsupported file format" in response.json()["detail"]


def test_websocket_stream_route_success():
    from tests.test_websocket_client import MockAsrWebSocket, AsyncContextManagerMock
    
    asr_ws = MockAsrWebSocket()
    asr_ws.recv_messages.append(json.dumps({
        "event": "handshake_ok",
        "session_id": "test_route_session_123"
    }))
    asr_ws.recv_messages.append(json.dumps({
        "event": "transcript_update",
        "text": "chào mừng bạn",
        "is_final": False
    }))
    asr_ws.recv_messages.append(json.dumps({
        "event": "finished",
        "full_transcript": "chào mừng bạn đến với hệ thống",
        "confidence": 0.96
    }))
    
    with patch("websockets.connect", return_value=AsyncContextManagerMock(asr_ws)):
        with client.websocket_connect("/api/stream") as websocket:
            websocket.send_json({
                "event": "handshake",
                "config": {
                    "method": "greedy_search",
                    "causal": True
                }
            })
            resp = websocket.receive_json()
            assert resp["event"] == "handshake_ok"
            assert resp["session_id"] == "test_route_session_123"
            
            websocket.send_bytes(b"some audio bytes")
            resp = websocket.receive_json()
            assert resp["event"] == "transcript_update"
            assert resp["text"] == "chào mừng bạn"
            
            websocket.send_json({"event": "stop"})
            resp = websocket.receive_json()
            assert resp["event"] == "finished"
            assert resp["full_transcript"] == "chào mừng bạn đến với hệ thống"

