import json
from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_transcribe_endpoint_without_config():
    files = {"file": ("test.wav", b"fake wav audio data content", "audio/wav")}
    response = client.post("/transcribe", files=files)
    
    assert response.status_code == 200
    data = response.json()
    assert "text" in data
    assert "confidence" in data
    assert "word_alignments" in data
    assert data["text"] == "chào mừng bạn đến với hệ thống nhận dạng giọng nói"
    assert len(data["word_alignments"]) > 0


def test_transcribe_endpoint_with_valid_config():
    files = {"file": ("test.wav", b"fake wav audio data content", "audio/wav")}
    config_data = {
        "method": "modified_beam_search",
        "beam_size": 5,
        "causal": True
    }
    data = {"config": json.dumps(config_data)}
    
    response = client.post("/transcribe", files=files, data=data)
    assert response.status_code == 200
    resp_data = response.json()
    assert resp_data["text"] == "chào mừng bạn đến với hệ thống nhận dạng giọng nói"


def test_transcribe_endpoint_with_invalid_config():
    files = {"file": ("test.wav", b"fake wav audio data content", "audio/wav")}
    data = {"config": "{malformed_json}"}
    
    response = client.post("/transcribe", files=files, data=data)
    assert response.status_code == 400
    assert "Invalid JSON" in response.json()["detail"]


def test_websocket_stream_success():
    with client.websocket_connect("/api/stream") as websocket:
        # 1. Send handshake config
        websocket.send_json({
            "event": "handshake",
            "config": {
                "method": "greedy_search",
                "causal": True,
                "chunk_size": 16,
                "left_context_frames": 128
            }
        })
        resp = websocket.receive_json()
        assert resp["event"] == "handshake_ok"
        assert "session_id" in resp
        
        # 2. Send binary audio chunk
        websocket.send_bytes(b"dummy raw pcm samples")
        resp = websocket.receive_json()
        assert resp["event"] == "transcript_update"
        assert resp["text"] == "chào mừng bạn đến với hệ thống nhận dạng giọng nói"
        assert resp["is_final"] is False
        
        # 3. Send stop event
        websocket.send_json({"event": "stop"})
        resp = websocket.receive_json()
        assert resp["event"] == "finished"
        assert "full_transcript" in resp
        assert resp["full_transcript"] == "chào mừng bạn đến với hệ thống nhận dạng giọng nói"


def test_websocket_stream_binary_before_handshake():
    # If client sends audio before handshake, connection should close with policy violation (1008)
    # Using try/except since client socket close raises RuntimeError in some test client implementations
    try:
        with client.websocket_connect("/api/stream") as websocket:
            websocket.send_bytes(b"early raw pcm samples")
            # Starlette TestClient WebSocket will close on invalid state
            # Reading should raise an error or return closed status
            resp = websocket.receive()
            # If we reached here, assert that it's closed
            assert resp.get("type") == "websocket.close"
    except RuntimeError:
        # TestClient can raise RuntimeError on forced socket close, which is expected
        pass
