import json
from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_transcribe_endpoint_without_config():
    # Send a mock wav file upload
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
    # Send malformed JSON string for config
    data = {"config": "{malformed_json}"}
    
    response = client.post("/transcribe", files=files, data=data)
    assert response.status_code == 400
    assert "Invalid JSON" in response.json()["detail"]
