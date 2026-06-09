"""
REST API and WebSocket integration tests for asr-server/src/main.py.

Tests:
  - /health endpoint
  - POST /transcribe: happy paths, invalid config JSON, invalid method/beam_size
  - WebSocket /api/stream: handshake flow, binary-before-handshake rejection

Tests use the real FastAPI TestClient. The K2Decoder is mocked to avoid
requiring model checkpoint files during CI test runs.

Run:
    uv run pytest tests/test_main.py -v
"""

import io
import json
import os
import pytest
import numpy as np
import soundfile as sf
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_wav_bytes(duration_s: float = 0.5, sample_rate: int = 16000) -> bytes:
    n_samples = int(duration_s * sample_rate)
    samples = np.zeros(n_samples, dtype=np.float32)
    buf = io.BytesIO()
    sf.write(buf, samples, sample_rate, format="WAV", subtype="FLOAT")
    buf.seek(0)
    return buf.read()


def _make_pcm_bytes(n_samples: int = 3200) -> bytes:
    pcm = np.zeros(n_samples, dtype=np.int16)
    return pcm.tobytes()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def mock_decoder():
    """Patch K2Decoder at module level so FastAPI app uses the mock."""
    with patch("src.main.decoder") as mocked:
        mocked.model = MagicMock()
        mocked.token_table = MagicMock()
        mocked.decode.return_value = {
            "text": "xin chào",
            "confidence": 0.95,
            "word_alignments": [],
        }
        mocked.decode_chunk.return_value = "xin chào"
        yield mocked


@pytest.fixture(scope="module")
def client(mock_decoder):
    from src.main import app
    return TestClient(app)


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_health_ok(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "model_ready" in data

    def test_health_model_ready_true_when_model_loaded(self, client, mock_decoder):
        response = client.get("/health")
        assert response.status_code == 200
        # model and token_table are mocked MagicMock (truthy), so model_ready=True
        assert response.json()["model_ready"] is True


# ---------------------------------------------------------------------------
# POST /transcribe
# ---------------------------------------------------------------------------

class TestTranscribeEndpoint:
    def test_transcribe_returns_valid_response_structure(self, client):
        """Response must include text, confidence, word_alignments."""
        files = {"file": ("test.wav", _make_wav_bytes(), "audio/wav")}
        response = client.post("/transcribe", files=files)
        assert response.status_code == 200
        data = response.json()
        assert "text" in data
        assert "confidence" in data
        assert "word_alignments" in data
        assert isinstance(data["text"], str)
        assert isinstance(data["confidence"], float)
        assert isinstance(data["word_alignments"], list)

    def test_transcribe_with_greedy_search_config(self, client, mock_decoder):
        """Greedy search config should be forwarded to decoder.decode()."""
        wav_bytes = _make_wav_bytes()
        config_data = {"method": "greedy_search"}
        files = {"file": ("test.wav", wav_bytes, "audio/wav")}
        data = {"config": json.dumps(config_data)}
        response = client.post("/transcribe", files=files, data=data)
        assert response.status_code == 200
        # Verify decoder was called with correct method
        mock_decoder.decode.assert_called()
        call_kwargs = mock_decoder.decode.call_args.kwargs
        assert call_kwargs["method"] == "greedy_search"

    def test_transcribe_with_beam_search_config(self, client, mock_decoder):
        """Modified beam search config should be forwarded."""
        wav_bytes = _make_wav_bytes()
        config_data = {"method": "modified_beam_search", "beam_size": 8, "causal": True}
        files = {"file": ("test.wav", wav_bytes, "audio/wav")}
        data = {"config": json.dumps(config_data)}
        response = client.post("/transcribe", files=files, data=data)
        assert response.status_code == 200
        call_kwargs = mock_decoder.decode.call_args.kwargs
        assert call_kwargs["method"] == "modified_beam_search"
        assert call_kwargs["beam_size"] == 8
        assert call_kwargs["causal"] is True

    def test_transcribe_invalid_json_config_returns_400(self, client):
        """Malformed JSON config must return HTTP 400."""
        files = {"file": ("test.wav", _make_wav_bytes(), "audio/wav")}
        data = {"config": "{bad json here"}
        response = client.post("/transcribe", files=files, data=data)
        assert response.status_code == 400
        assert "Invalid JSON" in response.json()["detail"]

    def test_transcribe_decoder_raises_value_error_returns_400(self, client, mock_decoder):
        """If decoder.decode raises ValueError, endpoint must return HTTP 400."""
        mock_decoder.decode.side_effect = ValueError("Unsupported decoding method: 'xyz'")
        files = {"file": ("test.wav", _make_wav_bytes(), "audio/wav")}
        config_data = {"method": "xyz"}
        data = {"config": json.dumps(config_data)}
        response = client.post("/transcribe", files=files, data=data)
        assert response.status_code == 400
        assert "Unsupported decoding method" in response.json()["detail"]
        # Restore
        mock_decoder.decode.side_effect = None
        mock_decoder.decode.return_value = {
            "text": "xin chào", "confidence": 0.95, "word_alignments": []
        }

    def test_transcribe_decoder_raises_runtime_error_returns_500(self, client, mock_decoder):
        """If decoder raises an unexpected exception, endpoint must return HTTP 500."""
        mock_decoder.decode.side_effect = RuntimeError("CUDA out of memory")
        files = {"file": ("test.wav", _make_wav_bytes(), "audio/wav")}
        response = client.post("/transcribe", files=files)
        assert response.status_code == 500
        assert "Inference processing error" in response.json()["detail"]
        # Restore
        mock_decoder.decode.side_effect = None
        mock_decoder.decode.return_value = {
            "text": "xin chào", "confidence": 0.95, "word_alignments": []
        }


# ---------------------------------------------------------------------------
# WebSocket /api/stream
# ---------------------------------------------------------------------------

class TestWebSocketStream:
    def test_handshake_ok(self, client):
        """Successful handshake should return handshake_ok with session_id."""
        with client.websocket_connect("/api/stream") as ws:
            ws.send_json({
                "event": "handshake",
                "config": {"method": "greedy_search", "beam_size": 4}
            })
            resp = ws.receive_json()
            assert resp["event"] == "handshake_ok"
            assert "session_id" in resp
            assert resp["session_id"].startswith("asr_sess_")
            # Close cleanly
            ws.send_json({"event": "stop"})
            ws.receive_json()

    def test_send_stop_after_handshake_returns_finished(self, client):
        """After handshake + stop, endpoint must return finished event."""
        with client.websocket_connect("/api/stream") as ws:
            ws.send_json({"event": "handshake", "config": {}})
            ws.receive_json()  # handshake_ok
            ws.send_json({"event": "stop"})
            resp = ws.receive_json()
            assert resp["event"] == "finished"
            assert "full_transcript" in resp
            assert isinstance(resp["full_transcript"], str)

    def test_binary_chunk_triggers_transcript_update(self, client):
        """Binary audio chunk after handshake must trigger transcript_update event."""
        with client.websocket_connect("/api/stream") as ws:
            ws.send_json({"event": "handshake", "config": {"method": "greedy_search"}})
            ws.receive_json()  # handshake_ok
            ws.send_bytes(_make_pcm_bytes(3200))
            resp = ws.receive_json()
            assert resp["event"] == "transcript_update"
            assert "text" in resp
            assert resp["is_final"] is False
            assert isinstance(resp["confidence"], float)
            # Stop cleanly
            ws.send_json({"event": "stop"})
            ws.receive_json()

    def test_binary_before_handshake_closes_connection(self, client):
        """Sending binary audio before handshake should close with code 1008."""
        try:
            with client.websocket_connect("/api/stream") as ws:
                ws.send_bytes(_make_pcm_bytes(3200))
                resp = ws.receive()
                # Starlette TestClient returns close frame or raises
                assert resp.get("type") == "websocket.close"
        except RuntimeError:
            # Expected: TestClient raises RuntimeError on forced close
            pass

    def test_unknown_event_is_ignored(self, client):
        """Unknown text events should not crash the server."""
        with client.websocket_connect("/api/stream") as ws:
            ws.send_json({"event": "handshake", "config": {}})
            ws.receive_json()  # handshake_ok
            ws.send_json({"event": "unknown_event_type"})
            # Server should not send a response for unknown events
            # Send stop to get a clean response to verify connection is alive
            ws.send_json({"event": "stop"})
            resp = ws.receive_json()
            assert resp["event"] == "finished"

    def test_invalid_json_closes_connection(self, client):
        """Malformed JSON text frame should close the connection (code 1008)."""
        try:
            with client.websocket_connect("/api/stream") as ws:
                ws.send_text("not valid json at all {{{{")
                resp = ws.receive()
                assert resp.get("type") == "websocket.close"
        except RuntimeError:
            pass
