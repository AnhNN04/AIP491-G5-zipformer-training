import json
import asyncio
import pytest
from unittest.mock import patch, AsyncMock
from fastapi import WebSocketDisconnect
import websockets
from src.adapters.asr_client.websocket_client import (
    map_handshake_config,
    ASRWebSocketRelayer
)

# ----------------- Unit Tests for map_handshake_config -----------------

def test_map_handshake_config_valid_defaults():
    config = {"method": "greedy_search"}
    mapped = map_handshake_config(config)
    assert mapped == {
        "method": "greedy_search",
        "causal": False
    }

def test_map_handshake_config_valid_all_params():
    config = {
        "method": "modified_beam_search",
        "beam_size": 10,
        "causal": True,
        "chunk_size": 32,
        "left_context_frames": 256
    }
    mapped = map_handshake_config(config)
    assert mapped == {
        "method": "modified_beam_search",
        "beam_size": 10,
        "causal": True,
        "chunk_size": 32,
        "left_context_frames": 256
    }

def test_map_handshake_config_invalid_method():
    with pytest.raises(ValueError, match="Invalid decoding method"):
        map_handshake_config({"method": "invalid_search"})

def test_map_handshake_config_invalid_beam_size_low():
    with pytest.raises(ValueError, match="Invalid beam_size"):
        map_handshake_config({"method": "modified_beam_search", "beam_size": 0})

def test_map_handshake_config_invalid_beam_size_high():
    with pytest.raises(ValueError, match="Invalid beam_size"):
        map_handshake_config({"method": "modified_beam_search", "beam_size": 21})

def test_map_handshake_config_invalid_causal():
    with pytest.raises(ValueError, match="Invalid causal parameter"):
        map_handshake_config({"causal": "yes"})

def test_map_handshake_config_invalid_chunk_size():
    with pytest.raises(ValueError, match="Invalid chunk_size"):
        map_handshake_config({"chunk_size": -5})

def test_map_handshake_config_invalid_left_context():
    with pytest.raises(ValueError, match="Invalid left_context_frames"):
        map_handshake_config({"left_context_frames": -1})


# ----------------- Mock Classes for WebSocket Relayer -----------------

class MockClientWebSocket:
    def __init__(self):
        self.received_messages = []
        self.sent_messages = []
        self.closed = False
        self.close_code = None
        
    async def receive(self):
        if not self.received_messages:
            # Simulate disconnect/end of messages
            raise WebSocketDisconnect()
        return self.received_messages.pop(0)
        
    async def send_text(self, text: str):
        self.sent_messages.append({"type": "text", "data": text})
        
    async def send_bytes(self, data: bytes):
        self.sent_messages.append({"type": "bytes", "data": data})
        
    async def close(self, code: int = 1000):
        self.closed = True
        self.close_code = code


class MockAsrWebSocket:
    def __init__(self):
        self.sent_messages = []
        self.recv_messages = []
        
    async def send(self, message):
        self.sent_messages.append(message)
        
    async def recv(self):
        if not self.recv_messages:
            raise websockets.exceptions.ConnectionClosed(None, None)
        return self.recv_messages.pop(0)
        
    def __aiter__(self):
        return self
        
    async def __anext__(self):
        if not self.recv_messages:
            raise StopAsyncIteration
        return self.recv_messages.pop(0)


class AsyncContextManagerMock:
    def __init__(self, target):
        self.target = target
        
    async def __aenter__(self):
        return self.target
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


# ----------------- Relayer Integration Tests -----------------

@pytest.mark.anyio
async def test_relayer_handshake_timeout():
    client_ws = MockClientWebSocket()
    relayer = ASRWebSocketRelayer()

    # Simulate timeout by causing wait_for to raise TimeoutError
    async def mock_wait_for(coro, timeout):
        try:
            coro.close()
        except RuntimeError:
            pass
        raise asyncio.TimeoutError()

    with patch("asyncio.wait_for", side_effect=mock_wait_for):
        await relayer.relay(client_ws)
        
    assert client_ws.closed is True
    assert client_ws.close_code == 1008  # Policy Violation


@pytest.mark.anyio
async def test_relayer_invalid_handshake_event():
    client_ws = MockClientWebSocket()
    # Send non-handshake event
    client_ws.received_messages.append({
        "type": "websocket.receive",
        "text": json.dumps({"event": "not_handshake"})
    })
    relayer = ASRWebSocketRelayer()

    await relayer.relay(client_ws)
    assert client_ws.closed is True
    assert client_ws.close_code == 1008


@pytest.mark.anyio
async def test_relayer_invalid_handshake_config():
    client_ws = MockClientWebSocket()
    # Send invalid config method
    client_ws.received_messages.append({
        "type": "websocket.receive",
        "text": json.dumps({
            "event": "handshake",
            "config": {"method": "invalid_method"}
        })
    })
    relayer = ASRWebSocketRelayer()

    await relayer.relay(client_ws)
    assert client_ws.closed is True
    assert client_ws.close_code == 1008


@pytest.mark.anyio
async def test_relayer_success_flow():
    client_ws = MockClientWebSocket()
    # 1. Enqueue client handshake config frame
    client_ws.received_messages.append({
        "type": "websocket.receive",
        "text": json.dumps({
            "event": "handshake",
            "config": {
                "method": "greedy_search",
                "causal": True
            }
        })
    })
    # 2. Enqueue client binary audio chunk
    client_ws.received_messages.append({
        "type": "websocket.receive",
        "bytes": b"fake pcm data"
    })
    # 3. Enqueue client stop event
    client_ws.received_messages.append({
        "type": "websocket.receive",
        "text": json.dumps({"event": "stop"})
    })

    asr_ws = MockAsrWebSocket()
    # Enqueue ASR handshake ok response
    asr_ws.recv_messages.append(json.dumps({
        "event": "handshake_ok",
        "session_id": "test_session_123"
    }))
    # Enqueue ASR transcript update response (triggered by audio chunk)
    asr_ws.recv_messages.append(json.dumps({
        "event": "transcript_update",
        "text": "chào mừng",
        "is_final": False
    }))
    # Enqueue ASR finished response (triggered by stop event)
    asr_ws.recv_messages.append(json.dumps({
        "event": "finished",
        "full_transcript": "chào mừng bạn",
        "confidence": 0.95
    }))

    relayer = ASRWebSocketRelayer()

    # Mock websockets.connect to return our mock ASR socket
    with patch("websockets.connect", return_value=AsyncContextManagerMock(asr_ws)):
        await relayer.relay(client_ws)

    # Assertions on ASR Server received data
    assert len(asr_ws.sent_messages) == 3
    # Handshake sent to ASR
    assert json.loads(asr_ws.sent_messages[0]) == {
        "event": "handshake",
        "config": {"method": "greedy_search", "causal": True}
    }
    # Audio bytes sent to ASR
    assert asr_ws.sent_messages[1] == b"fake pcm data"
    # Stop sent to ASR
    assert json.loads(asr_ws.sent_messages[2]) == {"event": "stop"}

    # Assertions on client received data (relayed from ASR)
    assert len(client_ws.sent_messages) == 3
    assert json.loads(client_ws.sent_messages[0]["data"])["event"] == "handshake_ok"
    assert json.loads(client_ws.sent_messages[1]["data"])["event"] == "transcript_update"
    assert json.loads(client_ws.sent_messages[2]["data"])["event"] == "finished"
