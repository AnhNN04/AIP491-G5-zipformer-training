import logging
from fastapi import WebSocket
from src.adapters.asr_client.websocket_client import ASRWebSocketRelayer

logger = logging.getLogger(__name__)

class StreamController:
    """
    Controller that accepts incoming WebSockets from frontend clients and delegates
    the connection to the downstream ASR WebSocket Relayer.
    """
    def __init__(self, relayer: ASRWebSocketRelayer = None):
        self.relayer = relayer or ASRWebSocketRelayer()

    async def handle_stream(self, websocket: WebSocket) -> None:
        """
        Accepts and handles a client WebSocket connection, starting the relay session.
        """
        logger.info("StreamController: Accepted client connection. Initializing stream relay...")
        await websocket.accept()
        await self.relayer.relay(websocket)
