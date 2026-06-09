import os
import json
import asyncio
import logging
from typing import Optional, Dict, Any
from fastapi import WebSocket, WebSocketDisconnect
import websockets

logger = logging.getLogger(__name__)

def map_handshake_config(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validates and maps frontend configuration fields to backend k2 search space.
    Enforces rules and bounds for decoding parameters.
    """
    method = config_dict.get("method", "greedy_search")
    if method not in ("greedy_search", "modified_beam_search"):
        raise ValueError("Invalid decoding method. Must be greedy_search or modified_beam_search.")
    
    mapped: Dict[str, Any] = {"method": method}
    
    beam_size = config_dict.get("beam_size")
    if beam_size is not None:
        try:
            beam_size = int(beam_size)
            if not (1 <= beam_size <= 20):
                raise ValueError()
        except ValueError:
            raise ValueError("Invalid beam_size. Must be an integer between 1 and 20.")
        mapped["beam_size"] = beam_size
    elif method == "modified_beam_search":
        mapped["beam_size"] = 4
        
    causal = config_dict.get("causal", False)
    if not isinstance(causal, bool):
        raise ValueError("Invalid causal parameter. Must be a boolean.")
    mapped["causal"] = causal
    
    chunk_size = config_dict.get("chunk_size")
    if chunk_size is not None:
        try:
            chunk_size = int(chunk_size)
            if chunk_size <= 0:
                raise ValueError()
        except ValueError:
            raise ValueError("Invalid chunk_size. Must be a positive integer.")
        mapped["chunk_size"] = chunk_size
        
    left_context_frames = config_dict.get("left_context_frames")
    if left_context_frames is not None:
        try:
            left_context_frames = int(left_context_frames)
            if left_context_frames < 0:
                raise ValueError()
        except ValueError:
            raise ValueError("Invalid left_context_frames. Must be a non-negative integer.")
        mapped["left_context_frames"] = left_context_frames
        
    return mapped


class ASRWebSocketRelayer:
    """
    Establishes and manages bidirectional WebSocket tunnels between the web backend client
    and the ASR Model Server. Relays text and audio frames with handshake validation.
    """
    def __init__(self, asr_server_url: Optional[str] = None):
        self.asr_server_url = asr_server_url or os.getenv("ASR_SERVER_URL", "http://localhost:8001")
        # Standardize ws/wss scheme
        ws_url = self.asr_server_url.replace("http://", "ws://").replace("https://", "wss://")
        if not ws_url.endswith("/api/stream"):
            ws_url = f"{ws_url.rstrip('/')}/api/stream"
        self.asr_ws_url = ws_url
        logger.info(f"Initialized ASRWebSocketRelayer pointing to {self.asr_ws_url}")

    async def relay(self, client_ws: WebSocket) -> None:
        """
        Accepts client connection, performs handshake verification, and opens WebSocket
        session with ASR Server. Relays all incoming and outgoing messages.
        """
        # 1. Enforce active handshake: client must send handshake config within 5.0 seconds
        try:
            message = await asyncio.wait_for(client_ws.receive(), timeout=5.0)
        except asyncio.TimeoutError:
            logger.warning("Handshake timeout: Client did not send handshake within 5.0 seconds")
            await client_ws.close(code=1008) # Policy Violation
            return

        if message.get("type") == "websocket.disconnect":
            logger.info("Client disconnected during handshake")
            return

        if "text" not in message:
            logger.warning("Invalid initial message: expected text frame handshake config")
            await client_ws.close(code=1008)
            return

        # 2. Parse and validate configuration
        try:
            handshake_data = json.loads(message["text"])
        except json.JSONDecodeError:
            logger.warning("Malformed JSON in handshake config")
            await client_ws.close(code=1008)
            return

        if handshake_data.get("event") != "handshake":
            logger.warning("Invalid event in handshake: expected 'handshake'")
            await client_ws.close(code=1008)
            return

        config_dict = handshake_data.get("config", {})
        try:
            mapped_config = map_handshake_config(config_dict)
        except ValueError as e:
            logger.warning(f"Handshake configuration invalid: {e}")
            await client_ws.close(code=1008)
            return

        # 3. Connect to downstream ASR Server
        logger.info(f"Connecting to ASR Server WebSocket at {self.asr_ws_url}...")
        try:
            async with websockets.connect(self.asr_ws_url) as asr_ws:
                # 4. Transmit handshake to ASR Server
                await asr_ws.send(json.dumps({
                    "event": "handshake",
                    "config": mapped_config
                }))

                # Wait for Handshake OK response
                asr_response = await asr_ws.recv()
                if isinstance(asr_response, str):
                    try:
                        resp_data = json.loads(asr_response)
                    except json.JSONDecodeError:
                        logger.error("ASR server returned malformed handshake response")
                        await client_ws.close(code=1011) # Internal Error
                        return

                    if resp_data.get("event") != "handshake_ok":
                        logger.error(f"ASR server handshake failed: {resp_data}")
                        await client_ws.close(code=1011)
                        return

                    # Forward Handshake OK to the client
                    await client_ws.send_text(asr_response)
                else:
                    logger.error("ASR server responded with binary data during handshake")
                    await client_ws.close(code=1011)
                    return

                # 5. Start bidirectional relay loops
                client_to_asr_task = asyncio.create_task(
                    self._relay_client_to_asr(client_ws, asr_ws)
                )
                asr_to_client_task = asyncio.create_task(
                    self._relay_asr_to_client(client_ws, asr_ws)
                )

                done, pending = await asyncio.wait(
                    [client_to_asr_task, asr_to_client_task],
                    return_when=asyncio.FIRST_COMPLETED
                )

                for task in pending:
                    task.cancel()

        except websockets.exceptions.ConnectionClosed as e:
            logger.warning(f"ASR WebSocket connection closed: {e}")
            try:
                await client_ws.close(code=1011)
            except Exception:
                pass
        except Exception as e:
            logger.error(f"Error establishing connection to ASR Server: {e}", exc_info=True)
            try:
                await client_ws.close(code=1011)
            except Exception:
                pass

    async def _relay_client_to_asr(self, client_ws: WebSocket, asr_ws: Any) -> None:
        """Forward frames from web client to ASR model server."""
        try:
            while True:
                message = await client_ws.receive()
                if message.get("type") == "websocket.disconnect":
                    logger.info("Client disconnected WebSocket stream")
                    break

                if "text" in message:
                    await asr_ws.send(message["text"])
                elif "bytes" in message:
                    await asr_ws.send(message["bytes"])
        except WebSocketDisconnect:
            logger.info("Client disconnected WebSocket stream (WebSocketDisconnect)")
        except Exception as e:
            logger.warning(f"Relaying client to ASR encountered error: {e}")

    async def _relay_asr_to_client(self, client_ws: WebSocket, asr_ws: Any) -> None:
        """Forward frames from ASR model server to web client."""
        try:
            async for message in asr_ws:
                if isinstance(message, str):
                    await client_ws.send_text(message)
                    try:
                        data = json.loads(message)
                        if data.get("event") == "finished":
                            logger.info("ASR session completed. Stopping relayer.")
                            break
                    except json.JSONDecodeError:
                        pass
                elif isinstance(message, bytes):
                    await client_ws.send_bytes(message)
        except Exception as e:
            logger.warning(f"Relaying ASR to client encountered error: {e}")
