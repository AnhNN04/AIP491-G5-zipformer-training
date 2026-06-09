import os
import json
import logging
import asyncio
from typing import Optional, Dict
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect
import uuid
from src.models.k2_decoder import K2Decoder
from src.models.decode_stream import DecodeStream

# Set up logging conforming to Constitution Principle V
logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    level=logging.INFO
)
logger = logging.getLogger("asr-server")

app = FastAPI(title="VietASR Model Serving API")

# Configuration from environment
CHECKPOINT_PATH = os.getenv("ASR_CHECKPOINT_PATH", "")
TOKENS_PATH = os.getenv("ASR_TOKENS_PATH", "")

if not CHECKPOINT_PATH:
    logger.warning("ASR_CHECKPOINT_PATH not set — model will not be loaded at startup.")
if not TOKENS_PATH:
    logger.warning("ASR_TOKENS_PATH not set — token table will not be loaded at startup.")

# Initialize K2Decoder — loads JIT model and BPE token table
decoder = K2Decoder(
    checkpoint_path=CHECKPOINT_PATH or None,
    tokens_path=TOKENS_PATH or None,
)

# Active WebSocket sessions: { session_id -> DecodeStream }
_sessions: Dict[str, DecodeStream] = {}


@app.get("/health")
def health_check():
    model_ready = decoder.model is not None and decoder.token_table is not None
    return {"status": "ok", "model_ready": model_ready}


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    config: Optional[str] = Form(None)
):
    logger.info(f"Received transcription request for file: {file.filename}")

    # Parse decoding parameters
    decoding_config = {}
    if config:
        try:
            decoding_config = json.loads(config)
            logger.info(f"Parsed decoding options: {decoding_config}")
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to decode config parameter JSON: {e}")
            raise HTTPException(status_code=400, detail="Invalid JSON format in config parameter.")

    method = decoding_config.get("method", "greedy_search")
    beam_size = decoding_config.get("beam_size", 4)
    causal = decoding_config.get("causal", False)
    chunk_size = decoding_config.get("chunk_size")
    left_context_frames = decoding_config.get("left_context_frames")

    audio_bytes = await file.read()

    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            lambda: decoder.decode(
                audio_data=audio_bytes,
                method=method,
                beam_size=beam_size,
                causal=causal,
                chunk_size=chunk_size,
                left_context_frames=left_context_frames,
            )
        )
        return result
    except ValueError as e:
        logger.warning(f"Decoder parameters invalid: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error executing decoding: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference processing error: {str(e)}")


@app.websocket("/api/stream")
async def websocket_stream(websocket: WebSocket):
    await websocket.accept()
    logger.info("ASR WS client connected")

    session_id = f"asr_sess_{uuid.uuid4().hex[:8]}"
    handshake_done = False
    stream: Optional[DecodeStream] = None

    try:
        while True:
            message = await websocket.receive()

            if "text" in message:
                text_data = message["text"]
                try:
                    data = json.loads(text_data)
                except json.JSONDecodeError:
                    logger.warning("Received invalid JSON on WS")
                    await websocket.close(code=1008)
                    break

                event = data.get("event")
                if event == "handshake":
                    ws_config = data.get("config", {})
                    logger.info(f"WS Handshake config received: {ws_config}")
                    stream = DecodeStream(session_id=session_id, config=ws_config)
                    _sessions[session_id] = stream
                    handshake_done = True
                    await websocket.send_json({
                        "event": "handshake_ok",
                        "session_id": session_id,
                    })

                elif event == "stop":
                    logger.info(f"WS Stop frame received for session {session_id}")
                    loop = asyncio.get_running_loop()
                    if stream is not None:
                        full_text = await loop.run_in_executor(
                            None,
                            lambda: decoder.decode_stream_session(stream),
                        )
                        stream.reset()
                    else:
                        full_text = ""
                    _sessions.pop(session_id, None)
                    await websocket.send_json({
                        "event": "finished",
                        "full_transcript": full_text,
                        "confidence": 0.95,
                    })
                    break
                else:
                    logger.warning(f"Unknown event received on WS: {event}")

            elif "bytes" in message:
                if not handshake_done or stream is None:
                    logger.warning("Received binary audio data before handshake")
                    await websocket.close(code=1008)
                    break

                audio_chunk = message["bytes"]
                logger.debug(f"Received audio chunk of size {len(audio_chunk)} bytes")

                # Accumulate PCM chunk into the session's DecodeStream
                stream.add_chunk(audio_chunk)

                # Run decode_stream_session in executor (non-blocking)
                loop = asyncio.get_running_loop()
                partial_text = await loop.run_in_executor(
                    None,
                    lambda: decoder.decode_stream_session(stream),
                )

                await websocket.send_json({
                    "event": "transcript_update",
                    "text": partial_text,
                    "is_final": False,
                    "confidence": 0.95,
                })

    except WebSocketDisconnect:
        logger.info(f"ASR WS client disconnected — session {session_id}")
        _sessions.pop(session_id, None)
    except Exception as e:
        logger.error(f"WS error: {e}", exc_info=True)
        _sessions.pop(session_id, None)
