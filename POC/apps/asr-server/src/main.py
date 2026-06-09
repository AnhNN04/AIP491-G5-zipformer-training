import os
import json
import logging
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect
import uuid
from src.models.k2_decoder import K2Decoder

# Set up logging conforming to Constitution Principle V
logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    level=logging.INFO
)
logger = logging.getLogger("asr-server")

app = FastAPI(title="VietASR Model Serving API")

# Checkpoint path and JIT loading attempt
CHECKPOINT_PATH = os.getenv("ASR_CHECKPOINT_PATH", "viet_iter3_pseudo_label/exp/jit_script.pt")

# Initialize K2Decoder search space wrapper
decoder = K2Decoder(CHECKPOINT_PATH)

@app.get("/health")
def health_check():
    return {"status": "ok"}

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
        # Delegate request parsing and search space mapping to K2Decoder
        result = decoder.decode(
            audio_data=audio_bytes,
            method=method,
            beam_size=beam_size,
            causal=causal,
            chunk_size=chunk_size,
            left_context_frames=left_context_frames
        )
        return result
    except ValueError as e:
        logger.warning(f"Decoder parameters invalid: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error executing decoding space: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference processing error: {str(e)}")


@app.websocket("/api/stream")
async def websocket_stream(websocket: WebSocket):
    await websocket.accept()
    logger.info("ASR WS client connected")
    
    session_id = f"asr_sess_{uuid.uuid4().hex[:8]}"
    handshake_done = False
    
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
                    config = data.get("config", {})
                    logger.info(f"WS Handshake config received: {config}")
                    handshake_done = True
                    await websocket.send_json({
                        "event": "handshake_ok",
                        "session_id": session_id
                    })
                elif event == "stop":
                    logger.info("WS Stop frame received")
                    await websocket.send_json({
                        "event": "finished",
                        "full_transcript": "chào mừng bạn đến với hệ thống nhận dạng giọng nói",
                        "confidence": 0.95
                    })
                    break
                else:
                    logger.warning(f"Unknown event received on WS: {event}")
            
            elif "bytes" in message:
                if not handshake_done:
                    logger.warning("Received binary audio data before handshake")
                    await websocket.close(code=1008)
                    break
                
                audio_chunk = message["bytes"]
                logger.debug(f"Received audio chunk of size {len(audio_chunk)}")
                
                # Send back simulated incremental update
                await websocket.send_json({
                    "event": "transcript_update",
                    "text": "chào mừng bạn đến với hệ thống nhận dạng giọng nói",
                    "is_final": False,
                    "confidence": 0.95
                })
                
    except WebSocketDisconnect:
        logger.info("ASR WS client disconnected")
    except Exception as e:
        logger.error(f"WS error: {e}", exc_info=True)

