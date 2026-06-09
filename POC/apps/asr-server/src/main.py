import os
import json
import logging
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
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
