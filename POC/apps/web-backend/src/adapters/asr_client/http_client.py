import os
import json
import asyncio
import logging
import httpx
from typing import Optional, Dict, Any
from src.domain.interfaces import IASRClient
from src.domain.entities import DecodingConfig

logger = logging.getLogger(__name__)

class ASRServerConnectionError(Exception):
    """Raised when the ASR Model Server is unreachable or returns a server error after retries."""
    pass

class ASRHTTPClient(IASRClient):
    def __init__(self, server_url: Optional[str] = None):
        self.server_url = server_url or os.getenv("ASR_SERVER_URL", "http://localhost:8001")
        # Ensure url does not end with a slash
        if self.server_url.endswith("/"):
            self.server_url = self.server_url[:-1]

    async def transcribe(
        self,
        audio_data: bytes,
        filename: str,
        decoding_config: Optional[DecodingConfig] = None
    ) -> Dict[str, Any]:
        """
        Send audio data to the ASR Model Server for transcription.
        Retries twice on connection issues before raising ASRServerConnectionError.
        """
        url = f"{self.server_url}/transcribe"
        data = {}
        
        if decoding_config:
            config_dict: Dict[str, Any] = {
                "method": decoding_config.method
            }
            if decoding_config.beam_size is not None:
                config_dict["beam_size"] = decoding_config.beam_size
            if decoding_config.causal is not None:
                config_dict["causal"] = decoding_config.causal
            if decoding_config.chunk_size is not None:
                config_dict["chunk_size"] = decoding_config.chunk_size
            if decoding_config.left_context_frames is not None:
                config_dict["left_context_frames"] = decoding_config.left_context_frames
            
            data["config"] = json.dumps(config_dict)

        files = {
            "file": (filename, audio_data, "audio/wav")
        }

        attempts = 3
        for attempt in range(attempts):
            try:
                # Use a reasonable timeout for audio file transcription
                async with httpx.AsyncClient(timeout=60.0) as client:
                    response = await client.post(url, files=files, data=data)
                    response.raise_for_status()
                    return response.json()
            except (httpx.RequestError, httpx.HTTPStatusError) as e:
                logger.warning(
                    f"ASR Server request failed (attempt {attempt + 1}/{attempts}) at URL {url}: {e}"
                )
                if attempt == attempts - 1:
                    raise ASRServerConnectionError(
                        f"ASR Inference Server at {self.server_url} is temporarily unreachable."
                    ) from e
                # Exponential backoff connection recovery (e.g. 0.5s, 1.0s) before next attempt
                backoff_duration = 0.5 * (2 ** attempt)
                logger.info(f"ASR HTTP Client: Connection recovery backing off for {backoff_duration} seconds before retry.")
                await asyncio.sleep(backoff_duration)
        
        raise ASRServerConnectionError("ASR Inference Server request failed after all attempts.")
