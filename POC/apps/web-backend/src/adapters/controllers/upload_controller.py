import os
import json
import logging
from typing import Optional, Dict, Any
from fastapi import UploadFile, HTTPException, Form
from src.domain.entities import DecodingConfig
from src.usecases.transcribe import TranscribeAudioUseCase
from src.adapters.asr_client.http_client import ASRServerConnectionError
from src.usecases.audio_process import InvalidAudioFormatException

logger = logging.getLogger(__name__)

class UploadController:
    def __init__(self, transcribe_usecase: TranscribeAudioUseCase):
        self.transcribe_usecase = transcribe_usecase

    async def upload_and_transcribe(
        self,
        file: UploadFile,
        config: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        REST Controller to handle multi-format audio uploads and transcribing.
        Validates size, format, decoding parameters, and returns structured JSON.
        """
        logger.info(f"UploadController received file: {file.filename}")
        
        # 1. Validate file format extension from name
        if not file.filename:
            raise HTTPException(status_code=400, detail="Filename is missing.")
            
        ext = os.path.splitext(file.filename)[1].lower().replace(".", "")
        if ext not in ('wav', 'mp3', 'flac', 'm4a', 'ogg', 'mp4'):
            raise HTTPException(
                status_code=400,
                detail="Unsupported file format. Supported formats: .wav, .mp3, .flac, .m4a, .ogg, .mp4"
            )

        # 2. Read bytes and enforce size limits (50MB default)
        file_data = await file.read()
        max_size_bytes = int(os.getenv("MAX_FILE_SIZE_BYTES", 52428800))
        
        if len(file_data) > max_size_bytes:
            logger.warning(
                f"Rejected file {file.filename} of size {len(file_data)} bytes: exceeds {max_size_bytes} limit"
            )
            raise HTTPException(
                status_code=413,
                detail="File size exceeds the 50MB maximum limit."
            )

        # 3. Deserialize and validate decoding configurations
        decoding_config = DecodingConfig(method="greedy_search")
        if config:
            try:
                config_dict = json.loads(config)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse config JSON in controller: {e}")
                raise HTTPException(status_code=400, detail="Invalid JSON format in config parameter.")
            
            method = config_dict.get("method", "greedy_search")
            if method not in ("greedy_search", "modified_beam_search"):
                raise HTTPException(
                    status_code=400,
                    detail="Invalid decoding method. Must be greedy_search or modified_beam_search."
                )
            
            beam_size = config_dict.get("beam_size")
            if beam_size is not None:
                try:
                    beam_size = int(beam_size)
                    if not (1 <= beam_size <= 20):
                        raise ValueError()
                except ValueError:
                    raise HTTPException(
                        status_code=400,
                        detail="Invalid beam_size. Must be an integer between 1 and 20."
                    )
            elif method == "modified_beam_search":
                # Default beam size for beam search
                beam_size = 4
            
            causal = config_dict.get("causal", False)
            if not isinstance(causal, bool):
                raise HTTPException(
                    status_code=400,
                    detail="Invalid causal parameter. Must be a boolean."
                )

            chunk_size = config_dict.get("chunk_size")
            if chunk_size is not None:
                try:
                    chunk_size = int(chunk_size)
                    if chunk_size <= 0:
                        raise ValueError()
                except ValueError:
                    raise HTTPException(
                        status_code=400,
                        detail="Invalid chunk_size. Must be a positive integer."
                    )

            left_context_frames = config_dict.get("left_context_frames")
            if left_context_frames is not None:
                try:
                    left_context_frames = int(left_context_frames)
                    if left_context_frames < 0:
                        raise ValueError()
                except ValueError:
                    raise HTTPException(
                        status_code=400,
                        detail="Invalid left_context_frames. Must be a non-negative integer."
                    )

            decoding_config = DecodingConfig(
                method=method,
                beam_size=beam_size,
                causal=causal,
                chunk_size=chunk_size,
                left_context_frames=left_context_frames
            )

        # 4. Dispatch coordinates to usecase
        try:
            result = await self.transcribe_usecase.execute(
                file_data=file_data,
                filename=file.filename,
                original_format=ext,
                decoding_config_input=decoding_config
            )
            return result
        except InvalidAudioFormatException as e:
            raise HTTPException(status_code=400, detail=str(e))
        except ASRServerConnectionError as e:
            logger.error(f"ASR Server connection error: {e}")
            raise HTTPException(
                status_code=503,
                detail="ASR Inference Server is temporarily unreachable. Please try again."
            )
        except Exception as e:
            logger.error(f"Internal error processing transcription: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"An internal server error occurred: {str(e)}"
            )
