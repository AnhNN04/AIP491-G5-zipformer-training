from typing import Optional
from fastapi import APIRouter, Depends, UploadFile, File, Form
from sqlalchemy.ext.asyncio import AsyncSession
from src.frameworks.database import get_db_session
from src.adapters.storage.minio_client import MinioStorageService
from src.adapters.asr_client.http_client import ASRHTTPClient
from src.adapters.database.sqlalchemy_repositories import (
    SQLAlchemyAudioFileRepository,
    SQLAlchemyTranscriptionRepository,
    SQLAlchemyDecodingConfigRepository,
    SQLAlchemyDialectStatsRepository
)
from src.usecases.audio_process import AudioProcessor
from src.usecases.transcribe import TranscribeAudioUseCase
from src.adapters.controllers.upload_controller import UploadController

router = APIRouter()

@router.post("/upload")
async def upload_audio(
    file: UploadFile = File(...),
    config: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db_session)
):
    """
    POST route for batch audio/video upload and transcription.
    Enforces format transcoding, storage upload, database logs, and returns ASR text.
    """
    # Instantiate repositories
    audio_file_repo = SQLAlchemyAudioFileRepository(db)
    transcription_repo = SQLAlchemyTranscriptionRepository(db)
    decoding_config_repo = SQLAlchemyDecodingConfigRepository(db)
    dialect_stats_repo = SQLAlchemyDialectStatsRepository(db)
    
    # Instantiate adapters
    storage_service = MinioStorageService()
    asr_client = ASRHTTPClient()
    audio_processor = AudioProcessor()
    
    # Instantiate usecase
    transcribe_usecase = TranscribeAudioUseCase(
        audio_file_repo=audio_file_repo,
        transcription_repo=transcription_repo,
        decoding_config_repo=decoding_config_repo,
        dialect_stats_repo=dialect_stats_repo,
        storage_service=storage_service,
        asr_client=asr_client,
        audio_processor=audio_processor
    )
    
    # Instantiate controller
    controller = UploadController(transcribe_usecase)
    
    return await controller.upload_and_transcribe(file=file, config=config)


from fastapi import WebSocket
from src.adapters.controllers.stream_controller import StreamController

@router.websocket("/stream")
async def websocket_stream(websocket: WebSocket):
    """
    WebSocket endpoint for real-time bidirectional audio streaming and transcript updates.
    """
    controller = StreamController()
    await controller.handle_stream(websocket)
