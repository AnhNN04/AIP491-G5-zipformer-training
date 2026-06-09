import os
import uuid
import logging
from typing import Optional, Dict, Any
from src.domain.entities import AudioFile, DecodingConfig, Transcription, DialectStats
from src.domain.interfaces import (
    IAudioFileRepository,
    ITranscriptionRepository,
    IDecodingConfigRepository,
    IDialectStatsRepository,
    IStorageService,
    IASRClient
)
from src.usecases.audio_process import AudioProcessor
from src.usecases.dialect_routing import DialectRouter

logger = logging.getLogger(__name__)

class TranscribeAudioUseCase:
    def __init__(
        self,
        audio_file_repo: IAudioFileRepository,
        transcription_repo: ITranscriptionRepository,
        decoding_config_repo: IDecodingConfigRepository,
        dialect_stats_repo: IDialectStatsRepository,
        storage_service: IStorageService,
        asr_client: IASRClient,
        audio_processor: AudioProcessor,
        dialect_router: Optional[DialectRouter] = None
    ):
        self.audio_file_repo = audio_file_repo
        self.transcription_repo = transcription_repo
        self.decoding_config_repo = decoding_config_repo
        self.dialect_stats_repo = dialect_stats_repo
        self.storage_service = storage_service
        self.asr_client = asr_client
        self.audio_processor = audio_processor
        self.dialect_router = dialect_router or DialectRouter()

    async def execute(
        self,
        file_data: bytes,
        filename: str,
        original_format: str,
        user_id: Optional[uuid.UUID] = None,
        decoding_config_input: Optional[DecodingConfig] = None
    ) -> Dict[str, Any]:
        """
        Coordinates the entire audio transcription workflow:
        1. Validates format.
        2. Transcodes to standard WAV via FFmpeg.
        3. Saves decoding configuration parameters.
        4. Uploads raw original and standard transcoded WAV files to MinIO.
        5. Saves AudioFile database metadata records.
        6. Submits transcription requests to ASR Model Server.
        7. Logs final transcriptions and dialect classification stats.
        """
        logger.info(f"Executing TranscribeAudioUseCase for file: {filename}")
        
        # 1. Validate file format extension
        self.audio_processor.validate_format(original_format)

        # 2. Asynchronously transcode raw audio bytes
        standard_wav_data, duration = await self.audio_processor.transcode(file_data, original_format)

        # 3. Store decoding configuration parameters
        config_entity = decoding_config_input or DecodingConfig(method="greedy_search")
        saved_config = await self.decoding_config_repo.save(config_entity)

        # 4. Generate unique ID and object names for storage
        file_uuid = uuid.uuid4()
        
        # Keep original extension for raw uploads
        ext = os.path.splitext(filename)[1]
        if not ext and original_format:
            ext = f".{original_format.replace('.', '')}"
        raw_object_name = f"{file_uuid}{ext}"
        std_object_name = f"{file_uuid}.wav"

        # Asynchronously upload files to raw and standardized buckets
        # Bucket names match MINIO_BUCKET_RAW and MINIO_BUCKET_STANDARD in .env
        raw_storage_path = await self.storage_service.upload_file(
            bucket_name="raw-uploads",
            object_name=raw_object_name,
            data=file_data,
            content_type=f"audio/{original_format.replace('.', '')}"
        )

        standard_storage_path = await self.storage_service.upload_file(
            bucket_name="standardized-wavs",
            object_name=std_object_name,
            data=standard_wav_data,
            content_type="audio/wav"
        )

        # 5. Log AudioFile metadata in database
        audio_file_entity = AudioFile(
            id=file_uuid,
            user_id=user_id,
            raw_storage_path=raw_storage_path,
            standard_storage_path=standard_storage_path,
            original_format=original_format.replace(".", ""),
            duration_seconds=duration,
            size_bytes=len(file_data)
        )
        saved_audio = await self.audio_file_repo.save(audio_file_entity)

        # 6. Relayout request to ASR inference engine
        asr_result = await self.asr_client.transcribe(
            audio_data=standard_wav_data,
            filename=std_object_name,
            decoding_config=saved_config
        )

        # 7. Log transcription results
        transcription_entity = Transcription(
            audio_file_id=saved_audio.id,
            decoding_config_id=saved_config.id,
            text=asr_result["text"],
            confidence=asr_result["confidence"],
            word_alignments=asr_result.get("word_alignments")
        )
        saved_transcription = await self.transcription_repo.save(transcription_entity)

        # 8. Log dialect/accent routing classification
        dialect_info = self.dialect_router.classify_dialect(asr_result["text"])
        dialect_entity = DialectStats(
            audio_file_id=saved_audio.id,
            inferred_dialect=dialect_info["inferred"],
            dialect_probability=dialect_info["probability"]
        )
        saved_dialect = await self.dialect_stats_repo.save(dialect_entity)

        logger.info(f"TranscribeAudioUseCase finished successfully. Audio ID: {saved_audio.id}")

        return {
            "audio_id": saved_audio.id,
            "duration_seconds": saved_audio.duration_seconds,
            "size_bytes": saved_audio.size_bytes,
            "status": "SUCCESS",
            "transcription": {
                "text": saved_transcription.text,
                "confidence": saved_transcription.confidence,
                "word_alignments": saved_transcription.word_alignments
            },
            "dialect": {
                "inferred": saved_dialect.inferred_dialect,
                "probability": saved_dialect.dialect_probability
            }
        }
