from abc import ABC, abstractmethod
from uuid import UUID
from typing import Optional, List, Dict, Any
from .entities import User, AudioFile, DecodingConfig, Transcription, DialectStats

class IUserRepository(ABC):
    @abstractmethod
    async def save(self, user: User) -> User:
        """Save a user entity to the database."""
        pass

    @abstractmethod
    async def get_by_id(self, user_id: UUID) -> Optional[User]:
        """Retrieve a user by their unique ID."""
        pass

    @abstractmethod
    async def get_by_username(self, username: str) -> Optional[User]:
        """Retrieve a user by their username."""
        pass


class IAudioFileRepository(ABC):
    @abstractmethod
    async def save(self, audio_file: AudioFile) -> AudioFile:
        """Save an audio file metadata entity to the database."""
        pass

    @abstractmethod
    async def get_by_id(self, audio_id: UUID) -> Optional[AudioFile]:
        """Retrieve audio file metadata by its unique ID."""
        pass

    @abstractmethod
    async def get_all_by_user_id(self, user_id: UUID) -> List[AudioFile]:
        """Retrieve all audio files uploaded by a specific user."""
        pass


class IDecodingConfigRepository(ABC):
    @abstractmethod
    async def save(self, config: DecodingConfig) -> DecodingConfig:
        """Save a decoding configuration entity to the database."""
        pass

    @abstractmethod
    async def get_by_id(self, config_id: UUID) -> Optional[DecodingConfig]:
        """Retrieve decoding configuration by its unique ID."""
        pass


class ITranscriptionRepository(ABC):
    @abstractmethod
    async def save(self, transcription: Transcription) -> Transcription:
        """Save a transcription entity to the database."""
        pass

    @abstractmethod
    async def get_by_id(self, transcription_id: UUID) -> Optional[Transcription]:
        """Retrieve transcription by its unique ID."""
        pass

    @abstractmethod
    async def get_by_audio_file_id(self, audio_file_id: UUID) -> Optional[Transcription]:
        """Retrieve transcription associated with a specific audio file."""
        pass


class IDialectStatsRepository(ABC):
    @abstractmethod
    async def save(self, stats: DialectStats) -> DialectStats:
        """Save dialect statistics entity to the database."""
        pass

    @abstractmethod
    async def get_by_id(self, stats_id: UUID) -> Optional[DialectStats]:
        """Retrieve dialect stats by its unique ID."""
        pass

    @abstractmethod
    async def get_by_audio_file_id(self, audio_file_id: UUID) -> Optional[DialectStats]:
        """Retrieve dialect stats associated with a specific audio file."""
        pass

    @abstractmethod
    async def get_all_dialect_stats(self) -> List[DialectStats]:
        """Retrieve all dialect statistics."""
        pass


class IStorageService(ABC):
    @abstractmethod
    async def upload_file(self, bucket_name: str, object_name: str, data: bytes, content_type: str) -> str:
        """
        Upload binary data to standard object storage and return its location/path.
        """
        pass

    @abstractmethod
    async def download_file(self, bucket_name: str, object_name: str) -> bytes:
        """
        Download binary data from standard object storage.
        """
        pass

    @abstractmethod
    async def get_presigned_url(self, bucket_name: str, object_name: str, expires: int = 3600) -> str:
        """
        Generate a presigned download URL for the object.
        """
        pass

    @abstractmethod
    async def delete_file(self, bucket_name: str, object_name: str) -> None:
        """
        Delete an object from storage.
        """
        pass


class IASRClient(ABC):
    @abstractmethod
    async def transcribe(
        self,
        audio_data: bytes,
        filename: str,
        decoding_config: Optional[DecodingConfig] = None
    ) -> Dict[str, Any]:
        """
        Send audio data to the ASR Model Server for transcription.
        Returns a dict containing text, confidence, word alignments, etc.
        """
        pass

