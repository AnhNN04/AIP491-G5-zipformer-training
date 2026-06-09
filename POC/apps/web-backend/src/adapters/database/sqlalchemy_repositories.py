from uuid import UUID
from typing import Optional, List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from src.domain.interfaces import (
    IUserRepository,
    IAudioFileRepository,
    IDecodingConfigRepository,
    ITranscriptionRepository,
    IDialectStatsRepository
)
from src.domain.entities import User, AudioFile, DecodingConfig, Transcription, DialectStats
from src.adapters.database.sqlalchemy_models import (
    UserModel,
    AudioFileModel,
    DecodingConfigModel,
    TranscriptionModel,
    DialectStatsModel
)

class SQLAlchemyUserRepository(IUserRepository):
    def __init__(self, session: AsyncSession):
        self.session = session

    async def save(self, user: User) -> User:
        orm_user = UserModel.from_domain(user)
        # Check if already exists in session or database to merge/add
        self.session.add(orm_user)
        await self.session.commit()
        return orm_user.to_domain()

    async def get_by_id(self, user_id: UUID) -> Optional[User]:
        result = await self.session.get(UserModel, user_id)
        if result:
            return result.to_domain()
        return None

    async def get_by_username(self, username: str) -> Optional[User]:
        stmt = select(UserModel).where(UserModel.username == username)
        result = await self.session.execute(stmt)
        orm_user = result.scalars().first()
        if orm_user:
            return orm_user.to_domain()
        return None


class SQLAlchemyAudioFileRepository(IAudioFileRepository):
    def __init__(self, session: AsyncSession):
        self.session = session

    async def save(self, audio_file: AudioFile) -> AudioFile:
        orm_audio = AudioFileModel.from_domain(audio_file)
        self.session.add(orm_audio)
        await self.session.commit()
        return orm_audio.to_domain()

    async def get_by_id(self, audio_id: UUID) -> Optional[AudioFile]:
        result = await self.session.get(AudioFileModel, audio_id)
        if result:
            return result.to_domain()
        return None

    async def get_all_by_user_id(self, user_id: UUID) -> List[AudioFile]:
        stmt = select(AudioFileModel).where(AudioFileModel.user_id == user_id)
        result = await self.session.execute(stmt)
        orm_audios = result.scalars().all()
        return [orm.to_domain() for orm in orm_audios]


class SQLAlchemyDecodingConfigRepository(IDecodingConfigRepository):
    def __init__(self, session: AsyncSession):
        self.session = session

    async def save(self, config: DecodingConfig) -> DecodingConfig:
        orm_config = DecodingConfigModel.from_domain(config)
        self.session.add(orm_config)
        await self.session.commit()
        return orm_config.to_domain()

    async def get_by_id(self, config_id: UUID) -> Optional[DecodingConfig]:
        result = await self.session.get(DecodingConfigModel, config_id)
        if result:
            return result.to_domain()
        return None


class SQLAlchemyTranscriptionRepository(ITranscriptionRepository):
    def __init__(self, session: AsyncSession):
        self.session = session

    async def save(self, transcription: Transcription) -> Transcription:
        orm_trans = TranscriptionModel.from_domain(transcription)
        self.session.add(orm_trans)
        await self.session.commit()
        return orm_trans.to_domain()

    async def get_by_id(self, transcription_id: UUID) -> Optional[Transcription]:
        result = await self.session.get(TranscriptionModel, transcription_id)
        if result:
            return result.to_domain()
        return None

    async def get_by_audio_file_id(self, audio_file_id: UUID) -> Optional[Transcription]:
        stmt = select(TranscriptionModel).where(TranscriptionModel.audio_file_id == audio_file_id)
        result = await self.session.execute(stmt)
        orm_trans = result.scalars().first()
        if orm_trans:
            return orm_trans.to_domain()
        return None


class SQLAlchemyDialectStatsRepository(IDialectStatsRepository):
    def __init__(self, session: AsyncSession):
        self.session = session

    async def save(self, stats: DialectStats) -> DialectStats:
        orm_stats = DialectStatsModel.from_domain(stats)
        self.session.add(orm_stats)
        await self.session.commit()
        return orm_stats.to_domain()

    async def get_by_id(self, stats_id: UUID) -> Optional[DialectStats]:
        result = await self.session.get(DialectStatsModel, stats_id)
        if result:
            return result.to_domain()
        return None

    async def get_by_audio_file_id(self, audio_file_id: UUID) -> Optional[DialectStats]:
        stmt = select(DialectStatsModel).where(DialectStatsModel.audio_file_id == audio_file_id)
        result = await self.session.execute(stmt)
        orm_stats = result.scalars().first()
        if orm_stats:
            return orm_stats.to_domain()
        return None

    async def get_all_dialect_stats(self) -> List[DialectStats]:
        stmt = select(DialectStatsModel)
        result = await self.session.execute(stmt)
        orm_stats = result.scalars().all()
        return [orm.to_domain() for orm in orm_stats]
