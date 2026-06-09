import pytest
from src.domain.interfaces import (
    IUserRepository,
    IAudioFileRepository,
    IDecodingConfigRepository,
    ITranscriptionRepository,
    IDialectStatsRepository,
    IStorageService,
)

def test_interfaces_are_abstract():
    # Attempting to instantiate abstract base classes should raise TypeError
    with pytest.raises(TypeError):
        IUserRepository()
    
    with pytest.raises(TypeError):
        IAudioFileRepository()

    with pytest.raises(TypeError):
        IDecodingConfigRepository()

    with pytest.raises(TypeError):
        ITranscriptionRepository()

    with pytest.raises(TypeError):
        IDialectStatsRepository()

    with pytest.raises(TypeError):
        IStorageService()


def test_dummy_implementation():
    # A complete subclass should be instantiable
    class DummyUserRepo(IUserRepository):
        async def save(self, user):
            return user
        async def get_by_id(self, user_id):
            return None
        async def get_by_username(self, username):
            return None

    repo = DummyUserRepo()
    assert repo is not None
