import pytest
from src.frameworks.database import engine, async_session_factory

@pytest.mark.anyio
async def test_database_connection_pool():
    assert engine is not None
    assert async_session_factory is not None
