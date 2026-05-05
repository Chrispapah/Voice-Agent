from __future__ import annotations

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def _normalize_database_url(database_url: str) -> str:
    if database_url.startswith("postgresql://"):
        return database_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    if database_url.startswith("postgres://"):
        return database_url.replace("postgres://", "postgresql+asyncpg://", 1)
    return database_url


def _get_engine(database_url: str) -> AsyncEngine:
    global _engine
    if _engine is None:
        _engine = create_async_engine(
            _normalize_database_url(database_url),
            echo=False,
            pool_size=5,
            max_overflow=10,
        )
    return _engine


async def init_db(database_url: str) -> None:
    """Initialise the async engine and session factory.

    Tables are managed by Supabase migrations -- this function only sets up
    the SQLAlchemy connection pool.
    """
    engine = _get_engine(database_url)
    global _session_factory
    _session_factory = async_sessionmaker(engine, expire_on_commit=False)


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    if _session_factory is None:
        raise RuntimeError("Database not initialised – call init_db() first")
    async with _session_factory() as session:
        yield session
