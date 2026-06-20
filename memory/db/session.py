"""
Async SQLAlchemy engine and session factory.

The engine is a module-level singleton initialized lazily on first use so
that importing memory.db.session in tooling (alembic, scripts) doesn't open
a connection unless required.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from graphrag.config.settings import settings

logger = logging.getLogger(__name__)


_engine: AsyncEngine | None = None
_factory: async_sessionmaker[AsyncSession] | None = None


def _build_engine() -> AsyncEngine:
    url = getattr(settings, "DATABASE_URL", None)
    if not url:
        raise RuntimeError(
            "DATABASE_URL is not set. Add it to your .env (see .env.example). "
            "Expected format: postgresql+asyncpg://user:pass@host:5432/db"
        )
    logger.info("Initializing async SQLAlchemy engine")
    return create_async_engine(
        url,
        pool_pre_ping=True,
        pool_size=10,
        max_overflow=20,
        echo=False,                          # flip to True for SQL trace in dev
        future=True,
    )


def AsyncSessionFactory() -> async_sessionmaker[AsyncSession]:
    """Return the process-wide async session factory, building it on first call."""
    global _engine, _factory
    if _factory is None:
        _engine = _build_engine()
        _factory = async_sessionmaker(
            bind=_engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )
    return _factory


@asynccontextmanager
async def get_session() -> AsyncIterator[AsyncSession]:
    """
    Async context manager yielding a session with automatic commit/rollback.

    Usage:
        async with get_session() as session:
            session.add(obj)
            # commit on exit, rollback on exception
    """
    factory = AsyncSessionFactory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def dispose_engine() -> None:
    """Close the engine and release the connection pool. Call on shutdown."""
    global _engine, _factory
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _factory = None
