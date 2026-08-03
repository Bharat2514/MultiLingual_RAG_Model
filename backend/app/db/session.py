"""Async engine/session factory plus a sync factory for Celery workers.

The API is fully async (asyncpg). The ingestion worker is CPU/IO-bound sync code
(PyMuPDF, Tesseract, torch), so it uses a separate synchronous engine rather
than bridging event loops.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager

from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import Session as SyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

from app.core.config import settings

_engine_kwargs: dict = {"echo": settings.db.echo, "future": True}
if not settings.db.is_sqlite:
    _engine_kwargs.update(
        pool_size=settings.db.pool_size,
        max_overflow=settings.db.max_overflow,
        pool_pre_ping=settings.db.pool_pre_ping,
    )
else:
    _engine_kwargs["poolclass"] = NullPool

async_engine: AsyncEngine = create_async_engine(settings.db.url, **_engine_kwargs)

AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)

_sync_engine = create_engine(
    settings.db.sync_url,
    echo=settings.db.echo,
    future=True,
    pool_pre_ping=settings.db.pool_pre_ping,
)

SyncSessionLocal = sessionmaker(bind=_sync_engine, class_=SyncSession, expire_on_commit=False)


async def get_db() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency. Commits on success, rolls back on any exception."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


@asynccontextmanager
async def async_session_scope() -> AsyncIterator[AsyncSession]:
    """Transactional scope for code outside the request lifecycle."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


@contextmanager
def sync_session_scope() -> Iterator[SyncSession]:
    """Transactional scope for Celery workers."""
    session = SyncSessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


async def dispose_engines() -> None:
    await async_engine.dispose()
    _sync_engine.dispose()
