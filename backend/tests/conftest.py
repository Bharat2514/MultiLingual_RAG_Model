"""Shared pytest fixtures.

Unit tests run against an in-memory SQLite database and never touch Postgres,
Redis, or Qdrant. Anything that needs those is marked ``integration``.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator

import pytest

# Must be set before app.core.config is first imported.
os.environ.setdefault("APP__ENV", "test")
os.environ.setdefault("APP__DEBUG", "true")
os.environ.setdefault("DB__URL", "sqlite+aiosqlite:///:memory:")
os.environ.setdefault("LLM__PROVIDER", "mock")
os.environ.setdefault("OBS__LOG_LEVEL", "WARNING")
os.environ.setdefault("OBS__METRICS_ENABLED", "false")
# Tests must not load 2.3 GB of weights during app startup.
os.environ.setdefault("RETRIEVAL__WARMUP_ON_STARTUP", "false")

import httpx
from asgi_lifespan import LifespanManager  # type: ignore[import-not-found]
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.db.base import Base
from app.db.models import Document, Session, User  # noqa: F401
from app.db.session import get_db
from app.main import create_app


@pytest.fixture(scope="session")
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
async def db_engine() -> AsyncIterator:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest.fixture
async def db_session(db_engine) -> AsyncIterator[AsyncSession]:
    factory = async_sessionmaker(bind=db_engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as session:
        yield session


@pytest.fixture
async def app(db_session: AsyncSession):
    application = create_app()

    async def _override_get_db() -> AsyncIterator[AsyncSession]:
        yield db_session

    application.dependency_overrides[get_db] = _override_get_db
    yield application
    application.dependency_overrides.clear()


@pytest.fixture
async def client(app) -> AsyncIterator[httpx.AsyncClient]:
    async with LifespanManager(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac
