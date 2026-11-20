"""Schema bootstrap.

Alembic owns migrations in staging/production. For local dev and tests this
creates the schema directly so ``make dev`` works from a clean machine (§22.1).
"""

from __future__ import annotations

import asyncio

from app.core.logging import get_logger
from app.db.base import Base
from app.db.models import (  # noqa: F401 - imported for metadata registration
    AnswerSnapshot,
    Chunk,
    Document,
    DocumentInsight,
    DocumentTimeline,
    Feedback,
    IngestionJob,
    QueryLog,
    Session,
    User,
)
from app.db.session import async_engine

log = get_logger(__name__)


async def create_all() -> None:
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    log.info("db_schema_ready", tables=sorted(Base.metadata.tables))


async def drop_all() -> None:
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    log.warning("db_schema_dropped")


if __name__ == "__main__":
    asyncio.run(create_all())
