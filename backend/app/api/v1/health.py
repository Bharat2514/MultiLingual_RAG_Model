"""Liveness and readiness (§9.1, NFR-5).

``/health`` is a cheap liveness probe. ``/health/ready`` actually touches
Postgres, Redis, and Qdrant so orchestrators do not route traffic to a worker
whose dependencies are down.
"""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends, Response, status
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.context import get_request_id
from app.core.logging import get_logger
from app.db.session import get_db
from app.schemas.common import ComponentHealth, HealthResponse

router = APIRouter(tags=["health"])
log = get_logger(__name__)


async def _check_postgres(db: AsyncSession) -> ComponentHealth:
    started = time.perf_counter()
    try:
        await db.execute(text("SELECT 1"))
    except Exception as exc:  # noqa: BLE001 - health checks must never raise
        return ComponentHealth(name="postgres", status="down", detail=str(exc)[:200])
    return ComponentHealth(
        name="postgres", status="ok", latency_ms=round((time.perf_counter() - started) * 1000, 2)
    )


async def _check_redis() -> ComponentHealth:
    started = time.perf_counter()
    try:
        from redis.asyncio import Redis

        client = Redis.from_url(settings.redis.url, socket_connect_timeout=2)
        try:
            await client.ping()
        finally:
            await client.aclose()
    except Exception as exc:  # noqa: BLE001
        return ComponentHealth(name="redis", status="down", detail=str(exc)[:200])
    return ComponentHealth(
        name="redis", status="ok", latency_ms=round((time.perf_counter() - started) * 1000, 2)
    )


async def _check_qdrant() -> ComponentHealth:
    started = time.perf_counter()
    try:
        from qdrant_client import AsyncQdrantClient

        client = AsyncQdrantClient(
            url=settings.qdrant.url, api_key=settings.qdrant.api_key, timeout=5
        )
        try:
            await client.get_collections()
            exists = await client.collection_exists(settings.qdrant.collection)
            detail = (
                f"collection '{settings.qdrant.collection}' ready"
                if exists
                else f"collection '{settings.qdrant.collection}' not created yet"
            )
        finally:
            await client.close()
    except Exception as exc:  # noqa: BLE001
        return ComponentHealth(name="qdrant", status="down", detail=str(exc)[:200])
    return ComponentHealth(
        name="qdrant",
        status="ok",
        detail=detail,
        latency_ms=round((time.perf_counter() - started) * 1000, 2),
    )


def _check_models() -> ComponentHealth:
    """Report the configured retrieval models without loading them — a health
    probe must never block on a multi-gigabyte download."""
    detail = f"embed={settings.embedding.model_name}"
    if settings.reranker.enabled:
        detail += f" rerank={settings.reranker.model_name}"
    else:
        detail += " rerank=disabled"

    try:
        from app.core.devices import resolve_device

        detail += f" device={resolve_device(settings.embedding.device)}"
    except Exception:  # noqa: BLE001
        pass
    return ComponentHealth(name="models", status="ok", detail=detail)


def _check_llm() -> ComponentHealth:
    provider = settings.llm.provider
    if provider == "mock":
        return ComponentHealth(name="llm", status="disabled", detail="mock provider")
    if provider == "ollama":
        # Local and keyless: report where it points; reachability is a runtime
        # concern surfaced as LLM_UNAVAILABLE on the first failed call.
        return ComponentHealth(
            name="llm",
            status="ok",
            detail=f"ollama:{settings.llm.resolved_model} @ {settings.llm.ollama_base_url}",
        )
    if not settings.llm_api_key:
        return ComponentHealth(
            name="llm",
            status="degraded",
            detail=f"{settings.llm.api_key_env} is not set",
        )
    return ComponentHealth(
        name="llm", status="ok", detail=f"{provider}:{settings.llm.resolved_model}"
    )


@router.get("/health", response_model=HealthResponse, summary="Liveness probe")
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        version=settings.app.version,
        environment=settings.app.env,
        request_id=get_request_id(),
    )


@router.get("/health/ready", response_model=HealthResponse, summary="Readiness probe")
async def readiness(response: Response, db: AsyncSession = Depends(get_db)) -> HealthResponse:
    components = [
        await _check_postgres(db),
        await _check_redis(),
        await _check_qdrant(),
        _check_models(),
        _check_llm(),
    ]

    if any(c.status == "down" for c in components):
        overall = "down"
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    elif any(c.status == "degraded" for c in components):
        overall = "degraded"
    else:
        overall = "ok"

    return HealthResponse(
        status=overall,
        version=settings.app.version,
        environment=settings.app.env,
        request_id=get_request_id(),
        components=components,
    )
