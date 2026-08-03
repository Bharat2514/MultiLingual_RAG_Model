"""FastAPI application factory.

Owns the lifespan (schema bootstrap, engine disposal), middleware stack, and the
exception handlers that guarantee every failure leaves as the §9.4 error
envelope — no unhandled 500 ever reaches the user (NFR-5).
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.gzip import GZipMiddleware

from app.api.v1.router import api_router
from app.core.config import settings
from app.core.context import get_request_id
from app.core.errors import ErrorCode, PolyDocError
from app.core.logging import configure_logging, get_logger
from app.core.middleware import AccessLogMiddleware, RequestContextMiddleware
from app.db.session import dispose_engines
from app.schemas.common import ErrorResponse

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_logging()
    log.info(
        "startup",
        app=settings.app.name,
        version=settings.app.version,
        env=settings.app.env,
        llm_provider=settings.llm.provider,
        llm_model=settings.llm.model,
    )

    settings.storage.local_dir.mkdir(parents=True, exist_ok=True)
    settings.embedding.cache_dir.mkdir(parents=True, exist_ok=True)

    warmup_task: asyncio.Task | None = None
    if settings.retrieval.warmup_on_startup:
        warmup_task = asyncio.create_task(_warm_models())

    if not settings.app.is_production:
        # Dev convenience; production applies Alembic migrations instead.
        from app.db.init_db import create_all

        try:
            await create_all()
        except Exception as exc:  # noqa: BLE001 - never block boot on a cold DB
            log.warning("db_bootstrap_skipped", error=str(exc))

    try:
        yield
    finally:
        if warmup_task is not None and not warmup_task.done():
            warmup_task.cancel()
        await dispose_engines()
        log.info("shutdown")


async def _warm_models() -> None:
    """Compile the retrieval models' kernels off the request path.

    Runs in a worker thread so it never blocks the event loop, and failures are
    swallowed: a cold model means a slow first query, not a reason to refuse
    traffic (NFR-5).
    """
    import time

    def _warm() -> None:
        from app.retrieval import factory

        factory.get_embedder().warmup()
        factory.get_reranker().warmup()

    started = time.perf_counter()
    try:
        await asyncio.to_thread(_warm)
    except asyncio.CancelledError:
        raise
    except Exception:
        log.warning("model_warmup_failed", exc_info=True)
        return
    log.info("models_warm", duration_ms=round((time.perf_counter() - started) * 1000, 2))


def _error_response(
    status_code: int, code: str, message: str, details: dict[str, Any] | None = None
) -> JSONResponse:
    error: dict[str, Any] = {"code": code, "message": message}
    if details:
        error["details"] = details
    return JSONResponse(
        status_code=status_code,
        content={"request_id": get_request_id(), "error": error},
    )


def create_app() -> FastAPI:
    configure_logging()

    app = FastAPI(
        title=f"{settings.app.name} API",
        version=settings.app.version,
        description=(
            "Multilingual (Hindi / English / Hinglish) document-grounded question "
            "answering with cross-lingual hybrid retrieval and inline citations."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
        responses={
            400: {"model": ErrorResponse},
            422: {"model": ErrorResponse},
            500: {"model": ErrorResponse},
        },
    )

    # Order matters: the outermost middleware runs first on the way in.
    app.add_middleware(AccessLogMiddleware)
    app.add_middleware(GZipMiddleware, minimum_size=1024)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.app.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-Response-Time-ms"],
    )
    app.add_middleware(RequestContextMiddleware)

    app.include_router(api_router, prefix=settings.app.api_prefix)

    @app.exception_handler(PolyDocError)
    async def polydoc_error_handler(_: Request, exc: PolyDocError):
        log.warning("domain_error", code=str(exc.code), message=exc.message, details=exc.details)
        return _error_response(exc.status_code, str(exc.code), exc.message, exc.details)

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(_: Request, exc: RequestValidationError):
        return _error_response(
            422,
            str(ErrorCode.VALIDATION_ERROR),
            "Request validation failed.",
            {"errors": exc.errors()[:10]},
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_error_handler(_: Request, exc: StarletteHTTPException):
        # Framework-level HTTP errors (bad route, bad method). Domain 404s come
        # through PolyDocError with the specific code, not this handler.
        code = {
            401: ErrorCode.UNAUTHORIZED,
            403: ErrorCode.FORBIDDEN,
            404: ErrorCode.NOT_FOUND,
            405: ErrorCode.METHOD_NOT_ALLOWED,
            422: ErrorCode.VALIDATION_ERROR,
            429: ErrorCode.RATE_LIMITED,
        }.get(exc.status_code, ErrorCode.INTERNAL)
        return _error_response(exc.status_code, str(code), str(exc.detail))

    @app.exception_handler(Exception)
    async def unhandled_error_handler(_: Request, exc: Exception):
        log.exception("unhandled_error", error_type=type(exc).__name__)
        message = (
            f"{type(exc).__name__}: {exc}"
            if settings.app.debug
            else "An unexpected error occurred."
        )
        return _error_response(500, str(ErrorCode.INTERNAL), message)

    @app.get("/", include_in_schema=False)
    async def root():
        return {
            "name": settings.app.name,
            "version": settings.app.version,
            "docs": "/docs",
            "api": settings.app.api_prefix,
        }

    if settings.obs.metrics_enabled:
        try:
            from prometheus_fastapi_instrumentator import Instrumentator

            Instrumentator(
                should_group_status_codes=False,
                excluded_handlers=["/metrics", "/health"],
            ).instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)
        except ImportError:  # pragma: no cover
            log.warning("metrics_disabled_missing_instrumentator")

    return app


app = create_app()
