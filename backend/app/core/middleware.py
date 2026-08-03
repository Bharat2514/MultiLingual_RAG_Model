"""Request-scoped middleware: correlation IDs, access logging, timing headers."""

from __future__ import annotations

import time

import structlog
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from app.core.context import (
    get_request_id,
    new_request_id,
    reset_timings,
    set_request_id,
    set_session_id,
)
from app.core.logging import get_logger

log = get_logger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"
SESSION_ID_HEADER = "X-Session-ID"
RESPONSE_TIME_HEADER = "X-Response-Time-ms"


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Establishes the per-request id, session id, and stage-timing store.

    An inbound ``X-Request-ID`` is honoured so a request can be traced across
    the frontend, API, and worker; otherwise we mint one (FR-32).
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = request.headers.get(REQUEST_ID_HEADER) or new_request_id()
        session_id = request.headers.get(SESSION_ID_HEADER)

        set_request_id(request_id)
        set_session_id(session_id)
        reset_timings()

        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id, session_id=session_id)

        request.state.request_id = request_id
        request.state.session_id = session_id

        started = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            request.state.elapsed_ms = elapsed_ms

        response.headers[REQUEST_ID_HEADER] = get_request_id()
        response.headers[RESPONSE_TIME_HEADER] = f"{elapsed_ms:.2f}"
        return response


class AccessLogMiddleware(BaseHTTPMiddleware):
    """One structured line per request. Health checks are logged at debug."""

    QUIET_PATHS = {"/health", "/api/v1/health", "/metrics", "/favicon.ico"}

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            log.exception(
                "http_request_failed",
                method=request.method,
                path=request.url.path,
                duration_ms=round(elapsed_ms, 2),
            )
            raise

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        emit = log.debug if request.url.path in self.QUIET_PATHS else log.info
        emit(
            "http_request",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=round(elapsed_ms, 2),
            client=request.client.host if request.client else None,
        )
        return response
