"""Per-request ambient context: request id, session id, stage timings.

The chat and ingestion pipelines report per-stage latency (NFR-2, NFR-7). Rather
than threading a timing dict through every call, stages record into a
context-local ``StageTimer`` that the API layer reads back when building the
response.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_request_id: ContextVar[str | None] = ContextVar("polydoc_request_id", default=None)
_session_id: ContextVar[str | None] = ContextVar("polydoc_session_id", default=None)
_timings: ContextVar[dict[str, float] | None] = ContextVar("polydoc_timings", default=None)


def new_request_id() -> str:
    return f"req_{uuid.uuid4().hex[:16]}"


def set_request_id(value: str) -> None:
    _request_id.set(value)


def get_request_id() -> str:
    value = _request_id.get()
    if value is None:
        value = new_request_id()
        _request_id.set(value)
    return value


def set_session_id(value: str | None) -> None:
    _session_id.set(value)


def get_session_id() -> str | None:
    return _session_id.get()


def reset_timings() -> None:
    _timings.set({})


def record_stage(stage: str, milliseconds: float) -> None:
    current = _timings.get()
    if current is None:
        current = {}
        _timings.set(current)
    current[stage] = round(current.get(stage, 0.0) + milliseconds, 2)


def get_timings() -> dict[str, float]:
    return dict(_timings.get() or {})


@contextmanager
def stage(name: str) -> Iterator[None]:
    """Time a pipeline stage and accumulate it under ``name``.

    >>> with stage("retrieval"):
    ...     candidates = retrieve(query)
    """
    started = time.perf_counter()
    try:
        yield
    finally:
        record_stage(name, (time.perf_counter() - started) * 1000.0)
