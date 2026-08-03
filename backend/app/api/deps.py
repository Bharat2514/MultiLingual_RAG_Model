"""Shared FastAPI dependencies."""

from __future__ import annotations

import uuid

from fastapi import Header, Request

SESSION_ID_HEADER = "X-Session-ID"


def get_session_id(
    request: Request,
    x_session_id: str | None = Header(default=None, alias=SESSION_ID_HEADER),
) -> str:
    """Resolve the caller's document scope (FR-19).

    v1 uses a client-supplied session key, per §9. Every document query filters
    on this value, so it is the isolation boundary — when auth lands, the
    user id is layered on top without changing call sites.
    """
    session_id = (x_session_id or "").strip()
    if not session_id:
        session_id = f"sess_{uuid.uuid4().hex[:16]}"
    request.state.session_id = session_id
    return session_id
