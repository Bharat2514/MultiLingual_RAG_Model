"""Shareable, verifiable answers (see `app.services.answers`)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Header
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import SESSION_ID_HEADER, get_session_id
from app.core.context import get_request_id
from app.core.errors import AnswerNotFoundError
from app.db.session import get_db
from app.services.answers import get_snapshot, verify_evidence

router = APIRouter(prefix="/answers", tags=["answers"])


@router.get("/{answer_id}", summary="A saved answer with a live check of its evidence")
async def read_answer(
    answer_id: str,
    db: AsyncSession = Depends(get_db),
    x_session_id: str | None = Header(default=None, alias=SESSION_ID_HEADER),
) -> dict:
    """Readable by its owner, or by anyone once shared — the id is the capability."""
    snapshot = await get_snapshot(db, answer_id, (x_session_id or "").strip() or None)
    return {
        "request_id": get_request_id(),
        "answer_id": snapshot.answer_id,
        "query": snapshot.query,
        "answer": snapshot.answer,
        "language": snapshot.language,
        "grounded": snapshot.grounded,
        "citations": [{k: v for k, v in c.items() if k != "text_hash"} for c in snapshot.citations],
        "claims": snapshot.claims or [],
        "model": snapshot.model,
        "created_at": snapshot.created_at.isoformat() if snapshot.created_at else None,
        "shared": snapshot.shared,
        "evidence": await verify_evidence(db, snapshot),
    }


async def _owned(db: AsyncSession, answer_id: str, session_id: str):
    snapshot = await get_snapshot(db, answer_id, session_id)
    if snapshot.session_id != session_id:
        raise AnswerNotFoundError()
    return snapshot


@router.post("/{answer_id}/share", summary="Make an answer readable by link")
async def share_answer(
    answer_id: str,
    session_id: str = Depends(get_session_id),
    db: AsyncSession = Depends(get_db),
) -> dict:
    snapshot = await _owned(db, answer_id, session_id)
    snapshot.shared = True
    await db.flush()
    return {"request_id": get_request_id(), "answer_id": answer_id, "shared": True}


@router.delete("/{answer_id}/share", summary="Revoke an answer's share link")
async def unshare_answer(
    answer_id: str,
    session_id: str = Depends(get_session_id),
    db: AsyncSession = Depends(get_db),
) -> dict:
    snapshot = await _owned(db, answer_id, session_id)
    snapshot.shared = False
    await db.flush()
    return {"request_id": get_request_id(), "answer_id": answer_id, "shared": False}
