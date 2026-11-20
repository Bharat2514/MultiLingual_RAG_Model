"""Answer feedback, and its export as draft gold-set questions.

A thumbs-down is the cheapest signal there is that the system failed on a real
question. The export turns those questions into draft entries for
`eval/gold/` — draft, because a gold label needs a human-chosen anchor (see
`eval/README.md`); the export supplies the question, what was answered, and the
passages that were cited, which is everything a reviewer needs to write one.
"""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, Query, Response
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_session_id
from app.core.context import get_request_id
from app.core.errors import NotFoundError
from app.db.models import AnswerSnapshot, Feedback, QueryLog
from app.db.session import get_db
from app.schemas.insights import FeedbackRequest

router = APIRouter(tags=["feedback"])


@router.post("/feedback", summary="Rate an answer")
async def submit_feedback(
    body: FeedbackRequest,
    session_id: str = Depends(get_session_id),
    db: AsyncSession = Depends(get_db),
) -> dict:
    trace = await db.scalar(
        select(QueryLog).where(
            QueryLog.request_id == body.request_id, QueryLog.session_id == session_id
        )
    )
    if trace is None:
        raise NotFoundError("No answer with this request id in this session.")

    existing = await db.scalar(
        select(Feedback).where(
            Feedback.request_id == body.request_id, Feedback.session_id == session_id
        )
    )
    if existing is None:
        existing = Feedback(request_id=body.request_id, session_id=session_id, rating=body.rating)
        db.add(existing)
    existing.rating = body.rating
    existing.comment = body.comment
    await db.flush()
    return {
        "request_id": get_request_id(),
        "feedback_id": existing.feedback_id,
        "rating": body.rating,
    }


@router.get(
    "/feedback/export",
    summary="Rated answers as draft gold-set entries (JSONL)",
    response_class=Response,
    responses={200: {"content": {"application/x-ndjson": {}}}},
)
async def export_feedback(
    rating: int | None = Query(default=-1, description="-1, 1, or omit for both."),
    session_id: str = Depends(get_session_id),
    db: AsyncSession = Depends(get_db),
) -> Response:
    query = (
        select(Feedback, QueryLog, AnswerSnapshot)
        .join(QueryLog, QueryLog.request_id == Feedback.request_id)
        .outerjoin(AnswerSnapshot, AnswerSnapshot.request_id == Feedback.request_id)
        .where(Feedback.session_id == session_id)
        .order_by(Feedback.created_at.asc())
    )
    if rating in (-1, 1):
        query = query.where(Feedback.rating == rating)

    lines = []
    for feedback, trace, snapshot in (await db.execute(query)).all():
        lang = trace.language_detected
        lines.append(
            json.dumps(
                {
                    "id": f"fb-{feedback.request_id[-10:]}",
                    "query": trace.query,
                    "lang": lang if lang in ("hi", "en", "hinglish") else "en",
                    "kind": "feedback",
                    "rating": feedback.rating,
                    "comment": feedback.comment,
                    "answer_given": snapshot.answer if snapshot else None,
                    "grounded": trace.grounded,
                    "cited": [
                        {"filename": c["filename"], "page": c["page"], "snippet": c["snippet"]}
                        for c in (snapshot.citations if snapshot else [])
                    ],
                    # A reviewer fills these in from the corpus — see eval/README.md.
                    "doc": None,
                    "anchors": [],
                    "answer": None,
                    "expect_language": lang if lang in ("hi", "en", "hinglish") else "en",
                },
                ensure_ascii=False,
            )
        )
    return Response(
        content="\n".join(lines) + ("\n" if lines else ""),
        media_type="application/x-ndjson",
        headers={"Content-Disposition": 'attachment; filename="feedback-gold-drafts.jsonl"'},
    )
