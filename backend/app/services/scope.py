"""Which documents a question may be answered from — now, or as of a date.

Retrieval already scopes by session. This adds *time*: a circular that has been
superseded should not answer today's question, and "what was the rule in
2023?" should be answered only from documents in force in 2023.

Rules, given the session's timeline rows:

* a document whose `effective_date` is after `as_of` is out of scope;
* a document superseded by another is out of scope once its replacement is in
  force (always, when no `as_of` is given; or when the replacement's effective
  date is on or before `as_of`);
* documents with no timeline row are always in scope — the feature is opt-in
  per document and never hides anything the user has not dated.

The result is expressed as an explicit `doc_ids` list on the search filter only
when something is excluded, so the common case adds nothing to the query.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import date

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Document, DocumentStatus, DocumentTimeline


@dataclass(slots=True)
class Scope:
    doc_ids: list[str] | None
    ready_count: int
    fingerprint: str
    excluded: list[dict] = field(default_factory=list)


def _in_force(row: DocumentTimeline | None, as_of: date | None) -> bool:
    if row is None or row.effective_date is None or as_of is None:
        return True
    return row.effective_date <= as_of


async def resolve_scope(
    db: AsyncSession,
    session_id: str,
    requested: list[str] | None,
    *,
    as_of: date | None = None,
    include_superseded: bool = False,
) -> Scope:
    documents = list(
        await db.scalars(
            select(Document).where(
                Document.session_id == session_id, Document.status == DocumentStatus.READY
            )
        )
    )
    timeline = {
        row.doc_id: row
        for row in await db.scalars(
            select(DocumentTimeline).where(DocumentTimeline.session_id == session_id)
        )
    }

    candidates = [d for d in documents if requested is None or d.doc_id in set(requested)]
    excluded: list[dict] = []
    allowed: list[str] = []
    for document in candidates:
        row = timeline.get(document.doc_id)
        if not _in_force(row, as_of):
            excluded.append(
                {
                    "doc_id": document.doc_id,
                    "filename": document.filename,
                    "reason": "not_yet_effective",
                }
            )
            continue
        if row is not None and row.superseded_by and not include_superseded:
            replacement = timeline.get(row.superseded_by)
            replacement_exists = any(d.doc_id == row.superseded_by for d in documents)
            if replacement_exists and (as_of is None or _in_force(replacement, as_of)):
                excluded.append(
                    {
                        "doc_id": document.doc_id,
                        "filename": document.filename,
                        "reason": "superseded",
                        "superseded_by": row.superseded_by,
                    }
                )
                continue
        allowed.append(document.doc_id)

    fingerprint = hashlib.sha256(
        "|".join(
            sorted(
                f"{d.doc_id}:{d.indexed_at.isoformat() if d.indexed_at else ''}" for d in documents
            )
        ).encode()
    ).hexdigest()

    # An explicit list only when something was excluded — otherwise the
    # caller's own scope (possibly "everything") stands unchanged.
    doc_ids = allowed if excluded else requested
    return Scope(
        doc_ids=doc_ids, ready_count=len(allowed), fingerprint=fingerprint, excluded=excluded
    )
