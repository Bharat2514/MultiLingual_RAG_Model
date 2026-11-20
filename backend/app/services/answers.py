"""Answer snapshots: shareable answers that can prove their evidence is unchanged.

Each answer is stored with a SHA-256 of every passage it cited. Opening a
shared answer later re-reads those passages and compares hashes, so the reader
sees one of:

* **intact** — every cited passage still says exactly what it said;
* **changed** — a passage was re-ingested with different text;
* **deleted** — the source document is gone.

That turns "the assistant told me X" into something auditable: the link carries
the answer, the quotes, and a live check that the quotes still stand.

Snapshots are private by default. Sharing flips `shared`, and the unguessable
`answer_id` is the capability — exactly how unlisted links work elsewhere.
"""

from __future__ import annotations

import hashlib

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.errors import AnswerNotFoundError
from app.db.models import AnswerSnapshot, Chunk
from app.generation.service import ChatResult


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def evidence_hash(answer: str, citations: list[dict]) -> str:
    material = (
        answer + "\n" + "\n".join(sorted(f"{c['chunk_id']}:{c['text_hash']}" for c in citations))
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def build_snapshot(request_id: str, session_id: str, result: ChatResult) -> AnswerSnapshot:
    chunks = {c.chunk_id: c for c in (result.retrieval.chunks if result.retrieval else [])}
    citations = []
    for citation in result.citations:
        chunk = chunks.get(citation.chunk_id)
        citations.append(
            {
                "chunk_id": citation.chunk_id,
                "doc_id": citation.doc_id,
                "filename": citation.filename,
                "page": citation.page,
                "snippet": citation.snippet,
                "context_index": citation.context_index,
                "text_hash": text_hash(chunk.text) if chunk else "",
            }
        )
    return AnswerSnapshot(
        request_id=request_id,
        session_id=session_id,
        query=result.query.raw if result.query else "",
        answer=result.answer,
        language=str(result.language_detected),
        grounded=result.grounded,
        citations=citations,
        claims=[c.to_dict() for c in result.claims.claims] if result.claims else None,
        evidence_hash=evidence_hash(result.answer, citations),
        model=result.model,
    )


async def get_snapshot(db: AsyncSession, answer_id: str, session_id: str | None) -> AnswerSnapshot:
    """Owner access by session, or public access to a shared snapshot."""
    snapshot = await db.get(AnswerSnapshot, answer_id)
    if snapshot is None:
        raise AnswerNotFoundError()
    if snapshot.session_id != session_id and not snapshot.shared:
        raise AnswerNotFoundError()
    return snapshot


async def verify_evidence(db: AsyncSession, snapshot: AnswerSnapshot) -> dict:
    ids = [c["chunk_id"] for c in snapshot.citations]
    current = {
        row.chunk_id: row.text_hash
        for row in await db.scalars(
            select(Chunk).where(Chunk.chunk_id.in_(ids), Chunk.session_id == snapshot.session_id)
        )
    }
    statuses = []
    for citation in snapshot.citations:
        now = current.get(citation["chunk_id"])
        status = (
            "deleted" if now is None else "intact" if now == citation["text_hash"] else "changed"
        )
        statuses.append({"chunk_id": citation["chunk_id"], "status": status})

    recomputed = evidence_hash(snapshot.answer, snapshot.citations)
    overall = (
        "intact"
        if statuses and all(s["status"] == "intact" for s in statuses)
        else "no_citations"
        if not statuses
        else "changed"
        if all(s["status"] != "deleted" for s in statuses)
        else "deleted"
    )
    return {
        "status": overall,
        "citations": statuses,
        "evidence_hash": snapshot.evidence_hash,
        "snapshot_untampered": recomputed == snapshot.evidence_hash,
    }
