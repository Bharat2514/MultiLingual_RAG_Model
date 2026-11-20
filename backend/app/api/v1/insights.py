"""Document intelligence endpoints — acting on documents, not just answering.

Everything here is session-scoped exactly like `/documents` and `/chat`: a
document from another session is a 404. Deterministic features (deadlines,
diffs, conflicts, security) need no model at all; LLM-assisted ones (FAQ,
eligibility rules, formulas, comparison, translation) cache their output per
document and verify every quote against the source before returning it.
"""

from __future__ import annotations

import itertools

from fastapi import APIRouter, Depends, Query, Response
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_session_id
from app.core.context import get_request_id
from app.core.errors import (
    DocumentNotFoundError,
    UnsupportedOperationError,
    ValidationError,
)
from app.core.logging import get_logger
from app.db.models import Chunk, Document, DocumentStatus, DocumentTimeline, InsightKind
from app.db.session import get_db
from app.generation.providers.base import LLMProvider
from app.intelligence import (
    calculator,
    corpus,
    deadlines,
    eligibility,
    highlight,
    readers,
    versions,
)
from app.retrieval.service import RetrievalService
from app.retrieval.vectorstore.base import SearchFilter
from app.schemas.insights import (
    CalculateRequest,
    CompareRequest,
    ConflictRequest,
    DiffRequest,
    EligibilityCheckRequest,
    TimelineUpdate,
    TranslateRequest,
)
from app.services import documents as doc_service

router = APIRouter(tags=["insights"])
log = get_logger(__name__)

#: Passages handed to a whole-document LLM reader. A coverage sample beyond this
#: costs tokens without adding rules — scheme conditions cluster in a few sections.
READER_CHUNKS = 24


def get_llm() -> LLMProvider:
    from app.generation.service import get_provider

    return get_provider()


def get_retrieval() -> RetrievalService:
    return RetrievalService()


async def _ready_documents(
    db: AsyncSession, session_id: str, doc_ids: list[str] | None = None
) -> list[Document]:
    query = select(Document).where(
        Document.session_id == session_id, Document.status == DocumentStatus.READY
    )
    if doc_ids is not None:
        query = query.where(Document.doc_id.in_(doc_ids))
    documents = list(await db.scalars(query.order_by(Document.created_at.asc())))
    if doc_ids is not None:
        missing = set(doc_ids) - {d.doc_id for d in documents}
        if missing:
            raise DocumentNotFoundError(details={"doc_ids": sorted(missing)})
    return documents


# --- deadlines & obligations (deterministic) -----------------------------------


async def _timeline_items(
    db: AsyncSession, documents: list[Document]
) -> list[deadlines.TimelineItem]:
    items: list[deadlines.TimelineItem] = []
    for document in documents:
        chunks = await corpus.load_chunks(db, document)
        items += deadlines.extract(corpus.as_dicts(chunks))
    dated = sorted((i for i in items if i.date), key=lambda i: i.date or "")
    return dated + [i for i in items if not i.date]


@router.get("/insights/deadlines", summary="Deadlines, obligations and penalties across documents")
async def session_deadlines(
    doc_ids: list[str] | None = Query(default=None),
    session_id: str = Depends(get_session_id),
    db: AsyncSession = Depends(get_db),
) -> dict:
    documents = await _ready_documents(db, session_id, doc_ids)
    items = await _timeline_items(db, documents)
    counts: dict[str, int] = {}
    for item in items:
        counts[item.kind] = counts.get(item.kind, 0) + 1
    return {
        "request_id": get_request_id(),
        "documents": len(documents),
        "counts": counts,
        "items": [i.to_dict() for i in items],
    }


@router.get(
    "/insights/deadlines.ics",
    summary="Deadlines as an iCalendar file",
    response_class=Response,
    responses={200: {"content": {"text/calendar": {}}}},
)
async def session_deadlines_ics(
    doc_ids: list[str] | None = Query(default=None),
    session_id: str = Depends(get_session_id),
    db: AsyncSession = Depends(get_db),
) -> Response:
    documents = await _ready_documents(db, session_id, doc_ids)
    items = await _timeline_items(db, documents)
    return Response(
        content=deadlines.to_ics(items),
        media_type="text/calendar; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="polydoc-deadlines.ics"'},
    )


# --- security ------------------------------------------------------------------


@router.get("/documents/{doc_id}/security", summary="Planted-instruction and hidden-text scan")
async def document_security(
    doc_id: str,
    session_id: str = Depends(get_session_id),
    db: AsyncSession = Depends(get_db),
) -> dict:
    document = await doc_service.get_document(db, doc_id, session_id)
    insight = await corpus.get_insight(db, document, InsightKind.SECURITY)
    if insight is None:
        # Ingested before the scanner existed: scan the stored text now.
        from app.intelligence import injection

        findings = []
        for chunk in await corpus.load_chunks(db, document):
            findings += injection.scan_text(
                chunk.text, page=chunk.page_start, chunk_id=chunk.chunk_id
            )
        payload = injection.summarize(findings)
        if document.status == DocumentStatus.READY:
            await corpus.save_insight(db, document, InsightKind.SECURITY, payload)
    else:
        payload = insight.payload
    return {"request_id": get_request_id(), "doc_id": doc_id, **payload}


# --- FAQ -------------------------------------------------------------------------


@router.get("/documents/{doc_id}/faq", summary="Questions this document answers, with answers")
async def document_faq(
    doc_id: str,
    language: str | None = Query(default=None, pattern="^(hi|en)$"),
    count: int = Query(default=8, ge=3, le=15),
    refresh: bool = False,
    session_id: str = Depends(get_session_id),
    db: AsyncSession = Depends(get_db),
    llm: LLMProvider = Depends(get_llm),
) -> dict:
    document = await corpus.ready_document(db, doc_id, session_id)
    lang = language or ("hi" if document.language_primary == "hi" else "en")
    cached = None if refresh else await corpus.get_insight(db, document, InsightKind.FAQ, lang)
    if cached is not None:
        return {"request_id": get_request_id(), "doc_id": doc_id, "cached": True, **cached.payload}

    chunks = corpus.sample(await corpus.load_chunks(db, document), READER_CHUNKS)
    payload = await readers.generate_faq(
        llm, chunks, language=lang, count=count, redact_pii=corpus.should_redact(llm)
    )
    await corpus.save_insight(
        db, document, InsightKind.FAQ, payload, variant=lang, model=payload["model"]
    )
    return {"request_id": get_request_id(), "doc_id": doc_id, "cached": False, **payload}


# --- eligibility -----------------------------------------------------------------


async def _rules_for(
    db: AsyncSession, document: Document, llm: LLMProvider, *, refresh: bool
) -> dict:
    cached = None if refresh else await corpus.get_insight(db, document, InsightKind.ELIGIBILITY)
    if cached is not None:
        return cached.payload
    chunks = corpus.sample(await corpus.load_chunks(db, document), READER_CHUNKS)
    payload = await eligibility.extract_rules(llm, chunks, redact_pii=corpus.should_redact(llm))
    await corpus.save_insight(
        db, document, InsightKind.ELIGIBILITY, payload, model=payload.get("model")
    )
    return payload


@router.get(
    "/insights/eligibility/fields", summary="Profile fields the eligibility checker understands"
)
async def eligibility_fields() -> dict:
    return {"fields": eligibility.PROFILE_FIELDS}


@router.post(
    "/documents/{doc_id}/eligibility/rules", summary="Extract a scheme's eligibility rules"
)
async def document_rules(
    doc_id: str,
    refresh: bool = False,
    session_id: str = Depends(get_session_id),
    db: AsyncSession = Depends(get_db),
    llm: LLMProvider = Depends(get_llm),
) -> dict:
    document = await corpus.ready_document(db, doc_id, session_id)
    payload = await _rules_for(db, document, llm, refresh=refresh)
    return {
        "request_id": get_request_id(),
        "doc_id": doc_id,
        "filename": document.filename,
        **payload,
    }


@router.post("/insights/eligibility", summary="Check a profile against every scheme's rules")
async def check_eligibility(
    body: EligibilityCheckRequest,
    session_id: str = Depends(get_session_id),
    db: AsyncSession = Depends(get_db),
    llm: LLMProvider = Depends(get_llm),
) -> dict:
    documents = await _ready_documents(db, session_id, body.doc_ids)
    results = []
    for document in documents:
        if body.extract_missing:
            rules = await _rules_for(db, document, llm, refresh=False)
        else:
            insight = await corpus.get_insight(db, document, InsightKind.ELIGIBILITY)
            if insight is None:
                continue
            rules = insight.payload
        verdict = eligibility.evaluate(rules.get("rules", []), dict(body.profile))
        results.append(
            {
                "doc_id": document.doc_id,
                "filename": document.filename,
                "scheme_name": rules.get("scheme_name"),
                "required_documents": rules.get("required_documents", []),
                "benefits": rules.get("benefits", []),
                **verdict,
            }
        )
    order = {"eligible": 0, "incomplete": 1, "not_eligible": 2, "no_rules": 3}
    results.sort(key=lambda r: order.get(r["verdict"], 9))
    return {
        "request_id": get_request_id(),
        "eligible": sum(1 for r in results if r["verdict"] == "eligible"),
        "checked": len(results),
        "results": results,
    }


# --- calculator --------------------------------------------------------------------


@router.post("/insights/calculate", summary="Compute an amount from a formula in the documents")
async def calculate(
    body: CalculateRequest,
    session_id: str = Depends(get_session_id),
    llm: LLMProvider = Depends(get_llm),
    retrieval: RetrievalService = Depends(get_retrieval),
) -> dict:
    found = await retrieval.retrieve(
        body.question, SearchFilter(session_id=session_id, doc_ids=body.doc_ids), top_n=6
    )
    if found.is_empty:
        return {"request_id": get_request_id(), "status": "not_found"}
    payload = await calculator.calculate(
        llm, body.question, found.chunks, body.inputs, redact_pii=corpus.should_redact(llm)
    )
    return {"request_id": get_request_id(), **payload}


# --- comparison ----------------------------------------------------------------------


@router.post("/insights/compare", summary="Comparison table across documents, every cell cited")
async def compare(
    body: CompareRequest,
    session_id: str = Depends(get_session_id),
    db: AsyncSession = Depends(get_db),
    llm: LLMProvider = Depends(get_llm),
    retrieval: RetrievalService = Depends(get_retrieval),
) -> dict:
    documents = await _ready_documents(db, session_id, body.doc_ids)
    attributes = [a.strip() for a in body.attributes if a.strip()]
    if not attributes:
        raise ValidationError("At least one non-empty attribute is required.")
    rows = []
    for document in documents:
        found = await retrieval.retrieve(
            "; ".join(attributes),
            SearchFilter(session_id=session_id, doc_ids=[document.doc_id]),
            top_n=8,
        )
        cells = await readers.compare_row(
            llm, document.filename, found.chunks, attributes, redact_pii=corpus.should_redact(llm)
        )
        rows.append({"doc_id": document.doc_id, "filename": document.filename, "cells": cells})
    return {"request_id": get_request_id(), "attributes": attributes, "rows": rows}


# --- versions & conflicts (deterministic) --------------------------------------------


@router.post("/insights/diff", summary="What changed between two versions of a document")
async def diff(
    body: DiffRequest,
    session_id: str = Depends(get_session_id),
    db: AsyncSession = Depends(get_db),
) -> dict:
    if body.old_doc_id == body.new_doc_id:
        raise ValidationError("Choose two different documents to compare.")
    old = await corpus.ready_document(db, body.old_doc_id, session_id)
    new = await corpus.ready_document(db, body.new_doc_id, session_id)
    result = versions.diff_documents(
        corpus.as_dicts(await corpus.load_chunks(db, old)),
        corpus.as_dicts(await corpus.load_chunks(db, new)),
    )
    return {
        "request_id": get_request_id(),
        "old": {"doc_id": old.doc_id, "filename": old.filename},
        "new": {"doc_id": new.doc_id, "filename": new.filename},
        **result,
    }


@router.post("/insights/conflicts", summary="Documents that disagree on a number")
async def conflicts(
    body: ConflictRequest,
    session_id: str = Depends(get_session_id),
    db: AsyncSession = Depends(get_db),
) -> dict:
    documents = await _ready_documents(db, session_id, body.doc_ids)
    texts = {d.doc_id: corpus.as_dicts(await corpus.load_chunks(db, d)) for d in documents}
    pairs = []
    for a, b in itertools.combinations(documents, 2):
        found = versions.find_conflicts(texts[a.doc_id], texts[b.doc_id])
        if found:
            pairs.append(
                {
                    "a": {"doc_id": a.doc_id, "filename": a.filename},
                    "b": {"doc_id": b.doc_id, "filename": b.filename},
                    "conflicts": found,
                }
            )
    return {
        "request_id": get_request_id(),
        "documents": len(documents),
        "total": sum(len(p["conflicts"]) for p in pairs),
        "pairs": pairs,
    }


# --- timeline (supersession / effective dates) -----------------------------------------


def _timeline_dict(doc_id: str, row: DocumentTimeline | None) -> dict:
    return {
        "doc_id": doc_id,
        "effective_date": row.effective_date.isoformat() if row and row.effective_date else None,
        "superseded_by": row.superseded_by if row else None,
        "note": row.note if row else None,
    }


@router.get("/documents/{doc_id}/timeline", summary="When a document is in force")
async def get_timeline(
    doc_id: str,
    session_id: str = Depends(get_session_id),
    db: AsyncSession = Depends(get_db),
) -> dict:
    document = await doc_service.get_document(db, doc_id, session_id)
    row = await db.get(DocumentTimeline, document.doc_id)
    suggestion = None
    if row is None or row.effective_date is None:
        # Offer the document's own "with effect from" date as a starting point.
        chunks = await corpus.load_chunks(db, document)
        effective = [i for i in deadlines.extract(corpus.as_dicts(chunks)) if i.kind == "effective"]
        if effective:
            suggestion = {
                "effective_date": effective[0].date,
                "text": effective[0].text,
                "page": effective[0].page,
            }
    return {"request_id": get_request_id(), **_timeline_dict(doc_id, row), "suggestion": suggestion}


@router.put("/documents/{doc_id}/timeline", summary="Set effective date / superseding document")
async def put_timeline(
    doc_id: str,
    body: TimelineUpdate,
    session_id: str = Depends(get_session_id),
    db: AsyncSession = Depends(get_db),
) -> dict:
    document = await doc_service.get_document(db, doc_id, session_id)
    if body.superseded_by:
        if body.superseded_by == doc_id:
            raise ValidationError("A document cannot supersede itself.")
        await doc_service.get_document(db, body.superseded_by, session_id)

    row = await db.get(DocumentTimeline, document.doc_id)
    if row is None:
        row = DocumentTimeline(doc_id=document.doc_id, session_id=session_id)
        db.add(row)
    row.effective_date = body.effective_date
    row.superseded_by = body.superseded_by
    row.note = body.note
    await db.flush()
    return {"request_id": get_request_id(), **_timeline_dict(doc_id, row)}


# --- bilingual reader ---------------------------------------------------------------------


@router.post(
    "/documents/{doc_id}/translate",
    summary="Side-by-side translation, a page of passages at a time",
)
async def translate(
    doc_id: str,
    body: TranslateRequest,
    session_id: str = Depends(get_session_id),
    db: AsyncSession = Depends(get_db),
    llm: LLMProvider = Depends(get_llm),
) -> dict:
    document = await corpus.ready_document(db, doc_id, session_id)
    chunks = await corpus.load_chunks(db, document)
    window = chunks[body.offset : body.offset + body.limit]

    insight = await corpus.get_insight(db, document, InsightKind.TRANSLATION, body.target)
    stored: dict[str, str] = dict(insight.payload.get("chunks", {})) if insight else {}
    missing = [c for c in window if c.chunk_id not in stored]
    if missing:
        stored.update(
            await readers.translate_chunks(
                llm, missing, target=body.target, redact_pii=corpus.should_redact(llm)
            )
        )
        await corpus.save_insight(
            db,
            document,
            InsightKind.TRANSLATION,
            {"chunks": stored},
            variant=body.target,
            model=llm.model,
        )

    next_offset = body.offset + body.limit
    return {
        "request_id": get_request_id(),
        "doc_id": doc_id,
        "target": body.target,
        "total": len(chunks),
        "next_offset": next_offset if next_offset < len(chunks) else None,
        "items": [
            {
                "chunk_id": c.chunk_id,
                "page": c.page_start,
                "original": c.text,
                "translation": stored.get(c.chunk_id),
            }
            for c in window
        ],
    }


# --- source viewer -------------------------------------------------------------------------


async def _pdf_path(db: AsyncSession, doc_id: str, session_id: str):
    from app.ingestion.storage import get_object_store

    document = await doc_service.get_document(db, doc_id, session_id)
    if document.mime_type != "application/pdf":
        raise UnsupportedOperationError(
            "Page view is available for PDFs; DOCX passages are shown as text.",
            details={"mime_type": document.mime_type},
        )
    return document, get_object_store().local_path(document.storage_key)


@router.get(
    "/documents/{doc_id}/pages/{page}/image",
    summary="A rendered PDF page",
    response_class=Response,
    responses={200: {"content": {"image/png": {}}}},
)
async def page_image(
    doc_id: str,
    page: int,
    dpi: int = Query(default=110, ge=50, le=200),
    session_id: str = Depends(get_session_id),
    db: AsyncSession = Depends(get_db),
) -> Response:
    document, path = await _pdf_path(db, doc_id, session_id)
    if not 1 <= page <= (document.pages or highlight.page_count(path)):
        raise ValidationError("Page out of range.", details={"pages": document.pages})
    return Response(
        content=highlight.render_page(path, page, dpi=dpi),
        media_type="image/png",
        headers={"Cache-Control": "private, max-age=3600"},
    )


@router.get("/documents/{doc_id}/pages/{page}/highlights", summary="Where a passage sits on a page")
async def page_highlights(
    doc_id: str,
    page: int,
    chunk_id: str,
    session_id: str = Depends(get_session_id),
    db: AsyncSession = Depends(get_db),
) -> dict:
    document, path = await _pdf_path(db, doc_id, session_id)
    chunk = await db.scalar(
        select(Chunk).where(
            Chunk.chunk_id == chunk_id, Chunk.doc_id == doc_id, Chunk.session_id == session_id
        )
    )
    if chunk is None:
        raise DocumentNotFoundError("Passage not found.", details={"chunk_id": chunk_id})
    located = highlight.locate(path, page, chunk.text, ocr_page=chunk.ocr_used)
    return {
        "request_id": get_request_id(),
        "doc_id": doc_id,
        "page": page,
        "pages": document.pages,
        "chunk_id": chunk_id,
        **located,
    }


@router.get("/chunks/{chunk_id}", summary="Full text of one passage")
async def get_chunk(
    chunk_id: str,
    session_id: str = Depends(get_session_id),
    db: AsyncSession = Depends(get_db),
) -> dict:
    row = await db.scalar(
        select(Chunk).where(Chunk.chunk_id == chunk_id, Chunk.session_id == session_id)
    )
    if row is None:
        raise DocumentNotFoundError("Passage not found.", details={"chunk_id": chunk_id})
    return {
        "chunk_id": row.chunk_id,
        "doc_id": row.doc_id,
        "page_start": row.page_start,
        "page_end": row.page_end,
        "text": row.text,
        "ocr_used": row.ocr_used,
        "language": row.language,
    }
