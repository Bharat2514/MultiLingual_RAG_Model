"""Ingestion pipeline: parse → OCR → clean → chunk (§8.1).

Pure orchestration over a file path — no database, no queue — so the whole
pipeline is testable end to end without infrastructure. The worker
(`app.workers.tasks`) wraps this and persists the result.

Stage progress is reported through a callback so the worker can write
`ingestion_jobs` rows and the UI can show a real pipeline rather than a
spinner (FR-26).
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path

from app.core.config import settings
from app.core.errors import ParseFailedError
from app.core.logging import get_logger
from app.db.models import JobStage, Language
from app.ingestion import ocr as ocr_module
from app.ingestion.chunking import chunk_pages
from app.ingestion.cleaning import clean_document
from app.ingestion.parsers import docx_parser, pdf
from app.ingestion.types import IngestionResult, PageText, ParsedDocument, SourceKind
from app.ingestion.validation import FileKind
from app.nlp.language import dominant_language

log = get_logger(__name__)

StageCallback = Callable[[JobStage, str, dict], None]


def _noop(stage: JobStage, status: str, meta: dict) -> None:  # pragma: no cover
    return None


def _apply_ocr(path: Path, parsed: ParsedDocument, on_stage: StageCallback) -> ParsedDocument:
    """Backfill sparse pages with OCR (§11.1)."""
    targets = pdf.pages_needing_ocr(parsed)
    if not targets:
        on_stage(JobStage.OCR, "skipped", {"reason": "text layer sufficient on every page"})
        return parsed

    if settings.ocr.engine == "none" or not ocr_module.is_available():
        # Degrade rather than fail: a document with *some* text layer is still
        # partially useful, and the flag lets the UI explain the gap (NFR-5).
        log.warning("ocr_unavailable_skipping_pages", pages=len(targets))
        on_stage(
            JobStage.OCR,
            "skipped",
            {"reason": "tesseract unavailable", "pages_affected": len(targets)},
        )
        return parsed

    by_page = {p.page_no: p for p in parsed.pages}
    recovered = 0
    low_confidence: list[int] = []

    for page_no in targets:
        try:
            image = pdf.render_page_png(path, page_no)
            text, confidence = ocr_module.ocr_image(image)
        except Exception as exc:  # noqa: BLE001 - one bad page must not fail the doc
            log.warning("ocr_page_failed", page=page_no, error=str(exc))
            continue

        existing = by_page[page_no]
        # Keep OCR only if it actually beat the text layer.
        if len(text.strip()) > existing.char_count:
            by_page[page_no] = PageText(
                page_no=page_no,
                text=text,
                source=SourceKind.OCR,
                ocr_confidence=confidence,
                width=existing.width,
                height=existing.height,
            )
            recovered += 1
            if confidence < ocr_module.LOW_CONFIDENCE_THRESHOLD:
                low_confidence.append(page_no)

    parsed.pages = [by_page[p.page_no] for p in parsed.pages]

    log.info(
        "ocr_complete",
        attempted=len(targets),
        recovered=recovered,
        low_confidence_pages=len(low_confidence),
    )
    on_stage(
        JobStage.OCR,
        "succeeded",
        {
            "pages_attempted": len(targets),
            "pages_recovered": recovered,
            "low_confidence_pages": low_confidence[:20],
        },
    )
    return parsed


def run_pipeline(
    path: Path,
    kind: FileKind,
    *,
    on_stage: StageCallback = _noop,
) -> IngestionResult:
    """Parse, OCR, clean, and chunk a document."""
    timings: dict[str, float] = {}

    def timed(stage: JobStage, fn):
        started = time.perf_counter()
        on_stage(stage, "running", {})
        try:
            result = fn()
        except Exception as exc:
            timings[str(stage)] = round((time.perf_counter() - started) * 1000, 2)
            on_stage(stage, "failed", {"error": str(exc)})
            raise
        timings[str(stage)] = round((time.perf_counter() - started) * 1000, 2)
        return result

    # --- parse ---------------------------------------------------------------
    def _parse() -> ParsedDocument:
        parsed = pdf.parse(path) if kind == FileKind.PDF else docx_parser.parse(path)
        on_stage(
            JobStage.PARSE,
            "succeeded",
            {"pages": parsed.page_count, "chars": parsed.total_chars},
        )
        return parsed

    parsed = timed(JobStage.PARSE, _parse)

    # --- ocr (PDF only) ------------------------------------------------------
    if kind == FileKind.PDF:
        parsed = timed(JobStage.OCR, lambda: _apply_ocr(path, parsed, on_stage))

    if parsed.is_empty:
        raise ParseFailedError(
            "No text could be extracted from this document. If it is a scan, "
            "try re-uploading at a higher resolution.",
            details={"pages": parsed.page_count, "ocr_used": parsed.ocr_page_count > 0},
        )

    # --- clean ---------------------------------------------------------------
    def _clean() -> list[PageText]:
        cleaned = clean_document(parsed.pages)
        on_stage(
            JobStage.CLEAN,
            "succeeded",
            {"chars_before": parsed.total_chars, "chars_after": sum(p.char_count for p in cleaned)},
        )
        return cleaned

    cleaned = timed(JobStage.CLEAN, _clean)

    # --- chunk ---------------------------------------------------------------
    def _chunk():
        chunks = chunk_pages(cleaned)
        on_stage(
            JobStage.CHUNK,
            "succeeded",
            {
                "chunks": len(chunks),
                "avg_tokens": round(sum(c.token_count for c in chunks) / len(chunks), 1)
                if chunks
                else 0,
            },
        )
        return chunks

    chunks = timed(JobStage.CHUNK, _chunk)

    if not chunks:
        raise ParseFailedError("Document produced no usable text chunks.")

    distribution = Counter(c.language for c in chunks)
    primary = dominant_language(dict(distribution))

    result = IngestionResult(
        chunks=chunks,
        page_count=parsed.page_count,
        language_primary=str(primary),
        language_distribution=dict(distribution),
        ocr_used=parsed.ocr_page_count > 0,
        ocr_page_ratio=round(parsed.ocr_ratio, 4),
        title=parsed.title,
        stage_timings_ms=timings,
    )

    log.info(
        "pipeline_complete",
        pages=result.page_count,
        chunks=len(result.chunks),
        language=result.language_primary,
        ocr_ratio=result.ocr_page_ratio,
        timings_ms=timings,
    )
    return result


def language_counts_to_str(counts: dict[str, int]) -> dict[str, int]:
    """Normalise Language enum keys to plain strings for JSON storage."""
    return {str(Language(k)) if k in set(Language) else str(k): v for k, v in counts.items()}
