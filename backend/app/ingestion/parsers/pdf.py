"""PDF parsing with text-layer extraction and OCR routing (FR-3, FR-5, §11.1).

The routing decision is per *page*, not per document: real Hindi circulars are
routinely a scanned cover sheet followed by born-digital annexures, so an
all-or-nothing choice either wastes minutes of OCR on clean text or drops the
scanned pages entirely.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from app.core.config import settings
from app.core.errors import ParseFailedError, TooManyPagesError
from app.core.logging import get_logger
from app.ingestion.types import PageText, ParsedDocument, SourceKind

if TYPE_CHECKING:  # pragma: no cover
    import fitz

log = get_logger(__name__)


def _open(path: Path) -> fitz.Document:
    import fitz

    try:
        return fitz.open(path)
    except Exception as exc:
        raise ParseFailedError(f"Could not open PDF: {exc}") from exc


def page_count(path: Path) -> int:
    doc = _open(path)
    try:
        return doc.page_count
    finally:
        doc.close()


def is_encrypted(path: Path) -> bool:
    doc = _open(path)
    try:
        return bool(doc.needs_pass)
    finally:
        doc.close()


def extract_text_layer(path: Path) -> ParsedDocument:
    """Extract the native text layer without touching OCR."""
    doc = _open(path)
    try:
        if doc.needs_pass:
            raise ParseFailedError("PDF is password-protected.")

        if doc.page_count > settings.limits.max_pages:
            raise TooManyPagesError(
                f"Document has {doc.page_count} pages; the limit is {settings.limits.max_pages}.",
                details={"pages": doc.page_count, "limit": settings.limits.max_pages},
            )

        pages: list[PageText] = []
        for index in range(doc.page_count):
            page = doc.load_page(index)
            # sort=True reconstructs reading order for multi-column layouts,
            # which Hindi government forms use heavily.
            text = page.get_text("text", sort=True) or ""
            rect = page.rect
            pages.append(
                PageText(
                    page_no=index + 1,
                    text=text,
                    source=SourceKind.TEXT_LAYER,
                    width=rect.width,
                    height=rect.height,
                )
            )

        metadata = {k: str(v) for k, v in (doc.metadata or {}).items() if v}
        title = metadata.get("title") or None
        return ParsedDocument(pages=pages, title=title, metadata=metadata)
    finally:
        doc.close()


def pages_needing_ocr(parsed: ParsedDocument, *, min_chars: int | None = None) -> list[int]:
    """Page numbers whose text layer is too sparse to trust (§11.1).

    A scanned page yields a handful of stray characters, not zero, so the test
    is a density threshold rather than emptiness.
    """
    threshold = settings.ocr.min_chars_per_page if min_chars is None else min_chars
    return [p.page_no for p in parsed.pages if p.char_count < threshold]


def render_page_png(path: Path, page_no: int, *, dpi: int | None = None) -> bytes:
    """Rasterise one page for OCR. `page_no` is 1-indexed."""
    import fitz

    resolution = dpi or settings.ocr.dpi
    doc = _open(path)
    try:
        page = doc.load_page(page_no - 1)
        pixmap = page.get_pixmap(matrix=fitz.Matrix(resolution / 72, resolution / 72))
        return pixmap.tobytes("png")
    finally:
        doc.close()


def parse(path: Path) -> ParsedDocument:
    """Text-layer parse. OCR backfill is applied by the pipeline."""
    parsed = extract_text_layer(path)
    log.info(
        "pdf_text_layer_extracted",
        pages=parsed.page_count,
        chars=parsed.total_chars,
        sparse_pages=len(pages_needing_ocr(parsed)),
    )
    return parsed
