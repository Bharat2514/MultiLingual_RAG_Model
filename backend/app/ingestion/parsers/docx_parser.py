"""DOCX parsing (FR-1, FR-5).

DOCX has no intrinsic pagination — pages only exist once a renderer lays the
document out. We therefore synthesise page numbers from *explicit* page breaks,
which is what a reader sees as a page boundary, and fall back to a size-based
split so citations stay usefully granular on long unbroken documents.

Tables are flattened to pipe-delimited rows: best-effort structure per the
non-goals in §4.2, but enough that a clause in a table is still retrievable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.core.errors import ParseFailedError
from app.core.logging import get_logger
from app.ingestion.types import PageText, ParsedDocument, SourceKind

log = get_logger(__name__)

# Characters per synthetic page when a document has no explicit page breaks.
# ~3000 keeps a citation pointing at roughly one printed page of text.
SYNTHETIC_PAGE_CHARS = 3000

_W_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def _has_page_break(paragraph: Any) -> bool:
    """True when the paragraph contains an explicit page break."""
    for br in paragraph._element.iter(f"{_W_NS}br"):
        if br.get(f"{_W_NS}type") == "page":
            return True
    # A "page break before" paragraph property counts too.
    p_pr = paragraph._element.find(f"{_W_NS}pPr")
    return p_pr is not None and p_pr.find(f"{_W_NS}pageBreakBefore") is not None


def _table_to_text(table: Any) -> str:
    rows: list[str] = []
    for row in table.rows:
        cells = [" ".join(c.text.split()) for c in row.cells]
        if any(cells):
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def _iter_block_items(document: Any):
    """Yield paragraphs and tables in true document order.

    python-docx exposes `.paragraphs` and `.tables` separately, which loses
    their interleaving — a table would otherwise land at the end of the
    document, far from the text that introduces it.
    """
    from docx.document import Document as DocxDocument
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    body = document.element.body if isinstance(document, DocxDocument) else document._element
    for child in body.iterchildren():
        if child.tag == f"{_W_NS}p":
            yield Paragraph(child, document)
        elif child.tag == f"{_W_NS}tbl":
            yield Table(child, document)


def parse(path: Path) -> ParsedDocument:
    try:
        import docx
        from docx.table import Table
        from docx.text.paragraph import Paragraph

        document = docx.Document(str(path))
    except Exception as exc:
        raise ParseFailedError(f"Could not open DOCX: {exc}") from exc

    pages: list[PageText] = []
    buffer: list[str] = []
    buffer_chars = 0
    page_no = 1

    def flush() -> None:
        nonlocal buffer, buffer_chars, page_no
        text = "\n".join(buffer).strip()
        if text:
            pages.append(PageText(page_no=page_no, text=text, source=SourceKind.DOCX))
            page_no += 1
        buffer, buffer_chars = [], 0

    for block in _iter_block_items(document):
        if isinstance(block, Paragraph):
            if _has_page_break(block) and buffer:
                flush()
            text = block.text.strip()
            if not text:
                buffer.append("")
                continue
            # Heading styles become their own line so chunk `section` metadata
            # can be derived later.
            buffer.append(text)
            buffer_chars += len(text)
        elif isinstance(block, Table):
            text = _table_to_text(block)
            if text:
                buffer.append(text)
                buffer_chars += len(text)

        if buffer_chars >= SYNTHETIC_PAGE_CHARS:
            flush()

    flush()

    if not pages:
        raise ParseFailedError("DOCX contains no extractable text.")

    core = document.core_properties
    metadata = {
        k: str(v)
        for k, v in {
            "title": core.title,
            "author": core.author,
            "subject": core.subject,
        }.items()
        if v
    }

    log.info("docx_parsed", pages=len(pages), chars=sum(p.char_count for p in pages))
    return ParsedDocument(pages=pages, title=core.title or None, metadata=metadata)
