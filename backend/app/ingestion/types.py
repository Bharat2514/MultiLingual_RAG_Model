"""Intermediate representations passed between ingestion stages.

Everything is a plain dataclass so stages stay pure and unit-testable without a
database, a file, or a model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class SourceKind(StrEnum):
    TEXT_LAYER = "text_layer"
    OCR = "ocr"
    DOCX = "docx"


@dataclass(slots=True)
class PageText:
    """Text of a single page, with the provenance needed for citations."""

    page_no: int  # 1-indexed, matches what the user sees in a viewer
    text: str
    source: SourceKind = SourceKind.TEXT_LAYER
    ocr_confidence: float | None = None
    width: float | None = None
    height: float | None = None

    @property
    def char_count(self) -> int:
        return len(self.text.strip())

    @property
    def ocr_used(self) -> bool:
        return self.source == SourceKind.OCR


@dataclass(slots=True)
class ParsedDocument:
    """Output of the parse (+ OCR) stage."""

    pages: list[PageText] = field(default_factory=list)
    title: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def total_chars(self) -> int:
        return sum(p.char_count for p in self.pages)

    @property
    def ocr_page_count(self) -> int:
        return sum(1 for p in self.pages if p.ocr_used)

    @property
    def ocr_ratio(self) -> float:
        return self.ocr_page_count / self.page_count if self.page_count else 0.0

    @property
    def is_empty(self) -> bool:
        return self.total_chars == 0


@dataclass(slots=True)
class TextChunk:
    """A retrieval unit. Mirrors the `chunks` table and the Qdrant payload."""

    index: int
    text: str
    page_start: int
    page_end: int
    char_start: int
    char_end: int
    token_count: int
    language: str
    ocr_used: bool = False
    section: str | None = None

    @property
    def spans_pages(self) -> bool:
        return self.page_end > self.page_start


@dataclass(slots=True)
class IngestionResult:
    """What the pipeline hands back to the worker for persistence."""

    chunks: list[TextChunk]
    page_count: int
    language_primary: str
    language_distribution: dict[str, int]
    ocr_used: bool
    ocr_page_ratio: float
    title: str | None = None
    stage_timings_ms: dict[str, float] = field(default_factory=dict)
