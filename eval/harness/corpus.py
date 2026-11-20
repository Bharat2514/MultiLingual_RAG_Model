"""Load the fixed evaluation corpus and turn it into indexable chunks.

The corpus is plain Markdown with a small front matter block and explicit page
breaks, rather than PDFs, for one reason: the gold set must stay valid when the
*chunker* changes. §12.4 sweeps chunk size and overlap, so every run re-chunks
the same source text — a corpus of binary PDFs would make each sweep also a
re-parse, and any parser drift would be silently attributed to the chunker.

OCR is exercised separately by the ingestion tests; what this harness measures is
retrieval and generation over known text.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from app.core.config import ChunkingSettings, settings
from app.ingestion.chunking import chunk_pages
from app.ingestion.cleaning import clean_document
from app.ingestion.types import PageText, SourceKind
from app.retrieval.vectorstore.base import PendingChunk

from . import CORPUS_DIR

PAGE_BREAK = re.compile(r"^<!--\s*page\s*-->\s*$", re.MULTILINE)
FRONT_MATTER = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)


def normalize(text: str) -> str:
    """Fold a string to the form anchor matching compares on.

    NFC because `क` + nukta and `क़` are the same word in different byte
    sequences; whitespace collapse because the chunker joins sentences with a
    single space and the corpus is hand-wrapped; casefold for English.
    """
    return " ".join(unicodedata.normalize("NFC", text).split()).casefold()


@dataclass(slots=True)
class CorpusDocument:
    doc_id: str
    filename: str
    language: str
    title: str
    pages: list[PageText] = field(default_factory=list)

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def text(self) -> str:
        return "\n\n".join(p.text for p in self.pages)


@dataclass(slots=True)
class EvalCorpus:
    documents: list[CorpusDocument]
    chunks: list[PendingChunk]
    session_id: str
    chunking: ChunkingSettings

    @property
    def by_doc(self) -> dict[str, CorpusDocument]:
        return {d.doc_id: d for d in self.documents}

    def chunks_for(self, doc_id: str) -> list[PendingChunk]:
        return [c for c in self.chunks if c.doc_id == doc_id]

    @property
    def fingerprint(self) -> str:
        """Stable id for (corpus content x chunking config).

        Used as the Qdrant collection suffix so a chunk-size sweep gets one
        collection per arm and reruns of the same arm reuse the index instead of
        re-embedding 2 000 chunks.
        """
        digest = hashlib.sha256()
        for chunk in self.chunks:
            digest.update(chunk.chunk_id.encode())
            digest.update(chunk.text.encode())
        cfg = self.chunking
        digest.update(
            f"{cfg.target_tokens}/{cfg.min_tokens}/{cfg.max_tokens}/{cfg.overlap_ratio}".encode()
        )
        return digest.hexdigest()[:12]


def _parse_front_matter(raw: str) -> tuple[dict[str, str], str]:
    match = FRONT_MATTER.match(raw)
    if not match:
        raise ValueError("corpus document is missing its front matter block")

    meta: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if not line.strip():
            continue
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip()

    missing = {"doc_id", "filename", "language", "title"} - meta.keys()
    if missing:
        raise ValueError(f"corpus front matter missing keys: {sorted(missing)}")
    return meta, raw[match.end() :]


def load_document(path: Path) -> CorpusDocument:
    meta, body = _parse_front_matter(path.read_text(encoding="utf-8"))
    pages = [
        PageText(page_no=i, text=part.strip(), source=SourceKind.TEXT_LAYER)
        for i, part in enumerate(PAGE_BREAK.split(body), start=1)
        if part.strip()
    ]
    if not pages:
        raise ValueError(f"{path.name} has no page content")

    return CorpusDocument(
        doc_id=meta["doc_id"],
        filename=meta["filename"],
        language=meta["language"],
        title=meta["title"],
        pages=pages,
    )


def load_corpus(
    *,
    directory: Path | None = None,
    session_id: str = "eval",
    chunking: ChunkingSettings | None = None,
) -> EvalCorpus:
    """Parse, clean and chunk the corpus exactly as ingestion would."""
    directory = directory or CORPUS_DIR
    cfg = chunking or settings.chunking

    documents: list[CorpusDocument] = []
    chunks: list[PendingChunk] = []

    for path in sorted(directory.glob("*.md")):
        if path.name.upper().startswith("README"):
            continue
        doc = load_document(path)
        doc.pages = clean_document(doc.pages)
        documents.append(doc)

        for chunk in chunk_pages(doc.pages, config=cfg):
            chunks.append(
                PendingChunk(
                    # Chunk ids must be stable across runs (so a rerun reuses the
                    # index) but must change when the chunking config does (so a
                    # sweep arm cannot read a previous arm's vectors).
                    chunk_id=f"{doc.doc_id}:{cfg.target_tokens}:{chunk.index}",
                    doc_id=doc.doc_id,
                    session_id=session_id,
                    text=chunk.text,
                    filename=doc.filename,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    chunk_index=chunk.index,
                    language=chunk.language,
                    ocr_used=chunk.ocr_used,
                )
            )

    if not documents:
        raise FileNotFoundError(f"no corpus documents found in {directory}")

    return EvalCorpus(
        documents=documents, chunks=chunks, session_id=session_id, chunking=cfg
    )
