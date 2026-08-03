"""Shared retrieval data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class SearchMode(StrEnum):
    """Retrieval strategy. The ablation harness (§12.4) sweeps these to
    quantify what hybrid search and reranking actually buy us."""

    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"


@dataclass(slots=True)
class SparseVector:
    """A learned lexical representation: token id -> weight.

    bge-m3 emits these alongside the dense vector, so one forward pass yields
    both halves of hybrid search.
    """

    indices: list[int]
    values: list[float]

    def __len__(self) -> int:
        return len(self.indices)

    @property
    def is_empty(self) -> bool:
        return not self.indices


@dataclass(slots=True)
class Embedding:
    dense: list[float]
    sparse: SparseVector | None = None

    @property
    def dim(self) -> int:
        return len(self.dense)


@dataclass(slots=True)
class ScoredChunk:
    """A retrieved chunk with everything needed to build a citation (FR-22)."""

    chunk_id: str
    doc_id: str
    text: str
    score: float
    filename: str
    page_start: int
    page_end: int
    chunk_index: int
    language: str = "unknown"
    ocr_used: bool = False
    session_id: str = ""
    # Populated after reranking so ablations can compare the two orderings.
    retrieval_score: float | None = None
    rerank_score: float | None = None
    rank: int = 0

    @property
    def page_label(self) -> str:
        return (
            f"p.{self.page_start}"
            if self.page_start == self.page_end
            else f"pp.{self.page_start}-{self.page_end}"
        )

    @property
    def citation_label(self) -> str:
        """The inline citation form the prompt contract asks for: [file p.N]."""
        return f"[{self.filename} {self.page_label}]"


@dataclass(slots=True)
class RetrievalResult:
    chunks: list[ScoredChunk] = field(default_factory=list)
    mode: SearchMode = SearchMode.HYBRID
    reranked: bool = False
    candidates_considered: int = 0
    timings_ms: dict[str, float] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return not self.chunks

    @property
    def top_score(self) -> float:
        return self.chunks[0].score if self.chunks else 0.0
