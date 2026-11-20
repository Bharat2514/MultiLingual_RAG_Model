"""The §12.4 ablation grid.

Each arm is one configuration of the *production* retrieval path — nothing here
reimplements retrieval, it only turns stages off. That is the whole point of an
ablation: the numbers must be attributable to a stage, not to a second
implementation of it.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.retrieval.types import SearchMode


@dataclass(frozen=True, slots=True)
class Arm:
    id: str
    label: str
    mode: SearchMode
    rerank: bool
    #: Hinglish dual-query (FR-15). Off means the romanized query alone is searched.
    dual_query: bool = True
    #: Coverage path for document-level questions (`app.generation.intent`).
    coverage: bool = True
    note: str = ""

    @property
    def is_baseline(self) -> bool:
        return self.id == "dense"


#: Ordered so the report reads as a story: each row adds one stage to the row above.
ABLATION_ARMS: tuple[Arm, ...] = (
    Arm(
        id="dense",
        label="dense only",
        mode=SearchMode.DENSE,
        rerank=False,
        note="bi-encoder cosine, no lexical arm, no cross-encoder",
    ),
    Arm(
        id="sparse",
        label="sparse only",
        mode=SearchMode.SPARSE,
        rerank=False,
        note="bge-m3 learned lexical weights alone",
    ),
    Arm(
        id="hybrid",
        label="+ sparse (weighted RRF)",
        mode=SearchMode.HYBRID,
        rerank=False,
        note="dense 0.75 / sparse 0.25, rank-fused",
    ),
    Arm(
        id="dense_rerank",
        label="dense + rerank",
        mode=SearchMode.DENSE,
        rerank=True,
        note="isolates the cross-encoder's contribution from the lexical arm's",
    ),
    Arm(
        id="hybrid_rerank",
        label="+ cross-encoder rerank",
        mode=SearchMode.HYBRID,
        rerank=True,
        note="the shipped configuration",
    ),
    Arm(
        id="hybrid_rerank_no_hinglish",
        label="shipped, minus Hinglish normalisation",
        mode=SearchMode.HYBRID,
        rerank=True,
        dual_query=False,
        note="romanized query only; no Devanagari transliteration variant",
    ),
)

ARMS_BY_ID: dict[str, Arm] = {arm.id: arm for arm in ABLATION_ARMS}

#: The arm the product actually ships, and the one the CI gate watches.
SHIPPED_ARM = "hybrid_rerank"


def resolve(ids: list[str] | None) -> list[Arm]:
    if not ids:
        return list(ABLATION_ARMS)
    unknown = [i for i in ids if i not in ARMS_BY_ID]
    if unknown:
        raise ValueError(f"unknown arm(s) {unknown}; choose from {sorted(ARMS_BY_ID)}")
    return [ARMS_BY_ID[i] for i in ids]
