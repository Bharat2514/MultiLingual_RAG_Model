"""The trilingual gold set (§12.1): schema, loading, and anchor validation.

Relevance is expressed as **text anchors**, not chunk ids. A gold label of
"chunk 7 of policy.pdf" is only true for one chunking configuration, and §12.4
sweeps chunk size and overlap — under a 300-token target the same answer lives in
chunk 12. Anchoring on a distinctive substring of the answer sentence keeps every
label valid across the sweep, and makes the labels reviewable by a human reading
the corpus rather than a chunk dump.

A chunk is relevant to a question when it contains **all** of that question's
anchors. `validate` fails loudly if an anchor matches zero chunks or every chunk
in its document — a label that cannot discriminate is a broken label, and a
silently unmatched anchor would depress recall for a reason that looks like a
retrieval bug.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from app.retrieval.vectorstore.base import PendingChunk

from . import GOLD_DIR
from .corpus import EvalCorpus, normalize


class QueryLang(StrEnum):
    HI = "hi"
    EN = "en"
    HINGLISH = "hinglish"


class QueryKind(StrEnum):
    #: Answer is a phrase in one passage.
    FACTOID = "factoid"
    #: Answer is a number, amount or duration — the lexical-recall probe.
    NUMERIC = "numeric"
    #: Answer is a clause/section/reference identifier.
    CLAUSE = "clause"
    #: Question language differs from the answering document's language.
    CROSSLINGUAL = "crosslingual"
    #: Document-level: no single passage answers it.
    SUMMARY = "summary"
    #: Answerable from no document in the corpus — must return "not found".
    ADVERSARIAL = "adversarial"


@dataclass(slots=True)
class GoldQuestion:
    id: str
    query: str
    lang: QueryLang
    kind: QueryKind
    doc: str | None
    anchors: list[str]
    answer: str | None
    expect_language: str
    answerable: bool = True
    notes: str = ""
    #: Filled by `validate`; the chunk ids that satisfy the anchors.
    relevant_chunk_ids: set[str] = field(default_factory=set)

    @property
    def is_crosslingual(self) -> bool:
        return self.kind is QueryKind.CROSSLINGUAL

    def matches(self, text: str) -> bool:
        """Whether a passage contains every anchor for this question."""
        if not self.anchors:
            return False
        folded = normalize(text)
        return all(normalize(a) in folded for a in self.anchors)


@dataclass(slots=True)
class GoldSet:
    questions: list[GoldQuestion]

    def __len__(self) -> int:
        return len(self.questions)

    def __iter__(self) -> Iterator[GoldQuestion]:
        return iter(self.questions)

    @property
    def answerable(self) -> list[GoldQuestion]:
        return [q for q in self.questions if q.answerable]

    @property
    def adversarial(self) -> list[GoldQuestion]:
        return [q for q in self.questions if not q.answerable]

    def of_kind(self, *kinds: QueryKind) -> list[GoldQuestion]:
        wanted = set(kinds)
        return [q for q in self.questions if q.kind in wanted]

    def of_lang(self, *langs: QueryLang) -> list[GoldQuestion]:
        wanted = set(langs)
        return [q for q in self.questions if q.lang in wanted]

    def subset(self, ids: Iterable[str]) -> GoldSet:
        wanted = set(ids)
        return GoldSet([q for q in self.questions if q.id in wanted])

    def sample(self, n: int) -> GoldSet:
        """A deterministic, stratified slice — the shape CI runs (see `make eval-ci`).

        Stratified by (kind, lang) and ordered by id so the CI subset is stable
        across runs: a gate that samples randomly reports a different number every
        build and cannot be compared to a baseline.
        """
        if n >= len(self.questions):
            return GoldSet(list(self.questions))

        buckets: dict[tuple[str, str], list[GoldQuestion]] = {}
        for q in sorted(self.questions, key=lambda q: q.id):
            buckets.setdefault((str(q.kind), str(q.lang)), []).append(q)

        picked: list[GoldQuestion] = []
        depth = 0
        while len(picked) < n:
            progressed = False
            for key in sorted(buckets):
                bucket = buckets[key]
                if depth < len(bucket):
                    picked.append(bucket[depth])
                    progressed = True
                    if len(picked) == n:
                        break
            if not progressed:
                break
            depth += 1
        return GoldSet(sorted(picked, key=lambda q: q.id))


def _coerce(raw: dict, source: Path) -> GoldQuestion:
    try:
        kind = QueryKind(raw["kind"])
        answerable = raw.get("answerable", kind is not QueryKind.ADVERSARIAL)
        return GoldQuestion(
            id=raw["id"],
            query=raw["query"],
            lang=QueryLang(raw["lang"]),
            kind=kind,
            doc=raw.get("doc"),
            anchors=list(raw.get("anchors", [])),
            answer=raw.get("answer"),
            expect_language=raw["expect_language"],
            answerable=answerable,
            notes=raw.get("notes", ""),
        )
    except (KeyError, ValueError) as exc:
        raise ValueError(f"{source.name}: bad gold record {raw.get('id', '?')}: {exc}") from exc


def load_goldset(directory: Path | None = None) -> GoldSet:
    directory = directory or GOLD_DIR
    questions: list[GoldQuestion] = []
    seen: set[str] = set()

    for path in sorted(directory.glob("*.jsonl")):
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path.name}:{lineno}: invalid JSON: {exc}") from exc
            question = _coerce(raw, path)
            if question.id in seen:
                raise ValueError(f"{path.name}:{lineno}: duplicate question id {question.id!r}")
            seen.add(question.id)
            questions.append(question)

    if not questions:
        raise FileNotFoundError(f"no gold questions found in {directory}")
    return GoldSet(questions)


@dataclass(slots=True)
class ValidationReport:
    total: int
    answerable: int
    adversarial: int
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


def validate(gold: GoldSet, corpus: EvalCorpus, *, strict: bool = True) -> ValidationReport:
    """Resolve every question's anchors against the chunked corpus.

    Populates `relevant_chunk_ids` in place, which is what the retrieval metrics
    score against.
    """
    by_doc: dict[str, list[PendingChunk]] = {}
    for chunk in corpus.chunks:
        by_doc.setdefault(chunk.doc_id, []).append(chunk)

    problems: list[str] = []

    for question in gold.questions:
        if not question.answerable:
            if question.anchors:
                problems.append(f"{question.id}: adversarial question must not carry anchors")
            question.relevant_chunk_ids = set()
            continue

        if not question.doc:
            problems.append(f"{question.id}: answerable question has no `doc`")
            continue
        chunks = by_doc.get(question.doc)
        if chunks is None:
            problems.append(f"{question.id}: unknown doc {question.doc!r}")
            continue

        if question.kind is QueryKind.SUMMARY:
            # A summary is grounded on the whole document, so every chunk of it
            # counts as relevant. Retrieval metrics on these are near-meaningless
            # by construction and are reported separately, never pooled.
            question.relevant_chunk_ids = {c.chunk_id for c in chunks}
            continue

        hits = {c.chunk_id for c in chunks if question.matches(c.text)}
        if not hits:
            problems.append(
                f"{question.id}: anchors {question.anchors!r} match no chunk of {question.doc}"
            )
        elif len(hits) == len(chunks):
            problems.append(
                f"{question.id}: anchors match every chunk of {question.doc} — not discriminating"
            )
        question.relevant_chunk_ids = hits

    report = ValidationReport(
        total=len(gold),
        answerable=len(gold.answerable),
        adversarial=len(gold.adversarial),
        problems=problems,
    )
    if strict and problems:
        raise ValueError(
            "gold set does not resolve against the corpus:\n  " + "\n  ".join(problems)
        )
    return report
