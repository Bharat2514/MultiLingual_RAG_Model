"""Clause-level diff between document versions, and cross-document conflicts.

A circular is amended by issuing a new circular. The reader's question is never
"which bytes changed" — it is "what is different now": the service requirement
dropped from 20 years to 15, a clause was added, a fee was removed. A line diff
over two PDFs is useless for that (re-flowed paragraphs differ everywhere), so
the comparison works on *sentences*:

1. both documents are reduced to their ordered sentences (chunk overlap removed);
2. the sentence sequences are aligned, and in each replaced region the sentences
   are paired by similarity, so a reworded clause shows as *modified*, not as a
   deletion plus an unrelated addition;
3. a modified pair whose numbers differ is reported as a **number change** —
   the most consequential kind of amendment, called out separately.

Conflict detection reuses the same pairing across two *different* documents:
near-identical sentences that disagree on a number are the classic sign of two
circulars that cannot both be current.
"""

from __future__ import annotations

import difflib
from collections import defaultdict
from dataclasses import asdict, dataclass, field

from app.intelligence.textutil import normalize, numbers_in, words
from app.nlp.sentences import split_sentences

#: Two sentences at or above this similarity are the same clause, edited.
PAIR_THRESHOLD = 0.55
#: Stricter bar for *conflicts* across documents, where a false pair produces a
#: false alarm rather than a slightly odd diff.
CONFLICT_THRESHOLD = 0.6
MAX_SENTENCES = 4000


@dataclass(slots=True)
class Sentence:
    text: str
    page: int
    chunk_id: str
    key: str = ""

    def __post_init__(self) -> None:
        self.key = normalize(self.text)


@dataclass(slots=True)
class Change:
    kind: str  # added | removed | modified
    old: str | None = None
    new: str | None = None
    old_page: int | None = None
    new_page: int | None = None
    similarity: float | None = None
    number_changes: list[dict] = field(default_factory=list)
    word_diff: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def sentences_of(chunks: list[dict]) -> list[Sentence]:
    """Ordered, de-duplicated sentences from chunk dicts (`text`, `page`, `chunk_id`).

    Chunks overlap by ~12%, so the sentences at each boundary appear twice;
    keeping only the first occurrence restores the document's own sequence.
    """
    out: list[Sentence] = []
    seen: set[str] = set()
    for chunk in chunks:
        for raw in split_sentences(chunk["text"]):
            text = " ".join(raw.text.split())
            if len(text) < 4:
                continue
            sentence = Sentence(text, chunk["page"], chunk["chunk_id"])
            if sentence.key in seen:
                continue
            seen.add(sentence.key)
            out.append(sentence)
            if len(out) >= MAX_SENTENCES:
                return out
    return out


def similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, words(a), words(b), autojunk=False).ratio()


def number_changes(old: str, new: str) -> list[dict]:
    """Numbers that differ between two versions of the same clause."""
    before, after = numbers_in(old), numbers_in(new)
    if before == after:
        return []
    matcher = difflib.SequenceMatcher(None, before, after, autojunk=False)
    changes: list[dict] = []
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == "equal":
            continue
        changes.append(
            {"old": ", ".join(before[i1:i2]) or None, "new": ", ".join(after[j1:j2]) or None}
        )
    return changes


def word_diff(old: str, new: str) -> list[dict]:
    """Inline diff as [{op: equal|insert|delete, text}] for highlighting."""
    a, b = old.split(), new.split()
    out: list[dict] = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if op == "equal":
            out.append({"op": "equal", "text": " ".join(a[i1:i2])})
        else:
            if i2 > i1:
                out.append({"op": "delete", "text": " ".join(a[i1:i2])})
            if j2 > j1:
                out.append({"op": "insert", "text": " ".join(b[j1:j2])})
    return out


def _pair_block(old: list[Sentence], new: list[Sentence]) -> list[Change]:
    """Greedy best-match pairing inside one replaced region."""
    if len(old) * len(new) > 40_000:
        # A rewrite this large is not an amendment; pairing it would cost
        # minutes of SequenceMatcher time for a diff nobody can read anyway.
        return [Change("removed", old=s.text, old_page=s.page) for s in old] + [
            Change("added", new=s.text, new_page=s.page) for s in new
        ]
    scored = sorted(
        ((similarity(o.text, n.text), i, j) for i, o in enumerate(old) for j, n in enumerate(new)),
        reverse=True,
    )
    used_old: set[int] = set()
    used_new: set[int] = set()
    pairs: dict[int, tuple[int, float]] = {}
    for score, i, j in scored:
        if score < PAIR_THRESHOLD:
            break
        if i in used_old or j in used_new:
            continue
        used_old.add(i)
        used_new.add(j)
        pairs[i] = (j, score)

    changes: list[Change] = []
    for i, sentence in enumerate(old):
        if i in pairs:
            j, score = pairs[i]
            other = new[j]
            changes.append(
                Change(
                    "modified",
                    old=sentence.text,
                    new=other.text,
                    old_page=sentence.page,
                    new_page=other.page,
                    similarity=round(score, 3),
                    number_changes=number_changes(sentence.text, other.text),
                    word_diff=word_diff(sentence.text, other.text),
                )
            )
        else:
            changes.append(Change("removed", old=sentence.text, old_page=sentence.page))
    for j, sentence in enumerate(new):
        if j not in used_new:
            changes.append(Change("added", new=sentence.text, new_page=sentence.page))
    return changes


def diff_documents(old_chunks: list[dict], new_chunks: list[dict]) -> dict:
    old = sentences_of(old_chunks)
    new = sentences_of(new_chunks)

    matcher = difflib.SequenceMatcher(
        None, [s.key for s in old], [s.key for s in new], autojunk=False
    )
    changes: list[Change] = []
    unchanged = 0
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == "equal":
            unchanged += i2 - i1
        elif op == "delete":
            changes += [Change("removed", old=s.text, old_page=s.page) for s in old[i1:i2]]
        elif op == "insert":
            changes += [Change("added", new=s.text, new_page=s.page) for s in new[j1:j2]]
        else:
            changes += _pair_block(old[i1:i2], new[j1:j2])

    counts = {"added": 0, "removed": 0, "modified": 0}
    for change in changes:
        counts[change.kind] += 1

    return {
        "summary": {
            **counts,
            "unchanged": unchanged,
            "number_changes": sum(1 for c in changes if c.number_changes),
            "old_sentences": len(old),
            "new_sentences": len(new),
        },
        # Number changes first: they are what a reader of an amendment needs.
        "changes": [
            c.to_dict()
            for c in sorted(changes, key=lambda c: (not c.number_changes, c.kind != "modified"))
        ],
    }


def find_conflicts(a_chunks: list[dict], b_chunks: list[dict], *, limit: int = 50) -> list[dict]:
    """Near-identical sentences in two documents that disagree on a number."""
    a = [s for s in sentences_of(a_chunks) if numbers_in(s.text)]
    b = [s for s in sentences_of(b_chunks) if numbers_in(s.text)]
    if not a or not b:
        return []

    # Inverted index on content words, so only sentences sharing vocabulary are
    # compared — keeps this near-linear instead of |a| × |b| SequenceMatchers.
    index: dict[str, set[int]] = defaultdict(set)
    for j, sentence in enumerate(b):
        for w in set(words(sentence.text)):
            if len(w) >= 4 and not w.isdigit():
                index[w].add(j)

    conflicts: list[dict] = []
    for sa in a:
        overlap: dict[int, int] = defaultdict(int)
        for w in set(words(sa.text)):
            for j in index.get(w, ()):
                overlap[j] += 1
        for j, shared in sorted(overlap.items(), key=lambda kv: -kv[1])[:5]:
            if shared < 3:
                continue
            sb = b[j]
            score = similarity(sa.text, sb.text)
            if score < CONFLICT_THRESHOLD:
                continue
            changes = number_changes(sa.text, sb.text)
            if not changes:
                continue
            conflicts.append(
                {
                    "a": sa.text,
                    "b": sb.text,
                    "a_page": sa.page,
                    "b_page": sb.page,
                    "similarity": round(score, 3),
                    "number_changes": changes,
                }
            )
    conflicts.sort(key=lambda c: -c["similarity"])
    return conflicts[:limit]
