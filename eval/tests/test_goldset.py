"""Gold-set loading, anchor resolution and the validation that guards the labels."""

from __future__ import annotations

import json

import pytest
from app.core.config import settings
from app.retrieval.vectorstore.base import PendingChunk
from harness.corpus import CorpusDocument, EvalCorpus, load_corpus, normalize
from harness.goldset import (
    GoldQuestion,
    GoldSet,
    QueryKind,
    QueryLang,
    load_goldset,
    validate,
)


def _chunk(chunk_id: str, text: str, doc: str = "d1") -> PendingChunk:
    return PendingChunk(
        chunk_id=chunk_id,
        doc_id=doc,
        session_id="eval",
        text=text,
        filename="f.pdf",
        page_start=1,
        page_end=1,
        chunk_index=0,
        language="en",
        ocr_used=False,
    )


def _corpus(chunks: list[PendingChunk]) -> EvalCorpus:
    return EvalCorpus(
        documents=[CorpusDocument("d1", "f.pdf", "en", "T")],
        chunks=chunks,
        session_id="eval",
        chunking=settings.chunking,
    )


def _question(anchors: list[str], **overrides) -> GoldQuestion:
    defaults = {
        "id": "q1",
        "query": "q",
        "lang": QueryLang.EN,
        "kind": QueryKind.FACTOID,
        "doc": "d1",
        "anchors": anchors,
        "answer": "a",
        "expect_language": "en",
    }
    defaults.update(overrides)
    return GoldQuestion(**defaults)


# --- anchor matching ---------------------------------------------------------


def test_anchors_match_across_nfc_normalisation():
    """`क` + nukta and `क़` are the same word in different byte sequences. Without
    folding, a correctly retrieved passage would be scored as a miss."""
    decomposed = "क" + "़" + " वर्ष"
    composed = "क़ वर्ष"
    assert normalize(decomposed) == normalize(composed)
    assert _question([composed]).matches(f"prefix {decomposed} suffix")


def test_anchors_match_across_whitespace_and_case():
    question = _question(["Forty Five   Days"])
    assert question.matches("a maximum of forty five\ndays of leave")


def test_all_anchors_must_be_present():
    question = _question(["earned leave", "forty five"])
    assert question.matches("earned leave up to forty five days")
    assert not question.matches("earned leave up to sixty days")


def test_a_question_with_no_anchors_matches_nothing():
    assert not _question([]).matches("anything at all")


# --- validation --------------------------------------------------------------


def test_validate_resolves_anchors_to_chunk_ids():
    corpus = _corpus(
        [_chunk("c1", "pension needs twenty years"), _chunk("c2", "unrelated")]
    )
    gold = GoldSet([_question(["twenty years"])])
    validate(gold, corpus)
    assert gold.questions[0].relevant_chunk_ids == {"c1"}


def test_validate_rejects_an_anchor_that_matches_nothing():
    corpus = _corpus([_chunk("c1", "pension needs twenty years")])
    gold = GoldSet([_question(["thirty years"])])
    with pytest.raises(ValueError, match="match no chunk"):
        validate(gold, corpus)


def test_validate_rejects_an_anchor_that_matches_every_chunk():
    """A label that cannot discriminate silently inflates recall to 1.0."""
    corpus = _corpus([_chunk("c1", "pension rules"), _chunk("c2", "pension tables")])
    gold = GoldSet([_question(["pension"])])
    with pytest.raises(ValueError, match="not discriminating"):
        validate(gold, corpus)


def test_validate_rejects_an_adversarial_question_carrying_anchors():
    corpus = _corpus([_chunk("c1", "pension needs twenty years")])
    gold = GoldSet(
        [_question(["twenty years"], kind=QueryKind.ADVERSARIAL, answerable=False)]
    )
    with pytest.raises(ValueError, match="must not carry anchors"):
        validate(gold, corpus)


def test_summary_questions_mark_their_whole_document_relevant():
    corpus = _corpus([_chunk("c1", "one"), _chunk("c2", "two")])
    gold = GoldSet([_question([], kind=QueryKind.SUMMARY)])
    validate(gold, corpus)
    assert gold.questions[0].relevant_chunk_ids == {"c1", "c2"}


# --- loading and sampling ----------------------------------------------------


def test_load_goldset_rejects_duplicate_ids(tmp_path):
    record = {
        "id": "dup",
        "query": "q",
        "lang": "en",
        "kind": "factoid",
        "doc": "d1",
        "anchors": ["x"],
        "answer": "a",
        "expect_language": "en",
    }
    (tmp_path / "a.jsonl").write_text(
        json.dumps(record) + "\n" + json.dumps(record) + "\n"
    )
    with pytest.raises(ValueError, match="duplicate question id"):
        load_goldset(tmp_path)


def test_load_goldset_skips_comment_lines(tmp_path):
    record = {
        "id": "q1",
        "query": "q",
        "lang": "en",
        "kind": "factoid",
        "doc": "d1",
        "anchors": ["x"],
        "answer": "a",
        "expect_language": "en",
    }
    (tmp_path / "a.jsonl").write_text("// a note\n\n" + json.dumps(record) + "\n")
    assert len(load_goldset(tmp_path)) == 1


def test_sample_is_deterministic_and_stratified():
    """The CI gate compares against a baseline, so its subset must be the same
    subset on every run."""
    gold = load_goldset()
    first, second = gold.sample(30), gold.sample(30)
    assert [q.id for q in first] == [q.id for q in second]
    assert len(first) == 30
    kinds = {str(q.kind) for q in first}
    assert len(kinds) >= 4, f"subset covers too few question kinds: {kinds}"
    assert {str(q.lang) for q in first} == {"hi", "en", "hinglish"}


def test_sample_larger_than_the_set_returns_everything():
    gold = load_goldset()
    assert len(gold.sample(10_000)) == len(gold)


# --- the shipped gold set ----------------------------------------------------


def test_shipped_goldset_resolves_against_the_shipped_corpus():
    """The check `make eval-validate` runs, as a test so CI catches it too."""
    report = validate(load_goldset(), load_corpus(), strict=False)
    assert report.ok, "unresolved gold labels:\n" + "\n".join(report.problems)
    assert report.total >= 150, "PRD §12.1 requires at least 150 gold questions"
    assert report.adversarial >= 20
