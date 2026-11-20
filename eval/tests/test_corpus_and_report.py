"""Corpus parsing, fingerprinting, judge parsing and table rendering."""

from __future__ import annotations

import pytest
from harness.corpus import load_corpus, load_document
from harness.judge import parse_judgement
from harness.report import table

from app.core.config import settings

FRONT_MATTER = """---
doc_id: sample
filename: sample.pdf
language: en
title: A Sample
---

First page body.

<!-- page -->

Second page body.
"""


def test_load_document_parses_front_matter_and_page_breaks(tmp_path):
    path = tmp_path / "sample.md"
    path.write_text(FRONT_MATTER, encoding="utf-8")
    doc = load_document(path)
    assert doc.doc_id == "sample"
    assert doc.filename == "sample.pdf"
    assert doc.page_count == 2
    assert doc.pages[0].page_no == 1
    assert "Second page" in doc.pages[1].text


def test_load_document_rejects_a_file_without_front_matter(tmp_path):
    path = tmp_path / "bad.md"
    path.write_text("no front matter here\n", encoding="utf-8")
    with pytest.raises(ValueError, match="front matter"):
        load_document(path)


def test_load_document_rejects_incomplete_front_matter(tmp_path):
    path = tmp_path / "bad.md"
    path.write_text("---\ndoc_id: x\n---\n\nbody\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing keys"):
        load_document(path)


def test_chunk_ids_and_fingerprint_change_with_the_chunking_config():
    """Two sweep arms must not be able to read each other's vectors."""
    default = load_corpus()
    smaller = load_corpus(
        chunking=settings.chunking.model_copy(update={"target_tokens": 250, "max_tokens": 400})
    )
    assert default.fingerprint != smaller.fingerprint
    assert len(smaller.chunks) > len(default.chunks)
    assert {c.chunk_id for c in default.chunks}.isdisjoint(c.chunk_id for c in smaller.chunks)


def test_fingerprint_is_stable_across_reloads():
    assert load_corpus().fingerprint == load_corpus().fingerprint


def test_shipped_corpus_has_labelled_and_distractor_documents():
    corpus = load_corpus()
    assert len(corpus.documents) >= 10
    languages = {d.language for d in corpus.documents}
    assert languages == {"en", "hi"}
    assert all(corpus.chunks_for(d.doc_id) for d in corpus.documents), (
        "a document produced no chunks"
    )


# --- judge -------------------------------------------------------------------


def test_parse_judgement_reads_a_plain_json_object():
    j = parse_judgement(
        '{"faithfulness": 0.9, "answer_relevancy": 0.8, "correctness": 1.0, '
        '"context_precision": 0.4, "reason": "fine"}'
    )
    assert j.ok
    assert j.faithfulness == pytest.approx(0.9)
    assert j.reason == "fine"


def test_parse_judgement_tolerates_code_fences_and_prose():
    j = parse_judgement(
        'Sure!\n```json\n{"faithfulness": 1, "answer_relevancy": 1, '
        '"correctness": 1, "context_precision": 1}\n```'
    )
    assert j.ok
    assert j.correctness == pytest.approx(1.0)


def test_parse_judgement_clamps_out_of_range_scores():
    j = parse_judgement(
        '{"faithfulness": 7, "answer_relevancy": -3, "correctness": "x", "context_precision": 0.5}'
    )
    assert j.faithfulness == 1.0
    assert j.answer_relevancy == 0.0
    assert j.correctness == 0.0


def test_a_failed_judgement_is_marked_not_ok_rather_than_scored_zero():
    """Missing data must be excluded from the mean, not averaged in as a zero —
    otherwise a provider outage reads as a model quality collapse."""
    j = parse_judgement("the model refused to answer")
    assert not j.ok
    assert "no JSON" in j.reason


# --- report ------------------------------------------------------------------


def test_table_renders_valid_markdown_with_aligned_columns():
    rendered = table(["arm", "Hit@5"], [["dense", "0.873"], ["hybrid+rerank", "0.940"]])
    lines = rendered.splitlines()
    assert len(lines) == 4
    assert lines[0].startswith("| arm")
    assert set(lines[1]) <= set("|-: ")
    assert all(line.count("|") == 3 for line in lines)
    # Right-aligned numeric column: every row is the same width.
    assert len({len(line) for line in lines}) == 1
