"""Document-level queries: intent detection, coverage selection, prompt branch.

Regression suite for the failure where "mujhe is document ka summary do" was
answered with "Mujhe ye jaankari aapke documents mein nahi mili". Retrieval was
healthy — 20 candidates, top dense score 0.41 — but the cross-encoder scored
every one around -8 against a -2.0 floor, because no single passage "answers" a
request to summarise. One arbitrary chunk reached the model, whose prompt also
forbade summarising, so it emitted the insufficient-context sentinel.
"""

from __future__ import annotations

import pytest

from app.db.models import Language
from app.generation import prompts
from app.generation.citations import extract_citations, is_grounded, strip_citation_markers
from app.generation.intent import QueryIntent, detect_intent
from app.generation.query import analyze_query
from app.retrieval.coverage import merge_for_summary, select_coverage
from app.retrieval.types import ScoredChunk


def chunk(index: int, doc_id: str = "doc1", filename: str = "policy.pdf") -> ScoredChunk:
    return ScoredChunk(
        chunk_id=f"{doc_id}-{index}",
        doc_id=doc_id,
        text=f"passage {index}",
        score=0.0,
        filename=filename,
        page_start=index,
        page_end=index,
        chunk_index=index,
    )


class TestIntentDetection:
    @pytest.mark.parametrize(
        "query",
        [
            # The reported failure, and its neighbours in each register.
            "mujhe is document ka summary do",
            "is document ka summary do",
            "summary batao",
            "ye document kis baare mein hai",
            "ye document kis bare me hai",
            "key points batao",
            "iske main points kya hain",
            "give me a summary of this document",
            "summarise this",
            "summarize the report",
            "what is this document about",
            "what's this about",
            "tldr",
            "give me an overview",
            "key takeaways?",
            "is document ka saaransh do",
            "इस दस्तावेज़ का सारांश दीजिए",
            "इसका संक्षेप बताइए",
            "मुख्य बिंदु क्या हैं",
            "यह दस्तावेज़ किस बारे में है",
        ],
    )
    def test_document_level_queries(self, query):
        assert detect_intent(query) == QueryIntent.SUMMARY

    @pytest.mark.parametrize(
        "query",
        [
            "notice period kitna hai",
            "what is the penalty for late submission",
            "pension ke liye kitne saal chahiye",
            "धारा 5 में क्या लिखा है",
            "who signed the agreement",
            "clause 7.2 batao",
        ],
    )
    def test_extractive_queries_are_not_misrouted(self, query):
        """Biased towards extractive: a factual question sent down the coverage
        path loses the one passage that held its answer."""
        assert detect_intent(query) == QueryIntent.EXTRACTIVE

    def test_empty_query(self):
        assert detect_intent("") == QueryIntent.EXTRACTIVE

    def test_analyze_query_carries_the_intent(self):
        analyzed = analyze_query("mujhe is document ka summary do")
        assert analyzed.intent == QueryIntent.SUMMARY
        assert analyzed.needs_coverage is True
        # Language handling must be untouched — this was never a register bug.
        assert analyzed.language == Language.HINGLISH

    def test_extractive_query_does_not_request_coverage(self):
        assert analyze_query("notice period kitna hai").needs_coverage is False


class TestCoverageSelection:
    def test_returns_everything_when_it_fits(self):
        chunks = [chunk(i) for i in range(5)]
        assert len(select_coverage(chunks, limit=24)) == 5

    def test_samples_down_to_the_limit(self):
        chunks = [chunk(i) for i in range(100)]
        assert len(select_coverage(chunks, limit=24)) == 24

    def test_keeps_the_head_of_the_document(self):
        """Title page, abstract and introduction carry most of the summary
        signal, so they are taken whole before the stride begins."""
        selected = select_coverage([chunk(i) for i in range(100)], limit=10)
        assert [c.chunk_index for c in selected[:3]] == [0, 1, 2]

    def test_spans_the_whole_document(self):
        selected = select_coverage([chunk(i) for i in range(100)], limit=10)
        # A relevance ranking would clump; coverage must reach the far end.
        assert max(c.chunk_index for c in selected) > 80

    def test_reading_order(self):
        selected = select_coverage([chunk(i) for i in range(100)], limit=12)
        indexes = [c.chunk_index for c in selected]
        assert indexes == sorted(indexes)

    def test_no_duplicates(self):
        selected = select_coverage([chunk(i) for i in range(100)], limit=24)
        assert len({c.chunk_id for c in selected}) == len(selected)

    def test_is_deterministic(self):
        chunks = [chunk(i) for i in range(100)]
        first = [c.chunk_id for c in select_coverage(chunks, limit=24)]
        second = [c.chunk_id for c in select_coverage(chunks, limit=24)]
        assert first == second

    def test_budget_splits_across_documents_by_length(self):
        chunks = [chunk(i, "long", "long.pdf") for i in range(90)]
        chunks += [chunk(i, "short", "short.pdf") for i in range(10)]
        selected = select_coverage(chunks, limit=20)

        per_doc = {"long": 0, "short": 0}
        for c in selected:
            per_doc[c.doc_id] += 1
        assert len(selected) == 20
        assert per_doc["long"] > per_doc["short"]
        # No document in scope goes unmentioned.
        assert per_doc["short"] >= 1

    def test_every_document_gets_at_least_one_chunk(self):
        chunks = [chunk(i, "big", "big.pdf") for i in range(200)]
        chunks += [chunk(0, "tiny", "tiny.pdf")]
        selected = select_coverage(chunks, limit=5)
        assert {c.doc_id for c in selected} == {"big", "tiny"}

    def test_empty_input(self):
        assert select_coverage([], limit=24) == []

    def test_zero_limit(self):
        assert select_coverage([chunk(0)], limit=0) == []


class TestMergeForSummary:
    def test_relevance_hits_survive_the_cut(self):
        """A scoped ask ("summarise the payment terms") must keep the passages
        that actually discuss it, not just an even stride."""
        relevant = [chunk(77), chunk(78)]
        spanning = [chunk(i) for i in range(0, 100, 10)]
        merged = merge_for_summary(relevant, spanning, limit=6)
        ids = {c.chunk_id for c in merged}
        assert "doc1-77" in ids
        assert "doc1-78" in ids

    def test_deduplicates_overlap(self):
        shared = [chunk(0), chunk(1)]
        merged = merge_for_summary(shared, [chunk(0), chunk(1), chunk(2)], limit=10)
        assert len(merged) == 3

    def test_respects_the_limit(self):
        merged = merge_for_summary(
            [chunk(i) for i in range(10)], [chunk(i) for i in range(50)], limit=8
        )
        assert len(merged) == 8

    def test_reading_order(self):
        merged = merge_for_summary([chunk(9), chunk(3)], [chunk(1), chunk(7)], limit=10)
        assert [c.chunk_index for c in merged] == [1, 3, 7, 9]


class TestSummaryPrompt:
    def test_extractive_prompt_still_forbids_summarising(self):
        system = prompts.build_system_prompt(Language.HINGLISH, QueryIntent.EXTRACTIVE)
        assert "do not summarise the whole document" in system

    def test_summary_prompt_permits_summarising(self):
        """Rule 5 vetoed the very thing the user asked for."""
        system = prompts.build_system_prompt(Language.HINGLISH, QueryIntent.SUMMARY)
        assert "do not summarise the whole document" not in system
        assert "summar" in system.lower()

    def test_summary_prompt_relaxes_the_sentinel_rule(self):
        """Told the context must be 'enough', a model handed a sample of a
        document reasons it cannot see all of it and refuses — which is how the
        summary came back as 'not found in your documents'."""
        system = prompts.build_system_prompt(Language.EN, QueryIntent.SUMMARY)
        assert "NOT a reason to refuse" in system
        # The sentinel must still exist for the genuinely-empty case.
        assert prompts.INSUFFICIENT_CONTEXT in system

    def test_summary_prompt_keeps_citations_and_register(self):
        system = prompts.build_system_prompt(Language.HINGLISH, QueryIntent.SUMMARY)
        assert "CITATIONS" in system
        assert "Do NOT reply in Devanagari" in system

    def test_default_intent_is_extractive(self):
        assert prompts.build_system_prompt(Language.EN) == prompts.build_system_prompt(
            Language.EN, QueryIntent.EXTRACTIVE
        )

    def test_user_prompt_flags_the_sample_as_ordered(self):
        user = prompts.build_user_prompt("summary do", [chunk(0)], QueryIntent.SUMMARY)
        assert "as they appear in the document" in user


class TestGroupedIndexCitations:
    """Summaries draw each bullet from several passages, so the model groups the
    indexes: "[1, 4]". Parsing only "[1]" left most bullets unsourced."""

    def test_grouped_indexes_all_resolve(self):
        chunks = [chunk(i) for i in range(5)]
        citations = extract_citations("SREGym is a benchmark [1, 4].", chunks)
        assert {c.chunk_id for c in citations} == {"doc1-0", "doc1-3"}

    def test_grouped_without_spaces(self):
        chunks = [chunk(i) for i in range(5)]
        assert len(extract_citations("Point one [2,3,5].", chunks)) == 3

    def test_single_index_still_works(self):
        chunks = [chunk(i) for i in range(5)]
        citations = extract_citations("Point one [2].", chunks)
        assert [c.chunk_id for c in citations] == ["doc1-1"]

    def test_out_of_range_indexes_are_dropped(self):
        chunks = [chunk(i) for i in range(2)]
        citations = extract_citations("Point one [1, 9].", chunks)
        assert [c.chunk_id for c in citations] == ["doc1-0"]

    def test_grouped_markers_are_stripped_for_the_grounding_check(self):
        assert strip_citation_markers("Fee is 500. [1, 4]") == "Fee is 500."

    def test_index_with_page_resolves_and_keeps_the_cited_page(self):
        """Half-compliance with the [file p.N] instruction: passage index plus
        page. Unparsed, a fully-sourced answer was reported as ungrounded."""
        chunks = [chunk(i) for i in range(5)]
        citations = extract_citations("Metastable failures are congestive [2 p.5, 3 p.19].", chunks)
        assert {(c.chunk_id, c.page) for c in citations} == {("doc1-1", 5), ("doc1-2", 19)}

    def test_index_with_page_range(self):
        chunks = [chunk(i) for i in range(5)]
        citations = extract_citations("A claim [2 pp.5-7].", chunks)
        assert [(c.chunk_id, c.page) for c in citations] == [("doc1-1", 5)]

    def test_index_page_answers_count_as_grounded(self):
        chunks = [chunk(i) for i in range(5)]
        answer = "Metastable failures are self-sustaining congestive collapses [2 p.5, 3 p.19]."
        assert is_grounded(answer, extract_citations(answer, chunks), chunks) is True

    def test_proper_filename_citations_still_win(self):
        """The [file p.N] form must keep taking precedence over the fallback."""
        chunks = [chunk(i) for i in range(5)]
        citations = extract_citations("A claim [policy.pdf p.3].", chunks)
        assert [(c.filename, c.page) for c in citations] == [("policy.pdf", 3)]


class TestContextIndex:
    """A bare `[7]` means the 7th CONTEXT passage. The citation list is
    deduplicated and re-sorted by (filename, page), so its array order does not
    recover that — a client mapping the Nth citation to the Nth passage links
    the reader to the wrong source, and drops any marker past the list length.
    """

    def test_records_the_position_in_the_context(self):
        chunks = [chunk(i) for i in range(10)]
        citations = extract_citations("A [3]. B [7].", chunks)
        assert sorted(c.context_index for c in citations) == [3, 7]

    def test_index_beyond_the_citation_count_still_resolves(self):
        """Two markers, one of them the 20th passage: the resulting list has
        length 2, so array-position mapping would lose [20] entirely."""
        chunks = [chunk(i) for i in range(24)]
        citations = extract_citations("A [1]. B [20].", chunks)
        assert len(citations) == 2
        assert {c.context_index for c in citations} == {1, 20}
        by_index = {c.context_index: c for c in citations}
        assert by_index[20].chunk_id == "doc1-19"

    def test_survives_the_reordering(self):
        """Citations sort by (filename, page); context_index must still point
        back at the passage the model cited."""
        chunks = [
            chunk(0, filename="zeta.pdf"),
            chunk(1, filename="alpha.pdf"),
        ]
        citations = extract_citations("A [1]. B [2].", chunks)
        # Sorted output puts alpha.pdf first, though it was passage 2.
        assert [c.filename for c in citations] == ["alpha.pdf", "zeta.pdf"]
        assert [c.context_index for c in citations] == [2, 1]

    def test_filename_form_records_it_too(self):
        chunks = [chunk(i) for i in range(5)]
        chunks[2].page_start = chunks[2].page_end = 3
        citations = extract_citations("A [policy.pdf p.3].", chunks)
        assert [c.context_index for c in citations] == [3]
