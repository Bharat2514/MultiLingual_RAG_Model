"""Query analysis, prompt contract, citations, and the grounding rules."""

from __future__ import annotations

import pytest

from app.db.models import Language
from app.generation import prompts
from app.generation.citations import extract_citations, is_grounded, strip_citation_markers
from app.generation.query import analyze_query
from app.nlp.transliterate import lexicon_coverage, to_devanagari
from app.retrieval.types import ScoredChunk


def chunk(chunk_id: str, text: str, filename: str = "policy.pdf", page: int = 4) -> ScoredChunk:
    return ScoredChunk(
        chunk_id=chunk_id,
        doc_id="doc1",
        text=text,
        score=0.9,
        filename=filename,
        page_start=page,
        page_end=page,
        chunk_index=int(chunk_id[-1]) if chunk_id[-1].isdigit() else 0,
    )


class TestTransliteration:
    @pytest.mark.parametrize(
        ("hinglish", "expected_fragment"),
        [
            ("mujhe batao", "मुझे"),
            ("kya likha hai", "क्या"),
            ("shiksha niti", "शिक्षा"),
            ("pension ke liye kitne saal", "कितने"),
        ],
    )
    def test_converts_common_hinglish(self, hinglish, expected_fragment):
        assert expected_fragment in to_devanagari(hinglish)

    def test_english_words_survive_intact(self):
        """Hinglish is code-mixed. Mangling "document" into Devanagari produces
        a token that matches nothing and pollutes the query vector."""
        result = to_devanagari("mujhe is document ka summary do")
        assert "document" in result
        assert "summary" in result
        assert "मुझे" in result

    def test_unknown_words_are_left_alone_by_default(self):
        assert "submission" in to_devanagari("late submission par kitna jurmana")

    def test_devanagari_input_is_untouched(self):
        text = "भारत सरकार की नीति"
        assert to_devanagari(text) == text

    def test_acronyms_survive(self):
        assert "RTI" in to_devanagari("RTI application kaise karein")

    def test_coverage_scores_hinglish_above_english(self):
        assert lexicon_coverage("mujhe batao kya hai") > lexicon_coverage(
            "what is written in the report"
        )


class TestQueryAnalysis:
    def test_english_query_is_not_transliterated(self):
        """Regression: Hindi/English homographs ("the" is also Hindi *थे*, "is"
        is *इस*) turned pure English queries into nonsense."""
        analyzed = analyze_query("What is the penalty for late submission?")

        assert analyzed.language == Language.EN
        assert analyzed.variants == ["What is the penalty for late submission?"]
        assert analyzed.transliterated is None

    def test_hindi_query_needs_no_variant(self):
        analyzed = analyze_query("पेंशन के लिए कितने साल की सेवा चाहिए?")

        assert analyzed.language == Language.HI
        assert not analyzed.uses_dual_query

    def test_hinglish_produces_a_dual_query(self):
        analyzed = analyze_query("shiksha niti ke bare mein batao")

        assert analyzed.language == Language.HINGLISH
        assert analyzed.uses_dual_query
        assert "शिक्षा" in (analyzed.transliterated or "")

    def test_empty_query(self):
        analyzed = analyze_query("   ")
        assert analyzed.language == Language.UNKNOWN
        assert analyzed.variants == []


class TestPromptContract:
    def test_system_prompt_carries_the_grounding_rules(self):
        prompt = prompts.build_system_prompt(Language.EN)

        assert prompts.INSUFFICIENT_CONTEXT in prompt
        assert "only" in prompt.lower()
        assert "cite" in prompt.lower()

    @pytest.mark.parametrize(
        ("language", "must_mention"),
        [
            (Language.HI, "Devanagari"),
            (Language.HINGLISH, "Latin script"),
            (Language.EN, "English"),
        ],
    )
    def test_register_instruction_matches_the_query_language(self, language, must_mention):
        assert must_mention in prompts.build_system_prompt(language)

    def test_hinglish_is_told_not_to_answer_in_devanagari(self):
        # Answering Devanagari to a romanized question fails language fidelity.
        assert "NOT" in prompts.build_system_prompt(Language.HINGLISH)

    def test_context_is_numbered_and_source_labelled(self):
        rendered = prompts.format_context([chunk("c1", "First."), chunk("c2", "Second.")])

        assert "[1] SOURCE: [policy.pdf p.4]" in rendered
        assert "[2] SOURCE:" in rendered

    def test_user_prompt_includes_the_question(self):
        built = prompts.build_user_prompt("What is the fee?", [chunk("c1", "The fee is 500.")])

        assert "QUESTION: What is the fee?" in built
        assert "The fee is 500." in built

    @pytest.mark.parametrize(
        ("language", "fragment"),
        [
            (Language.HI, "नहीं मिली"),
            (Language.HINGLISH, "nahi mili"),
            (Language.EN, "couldn't find"),
        ],
    )
    def test_not_found_message_is_localised(self, language, fragment):
        assert fragment in prompts.not_found_message(language)

    def test_detects_the_insufficient_sentinel(self):
        assert prompts.is_insufficient("INSUFFICIENT_CONTEXT")
        assert prompts.is_insufficient("Sorry — INSUFFICIENT_CONTEXT.")
        assert not prompts.is_insufficient("The fee is 500. [policy.pdf p.4]")


class TestCitations:
    def test_extracts_a_valid_citation(self):
        chunks = [chunk("c1", "The fee is 500 rupees.")]

        citations = extract_citations("The fee is 500 rupees. [policy.pdf p.4]", chunks)

        assert len(citations) == 1
        assert citations[0].filename == "policy.pdf"
        assert citations[0].page == 4
        assert citations[0].chunk_id == "c1"

    def test_discards_a_fabricated_source(self):
        """A citation to a document that was never in the context is a
        hallucination wearing a verifiable disguise — it must not be shown."""
        chunks = [chunk("c1", "Real content.")]

        citations = extract_citations("Made up. [does_not_exist.pdf p.9]", chunks)

        assert citations == []

    def test_handles_a_page_range(self):
        c = chunk("c1", "Spans pages.")
        c.page_end = 6
        assert len(extract_citations("Text. [policy.pdf pp.4-6]", [c])) == 1

    def test_deduplicates_repeated_citations(self):
        chunks = [chunk("c1", "Content.")]
        answer = "One. [policy.pdf p.4] Two. [policy.pdf p.4]"

        assert len(extract_citations(answer, chunks)) == 1

    def test_falls_back_to_passage_numbers(self):
        # Models sometimes cite [1] despite the instruction; the ordering is
        # ours, so this is still resolvable.
        chunks = [chunk("c1", "First."), chunk("c2", "Second.")]

        citations = extract_citations("The answer is here [2].", chunks)

        assert len(citations) == 1
        assert citations[0].chunk_id == "c2"

    def test_picks_the_chunk_whose_pages_contain_the_citation(self):
        chunks = [chunk("c1", "Page four.", page=4), chunk("c2", "Page nine.", page=9)]

        citations = extract_citations("Text. [policy.pdf p.9]", chunks)

        assert citations[0].chunk_id == "c2"

    def test_strips_markers(self):
        assert strip_citation_markers("Fee is 500. [policy.pdf p.4]") == "Fee is 500."


class TestGrounding:
    def test_cited_answer_is_grounded(self):
        chunks = [chunk("c1", "Content.")]
        citations = extract_citations("Answer. [policy.pdf p.4]", chunks)

        assert is_grounded("Answer. [policy.pdf p.4]", citations, chunks)

    def test_substantive_answer_without_citations_is_not_grounded(self):
        """The §11.4 post-check: prose we cannot tie to a source must not be
        presented as sourced."""
        chunks = [chunk("c1", "Content.")]

        answer = (
            "The penalty is five thousand rupees and must be paid within thirty "
            "days of the notice being issued."
        )
        assert not is_grounded(answer, [], chunks)

    def test_nothing_retrieved_is_never_grounded(self):
        assert not is_grounded("Some answer.", [], [])
