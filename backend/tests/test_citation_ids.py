"""The passage-ID citation contract and the forms that still resolve around it.

The reported failure: a single document named `2601.20882v1.pdf`, and Gemini
citing `[5.pdf p.22]` — passage 5's number glued to the extension. Nothing
resolved, so a fully sourced answer was shown as "Unverified" with "No
citation" on every sentence.
"""

from __future__ import annotations

from app.generation.citations import (
    canonicalize_citations,
    extract_citations,
    is_grounded,
    strip_citation_markers,
)
from app.intelligence.claims import UNCITED, verify_answer
from app.intelligence.llm_json import resolve_source
from app.retrieval.types import ScoredChunk

ARXIV = "2601.20882v1.pdf"


def chunk(index: int, *, filename: str = ARXIV, page: int | None = None) -> ScoredChunk:
    page = page if page is not None else index * 4
    return ScoredChunk(
        chunk_id=f"c{index}",
        doc_id="doc1",
        text=f"Monitoring tracks CPU, memory and disk usage in passage {index}.",
        score=0.5,
        filename=filename,
        page_start=page,
        page_end=page,
        chunk_index=index,
    )


def context(n: int = 6, **kwargs) -> list[ScoredChunk]:
    return [chunk(i, **kwargs) for i in range(1, n + 1)]


class TestPassageIds:
    def test_single_id_resolves(self):
        chunks = context()
        citations = extract_citations("Monitoring tracks CPU usage [S2].", chunks)
        assert [(c.chunk_id, c.context_index, c.page) for c in citations] == [("c2", 2, 8)]

    def test_grouped_ids_resolve(self):
        chunks = context()
        citations = extract_citations("Both are tracked [S1, S3].", chunks)
        assert sorted(c.context_index for c in citations) == [1, 3]

    def test_lowercase_and_spaced_ids(self):
        chunks = context()
        assert len(extract_citations("A [s2]. B [S 4].", chunks)) == 2

    def test_id_outside_the_context_is_fabricated(self):
        chunks = context(3)
        assert extract_citations("Made up [S9].", chunks) == []

    def test_ids_take_precedence_over_bare_numbers(self):
        """A bare [3] next to real IDs is a footnote or list number, not a citation."""
        chunks = context()
        citations = extract_citations("Rule [3] applies [S1].", chunks)
        assert [c.context_index for c in citations] == [1]

    def test_ids_work_with_any_filename(self):
        names = ["योजना दिशानिर्देश 2024.pdf", "scan (1) [final].pdf", "a, b.pdf"]
        chunks = [chunk(i, filename=name) for i, name in enumerate(names, start=1)]
        citations = extract_citations("X [S1]. Y [S2]. Z [S3].", chunks)
        assert {c.filename for c in citations} == set(names)


class TestPassageNumberFusedWithExtension:
    """`[5.pdf p.22]`: when no document is called `5.pdf`, it is passage 5."""

    REPORTED = (
        "Monitoring mein hum ye sab karte hain:\n\n"
        "*   **System behavior observe karna:** Server ke network requests aur "
        "resource usage ko monitor kiya jata hai [5.pdf p.22].\n"
        "*   **Specific metrics track karna:** CPU, memory aur disk usage [2.pdf p.3].\n"
        "*   **Dono:** logs aur metrics saath mein dekhe jaate hain [2.pdf p.3, 5.pdf p.22]."
    )

    def reported_context(self) -> list[ScoredChunk]:
        chunks = context()
        chunks[1].page_start = chunks[1].page_end = 3  # passage 2 is page 3
        chunks[4].page_start = chunks[4].page_end = 22  # passage 5 is page 22
        return chunks

    def test_the_reported_answer_is_grounded(self):
        chunks = self.reported_context()
        citations = extract_citations(self.REPORTED, chunks)
        assert {(c.context_index, c.page) for c in citations} == {(2, 3), (5, 22)}
        assert is_grounded(self.REPORTED, citations, chunks)

    def test_several_citations_in_one_bracket(self):
        chunks = self.reported_context()
        citations = extract_citations("Both [2.pdf p.3, 5.pdf p.22].", chunks)
        assert sorted(c.context_index for c in citations) == [2, 5]

    def test_semicolon_separated(self):
        chunks = self.reported_context()
        citations = extract_citations("Both [2.pdf p.3; 5.pdf p.22].", chunks)
        assert sorted(c.context_index for c in citations) == [2, 5]

    def test_page_outside_the_passage_falls_back_to_its_start(self):
        chunks = self.reported_context()
        [citation] = extract_citations("A claim [5.pdf p.99].", chunks)
        assert (citation.context_index, citation.page) == (5, 22)

    def test_a_real_file_named_5_pdf_wins(self):
        chunks = context()
        chunks[0].filename = "5.pdf"
        chunks[0].page_start = chunks[0].page_end = 1
        [citation] = extract_citations("A claim [5.pdf p.1].", chunks)
        assert citation.context_index == 1

    def test_not_mistaken_for_a_shortened_filename(self):
        """`5.pdf` must not suffix-match `report15.pdf` when passage 5 exists."""
        chunks = context()
        chunks[0].filename = "report15.pdf"
        [citation] = extract_citations("A claim [5.pdf p.20].", chunks)
        assert citation.context_index == 5

    def test_number_beyond_the_context_is_dropped(self):
        assert extract_citations("A claim [9.pdf p.3].", context(3)) == []

    def test_every_claim_now_has_a_citation(self):
        chunks = self.reported_context()
        answer = canonicalize_citations(self.REPORTED, chunks)
        report = verify_answer(answer, chunks)
        cited = [c for c in report.claims if c.text.startswith(("System", "Specific", "Dono"))]
        assert cited and all(c.verdict != UNCITED for c in cited)


class TestOtherForms:
    def test_filename_form_still_resolves(self):
        chunks = context()
        [citation] = extract_citations(f"A claim [{ARXIV} p.12].", chunks)
        assert citation.context_index == 3

    def test_filename_with_comma_and_page_word(self):
        chunks = context()
        [citation] = extract_citations(f"A claim [{ARXIV}, page 12].", chunks)
        assert citation.context_index == 3

    def test_filename_containing_a_comma_is_not_split(self):
        chunks = context(2, filename="a, b.pdf")
        [citation] = extract_citations("A claim [a, b.pdf p.8].", chunks)
        assert citation.context_index == 2

    def test_bare_numbers_still_resolve_on_their_own(self):
        chunks = context()
        assert len(extract_citations("A [1]. B [2, 3].", chunks)) == 3

    def test_ordinary_brackets_are_ignored(self):
        assert extract_citations("Rule [sic] and [Rule 3(a)] apply.", context()) == []


class TestCanonicalize:
    def test_every_form_becomes_a_passage_id(self):
        chunks = context()
        chunks[1].page_start = chunks[1].page_end = 3
        answer = f"A [5.pdf p.20]. B [2.pdf p.3, S4]. C [{ARXIV} p.12]. D [S6]."
        assert canonicalize_citations(answer, chunks) == "A [S5]. B [S2, S4]. C [S3]. D [S6]."

    def test_unresolved_entries_are_dropped_from_a_group(self):
        assert canonicalize_citations("A [S1, S9].", context(3)) == "A [S1]."

    def test_unresolved_brackets_are_left_alone(self):
        text = "Rule [sic] says [nowhere.pdf p.4]."
        assert canonicalize_citations(text, context()) == text

    def test_canonical_answer_resolves_to_the_same_citations(self):
        chunks = context()
        answer = "A [5.pdf p.20]. B [2.pdf p.8, 4.pdf p.16]."
        before = {c.chunk_id for c in extract_citations(answer, chunks)}
        after = {
            c.chunk_id for c in extract_citations(canonicalize_citations(answer, chunks), chunks)
        }
        assert before == after == {"c2", "c4", "c5"}

    def test_ids_are_stripped_for_the_grounding_check(self):
        assert strip_citation_markers("Fee is 500. [S1, S4]") == "Fee is 500."


class TestStructuredOutputSources:
    """The JSON tools (eligibility, FAQ, compare) cite by `source_index`."""

    def test_accepts_number_or_label(self):
        chunks = context(3)
        assert resolve_source(2, chunks) is chunks[1]
        assert resolve_source("S2", chunks) is chunks[1]
        assert resolve_source("[S3]", chunks) is chunks[2]
        assert resolve_source("S9", chunks) is None
