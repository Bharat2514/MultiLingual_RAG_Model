"""Text cleaning (FR-7). The regression cases here are the ones that silently
destroy retrieval quality if they break."""

from __future__ import annotations

import unicodedata

from app.ingestion.cleaning import (
    clean_document,
    dehyphenate,
    find_repeated_furniture,
    is_page_number,
    join_soft_wraps,
    normalize_unicode,
)
from app.ingestion.types import PageText


class TestUnicodeNormalization:
    def test_nfc_composes_devanagari_nukta(self):
        """The highest-impact Hindi fix: क + nukta and क़ are the same word but
        different byte sequences, so without NFC they never match."""
        decomposed = "क़"  # KA + NUKTA
        precomposed = "क़"  # QA
        assert decomposed != precomposed
        assert normalize_unicode(decomposed) == normalize_unicode(precomposed)

    def test_output_is_nfc(self):
        assert unicodedata.is_normalized("NFC", normalize_unicode("क़ fi test"))

    def test_strips_zero_width_space_but_keeps_zwnj(self):
        # ZWNJ is semantic in Devanagari conjuncts; ZWSP is extraction noise.
        assert "​" not in normalize_unicode("क​ख")
        assert "‌" in normalize_unicode("क‌ख")

    def test_expands_latin_ligatures(self):
        assert normalize_unicode("ofﬁce") == "office"

    def test_converts_nbsp_to_space(self):
        assert normalize_unicode("a b") == "a b"


class TestDehyphenation:
    def test_joins_hyphenated_line_break(self):
        assert dehyphenate("informa-\ntion") == "information"

    def test_preserves_genuine_compounds(self):
        # Capital after the break means a real compound, not a wrap artefact.
        assert "Indo-\nAryan" in dehyphenate("Indo-\nAryan")

    def test_preserves_inline_hyphen(self):
        assert dehyphenate("well-known fact") == "well-known fact"


class TestSoftWrap:
    def test_joins_wrapped_line(self):
        assert (
            join_soft_wraps("a sentence that\ncontinues here") == "a sentence that continues here"
        )

    def test_keeps_break_after_terminator(self):
        assert "\n" in join_soft_wraps("Sentence one.\nsentence two")


class TestPageNumbers:
    def test_recognises_common_forms(self):
        for line in ["4", "- 12 -", "Page 3", "Page 3 of 40", "[7]", "iv", "१२"]:
            assert is_page_number(line), line

    def test_rejects_prose(self):
        for line in ["Chapter 4 begins here", "The year 2024 was notable"]:
            assert not is_page_number(line), line


class TestFurniture:
    def _pages(self, body_template: str, count: int = 8) -> list[PageText]:
        return [
            PageText(
                page_no=i,
                text=f"ACME LTD — CONFIDENTIAL\n{body_template.format(i=i)}\nPage {i} of {count}",
            )
            for i in range(1, count + 1)
        ]

    def test_strips_repeated_header_and_footer(self):
        pages = self._pages("Unique body content for section {i} of the report.")
        cleaned = clean_document(pages)
        assert "ACME LTD" not in cleaned[0].text
        assert "Page 1 of 8" not in cleaned[0].text
        assert "Unique body content" in cleaned[0].text

    def test_does_not_delete_prose_containing_page_numbers(self):
        """Regression: digit-blind matching once made body sentences look
        identical across pages, and the whole page was deleted."""
        pages = self._pages(
            "On page {i} the committee reviewed applications from district {i} offices "
            "and recorded its detailed findings for later publication."
        )
        cleaned = clean_document(pages)
        for page in cleaned:
            assert "committee reviewed applications" in page.text

    def test_short_documents_are_left_alone(self):
        # "Repeated on 40% of pages" is meaningless on a two-page document.
        pages = [PageText(page_no=i, text=f"Header\nBody {i}") for i in (1, 2)]
        assert find_repeated_furniture(pages) == set()

    def test_hindi_furniture(self):
        pages = [
            PageText(
                page_no=i,
                text=f"भारत सरकार\nगृह मंत्रालय\nयह पृष्ठ {i} की विषयवस्तु है जिसमें नियमों का विवरण दिया गया है।\n— {i} —",
            )
            for i in range(1, 9)
        ]
        cleaned = clean_document(pages)
        assert "गृह मंत्रालय" not in cleaned[0].text
        assert "विषयवस्तु" in cleaned[0].text

    def test_page_numbering_is_preserved(self):
        cleaned = clean_document(self._pages("Body {i}."))
        assert [p.page_no for p in cleaned] == list(range(1, 9))
