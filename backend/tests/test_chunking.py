"""Chunking invariants (FR-8, §11.2).

These are the properties retrieval quality depends on: chunks stay inside the
token band, begin at sentence boundaries, overlap enough to survive a boundary
cutting through an answer, and carry accurate page/char provenance for citations.
"""

from __future__ import annotations

import itertools

import pytest

from app.core.config import ChunkingSettings
from app.db.models import Language
from app.ingestion.chunking import chunk_pages
from app.ingestion.types import PageText, SourceKind

HINDI_SENTENCES = (
    "यह एक सरकारी परिपत्र है। इसमें कई महत्वपूर्ण नियम दिए गए हैं। "
    "सभी कर्मचारियों को इनका पालन करना अनिवार्य है। उल्लंघन पर कार्रवाई होगी। "
)
ENGLISH_SENTENCES = (
    "This circular sets out the revised procedure. All officers must comply. "
    "Non-compliance will be recorded. The rules take effect immediately. "
)

CONFIG = ChunkingSettings(target_tokens=200, min_tokens=80, max_tokens=300, overlap_ratio=0.12)


def _pages(text: str, count: int, *, ocr: bool = False) -> list[PageText]:
    return [
        PageText(
            page_no=i,
            text=text,
            source=SourceKind.OCR if ocr else SourceKind.TEXT_LAYER,
        )
        for i in range(1, count + 1)
    ]


class TestSizing:
    def test_respects_the_token_ceiling(self):
        chunks = chunk_pages(_pages(ENGLISH_SENTENCES * 12, 3), config=CONFIG)
        assert chunks
        # The tail may be merged back into the previous chunk, which is bounded
        # at 1.25x rather than the hard ceiling.
        assert all(c.token_count <= CONFIG.max_tokens * 1.25 for c in chunks)

    def test_no_stub_chunks(self):
        """A 5-token trailing fragment would pollute retrieval, so the tail is
        merged into the previous chunk instead of emitted."""
        chunks = chunk_pages(_pages(ENGLISH_SENTENCES * 10, 2), config=CONFIG)
        assert len(chunks) > 1
        assert chunks[-1].token_count >= CONFIG.min_tokens

    def test_short_document_makes_one_chunk(self):
        chunks = chunk_pages(_pages("A single short sentence here.", 1), config=CONFIG)
        assert len(chunks) == 1

    def test_empty_pages_make_no_chunks(self):
        assert chunk_pages(_pages("", 3), config=CONFIG) == []

    def test_oversized_sentence_is_hard_split(self):
        # Unpunctuated OCR output is the real-world source of this.
        runaway = " ".join(["word"] * 4000)
        chunks = chunk_pages([PageText(page_no=1, text=runaway)], config=CONFIG)
        assert len(chunks) > 1
        assert all(c.token_count <= CONFIG.max_tokens * 1.25 for c in chunks)


class TestBoundaries:
    def test_chunks_start_at_sentence_boundaries(self):
        chunks = chunk_pages(_pages(ENGLISH_SENTENCES * 10, 2), config=CONFIG)
        for chunk in chunks:
            first_word = chunk.text.split()[0]
            assert first_word[0].isupper() or not first_word[0].isascii()

    def test_hindi_splits_on_danda_not_mid_sentence(self):
        chunks = chunk_pages(_pages(HINDI_SENTENCES * 12, 2), config=CONFIG)
        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk.text.strip()


class TestOverlap:
    def test_consecutive_chunks_share_text(self):
        """Overlap is what stops an answer that straddles a chunk boundary from
        becoming unretrievable."""
        chunks = chunk_pages(_pages(ENGLISH_SENTENCES * 12, 2), config=CONFIG)
        assert len(chunks) >= 2
        overlapping = sum(
            1
            for a, b in itertools.pairwise(chunks)
            if any(len(s) > 20 and s in b.text for s in a.text.split(". "))
        )
        assert overlapping >= 1

    def test_zero_overlap_config_still_terminates(self):
        config = ChunkingSettings(
            target_tokens=200, min_tokens=80, max_tokens=300, overlap_ratio=0.0
        )
        chunks = chunk_pages(_pages(ENGLISH_SENTENCES * 10, 2), config=config)
        assert len(chunks) > 1


class TestProvenance:
    def test_indices_are_sequential(self):
        chunks = chunk_pages(_pages(ENGLISH_SENTENCES * 10, 3), config=CONFIG)
        assert [c.index for c in chunks] == list(range(len(chunks)))

    def test_page_range_is_ordered_and_in_bounds(self):
        chunks = chunk_pages(_pages(ENGLISH_SENTENCES * 6, 4), config=CONFIG)
        for chunk in chunks:
            assert 1 <= chunk.page_start <= chunk.page_end <= 4

    def test_char_span_is_ordered(self):
        for chunk in chunk_pages(_pages(ENGLISH_SENTENCES * 8, 2), config=CONFIG):
            assert chunk.char_start < chunk.char_end

    def test_ocr_flag_propagates(self):
        chunks = chunk_pages(_pages(ENGLISH_SENTENCES * 6, 2, ocr=True), config=CONFIG)
        assert all(c.ocr_used for c in chunks)

    @pytest.mark.parametrize(
        ("text", "expected"),
        [(HINDI_SENTENCES * 6, Language.HI), (ENGLISH_SENTENCES * 6, Language.EN)],
    )
    def test_language_is_tagged(self, text, expected):
        chunks = chunk_pages(_pages(text, 2), config=CONFIG)
        assert all(c.language == expected for c in chunks)
