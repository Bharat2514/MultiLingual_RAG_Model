"""End-to-end ingestion over real generated PDF/DOCX files.

Fixtures are built with the same libraries that parse them, so these exercise
the actual parser code paths rather than hand-written `PageText` objects.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.core.errors import ParseFailedError
from app.db.models import JobStage
from app.ingestion.parsers import docx_parser, pdf
from app.ingestion.pipeline import run_pipeline
from app.ingestion.validation import FileKind

BODY = (
    "This circular sets out the revised procedure for district offices. "
    "All officers must comply with the stated requirements. "
    "Non-compliance will be recorded in the annual review. "
    "The rules take effect from the first of next month. "
)


@pytest.fixture
def text_pdf(tmp_path: Path) -> Path:
    import fitz

    doc = fitz.open()
    for page_index in range(3):
        page = doc.new_page()
        page.insert_textbox(
            fitz.Rect(50, 50, 550, 750),
            f"Section {page_index + 1}\n" + BODY * 4,
            fontsize=10,
        )
    path = tmp_path / "text.pdf"
    doc.save(path)
    doc.close()
    return path


@pytest.fixture
def scanned_like_pdf(tmp_path: Path) -> Path:
    """One page with a text layer, two without — the mixed case that makes
    per-page OCR routing necessary (§11.1)."""
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    page.insert_textbox(fitz.Rect(50, 50, 550, 750), BODY * 4, fontsize=10)
    doc.new_page()
    doc.new_page()
    path = tmp_path / "scanned.pdf"
    doc.save(path)
    doc.close()
    return path


@pytest.fixture
def sample_docx(tmp_path: Path) -> Path:
    import docx

    document = docx.Document()
    document.add_heading("Revised Procedure", level=1)
    document.add_paragraph(BODY * 3)

    table = document.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Clause"
    table.cell(0, 1).text = "Description"
    table.cell(1, 0).text = "4.2"
    table.cell(1, 1).text = "Applications must be filed within thirty days."

    document.add_page_break()
    document.add_paragraph(BODY * 3)

    path = tmp_path / "sample.docx"
    document.save(path)
    return path


class TestPdfPipeline:
    def test_produces_chunks_with_provenance(self, text_pdf):
        result = run_pipeline(text_pdf, FileKind.PDF)

        assert result.page_count == 3
        assert result.chunks
        assert result.language_primary == "en"
        assert not result.ocr_used
        for chunk in result.chunks:
            assert 1 <= chunk.page_start <= chunk.page_end <= 3
            assert chunk.char_start < chunk.char_end
            assert chunk.token_count > 0

    def test_reports_stage_timings(self, text_pdf):
        result = run_pipeline(text_pdf, FileKind.PDF)

        assert {"parse", "clean", "chunk"} <= set(result.stage_timings_ms)
        assert all(v >= 0 for v in result.stage_timings_ms.values())

    def test_emits_stage_callbacks(self, text_pdf):
        seen: list[tuple[JobStage, str]] = []
        run_pipeline(text_pdf, FileKind.PDF, on_stage=lambda s, st, m: seen.append((s, st)))

        stages = {stage for stage, _ in seen}
        assert {JobStage.PARSE, JobStage.CLEAN, JobStage.CHUNK} <= stages
        assert any(status == "succeeded" for _, status in seen)


class TestOcrRouting:
    def test_flags_only_the_pages_without_text(self, scanned_like_pdf):
        parsed = pdf.extract_text_layer(scanned_like_pdf)

        assert pdf.pages_needing_ocr(parsed) == [2, 3]

    def test_text_pages_are_not_routed_to_ocr(self, text_pdf):
        parsed = pdf.extract_text_layer(text_pdf)

        assert pdf.pages_needing_ocr(parsed) == []

    def test_degrades_gracefully_when_tesseract_is_missing(self, scanned_like_pdf, monkeypatch):
        """A missing OCR engine must not fail the document — the pages that do
        have a text layer are still worth indexing (NFR-5)."""
        monkeypatch.setattr("app.ingestion.ocr.is_available", lambda: False)

        seen: list[tuple[JobStage, str]] = []
        result = run_pipeline(
            scanned_like_pdf, FileKind.PDF, on_stage=lambda s, st, m: seen.append((s, st))
        )

        assert result.chunks
        assert (JobStage.OCR, "skipped") in seen

    def test_renders_a_page_to_png(self, text_pdf):
        image = pdf.render_page_png(text_pdf, 1, dpi=120)

        assert image.startswith(b"\x89PNG")
        assert len(image) > 1000

    def test_empty_document_raises_a_clear_error(self, tmp_path, monkeypatch):
        import fitz

        monkeypatch.setattr("app.ingestion.ocr.is_available", lambda: False)
        doc = fitz.open()
        doc.new_page()
        path = tmp_path / "blank.pdf"
        doc.save(path)
        doc.close()

        with pytest.raises(ParseFailedError):
            run_pipeline(path, FileKind.PDF)


class TestDocxPipeline:
    def test_extracts_text_tables_and_page_breaks(self, sample_docx):
        result = run_pipeline(sample_docx, FileKind.DOCX)

        assert result.page_count >= 2  # the explicit page break split it
        combined = " ".join(c.text for c in result.chunks)
        assert "Revised Procedure" in combined
        assert "within thirty days" in combined  # table content survived

    def test_table_rows_are_pipe_delimited(self, sample_docx):
        parsed = docx_parser.parse(sample_docx)

        assert any("Clause | Description" in p.text for p in parsed.pages)
