"""Upload validation (FR-1, FR-2) and filename safety."""

from __future__ import annotations

import pytest

from app.core.errors import EmptyFileError, FileTooLargeError, UnsupportedFileTypeError
from app.ingestion.validation import FileKind, safe_filename, sniff_kind, validate_upload

PDF_BYTES = b"%PDF-1.7\n" + b"0" * 500
DOCX_BYTES = b"PK\x03\x04" + b"\x00" * 30 + b"word/document.xml" + b"\x00" * 200


class TestContentSniffing:
    def test_detects_pdf(self):
        assert sniff_kind(PDF_BYTES) == FileKind.PDF

    def test_detects_docx(self):
        assert sniff_kind(DOCX_BYTES) == FileKind.DOCX

    def test_rejects_plain_zip(self):
        """A .zip renamed to .docx must not be accepted — the Word-specific
        part has to be present."""
        assert sniff_kind(b"PK\x03\x04" + b"\x00" * 400) is None

    def test_rejects_executable(self):
        assert sniff_kind(b"\x7fELF\x02\x01\x01" + b"\x00" * 100) is None

    def test_extension_is_not_trusted(self):
        """Content wins over the filename — the client controls the name."""
        with pytest.raises(UnsupportedFileTypeError):
            validate_upload("malware.pdf", b"\x7fELF" + b"\x00" * 100)


class TestLimits:
    def test_rejects_empty_file(self):
        with pytest.raises(EmptyFileError):
            validate_upload("empty.pdf", b"")

    def test_rejects_oversized_file(self):
        from app.core.config import settings

        oversized = b"%PDF-1.7\n" + b"0" * (settings.limits.max_file_bytes + 1)
        with pytest.raises(FileTooLargeError):
            validate_upload("huge.pdf", oversized)

    def test_accepts_valid_pdf(self):
        result = validate_upload("report.pdf", PDF_BYTES)
        assert result.kind == FileKind.PDF
        assert result.mime_type == "application/pdf"
        assert len(result.sha256) == 64


class TestFilenameSafety:
    @pytest.mark.parametrize(
        "attack",
        [
            "../../etc/passwd",
            "..\\..\\windows\\system32\\config",
            "/absolute/path/file.pdf",
            "....//....//secret.pdf",
        ],
    )
    def test_strips_path_traversal(self, attack):
        result = safe_filename(attack)
        assert "/" not in result
        assert "\\" not in result
        assert not result.startswith("..")

    def test_preserves_devanagari(self):
        """Hindi filenames are the norm in this corpus and must survive."""
        assert "हिन्दी" in safe_filename("हिन्दी परिपत्र.pdf")

    def test_removes_control_characters(self):
        assert "\x00" not in safe_filename("bad\x00name.pdf")

    def test_truncates_but_keeps_extension(self):
        result = safe_filename("a" * 400 + ".pdf")
        assert len(result) <= 200
        assert result.endswith(".pdf")

    def test_empty_becomes_placeholder(self):
        assert safe_filename("") == "document"
        assert safe_filename("...") == "document"
