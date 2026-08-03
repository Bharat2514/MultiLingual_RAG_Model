"""Upload validation (FR-1, FR-2).

Content is sniffed from magic bytes rather than trusted from the filename or
the client-supplied MIME type — both are attacker-controlled and neither
survives contact with real browsers.
"""

from __future__ import annotations

import hashlib
import unicodedata
from dataclasses import dataclass
from enum import StrEnum
from pathlib import PurePath

import regex

from app.core.config import settings
from app.core.errors import EmptyFileError, FileTooLargeError, UnsupportedFileTypeError


class FileKind(StrEnum):
    PDF = "pdf"
    DOCX = "docx"


MIME_TYPES: dict[FileKind, str] = {
    FileKind.PDF: "application/pdf",
    FileKind.DOCX: "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}

EXTENSIONS: dict[FileKind, str] = {FileKind.PDF: ".pdf", FileKind.DOCX: ".docx"}

_PDF_MAGIC = b"%PDF-"
_ZIP_MAGIC = (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")

_UNSAFE_CHARS_RE = regex.compile(r"[^\p{L}\p{M}\p{N}._ -]+")
_WS_RUN_RE = regex.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class ValidatedUpload:
    filename: str
    kind: FileKind
    mime_type: str
    size_bytes: int
    sha256: str

    @property
    def extension(self) -> str:
        return EXTENSIONS[self.kind]


def sniff_kind(content: bytes) -> FileKind | None:
    """Identify the file from its magic bytes."""
    if content.startswith(_PDF_MAGIC):
        return FileKind.PDF
    if content.startswith(_ZIP_MAGIC):
        # DOCX is a ZIP container; require the Word-specific part so a random
        # .zip (or an .xlsx) is not accepted as a document.
        head = content[:8192]
        if b"word/" in head or b"word/document.xml" in content[:65536]:
            return FileKind.DOCX
        return None
    return None


def safe_filename(filename: str, *, max_length: int = 200) -> str:
    """Sanitise a client-supplied filename for storage and display.

    Devanagari filenames are common in this corpus, so we keep all Unicode
    letters and marks and strip only path separators and control characters.
    """
    name = unicodedata.normalize("NFC", filename or "")
    # Defeat traversal: keep the final component only, for both separators.
    name = PurePath(name.replace("\\", "/")).name
    name = _UNSAFE_CHARS_RE.sub("_", name)
    name = _WS_RUN_RE.sub(" ", name).strip(" ._")

    if not name:
        name = "document"

    if len(name) > max_length:
        stem, dot, suffix = name.rpartition(".")
        if dot and len(suffix) <= 10:
            name = stem[: max_length - len(suffix) - 1] + "." + suffix
        else:
            name = name[:max_length]
    return name


def validate_upload(filename: str, content: bytes) -> ValidatedUpload:
    """Validate size, emptiness, and true content type."""
    size = len(content)
    if size == 0:
        raise EmptyFileError(details={"filename": filename})

    if size > settings.limits.max_file_bytes:
        raise FileTooLargeError(
            f"File is {size / 1024 / 1024:.1f} MB; the limit is {settings.limits.max_file_mb} MB.",
            details={"size_bytes": size, "limit_bytes": settings.limits.max_file_bytes},
        )

    kind = sniff_kind(content)
    if kind is None:
        suffix = PurePath(filename).suffix.lower() or "unknown"
        raise UnsupportedFileTypeError(
            "Only PDF and DOCX are supported.",
            details={"filename": filename, "detected_extension": suffix},
        )

    return ValidatedUpload(
        filename=safe_filename(filename),
        kind=kind,
        mime_type=MIME_TYPES[kind],
        size_bytes=size,
        sha256=hashlib.sha256(content).hexdigest(),
    )
