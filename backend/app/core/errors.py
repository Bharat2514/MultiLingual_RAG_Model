"""Single error vocabulary for the whole API (§9.4).

Every failure the user can observe is raised as a ``PolyDocError`` subclass and
rendered by the handlers in ``app.main`` as::

    {"request_id": "...", "error": {"code": "...", "message": "...", "details": {...}}}
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any


class ErrorCode(StrEnum):
    UNSUPPORTED_FILE_TYPE = "UNSUPPORTED_FILE_TYPE"
    FILE_TOO_LARGE = "FILE_TOO_LARGE"
    TOO_MANY_PAGES = "TOO_MANY_PAGES"
    TOO_MANY_FILES = "TOO_MANY_FILES"
    EMPTY_FILE = "EMPTY_FILE"
    PARSE_FAILED = "PARSE_FAILED"
    OCR_FAILED = "OCR_FAILED"
    DOCUMENT_NOT_FOUND = "DOCUMENT_NOT_FOUND"
    DOCUMENT_NOT_READY = "DOCUMENT_NOT_READY"
    NO_RELEVANT_CONTEXT = "NO_RELEVANT_CONTEXT"
    LLM_UNAVAILABLE = "LLM_UNAVAILABLE"
    EMBEDDING_UNAVAILABLE = "EMBEDDING_UNAVAILABLE"
    VECTOR_STORE_UNAVAILABLE = "VECTOR_STORE_UNAVAILABLE"
    RATE_LIMITED = "RATE_LIMITED"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    NOT_FOUND = "NOT_FOUND"
    METHOD_NOT_ALLOWED = "METHOD_NOT_ALLOWED"
    UNAUTHORIZED = "UNAUTHORIZED"
    FORBIDDEN = "FORBIDDEN"
    INTERNAL = "INTERNAL"


class PolyDocError(Exception):
    """Base class for every user-visible failure."""

    code: ErrorCode = ErrorCode.INTERNAL
    status_code: int = 500
    message: str = "An unexpected error occurred."

    def __init__(
        self,
        message: str | None = None,
        *,
        details: dict[str, Any] | None = None,
        status_code: int | None = None,
    ) -> None:
        self.message = message or self.__class__.message
        self.details = details or {}
        if status_code is not None:
            self.status_code = status_code
        super().__init__(self.message)

    def to_payload(self, request_id: str) -> dict[str, Any]:
        error: dict[str, Any] = {"code": str(self.code), "message": self.message}
        if self.details:
            error["details"] = self.details
        return {"request_id": request_id, "error": error}


# --- 4xx ---------------------------------------------------------------------


class UnsupportedFileTypeError(PolyDocError):
    code = ErrorCode.UNSUPPORTED_FILE_TYPE
    status_code = 415
    message = "Only PDF and DOCX are supported."


class FileTooLargeError(PolyDocError):
    code = ErrorCode.FILE_TOO_LARGE
    status_code = 413
    message = "File exceeds the maximum allowed size."


class TooManyPagesError(PolyDocError):
    code = ErrorCode.TOO_MANY_PAGES
    status_code = 413
    message = "Document exceeds the maximum allowed page count."


class TooManyFilesError(PolyDocError):
    code = ErrorCode.TOO_MANY_FILES
    status_code = 429
    message = "Too many documents in this session."


class EmptyFileError(PolyDocError):
    code = ErrorCode.EMPTY_FILE
    status_code = 400
    message = "The uploaded file is empty."


class DocumentNotFoundError(PolyDocError):
    code = ErrorCode.DOCUMENT_NOT_FOUND
    status_code = 404
    message = "Document not found."


class DocumentNotReadyError(PolyDocError):
    code = ErrorCode.DOCUMENT_NOT_READY
    status_code = 409
    message = "Document is still being processed."


class ValidationError(PolyDocError):
    code = ErrorCode.VALIDATION_ERROR
    status_code = 422
    message = "Request validation failed."


class UnauthorizedError(PolyDocError):
    code = ErrorCode.UNAUTHORIZED
    status_code = 401
    message = "Authentication required."


class ForbiddenError(PolyDocError):
    code = ErrorCode.FORBIDDEN
    status_code = 403
    message = "You do not have access to this resource."


class RateLimitedError(PolyDocError):
    code = ErrorCode.RATE_LIMITED
    status_code = 429
    message = "Too many requests. Please retry shortly."


# --- 5xx / pipeline ----------------------------------------------------------


class ParseFailedError(PolyDocError):
    code = ErrorCode.PARSE_FAILED
    status_code = 422
    message = "Could not extract text from this document."


class OCRFailedError(PolyDocError):
    code = ErrorCode.OCR_FAILED
    status_code = 422
    message = "OCR failed for this document."


class NoRelevantContextError(PolyDocError):
    """Not an HTTP error in the chat path — the answer layer converts this into
    a grounded 'not found in your documents' response (FR-23)."""

    code = ErrorCode.NO_RELEVANT_CONTEXT
    status_code = 200
    message = "No relevant content found in the selected documents."


class LLMUnavailableError(PolyDocError):
    code = ErrorCode.LLM_UNAVAILABLE
    status_code = 503
    message = "The language model is temporarily unavailable."


class EmbeddingUnavailableError(PolyDocError):
    code = ErrorCode.EMBEDDING_UNAVAILABLE
    status_code = 503
    message = "The embedding model is temporarily unavailable."


class VectorStoreUnavailableError(PolyDocError):
    code = ErrorCode.VECTOR_STORE_UNAVAILABLE
    status_code = 503
    message = "The vector store is temporarily unavailable."


class InternalError(PolyDocError):
    code = ErrorCode.INTERNAL
    status_code = 500
    message = "An unexpected error occurred."
