"""Response envelopes shared across every endpoint (§9.4, FR-32)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ErrorDetail(BaseModel):
    code: str = Field(description="Stable machine-readable error code.")
    message: str = Field(description="Human-readable explanation.")
    details: dict[str, Any] | None = Field(default=None, description="Optional structured context.")


class ErrorResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "request_id": "req_789",
                "error": {
                    "code": "UNSUPPORTED_FILE_TYPE",
                    "message": "Only PDF and DOCX are supported.",
                },
            }
        }
    )

    request_id: str
    error: ErrorDetail


class LatencyBreakdown(BaseModel):
    """Per-stage timings in milliseconds (NFR-2)."""

    retrieval: float = 0.0
    rerank: float = 0.0
    generation: float = 0.0
    total: float = 0.0


class TokenUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


class ComponentHealth(BaseModel):
    name: str
    status: Literal["ok", "degraded", "down", "disabled"]
    detail: str | None = None
    latency_ms: float | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "down"]
    version: str
    environment: str
    request_id: str
    components: list[ComponentHealth] = Field(default_factory=list)
