"""Request models for the document-intelligence endpoints.

Responses are returned as plain JSON objects documented on each route: their
shapes come straight from the `app.intelligence` modules, and duplicating every
nested field here would only create a second place for them to drift.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field


class EligibilityCheckRequest(BaseModel):
    profile: dict[str, str | float | int | bool | None] = Field(
        description="Applicant facts, e.g. {'age': 34, 'annual_income': 180000, 'is_farmer': true}."
    )
    doc_ids: list[str] | None = Field(
        default=None, description="Schemes to check against. Null checks every ready document."
    )
    extract_missing: bool = Field(
        default=True,
        description="Extract rules for documents that have none cached yet (one LLM call each).",
    )


class CalculateRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    inputs: dict[str, float] = Field(
        default_factory=dict, description="Values for the formula's variables, by name."
    )
    doc_ids: list[str] | None = None


class CompareRequest(BaseModel):
    doc_ids: list[str] = Field(min_length=2, max_length=5)
    attributes: list[str] = Field(min_length=1, max_length=10)


class DiffRequest(BaseModel):
    old_doc_id: str
    new_doc_id: str


class ConflictRequest(BaseModel):
    doc_ids: list[str] | None = Field(
        default=None, max_length=10, description="Null checks every pair of ready documents."
    )


class TimelineUpdate(BaseModel):
    effective_date: date | None = None
    superseded_by: str | None = None
    note: str | None = Field(default=None, max_length=500)


class TranslateRequest(BaseModel):
    target: Literal["hi", "en"]
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=4, ge=1, le=8)


class FeedbackRequest(BaseModel):
    request_id: str = Field(min_length=1, max_length=64)
    rating: Literal[-1, 1]
    comment: str | None = Field(default=None, max_length=2000)
