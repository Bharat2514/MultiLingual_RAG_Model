"""Structured output from the LLM, validated before use.

Every feature built on this treats the model as a *reader*: it is handed
numbered passages and asked to return JSON that points back into them by
passage number and quotes them verbatim. `verify_quote` then checks the quote
really is in the cited passage — output that cannot be traced to the source is
marked, not trusted.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import regex

from app.core.errors import InvalidModelOutputError
from app.core.logging import get_logger
from app.generation.prompts import format_context
from app.generation.providers.base import LLMProvider
from app.intelligence.pii import redact
from app.intelligence.textutil import contains_quote
from app.retrieval.types import ScoredChunk

log = get_logger(__name__)

_FENCE_RE = regex.compile(r"```(?:json)?\s*(.*?)```", regex.DOTALL)

JSON_RULES = """Return ONLY a JSON object — no prose, no Markdown fences.
Every value you extract must come from the numbered CONTEXT passages. For each,
give `source_index` (the passage's number: 2 for [S2]) and `quote`: a short span copied
VERBATIM from that passage that states it. Never use outside knowledge. If the
passages do not contain something, use null rather than guessing."""


@dataclass(slots=True)
class JSONResult:
    data: dict
    input_tokens: int
    output_tokens: int
    model: str


def _extract_object(text: str) -> dict:
    candidates = [m.group(1) for m in _FENCE_RE.finditer(text)] or [text]
    for candidate in candidates:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start == -1 or end <= start:
            continue
        try:
            value = json.loads(candidate[start : end + 1])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise InvalidModelOutputError(details={"preview": text[:200]})


def build_context(chunks: list[ScoredChunk], *, redact_pii: bool) -> str:
    context = format_context(chunks)
    return redact(context)[0] if redact_pii else context


async def generate_json(
    provider: LLMProvider, system: str, user: str, *, max_tokens: int = 2048
) -> JSONResult:
    response = await provider.generate(system, user, max_tokens=max_tokens, temperature=0.0)
    data = _extract_object(response.text or "")
    return JSONResult(data, response.input_tokens, response.output_tokens, response.model)


def resolve_source(index: object, chunks: list[ScoredChunk]) -> ScoredChunk | None:
    if isinstance(index, str):
        # "S2" / "[S2]" — the passage's label rather than its bare number.
        index = index.strip().strip("[]").strip().lstrip("Ss")
    try:
        position = int(index)  # type: ignore[call-overload]
    except (TypeError, ValueError):
        return None
    return chunks[position - 1] if 1 <= position <= len(chunks) else None


def verify_quote(quote: object, chunk: ScoredChunk | None) -> bool:
    return bool(chunk is not None and isinstance(quote, str) and contains_quote(chunk.text, quote))


def source_ref(chunk: ScoredChunk | None, quote: object = None) -> dict | None:
    """A citation-shaped dict for API responses."""
    if chunk is None:
        return None
    return {
        "doc_id": chunk.doc_id,
        "filename": chunk.filename,
        "page": chunk.page_start,
        "chunk_id": chunk.chunk_id,
        "quote": quote if isinstance(quote, str) else None,
        "quote_verified": verify_quote(quote, chunk),
    }
