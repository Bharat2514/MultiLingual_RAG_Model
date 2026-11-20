"""Whole-document LLM readers: FAQ generation, comparison tables, translation.

All three share one contract: the model is given numbered passages, returns
JSON that points back into them, and every quote is verified against the source
before it reaches the user.
"""

from __future__ import annotations

from app.core.errors import InvalidModelOutputError
from app.generation.providers.base import LLMProvider
from app.intelligence import llm_json
from app.retrieval.types import ScoredChunk

_LANGUAGE_NAMES = {"hi": "Hindi (Devanagari script)", "en": "English"}

# --- FAQ ---------------------------------------------------------------------

FAQ_SYSTEM = (
    """You write the questions a reader would actually ask about a document, and
answer each strictly from it. """
    + llm_json.JSON_RULES
)

FAQ_TEMPLATE = """CONTEXT
{context}

TASK
Write {count} frequently asked questions about this document, with answers, as JSON:
{{"items": [{{"question": string, "answer": string, "source_index": number, "quote": string}}]}}
Write the questions and answers in {language}. Prefer concrete questions —
amounts, dates, eligibility, procedures, penalties — over vague ones. Each answer
is one or two sentences and must be fully supported by its quoted passage."""


async def generate_faq(
    provider: LLMProvider, chunks: list[ScoredChunk], *, language: str, count: int, redact_pii: bool
) -> dict:
    user = FAQ_TEMPLATE.format(
        context=llm_json.build_context(chunks, redact_pii=redact_pii),
        count=count,
        language=_LANGUAGE_NAMES.get(language, "the document's own language"),
    )
    result = await llm_json.generate_json(provider, FAQ_SYSTEM, user, max_tokens=3000)
    items = []
    for item in result.data.get("items") or []:
        if not isinstance(item, dict) or not item.get("question") or not item.get("answer"):
            continue
        chunk = llm_json.resolve_source(item.get("source_index"), chunks)
        items.append(
            {
                "question": str(item["question"])[:400],
                "answer": str(item["answer"])[:1200],
                "source": llm_json.source_ref(chunk, item.get("quote")),
            }
        )
    if not items:
        raise InvalidModelOutputError("The model did not return any questions.")
    return {"language": language, "items": items, "model": result.model}


# --- comparison --------------------------------------------------------------

COMPARE_SYSTEM = (
    """You fill in one row of a comparison table from a single document. """ + llm_json.JSON_RULES
)

COMPARE_TEMPLATE = """CONTEXT (all passages are from "{filename}")
{context}

TASK
For each attribute below, give its value in this document, as JSON:
{{"values": {{"<attribute>": {{"value": string | null, "source_index": number | null, "quote": string | null}}}}}}
Attributes: {attributes}
Keep each value short (a number, date, amount or phrase). Use null when the
document does not state it — an empty cell is correct, a guessed one is not."""


async def compare_row(
    provider: LLMProvider,
    filename: str,
    chunks: list[ScoredChunk],
    attributes: list[str],
    *,
    redact_pii: bool,
) -> dict:
    if not chunks:
        return {a: {"value": None, "source": None} for a in attributes}
    user = COMPARE_TEMPLATE.format(
        filename=filename,
        context=llm_json.build_context(chunks, redact_pii=redact_pii),
        attributes="; ".join(attributes),
    )
    result = await llm_json.generate_json(provider, COMPARE_SYSTEM, user)
    values = result.data.get("values") or {}
    lowered = (
        {str(k).strip().casefold(): v for k, v in values.items()}
        if isinstance(values, dict)
        else {}
    )
    row: dict[str, dict] = {}
    for attribute in attributes:
        cell = lowered.get(attribute.strip().casefold())
        if not isinstance(cell, dict) or cell.get("value") in (None, ""):
            row[attribute] = {"value": None, "source": None}
            continue
        chunk = llm_json.resolve_source(cell.get("source_index"), chunks)
        row[attribute] = {
            "value": str(cell["value"])[:300],
            "source": llm_json.source_ref(chunk, cell.get("quote")),
        }
    return row


# --- translation -------------------------------------------------------------

TRANSLATE_SYSTEM = """You translate official documents between Hindi and English.
Translate faithfully: keep names, numbers, dates, amounts, clause and section
identifiers exactly as written. Do not summarise, omit or add anything.
Return ONLY a JSON object, no Markdown fences."""

TRANSLATE_TEMPLATE = """Translate each numbered passage into {language}.
Return JSON: {{"translations": [{{"index": number, "text": string}}]}}

{passages}"""


async def translate_chunks(
    provider: LLMProvider, chunks: list[ScoredChunk], *, target: str, redact_pii: bool
) -> dict[str, str]:
    from app.intelligence.pii import redact

    passages = "\n\n".join(
        f"[{i}]\n{redact(c.text)[0] if redact_pii else c.text}"
        for i, c in enumerate(chunks, start=1)
    )
    user = TRANSLATE_TEMPLATE.format(
        language=_LANGUAGE_NAMES.get(target, target), passages=passages
    )
    result = await llm_json.generate_json(provider, TRANSLATE_SYSTEM, user, max_tokens=8000)
    out: dict[str, str] = {}
    for item in result.data.get("translations") or []:
        if not isinstance(item, dict):
            continue
        chunk = llm_json.resolve_source(item.get("index"), chunks)
        if chunk is not None and isinstance(item.get("text"), str):
            out[chunk.chunk_id] = item["text"]
    return out
