"""The grounding contract (§11.4, FR-20 – FR-23).

Everything that keeps answers trustworthy lives here:

* answer **only** from the supplied context;
* cite every claim as ``[filename p.N]`` so it can be verified (US3);
* when the context does not contain the answer, say so explicitly instead of
  inventing one (FR-23, US4) — this is the behaviour the §12.3 anti-hallucination
  target is measured against;
* reply in the user's own language and register (FR-21).

The insufficient-context signal is a fixed sentinel rather than free prose. A
model asked to "say you couldn't find it" says it a hundred different ways in
three languages, which is impossible to detect reliably; a sentinel is exact,
and we substitute the localised wording ourselves.
"""

from __future__ import annotations

from app.db.models import Language
from app.generation.intent import QueryIntent
from app.retrieval.types import ScoredChunk

#: Emitted by the model when the context cannot answer the question.
INSUFFICIENT_CONTEXT = "INSUFFICIENT_CONTEXT"

# The user-facing "not found" text, in the register they asked in (FR-21).
NOT_FOUND_MESSAGES: dict[str, str] = {
    Language.HI: "मुझे यह जानकारी आपके दस्तावेज़ों में नहीं मिली।",
    Language.HINGLISH: "Mujhe ye jaankari aapke documents mein nahi mili.",
    Language.EN: "I couldn't find this in your documents.",
    Language.MIXED: "मुझे यह जानकारी आपके दस्तावेज़ों में नहीं मिली। "
    "(I couldn't find this in your documents.)",
    Language.UNKNOWN: "I couldn't find this in your documents.",
}

_REGISTER_INSTRUCTION: dict[str, str] = {
    Language.HI: ("The user wrote in Hindi (Devanagari). Reply in Hindi using Devanagari script."),
    Language.HINGLISH: (
        "The user wrote in Hinglish — romanized Hindi in Latin script, mixed with English. "
        "Reply in the same register: romanized Hindi in Latin script, keeping the English "
        "words the user would naturally use. Do NOT reply in Devanagari."
    ),
    Language.EN: "The user wrote in English. Reply in English.",
    Language.MIXED: (
        "The user mixed Hindi and English. Reply in the same mix, leaning on whichever "
        "language dominates their question."
    ),
    Language.UNKNOWN: "Reply in the same language the user used.",
}

SYSTEM_PROMPT = """You are PolyDoc, a document-grounded assistant for Hindi and English documents.

You answer strictly from the numbered CONTEXT passages given to you. You are not \
a general knowledge assistant: if the context does not contain the answer, you say so.

RULES

1. GROUNDING. {grounding}

2. CITATIONS. After each sentence or claim, cite the passage it came from using the \
source label shown for that passage, in square brackets — for example [policy.pdf p.4]. \
Cite every factual claim. If one sentence draws on two passages, cite both.

3. LANGUAGE. {register}

4. FIDELITY. Do not translate names, numbers, dates, clause identifiers, or amounts — \
reproduce them exactly as they appear in the context. If the context is in a different \
language from the user's question, translate the *meaning* into the user's language but \
keep such identifiers verbatim.

5. STYLE. {style}"""

_EXTRACTIVE_GROUNDING = """Use only the CONTEXT. Never use outside knowledge, and never \
guess. If the context does not contain enough to answer, reply with exactly this token and \
nothing else:
{sentinel}"""

_EXTRACTIVE_STYLE = """Be direct and concise. Answer the question asked; do not summarise \
the whole document. No preamble such as "Based on the context". If the context only partly \
answers the question, give what it supports and state plainly what is missing."""

# The summary variant has to relax rule 1 as well as rule 5. Told to emit the
# sentinel whenever the context is "not enough", a model handed a *sample* of a
# document reasons that it cannot see all of it and emits the sentinel — which
# is how a summary request came back as "not found in your documents". Partial
# coverage is the normal case here, so it must be stated as such.
_SUMMARY_GROUNDING = """Use only the CONTEXT. Never use outside knowledge, and never \
guess or extrapolate beyond what the passages say. The passages are a representative \
selection taken across the document(s) in reading order, so they may not be the complete \
text — that is expected, and is NOT a reason to refuse. Summarise what you were given. \
Only if the CONTEXT is empty or entirely unreadable, reply with exactly this token and \
nothing else:
{sentinel}"""

_SUMMARY_STYLE = """The user asked for a summary or overview, so summarising is exactly \
what is wanted here. Open with one or two sentences saying what the document is and what \
it covers, then give the main points as a short bulleted list — aim for 5 to 8 bullets, \
each one a substantive point rather than a heading. Cite every bullet. No preamble such as \
"Based on the context". Do not pad, and do not claim to cover what you cannot see: if the \
passages plainly skip parts of the document, say in one closing line what your summary is \
based on."""


def build_system_prompt(
    language: Language | str, intent: QueryIntent = QueryIntent.EXTRACTIVE
) -> str:
    register = _REGISTER_INSTRUCTION.get(str(language), _REGISTER_INSTRUCTION[Language.UNKNOWN])
    summary = intent == QueryIntent.SUMMARY
    grounding = (_SUMMARY_GROUNDING if summary else _EXTRACTIVE_GROUNDING).format(
        sentinel=INSUFFICIENT_CONTEXT
    )
    return SYSTEM_PROMPT.format(
        grounding=grounding,
        register=register,
        style=_SUMMARY_STYLE if summary else _EXTRACTIVE_STYLE,
    )


def format_context(chunks: list[ScoredChunk]) -> str:
    """Render retrieved chunks as numbered, source-labelled passages (FR-20)."""
    blocks: list[str] = []
    for position, chunk in enumerate(chunks, start=1):
        blocks.append(f"[{position}] SOURCE: {chunk.citation_label}\n{chunk.text.strip()}")
    return "\n\n".join(blocks)


def build_user_prompt(
    query: str, chunks: list[ScoredChunk], intent: QueryIntent = QueryIntent.EXTRACTIVE
) -> str:
    if intent == QueryIntent.SUMMARY:
        closing = (
            "The passages above are ordered as they appear in the document(s). "
            "Summarise them as instructed, citing each point."
        )
    else:
        closing = "Answer using only the context above, citing sources as instructed."

    return (
        f"CONTEXT\n{'=' * 60}\n{format_context(chunks)}\n{'=' * 60}\n\n"
        f"QUESTION: {query}\n\n{closing}"
    )


def not_found_message(language: Language | str) -> str:
    return NOT_FOUND_MESSAGES.get(str(language), NOT_FOUND_MESSAGES[Language.UNKNOWN])


def is_insufficient(answer: str) -> bool:
    """Whether the model signalled that the context could not answer.

    Matched loosely: models occasionally wrap the sentinel in punctuation or a
    short apology despite the instruction.
    """
    return INSUFFICIENT_CONTEXT in (answer or "").upper()
