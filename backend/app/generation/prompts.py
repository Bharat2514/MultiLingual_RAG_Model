"""The grounding contract (§11.4, FR-20 – FR-23).

Everything that keeps answers trustworthy lives here:

* answer **only** from the supplied context;
* cite every claim by passage ID — ``[S2]`` — so it can be verified (US3); the
  server maps IDs to file and page, see :mod:`app.generation.citations`;
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
from app.generation.citations import passage_id
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

2. CITATIONS. Every CONTEXT passage starts with an ID such as [S1] or [S2]. After each \
sentence or claim, cite the passage it came from by that ID in square brackets — for \
example [S2]. If one sentence draws on two passages, cite both: [S1, S3]. Cite every \
factual claim, including each bullet point. Use ONLY these IDs — never write a file \
name or page number in a citation.

3. LANGUAGE. {register}

4. FIDELITY. Do not translate names, numbers, dates, clause identifiers, or amounts — \
reproduce them exactly as they appear in the context. If the context is in a different \
language from the user's question, translate the *meaning* into the user's language but \
keep such identifiers verbatim.

5. STYLE. {style}{plain}"""

# Plain-language mode: official Hindi and English are dense with bureaucratic
# register ("अर्हकारी सेवा", "notwithstanding anything contained herein"). This
# rewrites the *wording* for a general reader without loosening any rule above.
_PLAIN_LANGUAGE = """

6. PLAIN LANGUAGE. Explain as you would to someone with no legal or administrative \
background — a farmer, a student, a first-time applicant. Use short sentences and \
everyday words; replace jargon with its plain meaning (you may keep the official term \
in brackets once). Keep every number, date and amount exact, and keep citing."""

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
    language: Language | str,
    intent: QueryIntent = QueryIntent.EXTRACTIVE,
    *,
    style: str = "default",
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
        plain=_PLAIN_LANGUAGE if style == "simple" else "",
    )


def format_context(chunks: list[ScoredChunk], *, redact_pii: bool = False) -> str:
    """Render retrieved chunks as ID-labelled passages (FR-20).

    Each passage opens with ``[S<n>]`` — the token the model cites — followed by
    its file and page in a form deliberately unlike a citation, so the model
    knows what it is reading without being tempted to copy the filename.

    With ``redact_pii`` the passage text — never the label, which the citation
    contract depends on — has Aadhaar, PAN, phone and similar identifiers
    replaced by typed placeholders before it leaves the machine.
    """
    from app.intelligence.pii import redact

    blocks: list[str] = []
    for position, chunk in enumerate(chunks, start=1):
        text = chunk.text.strip()
        if redact_pii:
            text = redact(text)[0]
        label = f"[{passage_id(position)}] (from: {chunk.filename}, {_page_words(chunk)})"
        blocks.append(f"{label}\n{text}")
    return "\n\n".join(blocks)


def _page_words(chunk: ScoredChunk) -> str:
    if chunk.page_end and chunk.page_end != chunk.page_start:
        return f"pages {chunk.page_start}-{chunk.page_end}"
    return f"page {chunk.page_start}"


def build_user_prompt(
    query: str,
    chunks: list[ScoredChunk],
    intent: QueryIntent = QueryIntent.EXTRACTIVE,
    *,
    history: str = "",
    redact_pii: bool = False,
) -> str:
    if intent == QueryIntent.SUMMARY:
        closing = (
            "The passages above are ordered as they appear in the document(s). "
            "Summarise them as instructed, citing each point."
        )
    else:
        closing = "Answer using only the context above, citing sources as instructed."

    conversation = f"{history}\n\n" if history else ""
    return (
        f"CONTEXT\n{'=' * 60}\n{format_context(chunks, redact_pii=redact_pii)}\n{'=' * 60}\n\n"
        f"{conversation}QUESTION: {query}\n\n{closing}"
    )


def not_found_message(language: Language | str) -> str:
    return NOT_FOUND_MESSAGES.get(str(language), NOT_FOUND_MESSAGES[Language.UNKNOWN])


# "Not found" with a reason: which documents were searched and what came
# closest. A bare refusal leaves the user guessing whether to rephrase, pick
# another document, or upload one; this tells them which.
_ABSTAIN_HINTS: dict[str, dict[str, str]] = {
    "no_documents": {
        Language.HI: "इस प्रश्न के लिए कोई तैयार दस्तावेज़ उपलब्ध नहीं है। पहले कोई दस्तावेज़ अपलोड करें।",
        Language.HINGLISH: "Is sawaal ke liye koi ready document nahi hai. Pehle ek document upload karein.",
        Language.EN: "There are no ready documents to search. Upload a document first.",
    },
    "nearest": {
        Language.HI: "{n} दस्तावेज़ खोजे गए। सबसे नज़दीकी अंश {source} में था, पर उसमें इसका उत्तर नहीं है। "
        "प्रश्न को दूसरे शब्दों में पूछें, या वह दस्तावेज़ अपलोड करें जिसमें यह विषय हो।",
        Language.HINGLISH: "{n} documents search kiye. Sabse kareeb ka hissa {source} mein tha, lekin usmein "
        "iska jawab nahi hai. Sawaal ko alag tareeke se poochein, ya woh document upload karein jismein "
        "yeh topic ho.",
        Language.EN: "Searched {n} document(s). The closest passage was in {source}, but it does not "
        "answer this. Try rephrasing, or upload a document that covers this topic.",
    },
    "filtered": {
        Language.HI: "कुछ अंश सुरक्षा जाँच में संदिग्ध निर्देशों के कारण हटाए गए।",
        Language.HINGLISH: "Kuch hisse security check mein suspicious instructions ki wajah se hata diye gaye.",
        Language.EN: "Some passages were withheld because they contained suspicious instructions.",
    },
}


def abstention_hint(
    language: Language | str,
    *,
    searched: int,
    nearest: str | None,
    filtered: int = 0,
) -> str:
    lang = str(language)
    lang = lang if lang in (Language.HI, Language.HINGLISH) else Language.EN
    if searched == 0:
        hint = _ABSTAIN_HINTS["no_documents"][lang]
    else:
        hint = _ABSTAIN_HINTS["nearest"][lang].format(n=searched, source=nearest or "—")
    if filtered:
        hint += " " + _ABSTAIN_HINTS["filtered"][lang]
    return hint


def is_insufficient(answer: str) -> bool:
    """Whether the model signalled that the context could not answer.

    Matched loosely: models occasionally wrap the sentinel in punctuation or a
    short apology despite the instruction.
    """
    return INSUFFICIENT_CONTEXT in (answer or "").upper()
