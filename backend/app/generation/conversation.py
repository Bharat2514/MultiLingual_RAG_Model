"""Multi-turn follow-ups: rewrite "aur uska penalty?" into a standalone question.

Retrieval is stateless — it embeds the text it is given. A follow-up like
"aur uska penalty kitna hai?" carries no topic of its own, so searching for it
literally returns whatever mentions "penalty" most, which is rarely the clause
the conversation was about. The fix is to resolve the reference *before*
retrieval: rewrite the follow-up into the question the user actually means, in
the user's own language, and search for that.

Only follow-ups are rewritten. A self-contained question goes through untouched
— an unnecessary rewrite costs an LLM round trip and risks drifting the query.
If the model is unavailable or answers off-contract, a deterministic fallback
prefixes the previous question, which keeps the topic words in the search.
"""

from __future__ import annotations

from dataclasses import dataclass

import regex

from app.core.config import settings
from app.core.logging import get_logger
from app.generation.prompts import INSUFFICIENT_CONTEXT
from app.generation.providers.base import LLMProvider

log = get_logger(__name__)

# Pronouns and connectives that only make sense relative to an earlier turn,
# in English, Devanagari Hindi and romanized Hinglish.
_ANAPHORA_RE = regex.compile(
    r"(?i)(?:^|\b)(?:it|its|that|this|those|these|they|them|their|he|she|his|her|"
    r"what about|how about|and|also|same|above|previous|"
    r"iska|uska|iske|uske|iski|uski|isme|usme|ismein|usmein|ye|yeh|wo|woh|aur|bhi|wahi|"
    r"aisa|waisa)\b|(?:इसका|उसका|इसके|उसके|इसकी|उसकी|इसमें|उसमें|यह|वह|ये|वो|और|भी|वही)"
)

REWRITE_SYSTEM = """You rewrite a follow-up question into a standalone question.
Use the conversation only to resolve what the follow-up refers to. Keep the
user's language and script exactly (Hindi stays Devanagari, Hinglish stays
romanized). Do not answer the question. Output only the rewritten question on
one line."""


@dataclass(slots=True)
class Turn:
    role: str
    content: str


@dataclass(slots=True)
class Resolution:
    query: str
    rewritten: bool
    method: str  # none | llm | fallback


def is_followup(query: str) -> bool:
    text = query.strip()
    if not text:
        return False
    return len(text.split()) <= 4 or bool(_ANAPHORA_RE.search(text))


def _transcript(history: list[Turn], limit_chars: int = 1800) -> str:
    lines: list[str] = []
    for turn in history[-settings.conversation.max_history_turns :]:
        speaker = "User" if turn.role == "user" else "Assistant"
        content = " ".join(turn.content.split())
        lines.append(f"{speaker}: {content[:500]}")
    return "\n".join(lines)[-limit_chars:]


def _fallback(query: str, history: list[Turn]) -> str:
    previous = next((t.content for t in reversed(history) if t.role == "user"), "")
    return f"{previous.strip()} — {query.strip()}" if previous else query


async def resolve(query: str, history: list[Turn], provider: LLMProvider) -> Resolution:
    if not history or not settings.conversation.rewrite_followups or not is_followup(query):
        return Resolution(query, False, "none")

    user = f"CONVERSATION\n{_transcript(history)}\n\nFOLLOW-UP: {query}\n\nSTANDALONE QUESTION:"
    try:
        response = await provider.generate(REWRITE_SYSTEM, user, max_tokens=200, temperature=0.0)
        text = (
            (response.text or "").strip().splitlines()[0].strip().strip('"')
            if response.text
            else ""
        )
    except Exception:
        log.warning("followup_rewrite_failed", exc_info=True)
        text = ""

    if not text or INSUFFICIENT_CONTEXT in text.upper() or len(text) > 4 * len(query) + 300:
        return Resolution(_fallback(query, history), True, "fallback")
    return Resolution(text, True, "llm")


def history_block(history: list[Turn]) -> str:
    """Prior turns for the answer prompt — context for wording, never a source."""
    if not history:
        return ""
    return (
        "CONVERSATION SO FAR (for reference only — it is NOT a source; cite only CONTEXT)\n"
        + _transcript(history, limit_chars=1200)
    )
