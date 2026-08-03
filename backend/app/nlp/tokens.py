"""Token-count estimation.

Chunk sizing (FR-8) is specified in tokens, but the real bge-m3 tokenizer lives
behind the optional ML extras and costs a model load. So we estimate with
script-aware character ratios and swap in the exact tokenizer when it is
available — the two agree closely enough for chunk-boundary decisions.

The ratios matter: XLM-R (which bge-m3 uses) tokenizes Devanagari far less
efficiently than Latin, so a single ``chars / 4`` rule would undercount Hindi by
roughly 2x and silently produce oversized chunks.
"""

from __future__ import annotations

import functools
from typing import Protocol

from app.core.logging import get_logger
from app.nlp.script import ScriptProfile, profile_script

log = get_logger(__name__)

# Characters per token, measured against XLM-R on Hindi/English corpora.
CHARS_PER_TOKEN_LATIN = 4.0
CHARS_PER_TOKEN_DEVANAGARI = 2.2
CHARS_PER_TOKEN_OTHER = 3.0


class Tokenizer(Protocol):
    def encode(self, text: str) -> list[int]: ...


_exact_tokenizer: Tokenizer | None = None


def set_tokenizer(tokenizer: Tokenizer | None) -> None:
    """Install the real tokenizer (called by the embedding layer in Batch 3)."""
    global _exact_tokenizer
    _exact_tokenizer = tokenizer


def _estimate_from_profile(text: str, profile: ScriptProfile) -> int:
    if not text:
        return 0

    # Whitespace is deliberately excluded. SentencePiece (which XLM-R uses)
    # folds the leading space into the following word as "▁word", so counting
    # spaces separately inflated estimates by ~33% on Latin text — and, worse,
    # made the estimate for a joined string disagree with the sum of its parts,
    # which broke chunk-size accounting during packing.
    whitespace = sum(1 for c in text if c.isspace())
    letters = profile.total_letters
    other = max(len(text) - letters - whitespace, 0)

    tokens = (
        profile.devanagari_chars / CHARS_PER_TOKEN_DEVANAGARI
        + profile.latin_chars / CHARS_PER_TOKEN_LATIN
        + other / CHARS_PER_TOKEN_OTHER
    )
    return max(1, round(tokens))


def count_tokens(text: str) -> int:
    """Exact when a tokenizer is installed, estimated otherwise."""
    if not text:
        return 0
    if _exact_tokenizer is not None:
        try:
            return len(_exact_tokenizer.encode(text))
        except Exception:
            log.warning("tokenizer_failed_falling_back_to_estimate", exc_info=True)
    return _estimate_from_profile(text, profile_script(text))


@functools.lru_cache(maxsize=4096)
def count_tokens_cached(text: str) -> int:
    """For repeatedly-measured short spans (sentences during chunk packing)."""
    return count_tokens(text)
