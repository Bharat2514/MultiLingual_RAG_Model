#!/usr/bin/env python
"""Verify the configured LLM provider end to end.

Confirms the credential works, lists the generation-capable models the key can
actually reach, and does one round-trip in Hindi + Hinglish so we know the
provider handles the registers PolyDoc targets (FR-21) before Batch 4 depends
on it.

    python scripts/check_llm.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.core.config import GEMINI_FALLBACK_ORDER, settings

GREEN, RED, YELLOW, BLUE, DIM, RESET = (
    "\033[0;32m",
    "\033[0;31m",
    "\033[0;33m",
    "\033[0;34m",
    "\033[2m",
    "\033[0m",
)


def ok(msg: str) -> None:
    print(f"{GREEN}✓{RESET} {msg}")


def fail(msg: str) -> None:
    print(f"{RED}✗{RESET} {msg}")


def warn(msg: str) -> None:
    print(f"{YELLOW}!{RESET} {msg}")


def info(msg: str) -> None:
    print(f"{BLUE}→{RESET} {msg}")


# Prompts that exercise the two registers most likely to break: Devanagari
# output and romanized code-mixed Hinglish.
PROBES = [
    ("Hindi", "एक वाक्य में उत्तर दें: भारत की राजधानी क्या है?"),
    ("Hinglish", "Ek line mein Hinglish me batao: RAG ka matlab kya hota hai?"),
]


def _is_quota_error(exc: Exception) -> bool:
    text = str(exc)
    return "RESOURCE_EXHAUSTED" in text or "429" in text


def check_gemini() -> int:
    from google import genai
    from google.genai import types

    api_key = settings.google_api_key
    if not api_key:
        fail("GOOGLE_API_KEY is empty in .env")
        print(f"  {DIM}Get one at https://aistudio.google.com/apikey{RESET}")
        return 1

    # Same API version as the app's provider, so this check matches production.
    client = genai.Client(
        api_key=api_key, http_options=types.HttpOptions(api_version="v1beta")
    )

    info("Listing models your key can reach...")
    try:
        available = [
            m.name.removeprefix("models/")
            for m in client.models.list()
            if "generateContent" in (m.supported_actions or [])
        ]
    except Exception as exc:  # noqa: BLE001
        fail(f"Could not list models: {type(exc).__name__}: {exc}")
        return 1

    if not available:
        fail("Key is valid but exposes no generateContent models.")
        return 1

    ok(f"{len(available)} generation models available")
    preferred = [m for m in available if m.startswith("gemini-") and "embedding" not in m]
    for name in sorted(preferred)[:12]:
        print(f"    {DIM}{name}{RESET}")

    target = settings.llm.resolved_model
    if target in available:
        ok(f"Configured model is listed: {target}")
    else:
        fail(f"Configured model '{target}' is NOT in your key's model list.")
        target = ""

    # A model appearing in list() only means it exists — not that your key has
    # quota for it. Free-tier keys get `limit: 0` on the Pro models, so probe
    # the real thing and fall through to a model that actually answers.
    candidates = [m for m in ([target] if target else []) if m]
    candidates += [m for m in GEMINI_FALLBACK_ORDER if m in available and m != target]

    working: str | None = None
    for candidate in candidates:
        try:
            client.models.generate_content(
                model=candidate,
                contents="Reply with the single word: OK",
                config=types.GenerateContentConfig(max_output_tokens=2000),
            )
        except Exception as exc:  # noqa: BLE001
            if _is_quota_error(exc):
                warn(f"{candidate}: no quota on this key — trying the next model")
                continue
            fail(f"{candidate}: {type(exc).__name__}: {str(exc)[:160]}")
            continue
        working = candidate
        break

    if working is None:
        fail("No Gemini model on this key has usable quota.")
        print(
            f"  {DIM}Enable billing: https://aistudio.google.com/apikey → project → Billing{RESET}"
        )
        return 1

    ok(f"Model with usable quota: {working}")
    if working != settings.llm.resolved_model:
        warn(f"Set LLM__MODEL={working} in .env (configured default is unusable)")

    for label, prompt in PROBES:
        try:
            response = client.models.generate_content(
                model=working,
                contents=prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=2000,
                    temperature=settings.llm.temperature,
                ),
            )
            text = (response.text or "").strip().replace("\n", " ")
            if not text:
                warn(f"{label} round-trip returned no text (finish={response.candidates})")
                continue
            ok(f"{label} round-trip: {text[:110]}")
        except Exception as exc:  # noqa: BLE001
            fail(f"{label} round-trip failed: {type(exc).__name__}: {str(exc)[:160]}")
            return 1

    return 0


def check_anthropic() -> int:
    import anthropic

    if not settings.anthropic_api_key:
        fail("ANTHROPIC_API_KEY is empty in .env")
        return 1

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    for label, prompt in PROBES:
        try:
            message = client.messages.create(
                model=settings.llm.resolved_model,
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
            )
            text = next((b.text for b in message.content if b.type == "text"), "")
            ok(f"{label} round-trip: {text.strip()[:110]}")
        except Exception as exc:  # noqa: BLE001
            fail(f"{label} round-trip failed: {type(exc).__name__}: {exc}")
            return 1
    return 0


def main() -> int:
    provider = settings.llm.provider
    print(f"\n{BLUE}PolyDoc — LLM connectivity check{RESET}")
    print(f"  provider : {provider}")
    print(f"  model    : {settings.llm.resolved_model or '(provider default)'}")
    print(f"  key env  : {settings.llm.api_key_env or '(none)'}\n")

    if provider == "mock":
        ok("Mock provider selected — nothing to verify.")
        return 0

    code = check_gemini() if provider == "gemini" else check_anthropic()
    print()
    if code == 0:
        ok("LLM provider is ready.")
    else:
        fail("LLM provider is not usable yet — fix the above, then re-run.")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
