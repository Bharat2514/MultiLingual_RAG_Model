"""Transient-failure retries in the Gemini provider.

`LLM__MAX_RETRIES` was honoured only by the Anthropic provider, which forwards it
to its SDK client. Gemini has no such knob, so one 503 — which this model returns
under load often enough to matter — reached the user as an outage. These tests
pin the retry policy, including the case it must *not* retry.
"""

from __future__ import annotations

import pytest

from app.core.errors import LLMUnavailableError
from app.generation.providers import gemini as gemini_module
from app.generation.providers.gemini import GeminiProvider

TRANSIENT = "503 UNAVAILABLE. {'error': {'code': 503, 'message': 'high demand'}}"
PERMANENT = "404 NOT_FOUND. {'error': {'code': 404, 'message': 'model not found'}}"


class _Usage:
    prompt_token_count = 11
    candidates_token_count = 7


class _Response:
    def __init__(self, text: str) -> None:
        self.text = text
        self.usage_metadata = _Usage()
        self.candidates = []


class _Event:
    def __init__(self, text: str) -> None:
        self.text = text
        self.usage_metadata = _Usage()


class _Models:
    """Fails the first `fail_times` calls with `error`, then succeeds."""

    def __init__(self, *, fail_times: int, error: str, stream_pieces=("Hello", " world")) -> None:
        self.fail_times = fail_times
        self.error = error
        self.stream_pieces = stream_pieces
        self.calls = 0

    async def generate_content(self, **_kwargs):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError(self.error)
        return _Response("an answer")

    async def generate_content_stream(self, **_kwargs):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError(self.error)
        pieces = self.stream_pieces

        async def _iterate():
            for piece in pieces:
                if piece == "__boom__":
                    raise RuntimeError(TRANSIENT)
                yield _Event(piece)

        return _iterate()


class _Client:
    def __init__(self, models: _Models) -> None:
        self.aio = type("Aio", (), {"models": models})()


@pytest.fixture(autouse=True)
def _no_real_sleeping(monkeypatch):
    async def _instant(_attempt: int) -> None:
        return None

    monkeypatch.setattr(gemini_module, "_backoff", _instant)


def _provider(models: _Models, *, max_retries: int = 2) -> GeminiProvider:
    from app.core.config import settings

    provider = GeminiProvider(model="test-model", api_key="test-key")
    provider._client = _Client(models)
    settings.llm.max_retries = max_retries
    return provider


@pytest.fixture(autouse=True)
def _restore_retries():
    from app.core.config import settings

    original = settings.llm.max_retries
    yield
    settings.llm.max_retries = original


class TestGenerate:
    async def test_retries_a_transient_failure_and_succeeds(self):
        models = _Models(fail_times=1, error=TRANSIENT)
        response = await _provider(models).generate("sys", "user")
        assert response.text == "an answer"
        assert models.calls == 2

    async def test_gives_up_after_max_retries(self):
        models = _Models(fail_times=99, error=TRANSIENT)
        with pytest.raises(LLMUnavailableError, match="temporarily unavailable"):
            await _provider(models, max_retries=2).generate("sys", "user")
        assert models.calls == 3, "max_retries=2 means one attempt plus two retries"

    async def test_does_not_retry_a_permanent_failure(self):
        """A 404 on the model name will fail identically forever; retrying it
        only makes the user wait longer for the same error."""
        models = _Models(fail_times=99, error=PERMANENT)
        with pytest.raises(LLMUnavailableError, match="not available to this API key"):
            await _provider(models).generate("sys", "user")
        assert models.calls == 1

    async def test_retries_disabled_means_a_single_attempt(self):
        models = _Models(fail_times=99, error=TRANSIENT)
        with pytest.raises(LLMUnavailableError):
            await _provider(models, max_retries=0).generate("sys", "user")
        assert models.calls == 1


class TestStream:
    async def _collect(self, provider):
        return [chunk async for chunk in provider.stream("sys", "user")]

    async def test_retries_a_failure_before_the_first_token(self):
        models = _Models(fail_times=1, error=TRANSIENT)
        chunks = await self._collect(_provider(models))
        assert "".join(c.text for c in chunks) == "Hello world"
        assert models.calls == 2

    async def test_does_not_retry_once_tokens_have_reached_the_client(self):
        """Restarting the call mid-stream would replay the opening of the answer
        into the middle of what the user is already reading."""
        models = _Models(fail_times=0, error=TRANSIENT, stream_pieces=("Hel", "__boom__"))
        provider = _provider(models)
        collected = []
        with pytest.raises(LLMUnavailableError):
            async for chunk in provider.stream("sys", "user"):
                collected.append(chunk.text)
        assert collected == ["Hel"]
        assert models.calls == 1

    async def test_final_chunk_carries_usage(self):
        models = _Models(fail_times=0, error=TRANSIENT)
        chunks = await self._collect(_provider(models))
        assert chunks[-1].done
        assert chunks[-1].usage.input_tokens == 11
        assert chunks[-1].usage.output_tokens == 7
