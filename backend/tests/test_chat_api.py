"""The /chat contract (§9.2, §9.3), end to end over HTTP with a stub LLM.

The mock provider obeys the prompt contract — it cites the passages it is given
and emits the insufficient-context sentinel when it has none — so these tests
exercise the real prompt assembly, citation resolution, and grounding logic
rather than a canned string.
"""

from __future__ import annotations

import json

import pytest

from app.generation.providers.mock import MockLLMProvider
from app.retrieval.types import RetrievalResult, ScoredChunk, SearchMode

pytestmark = pytest.mark.asyncio

SESSION = "sess_chat"


def _chunk(chunk_id: str, text: str, filename: str = "policy.pdf", page: int = 4) -> ScoredChunk:
    return ScoredChunk(
        chunk_id=chunk_id,
        doc_id="doc_policy",
        text=text,
        score=0.88,
        filename=filename,
        page_start=page,
        page_end=page,
        chunk_index=0,
        session_id=SESSION,
    )


class _StubRetrieval:
    """Stands in for RetrievalService so the API tests need no models."""

    def __init__(self, chunks: list[ScoredChunk]) -> None:
        self.chunks = chunks
        self.calls: list[dict] = []

    async def retrieve(self, query, scope, **kwargs):
        self.calls.append({"query": query, "scope": scope, **kwargs})
        return RetrievalResult(
            chunks=list(self.chunks),
            mode=kwargs.get("mode", SearchMode.HYBRID),
            reranked=True,
            candidates_considered=len(self.chunks),
            timings_ms={"embed": 5.0, "retrieval": 12.0, "rerank": 30.0},
        )


@pytest.fixture
def with_context(monkeypatch):
    retrieval = _StubRetrieval(
        [
            _chunk("c1", "A penalty of five thousand rupees applies to late submissions."),
            _chunk("c2", "Applications must be filed within thirty days.", page=5),
        ]
    )
    _patch(monkeypatch, retrieval, MockLLMProvider())
    return retrieval


@pytest.fixture
def without_context(monkeypatch):
    retrieval = _StubRetrieval([])
    _patch(monkeypatch, retrieval, MockLLMProvider())
    return retrieval


def _patch(monkeypatch, retrieval, provider):
    from app.generation import service as service_module

    original = service_module.ChatService.__init__

    def patched(self, **_kwargs):
        original(self, retrieval=retrieval, provider=provider)

    monkeypatch.setattr(service_module.ChatService, "__init__", patched)


def _post(client, query: str, **extra):
    return client.post(
        "/api/v1/chat",
        json={"query": query, **extra},
        headers={"X-Session-ID": SESSION},
    )


class TestAnswering:
    async def test_returns_a_grounded_answer_with_citations(self, client, with_context):
        response = await _post(client, "What is the penalty for late submission?")

        assert response.status_code == 200
        body = response.json()
        assert body["grounded"] is True
        assert body["citations"]
        assert body["citations"][0]["filename"] == "policy.pdf"
        assert body["citations"][0]["snippet"]
        assert body["language_detected"] == "en"

    async def test_reports_usage_and_latency(self, client, with_context):
        body = (await _post(client, "What is the penalty?")).json()

        assert body["usage"]["input_tokens"] > 0
        assert body["usage"]["output_tokens"] > 0
        assert body["latency_ms"]["total"] > 0
        assert body["latency_ms"]["retrieval"] > 0

    async def test_echoes_retrieval_diagnostics(self, client, with_context):
        body = (await _post(client, "What is the penalty?")).json()

        assert body["reranked"] is True
        assert body["candidates_considered"] == 2
        assert body["retrieval_mode"] == "hybrid"

    async def test_scopes_retrieval_to_the_session(self, client, with_context):
        await _post(client, "What is the penalty?")

        assert with_context.calls[0]["scope"].session_id == SESSION

    async def test_doc_ids_are_passed_through(self, client, with_context):
        await _post(client, "What is the penalty?", doc_ids=["doc_policy"])

        assert with_context.calls[0]["scope"].doc_ids == ["doc_policy"]


class TestAntiHallucination:
    """FR-23 / US4 — the §12.3 target this is measured against."""

    async def test_says_not_found_when_nothing_is_retrieved(self, client, without_context):
        response = await _post(client, "Who won the 1998 world cup?")

        assert response.status_code == 200  # a valid outcome, not an error
        body = response.json()
        assert body["grounded"] is False
        assert body["citations"] == []
        assert "couldn't find" in body["answer"]

    async def test_does_not_call_the_llm_when_there_is_no_context(
        self, client, monkeypatch, without_context
    ):
        """Answering "not found" is decided by us, not the model — which also
        avoids paying for a question we know cannot be answered."""
        calls: list[int] = []

        async def counting_generate(self, system, user, **kwargs):
            calls.append(1)
            raise AssertionError("the LLM must not be called without context")

        monkeypatch.setattr(MockLLMProvider, "generate", counting_generate)

        response = await _post(client, "Anything at all?")

        assert response.status_code == 200
        assert calls == []

    async def test_not_found_is_localised_to_the_query_language(self, client, without_context):
        hindi = (await _post(client, "पेंशन के लिए कितने साल चाहिए?")).json()
        hinglish = (await _post(client, "mujhe iska summary batao kya hai")).json()

        assert "नहीं मिली" in hindi["answer"]
        assert hindi["language_detected"] == "hi"
        assert "nahi mili" in hinglish["answer"]
        assert hinglish["language_detected"] == "hinglish"


class TestLanguageHandling:
    async def test_hinglish_query_searches_both_variants(self, client, with_context):
        await _post(client, "late submission par kitna jurmana lagta hai")

        variants = with_context.calls[0]["queries"]
        assert len(variants) == 2
        assert any("जुर्माना" in v for v in variants)

    async def test_english_query_searches_one_variant(self, client, with_context):
        await _post(client, "What is the penalty for late submission?")

        assert len(with_context.calls[0]["queries"]) == 1

    async def test_response_reports_the_variants_used(self, client, with_context):
        body = (await _post(client, "mujhe jurmana ke bare mein batao")).json()

        assert len(body["query_variants"]) == 2


class TestStreaming:
    async def test_streams_tokens_then_a_done_event(self, client, with_context):
        async with client.stream(
            "POST",
            "/api/v1/chat",
            json={"query": "What is the penalty?", "stream": True},
            headers={"X-Session-ID": SESSION},
        ) as response:
            assert response.status_code == 200
            assert "text/event-stream" in response.headers["content-type"]
            body = "".join([chunk async for chunk in response.aiter_text()])

        assert "event: start" in body
        assert "event: token" in body
        assert "event: done" in body

        final = json.loads(body.split("event: done\ndata: ", 1)[1].split("\n\n", 1)[0])
        assert final["grounded"] is True
        assert final["citations"]

    async def test_stream_never_leaks_the_sentinel(self, client, monkeypatch, without_context):
        """The raw INSUFFICIENT_CONTEXT marker must never flash in the UI before
        being replaced by the localised message."""
        async with client.stream(
            "POST",
            "/api/v1/chat",
            json={"query": "Something unrelated?", "stream": True},
            headers={"X-Session-ID": SESSION},
        ) as response:
            body = "".join([chunk async for chunk in response.aiter_text()])

        token_events = [line for line in body.splitlines() if line.startswith("data:")]
        assert not any("INSUFFICIENT_CONTEXT" in line for line in token_events)
        assert "couldn't find" in body


class TestValidation:
    async def test_rejects_an_empty_query(self, client, with_context):
        response = await _post(client, "")

        assert response.status_code == 422
        assert response.json()["error"]["code"] == "VALIDATION_ERROR"
