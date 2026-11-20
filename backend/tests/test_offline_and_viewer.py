"""Ollama provider (offline mode), PDF highlighting, and follow-up resolution."""

from __future__ import annotations

import json

import httpx
import pytest

from app.core.errors import LLMUnavailableError
from app.generation import conversation
from app.generation.conversation import Turn
from app.generation.providers.mock import MockLLMProvider
from app.generation.providers.ollama import OllamaProvider


def _patch_transport(monkeypatch, handler) -> None:
    real = httpx.AsyncClient

    def factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", factory)


class TestOllama:
    @pytest.mark.asyncio
    async def test_generate(self, monkeypatch):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen.update(json.loads(request.content))
            return httpx.Response(
                200,
                json={
                    "model": "qwen2.5:7b-instruct",
                    "message": {"role": "assistant", "content": "छुट्टी 20 दिन है। [1]"},
                    "prompt_eval_count": 120,
                    "eval_count": 12,
                    "done": True,
                },
            )

        _patch_transport(monkeypatch, handler)
        provider = OllamaProvider(model="qwen2.5:7b-instruct", base_url="http://ollama:11434")
        response = await provider.generate("system", "user", temperature=0.0)

        assert response.text == "छुट्टी 20 दिन है। [1]"
        assert (response.input_tokens, response.output_tokens) == (120, 12)
        assert seen["messages"][0] == {"role": "system", "content": "system"}
        assert seen["options"]["temperature"] == 0.0
        assert provider.estimate_cost_usd(1000, 1000) == 0.0

    @pytest.mark.asyncio
    async def test_stream(self, monkeypatch):
        lines = [
            {"message": {"content": "Leave "}, "done": False},
            {"message": {"content": "is 20 days."}, "done": False},
            {"done": True, "prompt_eval_count": 50, "eval_count": 4},
        ]

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="\n".join(json.dumps(x) for x in lines))

        _patch_transport(monkeypatch, handler)
        chunks = [c async for c in OllamaProvider(model="m").stream("s", "u")]
        assert "".join(c.text for c in chunks) == "Leave is 20 days."
        assert chunks[-1].done and chunks[-1].usage.output_tokens == 4

    @pytest.mark.asyncio
    async def test_unreachable_server_is_llm_unavailable(self, monkeypatch):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused", request=request)

        _patch_transport(monkeypatch, handler)
        with pytest.raises(LLMUnavailableError):
            await OllamaProvider(model="m").generate("s", "u")


class TestHighlight:
    def test_locates_a_passage_on_the_text_layer(self, tmp_path):
        fitz = pytest.importorskip("fitz")
        from app.intelligence import highlight

        path = tmp_path / "policy.pdf"
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 100), "Section 4. Leave rules apply to all staff.", fontsize=11)
        page.insert_text(
            (72, 300), "A penalty of five thousand rupees applies to late submission.", fontsize=11
        )
        doc.save(path)
        doc.close()

        result = highlight.locate(
            path, 1, "A penalty of five thousand rupees applies to late submission.", ocr_page=False
        )
        assert result["method"] == "text"
        assert result["rects"]
        # Highlighted near y=300pt on a 842pt page, not on the unrelated line.
        assert all(0.3 < r["y0"] < 0.4 for r in result["rects"])

    def test_png_render(self, tmp_path):
        fitz = pytest.importorskip("fitz")
        from app.intelligence import highlight

        path = tmp_path / "one.pdf"
        doc = fitz.open()
        doc.new_page().insert_text((72, 72), "hello")
        doc.save(path)
        doc.close()
        assert highlight.render_page(path, 1, dpi=60).startswith(b"\x89PNG")


class TestFollowups:
    @pytest.mark.parametrize(
        ("query", "expected"),
        [
            ("aur uska penalty?", True),
            ("इसका अंतिम दिनांक क्या है?", True),
            ("What about the deadline?", True),
            ("What is the notice period for resignation under the HR policy?", False),
        ],
    )
    def test_detection(self, query, expected):
        assert conversation.is_followup(query) is expected

    @pytest.mark.asyncio
    async def test_falls_back_when_the_model_is_off_contract(self):
        # The mock provider answers a rewrite prompt with the sentinel.
        resolution = await conversation.resolve(
            "aur uska penalty?",
            [Turn("user", "late submission ka niyam kya hai?"), Turn("assistant", "…")],
            MockLLMProvider(),
        )
        assert resolution.method == "fallback"
        assert resolution.query.startswith("late submission ka niyam kya hai?")

    @pytest.mark.asyncio
    async def test_no_history_means_no_rewrite(self):
        resolution = await conversation.resolve("aur uska penalty?", [], MockLLMProvider())
        assert resolution.rewritten is False
