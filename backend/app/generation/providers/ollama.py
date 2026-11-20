"""Local LLM through Ollama — the fully offline deployment (PRD OQ1).

With this provider nothing leaves the machine: embeddings and reranking are
already local (bge-m3), so swapping generation to a local model makes the whole
pipeline air-gapped. That is the requirement for most government and legal
buyers, and it is why PII redaction is skipped for this provider — there is no
third party to protect the data from.

Talks to Ollama's native `/api/chat` over plain HTTP, so no SDK is needed.
Multilingual instruction-tuned models (Qwen 2.5, Llama 3.x, Gemma) handle Hindi
and Hinglish; quality is lower than a frontier API model, which is exactly what
the eval harness is for — run `make eval-full` with `LLM__PROVIDER=ollama`.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx

from app.core.config import settings
from app.core.errors import LLMUnavailableError
from app.core.logging import get_logger
from app.generation.providers.base import LLMProvider, LLMResponse, StreamChunk

log = get_logger(__name__)


class OllamaProvider(LLMProvider):
    name = "ollama"

    def __init__(self, model: str | None = None, base_url: str | None = None) -> None:
        self.model = model or settings.llm.resolved_model
        self.base_url = (base_url or settings.llm.ollama_base_url).rstrip("/")
        self._timeout = httpx.Timeout(settings.llm.timeout_seconds, connect=5.0)

    def _payload(
        self,
        system: str,
        user: str,
        *,
        stream: bool,
        max_tokens: int | None,
        temperature: float | None,
    ) -> dict:
        return {
            "model": self.model,
            "stream": stream,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "options": {
                "temperature": settings.llm.temperature if temperature is None else temperature,
                "num_predict": max_tokens or settings.llm.max_tokens,
            },
        }

    def estimate_cost_usd(self, input_tokens: int, output_tokens: int) -> float:
        # Local inference has no per-token price.
        return 0.0

    async def generate(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        payload = self._payload(
            system, user, stream=False, max_tokens=max_tokens, temperature=temperature
        )
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(f"{self.base_url}/api/chat", json=payload)
                response.raise_for_status()
                data = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise LLMUnavailableError(
                f"Ollama at {self.base_url} is unreachable or failed: {exc}",
                details={"model": self.model},
            ) from exc

        return LLMResponse(
            text=(data.get("message") or {}).get("content", ""),
            model=data.get("model", self.model),
            input_tokens=int(data.get("prompt_eval_count") or 0),
            output_tokens=int(data.get("eval_count") or 0),
            finish_reason=data.get("done_reason"),
        )

    async def stream(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[StreamChunk]:
        payload = self._payload(
            system, user, stream=True, max_tokens=max_tokens, temperature=temperature
        )
        try:
            async with (
                httpx.AsyncClient(timeout=self._timeout) as client,
                client.stream("POST", f"{self.base_url}/api/chat", json=payload) as response,
            ):
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    if data.get("done"):
                        yield StreamChunk(
                            text="",
                            done=True,
                            usage=LLMResponse(
                                text="",
                                model=data.get("model", self.model),
                                input_tokens=int(data.get("prompt_eval_count") or 0),
                                output_tokens=int(data.get("eval_count") or 0),
                                finish_reason=data.get("done_reason"),
                            ),
                        )
                        return
                    text = (data.get("message") or {}).get("content", "")
                    if text:
                        yield StreamChunk(text=text)
        except (httpx.HTTPError, ValueError) as exc:
            raise LLMUnavailableError(
                f"Ollama at {self.base_url} is unreachable or failed: {exc}",
                details={"model": self.model},
            ) from exc
