"""Cross-encoder reranking (FR-18).

Bi-encoders embed query and document independently, so they never see the two
together. A cross-encoder scores the *pair*, which is far more precise — and
precision is what matters once we hand only top-N chunks to the LLM. Per §8.3
this is the single biggest accuracy lever after hybrid search.

bge-reranker-v2-m3 shares bge-m3's multilingual training, so it reranks a Hindi
query against an English passage correctly rather than penalising the language
mismatch.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod

from app.core.config import settings
from app.core.devices import resolve_device, supports_fp16
from app.core.logging import get_logger
from app.retrieval.types import ScoredChunk

log = get_logger(__name__)


class Reranker(ABC):
    name: str = "base"

    @abstractmethod
    def score(self, query: str, chunks: list[ScoredChunk]) -> list[float]: ...

    def rerank(
        self,
        query: str,
        chunks: list[ScoredChunk],
        *,
        top_n: int,
        min_score: float | None = None,
    ) -> list[ScoredChunk]:
        """Re-score and return the best `top_n` chunks."""
        if not chunks:
            return []

        scores = self.score(query, chunks)

        rescored: list[ScoredChunk] = []
        for chunk, score in zip(chunks, scores, strict=True):
            chunk.rerank_score = float(score)
            chunk.retrieval_score = (
                chunk.retrieval_score if chunk.retrieval_score is not None else chunk.score
            )
            chunk.score = float(score)
            rescored.append(chunk)

        rescored.sort(key=lambda c: c.score, reverse=True)

        if min_score is not None:
            filtered = [c for c in rescored if c.score >= min_score]
            if not filtered:
                # Never return nothing on the threshold alone — the grounding
                # check in the generation layer decides "not found", not the
                # reranker. Hand it the full top-N rather than a token single
                # chunk: keeping only one starved the model of context on any
                # query the cross-encoder cannot score (document-level asks
                # score around -8 because no passage "answers" them), so it
                # emitted the insufficient-context sentinel and the user was
                # told their document did not contain its own summary.
                log.debug(
                    "rerank_threshold_filtered_all_candidates",
                    min_score=min_score,
                    best_score=round(rescored[0].score, 3),
                    candidates=len(rescored),
                )
            rescored = filtered or rescored

        for position, chunk in enumerate(rescored):
            chunk.rank = position
        return rescored[:top_n]

    def warmup(self) -> None:  # pragma: no cover
        return None


class BGEReranker(Reranker):
    name = "bge-reranker-v2-m3"

    def __init__(
        self,
        model_name: str | None = None,
        *,
        device: str | None = None,
        use_fp16: bool | None = None,
        batch_size: int | None = None,
        max_length: int | None = None,
    ) -> None:
        cfg = settings.reranker
        self.model_name = model_name or cfg.model_name
        self.device = resolve_device(device or cfg.device)
        self.use_fp16 = supports_fp16(self.device) if use_fp16 is None else use_fp16
        self.batch_size = batch_size or cfg.batch_size
        self.max_length = max_length or cfg.max_length
        self._model = None
        self._tokenizer = None
        self._lock = threading.Lock()

    def _load(self):
        """Load the cross-encoder through plain `transformers`.

        Deliberately not `FlagEmbedding.FlagReranker`: its scoring path calls
        `tokenizer.prepare_for_model`, which transformers 5 removed from the
        slow tokenizers, so it raises `AttributeError` on every call.
        bge-reranker-v2-m3 is an ordinary sequence-classification model, so
        driving it directly is both fewer moving parts and one less dependency
        whose internals can break us.
        """
        if self._model is not None:
            return self._model, self._tokenizer

        with self._lock:
            if self._model is not None:
                return self._model, self._tokenizer

            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            log.info("loading_reranker", model=self.model_name, device=self.device)
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                dtype=torch.float16 if self.use_fp16 else torch.float32,
            )
            model.to(self.device)
            model.eval()

            self._tokenizer = tokenizer
            self._model = model
            log.info("reranker_ready", model=self.model_name, device=self.device)
            return self._model, self._tokenizer

    def score(self, query: str, chunks: list[ScoredChunk]) -> list[float]:
        import torch

        model, tokenizer = self._load()
        scores: list[float] = []

        with torch.inference_mode():
            for start in range(0, len(chunks), self.batch_size):
                batch = chunks[start : start + self.batch_size]
                encoded = tokenizer(
                    [query] * len(batch),
                    [c.text for c in batch],
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)
                # A single logit per pair: the relevance score.
                logits = model(**encoded).logits.view(-1).float()
                scores.extend(logits.cpu().tolist())

        return scores

    def warmup(self) -> None:
        # Loading weights is not enough — the first *forward pass* is what
        # triggers Metal/CUDA kernel compilation, so score a throwaway pair.
        self._load()
        from app.retrieval.types import ScoredChunk

        self.score(
            "warmup",
            [
                ScoredChunk(
                    chunk_id="warmup",
                    doc_id="",
                    text="warmup passage",
                    score=0.0,
                    filename="",
                    page_start=1,
                    page_end=1,
                    chunk_index=0,
                )
            ],
        )


class NoOpReranker(Reranker):
    """Passthrough used when reranking is disabled or unavailable.

    Keeps the retrieval-only fallback path in NFR-5 working: if the reranker
    fails to load, we still answer from hybrid-search order.
    """

    name = "none"

    def score(self, query: str, chunks: list[ScoredChunk]) -> list[float]:
        return [chunk.score for chunk in chunks]

    def rerank(
        self,
        query: str,
        chunks: list[ScoredChunk],
        *,
        top_n: int,
        min_score: float | None = None,
    ) -> list[ScoredChunk]:
        return chunks[:top_n]
