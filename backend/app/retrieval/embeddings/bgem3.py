"""bge-m3 embeddings — the cross-lingual core of the system (§8.3).

Why this model: it produces dense *and* learned-sparse vectors in a single
forward pass, and it was trained so that a Hindi sentence and its English
translation land close together in the same space. That is precisely the G3
requirement — a Hindi question retrieving an English passage.

Its learned sparse vectors also beat classical BM25 here: BM25 needs
language-specific tokenisation and stemming, and there is no good Hindi stemmer
in the usual stacks. bge-m3's lexical weights are multilingual by construction,
so exact terms (clause IDs, names, numbers) stay retrievable in both scripts.

**Implemented directly on `transformers` rather than via `FlagEmbedding`.**
FlagEmbedding's encoder calls `tokenizer.pad()` with a list, which transformers
5.x rejects — so it fails on any input long enough to take the batched path,
i.e. every real document chunk, while short strings sneak through and hide the
problem. Rather than pin the ecosystem to an old transformers, we run the model
ourselves: bge-m3 is an XLM-R encoder plus one linear head, so this is a small
amount of explicit code with no third-party layer that can break underneath us.
"""

from __future__ import annotations

import threading
from typing import Any

from app.core.config import settings
from app.core.devices import resolve_device, supports_fp16
from app.core.errors import EmbeddingUnavailableError
from app.core.logging import get_logger
from app.retrieval.embeddings.base import EmbeddingProvider
from app.retrieval.types import Embedding, SparseVector

log = get_logger(__name__)

# Lexical weights below this are noise and only bloat the sparse index.
MIN_SPARSE_WEIGHT = 1e-3

SPARSE_LINEAR_FILE = "sparse_linear.pt"


class BGEM3EmbeddingProvider(EmbeddingProvider):
    supports_sparse = True
    name = "bge-m3"

    def __init__(
        self,
        model_name: str | None = None,
        *,
        device: str | None = None,
        use_fp16: bool | None = None,
        max_length: int | None = None,
        batch_size: int | None = None,
    ) -> None:
        cfg = settings.embedding
        self.model_name = model_name or cfg.model_name
        self.device = resolve_device(device or cfg.device)
        # fp16 is a win on CUDA only; on MPS and CPU it is slower or unsupported.
        self.use_fp16 = supports_fp16(self.device) if use_fp16 is None else use_fp16
        self.max_length = max_length or cfg.max_length
        self.batch_size = batch_size or cfg.batch_size
        self.dense_dim = cfg.dense_dim

        self._model: Any = None
        self._tokenizer: Any = None
        self._sparse_linear: Any = None
        self._special_ids: set[int] = set()
        self._lock = threading.Lock()

    # --- model lifecycle -----------------------------------------------------

    def _load(self) -> None:
        """Load once, under a lock — the worker runs concurrent tasks and the
        weights are ~2.3 GB."""
        if self._model is not None:
            return

        with self._lock:
            if self._model is not None:
                return

            try:
                import torch
                from huggingface_hub import hf_hub_download
                from torch import nn
                from transformers import AutoModel, AutoTokenizer
            except ImportError as exc:  # pragma: no cover
                raise EmbeddingUnavailableError(
                    "The ML extras are not installed. Run `make install-ml`."
                ) from exc

            log.info(
                "loading_embedding_model",
                model=self.model_name,
                device=self.device,
                fp16=self.use_fp16,
            )
            try:
                tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                model = AutoModel.from_pretrained(
                    self.model_name,
                    dtype=torch.float16 if self.use_fp16 else torch.float32,
                )
                model.to(self.device)
                model.eval()

                # The sparse head is a single Linear(hidden, 1) shipped
                # alongside the backbone rather than inside it.
                sparse_linear = nn.Linear(model.config.hidden_size, 1)
                sparse_linear.load_state_dict(
                    torch.load(
                        hf_hub_download(self.model_name, SPARSE_LINEAR_FILE),
                        map_location="cpu",
                        weights_only=True,
                    )
                )
                sparse_linear.to(self.device)
                if self.use_fp16:
                    sparse_linear.half()
                sparse_linear.eval()
            except Exception as exc:
                raise EmbeddingUnavailableError(f"Could not load {self.model_name}: {exc}") from exc

            self._tokenizer = tokenizer
            self._model = model
            self._sparse_linear = sparse_linear
            # Special tokens carry lexical weight but no meaning; excluding them
            # stops the sparse index matching on <s>/</s>/<pad>.
            self._special_ids = {i for i in tokenizer.all_special_ids if i is not None}

            actual_dim = model.config.hidden_size
            if actual_dim != self.dense_dim:
                log.warning(
                    "dense_dim_mismatch_using_model_value",
                    configured=self.dense_dim,
                    actual=actual_dim,
                )
                self.dense_dim = actual_dim

            log.info("embedding_model_ready", model=self.model_name, device=self.device)

    def warmup(self) -> None:
        self._load()
        super().warmup()

    def close(self) -> None:
        self._model = self._tokenizer = self._sparse_linear = None

    # --- encoding ------------------------------------------------------------

    def _sparse_from_weights(
        self, token_ids: list[int], weights: list[float]
    ) -> SparseVector | None:
        """Collapse per-position weights into one weight per token id.

        A token appearing several times keeps its strongest activation, which is
        what bge-m3's own implementation does.
        """
        merged: dict[int, float] = {}
        for token_id, weight in zip(token_ids, weights, strict=True):
            if token_id in self._special_ids or weight <= MIN_SPARSE_WEIGHT:
                continue
            current = merged.get(token_id)
            if current is None or weight > current:
                merged[token_id] = weight

        if not merged:
            return None
        indices = sorted(merged)
        return SparseVector(indices, [round(merged[i], 5) for i in indices])

    def _encode(self, texts: list[str], *, with_sparse: bool) -> list[Embedding]:
        if not texts:
            return []

        import torch

        self._load()
        embeddings: list[Embedding] = []

        try:
            with torch.inference_mode():
                for start in range(0, len(texts), self.batch_size):
                    batch = texts[start : start + self.batch_size]
                    encoded = self._tokenizer(
                        batch,
                        padding=True,
                        truncation=True,
                        max_length=self.max_length,
                        return_tensors="pt",
                    ).to(self.device)

                    hidden = self._model(**encoded).last_hidden_state

                    # Dense: normalised CLS token, so cosine == dot product.
                    dense = torch.nn.functional.normalize(hidden[:, 0], p=2, dim=-1)
                    dense_list = dense.float().cpu().tolist()

                    if not with_sparse:
                        embeddings.extend(Embedding(dense=vector) for vector in dense_list)
                        continue

                    # relu(sparse_linear(h)) is bge-m3's lexical head: one
                    # non-negative weight per token position.
                    token_weights = (
                        torch.relu(self._sparse_linear(hidden)).squeeze(-1).float().cpu()
                    )
                    input_ids = encoded["input_ids"].cpu()
                    attention = encoded["attention_mask"].cpu()

                    for row in range(len(batch)):
                        # Padding positions carry weights too; mask them out.
                        keep = attention[row].bool()
                        embeddings.append(
                            Embedding(
                                dense=dense_list[row],
                                sparse=self._sparse_from_weights(
                                    input_ids[row][keep].tolist(),
                                    token_weights[row][keep].tolist(),
                                ),
                            )
                        )
        except Exception as exc:
            raise EmbeddingUnavailableError(f"Embedding failed: {exc}") from exc

        return embeddings

    def embed_documents(self, texts: list[str]) -> list[Embedding]:
        return self._encode(texts, with_sparse=True)

    def embed_queries(self, texts: list[str]) -> list[Embedding]:
        # bge-m3 is trained without asymmetric instruction prefixes, so queries
        # and documents are encoded identically.
        return self._encode(texts, with_sparse=True)
