"""Answer cache keyed on the question *and the state of the corpus*.

A repeated question is common (several people asking about the same circular,
a user re-asking after a page reload) and every repeat costs an embedding pass,
a rerank and an LLM call. The cache key includes a fingerprint of the session's
ready documents — their ids and index timestamps — so uploading, re-indexing or
deleting a document changes the key and stale answers are never served. No
explicit invalidation is needed.

Keys use a normalised question (case, whitespace, Devanagari digits), not
embedding similarity: "semantic" caching would return the answer to a
*different* question that happens to embed nearby — for "notice period for
resignation" vs "notice period for termination" that is a wrong answer.

In-process and bounded; a multi-worker deployment would move this to Redis.
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections import OrderedDict
from typing import Any

from app.core.config import settings
from app.intelligence.textutil import normalize


class AnswerCache:
    def __init__(self, *, max_entries: int, ttl_seconds: int) -> None:
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self._data: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key(**parts: object) -> str:
        material = "|".join(
            f"{k}={normalize(v) if isinstance(v, str) else v!r}" for k, v in sorted(parts.items())
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Any | None:
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                self.misses += 1
                return None
            stored_at, value = entry
            if time.monotonic() - stored_at > self.ttl_seconds:
                del self._data[key]
                self.misses += 1
                return None
            self._data.move_to_end(key)
            self.hits += 1
            return value

    def put(self, key: str, value: Any) -> None:
        with self._lock:
            self._data[key] = (time.monotonic(), value)
            self._data.move_to_end(key)
            while len(self._data) > self.max_entries:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self.hits = self.misses = 0


answer_cache = AnswerCache(
    max_entries=settings.cache.max_entries, ttl_seconds=settings.cache.ttl_seconds
)
