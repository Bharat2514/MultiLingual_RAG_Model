"""Original-file storage behind one interface (§8.2, NFR-8).

Local filesystem for development, S3-compatible object storage for production.
Keys are content-addressed and namespaced by session so a hard delete is a
single prefix operation and cross-session paths cannot collide (NFR-6).
"""

from __future__ import annotations

import shutil
from abc import ABC, abstractmethod
from pathlib import Path

from app.core.config import settings
from app.core.errors import InternalError
from app.core.logging import get_logger

log = get_logger(__name__)


def build_key(session_id: str, doc_id: str, extension: str) -> str:
    return f"{session_id}/{doc_id}{extension}"


class ObjectStore(ABC):
    @abstractmethod
    def put(self, key: str, content: bytes) -> str: ...

    @abstractmethod
    def get(self, key: str) -> bytes: ...

    @abstractmethod
    def local_path(self, key: str) -> Path:
        """A real filesystem path. Parsers and Tesseract need a file, so remote
        backends materialise into a temp file here."""

    @abstractmethod
    def delete(self, key: str) -> None: ...

    @abstractmethod
    def delete_prefix(self, prefix: str) -> int: ...

    @abstractmethod
    def exists(self, key: str) -> bool: ...


class LocalObjectStore(ObjectStore):
    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root or settings.storage.local_dir).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _resolve(self, key: str) -> Path:
        path = (self.root / key).resolve()
        # Reject traversal even though keys are server-generated — this is the
        # last line of defence for the isolation guarantee.
        if not path.is_relative_to(self.root):
            raise InternalError(f"Refusing to access path outside the store: {key}")
        return path

    def put(self, key: str, content: bytes) -> str:
        path = self._resolve(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write to a sibling temp file then rename, so a crash mid-write never
        # leaves a truncated document that parses as valid.
        temp = path.with_suffix(path.suffix + ".part")
        temp.write_bytes(content)
        temp.replace(path)
        log.debug("stored_object", key=key, bytes=len(content))
        return key

    def get(self, key: str) -> bytes:
        return self._resolve(key).read_bytes()

    def local_path(self, key: str) -> Path:
        return self._resolve(key)

    def delete(self, key: str) -> None:
        path = self._resolve(key)
        path.unlink(missing_ok=True)

    def delete_prefix(self, prefix: str) -> int:
        base = self._resolve(prefix)
        if not base.exists():
            return 0
        if base.is_dir():
            count = sum(1 for _ in base.rglob("*") if _.is_file())
            shutil.rmtree(base, ignore_errors=True)
            return count
        base.unlink(missing_ok=True)
        return 1

    def exists(self, key: str) -> bool:
        return self._resolve(key).is_file()


def get_object_store() -> ObjectStore:
    if settings.storage.backend == "local":
        return LocalObjectStore()
    # S3 lands with the production rollout (§18); the interface above is what
    # the rest of the codebase depends on.
    raise InternalError(f"Storage backend '{settings.storage.backend}' is not implemented yet.")
