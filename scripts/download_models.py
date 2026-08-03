#!/usr/bin/env python
"""Pre-fetch the retrieval models into the configured cache.

Run this before the first query (and during a Docker build) so a user request
never blocks on a multi-gigabyte download.

Both repositories ship artefacts we never load — an ONNX copy of the weights,
README images, benchmark plots. Downloading the whole snapshot pulls ~5.2 GB;
filtering to what the PyTorch path actually needs pulls ~2.3 GB.

    python scripts/download_models.py [--reranker-only|--embedder-only]
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.core.config import settings

GREEN, RED, BLUE, DIM, RESET = "\033[0;32m", "\033[0;31m", "\033[0;34m", "\033[2m", "\033[0m"

# Everything the PyTorch loader ignores. `onnx/` alone is 2.27 GB on bge-m3.
IGNORE = [
    "onnx/*",
    "imgs/*",
    "assets/*",
    "*.png",
    "*.jpg",
    "*.jpeg",
    "*.gif",
    "*.h5",
    "*.msgpack",
    "*.ot",
]


def human(num_bytes: int) -> str:
    return f"{num_bytes / 1e9:.2f} GB" if num_bytes >= 1e9 else f"{num_bytes / 1e6:.0f} MB"


def fetch(repo_id: str) -> bool:
    from huggingface_hub import snapshot_download

    print(f"{BLUE}→{RESET} {repo_id}")
    started = time.time()
    try:
        path = snapshot_download(
            repo_id=repo_id,
            cache_dir=str(settings.embedding.cache_dir / "hub"),
            ignore_patterns=IGNORE,
            max_workers=4,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"{RED}✗{RESET} {repo_id}: {type(exc).__name__}: {exc}")
        return False

    # HF stores real bytes in blobs/ and symlinks them into snapshots/; count
    # only the blobs so the reported size matches actual disk usage.
    size = sum(
        f.stat().st_size for f in Path(path).rglob("*") if f.is_file() and not f.is_symlink()
    )
    print(f"{GREEN}✓{RESET} {repo_id} — {human(size)} in {time.time() - started:.0f}s")
    print(f"  {DIM}{path}{RESET}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedder-only", action="store_true")
    parser.add_argument("--reranker-only", action="store_true")
    args = parser.parse_args()

    repos: list[str] = []
    if not args.reranker_only:
        repos.append(settings.embedding.model_name)
    if not args.embedder_only and settings.reranker.enabled:
        repos.append(settings.reranker.model_name)

    cache = settings.embedding.cache_dir
    cache.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(cache).free
    print(f"\n{BLUE}PolyDoc — model download{RESET}")
    print(f"  cache     : {cache}")
    print(f"  free disk : {human(free)}")
    print("  skipping  : onnx/, images (~2.9 GB of unused artefacts)\n")

    if free < 6e9:
        print(f"{RED}✗{RESET} Less than 6 GB free — free up space first.")
        return 1

    # Every repo is attempted even if an earlier one fails — `all(generator)`
    # would short-circuit and silently skip the reranker.
    ok = True
    for repo in repos:
        ok = fetch(repo) and ok

    total = sum(f.stat().st_size for f in cache.rglob("*") if f.is_file() and not f.is_symlink())
    print(f"\n  cache size: {human(total)}")
    print(f"{GREEN}✓ Models ready.{RESET}" if ok else f"{RED}✗ Some downloads failed.{RESET}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
