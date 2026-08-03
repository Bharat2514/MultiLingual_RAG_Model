"""Torch device selection.

Local dev on Apple silicon should use MPS; CI and Linux containers fall back to
CPU (or CUDA when present). Torch is an optional import so the API can boot for
health checks and metadata endpoints without the ML stack installed.
"""

from __future__ import annotations

import functools

from app.core.logging import get_logger

log = get_logger(__name__)


def in_forked_child() -> bool:
    """True when running inside a process created by `fork()`.

    Celery's default `prefork` pool forks its workers. Apple's Metal stack does
    not survive that: the child cannot reach `MTLCompilerService`, and the first
    MPS kernel launch aborts the process with SIGABRT rather than raising. So a
    forked child must never select MPS.
    """
    import multiprocessing

    return multiprocessing.parent_process() is not None


def resolve_device(preference: str = "auto") -> str:
    """Return a concrete torch device string for a configured preference."""
    # `forked` is part of the cache key on purpose: a forked child inherits the
    # parent's lru_cache, so a value computed pre-fork ("mps") would leak into
    # the child and abort it on the first kernel launch.
    return _resolve_device(preference, in_forked_child())


@functools.lru_cache(maxsize=8)
def _resolve_device(preference: str, forked: bool) -> str:
    try:
        import torch
    except ImportError:  # pragma: no cover - ML extras not installed
        log.warning("torch_unavailable_defaulting_to_cpu")
        return "cpu"

    mps_available = (
        getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    )

    if preference != "auto":
        # An explicit `mps` in a forked child would crash the process, so the
        # guard applies to the configured value too — degrade, never abort.
        if preference == "mps" and forked:
            log.warning("mps_unavailable_in_forked_process_using_cpu")
            return "cpu"
        return preference

    if torch.cuda.is_available():
        return "cuda"
    if mps_available:
        if forked:
            log.warning("mps_unavailable_in_forked_process_using_cpu")
            return "cpu"
        return "mps"
    return "cpu"


def supports_fp16(device: str) -> bool:
    """fp16 is a win on CUDA; on MPS and CPU it is slower or unsupported."""
    return device == "cuda"
