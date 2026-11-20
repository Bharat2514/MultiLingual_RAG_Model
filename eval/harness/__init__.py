"""Evaluation harness for PolyDoc (PRD §12).

Importing this package puts `backend/` on `sys.path` so the harness can use the
production retrieval and generation code directly rather than a reimplementation
of it — an eval that exercises a copy of the pipeline measures the copy.
"""

from __future__ import annotations

import sys
from pathlib import Path

EVAL_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EVAL_ROOT.parent
BACKEND_ROOT = REPO_ROOT / "backend"

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

CORPUS_DIR = EVAL_ROOT / "corpus"
GOLD_DIR = EVAL_ROOT / "gold"
RESULTS_DIR = EVAL_ROOT / "results"

__all__ = ["BACKEND_ROOT", "CORPUS_DIR", "EVAL_ROOT", "GOLD_DIR", "REPO_ROOT", "RESULTS_DIR"]
