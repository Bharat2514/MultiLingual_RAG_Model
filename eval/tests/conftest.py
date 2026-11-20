"""Put `eval/` and `backend/` on the path so the harness imports as it does at runtime."""

from __future__ import annotations

import sys
from pathlib import Path

EVAL_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = EVAL_ROOT.parent / "backend"

for path in (EVAL_ROOT, BACKEND_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
