"""Render evaluation results as console tables, Markdown, and JSON.

Three outputs because they have three consumers: a human watching `make eval`,
`docs/EVALUATION.md`, and the CI gate, which diffs a JSON baseline. They are
produced from the same objects so the published table cannot drift from the
number the gate enforces.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

Row = Sequence[str]


def table(headers: Row, rows: Sequence[Row], *, align_right_from: int = 1) -> str:
    """A fixed-width table that is also valid Markdown.

    One renderer for both outputs: a separate Markdown path is a second place for
    a number to be formatted differently from the one printed to the terminal.
    """
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def render(row: Row) -> str:
        cells = [
            cell.rjust(widths[i]) if i >= align_right_from else cell.ljust(widths[i])
            for i, cell in enumerate(row)
        ]
        return "| " + " | ".join(cells) + " |"

    separator = (
        "|"
        + "|".join(
            ("-" * (w + 2)) if i < align_right_from else ("-" * (w + 1) + ":")
            for i, w in enumerate(widths)
        )
        + "|"
    )

    return "\n".join([render(headers), separator, *(render(r) for r in rows)])


def pct(value: float) -> str:
    return f"{value:.3f}"


def delta(value: float, baseline: float) -> str:
    """Signed points difference. Blank on the baseline row itself."""
    if baseline is None:
        return ""
    diff = (value - baseline) * 100
    if abs(diff) < 0.05:
        return "  ±0.0"
    return f"{diff:+.1f}"


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {k: _jsonable(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    return value


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {"generated_at": datetime.now(UTC).isoformat(timespec="seconds"), **_jsonable(payload)}
    path.write_text(json.dumps(body, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def write_markdown(path: Path, sections: Sequence[tuple[str, str]], *, preamble: str = "") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    parts = [preamble.rstrip(), ""] if preamble else []
    for title, body in sections:
        parts.extend([f"### {title}", "", body, ""])
    path.write_text("\n".join(parts).rstrip() + "\n", encoding="utf-8")
    return path
