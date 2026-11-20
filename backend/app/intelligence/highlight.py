"""Locate a cited passage on its PDF page, for highlighting in the viewer.

A citation that opens the page with the exact sentence highlighted is
verifiable at a glance; a snippet in a sidebar asks the reader to trust that it
came from where it says. Two strategies, chosen per page:

* **text layer** — PyMuPDF `search_for` over short word windows of the passage.
  Whole sentences rarely match verbatim (line breaks, hyphenation, cleaning), but
  five-word windows do, and together they cover the passage.
* **OCR word boxes** — a scanned page has no text layer to search, so the page is
  OCR'd with Tesseract's word-level boxes and the longest run of words that
  belong to the passage is highlighted.

Rectangles are returned normalised to 0–1 of the page size, so the client can
overlay them on an image rendered at any resolution.
"""

from __future__ import annotations

import io
from pathlib import Path

from app.core.logging import get_logger
from app.intelligence.textutil import normalize, words
from app.nlp.sentences import split_sentences

log = get_logger(__name__)

WINDOW = 5
MAX_RECTS = 400


def render_page(path: Path, page_no: int, *, dpi: int = 110) -> bytes:
    from app.ingestion.parsers.pdf import render_page_png

    return render_page_png(path, page_no, dpi=dpi)


def page_count(path: Path) -> int:
    from app.ingestion.parsers.pdf import page_count as count

    return count(path)


def _windows(text: str) -> list[str]:
    out: list[str] = []
    for sentence in split_sentences(text):
        tokens = sentence.text.split()
        if len(tokens) <= WINDOW:
            if len(sentence.text.strip()) >= 6:
                out.append(sentence.text.strip())
            continue
        out.extend(" ".join(tokens[i : i + WINDOW]) for i in range(0, len(tokens), WINDOW))
    return out


def _merge(
    rects: list[tuple[float, float, float, float]],
) -> list[tuple[float, float, float, float]]:
    """Drop exact duplicates and rects contained in another."""
    unique = sorted(set(rects), key=lambda r: (r[1], r[0]))
    kept: list[tuple[float, float, float, float]] = []
    for r in unique:
        if any(k[0] <= r[0] and k[1] <= r[1] and k[2] >= r[2] and k[3] >= r[3] for k in kept):
            continue
        kept.append(r)
    return kept[:MAX_RECTS]


def text_layer_rects(
    path: Path, page_no: int, passage: str
) -> list[tuple[float, float, float, float]]:
    import fitz

    doc = fitz.open(path)
    try:
        page = doc.load_page(page_no - 1)
        width, height = page.rect.width, page.rect.height
        found: list[tuple[float, float, float, float]] = []
        for probe in _windows(passage):
            for rect in page.search_for(probe):
                found.append((rect.x0 / width, rect.y0 / height, rect.x1 / width, rect.y1 / height))
        return _merge(found)
    finally:
        doc.close()


def ocr_rects(png: bytes, passage: str) -> list[tuple[float, float, float, float]]:
    """Boxes of the longest run of OCR words that belong to the passage."""
    from app.ingestion import ocr as ocr_module

    if not ocr_module.is_available():
        return []

    import pytesseract
    from PIL import Image

    from app.core.config import settings

    image = Image.open(io.BytesIO(png))
    data = pytesseract.image_to_data(
        image,
        lang=settings.ocr.langs,
        config="--oem 3 --psm 3",
        output_type=pytesseract.Output.DICT,
    )
    wanted = {w for w in words(passage) if len(w) >= 2}
    width, height = image.size

    boxes: list[tuple[int, tuple[float, float, float, float]]] = []
    for i, raw in enumerate(data.get("text", [])):
        token = normalize(raw or "")
        if not token or float(data["conf"][i]) < 0:
            continue
        hit = any(w in wanted for w in words(token))
        if hit:
            x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
            boxes.append((i, (x / width, y / height, (x + w) / width, (y + h) / height)))

    # Longest run with gaps of at most 3 words: the passage, not every page-wide
    # occurrence of a common word.
    best: list[tuple[float, float, float, float]] = []
    run: list[tuple[float, float, float, float]] = []
    last = -10
    for index, box in boxes:
        if index - last > 4:
            if len(run) > len(best):
                best = run
            run = []
        run.append(box)
        last = index
    if len(run) > len(best):
        best = run
    return best if len(best) >= 3 else []


def locate(path: Path, page_no: int, passage: str, *, ocr_page: bool) -> dict:
    method = "none"
    rects: list[tuple[float, float, float, float]] = []
    try:
        rects = text_layer_rects(path, page_no, passage)
        method = "text" if rects else "none"
    except Exception:
        log.warning("highlight_text_layer_failed", page=page_no, exc_info=True)

    if not rects and ocr_page:
        try:
            rects = ocr_rects(render_page(path, page_no, dpi=200), passage)
            method = "ocr" if rects else "none"
        except Exception:
            log.warning("highlight_ocr_failed", page=page_no, exc_info=True)

    return {
        "method": method,
        "rects": [
            {"x0": round(r[0], 5), "y0": round(r[1], 5), "x1": round(r[2], 5), "y1": round(r[3], 5)}
            for r in rects
        ],
    }
