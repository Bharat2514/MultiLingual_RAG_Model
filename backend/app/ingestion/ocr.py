"""OCR for scanned Hindi/English pages (FR-4, risk R1).

Tesseract's accuracy on Devanagari is far more sensitive to image quality than
it is on Latin script — conjunct consonants and matras smear together on a
skewed or noisy scan. The preprocessing chain here (deskew → denoise → upscale
→ binarise) is what turns an unusable Hindi scan into a readable one, and it is
the difference between empty context and a correct answer.

Confidence is captured per page so low-quality scans can be surfaced to the
user rather than silently producing garbage (R1 mitigation).
"""

from __future__ import annotations

import functools
import io

from app.core.config import settings
from app.core.errors import OCRFailedError
from app.core.logging import get_logger

log = get_logger(__name__)

# Below this mean word confidence, OCR output is more likely to mislead than
# help; the page is flagged so the UI can prompt for a better scan.
LOW_CONFIDENCE_THRESHOLD = 55.0

# Tesseract works best around 300 DPI; upscale anything materially smaller.
MIN_OCR_HEIGHT_PX = 1000


@functools.lru_cache(maxsize=1)
def is_available() -> bool:
    """Whether a usable Tesseract binary with our languages is installed."""
    try:
        import pytesseract

        langs = set(pytesseract.get_languages(config=""))
    except Exception as exc:  # noqa: BLE001
        log.warning("tesseract_unavailable", error=str(exc))
        return False

    required = set(settings.ocr.langs.split("+"))
    missing = required - langs
    if missing:
        log.warning("tesseract_missing_languages", missing=sorted(missing), available=sorted(langs))
        return False
    return True


def _deskew(image, cv2, np):
    """Rotate the page so text lines are horizontal.

    Devanagari's shirorekha (the horizontal head-line) makes skew especially
    destructive: a few degrees merges adjacent lines into one blob.
    """
    inverted = cv2.bitwise_not(image)
    threshold = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = cv2.findNonZero(threshold)
    if coords is None:
        return image

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle
    elif angle > 45:
        angle = angle - 90

    # Ignore sub-degree noise; rotating costs quality for no gain.
    if abs(angle) < 0.35 or abs(angle) > 20:
        return image

    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width // 2, height // 2), angle, 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def preprocess(png_bytes: bytes) -> bytes:
    """Deskew, denoise, upscale, and binarise a rendered page."""
    if not settings.ocr.preprocess:
        return png_bytes

    try:
        import cv2
        import numpy as np
    except ImportError:  # pragma: no cover - OpenCV is a hard dep in the image
        log.warning("opencv_unavailable_skipping_preprocess")
        return png_bytes

    buffer = np.frombuffer(png_bytes, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return png_bytes

    image = _deskew(image, cv2, np)

    # Non-local-means preserves the thin matras that a blur would erase.
    image = cv2.fastNlMeansDenoising(image, None, h=7, templateWindowSize=7, searchWindowSize=21)

    height = image.shape[0]
    if height < MIN_OCR_HEIGHT_PX:
        scale = min(3.0, MIN_OCR_HEIGHT_PX / height)
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # Otsu beats adaptive thresholding on document scans with even lighting and
    # avoids the speckle adaptive methods introduce in Devanagari counters.
    image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    ok, encoded = cv2.imencode(".png", image)
    return encoded.tobytes() if ok else png_bytes


def ocr_image(png_bytes: bytes, *, langs: str | None = None) -> tuple[str, float]:
    """Run OCR on a page image. Returns (text, mean word confidence 0-100)."""
    if not is_available():
        raise OCRFailedError(
            "Tesseract is not installed or is missing the Hindi/English language data.",
            details={"required_langs": settings.ocr.langs},
        )

    import pytesseract
    from PIL import Image

    language = langs or settings.ocr.langs
    processed = preprocess(png_bytes)

    try:
        image = Image.open(io.BytesIO(processed))
        # PSM 3 = fully automatic page segmentation; correct for mixed-layout
        # circulars. OEM 3 uses the LSTM engine, which is what supports Hindi.
        data = pytesseract.image_to_data(
            image,
            lang=language,
            config="--oem 3 --psm 3",
            output_type=pytesseract.Output.DICT,
        )
    except Exception as exc:
        raise OCRFailedError(f"OCR failed: {exc}") from exc

    words: list[str] = []
    confidences: list[float] = []
    last_line: tuple[int, int, int] | None = None
    lines: list[list[str]] = []

    for i, raw_word in enumerate(data.get("text", [])):
        word = (raw_word or "").strip()
        confidence = float(data.get("conf", ["-1"])[i])
        if not word or confidence < 0:
            continue

        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        if key != last_line:
            lines.append([])
            last_line = key
        lines[-1].append(word)
        words.append(word)
        confidences.append(confidence)

    text = "\n".join(" ".join(line) for line in lines if line)
    mean_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    return text, round(mean_confidence, 2)


def detect_script_langs(png_bytes: bytes) -> str:
    """Per-page language choice for OCR (FR-4 auto-detect).

    Passing both languages to Tesseract at once is more robust than guessing
    wrong, so this only narrows when the configuration lists a single language.
    """
    return settings.ocr.langs
