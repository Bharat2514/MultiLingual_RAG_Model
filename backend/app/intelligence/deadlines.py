"""Deadline, obligation and penalty extraction — no model call.

"Submit by 31 March", "late fee ₹500 per day", "w.e.f. 01.04.2024": the
dates and duties in a circular are what its reader has to act on, and they are
exactly what a question-answering interface makes you ask for one at a time.
This pulls every one of them out in a single pass, each with its page, so the
document can be turned into a timeline and a calendar file.

Extraction is rule-based on purpose. Dates have a closed set of written forms
in Indian government English and Hindi, and a regex never invents a deadline —
a model asked to "list the deadlines" occasionally does.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime

import regex

from app.intelligence.textutil import ascii_digits
from app.nlp.sentences import split_sentences

_MONTHS: dict[str, int] = {
    # English
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9, "october": 10,
    "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
    # Hindi (both common spellings where they differ)
    "जनवरी": 1, "फरवरी": 2, "फ़रवरी": 2, "मार्च": 3, "अप्रैल": 4, "मई": 5,
    "जून": 6, "जुलाई": 7, "अगस्त": 8, "सितंबर": 9, "सितम्बर": 9,
    "अक्टूबर": 10, "अक्तूबर": 10, "नवंबर": 11, "नवम्बर": 11,
    "दिसंबर": 12, "दिसम्बर": 12,
}  # fmt: skip

_MONTH_ALT = "|".join(sorted((regex.escape(m) for m in _MONTHS), key=len, reverse=True))

_DATE_PATTERNS = [
    # 31/03/2025, 31-03-2025, 31.03.2025 — Indian documents are day-first.
    regex.compile(r"(?<!\d)(?P<d>\d{1,2})[./-](?P<m>\d{1,2})[./-](?P<y>(?:19|20)\d{2})(?!\d)"),
    # 31 March 2025, 31st March, 2025, 31 मार्च 2025
    regex.compile(
        rf"(?i)(?<!\d)(?P<d>\d{{1,2}})(?:st|nd|rd|th)?\s+(?P<mn>{_MONTH_ALT})\.?,?\s+(?P<y>(?:19|20)\d{{2}})(?!\d)"
    ),
    # March 31, 2025
    regex.compile(
        rf"(?i)\b(?P<mn>{_MONTH_ALT})\.?\s+(?P<d>\d{{1,2}})(?:st|nd|rd|th)?,?\s+(?P<y>(?:19|20)\d{{2}})(?!\d)"
    ),
]

# Classification looks at the text *around each date*, not the whole sentence:
# "submit by 31 March as per the circular dated 1 January" holds a deadline and
# an issue date, and a sentence-level rule would label both the same.
_EFFECTIVE_RE = regex.compile(
    r"(?i)with effect from|w\.?\s?e\.?\s?f\.?|effective (?:from|date|on)|comes? into force|"
    r"से प्रभावी|प्रभावी होगा|प्रभावी होगी|लागू होगा|लागू होगी|से लागू"
)
_ISSUED_BEFORE_RE = regex.compile(r"(?i)(?:\bdated|issued on|दिनांक|दि\.)\s*[:\-–]?\s*$")
_DEADLINE_BEFORE_RE = regex.compile(
    r"(?i)(?:\bby|\bbefore|\btill|\buntil|\bupto|\bup to|on or before|not later than|"
    r"no later than|last date(?: is| of [a-z ]+)?|deadline(?: is)?|due (?:date|on)|closing date(?: is)?)"
    r"\s*[:\-–]?\s*(?:the\s+)?$"
)
_DEADLINE_AFTER_RE = regex.compile(r"^\s*(?:तक|से पूर्व|से पहले|के पूर्व|तक अवश्य)")
_DEADLINE_SENTENCE_RE = regex.compile(
    r"(?i)last date|deadline|due date|closing date|अंतिम तिथि|अंतिम दिनांक|अन्तिम तिथि"
)
_EVENT_RE = regex.compile(
    r"(?i)meeting|will be held|opening of|pre-?bid|hearing|interview|examination|"
    r"बैठक|आयोजित|परीक्षा|साक्षात्कार"
)

_PENALTY_RE = regex.compile(
    r"(?i)penalt|fine\b|late fee|forfeit|liquidated damages|interest at|"
    r"जुर्माना|शास्ति|दंड|विलंब शुल्क|ज़ब्त|जब्त"
)
_OBLIGATION_RE = regex.compile(
    r"(?i)\bmust\b|\bshall\b|\bis required to\b|\bare required to\b|mandatory|compulsory|"
    r"अनिवार्य|आवश्यक होगा|आवश्यक है|करना होगा|करनी होगी|करने होंगे"
)
_AMOUNT_RE = regex.compile(
    r"(?i)(?:₹|rs\.?|inr|रु\.?)\s?\d[\d,]*(?:\.\d+)?(?:\s?(?:lakh|lakhs|crore|लाख|करोड़))?|"
    r"\d[\d,]*(?:\.\d+)?\s?(?:rupees|रुपये|रुपए|रूपये)"
)


@dataclass(slots=True)
class TimelineItem:
    kind: str  # deadline | effective | event | issued | date | penalty | obligation
    text: str
    date: str | None
    amount: str | None
    doc_id: str
    filename: str
    page: int
    chunk_id: str

    def to_dict(self) -> dict:
        return asdict(self)


def find_dates(text: str) -> list[tuple[date, int, int]]:
    """Every date in `text` with its span; impossible dates are skipped."""
    probe = ascii_digits(text)
    found: dict[tuple[int, int], date] = {}
    for pattern in _DATE_PATTERNS:
        for m in pattern.finditer(probe):
            if any(s <= m.start() < e for s, e in found):
                continue
            try:
                day = int(m.group("d"))
                groups = m.groupdict()
                month = int(groups["m"]) if groups.get("m") else _MONTHS[m.group("mn").casefold()]
                found[(m.start(), m.end())] = date(int(m.group("y")), month, day)
            except (KeyError, ValueError):
                continue  # 31/02/2025 and friends
    return sorted(((d, s, e) for (s, e), d in found.items()), key=lambda t: t[1])


def parse_dates(text: str) -> list[date]:
    return list(dict.fromkeys(d for d, _, _ in find_dates(text)))


def classify_date(sentence: str, start: int, end: int) -> str:
    before = sentence[max(0, start - 40) : start]
    after = sentence[end : end + 12]
    if _ISSUED_BEFORE_RE.search(before):
        return "issued"
    if _EFFECTIVE_RE.search(sentence[max(0, start - 60) : end + 30]):
        return "effective"
    if _DEADLINE_BEFORE_RE.search(before) or _DEADLINE_AFTER_RE.search(after):
        return "deadline"
    if _DEADLINE_SENTENCE_RE.search(sentence):
        return "deadline"
    if _EVENT_RE.search(sentence):
        return "event"
    return "date"


def extract(chunks: list[dict]) -> list[TimelineItem]:
    """Timeline items from chunk dicts (`text`, `page`, `doc_id`, `filename`, `chunk_id`).

    Chunks overlap by design, so the same sentence can appear twice; items are
    deduplicated on (kind, date, sentence).
    """
    items: list[TimelineItem] = []
    seen: set[tuple] = set()

    for chunk in chunks:
        for sentence in split_sentences(chunk["text"]):
            text = " ".join(sentence.text.split())
            if len(text) < 8:
                continue
            amount_match = _AMOUNT_RE.search(text)
            amount = amount_match.group(0).strip() if amount_match else None
            spans = find_dates(text)
            dates = [d for d, _, _ in spans]

            kinds: list[tuple[str, date | None]] = [
                (classify_date(text, s, e), d) for d, s, e in spans
            ]
            if _PENALTY_RE.search(text):
                kinds.append(("penalty", dates[0] if dates else None))
            elif not dates and _OBLIGATION_RE.search(text):
                kinds.append(("obligation", None))

            for kind, when in kinds:
                key = (kind, when, text.casefold())
                if key in seen:
                    continue
                seen.add(key)
                items.append(
                    TimelineItem(
                        kind=kind,
                        text=text[:600],
                        date=when.isoformat() if when else None,
                        amount=amount,
                        doc_id=chunk["doc_id"],
                        filename=chunk["filename"],
                        page=chunk["page"],
                        chunk_id=chunk["chunk_id"],
                    )
                )

    # Dated items chronologically first, then undated duties in reading order.
    dated = sorted((i for i in items if i.date), key=lambda i: i.date or "")
    undated = [i for i in items if not i.date]
    return dated + undated


# --- iCalendar export ----------------------------------------------------------


def _ics_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace(";", "\\;").replace(",", "\\,").replace("\n", "\\n")


def _fold(line: str) -> str:
    """RFC 5545 §3.1: lines over 75 octets are folded with CRLF + space."""
    encoded = line.encode("utf-8")
    if len(encoded) <= 75:
        return line
    parts: list[str] = []
    current = b""
    for ch in line:
        b = ch.encode("utf-8")
        if len(current) + len(b) > 74:
            parts.append(current.decode("utf-8"))
            current = b
        else:
            current += b
    parts.append(current.decode("utf-8"))
    return "\r\n ".join(parts)


_ICS_KINDS = {
    "deadline": "Deadline",
    "effective": "Takes effect",
    "event": "Event",
    "penalty": "Penalty",
}


def to_ics(items: list[TimelineItem], *, calendar_name: str = "PolyDoc deadlines") -> str:
    """All-day events for the actionable dated items."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//PolyDoc//Document deadlines//EN",
        "CALSCALE:GREGORIAN",
        f"X-WR-CALNAME:{_ics_escape(calendar_name)}",
    ]
    for n, item in enumerate(items):
        if not item.date or item.kind not in _ICS_KINDS:
            continue
        day = date.fromisoformat(item.date)
        next_day = date.fromordinal(day.toordinal() + 1)
        summary = f"{_ICS_KINDS[item.kind]}: {item.text[:80]}"
        lines += [
            "BEGIN:VEVENT",
            f"UID:{item.chunk_id}-{n}@polydoc",
            f"DTSTAMP:{stamp}",
            f"DTSTART;VALUE=DATE:{day.strftime('%Y%m%d')}",
            f"DTEND;VALUE=DATE:{next_day.strftime('%Y%m%d')}",
            f"SUMMARY:{_ics_escape(summary)}",
            f"DESCRIPTION:{_ics_escape(item.text + f' — {item.filename} p.{item.page}')}",
            "END:VEVENT",
        ]
    lines.append("END:VCALENDAR")
    return "\r\n".join(_fold(line) for line in lines) + "\r\n"
