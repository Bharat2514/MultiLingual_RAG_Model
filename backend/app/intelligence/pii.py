"""Detection and masking of Indian personal identifiers.

HR files, pension circulars and scheme applications are full of Aadhaar, PAN
and bank details. When the LLM is a remote API, sending those verbatim is a
DPDP Act problem, and none of them is ever needed to *answer* a question about
the document. So the context sent to the model is masked; the user's own view
of their documents is not.

Detectors favour precision over recall where the pattern alone is ambiguous:

* **Aadhaar** must pass the Verhoeff checksum UIDAI uses, so a random 12-digit
  reference number is not masked as an identity number;
* **PAN** must carry a valid holder-type letter in position 4;
* **bank account numbers** are bare digit runs, indistinguishable from any
  other number, so they are only masked next to an account keyword
  ("A/c", "account", "खाता").
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import regex

from app.intelligence.textutil import ascii_digits


class PIIKind(StrEnum):
    AADHAAR = "AADHAAR"
    PAN = "PAN"
    PHONE = "PHONE"
    EMAIL = "EMAIL"
    IFSC = "IFSC"
    BANK_ACCOUNT = "BANK_ACCOUNT"
    VOTER_ID = "VOTER_ID"
    PASSPORT = "PASSPORT"


@dataclass(frozen=True, slots=True)
class PIIMatch:
    kind: PIIKind
    start: int
    end: int
    text: str


# --- Verhoeff checksum (Aadhaar) ---------------------------------------------

_VERHOEFF_D = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 2, 3, 4, 0, 6, 7, 8, 9, 5),
    (2, 3, 4, 0, 1, 7, 8, 9, 5, 6),
    (3, 4, 0, 1, 2, 8, 9, 5, 6, 7),
    (4, 0, 1, 2, 3, 9, 5, 6, 7, 8),
    (5, 9, 8, 7, 6, 0, 4, 3, 2, 1),
    (6, 5, 9, 8, 7, 1, 0, 4, 3, 2),
    (7, 6, 5, 9, 8, 2, 1, 0, 4, 3),
    (8, 7, 6, 5, 9, 3, 2, 1, 0, 4),
    (9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
)
_VERHOEFF_P = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 5, 7, 6, 2, 8, 3, 0, 9, 4),
    (5, 8, 0, 3, 7, 9, 6, 1, 4, 2),
    (8, 9, 1, 6, 0, 4, 3, 5, 2, 7),
    (9, 4, 5, 3, 1, 2, 6, 8, 7, 0),
    (4, 2, 8, 6, 5, 7, 3, 9, 0, 1),
    (2, 7, 9, 3, 8, 0, 6, 4, 1, 5),
    (7, 0, 4, 6, 9, 1, 3, 2, 5, 8),
)


def verhoeff_valid(number: str) -> bool:
    if not number.isdigit():
        return False
    check = 0
    for i, digit in enumerate(reversed(number)):
        check = _VERHOEFF_D[check][_VERHOEFF_P[i % 8][int(digit)]]
    return check == 0


# --- patterns ----------------------------------------------------------------

_AADHAAR_RE = regex.compile(r"(?<![\d])([2-9]\d{3})[\s-]?(\d{4})[\s-]?(\d{4})(?![\d])")
# 4th character is the holder type: P(erson), C(ompany), H(UF), F(irm), A(OP),
# T(rust), B(OI), L(ocal authority), J(uridical), G(overnment).
_PAN_RE = regex.compile(r"(?<![A-Za-z0-9])[A-Z]{3}[PCHFATBLJG][A-Z]\d{4}[A-Z](?![A-Za-z0-9])")
_IFSC_RE = regex.compile(r"(?<![A-Za-z0-9])[A-Z]{4}0[A-Z0-9]{6}(?![A-Za-z0-9])")
_PHONE_RE = regex.compile(r"(?<![\d+])(?:(?:\+|00)?91[\s-]?)?[6-9]\d{4}[\s-]?\d{5}(?![\d])")
_EMAIL_RE = regex.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_VOTER_RE = regex.compile(r"(?<![A-Za-z0-9])[A-Z]{3}\d{7}(?![A-Za-z0-9])")
_ACCOUNT_RE = regex.compile(
    r"(?i)(?:a/c|acct|account|खाता|खाता\s*संख्या|khata)[^\d\n]{0,25}(\d{9,18})(?![\d])"
)
_PASSPORT_RE = regex.compile(
    r"(?i)(?:passport|पासपोर्ट)[^\n]{0,25}?(?<![A-Za-z0-9])([A-PR-WYa-pr-wy][1-9]\d{6})(?![\d])"
)


def find_pii(text: str) -> list[PIIMatch]:
    """All identifiers in `text`, non-overlapping, in order of appearance."""
    if not text:
        return []

    # Same length as the original, so spans found here are valid there too.
    probe = ascii_digits(text)
    found: list[PIIMatch] = []

    def add(kind: PIIKind, start: int, end: int) -> None:
        found.append(PIIMatch(kind, start, end, text[start:end]))

    for m in _AADHAAR_RE.finditer(probe):
        if verhoeff_valid("".join(m.groups())):
            add(PIIKind.AADHAAR, m.start(), m.end())
    for m in _PAN_RE.finditer(probe):
        add(PIIKind.PAN, m.start(), m.end())
    for m in _IFSC_RE.finditer(probe):
        add(PIIKind.IFSC, m.start(), m.end())
    for m in _EMAIL_RE.finditer(probe):
        add(PIIKind.EMAIL, m.start(), m.end())
    for m in _VOTER_RE.finditer(probe):
        add(PIIKind.VOTER_ID, m.start(), m.end())
    for m in _ACCOUNT_RE.finditer(probe):
        add(PIIKind.BANK_ACCOUNT, m.start(1), m.end(1))
    for m in _PASSPORT_RE.finditer(probe):
        add(PIIKind.PASSPORT, m.start(1), m.end(1))
    for m in _PHONE_RE.finditer(probe):
        add(PIIKind.PHONE, m.start(), m.end())

    # Resolve overlaps: earlier start wins, then the longer match. A 12-digit
    # Aadhaar must not also be reported as a phone number inside it.
    found.sort(key=lambda p: (p.start, -(p.end - p.start)))
    result: list[PIIMatch] = []
    cursor = -1
    for match in found:
        if match.start >= cursor:
            result.append(match)
            cursor = match.end
    return result


def redact(text: str) -> tuple[str, list[PIIMatch]]:
    """Replace each identifier with a typed placeholder like `[REDACTED:PAN]`.

    The placeholder keeps the *kind* so the model can still say "the applicant's
    PAN is listed on page 2" without ever seeing it.
    """
    matches = find_pii(text)
    if not matches:
        return text, []
    parts: list[str] = []
    cursor = 0
    for match in matches:
        parts.append(text[cursor : match.start])
        parts.append(f"[REDACTED:{match.kind}]")
        cursor = match.end
    parts.append(text[cursor:])
    return "".join(parts), matches
