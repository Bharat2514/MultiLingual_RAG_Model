"""Hinglish → Devanagari transliteration (FR-15, §11.3).

Romanized Hindi embeds poorly against Devanagari content, so we transliterate
and search both forms. Off-the-shelf ITRANS is not usable on its own here:

    kya        -> क्य        (should be क्या)
    batao      -> बतओ        (should be बताओ)
    document   -> दोचुमेन्त्   (should stay "document")

The last one matters most. Hinglish is *code-mixed* — "document ka summary do"
is half English — and blindly transliterating turns real English words into
nonsense that matches nothing. So the strategy is, per token:

1. A curated lexicon of common romanized Hindi (high precision, covers the
   function words and question words that carry most queries).
2. Recognised English words are left alone.
3. Anything else goes through ITRANS with Hinglish-specific spelling fixes.

Output is therefore genuinely code-mixed: "document का summary दो".

This is a recall booster, not a replacement — the original romanized string is
always searched too, and the two candidate lists are fused (FR-15).
"""

from __future__ import annotations

import functools

import regex

from app.core.logging import get_logger
from app.nlp.script import has_devanagari

log = get_logger(__name__)

# --- curated lexicon ---------------------------------------------------------
# Common romanized Hindi and its Devanagari form. Function words first (they
# survive code-mixing), then the content words that recur in the document
# domains this product targets: government circulars, policy, education.
HINGLISH_LEXICON: dict[str, str] = {
    # pronouns
    "main": "मैं",
    "mai": "मैं",
    "mujhe": "मुझे",
    "muje": "मुझे",
    "mera": "मेरा",
    "meri": "मेरी",
    "mere": "मेरे",
    "hum": "हम",
    "hume": "हमें",
    "humein": "हमें",
    "hamara": "हमारा",
    "hamari": "हमारी",
    "tum": "तुम",
    "tumhe": "तुम्हें",
    "aap": "आप",
    "aapko": "आपको",
    "aapka": "आपका",
    "aapki": "आपकी",
    "wo": "वो",
    "voh": "वह",
    "yeh": "यह",
    "ye": "ये",
    "iska": "इसका",
    "iski": "इसकी",
    "iske": "इसके",
    "uska": "उसका",
    "uski": "उसकी",
    "uske": "उसके",
    "unka": "उनका",
    "unki": "उनकी",
    "inka": "इनका",
    "is": "इस",
    "us": "उस",
    "in": "इन",
    "un": "उन",
    # question words
    "kya": "क्या",
    "kyu": "क्यों",
    "kyun": "क्यों",
    "kyon": "क्यों",
    "kaise": "कैसे",
    "kaisa": "कैसा",
    "kaisi": "कैसी",
    "kaun": "कौन",
    "kaunsa": "कौनसा",
    "konsa": "कौनसा",
    "kab": "कब",
    "kahan": "कहाँ",
    "kaha": "कहाँ",
    "kitna": "कितना",
    "kitne": "कितने",
    "kitni": "कितनी",
    # verbs / auxiliaries
    "hai": "है",
    "hain": "हैं",
    "tha": "था",
    "the": "थे",
    "thi": "थी",
    "ho": "हो",
    "hoga": "होगा",
    "hogi": "होगी",
    "hona": "होना",
    "hota": "होता",
    "hoti": "होती",
    "hote": "होते",
    "raha": "रहा",
    "rahi": "रही",
    "rahe": "रहे",
    "kar": "कर",
    "karo": "करो",
    "karna": "करना",
    "karne": "करने",
    "karta": "करता",
    "karti": "करती",
    "kiya": "किया",
    "kijiye": "कीजिये",
    "dijiye": "दीजिये",
    "dena": "देना",
    "do": "दो",
    "de": "दे",
    "diya": "दिया",
    "batao": "बताओ",
    "bataye": "बताये",
    "bataiye": "बताइये",
    "samjhao": "समझाओ",
    "likho": "लिखो",
    "padho": "पढ़ो",
    "dekho": "देखो",
    "milta": "मिलता",
    "milti": "मिलती",
    "mile": "मिले",
    "chahiye": "चाहिए",
    "sakta": "सकता",
    "sakti": "सकती",
    "jata": "जाता",
    "jati": "जाती",
    "gaya": "गया",
    "hui": "हुई",
    "hua": "हुआ",
    "huye": "हुए",
    "lagta": "लगता",
    "lagti": "लगती",
    "lagega": "लगेगा",
    # postpositions / connectives
    "ka": "का",
    "ki": "की",
    "ke": "के",
    "ko": "को",
    "se": "से",
    "me": "में",
    "mein": "में",
    "par": "पर",
    "pe": "पे",
    "tak": "तक",
    "aur": "और",
    "ya": "या",
    "lekin": "लेकिन",
    "magar": "मगर",
    "agar": "अगर",
    "toh": "तो",
    "phir": "फिर",
    "bhi": "भी",
    "nahi": "नहीं",
    "nahin": "नहीं",
    "haan": "हाँ",
    "han": "हाँ",
    "liye": "लिये",
    "bare": "बारे",
    "sath": "साथ",
    "bina": "बिना",
    "wala": "वाला",
    "wali": "वाली",
    "jaise": "जैसे",
    "waise": "वैसे",
    # quantity / time
    "sab": "सब",
    "sabhi": "सभी",
    "kuch": "कुछ",
    "kuchh": "कुछ",
    "bahut": "बहुत",
    "zyada": "ज्यादा",
    "jyada": "ज्यादा",
    "kam": "कम",
    "thoda": "थोड़ा",
    "poora": "पूरा",
    "pura": "पूरा",
    "adha": "आधा",
    "din": "दिन",
    "mahina": "महीना",
    "mahine": "महीने",
    "saal": "साल",
    "varsh": "वर्ष",
    "samay": "समय",
    "abhi": "अभी",
    "kabhi": "कभी",
    "hamesha": "हमेशा",
    "aaj": "आज",
    "kal": "कल",
    "pehle": "पहले",
    "baad": "बाद",
    "shuru": "शुरू",
    "ant": "अंत",
    # document / government domain
    "kaam": "काम",
    "niyam": "नियम",
    "adhikar": "अधिकार",
    "aavedan": "आवेदन",
    "avedan": "आवेदन",
    "praman": "प्रमाण",
    "pramaan": "प्रमाण",
    "patra": "पत्र",
    "praarthana": "प्रार्थना",
    "suchna": "सूचना",
    "soochna": "सूचना",
    "vibhag": "विभाग",
    "karyalay": "कार्यालय",
    "sarkar": "सरकार",
    "sarkari": "सरकारी",
    "mantralay": "मंत्रालय",
    "adhikari": "अधिकारी",
    "karmchari": "कर्मचारी",
    "naukri": "नौकरी",
    "vetan": "वेतन",
    "pension": "पेंशन",
    "seva": "सेवा",
    "niyukti": "नियुक्ति",
    "shiksha": "शिक्षा",
    "niti": "नीति",
    "neeti": "नीति",
    "yojana": "योजना",
    "vidyalay": "विद्यालय",
    "vidyarthi": "विद्यार्थी",
    "shikshak": "शिक्षक",
    "pariksha": "परीक्षा",
    "parinaam": "परिणाम",
    "praveshi": "प्रवेश",
    "pravesh": "प्रवेश",
    "shulk": "शुल्क",
    "jurmana": "जुर्माना",
    "danda": "दंड",
    "kanoon": "कानून",
    "adalat": "अदालत",
    "faisla": "फैसला",
    "report": "रिपोर्ट",
    "prativedan": "प्रतिवेदन",
    "sanshodhan": "संशोधन",
    "sudhar": "सुधार",
    "vivaran": "विवरण",
    "jankari": "जानकारी",
    "janakari": "जानकारी",
    "prakriya": "प्रक्रिया",
    "shart": "शर्त",
    "sharte": "शर्तें",
    "labh": "लाभ",
    "hani": "हानि",
    "raashi": "राशि",
    "rashi": "राशि",
    "rupaye": "रुपये",
    "rupay": "रुपये",
    "paisa": "पैसा",
    "khata": "खाता",
    "bhugtan": "भुगतान",
    "jama": "जमा",
    "sthiti": "स्थिति",
    "sthan": "स्थान",
    "nagar": "नगर",
    "jila": "जिला",
    "zila": "जिला",
    "rajya": "राज्य",
    "desh": "देश",
    "bharat": "भारत",
    "saransh": "सारांश",
    "mukhya": "मुख्य",
    "mahatvapurn": "महत्वपूर्ण",
    "avashyak": "आवश्यक",
    "anivarya": "अनिवार्य",
}

# Frequent English words that would otherwise be mangled by ITRANS. Only the
# ones plausible in a document question — this is a guard list, not a lexicon.
ENGLISH_GUARD: frozenset[str] = frozenset(
    [
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "if",
        "then",
        "than",
        "that",
        "this",
        "these",
        "those",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "do",
        "does",
        "did",
        "done",
        "have",
        "has",
        "had",
        "can",
        "could",
        "will",
        "would",
        "shall",
        "should",
        "may",
        "might",
        "must",
        "of",
        "in",
        "on",
        "at",
        "to",
        "for",
        "from",
        "with",
        "without",
        "within",
        "about",
        "above",
        "below",
        "under",
        "over",
        "between",
        "during",
        "after",
        "before",
        "by",
        "as",
        "into",
        "out",
        "up",
        "down",
        "off",
        "what",
        "when",
        "where",
        "which",
        "who",
        "whom",
        "whose",
        "why",
        "how",
        "all",
        "any",
        "both",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "nor",
        "not",
        "only",
        "own",
        "same",
        "so",
        "too",
        "very",
        "just",
        "document",
        "documents",
        "file",
        "files",
        "page",
        "pages",
        "report",
        "reports",
        "summary",
        "summarize",
        "policy",
        "policies",
        "rule",
        "rules",
        "law",
        "laws",
        "clause",
        "section",
        "chapter",
        "paragraph",
        "date",
        "time",
        "year",
        "month",
        "day",
        "deadline",
        "penalty",
        "fine",
        "fee",
        "amount",
        "total",
        "number",
        "name",
        "address",
        "email",
        "phone",
        "application",
        "form",
        "certificate",
        "licence",
        "license",
        "government",
        "office",
        "department",
        "ministry",
        "officer",
        "employee",
        "salary",
        "pension",
        "education",
        "school",
        "student",
        "teacher",
        "exam",
        "result",
        "admission",
        "course",
        "data",
        "table",
        "list",
        "detail",
        "details",
        "information",
        "content",
        "text",
        "please",
        "tell",
        "explain",
        "give",
        "show",
        "find",
        "search",
    ]
)

_TOKEN_RE = regex.compile(r"(\p{L}[\p{L}\p{M}']*|\d+|[^\p{L}\p{M}\d]+)")
_WORD_RE = regex.compile(r"^\p{L}[\p{L}\p{M}']*$")


def _to_itrans_spelling(token: str) -> str:
    """Nudge informal Hinglish spelling toward what ITRANS expects.

    Romanized Hindi drops vowel length that ITRANS needs: "kya" is written for
    "kyaa" (क्या), "shiksha" for "shikshaa" (शिक्षा). A word-final bare 'a' is
    almost always a long आ in Hinglish, so lengthening it fixes a large class of
    words at once.
    """
    word = token.lower()
    word = word.replace("ee", "ii").replace("oo", "uu").replace("w", "v")
    # sh -> ITRANS 'sh' is श; 'Sh' is ष. "ksh" is nearly always क्ष.
    word = word.replace("ksh", "kSh")
    if word.endswith("a") and not word.endswith("aa"):
        word += "a"
    return word


@functools.lru_cache(maxsize=4096)
def _transliterate_token(token: str, aggressive: bool) -> str:
    lowered = token.lower()

    if lowered in HINGLISH_LEXICON:
        return HINGLISH_LEXICON[lowered]
    if lowered in ENGLISH_GUARD:
        return token
    # Acronyms and identifiers ("RTI", "GST") stay verbatim.
    if token.isupper() and len(token) <= 5:
        return token
    if not aggressive:
        # Precision-first: an unknown token is far more likely an English word
        # than an unlisted Hindi one, and "submission" -> "सुब्मिस्सिओन्" is
        # noise that degrades the query vector. Leaving it romanized costs
        # nothing, because the untransliterated query is searched in parallel.
        return token

    try:
        from indic_transliteration import sanscript
        from indic_transliteration.sanscript import transliterate

        converted = transliterate(
            _to_itrans_spelling(token), sanscript.ITRANS, sanscript.DEVANAGARI
        )
    except Exception:
        log.warning("transliteration_failed", token=token, exc_info=True)
        return token

    # A conversion that produced no Devanagari is a failed conversion.
    return converted if has_devanagari(converted) else token


def to_devanagari(text: str, *, aggressive: bool = False) -> str:
    """Transliterate romanized Hindi in `text`, leaving English intact.

    ``aggressive`` additionally pushes unknown tokens through ITRANS. It raises
    coverage on Hindi content words the lexicon lacks, at the cost of mangling
    unrecognised English — off by default. Call this only on text already
    identified as Hinglish; a pure English query has nothing to gain and real
    homographs to lose ("the" is also Hindi *थे*, "is" is *इस*).
    """
    if not text or has_devanagari(text):
        # Already Devanagari (or mixed) — nothing to gain.
        return text

    return "".join(
        _transliterate_token(part, aggressive) if _WORD_RE.match(part) else part
        for part in _TOKEN_RE.findall(text)
    )


def lexicon_coverage(text: str) -> float:
    """Share of word tokens the curated lexicon recognises.

    Used to decide whether a transliteration is trustworthy enough to search as
    a separate query variant.
    """
    words = [p.lower() for p in _TOKEN_RE.findall(text) if _WORD_RE.match(p)]
    if not words:
        return 0.0
    known = sum(1 for w in words if w in HINGLISH_LEXICON)
    return known / len(words)
