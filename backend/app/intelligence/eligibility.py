"""Scheme eligibility: rules read from the document, evaluated as code.

"Am I eligible?" is the question every reader of a scheme circular has, and a
chat answer to it is unsatisfying in both directions — the model either hedges
or confidently applies a rule it half-read. Here the work is split:

1. **Extraction (LLM, once per document).** The model transcribes each
   eligibility condition into a typed rule — field, operator, value — and
   quotes the clause it came from. Rules whose quote is not found verbatim in
   the cited passage are kept but marked unverified.
2. **Evaluation (code, every time).** A user profile is checked against the
   rules deterministically. Each rule reports pass / fail / unknown with its
   clause and page, so a verdict is always explainable line by line.

A missing profile field yields `unknown`, never a guess, and the overall
verdict is `incomplete` until every rule can be decided.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from app.core.errors import InvalidModelOutputError
from app.generation.providers.base import LLMProvider
from app.intelligence import llm_json
from app.retrieval.types import ScoredChunk

OPERATORS = {
    "eq",
    "neq",
    "lt",
    "lte",
    "gt",
    "gte",
    "in",
    "not_in",
    "between",
    "is_true",
    "is_false",
}

#: Canonical profile fields the UI offers. Rules may use others; those simply
#: show up as extra inputs.
PROFILE_FIELDS: dict[str, str] = {
    "age": "number",
    "gender": "text",
    "annual_income": "number",
    "category": "text",
    "state": "text",
    "occupation": "text",
    "land_holding_acres": "number",
    "years_of_service": "number",
    "is_farmer": "boolean",
    "is_government_employee": "boolean",
    "is_bpl": "boolean",
    "has_bank_account": "boolean",
    "is_student": "boolean",
    "education_level": "text",
    "disability_percent": "number",
}

SYSTEM = (
    """You extract eligibility conditions from government scheme, policy and tender
documents so they can be checked by a program. """
    + llm_json.JSON_RULES
)

USER_TEMPLATE = """CONTEXT
{context}

TASK
List every eligibility condition an applicant must meet, as JSON:
{{
  "scheme_name": string | null,
  "rules": [
    {{
      "field": snake_case profile field — prefer one of: {fields},
      "label": short human description of the condition in the document's language,
      "operator": one of "eq","neq","lt","lte","gt","gte","in","not_in","between","is_true","is_false",
      "value": number | string | [values] (for in/not_in) | [min, max] (for between) | null (for is_true/is_false),
      "unit": string | null  (e.g. "years", "INR", "acres"),
      "source_index": passage number,
      "quote": verbatim span stating the condition
    }}
  ],
  "required_documents": [{{"name": string, "source_index": number, "quote": string}}],
  "benefits": [{{"text": string, "source_index": number, "quote": string}}]
}}
Convert amounts to plain numbers in rupees (e.g. "2.5 lakh" -> 250000). If the
document states no eligibility conditions, return "rules": []."""


@dataclass(slots=True)
class RuleResult:
    field: str
    label: str
    operator: str
    value: object
    unit: str | None
    outcome: str  # pass | fail | unknown
    provided: object
    source: dict | None

    def to_dict(self) -> dict:
        return asdict(self)


def _clean_rules(raw: dict, chunks: list[ScoredChunk]) -> dict:
    rules: list[dict] = []
    for item in raw.get("rules") or []:
        if not isinstance(item, dict):
            continue
        field = str(item.get("field") or "").strip().lower().replace(" ", "_")
        operator = str(item.get("operator") or "").strip().lower()
        if not field or operator not in OPERATORS:
            continue
        chunk = llm_json.resolve_source(item.get("source_index"), chunks)
        rules.append(
            {
                "field": field,
                "label": str(item.get("label") or field)[:300],
                "operator": operator,
                "value": item.get("value"),
                "unit": item.get("unit"),
                "source": llm_json.source_ref(chunk, item.get("quote")),
            }
        )

    def refs(key: str, name_key: str) -> list[dict]:
        out = []
        for item in raw.get(key) or []:
            if isinstance(item, dict) and item.get(name_key):
                chunk = llm_json.resolve_source(item.get("source_index"), chunks)
                out.append(
                    {
                        name_key: str(item[name_key])[:300],
                        "source": llm_json.source_ref(chunk, item.get("quote")),
                    }
                )
        return out

    return {
        "scheme_name": raw.get("scheme_name"),
        "rules": rules,
        "required_documents": refs("required_documents", "name"),
        "benefits": refs("benefits", "text"),
        "verified_rules": sum(1 for r in rules if r["source"] and r["source"]["quote_verified"]),
    }


async def extract_rules(
    provider: LLMProvider, chunks: list[ScoredChunk], *, redact_pii: bool
) -> dict:
    if not chunks:
        raise InvalidModelOutputError("The document has no text to read rules from.")
    user = USER_TEMPLATE.format(
        context=llm_json.build_context(chunks, redact_pii=redact_pii),
        fields=", ".join(PROFILE_FIELDS),
    )
    result = await llm_json.generate_json(provider, SYSTEM, user, max_tokens=3000)
    cleaned = _clean_rules(result.data, chunks)
    cleaned["model"] = result.model
    return cleaned


# --- evaluation ----------------------------------------------------------------


def _number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.replace(",", "").strip())
        except ValueError:
            return None
    return None


def _truthy(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"true", "yes", "y", "1", "haan", "हाँ", "हां"}:
            return True
        if v in {"false", "no", "n", "0", "nahi", "नहीं"}:
            return False
    if isinstance(value, int | float):
        return bool(value)
    return None


def _text(value: object) -> str:
    return str(value).strip().casefold()


def check_rule(rule: dict, profile: dict) -> str:
    field, op, expected = rule["field"], rule["operator"], rule.get("value")
    if field not in profile or profile[field] in (None, ""):
        return "unknown"
    actual = profile[field]

    if op in ("is_true", "is_false"):
        flag = _truthy(actual)
        if flag is None:
            return "unknown"
        return "pass" if flag == (op == "is_true") else "fail"

    if op in ("lt", "lte", "gt", "gte", "between"):
        value = _number(actual)
        if value is None:
            return "unknown"
        if op == "between":
            if not isinstance(expected, list) or len(expected) != 2:
                return "unknown"
            low, high = _number(expected[0]), _number(expected[1])
            if low is None or high is None:
                return "unknown"
            return "pass" if low <= value <= high else "fail"
        bound = _number(expected)
        if bound is None:
            return "unknown"
        ok = {
            "lt": value < bound,
            "lte": value <= bound,
            "gt": value > bound,
            "gte": value >= bound,
        }[op]
        return "pass" if ok else "fail"

    if op in ("in", "not_in"):
        options = expected if isinstance(expected, list) else [expected]
        member = _text(actual) in {_text(o) for o in options}
        return "pass" if member == (op == "in") else "fail"

    # eq / neq: numeric when both sides are numbers, else case-insensitive text.
    a, b = _number(actual), _number(expected)
    equal = a == b if a is not None and b is not None else _text(actual) == _text(expected)
    return "pass" if equal == (op == "eq") else "fail"


def evaluate(rules: list[dict], profile: dict) -> dict:
    results = [
        RuleResult(
            field=r["field"],
            label=r["label"],
            operator=r["operator"],
            value=r.get("value"),
            unit=r.get("unit"),
            outcome=check_rule(r, profile),
            provided=profile.get(r["field"]),
            source=r.get("source"),
        )
        for r in rules
    ]
    outcomes = [r.outcome for r in results]
    if not results:
        verdict = "no_rules"
    elif "fail" in outcomes:
        verdict = "not_eligible"
    elif "unknown" in outcomes:
        verdict = "incomplete"
    else:
        verdict = "eligible"
    return {
        "verdict": verdict,
        "passed": outcomes.count("pass"),
        "failed": outcomes.count("fail"),
        "unknown": outcomes.count("unknown"),
        "missing_fields": sorted({r.field for r in results if r.outcome == "unknown"}),
        "rules": [r.to_dict() for r in results],
    }
