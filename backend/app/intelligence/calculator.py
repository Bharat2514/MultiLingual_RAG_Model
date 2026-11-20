"""Document-grounded calculations: the model reads the formula, code computes it.

Flow for "how much pension will I get with basic pay 50,000?":

1. retrieve the passages that state the rule (normal hybrid retrieval);
2. the model transcribes the formula into an expression over named variables,
   reporting values the document itself fixes (rates, caps) and quoting the
   clause — it never does the arithmetic;
3. values the user must supply are requested if absent;
4. `safe_math` evaluates the expression, and the response shows the formula,
   the substituted working and the clause, so every number is reproducible.
"""

from __future__ import annotations

from app.generation.providers.base import LLMProvider
from app.intelligence import llm_json, safe_math
from app.retrieval.types import ScoredChunk

SYSTEM = (
    """You read formulas and rates out of documents so that a program can compute
them. You never compute results yourself. """
    + llm_json.JSON_RULES
)

USER_TEMPLATE = """CONTEXT
{context}

QUESTION: {question}

TASK
Find the rule in the CONTEXT that computes what the question asks for and
transcribe it as JSON:
{{
  "found": true | false,
  "target": short name of the quantity being computed,
  "expression": a Python arithmetic expression over snake_case variable names,
                e.g. "0.5 * last_basic_pay + dearness_allowance".
                Percentages become decimals (50% -> 0.5). Use min()/max() for caps,
                and "a if condition else b" for slabs,
  "variables": [
    {{"name": snake_case, "description": string, "unit": string | null,
      "value": number if the DOCUMENT fixes it (e.g. a rate or cap) else null}}
  ],
  "result_unit": string | null,
  "source_index": passage number,
  "quote": verbatim span stating the rule
}}
If the CONTEXT contains no such rule, return {{"found": false}}."""


def _number(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    try:
        return float(str(value).replace(",", "").strip())
    except ValueError:
        return None


async def calculate(
    provider: LLMProvider,
    question: str,
    chunks: list[ScoredChunk],
    inputs: dict[str, float],
    *,
    redact_pii: bool,
) -> dict:
    user = USER_TEMPLATE.format(
        context=llm_json.build_context(chunks, redact_pii=redact_pii), question=question
    )
    result = await llm_json.generate_json(provider, SYSTEM, user)
    data = result.data

    if not data.get("found") or not data.get("expression"):
        return {"status": "not_found", "model": result.model}

    expression = str(data["expression"])
    chunk = llm_json.resolve_source(data.get("source_index"), chunks)
    source = llm_json.source_ref(chunk, data.get("quote"))

    try:
        names = safe_math.variables_in(expression)
    except safe_math.FormulaError as exc:
        return {
            "status": "invalid_formula",
            "expression": expression,
            "error": str(exc),
            "source": source,
        }

    declared = {
        str(v.get("name")): v
        for v in data.get("variables") or []
        if isinstance(v, dict) and v.get("name")
    }
    values: dict[str, float] = {}
    variables: list[dict] = []
    for name in names:
        meta = declared.get(name, {})
        from_document = _number(meta.get("value"))
        provided = _number(inputs.get(name))
        value = provided if provided is not None else from_document
        if value is not None:
            values[name] = value
        variables.append(
            {
                "name": name,
                "description": meta.get("description") or name.replace("_", " "),
                "unit": meta.get("unit"),
                "value": value,
                "origin": "user"
                if provided is not None
                else "document"
                if from_document is not None
                else None,
            }
        )

    payload = {
        "target": data.get("target"),
        "expression": expression,
        "variables": variables,
        "result_unit": data.get("result_unit"),
        "source": source,
        "model": result.model,
    }

    missing = [v["name"] for v in variables if v["value"] is None]
    if missing:
        return {**payload, "status": "needs_inputs", "missing": missing}

    try:
        value = safe_math.evaluate(expression, values)
    except safe_math.FormulaError as exc:
        return {**payload, "status": "invalid_formula", "error": str(exc)}

    return {
        **payload,
        "status": "computed",
        "result": round(value, 2),
        "working": safe_math.substitute(expression, values),
    }
