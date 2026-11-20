"""Deterministic document intelligence: deadlines, versions, formulas, rules."""

from __future__ import annotations

from datetime import date

import pytest

from app.intelligence import deadlines, eligibility, safe_math, versions


def _c(text: str, page: int = 1, chunk_id: str = "c1") -> dict:
    return {"text": text, "page": page, "chunk_id": chunk_id, "doc_id": "d", "filename": "n.pdf"}


class TestDates:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("on 31/03/2025", date(2025, 3, 31)),
            ("on 31.03.2025", date(2025, 3, 31)),
            ("on 31st March, 2025", date(2025, 3, 31)),
            ("on March 31, 2025", date(2025, 3, 31)),
            ("दिनांक ३१ मार्च २०२५", date(2025, 3, 31)),
            ("15 सितम्बर 2024 तक", date(2024, 9, 15)),
        ],
    )
    def test_parses_indian_date_forms(self, text, expected):
        assert deadlines.parse_dates(text) == [expected]

    def test_impossible_dates_are_skipped(self):
        assert deadlines.parse_dates("on 31/02/2025") == []

    def test_deadline_and_issue_date_in_one_sentence_are_told_apart(self):
        items = deadlines.extract(
            [_c("Bids must be submitted by 15.04.2025 as per circular dated 01.01.2025.")]
        )
        kinds = {i.date: i.kind for i in items}
        assert kinds["2025-04-15"] == "deadline"
        assert kinds["2025-01-01"] == "issued"

    def test_issuer_by_is_not_a_deadline(self):
        items = deadlines.extract([_c("The scheme was sanctioned by the Ministry on 5 May 2024.")])
        assert items[0].kind != "deadline"

    def test_hindi_deadline_and_effective_date(self):
        items = deadlines.extract(
            [
                _c("आवेदन 30 जून 2025 तक जमा करें।"),
                _c("यह नियम 1 अप्रैल 2025 से प्रभावी होगा।", chunk_id="c2"),
            ]
        )
        assert [(i.kind, i.date) for i in items] == [
            ("effective", "2025-04-01"),
            ("deadline", "2025-06-30"),
        ]

    def test_penalty_with_amount_and_undated_obligation(self):
        items = deadlines.extract(
            [
                _c(
                    "A late fee of Rs. 500 per day shall be charged. "
                    "Every applicant must attach a caste certificate."
                )
            ]
        )
        penalty = next(i for i in items if i.kind == "penalty")
        assert penalty.amount == "Rs. 500"
        assert any(i.kind == "obligation" for i in items)

    def test_overlapping_chunks_do_not_duplicate_items(self):
        text = "Last date for submission is 30.06.2025."
        items = deadlines.extract([_c(text), _c(text, chunk_id="c2")])
        assert len(items) == 1

    def test_ics_has_one_all_day_event_per_actionable_date(self):
        items = deadlines.extract([_c("Submit by 30.06.2025, as per letter dated 01.01.2025.")])
        ics = deadlines.to_ics(items)
        assert ics.startswith("BEGIN:VCALENDAR\r\n")
        assert ics.count("BEGIN:VEVENT") == 1  # "issued" dates are not calendar events
        assert "DTSTART;VALUE=DATE:20250630" in ics
        assert all(len(line.encode()) <= 75 for line in ics.split("\r\n"))


class TestVersions:
    OLD = [
        _c(
            "Full pension requires a minimum of 20 years of qualifying service. "
            "Applications go to the district office. "
            "Medical allowance is paid monthly.",
            page=2,
        )
    ]
    NEW = [
        _c(
            "Full pension requires a minimum of 15 years of qualifying service. "
            "Applications go to the district office. "
            "Family pension is payable to the spouse.",
            page=3,
        )
    ]

    def test_reworded_clause_is_modified_with_number_change(self):
        result = versions.diff_documents(self.OLD, self.NEW)
        first = result["changes"][0]
        assert first["kind"] == "modified"
        assert first["number_changes"] == [{"old": "20", "new": "15"}]
        assert result["summary"]["unchanged"] == 1

    def test_added_and_removed_sentences(self):
        result = versions.diff_documents(self.OLD, self.NEW)
        kinds = sorted(c["kind"] for c in result["changes"])
        assert kinds == ["added", "modified", "removed"]

    def test_identical_documents_have_no_changes(self):
        assert versions.diff_documents(self.OLD, self.OLD)["changes"] == []

    def test_conflict_between_documents(self):
        conflicts = versions.find_conflicts(self.OLD, self.NEW)
        assert len(conflicts) == 1
        assert conflicts[0]["number_changes"] == [{"old": "20", "new": "15"}]


class TestSafeMath:
    def test_arithmetic_and_functions(self):
        assert safe_math.evaluate("0.5 * pay + da", {"pay": 50000, "da": 2000}) == 27000
        assert safe_math.evaluate("min(x * 0.1, 5000)", {"x": 80000}) == 5000
        assert safe_math.evaluate("x ^ 2", {"x": 3}) == 9

    def test_slab_ternary(self):
        expr = "0 if income <= 250000 else (income - 250000) * 0.05"
        assert safe_math.evaluate(expr, {"income": 200000}) == 0
        assert safe_math.evaluate(expr, {"income": 300000}) == 2500

    @pytest.mark.parametrize(
        "expr",
        [
            "__import__('os').system('ls')",
            "x.__class__",
            "[1, 2][0]",
            "open('f')",
            "x ** 100",
            "'a' * 3",
            "lambda: 1",
        ],
    )
    def test_rejects_anything_but_arithmetic(self, expr):
        with pytest.raises(safe_math.FormulaError):
            safe_math.evaluate(expr, {"x": 2})

    def test_missing_variable_and_division_by_zero(self):
        with pytest.raises(safe_math.FormulaError, match="Missing value"):
            safe_math.evaluate("a + b", {"a": 1})
        with pytest.raises(safe_math.FormulaError, match="Division by zero"):
            safe_math.evaluate("a / b", {"a": 1, "b": 0})

    def test_variables_and_working(self):
        assert safe_math.variables_in("max(a, b) * rate") == ["a", "b", "rate"]
        assert safe_math.substitute("0.5 * pay", {"pay": 100.0}) == "0.5 * 100.0"


class TestEligibilityEvaluation:
    RULES = [
        {"field": "age", "label": "18–40", "operator": "between", "value": [18, 40]},
        {"field": "annual_income", "label": "≤ 2.5 lakh", "operator": "lte", "value": 250000},
        {"field": "is_farmer", "label": "farmer", "operator": "is_true", "value": None},
        {"field": "category", "label": "SC/ST", "operator": "in", "value": ["SC", "ST"]},
    ]

    def test_eligible(self):
        result = eligibility.evaluate(
            self.RULES,
            {"age": 30, "annual_income": "1,80,000", "is_farmer": "haan", "category": "sc"},
        )
        assert result["verdict"] == "eligible"
        assert result["passed"] == 4

    def test_any_failure_is_not_eligible(self):
        result = eligibility.evaluate(
            self.RULES, {"age": 45, "annual_income": 100000, "is_farmer": True, "category": "ST"}
        )
        assert result["verdict"] == "not_eligible"
        assert result["rules"][0]["outcome"] == "fail"

    def test_missing_fields_are_unknown_not_guessed(self):
        result = eligibility.evaluate(self.RULES, {"age": 30})
        assert result["verdict"] == "incomplete"
        assert result["missing_fields"] == ["annual_income", "category", "is_farmer"]

    def test_rules_from_model_output_are_validated_against_the_source(self):
        from app.retrieval.types import ScoredChunk

        chunk = ScoredChunk(
            chunk_id="c1",
            doc_id="d",
            text="Applicants must be between 18 and 40 years of age.",
            score=0,
            filename="scheme.pdf",
            page_start=2,
            page_end=2,
            chunk_index=0,
        )
        raw = {
            "rules": [
                {
                    "field": "Age",
                    "operator": "between",
                    "value": [18, 40],
                    "source_index": 1,
                    "quote": "between 18 and 40 years of age",
                },
                {"field": "income", "operator": "approximately", "value": 1},  # bad operator
                {
                    "field": "state",
                    "operator": "eq",
                    "value": "Bihar",
                    "source_index": 1,
                    "quote": "resident of Bihar",  # not in the passage
                },
            ]
        }
        cleaned = eligibility._clean_rules(raw, [chunk])
        assert [r["field"] for r in cleaned["rules"]] == ["age", "state"]
        assert cleaned["rules"][0]["source"]["quote_verified"] is True
        assert cleaned["rules"][1]["source"]["quote_verified"] is False
        assert cleaned["verified_rules"] == 1
