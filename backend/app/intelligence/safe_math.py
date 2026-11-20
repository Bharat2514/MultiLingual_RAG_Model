"""A restricted arithmetic evaluator for formulas read out of documents.

LLMs are unreliable at arithmetic and reliable at *reading* a formula. So the
model's only job is to transcribe "pension = 50% of last basic pay + DA" into an
expression; this module computes it. That makes every numeric answer
reproducible and auditable — the user sees the formula, the inputs, and a
result that does not depend on sampling.

The evaluator walks the AST and allows only arithmetic, comparisons, a ternary
(for slab rules: ``0.05 * x if x <= 250000 else ...``) and a handful of pure
functions. Names must be declared variables. No attribute access, no
subscripts, no calls to anything else — `eval` is never used.
"""

from __future__ import annotations

import ast
import math
import operator
from collections.abc import Callable
from typing import Any

MAX_EXPRESSION_CHARS = 500
MAX_EXPONENT = 12
MAX_MAGNITUDE = 1e15


class FormulaError(ValueError):
    pass


_BINOPS: dict[type, Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY: dict[type, Callable[[Any], Any]] = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_COMPARE: dict[type, Callable[[Any, Any], bool]] = {
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
}
_FUNCTIONS: dict[str, Callable[..., float]] = {
    "min": min,
    "max": max,
    "abs": abs,
    "round": round,
    "floor": math.floor,
    "ceil": math.ceil,
}


def variables_in(expression: str) -> list[str]:
    tree = _parse(expression)
    names = sorted(
        (
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Name) and n.id not in _FUNCTIONS and n.id not in ("True", "False")
        ),
        key=lambda n: (n.lineno, n.col_offset),
    )
    # Source order, so the inputs are asked for in the order the formula reads.
    return list(dict.fromkeys(n.id for n in names))


def _parse(expression: str) -> ast.Expression:
    if not expression or len(expression) > MAX_EXPRESSION_CHARS:
        raise FormulaError("Formula is empty or too long.")
    try:
        return ast.parse(expression.replace("^", "**"), mode="eval")
    except SyntaxError as exc:
        raise FormulaError(f"Formula is not valid arithmetic: {exc.msg}") from exc


def evaluate(expression: str, variables: dict[str, float]) -> float:
    tree = _parse(expression)

    def check(value: Any) -> Any:
        if isinstance(value, bool):
            return value
        if not isinstance(value, int | float) or math.isnan(value) or abs(value) > MAX_MAGNITUDE:
            raise FormulaError("Result is out of range.")
        return value

    def visit(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool | int | float):
                return node.value
            raise FormulaError("Only numeric constants are allowed.")
        if isinstance(node, ast.Name):
            if node.id not in variables:
                raise FormulaError(f"Missing value for '{node.id}'.")
            return check(float(variables[node.id]))
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
                raise FormulaError("Exponent too large.")
            try:
                return check(_BINOPS[type(node.op)](left, right))
            except ZeroDivisionError as exc:
                raise FormulaError("Division by zero.") from exc
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            return check(_UNARY[type(node.op)](visit(node.operand)))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return not visit(node.operand)
        if isinstance(node, ast.BoolOp):
            values = [visit(v) for v in node.values]
            return all(values) if isinstance(node.op, ast.And) else any(values)
        if isinstance(node, ast.Compare):
            left = visit(node.left)
            for op, comparator in zip(node.ops, node.comparators, strict=True):
                if type(op) not in _COMPARE:
                    raise FormulaError("Unsupported comparison.")
                right = visit(comparator)
                if not _COMPARE[type(op)](left, right):
                    return False
                left = right
            return True
        if isinstance(node, ast.IfExp):
            return visit(node.body) if visit(node.test) else visit(node.orelse)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in _FUNCTIONS
            and not node.keywords
        ):
            return check(_FUNCTIONS[node.func.id](*[visit(a) for a in node.args]))
        raise FormulaError(f"'{type(node).__name__}' is not allowed in a formula.")

    result = visit(tree)
    if isinstance(result, bool):
        raise FormulaError("Formula evaluates to a condition, not an amount.")
    return float(result)


def substitute(expression: str, variables: dict[str, float]) -> str:
    """The expression with values in place of names — the "working" shown to users."""
    tree = _parse(expression)

    class _Sub(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name) -> ast.AST:
            if node.id in variables:
                return ast.copy_location(ast.Constant(variables[node.id]), node)
            return node

    return ast.unparse(_Sub().visit(tree))
