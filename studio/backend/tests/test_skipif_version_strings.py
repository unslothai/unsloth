# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A skipif / xfail condition must not end in a bare ``torch.version.hip`` / ``torch.version.cuda``.

Those are strings when set, and pytest evaluates a string condition as Python source: on a ROCm host
``"7.13.99004"`` is a SyntaxError, so the test errors at setup instead of skipping (seen on a two-device ROCm run).
Wrap them in ``bool()`` or compare with ``is not None``."""

from __future__ import annotations

import ast
from pathlib import Path

_TESTS = Path(__file__).resolve().parent
_VERSION_STRINGS = {"torch.version.hip", "torch.version.cuda"}


def _string_operands(condition: ast.expr) -> list[str]:
    """Operands a condition can evaluate to: ``and`` / ``or`` return one of their operands, nested ones included."""
    if isinstance(condition, ast.BoolOp):
        return [found for value in condition.values for found in _string_operands(value)]
    text = ast.unparse(condition)
    return [text] if text in _VERSION_STRINGS else []


def test_nested_boolean_operands_are_inspected():
    tree = ast.parse(
        "torch.cuda.is_available() and (torch.cuda.device_count() < 2 or torch.version.hip)",
        mode = "eval",
    )
    assert _string_operands(tree.body) == ["torch.version.hip"]
    assert _string_operands(ast.parse("bool(torch.version.hip)", mode = "eval").body) == []


def test_no_skip_condition_can_evaluate_to_a_version_string():
    offenders = []
    for path in sorted(_TESTS.rglob("*.py")):
        source = path.read_text(encoding = "utf-8", errors = "replace")
        if "torch.version" not in source:
            continue
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in ("skipif", "xfail")
                and node.args
            ):
                for operand in _string_operands(node.args[0]):
                    offenders.append(f"{path.relative_to(_TESTS)}:{node.lineno}: {operand}")
    assert not offenders, offenders
