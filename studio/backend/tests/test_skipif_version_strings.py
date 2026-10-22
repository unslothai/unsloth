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
    values = condition.values if isinstance(condition, ast.BoolOp) else [condition]
    return [ast.unparse(v) for v in values if ast.unparse(v) in _VERSION_STRINGS]


def test_no_skip_condition_can_evaluate_to_a_version_string():
    offenders = []
    for path in sorted(_TESTS.glob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding = "utf-8"))
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
                    offenders.append(f"{path.name}:{node.lineno}: {operand}")
    assert not offenders, offenders
