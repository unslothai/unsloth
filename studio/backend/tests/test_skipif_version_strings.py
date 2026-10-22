# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A skipif / xfail condition must not end in a bare ``torch.version.hip`` / ``torch.version.cuda``.

Those are strings when set, and pytest evaluates a string condition as Python source: on a ROCm host
``"7.13.99004"`` is a SyntaxError, so the test errors at setup instead of skipping (seen on a two-device ROCm run).
Wrap them in ``bool()`` or compare with ``is not None``."""

from __future__ import annotations

import ast
import sys
import warnings
from pathlib import Path

import pytest

_TESTS = Path(__file__).resolve().parent
_VERSION_ATTRS = ("hip", "cuda")


def _imported_names(tree: ast.AST, module: str, name: str) -> set[str]:
    """Local names bound by ``from <module> import <name> [as x]``."""
    return {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == module
        for alias in node.names
        if alias.name == name
    }


def _version_names(tree: ast.AST) -> set[str]:
    """Names bound to ``torch.version`` by ``from torch import version [as v]``."""
    return {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "torch"
        for alias in node.names
        if alias.name == "version"
    }


def _is_version_string(node: ast.expr, version_names: set[str]) -> bool:
    # <torch or any alias>.version.hip / .cuda, or <from-imported version>.hip / .cuda
    if not (isinstance(node, ast.Attribute) and node.attr in _VERSION_ATTRS):
        return False
    owner = node.value
    return (isinstance(owner, ast.Attribute) and owner.attr == "version") or (
        isinstance(owner, ast.Name) and owner.id in version_names
    )


def _string_operands(condition: ast.expr, version_names: set[str] = frozenset()) -> list[str]:
    """Operands a condition can evaluate to: ``or`` can return any operand, ``and`` only its last one, a
    conditional expression either branch."""
    if isinstance(condition, ast.BoolOp):
        values = condition.values if isinstance(condition.op, ast.Or) else condition.values[-1:]
        return [found for value in values for found in _string_operands(value, version_names)]
    if isinstance(condition, ast.NamedExpr):
        return _string_operands(condition.value, version_names)
    if isinstance(condition, ast.IfExp):
        return [
            *_string_operands(condition.body, version_names),
            *_string_operands(condition.orelse, version_names),
        ]
    return [ast.unparse(condition)] if _is_version_string(condition, version_names) else []


def _marker_conditions(node: ast.AST, mark_names: set[str] = frozenset()) -> list[ast.expr]:
    """Conditions of a ``*.mark.skipif`` / ``*.mark.xfail`` marker; ``condition=`` overrides positionals."""
    if not (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in ("skipif", "xfail")
        and (
            (isinstance(node.func.value, ast.Attribute) and node.func.value.attr == "mark")
            or (isinstance(node.func.value, ast.Name) and node.func.value.id in mark_names)
        )
    ):
        return []
    keyword = [k.value for k in node.keywords if k.arg == "condition"]
    return keyword or list(node.args)


def test_nested_boolean_operands_are_inspected():
    tree = ast.parse(
        "torch.cuda.is_available() and (torch.cuda.device_count() < 2 or torch.version.hip)",
        mode = "eval",
    )
    assert _string_operands(tree.body) == ["torch.version.hip"]
    assert _string_operands(ast.parse("bool(torch.version.hip)", mode = "eval").body) == []
    assert (
        _string_operands(ast.parse("torch.version.hip and unsupported_arch", mode = "eval").body)
        == []
    )
    assert _string_operands(ast.parse("t.version.cuda or x", mode = "eval").body) == [
        "t.version.cuda"
    ]


def test_every_condition_argument_is_inspected(tmp_path, monkeypatch):
    bad = tmp_path / "test_bad.py"
    bad.write_text(
        "import pytest, torch\n"
        "@pytest.mark.skipif(False, torch.version.hip, reason = 'x')\n"
        "def test_a(): pass\n"
        "@pytest.mark.xfail(condition = torch.version.cuda, reason = 'x')\n"
        "def test_b(): pass\n",
        encoding = "utf-8",
    )
    monkeypatch.setattr(sys.modules[__name__], "_TESTS", tmp_path)
    with pytest.raises(AssertionError) as info:
        test_no_skip_condition_can_evaluate_to_a_version_string()
    assert "test_bad.py:2: torch.version.hip" in str(info.value)
    assert "test_bad.py:4: torch.version.cuda" in str(info.value)


def test_ternaries_from_imports_and_marker_semantics(tmp_path, monkeypatch):
    bad = tmp_path / "test_more.py"
    bad.write_text(
        "import pytest\n"
        "from torch import version as v\n"
        "@pytest.mark.skipif(v.hip if flag else False, reason = 'x')\n"
        "def test_a(): pass\n"
        "@pytest.mark.skipif(v.hip, condition = False, reason = 'x')\n"
        "def test_b(): pass\n"
        "def test_c():\n"
        "    pytest.xfail(v.hip)\n"
        "from pytest import mark as m\n"
        "@m.skipif((h := v.cuda), reason = 'x')\n"
        "def test_d(): pass\n",
        encoding = "utf-8",
    )
    monkeypatch.setattr(sys.modules[__name__], "_TESTS", tmp_path)
    with pytest.raises(AssertionError) as info:
        test_no_skip_condition_can_evaluate_to_a_version_string()
    offenders = info.value.args[0] if info.value.args else str(info.value)
    assert "test_more.py:3: v.hip" in str(offenders)
    assert "test_more.py:5" not in str(offenders)  # condition= wins over positionals
    assert "test_more.py:8" not in str(offenders)  # imperative pytest.xfail takes a reason
    assert "test_more.py:10: v.cuda" in str(offenders)  # from pytest import mark, walrus


def test_no_skip_condition_can_evaluate_to_a_version_string():
    offenders = []
    for path in sorted(_TESTS.rglob("*.py")):
        source = path.read_text(encoding = "utf-8", errors = "replace")
        if "version" not in source:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)  # other files' escape sequences
                tree = ast.parse(source)
        except SyntaxError:
            continue
        names = _version_names(tree)
        marks = _imported_names(tree, "pytest", "mark")
        for node in ast.walk(tree):
            for operand in (
                found
                for c in _marker_conditions(node, marks)
                for found in _string_operands(c, names)
            ):
                offenders.append(f"{path.relative_to(_TESTS)}:{node.lineno}: {operand}")
    assert not offenders, offenders
