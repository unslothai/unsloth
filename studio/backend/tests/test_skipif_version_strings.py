# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A skipif / xfail condition must not end in a bare ``torch.version.hip`` / ``torch.version.cuda``.

Those are strings when set, and pytest evaluates a string condition as Python source: on a ROCm host
``"7.13.99004"`` is a SyntaxError, so the test errors at setup instead of skipping (seen on a two-device ROCm run).
Wrap them in ``bool()`` or compare with ``is not None``."""

from __future__ import annotations

import ast
import re
import sys
import warnings
from pathlib import Path
from typing import NamedTuple

import pytest

_TESTS = Path(__file__).resolve().parent
_VERSION_ATTRS = ("hip", "cuda")
# Only the spellings the scanner resolves: <torch>.version.hip / .cuda, ``from torch import version`` (parenthesised or
# aliased too), ``import torch.version as v`` and ``from torch.version import ...``. Checked before parsing so ~1,800
# modules stay cheap.
_CANDIDATE = re.compile(
    r"version\.(?:hip|cuda)\b|from\s+torch\s+import\s*\(?[\w\s,]*\bversion\b"
    r"|from\s+torch\.version\s+import|import\s+torch\.version\s+as\b"
)


class _Bindings(NamedTuple):
    torch: frozenset[str] = frozenset({"torch"})  # names bound to the torch module
    version: frozenset[str] = frozenset()  # names bound to torch.version
    values: frozenset[str] = frozenset()  # names bound to the version strings themselves


def _imported_names(tree: ast.AST, module: str, name: str) -> set[str]:
    """Local names bound by ``from <module> import <name> [as x]``."""
    return {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == module
        for alias in node.names
        if alias.name == name
    }


def _module_aliases(tree: ast.AST, module: str) -> set[str]:
    """Names bound to ``module`` itself by ``import <module> as x`` (and ``import torch`` for torch)."""
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == module:
                    names.add(alias.asname or alias.name.split(".")[0])
    return names


def _bindings(tree: ast.AST) -> _Bindings:
    torch_names = frozenset({"torch"} | _module_aliases(tree, "torch"))
    version = frozenset(
        _imported_names(tree, "torch", "version")
        | (_module_aliases(tree, "torch.version") - {"torch"})
    )
    values = {n for attr in _VERSION_ATTRS for n in _imported_names(tree, "torch.version", attr)}
    # ``hip = torch.version.hip`` (chains too): the name now holds the string.
    assigns = [
        (target.id, node.value)
        for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None
        for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
        if isinstance(target, ast.Name)
    ]
    while True:
        bound = _Bindings(torch_names, version, frozenset(values))
        grown = {
            name for name, value in assigns if name not in values and _string_operands(value, bound)
        }
        if not grown:
            return bound
        values |= grown


def _is_version_string(node: ast.expr, bound: _Bindings) -> bool:
    if isinstance(node, ast.Name):
        return node.id in bound.values
    if not (isinstance(node, ast.Attribute) and node.attr in _VERSION_ATTRS):
        return False
    owner = node.value
    # <torch alias>.version.hip / .cuda, or <bound torch.version>.hip / .cuda; driver.version.hip is not torch's
    return (
        isinstance(owner, ast.Attribute)
        and owner.attr == "version"
        and isinstance(owner.value, ast.Name)
        and owner.value.id in bound.torch
    ) or (isinstance(owner, ast.Name) and owner.id in bound.version)


def _string_operands(condition: ast.expr, bound: _Bindings = _Bindings()) -> list[str]:
    """Operands a condition can evaluate to as a version string: ``or`` can return any operand, ``and`` only its last
    one, a conditional expression either branch; ``str(...)`` and f-strings around a version string are strings too."""
    if isinstance(condition, ast.BoolOp):
        values = condition.values if isinstance(condition.op, ast.Or) else condition.values[-1:]
        return [found for value in values for found in _string_operands(value, bound)]
    if isinstance(condition, ast.NamedExpr):
        return _string_operands(condition.value, bound)
    if isinstance(condition, ast.IfExp):
        return [
            *_string_operands(condition.body, bound),
            *_string_operands(condition.orelse, bound),
        ]
    wrapped = []
    if isinstance(condition, ast.JoinedStr):
        wrapped = [v.value for v in condition.values if isinstance(v, ast.FormattedValue)]
    elif (
        isinstance(condition, ast.Call)
        and isinstance(condition.func, ast.Name)
        and condition.func.id == "str"
        and condition.args
    ):
        wrapped = condition.args[:1]
    if wrapped:
        return [ast.unparse(condition)] if any(_string_operands(w, bound) for w in wrapped) else []
    return [ast.unparse(condition)] if _is_version_string(condition, bound) else []


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
    aliased = _Bindings(torch = frozenset({"t"}))
    assert _string_operands(ast.parse("t.version.cuda or x", mode = "eval").body, aliased) == [
        "t.version.cuda"
    ]
    assert _string_operands(ast.parse("driver.version.hip", mode = "eval").body) == []


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
        "def test_d(): pass\n"
        "from torch.version import hip as hv\n"
        "@pytest.mark.skipif(hv, reason = 'x')\n"
        "def test_e(): pass\n",
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
    assert "test_more.py:13: hv" in str(offenders)  # from torch.version import hip


def test_local_aliases_wrappers_and_foreign_version_attrs(tmp_path, monkeypatch):
    bad = tmp_path / "test_alias.py"
    bad.write_text(
        "import pytest\n"
        "import torch as t\n"
        "import torch.version as tv\n"
        "hip = t.version.hip\n"
        "rocm: str = hip\n"
        "@pytest.mark.skipif(rocm, reason = 'x')\n"
        "def test_a(): pass\n"
        "@pytest.mark.skipif(f'{t.version.hip}', reason = 'x')\n"
        "def test_b(): pass\n"
        "@pytest.mark.skipif(str(tv.cuda), reason = 'x')\n"
        "def test_c(): pass\n"
        "@pytest.mark.skipif(driver.version.hip, reason = 'x')\n"
        "def test_d(): pass\n"
        "@pytest.mark.skipif(bool(hip), reason = 'x')\n"
        "def test_e(): pass\n",
        encoding = "utf-8",
    )
    monkeypatch.setattr(sys.modules[__name__], "_TESTS", tmp_path)
    with pytest.raises(AssertionError) as info:
        test_no_skip_condition_can_evaluate_to_a_version_string()
    offenders = str(info.value)
    assert "test_alias.py:6: rocm" in offenders  # assignment chain
    assert "test_alias.py:8:" in offenders  # f-string
    assert "test_alias.py:10: str(tv.cuda)" in offenders  # import torch.version as tv, str()
    assert "test_alias.py:12" not in offenders  # not a torch binding
    assert "test_alias.py:14" not in offenders  # bool() is fine


def test_prefilter_keeps_every_resolved_spelling():
    for text in (
        "torch.version.hip",
        "t.version.cuda",
        "from torch import version as v",
        "from torch import (\n    cuda,\n    version,\n)",
        "from torch.version import hip",
        "import torch.version as tv",
    ):
        assert _CANDIDATE.search(text), text
    assert not _CANDIDATE.search("from packaging import version\nversion.parse('1')")


def test_no_skip_condition_can_evaluate_to_a_version_string():
    offenders = []
    for path in sorted(_TESTS.rglob("*.py")):
        source = path.read_text(encoding = "utf-8", errors = "replace")
        if not ("skipif" in source or "xfail" in source) or not _CANDIDATE.search(source):
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)  # other files' escape sequences
                tree = ast.parse(source)
        except SyntaxError:
            continue
        bound = _bindings(tree)
        marks = _imported_names(tree, "pytest", "mark")
        for node in ast.walk(tree):
            for operand in (
                found
                for c in _marker_conditions(node, marks)
                for found in _string_operands(c, bound)
            ):
                offenders.append(f"{path.relative_to(_TESTS)}:{node.lineno}: {operand}")
    assert not offenders, offenders
