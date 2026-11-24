# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Studio installs on Python 3.9 and 3.10, where possessive quantifiers and atomic groups (3.11+) are
a re.error at import. CI runs 3.11+, so this parses every literal pattern and rejects that syntax."""

import ast
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 11), reason = "the 3.11 parser is what can name the 3.11-only opcodes"
)

_BACKEND = Path(__file__).resolve().parents[1]
_RE_FUNCTIONS = {
    "compile",
    "match",
    "search",
    "fullmatch",
    "finditer",
    "findall",
    "sub",
    "subn",
    "split",
}


def _uses_311_syntax(parsed) -> bool:
    from re import _constants, _parser

    newer = {_constants.POSSESSIVE_REPEAT, _constants.ATOMIC_GROUP}
    stack = [parsed]
    while stack:
        item = stack.pop()
        if isinstance(item, _parser.SubPattern):
            for op, av in item.data:
                if op in newer:
                    return True
                stack.append(av)
        elif isinstance(item, (list, tuple)):
            stack.extend(item)
    return False


def _literal_patterns():
    for path in sorted(_BACKEND.rglob("*.py")):
        if "node_modules" in path.parts:
            continue
        try:
            text = path.read_text(encoding = "utf-8")
        except UnicodeDecodeError:
            continue
        # Every possessive or atomic pattern contains one of these; most files have none to parse.
        if not any(token in text for token in ("*+", "++", "?+", "}+", "(?>")):
            continue
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and getattr(node.func, "attr", None) in _RE_FUNCTIONS
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                yield path, node.lineno, node.args[0].value


def test_the_check_sees_possessive_and_atomic_syntax():
    from re import _parser
    for pattern in (
        r'[^"{}]*+',
        r'(?:[^"{}<]++|<)*+',
        r"a?+",
        r"a{2,3}+",
        r"(?>ab|a)c",
        r"x(?:(?>y))",
    ):
        assert _uses_311_syntax(_parser.parse(pattern)), pattern
    for pattern in (r'[^"{}]*', r"\++", r"[*+]", r"a{2,3}", r"(?:ab|a)c", r"(?=a+)"):
        assert not _uses_311_syntax(_parser.parse(pattern)), pattern


def test_backend_patterns_compile_on_the_python_floor():
    import warnings

    from re import _parser

    found = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for path, line, pattern in _literal_patterns():
            try:
                parsed = _parser.parse(pattern)
            except Exception:
                continue  # Not a regex argument (str.split, a bytes-like helper); nothing to check.
            if _uses_311_syntax(parsed):
                found.append(f"{path.relative_to(_BACKEND)}:{line}: {pattern!r}")
    assert (_BACKEND / "core" / "tool_healing.py").is_file(), _BACKEND
    assert not found, "\n".join(found)
