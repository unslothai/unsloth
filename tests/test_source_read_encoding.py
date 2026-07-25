# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Guard: tests that read checked-in files must name their encoding.

`Path.read_text()` and `open()` with no encoding use
`locale.getpreferredencoding()`, which is UTF-8 on the Linux runners and cp1252
on a stock Windows install. A test that reads a repo file that way passes in CI
and raises UnicodeDecodeError for a Windows contributor the moment that file
gains a non-ASCII byte, which the source-scanning tests do constantly:

    studio/backend/routes/inference.py carries the DeepSeek tool-call token
    regexes, so it contains U+FF5C and U+2581. Reading it with cp1252 died on
    "byte 0x81 in position 97806", taking test_cancel_atomicity.py and
    test_cancel_id_wiring.py out at collection time.

Nothing that runs at import can see a tmp_path fixture, so a bare read there is
always touching a checked-in file. That makes the rule mechanical enough to
enforce with no allowlist, and it stays quiet about temp-dir I/O inside test
bodies where the platform default is harmless. The repo already spells this
correctly in 464 other places; this only stops the stragglers coming back.
"""

# `str | None` below is evaluated at import on Python 3.9 without this, and
# pyproject declares requires-python = ">=3.9,<3.15".
from __future__ import annotations

import ast
from pathlib import Path

TESTS = Path(__file__).resolve().parent
REPO = TESTS.parent
# tests/ and studio/backend/tests/ are collected by separate CI jobs
# (repo-cpu-tests and studio-backend-ci) and both collect on Windows, so the
# rule has to cover both trees.
ROOTS = (TESTS, REPO / "studio" / "backend" / "tests")
GUARDED_METHODS = {"read_text", "write_text"}


def _is_main_guard(node: ast.AST) -> bool:
    """True for `if __name__ == "__main__":`, whose body never runs at import."""
    if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
        return False
    left = node.test.left
    if not (isinstance(left, ast.Name) and left.id == "__name__"):
        return False
    return any(isinstance(c, ast.Constant) and c.value == "__main__" for c in node.test.comparators)


def _import_time_calls(tree: ast.Module):
    """Yield Call nodes that run at import time.

    That is module scope, class bodies (which execute on definition), and the
    bodies of module-level helpers invoked from either. A helper is the same
    hazard as an inline read: `CODE = _extract_mixed_precision_code()` runs its
    `read_text()` during collection, so skipping every def would let the
    Windows failure back in unreported.

    Bodies are only ever entered through an executed statement, never by
    walking into a def, so the "this definitely runs" property that makes the
    rule allowlist-free is preserved. Two things are deliberately not followed:
    `if __name__ == "__main__":` blocks, which pytest never executes, and
    non-name calls (attribute, dynamic), which aren't resolved rather than
    guessed at.
    """
    helpers = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    entered = set()
    frontier = [list(tree.body)]
    while frontier:
        stack = frontier.pop()
        while stack:
            node = stack.pop()
            # A def is only reached by calling it, handled below; a __main__
            # block is script-only.
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or _is_main_guard(node):
                continue
            if isinstance(node, ast.Call):
                yield node
                func = node.func
                if isinstance(func, ast.Name) and func.id in helpers and func.id not in entered:
                    entered.add(func.id)
                    frontier.append(list(helpers[func.id].body))
            stack.extend(ast.iter_child_nodes(node))


def _open_mode(call: ast.Call) -> str:
    """The mode string of an open() call, defaulting to text read."""
    if len(call.args) > 1 and isinstance(call.args[1], ast.Constant):
        return str(call.args[1].value)
    for kw in call.keywords:
        if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
            return str(kw.value.value)
    return "r"


def _offender(call: ast.Call) -> str | None:
    """The call's name if it reads text without an encoding, else None."""
    if any(kw.arg == "encoding" for kw in call.keywords):
        return None
    func = call.func
    if isinstance(func, ast.Attribute) and func.attr in GUARDED_METHODS:
        return f"{func.attr}()"
    # Binary handles have no encoding to name.
    if isinstance(func, ast.Name) and func.id == "open" and "b" not in _open_mode(call):
        return "open()"
    return None


def test_module_level_file_reads_name_an_encoding():
    offenders = []
    for root in ROOTS:
        for path in sorted(root.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding = "utf-8"), filename = str(path))
            for call in _import_time_calls(tree):
                name = _offender(call)
                if name is None:
                    continue
                rel = path.relative_to(REPO).as_posix()
                offenders.append(f"{rel}:{call.lineno}: {name}")
    assert offenders == [], (
        "Import-time file reads in the test trees touch a checked-in file with "
        "the platform default encoding, so they break on Windows as soon as "
        'that file gains a non-ASCII byte. Pass encoding = "utf-8": '
        f"{offenders[:10]}"
    )
