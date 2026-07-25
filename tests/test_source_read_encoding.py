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

At module scope there is no tmp_path fixture, so a bare read here is always
touching a checked-in file. That makes the rule mechanical enough to enforce
with no allowlist, and it stays quiet about temp-dir I/O inside test bodies
where the platform default is harmless. The repo already spells this correctly
in 464 other places; this only stops the stragglers coming back.
"""

import ast
from pathlib import Path

TESTS = Path(__file__).resolve().parent
GUARDED_METHODS = {"read_text", "write_text"}


def _module_level_calls(tree: ast.Module):
    """Yield Call nodes reachable without entering a def or class body."""
    stack = list(tree.body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if isinstance(node, ast.Call):
            yield node
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
    for path in sorted(TESTS.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding = "utf-8"), filename = str(path))
        for call in _module_level_calls(tree):
            name = _offender(call)
            if name is None:
                continue
            rel = path.relative_to(TESTS).as_posix()
            offenders.append(f"tests/{rel}:{call.lineno}: {name}")
    assert offenders == [], (
        "Module-level file reads in tests/ touch a checked-in file with the "
        "platform default encoding, so they break on Windows as soon as that "
        'file gains a non-ASCII byte. Pass encoding = "utf-8": '
        f"{offenders[:10]}"
    )
