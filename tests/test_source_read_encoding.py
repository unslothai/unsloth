# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Guard: tests that read checked-in files must name their encoding.

`Path.read_text()` and `open()` with no encoding use `locale.getencoding()`:
UTF-8 on the Linux and macOS runners, cp1252 on a stock Windows install. A test
that reads a repo file that way passes in CI and raises UnicodeDecodeError for a
Windows contributor as soon as that file gains a non-ASCII byte, which the
source-scanning tests do constantly:

    studio/backend/routes/inference.py carries the DeepSeek tool-call token
    regexes, so it holds U+FF5C and U+2581. Reading it as cp1252 dies on
    "byte 0x81", taking test_cancel_atomicity.py and test_cancel_id_wiring.py
    out at collection time.

Nothing that runs at import can see a tmp_path fixture, so a bare read there is
always touching a checked-in file. That makes the rule mechanical enough to
enforce with no allowlist, while staying quiet about temp-dir I/O inside test
bodies where the platform default is harmless.
"""

# `str | None` below is evaluated at import on Python 3.9 without this, and
# pyproject declares requires-python = ">=3.9,<3.15".
from __future__ import annotations

import ast
from pathlib import Path

TESTS = Path(__file__).resolve().parent
REPO = TESTS.parent
# Both trees ship to Windows contributors, and separate CI jobs collect them
# (repo-cpu-tests and the studio-backend matrix), so the rule covers both.
ROOTS = (TESTS, REPO / "studio" / "backend" / "tests")
GUARDED_METHODS = {"read_text", "write_text"}
NOT_PATH_RECEIVERS = {
    "bz2",
    "codecs",
    "dbm",
    "fitz",
    "gzip",
    "io",
    "lzma",
    "os",
    "pymupdf",
    "shelve",
    "sqlite3",
    "tarfile",
    "wave",
    "webbrowser",
    "zipfile",
}
# Distinct from None so that "no mode argument at all" still means text.
UNKNOWN_MODE = object()


def _is_main_guard(node: ast.AST) -> bool:
    """True for `if __name__ == "__main__":`, whose body never runs at import.

    The operator has to be `==`: `if __name__ != "__main__":` runs its body at
    import, so treating it as script-only would invert the rule.
    """
    if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
        return False
    left = node.test.left
    if not (isinstance(left, ast.Name) and left.id == "__name__"):
        return False
    if not all(isinstance(op, ast.Eq) for op in node.test.ops):
        return False
    return any(isinstance(c, ast.Constant) and c.value == "__main__" for c in node.test.comparators)


def _import_time_calls(tree: ast.Module):
    """Yield Call nodes that run at import time.

    That is module scope, class bodies, and the bodies of module-level helpers
    invoked from either. A helper is the same hazard as an inline read:
    `CODE = _extract_mixed_precision_code()` runs its `read_text()` during
    collection, so skipping every def would let the Windows failure back in.

    A def's body waits for a call, but its decorators and argument defaults run
    when the def executes, so those are followed. Lambda bodies and
    comprehension elements do not run at definition and are skipped, which
    keeps the rule free of false positives.

    A body is only ever entered through an executed statement, never by walking
    into a def, so the "this definitely runs" property that makes the rule
    allowlist-free holds. Not followed: the body of
    `if __name__ == "__main__":`, which pytest never runs (its `else` arm does,
    so that is walked), and non-name calls, which are left unresolved rather
    than guessed at.
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
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # The body waits for a call; these two run right now.
                stack.extend(node.decorator_list)
                stack.extend(d for d in node.args.defaults if d is not None)
                stack.extend(d for d in node.args.kw_defaults if d is not None)
                continue
            if isinstance(node, ast.Lambda):
                stack.extend(d for d in node.args.defaults if d is not None)
                stack.extend(d for d in node.args.kw_defaults if d is not None)
                continue
            if isinstance(node, (ast.GeneratorExp, ast.ListComp, ast.SetComp, ast.DictComp)):
                # Only the outermost iterable is evaluated where it is written.
                if node.generators:
                    stack.append(node.generators[0].iter)
                continue
            if _is_main_guard(node):
                stack.extend(node.orelse)  # the else arm runs at import
                continue
            if isinstance(node, ast.Call):
                yield node
                func = node.func
                if isinstance(func, ast.Name) and func.id in helpers and func.id not in entered:
                    entered.add(func.id)
                    frontier.append(list(helpers[func.id].body))
            stack.extend(ast.iter_child_nodes(node))


def _open_mode(call: ast.Call, mode_index: int):
    """The literal mode of an open() call, or UNKNOWN_MODE.

    A splat or a non-literal hides the mode. Defaulting those to "r" would
    demand an encoding on a call that may resolve to "rb", where passing one is
    a ValueError, so the contributor would have no compliant edit.
    """
    if any(isinstance(a, ast.Starred) for a in call.args):
        return UNKNOWN_MODE
    if any(kw.arg is None for kw in call.keywords):
        return UNKNOWN_MODE
    if len(call.args) > mode_index:
        node = call.args[mode_index]
        return node.value if isinstance(node, ast.Constant) else UNKNOWN_MODE
    for kw in call.keywords:
        if kw.arg == "mode":
            return kw.value.value if isinstance(kw.value, ast.Constant) else UNKNOWN_MODE
    return "r"


def _is_text(call: ast.Call, mode_index: int) -> bool:
    mode = _open_mode(call, mode_index)
    return mode is not UNKNOWN_MODE and "b" not in str(mode)


def _names_encoding(call: ast.Call) -> bool:
    """True only for an encoding that actually pins one.

    `encoding = None` and `encoding = "locale"` both re-select the platform
    default, so the keyword being present is not enough. A `**kwargs` may carry
    one we cannot see, so it counts as named rather than risking a false alarm.
    """
    for kw in call.keywords:
        if kw.arg is None:
            return True
        if kw.arg != "encoding":
            continue
        if isinstance(kw.value, ast.Constant) and kw.value.value in (None, "locale"):
            return False
        return True
    return False


def _offender(call: ast.Call) -> str | None:
    """The call's name if it reads text without an encoding, else None."""
    func = call.func
    if isinstance(func, ast.Attribute):
        if func.attr in GUARDED_METHODS:
            # importlib.metadata's Distribution.read_text takes a positional
            # filename and has no encoding parameter, so a positional argument
            # means the receiver is not a Path.
            if func.attr == "read_text" and call.args:
                return None
            return None if _names_encoding(call) else f"{func.attr}()"
        if func.attr == "open":
            # `<module>.open(...)` is somebody else's opener: tarfile.open takes
            # a compression mode, fitz.open takes filetype=.
            if isinstance(func.value, ast.Name) and func.value.id in NOT_PATH_RECEIVERS:
                return None
            if not _is_text(call, 0):
                return None
            return None if _names_encoding(call) else "Path.open()"
        return None
    # Binary handles have no encoding to name.
    if isinstance(func, ast.Name) and func.id == "open" and _is_text(call, 1):
        return None if _names_encoding(call) else "open()"
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
