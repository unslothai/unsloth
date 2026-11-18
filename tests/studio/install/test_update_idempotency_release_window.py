# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Every PyPI-path update test in test_update_idempotency.py must survive the release window.

Between a version bump on main and its publish, the checkout's unsloth is not PyPI's, so an update
that reaches PyPI (Windows: Invoke-RestMethod ignores HTTPS_PROXY) logs UPGRADE_MARKER and moves the
install to PyPI's release. A test that then asserts nothing on disk moved fails every Windows run
until the publish, and, without restoring the checkout, leaves the later tests running PyPI's code.
"""

from __future__ import annotations

import ast
import pathlib

SOURCE = pathlib.Path(__file__).with_name("test_update_idempotency.py")


def _kw(call: ast.Call, name: str):
    for kw in call.keywords:
        if kw.arg == name and isinstance(kw.value, ast.Constant):
            return kw.value.value
    return None


def _run_updates(fn: ast.FunctionDef) -> list[ast.Call]:
    return [
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "run_update"
    ]


def _disk_unchanged_asserts(fn: ast.FunctionDef) -> list[ast.Assert]:
    """`assert diff(before, snapshot(...)) == []`."""
    out = []
    for n in ast.walk(fn):
        if not isinstance(n, ast.Assert) or not isinstance(n.test, ast.Compare):
            continue
        left = n.test.left
        if (
            isinstance(left, ast.Call)
            and isinstance(left.func, ast.Name)
            and left.func.id == "diff"
        ):
            out.append(n)
    return out


def _guards_upgrade_before(fn: ast.FunctionDef, line: int) -> bool:
    """An `if` before `line` that reads UPGRADE_MARKER (in its test or body), restores the checkout
    (run_update local=True) and leaves the test (skip, fail or return)."""
    for n in ast.walk(fn):
        if not isinstance(n, ast.If) or n.lineno >= line:
            continue
        body = ast.Module(body = n.body, type_ignores = [])
        names = {x.id for x in ast.walk(n) if isinstance(x, ast.Name)}
        if "UPGRADE_MARKER" not in names:
            continue
        restores = any(
            isinstance(x, ast.Call)
            and isinstance(x.func, ast.Name)
            and x.func.id == "run_update"
            and _kw(x, "local") is True
            for x in ast.walk(body)
        )
        leaves = any(
            isinstance(x, ast.Call)
            and isinstance(x.func, ast.Attribute)
            and x.func.attr in ("skip", "fail")
            for x in ast.walk(body)
        ) or any(isinstance(x, ast.Return) for x in ast.walk(body))
        if restores and leaves:
            return True
    return False


def offenders(source: str) -> list[str]:
    bad = []
    for fn in ast.parse(source).body:
        if not isinstance(fn, ast.FunctionDef) or not fn.name.startswith("test_"):
            continue
        if not any(_kw(c, "local") is False for c in _run_updates(fn)):
            continue
        for a in _disk_unchanged_asserts(fn):
            if not _guards_upgrade_before(fn, a.lineno):
                bad.append(f"{fn.name}:{a.lineno}")
    return bad


def test_pypi_path_update_tests_handle_the_release_window():
    assert SOURCE.is_file(), SOURCE
    assert offenders(SOURCE.read_text(encoding = "utf-8")) == []


def test_the_checker_flags_an_unguarded_disk_assert():
    unguarded = (
        "def test_x(install, settled):\n"
        "    run = run_update(d, 'r', local = False)\n"
        "    assert diff(before, snapshot(install)) == []\n"
    )
    assert offenders(unguarded) == ["test_x:3"]
    no_restore = (
        "def test_x(install, settled):\n"
        "    run = run_update(d, 'r', local = False)\n"
        "    if UPGRADE_MARKER in run.log:\n"
        "        pytest.skip('x')\n"
        "    assert diff(before, snapshot(install)) == []\n"
    )
    assert offenders(no_restore) == ["test_x:5"]
    guarded = (
        "def test_x(install, settled):\n"
        "    run = run_update(d, 'r', local = False)\n"
        "    if UPGRADE_MARKER in run.log:\n"
        "        run_update(d, 'restore', local = True)\n"
        "        pytest.skip('x')\n"
        "    assert diff(before, snapshot(install)) == []\n"
    )
    assert offenders(guarded) == []
    local_only = (
        "def test_x(install, settled):\n"
        "    run = run_update(d, 'r', local = True)\n"
        "    assert diff(before, snapshot(install)) == []\n"
    )
    assert offenders(local_only) == []
