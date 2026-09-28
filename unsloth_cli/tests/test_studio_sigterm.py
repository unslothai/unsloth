# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""SIGTERM (docker stop through supervisord, `unsloth studio stop`) takes the same path as Ctrl+C.

`unsloth studio` and `unsloth start` run the server in-process and caught only
KeyboardInterrupt, so a SIGTERM died on Python's default action: no `_graceful_shutdown`,
no stop-and-save for a running training job, no child cleanup.
"""

import ast
import signal
from pathlib import Path

import pytest

from unsloth_cli.commands import studio as studio_mod

STUDIO_SRC = Path(studio_mod.__file__).read_text(encoding = "utf-8")


def _installs_before_waiting(fn: ast.FunctionDef) -> bool:
    waits = [
        node.lineno
        for node in ast.walk(fn)
        if isinstance(node, ast.While)
        and any(
            isinstance(inner, ast.Attribute) and inner.attr == "_shutdown_event"
            for inner in ast.walk(node)
        )
    ]
    if not waits:
        return False
    return any(
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Call)
        and isinstance(stmt.value.func, ast.Name)
        and stmt.value.func.id == "_graceful_shutdown_on_sigterm"
        and stmt.lineno < min(waits)
        for stmt in fn.body
    )


@pytest.fixture
def restore_sigterm():
    previous = signal.getsignal(signal.SIGTERM)
    yield
    signal.signal(signal.SIGTERM, previous)


def test_sigterm_becomes_a_keyboard_interrupt_once(restore_sigterm):
    studio_mod._graceful_shutdown_on_sigterm()
    handler = signal.getsignal(signal.SIGTERM)
    assert callable(handler) and handler is not signal.SIG_DFL
    with pytest.raises(KeyboardInterrupt):
        handler(signal.SIGTERM, None)
    assert signal.getsignal(signal.SIGTERM) is signal.SIG_DFL


def test_both_server_wait_loops_install_the_handler():
    """Both in-process commands, matched on the parsed tree so a commented-out call fails."""
    functions = [n for n in ast.walk(ast.parse(STUDIO_SRC)) if isinstance(n, ast.FunctionDef)]
    installed = sorted(fn.name for fn in functions if _installs_before_waiting(fn))
    assert installed == ["run", "studio_default"], installed
