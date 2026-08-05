# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""AST test that run.py survives being launched with no console.

A Windows process with no valid std handles (a GUI/pythonw or detached launch)
starts with sys.stdout / sys.stderr / sys.stdin as None.
`_normalize_standard_streams` replaces the missing ones at import time, and the
_TeeStream guards keep a direct `_TeeStream(None, ...)` from crashing with
`AttributeError: 'NoneType' object has no attribute 'write'`.

Source contract only. The behaviour is pinned at runtime in
studio/backend/tests/test_server_disk_logging.py, which runs on Python 3.10-3.13.
"""

from __future__ import annotations

import ast
from pathlib import Path

_RUN_PY = Path(__file__).resolve().parents[2] / "studio" / "backend" / "run.py"


def _module() -> ast.Module:
    return ast.parse(_RUN_PY.read_text(encoding = "utf-8"))


def _tee_stream_cls() -> ast.ClassDef:
    for node in _module().body:
        if isinstance(node, ast.ClassDef) and node.name == "_TeeStream":
            return node
    raise AssertionError("no _TeeStream class in studio/backend/run.py")


def _top_level_fn(name: str) -> ast.FunctionDef:
    for node in _module().body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"no {name} in studio/backend/run.py")


def _guards_target(fn: ast.FunctionDef, target: str) -> bool:
    """True if *fn* early-exits on `<target> is None` before dereferencing it.

    Deliberately strict: the guard must name the stream (not some other object),
    must compare it against None, and must return/raise, so an unrelated check
    like `if data is None:` cannot satisfy it.
    """
    for node in fn.body:
        if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
            continue
        names = {ast.unparse(node.test.left)}
        names.update(ast.unparse(c) for c in node.test.comparators)
        if target not in names:
            continue
        if not any(
            isinstance(op, ast.Is) and isinstance(c, ast.Constant) and c.value is None
            for op, c in zip(node.test.ops, node.test.comparators)
        ):
            continue
        if any(isinstance(stmt, (ast.Return, ast.Raise)) for stmt in node.body):
            return True
    return False


def test_streams_are_normalized_before_the_logger_import():
    """The fix has to land before structlog is imported, or it does nothing.

    structlog binds `from sys import stdout` at ITS import time and PrintLogger
    does `self._file = file or stdout`, so a None stdout is captured permanently
    the moment `from loggers import get_logger` runs. Normalizing later (inside
    run_server, say) leaves the first log call crashing exactly as before.
    """
    src = _RUN_PY.read_text(encoding = "utf-8")
    assert "\n_normalize_standard_streams()" in src, (
        "run.py never calls _normalize_standard_streams(); a console-less launch "
        "keeps sys.stdout/stderr as None and dies on the first log call"
    )
    assert src.index("\n_normalize_standard_streams()") < src.index(
        "\nfrom loggers import get_logger"
    ), (
        "_normalize_standard_streams() must be called before the loggers import; "
        "structlog captures sys.stdout at import time"
    )


def test_normalize_skips_streams_that_already_exist():
    """Replacing a live console would break Colab, Tauri and pytest capture."""
    fn = _top_level_fn("_normalize_standard_streams")
    assert any(
        isinstance(n, ast.If) and "is not None" in ast.unparse(n.test) for n in ast.walk(fn)
    ), "_normalize_standard_streams overwrites streams that are already present"


def test_tee_stream_write_guards_the_wrapped_stream():
    """write() must no-op on a None stream instead of delegating to it."""
    methods = {n.name: n for n in _tee_stream_cls().body if isinstance(n, ast.FunctionDef)}
    fn = methods.get("write")
    assert fn is not None, "_TeeStream has no write()"
    assert _guards_target(fn, "self._stream"), (
        "write() delegates straight to self._stream.write(data) with no None guard; "
        "_TeeStream(None, log) then crashes with "
        "AttributeError: 'NoneType' object has no attribute 'write'"
    )


def test_tee_stream_guards_flush_and_close_too():
    """flush()/close() must not delegate to a None stream either."""
    methods = {n.name: n for n in _tee_stream_cls().body if isinstance(n, ast.FunctionDef)}
    for name in ("flush", "close"):
        fn = methods.get(name)
        assert fn is not None, f"_TeeStream has no {name}()"
        assert _guards_target(
            fn, "self._stream"
        ), f"{name}() does not early-return on a None wrapped stream"


def test_harden_console_close_accepts_none():
    """_harden_console_close(None) must be a no-op (never read .close on None)."""
    assert _guards_target(
        _top_level_fn("_harden_console_close"), "stream"
    ), "_harden_console_close does not early-return on a None stream"
