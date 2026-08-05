"""AST test that the server's stream tee tolerates a missing console.

A hidden Windows subprocess (CREATE_NO_WINDOW, e.g. `unsloth studio` launched
from install.ps1 / a launcher) has no console, so the interpreter leaves
sys.stdout / sys.stderr as None. `_setup_server_disk_logging` wraps them in a
_TeeStream; write()/flush()/close() must no-op on the None side (the file log
copy still works) instead of crashing with
`AttributeError: 'NoneType' object has no attribute 'write'`.
Pinned by AST because importing run.py at runtime needs the full studio venv.
"""

from __future__ import annotations

import ast
from pathlib import Path

_RUN_PY = Path(__file__).resolve().parents[2] / "studio" / "backend" / "run.py"


def _tee_stream_cls() -> ast.ClassDef:
    tree = ast.parse(_RUN_PY.read_text(encoding = "utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "_TeeStream":
            return node
    raise AssertionError("no _TeeStream class in studio/backend/run.py")


def _harden_console_close() -> ast.FunctionDef:
    tree = ast.parse(_RUN_PY.read_text(encoding = "utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_harden_console_close":
            return node
    raise AssertionError("no _harden_console_close in studio/backend/run.py")


def _has_none_guard(node: ast.AST) -> bool:
    """True if *node* contains an `if ... is None` guard (or equivalent)."""
    for n in ast.walk(node):
        if not isinstance(n, ast.If) or n.test is None:
            continue
        if isinstance(n.test, ast.Compare) and any(
            isinstance(c, ast.Constant) and c.value is None for c in ast.walk(n.test)
        ):
            return True
    return False


def test_tee_stream_write_guards_the_wrapped_stream():
    """write() must no-op on a None stream instead of delegating to it."""
    methods = {n.name: n for n in _tee_stream_cls().body if isinstance(n, ast.FunctionDef)}
    fn = methods.get("write")
    assert fn is not None, "_TeeStream has no write()"
    assert _has_none_guard(fn), (
        "write() delegates straight to self._stream.write(data) with no None guard; "
        "a hidden Windows subprocess (sys.stdout/stderr = None) crashes on the first "
        "print with AttributeError: 'NoneType' object has no attribute 'write'"
    )


def test_tee_stream_guards_flush_and_close_too():
    """flush()/close() must not delegate to a None stream either."""
    methods = {n.name: n for n in _tee_stream_cls().body if isinstance(n, ast.FunctionDef)}
    for name in ("flush", "close"):
        fn = methods.get(name)
        assert fn is not None, f"_TeeStream has no {name}()"
        assert _has_none_guard(fn), f"{name}() does not guard against a None wrapped stream"


def test_harden_console_close_accepts_none():
    """_harden_console_close(None) must be a no-op (never read .close on None)."""
    fn = _harden_console_close()
    assert _has_none_guard(fn), "_harden_console_close does not early-return on a None stream"
