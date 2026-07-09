# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Stage 5: runtime realpath backstop injected into sandboxed _python_exec."""

import os
import sys
from pathlib import Path

import pytest

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from core.inference.tools import (
    _BLOCKED_COMMANDS_COMMON,
    _python_exec,
    get_sandbox_workdir,
)

_POSIX_ONLY = pytest.mark.skipif(
    sys.platform == "win32", reason = "preexec_fn / setsid are POSIX-only"
)


def test_ln_is_blocked_command():
    assert "ln" in _BLOCKED_COMMANDS_COMMON


@_POSIX_ONLY
def test_sandboxed_symlink_write_escape_denied(tmp_path):
    # A pre-existing symlink inside the workdir escapes to an outside dir. The
    # static gate sees only a relative literal (allowed); the runtime realpath
    # backstop follows the link and denies the write. This is the case static
    # analysis fundamentally cannot see.
    session = "backstop-symlink-write"
    workdir = get_sandbox_workdir(session)
    link = os.path.join(workdir, "escape_dir")
    if os.path.islink(link) or os.path.exists(link):
        os.remove(link)
    os.symlink(str(tmp_path), link)
    probe = tmp_path / "studio_escape_probe.txt"
    try:
        out = _python_exec(
            "open('escape_dir/studio_escape_probe.txt', 'w').write('x')",
            None, 30, session, disable_sandbox = False,
        )
        assert "sandbox:" in out or "PermissionError" in out
        assert not probe.exists()
    finally:
        os.remove(link)


@_POSIX_ONLY
def test_sandboxed_symlink_remove_escape_denied(tmp_path):
    session = "backstop-symlink-remove"
    workdir = get_sandbox_workdir(session)
    link = os.path.join(workdir, "escape_dir")
    if os.path.islink(link) or os.path.exists(link):
        os.remove(link)
    os.symlink(str(tmp_path), link)
    victim = tmp_path / "keep_me.txt"
    victim.write_text("important")
    try:
        out = _python_exec(
            "import os; os.remove('escape_dir/keep_me.txt')",
            None, 30, session, disable_sandbox = False,
        )
        assert "sandbox:" in out or "PermissionError" in out
        assert victim.exists()  # the guard blocked before the real call
    finally:
        os.remove(link)


@_POSIX_ONLY
def test_sandboxed_benign_relative_write_allowed():
    out = _python_exec(
        "f = open('backstop_ok.txt', 'w'); f.write('hi'); f.close(); print('WROTE_OK')",
        None, 30, "backstop-benign", disable_sandbox = False,
    )
    assert "WROTE_OK" in out
    assert "sandbox:" not in out


@_POSIX_ONLY
def test_bypass_open_write_not_guarded(monkeypatch, tmp_path):
    # Under bypass the guard is not injected; the write is not denied by us.
    target = tmp_path / "bypass_write.txt"
    out = _python_exec(
        f"open({str(target)!r}, 'w').write('x'); print('BYPASS_OK')",
        None, 30, "backstop-bypass", disable_sandbox = True,
    )
    assert "sandbox:" not in out
    assert "BYPASS_OK" in out


@_POSIX_ONLY
def test_sandboxed_imports_still_work_under_guard():
    # The guard must not break library imports (bytecode caching failures are
    # swallowed by importlib) or benign compute.
    out = _python_exec(
        "import json; print(json.dumps({'a': 1}))",
        None, 30, "backstop-imports", disable_sandbox = False,
    )
    assert '{"a": 1}' in out
    assert "sandbox:" not in out
