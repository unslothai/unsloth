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
            None,
            30,
            session,
            disable_sandbox = False,
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
            None,
            30,
            session,
            disable_sandbox = False,
        )
        assert "sandbox:" in out or "PermissionError" in out
        assert victim.exists()  # the guard blocked before the real call
    finally:
        os.remove(link)


@_POSIX_ONLY
def test_sandboxed_benign_relative_write_allowed():
    out = _python_exec(
        "f = open('backstop_ok.txt', 'w'); f.write('hi'); f.close(); print('WROTE_OK')",
        None,
        30,
        "backstop-benign",
        disable_sandbox = False,
    )
    assert "WROTE_OK" in out
    assert "sandbox:" not in out


@_POSIX_ONLY
def test_bypass_open_write_not_guarded(monkeypatch, tmp_path):
    # Under bypass the guard is not injected; the write is not denied by us.
    target = tmp_path / "bypass_write.txt"
    out = _python_exec(
        f"open({str(target)!r}, 'w').write('x'); print('BYPASS_OK')",
        None,
        30,
        "backstop-bypass",
        disable_sandbox = True,
    )
    assert "sandbox:" not in out
    assert "BYPASS_OK" in out


@_POSIX_ONLY
def test_sandboxed_os_open_write_escape_denied(tmp_path):
    # Low-level os.open with write flags to an absolute path outside the workdir.
    # The static gate allows it (write-confinement is now runtime-only); the guard
    # must deny it -- os.open is the classic builtins.open bypass.
    target = tmp_path / "osopen_escape.txt"
    out = _python_exec(
        f"import os; os.open({str(target)!r}, os.O_CREAT | os.O_WRONLY, 0o600); print('OPENED')",
        None,
        30,
        "backstop-osopen-write",
        disable_sandbox = False,
    )
    assert "sandbox:" in out or "PermissionError" in out
    assert not target.exists()


@_POSIX_ONLY
def test_sandboxed_os_open_read_local_allowed():
    # Read-only os.open of a workdir-local file is allowed (reads are not confined
    # by the backstop; host-secret reads are caught by the static scanner instead).
    out = _python_exec(
        "import os\n"
        "fd = os.open('ro_probe.txt', os.O_CREAT | os.O_WRONLY, 0o600)\n"
        "os.write(fd, b'hi'); os.close(fd)\n"
        "fd2 = os.open('ro_probe.txt', os.O_RDONLY); print('READ_OK'); os.close(fd2)",
        None,
        30,
        "backstop-osopen-read",
        disable_sandbox = False,
    )
    assert "READ_OK" in out
    assert "sandbox:" not in out


@_POSIX_ONLY
def test_sandboxed_os_open_dir_fd_denied(tmp_path):
    # A mutating os.open with dir_fd cannot be confined by a string realpath, so it
    # fails closed even though the relative name looks local.
    out = _python_exec(
        "import os\n"
        f"dfd = os.open({str(tmp_path)!r}, os.O_RDONLY)\n"
        "os.open('evil.txt', os.O_CREAT | os.O_WRONLY, dir_fd=dfd); print('OPENED')",
        None,
        30,
        "backstop-osopen-dirfd",
        disable_sandbox = False,
    )
    assert "sandbox:" in out or "PermissionError" in out
    assert not (tmp_path / "evil.txt").exists()


@_POSIX_ONLY
def test_sandboxed_io_open_write_escape_denied(tmp_path):
    target = tmp_path / "ioopen_escape.txt"
    out = _python_exec(
        f"import io; io.open({str(target)!r}, 'w').write('x'); print('WROTE')",
        None,
        30,
        "backstop-ioopen",
        disable_sandbox = False,
    )
    assert "sandbox:" in out or "PermissionError" in out
    assert not target.exists()


@_POSIX_ONLY
def test_sandboxed_pathlib_open_write_escape_denied(tmp_path):
    # Path.open("w") routes through io.open, which the guard now patches too.
    target = tmp_path / "pathopen_escape.txt"
    out = _python_exec(
        f"from pathlib import Path; Path({str(target)!r}).open('w').write('x'); print('WROTE')",
        None,
        30,
        "backstop-pathopen",
        disable_sandbox = False,
    )
    assert "sandbox:" in out or "PermissionError" in out
    assert not target.exists()


@_POSIX_ONLY
def test_sandboxed_imports_still_work_under_guard():
    # The guard must not break library imports (bytecode caching failures are
    # swallowed by importlib) or benign compute.
    out = _python_exec(
        "import json; print(json.dumps({'a': 1}))",
        None,
        30,
        "backstop-imports",
        disable_sandbox = False,
    )
    assert '{"a": 1}' in out
    assert "sandbox:" not in out
