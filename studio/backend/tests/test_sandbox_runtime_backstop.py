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
def test_sandboxed_os_rename_dir_fd_denied(tmp_path):
    # os.rename / os.replace with src_dir_fd / dst_dir_fd is fd-relative; a string
    # realpath against cwd cannot confine it (the relative names look local), so the
    # guard fails closed before the syscall.
    out = _python_exec(
        "import os\n"
        f"dfd = os.open({str(tmp_path)!r}, os.O_RDONLY)\n"
        "os.replace('a.txt', 'b.txt', src_dir_fd=dfd)\nprint('DONE-OK')",
        None,
        30,
        "backstop-osrename-dirfd",
        disable_sandbox = False,
    )
    assert "sandbox:" in out and "(dir_fd)" in out
    assert "DONE-OK" not in out


@_POSIX_ONLY
def test_sandboxed_pathlib_rename_target_kw_denied(tmp_path):
    # Path.rename / Path.replace accept the destination as the `target=` keyword; the
    # guard must confine the keyword target, not only the positional one.
    target = tmp_path / "pathrename_kw_escape.txt"
    session = "backstop-pathrename-kw"
    workdir = get_sandbox_workdir(session)
    src = os.path.join(workdir, "kw_src.txt")
    with open(src, "w") as f:
        f.write("x")
    try:
        out = _python_exec(
            "from pathlib import Path\n"
            f"Path('kw_src.txt').rename(target = {str(target)!r})\nprint('DONE-OK')",
            None,
            30,
            session,
            disable_sandbox = False,
        )
        assert "sandbox:" in out and "Path.rename target" in out
        assert "DONE-OK" not in out
        assert not target.exists()
    finally:
        if os.path.exists(src):
            os.remove(src)


@_POSIX_ONLY
def test_sandboxed_future_import_write_escape_denied(tmp_path):
    # A program that opens with `from __future__ import ...` must still be sandboxed.
    # The guard is spliced AFTER the (inert, compile-time) future import, so the
    # realpath backstop is active before the first real statement.
    target = tmp_path / "future_escape.txt"
    out = _python_exec(
        "from __future__ import annotations\n"
        f"open({str(target)!r}, 'w').write('x'); print('WROTE')",
        None,
        30,
        "backstop-future",
        disable_sandbox = False,
    )
    assert "sandbox:" in out or "PermissionError" in out
    assert not target.exists()


def test_sandboxed_future_import_program_runs():
    # The guard splice must not break an otherwise-benign future-import program
    # (a plain prepend would raise "from __future__ imports must occur at the
    # beginning of the file").
    out = _python_exec(
        "from __future__ import annotations\nx: int = 41\nprint(x + 1)",
        None,
        30,
        "backstop-future-ok",
        disable_sandbox = False,
    )
    assert "42" in out
    assert "sandbox:" not in out


def test_inject_sandbox_guard_preserves_leading_directives():
    from core.inference.tools import _inject_sandbox_guard

    prelude = "GUARD_LINE()\n"
    code = '"""doc"""\nfrom __future__ import annotations\nx = 1\n'
    out = _inject_sandbox_guard(code, prelude)
    lines = out.splitlines()
    fut = next(i for i, ln in enumerate(lines) if "__future__" in ln)
    guard = next(i for i, ln in enumerate(lines) if "GUARD_LINE" in ln)
    stmt = next(i for i, ln in enumerate(lines) if ln.strip() == "x = 1")
    # future import stays on top; guard runs before the first real statement.
    assert fut < guard < stmt


def test_inject_sandbox_guard_plain_prepend_without_future():
    from core.inference.tools import _inject_sandbox_guard

    prelude = "GUARD_LINE()\n"
    code = "import os\nx = 1\n"
    # No future import: behavior is unchanged (a simple prepend).
    assert _inject_sandbox_guard(code, prelude) == prelude + code


@_POSIX_ONLY
def test_sandboxed_open_wrapped_attr_removed(tmp_path):
    # functools.wraps would publish the ORIGINAL unguarded callable on __wrapped__;
    # the guard must not expose it (open.__wrapped__(outside, 'w') would bypass).
    target = tmp_path / "wrapped_escape.txt"
    out = _python_exec(
        f"open.__wrapped__({str(target)!r}, 'w').write('x'); print('WROTE')",
        None,
        30,
        "backstop-wrapped",
        disable_sandbox = False,
    )
    assert not target.exists()
    assert "AttributeError" in out or "sandbox:" in out


@_POSIX_ONLY
def test_sandboxed_low_level_io_open_denied(tmp_path):
    # io.open / builtins.open originate from the C module _io; patching io.open leaves
    # _io.open untouched, so it must be guarded too.
    target = tmp_path / "lowio_escape.txt"
    out = _python_exec(
        f"import _io; _io.open({str(target)!r}, 'w').write('x'); print('WROTE')",
        None,
        30,
        "backstop-lowio",
        disable_sandbox = False,
    )
    assert "sandbox:" in out
    assert not target.exists()


@_POSIX_ONLY
def test_sandboxed_chdir_escape_denied():
    # os.chdir outside the workdir would let a later relative read/write (which the
    # static scan treats as local) escape, so cwd changes are confined.
    out = _python_exec(
        "import os\nos.chdir('/etc')\nprint('CWD', os.getcwd())",
        None,
        30,
        "backstop-chdir",
        disable_sandbox = False,
    )
    assert "sandbox:" in out and "chdir" in out


@_POSIX_ONLY
def test_sandboxed_chdir_within_workdir_allowed():
    out = _python_exec(
        "import os\nos.chdir('.')\nprint('CWD-OK')",
        None,
        30,
        "backstop-chdir-ok",
        disable_sandbox = False,
    )
    assert "CWD-OK" in out
    assert "sandbox:" not in out


@_POSIX_ONLY
def test_sandboxed_fd_metadata_mutator_denied(tmp_path):
    # A read-only os.open of an outside file is allowed (reads are not confined), but
    # fd-based metadata mutators (os.fchmod/fchown) must be denied so they cannot be
    # reused to mutate host files.
    victim = tmp_path / "victim.txt"
    victim.write_text("x")
    os.chmod(victim, 0o600)
    out = _python_exec(
        "import os\n"
        f"fd = os.open({str(victim)!r}, os.O_RDONLY)\n"
        "os.fchmod(fd, 0o644); print('CHMODDED')",
        None,
        30,
        "backstop-fchmod",
        disable_sandbox = False,
    )
    assert "sandbox:" in out and "fchmod" in out
    assert oct(os.stat(victim).st_mode & 0o777) == "0o600"


@_POSIX_ONLY
def test_sandboxed_posix_module_open_denied(tmp_path):
    # os re-exports from the C module posix; posix.open must be guarded too.
    target = tmp_path / "posix_escape.txt"
    out = _python_exec(
        "import posix, os\n"
        f"fd = posix.open({str(target)!r}, os.O_CREAT | os.O_WRONLY, 0o600)\n"
        "posix.write(fd, b'x')\nprint('DONE-OK')",
        None,
        30,
        "backstop-posix",
        disable_sandbox = False,
    )
    assert "sandbox:" in out
    assert not target.exists()


@_POSIX_ONLY
def test_sandboxed_extra_os_mutators_denied(tmp_path):
    # os.mkfifo creates a host node; os.utime mutates host metadata.
    fifo = tmp_path / "escape.fifo"
    out = _python_exec(
        f"import os; os.mkfifo({str(fifo)!r}); print('DONE-OK')",
        None,
        30,
        "backstop-mkfifo",
        disable_sandbox = False,
    )
    assert "sandbox:" in out
    assert not fifo.exists()

    victim = tmp_path / "utime_victim.txt"
    victim.write_text("x")
    before = victim.stat().st_mtime
    out = _python_exec(
        f"import os; os.utime({str(victim)!r}, (0, 0)); print('DONE-OK')",
        None,
        30,
        "backstop-utime",
        disable_sandbox = False,
    )
    assert "sandbox:" in out
    assert victim.stat().st_mtime == before


@_POSIX_ONLY
def test_sandboxed_io_fileio_write_denied(tmp_path):
    # io.FileIO / _io.FileIO is a C constructor that opens a file without routing
    # through open(), so it needs its own guard.
    target = tmp_path / "fileio_escape.txt"
    out = _python_exec(
        f"import io; io.FileIO({str(target)!r}, 'w').write(b'x'); print('DONE-OK')",
        None,
        30,
        "backstop-fileio",
        disable_sandbox = False,
    )
    assert "sandbox:" in out
    assert not target.exists()


@_POSIX_ONLY
def test_sandboxed_stateful_fspath_toctou_denied(tmp_path):
    # A stateful __fspath__ returning a local path for the check and an outside path
    # for the real open must not escape: the guard materializes fspath once.
    target = tmp_path / "fspath_escape.txt"
    out = _python_exec(
        "class P:\n"
        "    n = 0\n"
        "    def __fspath__(self):\n"
        "        P.n += 1\n"
        f"        return 'ok.txt' if P.n == 1 else {str(target)!r}\n"
        "open(P(), 'w').write('x')\nprint('DONE')",
        None,
        30,
        "backstop-fspath",
        disable_sandbox = False,
    )
    # The escape target is never written (the single fspath call yields the local path).
    assert not target.exists()


@_POSIX_ONLY
def test_sandboxed_str_subclass_mode_denied(tmp_path):
    # A str-subclass mode whose __contains__ lies must not defeat the write check.
    target = tmp_path / "mode_escape.txt"
    out = _python_exec(
        "class M(str):\n"
        "    def __contains__(self, c):\n"
        "        return False\n"
        f"open({str(target)!r}, M('w')).write('x')\nprint('DONE-OK')",
        None,
        30,
        "backstop-mode",
        disable_sandbox = False,
    )
    assert "sandbox:" in out
    assert not target.exists()


@_POSIX_ONLY
def test_sandboxed_int_subclass_flags_denied(tmp_path):
    # An int-subclass flags whose __and__ lies must not defeat the os.open write check.
    target = tmp_path / "flags_escape.txt"
    out = _python_exec(
        "import os\n"
        "class F(int):\n"
        "    def __and__(self, o):\n"
        "        return 0\n"
        f"os.open({str(target)!r}, F(os.O_CREAT | os.O_WRONLY), 0o600)\nprint('DONE-OK')",
        None,
        30,
        "backstop-flags",
        disable_sandbox = False,
    )
    assert "sandbox:" in out
    assert not target.exists()


@_POSIX_ONLY
def test_sandboxed_fileio_stateful_fspath_denied(tmp_path):
    # io.FileIO must pass the MATERIALIZED path to the real constructor so a stateful
    # __fspath__ cannot return a different (outside) path than was checked.
    target = tmp_path / "fileio_fspath_escape.txt"
    out = _python_exec(
        "import io\n"
        "class P:\n"
        "    n = 0\n"
        "    def __fspath__(self):\n"
        "        P.n += 1\n"
        f"        return 'ok.txt' if P.n == 1 else {str(target)!r}\n"
        "io.FileIO(P(), 'w').write(b'x')\nprint('DONE')",
        None,
        30,
        "backstop-fileio-fspath",
        disable_sandbox = False,
    )
    assert not target.exists()


@_POSIX_ONLY
def test_sandboxed_os_chmod_fd_denied(tmp_path):
    # A read-only fd opened on an outside file must not be reusable via os.chmod(fd).
    victim = tmp_path / "chmod_fd_victim.txt"
    victim.write_text("x")
    os.chmod(victim, 0o600)
    out = _python_exec(
        "import os\n"
        f"fd = os.open({str(victim)!r}, os.O_RDONLY)\n"
        "os.chmod(fd, 0o644); print('CHMODDED')",
        None,
        30,
        "backstop-chmod-fd",
        disable_sandbox = False,
    )
    assert "sandbox:" in out and "(fd)" in out
    assert oct(os.stat(victim).st_mode & 0o777) == "0o600"


@_POSIX_ONLY
def test_sandboxed_in_workdir_ops_still_work():
    # The added guards must not break benign in-workdir writes.
    out = _python_exec(
        "import io, os\n"
        "io.FileIO('fio.txt', 'w').write(b'z')\n"
        "os.makedirs('subd', exist_ok = True)\n"
        "print(io.FileIO('fio.txt', 'r').read())",
        None,
        30,
        "backstop-inworkdir",
        disable_sandbox = False,
    )
    assert "z" in out
    assert "sandbox:" not in out


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
