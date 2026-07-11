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
    _SANDBOX_GUARD_SRC,
    _bash_exec,
    _command_reads_sensitive,
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
    # Read-only os.open of a workdir-local file is allowed: reads of non-sensitive paths
    # are not confined (only mutating opens are workdir-confined, and only host-secret
    # realpaths are denied by the runtime sensitive-read backstop).
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
def test_sandboxed_os_open_readonly_dir_fd_denied(tmp_path):
    # A READ-ONLY os.open with dir_fd can read a host file under a directory fd opened
    # outside the workdir (d = os.open('/etc', O_RDONLY); os.open('hostname', O_RDONLY,
    # dir_fd=d)); reads are not confined, so the fd-relative open must fail closed.
    victim = tmp_path / "secret.txt"
    victim.write_text("topsecret")
    out = _python_exec(
        "import os\n"
        f"dfd = os.open({str(tmp_path)!r}, os.O_RDONLY)\n"
        "fd = os.open('secret.txt', os.O_RDONLY, dir_fd=dfd)\n"
        "print('READ', os.read(fd, 64))",
        None,
        30,
        "backstop-osopen-ro-dirfd",
        disable_sandbox = False,
    )
    assert "sandbox:" in out or "PermissionError" in out
    assert "topsecret" not in out


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
def test_sandboxed_sqlite3_connect_escape_denied(tmp_path):
    # sqlite3.connect opens/creates the DB via the native _sqlite3 C extension (not
    # builtins.open), so the open-like backstop never sees it; the dedicated sqlite guard
    # must confine the database path to the workdir.
    target = tmp_path / "sqlite_escape.db"
    out = _python_exec(
        f"import sqlite3; sqlite3.connect({str(target)!r}); print('OPENED')",
        None,
        30,
        "backstop-sqlite-escape",
        disable_sandbox = False,
    )
    assert "sandbox:" in out or "PermissionError" in out
    assert not target.exists()


@_POSIX_ONLY
def test_sandboxed_sqlite3_connect_dynamic_escape_denied():
    # A dynamically built absolute path (os.sep + 'tmp/...') has no literal for the static
    # scanner; the runtime guard resolves and denies it.
    probe = os.path.join(os.sep, "tmp", "studio_sqlite_dyn_escape.db")
    if os.path.exists(probe):
        os.remove(probe)
    out = _python_exec(
        "import sqlite3, os; sqlite3.connect(os.sep + 'tmp/studio_sqlite_dyn_escape.db')",
        None,
        30,
        "backstop-sqlite-dyn",
        disable_sandbox = False,
    )
    assert "sandbox:" in out or "PermissionError" in out
    assert not os.path.exists(probe)


def test_sandboxed_sqlite3_connect_local_allowed():
    # A workdir-relative database opens and is usable; the guard confines but does not block
    # benign local DB work.
    out = _python_exec(
        "import sqlite3\n"
        "c = sqlite3.connect('backstop_local.db')\n"
        "c.execute('create table if not exists t(x)'); c.close(); print('DB_OK')",
        None,
        30,
        "backstop-sqlite-local",
        disable_sandbox = False,
    )
    assert "DB_OK" in out
    assert "sandbox:" not in out


def test_sandboxed_sqlite3_connect_memory_allowed():
    # :memory: never touches the filesystem, so it is allowed.
    out = _python_exec(
        "import sqlite3\n"
        "c = sqlite3.connect(':memory:')\n"
        "c.execute('create table t(x)'); c.close(); print('MEM_OK')",
        None,
        30,
        "backstop-sqlite-mem",
        disable_sandbox = False,
    )
    assert "MEM_OK" in out
    assert "sandbox:" not in out


@_POSIX_ONLY
def test_sandboxed_low_level_sqlite3_connect_escape_denied(tmp_path):
    # The native _sqlite3 C extension exposes the ORIGINAL connect and is importable directly,
    # bypassing the two Python bindings; the guard must wrap it too.
    target = tmp_path / "low_sqlite_escape.db"
    out = _python_exec(
        f"import _sqlite3; _sqlite3.connect({str(target)!r}); print('OPENED')",
        None,
        30,
        "backstop-low-sqlite-escape",
        disable_sandbox = False,
    )
    assert "sandbox:" in out or "PermissionError" in out
    assert not target.exists()


@_POSIX_ONLY
def test_sandboxed_sqlite3_uri_percent_encoded_escape_denied(tmp_path):
    # SQLite percent-decodes the URI filename, so an encoded absolute path (file:%2Ftmp%2Fx,
    # uri=True) must be decoded before the workdir check or it slips through as relative-looking.
    target = tmp_path / "uri_escape.db"
    enc = str(target).replace("/", "%2F")
    out = _python_exec(
        f"import sqlite3; sqlite3.connect('file:{enc}', uri=True); print('OPENED')",
        None,
        30,
        "backstop-sqlite-uri-escape",
        disable_sandbox = False,
    )
    assert "sandbox:" in out or "PermissionError" in out
    assert not target.exists()


def test_sandboxed_sqlite3_uri_local_allowed():
    # A workdir-local file: URI (no escaping percent-decode) still opens.
    out = _python_exec(
        "import sqlite3\n"
        "c = sqlite3.connect('file:uri_local.db', uri=True)\n"
        "c.execute('create table if not exists t(x)'); c.close(); print('URI_OK')",
        None,
        30,
        "backstop-sqlite-uri-local",
        disable_sandbox = False,
    )
    assert "URI_OK" in out
    assert "sandbox:" not in out


def test_sandboxed_sqlite3_attach_escape_denied(tmp_path):
    # ATTACH DATABASE '<outside>' creates/opens that file via the native extension, bypassing the
    # wrapped connect; the connection authorizer must deny an escaping ATTACH target.
    target = tmp_path / "attach_escape.db"
    out = _python_exec(
        "import sqlite3\n"
        "c = sqlite3.connect('backstop_attach.db')\n"
        f"c.execute(\"ATTACH DATABASE '{target}' AS ext\")\n"
        "print('ATTACHED_OK')",
        None,
        30,
        "backstop-sqlite-attach-escape",
        disable_sandbox = False,
    )
    assert "ATTACHED_OK" not in out
    assert not target.exists()


def test_sandboxed_sqlite3_vacuum_into_escape_denied(tmp_path):
    # VACUUM ... INTO '<outside>' writes a fresh database file outside the workdir via the native
    # extension; it fires the same SQLITE_ATTACH authorizer action and must be denied.
    target = tmp_path / "vacuum_escape.db"
    out = _python_exec(
        "import sqlite3\n"
        "c = sqlite3.connect('backstop_vacuum.db')\n"
        "c.execute('create table t(x)')\n"
        f"c.execute(\"VACUUM main INTO '{target}'\")\n"
        "print('VACUUMED_OK')",
        None,
        30,
        "backstop-sqlite-vacuum-escape",
        disable_sandbox = False,
    )
    assert "VACUUMED_OK" not in out
    assert not target.exists()


def test_sandboxed_sqlite3_attach_local_allowed():
    # A workdir-local ATTACH (and ordinary queries) stay allowed; the authorizer confines only
    # escaping targets, so benign multi-database work is not blocked.
    out = _python_exec(
        "import sqlite3\n"
        "c = sqlite3.connect('backstop_attach_main.db')\n"
        "c.execute(\"ATTACH DATABASE 'backstop_attach_side.db' AS ext\")\n"
        "c.execute('create table if not exists ext.t(x)')\n"
        "c.close(); print('ATTACH_LOCAL_OK')",
        None,
        30,
        "backstop-sqlite-attach-local",
        disable_sandbox = False,
    )
    assert "ATTACH_LOCAL_OK" in out
    assert "sandbox:" not in out


@_POSIX_ONLY
def test_sandboxed_sqlite3_connection_constructor_escape_denied(tmp_path):
    # The public sqlite3.Connection('/outside.db') constructor creates the DB via the native
    # extension without going through the guarded connect(); the guarded Connection subclass must
    # confine the path at construction.
    target = tmp_path / "conn_ctor_escape.db"
    out = _python_exec(
        f"import sqlite3\nc = sqlite3.Connection({str(target)!r})\n"
        "c.execute('create table t(x)'); c.commit(); print('CTOR_OK')",
        None,
        30,
        "backstop-sqlite-ctor-escape",
        disable_sandbox = False,
    )
    assert "CTOR_OK" not in out
    assert not target.exists()


@_POSIX_ONLY
def test_sandboxed_low_level_sqlite3_connection_constructor_escape_denied(tmp_path):
    # _sqlite3.Connection is the raw C constructor, importable directly; it must be guarded too.
    target = tmp_path / "low_conn_ctor_escape.db"
    out = _python_exec(
        f"import _sqlite3\nc = _sqlite3.Connection({str(target)!r})\n"
        "c.execute('create table t(x)'); print('LOW_CTOR_OK')",
        None,
        30,
        "backstop-sqlite-low-ctor-escape",
        disable_sandbox = False,
    )
    assert "LOW_CTOR_OK" not in out
    assert not target.exists()


def test_sandboxed_sqlite3_connection_constructor_local_allowed():
    # A workdir-relative sqlite3.Connection(...) opens and is usable.
    out = _python_exec(
        "import sqlite3\n"
        "c = sqlite3.Connection('ctor_local.db')\n"
        "c.execute('create table if not exists t(x)'); c.close(); print('CTOR_LOCAL_OK')",
        None,
        30,
        "backstop-sqlite-ctor-local",
        disable_sandbox = False,
    )
    assert "CTOR_LOCAL_OK" in out
    assert "sandbox:" not in out


@_POSIX_ONLY
def test_sandboxed_sqlite3_set_authorizer_none_attach_escape_denied(tmp_path):
    # Removing the confinement authorizer (set_authorizer(None)) must NOT re-open the ATTACH
    # escape: the guarded Connection composes its workdir confinement ahead of any caller
    # callback and keeps it on set_authorizer(None).
    target = tmp_path / "auth_removed_attach_escape.db"
    out = _python_exec(
        "import sqlite3\n"
        "c = sqlite3.connect('backstop_authrm.db')\n"
        "c.set_authorizer(None)\n"
        f"c.execute(\"ATTACH DATABASE '{target}' AS ext\")\n"
        "print('AUTH_REMOVED_ATTACH_OK')",
        None,
        30,
        "backstop-sqlite-authrm-attach",
        disable_sandbox = False,
    )
    assert "AUTH_REMOVED_ATTACH_OK" not in out
    assert not target.exists()


@_POSIX_ONLY
def test_sandboxed_sqlite3_set_authorizer_none_vacuum_escape_denied(tmp_path):
    # The same durability holds for VACUUM INTO after set_authorizer(None).
    target = tmp_path / "auth_removed_vacuum_escape.db"
    out = _python_exec(
        "import sqlite3\n"
        "c = sqlite3.connect('backstop_authrm_v.db')\n"
        "c.execute('create table t(x)')\n"
        "c.set_authorizer(None)\n"
        f"c.execute(\"VACUUM INTO '{target}'\")\n"
        "print('AUTH_REMOVED_VACUUM_OK')",
        None,
        30,
        "backstop-sqlite-authrm-vacuum",
        disable_sandbox = False,
    )
    assert "AUTH_REMOVED_VACUUM_OK" not in out
    assert not target.exists()


def test_sandboxed_sqlite3_user_authorizer_still_runs():
    # A caller-supplied authorizer still composes (benign work is not broken by the confinement).
    out = _python_exec(
        "import sqlite3\n"
        "c = sqlite3.connect('backstop_userauth.db')\n"
        "def ok(*a):\n    return sqlite3.SQLITE_OK\n"
        "c.set_authorizer(ok)\n"
        "c.execute('create table if not exists t(x)'); c.close(); print('USERAUTH_OK')",
        None,
        30,
        "backstop-sqlite-userauth",
        disable_sandbox = False,
    )
    assert "USERAUTH_OK" in out
    assert "sandbox:" not in out


@_POSIX_ONLY
def test_sandboxed_getattr_gadget_dunder_workdir_module_denied():
    # A workdir helper recovering the guard wrapper's original open via a getattr gadget dunder
    # (getattr(open, '__closure__')) must be refused by the vetter, like the direct attribute form.
    _assert_workdir_module_denied(
        "backstop-workdir-getattr-gadget",
        "gadgetdunder",
        "C = getattr(open, '__closure__')\nprint('GADGET_RAN')\n",
        "GADGET_RAN",
    )


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
def test_sandboxed_path_open_str_subclass_mode_denied(tmp_path):
    # Path.open must coerce a str-subclass mode through the base str (a lying
    # __contains__ must not defeat the write check). On 3.13 the underlying io.open
    # guard also catches it; this asserts the write never lands regardless.
    target = tmp_path / "pathopen_mode_escape.txt"
    out = _python_exec(
        "from pathlib import Path\n"
        "class M(str):\n"
        "    def __contains__(self, c):\n"
        "        return False\n"
        f"Path({str(target)!r}).open(M('w')).write('x')\nprint('DONE')",
        None,
        30,
        "backstop-pathmode",
        disable_sandbox = False,
    )
    assert not target.exists()


@_POSIX_ONLY
def test_sandboxed_path_rename_stateful_target_denied(tmp_path):
    # Path.rename must materialize the target once so a stateful __fspath__ cannot
    # return an in-workdir path for the check and an outside one for the real call.
    target = tmp_path / "pathrename_target_escape.txt"
    session = "backstop-pathrename-stateful"
    workdir = get_sandbox_workdir(session)
    src = os.path.join(workdir, "stateful_src.txt")
    with open(src, "w") as f:
        f.write("x")
    try:
        out = _python_exec(
            "from pathlib import Path\n"
            "class T:\n"
            "    n = 0\n"
            "    def __fspath__(self):\n"
            "        T.n += 1\n"
            f"        return 'okp.txt' if T.n == 1 else {str(target)!r}\n"
            "Path('stateful_src.txt').rename(T())\nprint('DONE')",
            None,
            30,
            session,
            disable_sandbox = False,
        )
        assert not target.exists()
    finally:
        if os.path.exists(src):
            os.remove(src)


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


@_POSIX_ONLY
def test_sandboxed_benign_workdir_module_import_allowed():
    # A benign sibling module (data / functions only) the user wrote must still import: the
    # workdir import vetter only refuses modules that reach a command-execution sink.
    session = "backstop-workdir-import-ok"
    workdir = get_sandbox_workdir(session)
    with open(os.path.join(workdir, "helper_ok.py"), "w") as f:
        f.write("VALUE = 42\ndef greet():\n    return 'hi'\n")
    try:
        out = _python_exec(
            "import helper_ok; print('HELPER', helper_ok.VALUE, helper_ok.greet())",
            None,
            30,
            session,
            disable_sandbox = False,
        )
        assert "HELPER 42 hi" in out
        assert "sandbox:" not in out
    finally:
        os.remove(os.path.join(workdir, "helper_ok.py"))


@_POSIX_ONLY
def test_sandboxed_malicious_workdir_module_import_denied():
    # A planted workdir module whose top-level code runs os.system was never seen by the static
    # analyzer; importing it would execute the sink in an unguarded child. The vetter refuses it.
    session = "backstop-workdir-import-evil"
    workdir = get_sandbox_workdir(session)
    with open(os.path.join(workdir, "evilmod.py"), "w") as f:
        f.write("import os\nos.system('echo PWNED')\n")
    try:
        out = _python_exec(
            "import evilmod; print('REACHED_' + 'BODY')",
            None,
            30,
            session,
            disable_sandbox = False,
        )
        # The import is refused before the module body (its os.system) runs, and before the
        # trailing print. (The source line is echoed in the traceback, so assert on the sink
        # output + the printed marker, not on the source text.)
        assert "PWNED" not in out
        assert "REACHED_BODY" not in out
        assert "sandbox:" in out or "ImportError" in out
    finally:
        os.remove(os.path.join(workdir, "evilmod.py"))


@_POSIX_ONLY
def test_sandboxed_pyc_only_workdir_module_import_denied():
    # A sandboxed snippet can write a legacy sourceless `evil.pyc` directly in the workdir and
    # `import evil`; PathFinder returns a `.pyc` origin the source scanner cannot read. The vetter
    # must refuse any non-source workdir module rather than letting the default sourceless loader
    # execute its bytecode (which would run an unguarded os.system child).
    import importlib.util
    import marshal

    session = "backstop-workdir-pyc"
    workdir = get_sandbox_workdir(session)
    src = "import os\nos.system('echo PWNED_PYC')\n"
    pyc = (
        importlib.util.MAGIC_NUMBER
        + (b"\x00" * 12)
        + marshal.dumps(compile(src, "evilpyc.py", "exec"))
    )
    target = os.path.join(workdir, "evilpyc.pyc")
    with open(target, "wb") as f:
        f.write(pyc)
    try:
        out = _python_exec(
            "import evilpyc; print('REACHED_' + 'BODY')",
            None,
            30,
            session,
            disable_sandbox = False,
        )
        assert "PWNED_PYC" not in out
        assert "REACHED_BODY" not in out
        assert "sandbox:" in out or "ImportError" in out
    finally:
        os.remove(target)


@_POSIX_ONLY
def test_sandboxed_from_os_import_sink_workdir_module_denied():
    # `from os import system; system('...')` binds a BARE sink name; the earlier vetter only
    # rejected `import subprocess` / `from subprocess ...`, so the from-os form slipped through
    # and ran an unguarded shell child at import time. The vetter now refuses it.
    session = "backstop-workdir-fromos"
    workdir = get_sandbox_workdir(session)
    with open(os.path.join(workdir, "evilfromos.py"), "w") as f:
        f.write("from os import system\nsystem('echo PWNED_FROMOS')\n")
    try:
        out = _python_exec(
            "import evilfromos; print('REACHED_' + 'BODY')",
            None,
            30,
            session,
            disable_sandbox = False,
        )
        assert "PWNED_FROMOS" not in out
        assert "REACHED_BODY" not in out
        assert "sandbox:" in out or "ImportError" in out
    finally:
        os.remove(os.path.join(workdir, "evilfromos.py"))


@_POSIX_ONLY
def test_sandboxed_benign_attr_named_sink_workdir_module_allowed():
    # A benign workdir module with a DATA attribute that merely shares a name with an os sink
    # (p.system = 'linux') must still import: the vetter scopes rejection to actual sink calls
    # and to sink references rooted at os / posix, not any same-named attribute.
    session = "backstop-workdir-attrfp"
    workdir = get_sandbox_workdir(session)
    with open(os.path.join(workdir, "helper_attr.py"), "w") as f:
        f.write("class P:\n    pass\np = P()\np.system = 'linux'\nVALUE = p.system\n")
    try:
        out = _python_exec(
            "import helper_attr; print('HELPER', helper_attr.VALUE)",
            None,
            30,
            session,
            disable_sandbox = False,
        )
        assert "HELPER linux" in out
        assert "sandbox:" not in out
    finally:
        os.remove(os.path.join(workdir, "helper_attr.py"))


@_POSIX_ONLY
def test_sandboxed_benign_called_sink_name_workdir_module_allowed():
    # A workdir helper that CALLS a method merely sharing a name with an os sink -- the ubiquitous
    # platform.system(), or the module's own object method obj.system() -- must still import. The
    # vetter now roots the exec-attr CALL rejection at an os / posix receiver (like the reference
    # check), so a same-named call on an unrelated object is no longer misread as a shell escape.
    session = "backstop-workdir-callfp"
    workdir = get_sandbox_workdir(session)
    with open(os.path.join(workdir, "helper_call.py"), "w") as f:
        f.write(
            "import platform\n"
            "class Runner:\n"
            "    def system(self, x):\n"
            "        return x * 2\n"
            "PLAT = bool(platform.system())\n"
            "VALUE = Runner().system(21)\n"
        )
    try:
        out = _python_exec(
            "import helper_call; print('HELPER', helper_call.VALUE)",
            None,
            30,
            session,
            disable_sandbox = False,
        )
        assert "HELPER 42" in out
        assert "sandbox:" not in out
    finally:
        os.remove(os.path.join(workdir, "helper_call.py"))


@_POSIX_ONLY
def test_sandboxed_os_system_call_workdir_module_still_denied():
    # The item-511 loosening must NOT reopen a real os.system escape: a workdir helper that calls
    # os.system (rooted at the os module) still spawns an unguarded child, so it stays refused.
    # (The command is assembled at runtime so the source echoed in the traceback does not itself
    # contain the marker -- proving the sink never actually ran.)
    session = "backstop-workdir-ossys"
    workdir = get_sandbox_workdir(session)
    with open(os.path.join(workdir, "ossys_helper.py"), "w") as f:
        f.write("import os\nos.system('echo ' + 'PWN' + 'MARK')\n")
    try:
        out = _python_exec(
            "import ossys_helper; print('REACHED_' + 'BODY')",
            None,
            30,
            session,
            disable_sandbox = False,
        )
        assert "PWNMARK" not in out
        assert "REACHED_BODY" not in out
        assert "sandbox:" in out or "ImportError" in out
    finally:
        os.remove(os.path.join(workdir, "ossys_helper.py"))


@_POSIX_ONLY
def test_sandboxed_workdir_module_meta_path_mutation_denied():
    # A workdir module that mutates the import machinery (sys.meta_path.pop(0)) would remove THIS
    # vetter, after which a second workdir module could import unscanned and run an unguarded
    # sink. The top-level analyzer blocks meta_path mutation in submitted code; the vetter must
    # refuse it inside a workdir module too.
    session = "backstop-workdir-metapop"
    workdir = get_sandbox_workdir(session)
    with open(os.path.join(workdir, "mp_popper.py"), "w") as f:
        f.write("import sys\nsys.meta_path.pop(0)\nprint('POPPED_OK')\n")
    try:
        out = _python_exec(
            "import mp_popper; print('REACHED_' + 'BODY')",
            None,
            30,
            session,
            disable_sandbox = False,
        )
        assert "POPPED_OK" not in out
        assert "REACHED_BODY" not in out
        assert "sandbox:" in out or "ImportError" in out
    finally:
        os.remove(os.path.join(workdir, "mp_popper.py"))


@_POSIX_ONLY
def test_sandboxed_utf7_encoded_workdir_module_denied():
    # A `# coding: utf_7` module hides os.system in what a UTF-8 scan reads as a comment (the raw
    # +AAo- bytes are a newline under UTF-7). The vetter must decode with PEP 263 like the loader
    # will, so the real os.system is seen and refused.
    session = "backstop-workdir-utf7"
    workdir = get_sandbox_workdir(session)
    data = b"# coding: utf_7\nimport os\npass  #+AAo-os.system('echo PWNED_UTF7')\n"
    with open(os.path.join(workdir, "evilenc.py"), "wb") as f:
        f.write(data)
    try:
        out = _python_exec(
            "import evilenc; print('REACHED_' + 'BODY')",
            None,
            30,
            session,
            disable_sandbox = False,
        )
        assert "PWNED_UTF7" not in out
        assert "REACHED_BODY" not in out
        assert "sandbox:" in out or "ImportError" in out
    finally:
        os.remove(os.path.join(workdir, "evilenc.py"))


@_POSIX_ONLY
def test_sandboxed_forged_pyc_workdir_module_ignored():
    # A harmless source plus a planted __pycache__ .pyc whose header matches it but whose body is
    # malicious: the vetter scans the safe source, but the module must run the VETTED SOURCE
    # (not the cached bytecode), so the planted payload never executes.
    import importlib.util
    import marshal
    import struct

    session = "backstop-workdir-forgedpyc"
    workdir = get_sandbox_workdir(session)
    src_path = os.path.join(workdir, "forged.py")
    with open(src_path, "w") as f:
        f.write("VALUE = 7\nprint('SOURCE_RAN')\n")
    st = os.stat(src_path)
    mal = compile("import os\nos.system('echo PWNED_FORGEDPYC')\n", "forged.py", "exec")
    pyc_dir = os.path.join(workdir, "__pycache__")
    os.makedirs(pyc_dir, exist_ok = True)
    pyc_path = os.path.join(pyc_dir, f"forged.{sys.implementation.cache_tag}.pyc")
    with open(pyc_path, "wb") as f:
        f.write(importlib.util.MAGIC_NUMBER)
        f.write(struct.pack("<I", 0))
        f.write(struct.pack("<I", int(st.st_mtime) & 0xFFFFFFFF))
        f.write(struct.pack("<I", st.st_size & 0xFFFFFFFF))
        f.write(marshal.dumps(mal))
    try:
        out = _python_exec(
            "import forged; print('REACHED', forged.VALUE)",
            None,
            30,
            session,
            disable_sandbox = False,
        )
        assert "PWNED_FORGEDPYC" not in out  # the cached malicious bytecode never runs
        assert "SOURCE_RAN" in out and "REACHED 7" in out  # the vetted source runs
    finally:
        os.remove(src_path)
        if os.path.exists(pyc_path):
            os.remove(pyc_path)


@_POSIX_ONLY
def test_sandboxed_symlinked_workdir_module_denied(tmp_path):
    # A workdir module that is a symlink to a file OUTSIDE the workdir: its realpath escapes, so
    # the vetter must fail closed rather than hand the outside file to the default loader unvetted.
    session = "backstop-workdir-symlinkmod"
    workdir = get_sandbox_workdir(session)
    outside = tmp_path / "outside_evil.py"
    outside.write_text("import os\nos.system('echo PWNED_SYMLINKMOD')\n")
    link = os.path.join(workdir, "evillink.py")
    if os.path.islink(link) or os.path.exists(link):
        os.remove(link)
    os.symlink(str(outside), link)
    try:
        out = _python_exec(
            "import evillink; print('REACHED_' + 'BODY')",
            None,
            30,
            session,
            disable_sandbox = False,
        )
        assert "PWNED_SYMLINKMOD" not in out
        assert "REACHED_BODY" not in out
        assert "sandbox:" in out or "ImportError" in out
    finally:
        os.remove(link)


@_POSIX_ONLY
def test_sandboxed_network_workdir_module_denied():
    # A workdir helper that opens a socket bypasses the static network policy (no runtime network
    # backstop), so the vetter refuses a module importing a network primitive.
    session = "backstop-workdir-net"
    workdir = get_sandbox_workdir(session)
    with open(os.path.join(workdir, "evilnet.py"), "w") as f:
        f.write("print('NET_REACHED')\nimport socket\ns = socket.socket()\ns.close()\n")
    try:
        out = _python_exec(
            "import evilnet; print('REACHED_' + 'BODY')",
            None,
            30,
            session,
            disable_sandbox = False,
        )
        assert "NET_REACHED" not in out
        assert "REACHED_BODY" not in out
        assert "sandbox:" in out or "ImportError" in out
    finally:
        os.remove(os.path.join(workdir, "evilnet.py"))


@_POSIX_ONLY
def test_sandboxed_os_alias_workdir_module_denied():
    # import os as o; s = o.system; s(...) -- an os import ALIAS whose sink is only assigned (not
    # directly called) must be recognized: record the alias before checking sink references.
    session = "backstop-workdir-osalias"
    workdir = get_sandbox_workdir(session)
    with open(os.path.join(workdir, "evilalias.py"), "w") as f:
        f.write("import os as o\ns = o.system\ns('echo PWNED_OSALIAS')\n")
    try:
        out = _python_exec(
            "import evilalias; print('REACHED_' + 'BODY')",
            None,
            30,
            session,
            disable_sandbox = False,
        )
        assert "PWNED_OSALIAS" not in out
        assert "REACHED_BODY" not in out
        assert "sandbox:" in out or "ImportError" in out
    finally:
        os.remove(os.path.join(workdir, "evilalias.py"))


@_POSIX_ONLY
def test_sandboxed_builtins_exec_workdir_module_denied():
    # import builtins; builtins.eval("__import__('os').system(...)") -- the execution builtins
    # reached as an attribute of the builtins module (not the bare eval/exec name) must be
    # recognized as an exec sink so a workdir helper cannot run arbitrary code at import.
    session = "backstop-workdir-builtins"
    workdir = get_sandbox_workdir(session)
    with open(os.path.join(workdir, "evilbi.py"), "w") as f:
        f.write("import builtins\nbuiltins.eval(\"__import__('os').system('echo PWNED_BI')\")\n")
    try:
        out = _python_exec(
            "import evilbi; print('REACHED_' + 'BODY')",
            None,
            30,
            session,
            disable_sandbox = False,
        )
        assert "PWNED_BI" not in out
        assert "REACHED_BODY" not in out
        assert "sandbox:" in out or "ImportError" in out
    finally:
        os.remove(os.path.join(workdir, "evilbi.py"))


@_POSIX_ONLY
def test_sandboxed_dotted_network_workdir_module_denied():
    # import urllib.request -- the bare top (urllib) is benign, but the dotted network submodule
    # opens outbound connections the static policy never saw, so the vetter refuses it.
    session = "backstop-workdir-urlreq"
    workdir = get_sandbox_workdir(session)
    with open(os.path.join(workdir, "evilurl.py"), "w") as f:
        f.write("print('URL_REACHED')\nimport urllib.request\n")
    try:
        out = _python_exec(
            "import evilurl; print('REACHED_' + 'BODY')",
            None,
            30,
            session,
            disable_sandbox = False,
        )
        assert "URL_REACHED" not in out
        assert "REACHED_BODY" not in out
        assert "sandbox:" in out or "ImportError" in out
    finally:
        os.remove(os.path.join(workdir, "evilurl.py"))


@_POSIX_ONLY
def test_sandboxed_benign_urllib_parse_workdir_module_allowed():
    # The benign urllib sibling (urllib.parse) is NOT a network submodule and must still import
    # from a workdir helper -- the dotted-network refusal keys on the full dotted name.
    session = "backstop-workdir-urlparse"
    workdir = get_sandbox_workdir(session)
    with open(os.path.join(workdir, "okparse.py"), "w") as f:
        f.write("import urllib.parse\nVALUE = urllib.parse.quote('a b')\nprint('PARSE_OK')\n")
    try:
        out = _python_exec(
            "import okparse; print('REACHED', okparse.VALUE)",
            None,
            30,
            session,
            disable_sandbox = False,
        )
        assert "PARSE_OK" in out
        assert "REACHED a%20b" in out
        assert "sandbox:" not in out
    finally:
        os.remove(os.path.join(workdir, "okparse.py"))


@_POSIX_ONLY
def test_sandboxed_deserializer_workdir_module_denied():
    # A workdir helper that calls pickle.loads runs an attacker-controlled reduce payload (whose
    # bytes live outside the helper source) -- here the reducer spawns os.system -- so the import
    # vetter must refuse it even though the malicious call is not spelled in the source.
    import pickle

    class _Evil:
        def __reduce__(self):
            return (os.system, ("echo PWNED_DESER",))

    evil = pickle.dumps(_Evil())
    helper = "import pickle\nprint('DESER_REACHED')\npickle.loads(%r)\n" % (evil,)
    session = "backstop-workdir-deser"
    workdir = get_sandbox_workdir(session)
    with open(os.path.join(workdir, "evildeser.py"), "w") as f:
        f.write(helper)
    try:
        out = _python_exec(
            "import evildeser; print('REACHED_' + 'BODY')",
            None,
            30,
            session,
            disable_sandbox = False,
        )
        assert "PWNED_DESER" not in out
        assert "DESER_REACHED" not in out
        assert "REACHED_BODY" not in out
        assert "sandbox:" in out or "ImportError" in out
    finally:
        os.remove(os.path.join(workdir, "evildeser.py"))


@_POSIX_ONLY
def test_sandboxed_benign_json_workdir_module_allowed():
    # json.load / json.loads do NOT run a reduce payload and are not in the deserializer set, so a
    # workdir helper using json must still import.
    session = "backstop-workdir-json"
    workdir = get_sandbox_workdir(session)
    with open(os.path.join(workdir, "okjson.py"), "w") as f:
        f.write("import json\nVALUE = json.loads('[1, 2, 3]')\nprint('JSON_OK')\n")
    try:
        out = _python_exec(
            "import okjson; print('REACHED', okjson.VALUE)",
            None,
            30,
            session,
            disable_sandbox = False,
        )
        assert "JSON_OK" in out
        assert "REACHED [1, 2, 3]" in out
        assert "sandbox:" not in out
    finally:
        os.remove(os.path.join(workdir, "okjson.py"))


@_POSIX_ONLY
def test_sandboxed_getattr_obfuscated_sink_workdir_module_denied():
    # getattr(os, 'system')('...') in a workdir helper is the obfuscated twin of os.system, which
    # the direct-attribute checks miss -- the dynamic-attribute sink must be refused at import.
    session = "backstop-workdir-getattr"
    workdir = get_sandbox_workdir(session)
    with open(os.path.join(workdir, "evilga.py"), "w") as f:
        f.write("import os\nprint('OBF_REACHED')\ngetattr(os, 'system')('echo PWNED_GA')\n")
    try:
        out = _python_exec(
            "import evilga; print('REACHED_' + 'BODY')",
            None,
            30,
            session,
            disable_sandbox = False,
        )
        assert "PWNED_GA" not in out
        assert "OBF_REACHED" not in out
        assert "REACHED_BODY" not in out
        assert "sandbox:" in out or "ImportError" in out
    finally:
        os.remove(os.path.join(workdir, "evilga.py"))


@_POSIX_ONLY
def test_sandboxed_benign_getattr_workdir_module_allowed():
    # getattr on a non-sink receiver (a plain object attribute) is ordinary reflection, not a sink,
    # so a workdir helper using it must still import.
    session = "backstop-workdir-okgetattr"
    workdir = get_sandbox_workdir(session)
    with open(os.path.join(workdir, "okga.py"), "w") as f:
        f.write("class K:\n    v = 7\nVALUE = getattr(K, 'v')\nprint('GA_OK')\n")
    try:
        out = _python_exec(
            "import okga; print('REACHED', okga.VALUE)",
            None,
            30,
            session,
            disable_sandbox = False,
        )
        assert "GA_OK" in out
        assert "REACHED 7" in out
        assert "sandbox:" not in out
    finally:
        os.remove(os.path.join(workdir, "okga.py"))


def _assert_workdir_module_denied(session, modname, src, marker):
    workdir = get_sandbox_workdir(session)
    path = os.path.join(workdir, modname + ".py")
    with open(path, "w") as f:
        f.write(src)
    try:
        out = _python_exec(
            "import %s; print('REACHED_' + 'BODY')" % modname,
            None,
            30,
            session,
            disable_sandbox = False,
        )
        assert marker not in out
        assert "REACHED_BODY" not in out
        assert "sandbox:" in out or "ImportError" in out
    finally:
        os.remove(path)


@_POSIX_ONLY
def test_sandboxed_ctypes_workdir_module_denied():
    # import ctypes gives a workdir helper UNGUARDED native libc, bypassing the patched open/os.open.
    _assert_workdir_module_denied(
        "backstop-workdir-ctypes",
        "evilct",
        "print('CT_REACHED')\nimport ctypes\nctypes.CDLL(None)\n",
        "CT_REACHED",
    )


@_POSIX_ONLY
def test_sandboxed_dynamic_import_workdir_module_denied():
    # importlib.import_module('subprocess') re-obtains a denied module without a literal import.
    _assert_workdir_module_denied(
        "backstop-workdir-dynimp",
        "evildi",
        "import importlib\nprint('DI_REACHED')\n"
        "sp = importlib.import_module('subprocess')\nsp.run(['echo', 'x'])\n",
        "DI_REACHED",
    )


@_POSIX_ONLY
def test_sandboxed_closure_gadget_workdir_module_denied():
    # __closure__ / cell_contents recover a guard wrapper's original unguarded callable.
    _assert_workdir_module_denied(
        "backstop-workdir-clo",
        "evilclo",
        "import builtins\nprint('CLO_REACHED')\nc = builtins.open.__closure__\n",
        "CLO_REACHED",
    )


@_POSIX_ONLY
def test_sandboxed_indirect_metapath_workdir_module_denied():
    # vars(sys)['meta_path'] reaches the import machinery without the literal .meta_path attribute.
    _assert_workdir_module_denied(
        "backstop-workdir-meta",
        "evilmeta",
        "import sys\nprint('META_REACHED')\nmp = vars(sys)['meta_' + 'path']\nmp[:] = []\n",
        "META_REACHED",
    )


@_POSIX_ONLY
def test_sandboxed_subscripted_builtins_workdir_module_denied():
    # __builtins__['eval'] reaches the execution builtins via the module's builtins dict.
    _assert_workdir_module_denied(
        "backstop-workdir-subbi",
        "evilsub",
        "print('SUB_REACHED')\n__builtins__['ev' + 'al'](\"__import__('os').system('echo x')\")\n",
        "SUB_REACHED",
    )


@_POSIX_ONLY
def test_sandboxed_benign_dynamic_import_workdir_module_allowed():
    # importlib.import_module of a NON-denied module (json) stays allowed.
    session = "backstop-workdir-okdi"
    workdir = get_sandbox_workdir(session)
    with open(os.path.join(workdir, "okdi.py"), "w") as f:
        f.write(
            "import importlib\nm = importlib.import_module('json')\n"
            "VALUE = m.dumps({'a': 1})\nprint('DI_OK')\n"
        )
    try:
        out = _python_exec(
            "import okdi; print('REACHED', okdi.VALUE)",
            None,
            30,
            session,
            disable_sandbox = False,
        )
        assert "DI_OK" in out
        assert '{"a": 1}' in out
        assert "sandbox:" not in out
    finally:
        os.remove(os.path.join(workdir, "okdi.py"))


@_POSIX_ONLY
def test_sandboxed_realpath_monkeypatch_write_escape_denied(tmp_path):
    # Sandboxed code reassigns os.path.realpath to a lambda that echoes an in-workdir
    # path, then writes to an absolute path outside the workdir. If the guard read
    # os.path.realpath off the live module every call, the reassignment would fool
    # _within() into approving the outside write. The guard captures the original path
    # helpers at prelude time, so the write is still denied.
    session = "backstop-realpath-monkeypatch"
    workdir = get_sandbox_workdir(session)
    target = tmp_path / "realpath_escape_probe.txt"
    if target.exists():
        target.unlink()
    out = _python_exec(
        "import os\n"
        f"os.path.realpath = lambda p: {workdir!r} + '/ok'\n"
        f"os.fspath = lambda p: {workdir!r} + '/ok'\n"
        f"open({str(target)!r}, 'w').write('escaped')\n"
        "print('WROTE-VIA-MONKEYPATCH')\n",
        None,
        30,
        session,
        disable_sandbox = False,
    )
    assert "sandbox:" in out or "PermissionError" in out
    assert not target.exists()


@_POSIX_ONLY
def test_sandboxed_fspath_monkeypatch_write_escape_denied(tmp_path):
    # The os.fspath twin of the realpath monkeypatch: reassigning os.fspath must not
    # let a materialized outside path slip past the confinement check.
    session = "backstop-fspath-monkeypatch"
    workdir = get_sandbox_workdir(session)
    target = tmp_path / "fspath_escape_probe.txt"
    if target.exists():
        target.unlink()
    out = _python_exec(
        "import os\n"
        f"os.fspath = lambda p: {workdir!r} + '/ok'\n"
        f"open({str(target)!r}, 'w').write('escaped')\n"
        "print('WROTE-VIA-MONKEYPATCH')\n",
        None,
        30,
        session,
        disable_sandbox = False,
    )
    assert "sandbox:" in out or "PermissionError" in out
    assert not target.exists()


@_POSIX_ONLY
def test_sandboxed_bytes_path_in_workdir_write_allowed():
    # A bytes path resolves to bytes from os.path.realpath; the guard must normalize it
    # (fsdecode) so a legitimate in-workdir bytes write is not denied by a str/bytes
    # prefix-compare TypeError.
    out = _python_exec(
        "f = open(b'bytes_local.txt', 'w'); f.write('hi'); f.close(); print('BYTES_OK')",
        None,
        30,
        "backstop-bytes-path",
        disable_sandbox = False,
    )
    assert "BYTES_OK" in out
    assert "sandbox:" not in out


@_POSIX_ONLY
def test_sandboxed_bytes_path_out_of_workdir_write_denied(tmp_path):
    # The bytes-path normalization must not weaken confinement: an outside bytes write
    # is still denied.
    target = tmp_path / "bytes_escape.txt"
    out = _python_exec(
        f"open({bytes(str(target), 'utf-8')!r}, 'w').write('x'); print('WROTE')",
        None,
        30,
        "backstop-bytes-escape",
        disable_sandbox = False,
    )
    assert "sandbox:" in out or "PermissionError" in out
    assert not target.exists()


@_POSIX_ONLY
def test_sandboxed_closure_recovery_of_open_blocked():
    # object.__getattribute__(builtins.open, '__closure__') recovers the original
    # unguarded open from the wrapper closure. The static gate now blocks the
    # introspection (gadget dunder via __getattribute__), so it never runs.
    out = _python_exec(
        "import builtins\n"
        "object.__getattribute__(builtins.open, '__closure__')[0].cell_contents"
        "('/tmp/studio_closure_escape.txt', 'w').write('x')\n"
        "print('CLOSURE_WROTE')\n",
        None,
        30,
        "backstop-closure",
        disable_sandbox = False,
    )
    assert "unsafe code detected" in out or "sandbox:" in out or "PermissionError" in out
    assert not os.path.exists("/tmp/studio_closure_escape.txt")


@_POSIX_ONLY
def test_sandboxed_dynamic_closure_name_recovery_blocked():
    # __closure__ built at runtime via chr(): the static gate must block the
    # .cell_contents recovery step so the original open is never reached.
    name = "''.join(map(chr,[95,95,99,108,111,115,117,114,101,95,95]))"
    out = _python_exec(
        f"getattr(open, {name})[0].cell_contents('/tmp/studio_dyn_closure.txt', 'w').write('x')\n"
        "print('DYN_CLOSURE_WROTE')\n",
        None,
        30,
        "backstop-dyn-closure",
        disable_sandbox = False,
    )
    assert "unsafe code detected" in out or "sandbox:" in out or "PermissionError" in out
    assert not os.path.exists("/tmp/studio_dyn_closure.txt")


@_POSIX_ONLY
def test_sandboxed_fileio_base_via_mro_blocked():
    # The guarded io.FileIO subclass exposes the unguarded C base at __mro__[1]; the
    # static gate now blocks the integer-indexed __mro__ base extraction.
    out = _python_exec(
        "import io\n"
        "io.FileIO.__mro__[1]('/tmp/studio_mro_escape.txt', 'w').write(b'x')\n"
        "print('MRO_WROTE')\n",
        None,
        30,
        "backstop-mro",
        disable_sandbox = False,
    )
    assert "unsafe code detected" in out or "sandbox:" in out or "PermissionError" in out
    assert not os.path.exists("/tmp/studio_mro_escape.txt")


@_POSIX_ONLY
def test_sandboxed_lstat_monkeypatch_symlink_escape_denied(tmp_path):
    # A pre-existing in-workdir symlink points outside. Sandboxed code monkeypatches
    # os.lstat to raise so os.path.realpath stops FOLLOWING the link, which would make
    # _within() resolve to the in-workdir link path while the real open() escapes through
    # it. The guard captures os.lstat/os.readlink and re-pins them before resolving, so
    # the write is still denied.
    session = "backstop-lstat-monkeypatch"
    workdir = get_sandbox_workdir(session)
    link = os.path.join(workdir, "lstat_escape_link")
    if os.path.islink(link) or os.path.exists(link):
        os.remove(link)
    os.symlink(str(tmp_path), link)
    target = tmp_path / "lstat_escape_probe.txt"
    if target.exists():
        target.unlink()
    try:
        out = _python_exec(
            "import os\n"
            "def _boom(*a, **k):\n"
            "    raise OSError('nope')\n"
            "os.lstat = _boom\n"
            "open('lstat_escape_link/lstat_escape_probe.txt', 'w').write('escaped')\n"
            "print('LSTAT_WROTE')\n",
            None,
            30,
            session,
            disable_sandbox = False,
        )
        assert "sandbox:" in out or "PermissionError" in out
        assert not target.exists()
    finally:
        os.remove(link)


@_POSIX_ONLY
def test_sandboxed_posix_chdir_escape_denied():
    # os re-exports chdir from the low-level C module posix; patching os.chdir leaves
    # posix.chdir importable with the original, so posix.chdir('/etc') would move cwd
    # outside the workdir and let a later relative read escape. The low-level module
    # must be guarded too.
    out = _python_exec(
        "import posix\nposix.chdir('/etc')\nimport os\nprint('CWD', os.getcwd())",
        None,
        30,
        "backstop-posix-chdir",
        disable_sandbox = False,
    )
    assert "sandbox:" in out and "chdir" in out


@_POSIX_ONLY
def test_sandboxed_posix_fd_metadata_mutator_denied(tmp_path):
    # posix.fchmod / posix.fchown are the low-level twins of os.fchmod/fchown; after a
    # read-only outside fd is allowed, they must still be denied so host metadata cannot
    # be mutated through the unwrapped C module.
    victim = tmp_path / "posix_victim.txt"
    victim.write_text("x")
    os.chmod(victim, 0o600)
    out = _python_exec(
        "import posix, os\n"
        f"fd = posix.open({str(victim)!r}, os.O_RDONLY)\n"
        "posix.fchmod(fd, 0o777); print('CHMODDED')",
        None,
        30,
        "backstop-posix-fchmod",
        disable_sandbox = False,
    )
    assert "sandbox:" in out and "fchmod" in out
    assert oct(os.stat(victim).st_mode & 0o777) == "0o600"


_SECRET_ABS = "/" + "etc" + "/" + "passwd"


@_POSIX_ONLY
def test_sandboxed_opaque_read_of_secret_denied():
    # The static scanner cannot fold an opaque read path (globals()['x']), and reads are
    # otherwise unconfined, so the runtime sensitive-read backstop must deny a read whose
    # realpath resolves to a host secret regardless of how the path was computed.
    out = _python_exec(
        "x = " + repr(_SECRET_ABS) + "\np = globals()['x']\nprint('LEN', len(open(p).read()))\n",
        None,
        30,
        "backstop-opaque-read",
        disable_sandbox = False,
    )
    assert "sandbox:" in out or "PermissionError" in out
    assert "reading a sensitive host path" in out
    assert "LEN " not in out


@_POSIX_ONLY
def test_sandboxed_opaque_os_open_read_of_secret_denied():
    # The same opaque path routed through the low-level os.open read entry point.
    out = _python_exec(
        "import os\n"
        "x = " + repr(_SECRET_ABS) + "\n"
        "p = globals()['x']\n"
        "fd = os.open(p, os.O_RDONLY); print('FD', fd)\n",
        None,
        30,
        "backstop-opaque-osopen",
        disable_sandbox = False,
    )
    assert "sandbox:" in out or "PermissionError" in out
    assert "reading a sensitive host path" in out
    assert "FD " not in out


@_POSIX_ONLY
def test_sandboxed_symlink_read_of_secret_denied(tmp_path):
    # A pre-existing in-workdir symlink pointing at a host secret: the static scanner sees a
    # benign local name ('notes.txt'), only the runtime realpath backstop can follow the
    # link and deny the read. (Sandboxed code cannot create the symlink; this is the
    # defense-in-depth the runtime layer adds over static analysis.)
    session = "backstop-symlink-read"
    workdir = get_sandbox_workdir(session)
    link = os.path.join(workdir, "notes.txt")
    if os.path.islink(link) or os.path.exists(link):
        os.remove(link)
    os.symlink(_SECRET_ABS, link)
    try:
        out = _python_exec(
            "print('LEN', len(open('notes.txt').read()))\n",
            None,
            30,
            session,
            disable_sandbox = False,
        )
        assert "sandbox:" in out or "PermissionError" in out
        assert "reading a sensitive host path" in out
        assert "LEN " not in out
    finally:
        os.remove(link)


@_POSIX_ONLY
def test_sandboxed_benign_outside_read_allowed():
    # Reads are not confined to the workdir; only sensitive realpaths are denied. A benign
    # outside read (and importing libraries whose files carry 'credentials'/'.pem' in the
    # name) must stay allowed so the backstop does not break normal computation.
    out = _python_exec(
        "print('HOST', open('/etc/hostname').read().strip()[:0] == '')\n"
        "import json, urllib.request, ssl, email\nprint('IMPORTS_OK')",
        None,
        30,
        "backstop-benign-read",
        disable_sandbox = False,
    )
    assert "IMPORTS_OK" in out
    assert "sandbox:" not in out


@_POSIX_ONLY
def test_sandboxed_opaque_pathlib_read_of_secret_denied():
    # A dynamically assembled pathlib receiver (Path(globals()['P']).read_text()) has no
    # literal for the static scanner. Path.open must apply the runtime sensitive-read
    # backstop so pathlib reads cannot exfiltrate a host secret.
    out = _python_exec(
        "from pathlib import Path\n"
        "P = " + repr(_SECRET_ABS) + "\n"
        "print('LEN', len(Path(globals()['P']).read_text()))\n",
        None,
        30,
        "backstop-pathlib-read",
        disable_sandbox = False,
    )
    assert "sandbox:" in out or "PermissionError" in out
    assert "reading a sensitive host path" in out
    assert "LEN " not in out


@_POSIX_ONLY
def test_sandboxed_pathlib_open_read_of_secret_denied():
    # The same via Path(...).open().read() rather than read_text().
    out = _python_exec(
        "from pathlib import Path\n"
        "P = " + repr(_SECRET_ABS) + "\n"
        "print('LEN', len(Path(globals()['P']).open().read()))\n",
        None,
        30,
        "backstop-pathlib-open-read",
        disable_sandbox = False,
    )
    assert "sandbox:" in out or "PermissionError" in out
    assert "reading a sensitive host path" in out
    assert "LEN " not in out


@_POSIX_ONLY
def test_sandboxed_pathlib_local_read_allowed():
    # A workdir-local pathlib read stays allowed.
    out = _python_exec(
        "from pathlib import Path\n"
        "Path('note.txt').write_text('hi')\n"
        "print('GOT', Path('note.txt').read_text())\n",
        None,
        30,
        "backstop-pathlib-local",
        disable_sandbox = False,
    )
    assert "GOT hi" in out
    assert "sandbox:" not in out


@_POSIX_ONLY
def test_sandboxed_workdir_module_shadowing_neutralized(tmp_path):
    # The exec script lives in the workdir, so Python prepends the workdir to sys.path[0].
    # A malicious re.py / pathlib.py / os.py / io.py / shutil.py dropped in the workdir must
    # NOT shadow the guard's own imports (which would run unguarded at import time before any
    # patch). shutil is imported late in the prelude, so sys.path stays stripped throughout.
    session = "backstop-shadow"
    workdir = get_sandbox_workdir(session)
    marker = os.path.join(str(tmp_path), "shadow_ran.marker")
    evil = "import builtins as _b\n_b.open(%r, 'w').write('pwned')\nraise SystemExit\n" % marker
    written = []
    for name in ("re.py", "pathlib.py", "os.py", "io.py", "shutil.py"):
        p = os.path.join(workdir, name)
        with open(p, "w") as fh:
            fh.write(evil)
        written.append(p)
    try:
        # A benign snippet: if any guard import is shadowed, evil runs and writes the marker.
        out = _python_exec("print('OK', 1 + 1)", None, 30, session, disable_sandbox = False)
        assert "OK 2" in out
        assert not os.path.exists(marker), "workdir module shadowed a guard import"
        # The real, patched modules stay usable for ordinary user imports.
        out2 = _python_exec(
            "import re, pathlib\nprint('REOK', bool(re.match('a', 'abc')))\n",
            None,
            30,
            session,
            disable_sandbox = False,
        )
        assert "REOK True" in out2
    finally:
        for p in written:
            if os.path.exists(p):
                os.remove(p)
        _pyc = os.path.join(workdir, "__pycache__")
        if os.path.isdir(_pyc):
            import shutil
            shutil.rmtree(_pyc, ignore_errors = True)
        if os.path.exists(marker):
            os.remove(marker)


@_POSIX_ONLY
def test_sandboxed_keyword_path_mutator_allowed_and_confined():
    # The mutator guard must accept the path via its public keyword (os.makedirs(name=...),
    # os.mkdir(path=...)) instead of raising TypeError on a missing positional argument, while
    # still confining the write. Clean any leftover dir first so the test is idempotent.
    session = "backstop-kwpath"
    workdir = get_sandbox_workdir(session)
    import shutil as _sh

    _sh.rmtree(os.path.join(workdir, "kwdir"), ignore_errors = True)
    out = _python_exec(
        "import os\nos.makedirs(name='kwdir/sub', exist_ok=True)\n"
        "os.mkdir(path='kwdir/one')\nprint('KW-OK', os.path.isdir('kwdir/sub'))",
        None,
        30,
        session,
        disable_sandbox = False,
    )
    assert "KW-OK True" in out
    assert "sandbox:" not in out
    _sh.rmtree(os.path.join(workdir, "kwdir"), ignore_errors = True)


@_POSIX_ONLY
def test_sandboxed_keyword_path_mutator_escape_denied(tmp_path):
    # A keyword path outside the workdir is still confined.
    target = tmp_path / "kw_escape"
    out = _python_exec(
        f"import os\nos.makedirs(name={str(target)!r}, exist_ok=True)\nprint('MADE')",
        None,
        30,
        "backstop-kwpath-escape",
        disable_sandbox = False,
    )
    assert "sandbox:" in out or "PermissionError" in out
    assert not target.exists()


# The sensitive directory path is assembled from chr() codepoints at runtime so the static
# scanner cannot const-fold it (proving the RUNTIME dir-reader guard, not the static layer).
_OPAQUE_SSH = "P=''.join(chr(c) for c in [47,114,111,111,116,47,46,115,115,104])\n"


@_POSIX_ONLY
@pytest.mark.parametrize(
    "reader",
    [
        "import os\nos.listdir(P)",
        "import os\nlist(os.scandir(P))",
        "import pathlib\nlist(pathlib.Path(P).iterdir())",
    ],
)
def test_sandboxed_opaque_sensitive_dir_read_denied(reader):
    # A directory-enumeration API (os.listdir / os.scandir / Path.iterdir) does not route
    # through open(), so an opaque sensitive host directory would leak its contents/names past
    # the open-like backstop. The runtime guard applies the sensitive-read check to the dir path.
    out = _python_exec(_OPAQUE_SSH + reader, None, 30, "backstop-diropaque", disable_sandbox = False)
    assert "sandbox:" in out or "PermissionError" in out


@_POSIX_ONLY
def test_sandboxed_workdir_dir_read_allowed():
    # Enumerating the sandbox's own workdir stays allowed (positional and keyword forms).
    out = _python_exec(
        "import os\nopen('a.txt', 'w').write('x')\n"
        "print('LS', 'a.txt' in os.listdir('.'))\n"
        "print('SC', any(e.name == 'a.txt' for e in os.scandir(path='.')))\n",
        None,
        30,
        "backstop-dirlocal",
        disable_sandbox = False,
    )
    assert "LS True" in out
    assert "SC True" in out
    assert "sandbox:" not in out


@_POSIX_ONLY
def test_sandboxed_fresh_builtin_module_open_confined(tmp_path):
    # _imp.create_builtin(posix.__spec__) mints a FRESH posix module with the original
    # unwrapped C open(), sidestepping the guards on the existing posix object. The guard
    # re-wraps a freshly created builtin module, so its open() to an outside path is confined.
    target = tmp_path / "fresh_builtin_escape.txt"
    out = _python_exec(
        "import _imp, posix\n"
        f"m = _imp.create_builtin(posix.__spec__)\n"
        f"m.open({str(target)!r}, posix.O_CREAT | posix.O_WRONLY)\n"
        "print('MADE')\n",
        None,
        30,
        "backstop-freshbuiltin",
        disable_sandbox = False,
    )
    assert "sandbox:" in out or "PermissionError" in out
    assert not target.exists()


@_POSIX_ONLY
def test_sandboxed_fresh_builtin_local_write_allowed():
    # A freshly created posix module can still write INSIDE the workdir (the re-applied guard
    # only confines escapes), so ordinary use keeps working.
    workdir = get_sandbox_workdir("backstop-freshbuiltin-local")
    out = _python_exec(
        "import _imp, posix, os\n"
        "m = _imp.create_builtin(posix.__spec__)\n"
        "fd = m.open('fresh_local.txt', posix.O_CREAT | posix.O_WRONLY)\n"
        "os.write(fd, b'x')\nos.close(fd)\nprint('LOCAL-OK')\n",
        None,
        30,
        "backstop-freshbuiltin-local",
        disable_sandbox = False,
    )
    assert "LOCAL-OK" in out
    assert "sandbox:" not in out
    _p = os.path.join(workdir, "fresh_local.txt")
    if os.path.exists(_p):
        os.remove(_p)


@_POSIX_ONLY
def test_sandboxed_user_site_usercustomize_not_run():
    # HOME points at the workdir, so one run could drop
    # .local/lib/pythonX.Y/site-packages/usercustomize.py that Python imports at the NEXT
    # child's startup, before the injected guard runs, executing writers unpatched. The
    # sandboxed interpreter runs with -s (user site disabled), so it is never imported.
    session = "backstop-usersite"
    workdir = get_sandbox_workdir(session)
    ver = "python%d.%d" % (sys.version_info[0], sys.version_info[1])
    usdir = os.path.join(workdir, ".local", "lib", ver, "site-packages")
    os.makedirs(usdir, exist_ok = True)
    marker = os.path.join(workdir, "usercustomize_ran.marker")
    if os.path.exists(marker):
        os.remove(marker)
    with open(os.path.join(usdir, "usercustomize.py"), "w") as fh:
        fh.write("open(%r, 'w').write('pwned')\n" % marker)
    try:
        out = _python_exec("print('OK', 1 + 1)", None, 30, session, disable_sandbox = False)
        assert "OK 2" in out
        assert not os.path.exists(marker), "user-site usercustomize.py ran before the guard"
    finally:
        import shutil
        shutil.rmtree(os.path.join(workdir, ".local"), ignore_errors = True)
        if os.path.exists(marker):
            os.remove(marker)


@_POSIX_ONLY
def test_sandboxed_dir_reader_stateful_fspath_confined(tmp_path):
    # A stateful __fspath__ returns an in-workdir path for the guard's check, then a sensitive
    # outside directory for the real listdir (a TOCTOU). The guard materializes the path ONCE
    # and passes that same value to listdir, so the second (outside) resolution never reaches
    # the real call: the workdir is listed, not the outside directory.
    secret = tmp_path / "SECRET_MARKER_FILE.txt"
    secret.write_text("x")
    code = (
        "import os\n"
        "class Evil:\n"
        "    def __init__(self):\n        self.n = 0\n"
        "    def __fspath__(self):\n"
        "        self.n += 1\n"
        f"        return '.' if self.n == 1 else {str(tmp_path)!r}\n"
        "print('LIST', os.listdir(Evil()))\n"
    )
    out = _python_exec(code, None, 30, "backstop-dir-toctou", disable_sandbox = False)
    # The outside directory's contents must not leak through the re-resolving path object.
    assert "SECRET_MARKER_FILE.txt" not in out


@pytest.mark.parametrize(
    "command",
    [
        "cat /etc/passwd | head -1",
        "head -1 /etc/shadow",
        "tr a b < /etc/passwd",
        "cat ~/.ssh/id_rsa",
        "cat ../../../../etc/passwd",
        "cat${IFS}/etc/shadow",
        "head /etc/shad*",
    ],
)
def test_bash_sensitive_read_blocked(command):
    # The terminal tool runs an unguarded shell child (no open() backstop), so an embedded
    # host-secret read must be refused statically before the subprocess is spawned.
    out = _bash_exec(command, None, 30, "bash-read-block", disable_sandbox = False)
    assert "sensitive file read" in out, out


@pytest.mark.parametrize(
    "command",
    [
        "echo hello world",
        "cat notes.txt",
        "ls ../src",
        "cat ../sibling/data.txt",
        "sort input.txt",
    ],
)
def test_bash_benign_read_scan_allows(command):
    # Ordinary in-tree relative navigation and non-sensitive reads are not flagged by the
    # sensitive-read scanner (they may still be shaped/executed, but not blocked as a read).
    assert _command_reads_sensitive(command) is None, command


def test_bash_bypass_permissions_skips_read_scan():
    # Bypass Permissions intentionally disables the static command scans.
    assert _command_reads_sensitive("cat /etc/passwd") is not None


@pytest.mark.parametrize(
    "command",
    [
        # Sensitive reads hidden behind an assignment / wrapper / nested-shell prefix, and
        # brace-expanded readers, must be refused before the unguarded bash child runs.
        "P=/etc/passwd cat ${P-/etc/passwd}",
        "bash -c 'cat /etc/passwd'",
        "sh -c 'cat ../../../../etc/passwd'",
        "bash -c 'sh -c \"cat /etc/passwd\"'",
        "{cat,/etc/passwd}",
    ],
)
def test_bash_prefixed_and_brace_read_blocked(command):
    out = _bash_exec(command, None, 30, "bash-read-prefix", disable_sandbox = False)
    assert "sensitive file read" in out, out


@pytest.mark.parametrize(
    "command",
    [
        # Brace expansion of a writer / interpreter must still be caught by the command scan.
        "{touch,/tmp/escape}",
        "{rm,-rf,/tmp/x}",
    ],
)
def test_bash_brace_expanded_writer_blocked(command):
    out = _bash_exec(command, None, 30, "bash-brace-writer", disable_sandbox = False)
    assert "Blocked command" in out, out


@pytest.mark.parametrize(
    "command",
    [
        # Benign prefixed / brace forms are not flagged as sensitive reads.
        "env FOO=bar grep pattern src/app.py",
        "bash -c 'ls -la'",
        "echo {a,b,c}",
        "X=1 cat notes.txt",
    ],
)
def test_bash_benign_prefixed_read_scan_allows(command):
    assert _command_reads_sensitive(command) is None, command


@_POSIX_ONLY
def test_sandboxed_poisoned_isinstance_write_denied(tmp_path):
    # Reassigning builtins.isinstance must not make the guard's isinstance(path, int) fd check
    # lie and approve an absolute write outside the workdir; the guard uses pinned builtins.
    target = tmp_path / "poison_isinstance.txt"
    out = _python_exec(
        "import builtins\n"
        "builtins.isinstance = lambda *a, **k: True\n"
        f"open({str(target)!r}, 'w').write('x'); print('WROTE')\n",
        None,
        30,
        "backstop-poison-isinstance",
        disable_sandbox = False,
    )
    assert "sandbox:" in out or "PermissionError" in out
    assert not target.exists()


@_POSIX_ONLY
def test_sandboxed_poisoned_s_islnk_symlink_write_denied(tmp_path):
    # Reassigning os.path.stat.S_ISLNK so realpath stops following an in-workdir symlink that
    # escapes must not let the write through; the guard re-pins S_ISLNK before each resolve.
    session = "backstop-poison-islnk"
    workdir = get_sandbox_workdir(session)
    link = os.path.join(workdir, "islnk_escape")
    if os.path.islink(link) or os.path.exists(link):
        os.remove(link)
    os.symlink(str(tmp_path), link)
    victim = tmp_path / "poison_islnk.txt"
    try:
        out = _python_exec(
            "import os.path\n"
            "os.path.stat.S_ISLNK = lambda mode: False\n"
            "open('islnk_escape/poison_islnk.txt', 'w').write('x'); print('WROTE')\n",
            None,
            30,
            session,
            disable_sandbox = False,
        )
        assert "sandbox:" in out or "PermissionError" in out
        assert not victim.exists()
    finally:
        os.remove(link)


def test_runtime_is_sensitive_read_covers_root_home():
    # The runtime backstop's _is_sensitive_read must protect /root dotfiles/caches while
    # carving out package/library trees so imports under a root home are not broken.
    import re as _re

    src = _SANDBOX_GUARD_SRC
    ns = {"_re": _re}
    block = src[src.index("_SENS_EXACT = ") : src.index("def _read_realpath")]
    exec(block, ns)
    f = ns["_is_sensitive_read"]
    assert f("/root/.bashrc") is True
    assert f("/root/.cache/secret") is True
    assert f("/root/.local/lib/python3.13/site-packages/certifi/cacert.pem") is False
    assert f("/root/miniconda3/lib/python3.13/os.py") is False
    assert f("/home/ubuntu/project/data.txt") is False


def test_runtime_is_sensitive_read_covers_exact_root():
    # os.listdir('/root') (P = '/root' computed dynamically) enumerates the root home itself;
    # the runtime backstop must treat the exact /root path as sensitive, not only /root/*.
    import re as _re

    ns = {"_re": _re}
    block = _SANDBOX_GUARD_SRC[
        _SANDBOX_GUARD_SRC.index("_SENS_EXACT = ") : _SANDBOX_GUARD_SRC.index("def _read_realpath")
    ]
    exec(block, ns)
    f = ns["_is_sensitive_read"]
    assert f("/root") is True
    assert f("/root/") is True
    assert f("/root/.local/lib/python3.13/site-packages/x.py") is False


@_POSIX_ONLY
@pytest.mark.parametrize("meth", ["glob", "rglob"])
def test_sandboxed_pathlib_glob_sensitive_dir_denied(meth):
    # Path(P).glob('*') / rglob enumerate a directory through pathlib internals; a dynamically
    # built receiver pointing (via an in-workdir symlink) at a sensitive dir must be screened
    # the same way Path.iterdir is.
    session = "backstop-glob-" + meth
    workdir = get_sandbox_workdir(session)
    link = os.path.join(workdir, "ssh_link_" + meth)
    if os.path.islink(link) or os.path.exists(link):
        os.remove(link)
    os.symlink("/etc/ssh", link)  # /etc/ssh/ is a sensitive directory
    try:
        out = _python_exec(
            "from pathlib import Path\n"
            f"print('N', len(list(Path('ssh_link_{meth}').{meth}('*'))))\n",
            None,
            30,
            session,
            disable_sandbox = False,
        )
        assert "sandbox:" in out or "PermissionError" in out
    finally:
        os.remove(link)


@_POSIX_ONLY
@pytest.mark.parametrize("path", ["/dev/null", "/dev/stdout", "/dev/stderr"])
def test_sandboxed_device_sink_write_allowed(path):
    # A write to a standard device sink cannot persist data outside the workspace, so it is
    # allowed (mirrors the terminal redirect allowlist) rather than denied by the workdir check.
    out = _python_exec(
        f"open({path!r}, 'w').write('x'); print('WROTE_SINK')",
        None,
        30,
        "backstop-devsink",
        disable_sandbox = False,
    )
    assert "WROTE_SINK" in out
    assert "sandbox:" not in out


@_POSIX_ONLY
def test_sandboxed_os_devnull_write_allowed():
    out = _python_exec(
        "import os\nopen(os.devnull, 'w').write('x'); print('WROTE_OSDEVNULL')",
        None,
        30,
        "backstop-osdevnull",
        disable_sandbox = False,
    )
    assert "WROTE_OSDEVNULL" in out
    assert "sandbox:" not in out


@_POSIX_ONLY
def test_sandboxed_device_sink_str_subclass_escape_denied(tmp_path):
    # The device-sink allowlist must not trust a str subclass's replace(): a subclass whose
    # replace() returns '/dev/null' while its real value is an outside file would otherwise
    # pass the sink shortcut and write outside the workdir. The guard normalizes via the base
    # str.replace, so the real (outside) path is seen and the write is denied.
    target = tmp_path / "sink_escape.txt"
    code = (
        "class P(str):\n"
        "    def replace(self, *a, **k):\n"
        "        return '/dev/null'\n"
        f"open(P({str(target)!r}), 'w').write('x'); print('WROTE_ESCAPE')"
    )
    out = _python_exec(code, None, 30, "backstop-devsink-subclass", disable_sandbox = False)
    assert "sandbox:" in out or "PermissionError" in out
    assert not target.exists()


# '/root' assembled from chr() codepoints so the static scanner cannot const-fold it, proving
# the RUNTIME guard on the low-level posix module (not the static layer).
_OPAQUE_ROOT = "P=''.join(chr(c) for c in [47,114,111,111,116])\n"


@_POSIX_ONLY
@pytest.mark.parametrize(
    "reader",
    [
        "import posix\nposix.listdir(P)",
        "import posix\nlist(posix.scandir(P))",
    ],
)
def test_sandboxed_low_level_posix_dir_read_denied(reader):
    # posix.listdir / posix.scandir re-export the ORIGINAL enumerators, so the os.* dir guard
    # left them reachable for an opaque sensitive path; the runtime guard now confines them too.
    out = _python_exec(_OPAQUE_ROOT + reader, None, 30, "backstop-posixdir", disable_sandbox = False)
    assert "sandbox:" in out or "PermissionError" in out


@_POSIX_ONLY
def test_sandboxed_fresh_posix_module_fd_denier_reapplied():
    # A fresh posix module built via _imp.create_builtin re-exposes the original fd metadata
    # mutators; _reguard_created must reapply the fchmod/fchown deniers so a read-only open of an
    # outside file cannot be reused to mutate host metadata.
    code = (
        "import _imp, posix\n"
        "m = _imp.create_builtin(posix.__spec__)\n"
        "try:\n"
        "    fd = m.open('/etc/hostname', 0)\n"
        "    m.fchmod(fd, 0o777)\n"
        "    print('MUTATED')\n"
        "except Exception as e:\n"
        "    print(repr(e))"
    )
    out = _python_exec(code, None, 30, "backstop-freshfchmod", disable_sandbox = False)
    assert "MUTATED" not in out
    assert "sandbox:" in out or "PermissionError" in out


@_POSIX_ONLY
def test_sandboxed_fresh_posix_module_dir_read_denied():
    # The fresh posix module's directory readers are guarded too (opaque sensitive path).
    code = (
        "import _imp, posix\n" + _OPAQUE_ROOT + "m = _imp.create_builtin(posix.__spec__)\n"
        "try:\n"
        "    print(m.listdir(P))\n"
        "except Exception as e:\n"
        "    print(repr(e))"
    )
    out = _python_exec(code, None, 30, "backstop-freshlistdir", disable_sandbox = False)
    assert "sandbox:" in out or "PermissionError" in out


@_POSIX_ONLY
def test_sandboxed_low_level_posix_workdir_read_allowed():
    # Enumerating the sandbox's own workdir through the low-level module stays allowed.
    out = _python_exec(
        "import posix, os\nos.makedirs('subd', exist_ok=True)\nprint('LS', posix.listdir('subd'))",
        None,
        30,
        "backstop-posixdir-ok",
        disable_sandbox = False,
    )
    assert "LS" in out
    assert "sandbox:" not in out


@_POSIX_ONLY
def test_sandboxed_future_import_same_line_write_denied(tmp_path):
    # A `from __future__ import ...; open(<outside>, 'w')` puts a real write on the SAME physical
    # line as the future import; the guard prelude must still be installed BEFORE that write, so it
    # is confined by the runtime backstop rather than running unguarded.
    target = tmp_path / "future_sameline_escape.txt"
    out = _python_exec(
        f"from __future__ import annotations; open({str(target)!r}, 'w').write('x'); print('DONE')",
        None,
        30,
        "backstop-future-sameline",
        disable_sandbox = False,
    )
    assert "sandbox:" in out or "PermissionError" in out
    assert not target.exists()


@_POSIX_ONLY
def test_sandboxed_future_import_own_line_benign_allowed():
    # The ordinary form (future import on its own line, an in-workdir write after) still works.
    out = _python_exec(
        "from __future__ import annotations\n"
        "open('future_ok.txt', 'w').write('hi')\nprint('WROTE_OK')",
        None,
        30,
        "backstop-future-ok",
        disable_sandbox = False,
    )
    assert "WROTE_OK" in out
    assert "sandbox:" not in out


@_POSIX_ONLY
def test_sandboxed_module_docstring_preserved():
    # A leading string literal is the module docstring only while it is the FIRST statement, so the
    # guard prelude must be spliced AFTER it (not prepended) -- otherwise __doc__ becomes None.
    out = _python_exec(
        '"""studio doc marker"""\nprint(__doc__)',
        None,
        30,
        "backstop-docstring",
        disable_sandbox = False,
    )
    assert "studio doc marker" in out
    assert "sandbox:" not in out


@_POSIX_ONLY
def test_sandboxed_docstring_same_line_write_denied(tmp_path):
    # A `"""doc"""; open(<outside>, 'w')` puts a real write on the SAME line as the docstring; the
    # guard prelude must still be installed BEFORE that write while keeping the docstring first.
    target = tmp_path / "docstring_sameline_escape.txt"
    out = _python_exec(
        f'"""doc"""; open({str(target)!r}, "w").write("x"); print("DONE")',
        None,
        30,
        "backstop-docstring-sameline",
        disable_sandbox = False,
    )
    assert "sandbox:" in out or "PermissionError" in out
    assert not target.exists()


@_POSIX_ONLY
def test_sandboxed_assigned_os_alias_workdir_module_denied():
    # import os; o = os; o.system(...) -- a whole-module assignment alias (not `import os as o`)
    # must be followed in the vetter pre-pass so the aliased sink receiver is recognized.
    _assert_workdir_module_denied(
        "backstop-workdir-osassign",
        "evilassign",
        "import os\no = os\nprint('R61_ASSIGN')\no.system('echo PWN')\n",
        "R61_ASSIGN",
    )


@_POSIX_ONLY
def test_sandboxed_module_dict_subscript_workdir_module_denied():
    # import os; os.__dict__['system'](...) -- a namespace-dict subscript through a guarded
    # module's __dict__ reaches the sink the attribute checks miss; fail closed like vars(os).
    _assert_workdir_module_denied(
        "backstop-workdir-osdict",
        "evildict",
        "import os\nprint('R61_DICT')\nos.__dict__['system']('echo PWN')\n",
        "R61_DICT",
    )


@_POSIX_ONLY
def test_sandboxed_module_getattribute_workdir_module_denied():
    # import os; os.__getattribute__('system')(...) -- a dynamic attribute lookup via the module
    # dunder reaches the sink the builtin getattr(...) branch misses.
    _assert_workdir_module_denied(
        "backstop-workdir-osgetattr",
        "evilga",
        "import os\nprint('R61_GA')\nos.__getattribute__('system')('echo PWN')\n",
        "R61_GA",
    )


@_POSIX_ONLY
def test_sandboxed_mro_recovery_workdir_module_denied():
    # io.FileIO.__mro__[1](...) recovers an unguarded base class in a vetted workdir helper; the
    # vetter must treat __mro__ / mro as a gadget.
    _assert_workdir_module_denied(
        "backstop-workdir-mro",
        "evilmro",
        "import io\nprint('R61_MRO')\nc = io.FileIO.__mro__[1]\n",
        "R61_MRO",
    )


@_POSIX_ONLY
def test_sandboxed_sys_modules_subscript_workdir_module_denied():
    # import sys; sys.modules['os'].system(...) recovers the guard-cached os module without an
    # import, bypassing the denied-import path; access to sys.modules must be denied.
    _assert_workdir_module_denied(
        "backstop-workdir-sysmods",
        "evilsysmods",
        "import sys\nprint('R61_SYSMODS')\nsys.modules['os'].system('echo PWN')\n",
        "R61_SYSMODS",
    )


@_POSIX_ONLY
def test_sandboxed_benign_os_alias_workdir_module_allowed():
    # A benign whole-module alias that only calls a NON-sink os attribute (o = os; o.getcwd())
    # must still import -- the alias-following must not over-block ordinary os use.
    session = "backstop-workdir-benignalias"
    workdir = get_sandbox_workdir(session)
    with open(os.path.join(workdir, "okalias.py"), "w") as f:
        f.write("import os\no = os\nCWD = o.getcwd()\nprint('OKALIAS_' + 'BODY')\n")
    try:
        out = _python_exec(
            "import okalias; print('IMPORTED_' + 'OK')",
            None,
            30,
            session,
            disable_sandbox = False,
        )
        assert "IMPORTED_OK" in out
        assert "sandbox:" not in out
    finally:
        os.remove(os.path.join(workdir, "okalias.py"))


@_POSIX_ONLY
def test_sandboxed_sqlite_uri_xmode_memory_escape_denied(tmp_path):
    # file:<path>?xmode=memory is an on-disk file (SQLite ignores the unknown xmode key), so the
    # in-memory skip must NOT apply -- an escaping path via a uri connection is confined.
    target = tmp_path / "sqlite_uri_escape.db"
    out = _python_exec(
        f"import sqlite3\nsqlite3.connect('file:{target}?xmode=memory', uri=True)\nprint('OPENED')",
        None,
        30,
        "backstop-sqlite-uri-xmode",
        disable_sandbox = False,
    )
    assert "sandbox:" in out or "PermissionError" in out
    assert not target.exists()


@_POSIX_ONLY
def test_sandboxed_sqlite_uri_real_memory_allowed():
    # A genuine mode=memory URI parameter is a real in-memory database and must stay allowed.
    out = _python_exec(
        "import sqlite3\n"
        "c = sqlite3.connect('file:r61mem?mode=memory&cache=shared', uri=True)\n"
        "c.execute('create table t(x)')\nprint('MEM_OK')",
        None,
        30,
        "backstop-sqlite-uri-mem",
        disable_sandbox = False,
    )
    assert "MEM_OK" in out
    assert "sandbox:" not in out
