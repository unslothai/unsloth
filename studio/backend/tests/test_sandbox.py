# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
End-to-end tests for the OS-level sandbox wired into ``_python_exec``
and ``_bash_exec``.

The platform-agnostic tests (workdir write, $HOME read deny, bash $HOME
read deny, network deny) run on both macOS (Seatbelt) and Linux
(bubblewrap) — same security claims, different mechanisms. The
``/System/Applications``-enumeration test is darwin-specific because
that path only exists on macOS.

These tests are the only layer that proves the sandbox does what it
claims — anything that only inspects the profile string is checking
typography, not enforcement.
"""

import importlib.util
import os
import shlex
import sys
import uuid
from pathlib import Path

import pytest

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

pytestmark = pytest.mark.skipif(
    sys.platform not in ("darwin", "linux"),
    reason = "sandbox tests run on macOS and Linux only",
)


def _load_sandbox_module():
    # Bypass core.inference.__init__ (pulls orchestrator/fastapi/structlog).
    path = _BACKEND_ROOT / "core" / "inference" / "sandbox.py"
    spec = importlib.util.spec_from_file_location("_studio_sandbox_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def sandboxed_workdir(tmp_path, monkeypatch):
    """Point tool execution's workdir lookup at a pytest tmp_path."""
    sandbox = _load_sandbox_module()

    if not sandbox.sandbox_available():
        if os.environ.get("CI") == "true":
            pytest.fail(
                "sandbox unavailable: CI must install/enable bubblewrap or "
                "sandbox-exec so this enforcement test actually runs"
            )
        pytest.skip("sandbox unavailable (binary missing or cannot apply policy)")

    from core.inference import tools

    sid = "_sbtest"
    monkeypatch.setitem(tools._workdirs, sid, str(tmp_path))
    yield sid, str(tmp_path)


@pytest.fixture
def home_sentinel(tmp_path_factory):
    """Yield a sentinel file path + secret, kept outside the sandbox workdir.

    The sentinel proves *negative*: if the sandboxed code reads the
    file, the secret appears in the tool output. Placed under a
    pytest tmp_path so the test is hermetic and works on rootless CI
    where the real $HOME is read-only.
    """
    secret = f"SECRET-{uuid.uuid4().hex}"
    sentinel_dir = tmp_path_factory.mktemp("studio_sandbox_sentinel")
    path = str(sentinel_dir / f"sentinel_{uuid.uuid4().hex}.txt")
    Path(path).write_text(secret)
    try:
        yield path, secret
    finally:
        if os.path.exists(path):
            os.unlink(path)


def _run_python(code: str, sid: str) -> str:
    from core.inference.tools import _python_exec

    return _python_exec(code, session_id = sid, timeout = 30)


def _run_bash(command: str, sid: str) -> str:
    from core.inference.tools import _bash_exec

    return _bash_exec(command, session_id = sid, timeout = 30)


def test_workdir_write_succeeds(sandboxed_workdir):
    sid, wd = sandboxed_workdir
    code = (
        "from pathlib import Path\n"
        'Path("hi.txt").write_text("ok")\n'
        'print("done")\n'
    )
    out = _run_python(code, sid)
    assert "done" in out, out
    assert os.path.exists(os.path.join(wd, "hi.txt"))


def test_home_read_denied(sandboxed_workdir, home_sentinel):
    sid, _ = sandboxed_workdir
    path, secret = home_sentinel
    code = (
        f"try:\n"
        f"    with open({path!r}) as f: print('LEAKED:', f.read())\n"
        f"except (PermissionError, FileNotFoundError, OSError) as e:\n"
        f"    print('DENIED:', type(e).__name__)\n"
    )
    out = _run_python(code, sid)
    assert secret not in out, out
    assert "LEAKED:" not in out, out
    assert "DENIED:" in out, out


def test_bash_home_read_denied(sandboxed_workdir, home_sentinel):
    """The terminal tool must enforce the same $HOME-denial as the python tool."""
    sid, _ = sandboxed_workdir
    path, secret = home_sentinel
    out = _run_bash(f"/bin/cat {shlex.quote(path)}", sid)
    assert secret not in out, out
    # Confirm cat actually ran and was denied, not silently no-op'd.
    assert any(
        s in out
        for s in ("Permission denied", "Operation not permitted", "No such file")
    ), out


def test_network_denied(sandboxed_workdir):
    """Hit a routable IP so the test does not depend on DNS or external service.

    Host is assembled at runtime so the static AST allowlist does not
    pre-block it; only the sandbox can deny the egress.
    """
    sid, _ = sandboxed_workdir
    code = (
        "try:\n"
        "    import socket\n"
        "    host = '.'.join(['8', '8', '8', '8'])\n"
        "    s = socket.create_connection((host, 80), timeout=5)\n"
        "    s.close()\n"
        "    print('LEAKED')\n"
        "except Exception as e:\n"
        "    print('DENIED:', type(e).__name__)\n"
    )
    out = _run_python(code, sid)
    assert "LEAKED" not in out, out
    assert "DENIED:" in out, out


def test_sandbox_off_actually_leaks(tmp_path, monkeypatch, home_sentinel):
    """Control test: with the sandbox disabled, the sentinel IS readable.

    Without this, ``test_bash_home_read_denied`` would pass even if the
    sandbox silently no-op'd (binary missing, probe failed) — proving
    only that the sentinel UUID doesn't appear by chance, not that the
    sandbox is the thing blocking it.
    """
    from core.inference import tools

    monkeypatch.setattr(tools, "sandbox_available", lambda: False)
    sid = "_sbtest_off"
    monkeypatch.setitem(tools._workdirs, sid, str(tmp_path))

    path, secret = home_sentinel
    out = _run_bash(f"/bin/cat {shlex.quote(path)}", sid)
    assert secret in out, out


@pytest.mark.skipif(
    sys.platform != "darwin",
    reason = "/System/Applications is a macOS path",
)
def test_system_applications_enumeration_denied(sandboxed_workdir):
    """Pin the macOS narrowing: /System/Applications should not be readable.

    v1 of the macOS profile allowed all of /System; v2 narrowed it to
    Frameworks + dyld only. The Frameworks dir must remain readable
    (loading still works), while /System/Applications and
    /System/iOSSupport must not.
    """
    sid, _ = sandboxed_workdir
    out = _run_bash("ls /System/Applications 2>&1; ls /System/iOSSupport 2>&1", sid)
    assert "Operation not permitted" in out, out
    out_fw = _run_bash("ls /System/Library/Frameworks | head -1", sid)
    assert "Operation not permitted" not in out_fw, out_fw
    assert ".framework" in out_fw, out_fw


# ---------------------------------------------------------------------------
# Profile / argv construction tests. These run on any macOS or Linux host
# regardless of whether the sandbox can be applied in this process context.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(sys.platform != "darwin", reason = "Seatbelt is macOS-only")
def test_macos_profile_omits_dev_tty(tmp_path):
    sandbox = _load_sandbox_module()
    profile = sandbox._macos_seatbelt_profile(str(tmp_path))
    assert "/dev/tty" not in profile, profile


@pytest.mark.skipif(sys.platform != "darwin", reason = "Seatbelt is macOS-only")
def test_macos_profile_narrows_private_etc(tmp_path):
    sandbox = _load_sandbox_module()
    profile = sandbox._macos_seatbelt_profile(str(tmp_path))
    assert '(subpath "/private/etc")' not in profile, profile
    for required in (
        '(literal "/private/etc/hosts")',
        '(literal "/private/etc/resolv.conf")',
        '(subpath "/private/etc/ssl")',
    ):
        assert required in profile, required
    for forbidden in (
        "/private/etc/passwd",
        "/private/etc/shadow",
        "/private/etc/sudoers",
    ):
        assert forbidden not in profile, forbidden


@pytest.mark.skipif(sys.platform != "darwin", reason = "Seatbelt is macOS-only")
def test_macos_profile_constrains_process_exec(tmp_path):
    sandbox = _load_sandbox_module()
    profile = sandbox._macos_seatbelt_profile(str(tmp_path))
    assert "(allow process-exec)" not in profile
    assert "(allow process-exec\n" in profile


@pytest.mark.skipif(sys.platform != "linux", reason = "bwrap argv is Linux-only")
def test_linux_argv_narrows_etc(tmp_path, monkeypatch):
    sandbox = _load_sandbox_module()
    monkeypatch.setattr(sandbox, "_linux_bwrap_path", "/usr/bin/bwrap")
    argv = sandbox._linux_bwrap_argv(["/usr/bin/true"], str(tmp_path))
    bound_targets = set()
    bind_flags = ("--ro-bind", "--ro-bind-try", "--bind", "--bind-try")
    for i, token in enumerate(argv):
        if token in bind_flags and i + 2 < len(argv):
            bound_targets.add(argv[i + 2])
    assert "/etc" not in bound_targets
    for required in ("/etc/hosts", "/etc/resolv.conf", "/etc/ssl"):
        assert required in bound_targets, (required, bound_targets)


@pytest.mark.skipif(sys.platform != "linux", reason = "bwrap argv is Linux-only")
def test_linux_argv_asserts_when_bwrap_path_unset(tmp_path, monkeypatch):
    sandbox = _load_sandbox_module()
    monkeypatch.setattr(sandbox, "_linux_bwrap_path", None)
    with pytest.raises(AssertionError):
        sandbox._linux_bwrap_argv(["/usr/bin/true"], str(tmp_path))


def test_python_read_paths_excludes_user_site_by_default(monkeypatch, tmp_path):
    sandbox = _load_sandbox_module()
    import site

    fake_user_site = tmp_path / "fake_user_site_default"
    fake_user_site.mkdir()
    monkeypatch.setattr(site, "getusersitepackages", lambda: str(fake_user_site))
    monkeypatch.delenv("UNSLOTH_STUDIO_SANDBOX_ALLOW_USER_SITE", raising = False)
    paths = sandbox._python_read_paths()
    assert os.path.realpath(str(fake_user_site)) not in paths


def test_python_read_paths_includes_user_site_when_opted_in(monkeypatch, tmp_path):
    sandbox = _load_sandbox_module()
    import site

    fake_user_site = tmp_path / "fake_user_site_opt_in"
    fake_user_site.mkdir()
    monkeypatch.setattr(site, "getusersitepackages", lambda: str(fake_user_site))
    monkeypatch.setenv("UNSLOTH_STUDIO_SANDBOX_ALLOW_USER_SITE", "1")
    paths = sandbox._python_read_paths()
    assert os.path.realpath(str(fake_user_site)) in paths


def test_bwrap_probe_bin_exists_and_executable():
    sandbox = _load_sandbox_module()
    bin_path = sandbox._BWRAP_PROBE_BIN
    assert os.path.exists(bin_path), bin_path
    assert os.access(bin_path, os.X_OK), bin_path


def test_build_sandbox_argv_rejects_empty_inner(tmp_path):
    sandbox = _load_sandbox_module()
    with pytest.raises(ValueError):
        sandbox.build_sandbox_argv([], str(tmp_path))


def test_build_sandbox_argv_asserts_on_unsupported_platform(tmp_path, monkeypatch):
    sandbox = _load_sandbox_module()
    monkeypatch.setattr(sandbox.sys, "platform", "freebsd14")
    with pytest.raises(AssertionError):
        sandbox.build_sandbox_argv(["/usr/bin/true"], str(tmp_path))


# ---------------------------------------------------------------------------
# Workdir realpath: bwrap binds os.path.realpath(workdir); tmp_path inside
# inner_argv must resolve to the same bound path when $HOME is symlinked.
# ---------------------------------------------------------------------------


def test_get_workdir_returns_realpath_when_home_is_symlinked(tmp_path, monkeypatch):
    real_home = tmp_path / "real_home"
    real_home.mkdir()
    home_symlink = tmp_path / "home_symlink"
    os.symlink(real_home, home_symlink)
    monkeypatch.setattr(
        os.path,
        "expanduser",
        lambda p: str(home_symlink) if p == "~" else p,
    )

    from core.inference import tools

    tools._workdirs.pop("_realpath_test", None)
    wd = tools._get_workdir("_realpath_test")
    try:
        assert os.path.realpath(wd) == wd, wd
        assert str(home_symlink) not in wd, wd
        assert str(real_home) in wd, wd
    finally:
        tools._workdirs.pop("_realpath_test", None)


def test_get_workdir_idempotent(tmp_path, monkeypatch):
    monkeypatch.setattr(
        os.path,
        "expanduser",
        lambda p: str(tmp_path) if p == "~" else p,
    )

    from core.inference import tools

    tools._workdirs.pop("_idem_test", None)
    first = tools._get_workdir("_idem_test")
    second = tools._get_workdir("_idem_test")
    try:
        assert first == second
    finally:
        tools._workdirs.pop("_idem_test", None)
