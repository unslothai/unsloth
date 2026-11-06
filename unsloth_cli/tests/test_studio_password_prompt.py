# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The seeded admin password must change before public exposure; no terminal means the timeout."""

from __future__ import annotations

import io
import sqlite3
import os
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _studio():
    from unsloth_cli.commands import studio as _studio_mod
    return _studio_mod


_BASE = ["--model", "unsloth/Qwen3-1.7B-GGUF"]


@pytest.fixture(autouse = True)
def _no_leaked_unattended_marker(monkeypatch):
    """The gate sets the unattended marker in os.environ; left set, it silently suppresses prompts."""
    import unsloth_cli.commands.studio as studio_mod
    monkeypatch.delenv(studio_mod._UNATTENDED_PROMPT_DONE_ENV, raising = False)


_NEW_PW = "brand-new-password"


_TUNNEL_MATRIX = [
    (None, "127.0.0.1", True, False, True),
    (True, "127.0.0.1", True, False, True),
    (None, "127.0.0.1", True, True, True),
    (True, "0.0.0.0", False, False, True),
    (True, "::", False, False, True),
    (True, "::0", False, False, True),
    (True, "0:0:0:0:0:0:0:0", False, False, True),
    (True, "0", False, False, True),
    (True, "::ffff:0.0.0.0", False, False, True),
    (True, "", False, False, False),
    (True, "127.0.0.1", False, False, False),
    (True, "0.0.0.0", False, True, False),
    (None, "0.0.0.0", False, False, False),
    (False, "0.0.0.0", False, False, False),
    (None, "127.0.0.1", False, False, False),
]


@pytest.mark.parametrize("cloudflare,host,secure,api_only,expected", _TUNNEL_MATRIX)
def test_launch_publishes_tunnel_matrix(cloudflare, host, secure, api_only, expected):
    """A raw wildcard bind never strips .bootstrap_password, so the tunnel predicate must exclude it."""
    assert (
        _studio()._launch_publishes_tunnel(
            cloudflare = cloudflare, host = host, secure = secure, api_only = api_only
        )
        is expected
    )


@pytest.mark.parametrize(
    "cloudflare,host,secure,api_only,interactive,expected",
    [
        *[(c, h, s, a, False, e) for c, h, s, a, e in _TUNNEL_MATRIX if e],
        *[(c, h, s, a, True, e) for c, h, s, a, e in _TUNNEL_MATRIX if e],
        (None, "0.0.0.0", False, False, True, True),
        (False, "0.0.0.0", False, False, True, True),
        (None, "::", False, False, True, True),
        (None, "0", False, False, True, True),
        (None, "::ffff:0.0.0.0", False, False, True, True),
        (None, "192.168.1.50", False, False, True, True),
        (None, "10.0.0.5", False, False, True, True),
        (None, "172.16.4.9", False, False, True, True),
        (None, "example.com", False, False, True, True),
        (None, "myhost.local", False, False, True, True),
        (None, "[::]", False, False, True, True),
        (None, "0.0.0.0", False, False, False, False),
        (False, "0.0.0.0", False, False, False, False),
        (None, "::", False, False, False, False),
        (None, "192.168.1.50", False, False, False, False),
        (None, "0.0.0.0", False, True, True, False),
        (None, "192.168.1.50", False, True, True, False),
        (None, "127.0.0.1", False, False, True, False),
        (None, "localhost", False, False, True, False),
        (None, "::1", False, False, True, False),
        (None, "", False, False, True, False),
    ],
)
def test_should_prompt_password_change_matrix(
    monkeypatch, cloudflare, host, secure, api_only, interactive, expected
):
    monkeypatch.setattr(_studio(), "_prompt_streams_interactive", lambda: interactive)
    assert (
        _studio()._should_prompt_password_change(
            cloudflare = cloudflare, host = host, secure = secure, api_only = api_only
        )
        is expected
    )


class _ExecCaptured(SystemExit):
    def __init__(self, argv):
        super().__init__(0)
        self.argv = list(argv)


def _auth_db(studio_home: Path) -> Path:
    return studio_home / "auth" / "auth.db"


def _seed_auth(studio_mod, *, must_change = True):
    """Create the CLI-side default admin (must_change_password=1) plus one
    refresh token, mirroring a fresh install that served a login."""
    conn = studio_mod._connect_auth_db()
    try:
        studio_mod._ensure_cli_default_admin(conn)
        if not must_change:
            conn.execute("UPDATE auth_user SET must_change_password = 0")
        conn.execute(
            "INSERT INTO refresh_tokens (token_hash, username, expires_at) VALUES (?, ?, ?)",
            ("deadbeef", studio_mod.DEFAULT_ADMIN_USERNAME, "2099-01-01T00:00:00"),
        )
        conn.commit()
        row = conn.execute(
            "SELECT password_hash, jwt_secret FROM auth_user WHERE username = ?",
            (studio_mod.DEFAULT_ADMIN_USERNAME,),
        ).fetchone()
        return {"password_hash": row[0], "jwt_secret": row[1]}
    finally:
        conn.close()


def _auth_state(studio_mod):
    conn = sqlite3.connect(_auth_db(studio_mod.STUDIO_HOME))
    try:
        row = conn.execute(
            "SELECT password_hash, jwt_secret, must_change_password FROM auth_user "
            "WHERE username = ?",
            (studio_mod.DEFAULT_ADMIN_USERNAME,),
        ).fetchone()
        n_refresh = conn.execute("SELECT COUNT(*) FROM refresh_tokens").fetchone()[0]
        return {
            "password_hash": row[0],
            "jwt_secret": row[1],
            "must_change_password": row[2],
            "n_refresh": n_refresh,
        }
    finally:
        conn.close()


def _install_prompt_env(
    monkeypatch,
    tmp_path,
    *,
    interactive,
    scripted = _NEW_PW,
):
    """Tmp STUDIO_HOME + fake tty + scripted prompt. Returns the event log that
    records prompt calls and re-exec argv in order."""
    studio_mod = _studio()
    events = []

    monkeypatch.setattr(studio_mod, "STUDIO_HOME", tmp_path)
    monkeypatch.setattr(studio_mod, "_prompt_streams_interactive", lambda: interactive)
    monkeypatch.setattr(studio_mod, "_tunnel_binary_confirmed_unavailable", lambda: False)

    def fake_prompt(
        verify_current,
        out = None,
        **_kw,
    ):
        events.append(("prompt", verify_current))
        if isinstance(scripted, BaseException):
            raise scripted
        return scripted

    monkeypatch.setattr(studio_mod._password_prompt, "prompt_new_password", fake_prompt)
    return events


def _install_studio_default_reexec(monkeypatch, events):
    studio_mod = _studio()
    monkeypatch.setattr(sys, "prefix", "/nonexistent/outer/venv")
    monkeypatch.setattr(studio_mod, "_ensure_studio_env_exported", lambda: None)
    fake_venv = Path("/fake/studio/venv/unsloth_studio")
    monkeypatch.setattr(studio_mod, "_studio_venv_python", lambda: fake_venv / "bin" / "python")
    monkeypatch.setattr(studio_mod, "_find_run_py", lambda: Path("/fake/studio/run.py"))
    monkeypatch.setattr(
        studio_mod, "_find_frontend_dist", lambda: Path("/fake/studio/frontend/dist")
    )
    monkeypatch.setattr(sys, "platform", "linux")

    def fake_execvp(file, argv):
        events.append(("exec", list(argv)))
        raise _ExecCaptured(argv)

    monkeypatch.setattr(studio_mod.os, "execvp", fake_execvp)


def _install_run_reexec(monkeypatch, events):
    studio_mod = _studio()
    monkeypatch.setattr(sys, "prefix", "/nonexistent/outer/venv")
    fake_venv = Path("/fake/studio/venv/unsloth_studio")
    monkeypatch.setattr(studio_mod, "_studio_venv_python", lambda: fake_venv / "bin" / "python")
    monkeypatch.setattr(
        studio_mod, "_find_frontend_dist", lambda: Path("/fake/studio/frontend/dist")
    )
    fake_bin = fake_venv / "bin" / "unsloth"
    real_is_file = Path.is_file
    # Pin platform.system() too: the launcher name comes from it, so a Windows runner would miss the
    # POSIX fixture.
    monkeypatch.setattr(studio_mod.platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        Path,
        "is_file",
        lambda self: True if str(self) == str(fake_bin) else real_is_file(self),
    )
    from unsloth_cli import _tool_policy as _tp_mod

    monkeypatch.setattr(
        _tp_mod,
        "resolve_tool_policy",
        lambda host, flag, yes, silent: False if flag is None else bool(flag),
    )
    monkeypatch.setattr(sys, "platform", "linux")

    def fake_execvp(file, argv):
        events.append(("exec", list(argv)))
        raise _ExecCaptured(argv)

    monkeypatch.setattr(studio_mod.os, "execvp", fake_execvp)


def _invoke_studio_default(monkeypatch, events, args):
    import typer as _typer

    studio_mod = _studio()
    _install_studio_default_reexec(monkeypatch, events)
    app = _typer.Typer()
    app.command()(studio_mod.studio_default)
    return CliRunner().invoke(app, args, catch_exceptions = True)


def _invoke_run(monkeypatch, events, args):
    import typer as _typer

    studio_mod = _studio()
    _install_run_reexec(monkeypatch, events)
    app = _typer.Typer()
    app.command(
        context_settings = {"allow_extra_args": True, "ignore_unknown_options": True},
    )(studio_mod.run)
    return CliRunner().invoke(app, args, catch_exceptions = True)


def test_run_reexecs_through_the_windows_console_script(monkeypatch):
    import typer as _typer

    studio_mod = _studio()
    events = []
    _install_run_reexec(monkeypatch, events)
    monkeypatch.setattr(studio_mod.platform, "system", lambda: "Windows")
    windows_bin = Path("/fake/studio/venv/unsloth_studio/bin/unsloth.exe")
    real_is_file = Path.is_file
    monkeypatch.setattr(
        Path,
        "is_file",
        lambda self: True if str(self) == str(windows_bin) else real_is_file(self),
    )
    monkeypatch.setattr(studio_mod, "_managed_cli_package_present", lambda _python: False)

    app = _typer.Typer()
    app.command(
        context_settings = {"allow_extra_args": True, "ignore_unknown_options": True},
    )(studio_mod.run)
    result = CliRunner().invoke(app, _BASE + ["--api-only"], catch_exceptions = True)

    assert result.exit_code == 0, result.output
    assert [kind for kind, _ in events] == ["exec"], events


@pytest.mark.parametrize("command", ["default", "run"])
def test_an_empty_bind_is_rejected_before_password_or_launch(monkeypatch, command):
    studio_mod = _studio()
    events = []
    monkeypatch.setattr(
        studio_mod,
        "_enforce_password_change_before_exposure",
        lambda **_kwargs: events.append(("password", None)),
    )
    if command == "default":
        result = _invoke_studio_default(monkeypatch, events, ["--host", "", "--cloudflare"])
    else:
        result = _invoke_run(
            monkeypatch,
            events,
            _BASE + ["--host", "", "--cloudflare"],
        )

    assert result.exit_code == 2, result.output
    assert "--host cannot be empty" in result.output
    assert events == []


@pytest.mark.parametrize("command", ["default", "run"])
def test_a_mixed_family_wildcard_is_rejected_before_password_or_launch(monkeypatch, command):
    import socket

    from unsloth_cli import _tool_policy

    studio_mod = _studio()
    events = []
    monkeypatch.setattr(
        studio_mod,
        "_enforce_password_change_before_exposure",
        lambda **_kwargs: events.append(("password", None)),
    )
    monkeypatch.setattr(
        _tool_policy.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("0.0.0.0", 0)),
            (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("fd00::24", 0, 0, 0)),
        ],
    )
    if command == "default":
        result = _invoke_studio_default(
            monkeypatch,
            events,
            ["--host", "mixed-wildcard.test", "--cloudflare"],
        )
    else:
        result = _invoke_run(
            monkeypatch,
            events,
            _BASE + ["--host", "mixed-wildcard.test", "--cloudflare"],
        )

    assert result.exit_code == 2, result.output
    assert "mixes wildcard and specific address families" in result.output
    assert events == []


@pytest.mark.parametrize("command", ["default", "run"])
def test_a_mapped_wildcard_is_canonicalized_before_reexec(monkeypatch, command):
    studio_mod = _studio()
    events = []
    monkeypatch.setattr(studio_mod, "_enforce_password_change_before_exposure", lambda **_kw: None)
    if command == "default":
        result = _invoke_studio_default(
            monkeypatch,
            events,
            ["--host", "::ffff:0.0.0.0", "--api-only"],
        )
    else:
        result = _invoke_run(
            monkeypatch,
            events,
            _BASE + ["--host", "::ffff:0.0.0.0", "--api-only"],
        )

    assert result.exit_code == 0, result.output
    argv = next(payload for kind, payload in events if kind == "exec")
    assert argv[argv.index("--host") + 1] == "0.0.0.0"


@pytest.mark.parametrize("command", ["default", "run"])
def test_a_mapped_specific_bind_is_canonicalized_before_reexec(monkeypatch, command):
    studio_mod = _studio()
    events = []
    monkeypatch.setattr(studio_mod, "_enforce_password_change_before_exposure", lambda **_kw: None)
    if command == "default":
        result = _invoke_studio_default(
            monkeypatch,
            events,
            ["--host", "::ffff:127.0.0.1", "--api-only"],
        )
    else:
        result = _invoke_run(
            monkeypatch,
            events,
            _BASE + ["--host", "::ffff:127.0.0.1", "--api-only"],
        )

    assert result.exit_code == 0, result.output
    argv = next(payload for kind, payload in events if kind == "exec")
    assert argv[argv.index("--host") + 1] == "127.0.0.1"


@pytest.mark.parametrize("command", ["default", "run"])
def test_a_resolved_mapped_bind_is_canonicalized_before_reexec(monkeypatch, command):
    import socket

    from unsloth_cli import _tool_policy

    studio_mod = _studio()
    events = []
    monkeypatch.setattr(studio_mod, "_enforce_password_change_before_exposure", lambda **_kw: None)
    monkeypatch.setattr(
        _tool_policy.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("::ffff:127.0.0.1", 0, 0, 0))
        ],
    )
    if command == "default":
        result = _invoke_studio_default(
            monkeypatch,
            events,
            ["--host", "mapped.test", "--api-only"],
        )
    else:
        result = _invoke_run(
            monkeypatch,
            events,
            _BASE + ["--host", "mapped.test", "--api-only"],
        )

    assert result.exit_code == 0, result.output
    argv = next(payload for kind, payload in events if kind == "exec")
    assert argv[argv.index("--host") + 1] == "127.0.0.1"


@pytest.mark.parametrize("command", ["default", "run"])
def test_ambiguous_resolved_mapped_binds_are_rejected_before_launch(monkeypatch, command):
    import socket

    from unsloth_cli import _tool_policy

    studio_mod = _studio()
    events = []
    monkeypatch.setattr(studio_mod, "_enforce_password_change_before_exposure", lambda **_kw: None)
    monkeypatch.setattr(
        _tool_policy.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("::ffff:127.0.0.1", 0, 0, 0)),
            (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("::ffff:192.168.1.24", 0, 0, 0)),
        ],
    )
    if command == "default":
        result = _invoke_studio_default(
            monkeypatch,
            events,
            ["--host", "ambiguous-mapped.test", "--api-only"],
        )
    else:
        result = _invoke_run(
            monkeypatch,
            events,
            _BASE + ["--host", "ambiguous-mapped.test", "--api-only"],
        )

    assert result.exit_code == 2, result.output
    assert "resolves to ambiguous IPv4-mapped addresses" in result.output
    assert events == []


@pytest.mark.parametrize("command", ["default", "run"])
def test_a_dual_stack_wildcard_hostname_is_preserved_for_reexec(monkeypatch, command):
    import socket

    from unsloth_cli import _tool_policy

    studio_mod = _studio()
    events = []
    monkeypatch.setattr(studio_mod, "_enforce_password_change_before_exposure", lambda **_kw: None)
    monkeypatch.setattr(
        _tool_policy.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("0.0.0.0", 0)),
            (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("::", 0, 0, 0)),
        ],
    )
    if command == "default":
        result = _invoke_studio_default(
            monkeypatch,
            events,
            ["--host", "dual-wildcard.test", "--api-only"],
        )
    else:
        result = _invoke_run(
            monkeypatch,
            events,
            _BASE + ["--host", "dual-wildcard.test", "--api-only"],
        )

    assert result.exit_code == 0, result.output
    argv = next(payload for kind, payload in events if kind == "exec")
    assert argv[argv.index("--host") + 1] == "dual-wildcard.test"


@pytest.mark.parametrize("command", ["default", "run"])
def test_an_ephemeral_multi_address_bind_is_rejected_before_password_or_launch(
    monkeypatch, command
):
    import socket

    from unsloth_cli import _tool_policy

    studio_mod = _studio()
    events = []
    monkeypatch.setattr(
        studio_mod,
        "_enforce_password_change_before_exposure",
        lambda **_kwargs: events.append(("password", None)),
    )
    monkeypatch.setattr(
        _tool_policy.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("fe80::1", 0, 0, 2)),
            (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("fe80::1", 0, 0, 3)),
        ],
    )
    if command == "default":
        result = _invoke_studio_default(
            monkeypatch,
            events,
            ["--host", "scoped.test", "--port", "0", "--cloudflare"],
        )
    else:
        result = _invoke_run(
            monkeypatch,
            events,
            _BASE + ["--host", "scoped.test", "--port", "0", "--cloudflare"],
        )

    assert result.exit_code == 2, result.output
    assert "--port 0 cannot be used" in result.output
    assert events == []


def test_studio_default_secure_prompts_and_updates_before_reexec(monkeypatch, tmp_path):
    studio_mod = _studio()
    events = _install_prompt_env(monkeypatch, tmp_path, interactive = True)
    before = _seed_auth(studio_mod)

    _invoke_studio_default(monkeypatch, events, ["--secure"])

    kinds = [kind for kind, _ in events]
    assert kinds == ["prompt", "exec"], events

    after = _auth_state(studio_mod)
    assert after["must_change_password"] == 0
    assert after["password_hash"] != before["password_hash"]
    assert after["jwt_secret"] != before["jwt_secret"]
    assert after["n_refresh"] == 0
    assert not (tmp_path / "auth" / studio_mod.BOOTSTRAP_PASSWORD_FILE).exists()


def test_studio_default_prompt_rejects_current_password(monkeypatch, tmp_path):
    studio_mod = _studio()
    events = _install_prompt_env(monkeypatch, tmp_path, interactive = True)
    _seed_auth(studio_mod)
    bootstrap_pw = (tmp_path / "auth" / studio_mod.BOOTSTRAP_PASSWORD_FILE).read_text().strip()

    _invoke_studio_default(monkeypatch, events, ["--secure"])

    verify_current = events[0][1]
    assert verify_current(bootstrap_pw) is True
    assert verify_current("something-else-entirely") is False


def test_studio_default_non_tty_warns_and_proceeds(monkeypatch, tmp_path):
    studio_mod = _studio()
    events = _install_prompt_env(monkeypatch, tmp_path, interactive = False)
    _seed_auth(studio_mod)

    result = _invoke_studio_default(monkeypatch, events, ["--secure"])

    kinds = [kind for kind, _ in events]
    assert kinds == ["exec"], events
    combined = (result.output or "") + (getattr(result, "stderr", "") or "")
    assert "bootstrap password" in combined
    assert _auth_state(studio_mod)["must_change_password"] == 1


def test_studio_default_non_tty_deletes_bootstrap_password_file(monkeypatch, tmp_path):
    studio_mod = _studio()
    events = _install_prompt_env(monkeypatch, tmp_path, interactive = False)
    _seed_auth(studio_mod)
    bootstrap_file = tmp_path / "auth" / studio_mod.BOOTSTRAP_PASSWORD_FILE
    assert bootstrap_file.exists()

    _invoke_studio_default(monkeypatch, events, ["--secure"])

    assert not bootstrap_file.exists()
    kinds = [kind for kind, _ in events]
    assert kinds == ["exec"], events
    assert _auth_state(studio_mod)["must_change_password"] == 1


def test_studio_default_reexec_outer_runpy_keeps_bootstrap_for_local_recovery(
    monkeypatch, tmp_path
):
    import typer as _typer

    studio_mod = _studio()
    events = _install_prompt_env(monkeypatch, tmp_path, interactive = False)
    _seed_auth(studio_mod)
    bootstrap_file = tmp_path / "auth" / studio_mod.BOOTSTRAP_PASSWORD_FILE
    assert bootstrap_file.exists()

    _install_studio_default_reexec(monkeypatch, events)
    outer_run_py = studio_mod._PACKAGE_ROOT / "studio" / "backend" / "run.py"
    monkeypatch.setattr(studio_mod, "_find_run_py", lambda: outer_run_py)

    app = _typer.Typer()
    app.command()(studio_mod.studio_default)
    result = CliRunner().invoke(app, ["--secure"], catch_exceptions = True)

    assert bootstrap_file.exists(), result.output
    assert _auth_state(studio_mod)["must_change_password"] == 1
    assert "exec" in [k for k, _ in events], events


def test_studio_default_non_tty_persists_seeded_admin_on_fresh_home(monkeypatch, tmp_path):
    studio_mod = _studio()
    events = _install_prompt_env(monkeypatch, tmp_path, interactive = False)
    # Deliberately NO _seed_auth(): exercise the gate seeding a fresh DB itself.

    _invoke_studio_default(monkeypatch, events, ["--secure"])

    state = _auth_state(studio_mod)
    assert state["must_change_password"] == 1
    assert not (tmp_path / "auth" / studio_mod.BOOTSTRAP_PASSWORD_FILE).exists()
    kinds = [kind for kind, _ in events]
    assert kinds == ["exec"], events


def test_studio_default_non_tty_fails_closed_when_bootstrap_removal_fails(monkeypatch, tmp_path):
    import pathlib

    studio_mod = _studio()
    events = _install_prompt_env(monkeypatch, tmp_path, interactive = False)
    _seed_auth(studio_mod)
    bootstrap_file = tmp_path / "auth" / studio_mod.BOOTSTRAP_PASSWORD_FILE
    assert bootstrap_file.exists()

    _real_unlink = pathlib.Path.unlink

    def _boom_unlink(self, *a, **k):
        if self.name == studio_mod.BOOTSTRAP_PASSWORD_FILE:
            raise OSError("locked")
        return _real_unlink(self, *a, **k)

    monkeypatch.setattr(pathlib.Path, "unlink", _boom_unlink)

    result = _invoke_studio_default(monkeypatch, events, ["--secure"])

    kinds = [kind for kind, _ in events]
    assert "exec" not in kinds, events
    assert result.exit_code == 1, result.output
    combined = (result.output or "") + (getattr(result, "stderr", "") or "")
    assert "refusing to publish" in combined.lower()
    assert bootstrap_file.exists()
    assert _auth_state(studio_mod)["must_change_password"] == 1


class _FailingSelectConn:
    """Wrap a real auth connection but raise on the gate's must_change SELECT,
    so seeding + commit still happen and only the read-back fails (a locked-DB
    window that lands after _ensure_cli_default_admin already wrote the file)."""

    def __init__(self, inner):
        self._inner = inner

    def execute(self, sql, *args, **kwargs):
        if sql.lstrip().startswith("SELECT password_salt"):
            raise sqlite3.OperationalError("database is locked")
        return self._inner.execute(sql, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._inner, name)


class _FailingCommitConn:
    """Wraps a real auth connection; commit() raises, so the seeded admin INSERT rolls back on close."""

    def __init__(self, inner):
        self._inner = inner

    def commit(self):
        raise sqlite3.OperationalError("database is locked")

    def __getattr__(self, name):
        return getattr(self._inner, name)


def test_studio_default_connect_failure_fails_closed(monkeypatch, tmp_path):
    studio_mod = _studio()
    events = _install_prompt_env(monkeypatch, tmp_path, interactive = True)
    _seed_auth(studio_mod)
    bootstrap_file = tmp_path / "auth" / studio_mod.BOOTSTRAP_PASSWORD_FILE
    assert bootstrap_file.exists()

    monkeypatch.setattr(
        studio_mod,
        "_connect_auth_db",
        lambda: (_ for _ in ()).throw(sqlite3.OperationalError("database is locked")),
    )

    result = _invoke_studio_default(monkeypatch, events, ["--secure"])

    kinds = [kind for kind, _ in events]
    assert "exec" not in kinds, events
    assert result.exit_code == 1, result.output
    combined = (result.output or "") + (getattr(result, "stderr", "") or "")
    assert "refusing to expose" in combined.lower()
    assert bootstrap_file.exists()


def test_studio_default_seed_commit_failure_fails_closed(monkeypatch, tmp_path):
    studio_mod = _studio()
    events = _install_prompt_env(monkeypatch, tmp_path, interactive = False)
    # Deliberately NO _seed_auth(): the gate seeds the fresh DB itself, then commit fails.
    real_connect = studio_mod._connect_auth_db
    monkeypatch.setattr(studio_mod, "_connect_auth_db", lambda: _FailingCommitConn(real_connect()))

    result = _invoke_studio_default(monkeypatch, events, ["--secure"])

    kinds = [kind for kind, _ in events]
    assert "exec" not in kinds, events
    assert result.exit_code == 1, result.output
    combined = (result.output or "") + (getattr(result, "stderr", "") or "")
    assert "refusing to expose" in combined.lower()
    assert not (tmp_path / "auth" / studio_mod.BOOTSTRAP_PASSWORD_FILE).exists()
    verify = sqlite3.connect(_auth_db(tmp_path))
    try:
        assert verify.execute("SELECT COUNT(*) FROM auth_user").fetchone()[0] == 0
    finally:
        verify.close()


def test_studio_default_missing_venv_exits_before_stripping_bootstrap(monkeypatch, tmp_path):
    import typer as _typer

    studio_mod = _studio()
    events = _install_prompt_env(monkeypatch, tmp_path, interactive = False)
    _seed_auth(studio_mod)
    bootstrap_file = tmp_path / "auth" / studio_mod.BOOTSTRAP_PASSWORD_FILE
    assert bootstrap_file.exists()

    monkeypatch.setattr(sys, "prefix", "/nonexistent/outer/venv")
    monkeypatch.setattr(studio_mod, "_studio_venv_python", lambda: None)
    monkeypatch.setattr(studio_mod, "_find_run_py", lambda: None)

    app = _typer.Typer()
    app.command()(studio_mod.studio_default)
    result = CliRunner().invoke(app, ["--secure"], catch_exceptions = True)

    assert result.exit_code == 1, result.output
    assert bootstrap_file.exists()
    assert _auth_state(studio_mod)["must_change_password"] == 1
    assert events == [], events
    combined = (result.output or "") + (getattr(result, "stderr", "") or "")
    assert "not set up" in combined.lower()


def test_studio_default_missing_frontend_exits_before_stripping_bootstrap(monkeypatch, tmp_path):
    import typer as _typer

    studio_mod = _studio()
    events = _install_prompt_env(monkeypatch, tmp_path, interactive = False)
    _seed_auth(studio_mod)
    bootstrap_file = tmp_path / "auth" / studio_mod.BOOTSTRAP_PASSWORD_FILE
    assert bootstrap_file.exists()

    monkeypatch.setattr(sys, "prefix", "/nonexistent/outer/venv")
    fake_venv = Path("/fake/studio/venv/unsloth_studio")
    monkeypatch.setattr(studio_mod, "_studio_venv_python", lambda: fake_venv / "bin" / "python")
    monkeypatch.setattr(studio_mod, "_find_run_py", lambda: Path("/fake/studio/run.py"))
    monkeypatch.setattr(studio_mod, "_find_frontend_dist", lambda: None)

    app = _typer.Typer()
    app.command()(studio_mod.studio_default)
    result = CliRunner().invoke(app, ["--secure"], catch_exceptions = True)

    assert result.exit_code == 1, result.output
    assert bootstrap_file.exists()
    assert _auth_state(studio_mod)["must_change_password"] == 1
    assert events == [], events
    combined = (result.output or "") + (getattr(result, "stderr", "") or "")
    assert "frontend is not built" in combined.lower()


def test_studio_default_bad_frontend_path_exits_before_stripping_bootstrap(monkeypatch, tmp_path):
    import typer as _typer

    studio_mod = _studio()
    events = _install_prompt_env(monkeypatch, tmp_path, interactive = False)
    _seed_auth(studio_mod)
    bootstrap_file = tmp_path / "auth" / studio_mod.BOOTSTRAP_PASSWORD_FILE
    assert bootstrap_file.exists()

    monkeypatch.setattr(sys, "prefix", "/nonexistent/outer/venv")
    fake_venv = Path("/fake/studio/venv/unsloth_studio")
    monkeypatch.setattr(studio_mod, "_studio_venv_python", lambda: fake_venv / "bin" / "python")
    monkeypatch.setattr(studio_mod, "_find_run_py", lambda: Path("/fake/studio/run.py"))
    monkeypatch.setattr(
        studio_mod, "_find_frontend_dist", lambda: Path("/fake/studio/frontend/dist")
    )
    empty_dir = tmp_path / "empty_frontend"
    empty_dir.mkdir()

    app = _typer.Typer()
    app.command()(studio_mod.studio_default)
    result = CliRunner().invoke(
        app, ["--secure", "--frontend", str(empty_dir)], catch_exceptions = True
    )

    assert result.exit_code == 1, result.output
    assert bootstrap_file.exists()
    assert _auth_state(studio_mod)["must_change_password"] == 1
    assert events == [], events
    combined = (result.output or "") + (getattr(result, "stderr", "") or "")
    assert "index.html" in combined.lower()


def test_studio_default_missing_frontend_loopback_cloudflare_still_launches(monkeypatch, tmp_path):
    import typer as _typer

    studio_mod = _studio()
    events = _install_prompt_env(monkeypatch, tmp_path, interactive = True)
    _seed_auth(studio_mod)

    monkeypatch.setattr(sys, "prefix", "/nonexistent/outer/venv")
    monkeypatch.setattr(studio_mod, "_ensure_studio_env_exported", lambda: None)
    fake_venv = Path("/fake/studio/venv/unsloth_studio")
    monkeypatch.setattr(studio_mod, "_studio_venv_python", lambda: fake_venv / "bin" / "python")
    monkeypatch.setattr(studio_mod, "_find_run_py", lambda: Path("/fake/studio/run.py"))
    monkeypatch.setattr(studio_mod, "_find_frontend_dist", lambda: None)
    monkeypatch.setattr(sys, "platform", "linux")

    def fake_execvp(file, argv):
        events.append(("exec", list(argv)))
        raise _ExecCaptured(argv)

    monkeypatch.setattr(studio_mod.os, "execvp", fake_execvp)

    app = _typer.Typer()
    app.command()(studio_mod.studio_default)
    result = CliRunner().invoke(app, ["--cloudflare"], catch_exceptions = True)

    kinds = [kind for kind, _ in events]
    assert kinds == ["exec"], (events, result.output)
    combined = (result.output or "") + (getattr(result, "stderr", "") or "")
    assert "frontend not built" not in combined.lower()


def test_studio_default_in_venv_broken_backend_exits_before_stripping_bootstrap(
    monkeypatch, tmp_path
):
    import typer as _typer

    studio_mod = _studio()
    events = _install_prompt_env(monkeypatch, tmp_path, interactive = False)
    _seed_auth(studio_mod)
    bootstrap_file = tmp_path / "auth" / studio_mod.BOOTSTRAP_PASSWORD_FILE
    assert bootstrap_file.exists()

    monkeypatch.setattr(sys, "prefix", str(tmp_path / "unsloth_studio"))
    # Stub the frontend gate, which runs first, so this reaches the backend check.
    monkeypatch.setattr(
        studio_mod, "_find_frontend_dist", lambda: Path("/fake/studio/frontend/dist")
    )

    def _boom():
        raise ImportError("cannot import backend run.py")

    monkeypatch.setattr(studio_mod, "_load_run_module", _boom)

    app = _typer.Typer()
    app.command()(studio_mod.studio_default)
    result = CliRunner().invoke(app, ["--secure"], catch_exceptions = True)

    assert result.exit_code == 1, result.output
    assert bootstrap_file.exists()
    assert _auth_state(studio_mod)["must_change_password"] == 1
    assert events == [], events
    combined = (result.output or "") + (getattr(result, "stderr", "") or "")
    assert "backend could not be loaded" in combined.lower()


def test_studio_default_in_venv_missing_frontend_exits_before_stripping_bootstrap(
    monkeypatch, tmp_path
):
    import typer as _typer

    studio_mod = _studio()
    events = _install_prompt_env(monkeypatch, tmp_path, interactive = False)
    _seed_auth(studio_mod)
    bootstrap_file = tmp_path / "auth" / studio_mod.BOOTSTRAP_PASSWORD_FILE
    assert bootstrap_file.exists()

    monkeypatch.setattr(sys, "prefix", str(tmp_path / "unsloth_studio"))
    monkeypatch.setattr(studio_mod, "_find_frontend_dist", lambda: None)
    monkeypatch.setattr(studio_mod, "_load_run_module", lambda: None)

    app = _typer.Typer()
    app.command()(studio_mod.studio_default)
    result = CliRunner().invoke(app, ["--secure"], catch_exceptions = True)

    assert result.exit_code == 1, result.output
    assert bootstrap_file.exists()
    assert _auth_state(studio_mod)["must_change_password"] == 1
    assert events == [], events
    combined = (result.output or "") + (getattr(result, "stderr", "") or "")
    assert "frontend is not built" in combined.lower()


def test_studio_default_secure_tunnel_unavailable_preserves_bootstrap(monkeypatch, tmp_path):
    import typer as _typer

    studio_mod = _studio()
    events = _install_prompt_env(monkeypatch, tmp_path, interactive = False)
    _seed_auth(studio_mod)
    bootstrap_file = tmp_path / "auth" / studio_mod.BOOTSTRAP_PASSWORD_FILE
    assert bootstrap_file.exists()

    _install_studio_default_reexec(monkeypatch, events)
    monkeypatch.setattr(studio_mod, "_tunnel_binary_confirmed_unavailable", lambda: True)

    app = _typer.Typer()
    app.command()(studio_mod.studio_default)
    result = CliRunner().invoke(app, ["--secure"], catch_exceptions = True)

    assert result.exit_code == 1, result.output
    assert bootstrap_file.exists()
    assert _auth_state(studio_mod)["must_change_password"] == 1
    assert "exec" not in [k for k, _ in events], events
    combined = (result.output or "") + (getattr(result, "stderr", "") or "")
    assert "cloudflared" in combined.lower()


def test_studio_default_wildcard_cloudflare_strips_even_if_tunnel_unavailable(
    monkeypatch, tmp_path
):
    import typer as _typer

    studio_mod = _studio()
    events = _install_prompt_env(monkeypatch, tmp_path, interactive = False)
    _seed_auth(studio_mod)
    bootstrap_file = tmp_path / "auth" / studio_mod.BOOTSTRAP_PASSWORD_FILE
    assert bootstrap_file.exists()

    _install_studio_default_reexec(monkeypatch, events)
    monkeypatch.setattr(studio_mod, "_tunnel_binary_confirmed_unavailable", lambda: True)

    app = _typer.Typer()
    app.command()(studio_mod.studio_default)
    result = CliRunner().invoke(app, ["-H", "0.0.0.0", "--cloudflare"], catch_exceptions = True)

    assert not bootstrap_file.exists(), result.output
    assert "exec" in [k for k, _ in events], events


def test_tunnel_probe_adds_backend_to_syspath(monkeypatch, tmp_path):
    studio_mod = _studio()
    # Model a cloudflare_tunnel whose ensure_cloudflared resolves only with backend on sys.path.
    backend = tmp_path / "backend"
    backend.mkdir()
    (backend / "cloudflare_tunnel.py").write_text(
        "import sys\n"
        f"_BACKEND = {str(backend)!r}\n"
        "def ensure_cloudflared():\n"
        "    # Resolvable (cached) ONLY when the backend dir is importable.\n"
        "    return '/fake/cloudflared' if _BACKEND in sys.path else None\n"
    )
    monkeypatch.setattr(studio_mod, "_find_run_py", lambda: backend / "run.py")
    assert str(backend) not in sys.path

    result = studio_mod._tunnel_binary_confirmed_unavailable()

    assert result is False
    assert str(backend) not in sys.path


def test_studio_default_query_failure_strips_bootstrap_file(monkeypatch, tmp_path):
    studio_mod = _studio()
    events = _install_prompt_env(monkeypatch, tmp_path, interactive = True)
    _seed_auth(studio_mod)
    bootstrap_file = tmp_path / "auth" / studio_mod.BOOTSTRAP_PASSWORD_FILE
    assert bootstrap_file.exists()

    real_connect = studio_mod._connect_auth_db
    monkeypatch.setattr(studio_mod, "_connect_auth_db", lambda: _FailingSelectConn(real_connect()))

    result = _invoke_studio_default(monkeypatch, events, ["--secure"])

    assert not bootstrap_file.exists()
    kinds = [kind for kind, _ in events]
    assert kinds == ["exec"], events
    assert _auth_state(studio_mod)["must_change_password"] == 1
    combined = (result.output or "") + (getattr(result, "stderr", "") or "")
    assert "removing the seeded bootstrap password" in combined.lower()


def test_studio_default_loopback_cloudflare_never_prompts(monkeypatch, tmp_path):
    studio_mod = _studio()
    events = _install_prompt_env(monkeypatch, tmp_path, interactive = True)
    _seed_auth(studio_mod)

    result = _invoke_studio_default(monkeypatch, events, ["--cloudflare"])

    kinds = [kind for kind, _ in events]
    assert "prompt" not in kinds, events
    combined = (result.output or "") + (getattr(result, "stderr", "") or "")
    assert "bootstrap password" not in combined


def test_studio_default_changed_password_never_prompts(monkeypatch, tmp_path):
    studio_mod = _studio()
    events = _install_prompt_env(monkeypatch, tmp_path, interactive = True)
    _seed_auth(studio_mod, must_change = False)

    _invoke_studio_default(monkeypatch, events, ["--secure"])

    kinds = [kind for kind, _ in events]
    assert kinds == ["exec"], events


def test_studio_default_refusal_aborts_launch(monkeypatch, tmp_path):
    studio_mod = _studio()
    events = _install_prompt_env(
        monkeypatch, tmp_path, interactive = True, scripted = KeyboardInterrupt()
    )
    _seed_auth(studio_mod)

    result = _invoke_studio_default(monkeypatch, events, ["--secure"])

    assert result.exit_code == 1, result.output
    kinds = [kind for kind, _ in events]
    assert "exec" not in kinds, events
    assert _auth_state(studio_mod)["must_change_password"] == 1


def test_studio_default_wildcard_cloudflare_prompts(monkeypatch, tmp_path):
    studio_mod = _studio()
    events = _install_prompt_env(monkeypatch, tmp_path, interactive = True)
    _seed_auth(studio_mod)

    _invoke_studio_default(monkeypatch, events, ["-H", "0.0.0.0", "--cloudflare"])

    kinds = [kind for kind, _ in events]
    assert kinds == ["prompt", "exec"], events
    assert _auth_state(studio_mod)["must_change_password"] == 0


def test_run_secure_prompts_and_updates_before_reexec(monkeypatch, tmp_path):
    studio_mod = _studio()
    events = _install_prompt_env(monkeypatch, tmp_path, interactive = True)
    before = _seed_auth(studio_mod)

    _invoke_run(monkeypatch, events, _BASE + ["--secure"])

    kinds = [kind for kind, _ in events]
    assert kinds == ["prompt", "exec"], events

    after = _auth_state(studio_mod)
    assert after["must_change_password"] == 0
    assert after["password_hash"] != before["password_hash"]
    assert after["n_refresh"] == 0


def test_run_non_tty_warns_and_proceeds(monkeypatch, tmp_path):
    studio_mod = _studio()
    events = _install_prompt_env(monkeypatch, tmp_path, interactive = False)
    _seed_auth(studio_mod)

    result = _invoke_run(monkeypatch, events, _BASE + ["--secure"])

    kinds = [kind for kind, _ in events]
    assert kinds == ["exec"], events
    combined = (result.output or "") + (getattr(result, "stderr", "") or "")
    assert "bootstrap password" in combined


def test_run_non_tty_deletes_bootstrap_password_file(monkeypatch, tmp_path):
    studio_mod = _studio()
    events = _install_prompt_env(monkeypatch, tmp_path, interactive = False)
    _seed_auth(studio_mod)
    bootstrap_file = tmp_path / "auth" / studio_mod.BOOTSTRAP_PASSWORD_FILE
    assert bootstrap_file.exists()

    _invoke_run(monkeypatch, events, _BASE + ["--secure"])

    assert not bootstrap_file.exists()
    kinds = [kind for kind, _ in events]
    assert kinds == ["exec"], events
    assert _auth_state(studio_mod)["must_change_password"] == 1


def test_run_missing_frontend_exits_before_stripping_bootstrap(monkeypatch, tmp_path):
    import typer as _typer

    studio_mod = _studio()
    events = _install_prompt_env(monkeypatch, tmp_path, interactive = False)
    _seed_auth(studio_mod)
    bootstrap_file = tmp_path / "auth" / studio_mod.BOOTSTRAP_PASSWORD_FILE
    assert bootstrap_file.exists()

    _install_run_reexec(monkeypatch, events)
    monkeypatch.setattr(studio_mod, "_find_frontend_dist", lambda: None)

    app = _typer.Typer()
    app.command(
        context_settings = {"allow_extra_args": True, "ignore_unknown_options": True},
    )(studio_mod.run)
    result = CliRunner().invoke(app, _BASE + ["--secure"], catch_exceptions = True)

    assert result.exit_code == 1, result.output
    assert bootstrap_file.exists()
    assert _auth_state(studio_mod)["must_change_password"] == 1
    assert events == [], events
    combined = (result.output or "") + (getattr(result, "stderr", "") or "")
    assert "frontend is not built" in combined.lower()


def test_run_in_venv_missing_frontend_exits_before_stripping_bootstrap(monkeypatch, tmp_path):
    import typer as _typer

    studio_mod = _studio()
    events = _install_prompt_env(monkeypatch, tmp_path, interactive = False)
    _seed_auth(studio_mod)
    bootstrap_file = tmp_path / "auth" / studio_mod.BOOTSTRAP_PASSWORD_FILE
    assert bootstrap_file.exists()

    monkeypatch.setattr(sys, "prefix", str(tmp_path / "unsloth_studio"))
    monkeypatch.setattr(studio_mod, "_find_frontend_dist", lambda: None)
    monkeypatch.setattr(studio_mod, "_load_run_module", lambda: None)

    app = _typer.Typer()
    app.command(
        context_settings = {"allow_extra_args": True, "ignore_unknown_options": True},
    )(studio_mod.run)
    result = CliRunner().invoke(app, _BASE + ["--secure"], catch_exceptions = True)

    assert result.exit_code == 1, result.output
    assert bootstrap_file.exists()
    assert _auth_state(studio_mod)["must_change_password"] == 1
    assert events == [], events
    combined = (result.output or "") + (getattr(result, "stderr", "") or "")
    assert "frontend is not built" in combined.lower()


def test_run_reexec_forwards_resolved_frontend_on_public_launch(monkeypatch, tmp_path):
    studio_mod = _studio()
    events = _install_prompt_env(monkeypatch, tmp_path, interactive = True)
    _seed_auth(studio_mod, must_change = False)

    # _install_run_reexec resolves _find_frontend_dist -> /fake/studio/frontend/dist.
    _invoke_run(monkeypatch, events, _BASE + ["--secure"])

    exec_argv = [argv for kind, argv in events if kind == "exec"][0]
    assert "--frontend" in exec_argv, exec_argv
    # str(Path(...)), not the literal: Windows renders it with backslashes.
    expected_dist = str(Path("/fake/studio/frontend/dist"))
    assert exec_argv[exec_argv.index("--frontend") + 1] == expected_dist, exec_argv


def test_run_non_tty_persists_seeded_admin_on_fresh_home(monkeypatch, tmp_path):
    studio_mod = _studio()
    events = _install_prompt_env(monkeypatch, tmp_path, interactive = False)

    _invoke_run(monkeypatch, events, _BASE + ["--secure"])

    state = _auth_state(studio_mod)
    assert state["must_change_password"] == 1
    assert not (tmp_path / "auth" / studio_mod.BOOTSTRAP_PASSWORD_FILE).exists()
    kinds = [kind for kind, _ in events]
    assert kinds == ["exec"], events


def test_run_non_tty_api_only_fails_closed(monkeypatch, tmp_path):
    studio_mod = _studio()
    events = _install_prompt_env(monkeypatch, tmp_path, interactive = False)
    _seed_auth(studio_mod)

    result = _invoke_run(monkeypatch, events, _BASE + ["--secure", "--api-only"])

    kinds = [kind for kind, _ in events]
    assert "exec" not in kinds, events
    assert result.exit_code == 1, result.output
    combined = (result.output or "") + (getattr(result, "stderr", "") or "")
    assert "refusing to publish" in combined.lower()
    assert _auth_state(studio_mod)["must_change_password"] == 1


def test_studio_default_non_tty_disabled_deadline_fails_closed(monkeypatch, tmp_path):
    studio_mod = _studio()
    events = _install_prompt_env(monkeypatch, tmp_path, interactive = False)
    _seed_auth(studio_mod)
    monkeypatch.setenv("UNSLOTH_STUDIO_BOOTSTRAP_TIMEOUT", "0")

    result = _invoke_studio_default(monkeypatch, events, ["--secure"])

    kinds = [kind for kind, _ in events]
    assert "exec" not in kinds, events
    assert result.exit_code == 1, result.output
    combined = (result.output or "") + (getattr(result, "stderr", "") or "")
    assert "refusing to publish" in combined.lower()


@pytest.mark.parametrize(
    "raw,expected",
    [
        (None, True),
        ("", True),
        ("garbage", True),
        ("3600", True),
        ("1", True),
        ("0", False),
        ("-5", False),
    ],
)
def test_bootstrap_deadline_active_mirrors_backend_parsing(monkeypatch, raw, expected):
    studio_mod = _studio()
    if raw is None:
        monkeypatch.delenv("UNSLOTH_STUDIO_BOOTSTRAP_TIMEOUT", raising = False)
    else:
        monkeypatch.setenv("UNSLOTH_STUDIO_BOOTSTRAP_TIMEOUT", raw)
    assert studio_mod._bootstrap_deadline_active() is expected


def _reset_password_cli(studio_mod):
    import typer as _typer

    app = _typer.Typer()
    app.command()(studio_mod.reset_password)
    return CliRunner().invoke(app, [], catch_exceptions = True)


def _password_works(studio_mod, candidate):
    conn = studio_mod._connect_auth_db()
    try:
        row = conn.execute(
            "SELECT password_salt, password_hash FROM auth_user WHERE username = ?",
            (studio_mod.DEFAULT_ADMIN_USERNAME,),
        ).fetchone()
    finally:
        conn.close()
    return studio_mod._pbkdf2_hex(candidate, row[0].encode("utf-8")) == row[1]


def _printed_password(result):
    line = next(l for l in result.output.splitlines() if l.startswith("New password for"))
    return line.split(": ", 1)[1].strip()


def test_reset_password_rotates_in_place_without_deleting_the_db(monkeypatch, tmp_path):
    studio_mod = _studio()
    monkeypatch.setattr(studio_mod, "STUDIO_HOME", tmp_path)
    _seed_auth(studio_mod)
    db_file = tmp_path / "auth" / "auth.db"
    before = _auth_state(studio_mod)

    result = _reset_password_cli(studio_mod)

    assert result.exit_code == 0, result.output
    assert db_file.exists()
    after = _auth_state(studio_mod)
    assert after["password_hash"] != before["password_hash"]
    assert after["jwt_secret"] != before["jwt_secret"]
    assert _password_works(studio_mod, _printed_password(result))


def test_reset_password_waits_out_a_concurrent_writer(monkeypatch, tmp_path):
    import threading
    import time

    studio_mod = _studio()
    monkeypatch.setattr(studio_mod, "STUDIO_HOME", tmp_path)
    _seed_auth(studio_mod)
    released = threading.Event()

    def hold_write_lock():
        conn = sqlite3.connect(_auth_db(tmp_path))
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "INSERT INTO refresh_tokens (token_hash, username, expires_at) "
            "VALUES ('held', 'unsloth', '2099-01-01T00:00:00')"
        )
        time.sleep(0.5)
        conn.rollback()
        conn.close()
        released.set()

    holder = threading.Thread(target = hold_write_lock)
    holder.start()
    time.sleep(0.1)
    result = _reset_password_cli(studio_mod)
    holder.join()

    assert released.is_set()
    assert result.exit_code == 0, result.output
    assert _password_works(studio_mod, _printed_password(result))


def test_reset_password_revokes_sessions_and_api_keys(monkeypatch, tmp_path):
    studio_mod = _studio()
    monkeypatch.setattr(studio_mod, "STUDIO_HOME", tmp_path)
    _seed_auth(studio_mod)
    conn = studio_mod._connect_auth_db()
    conn.execute(
        "INSERT INTO api_keys (username, key_prefix, key_hash, name, created_at) "
        "VALUES (?, 'sk-x', 'hash', 'k', '2026-01-01T00:00:00')",
        (studio_mod.DEFAULT_ADMIN_USERNAME,),
    )
    conn.commit()
    conn.close()

    assert _reset_password_cli(studio_mod).exit_code == 0

    conn = studio_mod._connect_auth_db()
    try:
        assert conn.execute("SELECT COUNT(*) FROM api_keys").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM refresh_tokens").fetchone()[0] == 0
    finally:
        conn.close()


def test_reset_password_leaves_the_account_ready_to_log_in(monkeypatch, tmp_path):
    studio_mod = _studio()
    monkeypatch.setattr(studio_mod, "STUDIO_HOME", tmp_path)
    _seed_auth(studio_mod)

    assert _reset_password_cli(studio_mod).exit_code == 0

    assert _auth_state(studio_mod)["must_change_password"] == 0
    assert not (tmp_path / "auth" / studio_mod.BOOTSTRAP_PASSWORD_FILE).exists()


def test_reset_password_seeds_the_admin_when_no_db_exists(monkeypatch, tmp_path):
    studio_mod = _studio()
    monkeypatch.setattr(studio_mod, "STUDIO_HOME", tmp_path)

    result = _reset_password_cli(studio_mod)

    assert result.exit_code == 0, result.output
    assert _password_works(studio_mod, _printed_password(result))


def test_reset_password_reports_an_unwritable_auth_dir(monkeypatch, tmp_path):
    import pathlib

    studio_mod = _studio()
    monkeypatch.setattr(studio_mod, "STUDIO_HOME", tmp_path)

    def _boom_mkdir(self, *a, **k):
        raise PermissionError("read-only")

    monkeypatch.setattr(pathlib.Path, "mkdir", _boom_mkdir)

    result = _reset_password_cli(studio_mod)

    assert result.exit_code == 1, result.output
    assert not isinstance(result.exception, OSError)
    combined = (result.output or "") + (getattr(result, "stderr", "") or "")
    assert "could not open the auth database" in combined.lower()


def test_reset_password_reports_an_unreadable_db(monkeypatch, tmp_path):
    studio_mod = _studio()
    monkeypatch.setattr(studio_mod, "STUDIO_HOME", tmp_path)
    auth_dir = tmp_path / "auth"
    auth_dir.mkdir()
    (auth_dir / "auth.db").write_text("not a database")

    result = _reset_password_cli(studio_mod)

    assert result.exit_code == 1, result.output
    assert (auth_dir / "auth.db").exists()
    combined = (result.output or "") + (getattr(result, "stderr", "") or "")
    assert "could not open the auth database" in combined.lower()


def test_cli_update_password_truncates_locked_bootstrap_after_change(monkeypatch, tmp_path):
    import pathlib

    studio_mod = _studio()
    monkeypatch.setattr(studio_mod, "STUDIO_HOME", tmp_path)
    _seed_auth(studio_mod)
    bootstrap_file = tmp_path / "auth" / studio_mod.BOOTSTRAP_PASSWORD_FILE
    assert bootstrap_file.read_text().strip()

    _real_unlink = pathlib.Path.unlink

    def _boom_unlink(self, *a, **k):
        if self.name == studio_mod.BOOTSTRAP_PASSWORD_FILE:
            raise OSError("locked")
        return _real_unlink(self, *a, **k)

    monkeypatch.setattr(pathlib.Path, "unlink", _boom_unlink)

    conn = studio_mod._connect_auth_db()
    studio_mod._cli_update_password(conn, studio_mod.DEFAULT_ADMIN_USERNAME, "fresh-new-pw-123")
    conn.close()

    assert _auth_state(studio_mod)["must_change_password"] == 0
    assert bootstrap_file.exists()
    assert bootstrap_file.read_text() == ""


def test_reset_clears_cached_cli_api_keys(monkeypatch, tmp_path):
    studio_mod = _studio()
    monkeypatch.setattr(studio_mod, "STUDIO_HOME", tmp_path)
    _seed_auth(studio_mod)
    for name in ("cli", "my key"):
        studio_mod._write_auth_secret(
            studio_mod._cli_api_key_secret_path(name), "sk-unsloth-" + "0" * 32
        )
    auth_dir = tmp_path / "auth"
    assert len(list(auth_dir.glob(f"{studio_mod.CLI_API_KEY_FILE_PREFIX}*"))) == 2

    conn = studio_mod._connect_auth_db()
    studio_mod._cli_update_password(
        conn, studio_mod.DEFAULT_ADMIN_USERNAME, "fresh-new-pw-123", revoke_api_keys = True
    )
    conn.close()

    assert list(auth_dir.glob(f"{studio_mod.CLI_API_KEY_FILE_PREFIX}*")) == []


def test_ordinary_password_change_keeps_cached_cli_api_keys(monkeypatch, tmp_path):
    studio_mod = _studio()
    monkeypatch.setattr(studio_mod, "STUDIO_HOME", tmp_path)
    _seed_auth(studio_mod)
    path = studio_mod._cli_api_key_secret_path("cli")
    studio_mod._write_auth_secret(path, "sk-unsloth-" + "0" * 32)

    conn = studio_mod._connect_auth_db()
    studio_mod._cli_update_password(conn, studio_mod.DEFAULT_ADMIN_USERNAME, "fresh-new-pw-123")
    conn.close()

    assert path.exists()


def test_connect_auth_db_creates_private_files(monkeypatch, tmp_path):
    import os as _os
    import stat

    if _os.name == "nt":
        pytest.skip("POSIX permission bits")
    studio_mod = _studio()
    monkeypatch.setattr(studio_mod, "STUDIO_HOME", tmp_path)
    conn = studio_mod._connect_auth_db()
    conn.close()
    auth_dir = tmp_path / "auth"
    assert stat.S_IMODE(auth_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE((auth_dir / "auth.db").stat().st_mode) == 0o600


def test_write_auth_secret_terminates_the_file_with_a_newline(monkeypatch, tmp_path):
    studio_mod = _studio()
    path = tmp_path / ".desktop_secret"

    studio_mod._write_auth_secret(path, "desktop-abc123")

    # Bytes: read_text would decode CRLF back to "\n" and hide a CR.
    assert path.read_bytes() == b"desktop-abc123\n"


def test_seeded_bootstrap_file_ends_with_a_newline(monkeypatch, tmp_path):
    studio_mod = _studio()
    monkeypatch.setattr(studio_mod, "STUDIO_HOME", tmp_path)
    _seed_auth(studio_mod)

    raw = (tmp_path / "auth" / studio_mod.BOOTSTRAP_PASSWORD_FILE).read_bytes()

    assert raw.endswith(b"\n") and not raw.endswith(b"\r\n")

    conn = sqlite3.connect(_auth_db(tmp_path))
    try:
        salt, pwd_hash = conn.execute(
            "SELECT password_salt, password_hash FROM auth_user WHERE username = ?",
            (studio_mod.DEFAULT_ADMIN_USERNAME,),
        ).fetchone()
    finally:
        conn.close()
    assert studio_mod._pbkdf2_hex(raw.decode("utf-8").strip(), salt.encode("utf-8")) == pwd_hash


def _exec_argv(events):
    return next(argv for kind, argv in events if kind == "exec")


def test_studio_default_password_sets_initial_no_prompt_no_forward(monkeypatch, tmp_path):
    studio_mod = _studio()
    events = _install_prompt_env(monkeypatch, tmp_path, interactive = True)
    before = _seed_auth(studio_mod)

    _invoke_studio_default(monkeypatch, events, ["--secure", "--password", "cli-supplied-pw12"])

    assert [kind for kind, _ in events] == ["exec"], events
    after = _auth_state(studio_mod)
    assert after["must_change_password"] == 0
    assert after["password_hash"] != before["password_hash"]
    assert after["jwt_secret"] != before["jwt_secret"]
    assert after["n_refresh"] == 0
    assert not (tmp_path / "auth" / studio_mod.BOOTSTRAP_PASSWORD_FILE).exists()
    assert "--password" not in _exec_argv(events)


def test_studio_default_password_via_env_strips_child_env(monkeypatch, tmp_path):
    import os

    studio_mod = _studio()
    events = _install_prompt_env(monkeypatch, tmp_path, interactive = False)
    _seed_auth(studio_mod)
    monkeypatch.setenv("UNSLOTH_STUDIO_PASSWORD", "env-supplied-pw12")

    _invoke_studio_default(monkeypatch, events, ["--secure"])

    assert [kind for kind, _ in events] == ["exec"], events
    assert _auth_state(studio_mod)["must_change_password"] == 0
    assert "UNSLOTH_STUDIO_PASSWORD" not in os.environ


def test_studio_default_password_via_stdin(monkeypatch, tmp_path):
    # CliRunner owns stdin during invoke, so feed `--password -` via input=.
    import typer as _typer

    studio_mod = _studio()
    events = _install_prompt_env(monkeypatch, tmp_path, interactive = False)
    _seed_auth(studio_mod)
    _install_studio_default_reexec(monkeypatch, events)
    app = _typer.Typer()
    app.command()(studio_mod.studio_default)
    CliRunner().invoke(
        app,
        ["--secure", "--password", "-"],
        input = "stdin-supplied-pw12\n",
        catch_exceptions = True,
    )

    assert [kind for kind, _ in events] == ["exec"], events
    assert _auth_state(studio_mod)["must_change_password"] == 0


def test_studio_default_password_too_short_fails_closed(monkeypatch, tmp_path):
    studio_mod = _studio()
    events = _install_prompt_env(monkeypatch, tmp_path, interactive = True)
    _seed_auth(studio_mod)

    result = _invoke_studio_default(monkeypatch, events, ["--secure", "--password", "short"])

    assert result.exit_code == 1
    assert [kind for kind, _ in events] == []
    assert _auth_state(studio_mod)["must_change_password"] == 1


def test_studio_default_password_must_differ_fails_closed(monkeypatch, tmp_path):
    studio_mod = _studio()
    events = _install_prompt_env(monkeypatch, tmp_path, interactive = True)
    _seed_auth(studio_mod)
    bootstrap_pw = (tmp_path / "auth" / studio_mod.BOOTSTRAP_PASSWORD_FILE).read_text().strip()

    result = _invoke_studio_default(monkeypatch, events, ["--secure", "--password", bootstrap_pw])

    assert result.exit_code == 1
    assert _auth_state(studio_mod)["must_change_password"] == 1


def test_studio_default_password_already_set_fails_closed(monkeypatch, tmp_path):
    studio_mod = _studio()
    events = _install_prompt_env(monkeypatch, tmp_path, interactive = True)
    _seed_auth(studio_mod, must_change = False)

    result = _invoke_studio_default(
        monkeypatch, events, ["--secure", "--password", "another-pw-12345"]
    )

    assert result.exit_code == 1
    assert [kind for kind, _ in events] == []


def test_studio_default_password_before_subcommand_errors(monkeypatch, tmp_path):
    import typer as _typer

    studio_mod = _studio()
    monkeypatch.setattr(studio_mod, "_ensure_studio_env_exported", lambda: None)
    app = _typer.Typer()
    app.add_typer(studio_mod.studio_app, name = "studio")
    result = CliRunner().invoke(app, ["studio", "--password", "x", "run", "--model", "X"])
    assert result.exit_code == 2
    combined = (result.output or "") + (getattr(result, "stderr", "") or "")
    assert "--password" in combined


def test_run_password_sets_initial_no_prompt_no_forward(monkeypatch, tmp_path):
    studio_mod = _studio()
    events = _install_prompt_env(monkeypatch, tmp_path, interactive = True)
    before = _seed_auth(studio_mod)

    _invoke_run(monkeypatch, events, _BASE + ["--secure", "--password", "cli-supplied-pw12"])

    assert [kind for kind, _ in events] == ["exec"], events
    after = _auth_state(studio_mod)
    assert after["must_change_password"] == 0
    assert after["password_hash"] != before["password_hash"]
    assert "--password" not in _exec_argv(events)


def test_run_password_via_env_strips_child_env(monkeypatch, tmp_path):
    import os

    studio_mod = _studio()
    events = _install_prompt_env(monkeypatch, tmp_path, interactive = False)
    _seed_auth(studio_mod)
    monkeypatch.setenv("UNSLOTH_STUDIO_PASSWORD", "env-supplied-pw12")

    _invoke_run(monkeypatch, events, _BASE + ["--secure"])

    assert [kind for kind, _ in events] == ["exec"], events
    assert _auth_state(studio_mod)["must_change_password"] == 0
    assert "UNSLOTH_STUDIO_PASSWORD" not in os.environ


def test_studio_default_password_applies_on_headless_wildcard_no_tunnel(monkeypatch, tmp_path):
    studio_mod = _studio()
    events = _install_prompt_env(monkeypatch, tmp_path, interactive = True)
    before = _seed_auth(studio_mod)

    _invoke_studio_default(
        monkeypatch, events, ["-H", "0.0.0.0", "--password", "headless-set-pw12"]
    )

    assert [kind for kind, _ in events] == ["exec"], events
    after = _auth_state(studio_mod)
    assert after["must_change_password"] == 0
    assert after["password_hash"] != before["password_hash"]
    assert "--password" not in _exec_argv(events)


_EXPOSURE_HOSTS = [
    "0.0.0.0",
    "::",
    "::0",
    "0:0:0:0:0:0:0:0",
    "0",
    "::ffff:0.0.0.0",
    "127.0.0.1",
    "localhost",
    "::1",
    "192.168.1.50",
    "10.0.0.5",
    "172.16.4.9",
    "example.com",
    "myhost.local",
    "[::]",
    "0.0.0.0.0",
]


@pytest.mark.parametrize("host", _EXPOSURE_HOSTS)
def test_cli_and_backend_agree_on_which_hosts_are_exposed(monkeypatch, host):
    """CLI and backend must agree on exposed hosts, or the seeded password can be served unprompted."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_bootstrap_timeout_probe",
        _REPO_ROOT / "studio" / "backend" / "auth" / "bootstrap_timeout.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(_REPO_ROOT / "studio" / "backend"))
    try:
        spec.loader.exec_module(module)
        backend_exposed = module._is_exposed_bind(host, False)
    finally:
        sys.path.remove(str(_REPO_ROOT / "studio" / "backend"))

    monkeypatch.setattr(_studio(), "_prompt_streams_interactive", lambda: True)
    cli_prompts = _studio()._should_prompt_password_change(
        cloudflare = None, host = host, secure = False, api_only = False
    )
    assert cli_prompts is backend_exposed, (
        f"{host!r}: CLI prompts={cli_prompts} but the backend considers it "
        f"exposed={backend_exposed}; the parent gate and the child gate disagree"
    )


@pytest.mark.skipif(
    os.name == "nt",
    reason = "POSIX terminal semantics: Windows has no process groups, no SIGTTOU and no pty, "
    "so there is nothing here to assert. _prompt_owns_the_terminal fails open there, "
    "which test_windows_has_no_terminal_ownership_to_lose pins.",
)
def test_a_backgrounded_raw_bind_does_not_prompt(monkeypatch):
    """`unsloth studio -H 0.0.0.0 &` must still launch.

    A background job inherits the terminal, so isatty() is True on both streams,
    but the masked prompt calls termios.tcsetattr; POSIX SIGTTOUs a background
    process group that does, and the default action STOPS the process. The launch
    would freeze before the socket binds, so a launch that worked before this gate
    widened stops working. It keeps the bootstrap deadline it already had.
    """
    studio = _studio()
    monkeypatch.setattr(studio, "_prompt_streams_interactive", lambda: True)
    monkeypatch.setattr(studio.os, "tcgetpgrp", lambda _fd: 4242)
    monkeypatch.setattr(studio.os, "getpgrp", lambda: 99)
    monkeypatch.setattr(studio.sys, "stdin", _FdStream())

    assert (
        studio._should_prompt_password_change(
            cloudflare = None, host = "0.0.0.0", secure = False, api_only = False
        )
        is False
    )
    assert (
        studio._should_prompt_password_change(
            cloudflare = None, host = "0.0.0.0", secure = True, api_only = False
        )
        is True
    )


@pytest.mark.skipif(
    os.name == "nt",
    reason = "POSIX terminal semantics: Windows has no process groups, no SIGTTOU and no pty, "
    "so there is nothing here to assert. _prompt_owns_the_terminal fails open there, "
    "which test_windows_has_no_terminal_ownership_to_lose pins.",
)
def test_a_foreground_raw_bind_still_prompts(monkeypatch):
    """The ordinary interactive case is untouched."""
    studio = _studio()
    monkeypatch.setattr(studio, "_prompt_streams_interactive", lambda: True)
    monkeypatch.setattr(studio.os, "tcgetpgrp", lambda _fd: 4242)
    monkeypatch.setattr(studio.os, "getpgrp", lambda: 4242)
    monkeypatch.setattr(studio.sys, "stdin", _FdStream())

    assert (
        studio._should_prompt_password_change(
            cloudflare = None, host = "0.0.0.0", secure = False, api_only = False
        )
        is True
    )


@pytest.mark.parametrize("raised", [OSError("ENOTTY"), AttributeError(), ValueError()])
def test_no_job_control_falls_back_to_the_isatty_answer(monkeypatch, raised):
    """Windows / no controlling terminal: nothing can stop us, so still prompt."""
    studio = _studio()
    monkeypatch.setattr(studio, "_prompt_streams_interactive", lambda: True)

    def _boom(_fd):
        raise raised

    monkeypatch.setattr(studio.os, "tcgetpgrp", _boom, raising = False)
    monkeypatch.setattr(studio.sys, "stdin", _FdStream())

    assert (
        studio._should_prompt_password_change(
            cloudflare = None, host = "0.0.0.0", secure = False, api_only = False
        )
        is True
    )


class _FdStream:
    def fileno(self):
        return 0

    def isatty(self):
        return True


def test_an_unattended_pty_still_launches_a_raw_bind(monkeypatch, tmp_path):
    """Detached ptys look interactive but nobody types; the prompt must time out, not hang the launch."""
    studio_mod = _studio()
    events = _install_prompt_env(
        monkeypatch,
        tmp_path,
        interactive = True,
        scripted = studio_mod._password_prompt.PromptUnattended(),
    )
    monkeypatch.setattr(studio_mod, "_prompt_owns_the_terminal", lambda: True)
    _seed_auth(studio_mod)

    result = _invoke_studio_default(monkeypatch, events, ["-H", "0.0.0.0"])

    kinds = [kind for kind, _ in events]
    assert kinds == ["prompt", "exec"], events
    assert result.exit_code == 0, result.output
    assert _auth_state(studio_mod)["must_change_password"] == 1
    assert (tmp_path / "auth" / studio_mod.BOOTSTRAP_PASSWORD_FILE).exists()


def test_an_unattended_pty_still_aborts_a_tunnel_launch(monkeypatch, tmp_path):
    """A tunnel publishes a public URL, so it gets no deadline and no fallback."""
    studio_mod = _studio()
    seen = {}

    def _fake_prompt(
        verify_current,
        out = None,
        **kw,
    ):
        seen.update(kw)
        return _NEW_PW

    events = _install_prompt_env(monkeypatch, tmp_path, interactive = True)
    monkeypatch.setattr(studio_mod._password_prompt, "prompt_new_password", _fake_prompt)
    _seed_auth(studio_mod)

    _invoke_studio_default(monkeypatch, events, ["--secure"])
    assert seen["first_key_timeout"] is None


def test_a_raw_bind_prompt_carries_the_unattended_deadline(monkeypatch, tmp_path):
    """The other half: only the launch that must not be blocked is deadlined."""
    studio_mod = _studio()
    seen = {}

    def _fake_prompt(
        verify_current,
        out = None,
        **kw,
    ):
        seen.update(kw)
        return _NEW_PW

    events = _install_prompt_env(monkeypatch, tmp_path, interactive = True)
    monkeypatch.setattr(studio_mod._password_prompt, "prompt_new_password", _fake_prompt)
    monkeypatch.setattr(studio_mod, "_prompt_owns_the_terminal", lambda: True)
    _seed_auth(studio_mod)

    _invoke_studio_default(monkeypatch, events, ["-H", "0.0.0.0"])
    assert seen["first_key_timeout"] == studio_mod._UNATTENDED_PROMPT_SECONDS


@pytest.mark.skipif(
    os.name == "nt",
    reason = "POSIX terminal semantics: Windows has no process groups, no SIGTTOU and no pty, "
    "so there is nothing here to assert. _prompt_owns_the_terminal fails open there, "
    "which test_windows_has_no_terminal_ownership_to_lose pins.",
)
def test_read_masked_gives_up_on_a_pty_nobody_types_into(monkeypatch):
    """The mechanism itself, against a real pty with no writer."""
    import os
    import pty

    from unsloth_cli.commands import _password_prompt

    master, slave = pty.openpty()

    class _PtyStdin:
        encoding = "utf-8"

        def fileno(self):
            return slave

        def isatty(self):
            return True

    try:
        monkeypatch.setattr(_password_prompt.sys, "stdin", _PtyStdin())
        out = io.StringIO()
        with pytest.raises(_password_prompt.PromptUnattended):
            _password_prompt.read_masked("New password: ", out, first_key_timeout = 0.25)
    finally:
        os.close(master)
        os.close(slave)


@pytest.mark.parametrize(
    "cloudflare,host,secure,expect_tunnel_wording",
    [
        (None, "127.0.0.1", True, True),
        (True, "0.0.0.0", False, True),
        (True, "192.168.1.50", False, False),
        (True, "example.com", False, False),
        (None, "0.0.0.0", False, False),
        (None, "192.168.1.50", False, False),
    ],
)
def test_the_exposure_wording_matches_what_will_actually_happen(
    monkeypatch, tmp_path, cloudflare, host, secure, expect_tunnel_wording
):
    """Prompt wording follows the tunnel predicate, not the request; a LAN bind is not a public URL."""
    studio_mod = _studio()
    monkeypatch.setattr(studio_mod, "_prompt_streams_interactive", lambda: True)
    tunnel = studio_mod._launch_publishes_tunnel(
        cloudflare = cloudflare, host = host, secure = secure, api_only = False
    )
    assert tunnel is expect_tunnel_wording


def test_the_cli_deadline_sentence_tracks_the_configured_timeout(monkeypatch):
    """The CLI fallback must not promise a shutdown that will never arm."""
    studio_mod = _studio()
    monkeypatch.delenv("UNSLOTH_STUDIO_BOOTSTRAP_TIMEOUT", raising = False)
    assert "shuts down after the bootstrap deadline" in studio_mod._deadline_sentence()
    for disabled in ("0", "-1"):
        monkeypatch.setenv("UNSLOTH_STUDIO_BOOTSTRAP_TIMEOUT", disabled)
        sentence = studio_mod._deadline_sentence()
        assert "DISABLED for this launch" in sentence, disabled
        assert "shuts down after" not in sentence, disabled


def test_a_raw_bind_ctrl_c_aborts_the_launch(monkeypatch, tmp_path):
    """Ctrl+C on `unsloth studio -H 0.0.0.0` is an explicit refusal."""
    import os as _os
    import typer

    studio_mod = _studio()
    events = _install_prompt_env(monkeypatch, tmp_path, interactive = True)
    _seed_auth(studio_mod)

    def _abort(*_a, **_kw):
        raise KeyboardInterrupt

    monkeypatch.setattr(studio_mod._password_prompt, "prompt_new_password", _abort)

    with pytest.raises(typer.Exit):
        studio_mod._enforce_password_change_before_exposure(
            cloudflare = None, host = "0.0.0.0", secure = False, api_only = False
        )
    assert _auth_state(studio_mod)["must_change_password"] == 1
    assert _os.environ.get(studio_mod._UNATTENDED_PROMPT_DONE_ENV) is None
    del events


@pytest.mark.parametrize(
    "args,present,absent",
    [
        (dict(cloudflare = None, host = "0.0.0.0", secure = False), "-H 127.0.0.1", "--cloudflare"),
        (
            dict(cloudflare = None, host = "127.0.0.1", secure = True),
            "--secure/--cloudflare",
            "-H 127.0.0.1",
        ),
    ],
)
def test_the_abort_names_a_remedy_this_launch_actually_has(
    monkeypatch, tmp_path, capsys, args, present, absent
):
    """An abort that leaves no way forward just gets retried the same way."""
    import typer

    studio_mod = _studio()
    _install_prompt_env(monkeypatch, tmp_path, interactive = True)
    _seed_auth(studio_mod)

    def _abort(*_a, **_kw):
        raise KeyboardInterrupt

    monkeypatch.setattr(studio_mod._password_prompt, "prompt_new_password", _abort)

    with pytest.raises(typer.Exit):
        studio_mod._enforce_password_change_before_exposure(api_only = False, **args)

    err = capsys.readouterr().err
    assert "UNSLOTH_STUDIO_PASSWORD" in err, err
    assert present in err, err
    assert absent not in err, err


def test_a_tunnel_ctrl_c_still_aborts(monkeypatch, tmp_path):
    studio_mod = _studio()
    _install_prompt_env(monkeypatch, tmp_path, interactive = True)
    _seed_auth(studio_mod)

    def _abort(*_a, **_kw):
        raise KeyboardInterrupt

    monkeypatch.setattr(studio_mod._password_prompt, "prompt_new_password", _abort)

    import typer

    with pytest.raises(typer.Exit):
        studio_mod._enforce_password_change_before_exposure(
            cloudflare = None, host = "127.0.0.1", secure = True, api_only = False
        )


def test_a_second_cli_gate_does_not_re_wait_the_same_dead_terminal(monkeypatch, tmp_path):
    """Re-exec'd child must honour the parent's already-tried mark on the same pty; peek, never pop."""
    studio_mod = _studio()
    calls = []

    def _fake_prompt(
        verify_current,
        out = None,
        **kw,
    ):
        calls.append(kw)
        raise studio_mod._password_prompt.PromptUnattended

    events = _install_prompt_env(monkeypatch, tmp_path, interactive = True)
    monkeypatch.setattr(studio_mod._password_prompt, "prompt_new_password", _fake_prompt)
    monkeypatch.setattr(studio_mod, "_prompt_owns_the_terminal", lambda: True)
    _seed_auth(studio_mod)

    _invoke_studio_default(monkeypatch, events, ["-H", "0.0.0.0"])
    assert len(calls) == 1
    import os as _os

    assert _os.environ.get(studio_mod._UNATTENDED_PROMPT_DONE_ENV) == "1"

    _invoke_studio_default(monkeypatch, events, ["-H", "0.0.0.0"])
    assert len(calls) == 1, "the re-executed gate waited on the dead terminal again"
    assert _os.environ.get(studio_mod._UNATTENDED_PROMPT_DONE_ENV) == "1", "popped, not peeked"


def test_the_mark_never_lets_a_tunnel_skip_its_prompt(monkeypatch, tmp_path):
    """A public URL fails closed, mark or no mark."""
    studio_mod = _studio()
    calls = []

    def _fake_prompt(
        verify_current,
        out = None,
        **kw,
    ):
        calls.append(kw)
        return _NEW_PW

    events = _install_prompt_env(monkeypatch, tmp_path, interactive = True)
    monkeypatch.setattr(studio_mod._password_prompt, "prompt_new_password", _fake_prompt)
    monkeypatch.setattr(studio_mod, "_prompt_owns_the_terminal", lambda: True)
    monkeypatch.setenv(studio_mod._UNATTENDED_PROMPT_DONE_ENV, "1")
    _seed_auth(studio_mod)

    _invoke_studio_default(monkeypatch, events, ["--secure"])
    assert len(calls) == 1, "a tunnel launch skipped its prompt because of the mark"


def _banner(monkeypatch, tmp_path, args):
    """Clears the unattended marker first, so one banner assertion cannot inherit it from another."""
    import os as _os

    studio_mod = _studio()
    _os.environ.pop(studio_mod._UNATTENDED_PROMPT_DONE_ENV, None)

    def _fake_prompt(
        verify_current,
        out = None,
        **kw,
    ):
        raise KeyboardInterrupt

    events = _install_prompt_env(monkeypatch, tmp_path, interactive = True)
    monkeypatch.setattr(studio_mod._password_prompt, "prompt_new_password", _fake_prompt)
    monkeypatch.setattr(studio_mod, "_prompt_owns_the_terminal", lambda: True)
    _seed_auth(studio_mod)
    result = _invoke_studio_default(monkeypatch, events, args)
    return (result.output or "") + (getattr(result, "stderr", "") or "")


def test_the_banner_promises_abort_for_every_exposed_bind(monkeypatch, tmp_path):
    raw = _banner(monkeypatch, tmp_path, ["-H", "0.0.0.0"])
    assert "Ctrl+C to abort" in raw
    assert "Ctrl+C to skip" not in raw

    tunnel = _banner(monkeypatch, tmp_path, ["--secure"])
    assert "Ctrl+C to abort" in tunnel


def test_a_concrete_bind_is_not_described_as_every_interface(monkeypatch, tmp_path):
    """A concrete bind like -H 192.168.1.50 listens on one address; it must not claim every interface."""
    concrete = _banner(monkeypatch, tmp_path, ["-H", "192.168.1.50"])
    assert "on every network interface" not in concrete
    assert "192.168.1.50" in concrete

    wildcard = _banner(monkeypatch, tmp_path, ["-H", "0.0.0.0"])
    assert "on every network interface" in wildcard
