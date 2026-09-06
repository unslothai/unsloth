# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the forced terminal password change before public (tunnel) exposure.

`unsloth studio --secure` / `--cloudflare` (wildcard bind) must, when the admin
account still has its seeded bootstrap password, prompt for a new password in
the terminal BEFORE any re-exec or server exists; without a terminal it warns
and falls back to the backend bootstrap timeout. Modeled on
test_studio_cloudflare_flag.py.
"""

from __future__ import annotations

import sqlite3
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
_NEW_PW = "brand-new-password"


# ── pure trigger matrix ──────────────────────────────────────────────


@pytest.mark.parametrize(
    "cloudflare,host,secure,api_only,expected",
    [
        # --secure always implies the tunnel (host already forced to loopback).
        (None, "127.0.0.1", True, False, True),
        (True, "127.0.0.1", True, False, True),
        (None, "127.0.0.1", True, True, True),
        # --cloudflare tunnels only non-api-only wildcard binds.
        (True, "0.0.0.0", False, False, True),
        (True, "::", False, False, True),
        (True, "::0", False, False, True),
        (True, "0:0:0:0:0:0:0:0", False, False, True),
        (True, "0", False, False, True),
        (True, "::ffff:0.0.0.0", False, False, True),
        (True, "", False, False, False),
        (True, "127.0.0.1", False, False, False),
        (True, "0.0.0.0", False, True, False),
        # Off/unset never prompts without --secure.
        (None, "0.0.0.0", False, False, False),
        (False, "0.0.0.0", False, False, False),
        (None, "127.0.0.1", False, False, False),
    ],
)
def test_should_prompt_password_change_matrix(cloudflare, host, secure, api_only, expected):
    assert (
        _studio()._should_prompt_password_change(
            cloudflare = cloudflare, host = host, secure = secure, api_only = api_only
        )
        is expected
    )


# ── shared harness ───────────────────────────────────────────────────


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
    # cloudflared is "available" by default so the headless --secure strip path
    # proceeds without a real download; the unavailable-tunnel guard has its own
    # dedicated test that overrides this.
    monkeypatch.setattr(studio_mod, "_tunnel_binary_confirmed_unavailable", lambda: False)
    # The one-time-secret selector now requires a real tty (a redirected/persisted
    # stream would leak the credential -- see _one_time_secret_console_stream and
    # test_one_time_secret_console_stream_requires_tty). CliRunner's captured stderr
    # is not a tty, so provide a usable console here: these tests exercise the
    # rotation/re-exec logic given a surface, not the tty preflight. Tests for the
    # no-console fail-closed path override this back to None afterwards.
    monkeypatch.setattr(studio_mod, "_one_time_secret_console_stream", lambda **_kw: sys.stderr)

    def fake_prompt(verify_current, out = None):
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
    # A built frontend dist is present by default so the public-launch UI check
    # passes; the no-dist lockout guard has its own dedicated test.
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
    # A built frontend dist is present by default so the public-launch UI check
    # passes deterministically (independent of whether the repo dist was built);
    # the missing-dist lockout guard has its own dedicated test.
    monkeypatch.setattr(
        studio_mod, "_find_frontend_dist", lambda: Path("/fake/studio/frontend/dist")
    )
    fake_bin = fake_venv / "bin" / "unsloth"
    real_is_file = Path.is_file
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


# ── plain `unsloth studio` ───────────────────────────────────────────


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
    # The verify_current callback handed to the prompt must recognize the
    # seeded bootstrap password (hash compare with the stored salt).
    studio_mod = _studio()
    events = _install_prompt_env(monkeypatch, tmp_path, interactive = True)
    _seed_auth(studio_mod)
    bootstrap_pw = (tmp_path / "auth" / studio_mod.BOOTSTRAP_PASSWORD_FILE).read_text().strip()

    _invoke_studio_default(monkeypatch, events, ["--secure"])

    verify_current = events[0][1]
    assert verify_current(bootstrap_pw) is True
    assert verify_current("something-else-entirely") is False


def test_studio_default_non_tty_autogenerates_and_proceeds(monkeypatch, tmp_path):
    # No terminal + seeded default password: auto-generate a strong admin
    # password, commit it (clears must_change so the child launches cleanly), and
    # print it once before re-exec.
    studio_mod = _studio()
    events = _install_prompt_env(monkeypatch, tmp_path, interactive = False)
    _seed_auth(studio_mod)

    result = _invoke_studio_default(monkeypatch, events, ["--secure"])

    kinds = [kind for kind, _ in events]
    assert kinds == ["exec"], events
    combined = (result.output or "") + (getattr(result, "stderr", "") or "")
    assert "auto-generated" in combined.lower()
    state = _auth_state(studio_mod)
    assert state["must_change_password"] == 0
    assert state["n_refresh"] == 0  # refresh tokens revoked in the same transaction


def test_studio_default_non_tty_deletes_bootstrap_password_file(monkeypatch, tmp_path):
    # Auto-generation commits a new password and deletes the seeded plaintext
    # credential before re-exec, so a fresh child of ANY version reads None from
    # disk and never injects it into the public HTML. The launch still proceeds.
    studio_mod = _studio()
    events = _install_prompt_env(monkeypatch, tmp_path, interactive = False)
    _seed_auth(studio_mod)
    bootstrap_file = tmp_path / "auth" / studio_mod.BOOTSTRAP_PASSWORD_FILE
    assert bootstrap_file.exists()

    _invoke_studio_default(monkeypatch, events, ["--secure"])

    assert not bootstrap_file.exists()
    kinds = [kind for kind, _ in events]
    assert kinds == ["exec"], events
    assert _auth_state(studio_mod)["must_change_password"] == 0


def test_one_time_secret_console_stream_prefers_stderr_then_stdout(monkeypatch):
    # Unit: prefer stderr, fall back to stdout, None when neither is usable (absent
    # or closed). Mirrors run._one_time_secret_stream so the CLI parent never rotates
    # the recovery credential when there is nowhere to surface the replacement.
    studio_mod = _studio()

    class _Stream:
        def __init__(
            self,
            *,
            closed = False,
            tty = True,
        ):
            self.closed = closed
            self._tty = tty

        def write(self, *_a, **_k):
            pass

        def isatty(self):
            return self._tty

    real_err, real_out = _Stream(), _Stream()

    monkeypatch.setattr(sys, "stderr", real_err)
    monkeypatch.setattr(sys, "stdout", real_out)
    assert studio_mod._one_time_secret_console_stream() is real_err  # stderr wins

    monkeypatch.setattr(sys, "stderr", None)
    assert studio_mod._one_time_secret_console_stream() is real_out  # falls back

    monkeypatch.setattr(sys, "stderr", _Stream(closed = True))
    monkeypatch.setattr(sys, "stdout", _Stream(closed = True))
    assert studio_mod._one_time_secret_console_stream() is None  # no usable console

    monkeypatch.setattr(sys, "stderr", None)
    monkeypatch.setattr(sys, "stdout", None)
    assert studio_mod._one_time_secret_console_stream() is None  # pythonw wrapper


def test_one_time_secret_console_stream_requires_tty(monkeypatch):
    # Regression (Codex 3644671925, P1): a headless CLI launch (nohup / systemd /
    # `> log 2>&1`) inherits a stderr/stdout that is open and writable but NOT a
    # tty. Surfacing the auto-generated password there PERSISTS the plaintext to
    # the file/journal (CWE-532), so the selector must skip a non-tty stream and
    # fall back to a real terminal, returning None when neither is a tty.
    studio_mod = _studio()

    class _Stream:
        def __init__(self, *, tty):
            self.closed = False
            self._tty = tty

        def write(self, *_a, **_k):
            pass

        def isatty(self):
            return self._tty

    redirected_err, tty_out = _Stream(tty = False), _Stream(tty = True)
    monkeypatch.setattr(sys, "stderr", redirected_err)
    monkeypatch.setattr(sys, "stdout", tty_out)
    # Redirected stderr is skipped; falls back to the tty stdout.
    assert studio_mod._one_time_secret_console_stream() is tty_out

    # Both redirected (fully headless) -> None, so the caller fails closed rather
    # than write the credential into a retained file/journal.
    monkeypatch.setattr(sys, "stderr", _Stream(tty = False))
    monkeypatch.setattr(sys, "stdout", _Stream(tty = False))
    assert studio_mod._one_time_secret_console_stream() is None


def test_studio_default_non_tty_no_console_preserves_bootstrap(monkeypatch, tmp_path):
    # Non-interactive launch with no usable console (a Windows pythonw/service
    # wrapper leaves stderr/stdout absent): the auto-generated password could never
    # be shown, so rotating the seeded recovery credential would lock the operator
    # out. Fail closed WITHOUT rotating -- the bootstrap password and must_change
    # flag are preserved, and the launch aborts before any re-exec.
    studio_mod = _studio()
    events = _install_prompt_env(monkeypatch, tmp_path, interactive = False)
    before = _seed_auth(studio_mod)
    bootstrap_file = tmp_path / "auth" / studio_mod.BOOTSTRAP_PASSWORD_FILE
    assert bootstrap_file.exists()
    monkeypatch.setattr(studio_mod, "_one_time_secret_console_stream", lambda **_kw: None)

    result = _invoke_studio_default(monkeypatch, events, ["--secure"])

    assert result.exit_code == 1, result.output
    # No rotation and no re-exec: recovery credential intact.
    assert bootstrap_file.exists()
    after = _auth_state(studio_mod)
    assert after["must_change_password"] == 1
    assert after["password_hash"] == before["password_hash"]
    assert after["jwt_secret"] == before["jwt_secret"]
    assert "exec" not in [k for k, _ in events], events


def test_studio_default_reexec_outer_runpy_autogenerates(monkeypatch, tmp_path):
    # Even when the re-exec target is THIS install's own run.py (self-suppressing),
    # the parent auto-generates and commits a strong password so it is surfaced
    # once here and the child sees must_change=0 and no-ops.
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

    assert not bootstrap_file.exists(), result.output
    assert _auth_state(studio_mod)["must_change_password"] == 0
    assert "exec" in [k for k, _ in events], events


def test_studio_default_non_tty_persists_seeded_admin_on_fresh_home(monkeypatch, tmp_path):
    # Fresh STUDIO_HOME (no pre-seed): the gate's own _ensure_cli_default_admin
    # does the INSERT, then auto-generation commits a strong password over it. The
    # committed change persists (must_change=0) and the launch re-execs.
    studio_mod = _studio()
    events = _install_prompt_env(monkeypatch, tmp_path, interactive = False)
    # Deliberately NO _seed_auth(): exercise the gate seeding a fresh DB itself.

    _invoke_studio_default(monkeypatch, events, ["--secure"])

    state = _auth_state(studio_mod)
    assert state["must_change_password"] == 0
    assert not (tmp_path / "auth" / studio_mod.BOOTSTRAP_PASSWORD_FILE).exists()
    kinds = [kind for kind, _ in events]
    assert kinds == ["exec"], events


def test_studio_default_non_tty_autogen_survives_locked_bootstrap_file(monkeypatch, tmp_path):
    # The new password is already committed before the seeded file is removed, so a
    # locked/undeletable .bootstrap_password (Windows AV / read-only dir) must NOT
    # fail the launch: it is truncated instead and the launch proceeds.
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
    assert kinds == ["exec"], events
    assert result.exit_code == 0, result.output
    # Password rotated (must_change cleared); the locked file is truncated so its
    # stale plaintext cannot be reused.
    assert _auth_state(studio_mod)["must_change_password"] == 0
    assert bootstrap_file.read_text() == ""


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
    """Wrap a real auth connection but raise on commit(), so a fresh install's
    seeded admin INSERT rolls back on close() -- the seed-committed guarantee the
    gate depends on is not met, even though _ensure_cli_default_admin already
    wrote the .bootstrap_password file."""

    def __init__(self, inner):
        self._inner = inner

    def commit(self):
        raise sqlite3.OperationalError("database is locked")

    def __getattr__(self, name):
        return getattr(self._inner, name)


def test_studio_default_connect_failure_fails_closed(monkeypatch, tmp_path):
    # If the auth DB cannot even be opened (transient lock / unwritable home) we
    # cannot confirm a committed admin exists, so a re-exec'd old studio-venv child
    # could find no admin, regenerate a fresh bootstrap credential, and serve it
    # publicly -- stripping a file we cannot vouch for would not stop that. Refuse
    # rather than publish; a transient lock clears on retry, and the existing
    # credential file is left untouched so a retry can still prompt.
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
    assert "refusing to publish" in combined.lower()
    # Not stripped: a retry can still prompt/strip once the lock clears.
    assert bootstrap_file.exists()


def test_studio_default_seed_commit_failure_fails_closed(monkeypatch, tmp_path):
    # Fresh install: the gate's own _ensure_cli_default_admin does the INSERT and
    # writes .bootstrap_password, but the commit fails (write lock held past
    # busy_timeout). The uncommitted admin rolls back on close, so a re-exec'd old
    # child would find no admin and regenerate + serve a fresh default credential;
    # stripping cannot stop a regeneration. The gate must fail closed.
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
    assert "refusing to publish" in combined.lower()
    # The half-written seed file is stripped, and no admin row was committed.
    assert not (tmp_path / "auth" / studio_mod.BOOTSTRAP_PASSWORD_FILE).exists()
    verify = sqlite3.connect(_auth_db(tmp_path))
    try:
        assert verify.execute("SELECT COUNT(*) FROM auth_user").fetchone()[0] == 0
    finally:
        verify.close()


def test_studio_default_missing_venv_exits_before_stripping_bootstrap(monkeypatch, tmp_path):
    # Regression: the venv/run.py launchability check must run BEFORE the headless
    # gate strips .bootstrap_password. Otherwise a failed launch leaves the admin
    # at must_change_password=1 with no password to log in (lockout until
    # reset-password). With the venv missing, exit without stripping the file.
    import typer as _typer

    studio_mod = _studio()
    events = _install_prompt_env(monkeypatch, tmp_path, interactive = False)
    _seed_auth(studio_mod)
    bootstrap_file = tmp_path / "auth" / studio_mod.BOOTSTRAP_PASSWORD_FILE
    assert bootstrap_file.exists()

    monkeypatch.setattr(sys, "prefix", "/nonexistent/outer/venv")
    monkeypatch.setattr(studio_mod, "_studio_venv_python", lambda: None)  # venv missing
    monkeypatch.setattr(studio_mod, "_find_run_py", lambda: None)

    app = _typer.Typer()
    app.command()(studio_mod.studio_default)
    result = CliRunner().invoke(app, ["--secure"], catch_exceptions = True)

    assert result.exit_code == 1, result.output
    # The seeded file survives: launchability failed BEFORE the gate could strip it.
    assert bootstrap_file.exists()
    assert _auth_state(studio_mod)["must_change_password"] == 1
    # The gate never ran (no prompt, no strip).
    assert events == [], events
    combined = (result.output or "") + (getattr(result, "stderr", "") or "")
    assert "not set up" in combined.lower()


def test_studio_default_missing_frontend_exits_before_stripping_bootstrap(monkeypatch, tmp_path):
    # Regression (item B): a public UI launch needs a built frontend dist -- the
    # login page is the ONLY way to change the seeded password. Resolve it BEFORE
    # the headless gate strips .bootstrap_password, so a missing dist aborts the
    # launch without stripping (no lockout at must_change_password=1 with nothing
    # left to log in with).
    import typer as _typer

    studio_mod = _studio()
    events = _install_prompt_env(monkeypatch, tmp_path, interactive = False)
    _seed_auth(studio_mod)
    bootstrap_file = tmp_path / "auth" / studio_mod.BOOTSTRAP_PASSWORD_FILE
    assert bootstrap_file.exists()

    # Launcher present, but no built frontend dist.
    monkeypatch.setattr(sys, "prefix", "/nonexistent/outer/venv")
    fake_venv = Path("/fake/studio/venv/unsloth_studio")
    monkeypatch.setattr(studio_mod, "_studio_venv_python", lambda: fake_venv / "bin" / "python")
    monkeypatch.setattr(studio_mod, "_find_run_py", lambda: Path("/fake/studio/run.py"))
    monkeypatch.setattr(studio_mod, "_find_frontend_dist", lambda: None)

    app = _typer.Typer()
    app.command()(studio_mod.studio_default)
    result = CliRunner().invoke(app, ["--secure"], catch_exceptions = True)

    assert result.exit_code == 1, result.output
    # The seeded file survives: the frontend check failed BEFORE the gate stripped it.
    assert bootstrap_file.exists()
    assert _auth_state(studio_mod)["must_change_password"] == 1
    # The gate never ran (no prompt, no strip, no exec).
    assert events == [], events
    combined = (result.output or "") + (getattr(result, "stderr", "") or "")
    assert "frontend is not built" in combined.lower()


def test_studio_default_bad_frontend_path_exits_before_stripping_bootstrap(monkeypatch, tmp_path):
    # Regression (item B / reviewer finding): a user-supplied --frontend that does
    # not contain index.html must NOT bypass the servable-UI guard. Otherwise the
    # headless gate strips .bootstrap_password and the child serves no login page
    # -> lockout. Validate the path BEFORE the gate and abort without stripping.
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
    # Auto-resolution would find a dist, but the user forced an empty one (no
    # index.html): the guard must reject it rather than trust it.
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
    assert bootstrap_file.exists()  # not stripped
    assert _auth_state(studio_mod)["must_change_password"] == 1
    assert events == [], events
    combined = (result.output or "") + (getattr(result, "stderr", "") or "")
    assert "index.html" in combined.lower()


def test_studio_default_missing_frontend_loopback_cloudflare_still_launches(monkeypatch, tmp_path):
    # The dist guard is scoped to public exposure only. A loopback --cloudflare
    # (default host) does not tunnel, so a missing dist must NOT abort it -- the
    # launch proceeds exactly as before.
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
    # Regression (item B / reviewer finding): the in-venv (in-process) path skips
    # the re-exec launcher check, so a headless public launch would seed + strip
    # the seeded .bootstrap_password in the gate before _load_run_module() later
    # fails on a broken/partial venv -> lockout. Validate the backend is
    # importable BEFORE the strip and abort without stripping.
    import typer as _typer

    studio_mod = _studio()
    events = _install_prompt_env(monkeypatch, tmp_path, interactive = False)
    _seed_auth(studio_mod)
    bootstrap_file = tmp_path / "auth" / studio_mod.BOOTSTRAP_PASSWORD_FILE
    assert bootstrap_file.exists()

    # Pretend we are already inside the studio venv, with a broken backend. The
    # servable-frontend guard runs first (and would otherwise exit on the missing
    # dist), so stub it satisfied to isolate the backend-load guard under test.
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "unsloth_studio"))
    # A built dist is not present in a fresh clone. The missing-frontend gate
    # runs first and has its own test below; stub it so this one reaches the
    # backend check it is actually about.
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
    assert bootstrap_file.exists()  # not stripped
    assert _auth_state(studio_mod)["must_change_password"] == 1
    assert events == [], events  # gate never stripped/prompted
    combined = (result.output or "") + (getattr(result, "stderr", "") or "")
    assert "backend could not be loaded" in combined.lower()


def test_studio_default_in_venv_missing_frontend_exits_before_stripping_bootstrap(
    monkeypatch, tmp_path
):
    # Regression (Codex): the in-venv (in-process) path validated the backend but
    # not the frontend, so a headless public launch would strip the seeded
    # password in the gate before run_server() aborted on a missing dist. Validate
    # the servable frontend BEFORE the strip, same as the re-exec path.
    import typer as _typer

    studio_mod = _studio()
    events = _install_prompt_env(monkeypatch, tmp_path, interactive = False)
    _seed_auth(studio_mod)
    bootstrap_file = tmp_path / "auth" / studio_mod.BOOTSTRAP_PASSWORD_FILE
    assert bootstrap_file.exists()

    monkeypatch.setattr(sys, "prefix", str(tmp_path / "unsloth_studio"))  # in-venv
    monkeypatch.setattr(studio_mod, "_find_frontend_dist", lambda: None)  # no built dist
    monkeypatch.setattr(studio_mod, "_load_run_module", lambda: None)  # backend fine

    app = _typer.Typer()
    app.command()(studio_mod.studio_default)
    result = CliRunner().invoke(app, ["--secure"], catch_exceptions = True)

    assert result.exit_code == 1, result.output
    assert bootstrap_file.exists()  # not stripped
    assert _auth_state(studio_mod)["must_change_password"] == 1
    assert events == [], events
    combined = (result.output or "") + (getattr(result, "stderr", "") or "")
    assert "frontend is not built" in combined.lower()


def test_studio_default_secure_tunnel_unavailable_preserves_bootstrap(monkeypatch, tmp_path):
    # Regression (Codex): a headless --secure launch strips the only plaintext
    # recovery credential before the child proves the tunnel can start. If
    # cloudflared is provably unavailable no public URL comes up (loopback bind),
    # so the strip must be skipped and the launch refused, preserving recovery.
    import typer as _typer

    studio_mod = _studio()
    events = _install_prompt_env(monkeypatch, tmp_path, interactive = False)
    _seed_auth(studio_mod)
    bootstrap_file = tmp_path / "auth" / studio_mod.BOOTSTRAP_PASSWORD_FILE
    assert bootstrap_file.exists()

    _install_studio_default_reexec(monkeypatch, events)
    # cloudflared cannot be found or downloaded -> the --secure tunnel is dead.
    monkeypatch.setattr(studio_mod, "_tunnel_binary_confirmed_unavailable", lambda: True)

    app = _typer.Typer()
    app.command()(studio_mod.studio_default)
    result = CliRunner().invoke(app, ["--secure"], catch_exceptions = True)

    assert result.exit_code == 1, result.output
    assert bootstrap_file.exists()  # preserved for recovery, NOT stripped
    assert _auth_state(studio_mod)["must_change_password"] == 1
    assert "exec" not in [k for k, _ in events], events  # never re-exec'd
    combined = (result.output or "") + (getattr(result, "stderr", "") or "")
    assert "cloudflared" in combined.lower()


def test_studio_default_wildcard_cloudflare_strips_even_if_tunnel_unavailable(
    monkeypatch, tmp_path
):
    # The unavailable-tunnel skip is --secure-only: a wildcard --cloudflare binds
    # 0.0.0.0 publicly regardless of the tunnel, so the seeded password must still
    # be stripped even when cloudflared is unavailable.
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

    # Still strips (raw public bind) and re-execs.
    assert not bootstrap_file.exists(), result.output
    assert "exec" in [k for k, _ in events], events


def test_tunnel_probe_adds_backend_to_syspath(monkeypatch, tmp_path):
    # Regression (Codex 3572165922): ensure_cloudflared -> _cache_path lazily
    # imports utils.paths.storage_roots, which only resolves when studio/backend is
    # on sys.path. From the outer CLI it is not, so the probe must add it or it
    # false-reports "unavailable" and wrongly refuses --secure. Model that with a
    # cloudflare_tunnel whose ensure_cloudflared resolves ONLY when backend is on
    # sys.path.
    studio_mod = _studio()
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
    assert str(backend) not in sys.path  # precondition

    result = studio_mod._tunnel_binary_confirmed_unavailable()

    # ensure_cloudflared resolved (backend was on sys.path) -> available -> not
    # "confirmed unavailable"; without the fix it would false-report True.
    assert result is False
    # The probe cleans up the sys.path entry it added.
    assert str(backend) not in sys.path


def test_studio_default_query_failure_strips_bootstrap_file(monkeypatch, tmp_path):
    # The DB opens and the admin is seeded + committed (so .bootstrap_password is
    # on disk), but reading must_change_password back fails. Returning here would
    # re-exec with the freshly seeded credential still on disk; strip it first.
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


# ── `unsloth studio run` ─────────────────────────────────────────────


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


def test_run_non_tty_autogenerates_and_proceeds(monkeypatch, tmp_path):
    studio_mod = _studio()
    events = _install_prompt_env(monkeypatch, tmp_path, interactive = False)
    _seed_auth(studio_mod)

    result = _invoke_run(monkeypatch, events, _BASE + ["--secure"])

    kinds = [kind for kind, _ in events]
    assert kinds == ["exec"], events
    combined = (result.output or "") + (getattr(result, "stderr", "") or "")
    assert "auto-generated" in combined.lower()
    assert _auth_state(studio_mod)["must_change_password"] == 0


def test_run_non_tty_deletes_bootstrap_password_file(monkeypatch, tmp_path):
    # Auto-generation on the `unsloth studio run` re-exec path commits a new
    # password and deletes the seeded credential file before re-exec, so a fresh
    # child of ANY version reads None from disk. The launch still proceeds.
    studio_mod = _studio()
    events = _install_prompt_env(monkeypatch, tmp_path, interactive = False)
    _seed_auth(studio_mod)
    bootstrap_file = tmp_path / "auth" / studio_mod.BOOTSTRAP_PASSWORD_FILE
    assert bootstrap_file.exists()

    _invoke_run(monkeypatch, events, _BASE + ["--secure"])

    assert not bootstrap_file.exists()
    kinds = [kind for kind, _ in events]
    assert kinds == ["exec"], events
    assert _auth_state(studio_mod)["must_change_password"] == 0


def test_run_missing_frontend_exits_before_stripping_bootstrap(monkeypatch, tmp_path):
    # Regression (item B / reviewer finding 4): `unsloth studio run` serves the
    # same Unsloth UI and strips the seeded password on a headless public launch,
    # so a missing frontend dist must abort BEFORE the strip -- the same lockout
    # guard as `unsloth studio`, not just `studio run`'s model-load residual.
    import typer as _typer

    studio_mod = _studio()
    events = _install_prompt_env(monkeypatch, tmp_path, interactive = False)
    _seed_auth(studio_mod)
    bootstrap_file = tmp_path / "auth" / studio_mod.BOOTSTRAP_PASSWORD_FILE
    assert bootstrap_file.exists()

    _install_run_reexec(monkeypatch, events)
    monkeypatch.setattr(studio_mod, "_find_frontend_dist", lambda: None)  # no built dist

    app = _typer.Typer()
    app.command(
        context_settings = {"allow_extra_args": True, "ignore_unknown_options": True},
    )(studio_mod.run)
    result = CliRunner().invoke(app, _BASE + ["--secure"], catch_exceptions = True)

    assert result.exit_code == 1, result.output
    assert bootstrap_file.exists()  # not stripped
    assert _auth_state(studio_mod)["must_change_password"] == 1
    assert events == [], events  # no strip, no exec
    combined = (result.output or "") + (getattr(result, "stderr", "") or "")
    assert "frontend is not built" in combined.lower()


def test_run_in_venv_missing_frontend_exits_before_stripping_bootstrap(monkeypatch, tmp_path):
    # Regression (Codex 3571888563): the in-venv `studio run` path validated only
    # the backend, so a headless public launch would strip the seeded password
    # before run_server() aborted on a missing dist. Validate the frontend first.
    import typer as _typer

    studio_mod = _studio()
    events = _install_prompt_env(monkeypatch, tmp_path, interactive = False)
    _seed_auth(studio_mod)
    bootstrap_file = tmp_path / "auth" / studio_mod.BOOTSTRAP_PASSWORD_FILE
    assert bootstrap_file.exists()

    monkeypatch.setattr(sys, "prefix", str(tmp_path / "unsloth_studio"))  # in-venv
    monkeypatch.setattr(studio_mod, "_find_frontend_dist", lambda: None)  # no built dist
    monkeypatch.setattr(studio_mod, "_load_run_module", lambda: None)  # backend fine

    app = _typer.Typer()
    app.command(
        context_settings = {"allow_extra_args": True, "ignore_unknown_options": True},
    )(studio_mod.run)
    result = CliRunner().invoke(app, _BASE + ["--secure"], catch_exceptions = True)

    assert result.exit_code == 1, result.output
    assert bootstrap_file.exists()  # not stripped
    assert _auth_state(studio_mod)["must_change_password"] == 1
    assert events == [], events
    combined = (result.output or "") + (getattr(result, "stderr", "") or "")
    assert "frontend is not built" in combined.lower()


def test_run_reexec_forwards_resolved_frontend_on_public_launch(monkeypatch, tmp_path):
    # Regression (Codex 3571888570): the run re-exec discarded the dist resolved
    # by the pre-strip check and only forwarded a user-supplied --frontend. On a
    # public launch it must forward the resolved dist so a shadowed child that
    # cannot self-resolve one still serves it (no post-strip lockout).
    studio_mod = _studio()
    events = _install_prompt_env(monkeypatch, tmp_path, interactive = True)
    _seed_auth(studio_mod, must_change = False)  # gate is a no-op -> straight to re-exec

    # _install_run_reexec resolves _find_frontend_dist -> /fake/studio/frontend/dist.
    _invoke_run(monkeypatch, events, _BASE + ["--secure"])  # no user --frontend

    exec_argv = [argv for kind, argv in events if kind == "exec"][0]
    assert "--frontend" in exec_argv, exec_argv
    # str(Path(...)), not the literal: Windows renders it with backslashes.
    expected_dist = str(Path("/fake/studio/frontend/dist"))
    assert exec_argv[exec_argv.index("--frontend") + 1] == expected_dist, exec_argv


def test_run_non_tty_persists_seeded_admin_on_fresh_home(monkeypatch, tmp_path):
    # Fresh STUDIO_HOME on the `run` re-exec path: the gate seeds the admin, then
    # auto-generation commits a strong password over it (must_change=0) before
    # re-exec, so no old console-script child ever serves a default credential.
    studio_mod = _studio()
    events = _install_prompt_env(monkeypatch, tmp_path, interactive = False)

    _invoke_run(monkeypatch, events, _BASE + ["--secure"])

    state = _auth_state(studio_mod)
    assert state["must_change_password"] == 0
    assert not (tmp_path / "auth" / studio_mod.BOOTSTRAP_PASSWORD_FILE).exists()
    kinds = [kind for kind, _ in events]
    assert kinds == ["exec"], events


def test_run_non_tty_api_only_autogenerates(monkeypatch, tmp_path):
    # api-only headless public serving used to fail closed for lack of a deadline;
    # now a strong password is auto-generated instead, so the launch proceeds.
    studio_mod = _studio()
    events = _install_prompt_env(monkeypatch, tmp_path, interactive = False)
    _seed_auth(studio_mod)

    result = _invoke_run(monkeypatch, events, _BASE + ["--secure", "--api-only"])

    kinds = [kind for kind, _ in events]
    assert kinds == ["exec"], events
    assert result.exit_code == 0, result.output
    combined = (result.output or "") + (getattr(result, "stderr", "") or "")
    assert "auto-generated" in combined.lower()
    assert _auth_state(studio_mod)["must_change_password"] == 0


def test_studio_default_non_tty_disabled_deadline_autogenerates(monkeypatch, tmp_path):
    # UNSLOTH_STUDIO_BOOTSTRAP_TIMEOUT=0 disables the deadline; the auto-generated
    # password is the safeguard now, so the launch still proceeds.
    studio_mod = _studio()
    events = _install_prompt_env(monkeypatch, tmp_path, interactive = False)
    _seed_auth(studio_mod)
    monkeypatch.setenv("UNSLOTH_STUDIO_BOOTSTRAP_TIMEOUT", "0")

    result = _invoke_studio_default(monkeypatch, events, ["--secure"])

    kinds = [kind for kind, _ in events]
    assert kinds == ["exec"], events
    combined = (result.output or "") + (getattr(result, "stderr", "") or "")
    assert "auto-generated" in combined.lower()
    assert _auth_state(studio_mod)["must_change_password"] == 0


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
    # The DB survives, so a running server keeps its admin row and the new password.
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
    # The CLI now writes while the server does; without a busy_timeout this fails.
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
    # Deleting auth.db used to drop these implicitly.
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
    # must_change_password stays 0 on purpose: at 1 a running server injects its
    # startup-cached (now wrong) bootstrap password into the login page.
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
    # _connect_auth_db creates auth/ before it opens SQLite, so a read-only Unsloth
    # home raises OSError, not sqlite3.Error.
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
    # Deleting a corrupt DB here would revive the bug: a running server would be
    # left with no admin row, rejecting the correct password until restarted.
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
    # After a CLI/interactive password change the seeded .bootstrap_password is
    # deleted. If it cannot be unlinked but is still writable (locked file /
    # read-only dir), it must be TRUNCATED so its stale plaintext cannot be
    # re-seeded by generate_bootstrap_password() if auth.db is ever recreated. The
    # change is already committed, so it must NOT roll back.
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

    # The change committed (must_change cleared) AND the locked file is truncated.
    assert _auth_state(studio_mod)["must_change_password"] == 0
    assert bootstrap_file.exists()
    assert bootstrap_file.read_text() == ""



def test_cli_update_password_revokes_link_tokens(monkeypatch, tmp_path):
    # Mirror backend storage.update_password: a link token is signed with a key
    # derived from the JWT secret rotated in this transaction, so the CLI password
    # change must delete outstanding link_tokens in the SAME transaction. A
    # leftover row would let a concurrent exchange that read the pre-rotation key
    # still consume its jti and mint a session under the new secret.
    studio_mod = _studio()
    monkeypatch.setattr(studio_mod, "STUDIO_HOME", tmp_path)
    _seed_auth(studio_mod)

    conn = studio_mod._connect_auth_db()
    # The CLI never mints these, but shares the auth.db with the backend that does;
    # _connect_auth_db must create the table so the revoke DELETE never errors.
    conn.execute(
        "INSERT INTO link_tokens (jti, username, expires_at) VALUES (?, ?, ?)",
        ("jti-live", studio_mod.DEFAULT_ADMIN_USERNAME, "2099-01-01T00:00:00"),
    )
    conn.commit()

    studio_mod._cli_update_password(conn, studio_mod.DEFAULT_ADMIN_USERNAME, "fresh-new-pw-123")
    remaining = conn.execute("SELECT COUNT(*) FROM link_tokens").fetchone()[0]
    conn.close()

    assert remaining == 0
    assert _auth_state(studio_mod)["must_change_password"] == 0


def test_cli_update_password_compare_and_set_guard(monkeypatch, tmp_path):
    # Mirror backend storage.update_password: the auto-generated launch credential
    # is committed only while must_change_password is still 1. A user finishing
    # /change-password in a Studio tab between the must_change read and this write
    # must not be silently overwritten, and nothing else may be revoked when the
    # guard rejects the update.
    studio_mod = _studio()
    monkeypatch.setattr(studio_mod, "STUDIO_HOME", tmp_path)
    _seed_auth(studio_mod)

    conn = studio_mod._connect_auth_db()
    admin = studio_mod.DEFAULT_ADMIN_USERNAME
    assert (
        studio_mod._cli_update_password(
            conn, admin, "first-generated-pw-1", require_must_change = True
        )
        is True
    )
    before = conn.execute(
        "SELECT password_hash, jwt_secret FROM auth_user WHERE username = ?", (admin,)
    ).fetchone()
    # Two credentials a rejected write must NOT destroy. api_keys is the one that
    # matters most: _cli_update_password deletes it with no WHERE clause, so a
    # revocation that ran past a failed compare-and-set would wipe every key for
    # every user while leaving the password exactly as the winning writer set it.
    conn.execute(
        "INSERT INTO refresh_tokens (token_hash, username, expires_at) VALUES (?, ?, ?)",
        ("refresh-hash-guarded", admin, "2099-01-01T00:00:00"),
    )
    conn.execute(
        "INSERT INTO api_keys (username, key_prefix, key_hash, name, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (admin, "sk-unsloth-", "api-key-hash-guarded", "guarded", "2026-01-01T00:00:00"),
    )
    conn.execute(
        "INSERT INTO link_tokens (jti, username, expires_at) VALUES (?, ?, ?)",
        ("jti-guarded", admin, "2099-01-01T00:00:00"),
    )
    conn.commit()

    # must_change is 0 now, so the guarded write is refused and changes nothing.
    assert (
        studio_mod._cli_update_password(
            conn, admin, "second-generated-pw-2", require_must_change = True
        )
        is False
    )
    after = conn.execute(
        "SELECT password_hash, jwt_secret FROM auth_user WHERE username = ?", (admin,)
    ).fetchone()
    remaining_refresh = conn.execute("SELECT COUNT(*) FROM refresh_tokens").fetchone()[0]
    remaining_keys = conn.execute("SELECT COUNT(*) FROM api_keys").fetchone()[0]
    remaining_links = conn.execute("SELECT COUNT(*) FROM link_tokens").fetchone()[0]

    # An explicit (unguarded) change still applies.
    assert studio_mod._cli_update_password(conn, admin, "explicit-change-pw-3") is True
    conn.close()

    assert tuple(after) == tuple(before)  # password and JWT secret untouched
    # No collateral revocation on a rejected write.
    assert remaining_refresh == 1
    assert remaining_keys == 1
    assert remaining_links == 1


def test_connect_auth_db_creates_private_files(monkeypatch, tmp_path):
    # Fresh install: the CLI gate writes the password hash + JWT secret before
    # the backend ever runs, so this path must apply the same 0700/0600 modes
    # as backend storage.get_connection (sqlite3.connect creates 0644 files
    # under a 022 umask).
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
    # Shared by .bootstrap_password and .desktop_secret; every reader strips.
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


# ── non-interactive --password / UNSLOTH_STUDIO_PASSWORD / stdin ──────


def _exec_argv(events):
    return next(argv for kind, argv in events if kind == "exec")


def test_studio_default_password_sets_initial_no_prompt_no_forward(monkeypatch, tmp_path):
    studio_mod = _studio()
    events = _install_prompt_env(monkeypatch, tmp_path, interactive = True)
    before = _seed_auth(studio_mod)

    _invoke_studio_default(monkeypatch, events, ["--secure", "--password", "cli-supplied-pw12"])

    # No interactive prompt: --password applied in the parent, so the gate no-ops.
    assert [kind for kind, _ in events] == ["exec"], events
    after = _auth_state(studio_mod)
    assert after["must_change_password"] == 0
    assert after["password_hash"] != before["password_hash"]
    assert after["jwt_secret"] != before["jwt_secret"]
    assert after["n_refresh"] == 0
    assert not (tmp_path / "auth" / studio_mod.BOOTSTRAP_PASSWORD_FILE).exists()
    # The secret never crosses to the child argv.
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
    # Env var stripped so a re-exec'd child cannot re-read it.
    assert "UNSLOTH_STUDIO_PASSWORD" not in os.environ


def test_studio_default_password_via_stdin(monkeypatch, tmp_path):
    # `--password -` reads one line from stdin. CliRunner owns stdin during
    # invoke, so feed it via input= rather than patching sys.stdin.
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
    assert [kind for kind, _ in events] == []  # never reached the gate / re-exec
    assert _auth_state(studio_mod)["must_change_password"] == 1  # unchanged


def test_studio_default_password_must_differ_fails_closed(monkeypatch, tmp_path):
    studio_mod = _studio()
    events = _install_prompt_env(monkeypatch, tmp_path, interactive = True)
    _seed_auth(studio_mod)
    bootstrap_pw = (tmp_path / "auth" / studio_mod.BOOTSTRAP_PASSWORD_FILE).read_text().strip()

    result = _invoke_studio_default(monkeypatch, events, ["--secure", "--password", bootstrap_pw])

    assert result.exit_code == 1
    assert _auth_state(studio_mod)["must_change_password"] == 1  # unchanged


def test_studio_default_password_already_set_fails_closed(monkeypatch, tmp_path):
    studio_mod = _studio()
    events = _install_prompt_env(monkeypatch, tmp_path, interactive = True)
    _seed_auth(studio_mod, must_change = False)  # a password is already set

    result = _invoke_studio_default(
        monkeypatch, events, ["--secure", "--password", "another-pw-12345"]
    )

    assert result.exit_code == 1
    assert [kind for kind, _ in events] == []


def test_studio_default_password_before_subcommand_errors(monkeypatch, tmp_path):
    # --password on `unsloth studio` (before a subcommand) is a plain-only option;
    # like --secure/--cloudflare it must error, not be silently dropped.
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
    # The `run` mirror must also strip UNSLOTH_STUDIO_PASSWORD before re-exec so a
    # shadowed child cannot re-read the secret (parity with studio_default).
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
    # The apply is scoped to "any launch", not just --secure/--cloudflare: a raw
    # public wildcard bind (-H 0.0.0.0, no tunnel) must set the initial password
    # before bind and re-exec, with the gate no-op'ing (must_change now 0).
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


class _DyingConsole:
    """A terminal that passes the preflight and then raises on write.

    Models the console going away between _one_time_secret_console_stream()'s
    checks and the post-commit banner (a dropped SSH session leaves an orphaned
    pty whose writes fail with OSError EIO).
    """

    closed = False

    def __init__(self):
        self.writes = 0

    def write(self, *_a, **_k):
        self.writes += 1
        raise OSError(5, "Input/output error")

    def flush(self, *_a, **_k):
        pass

    def isatty(self):
        return True


class _LiveConsole:
    closed = False

    def __init__(self):
        self.text = ""

    def write(self, data):
        self.text += data
        return len(data)

    def flush(self, *_a, **_k):
        pass

    def isatty(self):
        return True


def test_credential_delivery_retries_the_other_console(monkeypatch):
    # Regression (Codex 3651035060, P2): _echo_auto_generated_credentials runs
    # AFTER _cli_update_password committed the generated password and removed the
    # seeded bootstrap file, so a raise there used to abort the launch with a live
    # password nobody had ever seen. Retry the other terminal instead.
    studio_mod = _studio()
    dying, alive = _DyingConsole(), _LiveConsole()
    monkeypatch.setattr(sys, "stderr", dying)
    monkeypatch.setattr(sys, "stdout", alive)

    delivered = studio_mod._deliver_auto_generated_credentials(
        "unsloth", "Cli-Retry-Pw-1", out = dying
    )

    assert delivered is True
    assert dying.writes >= 1
    assert "Cli-Retry-Pw-1" in alive.text


def test_credential_delivery_reports_failure_without_a_usable_console(monkeypatch):
    # No console accepts the write, and a redirected non-tty stream must NOT be
    # used as a fallback (it would persist the plaintext, CWE-532). The caller
    # then exits non-zero with a secret-free message instead of a traceback.
    studio_mod = _studio()
    dying = _DyingConsole()

    class _Redirected(_LiveConsole):
        def isatty(self):
            return False

    redirected = _Redirected()
    monkeypatch.setattr(sys, "stderr", dying)
    monkeypatch.setattr(sys, "stdout", redirected)

    assert (
        studio_mod._deliver_auto_generated_credentials("unsloth", "Cli-Lost-Pw-2", out = dying)
        is False
    )
    assert redirected.text == ""


def test_delivery_failure_message_carries_no_credential(monkeypatch):
    # The advisory printed after an undeliverable credential must name the
    # recovery command and nothing else; it can reach a log the password may not.
    studio_mod = _studio()
    printed: list[str] = []
    monkeypatch.setattr(studio_mod.typer, "echo", lambda msg, **k: printed.append(str(msg)))

    studio_mod._log_secret_free_delivery_failure()

    assert printed and "reset-password" in printed[0]
    assert "Password:" not in printed[0]
