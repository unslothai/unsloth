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
    monkeypatch.setattr(studio_mod, "_find_frontend_dist", lambda: Path("/fake/studio/frontend/dist"))
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
    monkeypatch.setattr(studio_mod, "_find_frontend_dist", lambda: Path("/fake/studio/frontend/dist"))
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
    bootstrap_pw = (tmp_path / "auth" / studio_mod.BOOTSTRAP_PASSWORD_FILE).read_text()

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
    # Mixed-version safety: a headless public launch must delete the seeded
    # plaintext credential before re-exec so a fresh child of ANY version reads
    # None from disk and never injects it into the public HTML. The launch still
    # proceeds (re-exec captured), and the DB flag stays set so the login page
    # still forces a change and the bootstrap shutdown timer still arms.
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


def test_studio_default_non_tty_persists_seeded_admin_on_fresh_home(monkeypatch, tmp_path):
    # Fresh STUDIO_HOME (no pre-seed): the gate's own _ensure_cli_default_admin
    # does the INSERT. It must COMMIT that seed before re-exec, or conn.close()
    # rolls it back and an OLD child would find no admin, regenerate a fresh
    # bootstrap password + file, and inject THAT -- defeating the file deletion.
    studio_mod = _studio()
    events = _install_prompt_env(monkeypatch, tmp_path, interactive = False)
    # Deliberately NO _seed_auth(): exercise the gate seeding a fresh DB itself.

    _invoke_studio_default(monkeypatch, events, ["--secure"])

    # The seeded admin persists (committed) so an old child sees it and does not
    # regenerate; the bootstrap file stays deleted; the launch still re-execs.
    state = _auth_state(studio_mod)
    assert state["must_change_password"] == 1
    assert not (tmp_path / "auth" / studio_mod.BOOTSTRAP_PASSWORD_FILE).exists()
    kinds = [kind for kind, _ in events]
    assert kinds == ["exec"], events


def test_studio_default_non_tty_fails_closed_when_bootstrap_removal_fails(monkeypatch, tmp_path):
    # Removing .bootstrap_password IS the protection on this path. If unlink
    # fails (locked file / read-only auth dir) the credential is still on disk
    # for an old child to inject, so the launch must fail closed, not publish.
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
    # The file remains (removal failed) and the DB flag is untouched.
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
    monkeypatch.setattr(studio_mod, "_find_frontend_dist", lambda: Path("/fake/studio/frontend/dist"))
    empty_dir = tmp_path / "empty_frontend"
    empty_dir.mkdir()

    app = _typer.Typer()
    app.command()(studio_mod.studio_default)
    result = CliRunner().invoke(app, ["--secure", "--frontend", str(empty_dir)], catch_exceptions = True)

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
    # Same mixed-version safety for the `unsloth studio run` re-exec path (which
    # cannot fail-close an old child via a CLI flag): the seeded credential file
    # is deleted before re-exec, the launch still proceeds, and the DB flag holds.
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
    # Regression (item B / reviewer finding 4): `unsloth studio run` serves the
    # same Studio UI and strips the seeded password on a headless public launch,
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


def test_run_non_tty_persists_seeded_admin_on_fresh_home(monkeypatch, tmp_path):
    # Fresh STUDIO_HOME on the `run` re-exec path: the seeded admin must be
    # committed before re-exec so an old console-script child does not regenerate
    # and inject a fresh bootstrap credential.
    studio_mod = _studio()
    events = _install_prompt_env(monkeypatch, tmp_path, interactive = False)

    _invoke_run(monkeypatch, events, _BASE + ["--secure"])

    state = _auth_state(studio_mod)
    assert state["must_change_password"] == 1
    assert not (tmp_path / "auth" / studio_mod.BOOTSTRAP_PASSWORD_FILE).exists()
    kinds = [kind for kind, _ in events]
    assert kinds == ["exec"], events


def test_run_non_tty_api_only_fails_closed(monkeypatch, tmp_path):
    # api-only serving never arms the bootstrap shutdown deadline, so a
    # headless public launch with the default password has no safeguard at
    # all: the CLI must refuse rather than promise a shutdown that never comes.
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
    # UNSLOTH_STUDIO_BOOTSTRAP_TIMEOUT=0 disables the deadline; headless +
    # default password + public tunnel then has no protection -> refuse.
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
        (None, True),  # unset -> default 1h
        ("", True),
        ("garbage", True),  # malformed must not remove protection
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


def test_reset_password_invalidates_locked_bootstrap_before_db_delete(monkeypatch, tmp_path):
    # reset-password must invalidate the seeded credential files BEFORE deleting
    # auth.db, and survive a locked/undeletable .bootstrap_password by truncating
    # it. Otherwise the file's stale plaintext outlives auth.db and the next
    # re-seed re-validates the revoked bootstrap password.
    import pathlib

    studio_mod = _studio()
    monkeypatch.setattr(studio_mod, "STUDIO_HOME", tmp_path)
    _seed_auth(studio_mod)
    auth_dir = tmp_path / "auth"
    bootstrap_file = auth_dir / studio_mod.BOOTSTRAP_PASSWORD_FILE
    db_file = auth_dir / "auth.db"
    assert bootstrap_file.exists() and db_file.exists()
    assert bootstrap_file.read_text().strip()

    _real_unlink = pathlib.Path.unlink

    def _boom_unlink(self, *a, **k):
        if self.name == studio_mod.BOOTSTRAP_PASSWORD_FILE:
            raise OSError("locked")
        return _real_unlink(self, *a, **k)

    monkeypatch.setattr(pathlib.Path, "unlink", _boom_unlink)

    import typer as _typer

    app = _typer.Typer()
    app.command()(studio_mod.reset_password)
    result = CliRunner().invoke(app, [], catch_exceptions = True)

    assert result.exit_code == 0, result.output
    assert not db_file.exists()
    # The locked file survives, but truncated -- no reusable plaintext.
    assert bootstrap_file.exists()
    assert bootstrap_file.read_text() == ""


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
