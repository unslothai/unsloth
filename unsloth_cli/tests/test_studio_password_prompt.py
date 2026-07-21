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
def test_should_prompt_password_change_matrix(
    cloudflare, host, secure, api_only, expected
):
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
    monkeypatch.setattr(
        studio_mod, "_tunnel_binary_confirmed_unavailable", lambda: False
    )

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
    monkeypatch.setattr(
        studio_mod, "_studio_venv_python", lambda: fake_venv / "bin" / "python"
    )
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
    monkeypatch.setattr(
        studio_mod, "_studio_venv_python", lambda: fake_venv / "bin" / "python"
    )
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


def test_studio_default_reexec_outer_runpy_keeps_bootstrap_for_local_recovery(
    monkeypatch, tmp_path
):
    # Regression (Codex 3572165931): when the re-exec target is THIS install's own
    # run.py, the child's pre-bind gate sets suppress_bootstrap_injection and never
    # serves the seeded credential publicly, so the parent strip is unnecessary.
    # Skipping it means a --secure launch whose tunnel later fails to connect does
    # not lock the user out, and .bootstrap_password stays for local recovery.
    import typer as _typer

    studio_mod = _studio()
    events = _install_prompt_env(monkeypatch, tmp_path, interactive = False)
    _seed_auth(studio_mod)
    bootstrap_file = tmp_path / "auth" / studio_mod.BOOTSTRAP_PASSWORD_FILE
    assert bootstrap_file.exists()

    _install_studio_default_reexec(monkeypatch, events)
    # Re-exec target IS this install's outer run.py -> child self-suppresses.
    outer_run_py = studio_mod._PACKAGE_ROOT / "studio" / "backend" / "run.py"
    monkeypatch.setattr(studio_mod, "_find_run_py", lambda: outer_run_py)

    app = _typer.Typer()
    app.command()(studio_mod.studio_default)
    result = CliRunner().invoke(app, ["--secure"], catch_exceptions = True)

    # Strip skipped: file preserved, must_change still set, launch still re-execs.
    assert bootstrap_file.exists(), result.output
    assert _auth_state(studio_mod)["must_change_password"] == 1
    assert "exec" in [k for k, _ in events], events


def test_studio_default_non_tty_persists_seeded_admin_on_fresh_home(
    monkeypatch, tmp_path
):
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


def test_studio_default_non_tty_fails_closed_when_bootstrap_removal_fails(
    monkeypatch, tmp_path
):
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
    monkeypatch.setattr(
        studio_mod, "_connect_auth_db", lambda: _FailingCommitConn(real_connect())
    )

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


def test_studio_default_missing_venv_exits_before_stripping_bootstrap(
    monkeypatch, tmp_path
):
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


def test_studio_default_missing_frontend_exits_before_stripping_bootstrap(
    monkeypatch, tmp_path
):
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
    monkeypatch.setattr(
        studio_mod, "_studio_venv_python", lambda: fake_venv / "bin" / "python"
    )
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


def test_studio_default_bad_frontend_path_exits_before_stripping_bootstrap(
    monkeypatch, tmp_path
):
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
    monkeypatch.setattr(
        studio_mod, "_studio_venv_python", lambda: fake_venv / "bin" / "python"
    )
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


def test_studio_default_missing_frontend_loopback_cloudflare_still_launches(
    monkeypatch, tmp_path
):
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
    monkeypatch.setattr(
        studio_mod, "_studio_venv_python", lambda: fake_venv / "bin" / "python"
    )
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

    # Pretend we are already inside the studio venv, with a broken backend.
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "unsloth_studio"))

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
    monkeypatch.setattr(
        studio_mod, "_find_frontend_dist", lambda: None
    )  # no built dist
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


def test_studio_default_secure_tunnel_unavailable_preserves_bootstrap(
    monkeypatch, tmp_path
):
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
    monkeypatch.setattr(
        studio_mod, "_tunnel_binary_confirmed_unavailable", lambda: True
    )

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
    monkeypatch.setattr(
        studio_mod, "_tunnel_binary_confirmed_unavailable", lambda: True
    )

    app = _typer.Typer()
    app.command()(studio_mod.studio_default)
    result = CliRunner().invoke(
        app, ["-H", "0.0.0.0", "--cloudflare"], catch_exceptions = True
    )

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
    monkeypatch.setattr(
        studio_mod, "_connect_auth_db", lambda: _FailingSelectConn(real_connect())
    )

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
    monkeypatch.setattr(
        studio_mod, "_find_frontend_dist", lambda: None
    )  # no built dist

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


def test_run_in_venv_missing_frontend_exits_before_stripping_bootstrap(
    monkeypatch, tmp_path
):
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
    monkeypatch.setattr(
        studio_mod, "_find_frontend_dist", lambda: None
    )  # no built dist
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
    assert (
        exec_argv[exec_argv.index("--frontend") + 1] == "/fake/studio/frontend/dist"
    ), exec_argv


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


def test_reset_password_truncates_locked_bootstrap_after_db_delete(
    monkeypatch, tmp_path
):
    # reset-password deletes auth.db first, then invalidates the seeded credential
    # files. A locked/undeletable .bootstrap_password must be truncated so its
    # stale plaintext cannot be re-seeded (generate_bootstrap_password reuses a
    # non-empty file), while the reset still succeeds.
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


def test_cli_update_password_truncates_locked_bootstrap_after_change(
    monkeypatch, tmp_path
):
    # After a CLI/interactive password change the seeded .bootstrap_password is
    # deleted. If it cannot be unlinked but is still writable (locked file /
    # read-only dir), it must be TRUNCATED so its stale plaintext cannot be
    # re-seeded by generate_bootstrap_password() after a later reset-password
    # deletes auth.db. The change is already committed, so it must NOT roll back.
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
    studio_mod._cli_update_password(
        conn, studio_mod.DEFAULT_ADMIN_USERNAME, "fresh-new-pw-123"
    )
    conn.close()

    # The change committed (must_change cleared) AND the locked file is truncated.
    assert _auth_state(studio_mod)["must_change_password"] == 0
    assert bootstrap_file.exists()
    assert bootstrap_file.read_text() == ""


def test_reset_password_fails_closed_when_db_cannot_be_deleted(monkeypatch, tmp_path):
    # If auth.db cannot be removed (running Unsloth / Windows lock, read-only dir),
    # reset must abort BEFORE touching the credential files -- deleting them while
    # an un-resettable must_change_password=1 DB survives would lock a
    # forgotten-password reset out with no recovery credential.
    import pathlib

    studio_mod = _studio()
    monkeypatch.setattr(studio_mod, "STUDIO_HOME", tmp_path)
    _seed_auth(studio_mod)
    auth_dir = tmp_path / "auth"
    bootstrap_file = auth_dir / studio_mod.BOOTSTRAP_PASSWORD_FILE
    db_file = auth_dir / "auth.db"
    assert bootstrap_file.exists() and db_file.exists()

    _real_unlink = pathlib.Path.unlink

    def _boom_unlink(self, *a, **k):
        if self.name == "auth.db":
            raise OSError("database is locked")
        return _real_unlink(self, *a, **k)

    monkeypatch.setattr(pathlib.Path, "unlink", _boom_unlink)

    import typer as _typer

    app = _typer.Typer()
    app.command()(studio_mod.reset_password)
    result = CliRunner().invoke(app, [], catch_exceptions = True)

    assert result.exit_code == 1, result.output
    # DB still there; credential files untouched (no lockout, no half-done reset).
    assert db_file.exists()
    assert bootstrap_file.exists()
    assert bootstrap_file.read_text().strip()
    combined = (result.output or "") + (getattr(result, "stderr", "") or "")
    assert "could not delete the auth database" in combined.lower()


def test_reset_password_fails_closed_when_credential_cannot_be_invalidated(
    monkeypatch, tmp_path
):
    # If a seeded credential file can be neither unlinked nor truncated, reset must
    # fail closed: auth.db is already gone, so a surviving plaintext would be
    # re-seeded and re-validate the revoked password.
    import pathlib

    studio_mod = _studio()
    monkeypatch.setattr(studio_mod, "STUDIO_HOME", tmp_path)
    _seed_auth(studio_mod)
    auth_dir = tmp_path / "auth"
    bootstrap_file = auth_dir / studio_mod.BOOTSTRAP_PASSWORD_FILE
    db_file = auth_dir / "auth.db"
    assert bootstrap_file.exists() and db_file.exists()

    _real_unlink = pathlib.Path.unlink
    _real_write_text = pathlib.Path.write_text

    def _boom_unlink(self, *a, **k):
        if self.name == studio_mod.BOOTSTRAP_PASSWORD_FILE:
            raise OSError("locked")
        return _real_unlink(self, *a, **k)

    def _boom_write_text(self, *a, **k):
        if self.name == studio_mod.BOOTSTRAP_PASSWORD_FILE:
            raise OSError("read-only")
        return _real_write_text(self, *a, **k)

    monkeypatch.setattr(pathlib.Path, "unlink", _boom_unlink)
    monkeypatch.setattr(pathlib.Path, "write_text", _boom_write_text)

    import typer as _typer

    app = _typer.Typer()
    app.command()(studio_mod.reset_password)
    result = CliRunner().invoke(app, [], catch_exceptions = True)

    assert result.exit_code == 1, result.output
    # auth.db was deleted first; the un-invalidatable file is reported for manual removal.
    assert not db_file.exists()
    combined = (result.output or "") + (getattr(result, "stderr", "") or "")
    assert "delete it manually" in combined.lower()


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


# ── non-interactive --password / UNSLOTH_STUDIO_PASSWORD / stdin ──────


def _exec_argv(events):
    return next(argv for kind, argv in events if kind == "exec")


def test_studio_default_password_sets_initial_no_prompt_no_forward(
    monkeypatch, tmp_path
):
    studio_mod = _studio()
    events = _install_prompt_env(monkeypatch, tmp_path, interactive = True)
    before = _seed_auth(studio_mod)

    _invoke_studio_default(
        monkeypatch, events, ["--secure", "--password", "cli-supplied-pw12"]
    )

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

    result = _invoke_studio_default(
        monkeypatch, events, ["--secure", "--password", "short"]
    )

    assert result.exit_code == 1
    assert [kind for kind, _ in events] == []  # never reached the gate / re-exec
    assert _auth_state(studio_mod)["must_change_password"] == 1  # unchanged


def test_studio_default_password_must_differ_fails_closed(monkeypatch, tmp_path):
    studio_mod = _studio()
    events = _install_prompt_env(monkeypatch, tmp_path, interactive = True)
    _seed_auth(studio_mod)
    bootstrap_pw = (tmp_path / "auth" / studio_mod.BOOTSTRAP_PASSWORD_FILE).read_text()

    result = _invoke_studio_default(
        monkeypatch, events, ["--secure", "--password", bootstrap_pw]
    )

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
    result = CliRunner().invoke(
        app, ["studio", "--password", "x", "run", "--model", "X"]
    )
    assert result.exit_code == 2
    combined = (result.output or "") + (getattr(result, "stderr", "") or "")
    assert "--password" in combined


def test_run_password_sets_initial_no_prompt_no_forward(monkeypatch, tmp_path):
    studio_mod = _studio()
    events = _install_prompt_env(monkeypatch, tmp_path, interactive = True)
    before = _seed_auth(studio_mod)

    _invoke_run(
        monkeypatch, events, _BASE + ["--secure", "--password", "cli-supplied-pw12"]
    )

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


def test_studio_default_password_applies_on_headless_wildcard_no_tunnel(
    monkeypatch, tmp_path
):
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


def test_reset_password_then_password_roundtrip(monkeypatch, tmp_path):
    # After reset-password wipes the DB, the next start re-seeds a fresh admin
    # that again requires a change, so --password can set a new initial password.
    import typer

    studio_mod = _studio()
    monkeypatch.setattr(studio_mod, "STUDIO_HOME", tmp_path)
    _seed_auth(studio_mod)
    conn = studio_mod._connect_auth_db()
    studio_mod._cli_update_password(
        conn, studio_mod.DEFAULT_ADMIN_USERNAME, "first-password-1"
    )
    conn.close()
    assert _auth_state(studio_mod)["must_change_password"] == 0

    # reset-password deletes the auth DB + seeded credential files.
    try:
        studio_mod.reset_password()
    except typer.Exit:
        pass
    assert not (tmp_path / "auth" / "auth.db").exists()

    # A restart re-seeds (ensure_default_admin, must_change=1); --password sets anew.
    events = _install_prompt_env(monkeypatch, tmp_path, interactive = True)
    _invoke_studio_default(
        monkeypatch, events, ["--secure", "--password", "second-password-2"]
    )
    assert [kind for kind, _ in events] == ["exec"], events
    assert _auth_state(studio_mod)["must_change_password"] == 0
