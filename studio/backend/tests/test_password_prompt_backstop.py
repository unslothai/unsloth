# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Pre-tunnel terminal password gate: never publish a public Cloudflare URL
while the seeded default admin password is active. Imports run.py directly,
so run under the Unsloth venv."""

from __future__ import annotations

import io
import os
import re
import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import run  # noqa: E402
from auth import storage as auth_storage  # noqa: E402
from auth import terminal_prompt  # noqa: E402
from auth.terminal_prompt import should_prompt_password_change  # noqa: E402

_GATE_KWARGS = dict(
    host = "127.0.0.1",
    secure = True,
    api_only = False,
    frontend_served = True,
)


# ── pure decision matrix ─────────────────────────────────────────────


@pytest.mark.parametrize(
    "tunnel_will_start,requires_change,stdin_isatty,stderr_isatty,expected",
    [
        (True, True, True, True, True),
        # Any missing precondition suppresses the prompt.
        (False, True, True, True, False),
        (True, False, True, True, False),
        (True, True, False, True, False),
        (True, True, True, False, False),
        (False, False, False, False, False),
    ],
)
def test_should_prompt_matrix(
    tunnel_will_start, requires_change, stdin_isatty, stderr_isatty, expected
):
    assert (
        should_prompt_password_change(
            tunnel_will_start = tunnel_will_start,
            requires_change = requires_change,
            stdin_isatty = stdin_isatty,
            stderr_isatty = stderr_isatty,
        )
        is expected
    )


# ── _terminal_password_gate unit tests ───────────────────────────────


class _Stream(io.StringIO):
    def __init__(self, isatty: bool):
        super().__init__()
        self._isatty = isatty

    def isatty(self) -> bool:
        return self._isatty


class _BrokenStream(io.StringIO):
    """Service-wrapper stand-in whose isatty() raises (closed stdin)."""

    def isatty(self) -> bool:
        raise ValueError("I/O operation on closed file")


def _patch_streams(monkeypatch, *, tty: bool) -> _Stream:
    stderr = _Stream(isatty = tty)
    monkeypatch.setattr(sys, "stdin", _Stream(isatty = tty))
    monkeypatch.setattr(sys, "stderr", stderr)
    return stderr


def _patch_seeded_admin(monkeypatch, *, requires_change: bool) -> None:
    # The gate seeds the admin row itself (it can run before lifespan startup);
    # tests fake both the seeding no-op and the flag.
    monkeypatch.setattr(auth_storage, "ensure_default_admin", lambda: False)
    monkeypatch.setattr(auth_storage, "requires_password_change", lambda u: requires_change)


def test_gate_skips_when_tunnel_off(monkeypatch):
    # Short-circuits before touching auth storage at all.
    def _boom(*a, **k):
        raise AssertionError("storage must not be consulted when the tunnel is off")

    monkeypatch.setattr(auth_storage, "requires_password_change", _boom)
    monkeypatch.setattr(auth_storage, "ensure_default_admin", _boom)
    assert run._terminal_password_gate(tunnel_will_start = False, **_GATE_KWARGS) == (True, False)


def test_gate_skips_when_password_already_changed(monkeypatch):
    _patch_streams(monkeypatch, tty = True)
    _patch_seeded_admin(monkeypatch, requires_change = False)
    monkeypatch.setattr(
        terminal_prompt,
        "prompt_for_password_change",
        lambda **k: pytest.fail("prompt must not run when no change is required"),
    )
    assert run._terminal_password_gate(tunnel_will_start = True, **_GATE_KWARGS) == (True, False)


def test_gate_warns_and_proceeds_without_tty_when_deadline_arms(monkeypatch):
    stderr = _patch_streams(monkeypatch, tty = False)
    _patch_seeded_admin(monkeypatch, requires_change = True)
    monkeypatch.delenv("UNSLOTH_STUDIO_BOOTSTRAP_TIMEOUT", raising = False)
    monkeypatch.setattr(
        terminal_prompt,
        "prompt_for_password_change",
        lambda **k: pytest.fail("prompt must not run without a tty"),
    )
    # Proceeds, but the public HTML must not auto-fill the default credential.
    assert run._terminal_password_gate(tunnel_will_start = True, **_GATE_KWARGS) == (True, True)
    out = stderr.getvalue()
    assert "default admin password is still active" in out
    assert "UNSLOTH_STUDIO_BOOTSTRAP_TIMEOUT" in out
    # The seeded file may already be gone (the CLI parent deletes it before
    # re-exec), so the warning must point at the reset-password recovery path
    # instead of promising a file to read.
    assert "reset-password" in out
    assert ".bootstrap_password" not in out


def test_gate_fails_closed_without_tty_when_deadline_cannot_arm(monkeypatch):
    # api-only launches never arm the bootstrap deadline, so a headless public
    # launch with the default password has NO safeguard: refuse to start.
    stderr = _patch_streams(monkeypatch, tty = False)
    _patch_seeded_admin(monkeypatch, requires_change = True)
    monkeypatch.delenv("UNSLOTH_STUDIO_BOOTSTRAP_TIMEOUT", raising = False)
    kwargs = dict(_GATE_KWARGS)
    kwargs["api_only"] = True
    kwargs["frontend_served"] = False
    assert run._terminal_password_gate(tunnel_will_start = True, **kwargs) == (False, False)
    assert "Refusing to publish" in stderr.getvalue()


def test_gate_fails_closed_without_tty_when_deadline_disabled(monkeypatch):
    stderr = _patch_streams(monkeypatch, tty = False)
    _patch_seeded_admin(monkeypatch, requires_change = True)
    monkeypatch.setenv("UNSLOTH_STUDIO_BOOTSTRAP_TIMEOUT", "0")
    assert run._terminal_password_gate(tunnel_will_start = True, **_GATE_KWARGS) == (False, False)
    assert "Refusing to publish" in stderr.getvalue()


def test_gate_treats_broken_streams_as_non_interactive(monkeypatch):
    # A closed/None stdin must take the headless path, not blow up.
    stderr = _Stream(isatty = False)
    monkeypatch.setattr(sys, "stdin", _BrokenStream())
    monkeypatch.setattr(sys, "stderr", stderr)
    _patch_seeded_admin(monkeypatch, requires_change = True)
    monkeypatch.delenv("UNSLOTH_STUDIO_BOOTSTRAP_TIMEOUT", raising = False)
    assert run._terminal_password_gate(tunnel_will_start = True, **_GATE_KWARGS) == (True, True)


def test_gate_refusal_fails_closed(monkeypatch):
    _patch_streams(monkeypatch, tty = True)
    _patch_seeded_admin(monkeypatch, requires_change = True)
    monkeypatch.setattr(terminal_prompt, "prompt_for_password_change", lambda **k: False)
    assert run._terminal_password_gate(tunnel_will_start = True, **_GATE_KWARGS) == (False, False)


def test_gate_success_applies_route_equivalent_change(monkeypatch):
    _patch_streams(monkeypatch, tty = True)
    calls = []
    _patch_seeded_admin(monkeypatch, requires_change = True)
    monkeypatch.setattr(
        auth_storage,
        "get_user_and_secret",
        lambda u: ("salt", "hash", "jwt", True),
    )
    monkeypatch.setattr(
        auth_storage,
        "update_password",
        lambda u, p, **kw: calls.append(("update", u, p, kw)),
    )

    def _fake_prompt(*, min_length, is_current_password, apply_change, out, **_kw):
        # The gate wires the policy constant and route-equivalent apply hook.
        assert min_length == auth_storage.MIN_PASSWORD_LENGTH
        # Wired to the real hash comparison: a wrong guess is rejected.
        assert is_current_password("wrong-guess") is False
        apply_change("brand-new-password")
        return True

    monkeypatch.setattr(terminal_prompt, "prompt_for_password_change", _fake_prompt)
    assert run._terminal_password_gate(tunnel_will_start = True, **_GATE_KWARGS) == (True, True)
    admin = auth_storage.DEFAULT_ADMIN_USERNAME
    # One atomic call: refresh tokens revoked in the same transaction as the
    # password commit (a separable follow-up delete can fail and leave a
    # pre-change refresh token able to mint access tokens).
    assert calls == [("update", admin, "brand-new-password", {"revoke_refresh_tokens": True})]


# ── ordering inside run_server (source-level, repo convention) ───────


def test_gate_runs_before_server_bind_in_source():
    app_state = type("State", (), {})()
    run._publish_cloudflare_url(app_state, "https://live.trycloudflare.com")
    assert app_state.cloudflare_url == run._cloudflare_url == "https://live.trycloudflare.com"
    run._publish_cloudflare_url(app_state, None)
    # The gate must run before the uvicorn socket binds: on a wildcard bind
    # the served HTML injects the bootstrap credential for first login, so a
    # pre-gate listener would hand out the default password mid-prompt.
    src = (_BACKEND / "run.py").read_text(encoding = "utf-8")
    gate_call = src.index("_pw_proceed, _pw_drop_bootstrap = _terminal_password_gate(")
    thread_start = src.index("thread.start()")
    callback_bind = src.index("set_studio_tunnel_url_callback(")
    tunnel_start = src.index("start_studio_tunnel(", callback_bind)
    assert gate_call < thread_start < callback_bind < tunnel_start
    assert "_cloudflare_url = start_studio_tunnel" not in src
    # The fail-closed branch exits before any server exists.
    refusal = src[gate_call:thread_start]
    assert "sys.exit(1)" in refusal


def test_min_password_length_single_source():
    # models/auth.py must reference the storage constant, not a literal.
    models_src = (_BACKEND / "models" / "auth.py").read_text(encoding = "utf-8")
    assert "MIN_PASSWORD_LENGTH" in models_src
    assert not re.search(r"min_length\s*=\s*8\b", models_src)
    assert auth_storage.MIN_PASSWORD_LENGTH == 8


def test_lifespan_honors_bootstrap_suppression_in_source():
    # The lifespan runs AFTER the gate and re-reads the bootstrap password
    # into app.state; without the suppress flag it would overwrite the gate's
    # None and the public HTML would inject the default credential again.
    main_src = (_BACKEND / "main.py").read_text(encoding = "utf-8")
    assert "suppress_bootstrap_injection" in main_src
    # Every lifespan capture of the bootstrap password must be flag-guarded.
    for line in main_src.splitlines():
        if "storage.get_bootstrap_password()" in line and "=" in line:
            assert "_suppress_bootstrap" in line, line
    run_src = (_BACKEND / "run.py").read_text(encoding = "utf-8")
    assert "app.state.suppress_bootstrap_injection = True" in run_src


def test_clear_bootstrap_password_truncates_when_unlink_fails(monkeypatch, tmp_path):
    # If the file cannot be unlinked (Windows AV / read-only auth dir), clear must
    # truncate it so its stale plaintext cannot be re-seeded by
    # generate_bootstrap_password() if auth.db is ever recreated, which would
    # re-validate the revoked bootstrap password.
    import pathlib

    pw_path = tmp_path / ".bootstrap_password"
    pw_path.write_text("old-diceware-passphrase")
    monkeypatch.setattr(auth_storage, "_BOOTSTRAP_PW_PATH", pw_path)
    monkeypatch.setattr(auth_storage, "_bootstrap_password", "old-diceware-passphrase")

    _real_unlink = pathlib.Path.unlink

    def _boom(self, *a, **k):
        if self == pw_path:
            raise OSError("locked")
        return _real_unlink(self, *a, **k)

    monkeypatch.setattr(pathlib.Path, "unlink", _boom)

    auth_storage.clear_bootstrap_password()

    assert pw_path.exists()  # unlink failed
    assert pw_path.read_text() == ""  # but truncated -> no reusable plaintext

    # The stale value must not load back (empty file -> None), so a later re-seed
    # generates fresh rather than resurrecting the revoked credential.
    monkeypatch.setattr(auth_storage, "_bootstrap_password", None)
    assert auth_storage._load_bootstrap_password() is None


def test_clear_bootstrap_password_warns_truthfully_when_not_cleared(monkeypatch, tmp_path, capsys):
    # If the file can be neither unlinked NOR truncated, the stale plaintext stays
    # on disk. The warning must NOT claim it was made unreusable (Codex 3571888584):
    # it must say it could not be cleared and ask the user to remove it manually.
    import pathlib

    pw_path = tmp_path / ".bootstrap_password"
    pw_path.write_text("old-diceware-passphrase")
    monkeypatch.setattr(auth_storage, "_BOOTSTRAP_PW_PATH", pw_path)
    monkeypatch.setattr(auth_storage, "_bootstrap_password", "old-diceware-passphrase")

    _real_unlink = pathlib.Path.unlink
    _real_write_text = pathlib.Path.write_text

    def _boom_unlink(self, *a, **k):
        if self == pw_path:
            raise OSError("locked")
        return _real_unlink(self, *a, **k)

    def _boom_write_text(self, *a, **k):
        if self == pw_path:
            raise OSError("read-only")
        return _real_write_text(self, *a, **k)

    monkeypatch.setattr(pathlib.Path, "unlink", _boom_unlink)
    monkeypatch.setattr(pathlib.Path, "write_text", _boom_write_text)

    auth_storage.clear_bootstrap_password()

    # The stale plaintext survives untouched.
    assert pw_path.read_text() == "old-diceware-passphrase"
    warning = capsys.readouterr().err.lower()
    assert "could not delete or clear" in warning
    assert "still on disk" in warning
    assert "remove it manually" in warning
    # Must not falsely claim the contents were cleared (the bug being fixed).
    assert "cleared its contents" not in warning


# ── _apply_supplied_password: non-interactive initial password (direct run.py) ──


def _seed_stub_admin(
    monkeypatch,
    *,
    requires_change,
    bootstrap_pw = "bootstrap-secret",
):
    """Stub storage so _apply_supplied_password sees a seeded admin whose current
    password is ``bootstrap_pw`` and whose must-change flag is ``requires_change``;
    return the recorded update_password calls."""
    from auth import hashing

    salt, pwd_hash = hashing.hash_password(bootstrap_pw)
    monkeypatch.setattr(auth_storage, "ensure_default_admin", lambda: False)
    monkeypatch.setattr(auth_storage, "requires_password_change", lambda u: requires_change)
    monkeypatch.setattr(
        auth_storage, "get_user_and_secret", lambda u: (salt, pwd_hash, "jwt", requires_change)
    )
    calls = []
    monkeypatch.setattr(
        auth_storage, "update_password", lambda u, p, **kw: calls.append((u, p, kw))
    )
    return calls


def test_apply_supplied_password_sets_initial(monkeypatch):
    calls = _seed_stub_admin(monkeypatch, requires_change = True)
    monkeypatch.setenv(terminal_prompt.SUPPLIED_PASSWORD_ENV, "brand-new-password")
    run._apply_supplied_password(None)  # resolves from the env var
    admin = auth_storage.DEFAULT_ADMIN_USERNAME
    assert calls == [(admin, "brand-new-password", {"revoke_refresh_tokens": True})]


def test_apply_supplied_password_off_is_noop(monkeypatch):
    calls = _seed_stub_admin(monkeypatch, requires_change = True)
    monkeypatch.delenv(terminal_prompt.SUPPLIED_PASSWORD_ENV, raising = False)
    run._apply_supplied_password(None)
    run._apply_supplied_password("")
    assert calls == []


def test_apply_supplied_password_already_set_fails_closed(monkeypatch):
    calls = _seed_stub_admin(monkeypatch, requires_change = False)
    monkeypatch.setenv(terminal_prompt.SUPPLIED_PASSWORD_ENV, "brand-new-password")
    with pytest.raises(SystemExit) as exc:
        run._apply_supplied_password(None)
    assert exc.value.code == 1
    assert calls == []  # never overrides an existing password


def test_apply_supplied_password_too_short_fails_closed(monkeypatch):
    calls = _seed_stub_admin(monkeypatch, requires_change = True)
    monkeypatch.setenv(terminal_prompt.SUPPLIED_PASSWORD_ENV, "short")
    with pytest.raises(SystemExit) as exc:
        run._apply_supplied_password(None)
    assert exc.value.code == 1
    assert calls == []


def test_apply_supplied_password_must_differ_fails_closed(monkeypatch):
    calls = _seed_stub_admin(monkeypatch, requires_change = True, bootstrap_pw = "bootstrap-secret")
    monkeypatch.setenv(terminal_prompt.SUPPLIED_PASSWORD_ENV, "bootstrap-secret")
    with pytest.raises(SystemExit) as exc:
        run._apply_supplied_password(None)
    assert exc.value.code == 1
    assert calls == []


def test_apply_supplied_password_strips_env_from_subprocess_environment(monkeypatch):
    # The plaintext password must not linger in os.environ: run_server later spawns
    # cloudflared/llama-server/code-exec tools that would otherwise inherit it (also
    # readable via /proc/PID/environ). The direct-run.py path pops it itself; the CLI
    # pops it before re-exec. Assert the pop happens on the apply path...
    _seed_stub_admin(monkeypatch, requires_change = True)
    monkeypatch.setenv(terminal_prompt.SUPPLIED_PASSWORD_ENV, "brand-new-password")
    run._apply_supplied_password(None)
    assert terminal_prompt.SUPPLIED_PASSWORD_ENV not in run.os.environ


def test_apply_supplied_password_strips_env_even_when_literal_wins(monkeypatch):
    # A literal --password wins over the env var, but a stale env value would still
    # leak to subprocesses; the unconditional pop must clear it regardless of source.
    _seed_stub_admin(monkeypatch, requires_change = True)
    monkeypatch.setenv(terminal_prompt.SUPPLIED_PASSWORD_ENV, "env-should-be-stripped")
    run._apply_supplied_password("literal-new-password")
    assert terminal_prompt.SUPPLIED_PASSWORD_ENV not in run.os.environ


# ──────────────────────────────────────────────────────────────────────
# The gate now covers a raw exposed bind, not just the tunnel.
# ──────────────────────────────────────────────────────────────────────


_RAW_BIND_KWARGS = dict(
    host = "0.0.0.0",
    secure = False,
    api_only = False,
    frontend_served = True,
)


def test_a_headless_raw_bind_is_byte_for_byte_unchanged(monkeypatch):
    """The compatibility promise of this change.

    Headless `-H 0.0.0.0` is the long-running container case: it must still
    proceed, keep serving the bootstrap credential, and above all not reach the
    strip-and-refuse handling a public tunnel launch uses, which would delete the
    .bootstrap_password such deployments are logged into with.
    """
    _patch_streams(monkeypatch, tty = False)
    _patch_seeded_admin(monkeypatch, requires_change = True)
    assert run._terminal_password_gate(tunnel_will_start = False, **_RAW_BIND_KWARGS) == (True, False)


def test_a_headless_raw_bind_does_not_open_auth_storage(monkeypatch):
    """ "Unchanged" has to mean it does not touch the database either.

    A headless raw bind used to return at `if not tunnel_will_start` without
    importing auth storage. Calling ensure_default_admin() first would move
    seeding earlier and open the SQLite file sooner, giving a read-only or locked
    STUDIO_HOME a new place to fail on a launch that used to work, so
    promptability must be decided before storage is consulted.
    """
    _patch_streams(monkeypatch, tty = False)

    def _boom(*_a, **_k):
        raise AssertionError(
            "auth storage was opened on a headless raw bind; the gate must "
            "decide it cannot prompt before touching the database"
        )

    from auth import storage as _storage

    monkeypatch.setattr(_storage, "ensure_default_admin", _boom)
    monkeypatch.setattr(_storage, "requires_password_change", _boom)

    assert run._terminal_password_gate(tunnel_will_start = False, **_RAW_BIND_KWARGS) == (True, False)


def test_refusing_the_prompt_on_a_raw_bind_still_launches(monkeypatch):
    """Ctrl+C must not turn a working launch into no Studio.

    `docker/studio_run.sh` execs `unsloth studio -H 0.0.0.0` and only supplies a
    password when the initial-password file is non-empty, so a fresh
    `docker run -it` meets this prompt and aborting would stop a container that
    starts today. It proceeds on the bootstrap deadline it already had.
    """
    _patch_streams(monkeypatch, tty = True)
    _patch_seeded_admin(monkeypatch, requires_change = True)

    from auth import terminal_prompt

    monkeypatch.setattr(
        terminal_prompt,
        "prompt_for_password_change",
        lambda **_kw: False,  # Ctrl+C / EOF
    )

    # proceed = True, and the bootstrap credential is still injected as before.
    assert run._terminal_password_gate(tunnel_will_start = False, **_RAW_BIND_KWARGS) == (True, False)


def test_refusing_the_prompt_on_a_tunnel_still_aborts(monkeypatch):
    """The tunnel case is unchanged: refusing to secure a public URL fails closed."""
    _patch_streams(monkeypatch, tty = True)
    _patch_seeded_admin(monkeypatch, requires_change = True)

    from auth import terminal_prompt

    monkeypatch.setattr(terminal_prompt, "prompt_for_password_change", lambda **_kw: False)

    assert run._terminal_password_gate(tunnel_will_start = True, **_GATE_KWARGS) == (False, False)


def test_a_raw_bind_with_a_terminal_reaches_the_prompt(monkeypatch):
    """The fix. Before this, `if not tunnel_will_start` returned first."""
    _patch_streams(monkeypatch, tty = True)
    _patch_seeded_admin(monkeypatch, requires_change = True)

    seen = {}

    def _fake_prompt(*, min_length, is_current_password, apply_change, out, **kw):
        seen.update(kw)
        apply_change("a-brand-new-password")
        return True

    from auth import terminal_prompt

    monkeypatch.setattr(terminal_prompt, "prompt_for_password_change", _fake_prompt)

    assert run._terminal_password_gate(tunnel_will_start = False, **_RAW_BIND_KWARGS) == (True, True)
    # And it must not tell a LAN operator they are on the public internet.
    assert seen.get("exposure") == "on every network interface"


def test_a_loopback_launch_still_short_circuits(monkeypatch):
    """Plain `unsloth studio` must not consult storage at all."""

    def _boom(*_a, **_k):
        raise AssertionError("storage must not be consulted for a loopback launch")

    monkeypatch.setattr(run, "_stream_isatty", lambda _s: True)
    from auth import storage as _storage

    monkeypatch.setattr(_storage, "ensure_default_admin", _boom)

    assert run._terminal_password_gate(
        tunnel_will_start = False,
        host = "127.0.0.1",
        secure = False,
        api_only = False,
        frontend_served = True,
    ) == (True, False)


def test_api_only_and_colab_raw_binds_do_not_prompt(monkeypatch):
    """Scoped like the bootstrap deadline: web UI only, never api-only or Colab."""

    def _boom(*_a, **_k):
        raise AssertionError("storage must not be consulted")

    monkeypatch.setattr(run, "_stream_isatty", lambda _s: True)
    from auth import storage as _storage

    monkeypatch.setattr(_storage, "ensure_default_admin", _boom)

    assert run._terminal_password_gate(
        tunnel_will_start = False,
        host = "0.0.0.0",
        secure = False,
        api_only = True,
        frontend_served = True,
    ) == (True, False)
    assert run._terminal_password_gate(
        tunnel_will_start = False,
        host = "0.0.0.0",
        secure = False,
        api_only = False,
        frontend_served = True,
        is_colab = True,
    ) == (True, False)


# ──────────────────────────────────────────────────────────────────────
# A backgrounded shell job is not a usable terminal.
# ──────────────────────────────────────────────────────────────────────


class _FdStream(_Stream):
    def __init__(self):
        super().__init__(isatty = True)

    def fileno(self):
        return 0


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
    would freeze before the socket binds, so it takes the old headless path:
    proceed on the bootstrap deadline, without consulting auth storage.
    """
    monkeypatch.setattr(sys, "stdin", _FdStream())
    monkeypatch.setattr(sys, "stderr", _FdStream())
    monkeypatch.setattr(run.os, "tcgetpgrp", lambda _fd: 4242)
    monkeypatch.setattr(run.os, "getpgrp", lambda: 99)

    def _boom(*_a, **_k):
        raise AssertionError("storage must not be opened for a background job")

    monkeypatch.setattr(auth_storage, "ensure_default_admin", _boom)

    assert run._terminal_password_gate(tunnel_will_start = False, **_RAW_BIND_KWARGS) == (True, False)


@pytest.mark.skipif(
    os.name == "nt",
    reason = "POSIX terminal semantics: Windows has no process groups, no SIGTTOU and no pty, "
    "so there is nothing here to assert. _prompt_owns_the_terminal fails open there, "
    "which test_windows_has_no_terminal_ownership_to_lose pins.",
)
def test_a_backgrounded_tunnel_launch_still_fails_closed(monkeypatch):
    """A tunnel publishes a public URL: it must not quietly proceed unprompted."""
    monkeypatch.setattr(sys, "stdin", _FdStream())
    monkeypatch.setattr(sys, "stderr", _FdStream())
    monkeypatch.setattr(run.os, "tcgetpgrp", lambda _fd: 4242)
    monkeypatch.setattr(run.os, "getpgrp", lambda: 99)
    _patch_seeded_admin(monkeypatch, requires_change = True)
    monkeypatch.setattr(terminal_prompt, "prompt_for_password_change", lambda **_kw: False)

    # The process group check is scoped to the raw-bind branch, so a tunnel still
    # reaches the prompt and still aborts when it is refused.
    assert run._prompt_owns_the_terminal() is False
    assert run._terminal_password_gate(tunnel_will_start = True, **_GATE_KWARGS) == (False, False)


@pytest.mark.skipif(
    os.name == "nt",
    reason = "POSIX terminal semantics: Windows has no process groups, no SIGTTOU and no pty, "
    "so there is nothing here to assert. _prompt_owns_the_terminal fails open there, "
    "which test_windows_has_no_terminal_ownership_to_lose pins.",
)
def test_a_foreground_raw_bind_still_reaches_the_prompt(monkeypatch):
    """The ordinary interactive case is untouched."""
    monkeypatch.setattr(sys, "stdin", _FdStream())
    monkeypatch.setattr(sys, "stderr", _FdStream())
    monkeypatch.setattr(run.os, "tcgetpgrp", lambda _fd: 4242)
    monkeypatch.setattr(run.os, "getpgrp", lambda: 4242)
    _patch_seeded_admin(monkeypatch, requires_change = True)
    monkeypatch.setattr(terminal_prompt, "prompt_for_password_change", lambda **_kw: True)

    assert run._terminal_password_gate(tunnel_will_start = False, **_RAW_BIND_KWARGS) == (True, True)


def test_no_job_control_falls_back_to_the_isatty_answer(monkeypatch):
    """Windows / no controlling terminal: nothing can stop us, so still prompt."""
    for exc in (OSError("ENOTTY"), AttributeError(), ValueError()):

        def _boom(_fd, _exc = exc):
            raise _exc

        monkeypatch.setattr(run.os, "tcgetpgrp", _boom, raising = False)
        monkeypatch.setattr(sys, "stdin", _FdStream())
        assert run._prompt_owns_the_terminal() is True


# ──────────────────────────────────────────────────────────────────────
# A pty is not a person: an unattended terminal must not hold the launch.
# ──────────────────────────────────────────────────────────────────────


class _PtyStdin:
    """sys.stdin standing on a real pty slave, as tmux/screen/docker -t give it."""

    def __init__(self, fd):
        self._fd = fd
        self.encoding = "utf-8"

    def fileno(self):
        return self._fd

    def isatty(self):
        return True


@pytest.mark.skipif(
    os.name == "nt",
    reason = "POSIX terminal semantics: Windows has no process groups, no SIGTTOU and no pty, "
    "so there is nothing here to assert. _prompt_owns_the_terminal fails open there, "
    "which test_windows_has_no_terminal_ownership_to_lose pins.",
)
def test_an_unattended_pty_does_not_block_a_raw_bind_forever(monkeypatch):
    """`tmux new -d 'unsloth studio -H 0.0.0.0'` must still bind its socket.

    A detached pty is a real, foreground terminal nobody will ever type into:
    isatty() is True on both streams and the process owns the terminal, so every
    interactivity test says "prompt". The read never returns, and the gate runs
    BEFORE uvicorn binds, so the launch hangs forever instead of starting.
    """
    import io
    import pty

    master, slave = pty.openpty()
    try:
        monkeypatch.setattr(sys, "stdin", _PtyStdin(slave))
        out = io.StringIO()
        # Nothing is written to `master`: the pty exists, the human does not.
        changed = terminal_prompt.prompt_for_password_change(
            min_length = 8,
            is_current_password = lambda _c: False,
            apply_change = lambda _p: pytest.fail("nothing was typed"),
            out = out,
            first_key_timeout = 0.25,
        )
    finally:
        os.close(master)
        os.close(slave)

    assert changed is False
    assert "No response at the terminal" in out.getvalue()


@pytest.mark.skipif(
    os.name == "nt",
    reason = "POSIX terminal semantics: Windows has no process groups, no SIGTTOU and no pty, "
    "so there is nothing here to assert. _prompt_owns_the_terminal fails open there, "
    "which test_windows_has_no_terminal_ownership_to_lose pins.",
)
def test_a_pty_someone_types_into_is_not_treated_as_unattended(monkeypatch):
    """The deadline is on the FIRST keystroke only, and a real one clears it."""
    import io
    import pty

    applied = []
    master, slave = pty.openpty()
    try:
        os.write(master, b"abcdefgh12\rabcdefgh12\r")
        monkeypatch.setattr(sys, "stdin", _PtyStdin(slave))
        out = io.StringIO()
        changed = terminal_prompt.prompt_for_password_change(
            min_length = 8,
            is_current_password = lambda _c: False,
            apply_change = applied.append,
            out = out,
            first_key_timeout = 0.25,
        )
    finally:
        os.close(master)
        os.close(slave)

    assert changed is True
    assert applied == ["abcdefgh12"]


def test_the_gate_deadlines_a_raw_bind_prompt_and_never_the_tunnel(monkeypatch):
    """Scope: only the launch that must not be blocked gets a deadline.

    A tunnel publishes a public URL, so it keeps waiting indefinitely and fails
    closed; a raw bind falls back to the protection it already had.
    """
    seen = {}

    def _fake_prompt(**kwargs):
        seen.clear()
        seen.update(kwargs)
        return True

    monkeypatch.setattr(terminal_prompt, "prompt_for_password_change", _fake_prompt)

    _patch_streams(monkeypatch, tty = True)
    _patch_seeded_admin(monkeypatch, requires_change = True)
    run._terminal_password_gate(tunnel_will_start = False, **_RAW_BIND_KWARGS)
    assert seen["first_key_timeout"] == run._UNATTENDED_PROMPT_SECONDS

    _patch_streams(monkeypatch, tty = True)
    _patch_seeded_admin(monkeypatch, requires_change = True)
    run._terminal_password_gate(tunnel_will_start = True, **_GATE_KWARGS)
    assert seen["first_key_timeout"] is None


def test_a_refusal_does_not_promise_a_deadline_that_is_disabled(monkeypatch):
    """With TIMEOUT=0 nothing will shut this instance down; do not say otherwise.

    The refusal path proceeds on purpose (the launch worked before the prompt
    existed), but must not tell the operator a deadline will rescue them when
    `should_arm_bootstrap_timeout` will not arm one: that is the single sentence
    they would act on.
    """
    monkeypatch.setenv("UNSLOTH_STUDIO_BOOTSTRAP_TIMEOUT", "0")
    stderr = _patch_streams(monkeypatch, tty = True)
    _patch_seeded_admin(monkeypatch, requires_change = True)

    from auth import terminal_prompt

    monkeypatch.setattr(terminal_prompt, "prompt_for_password_change", lambda **_kw: False)

    assert run._terminal_password_gate(tunnel_will_start = False, **_RAW_BIND_KWARGS) == (True, False)
    err = stderr.getvalue()
    assert "DISABLED for this launch" in err, err
    assert "shuts down after the bootstrap deadline" not in err, err


def test_a_refusal_still_names_the_deadline_when_one_will_arm(monkeypatch):
    monkeypatch.delenv("UNSLOTH_STUDIO_BOOTSTRAP_TIMEOUT", raising = False)
    stderr = _patch_streams(monkeypatch, tty = True)
    _patch_seeded_admin(monkeypatch, requires_change = True)

    from auth import terminal_prompt

    monkeypatch.setattr(terminal_prompt, "prompt_for_password_change", lambda **_kw: False)

    assert run._terminal_password_gate(tunnel_will_start = False, **_RAW_BIND_KWARGS) == (True, False)
    err = stderr.getvalue()
    assert "shuts down after the bootstrap deadline" in err, err
    assert "DISABLED" not in err, err


def test_the_child_does_not_repeat_a_prompt_the_parent_already_gave_up_on(monkeypatch):
    """A 30s fallback must not become 60s across the re-exec.

    The CLI parent holds the terminal for its deadline, gets nothing, warns and
    launches. The child gate then sees the SAME unattended pty; without a handoff
    it waits the whole deadline again, and a third time on the `studio run` path,
    which is the startup stall the 30s value was chosen to stay under.
    """
    monkeypatch.setenv("UNSLOTH_STUDIO_UNATTENDED_PROMPT_DONE", "1")
    _patch_streams(monkeypatch, tty = True)

    def _boom(*_a, **_k):
        raise AssertionError("the child prompted again after the parent gave up")

    from auth import storage as _storage

    monkeypatch.setattr(_storage, "ensure_default_admin", _boom)

    assert run._terminal_password_gate(tunnel_will_start = False, **_RAW_BIND_KWARGS) == (True, False)
    # Consumed, so a later launch from the same environment keeps its own prompt.
    import os as _os

    assert _os.environ.get("UNSLOTH_STUDIO_UNATTENDED_PROMPT_DONE") is None


def test_the_marker_never_silences_a_tunnel_launch(monkeypatch):
    """A public URL fails closed regardless of what the parent did."""
    monkeypatch.setenv("UNSLOTH_STUDIO_UNATTENDED_PROMPT_DONE", "1")
    _patch_streams(monkeypatch, tty = True)
    _patch_seeded_admin(monkeypatch, requires_change = True)

    from auth import terminal_prompt

    monkeypatch.setattr(terminal_prompt, "prompt_for_password_change", lambda **_kw: False)

    assert run._terminal_password_gate(tunnel_will_start = True, **_GATE_KWARGS) == (False, False)


def test_windows_has_no_terminal_ownership_to_lose(monkeypatch):
    """The Windows half of the tests skipped above, and it runs everywhere.

    `_prompt_owns_the_terminal` exists to catch a backgrounded POSIX job, where
    driving the terminal raises SIGTTOU and stops the process. Windows has no
    process groups and no `os.tcgetpgrp`, so the call raises AttributeError and
    the answer must be True: there is nothing there that can stop us, so the
    isatty verdict stands and an interactive Windows launch still prompts. A
    False would silently drop the prompt on every Windows terminal.
    """
    monkeypatch.delattr(run.os, "tcgetpgrp", raising = False)
    assert run._prompt_owns_the_terminal() is True
