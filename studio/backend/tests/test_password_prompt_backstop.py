# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Pre-tunnel terminal password gate: never publish a public Cloudflare URL
while the seeded default admin password is active. Imports run.py directly,
so run under the Unsloth venv."""

from __future__ import annotations

import io
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


@pytest.fixture(autouse = True)
def _tunnel_available_by_default(monkeypatch):
    # The headless auto-generate branch now consults the real cloudflared binary on
    # the --secure path (a missing tunnel means no public URL, so rotating the only
    # recovery password would strip it behind a droppable one-time banner). Default
    # to "available" so the existing auto-generate tests never touch the network;
    # the dedicated test below flips this to exercise the fail-closed refusal.
    monkeypatch.setattr(run, "_tunnel_binary_confirmed_unavailable", lambda: False)


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


def _patch_streams_autogen(monkeypatch) -> _Stream:
    # The auto-generate branch fires when the interactive prompt is unavailable
    # (stdin not a tty) but a real terminal still exists to surface the one-time
    # credential on an EPHEMERAL console: stdin non-tty, stderr a tty. A fully
    # redirected launch (stderr also non-tty) instead fails closed -- see
    # test_gate_fails_closed_when_console_is_redirected_file.
    stderr = _Stream(isatty = True)
    monkeypatch.setattr(sys, "stdin", _Stream(isatty = False))
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


def _stub_update_password(monkeypatch, *, committed: bool = True) -> list:
    """Record update_password calls so the auto-generate path can be asserted
    without touching the real auth DB.

    Returns what the real function returns -- the rotated JWT secret on a write,
    None when a guard rejected it -- so `committed = False` really does read as a
    lost compare-and-set at the call site. A stub that returned the bool itself
    would make the lost case look committed, since `False is not None`.
    """
    calls = []

    def _update(u, p, **kw):
        calls.append((u, p, kw))
        return "rotated-secret" if committed else None

    monkeypatch.setattr(auth_storage, "update_password", _update)
    return calls


def test_gate_autogenerates_password_on_tty_console(monkeypatch):
    # No interactive prompt (stdin not a tty) and no supplied password, but a real
    # terminal (stderr) is present to surface the credential ephemerally: the gate
    # auto-generates a strong admin password, commits it (clearing must_change so
    # the tunnel proceeds headlessly), and surfaces it once. The public HTML must
    # still not auto-fill any stale bootstrap credential (drop_bootstrap True).
    stderr = _patch_streams_autogen(monkeypatch)
    _patch_seeded_admin(monkeypatch, requires_change = True)
    monkeypatch.delenv("UNSLOTH_STUDIO_BOOTSTRAP_TIMEOUT", raising = False)
    monkeypatch.setattr(
        terminal_prompt,
        "prompt_for_password_change",
        lambda **k: pytest.fail("prompt must not run without a tty"),
    )
    calls = _stub_update_password(monkeypatch)
    assert run._terminal_password_gate(tunnel_will_start = True, **_GATE_KWARGS) == (True, True)
    admin = auth_storage.DEFAULT_ADMIN_USERNAME
    # Committed via the route-equivalent atomic update (refresh tokens revoked in
    # the same transaction).
    assert len(calls) == 1, calls
    username, password, kwargs = calls[0]
    assert username == admin
    assert isinstance(password, str) and len(password) >= auth_storage.MIN_PASSWORD_LENGTH
    # Compare-and-set: the write lands only while must_change_password is still 1,
    # so a password chosen elsewhere in the meantime is never overwritten.
    assert kwargs == {"revoke_refresh_tokens": True, "require_must_change": True}
    out = stderr.getvalue()
    assert "auto-generated" in out
    assert admin in out
    assert password in out  # the credential is printed exactly once


def test_gate_autogenerated_credential_stays_out_of_session_log(monkeypatch):
    # run_server() installs a _TeeStream over sys.stderr (see
    # _setup_server_disk_logging) that mirrors everything into a RETAINED
    # logs/server/server-*.log. The one-time auto-generated admin password must
    # reach the operator's console but must NEVER land in that persisted log
    # (OWASP CWE-532: no credentials in log files). The gate writes it to the raw
    # stream behind the tee, so the log copy stays clean.
    console = _Stream(isatty = True)
    log_fh = io.StringIO()
    tee = run._TeeStream(console, log_fh)
    monkeypatch.setattr(sys, "stdin", _Stream(isatty = False))
    monkeypatch.setattr(sys, "stderr", tee)
    _patch_seeded_admin(monkeypatch, requires_change = True)
    monkeypatch.delenv("UNSLOTH_STUDIO_BOOTSTRAP_TIMEOUT", raising = False)
    calls = _stub_update_password(monkeypatch)

    assert run._terminal_password_gate(tunnel_will_start = True, **_GATE_KWARGS) == (True, True)
    assert len(calls) == 1, calls
    _username, password, _kwargs = calls[0]

    console_out = console.getvalue()
    log_out = log_fh.getvalue()
    # The operator sees the banner and the credential on the console...
    assert "auto-generated" in console_out
    assert password in console_out
    # ...but nothing was mirrored into the retained on-disk session log.
    assert password not in log_out
    assert "auto-generated" not in log_out


def test_console_only_stream_unwraps_tee():
    # Unit: _console_only_stream returns the raw stream behind a _TeeStream and is
    # a no-op for a plain stream, so a secret written through it never reaches the
    # log file handle the tee mirrors into.
    console = io.StringIO()
    log_fh = io.StringIO()
    tee = run._TeeStream(console, log_fh)
    assert run._console_only_stream(tee) is console
    plain = io.StringIO()
    assert run._console_only_stream(plain) is plain


def test_console_only_stream_unwraps_nested_tees(monkeypatch):
    # run_server() can run twice in one process (a local run, then a public one),
    # and each call re-wraps the already-wrapped sys.stdout/stderr, so the tees
    # nest. Peeling a single layer returned the INNER _TeeStream -- which forwards
    # isatty() to the console and so passes the tty check -- and the one-time
    # credential was mirrored into the first run's retained server-*.log
    # (CWE-532). Every layer must be unwrapped.
    console = _Stream(isatty = True)
    first_log, second_log = io.StringIO(), io.StringIO()
    nested = run._TeeStream(run._TeeStream(console, first_log), second_log)
    assert run._console_only_stream(nested) is console

    # End to end: the secret reaches the console and neither retained log.
    monkeypatch.setattr(sys, "stderr", nested)
    monkeypatch.setattr(sys, "stdout", nested)
    out = run._one_time_secret_stream()
    assert out is console
    run._print_auto_generated_credentials("unsloth", "Nested-Tee-Pw-1", out = out)
    assert "Nested-Tee-Pw-1" in console.getvalue()
    assert "Nested-Tee-Pw-1" not in first_log.getvalue()
    assert "Nested-Tee-Pw-1" not in second_log.getvalue()


def test_gate_does_not_show_password_when_rotation_loses_the_race(monkeypatch):
    # Another Studio process/tab sharing this auth DB can finish /change-password
    # between the gate's must_change read and the rotation. The guarded update then
    # writes nothing, so the generated value never took effect: it must not be
    # displayed (it would not authenticate), and the launch proceeds under the
    # password that did land.
    stderr = _patch_streams_autogen(monkeypatch)
    _patch_seeded_admin(monkeypatch, requires_change = True)
    monkeypatch.delenv("UNSLOTH_STUDIO_BOOTSTRAP_TIMEOUT", raising = False)
    calls = _stub_update_password(monkeypatch, committed = False)

    assert run._terminal_password_gate(tunnel_will_start = True, **_GATE_KWARGS) == (True, True)

    assert len(calls) == 1, calls
    _username, password, kwargs = calls[0]
    assert kwargs["require_must_change"] is True
    out = stderr.getvalue()
    assert password not in out
    assert "auto-generated" not in out


def test_gate_autogenerates_even_when_deadline_cannot_arm(monkeypatch):
    # api-only launches never armed the bootstrap deadline and used to fail closed;
    # now a strong password is set instead, so the launch proceeds with real
    # protection rather than being refused.
    _patch_streams_autogen(monkeypatch)
    _patch_seeded_admin(monkeypatch, requires_change = True)
    monkeypatch.delenv("UNSLOTH_STUDIO_BOOTSTRAP_TIMEOUT", raising = False)
    calls = _stub_update_password(monkeypatch)
    kwargs = dict(_GATE_KWARGS)
    kwargs["api_only"] = True
    kwargs["frontend_served"] = False
    assert run._terminal_password_gate(tunnel_will_start = True, **kwargs) == (True, True)
    assert len(calls) == 1, calls


def test_gate_autogenerates_when_deadline_disabled(monkeypatch):
    # UNSLOTH_STUDIO_BOOTSTRAP_TIMEOUT=0 disables the deadline; the auto-generated
    # password is the safeguard now, so the launch still proceeds.
    _patch_streams_autogen(monkeypatch)
    _patch_seeded_admin(monkeypatch, requires_change = True)
    monkeypatch.setenv("UNSLOTH_STUDIO_BOOTSTRAP_TIMEOUT", "0")
    calls = _stub_update_password(monkeypatch)
    assert run._terminal_password_gate(tunnel_will_start = True, **_GATE_KWARGS) == (True, True)
    assert len(calls) == 1, calls


def test_gate_refuses_secure_when_tunnel_binary_unavailable(monkeypatch):
    # --secure exposes ONLY the loopback-bound tunnel. When cloudflared is provably
    # unavailable no public URL can come up, so the gate must NOT rotate the seeded
    # recovery password (which a headless launch may only surface on a droppable
    # one-time banner, locking the operator out). Mirrors the CLI: fail closed with
    # the credential untouched, matching run_server's later secure-tunnel guard.
    stderr = _patch_streams_autogen(monkeypatch)
    _patch_seeded_admin(monkeypatch, requires_change = True)
    monkeypatch.setattr(run, "_tunnel_binary_confirmed_unavailable", lambda: True)
    calls = _stub_update_password(monkeypatch)
    assert run._terminal_password_gate(tunnel_will_start = True, **_GATE_KWARGS) == (False, False)
    # No rotation happened: the existing recovery credential is intact.
    assert calls == []
    # And nothing that looks like a credential was surfaced.
    assert "auto-generated" not in stderr.getvalue()


def test_gate_non_secure_still_rotates_when_tunnel_unavailable(monkeypatch):
    # A --cloudflare wildcard bind (NOT --secure) still serves the raw 0.0.0.0 port
    # even if the tunnel fails, so the default credential must be replaced: the
    # tunnel-availability refusal is scoped to --secure only.
    _patch_streams_autogen(monkeypatch)
    _patch_seeded_admin(monkeypatch, requires_change = True)
    monkeypatch.setattr(run, "_tunnel_binary_confirmed_unavailable", lambda: True)
    monkeypatch.delenv("UNSLOTH_STUDIO_BOOTSTRAP_TIMEOUT", raising = False)
    calls = _stub_update_password(monkeypatch)
    kwargs = dict(_GATE_KWARGS)
    kwargs["secure"] = False
    kwargs["host"] = "0.0.0.0"
    assert run._terminal_password_gate(tunnel_will_start = True, **kwargs) == (True, True)
    assert len(calls) == 1, calls


def test_gate_credential_falls_back_to_stdout_when_stderr_absent(monkeypatch):
    # A GUI/service wrapper can leave sys.stderr's raw stream absent (None) while
    # _setup_server_disk_logging() still tees sys.stdout into a retained log. The
    # one-time credential must fall back to the raw console behind the stdout tee,
    # NEVER to print(file=None) which would write it into that log (CWE-532).
    log_fh = io.StringIO()
    stderr_tee = run._TeeStream(None, log_fh)  # raw console absent behind the tee
    console = _Stream(isatty = True)
    stdout_tee = run._TeeStream(console, log_fh)
    monkeypatch.setattr(sys, "stdin", _Stream(isatty = False))
    monkeypatch.setattr(sys, "stderr", stderr_tee)
    monkeypatch.setattr(sys, "stdout", stdout_tee)
    _patch_seeded_admin(monkeypatch, requires_change = True)
    monkeypatch.delenv("UNSLOTH_STUDIO_BOOTSTRAP_TIMEOUT", raising = False)
    calls = _stub_update_password(monkeypatch)

    assert run._terminal_password_gate(tunnel_will_start = True, **_GATE_KWARGS) == (True, True)
    assert len(calls) == 1, calls
    _username, password, _kwargs = calls[0]
    # Surfaced on the real console (behind the stdout tee)...
    assert password in console.getvalue()
    assert "auto-generated" in console.getvalue()
    # ...but never mirrored into the retained on-disk session log.
    assert password not in log_fh.getvalue()
    assert "auto-generated" not in log_fh.getvalue()


def test_gate_fails_closed_when_no_console_stream(monkeypatch):
    # A Windows pythonw/service wrapper can leave BOTH sys.stderr and sys.stdout
    # absent (None) even while a session log is tee'd. There is then no real console
    # to surface the one-time credential without persisting it, so the gate must
    # fail closed WITHOUT rotating the only recovery credential -- rather than let
    # print(file=None) fall back to the tee'd stdout and write it to disk.
    log_fh = io.StringIO()
    monkeypatch.setattr(sys, "stdin", _Stream(isatty = False))
    monkeypatch.setattr(sys, "stderr", run._TeeStream(None, log_fh))
    monkeypatch.setattr(sys, "stdout", run._TeeStream(None, log_fh))
    _patch_seeded_admin(monkeypatch, requires_change = True)
    monkeypatch.delenv("UNSLOTH_STUDIO_BOOTSTRAP_TIMEOUT", raising = False)
    calls = _stub_update_password(monkeypatch)

    assert run._terminal_password_gate(tunnel_will_start = True, **_GATE_KWARGS) == (False, False)
    # No rotation, and nothing written into the retained on-disk session log.
    assert calls == []
    assert log_fh.getvalue() == ""


def test_one_time_secret_stream_prefers_stderr_then_stdout():
    # Unit: prefer stderr, fall back to stdout, unwrap the tee, None when neither
    # has a usable raw stream. The real console behind the tee must be a tty -- only
    # a terminal is an ephemeral surface for the one-time secret.
    real_err, real_out, log = _Stream(isatty = True), _Stream(isatty = True), io.StringIO()
    import contextlib

    @contextlib.contextmanager
    def _streams(stderr, stdout):
        old_err, old_out = sys.stderr, sys.stdout
        sys.stderr, sys.stdout = stderr, stdout
        try:
            yield
        finally:
            sys.stderr, sys.stdout = old_err, old_out

    with _streams(run._TeeStream(real_err, log), run._TeeStream(real_out, log)):
        assert run._one_time_secret_stream() is real_err  # stderr wins
    with _streams(run._TeeStream(None, log), run._TeeStream(real_out, log)):
        assert run._one_time_secret_stream() is real_out  # falls back past absent stderr
    with _streams(run._TeeStream(None, log), run._TeeStream(None, log)):
        assert run._one_time_secret_stream() is None  # no real console anywhere


def test_one_time_secret_stream_skips_closed_and_nonwritable(monkeypatch):
    # A headless launch can inherit a stderr/stdout whose raw stream is a CLOSED
    # or non-writable stream (not None). Printing the one-time credential to it
    # raises ValueError -- but only AFTER _auto_generate_admin_password already
    # rotated the seeded recovery credential, locking the operator out. The stream
    # selector must treat such streams as unusable and skip them, mirroring the CLI
    # _one_time_secret_console_stream closed/writable preflight.
    log = io.StringIO()

    class _NonWritable:
        closed = False
        write = None  # not callable -> unusable

        def isatty(self):
            return True

    closed_err = io.StringIO()
    closed_err.close()
    real_out = _Stream(isatty = True)

    # A closed raw stderr behind the tee is skipped; falls back to the usable stdout.
    monkeypatch.setattr(sys, "stderr", run._TeeStream(closed_err, log))
    monkeypatch.setattr(sys, "stdout", run._TeeStream(real_out, log))
    assert run._one_time_secret_stream() is real_out

    # A non-writable raw stderr is skipped too.
    monkeypatch.setattr(sys, "stderr", run._TeeStream(_NonWritable(), log))
    monkeypatch.setattr(sys, "stdout", run._TeeStream(real_out, log))
    assert run._one_time_secret_stream() is real_out

    # Closed on both -> no usable console anywhere -> None (caller fails closed).
    closed_out = io.StringIO()
    closed_out.close()
    monkeypatch.setattr(sys, "stderr", run._TeeStream(closed_err, log))
    monkeypatch.setattr(sys, "stdout", run._TeeStream(closed_out, log))
    assert run._one_time_secret_stream() is None


def test_one_time_secret_stream_requires_tty(monkeypatch):
    # A headless launch with stderr/stdout redirected to a file (`> log 2>&1`),
    # nohup.out, or a systemd-journald socket inherits a stream that is open and
    # writable but NOT a tty. Writing the one-time credential there PERSISTS the
    # plaintext (CWE-532), breaking the banner's "not written to disk" promise, so
    # the selector must skip a non-tty stream and prefer a real terminal.
    log = io.StringIO()
    redirected_err = _Stream(isatty = False)  # e.g. `> server.log 2>&1`
    tty_out = _Stream(isatty = True)

    # A non-tty (redirected) stderr is skipped; falls back to the tty stdout.
    monkeypatch.setattr(sys, "stderr", run._TeeStream(redirected_err, log))
    monkeypatch.setattr(sys, "stdout", run._TeeStream(tty_out, log))
    assert run._one_time_secret_stream() is tty_out

    # Both streams redirected (fully headless) -> no ephemeral surface -> None, so
    # the caller fails closed rather than persist the credential.
    monkeypatch.setattr(sys, "stderr", run._TeeStream(_Stream(isatty = False), log))
    monkeypatch.setattr(sys, "stdout", run._TeeStream(_Stream(isatty = False), log))
    assert run._one_time_secret_stream() is None


def test_gate_fails_closed_when_console_is_redirected_file(monkeypatch):
    # Regression (Codex 3644549779, P1): a headless `python run.py --secure` with
    # stderr/stdout redirected to a file/journal (nohup, systemd, `> log 2>&1`) has
    # a writable but non-tty console. Auto-generating there would write the admin
    # password into that RETAINED file, leaking the only credential to log consumers
    # despite the no-persistence guarantee. The gate must fail closed WITHOUT
    # rotating -- the operator supplies --password for a truly headless launch.
    redirected = _Stream(isatty = False)  # simulates `> server.log 2>&1`
    monkeypatch.setattr(sys, "stdin", _Stream(isatty = False))
    monkeypatch.setattr(sys, "stderr", redirected)
    monkeypatch.setattr(sys, "stdout", _Stream(isatty = False))
    _patch_seeded_admin(monkeypatch, requires_change = True)
    monkeypatch.delenv("UNSLOTH_STUDIO_BOOTSTRAP_TIMEOUT", raising = False)
    calls = _stub_update_password(monkeypatch)

    assert run._terminal_password_gate(tunnel_will_start = True, **_GATE_KWARGS) == (False, False)
    # No rotation, and nothing that looks like a credential written to the file.
    assert calls == []
    assert "auto-generated" not in redirected.getvalue()


def test_gate_fails_closed_when_console_stream_closed(monkeypatch):
    # Regression: a headless `python run.py --secure` launch whose raw stderr is a
    # CLOSED stream (and stdout absent) must NOT rotate the seeded recovery
    # credential. Previously _one_time_secret_stream only checked for None, so it
    # returned the closed stream; the gate committed the generated password (clearing
    # the bootstrap credential) and then _print_auto_generated_credentials raised
    # ValueError on the closed stream, so the operator never received the only new
    # password and was locked out. It must fail closed WITHOUT rotating.
    log_fh = io.StringIO()
    closed_err = io.StringIO()
    closed_err.close()
    monkeypatch.setattr(sys, "stdin", _Stream(isatty = False))
    monkeypatch.setattr(sys, "stderr", run._TeeStream(closed_err, log_fh))
    monkeypatch.setattr(sys, "stdout", run._TeeStream(None, log_fh))
    _patch_seeded_admin(monkeypatch, requires_change = True)
    monkeypatch.delenv("UNSLOTH_STUDIO_BOOTSTRAP_TIMEOUT", raising = False)
    calls = _stub_update_password(monkeypatch)

    assert run._terminal_password_gate(tunnel_will_start = True, **_GATE_KWARGS) == (False, False)
    # No rotation, and nothing written into the retained on-disk session log.
    assert calls == []
    assert log_fh.getvalue() == ""


def test_gate_treats_broken_streams_as_non_interactive(monkeypatch):
    # A closed/None stdin must take the headless path (auto-generate), not blow up.
    # stderr is a real terminal, so the one-time credential is surfaced ephemerally.
    stderr = _Stream(isatty = True)
    monkeypatch.setattr(sys, "stdin", _BrokenStream())
    monkeypatch.setattr(sys, "stderr", stderr)
    _patch_seeded_admin(monkeypatch, requires_change = True)
    monkeypatch.delenv("UNSLOTH_STUDIO_BOOTSTRAP_TIMEOUT", raising = False)
    calls = _stub_update_password(monkeypatch)
    assert run._terminal_password_gate(tunnel_will_start = True, **_GATE_KWARGS) == (True, True)
    assert len(calls) == 1, calls


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

    def _fake_prompt(*, min_length, is_current_password, apply_change, out):
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
    tunnel_start = src.index('start_studio_tunnel(port, managed_by = "launch")')
    callback_bind = src.index("set_studio_tunnel_url_callback(")
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


# ── partial-success recovery on the direct-server path ───────────────


@pytest.fixture
def real_auth_db(tmp_path, monkeypatch):
    """Point auth storage at a throwaway DB for the tests that need the REAL
    update_password commit (the stubs above never touch SQLite)."""
    monkeypatch.setattr(auth_storage, "DB_PATH", tmp_path / "auth.db")
    monkeypatch.setattr(auth_storage, "_BOOTSTRAP_PW_PATH", tmp_path / ".bootstrap_password")
    monkeypatch.setattr(auth_storage, "_bootstrap_password", None)
    monkeypatch.setattr(auth_storage, "_api_key_pbkdf2_salt_cache", None)
    return tmp_path


def _seed_real_admin(*, must_change_password: bool = True) -> str:
    import secrets as _secrets
    auth_storage.create_initial_user(
        username = auth_storage.DEFAULT_ADMIN_USERNAME,
        password = "bootstrap-secret-123",
        jwt_secret = _secrets.token_urlsafe(64),
        must_change_password = must_change_password,
    )
    return auth_storage.DEFAULT_ADMIN_USERNAME


def _commit_then_raise(monkeypatch, exc: Exception) -> None:
    """Make update_password commit for real and only then raise, reproducing a
    failure in the post-commit cleanup that runs outside the transaction."""
    real_update = auth_storage.update_password

    def _update(username, new_password, **kwargs):
        real_update(username, new_password, **kwargs)
        raise exc

    monkeypatch.setattr(auth_storage, "update_password", _update)


def test_auto_generate_keeps_credential_when_post_commit_cleanup_raises(real_auth_db, monkeypatch):
    # update_password commits the row BEFORE its best-effort bootstrap-file
    # cleanup, so a raise there leaves the generated password live while the
    # seeded recovery credential is already gone. Discarding it (the exception
    # used to propagate out of the gate) aborted the launch behind a password
    # nobody had ever seen -- an unrecoverable lockout. Resolve the partial
    # success against the stored hash instead, as the Colab path does.
    admin = _seed_real_admin()
    _commit_then_raise(monkeypatch, OSError("database is locked"))
    console = _Stream(isatty = True)

    generated = run._auto_generate_admin_password(admin, out = console)

    assert isinstance(generated, str) and generated
    from auth import hashing as _hashing

    salt, pwd_hash, _jwt, _mc = auth_storage.get_user_and_secret(admin)
    # The returned value is exactly the one that landed, so the banner shows a
    # password that actually authenticates.
    assert _hashing.verify_password(generated, salt, pwd_hash) is True
    assert auth_storage.requires_password_change(admin) is False
    notice = console.getvalue()
    assert "reported an error" in notice
    assert generated not in notice  # the notice never repeats the credential


def test_auto_generate_reraises_when_the_commit_itself_fails(real_auth_db, monkeypatch):
    # The converse: nothing was committed, so the seeded credential is still the
    # live one. Returning None here would tell the gate "someone else set a
    # password" and publish the tunnel; re-raise so the launch fails closed.
    admin = _seed_real_admin()

    def _explode(username, new_password, **kwargs):
        raise OSError("database is locked")

    monkeypatch.setattr(auth_storage, "update_password", _explode)

    with pytest.raises(OSError):
        run._auto_generate_admin_password(admin)
    assert auth_storage.requires_password_change(admin) is True
    from auth import hashing as _hashing

    salt, pwd_hash, _jwt, _mc = auth_storage.get_user_and_secret(admin)
    assert _hashing.verify_password("bootstrap-secret-123", salt, pwd_hash) is True


def test_auto_generate_returns_none_when_the_compare_and_set_loses(real_auth_db, monkeypatch):
    # No exception, just a lost CAS (another tab completed /change-password): the
    # generated value was never written, so it must not be shown.
    admin = _seed_real_admin(must_change_password = False)
    assert run._auto_generate_admin_password(admin) is None


def test_gate_surfaces_the_password_when_post_commit_cleanup_raises(real_auth_db, monkeypatch):
    # End to end on the direct `python run.py --secure` path: the gate must still
    # proceed and print the live credential instead of aborting after the rotation.
    _seed_real_admin()
    stderr = _patch_streams_autogen(monkeypatch)
    monkeypatch.delenv("UNSLOTH_STUDIO_BOOTSTRAP_TIMEOUT", raising = False)
    _commit_then_raise(monkeypatch, OSError("database is locked"))

    assert run._terminal_password_gate(tunnel_will_start = True, **_GATE_KWARGS) == (True, True)

    admin = auth_storage.DEFAULT_ADMIN_USERNAME
    out = stderr.getvalue()
    assert "auto-generated" in out
    printed = re.search(r"Password: (\S+)", out)
    assert printed is not None, out
    from auth import hashing as _hashing

    salt, pwd_hash, _jwt, _mc = auth_storage.get_user_and_secret(admin)
    assert _hashing.verify_password(printed.group(1), salt, pwd_hash) is True


def test_gate_aborts_when_the_rotation_never_landed(real_auth_db, monkeypatch):
    # Same shape, but the commit failed outright: the gate must not swallow it and
    # publish under the seeded credential.
    admin = _seed_real_admin()
    _patch_streams_autogen(monkeypatch)
    monkeypatch.delenv("UNSLOTH_STUDIO_BOOTSTRAP_TIMEOUT", raising = False)

    def _explode(username, new_password, **kwargs):
        raise OSError("database is locked")

    monkeypatch.setattr(auth_storage, "update_password", _explode)

    with pytest.raises(OSError):
        run._terminal_password_gate(tunnel_will_start = True, **_GATE_KWARGS)
    assert auth_storage.requires_password_change(admin) is True


class _DyingStream(_Stream):
    """A terminal that passes the preflight and then raises on write.

    Models the console going away between _one_time_secret_stream()'s checks and
    the post-commit banner: an SSH session that drops, or a closed terminal window
    whose orphaned pty fails writes with OSError EIO.
    """

    def __init__(self, isatty: bool = True):
        super().__init__(isatty = isatty)
        self.writes = 0

    def write(self, data):
        self.writes += 1
        raise OSError(5, "Input/output error")


def test_delivery_retries_the_other_console_when_the_first_write_dies(monkeypatch):
    # Regression (Codex 3651035060, P2): the banner is written AFTER
    # update_password committed the generated password and deleted the seeded
    # bootstrap credential, so a write failure there used to propagate and abort
    # the launch with a live password nobody had ever seen. Retry the other real
    # terminal instead of losing the only copy.
    dying = _DyingStream()
    alive = _Stream(isatty = True)
    monkeypatch.setattr(sys, "stderr", dying)
    monkeypatch.setattr(sys, "stdout", alive)

    assert run._deliver_one_time_credential("unsloth", "Retry-Console-Pw-1", out = dying) is True
    assert dying.writes >= 1  # the dead console was tried first
    assert "Retry-Console-Pw-1" in alive.getvalue()


def test_delivery_never_falls_back_to_a_tee_or_a_redirected_stream(monkeypatch):
    # The retry must not downgrade the surface: the only other candidates here are
    # a tee'd session log (retained on disk) and a redirected non-tty file, both of
    # which would PERSIST the plaintext (CWE-532). Report failure instead.
    log_fh = io.StringIO()
    dying = _DyingStream()
    redirected = _Stream(isatty = False)  # `> server.log`
    monkeypatch.setattr(sys, "stderr", run._TeeStream(dying, log_fh))
    monkeypatch.setattr(sys, "stdout", run._TeeStream(redirected, log_fh))

    assert run._deliver_one_time_credential("unsloth", "No-Tee-Pw-2", out = dying) is False
    assert log_fh.getvalue() == ""
    assert redirected.getvalue() == ""


def test_gate_fails_closed_when_the_credential_cannot_be_delivered(real_auth_db, monkeypatch):
    # End to end: every console dies after the preflight. The gate must not raise
    # (an unhandled OSError aborted startup with an opaque traceback); it fails
    # closed with a secret-free message so the operator knows to run
    # `unsloth studio reset-password`.
    _seed_real_admin()
    monkeypatch.setattr(sys, "stdin", _Stream(isatty = False))
    monkeypatch.setattr(sys, "stderr", _DyingStream())
    monkeypatch.setattr(sys, "stdout", _DyingStream())
    monkeypatch.delenv("UNSLOTH_STUDIO_BOOTSTRAP_TIMEOUT", raising = False)
    logged: list[str] = []
    monkeypatch.setattr(run.logger, "error", lambda msg, *a, **k: logged.append(str(msg)))

    assert run._terminal_password_gate(tunnel_will_start = True, **_GATE_KWARGS) == (False, False)

    assert logged and "reset-password" in logged[0]
    # The message names no credential: it is written through the logger, which the
    # session-log tee persists.
    assert "Password:" not in logged[0]
