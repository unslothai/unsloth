# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Pre-tunnel terminal password gate: never publish a public Cloudflare URL
while the seeded default admin password is active. Imports run.py directly,
so run under the Studio venv."""

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
    # The gate must run before the uvicorn socket binds: on a wildcard bind
    # the served HTML injects the bootstrap credential for first login, so a
    # pre-gate listener would hand out the default password mid-prompt.
    src = (_BACKEND / "run.py").read_text(encoding = "utf-8")
    gate_call = src.index("_pw_proceed, _pw_drop_bootstrap = _terminal_password_gate(")
    thread_start = src.index("thread.start()")
    tunnel_start = src.index("_cloudflare_url = start_studio_tunnel(port)")
    assert gate_call < thread_start < tunnel_start
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
    # generate_bootstrap_password() after a later reset-password deletes auth.db,
    # which would re-validate the revoked bootstrap password.
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
