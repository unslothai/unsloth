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


def _patch_streams(monkeypatch, *, tty: bool) -> _Stream:
    stderr = _Stream(isatty = tty)
    monkeypatch.setattr(sys, "stdin", _Stream(isatty = tty))
    monkeypatch.setattr(sys, "stderr", stderr)
    return stderr


def test_gate_skips_when_tunnel_off(monkeypatch):
    # Short-circuits before touching auth storage at all.
    def _boom(*a, **k):
        raise AssertionError("storage must not be consulted when the tunnel is off")

    monkeypatch.setattr(auth_storage, "requires_password_change", _boom)
    assert run._terminal_password_gate(tunnel_will_start = False) == (True, False)


def test_gate_skips_when_password_already_changed(monkeypatch):
    _patch_streams(monkeypatch, tty = True)
    monkeypatch.setattr(auth_storage, "requires_password_change", lambda u: False)
    monkeypatch.setattr(
        terminal_prompt,
        "prompt_for_password_change",
        lambda **k: pytest.fail("prompt must not run when no change is required"),
    )
    assert run._terminal_password_gate(tunnel_will_start = True) == (True, False)


def test_gate_warns_and_proceeds_without_tty(monkeypatch):
    stderr = _patch_streams(monkeypatch, tty = False)
    monkeypatch.setattr(auth_storage, "requires_password_change", lambda u: True)
    monkeypatch.setattr(
        terminal_prompt,
        "prompt_for_password_change",
        lambda **k: pytest.fail("prompt must not run without a tty"),
    )
    assert run._terminal_password_gate(tunnel_will_start = True) == (True, False)
    out = stderr.getvalue()
    assert "default admin password is still active" in out
    assert "UNSLOTH_STUDIO_BOOTSTRAP_TIMEOUT" in out


def test_gate_refusal_fails_closed(monkeypatch):
    _patch_streams(monkeypatch, tty = True)
    monkeypatch.setattr(auth_storage, "requires_password_change", lambda u: True)
    monkeypatch.setattr(terminal_prompt, "prompt_for_password_change", lambda **k: False)
    assert run._terminal_password_gate(tunnel_will_start = True) == (False, False)


def test_gate_success_applies_route_equivalent_change(monkeypatch):
    _patch_streams(monkeypatch, tty = True)
    calls = []
    monkeypatch.setattr(auth_storage, "requires_password_change", lambda u: True)
    monkeypatch.setattr(
        auth_storage,
        "get_user_and_secret",
        lambda u: ("salt", "hash", "jwt", True),
    )
    monkeypatch.setattr(
        auth_storage,
        "update_password",
        lambda u, p: calls.append(("update", u, p)),
    )
    monkeypatch.setattr(
        auth_storage,
        "revoke_user_refresh_tokens",
        lambda u: calls.append(("revoke", u)),
    )

    def _fake_prompt(*, min_length, is_current_password, apply_change, out):
        # The gate wires the policy constant and route-equivalent apply hook.
        assert min_length == auth_storage.MIN_PASSWORD_LENGTH
        # Wired to the real hash comparison: a wrong guess is rejected.
        assert is_current_password("wrong-guess") is False
        apply_change("brand-new-password")
        return True

    monkeypatch.setattr(terminal_prompt, "prompt_for_password_change", _fake_prompt)
    assert run._terminal_password_gate(tunnel_will_start = True) == (True, True)
    admin = auth_storage.DEFAULT_ADMIN_USERNAME
    assert ("update", admin, "brand-new-password") in calls
    assert ("revoke", admin) in calls
    # Revocation must happen with (not before) the update, mirroring the route.
    assert calls.index(("update", admin, "brand-new-password")) < calls.index(
        ("revoke", admin)
    )


# ── ordering inside run_server (source-level, repo convention) ───────


def test_gate_runs_before_tunnel_start_in_source():
    src = (_BACKEND / "run.py").read_text(encoding = "utf-8")
    gate_call = src.index("_pw_proceed, _pw_changed = _terminal_password_gate(")
    tunnel_start = src.index("_cloudflare_url = start_studio_tunnel(port)")
    assert gate_call < tunnel_start, (
        "the terminal password gate must run before start_studio_tunnel"
    )
    # The fail-closed branch mirrors the secure gate: shutdown + exit(1).
    refusal = src[gate_call:tunnel_start]
    assert "_graceful_shutdown(_server)" in refusal
    assert "sys.exit(1)" in refusal


def test_gate_runs_after_tunnel_decision_in_source():
    # The gate consumes the real tunnel decision (no prompt for loopback
    # --cloudflare no-ops), so it must come after the gate computation.
    src = (_BACKEND / "run.py").read_text(encoding = "utf-8")
    decision = src.index("_cloudflare_enabled = _cloudflare_tunnel_should_start(")
    gate_call = src.index("_pw_proceed, _pw_changed = _terminal_password_gate(")
    assert decision < gate_call


def test_min_password_length_single_source():
    # models/auth.py must reference the storage constant, not a literal.
    models_src = (_BACKEND / "models" / "auth.py").read_text(encoding = "utf-8")
    assert "MIN_PASSWORD_LENGTH" in models_src
    assert not re.search(r"min_length\s*=\s*8\b", models_src)
    assert auth_storage.MIN_PASSWORD_LENGTH == 8
