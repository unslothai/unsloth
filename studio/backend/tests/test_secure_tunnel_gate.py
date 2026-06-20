# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Cloudflare tunnel start gate, incl. --secure on loopback. Imports run.py
directly, so run under the Studio venv."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from run import _cloudflare_tunnel_should_start as should_start  # noqa: E402


@pytest.mark.parametrize(
    "cloudflare,host,secure,api_only,is_colab,expected",
    [
        # Non-secure: historical 0.0.0.0-only behaviour preserved.
        (True, "0.0.0.0", False, False, False, True),
        (True, "127.0.0.1", False, False, False, False),
        (True, "localhost", False, False, False, False),
        # --secure tunnels a loopback bind too.
        (True, "127.0.0.1", True, False, False, True),
        (True, "0.0.0.0", True, False, False, True),
        # --no-cloudflare always wins.
        (False, "0.0.0.0", False, False, False, False),
        (False, "127.0.0.1", True, False, False, False),
        # api-only and Colab never tunnel.
        (True, "0.0.0.0", False, True, False, False),
        (True, "127.0.0.1", True, True, False, False),
        (True, "0.0.0.0", False, False, True, False),
        (True, "127.0.0.1", True, False, True, False),
    ],
)
def test_cloudflare_gate(cloudflare, host, secure, api_only, is_colab, expected):
    assert (
        should_start(
            cloudflare = cloudflare,
            host = host,
            secure = secure,
            api_only = api_only,
            is_colab = is_colab,
        )
        is expected
    )


def test_run_server_accepts_secure_kwarg():
    import inspect

    import run

    assert "secure" in inspect.signature(run.run_server).parameters
    assert inspect.signature(run.run_server).parameters["secure"].default is False


def test_run_server_accepts_enable_tools_kwarg():
    import inspect

    import run

    params = inspect.signature(run.run_server).parameters
    assert "enable_tools" in params
    assert params["enable_tools"].default is None  # default: leave policy unset


def test_tool_policy_not_auto_disabled_by_bind():
    # Tools default on for every bind; the backend only changes the policy from
    # an explicit --enable-tools/--disable-tools, never from host/secure.
    import run
    from state.tool_policy import get_tool_policy, reset_tool_policy

    for host in ("127.0.0.1", "localhost", "0.0.0.0"):
        reset_tool_policy()
        run._apply_cli_tool_policy(None)  # no flag, on any bind
        assert get_tool_policy() is None, host  # untouched: per-request honored

    reset_tool_policy()
    run._apply_cli_tool_policy(True)  # --enable-tools: forced on
    assert get_tool_policy() is True

    reset_tool_policy()
    run._apply_cli_tool_policy(False)  # --disable-tools: forced off
    assert get_tool_policy() is False
    reset_tool_policy()


def test_tool_policy_notice_wording():
    # The plain-server startup banner states the resolved policy for every bind.
    import run

    loopback = run._tool_policy_notice("127.0.0.1", False, None)
    assert "ENABLED by default" in loopback and "loopback" in loopback

    network = run._tool_policy_notice("0.0.0.0", False, None)
    assert "ENABLED by default" in network and "network-reachable" in network

    secure = run._tool_policy_notice("127.0.0.1", True, None)
    assert "Cloudflare HTTPS tunnel" in secure

    assert run._tool_policy_notice("0.0.0.0", False, False) == (
        "Server-side tools are DISABLED (--disable-tools)."
    )
    assert "ENABLED (--enable-tools)" in run._tool_policy_notice("0.0.0.0", False, True)


def test_startup_output_emits_tool_notice_on_network_bind(capsys, monkeypatch):
    # Plain `unsloth studio -H 0.0.0.0` must not be silent about tools now.
    import run

    monkeypatch.setattr(run, "_verify_global_reachability", lambda *a, **k: None)
    monkeypatch.setattr(run, "_print_cloudflare_line", lambda: None)
    monkeypatch.setattr(run, "_localhost_ipv6_mismatch_url", lambda *a, **k: None)

    run._emit_startup_output("0.0.0.0", 8000, "0.0.0.0", secure = False, enable_tools = None)
    out = capsys.readouterr().out
    assert "Server-side tools" in out
    assert "network-reachable" in out


def test_startup_output_emits_disabled_notice(capsys, monkeypatch):
    import run

    monkeypatch.setattr(run, "_localhost_ipv6_mismatch_url", lambda *a, **k: None)
    run._emit_startup_output("127.0.0.1", 8000, "127.0.0.1", secure = False, enable_tools = False)
    out = capsys.readouterr().out
    assert "Server-side tools are DISABLED" in out


def test_run_server_rejects_secure_without_cloudflare():
    # Direct backend callers (not just the CLI) must reject the contradictory combo.
    import run
    with pytest.raises(SystemExit) as exc:
        run.run_server(secure = True, cloudflare = False)
    assert "A secure Cloudflare link is not allowed" in str(exc.value)


def test_failclosed_message_present_in_source():
    # The exact, user-facing fail-closed message must not drift.
    src = (_BACKEND / "run.py").read_text(encoding = "utf-8")
    assert (
        "A secure Cloudflare link is not allowed, use --not-secure which provides a 0.0.0.0 link"
        in src
    )
