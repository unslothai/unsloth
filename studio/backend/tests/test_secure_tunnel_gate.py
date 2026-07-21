# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Cloudflare tunnel start gate, incl. --secure on loopback. Imports run.py
directly, so run under the Unsloth venv."""

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
        # Non-secure wildcard binds tunnel only when --cloudflare is passed (True).
        (True, "0.0.0.0", False, False, False, True),
        (True, "::", False, False, False, True),
        (True, "127.0.0.1", False, False, False, False),
        (True, "localhost", False, False, False, False),
        # --secure tunnels a loopback bind too.
        (True, "127.0.0.1", True, False, False, True),
        (True, "0.0.0.0", True, False, False, True),
        # --no-cloudflare always wins.
        (False, "0.0.0.0", False, False, False, False),
        (False, "::", False, False, False, False),
        (False, "127.0.0.1", True, False, False, False),
        # Unset (None, no flag) behaves as off for non-secure binds.
        (None, "0.0.0.0", False, False, False, False),
        (None, "::", False, False, False, False),
        (None, "127.0.0.1", False, False, False, False),
        # Non-secure api-only never tunnels (Tauri).
        (True, "0.0.0.0", False, True, False, False),
        (True, "::", False, True, False, False),
        # --secure tunnels even api-only (headless secure API server).
        (True, "127.0.0.1", True, True, False, True),
        # Colab never tunnels, even --secure.
        (True, "0.0.0.0", False, False, True, False),
        (True, "::", False, False, True, False),
        (True, "127.0.0.1", True, False, True, False),
        (True, "127.0.0.1", True, True, True, False),
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


def test_arg_parser_secure_polarity_and_not_secure_alias():
    # --secure/--no-secure is the documented flag; --not-secure is a hidden,
    # back-compat alias for --no-secure. Last flag wins (BooleanOptionalAction).
    import run

    parser = run._build_arg_parser()
    assert parser.parse_args([]).secure is False
    assert parser.parse_args(["--secure"]).secure is True
    assert parser.parse_args(["--no-secure"]).secure is False
    assert parser.parse_args(["--not-secure"]).secure is False
    assert parser.parse_args(["--secure", "--not-secure"]).secure is False
    assert parser.parse_args(["--not-secure", "--secure"]).secure is True


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
    monkeypatch.setattr(run, "_print_cloudflare_line", lambda *a, **k: None)
    monkeypatch.setattr(run, "_localhost_ipv6_mismatch_url", lambda *a, **k: None)

    run._emit_startup_output(
        "0.0.0.0", 8000, "0.0.0.0", secure = False, enable_tools = None
    )
    out = capsys.readouterr().out
    assert "Server-side tools" in out
    assert "network-reachable" in out


def test_startup_output_emits_disabled_notice(capsys, monkeypatch):
    import run

    monkeypatch.setattr(run, "_localhost_ipv6_mismatch_url", lambda *a, **k: None)
    run._emit_startup_output(
        "127.0.0.1", 8000, "127.0.0.1", secure = False, enable_tools = False
    )
    out = capsys.readouterr().out
    assert "Server-side tools are DISABLED" in out


def test_run_server_rejects_secure_without_cloudflare():
    # Direct backend callers (not just the CLI) must reject the contradictory
    # combo: --secure asks for the tunnel, --no-cloudflare (cloudflare=False) forbids it.
    import run
    with pytest.raises(SystemExit) as exc:
        run.run_server(secure = True, cloudflare = False)
    assert "do not combine it with --no-cloudflare" in str(exc.value)


def test_failclosed_message_present_in_source():
    # The exact, user-facing fail-closed message must not drift.
    src = (_BACKEND / "run.py").read_text(encoding = "utf-8")
    assert (
        "A secure Cloudflare link is not allowed, use --no-secure which provides a 0.0.0.0 link"
        in src
    )


@pytest.mark.parametrize(
    "api_only,secure,expected",
    [
        (False, False, ["*"]),  # plain server: any origin
        (False, True, ["*"]),  # secure UI server: any origin
        (True, True, ["*"]),  # secure api-only: remote browsers need any origin
        (True, False, "tauri"),  # local api-only: locked to the Tauri app
    ],
)
def test_cors_origins_for_mode(api_only, secure, expected):
    from utils.host_policy import cors_origins_for_mode
    origins = cors_origins_for_mode(api_only = api_only, secure = secure)
    if expected == "tauri":
        assert origins != ["*"] and any(o.startswith("tauri://") for o in origins)
    else:
        assert origins == expected


def test_run_server_exports_secure_env_for_cors():
    # run_server must export UNSLOTH_SECURE before importing main so the CORS
    # profile can tell remote secure serving from local Tauri use.
    src = (_BACKEND / "run.py").read_text(encoding = "utf-8")
    assert 'os.environ["UNSLOTH_SECURE"] = "1"' in src


def test_run_server_emit_tauri_port_defaults_on():
    # Default on keeps the desktop app's stdout contract; the headless
    # `run --api-only` path opts out explicitly.
    import inspect

    import run

    params = inspect.signature(run.run_server).parameters
    assert "emit_tauri_port" in params
    assert params["emit_tauri_port"].default is True


def test_tauri_port_print_is_gated_in_source():
    # The TAURI_PORT line must depend on emit_tauri_port, not api_only alone.
    src = (_BACKEND / "run.py").read_text(encoding = "utf-8")
    assert "if api_only and emit_tauri_port:" in src
