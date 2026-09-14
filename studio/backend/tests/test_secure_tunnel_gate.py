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
        (True, "::0", False, False, False, True),
        (True, "0:0:0:0:0:0:0:0", False, False, False, True),
        (True, "0", False, False, False, True),
        (True, "::ffff:0.0.0.0", False, False, False, True),
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


@pytest.mark.parametrize(
    "inherited,cloudflare,secure,expected",
    [
        ("unset", False, False, "unset"),
        ("disabled", False, False, "disabled"),
        (None, False, False, "disabled"),
        (None, None, False, "unset"),
        (None, None, True, "enabled"),
    ],
)
def test_cloudflare_intent_preserves_compatibility_provenance(
    monkeypatch, inherited, cloudflare, secure, expected
):
    import run

    if inherited is None:
        monkeypatch.delenv(run._CLOUDFLARE_INTENT_ENV, raising = False)
    else:
        monkeypatch.setenv(run._CLOUDFLARE_INTENT_ENV, inherited)
    assert run._consume_cloudflare_intent(cloudflare, secure) == expected
    assert run._CLOUDFLARE_INTENT_ENV not in run.os.environ


def test_run_server_accepts_secure_kwarg():
    import inspect

    import run

    assert "secure" in inspect.signature(run.run_server).parameters
    assert inspect.signature(run.run_server).parameters["secure"].default is False


def test_final_bound_port_uses_uvicorn_listener_for_ephemeral_bind():
    from types import SimpleNamespace

    import run

    sock = SimpleNamespace(getsockname = lambda: ("127.0.0.1", 43123))
    server = SimpleNamespace(servers = [SimpleNamespace(sockets = [sock])])
    assert run._final_bound_port(server, 0) == 43123
    assert run._final_bound_port(server, 8888) == 8888
    source = (_BACKEND / "run.py").read_text(encoding = "utf-8")
    resolved = source.index("port = _final_bound_port(_server, port)")
    assert all(
        resolved < source.index(consumer, resolved)
        for consumer in (
            "app.state.server_request_host",
            "app.state.remote_access_port",
            "TAURI_PORT={port}",
            "start_studio_tunnel(",
        )
    )
    assert "origin_host = app.state.server_request_host" in source[resolved:]


@pytest.mark.parametrize(
    "address,expected",
    [
        (("0.0.0.0", 43123), "127.0.0.1"),
        (("::", 43123, 0, 0), "::1"),
        (("192.0.2.24", 43123), "192.0.2.24"),
        (("fe80::1234", 43123, 0, 7), "fe80::1234%7"),
    ],
)
def test_bound_request_host_uses_the_active_listener_address(address, expected):
    from types import SimpleNamespace

    import run

    sock = SimpleNamespace(getsockname = lambda: address)
    server = SimpleNamespace(servers = [SimpleNamespace(sockets = [sock])])
    assert run._bound_request_host(server) == expected


def test_bound_request_host_fails_closed_without_a_listener():
    from types import SimpleNamespace

    import run
    with pytest.raises(RuntimeError, match = "did not expose its bound address"):
        run._bound_request_host(SimpleNamespace(servers = []))


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


def test_arg_parser_dns_pinning_opt_out_defaults_off():
    import run

    parser = run._build_arg_parser()
    assert parser.parse_args([]).disable_dns_pinning is False
    assert parser.parse_args(["--disable-dns-pinning"]).disable_dns_pinning is True


def test_run_server_accepts_enable_tools_kwarg():
    import inspect

    import run

    params = inspect.signature(run.run_server).parameters
    assert "enable_tools" in params
    assert params["enable_tools"].default is None  # default: leave policy unset


def test_tool_policy_not_auto_disabled_by_bind():
    # No flag installs neither an override nor a tools-on default on any bind: the
    # default belongs to `unsloth studio run`, which installs it itself. The
    # backend never changes the policy from host/secure.
    import run
    from state.tool_policy import (
        get_tool_policy,
        get_tool_policy_default,
        reset_tool_policy,
    )

    for host in ("127.0.0.1", "localhost", "0.0.0.0"):
        reset_tool_policy()
        run._apply_cli_tool_policy(None)  # no flag, on any bind
        assert get_tool_policy() is None, host  # no override: request off honored
        assert get_tool_policy_default() is None, host  # no default from this path

    reset_tool_policy()
    run._apply_cli_tool_policy(True)  # --enable-tools: forced on
    assert get_tool_policy() is True

    reset_tool_policy()
    run._apply_cli_tool_policy(False)  # --disable-tools: forced off
    assert get_tool_policy() is False
    reset_tool_policy()


def test_apply_cli_tool_policy_is_idempotent():
    # Both the CLI and run_server() apply the pair; re-applying must not drift.
    import run
    from state.tool_policy import get_tool_policy, get_tool_policy_default, reset_tool_policy

    for flag in (None, True, False):
        reset_tool_policy()
        run._apply_cli_tool_policy(flag)
        first = (get_tool_policy(), get_tool_policy_default())
        run._apply_cli_tool_policy(flag)
        assert (get_tool_policy(), get_tool_policy_default()) == first, flag
    reset_tool_policy()


def test_tool_policy_notice_wording():
    # The plain-server startup banner states the resolved policy for every bind.
    import run

    # No flag on this launcher: no tools-on default, so the request decides.
    for host, secure_mode in (("127.0.0.1", False), ("0.0.0.0", False), ("127.0.0.1", True)):
        notice = run._tool_policy_notice(host, secure_mode, None)
        assert "follow each request's enable_tools" in notice, notice
        assert "--enable-tools to force them on" in notice, notice

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

    run._emit_startup_output("0.0.0.0", 8000, "0.0.0.0", secure = False, enable_tools = None)
    out = capsys.readouterr().out
    assert "Server-side tools" in out
    assert "follow each request's enable_tools" in out


def test_startup_output_emits_disabled_notice(capsys, monkeypatch):
    import run

    monkeypatch.setattr(run, "_localhost_ipv6_mismatch_url", lambda *a, **k: None)
    run._emit_startup_output("127.0.0.1", 8000, "127.0.0.1", secure = False, enable_tools = False)
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


def test_api_only_cors_tracks_published_public_url():
    from types import SimpleNamespace

    from starlette.datastructures import Headers

    from main import RemoteAccessCORSMiddleware

    state = SimpleNamespace(cloudflare_url = None)
    middleware = RemoteAccessCORSMiddleware(
        lambda *_: None,
        remote_access_state = state,
        allow_origins = ["tauri://localhost"],
        allow_credentials = True,
        allow_methods = ["*"],
        allow_headers = ["*"],
    )
    request = Headers(
        {
            "origin": "https://browser-client.example",
            "access-control-request-method": "POST",
            "access-control-request-headers": "authorization,content-type",
        }
    )
    assert middleware.preflight_response(request).status_code == 400
    state.cloudflare_url = "https://public.trycloudflare.com"
    response = middleware.preflight_response(request)
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "https://browser-client.example"
    state.cloudflare_url = None
    assert middleware.preflight_response(request).status_code == 400


def test_run_server_exports_secure_env_for_cors():
    # run_server must export UNSLOTH_SECURE before importing main so the CORS
    # profile can tell remote secure serving from local Tauri use.
    src = (_BACKEND / "run.py").read_text(encoding = "utf-8")
    assert 'os.environ["UNSLOTH_SECURE"] = "1"' in src
    assert "set_studio_tunnel_runtime_callback(set_remote_connector_active)" in src
    main_src = (_BACKEND / "main.py").read_text(encoding = "utf-8")
    assert "RemoteAccessCORSMiddleware,\n    remote_access_state = app.state" in main_src


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


def test_cors_preflight_cache_window_is_short():
    # is_allowed_origin closes the instant the tunnel URL clears, but a preflight
    # the browser already cached does not. Measured in WebKit: with Starlette's
    # 600s default a state-changing request still reached the server after Stop.
    from types import SimpleNamespace

    from starlette.datastructures import Headers

    from main import RemoteAccessCORSMiddleware

    middleware = RemoteAccessCORSMiddleware(
        lambda *_: None,
        remote_access_state = SimpleNamespace(cloudflare_url = "https://x.trycloudflare.com"),
        allow_origins = ["tauri://localhost"],
        allow_credentials = True,
        allow_methods = ["*"],
        allow_headers = ["*"],
        max_age = 60,
    )
    response = middleware.preflight_response(
        Headers(
            {
                "origin": "https://browser-client.example",
                "access-control-request-method": "POST",
                "access-control-request-headers": "authorization",
            }
        )
    )
    assert response.status_code == 200
    assert int(response.headers["access-control-max-age"]) <= 60

    main_src = (_BACKEND / "main.py").read_text(encoding = "utf-8")
    assert "max_age = 60" in main_src, "the mounted middleware must pin max_age"
