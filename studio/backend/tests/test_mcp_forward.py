# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import secrets

import httpx
import pytest
from fastapi import Depends, Request
from fastapi.testclient import TestClient
from fastmcp import FastMCP

from auth import policy, storage
from auth.authentication import get_current_subject
from studio_mcp.caller import Caller, current_caller
from studio_mcp.forward import (
    FORWARD_BASE_URL,
    checked_path,
    forward,
    ndjson_last,
    parse_json,
)
from utils import keyless_api_access
from utils.account_context import AccountContext, bind_account, current_account, reset_account
from utils.client_ip import client_ip, is_direct_local_request

from .mcp_harness import call_tool, fake_studio, served

PROBE_PATH = "/api/inference/__mcp_forward_probe"


@pytest.fixture
def auth_db(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "auth.db")
    monkeypatch.setattr(storage, "_BOOTSTRAP_PW_PATH", tmp_path / ".bootstrap_password")
    monkeypatch.setattr(storage, "_bootstrap_password", None)
    monkeypatch.setattr(storage, "_api_key_pbkdf2_salt_cache", None)
    storage._reset_api_key_hash_cache()
    policy.invalidate_account_cache()
    keyless_api_access._reset_scope_cache()
    storage.create_initial_user(
        username = storage.DEFAULT_ADMIN_USERNAME,
        password = "human-password-123",
        jwt_secret = secrets.token_urlsafe(64),
        must_change_password = False,
    )
    yield
    storage._reset_api_key_hash_cache()
    policy.invalidate_account_cache()
    keyless_api_access._reset_scope_cache()


@pytest.fixture
def studio_with_probe(monkeypatch, auth_db):
    """The real main.app with a throwaway route, keyless access at its widest and Studio on loopback."""
    import main

    app = main.app

    async def probe(request: Request, subject: str = Depends(get_current_subject)):
        return {
            "subject": subject,
            "account": current_account().account_id,
            "client_ip": client_ip(request),
            "is_direct_local_request": is_direct_local_request(request),
            "keyless_request_allowed": keyless_api_access.keyless_request_allowed(request),
            "headers": {
                name: value for name, value in request.headers.items() if name != "authorization"
            },
        }

    app.add_api_route(PROBE_PATH, probe, methods = ["GET"])
    routes = app.router.routes
    routes.insert(0, routes.pop())
    app.middleware_stack = None
    for name, value in {
        "bind_host": "127.0.0.1",
        "secure": False,
        "remote_access_is_colab": False,
        "lan_access_is_colab": False,
        "lan_access_secure_launch": False,
        "cloudflare_url": None,
    }.items():
        monkeypatch.setattr(app.state, name, value, raising = False)
    keyless_api_access.set_keyless_api_access("full", tools = True)
    yield app
    app.router.routes[:] = [r for r in app.router.routes if getattr(r, "path", None) != PROBE_PATH]
    app.middleware_stack = None


def _caller(
    app,
    token,
    hf_token = None,
):
    return Caller(
        token = token,
        account_id = "owner",
        direct_local = True,
        public_base = "http://127.0.0.1:8888",
        studio_app = app,
        hf_token = hf_token,
    )


def test_keyless_full_on_loopback_still_works_for_a_direct_caller(studio_with_probe):
    # Control: the same route admits a keyless caller that really is local.
    local = TestClient(
        studio_with_probe, base_url = "http://127.0.0.1:8888", client = ("127.0.0.1", 50000)
    )
    response = local.get(PROBE_PATH)
    assert response.status_code == 200, response.text
    assert response.json()["keyless_request_allowed"] is True


@pytest.mark.parametrize("token", ["", "not-needed", "lm-studio", "ollama", "no-key-required"])
def test_a_forwarded_call_cannot_use_keyless(studio_with_probe, token):
    response = asyncio.run(forward(_caller(studio_with_probe, token), "GET", PROBE_PATH))
    assert response.status_code == 401


def test_a_forwarded_call_is_remote_and_authenticated_by_the_key(studio_with_probe):
    raw_key, _row = storage.create_api_key(storage.DEFAULT_ADMIN_USERNAME, name = "agent")
    response = asyncio.run(forward(_caller(studio_with_probe, raw_key), "GET", PROBE_PATH))
    assert response.status_code == 200, response.text
    seen = response.json()
    assert seen["subject"] == storage.DEFAULT_ADMIN_USERNAME
    assert seen["client_ip"] == "192.0.2.1"
    assert seen["is_direct_local_request"] is False
    assert seen["keyless_request_allowed"] is False
    assert seen["headers"]["host"] == "unsloth-mcp.invalid"


def _echo_server(monkeypatch, routes = None):
    studio = fake_studio(
        {
            ("GET", "/api/inference/echo"): lambda request, body: {"ok": True},
            ("POST", "/v1/audio/inputs"): lambda request, body: {"size": len(body)},
            **(routes or {}),
        }
    )
    mcp = FastMCP("probe")

    @mcp.tool
    async def echo(hub: bool = False) -> dict:
        response = await forward(current_caller(), "GET", "/api/inference/echo", hub_header = hub)
        return {"status": response.status_code}

    @mcp.tool
    async def upload(size: int) -> dict:
        response = await forward(
            current_caller(),
            "POST",
            "/v1/audio/inputs",
            params = {"name": "a.wav"},
            content = b"x" * size,
        )
        return response.json()

    return studio, served(mcp, studio, monkeypatch = monkeypatch)


def test_only_the_key_is_forwarded_from_the_inbound_request(monkeypatch):
    studio, app = _echo_server(monkeypatch)
    inbound = {
        "CF-Connecting-IP": "127.0.0.1",
        "Cookie": "session=abc",
        "X-Forwarded-For": "127.0.0.1",
        "X-Unsloth-HF-Token": "hf_secret",
        "X-Custom": "1",
    }
    with TestClient(app, base_url = "http://127.0.0.1:8888", client = ("127.0.0.1", 50000)) as http:
        result = call_tool(http, "echo", headers = inbound)
    assert result["structuredContent"] == {"status": 200}
    (_method, path, headers, _body) = studio.state.calls[-1]
    assert path == "/api/inference/echo"
    assert headers["authorization"] == "Bearer sk-unsloth-test"
    assert headers["host"] == "unsloth-mcp.invalid"
    for name in ("cf-connecting-ip", "cookie", "x-forwarded-for", "x-custom", "x-unsloth-hf-token"):
        assert name not in headers


def test_the_hub_token_is_sent_only_when_the_call_opts_in(monkeypatch):
    studio, app = _echo_server(monkeypatch)
    with TestClient(app) as http:
        call_tool(http, "echo", {"hub": True}, headers = {"X-Unsloth-HF-Token": "hf_secret"})
        call_tool(http, "echo", {"hub": True})
    assert studio.state.calls[0][2]["x-unsloth-hf-token"] == "hf_secret"
    assert "x-unsloth-hf-token" not in studio.state.calls[1][2]


def test_a_bytes_upload_carries_content_length(monkeypatch):
    studio, app = _echo_server(monkeypatch)
    with TestClient(app) as http:
        result = call_tool(http, "upload", {"size": 4096})
    assert result["structuredContent"] == {"size": 4096}
    (_method, path, headers, body) = studio.state.calls[-1]
    assert path == "/v1/audio/inputs"
    assert headers["content-length"] == "4096"
    assert "transfer-encoding" not in headers
    assert len(body) == 4096


def test_a_streamed_upload_is_refused():
    caller = _caller(None, "sk-unsloth-test")

    async def chunks():
        yield b"x"

    with pytest.raises(TypeError):
        asyncio.run(forward(caller, "POST", "/v1/audio/inputs", content = chunks()))


@pytest.mark.parametrize(
    "path",
    [
        "/api/auth/api-keys",
        "/api/settings/sandbox",
        "/api/settings/mcp-access",
        "/api/hub/token",
        "/mcp/",
        "/v1/../api/auth/api-keys",
        "/api/inference/../../api/auth/api-keys",
        "/v1/videos/%2e%2e/%2e%2e/api/auth/api-keys",
        "/api/inference",
    ],
)
def test_a_path_outside_the_allowlist_raises(path):
    caller = _caller(None, "sk-unsloth-test")
    with pytest.raises(RuntimeError):
        asyncio.run(forward(caller, "GET", path))


@pytest.mark.parametrize(
    "path",
    [
        "/v1/models",
        "/api/inference/status",
        "/api/hub/gguf-variants",
        "/api/settings/embedding-model",
    ],
)
def test_allowlisted_paths_pass_the_check(path):
    assert checked_path(httpx.URL(FORWARD_BASE_URL + path)) == path


def test_padded_bodies_parse():
    assert parse_json(b'      \n  {"loaded": true}') == {"loaded": True}
    assert parse_json(b'{"a": 1}   ') == {"a": 1}
    assert ndjson_last(b'{"type":"progress"}\n{"type":"done","text":"hi"}\n\n') == {
        "type": "done",
        "text": "hi",
    }
    with pytest.raises(ValueError):
        ndjson_last(b"\n \n")


def test_the_routes_account_binding_never_reaches_the_tool(studio_with_probe):
    raw_key, _row = storage.create_api_key(storage.DEFAULT_ADMIN_USERNAME, name = "agent")
    sentinel = AccountContext("sentinel-account", "sentinel")

    async def run():
        token = bind_account(sentinel)
        try:
            response = await forward(_caller(studio_with_probe, raw_key), "GET", PROBE_PATH)
            return response.json()["account"], current_account()
        finally:
            reset_account(token)

    route_account, after = asyncio.run(run())
    assert route_account == "owner"
    assert after == sentinel


def test_cancelling_the_tool_cancels_the_forwarded_call():
    state = {}

    async def run():
        started = asyncio.Event()

        async def hang(request, body):
            started.set()
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                state["route_cancelled"] = True
                raise

        studio = fake_studio({("GET", "/api/inference/hang"): hang})
        tool = asyncio.create_task(
            forward(_caller(studio, "sk-unsloth-test"), "GET", "/api/inference/hang")
        )
        await asyncio.wait_for(started.wait(), 10)
        tool.cancel()
        with pytest.raises(asyncio.CancelledError):
            await tool
        await asyncio.sleep(0)

    asyncio.run(run())
    assert state == {"route_cancelled": True}


def test_concurrent_calls_under_different_keys_keep_their_own_account(studio_with_probe):
    owner_key, _row = storage.create_api_key(storage.DEFAULT_ADMIN_USERNAME, name = "owner-agent")
    alice = storage.issue_account_setup_code(username = "alice")["account"]["account_id"]
    alice_key, _row = storage.create_api_key("alice", name = "alice-agent", account_id = alice)

    async def run():
        calls = [
            forward(_caller(studio_with_probe, key), "GET", PROBE_PATH)
            for key in (owner_key, alice_key) * 4
        ]
        responses = await asyncio.gather(*calls)
        return [response.json()["account"] for response in responses], current_account()

    accounts, after = asyncio.run(run())
    assert accounts == ["owner", alice] * 4
    assert after.account_id == "owner"
