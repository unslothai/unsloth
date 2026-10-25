# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio

import httpx
import pytest
from fastapi import Depends, Request
from fastapi.testclient import TestClient
from fastmcp import FastMCP

from auth import storage
from auth.authentication import get_current_subject
from studio_mcp.caller import current_caller
from studio_mcp.forward import FORWARD_BASE_URL, checked_path, forward, ndjson_last, parse_json
from utils import keyless_api_access
from utils.account_context import AccountContext, bind_account, current_account, reset_account
from utils.client_ip import client_ip, is_direct_local_request

from .mcp_harness import (
    call_tool,
    fake_studio,
    isolated_auth,  # noqa: F401  (fixture)
    make_caller,
    owner_auth,  # noqa: F401  (fixture)
    served,
)

PROBE_PATH = "/api/inference/__mcp_forward_probe"


@pytest.fixture
def studio_with_probe(monkeypatch, owner_auth):
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


def test_keyless_full_on_loopback_still_works_for_a_direct_caller(studio_with_probe):
    local = TestClient(
        studio_with_probe, base_url = "http://127.0.0.1:8888", client = ("127.0.0.1", 50000)
    )
    response = local.get(PROBE_PATH)
    assert response.status_code == 200, response.text
    assert response.json()["keyless_request_allowed"] is True


@pytest.mark.parametrize("token", ["", "not-needed", "lm-studio", "ollama", "no-key-required"])
def test_a_forwarded_call_cannot_use_keyless(studio_with_probe, token):
    response = asyncio.run(
        forward(make_caller(studio_app = studio_with_probe, token = token), "GET", PROBE_PATH)
    )
    assert response.status_code == 401


def test_a_forwarded_call_is_remote_keyed_and_leaves_the_tools_account_alone(studio_with_probe):
    raw_key, _row = storage.create_api_key(storage.DEFAULT_ADMIN_USERNAME, name = "agent")
    sentinel = AccountContext("sentinel-account", "sentinel")

    async def run():
        token = bind_account(sentinel)
        try:
            caller = make_caller(studio_app = studio_with_probe, token = raw_key)
            return await forward(caller, "GET", PROBE_PATH), current_account()
        finally:
            reset_account(token)

    response, after = asyncio.run(run())
    assert response.status_code == 200, response.text
    seen = response.json()
    assert seen["subject"] == storage.DEFAULT_ADMIN_USERNAME
    assert seen["account"] == "owner"
    assert seen["client_ip"] == "192.0.2.1"
    assert seen["is_direct_local_request"] is False
    assert seen["keyless_request_allowed"] is False
    assert seen["headers"]["host"] == "unsloth-mcp.invalid"
    assert after == sentinel


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


def test_only_the_key_and_an_opted_in_hub_token_are_forwarded(monkeypatch):
    studio, app = _echo_server(monkeypatch)
    inbound = {
        "CF-Connecting-IP": "127.0.0.1",
        "Cookie": "session=abc",
        "X-Forwarded-For": "127.0.0.1",
        "X-Unsloth-HF-Token": "hf_secret",
        "X-Custom": "1",
    }
    with TestClient(app, base_url = "http://127.0.0.1:8888", client = ("127.0.0.1", 50000)) as http:
        assert call_tool(http, "echo", headers = inbound)["structuredContent"] == {"status": 200}
        call_tool(http, "echo", {"hub": True}, headers = {"X-Unsloth-HF-Token": "hf_secret"})
        call_tool(http, "echo", {"hub": True})
    (_method, path, headers, _body), with_hub, without_hub = studio.state.calls
    assert path == "/api/inference/echo"
    assert headers["authorization"] == "Bearer sk-unsloth-test"
    assert headers["host"] == "unsloth-mcp.invalid"
    for name in ("cf-connecting-ip", "cookie", "x-forwarded-for", "x-custom", "x-unsloth-hf-token"):
        assert name not in headers
    assert with_hub[2]["x-unsloth-hf-token"] == "hf_secret"
    assert "x-unsloth-hf-token" not in without_hub[2]


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
    caller = make_caller(studio_app = None)

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
        "/api/hub/active-downloads",
        "/api/hub/gguf-download-progress",
        "/api/hub/token/validate",
        "/api/hub/transport-status",
        "/api/hub/cached-gguf",
        "/api/hub/download/../token",
    ],
)
def test_a_path_outside_the_allowlist_raises(path):
    caller = make_caller(studio_app = None)
    with pytest.raises(RuntimeError):
        asyncio.run(forward(caller, "GET", path))


@pytest.mark.parametrize(
    "path",
    [
        "/v1/models",
        "/api/inference/status",
        "/api/hub/gguf-variants",
        "/api/settings/embedding-model",
        "/api/hub/download",
        "/api/hub/download-status",
        "/api/hub/download-progress",
        "/api/hub/download/cancel",
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
            forward(make_caller(studio_app = studio), "GET", "/api/inference/hang")
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
            forward(make_caller(studio_app = studio_with_probe, token = key), "GET", PROBE_PATH)
            for key in (owner_key, alice_key) * 4
        ]
        responses = await asyncio.gather(*calls)
        return [response.json()["account"] for response in responses], current_account()

    accounts, after = asyncio.run(run())
    assert accounts == ["owner", alice] * 4
    assert after.account_id == "owner"
