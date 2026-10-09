# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio

import pytest
from fastapi.testclient import TestClient
from starlette.applications import Starlette
from starlette.routing import Mount

from auth import storage
from mcp_server import BearerTokenMiddleware, create_studio_mcp
from studio_mcp.gate import StudioMcpGate
from utils import mcp_access
from utils.keyless_api_access import _reset_scope_cache
from utils.mcp_access import ENV_FORCE, set_mcp_enabled

MCP_HEADERS = {"Accept": "application/json, text/event-stream"}
LISTING = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}


@pytest.fixture(autouse = True)
def isolated_state(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "auth.db")
    monkeypatch.setattr(storage, "_BOOTSTRAP_PW_PATH", tmp_path / ".bootstrap_password")
    monkeypatch.setattr(storage, "_bootstrap_password", None)
    monkeypatch.setattr(storage, "_api_key_pbkdf2_salt_cache", None)
    monkeypatch.delenv(ENV_FORCE, raising = False)
    storage._reset_api_key_hash_cache()
    _reset_scope_cache()
    mcp_access._reset_cache()
    yield
    storage._reset_api_key_hash_cache()
    _reset_scope_cache()
    mcp_access._reset_cache()


@pytest.fixture
def studio():
    from main import app
    return TestClient(app)


@pytest.fixture
def key_spy(monkeypatch):
    calls = []
    real = storage.validate_api_key_account

    def spy(*args, **kwargs):
        calls.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(storage, "validate_api_key_account", spy)
    return calls


@pytest.mark.parametrize("method", ["GET", "POST", "PUT", "DELETE", "PATCH"])
@pytest.mark.parametrize("path", ["/mcp/", "/mcp/x"])
@pytest.mark.parametrize("authorization", [None, "Bearer sk-unsloth-0123456789abcdef", "Bearer x"])
def test_off_is_a_plain_404_before_any_auth(studio, key_spy, method, path, authorization):
    headers = {**MCP_HEADERS, **({"Authorization": authorization} if authorization else {})}
    response = studio.request(method, path, json = LISTING, headers = headers)
    assert response.status_code == 404
    assert response.json() == {"detail": "Not Found"}
    assert "www-authenticate" not in response.headers
    assert key_spy == []


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("method", ["GET", "POST"])
def test_decisions_without_the_slash_still_redirects(studio, enabled, method):
    if enabled:
        set_mcp_enabled(True)
    response = studio.request(method, "/mcp/decisions?x=1", follow_redirects = False)
    assert response.status_code == 307
    assert response.headers["location"] == "http://testserver/mcp/decisions/?x=1"


@pytest.mark.parametrize("enabled", [False, True])
def test_decisions_with_the_slash_keeps_its_own_auth(studio, enabled):
    if enabled:
        set_mcp_enabled(True)
    response = studio.post("/mcp/decisions/", json = LISTING, headers = MCP_HEADERS)
    assert response.status_code == 401


def test_on_without_a_static_token_refuses_everyone(studio):
    set_mcp_enabled(True)
    response = studio.post(
        "/mcp/", json = LISTING, headers = {**MCP_HEADERS, "Authorization": "Bearer anything"}
    )
    assert response.status_code == 401


def test_the_switch_flips_live_without_a_restart(studio):
    assert studio.post("/mcp/", json = LISTING, headers = MCP_HEADERS).status_code == 404
    set_mcp_enabled(True)
    assert studio.post("/mcp/", json = LISTING, headers = MCP_HEADERS).status_code == 401
    set_mcp_enabled(False)
    assert studio.post("/mcp/", json = LISTING, headers = MCP_HEADERS).status_code == 404


def test_the_env_var_forces_it_on(studio, monkeypatch):
    monkeypatch.setenv(ENV_FORCE, "1")
    assert studio.post("/mcp/", json = LISTING, headers = MCP_HEADERS).status_code == 401


def _run_gate(scope):
    events = []

    async def inner(scope, receive, send):
        events.append("app")

    async def send(message):
        events.append(message)

    asyncio.run(StudioMcpGate(inner)(scope, None, send))
    return events


def test_lifespan_passes_through():
    assert _run_gate({"type": "lifespan"}) == ["app"]


def test_a_websocket_is_closed():
    set_mcp_enabled(True)
    scope = {"type": "websocket", "path": "/mcp/", "root_path": "/mcp", "headers": []}
    assert _run_gate(scope) == [{"type": "websocket.close", "code": 4404}]


def test_the_mcp_app_is_stateless_and_streams():
    set_mcp_enabled(True)
    mcp_app = create_studio_mcp().http_app(path = "/", stateless_http = True)
    served = Starlette(
        routes = [Mount("/mcp", StudioMcpGate(BearerTokenMiddleware(mcp_app, "static-token")))],
        lifespan = mcp_app.lifespan,
    )
    headers = {**MCP_HEADERS, "Authorization": "Bearer static-token"}
    with TestClient(served) as http:
        first = http.post("/mcp/", json = LISTING, headers = headers)
        second = http.post("/mcp/", json = {**LISTING, "id": 2}, headers = headers)
    for response in (first, second):
        assert response.status_code == 200, response.text
        assert "mcp-session-id" not in response.headers
        assert response.headers["content-type"].startswith("text/event-stream")
        assert '"tools"' in response.text
