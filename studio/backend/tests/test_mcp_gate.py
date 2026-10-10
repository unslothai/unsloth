# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.applications import Starlette
from starlette.routing import Mount

from fastmcp import FastMCP

from auth import storage
from auth.authentication import create_access_token
from mcp_server import create_studio_mcp
from studio_mcp.caller import current_caller
from studio_mcp.gate import MAX_REQUEST_BYTES, NEED_KEY, TOO_LARGE, StudioMcpGate
from utils.account_context import AccountContext, bind_account, current_account, reset_account
from utils.keyless_api_access import APPROVED_DUMMY_BEARERS, KEYLESS_SCOPES, set_keyless_api_access
from utils.mcp_access import ENV_FORCE, set_mcp_enabled

from .mcp_harness import LOCAL, MCP_HEADERS, REMOTE, isolated_auth, seed_owner  # noqa: F401  (fixture)

LISTING = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
# Every tool call here asks the probe's one tool, whoami, which takes no arguments.
WHOAMI = {**LISTING, "id": 7, "method": "tools/call", "params": {"name": "whoami", "arguments": {}}}
pytestmark = pytest.mark.usefixtures("isolated_auth")


def owner_key(**kwargs):
    return storage.create_api_key(storage.DEFAULT_ADMIN_USERNAME, name = "agent", **kwargs)


@pytest.fixture
def live_key():
    """The owner's API key, with the switch on."""
    seed_owner()
    raw_key, _row = owner_key()
    set_mcp_enabled(True)
    return raw_key


def probe_mcp():
    mcp = FastMCP("probe")

    @mcp.tool
    async def whoami() -> dict:
        caller = current_caller()
        return {
            "token": caller.token,
            "account_id": caller.account_id,
            "direct_local": caller.direct_local,
            "public_base": caller.public_base,
            "hf_token": caller.hf_token,
            "has_studio_app": caller.studio_app is not None,
            "bound_account": current_account().account_id,
        }

    return mcp


def served(mcp, **state):
    mcp_app = mcp.http_app(path = "/", stateless_http = True)
    app = Starlette(routes = [Mount("/mcp", StudioMcpGate(mcp_app))], lifespan = mcp_app.lifespan)
    for name, value in {"server_port": 8888, "cloudflare_url": None, **state}.items():
        setattr(app.state, name, value)
    return app


def call_tool(http, headers):
    response = http.post("/mcp/", json = WHOAMI, headers = {**MCP_HEADERS, **headers})
    if response.status_code != 200:
        return response, None
    data = [line[5:].strip() for line in response.text.splitlines() if line.startswith("data:")]
    return response, json.loads(data[-1])["result"]["structuredContent"]


def bearer(token):
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def studio():
    from main import app
    return TestClient(app)


@pytest.fixture
def key_spy(monkeypatch):
    calls = []
    real = storage.validate_api_key_account

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(storage, "validate_api_key_account", spy)
    return calls


@pytest.mark.parametrize("method", ["GET", "POST", "PUT", "DELETE", "PATCH"])
@pytest.mark.parametrize("path", ["/mcp/", "/mcp/x"])
@pytest.mark.parametrize("authorization", [None, "Bearer sk-unsloth-0123456789abcdef", "Bearer x"])
@pytest.mark.parametrize("origin", [None, "https://evil.example"])
def test_off_is_a_plain_404_before_any_auth(studio, key_spy, method, path, authorization, origin):
    headers = {**MCP_HEADERS, **({"Authorization": authorization} if authorization else {})}
    if origin:
        headers["Origin"] = origin
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


def test_the_mcp_app_is_stateless_and_streams(live_key):
    headers = {**MCP_HEADERS, **bearer(live_key)}
    with TestClient(served(create_studio_mcp())) as http:
        first = http.post("/mcp/", json = LISTING, headers = headers)
        second = http.post("/mcp/", json = {**LISTING, "id": 2}, headers = headers)
    for response in (first, second):
        assert response.status_code == 200, response.text
        assert "mcp-session-id" not in response.headers
        assert response.headers["content-type"].startswith("text/event-stream")
        assert '"tools"' in response.text


KEYLESS_BEARERS = [
    None,
    "Bearer ",
    "Bearer",
    *(f"Bearer {dummy}" for dummy in sorted(APPROVED_DUMMY_BEARERS)),
]


@pytest.mark.parametrize("scope", KEYLESS_SCOPES)
@pytest.mark.parametrize("authorization", KEYLESS_BEARERS)
def test_keyless_is_never_admitted(scope, authorization):
    seed_owner()
    set_keyless_api_access(scope, tools = scope != "off")
    set_mcp_enabled(True)
    headers = {"Authorization": authorization} if authorization is not None else {}
    with TestClient(served(probe_mcp()), **LOCAL) as http:
        response, result = call_tool(http, headers)
    assert response.status_code == 401
    assert response.json() == {"detail": NEED_KEY}
    assert response.headers["www-authenticate"] == "Bearer"
    assert result is None


@pytest.mark.parametrize(
    "authorization",
    [
        lambda: "Basic dXNlcjpwYXNz",
        lambda: "Token sk-unsloth-x",
        lambda: "Bearer not-a-key",
        # A UI session's JWT.
        lambda: f"Bearer {create_access_token(subject = storage.DEFAULT_ADMIN_USERNAME)}",
    ],
)
def test_other_schemes_and_non_keys_need_a_key(authorization):
    seed_owner()
    set_mcp_enabled(True)
    with TestClient(served(probe_mcp())) as http:
        response, _result = call_tool(http, {"Authorization": authorization()})
    assert response.status_code == 401
    assert response.json() == {"detail": NEED_KEY}


def test_a_valid_key_passes_and_a_revoked_one_does_not():
    seed_owner()
    raw_key, row = owner_key()
    set_mcp_enabled(True)
    with TestClient(served(probe_mcp())) as http:
        response, result = call_tool(http, bearer(raw_key))
        assert response.status_code == 200, response.text
        assert result["token"] == raw_key
        assert result["account_id"] == storage.get_user_record("unsloth")["account_id"]
        assert storage.revoke_api_key(storage.DEFAULT_ADMIN_USERNAME, row["id"])
        storage._reset_api_key_hash_cache()
        revoked, _result = call_tool(http, bearer(raw_key))
    assert revoked.status_code == 401
    assert revoked.json() == {"detail": "Invalid or expired API key"}


@pytest.mark.parametrize(
    "workflow,detail",
    [(False, "Invalid or expired API key"), (True, "Workflow keys cannot use Unsloth Studio MCP")],
)
def test_an_unknown_or_workflow_key_is_refused(workflow, detail):
    seed_owner()
    raw_key = owner_key(internal = True)[0] if workflow else "sk-unsloth-" + "0" * 32
    set_mcp_enabled(True)
    with TestClient(served(probe_mcp())) as http:
        response, _result = call_tool(http, bearer(raw_key))
    assert response.status_code == 401
    assert response.json() == {"detail": detail}


def test_a_managed_account_key_carries_its_account():
    seed_owner()
    alice = storage.issue_account_setup_code(username = "alice")["account"]["account_id"]
    raw_key, _row = storage.create_api_key("alice", name = "agent", account_id = alice)
    set_mcp_enabled(True)
    with TestClient(served(probe_mcp())) as http:
        response, result = call_tool(http, bearer(raw_key))
    assert response.status_code == 200, response.text
    assert result["account_id"] == alice
    # The gate validates; the forwarded route call is what binds the account.
    assert result["bound_account"] == "owner"


def test_two_authorization_headers_are_refused(live_key):
    with TestClient(served(probe_mcp())) as http:
        response = http.post(
            "/mcp/",
            json = LISTING,
            headers = [
                ("accept", MCP_HEADERS["Accept"]),
                ("authorization", f"Bearer {live_key}"),
                ("authorization", f"Bearer {live_key}"),
            ],
        )
    assert response.status_code == 401
    assert response.json() == {"detail": "Send one Authorization header"}


def test_the_gate_validates_without_touching_last_used(key_spy):
    seed_owner()
    raw_key, row = owner_key()
    set_mcp_enabled(True)
    with TestClient(served(probe_mcp())) as http:
        response, _result = call_tool(http, bearer(raw_key))
    assert response.status_code == 200, response.text
    assert key_spy == [{"touch": False}]
    conn = sqlite3.connect(storage.DB_PATH)
    try:
        last_used = conn.execute(
            "SELECT last_used_at FROM api_keys WHERE id = ?", (row["id"],)
        ).fetchone()[0]
    finally:
        conn.close()
    assert last_used is None


def test_the_gate_never_binds_an_account(live_key):
    sentinel = AccountContext("sentinel-account", "sentinel")
    seen = []

    async def inner(scope, receive, send):
        seen.append(current_account())
        await send({"type": "http.response.start", "status": 204, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    async def run():
        sent = []

        async def send(message):
            sent.append(message)

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        token = bind_account(sentinel)
        try:
            scope = {
                "type": "http",
                "method": "POST",
                "path": "/mcp/",
                "root_path": "/mcp",
                "scheme": "http",
                "query_string": b"",
                "server": ("127.0.0.1", 8888),
                "client": ("127.0.0.1", 50000),
                "headers": [
                    (b"host", b"127.0.0.1:8888"),
                    (b"authorization", f"Bearer {live_key}".encode()),
                ],
            }
            await StudioMcpGate(inner)(scope, receive, send)
            return sent, current_account()
        finally:
            reset_account(token)

    sent, after = asyncio.run(run())
    assert sent[0]["status"] == 204
    assert seen == [sentinel]
    assert after == sentinel


def test_tools_see_the_outer_request(live_key):
    with TestClient(served(probe_mcp()), **LOCAL) as http:
        _response, local = call_tool(http, {**bearer(live_key), "X-Unsloth-HF-Token": " hf_abc "})
    with TestClient(served(probe_mcp()), **REMOTE) as http:
        _response, remote = call_tool(http, bearer(live_key))
    assert local["direct_local"] is True
    assert local["public_base"] == "http://127.0.0.1:8888"
    assert local["hf_token"] == "hf_abc"
    assert local["has_studio_app"] is True
    assert remote["direct_local"] is False
    assert remote["public_base"] == "http://192.168.1.20:8888"
    assert remote["hf_token"] is None


def test_a_root_path_carries_into_the_public_base(live_key):
    # The client's base URL carries the prefix, so call_tool's /mcp/ goes to /studio/mcp/.
    client = {**LOCAL, "base_url": "http://127.0.0.1:8888/studio"}
    with TestClient(served(probe_mcp()), root_path = "/studio", **client) as http:
        response, result = call_tool(http, bearer(live_key))
    assert response.status_code == 200
    assert result["public_base"] == "http://127.0.0.1:8888/studio"


def test_an_oversized_hf_token_is_refused(live_key):
    with TestClient(served(probe_mcp())) as http:
        response, _result = call_tool(http, {**bearer(live_key), "X-Unsloth-HF-Token": "h" * 513})
    assert response.status_code == 400


def _padded_call(pad):
    body = {**WHOAMI, "params": {**WHOAMI["params"], "_meta": {"pad": "x" * pad}}}
    return json.dumps(body).encode()


@pytest.mark.parametrize("chunked", [False, True])
@pytest.mark.parametrize("over", [True, False])
def test_a_request_over_4_mib_is_refused_and_one_under_reaches_the_tool(live_key, chunked, over):
    payload = _padded_call(MAX_REQUEST_BYTES if over else MAX_REQUEST_BYTES - 4096)
    assert (len(payload) > MAX_REQUEST_BYTES) is over
    content = (
        (payload[i : i + 65536] for i in range(0, len(payload), 65536)) if chunked else payload
    )
    with TestClient(served(probe_mcp())) as http:
        response = http.post(
            "/mcp/",
            content = content,
            headers = {**MCP_HEADERS, **bearer(live_key), "Content-Type": "application/json"},
        )
    if over:
        assert response.status_code == 413
        assert response.json() == {"detail": TOO_LARGE}
    else:
        assert response.status_code == 200
        assert '"isError":false' in response.text.replace(" ", "")


def test_current_caller_outside_the_gate_is_an_error():
    with pytest.raises(RuntimeError):
        current_caller()


def test_the_static_token_is_retired(monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_MCP_TOKEN", "static-token")
    set_mcp_enabled(True)
    with TestClient(served(create_studio_mcp())) as http:
        response = http.post(
            "/mcp/", json = LISTING, headers = {**MCP_HEADERS, **bearer("static-token")}
        )
    assert response.status_code == 401
    assert response.json() == {
        "detail": "The MCP static token is no longer supported; use an Unsloth Studio API key (sk-unsloth-…)"
    }
    assert response.headers["www-authenticate"] == "Bearer"


def test_a_non_ascii_authorization_header_is_a_clean_401(monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_MCP_TOKEN", "static-token")
    set_mcp_enabled(True)
    with TestClient(served(probe_mcp())) as http:
        response = http.post(
            "/mcp/",
            json = LISTING,
            headers = [("accept", MCP_HEADERS["Accept"]), ("authorization", b"Bearer \xff\xff")],
        )
    assert response.status_code == 401
    assert response.json() == {"detail": NEED_KEY}


def _import_main(extra_env):
    env = {
        name: value
        for name, value in os.environ.items()
        if name not in ("UNSLOTH_STUDIO_ENABLE_MCP", "UNSLOTH_STUDIO_MCP_TOKEN")
    }
    env.update(extra_env)
    return subprocess.run(
        [sys.executable, "-c", "import main"],
        cwd = Path(__file__).resolve().parent.parent,
        env = env,
        capture_output = True,
        text = True,
        timeout = 600,
    )


def test_enabling_by_env_without_a_token_starts():
    result = _import_main({"UNSLOTH_STUDIO_ENABLE_MCP": "1"})
    assert result.returncode == 0, result.stderr[-2000:]
    assert "UNSLOTH_STUDIO_MCP_TOKEN" not in result.stdout + result.stderr


def test_a_leftover_static_token_logs_one_warning():
    result = _import_main({"UNSLOTH_STUDIO_ENABLE_MCP": "1", "UNSLOTH_STUDIO_MCP_TOKEN": "legacy"})
    assert result.returncode == 0, result.stderr[-2000:]
    lines = [
        line
        for line in (result.stdout + result.stderr).splitlines()
        if "UNSLOTH_STUDIO_MCP_TOKEN" in line
    ]
    assert len(lines) == 1, lines
    assert json.loads(lines[0])["level"] == "warning"


TUNNEL = "https://abc-def.trycloudflare.com"


@pytest.mark.parametrize(
    "base_url,origin,allowed",
    [
        ("http://127.0.0.1:8888", None, True),
        ("http://127.0.0.1:8888", "http://localhost:8888", True),
        ("http://127.0.0.1:8888", "http://127.0.0.1:8888", True),
        ("http://127.0.0.1:8888", "http://[::1]:8888", True),
        ("http://192.168.1.20:8888", "http://192.168.1.20:8888", True),
        (TUNNEL, TUNNEL, True),
        ("http://127.0.0.1:8888", "tauri://localhost", True),
        ("http://127.0.0.1:8888", "http://tauri.localhost", True),
        ("http://127.0.0.1:8888", "https://evil.example", False),
        ("http://evil.example:8888", "http://evil.example:8888", False),
        ("http://127.0.0.1:8888", "null", False),
        ("http://127.0.0.1:8888", "http://localhost:3000", False),
    ],
)
def test_origin_allowlist(key_spy, base_url, origin, allowed, live_key):
    headers = {**bearer(live_key), **({"Origin": origin} if origin else {})}
    with TestClient(served(probe_mcp(), cloudflare_url = TUNNEL), base_url = base_url) as http:
        response, _result = call_tool(http, headers)
    if allowed:
        assert response.status_code == 200, response.text
    else:
        assert response.status_code == 403
        assert response.json() == {"detail": "Origin not allowed for Unsloth Studio MCP"}
        assert key_spy == []


def test_a_foreign_origin_is_refused_before_auth():
    set_mcp_enabled(True)
    with TestClient(served(probe_mcp())) as http:
        response, _result = call_tool(http, {"Origin": "https://evil.example"})
    assert response.status_code == 403


def test_two_origin_headers_are_refused(live_key):
    with TestClient(served(probe_mcp()), base_url = "http://127.0.0.1:8888") as http:
        response = http.post(
            "/mcp/",
            json = LISTING,
            headers = [
                ("accept", MCP_HEADERS["Accept"]),
                ("authorization", f"Bearer {live_key}"),
                ("origin", "http://127.0.0.1:8888"),
                ("origin", "https://evil.example"),
            ],
        )
    assert response.status_code == 403


@pytest.mark.parametrize(
    "path",
    [
        "/.well-known/oauth-protected-resource",
        "/.well-known/oauth-protected-resource/mcp/",
        "/.well-known/oauth-authorization-server",
        "/.well-known/oauth-authorization-server/mcp",
    ],
)
def test_oauth_discovery_is_a_404_not_the_app_shell(tmp_path, path):
    import main

    (tmp_path / "index.html").write_text("<!doctype html><title>studio</title>")
    app = FastAPI()
    assert main.setup_frontend(app, tmp_path)
    client = TestClient(app, base_url = "http://127.0.0.1:8888", client = ("127.0.0.1", 40000))
    response = client.get(path)
    assert response.status_code == 404
    assert response.json() == {"detail": "Not Found"}
    root = client.get("/")
    assert root.status_code == 200
    assert "<title>studio</title>" in root.text


def test_the_gate_401_advertises_no_oauth_metadata():
    set_mcp_enabled(True)
    with TestClient(served(probe_mcp())) as http:
        response, _result = call_tool(http, {})
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    assert "resource_metadata" not in response.text
