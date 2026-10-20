# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The chat's "Unsloth Studio" MCP row: the /mcp tools minus the chat denylist, called in process as the chat's own
account through a per-call internal key. The fake Unsloth Studio here runs the real auth dependency, so every route
sees the account the minted key belongs to."""

import asyncio
import inspect
import threading
import time
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest
from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse, Response
from fastapi.testclient import TestClient

from auth import storage
from auth.authentication import get_current_subject
from auth.policy import OwnerOnly
from core.inference import mcp_client, media_chat_tools, studio_tools_mcp
from core.inference import tools as tools_mod
from mcp_server import CHAT_DENYLIST, create_studio_mcp
from storage import mcp_servers_db
from studio_mcp import caller as caller_mod
from studio_mcp import forward as forward_mod
from studio_mcp.inputs import PATH_REMOTE
from utils.account_context import OWNER, AccountContext, current_account, run_as
from utils.mcp_access import set_mcp_enabled

from .mcp_harness import (
    MCP_HEADERS,
    SENTINEL_ROOTS,
    WAV,
    b64,
    isolated_auth,
    make_caller,
    poison,
    seed_owner,
)  # noqa: F401
from .test_mcp_tools_training import RUN_PAYLOADS

URL = studio_tools_mcp.STUDIO_TOOLS_MCP_URL
RUNS = ("GET", "/api/train/runs")
pytestmark = pytest.mark.usefixtures("isolated_auth")


@pytest.fixture
def studio_home(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))
    monkeypatch.setattr(mcp_servers_db, "_schema_ready", set())
    monkeypatch.setattr(mcp_client, "_tool_cache", {})
    monkeypatch.setattr(mcp_client, "_probe_cooloff_until", {})


@pytest.fixture
def server_loop(monkeypatch):
    """Stands in for Unsloth Studio's own event loop, which the chat tool loop hands work to."""
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target = loop.run_forever, daemon = True)
    thread.start()
    monkeypatch.setattr(media_chat_tools, "_server_loop", loop)
    yield loop
    loop.call_soon_threadsafe(loop.stop)
    thread.join(5)


@pytest.fixture
def accounts():
    seed_owner()
    alice = storage.issue_account_setup_code(username = "alice")["account"]
    return SimpleNamespace(
        owner = storage.get_account_by_id("owner"),
        alice = AccountContext(alice["account_id"], "alice", "user"),
    )


def studio(
    monkeypatch,
    routes,
    *,
    owner_only = (),
):
    """A fake Unsloth Studio behind the real auth dependency; ``app.state.seen`` records who each route ran as."""
    app = FastAPI()
    app.state.server_port = 8888
    app.state.cloudflare_url = None
    app.state.seen = []

    def bind(method, path, answer):
        async def endpoint(request: Request) -> Response:
            token = request.headers["authorization"].removeprefix("Bearer ")
            app.state.seen.append({"path": path, "account": current_account(), "token": token})
            result = answer(request) if callable(answer) else answer
            if inspect.isawaitable(result):
                result = await result
            return result if isinstance(result, Response) else JSONResponse(result)

        deps = [Depends(get_current_subject)] + ([OwnerOnly] if path in owner_only else [])
        app.add_api_route(path, endpoint, methods = [method], dependencies = deps)

    for (method, path), answer in routes.items():
        bind(method, path, answer)
    monkeypatch.setattr(studio_tools_mcp, "_studio_app", app)
    return app


def bridge(
    account,
    name,
    args = None,
    **kw,
):
    kw.setdefault("timeout", 30.0)
    return run_as(account, studio_tools_mcp.call, name, args or {}, **kw)


def chat_keys(account_id = None):
    rows = storage.list_api_keys("alice" if account_id else "unsloth", True, account_id)
    return [row for row in rows if row["name"] == storage.STUDIO_CHAT_TOOLS_KEY_NAME]


def add_row(account, **fields):
    from models.mcp_servers import McpServerCreate
    from routes.mcp_servers import create_mcp_server

    payload = McpServerCreate(display_name = "Unsloth Studio", url = URL, **fields)
    return run_as(
        account, asyncio.run, create_mcp_server(payload, current_subject = account.username)
    )


def listed(account):
    return run_as(account, asyncio.run, tools_mod.get_enabled_mcp_tools())


# -- the row and the chat instance ------------------------------------------------------------------------------


def test_the_reserved_url_is_port_agnostic_and_distinct_from_decisions():
    # The frontend's UNSLOTH_STUDIO_TOOLS_MCP_URL must match this literal.
    assert URL == "http://127.0.0.1/mcp/studio-tools"
    for url in (
        URL,
        URL + "/",
        "http://localhost:8888/mcp/studio-tools",
        "http://[::1]:1/mcp/studio-tools",
    ):
        assert mcp_client.is_studio_tools(url), url
        assert not mcp_client.is_studio_decisions(url)
    for url in (
        "https://127.0.0.1/mcp/studio-tools",
        "http://192.168.1.2/mcp/studio-tools",
        "http://127.0.0.1/mcp/studio-tools/x",
        "http://127.0.0.1/mcp",
        "http://[::1/mcp/studio-tools",
        "npx some-server",
    ):
        assert not mcp_client.is_studio_tools(url), url


def test_the_frontend_preset_uses_the_same_url():
    from pathlib import Path
    source = Path(__file__).resolve().parents[2] / "frontend/src/features/chat/mcp-server-url.ts"
    assert f'export const UNSLOTH_STUDIO_TOOLS_MCP_URL = "{URL}";' in source.read_text("utf-8")


def test_the_chat_instance_drops_exactly_the_denylist_and_mcp_keeps_every_tool():
    everything = [tool.name for tool in asyncio.run(create_studio_mcp().list_tools())]
    chat = [tool.name for tool in asyncio.run(studio_tools_mcp.chat_tools_mcp().list_tools())]
    assert CHAT_DENYLIST == {
        "chat",
        "load_model",
        "unload_model",
        "cancel",
        "start_training",
        "export_model",
    }
    assert CHAT_DENYLIST <= set(everything)
    assert chat == [name for name in everything if name not in CHAT_DENYLIST]
    assert studio_tools_mcp.chat_tools_mcp() is studio_tools_mcp.chat_tools_mcp()


@pytest.mark.parametrize("who", ["owner", "alice"])
def test_the_chat_lists_only_chat_tools_under_the_row_id(studio_home, accounts, who):
    account = getattr(accounts, who)
    # A managed account's UI session may add the row: it is answered in process, never dialled.
    row = add_row(account)
    names = [tool["function"]["name"] for tool in listed(account)]
    prefix = f"mcp__{row.id}__"
    raw = [name.removeprefix(prefix) for name in names]
    assert all(name.startswith(prefix) for name in names)
    assert "list_models" in raw and "studio_status" in raw
    assert not CHAT_DENYLIST & set(raw)
    assert len(raw) == len(asyncio.run(create_studio_mcp().list_tools())) - len(CHAT_DENYLIST)


def test_without_the_row_nothing_is_listed_or_probed(studio_home, accounts, monkeypatch):
    probes = []
    monkeypatch.setattr(tools_mod, "list_tools_async", lambda **kw: probes.append(kw))
    assert listed(OWNER) == []
    row = add_row(OWNER, is_enabled = False)
    assert listed(OWNER) == []
    assert probes == []
    assert row.is_enabled is False


def test_the_row_is_relisted_every_send_so_a_tool_turned_off_drops_out_without_eviction(
    studio_home, accounts, monkeypatch
):
    # Stands in for PR A's owner toggle, which filters this same instance's list: swap in an instance without
    # list_models and the next send must see it, with no cache eviction in between.
    add_row(OWNER)
    first = [tool["function"]["name"] for tool in listed(OWNER)]
    assert any(name.endswith("__list_models") for name in first)
    narrower = create_studio_mcp(exclude = CHAT_DENYLIST | {"list_models"})
    monkeypatch.setattr(studio_tools_mcp, "_mcp", narrower)
    second = [tool["function"]["name"] for tool in listed(OWNER)]
    assert not any(name.endswith("__list_models") for name in second)
    assert len(second) == len(first) - 1


def test_an_owner_disabled_tool_is_unlisted_and_refused(
    studio_home, accounts, server_loop, monkeypatch
):
    from utils import mcp_access

    app = studio(monkeypatch, RUN_PAYLOADS)
    add_row(accounts.alice)
    mcp_access.set_mcp_settings(disabled_tools = ["list_training_runs"])
    names = [tool["function"]["name"] for tool in listed(accounts.alice)]
    assert names and not any(name.endswith("__list_training_runs") for name in names)
    refused = bridge(accounts.alice, "list_training_runs")
    assert refused.startswith("Error: ") and "[tool_disabled]" in refused, refused
    assert app.state.seen == []
    # Back on: the very next send lists it and the call runs, with no cache eviction.
    mcp_access.set_mcp_settings(disabled_tools = [])
    assert any(
        name.endswith("__list_training_runs")
        for name in (tool["function"]["name"] for tool in listed(accounts.alice))
    )
    assert "qwen-lora" in bridge(accounts.alice, "list_training_runs")


# -- calls ------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(CHAT_DENYLIST))
def test_a_denylisted_tool_is_unknown_and_never_forwarded(
    studio_home, accounts, server_loop, monkeypatch, name
):
    app = studio(monkeypatch, {})
    sent = []
    monkeypatch.setattr(forward_mod, "_send", lambda *a, **k: sent.append(a))
    result = bridge(OWNER, name, {"model": "x", "config": {}, "messages": []})
    assert result.startswith("Error: ") and "Unknown tool" in result, result
    assert sent == [] and app.state.seen == []
    assert [row["is_active"] for row in chat_keys()] == [0]


def test_a_call_runs_through_routes_as_the_owner_with_a_revoked_internal_key(
    studio_home, accounts, server_loop, monkeypatch
):
    app = studio(monkeypatch, RUN_PAYLOADS)
    result = bridge(OWNER, "list_training_runs", timeout = None)
    assert "qwen-lora" in result and not result.startswith("Error"), result
    seen = {entry["path"]: entry for entry in app.state.seen}
    assert seen["/api/train/runs"]["account"].is_owner
    (key,) = chat_keys()
    assert key["is_internal"] == 1 and key["is_active"] == 0
    # A timeout of None waits up to an hour; the key outlives that by its grace and never passes 6 h.
    lifetime = datetime.fromisoformat(key["expires_at"]) - datetime.fromisoformat(key["created_at"])
    assert timedelta(minutes = 64) < lifetime <= timedelta(minutes = 65)
    token = seen["/api/train/runs"]["token"]
    assert storage.validate_api_key_account(token, touch = False) is None


def test_a_managed_account_runs_as_itself_and_owner_only_routes_refuse_it(
    studio_home, accounts, server_loop, monkeypatch
):
    app = studio(monkeypatch, RUN_PAYLOADS, owner_only = {"/api/train/runs"})
    refused = bridge(accounts.alice, "list_training_runs")
    assert refused.startswith("Error: ") and "installation owner" in refused, refused
    assert {entry["account"].account_id for entry in app.state.seen} == {accounts.alice.account_id}
    assert bridge(OWNER, "list_training_runs").find("qwen-lora") >= 0
    (key,) = chat_keys(accounts.alice.account_id)
    assert key["is_internal"] == 1 and key["is_active"] == 0


def test_the_key_never_passes_six_hours():
    seed_owner()
    key, row = studio_tools_mcp._mint(OWNER, 10 * 24 * 3600.0)
    lifetime = datetime.fromisoformat(row["expires_at"]) - datetime.fromisoformat(row["created_at"])
    assert lifetime <= timedelta(hours = 6)
    assert storage.is_internal_api_key(key)


def _blocking_route(app_state):
    async def answer(request):
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            app_state.cancelled = True
            raise
        return RUN_PAYLOADS[RUNS]

    return answer


def test_cancel_stops_the_route_and_revokes_the_key(
    studio_home, accounts, server_loop, monkeypatch
):
    holder = SimpleNamespace(cancelled = False)
    app = studio(monkeypatch, {**RUN_PAYLOADS, RUNS: _blocking_route(holder)})
    stop = threading.Event()

    def press_stop():
        deadline = time.monotonic() + 10
        while (
            not any(e["path"] == "/api/train/runs" for e in app.state.seen)
            and time.monotonic() < deadline
        ):
            time.sleep(0.02)
        stop.set()

    threading.Thread(target = press_stop, daemon = True).start()
    started = time.monotonic()
    result = bridge(accounts.alice, "list_training_runs", cancel_event = stop)
    assert result == "Error: MCP tool 'list_training_runs' cancelled"
    assert time.monotonic() - started < 10
    assert [row["is_active"] for row in chat_keys(accounts.alice.account_id)] == [0]
    deadline = time.monotonic() + 5
    while not holder.cancelled and time.monotonic() < deadline:
        time.sleep(0.02)
    assert holder.cancelled


def test_a_timeout_revokes_the_key(studio_home, accounts, server_loop, monkeypatch):
    studio(monkeypatch, {**RUN_PAYLOADS, RUNS: _blocking_route(SimpleNamespace())})
    result = bridge(OWNER, "list_training_runs", timeout = 1.0)
    assert result == "Error: MCP tool 'list_training_runs' timed out after 1s"
    assert [row["is_active"] for row in chat_keys()] == [0]


def test_a_tool_error_revokes_the_key(studio_home, accounts, server_loop, monkeypatch):
    studio(monkeypatch, {**RUN_PAYLOADS, RUNS: JSONResponse({"detail": "boom"}, status_code = 500)})
    result = bridge(OWNER, "list_training_runs")
    assert result.startswith("Error: "), result
    assert [row["is_active"] for row in chat_keys()] == [0]


def test_no_mint_means_no_call_and_never_the_owner(studio_home, accounts, server_loop, monkeypatch):
    app = studio(monkeypatch, RUN_PAYLOADS)
    storage.set_account_active(accounts.alice.account_id, False)
    result = bridge(accounts.alice, "list_training_runs")
    assert result == (
        "Error: MCP tool 'list_training_runs' failed: Unsloth Studio could not authorize this call."
    )
    assert app.state.seen == [] and chat_keys() == []


def test_without_a_server_loop_or_app_nothing_is_minted(studio_home, accounts, monkeypatch):
    monkeypatch.setattr(media_chat_tools, "_server_loop", None)
    studio(monkeypatch, RUN_PAYLOADS)
    assert "not ready" in bridge(OWNER, "list_training_runs")
    assert chat_keys() == []


# -- host paths ---------------------------------------------------------------------------------------------------


def test_host_paths_are_refused_even_for_the_owner(
    studio_home, accounts, server_loop, monkeypatch, tmp_path
):
    clip = tmp_path / "clip.wav"
    clip.write_bytes(WAV)
    app = studio(monkeypatch, {})
    result = bridge(OWNER, "transcribe", {"audio": {"path": str(clip)}})
    assert result.startswith("Error: ") and PATH_REMOTE in result, result
    assert app.state.seen == []


def test_route_host_paths_do_not_reach_the_chat(studio_home, accounts, server_loop, monkeypatch):
    studio(monkeypatch, {route: poison(payload) for route, payload in RUN_PAYLOADS.items()})
    result = bridge(OWNER, "list_training_runs", {"run_id": "run-1", "include_checkpoints": True})
    assert "qwen-lora" in result
    assert not any(root in result for root in SENTINEL_ROOTS), result


# -- the caller ContextVar ---------------------------------------------------------------------------------------


def test_after_a_call_no_caller_is_left_bound(studio_home, accounts, server_loop, monkeypatch):
    studio(monkeypatch, RUN_PAYLOADS)
    bridge(OWNER, "list_training_runs")
    with pytest.raises(RuntimeError):
        caller_mod.current_caller()

    async def on_the_loop():
        return caller_mod._bound.get()

    assert asyncio.run_coroutine_threadsafe(on_the_loop(), server_loop).result(5) is None


def test_two_accounts_at_once_each_run_as_themselves(
    studio_home, accounts, server_loop, monkeypatch
):
    both_in = asyncio.Event()
    arrived = []

    async def rendezvous(request):
        arrived.append(1)
        if len(arrived) == 2:
            both_in.set()
        await asyncio.wait_for(both_in.wait(), 10)
        return RUN_PAYLOADS[RUNS]

    app = studio(monkeypatch, {**RUN_PAYLOADS, RUNS: rendezvous})
    results = {}

    def run(who):
        results[who] = bridge(getattr(accounts, who), "list_training_runs")

    threads = [threading.Thread(target = run, args = (who,)) for who in ("owner", "alice")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(20)
    assert all("qwen-lora" in result for result in results.values()), results
    by_token = {}
    for entry in app.state.seen:
        by_token.setdefault(entry["token"], set()).add(entry["account"].account_id)
    assert sorted(map(sorted, by_token.values())) == sorted(
        [["owner"], [accounts.alice.account_id]]
    )


def _fake_http_request(caller = None):
    from starlette.requests import Request as StarletteRequest

    scope = {"type": "http", "method": "POST", "path": "/mcp/", "headers": [], "state": {}}
    if caller is not None:
        scope["state"][caller_mod.STATE_KEY] = caller
    return StarletteRequest(scope)


def test_an_http_request_never_sees_a_bound_caller():
    from fastmcp.server.dependencies import _current_http_request
    bound = caller_mod.bind_caller(make_caller(token = "bound"))
    try:
        assert caller_mod.current_caller().token == "bound"
        request = _current_http_request.set(_fake_http_request())
        try:
            with pytest.raises(RuntimeError):
                caller_mod.current_caller()
        finally:
            _current_http_request.reset(request)
        request = _current_http_request.set(_fake_http_request(make_caller(token = "gate")))
        try:
            assert caller_mod.current_caller().token == "gate"
        finally:
            _current_http_request.reset(request)
    finally:
        caller_mod.reset_caller(bound)


def test_an_outer_http_request_on_the_tool_thread_does_not_outrank_the_bridge(
    studio_home, accounts, server_loop, monkeypatch
):
    from fastmcp.server.dependencies import _current_http_request

    app = studio(monkeypatch, RUN_PAYLOADS)
    outer = _current_http_request.set(_fake_http_request(make_caller(token = "outer-agent-key")))
    try:
        result = bridge(OWNER, "list_training_runs")
    finally:
        _current_http_request.reset(outer)
    assert "qwen-lora" in result, result
    assert app.state.seen and "outer-agent-key" not in {entry["token"] for entry in app.state.seen}


# -- the external /mcp gate is untouched -------------------------------------------------------------------------


def test_chat_works_with_agent_access_off_and_its_key_cannot_use_mcp(
    studio_home, accounts, server_loop, monkeypatch
):
    from main import app as main_app

    app = studio(monkeypatch, RUN_PAYLOADS)
    revoked = []
    monkeypatch.setattr(studio_tools_mcp, "_revoke", revoked.append)
    set_mcp_enabled(False)
    http = TestClient(main_app)
    listing = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
    assert http.post("/mcp/", json = listing, headers = MCP_HEADERS).status_code == 404
    assert "qwen-lora" in bridge(OWNER, "list_training_runs")
    token = app.state.seen[0]["token"]
    assert http.post("/mcp/", json = listing, headers = MCP_HEADERS).status_code == 404

    set_mcp_enabled(True)
    refused = http.post(
        "/mcp/", json = listing, headers = {**MCP_HEADERS, "Authorization": f"Bearer {token}"}
    )
    assert refused.status_code == 401
    assert refused.json() == {"detail": "Workflow keys cannot use Unsloth Studio MCP"}
    (key,) = chat_keys()
    assert revoked == [key["id"]]
    storage.revoke_internal_api_key(key["id"])


# -- mcp_client and the chat request ----------------------------------------------------------------------------


def test_call_tool_sync_uses_the_bridge_and_no_pooled_session(
    studio_home, accounts, server_loop, monkeypatch
):
    studio(monkeypatch, RUN_PAYLOADS)

    def no_session(*a, **k):
        raise AssertionError("the Unsloth Studio tools row must not use a pooled session")

    monkeypatch.setattr(mcp_client, "_call_session_tool", no_session)
    result = run_as(
        OWNER,
        mcp_client.call_tool_sync,
        URL,
        None,
        "list_training_runs",
        {},
        timeout = 30.0,
        scope = "thread:1",
    )
    assert "qwen-lora" in result, result
    stale = run_as(
        OWNER,
        mcp_client.call_tool_sync,
        URL,
        None,
        "list_training_runs",
        {},
        config_check = lambda: False,
    )
    assert stale.startswith("Error: ") and len(chat_keys()) == 1


def test_a_structured_call_outside_the_bridge_has_no_caller(studio_home, accounts):
    result = mcp_client.call_tool_structured_sync(URL, None, "studio_status", {}, timeout = 30.0)
    assert result["is_error"] is True
    assert chat_keys() == []


def test_a_chat_request_with_mcp_remembers_the_server_loop(monkeypatch):
    from routes import inference

    monkeypatch.setattr(media_chat_tools, "_server_loop", None)

    async def no_tools(*a, **k):
        return []

    monkeypatch.setattr(tools_mod, "get_enabled_mcp_tools", no_tools)
    payload = SimpleNamespace(
        enabled_tools = None,
        rag_scope = None,
        bypass_permissions = False,
        sandbox_level = None,
        thread_id = None,
        deep_research_armed = None,
    )

    async def select():
        await inference._select_request_tools(payload, tools_on = False, mcp_allowed = True)
        return asyncio.get_running_loop()

    loop = asyncio.new_event_loop()
    try:
        assert loop.run_until_complete(select()) is media_chat_tools.server_loop()
    finally:
        loop.close()
