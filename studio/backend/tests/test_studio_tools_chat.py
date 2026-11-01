# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The chat's "Unsloth Studio" MCP row: the /mcp tools minus the chat denylist, called in process as
the chat's own account through a per-call internal key, against a fake Studio behind the real auth."""

import asyncio
import sqlite3
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from fastapi.testclient import TestClient
from fastmcp.server.dependencies import _current_http_request
from starlette.requests import Request as StarletteRequest

from auth import storage
from auth.authentication import get_current_subject
from auth.policy import OwnerOnly
from core.inference import mcp_client, media_chat_tools, studio_tools_mcp
from core.inference import tools as tools_mod
from mcp_server import create_studio_mcp, withheld
from storage import mcp_servers_db
from studio_mcp import caller as caller_mod
from studio_mcp import forward as forward_mod
from studio_mcp.inputs import PATH_REMOTE
from studio_mcp.profile import CHAT
from utils import mcp_access
from utils.account_context import OWNER, AccountContext, as_api_caller, current_account, run_as

from .mcp_harness import (
    MCP_HEADERS,
    PNG,
    SENTINEL_ROOTS,
    WAV,
    isolated_auth,  # noqa: F401  (fixture)
    make_caller,
    poison,
    seed_owner,
)
from .test_mcp_tools_training import RUN_PAYLOADS

CHAT_WITHHELD = frozenset(withheld())
URL = studio_tools_mcp.STUDIO_TOOLS_MCP_URL
RUNS = ("GET", "/api/train/runs")
KEY_NAME = storage.STUDIO_CHAT_TOOLS_KEY_NAME
CANCELLED = "Error: MCP tool 'list_training_runs' cancelled"
pytestmark = pytest.mark.usefixtures("isolated_auth", "studio_home")


@pytest.fixture
def studio_home(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))
    monkeypatch.setattr(mcp_servers_db, "_schema_ready", set())
    monkeypatch.setattr(mcp_client, "_tool_cache", {})
    monkeypatch.setattr(mcp_client, "_probe_cooloff_until", {})
    seed_owner()


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
def alice():
    row = storage.issue_account_setup_code(username = "alice")["account"]
    return AccountContext(row["account_id"], "alice", "user")


def studio(monkeypatch, routes, *owner_only):
    """A fake Studio behind the real auth dependency; ``app.state.seen`` records who each route ran as."""
    app = FastAPI()
    app.state.server_port, app.state.cloudflare_url, app.state.seen = 8888, None, []

    def bind(method, path, answer):
        async def endpoint(request: Request) -> Response:
            token = request.headers["authorization"].removeprefix("Bearer ")
            app.state.seen.append({"path": path, "account": current_account(), "token": token})
            result = await answer(request) if callable(answer) else answer
            return result if isinstance(result, Response) else JSONResponse(result)

        deps = [Depends(get_current_subject)] + ([OwnerOnly] if path in owner_only else [])
        app.add_api_route(path, endpoint, methods = [method], dependencies = deps)

    for (method, path), answer in routes.items():
        bind(method, path, answer)
    monkeypatch.setattr(studio_tools_mcp, "_studio_app", app)
    return app


def bridge(account, name, *args, **kw):
    return run_as(account, studio_tools_mcp.call, name, *(args or [{}]), **{"timeout": 30.0, **kw})


def runs(account = OWNER, **kw):
    return bridge(account, "list_training_runs", **kw)


def call_tool_sync(**kw):
    call, kw = mcp_client.call_tool_sync, {"timeout": 30.0, **kw}
    return run_as(OWNER, call, URL, None, "list_training_runs", {}, **kw)


def minted_keys(monkeypatch):
    rows, real = [], storage.create_api_key

    def spy(*args, **kwargs):
        key, row = real(*args, **kwargs)
        if kwargs.get("name") == KEY_NAME:
            rows.append(dict(row, is_internal = bool(kwargs.get("internal"))))
        return key, row

    monkeypatch.setattr(storage, "create_api_key", spy)
    return rows


def chat_keys():
    """Chat tools key rows left in either key table: the bridge deletes them when its call ends."""
    conn = sqlite3.connect(storage.DB_PATH)
    try:
        tables = ("api_keys", "account_api_keys")
        query = " UNION ALL ".join(f"SELECT id FROM {t} WHERE name = ?1" for t in tables)
        return conn.execute(query, (KEY_NAME,)).fetchall()
    finally:
        conn.close()


def add_row(account, **fields):
    from models.mcp_servers import McpServerCreate
    from routes.mcp_servers import create_mcp_server

    payload = McpServerCreate(display_name = "Unsloth Studio", url = URL, **fields)
    create = create_mcp_server(payload, current_subject = account.username)
    return run_as(account, asyncio.run, create)


def listed(account):
    tools = run_as(account, asyncio.run, tools_mod.get_enabled_mcp_tools())
    return [tool["function"]["name"] for tool in tools]


def chat_tool_names():
    return [tool.name for tool in asyncio.run(studio_tools_mcp.chat_tools_mcp().list_tools())]


def test_the_reserved_url_is_port_agnostic_and_matches_the_frontend():
    assert URL == "http://127.0.0.1/mcp/studio-tools"
    ok = (URL, URL + "/", "http://localhost:8888/mcp/studio-tools")
    for url in ok + ("http://[::1]:1/mcp/studio-tools",):
        assert mcp_client.is_studio_tools(url) and not mcp_client.is_studio_decisions(url), url
    bad = ("https://127.0.0.1/mcp/studio-tools", "http://192.168.1.2/mcp/studio-tools")
    bad += ("http://127.0.0.1/mcp/studio-tools/x", "http://127.0.0.1/mcp")
    for url in bad + ("http://[::1/mcp/studio-tools", "npx some-server"):
        assert not mcp_client.is_studio_tools(url), url
    source = Path(__file__).resolve().parents[2] / "frontend/src/features/chat/mcp-server-url.ts"
    assert f'export const UNSLOTH_STUDIO_TOOLS_MCP_URL = "{URL}";' in source.read_text("utf-8")


def test_the_chat_instance_drops_exactly_the_withheld_tools():
    everything = [tool.name for tool in asyncio.run(create_studio_mcp().list_tools())]
    loading = {"load_model", "unload_model", "plan_load"}
    jobs = {"start_training", "export_model", "run_recipe"}
    assert CHAT_WITHHELD == {"chat", "cancel"} | loading | jobs
    assert chat_tool_names() == [name for name in everything if name not in CHAT_WITHHELD]
    assert studio_tools_mcp.chat_tools_mcp() is studio_tools_mcp.chat_tools_mcp()


def test_the_row_lists_chat_tools_fresh_every_send_minus_disabled_ones(
    alice, server_loop, monkeypatch
):
    app = studio(monkeypatch, RUN_PAYLOADS)
    probes, real_probe = [], tools_mod.list_tools_async
    monkeypatch.setattr(tools_mod, "list_tools_async", lambda **kw: probes.append(kw))
    assert listed(alice) == []
    add_row(alice, is_enabled = False)
    assert listed(alice) == [] and probes == []
    monkeypatch.setattr(tools_mod, "list_tools_async", real_probe)
    prefix = f"mcp__{add_row(alice).id}__"
    names = listed(alice)
    assert sorted(names) == sorted(prefix + name for name in chat_tool_names())
    keyless = as_api_caller(alice)
    assert listed(keyless) == [] and run_as(keyless, tools_mod.cached_mcp_tools)[0] == []
    mcp_access.set_mcp_settings(disabled_tools = ["list_training_runs"])
    assert prefix + "list_training_runs" not in listed(alice)
    assert "[tool_disabled]" in runs(alice) and app.state.seen == []
    mcp_access.set_mcp_settings(disabled_tools = [])
    assert "qwen-lora" in runs(alice)
    narrower = create_studio_mcp(CHAT, exclude = frozenset({"list_models"}))
    monkeypatch.setattr(studio_tools_mcp, "_mcp", narrower)
    assert sorted(listed(alice)) == sorted(set(names) - {prefix + "list_models"})


def test_a_call_runs_as_the_owner_through_the_bridge_with_a_deleted_short_lived_key(
    server_loop, monkeypatch
):
    app = studio(monkeypatch, RUN_PAYLOADS)
    minted = minted_keys(monkeypatch)

    def no_session(*a, **k):
        raise AssertionError("the Unsloth Studio tools row must not use a pooled session")

    monkeypatch.setattr(mcp_client, "_call_session_tool", no_session)
    result = call_tool_sync(timeout = None, scope = "thread:1")
    assert "qwen-lora" in result and not result.startswith("Error"), result
    assert ":null" not in result and ": null" not in result, result
    (seen,), (key,) = [e for e in app.state.seen if e["path"] == RUNS[1]], minted
    assert seen["account"].is_owner and key["is_internal"] and chat_keys() == []
    # A timeout of None waits up to an hour; the key outlives that only by its grace, never 6 h.
    lifetime = datetime.fromisoformat(key["expires_at"]) - datetime.fromisoformat(key["created_at"])
    assert timedelta(minutes = 64) < lifetime <= timedelta(minutes = 65)
    assert storage.validate_api_key_account(seen["token"], touch = False) is None
    _key, row = studio_tools_mcp._mint(OWNER, 10 * 24 * 3600.0)
    lifetime = datetime.fromisoformat(row["expires_at"]) - datetime.fromisoformat(row["created_at"])
    assert lifetime <= timedelta(hours = 6)


def test_each_account_runs_as_itself_even_at_once(alice, server_loop, monkeypatch):
    both_in, arrived, results = asyncio.Event(), [], []

    async def rendezvous(request):
        arrived.append(1)
        if len(arrived) == 2:
            both_in.set()
        await asyncio.wait_for(both_in.wait(), 10)
        return RUN_PAYLOADS[RUNS]

    app = studio(monkeypatch, {**RUN_PAYLOADS, RUNS: rendezvous})
    run = lambda account: results.append(runs(account))
    threads = [threading.Thread(target = run, args = (account,)) for account in (OWNER, alice)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(20)
    assert len(results) == 2 and all("qwen-lora" in result for result in results), results
    by_token = {}
    for entry in app.state.seen:
        by_token.setdefault(entry["token"], set()).add(entry["account"].account_id)
    assert sorted(map(sorted, by_token.values())) == sorted([["owner"], [alice.account_id]])
    studio(monkeypatch, RUN_PAYLOADS, "/api/train/runs")
    assert "installation owner" in runs(alice) and chat_keys() == []


def test_stop_cancels_the_route_and_an_image_render_on_the_gpu(alice, server_loop, monkeypatch):
    from core.inference.generation_ticket import bind_generation_cancel

    cancelled, render, bound = threading.Event(), threading.Event(), threading.Event()

    async def blocking(request):
        try:
            bound.set()
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    def engine():
        bind_generation_cancel(render)
        bound.set()
        render.wait(10)

    async def generating(request):
        await asyncio.to_thread(engine)
        return RUN_PAYLOADS[RUNS]

    for answer, done in ((blocking, cancelled), (generating, render)):
        studio(monkeypatch, {**RUN_PAYLOADS, RUNS: answer})
        bound.clear()
        stop = threading.Event()
        # Stop once the route is running: a fixed delay can fire before a loaded host reaches it.
        threading.Thread(target = lambda: bound.wait(10) and stop.set(), daemon = True).start()
        started = time.monotonic()
        assert runs(alice, cancel_event = stop) == CANCELLED
        assert time.monotonic() - started < 10 and chat_keys() == []
        assert done.wait(5)


async def _hang(request):
    await asyncio.sleep(30)


@pytest.mark.parametrize(
    "answer, timeout, expected",
    [
        (_hang, 1.0, "Error: MCP tool 'list_training_runs' timed out after 1s"),
        (JSONResponse({"detail": "boom"}, status_code = 500), 30.0, "Error: "),
    ],
)
def test_a_timeout_or_error_deletes_the_key(server_loop, monkeypatch, answer, timeout, expected):
    studio(monkeypatch, {**RUN_PAYLOADS, RUNS: answer})
    assert runs(timeout = timeout).startswith(expected)
    assert chat_keys() == []


def _stop_during_mint(monkeypatch, alice):
    stop, real = threading.Event(), studio_tools_mcp._mint
    monkeypatch.setattr(studio_tools_mcp, "_mint", lambda *a: (real(*a), stop.set())[0])
    return lambda: runs(cancel_event = stop)


def _inactive(monkeypatch, alice):
    storage.set_account_active(alice.account_id, False)
    return lambda: runs(alice)


def _password_reset(monkeypatch, alice):
    generation = storage.credential_generation(storage.get_user_record("unsloth")["jwt_secret"])
    account = AccountContext(OWNER.account_id, OWNER.username, OWNER.role, generation)
    assert account == OWNER and "qwen-lora" in runs(account), "the current generation mints"
    storage.update_password("unsloth", "a-new-password-123", revoke_refresh_tokens = True)
    return lambda: runs(account)


def _stale_row(*checks):
    def scenario(monkeypatch, alice):
        answers = iter(checks)
        return lambda: call_tool_sync(config_check = lambda: next(answers))

    return scenario


def _withheld(name):
    return lambda m, a: lambda: bridge(OWNER, name, {"model": "x", "config": {}, "messages": []})


STOPPED, HOST_FILE = SimpleNamespace(is_set = lambda: True), {"audio": {"path": __file__}}
REFUSED = [(f"withheld-{name}", "Unknown tool", _withheld(name)) for name in sorted(CHAT_WITHHELD)]
REFUSED += [
    ("keyless", "API key", lambda m, a: lambda: runs(as_api_caller(OWNER))),
    ("inactive", "authorize", _inactive),
    ("password-reset", "authorize", _password_reset),
    (
        "no-loop",
        "not ready",
        lambda m, a: m.setattr(media_chat_tools, "_server_loop", None) or runs,
    ),
    ("host-path", PATH_REMOTE, lambda m, a: lambda: bridge(OWNER, "transcribe", HOST_FILE)),
    ("stopped-first", "cancelled", lambda m, a: lambda: runs(cancel_event = STOPPED)),
    ("stopped-during-mint", "cancelled", _stop_during_mint),
    ("stale-before-mint", "updated or removed", _stale_row(False)),
    ("stale-after-mint", "updated or removed", _stale_row(True, False)),
    ("stale-on-the-loop", "updated or removed", _stale_row(True, True, False)),
]


@pytest.mark.parametrize(
    "expected, scenario", [r[1:] for r in REFUSED], ids = [r[0] for r in REFUSED]
)
def test_a_refused_call_never_reaches_a_route_and_leaves_no_key(
    alice, server_loop, monkeypatch, expected, scenario
):
    app = studio(monkeypatch, RUN_PAYLOADS)
    invoke, sent = scenario(monkeypatch, alice), []
    monkeypatch.setattr(forward_mod, "_send", lambda *a, **k: sent.append(a))
    app.state.seen.clear()
    result = invoke()
    assert result.startswith("Error: ") and expected in result, result
    assert app.state.seen == [] and sent == [] and chat_keys() == []


def test_route_host_paths_do_not_reach_the_chat(server_loop, monkeypatch):
    studio(monkeypatch, {route: poison(payload) for route, payload in RUN_PAYLOADS.items()})
    result = bridge(OWNER, "list_training_runs", {"run_id": "run-1", "include_checkpoints": True})
    assert "qwen-lora" in result
    assert not any(root in result for root in SENTINEL_ROOTS), result


def test_a_chat_tool_loads_only_beside_and_is_told_to_ask_the_user(server_loop, monkeypatch):
    from core.inference.load_policy import current_load_policy

    music, policies = "audio-cpp/audio.cpp-gguf/Stable-Audio-3-Small-Music-GGUF", []

    async def not_loaded(_request):
        policies.append(current_load_policy().value)
        problem = f"not_loaded kind=audio model={music} page=Audio"
        body = {"error": {"message": "Load a music model first.", "type": "x", "code": None}}
        return JSONResponse(body, status_code = 400, headers = {"X-Unsloth-Problem": problem})

    studio(monkeypatch, {("POST", "/v1/audio/run"): not_loaded})
    result = bridge(OWNER, "generate_audio", {"workflow": "music", "text": "lofi", "model": music})
    assert policies == ["beside_only"] and "[not_loaded kind=audio]" in result, result
    assert "load_model" not in result and "unload_model" not in result
    assert "on the Audio page in Unsloth Studio" in result


def _http_request(caller = None):
    state = {} if caller is None else {caller_mod.STATE_KEY: caller}
    scope = {"type": "http", "method": "POST", "path": "/mcp/", "headers": [], "state": state}
    return _current_http_request.set(StarletteRequest(scope))


def test_an_http_request_never_sees_a_bound_caller():
    bound = caller_mod.bind_caller(make_caller(token = "bound"))
    try:
        assert caller_mod.current_caller().token == "bound"
        request = _http_request()
        with pytest.raises(RuntimeError):
            caller_mod.current_caller()
        _current_http_request.reset(request)
        request = _http_request(make_caller(token = "gate"))
        assert caller_mod.current_caller().token == "gate"
        _current_http_request.reset(request)
    finally:
        caller_mod.reset_caller(bound)


def test_the_bridge_outranks_an_outer_http_request_and_unbinds_after(server_loop, monkeypatch):
    app = studio(monkeypatch, RUN_PAYLOADS)
    outer = _http_request(make_caller(token = "outer-agent-key"))
    try:
        assert "qwen-lora" in runs()
    finally:
        _current_http_request.reset(outer)
    assert app.state.seen and "outer-agent-key" not in {entry["token"] for entry in app.state.seen}
    with pytest.raises(RuntimeError):
        caller_mod.current_caller()

    async def on_the_loop():
        return caller_mod._bound.get()

    assert asyncio.run_coroutine_threadsafe(on_the_loop(), server_loop).result(5) is None


def test_chat_works_with_agent_access_off_and_its_key_cannot_use_mcp(server_loop, monkeypatch):
    from main import app as main_app

    app = studio(monkeypatch, RUN_PAYLOADS)
    revoked = []
    monkeypatch.setattr(studio_tools_mcp, "_revoke", revoked.append)
    mcp_access.set_mcp_enabled(False)
    http = TestClient(main_app)
    listing = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
    assert http.post("/mcp/", json = listing, headers = MCP_HEADERS).status_code == 404
    assert "qwen-lora" in runs()
    mcp_access.set_mcp_enabled(True)
    headers = {**MCP_HEADERS, "Authorization": f"Bearer {app.state.seen[0]['token']}"}
    refused = http.post("/mcp/", json = listing, headers = headers)
    assert refused.status_code == 401
    assert refused.json() == {"detail": "Workflow keys cannot use Unsloth Studio MCP"}
    (key,) = [k for k in storage.list_api_keys("unsloth", True) if k["name"] == KEY_NAME]
    assert revoked == [key["id"]]
    storage.revoke_internal_api_key(key["id"])


def test_a_structured_call_outside_the_bridge_has_no_caller():
    result = mcp_client.call_tool_structured_sync(URL, None, "studio_status", {}, timeout = 30.0)
    assert result["is_error"] is True and chat_keys() == []


def test_a_chat_request_with_mcp_remembers_the_server_loop(monkeypatch):
    from routes import inference

    async def no_tools(*a, **k):
        return []

    monkeypatch.setattr(media_chat_tools, "_server_loop", None)
    monkeypatch.setattr(tools_mod, "get_enabled_mcp_tools", no_tools)
    fields = ("enabled_tools", "rag_scope", "sandbox_level", "thread_id", "deep_research_armed")
    payload = SimpleNamespace(bypass_permissions = False, **dict.fromkeys(fields))

    async def select():
        await inference._select_request_tools(payload, tools_on = False, mcp_allowed = True)
        return asyncio.get_running_loop()

    loop = asyncio.new_event_loop()
    try:
        assert loop.run_until_complete(select()) is media_chat_tools.server_loop()
    finally:
        loop.close()


GALLERY, AUDIO = "/api/inference/images/gallery/img-1", "/v1/audio/gallery/clip-1"
IMAGE = dict(id = "img-1", url = f"{GALLERY}/file", prompt = "a red fox", width = 64, height = 64)
IMAGE.update(steps = 4, guidance = 0.0, seed = 1, model = "black-forest-labs/FLUX.1-schnell")
CLIP = dict(id = "clip-1", url = f"{AUDIO}/file", duration_s = 1.0, sample_rate = 24000)
CLIP.update(workflow = "speak", text = "hi", model = "m")


async def _media(request):
    is_png = request.url.path.startswith("/api")
    return Response(PNG if is_png else WAV, media_type = "image/png" if is_png else "audio/wav")


MEDIA_ROUTES = {
    ("POST", "/api/inference/images/generate"): {"images": [IMAGE]},
    ("GET", "/api/inference/images/generate-progress"): {"active": False},
    ("GET", f"{GALLERY}/file"): _media,
    ("GET", f"{GALLERY}/signed-url"): {"url": f"{GALLERY}/file-signed?token=t.img-1"},
    ("POST", "/v1/audio/run"): {"clips": [CLIP], "group_id": None, "model": "m", "audio": None},
    ("GET", f"{AUDIO}/file"): _media,
    ("GET", f"{AUDIO}/signed-url"): {"url": f"{AUDIO}/file-signed?token=t.clip-1"},
    ("GET", "/api/inference/status"): {"loaded": []},
}
LINK = {"prompt": "a red fox", "return_media": "link"}


@pytest.mark.parametrize(
    "root, name, args, link",
    [
        ("", "generate_image", LINK, f"img-1.png: {GALLERY}/file-signed"),
        ("/studio", "generate_image", LINK, f"img-1.png: /studio{GALLERY}/file-signed"),
        ("", "generate_image", {"prompt": "a red fox"}, ""),
        ("", "generate_audio", {"workflow": "speak", "text": "hi"}, f"clip-1.wav: {AUDIO}/"),
    ],
)
def test_chat_media_links_are_origin_relative_text(
    server_loop, monkeypatch, root, name, args, link
):
    # A ResourceLink needs an absolute URI and a chat call's links are origin-relative.
    studio(monkeypatch, MEDIA_ROUTES)
    callers, real = [], studio_tools_mcp._call
    monkeypatch.setattr(
        studio_tools_mcp, "_call", lambda c, *rest: callers.append(c) or real(c, *rest)
    )
    token = studio_tools_mcp._root_path.set(root)
    try:
        result = bridge(OWNER, name, args)
    finally:
        studio_tools_mcp._root_path.reset(token)
    assert not result.startswith("Error") and link in result, result
    (caller,) = callers
    assert caller.public_base == root and caller.call_scoped and not caller.direct_local


def test_the_middleware_hands_each_request_its_root_path():
    inner = FastAPI()

    @inner.get("/seen")
    def seen():
        return PlainTextResponse(studio_tools_mcp._root_path.get())

    inner.add_middleware(studio_tools_mcp.ChatRootPathMiddleware)
    with TestClient(inner, root_path = "/studio/") as http:
        assert http.get("/seen").text == "/studio"
    with TestClient(inner) as http:
        assert http.get("/seen").text == ""
    assert studio_tools_mcp._root_path.get() == ""


def test_dropping_nulls_keeps_empty_lists_false_and_zero():
    from mcp.types import TextContent

    text = '{"a":null,"b":[],"c":false,"d":0,"e":[{"f":null}]}'
    blocks = [TextContent(type = "text", text = text), TextContent(type = "text", text = "not json {")]
    structured = {"a": None, "b": {"c": None, "d": 1}}
    result = studio_tools_mcp._without_nulls(
        SimpleNamespace(content = blocks, structured_content = structured)
    )
    texts = [block.text for block in result.content]
    assert texts == ['{"b":[],"c":false,"d":0,"e":[{}]}', "not json {"]
    assert result.structured_content == {"b": {"d": 1}}
