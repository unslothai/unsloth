# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from fastapi import HTTPException
from starlette.requests import Request

from auth import storage as auth_storage
from core.inference import linked_instances
from storage import credential_secrets, linked_instances_db

REMOTE_KEY = "sk-unsloth-" + "a" * 32
LAN_ADDRESS = {"remote": "192.168.1.20"}


@pytest.fixture(autouse = True)
def isolated_databases(tmp_path, monkeypatch):
    studio_db = tmp_path / "studio.db"
    monkeypatch.setattr(auth_storage, "DB_PATH", tmp_path / "auth.db")
    monkeypatch.setattr(auth_storage, "_credential_encryption_key_cache", None)
    for module in (credential_secrets, linked_instances_db):
        monkeypatch.setattr(module, "studio_db_path", lambda: studio_db)
        monkeypatch.setattr(module, "ensure_dir", lambda p: p.mkdir(parents = True, exist_ok = True))
        monkeypatch.setattr(module, "_schema_ready", set())
    monkeypatch.setattr(
        credential_secrets,
        "get_or_create_credential_encryption_key",
        auth_storage.get_or_create_credential_encryption_key,
    )
    monkeypatch.setattr(linked_instances, "_catalog_cache", {})
    # The test remotes live on the LAN.
    real_getaddrinfo = linked_instances.socket.getaddrinfo

    def getaddrinfo(host, *args, **kwargs):
        if host == "remote":
            return [(2, 1, 6, "", (LAN_ADDRESS["remote"], 0))]
        return real_getaddrinfo(host, *args, **kwargs)

    monkeypatch.setattr(linked_instances.socket, "getaddrinfo", getaddrinfo)
    # Owner with an API key: no managed account, not keyless.
    monkeypatch.setattr(
        linked_instances,
        "_may_use_linked",
        lambda r: not r.headers.get(linked_instances.HOP_HEADER),
    )
    yield studio_db


def _remote(handler, monkeypatch):
    monkeypatch.setattr(
        linked_instances, "_http_client", httpx.AsyncClient(transport = httpx.MockTransport(handler))
    )


def _request(body: dict, headers: dict | None = None) -> Request:
    raw = json.dumps(body).encode()
    sent = False

    async def receive():
        nonlocal sent
        if sent:
            return {"type": "http.disconnect"}
        sent = True
        return {"type": "http.request", "body": raw, "more_body": False}

    header_list = [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()]
    return Request(
        {"type": "http", "method": "POST", "path": "/v1/x", "headers": header_list}, receive
    )


def test_key_is_encrypted_and_names_are_validated(isolated_databases):
    instance = linked_instances_db.create_instance("WSL", "http://127.0.0.1:8890", REMOTE_KEY)
    assert instance["name"] == "wsl"
    assert linked_instances_db.get_api_key(instance["id"]) == REMOTE_KEY
    assert REMOTE_KEY.encode() not in isolated_databases.read_bytes()

    with pytest.raises(linked_instances_db.DuplicateName):
        linked_instances_db.create_instance("wsl", "http://127.0.0.1:1", "k")
    with pytest.raises(ValueError):
        linked_instances_db.create_instance("has/slash", "http://127.0.0.1:1", "k")

    assert linked_instances_db.delete_instance(instance["id"]) is True
    assert linked_instances_db.get_api_key(instance["id"]) is None


@pytest.mark.parametrize(
    "model, expected",
    [
        ("@wsl/unsloth/Qwen3-0.6B-GGUF:Q4_K_M", ("wsl", "unsloth/Qwen3-0.6B-GGUF:Q4_K_M")),
        ("@Colab-1/m", ("colab-1", "m")),
        ("unsloth/Qwen3-0.6B-GGUF", None),
        ("@wsl", None),
        (None, None),
    ],
)
def test_split_model(model, expected):
    assert linked_instances.split_model(model) == expected


def test_normalize_accepts_a_pasted_v1_url():
    assert (
        linked_instances.normalize_base_url("https://abc.trycloudflare.com/v1/")
        == "https://abc.trycloudflare.com"
    )


def test_catalog_prefixes_ids_and_drops_the_remotes_own_links(monkeypatch):
    linked_instances_db.create_instance("wsl", "http://remote", REMOTE_KEY)
    seen = {}

    def handler(request: httpx.Request):
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(
            200, json = {"data": [{"id": "unsloth/a", "loaded": True}, {"id": "@other/b"}]}
        )

    _remote(handler, monkeypatch)
    models = asyncio.run(linked_instances.catalog_objects(_request({})))
    assert [m["id"] for m in models] == ["@wsl/unsloth/a"]
    assert models[0]["loaded"] is True and models[0]["owned_by"] == "wsl"
    assert seen["auth"] == f"Bearer {REMOTE_KEY}"


def test_offline_instance_contributes_no_models(monkeypatch):
    linked_instances_db.create_instance("wsl", "http://remote", REMOTE_KEY)

    def handler(request):
        raise httpx.ConnectError("down", request = request)

    _remote(handler, monkeypatch)
    assert asyncio.run(linked_instances.catalog_objects(_request({}))) == []


def test_forward_unwraps_the_model_and_marks_the_hop(monkeypatch):
    linked_instances_db.create_instance("wsl", "http://remote", REMOTE_KEY)
    seen = {}

    def handler(request: httpx.Request):
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        seen["headers"] = request.headers
        return httpx.Response(200, json = {"ok": True})

    _remote(handler, monkeypatch)
    body = {"model": "@wsl/unsloth/a", "max_tokens": 8, "messages": []}
    request = _request(body, {"anthropic-version": "2023-06-01"})

    async def run():
        target = await linked_instances.resolve(request, body["model"])
        return await linked_instances.forward(request, "messages", target)

    response = asyncio.run(run())
    assert response.status_code == 200
    assert seen["url"] == "http://remote/v1/messages"
    assert seen["body"]["model"] == "unsloth/a"
    assert seen["headers"]["authorization"] == f"Bearer {REMOTE_KEY}"
    assert seen["headers"][linked_instances.HOP_HEADER] == "1"
    assert seen["headers"]["anthropic-version"] == "2023-06-01"


def test_forward_streams_sse_through(monkeypatch):
    linked_instances_db.create_instance("wsl", "http://remote", REMOTE_KEY)
    sse = b'data: {"x": 1}\n\ndata: [DONE]\n\n'
    _remote(
        lambda r: httpx.Response(200, content = sse, headers = {"content-type": "text/event-stream"}),
        monkeypatch,
    )
    body = {"model": "@wsl/a", "stream": True, "messages": []}
    request = _request(body)

    async def run():
        response = await linked_instances.forward(
            request, "chat/completions", await linked_instances.resolve(request, body["model"])
        )
        return response, b"".join([chunk async for chunk in response.body_iterator])

    response, content = asyncio.run(run())
    assert response.media_type == "text/event-stream"
    assert content == sse


def test_unreachable_remote_is_a_502(monkeypatch):
    linked_instances_db.create_instance("wsl", "http://remote", REMOTE_KEY)

    def handler(request):
        raise httpx.ConnectError("down", request = request)

    _remote(handler, monkeypatch)
    request = _request({"model": "@wsl/a"})

    async def run():
        await linked_instances.forward(
            request, "chat/completions", await linked_instances.resolve(request, "@wsl/a")
        )

    with pytest.raises(HTTPException) as exc:
        asyncio.run(run())
    assert exc.value.status_code == 502


def test_unknown_instance_is_a_404_and_a_forwarded_request_never_forwards_again():
    with pytest.raises(HTTPException) as exc:
        asyncio.run(linked_instances.resolve(_request({}), "@nope/a"))
    assert exc.value.status_code == 404

    hopped = _request({}, {linked_instances.HOP_HEADER: "1"})
    with pytest.raises(HTTPException) as exc:
        asyncio.run(linked_instances.resolve(hopped, "@nope/a"))
    assert exc.value.status_code == 403
    assert asyncio.run(linked_instances.catalog_objects(hopped)) == []


def test_local_model_ids_are_not_routed():
    assert asyncio.run(linked_instances.resolve(_request({}), "unsloth/Qwen3-0.6B-GGUF")) is None


@pytest.mark.parametrize(
    "url, ok",
    [
        ("http://127.0.0.1:8895/v1", True),
        ("http://localhost:8888", True),
        ("http://192.168.1.20:8888", True),
        ("https://abc.trycloudflare.com", True),
        ("http://8.8.8.8:8888", False),
    ],
)
def test_plain_http_only_on_private_hosts(url, ok):
    if ok:
        assert linked_instances.normalize_base_url(url).startswith(url.split("/v1")[0])
    else:
        with pytest.raises(ValueError, match = "https"):
            linked_instances.normalize_base_url(url)


class _Monitor:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        return lambda *args, **kwargs: self.calls.append((name, args, kwargs)) or "entry-1"


def test_forwarded_requests_show_up_in_the_api_monitor(monkeypatch):
    linked_instances_db.create_instance("wsl", "http://remote", REMOTE_KEY)
    monitor = _Monitor()
    monkeypatch.setattr(linked_instances, "api_monitor", monitor)
    _remote(
        lambda r: httpx.Response(
            200,
            json = {
                "choices": [{"message": {"content": "pong"}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 1},
            },
        ),
        monkeypatch,
    )
    body = {"model": "@wsl/unsloth/a", "messages": [{"role": "user", "content": "ping"}]}
    request = _request(body)

    async def run():
        target = await linked_instances.resolve(request, body["model"])
        await linked_instances.forward(request, "chat/completions", target, subject = "u")

    asyncio.run(run())
    names = [c[0] for c in monitor.calls]
    assert names[0] == "start" and names[-1] == "finish"
    start = monitor.calls[0][2]
    assert start["model"] == "@wsl/unsloth/a" and start["prompt"] == "ping"
    assert ("append_reply", ("entry-1", "pong"), {"stamp_first_token": False}) in monitor.calls


def test_a_remote_error_fails_the_monitor_row(monkeypatch):
    linked_instances_db.create_instance("wsl", "http://remote", REMOTE_KEY)
    monitor = _Monitor()
    monkeypatch.setattr(linked_instances, "api_monitor", monitor)
    _remote(
        lambda r: httpx.Response(404, json = {"error": {"message": "model not found"}}), monkeypatch
    )
    request = _request({"model": "@wsl/x"})

    async def run():
        return await linked_instances.forward(
            request, "chat/completions", await linked_instances.resolve(request, "@wsl/x")
        )

    response = asyncio.run(run())
    assert response.status_code == 404
    assert ("fail", ("entry-1", "model not found"), {}) in monitor.calls


@pytest.mark.parametrize(
    "upstream_type, relayed, attachment",
    [
        ("image/png", "image/png", False),
        ("image/svg+xml", "image/svg+xml", False),
        ("text/html; charset=utf-8", "application/octet-stream", True),
        ("application/xhtml+xml", "application/octet-stream", True),
        ("text/javascript", "application/octet-stream", True),
        ("", "application/octet-stream", True),
    ],
)
def test_proxy_never_repeats_a_document_type_from_a_remote(
    monkeypatch, upstream_type, relayed, attachment
):
    """A relayed body is served from this origin, so only media types survive the trip."""
    instance = linked_instances_db.create_instance("colab", "https://remote.example", REMOTE_KEY)
    headers = {"content-disposition": "inline; filename=x.html"}
    if upstream_type:
        headers["content-type"] = upstream_type
    _remote(
        lambda r: httpx.Response(200, content = b"<script>x</script>", headers = headers), monkeypatch
    )
    response = asyncio.run(
        linked_instances.proxy(
            _proxy_request("GET"), instance, "api/inference/images/gallery/1/file"
        )
    )
    assert response.media_type == relayed
    assert response.headers["x-content-type-options"] == "nosniff"
    assert "sandbox" in response.headers["content-security-policy"]
    assert response.headers.get("content-disposition") == (
        "attachment" if attachment else "inline; filename=x.html"
    )


def test_tool_fields_do_not_reach_an_instance_that_is_not_trusted_with_tools():
    instance = linked_instances_db.create_instance("colab", "https://remote.example", REMOTE_KEY)
    body = {
        "model": "unsloth/x",
        "enable_tools": True,
        "enabled_tools": ["python", "terminal"],
        "mcp_enabled": True,
        "permission_mode": "off",
        "bypass_permissions": True,
        "confirm_tool_calls": True,
        "deep_research_armed": True,
        "messages": [{"role": "user", "content": "hi"}],
    }
    dropped = linked_instances.strip_tool_fields(body, instance)

    assert set(dropped) == {
        "enable_tools",
        "enabled_tools",
        "mcp_enabled",
        "permission_mode",
        "bypass_permissions",
        "confirm_tool_calls",
        "deep_research_armed",
    }
    # An explicit refusal, not just an omission: the remote's own default cannot re-enable them.
    assert body["enable_tools"] is False
    assert not any(
        f in body for f in ("enabled_tools", "mcp_enabled", "permission_mode", "bypass_permissions")
    )
    assert body["messages"] == [{"role": "user", "content": "hi"}]


def test_tool_fields_travel_once_the_owner_allows_tools_for_that_instance():
    instance = linked_instances_db.create_instance("colab", "https://remote.example", REMOTE_KEY)
    linked_instances_db.update_instance(instance["id"], allow_tools = True)
    trusted = linked_instances_db.get_instance(instance["id"])
    body = {"enable_tools": True, "enabled_tools": ["python"], "permission_mode": "auto"}

    assert linked_instances.strip_tool_fields(body, trusted) == []
    assert body == {"enable_tools": True, "enabled_tools": ["python"], "permission_mode": "auto"}


def test_allow_tools_is_off_for_a_new_instance_and_survives_a_rename():
    instance = linked_instances_db.create_instance("colab", "https://remote.example", REMOTE_KEY)
    assert instance["allow_tools"] is False

    linked_instances_db.update_instance(instance["id"], allow_tools = True)
    renamed = linked_instances_db.update_instance(instance["id"], name = "colab2")
    assert renamed["allow_tools"] is True

    assert (
        linked_instances_db.update_instance(instance["id"], allow_tools = False)["allow_tools"]
        is False
    )


def test_forward_strips_tool_fields_before_they_leave_this_machine(monkeypatch):
    """The end-to-end shape of the fix: what the remote actually receives."""
    linked_instances_db.create_instance("wsl", "http://remote", REMOTE_KEY)
    monkeypatch.setattr(linked_instances, "api_monitor", _Monitor())
    seen = {}

    def handler(request: httpx.Request):
        seen.update(json.loads(request.content))
        return httpx.Response(200, json = {"choices": [{"message": {"content": "ok"}}]})

    _remote(handler, monkeypatch)
    body = {
        "model": "@wsl/unsloth/a",
        "enable_tools": True,
        "enabled_tools": ["python"],
        "permission_mode": "off",
        "messages": [{"role": "user", "content": "run something"}],
    }
    request = _request(body)

    async def run():
        target = await linked_instances.resolve(request, body["model"])
        return await linked_instances.forward(request, "chat/completions", target)

    asyncio.run(run())
    assert seen["enable_tools"] is False
    assert "enabled_tools" not in seen and "permission_mode" not in seen
    assert seen["model"] == "unsloth/a"


def test_info_merges_the_remotes_system_endpoints(monkeypatch):
    instance = linked_instances_db.create_instance("colab", "http://remote", REMOTE_KEY)

    def handler(request: httpx.Request):
        assert request.headers["authorization"] == f"Bearer {REMOTE_KEY}"
        if request.url.path == "/api/system":
            return httpx.Response(
                200,
                json = {
                    "platform": "Linux",
                    "cpu_count": 12,
                    "memory": {"total_gb": 53.0},
                    "gpu": {
                        "devices": [
                            {"name": "NVIDIA L4", "memory_total_gb": 22.5, "vram_used_gb": 11.7}
                        ]
                    },
                },
            )
        if request.url.path == "/api/system/hardware":
            return httpx.Response(
                200, json = {"versions": {"unsloth": "2026.9.1", "cuda": "12.8"}, "llama_cpp": "b1"}
            )
        return httpx.Response(404)

    _remote(handler, monkeypatch)
    info = asyncio.run(linked_instances.fetch_info(instance))
    assert info["online"] and info["version"] == "2026.9.1" and info["cuda"] == "12.8"
    assert info["gpus"] == [
        {"name": "NVIDIA L4", "vram_total_gb": 22.5, "vram_used_gb": 11.7, "utilization_pct": None}
    ]
    assert info["cpu_count"] == 12 and info["install_source"] is None
    assert info["image_model"] is None  # the images endpoint 404s, as on an older release


def test_info_names_a_loaded_image_model(monkeypatch):
    instance = linked_instances_db.create_instance("colab", "http://remote", REMOTE_KEY)

    def handler(request: httpx.Request):
        if request.url.path == "/api/inference/images/status":
            return httpx.Response(
                200, json = {"loaded": True, "repo_id": "unsloth/Qwen-Image-2.1-GGUF"}
            )
        return httpx.Response(200, json = {})

    _remote(handler, monkeypatch)
    assert (
        asyncio.run(linked_instances.fetch_info(instance))["image_model"]
        == "unsloth/Qwen-Image-2.1-GGUF"
    )


def test_info_reports_a_rejected_key(monkeypatch):
    instance = linked_instances_db.create_instance("colab", "http://remote", REMOTE_KEY)
    _remote(lambda request: httpx.Response(401), monkeypatch)
    assert asyncio.run(linked_instances.fetch_info(instance)) == {
        "online": False,
        "error": "The API key was rejected.",
    }


def test_reasoning_deltas_stamp_the_first_token(monkeypatch):
    monitor = _Monitor()
    monkeypatch.setattr(linked_instances, "api_monitor", monitor)
    linked_instances._record_sse_line(
        "e", 'data: {"choices": [{"delta": {"reasoning_content": "hmm"}}]}'
    )
    linked_instances._record_sse_line("e", 'data: {"choices": [{"delta": {"content": "hi"}}]}')
    names = [c[0] for c in monitor.calls if c[0] in ("mark_first_token", "append_reply")]
    assert names == ["mark_first_token", "append_reply"]


@pytest.mark.parametrize("caller_asked", [False, True])
def test_stream_usage_is_requested_counted_and_hidden_unless_asked(monkeypatch, caller_asked):
    linked_instances_db.create_instance("wsl", "http://remote", REMOTE_KEY)
    monitor = _Monitor()
    monkeypatch.setattr(linked_instances, "api_monitor", monitor)
    usage = b'data: {"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 7}}\n\n'
    sse = (
        b'data: {"choices": [{"delta": {"content": "hi \xc3\xa9"}}]}\n\n'
        + usage
        + b"data: [DONE]\n\n"
    )
    sent = {}

    def handler(request: httpx.Request):
        sent.update(json.loads(request.content))
        return httpx.Response(200, content = sse, headers = {"content-type": "text/event-stream"})

    _remote(handler, monkeypatch)
    body = {"model": "@wsl/a", "stream": True, "messages": []}
    if caller_asked:
        body["stream_options"] = {"include_usage": True}
    request = _request(body)

    async def run():
        response = await linked_instances.forward(
            request, "chat/completions", await linked_instances.resolve(request, body["model"])
        )
        return b"".join([chunk async for chunk in response.body_iterator])

    content = asyncio.run(run())
    assert sent["stream_options"] == {"include_usage": True}
    assert content == (sse if caller_asked else sse.replace(usage, b"\n"))
    assert (
        "set_usage",
        ("entry-1",),
        {"prompt_tokens": 5, "completion_tokens": 7},
    ) in monitor.calls
    assert ("append_reply", ("entry-1", "hi é"), {}) in monitor.calls


@pytest.mark.parametrize(
    "method, path, ok",
    [
        ("GET", "api/hub/cached-gguf", True),
        ("GET", "api/hub/gguf-variants", True),
        ("POST", "api/hub/download", True),
        ("POST", "api/inference/load", True),
        ("POST", "api/inference/images/generate", True),
        ("GET", "api/inference/images/gallery/abc/file", True),
        ("DELETE", "api/hub/download", False),
        ("GET", "api/auth/api-keys", False),
        ("POST", "api/settings/anything", False),
        ("GET", "api/inference/images/../../auth/api-keys", False),
        ("POST", "api/train/start", False),
    ],
)
def test_proxy_only_reaches_the_picker_and_image_routes(method, path, ok):
    assert linked_instances.proxy_allowed(method, path) is ok


def _proxy_request(
    method: str,
    query: bytes = b"",
    body: bytes = b"",
    headers: dict | None = None,
) -> Request:
    sent = False

    async def receive():
        nonlocal sent
        if sent:
            return {"type": "http.disconnect"}
        sent = True
        return {"type": "http.request", "body": body, "more_body": False}

    header_list = [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()]
    return Request(
        {
            "type": "http",
            "method": method,
            "path": "/p",
            "query_string": query,
            "headers": header_list,
        },
        receive,
    )


def test_proxy_sends_the_key_and_rewrites_gallery_urls(monkeypatch):
    instance = linked_instances_db.create_instance("colab", "https://remote.example", REMOTE_KEY)
    seen = {}

    def handler(request: httpx.Request):
        seen.update(
            url = str(request.url),
            auth = request.headers.get("authorization"),
            hop = request.headers.get(linked_instances.HOP_HEADER),
            hf = request.headers.get("x-unsloth-hf-token"),
            body = request.content,
        )
        return httpx.Response(
            200,
            json = {
                "images": [
                    {"id": "i1", "url": "/api/inference/images/gallery/i1/file", "prompt": "/api/x"}
                ]
            },
        )

    _remote(handler, monkeypatch)
    request = _proxy_request(
        "POST",
        body = b'{"prompt": "a sloth"}',
        headers = {"content-type": "application/json", "x-unsloth-hf-token": "hf_x", "cookie": "s=1"},
    )
    response = asyncio.run(
        linked_instances.proxy(request, instance, "api/inference/images/generate")
    )
    assert seen["url"] == "https://remote.example/api/inference/images/generate"
    assert seen["auth"] == f"Bearer {REMOTE_KEY}" and seen["hop"] == "1" and seen["hf"] == "hf_x"
    assert seen["body"] == b'{"prompt": "a sloth"}'
    image = json.loads(response.body)["images"][0]
    assert (
        image["url"]
        == f"/api/linked-instances/{instance['id']}/proxy/api/inference/images/gallery/i1/file"
    )
    assert image["prompt"] == "/api/x"


def test_proxy_keeps_the_query_and_streams_binary(monkeypatch):
    instance = linked_instances_db.create_instance("colab", "https://remote.example", REMOTE_KEY)
    seen = {}

    def handler(request: httpx.Request):
        seen["url"] = str(request.url)
        return httpx.Response(200, content = b"\x89PNG", headers = {"content-type": "image/png"})

    _remote(handler, monkeypatch)
    request = _proxy_request("GET", query = b"repo_id=unsloth%2Fa")
    response = asyncio.run(linked_instances.proxy(request, instance, "api/hub/gguf-variants"))

    async def body():
        return b"".join([chunk async for chunk in response.body_iterator])

    assert seen["url"] == "https://remote.example/api/hub/gguf-variants?repo_id=unsloth%2Fa"
    assert response.media_type == "image/png" and asyncio.run(body()) == b"\x89PNG"


def test_the_tools_off_header_rides_until_the_owner_allows_tools(monkeypatch):
    row = linked_instances_db.create_instance("wsl", "http://remote", REMOTE_KEY)
    seen = []

    def handler(request: httpx.Request):
        seen.append(request.headers.get(linked_instances.TOOLS_OFF_HEADER))
        return httpx.Response(200, json = {"ok": True})

    _remote(handler, monkeypatch)
    body = {"model": "@wsl/unsloth/a", "messages": [], "enable_tools": True}

    async def run():
        request = _request(dict(body))
        target = await linked_instances.resolve(request, body["model"])
        return await linked_instances.forward(request, "chat/completions", target)

    asyncio.run(run())
    linked_instances_db.update_instance(row["id"], allow_tools = True)
    linked_instances.forget(row["id"])
    asyncio.run(run())
    assert seen == ["1", None]


def test_the_tools_off_header_beats_a_remote_launched_with_enable_tools():
    from types import SimpleNamespace

    from starlette.applications import Starlette
    from starlette.responses import JSONResponse
    from starlette.routing import Route
    from starlette.testclient import TestClient

    from routes.inference import _effective_enable_tools
    from state.tool_policy import reset_tool_policy, set_tool_policy

    async def endpoint(request):
        return JSONResponse({"tools": _effective_enable_tools(SimpleNamespace(enable_tools = False))})

    app = linked_instances.LinkedToolsOffMiddleware(Starlette(routes = [Route("/", endpoint)]))
    set_tool_policy(True)
    try:
        with TestClient(app) as client:
            assert client.get("/").json() == {"tools": True}
            off = client.get("/", headers = {linked_instances.TOOLS_OFF_HEADER: "1"})
            assert off.json() == {"tools": False}
    finally:
        reset_tool_policy()


def test_a_plain_http_remote_that_now_resolves_public_never_gets_the_key(monkeypatch):
    linked_instances_db.create_instance("wsl", "http://remote", REMOTE_KEY)
    sent = []

    def handler(request: httpx.Request):
        sent.append(request.headers.get("authorization"))
        return httpx.Response(200, json = {"ok": True})

    _remote(handler, monkeypatch)
    monkeypatch.setitem(LAN_ADDRESS, "remote", "8.8.8.8")
    body = {"model": "@wsl/unsloth/a", "messages": []}
    request = _request(body)

    async def run():
        target = await linked_instances.resolve(request, body["model"])
        return await linked_instances.forward(request, "chat/completions", target)

    with pytest.raises(HTTPException) as refused:
        asyncio.run(run())
    assert refused.value.status_code == 502
    assert sent == []


def test_status_and_info_report_a_rebound_http_remote_offline(monkeypatch):
    row = linked_instances_db.create_instance("wsl", "http://remote", REMOTE_KEY)
    sent = []

    def handler(request: httpx.Request):
        sent.append(request.url.path)
        return httpx.Response(200, json = {"data": []})

    _remote(handler, monkeypatch)
    monkeypatch.setitem(LAN_ADDRESS, "remote", "8.8.8.8")
    instance = linked_instances_db.get_instance(row["id"])
    assert asyncio.run(linked_instances.probe(instance))["online"] is False
    assert asyncio.run(linked_instances.fetch_info(instance))["online"] is False
    assert sent == []


def test_linked_url_images_are_served_from_this_machine(monkeypatch):
    import base64
    import io

    from PIL import Image

    import routes.inference as inference
    from core.inference import image_gallery

    buffer = io.BytesIO()
    Image.new("RGB", (16, 8), (1, 2, 3)).save(buffer, format = "PNG")
    remote = {
        "created": 5,
        "data": [{"b64_json": base64.b64encode(buffer.getvalue()).decode(), "revised_prompt": "p"}],
    }
    saved = []

    def save(image, meta):
        saved.append((image.size, meta))
        return {"id": "local-1"}

    monkeypatch.setattr(image_gallery, "save", save)
    monkeypatch.setattr(inference, "_sign_image_id", lambda image_id: "tok")
    request = Request(
        {
            "type": "http",
            "scheme": "http",
            "server": ("studio.local", 8888),
            "path": "/v1/x",
            "headers": [(b"host", b"studio.local:8888")],
        }
    )
    response = inference._linked_images_as_local_urls(
        request, json.dumps(remote).encode(), "a cat", ({"name": "wsl"}, "unsloth/flux")
    )
    item = json.loads(response.body)["data"][0]
    assert item == {
        "revised_prompt": "p",
        "url": "http://studio.local:8888/api/inference/images/gallery/local-1/file-signed?token=tok",
    }
    assert saved[0][0] == (16, 8)
    assert saved[0][1]["model"] == "@wsl/unsloth/flux"


def test_forward_applies_body_overrides(monkeypatch):
    linked_instances_db.create_instance("wsl", "http://remote", REMOTE_KEY)
    seen = {}

    def handler(request: httpx.Request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json = {"data": []})

    _remote(handler, monkeypatch)
    body = {"model": "@wsl/unsloth/flux", "prompt": "x", "response_format": "url"}
    request = _request(body)

    async def run():
        target = await linked_instances.resolve(request, body["model"])
        return await linked_instances.forward(
            request, "images/generations", target, body_overrides = {"response_format": "b64_json"}
        )

    asyncio.run(run())
    assert seen["body"]["response_format"] == "b64_json"
    assert seen["body"]["model"] == "unsloth/flux"


def test_a_linked_url_image_request_never_returns_the_remotes_own_url(monkeypatch):
    import base64
    import io

    from fastapi import Response
    from PIL import Image

    import routes.inference as inference
    from core.inference import image_gallery
    from models.inference import ImageGenerationRequest

    buffer = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buffer, format = "PNG")
    b64 = base64.b64encode(buffer.getvalue()).decode()
    remote_url = "http://10.0.0.5:8888/api/inference/images/gallery/r1/file-signed?token=x"

    async def resolve(request, model):
        return ({"id": "i", "name": "wsl"}, "unsloth/flux")

    async def forward(request, path, target, **kwargs):
        if (kwargs.get("body_overrides") or {}).get("response_format") == "b64_json":
            data = {"created": 1, "data": [{"b64_json": b64}]}
        else:
            data = {"created": 1, "data": [{"url": remote_url}]}
        return Response(json.dumps(data), media_type = "application/json")

    monkeypatch.setattr(inference.linked_instances, "resolve", resolve)
    monkeypatch.setattr(inference.linked_instances, "forward", forward)
    monkeypatch.setattr(image_gallery, "save", lambda image, meta: {"id": "local-1"})
    monkeypatch.setattr(inference, "_sign_image_id", lambda image_id: "tok")
    request = Request(
        {
            "type": "http",
            "scheme": "http",
            "server": ("studio.local", 8888),
            "path": "/v1/x",
            "headers": [(b"host", b"studio.local:8888")],
        }
    )
    endpoint = getattr(
        inference.openai_image_generations, "__wrapped__", inference.openai_image_generations
    )
    response = asyncio.run(
        endpoint(
            ImageGenerationRequest(prompt = "a cat", model = "@wsl/unsloth/flux"),
            request,
            current_subject = "unsloth",
            hf_token = None,
        )
    )
    url = json.loads(response.body)["data"][0]["url"]
    assert url.startswith("http://studio.local:8888/api/inference/images/gallery/local-1/")
