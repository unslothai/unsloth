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
