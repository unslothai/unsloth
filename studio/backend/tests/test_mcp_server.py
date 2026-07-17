# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import sys
import types

import pytest

from mcp_server import BearerTokenMiddleware, _clamp, _dump, create_studio_mcp


def _get_tool(name):
    tools = asyncio.run(create_studio_mcp().list_tools())
    return {tool.name: tool for tool in tools}[name]


def test_studio_mcp_registers_control_plane_tools():
    tools = asyncio.run(create_studio_mcp().list_tools())

    assert {tool.name for tool in tools} == {
        "studio_status",
        "list_local_models",
        "get_training_status",
        "start_training",
        "stop_training",
        "list_training_runs",
        "validate_recipe",
        "get_recipe_job_status",
        "get_recipe_job_dataset",
        "load_checkpoint",
        "export_gguf",
    }


def test_dump_serializes_pydantic_values():
    class Response:
        def model_dump(self, *, mode):
            assert mode == "json"
            return {"ok": True}

    assert _dump(Response()) == {"ok": True}
    assert _dump({"already": "json"}) == {"already": "json"}


def test_bearer_token_middleware_rejects_wrong_token():
    events = []

    async def app(scope, receive, send):
        events.append("app")

    async def send(message):
        events.append(message)

    middleware = BearerTokenMiddleware(app, "secret")
    asyncio.run(
        middleware(
            {"type": "http", "headers": [(b"authorization", b"Bearer wrong")]},
            None,
            send,
        )
    )

    assert events[0]["status"] == 401
    assert "app" not in events


def test_bearer_token_middleware_closes_unauthorized_websocket():
    events = []

    async def app(scope, receive, send):
        events.append("app")

    async def send(message):
        events.append(message)

    middleware = BearerTokenMiddleware(app, "secret")
    asyncio.run(
        middleware(
            {"type": "websocket", "headers": []},
            None,
            send,
        )
    )

    assert events == [{"type": "websocket.close", "code": 4401}]


def test_bearer_token_middleware_rejects_non_ascii_authorization():
    # A non-ASCII bearer value must produce a clean 401, not a 500. Comparing on
    # bytes avoids the str hmac.compare_digest TypeError on non-ASCII input.
    events = []

    async def app(scope, receive, send):
        events.append("app")

    async def send(message):
        events.append(message)

    middleware = BearerTokenMiddleware(app, "secret")
    asyncio.run(
        middleware(
            {"type": "http", "headers": [(b"authorization", b"Bearer \xff\xff")]},
            None,
            send,
        )
    )

    assert events[0]["status"] == 401
    assert "app" not in events


def test_bearer_token_middleware_accepts_correct_token():
    events = []

    async def app(scope, receive, send):
        events.append("app")

    async def send(message):
        events.append(message)

    middleware = BearerTokenMiddleware(app, "secret")
    asyncio.run(
        middleware(
            {"type": "http", "headers": [(b"authorization", b"Bearer secret")]},
            None,
            send,
        )
    )

    assert events == ["app"]


def test_bearer_token_middleware_requires_non_empty_token():
    async def app(scope, receive, send):
        pass

    for bad in ("", "   "):
        with pytest.raises(ValueError):
            BearerTokenMiddleware(app, bad)


def test_bearer_token_middleware_passes_through_non_http_scopes():
    events = []

    async def app(scope, receive, send):
        events.append("app")

    async def send(message):
        events.append(message)

    middleware = BearerTokenMiddleware(app, "secret")
    asyncio.run(middleware({"type": "lifespan"}, None, send))

    assert events == ["app"]


def test_clamp_restricts_to_inclusive_bounds():
    assert _clamp(5, 1, 200) == 5
    assert _clamp(-10, 1, 200) == 1
    assert _clamp(10_000, 1, 200) == 200
    assert _clamp(0, 1, 500) == 1
    assert _clamp(1_000, 1, 500) == 500


def test_export_and_checkpoint_tools_expose_forwarded_fields():
    export_props = set(_get_tool("export_gguf").parameters["properties"])
    assert {"hf_token", "imatrix", "imatrix_path"} <= export_props

    checkpoint_props = set(_get_tool("load_checkpoint").parameters["properties"])
    assert {"hf_token", "approved_remote_code_fingerprint"} <= checkpoint_props


def _stub_module(monkeypatch, name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    if "." in name:
        module.__path__ = []  # mark package-like so submodule imports resolve
    monkeypatch.setitem(sys.modules, name, module)
    return module


def test_export_gguf_forwards_hf_token_and_imatrix(monkeypatch):
    captured = {}

    class FakeExportGGUFRequest:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    async def fake_export(request, current_subject):
        return {"current_subject": current_subject}

    _stub_module(monkeypatch, "models", ExportGGUFRequest = FakeExportGGUFRequest)
    _stub_module(monkeypatch, "routes")
    _stub_module(monkeypatch, "routes.export", export_gguf = fake_export)

    tool = _get_tool("export_gguf")
    result = asyncio.run(
        tool.fn(
            save_directory = "/tmp/out",
            quantization_method = ["Q4_K_M", "Q8_0"],
            push_to_hub = True,
            repo_id = "me/model",
            hf_token = "hf_secret",
            imatrix = True,
            imatrix_path = "/tmp/imatrix.dat",
        )
    )

    assert captured["hf_token"] == "hf_secret"
    assert captured["imatrix"] is True
    assert captured["imatrix_path"] == "/tmp/imatrix.dat"
    assert captured["quantization_method"] == ["Q4_K_M", "Q8_0"]
    assert result["current_subject"] == "mcp"


def test_load_checkpoint_forwards_token_and_fingerprint(monkeypatch):
    captured = {}

    class FakeLoadCheckpointRequest:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    async def fake_load(request, current_subject):
        return {"current_subject": current_subject}

    _stub_module(monkeypatch, "models", LoadCheckpointRequest = FakeLoadCheckpointRequest)
    _stub_module(monkeypatch, "routes")
    _stub_module(monkeypatch, "routes.export", load_checkpoint = fake_load)

    tool = _get_tool("load_checkpoint")
    asyncio.run(
        tool.fn(
            checkpoint_path = "/tmp/ckpt",
            approved_remote_code_fingerprint = "sha256:abc",
            hf_token = "hf_secret",
        )
    )

    assert captured["hf_token"] == "hf_secret"
    assert captured["approved_remote_code_fingerprint"] == "sha256:abc"


def test_list_training_runs_clamps_pagination(monkeypatch):
    captured = {}

    async def fake_list_runs(limit, offset, current_subject):
        captured["limit"] = limit
        captured["offset"] = offset
        return {"ok": True}

    _stub_module(monkeypatch, "routes")
    _stub_module(monkeypatch, "routes.training_history", list_training_runs = fake_list_runs)

    tool = _get_tool("list_training_runs")
    asyncio.run(tool.fn(limit = 10_000, offset = -5))

    assert captured["limit"] == 200
    assert captured["offset"] == 0


def test_get_recipe_job_dataset_clamps_pagination(monkeypatch):
    captured = {}

    def fake_job_dataset(job_id, limit, offset):
        captured["limit"] = limit
        captured["offset"] = offset
        return {"ok": True}

    _stub_module(monkeypatch, "routes")
    _stub_module(monkeypatch, "routes.data_recipe")
    _stub_module(monkeypatch, "routes.data_recipe.jobs", job_dataset = fake_job_dataset)

    tool = _get_tool("get_recipe_job_dataset")  # this tool is synchronous
    tool.fn(job_id = "job-1", limit = -1, offset = -9)

    assert captured["limit"] == 1
    assert captured["offset"] == 0
