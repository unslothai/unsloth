# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Per-connection Custom reasoning contracts, including actual HTTP payloads."""

import asyncio
import base64
import json
import os
import re
import sqlite3
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import ValidationError

from auth import storage as auth_storage
from core.inference import external_provider as ep, key_exchange
from models.inference import ChatCompletionRequest
from models.providers import ProviderCreate, ProviderUpdate
from routes import inference, providers
from storage import credential_secrets, providers_db
from .test_external_provider_sampling_over_the_wire import _Server

STYLES = ("reasoning_effort", "reasoning", "thinking", "chat_template_kwargs.enable_thinking")
BAD = [
    {"enabled": "true", "style": "thinking"},
    {"enabled": 1, "style": "thinking"},
    {"enabled": True, "style": "unknown"},
    {"enabled": True, "style": "thinking", "extra": 1},
    {"style": "thinking"},
    ["thinking"],
]
KEYS = {"reasoning_effort", "reasoning", "thinking", "chat_template_kwargs", "enable_thinking"}
PROVIDER = {"provider_type": "custom", "display_name": "Gateway"}
CHAT = {"messages": [{"role": "user", "content": "hi"}], "model": "test-model"}


def config(style = "reasoning_effort", enabled = True):
    return {"enabled": enabled, "style": style}


def expected(style, effort = "medium"):
    off = effort == "none"
    return {
        "reasoning_effort": {"reasoning_effort": effort},
        "reasoning": {"reasoning": {"enabled": False} if off else {"effort": effort}},
        "thinking": {"thinking": {"type": "disabled" if off else "enabled"}},
        "chat_template_kwargs.enable_thinking": {
            "chat_template_kwargs": {"enable_thinking": not off}
        },
    }[style]


def reasoning(body):
    return {key: value for key, value in body.items() if key in KEYS}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.recorded.append(
            {"path": self.path, "body": body, "auth": self.headers.get("Authorization")}
        )
        data = (
            b'data: {"type":"response.output_text.delta","delta":"ok"}\n\n'
            b'data: {"type":"response.completed","response":{"id":"resp_test"}}\n\n'
            if self.path.endswith("/responses")
            else b'data: {"choices":[{"index":0,"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n'
        )
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture()
def harness(tmp_path, monkeypatch, request):
    # conftest isolates studio.db and schema caches; auth.DB_PATH is import-time state.
    monkeypatch.setattr(auth_storage, "DB_PATH", tmp_path / "auth.db")
    monkeypatch.setattr(auth_storage, "_credential_encryption_key_cache", None)
    app = FastAPI()
    app.include_router(providers.router, prefix = "/api/providers")
    app.dependency_overrides[providers.get_current_subject] = lambda: "alice"
    app.dependency_overrides[providers.get_current_credential] = lambda: ("alice", None)
    app.dependency_overrides[providers.authenticated_via_api_key] = lambda: False

    async def disconnected():
        return False

    route_request = SimpleNamespace(
        headers = {}, state = SimpleNamespace(skip_api_monitor = True), is_disconnected = disconnected
    )
    with _Server(Handler) as server, TestClient(app) as client:

        async def send(
            mode = "client",
            cfg = None,
            api_type = "chat_completions",
            **fields,
        ):
            async with httpx.AsyncClient(trust_env = False) as transport:
                monkeypatch.setattr(ep, "_http_client", transport)
                if mode == "client":
                    upstream = ep.ExternalProviderClient(
                        "custom", server.base_url, "", api_type = api_type, reasoning_config = cfg
                    )
                    output = upstream.stream_chat_completion(**CHAT, **fields)
                else:
                    defaults = dict(
                        CHAT,
                        stream = True,
                        provider_type = "custom",
                        provider_base_url = server.base_url,
                        provider_reasoning_config = cfg,
                    )
                    payload = ChatCompletionRequest(**{**defaults, **fields})
                    response = await inference._proxy_to_external_provider(payload, route_request)
                    output = response.body_iterator
                assert "ok" in "".join([line async for line in output])
            path = "responses" if api_type == "responses" else "chat/completions"
            assert server.bodies[-1]["path"] == f"/v1/{path}"
            return server.bodies[-1]["body"]

        def api(
            method,
            path = "/",
            status = 200,
            **payload,
        ):
            response = client.request(method, "/api/providers" + path, json = payload)
            assert response.status_code == status, response.text
            return response.json()

        yield SimpleNamespace(
            db = providers_db.studio_db_path(),
            url = server.base_url,
            api = api,
            send = lambda *args, **kwargs: asyncio.run(send(*args, **kwargs)),
            recorded = server.bodies,
        )
    root = os.environ.get("UNSLOTH_REASONING_EVIDENCE_DIR")
    if root:
        path = Path(root)
        path.mkdir(parents = True, exist_ok = True)
        name = re.sub(r"[^a-zA-Z0-9_.-]", "_", request.node.name)
        (path / f"{name}.json").write_text(json.dumps(server.bodies, indent = 2) + "\n")


@pytest.mark.parametrize("schema", [ProviderCreate, ProviderUpdate, ChatCompletionRequest])
def test_strict_schema(schema):
    field = "provider_reasoning_config" if schema is ChatCompletionRequest else "reasoning_config"
    payload = dict(PROVIDER, messages = [])
    for value in [None, *[config(style, enabled) for style in STYLES for enabled in (True, False)]]:
        assert schema(**payload, **{field: value}).model_dump()[field] == value
    for bad in BAD:
        with pytest.raises(ValidationError):
            schema(**payload, **{field: bad})


@pytest.mark.parametrize("style", STYLES)
def test_api_persistence_and_wire(harness, style):
    h = harness
    created = h.api("POST", status = 201, **PROVIDER, base_url = h.url, reasoning_config = config(style))
    assert created["reasoning_config"] == config(style)
    provider_id = created["id"]
    for update, value in [
        ({"display_name": "Renamed"}, config(style)),
        ({"reasoning_config": config(style, False)}, config(style, False)),
        ({"reasoning_config": None}, None),
    ]:
        assert h.api("PUT", f"/{provider_id}", **update)["reasoning_config"] == value
        providers_db.reset_schema_state_for_tests()
        assert providers_db.get_provider(provider_id)["reasoning_config"] == value
        assert h.api("GET")[0]["reasoning_config"] == value
        assert providers_db.list_providers()[0]["reasoning_config"] == value
        body = h.send("route", config("thinking"), provider_id = provider_id, reasoning_effort = "high")
        assert reasoning(body) == (expected(style, "high") if value and value["enabled"] else {})


@pytest.mark.parametrize("style", STYLES)
@pytest.mark.parametrize("mode", ["client", "route"])
def test_style_controls_real_http(harness, style, mode):
    for controls, effort in [
        ({}, "medium"),
        ({"enable_thinking": True}, "medium"),
        ({"enable_thinking": False}, "none"),
        ({"reasoning_effort": "none"}, "none"),
        ({"enable_thinking": False, "reasoning_effort": "max"}, "none"),
        *[({"reasoning_effort": e}, e) for e in ("low", "medium", "high")],
        ({"reasoning_effort": "max"}, None),
    ]:
        body = harness.send(mode, config(style), **controls)
        assert reasoning(body) == (expected(style, effort) if effort else {}), controls


def test_fail_closed_and_legacy_body(harness):
    h = harness
    controls = {"enable_thinking": True, "reasoning_effort": "high"}
    legacy = h.send()
    for value in [None, *[config(style, False) for style in STYLES], *BAD]:
        assert h.send(cfg = value, **controls) == legacy
    with sqlite3.connect(h.db) as conn:
        conn.execute(
            "CREATE TABLE llm_providers (id TEXT PRIMARY KEY, provider_type TEXT, display_name TEXT, base_url TEXT, is_enabled INTEGER, created_at TEXT, updated_at TEXT)"
        )
        conn.execute(
            "INSERT INTO llm_providers VALUES ('legacy', 'custom', 'Gateway', ?, 1, '', '')",
            (h.url,),
        )
    assert providers_db.get_provider("legacy")["reasoning_config"] is None
    legacy_route = h.send("route", provider_id = "legacy")
    assert reasoning(legacy_route) == {}
    for raw in ["not json", "[]", *[json.dumps(bad) for bad in BAD]]:
        with sqlite3.connect(h.db) as conn:
            conn.execute("UPDATE llm_providers SET reasoning_config_json = ?", (raw,))
        assert providers_db.get_provider("legacy")["reasoning_config"] is None
        assert providers_db.list_providers()[0]["reasoning_config"] is None
        assert h.send("route", config(), provider_id = "legacy", **controls) == legacy_route


@pytest.mark.parametrize("api_type", ["responses", "systemone"])
def test_incompatible_contracts(harness, api_type):
    h = harness
    payload = dict(PROVIDER, api_type = api_type, reasoning_config = config())
    h.api("POST", status = 400, **payload)
    with pytest.raises(ValueError, match = "Chat Completions"):
        ep.ExternalProviderClient("custom", h.url, "", api_type = api_type, reasoning_config = config())
    if api_type == "responses":
        with pytest.raises(HTTPException) as exc:
            h.send("route", config(), provider_api_type = api_type)
        assert exc.value.status_code == 400
    assert h.recorded == []
    payload["api_type"] = "chat_completions"
    provider_id = h.api("POST", status = 201, **payload)["id"]
    for update, status in [
        ({"api_type": api_type}, 400),
        ({"api_type": api_type, "reasoning_config": config(enabled = False)}, 200),
        ({"reasoning_config": config()}, 400),
        ({"reasoning_config": None}, 200),
    ]:
        response = h.api("PUT", f"/{provider_id}", status, **update)
    assert response["api_type"] == api_type
    for extra, status in [({"reasoning_config": BAD[3]}, 422), ({"provider_type": "ollama"}, 400)]:
        h.api("POST", status = status, **{**payload, **extra})


@pytest.mark.parametrize("encrypted", [False, True])
def test_saved_authority_and_connection_isolation(harness, monkeypatch, encrypted):
    if encrypted:
        private = rsa.generate_private_key(public_exponent = 65537, key_size = 2048)
        monkeypatch.setattr(key_exchange, "_private_key", private)
        oaep = padding.OAEP(
            mgf = padding.MGF1(hashes.SHA256()), algorithm = hashes.SHA256(), label = None
        )
        cipher = base64.b64encode(private.public_key().encrypt(b"browser-test-key", oaep)).decode()
    for provider_id, value in [
        ("enabled", config("thinking")),
        ("disabled", config(enabled = False)),
        ("legacy", None),
        ("enabled-again", config("thinking")),
    ]:
        providers_db.create_provider(
            provider_id, "custom", provider_id, harness.url, reasoning_config = value
        )
        credential_secrets.save_provider_api_key(provider_id, "local-test-key")
        fields = dict(
            provider_id = provider_id,
            provider_type = "openai",
            provider_api_type = "responses",
            provider_base_url = "https://unrelated.invalid/v1",
            reasoning_effort = "high",
            enable_thinking = True,
        )
        if encrypted:
            fields["encrypted_api_key"] = cipher
        body = harness.send("route", config(), **fields)
        assert reasoning(body) == (expected("thinking") if value and value["enabled"] else {})
        assert harness.recorded[-1]["auth"] == (
            "Bearer browser-test-key" if encrypted else "Bearer local-test-key"
        )


def test_saved_config_race_refused(harness, monkeypatch):
    rows = iter([config(), config("thinking")])
    row = dict(PROVIDER, base_url = harness.url, api_type = "chat_completions", is_enabled = True)
    monkeypatch.setattr(
        providers_db, "get_provider", lambda _: dict(row, reasoning_config = next(rows))
    )
    with pytest.raises(HTTPException) as exc:
        harness.send("route", provider_id = "saved")
    assert exc.value.status_code == 409
    assert harness.recorded == []


def test_responses_legacy_semantics(harness):
    bodies = [
        harness.send(cfg = value, api_type = "responses", reasoning_effort = "high")
        for value in (None, config("thinking", False))
    ]
    assert bodies[0] == bodies[1]
    assert bodies[0]["reasoning"] == {"effort": "high", "summary": "auto"}
