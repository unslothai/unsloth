# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Strict, per-connection Custom reasoning configuration and its actual HTTP payloads."""

import asyncio
import json
import os
import re
import sqlite3
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from core.inference import external_provider as ep
from core.inference.external_provider import ExternalProviderClient
from models.inference import ChatCompletionRequest
from models.providers import ProviderCreate, ProviderUpdate
from storage import providers_db


STYLES = ("reasoning_effort", "reasoning", "thinking", "chat_template_kwargs.enable_thinking")
REASONING_KEYS = {
    "reasoning_effort",
    "reasoning",
    "thinking",
    "chat_template_kwargs",
    "enable_thinking",
}


def _config(style = "reasoning_effort", enabled = True):
    return {"enabled": enabled, "style": style}


@pytest.mark.parametrize("schema", [ProviderCreate, ProviderUpdate, ChatCompletionRequest])
def test_schema_retains_explicit_reasoning_config(schema):
    field = "provider_reasoning_config" if schema is ChatCompletionRequest else "reasoning_config"
    payload = schema(
        provider_type = "custom", display_name = "Gateway", messages = [], **{field: _config()}
    )
    assert payload.model_dump().get(field) == _config()


@pytest.mark.parametrize("schema", [ProviderCreate, ProviderUpdate, ChatCompletionRequest])
@pytest.mark.parametrize(
    "bad",
    [
        {"enabled": "true", "style": "thinking"},
        {"enabled": 1, "style": "thinking"},
        {"enabled": True, "style": "unknown"},
        {"enabled": True, "style": "thinking", "extra": 1},
        {"style": "thinking"},
        ["thinking"],
    ],
)
def test_config_schema_rejects_malformed(schema, bad):
    field = "provider_reasoning_config" if schema is ChatCompletionRequest else "reasoning_config"
    with pytest.raises(ValidationError):
        schema(provider_type = "custom", display_name = "Gateway", messages = [], **{field: bad})


@pytest.fixture()
def db(tmp_path, monkeypatch):
    path = tmp_path / "studio.db"
    monkeypatch.setattr(providers_db, "studio_db_path", lambda: path)
    providers_db.reset_schema_state_for_tests()
    yield path
    providers_db.reset_schema_state_for_tests()


@pytest.mark.parametrize("style", STYLES)
def test_config_persists_preserves_and_clears(db, style):
    payload = ProviderCreate(
        provider_type = "custom",
        display_name = "Gateway",
        base_url = "https://gateway.example/v1",
        reasoning_config = _config(style),
    )
    providers_db.create_provider(id = "gateway", **payload.model_dump(exclude = {"encrypted_api_key"}))
    providers_db.update_provider("gateway", display_name = "Renamed")
    providers_db.reset_schema_state_for_tests()
    assert providers_db.get_provider("gateway")["reasoning_config"] == _config(style)
    assert providers_db.list_providers()[0]["reasoning_config"] == _config(style)
    update = ProviderUpdate(reasoning_config = _config(style, False))
    providers_db.update_provider("gateway", **update.model_dump(exclude_unset = True))
    assert providers_db.get_provider("gateway")["reasoning_config"] == _config(style, False)
    clear = ProviderUpdate(reasoning_config = None)
    providers_db.update_provider("gateway", **clear.model_dump(exclude_unset = True))
    assert providers_db.get_provider("gateway")["reasoning_config"] is None


def test_legacy_schema_has_no_reasoning_config(db):
    with sqlite3.connect(db) as conn:
        conn.execute("""CREATE TABLE llm_providers (id TEXT PRIMARY KEY, provider_type TEXT,
                        display_name TEXT, base_url TEXT, is_enabled INTEGER,
                        created_at TEXT, updated_at TEXT)""")
        conn.execute(
            "INSERT INTO llm_providers VALUES ('legacy', 'custom', 'Gateway', 'https://example.com', 1, '', '')"
        )
    assert providers_db.get_provider("legacy")["reasoning_config"] is None


@pytest.mark.parametrize(
    "raw",
    [
        "not json",
        "[]",
        '{"enabled":"true","style":"thinking"}',
        '{"enabled":true,"style":"invalid"}',
        '{"enabled":true,"style":"thinking","extra":1}',
    ],
)
def test_malformed_storage_fails_closed(db, raw, monkeypatch, endpoint):
    _, base_url = endpoint
    providers_db.create_provider("gateway", "custom", "Gateway", base_url)
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE llm_providers SET reasoning_config_json = ?", (raw,))
    assert providers_db.get_provider("gateway")["reasoning_config"] is None
    assert providers_db.list_providers()[0]["reasoning_config"] is None
    assert (
        _route_capture(
            monkeypatch,
            endpoint,
            "malformed-storage-" + raw,
            provider_id = "gateway",
            provider_reasoning_config = _config(),
            enable_thinking = True,
            reasoning_effort = "high",
        )
        == {}
    )


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.recorded.append({"path": self.path, "body": body})
        self.server.auths.append(self.headers.get("Authorization"))
        if self.path.endswith("/responses"):
            sse = (
                b'data: {"type":"response.output_text.delta","delta":"ok"}\n\n'
                b'data: {"type":"response.completed","response":{"id":"resp_test"}}\n\n'
            )
        else:
            sse = b'data: {"choices":[{"index":0,"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n'
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(sse)))
        self.end_headers()
        self.wfile.write(sse)


@pytest.fixture()
def endpoint():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    server.recorded = []
    server.auths = []
    thread = threading.Thread(target = server.serve_forever, daemon = True)
    thread.start()
    yield server, f"http://127.0.0.1:{server.server_address[1]}/v1"
    server.shutdown()
    server.server_close()
    thread.join(timeout = 10)


def _save_evidence(name, recorded):
    root = os.environ.get("UNSLOTH_REASONING_EVIDENCE_DIR")
    if root:
        path = Path(root)
        path.mkdir(parents = True, exist_ok = True)
        filename = re.sub(r"[^a-zA-Z0-9_.-]", "_", name)
        (path / f"{filename}.json").write_text(json.dumps(recorded, indent = 2) + "\n")


def _capture(monkeypatch, endpoint, config, name, **controls):
    server, base_url = endpoint

    async def run():
        async with httpx.AsyncClient(trust_env = False) as transport:
            monkeypatch.setattr(ep, "_http_client", transport)
            client = ExternalProviderClient("custom", base_url, "", reasoning_config = config)
            output = [
                line
                async for line in client.stream_chat_completion(
                    messages = [{"role": "user", "content": "hi"}], model = "test-model", **controls
                )
            ]
            assert "ok" in "".join(output)

    asyncio.run(run())
    assert server.recorded[-1]["path"] == "/v1/chat/completions"
    _save_evidence(name, server.recorded)
    return {
        key: value for key, value in server.recorded[-1]["body"].items() if key in REASONING_KEYS
    }


@pytest.mark.parametrize(
    "style,on,off",
    [
        ("reasoning_effort", {"reasoning_effort": "medium"}, {"reasoning_effort": "none"}),
        ("reasoning", {"reasoning": {"effort": "medium"}}, {"reasoning": {"enabled": False}}),
        ("thinking", {"thinking": {"type": "enabled"}}, {"thinking": {"type": "disabled"}}),
        (
            "chat_template_kwargs.enable_thinking",
            {"chat_template_kwargs": {"enable_thinking": True}},
            {"chat_template_kwargs": {"enable_thinking": False}},
        ),
    ],
)
def test_all_styles_real_http_on_off_and_default(monkeypatch, endpoint, style, on, off):
    assert _capture(monkeypatch, endpoint, _config(style), style + "-default") == on
    assert (
        _capture(monkeypatch, endpoint, _config(style), style + "-on", enable_thinking = True) == on
    )
    assert (
        _capture(monkeypatch, endpoint, _config(style), style + "-off", enable_thinking = False)
        == off
    )
    assert (
        _capture(monkeypatch, endpoint, _config(style), style + "-none", reasoning_effort = "none")
        == off
    )


@pytest.mark.parametrize("style", ["reasoning_effort", "reasoning"])
@pytest.mark.parametrize("effort", ["low", "medium", "high"])
def test_effort_real_http(monkeypatch, endpoint, style, effort):
    expected = (
        {"reasoning_effort": effort}
        if style == "reasoning_effort"
        else {"reasoning": {"effort": effort}}
    )
    assert (
        _capture(
            monkeypatch, endpoint, _config(style), style + "-" + effort, reasoning_effort = effort
        )
        == expected
    )


@pytest.mark.parametrize(
    "config",
    [
        None,
        _config(enabled = False),
        {"enabled": "true", "style": "thinking"},
        {"enabled": True, "style": "unknown"},
        {"enabled": True, "style": "thinking", "extra": 1},
    ],
)
def test_unconfigured_disabled_malformed_real_http(monkeypatch, endpoint, config):
    assert (
        _capture(
            monkeypatch,
            endpoint,
            config,
            "fail-closed-" + str(config),
            enable_thinking = True,
            reasoning_effort = "high",
        )
        == {}
    )


def test_disabled_contracts_keep_the_entire_legacy_http_body(monkeypatch, endpoint):
    server, _ = endpoint
    _capture(monkeypatch, endpoint, None, "legacy-full-body")
    legacy_body = server.recorded[-1]["body"]
    for config in [
        None,
        *[_config(style, False) for style in STYLES],
        {"enabled": "true", "style": "thinking"},
    ]:
        _capture(
            monkeypatch,
            endpoint,
            config,
            "unchanged-full-body-" + str(config),
            enable_thinking = True,
            reasoning_effort = "high",
        )
        assert server.recorded[-1]["body"] == legacy_body


@pytest.mark.parametrize("style", STYLES)
def test_unknown_effort_fails_closed(monkeypatch, endpoint, style):
    assert (
        _capture(
            monkeypatch, endpoint, _config(style), style + "-invalid-effort", reasoning_effort = "max"
        )
        == {}
    )


@pytest.mark.parametrize("style", STYLES)
def test_explicit_off_wins_over_an_unsupported_stale_effort(monkeypatch, endpoint, style):
    expected = {
        "reasoning_effort": {"reasoning_effort": "none"},
        "reasoning": {"reasoning": {"enabled": False}},
        "thinking": {"thinking": {"type": "disabled"}},
        "chat_template_kwargs.enable_thinking": {
            "chat_template_kwargs": {"enable_thinking": False}
        },
    }[style]
    assert (
        _capture(
            monkeypatch,
            endpoint,
            _config(style),
            style + "-off-stale-effort",
            enable_thinking = False,
            reasoning_effort = "max",
        )
        == expected
    )


@pytest.fixture()
def credential_db(db, tmp_path, monkeypatch):
    from auth import storage as auth_storage
    from storage import credential_secrets

    monkeypatch.setattr(auth_storage, "DB_PATH", tmp_path / "auth.db")
    monkeypatch.setattr(auth_storage, "_credential_encryption_key_cache", None)
    monkeypatch.setattr(credential_secrets, "studio_db_path", lambda: db)
    monkeypatch.setattr(
        credential_secrets,
        "get_or_create_credential_encryption_key",
        auth_storage.get_or_create_credential_encryption_key,
    )
    credential_secrets._schema_ready = set()
    yield db
    credential_secrets._schema_ready = set()


@pytest.fixture()
def provider_app(credential_db):
    from fastapi import FastAPI
    from routes import providers as route

    app = FastAPI()
    app.include_router(route.router, prefix = "/api/providers")
    app.dependency_overrides[route.get_current_subject] = lambda: "alice"
    app.dependency_overrides[route.get_current_credential] = lambda: ("alice", None)
    app.dependency_overrides[route.authenticated_via_api_key] = lambda: False
    return app


@pytest.mark.parametrize("style", STYLES)
def test_provider_api_create_edit_reload_and_clear(provider_app, style):
    async def run():
        async with httpx.AsyncClient(
            transport = httpx.ASGITransport(app = provider_app), base_url = "http://studio"
        ) as client:
            created = await client.post(
                "/api/providers/",
                json = {
                    "provider_type": "custom",
                    "display_name": "Gateway",
                    "base_url": "https://gateway.example/v1",
                    "reasoning_config": _config(style),
                },
            )
            assert created.status_code == 201, created.text
            assert created.json()["reasoning_config"] == _config(style)
            provider_id = created.json()["id"]
            edited = await client.put(
                f"/api/providers/{provider_id}", json = {"display_name": "Renamed"}
            )
            assert edited.status_code == 200, edited.text
            assert edited.json()["reasoning_config"] == _config(style)
            providers_db.reset_schema_state_for_tests()
            listed = await client.get("/api/providers/")
            assert listed.json()[0]["reasoning_config"] == _config(style)
            disabled = await client.put(
                f"/api/providers/{provider_id}", json = {"reasoning_config": _config(style, False)}
            )
            assert disabled.status_code == 200
            assert disabled.json()["reasoning_config"] == _config(style, False)
            cleared = await client.put(
                f"/api/providers/{provider_id}", json = {"reasoning_config": None}
            )
            assert cleared.status_code == 200
            assert cleared.json()["reasoning_config"] is None

    asyncio.run(run())


@pytest.mark.parametrize("api_type", ["responses", "systemone"])
def test_provider_api_rejects_enabled_incompatible_protocol(provider_app, api_type):
    async def run():
        async with httpx.AsyncClient(
            transport = httpx.ASGITransport(app = provider_app), base_url = "http://studio"
        ) as client:
            payload = {
                "provider_type": "custom",
                "display_name": "Gateway",
                "api_type": api_type,
                "reasoning_config": _config(),
            }
            refused = await client.post("/api/providers/", json = payload)
            assert refused.status_code == 400
            payload["api_type"] = "chat_completions"
            created = await client.post("/api/providers/", json = payload)
            provider_id = created.json()["id"]
            refused_edit = await client.put(
                f"/api/providers/{provider_id}", json = {"api_type": api_type}
            )
            assert refused_edit.status_code == 400
            # Both a clear and a disable can accompany a protocol change.
            accepted = await client.put(
                f"/api/providers/{provider_id}",
                json = {"api_type": api_type, "reasoning_config": _config(enabled = False)},
            )
            assert accepted.status_code == 200, accepted.text
            refused_enable = await client.put(
                f"/api/providers/{provider_id}", json = {"reasoning_config": _config()}
            )
            assert refused_enable.status_code == 400
            cleared = await client.put(
                f"/api/providers/{provider_id}", json = {"reasoning_config": None}
            )
            assert cleared.status_code == 200
            assert cleared.json()["api_type"] == api_type

    asyncio.run(run())


def test_provider_api_rejects_unknown_config_fields_and_non_custom(provider_app):
    async def run():
        async with httpx.AsyncClient(
            transport = httpx.ASGITransport(app = provider_app), base_url = "http://studio"
        ) as client:
            malformed = await client.post(
                "/api/providers/",
                json = {
                    "provider_type": "custom",
                    "display_name": "Gateway",
                    "reasoning_config": {"enabled": True, "style": "thinking", "budget": 123},
                },
            )
            assert malformed.status_code == 422
            non_custom = await client.post(
                "/api/providers/",
                json = {
                    "provider_type": "ollama",
                    "display_name": "Ollama",
                    "reasoning_config": _config(),
                },
            )
            assert non_custom.status_code == 400

    asyncio.run(run())


def _request():
    async def disconnected():
        return False

    return SimpleNamespace(
        headers = {}, state = SimpleNamespace(skip_api_monitor = True), is_disconnected = disconnected
    )


def _route_capture(monkeypatch, endpoint, name, **fields):
    from routes import inference

    server, base_url = endpoint
    payload = ChatCompletionRequest(
        **{
            "messages": [{"role": "user", "content": "hi"}],
            "model": "test-model",
            "stream": True,
            "provider_type": "custom",
            "provider_base_url": base_url,
            **fields,
        }
    )

    async def run():
        async with httpx.AsyncClient(trust_env = False) as transport:
            monkeypatch.setattr(ep, "_http_client", transport)
            response = await inference._proxy_to_external_provider(payload, _request())
            output = [line async for line in response.body_iterator]
            assert "ok" in "".join(output)

    asyncio.run(run())
    _save_evidence(name, server.recorded)
    assert server.recorded[-1]["path"] == "/v1/chat/completions"
    return {
        key: value for key, value in server.recorded[-1]["body"].items() if key in REASONING_KEYS
    }


@pytest.mark.parametrize(
    "style,expected",
    [
        ("reasoning_effort", {"reasoning_effort": "low"}),
        ("reasoning", {"reasoning": {"effort": "low"}}),
        ("thinking", {"thinking": {"type": "enabled"}}),
        (
            "chat_template_kwargs.enable_thinking",
            {"chat_template_kwargs": {"enable_thinking": True}},
        ),
    ],
)
def test_explicit_route_config_real_http(monkeypatch, endpoint, style, expected):
    assert (
        _route_capture(
            monkeypatch,
            endpoint,
            "explicit-route-" + style,
            provider_reasoning_config = _config(style),
            reasoning_effort = "low",
        )
        == expected
    )


@pytest.mark.parametrize("encrypted", [False, True])
@pytest.mark.parametrize(
    "config,expected",
    [
        (_config("thinking"), {"thinking": {"type": "enabled"}}),
        (_config(enabled = False), {}),
        (None, {}),
    ],
)
def test_saved_config_and_routing_authoritative_for_both_key_paths(
    monkeypatch, endpoint, credential_db, encrypted, config, expected
):
    from core.inference import key_exchange
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import padding, rsa
    from storage import credential_secrets
    import base64

    _, base_url = endpoint
    providers_db.create_provider("saved", "custom", "Saved", base_url, reasoning_config = config)
    credential_secrets.save_provider_api_key("saved", "local-test-key")
    fields = {
        "provider_id": "saved",
        "provider_reasoning_config": _config("reasoning_effort"),
        "reasoning_effort": "high",
        "provider_api_type": "responses",
        "provider_type": "openai",
        "provider_base_url": "https://unrelated.invalid/v1",
    }
    if encrypted:
        private_key = rsa.generate_private_key(public_exponent = 65537, key_size = 2048)
        monkeypatch.setattr(key_exchange, "_private_key", private_key)
        fields["encrypted_api_key"] = base64.b64encode(
            private_key.public_key().encrypt(
                b"browser-test-key",
                padding.OAEP(
                    mgf = padding.MGF1(hashes.SHA256()), algorithm = hashes.SHA256(), label = None
                ),
            )
        ).decode()
    # Both requested API type and requested reasoning dialect conflict with the saved row.
    assert (
        _route_capture(monkeypatch, endpoint, f"saved-route-{encrypted}-{config}", **fields)
        == expected
    )
    assert endpoint[0].auths[-1] == (
        "Bearer browser-test-key" if encrypted else "Bearer local-test-key"
    )


def test_saved_connection_switch_does_not_leak_config(monkeypatch, endpoint, credential_db):
    _, base_url = endpoint
    providers_db.create_provider(
        "enabled", "custom", "Enabled", base_url, reasoning_config = _config("thinking")
    )
    providers_db.create_provider("legacy", "custom", "Legacy", base_url)
    assert _route_capture(
        monkeypatch, endpoint, "connection-enabled", provider_id = "enabled", enable_thinking = True
    ) == {"thinking": {"type": "enabled"}}
    assert (
        _route_capture(
            monkeypatch, endpoint, "connection-legacy", provider_id = "legacy", enable_thinking = True
        )
        == {}
    )


def test_explicit_responses_config_rejected_before_http(monkeypatch, endpoint):
    from routes import inference

    server, base_url = endpoint
    payload = ChatCompletionRequest(
        messages = [],
        model = "test-model",
        provider_type = "custom",
        provider_base_url = base_url,
        provider_api_type = "responses",
        provider_reasoning_config = _config(),
    )
    with pytest.raises(HTTPException) as exc:
        asyncio.run(inference._proxy_to_external_provider(payload, _request()))
    assert exc.value.status_code == 400
    assert server.recorded == []


@pytest.mark.parametrize("api_type", ["responses", "systemone"])
def test_client_rejects_enabled_incompatible_protocol(api_type):
    with pytest.raises(ValueError, match = "Chat Completions"):
        ExternalProviderClient(
            "custom",
            "https://gateway.example/v1",
            "",
            api_type = api_type,
            reasoning_config = _config(),
        )


def test_saved_config_change_during_lookup_is_refused(monkeypatch):
    from routes import inference

    row = {
        "provider_type": "custom",
        "base_url": "https://gateway.example/v1",
        "api_type": "chat_completions",
        "is_enabled": True,
        "reasoning_config": _config(),
    }
    reads = []

    def get_provider(_id):
        reads.append(True)
        return {**row, "reasoning_config": _config("thinking") if len(reads) > 1 else _config()}

    monkeypatch.setattr(providers_db, "get_provider", get_provider)
    payload = ChatCompletionRequest(messages = [], provider_id = "saved", model = "test-model")
    with pytest.raises(HTTPException) as exc:
        asyncio.run(inference._proxy_to_external_provider(payload, _request()))
    assert exc.value.status_code == 409


@pytest.mark.parametrize("style", STYLES)
def test_saved_style_real_http_after_reload(monkeypatch, endpoint, credential_db, style):
    _, base_url = endpoint
    providers_db.create_provider(
        "saved", "custom", "Saved", base_url, reasoning_config = _config(style)
    )
    providers_db.reset_schema_state_for_tests()
    expected = {
        "reasoning_effort": {"reasoning_effort": "high"},
        "reasoning": {"reasoning": {"effort": "high"}},
        "thinking": {"thinking": {"type": "enabled"}},
        "chat_template_kwargs.enable_thinking": {"chat_template_kwargs": {"enable_thinking": True}},
    }[style]
    assert (
        _route_capture(
            monkeypatch,
            endpoint,
            "saved-style-" + style,
            provider_id = "saved",
            reasoning_effort = "high",
        )
        == expected
    )


def test_responses_retains_preexisting_reasoning_semantics(monkeypatch, endpoint):
    server, base_url = endpoint

    async def run():
        async with httpx.AsyncClient(trust_env = False) as transport:
            monkeypatch.setattr(ep, "_http_client", transport)
            for config in (None, _config("thinking", False)):
                client = ExternalProviderClient(
                    "custom", base_url, "", api_type = "responses", reasoning_config = config
                )
                output = [
                    line
                    async for line in client.stream_chat_completion(
                        messages = [{"role": "user", "content": "hi"}],
                        model = "test-model",
                        reasoning_effort = "high",
                    )
                ]
                assert "ok" in "".join(output)

    asyncio.run(run())
    assert len(server.recorded) == 2
    assert all(request["path"] == "/v1/responses" for request in server.recorded)
    assert server.recorded[0]["body"] == server.recorded[1]["body"]
    assert server.recorded[0]["body"]["reasoning"] == {"effort": "high", "summary": "auto"}
    _save_evidence("responses-unchanged", server.recorded)
