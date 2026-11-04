# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Saved connection extra headers, including {{CHAT_SESSION_ID}} interpolation."""

import asyncio
import json
from http.server import BaseHTTPRequestHandler
from types import SimpleNamespace

import httpx
import pytest
from pydantic import ValidationError

from core.inference import external_provider as ep
from models.providers import (
    ProviderCreate,
    ProviderUpdate,
    normalize_provider_extra_headers,
    render_provider_extra_headers,
)
from storage import providers_db
from .test_external_provider_sampling_over_the_wire import _Server


HEADERS = {"X-LiteLLM-Session-Id": "unsloth-{{CHAT_SESSION_ID}}", "X-Title": "Unsloth"}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.recorded.append(
            {
                "path": self.path,
                "body": body,
                "session": self.headers.get("X-LiteLLM-Session-Id"),
                "title": self.headers.get("X-Title"),
                "authorization": self.headers.get("Authorization"),
            }
        )
        data = b'data: {"choices":[{"index":0,"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n'
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def test_reserved_headers_are_rejected():
    with pytest.raises(ValueError, match = "reserved"):
        normalize_provider_extra_headers({"Authorization": "Bearer secret"})
    with pytest.raises(ValidationError):
        ProviderCreate(
            provider_type = "custom",
            display_name = "Gateway",
            extra_headers = {"Authorization": "Bearer secret"},
        )


def test_header_values_must_be_sendable():
    for value in ("Ünsloth", "a\nb", "a\x00b"):
        with pytest.raises(ValueError, match = "printable ASCII"):
            normalize_provider_extra_headers({"X-Title": value})
    assert normalize_provider_extra_headers({"X-Title": "a\tb c"}) == {"X-Title": "a\tb c"}


def test_template_interpolation():
    rendered = render_provider_extra_headers(
        HEADERS, {"CHAT_SESSION_ID": "thread-1", "MODEL": "gpt"}
    )
    assert rendered["X-LiteLLM-Session-Id"] == "unsloth-thread-1"
    assert rendered["X-Title"] == "Unsloth"


def test_extra_headers_persist_and_reach_the_wire(tmp_path, monkeypatch):
    monkeypatch.setattr(providers_db, "studio_db_path", lambda: tmp_path / "studio.db")
    providers_db.reset_schema_state_for_tests()
    payload = ProviderCreate(
        provider_type = "custom",
        display_name = "LiteLLM",
        base_url = "https://gateway.example/v1",
        extra_headers = HEADERS,
    )
    providers_db.create_provider(id = "litellm", **payload.model_dump(exclude = {"encrypted_api_key"}))
    row = providers_db.get_provider("litellm")
    assert row["extra_headers"] == HEADERS
    assert providers_db.update_provider("litellm", extra_headers = {"X-Title": "Studio"})
    assert providers_db.get_provider("litellm")["extra_headers"] == {"X-Title": "Studio"}
    assert ProviderUpdate(extra_headers = None).extra_headers is None

    with _Server(Handler) as server:

        async def send():
            async with httpx.AsyncClient(trust_env = False) as transport:
                monkeypatch.setattr(ep, "_http_client", transport)
                client = ep.ExternalProviderClient(
                    "custom",
                    server.base_url,
                    "sk-test",
                    extra_headers = HEADERS,
                )
                output = client.stream_chat_completion(
                    messages = [{"role": "user", "content": "hi"}],
                    model = "proxy-model",
                    thread_id = "chat-42",
                )
                assert "ok" in "".join([line async for line in output])
                await client.close()

        asyncio.run(send())
        recorded = server.bodies[-1]
        assert recorded["session"] == "unsloth-chat-42"
        assert recorded["title"] == "Unsloth"
        assert recorded["authorization"] == "Bearer sk-test"


def _drive(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def test_explicit_key_sends_request_headers_without_saved_lookup(monkeypatch):
    from models.inference import ChatCompletionRequest
    from routes import inference as inf_mod

    class ClientReached(Exception):
        pass

    monkeypatch.setattr(
        inf_mod.providers_db,
        "get_provider",
        lambda _provider_id: (_ for _ in ()).throw(
            AssertionError("explicit keys must not bind saved metadata")
        ),
    )
    monkeypatch.setattr(
        inf_mod,
        "resolve_provider_api_key_or_400",
        lambda *_args, **_kwargs: "replacement-key",
    )

    def client(**kwargs):
        raise ClientReached(kwargs)

    monkeypatch.setattr(inf_mod, "ExternalProviderClient", client)
    payload = ChatCompletionRequest(
        messages = [{"role": "user", "content": "hello"}],
        provider_id = "provider-1",
        provider_type = "custom",
        provider_base_url = "https://new.example/v1",
        encrypted_api_key = "encrypted-replacement",
        extra_headers = HEADERS,
        external_model = "new-model",
    )
    request = SimpleNamespace(headers = {}, state = SimpleNamespace(skip_api_monitor = True))
    with pytest.raises(ClientReached) as reached:
        _drive(inf_mod._proxy_to_external_provider(payload, request))
    assert reached.value.args[0] == {
        "provider_type": "custom",
        "base_url": "https://new.example/v1",
        "api_key": "replacement-key",
        "api_type": "chat_completions",
        "extra_headers": HEADERS,
    }


def test_chat_request_rejects_reserved_extra_headers():
    from models.inference import ChatCompletionRequest
    with pytest.raises(ValidationError, match = "reserved"):
        ChatCompletionRequest(
            messages = [{"role": "user", "content": "hello"}],
            extra_headers = {"Authorization": "Bearer secret"},
        )


@pytest.mark.parametrize("may_use_saved", [True, False])
def test_saved_headers_follow_the_saved_key_gate(monkeypatch, may_use_saved):
    from models.inference import ChatCompletionRequest
    from routes import inference as inf_mod

    class ClientReached(Exception):
        pass

    row = {
        "id": "provider-1",
        "provider_type": "custom",
        "display_name": "Gateway",
        "base_url": "https://gw.example/v1",
        "api_type": "chat_completions",
        "is_enabled": True,
        "reasoning_config": None,
        "extra_headers": {"X-Gateway-Token": "saved-secret"},
    }
    monkeypatch.setattr(inf_mod.providers_db, "get_provider", lambda _pid: dict(row))
    monkeypatch.setattr(inf_mod, "_request_may_use_saved_provider_key", lambda _r: may_use_saved)
    monkeypatch.setattr(inf_mod, "resolve_provider_api_key_or_400", lambda *a, **k: None)

    def client(**kwargs):
        raise ClientReached(kwargs)

    monkeypatch.setattr(inf_mod, "ExternalProviderClient", client)
    payload = ChatCompletionRequest(
        messages = [{"role": "user", "content": "hello"}],
        provider_id = "provider-1",
        external_model = "m",
    )
    request = SimpleNamespace(headers = {}, state = SimpleNamespace(skip_api_monitor = True))
    with pytest.raises(ClientReached) as reached:
        _drive(inf_mod._proxy_to_external_provider(payload, request))
    sent = reached.value.args[0].get("extra_headers")
    assert sent == ({"X-Gateway-Token": "saved-secret"} if may_use_saved else None)


def test_saved_extra_headers_helper_honours_api_keys(monkeypatch):
    from routes import inference as inf_mod

    row = {"extra_headers": {"X-Gateway-Token": "saved-secret"}}
    monkeypatch.setattr(inf_mod, "_request_has_api_key", lambda _r: False)
    assert inf_mod._saved_extra_headers(row, object()) == row["extra_headers"]
    monkeypatch.setattr(inf_mod, "_request_has_api_key", lambda _r: True)
    assert inf_mod._saved_extra_headers(row, object()) == {}


def test_values_are_trimmed_and_substitutions_stay_sendable():
    assert normalize_provider_extra_headers({"X-Token": " secret "}) == {"X-Token": "secret"}
    rendered = render_provider_extra_headers(
        {"X-Session": "unsloth-{{CHAT_SESSION_ID}}", "X-Model": "{{MODEL}} "},
        {"CHAT_SESSION_ID": "会話-1\r\n", "MODEL": ""},
    )
    assert rendered == {"X-Session": "unsloth-%E4%BC%9A%E8%A9%B1-1%0D%0A", "X-Model": ""}
    httpx.Request("GET", "http://x", headers = rendered)


@pytest.mark.parametrize("allow_saved_key", [True, False])
def test_container_client_keeps_the_connection_scope(monkeypatch, allow_saved_key):
    from models.inference import OpenAIContainerRequest
    from routes import inference as inf_mod

    row = {
        "id": "openai-1",
        "provider_type": "openai",
        "display_name": "OpenAI",
        "base_url": "https://api.openai.com/v1",
        "is_enabled": True,
        "extra_headers": {"OpenAI-Project": "proj_123"},
    }
    monkeypatch.setattr(inf_mod.providers_db, "get_provider", lambda _pid: dict(row))
    monkeypatch.setattr(inf_mod, "resolve_provider_api_key_or_400", lambda *a, **k: "sk-test")
    captured = {}
    monkeypatch.setattr(inf_mod, "ExternalProviderClient", lambda **kw: captured.update(kw))
    inf_mod._resolve_openai_cloud_client(
        OpenAIContainerRequest(provider_id = "openai-1"), allow_saved_key = allow_saved_key
    )
    assert captured.get("extra_headers") == (
        {"OpenAI-Project": "proj_123"} if allow_saved_key else None
    )


def test_served_metadata_cache_identity_includes_headers():
    import hashlib

    plain = ep.ExternalProviderClient("custom", "https://gw.example/v1", "sk")
    assert plain._credential_digest() == hashlib.sha256(b"sk").hexdigest()
    a = ep.ExternalProviderClient(
        "custom", "https://gw.example/v1", "sk", extra_headers = {"X-Tenant": "a"}
    )
    b = ep.ExternalProviderClient(
        "custom", "https://gw.example/v1", "sk", extra_headers = {"X-Tenant": "b"}
    )
    assert len({plain._credential_digest(), a._credential_digest(), b._credential_digest()}) == 3
    routed = ep.ExternalProviderClient(
        "custom", "https://gw.example/v1", "sk", extra_headers = {"X-Route": "{{CHAT_SESSION_ID}}"}
    )
    routed.set_header_context("chat-1", "m")
    first = routed._credential_digest()
    routed.set_header_context("chat-2", "m")
    assert routed._credential_digest() != first


def test_extra_headers_replace_registry_headers_case_insensitively():
    client = ep.ExternalProviderClient(
        "openrouter", "https://openrouter.ai/api/v1", "sk", extra_headers = {"x-title": "My App"}
    )
    headers = client._auth_headers()
    assert [k for k in headers if k.lower() == "x-title"] == ["x-title"]
    assert headers["x-title"] == "My App"


def test_route_sets_header_context_before_any_upstream_call(monkeypatch):
    from models.inference import ChatCompletionRequest
    from routes import inference as inf_mod

    class ContextSet(Exception):
        pass

    class FakeClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def set_header_context(self, thread_id, model):
            raise ContextSet(thread_id, model)

    monkeypatch.setattr(inf_mod, "resolve_provider_api_key_or_400", lambda *a, **k: "key")
    monkeypatch.setattr(inf_mod, "ExternalProviderClient", FakeClient)
    payload = ChatCompletionRequest(
        messages = [{"role": "user", "content": "hello"}],
        provider_type = "custom",
        provider_base_url = "https://gw.example/v1",
        encrypted_api_key = "encrypted",
        extra_headers = HEADERS,
        external_model = "m",
        thread_id = "chat-7",
    )
    request = SimpleNamespace(headers = {}, state = SimpleNamespace(skip_api_monitor = True))
    with pytest.raises(ContextSet) as reached:
        _drive(inf_mod._proxy_to_external_provider(payload, request))
    assert reached.value.args == ("chat-7", "m")


def test_decision_key_refuses_headers_changed_mid_start(monkeypatch):
    from fastapi import HTTPException
    from routes import systemone

    row = {
        "provider_type": "custom",
        "base_url": "https://gw.example/v1",
        "api_type": "systemone",
        "is_enabled": True,
        "extra_headers": {"X-Gateway-Token": "new"},
    }
    monkeypatch.setattr(systemone.providers_db, "get_provider", lambda _pid: dict(row))
    monkeypatch.setattr(systemone, "resolve_provider_api_key_or_400", lambda *a, **k: "new-key")
    stale = {**row, "extra_headers": {"X-Gateway-Token": "old"}}
    with pytest.raises(HTTPException) as refused:
        _drive(systemone._connection_key("p", stale))
    assert refused.value.status_code == 409
    assert _drive(systemone._connection_key("p", dict(row))) == "new-key"
