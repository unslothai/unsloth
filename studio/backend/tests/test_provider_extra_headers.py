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
