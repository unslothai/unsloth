# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import asyncio
from types import SimpleNamespace

import pytest
from starlette.requests import ClientDisconnect

from models.inference import ChatCompletionRequest
from routes import inference


@pytest.mark.parametrize("tools", [None, []])
def test_no_image_or_tools_does_not_start_preparation_worker(monkeypatch, tools):
    def unexpected(*args, **kwargs):
        pytest.fail("ordinary requests must not enter image preparation")

    monkeypatch.setattr(inference.asyncio, "to_thread", unexpected)
    result = asyncio.run(
        inference._prepare_mcp_image_for_route(
            SimpleNamespace(mcp_image_attachment = None),
            "user",
            tools,
            None,
            False,
        )
    )
    assert result == (None, tools)


@pytest.mark.parametrize("provider", ["custom", "openai_codex"])
def test_provider_disconnect_before_body_closes_eager_image_run(monkeypatch, provider):
    from core.inference import external_provider, openai_codex_auth, openai_codex_client
    from core.inference.providers import get_provider_info
    from core.inference.openai_codex_auth import OPENAI_CODEX_API_BASE

    closed = []
    clients = []
    image_run = SimpleNamespace(close = lambda: closed.append("image"))

    class Client:
        def __init__(self, *args, **kwargs):
            clients.append(self)
            self.closed = False

        async def close(self):
            self.closed = True

    monkeypatch.setattr(external_provider, "ExternalProviderClient", Client)
    monkeypatch.setattr(inference, "ExternalProviderClient", Client)
    monkeypatch.setattr(openai_codex_client, "OpenAICodexClient", Client)

    async def prepare(payload, subject, tools, cancel_event, ui_events):
        return (image_run, tools) if tools is not None else (None, None)

    monkeypatch.setattr(inference, "_prepare_mcp_image_for_route", prepare)
    if provider == "openai_codex":
        model = get_provider_info(provider)["default_models"][0]
        monkeypatch.setattr(
            inference.providers_db,
            "get_provider",
            lambda pid: {
                "id": pid,
                "provider_type": provider,
                "base_url": OPENAI_CODEX_API_BASE,
                "display_name": "ChatGPT",
                "is_enabled": True,
                "models": [model],
            },
        )
        monkeypatch.setattr(
            openai_codex_auth, "load_oauth_bundle", lambda pid: {"account_id": "account"}
        )
        monkeypatch.setattr(
            openai_codex_client, "subscription_catalog_matches_account", lambda *args: True
        )
        monkeypatch.setattr(openai_codex_client, "subscription_catalog_known", lambda *args: False)
        monkeypatch.setattr(openai_codex_client, "subscription_catalog_stale", lambda *args: False)
        monkeypatch.setattr(openai_codex_client, "saved_models_proven_for", lambda *args: True)

        async def resolve(*args, **kwargs):
            return "token", "account"

        monkeypatch.setattr(openai_codex_auth, "resolve_access", resolve)
        provider_fields = {"provider_id": "codex", "external_model": model}
    else:
        provider_fields = {
            "provider_type": "custom",
            "provider_base_url": "https://example.test/v1",
            "external_model": "test",
        }

    async def disconnected():
        return False

    request = SimpleNamespace(
        headers = {}, state = SimpleNamespace(skip_api_monitor = True), is_disconnected = disconnected
    )
    payload = ChatCompletionRequest(
        messages = [{"role": "user", "content": "hello"}], stream = True, **provider_fields
    )

    async def run():
        response = await inference._proxy_to_external_provider(payload, request, "user")
        assert isinstance(response, inference._SameTaskStreamingResponse)

        async def send(message):
            assert message["type"] == "http.response.start"
            raise OSError("client disconnected")

        async def receive():
            return {"type": "http.disconnect"}

        with pytest.raises(ClientDisconnect):
            await response({}, receive, send)

    asyncio.run(run())
    assert closed == ["image"]
    if provider == "custom":
        assert clients and all(client.closed for client in clients)
    else:
        assert not clients  # Codex constructs its client only when the body starts.
