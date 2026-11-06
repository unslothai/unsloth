# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import json

import httpx

from core.inference import external_provider as ep_mod
from core.inference.external_provider import ExternalProviderClient

_STOP = b'event: message_stop\ndata: {"type": "message_stop"}\n\n'


def _sent_max_tokens(monkeypatch, models_response, max_tokens = None, model = "claude-haiku-4-5"):
    sent: list = []
    model_lookups: list = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.startswith("/v1/models/"):
            model_lookups.append(request.url.path)
            return models_response
        sent.append(json.loads(request.content)["max_tokens"])
        return httpx.Response(200, content = _STOP, headers = {"content-type": "text/event-stream"})

    monkeypatch.setattr(ep_mod, "_anthropic_output_caps", {}, raising = False)
    monkeypatch.setattr(
        ep_mod, "_http_client", httpx.AsyncClient(transport = httpx.MockTransport(handler))
    )

    async def run():
        client = ExternalProviderClient(
            provider_type = "anthropic",
            base_url = "https://api.anthropic.com/v1",
            api_key = "sk-ant-test",
        )
        for _ in range(2):
            async for _ in client.stream_chat_completion(
                messages = [{"role": "user", "content": "hi"}],
                model = model,
                max_tokens = max_tokens,
            ):
                pass
        await client.close()

    asyncio.new_event_loop().run_until_complete(run())
    return sent, model_lookups


def test_omitted_max_tokens_uses_the_models_output_limit(monkeypatch):
    sent, lookups = _sent_max_tokens(
        monkeypatch,
        httpx.Response(200, json = {"id": "claude-haiku-4-5", "max_tokens": 64000}),
    )
    assert sent == [64000, 64000]
    assert lookups == ["/v1/models/claude-haiku-4-5"]


def test_explicit_max_tokens_is_sent_unchanged(monkeypatch):
    sent, lookups = _sent_max_tokens(
        monkeypatch,
        httpx.Response(200, json = {"id": "claude-haiku-4-5", "max_tokens": 64000}),
        max_tokens = 300,
    )
    assert sent == [300, 300]
    assert lookups == []


def test_unknown_output_limit_keeps_the_old_default(monkeypatch):
    sent, _ = _sent_max_tokens(monkeypatch, httpx.Response(404, json = {"type": "error"}))
    assert sent == [1024, 1024]
