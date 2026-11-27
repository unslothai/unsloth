# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import json

import httpx
import pytest

from core.inference import external_provider as ep_mod
from core.inference.external_provider import ExternalProviderClient


def _sent_body(monkeypatch, model, temperature, **kwargs) -> dict:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            content = b'event: message_stop\ndata: {"type": "message_stop"}\n\n',
            headers = {"content-type": "text/event-stream"},
        )

    monkeypatch.setattr(
        ep_mod, "_http_client", httpx.AsyncClient(transport = httpx.MockTransport(handler))
    )

    async def run():
        client = ExternalProviderClient(
            provider_type = "anthropic",
            base_url = "https://api.anthropic.com/v1",
            api_key = "sk-ant-test",
        )
        async for _ in client.stream_chat_completion(
            messages = [{"role": "user", "content": "hi"}],
            model = model,
            temperature = temperature,
            top_p = None,
            max_tokens = 32,
            **kwargs,
        ):
            pass
        await client.close()

    asyncio.new_event_loop().run_until_complete(run())
    return captured["body"]


@pytest.mark.parametrize("model", ["claude-haiku-4-5", "claude-sonnet-4-6"])
def test_no_temperature_is_left_out_instead_of_sent_as_null(monkeypatch, model):
    body = _sent_body(monkeypatch, model, None, enable_thinking = False)
    assert "temperature" not in body
    assert "top_p" not in body


@pytest.mark.parametrize("model", ["claude-haiku-4-5", "claude-sonnet-4-6"])
def test_a_set_temperature_is_still_sent(monkeypatch, model):
    body = _sent_body(monkeypatch, model, 0.3, enable_thinking = False)
    assert body["temperature"] == 0.3


@pytest.mark.parametrize("temperature", [None, 0.3])
def test_thinking_still_sends_temperature_one(monkeypatch, temperature):
    body = _sent_body(monkeypatch, "claude-sonnet-4-6", temperature, reasoning_effort = "medium")
    assert body["temperature"] == 1
    assert "thinking" in body
