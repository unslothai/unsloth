# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit tests for MiniMax provider routing and request metadata."""

import asyncio
import json

import httpx
import pytest

from core.inference import external_provider as ep_mod
from core.inference.external_provider import ExternalProviderClient
from core.inference.providers import PROVIDER_REGISTRY


def _drive(coro):
    return asyncio.run(coro)


async def _collect(agen):
    lines = []
    async for line in agen:
        lines.append(line)
    return lines


def _response_for_request(request: httpx.Request) -> httpx.Response:
    if request.url.path.endswith("/messages"):
        content = b'event: message_stop\ndata: {"type":"message_stop"}\n\n'
    else:
        content = b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n'
    return httpx.Response(
        200,
        content = content,
        headers = {"content-type": "text/event-stream"},
    )


@pytest.mark.parametrize(
    ("base_url", "expected_url"),
    [
        (
            "https://api.minimax.io/v1",
            "https://api.minimax.io/v1/chat/completions",
        ),
        (
            "https://api.minimaxi.com/v1",
            "https://api.minimaxi.com/v1/chat/completions",
        ),
        (
            "https://api.minimax.io/anthropic",
            "https://api.minimax.io/anthropic/v1/messages",
        ),
        (
            "https://api.minimaxi.com/anthropic",
            "https://api.minimaxi.com/anthropic/v1/messages",
        ),
    ],
)
def test_minimax_endpoint_matrix_routes_to_the_selected_protocol(
    monkeypatch, base_url: str, expected_url: str
):
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = request.headers
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return _response_for_request(request)

    monkeypatch.setattr(
        ep_mod,
        "_http_client",
        httpx.AsyncClient(transport = httpx.MockTransport(handler)),
    )

    async def run():
        client = ExternalProviderClient(
            provider_type = "minimax",
            base_url = base_url,
            api_key = "test-key",
        )
        await _collect(
            client.stream_chat_completion(
                messages = [{"role": "user", "content": "hello"}],
                model = "MiniMax-M3",
                enable_thinking = True,
            )
        )
        await client.close()

    _drive(run())

    assert captured["url"] == expected_url
    assert captured["headers"]["authorization"] == "Bearer test-key"
    if base_url.endswith("/anthropic"):
        assert captured["headers"]["anthropic-version"] == "2023-06-01"
    else:
        assert "anthropic-version" not in captured["headers"]
    assert captured["body"]["thinking"] == {"type": "adaptive"}
    assert "presence_penalty" not in captured["body"]


@pytest.mark.parametrize(
    ("base_url", "expected_url"),
    [
        ("https://api.minimax.io/v1", "https://api.minimax.io/v1/models"),
        ("https://api.minimaxi.com/v1", "https://api.minimaxi.com/v1/models"),
        (
            "https://api.minimax.io/anthropic",
            "https://api.minimax.io/anthropic/v1/models",
        ),
        (
            "https://api.minimaxi.com/anthropic",
            "https://api.minimaxi.com/anthropic/v1/models",
        ),
    ],
)
def test_minimax_endpoint_matrix_routes_model_discovery(
    monkeypatch, base_url: str, expected_url: str
):
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = request.headers
        return httpx.Response(200, json = {"data": [{"id": "MiniMax-M3"}]})

    monkeypatch.setattr(
        ep_mod,
        "_http_client",
        httpx.AsyncClient(transport = httpx.MockTransport(handler)),
    )

    async def run():
        client = ExternalProviderClient(
            provider_type = "minimax",
            base_url = base_url,
            api_key = "test-key",
        )
        models = await client.list_models()
        await client.close()
        return models

    assert _drive(run()) == [{"id": "MiniMax-M3"}]
    assert captured["url"] == expected_url
    if base_url.endswith("/anthropic"):
        assert captured["headers"]["x-api-key"] == "test-key"
        assert "authorization" not in captured["headers"]
    else:
        assert captured["headers"]["authorization"] == "Bearer test-key"
        assert "x-api-key" not in captured["headers"]


@pytest.mark.parametrize(
    "base_url",
    [
        "https://api.minimax.io/v1",
        "https://api.minimax.io/anthropic",
    ],
)
def test_minimax_m3_thinking_can_be_disabled(monkeypatch, base_url: str):
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return _response_for_request(request)

    monkeypatch.setattr(
        ep_mod,
        "_http_client",
        httpx.AsyncClient(transport = httpx.MockTransport(handler)),
    )

    async def run():
        client = ExternalProviderClient(
            provider_type = "minimax",
            base_url = base_url,
            api_key = "test-key",
        )
        await _collect(
            client.stream_chat_completion(
                messages = [{"role": "user", "content": "hello"}],
                model = "MiniMax-M3",
                enable_thinking = False,
            )
        )
        await client.close()

    _drive(run())

    assert captured["body"]["thinking"] == {"type": "disabled"}


def test_minimax_m27_keeps_always_on_thinking_implicit(monkeypatch):
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return _response_for_request(request)

    monkeypatch.setattr(
        ep_mod,
        "_http_client",
        httpx.AsyncClient(transport = httpx.MockTransport(handler)),
    )

    async def run():
        client = ExternalProviderClient(
            provider_type = "minimax",
            base_url = "https://api.minimax.io/v1",
            api_key = "test-key",
        )
        await _collect(
            client.stream_chat_completion(
                messages = [{"role": "user", "content": "hello"}],
                model = "MiniMax-M2.7",
                enable_thinking = False,
            )
        )
        await client.close()

    _drive(run())

    assert "thinking" not in captured["body"]


@pytest.mark.parametrize(
    ("model", "expects_explicit_cache"),
    [
        ("MiniMax-M3", False),
        ("MiniMax-M2.7", True),
    ],
)
def test_minimax_anthropic_prompt_caching_matches_model_support(
    monkeypatch, model: str, expects_explicit_cache: bool
):
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return _response_for_request(request)

    monkeypatch.setattr(
        ep_mod,
        "_http_client",
        httpx.AsyncClient(transport = httpx.MockTransport(handler)),
    )

    async def run():
        client = ExternalProviderClient(
            provider_type = "minimax",
            base_url = "https://api.minimax.io/anthropic",
            api_key = "test-key",
        )
        await _collect(
            client.stream_chat_completion(
                messages = [{"role": "user", "content": "hello"}],
                model = model,
            )
        )
        await client.close()

    _drive(run())

    has_explicit_cache = "cache_control" in json.dumps(captured["body"])
    assert has_explicit_cache is expects_explicit_cache


def test_minimax_registry_uses_current_models_and_request_capabilities():
    entry = PROVIDER_REGISTRY["minimax"]

    assert entry["default_models"][:2] == ["MiniMax-M3", "MiniMax-M2.7"]
    assert entry["supports_vision"] is True
    assert entry["supports_tool_calling"] is True
    assert entry["body_omit"] == ("presence_penalty",)
