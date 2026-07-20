# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit tests for the OrcaRouter provider registry entry and wire shape.

OrcaRouter is an OpenAI-compatible meta-router, so requests flow through the
generic /chat/completions passthrough. These tests pin the registry contract
(curated model list, Bearer auth, attribution headers) and the outbound
request shape without needing a live server or API key.
"""

import asyncio
import json

import httpx

from core.inference import external_provider as ep_mod
from core.inference.external_provider import ExternalProviderClient
from core.inference.providers import get_provider_info, list_available_providers


# ── Registry contract ───────────────────────────────────────────────


def test_orcarouter_registry_entry():
    info = get_provider_info("orcarouter")
    assert info is not None
    assert info["display_name"] == "OrcaRouter"
    assert info["base_url"] == "https://api.orcarouter.ai/v1"
    assert info["auth_header"] == "Authorization"
    assert info["auth_prefix"] == "Bearer "
    # Curated picker: /api/providers/models returns default_models verbatim.
    assert info["model_list_mode"] == "curated"
    assert "orcarouter/auto" in info["default_models"]
    # Attribution headers, same convention as OpenRouter.
    assert info["extra_headers"]["HTTP-Referer"] == "https://unsloth.ai"
    assert info["extra_headers"]["X-Title"] == "Unsloth Studio"


def test_orcarouter_listed_in_registry_endpoint_payload():
    entries = {p["provider_type"]: p for p in list_available_providers()}
    assert "orcarouter" in entries
    entry = entries["orcarouter"]
    assert entry["model_list_mode"] == "curated"
    assert entry["supports_streaming"] is True
    assert entry["supports_vision"] is True
    assert len(entry["default_models"]) > 0


# ── Outbound request shape (generic chat-completions passthrough) ───


def _drive(coro):
    return asyncio.run(coro)


async def _collect(agen):
    out = []
    async for line in agen:
        out.append(line)
    return out


def test_orcarouter_stream_uses_chat_completions_with_auth_and_attribution(monkeypatch):
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            content = b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n',
            headers = {"content-type": "text/event-stream"},
        )

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(ep_mod, "_http_client", httpx.AsyncClient(transport = transport))

    async def run():
        client = ExternalProviderClient(
            provider_type = "orcarouter",
            base_url = "https://api.orcarouter.ai/v1",
            api_key = "sk-orca-test",
        )
        lines = await _collect(
            client.stream_chat_completion(
                messages = [{"role": "user", "content": "ping"}],
                model = "orcarouter/auto",
                temperature = 0.7,
                top_p = 0.95,
                max_tokens = 64,
            )
        )
        await client.close()
        return lines

    lines = _drive(run())
    assert captured["url"] == "https://api.orcarouter.ai/v1/chat/completions"
    headers = {k.lower(): v for k, v in captured["headers"].items()}
    assert headers["authorization"] == "Bearer sk-orca-test"
    # Attribution headers from the registry entry must reach the wire.
    assert headers["http-referer"] == "https://unsloth.ai"
    assert headers["x-title"] == "Unsloth Studio"
    # Namespaced model id passes through verbatim (`orcarouter/` prefix kept —
    # the backend routes on the full id).
    assert captured["body"]["model"] == "orcarouter/auto"
    assert captured["body"]["stream"] is True
    assert any("ok" in line for line in lines)


# ── Sampling-param stripping for reasoning-class ids ────────────────


def _capture_orcarouter_body(monkeypatch, model: str) -> dict:
    """Drive one streamed request and return the outbound JSON body."""
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            content = b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n',
            headers = {"content-type": "text/event-stream"},
        )

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(ep_mod, "_http_client", httpx.AsyncClient(transport = transport))

    async def run():
        client = ExternalProviderClient(
            provider_type = "orcarouter",
            base_url = "https://api.orcarouter.ai/v1",
            api_key = "sk-orca-test",
        )
        await _collect(
            client.stream_chat_completion(
                messages = [{"role": "user", "content": "ping"}],
                model = model,
                temperature = 0.7,
                top_p = 0.95,
                presence_penalty = 0.0,
                max_tokens = 64,
            )
        )
        await client.close()

    _drive(run())
    return captured["body"]


def test_orcarouter_strips_sampling_for_reasoning_ids(monkeypatch):
    # Reasoning-class upstreams (and `auto`, which can route to one) 400 on a
    # non-default temperature/top_p/presence_penalty, so those must not reach
    # the wire.
    for model in (
        "orcarouter/auto",
        "openai/gpt-5.5",
        "anthropic/claude-opus-4.8",
        "deepseek/deepseek-v4-pro",
    ):
        body = _capture_orcarouter_body(monkeypatch, model)
        assert body["model"] == model
        assert "temperature" not in body, model
        assert "top_p" not in body, model
        assert "presence_penalty" not in body, model


def test_orcarouter_keeps_sampling_for_non_reasoning_ids(monkeypatch):
    # Non-reasoning upstreams accept the sampling knobs; they must pass through
    # so the UI controls stay functional.
    for model in ("google/gemini-3.5-flash", "grok/grok-4.3", "qwen/qwen3.7-max"):
        body = _capture_orcarouter_body(monkeypatch, model)
        assert body["model"] == model
        assert body["temperature"] == 0.7, model
        assert body["top_p"] == 0.95, model
        assert "presence_penalty" in body, model
