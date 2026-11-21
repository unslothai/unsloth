# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio

import httpx

from core.inference import external_provider as ep
from core.inference.providers import get_base_url, get_provider_info, list_available_providers


def _capture(
    monkeypatch,
    provider_type,
    model,
    thread_id = None,
) -> dict:
    captured: dict = {}

    def handler(request):
        captured["url"] = str(request.url)
        captured["headers"] = request.headers
        return httpx.Response(
            200, headers = {"content-type": "text/event-stream"}, content = "data: [DONE]\n\n"
        )

    async def run():
        async with httpx.AsyncClient(transport = httpx.MockTransport(handler)) as http:
            monkeypatch.setattr(ep, "_client", lambda: http)
            client = ep.ExternalProviderClient(
                provider_type, get_base_url(provider_type), "sk-test"
            )
            async for _ in client.stream_chat_completion(
                messages = [{"role": "user", "content": "hi"}], model = model, thread_id = thread_id
            ):
                pass

    asyncio.run(run())
    return captured


def test_opencode_providers_listed():
    types = {p["provider_type"] for p in list_available_providers()}
    assert {"opencode", "opencode_go"} <= types


def test_opencode_go_sends_session(monkeypatch):
    cap = _capture(monkeypatch, "opencode_go", "kimi-k3", thread_id = "thread-abc")
    assert cap["url"] == "https://opencode.ai/inference/go/v1/chat/completions"
    assert cap["headers"]["x-opencode-session"]
    assert cap["headers"]["user-agent"] == "unsloth-studio"
    assert cap["headers"]["authorization"] == "Bearer sk-test"


def test_opencode_zen_identifies_studio(monkeypatch):
    cap = _capture(monkeypatch, "opencode", "glm-5.3")
    assert cap["url"] == "https://opencode.ai/inference/v1/chat/completions"
    assert cap["headers"]["user-agent"] == "unsloth-studio"
    assert "x-opencode-session" not in cap["headers"]


def test_opencode_hides_free_tier_models():
    zen = get_provider_info("opencode")["model_id_denylist"]
    go = get_provider_info("opencode_go")["model_id_denylist"]
    zen_ids = ("big-pickle", "glm-5.3-flash-free", "jev-1.13", "glm-5.3")
    assert [m for m in zen_ids if not zen.search(m)] == ["glm-5.3"]
    go_ids = ("beam-free", "iceman-contributor-free", "kimi-k3")
    assert [m for m in go_ids if not go.search(m)] == ["kimi-k3"]


def test_opencode_mcp_images_follow_the_models_dev_catalog(monkeypatch):
    from core.inference.provider_model_capabilities import trim_models_dev_catalog
    from routes import providers as provider_routes
    from routes.inference import _external_takes_mcp_images

    models = {
        "kimi-k3": {"modalities": {"input": ["text", "image"]}},
        "glm-5.3": {"modalities": {"input": ["text"]}},
    }
    catalog = trim_models_dev_catalog(
        {"opencode": {"models": models}, "opencode-go": {"models": models}}
    )
    monkeypatch.setattr(
        provider_routes, "_model_catalog_cache", {"fetched_at": 0, "providers": catalog}
    )
    for provider_type in ("opencode", "opencode_go"):
        info = get_provider_info(provider_type)
        vision = info["supports_vision"]
        assert _external_takes_mcp_images(provider_type, vision, "kimi-k3", info) is True
        assert _external_takes_mcp_images(provider_type, vision, "glm-5.3", info) is False
        assert _external_takes_mcp_images(provider_type, vision, "unknown-model", info) is False
