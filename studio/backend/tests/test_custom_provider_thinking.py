# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A "custom" provider must receive the thinking toggle too.

``ExternalProviderClient.stream_chat_completion`` dispatches ``enable_thinking``
per provider type -- Kimi, Mistral, vLLM and Ollama each have a branch -- and
"custom" had none, so the toggle survived only as a top-level body field. Neither
vLLM nor llama.cpp reads that, and both register as "custom" when added by
base_url without a preset, which ``_TEMPLATE_APPLYING_PROVIDERS`` already assumes
of it. Studio asks the Deep Research planner for ``enable_thinking=False``; with
the request ignored, the planner spent its whole 4096-token budget reasoning and
returned no plan.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from core.inference import external_provider as ep_mod
from core.inference.external_provider import ExternalProviderClient


def _capture_body(provider_type: str, model: str, **kwargs) -> dict:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode())
        sse = 'data: {"choices":[{"index":0,"delta":{"content":"ok"}}]}\n\n' "data: [DONE]\n\n"
        return httpx.Response(200, content = sse, headers = {"content-type": "text/event-stream"})

    mock_client = httpx.AsyncClient(transport = httpx.MockTransport(handler))
    client = ExternalProviderClient(
        provider_type = provider_type,
        base_url = "http://127.0.0.1:11434/v1",
        api_key = "",
    )

    async def run() -> None:
        try:
            async for _ in client.stream_chat_completion(
                messages = [{"role": "user", "content": "hi"}],
                model = model,
                **kwargs,
            ):
                pass
        finally:
            await mock_client.aclose()

    event_loop = asyncio.new_event_loop()
    previous_client = ep_mod._http_client
    ep_mod._http_client = mock_client
    try:
        event_loop.run_until_complete(run())
    finally:
        ep_mod._http_client = previous_client
        event_loop.close()
    return captured["body"]


def test_custom_provider_without_a_toggle_sends_no_template_kwargs():
    body = _capture_body("custom", "some-local-model")
    assert "chat_template_kwargs" not in body


@pytest.mark.parametrize("provider_type", ["vllm", "custom"])
def test_template_appliers_gate_thinking_off_via_chat_template_kwargs(provider_type):
    body = _capture_body(provider_type, "some-local-model", enable_thinking = False)
    assert body["chat_template_kwargs"] == {"enable_thinking": False}


@pytest.mark.parametrize("provider_type", ["vllm", "custom"])
def test_template_appliers_gate_thinking_on_via_chat_template_kwargs(provider_type):
    body = _capture_body(provider_type, "some-local-model", enable_thinking = True)
    assert body["chat_template_kwargs"] == {"enable_thinking": True}


def test_a_hosted_provider_is_not_given_template_kwargs():
    # Only the local template-appliers take this; a hosted API has its own mechanism
    # (Kimi uses a top-level `thinking` field) and would reject the unknown key.
    body = _capture_body("kimi", "kimi-k2.6", enable_thinking = False)
    assert "chat_template_kwargs" not in body
    assert body["thinking"] == {"type": "disabled"}
