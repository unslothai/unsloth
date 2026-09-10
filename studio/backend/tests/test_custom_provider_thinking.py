# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Who gets ``chat_template_kwargs.enable_thinking``, and who must never get it.

``ExternalProviderClient.stream_chat_completion`` dispatches ``enable_thinking`` per provider
type and "llama_cpp" had no branch, so on a llama.cpp connection the toggle survived only as a
top-level body field llama-server does not read: the Deep Research planner, asked for
``enable_thinking=False``, spent its whole 4096-token budget reasoning and returned no plan.

"custom" is any user-supplied base_url, including a strict gateway that 400s on a nonstandard
key -- why the registry gives it a ``body_omit`` list and keeps it out of
``_CONTINUATION_FLAG_PROVIDERS``. Sending it there would fail every Deep Research call, which
carries ``enable_thinking=False`` whether or not the user touched the toggle; a llama.cpp or
vLLM server reaches the working branch by being added with its preset.
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


@pytest.mark.parametrize("provider_type", ["vllm", "llama_cpp"])
def test_template_appliers_gate_thinking_off_via_chat_template_kwargs(provider_type):
    body = _capture_body(provider_type, "some-local-model", enable_thinking = False)
    assert body["chat_template_kwargs"] == {"enable_thinking": False}


@pytest.mark.parametrize("provider_type", ["vllm", "llama_cpp"])
def test_template_appliers_gate_thinking_on_via_chat_template_kwargs(provider_type):
    body = _capture_body(provider_type, "some-local-model", enable_thinking = True)
    assert body["chat_template_kwargs"] == {"enable_thinking": True}


@pytest.mark.parametrize("provider_type", ["vllm", "llama_cpp"])
def test_no_toggle_sends_no_template_kwargs(provider_type):
    body = _capture_body(provider_type, "some-local-model")
    assert "chat_template_kwargs" not in body


@pytest.mark.parametrize("enable_thinking", [True, False])
def test_a_custom_base_url_is_never_given_template_kwargs(enable_thinking):
    # The Deep Research planner sends enable_thinking=False on every call, so a strict gateway
    # behind a "custom" connection would 400 on requests the user never asked to change.
    body = _capture_body("custom", "some-local-model", enable_thinking = enable_thinking)
    assert "chat_template_kwargs" not in body


def test_a_hosted_provider_is_not_given_template_kwargs():
    # A hosted API has its own mechanism (Kimi uses a top-level `thinking` field) and would
    # reject the unknown key.
    body = _capture_body("kimi", "kimi-k2.6", enable_thinking = False)
    assert "chat_template_kwargs" not in body
    assert body["thinking"] == {"type": "disabled"}
