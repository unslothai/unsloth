# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Ollama's OpenAI-compatible proxy must carry API thinking controls. #9649

``ExternalProviderClient.stream_chat_completion`` already maps thinking for
Kimi, Mistral, vLLM and OpenRouter. Ollama documents ``reasoning_effort``
values ``high`` / ``medium`` / ``low`` / ``none`` on ``/v1/chat/completions``,
but the outbound body omitted the field.
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


def test_ollama_request_without_controls_does_not_send_reasoning_effort():
    body = _capture_body("ollama", "thinkingcap-27b-bottlecap:latest")
    assert "reasoning_effort" not in body
    assert "thinking" not in body
    assert "chat_template_kwargs" not in body


@pytest.mark.parametrize("effort", ["none", "low", "medium", "high", "max"])
def test_ollama_forwards_reasoning_effort(effort):
    body = _capture_body(
        "ollama",
        "thinkingcap-27b-bottlecap:latest",
        reasoning_effort = effort,
    )
    assert body["reasoning_effort"] == effort


def test_ollama_thinking_off_maps_to_reasoning_effort_none():
    body = _capture_body(
        "ollama",
        "thinkingcap-27b-bottlecap:latest",
        enable_thinking = False,
    )
    assert body["reasoning_effort"] == "none"


def test_ollama_thinking_on_defaults_to_medium():
    body = _capture_body(
        "ollama",
        "thinkingcap-27b-bottlecap:latest",
        enable_thinking = True,
    )
    assert body["reasoning_effort"] == "medium"


def test_ollama_explicit_effort_wins_over_enable_thinking():
    body = _capture_body(
        "ollama",
        "thinkingcap-27b-bottlecap:latest",
        enable_thinking = True,
        reasoning_effort = "high",
    )
    assert body["reasoning_effort"] == "high"


@pytest.mark.parametrize(
    "incoming, expected",
    [("minimal", "low"), ("xhigh", "max")],
)
def test_ollama_maps_reasoning_effort_aliases(incoming, expected):
    body = _capture_body(
        "ollama",
        "thinkingcap-27b-bottlecap:latest",
        reasoning_effort = incoming,
    )
    assert body["reasoning_effort"] == expected
