# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Min P / Repetition Penalty / Top K must reach a self-hosted OpenAI-compatible server.

The panel offered all three and persisted them, but ``stream_chat_completion`` built only
``temperature`` / ``top_p`` / ``presence_penalty`` / ``max_tokens``, so the sliders moved
and nothing changed.

They stay opt-in: the schema defaults (20 / 0.01 / 1.0) are non-None for the local path, so
forwarding unconditionally would start sending sampling to providers that never got any.
"""

from __future__ import annotations

import ast
import asyncio
import json
import pathlib

import httpx
import pytest

from core.inference import external_provider as ep_mod
from core.inference.external_provider import ExternalProviderClient
from models.inference import ChatCompletionRequest


_ROUTE_SOURCE = pathlib.Path(__file__).resolve().parents[1] / "routes" / "inference.py"


def _capture_body(provider_type: str, **kwargs) -> dict:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode())
        sse = 'data: {"choices":[{"index":0,"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n'
        return httpx.Response(200, content = sse, headers = {"content-type": "text/event-stream"})

    mock_client = httpx.AsyncClient(transport = httpx.MockTransport(handler))
    client = ExternalProviderClient(
        provider_type = provider_type,
        base_url = "http://127.0.0.1:8000/v1",
        api_key = "",
    )

    async def run() -> None:
        try:
            async for _ in client.stream_chat_completion(
                messages = [{"role": "user", "content": "hi"}],
                model = "a-model",
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


@pytest.mark.parametrize("provider_type", ["vllm", "openrouter"])
def test_the_sliders_reach_the_server(provider_type):
    body = _capture_body(
        provider_type,
        top_k = 40,
        min_p = 0.07,
        repetition_penalty = 1.15,
    )
    assert body["top_k"] == 40
    assert body["min_p"] == 0.07
    assert body["repetition_penalty"] == 1.15


@pytest.mark.parametrize("provider_type", ["vllm", "openrouter", "custom", "llama_cpp"])
def test_a_request_that_set_none_of_them_sends_none(provider_type):
    body = _capture_body(provider_type)
    for field in ("top_k", "min_p", "repetition_penalty", "repeat_penalty"):
        assert field not in body, field
    assert body["temperature"] == 0.7
    assert body["top_p"] == 0.95
    assert body["presence_penalty"] == 0.0


def test_llama_server_gets_repeat_penalty_not_repetition_penalty():
    body = _capture_body("llama_cpp", min_p = 0.07, repetition_penalty = 1.15, top_k = 40)
    assert body["repeat_penalty"] == 1.15
    assert "repetition_penalty" not in body
    assert body["min_p"] == 0.07
    assert body["top_k"] == 40


def test_a_zero_value_is_forwarded_rather_than_read_as_unset():
    # 0 means "disabled" here, so a truthiness gate would restore the server's default.
    body = _capture_body("vllm", top_k = 0, min_p = 0.0)
    assert body["top_k"] == 0
    assert body["min_p"] == 0.0


def test_the_schema_defaults_are_not_none_so_the_route_cannot_test_for_none():
    bare = ChatCompletionRequest(messages = [{"role": "user", "content": "hi"}])
    assert bare.top_k == 20
    assert bare.min_p == 0.01
    assert bare.repetition_penalty == 1.0
    assert "min_p" not in bare.model_fields_set
    assert "repetition_penalty" not in bare.model_fields_set

    asked = ChatCompletionRequest(
        messages = [{"role": "user", "content": "hi"}],
        min_p = 0.0,
        repetition_penalty = 1.0,
    )
    # Both equal a default, so only model_fields_set can tell them from an omission.
    assert {"min_p", "repetition_penalty"} <= asked.model_fields_set


def test_the_proxy_hands_the_client_explicit_values_not_schema_defaults():
    tree = ast.parse(_ROUTE_SOURCE.read_text(encoding = "utf-8"))
    proxy = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "_proxy_to_external_provider"
    )
    kwargs = next(
        node
        for node in ast.walk(proxy)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "dict"
        and any(keyword.arg == "top_k" for keyword in node.keywords)
    )
    passed = {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in kwargs.keywords
        if keyword.arg in ("top_k", "min_p", "repetition_penalty")
    }
    assert passed == {
        "top_k": "_top_k_explicit",
        "min_p": "_min_p_explicit",
        "repetition_penalty": "_repetition_penalty_explicit",
    }


@pytest.mark.parametrize("provider_type", ["ollama", "custom"])
def test_the_registry_strips_the_three_for_providers_that_cannot_take_them(provider_type):
    # Ollama ignores them; a gateway behind a Custom base URL rejects the whole turn.
    body = _capture_body(
        provider_type,
        temperature = 0.31,
        top_k = 42,
        min_p = 0.07,
        repetition_penalty = 1.23,
        presence_penalty = 0.11,
    )
    assert "top_k" not in body
    assert "min_p" not in body
    assert "repetition_penalty" not in body
    assert body["temperature"] == 0.31
    assert body["presence_penalty"] == 0.11
