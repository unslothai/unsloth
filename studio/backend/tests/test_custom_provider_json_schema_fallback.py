# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""JSON mode against an OpenAI-compatible server that only takes ``json_schema`` (LM Studio)."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from core.inference import external_provider as ep_mod
from core.inference.external_provider import ExternalProviderClient

_OK_SSE = 'data: {"choices":[{"index":0,"delta":{"content":"{}"}}]}\n\n' "data: [DONE]\n\n"
_LMSTUDIO_REFUSAL = {"error": "'response_format.type' must be 'json_schema' or 'text'"}


def _run(
    handler,
    response_format,
    *,
    provider_type = "custom",
    **kwargs,
) -> list[str]:
    mock_client = httpx.AsyncClient(transport = httpx.MockTransport(handler))
    client = ExternalProviderClient(
        provider_type = provider_type,
        base_url = "http://127.0.0.1:1234/v1",
        api_key = "",
    )
    lines: list[str] = []

    async def run() -> None:
        try:
            async for line in client.stream_chat_completion(
                messages = [{"role": "user", "content": "plan"}],
                model = "some-local-model",
                response_format = response_format,
                **kwargs,
            ):
                lines.append(line)
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
    return lines


def _schema_only_server(bodies: list[dict]):
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        bodies.append(body)
        if (body.get("response_format") or {}).get("type") == "json_object":
            return httpx.Response(400, json = _LMSTUDIO_REFUSAL)
        return httpx.Response(200, content = _OK_SSE, headers = {"content-type": "text/event-stream"})

    return handler


def test_json_object_refusal_is_retried_once_as_json_schema():
    bodies: list[dict] = []
    lines = _run(_schema_only_server(bodies), {"type": "json_object"})

    assert [b["response_format"]["type"] for b in bodies] == ["json_object", "json_schema"]
    assert bodies[1]["response_format"]["json_schema"]["schema"] == {"type": "object"}
    assert bodies[0]["messages"] == bodies[1]["messages"]
    assert bodies[0]["model"] == bodies[1]["model"]
    assert any('"content": "{}"' in line or '"content":"{}"' in line for line in lines)
    assert not any('"error"' in line for line in lines)


def test_schema_retry_keeps_a_supplied_json_object_schema():
    schema = {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]}
    bodies: list[dict] = []
    _run(_schema_only_server(bodies), {"type": "json_object", "schema": schema})

    assert [b["response_format"]["type"] for b in bodies] == ["json_object", "json_schema"]
    assert bodies[1]["response_format"]["json_schema"]["schema"] == schema


def test_servers_that_accept_json_object_are_sent_it_unchanged():
    bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content.decode()))
        return httpx.Response(200, content = _OK_SSE, headers = {"content-type": "text/event-stream"})

    _run(handler, {"type": "json_object"})
    assert [b["response_format"] for b in bodies] == [{"type": "json_object"}]


def test_unrelated_400s_are_not_retried():
    bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content.decode()))
        return httpx.Response(400, json = {"error": "context length exceeded"})

    lines = _run(handler, {"type": "json_object"})
    assert len(bodies) == 1
    assert any("error" in line for line in lines)


def test_a_json_schema_refusal_is_not_retried_again():
    # No loop: the retried request already carries json_schema.
    bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content.decode()))
        return httpx.Response(400, json = _LMSTUDIO_REFUSAL)

    _run(handler, {"type": "json_object"})
    assert len(bodies) == 2


def test_schema_retry_preserves_openrouter_session_and_request_options():
    bodies: list[dict] = []
    lines = _run(
        _schema_only_server(bodies),
        {"type": "json_object"},
        provider_type = "openrouter",
        thread_id = "research-thread",
        temperature = 0.2,
        top_p = None,
        max_tokens = 128,
        presence_penalty = 0.4,
        top_k = 30,
        min_p = 0.1,
        repetition_penalty = 1.1,
        reasoning_effort = "high",
        tools = [
            {"type": "function", "function": {"name": "search", "parameters": {"type": "object"}}}
        ],
        tool_choice = "none",
    )
    assert len(bodies) == 2
    assert bodies[0]["session_id"] == "research-thread"
    assert bodies[1] == {**bodies[0], "response_format": bodies[1]["response_format"]}
    assert not any('"error"' in line for line in lines)


@pytest.mark.parametrize("first_refusal", ["max_tokens", "response_format"])
@pytest.mark.parametrize("schema_accepted", [True, False])
def test_schema_and_token_limit_retries_preserve_each_others_changes(
    first_refusal, schema_accepted
):
    bodies: list[dict] = []
    token_refusal = {"error": "Unsupported parameter: 'max_tokens'. Use 'max_completion_tokens'."}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        bodies.append(body)
        refusals = {
            "max_tokens": token_refusal if "max_tokens" in body else None,
            "response_format": _LMSTUDIO_REFUSAL
            if body["response_format"]["type"] == "json_object"
            else None,
        }
        error = refusals[first_refusal] or next((r for r in refusals.values() if r), None)
        if error is not None or not schema_accepted:
            return httpx.Response(400, json = error or _LMSTUDIO_REFUSAL)
        return httpx.Response(200, content = _OK_SSE, headers = {"content-type": "text/event-stream"})

    lines = _run(handler, {"type": "json_object"}, max_tokens = 128)

    assert len(bodies) == 3
    assert bodies[-1]["max_completion_tokens"] == 128
    assert "max_tokens" not in bodies[-1]
    assert bodies[-1]["response_format"]["json_schema"]["schema"] == {"type": "object"}
    for previous, current in zip(bodies, bodies[1:]):
        if "max_completion_tokens" in previous:
            assert current["max_completion_tokens"] == 128
            assert "max_tokens" not in current
        if previous["response_format"]["type"] == "json_schema":
            assert current["response_format"] == previous["response_format"]
    assert any('"error"' in line for line in lines) is not schema_accepted
