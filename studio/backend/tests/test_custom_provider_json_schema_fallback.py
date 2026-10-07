# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""JSON mode against an OpenAI-compatible server that only takes ``json_schema``.

LM Studio answers ``{"type": "json_object"}`` with a 400 ("'response_format.type' must be
'json_schema' or 'text'"), so every Deep Research run on a "custom" connection to it failed while
planning. The client now re-sends once with an any-object schema, and only on that refusal: servers
that accept json_object keep getting it unchanged.
"""

from __future__ import annotations

import asyncio
import json

import httpx

from core.inference import external_provider as ep_mod
from core.inference.external_provider import ExternalProviderClient

_OK_SSE = 'data: {"choices":[{"index":0,"delta":{"content":"{}"}}]}\n\n' "data: [DONE]\n\n"
_LMSTUDIO_REFUSAL = {"error": "'response_format.type' must be 'json_schema' or 'text'"}


def _run(handler, response_format) -> list[str]:
    mock_client = httpx.AsyncClient(transport = httpx.MockTransport(handler))
    client = ExternalProviderClient(
        provider_type = "custom",
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
    # The retry is otherwise the same request.
    assert bodies[0]["messages"] == bodies[1]["messages"]
    assert bodies[0]["model"] == bodies[1]["model"]
    assert any('"content": "{}"' in line or '"content":"{}"' in line for line in lines)
    assert not any('"error"' in line for line in lines)


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
    # Guards against a loop: the retried request already carries json_schema.
    bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content.decode()))
        return httpx.Response(400, json = _LMSTUDIO_REFUSAL)

    _run(handler, {"type": "json_object"})
    assert len(bodies) == 2
