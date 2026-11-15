# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import copy
import json

import httpx
import pytest
from fastapi import FastAPI, Request

from auth import authentication
from core.inference import external_provider as ep_mod
from core.inference.external_provider import ExternalProviderClient
from models.inference import ChatCompletionRequest
from routes import inference


def _capture(monkeypatch, messages, **options):
    captured = {}

    def handler(request):
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            content = b'event: message_stop\ndata: {"type":"message_stop"}\n\n',
            headers = {"content-type": "text/event-stream"},
        )

    async def run():
        async with httpx.AsyncClient(transport = httpx.MockTransport(handler)) as transport:
            monkeypatch.setattr(ep_mod, "_http_client", transport)
            client = ExternalProviderClient(
                provider_type = "anthropic",
                base_url = "https://api.anthropic.com/v1",
                api_key = "sk-ant-test",
            )
            try:
                async for _ in client.stream_chat_completion(
                    messages = messages,
                    model = "claude-sonnet-4-6",
                    temperature = 0.7,
                    top_p = 0.95,
                    max_tokens = 64,
                    **options,
                ):
                    pass
            finally:
                await client.close()

    asyncio.run(run())
    return captured


@pytest.mark.parametrize("caching", [None, True, False])
def test_all_system_messages_reach_anthropic_in_order(monkeypatch, caching):
    messages = [
        {"role": "system", "content": "Answer in French."},
        {"role": "system", "content": "Keep the answer short."},
        {"role": "user", "content": "Hello"},
    ]
    original = copy.deepcopy(messages)
    body = _capture(monkeypatch, messages, enable_prompt_caching = caching)
    expected = "Answer in French.\n\nKeep the answer short."
    assert body["system"] == (
        expected
        if caching is False
        else [{"type": "text", "text": expected, "cache_control": {"type": "ephemeral"}}]
    )
    assert [message["role"] for message in body["messages"]] == ["user"]
    assert messages[:2] == original[:2]


def test_multipart_system_text_keeps_inner_and_message_order(monkeypatch):
    body = _capture(
        monkeypatch,
        [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "First instruction."},
                    {"type": "text", "text": "Second instruction."},
                ],
            },
            {"role": "system", "content": "Third instruction."},
            {"role": "user", "content": "Hello"},
        ],
        enable_prompt_caching = False,
    )
    assert body["system"] == "First instruction.\nSecond instruction.\n\nThird instruction."


@pytest.mark.parametrize("empty", ["", []])
def test_later_empty_system_message_keeps_earlier_instruction(monkeypatch, empty):
    body = _capture(
        monkeypatch,
        [
            {"role": "system", "content": "Keep this instruction."},
            {"role": "system", "content": empty},
            {"role": "user", "content": "Hello"},
        ],
        enable_prompt_caching = False,
    )
    assert body["system"] == "Keep this instruction."


def test_many_system_messages_share_one_cache_block_and_keep_ttl(monkeypatch):
    body = _capture(
        monkeypatch,
        [
            {"role": "system", "content": text}
            for text in ["One.", "Two.", "Three.", "Four.", "Five."]
        ]
        + [{"role": "user", "content": "Hello"}],
        enable_prompt_caching = True,
        prompt_cache_ttl = "1h",
    )
    assert body["system"] == [
        {
            "type": "text",
            "text": "One.\n\nTwo.\n\nThree.\n\nFour.\n\nFive.",
            "cache_control": {"type": "ephemeral", "ttl": "1h"},
        }
    ]
    assert body["messages"] == [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Hello",
                    "cache_control": {"type": "ephemeral", "ttl": "1h"},
                }
            ],
        }
    ]


@pytest.mark.parametrize("system", [None, "Keep this instruction."])
def test_zero_or_one_system_message_retains_wire_shape(monkeypatch, system):
    messages = [{"role": "user", "content": "Hello"}]
    if system is not None:
        messages.insert(0, {"role": "system", "content": system})
    body = _capture(monkeypatch, messages, enable_prompt_caching = False)
    assert body.get("system") == system
    assert body["messages"] == [{"role": "user", "content": "Hello"}]


def test_chat_completions_preserves_system_and_developer_instructions(monkeypatch):
    captured = {}

    def handler(request):
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            content = b'event: message_stop\ndata: {"type":"message_stop"}\n\n',
            headers = {"content-type": "text/event-stream"},
        )

    monkeypatch.setattr(
        authentication, "request_admitted_without_credential", lambda request: False
    )
    monkeypatch.setattr(
        inference, "resolve_provider_api_key_or_400", lambda *args, **kwargs: "test"
    )
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def endpoint(payload: ChatCompletionRequest, request: Request):
        request.state.skip_api_monitor = True
        return await inference.produce_openai_chat_completions(
            payload, request, "test-subject", cancel_on_disconnect = False
        )

    async def run():
        async with httpx.AsyncClient(transport = httpx.MockTransport(handler)) as transport:
            monkeypatch.setattr(ep_mod, "_http_client", transport)
            async with httpx.AsyncClient(
                transport = httpx.ASGITransport(app = app), base_url = "http://test"
            ) as client:
                response = await client.post(
                    "/v1/chat/completions",
                    json = {
                        "provider_type": "anthropic",
                        "provider_base_url": "https://api.anthropic.com/v1",
                        "model": "claude-sonnet-4-6",
                        "stream": True,
                        "max_tokens": 64,
                        "messages": [
                            {"role": "system", "content": "Answer in French."},
                            {"role": "developer", "content": "Keep the answer short."},
                            {"role": "user", "content": "Hello"},
                        ],
                    },
                )
                assert response.status_code == 200, response.text

    asyncio.run(run())
    [system] = captured["system"]
    assert system["type"] == "text"
    assert system["cache_control"] == {"type": "ephemeral"}
    assert system["text"].endswith("Answer in French.\n\nKeep the answer short.")
