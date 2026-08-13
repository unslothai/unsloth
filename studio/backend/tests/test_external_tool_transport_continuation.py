# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""``continue_final_message`` must describe the turn actually being sent.

vLLM and llama.cpp forward the flag to the HF chat template, which appends a
sentinel to whatever the last message is and truncates the rendered prompt
there. With a trailing tool result that removes the assistant generation prompt
and asks the model to continue the tool output, so the flag has to drop as soon
as the loop stops ending its conversation on an assistant turn.
"""

import asyncio
import json
import threading

import httpx
import pytest

from core.inference import external_provider as ep_mod
from core.inference.external_provider import ExternalProviderClient
from core.inference.external_tool_transport import OAICompatTransport


def _drive(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def _transport(
    monkeypatch,
    captured: dict,
    provider_type = "vllm",
):
    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            content = b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n',
            headers = {"content-type": "text/event-stream"},
        )

    monkeypatch.setattr(
        ep_mod, "_http_client", httpx.AsyncClient(transport = httpx.MockTransport(handler))
    )
    client = ExternalProviderClient(
        provider_type = provider_type,
        base_url = "http://self-hosted.example/v1",
        api_key = "",
    )
    return client, OAICompatTransport(client, model = "local-model", continue_final_message = True)


_ASSISTANT_TAIL = [
    {"role": "user", "content": "search the docs"},
    {"role": "assistant", "content": "Sure, let me"},
]
_TOOL_TAIL = _ASSISTANT_TAIL + [
    {
        "role": "assistant",
        "content": "Sure, let me",
        "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "web_search", "arguments": "{}"}}
        ],
    },
    {"role": "tool", "tool_call_id": "c1", "content": "result text"},
]
_USER_TAIL = _TOOL_TAIL + [{"role": "user", "content": "That tool could not run."}]


@pytest.mark.parametrize(
    "messages, expected",
    [(_ASSISTANT_TAIL, True), (_TOOL_TAIL, False), (_USER_TAIL, False)],
)
def test_continuation_flags_track_the_trailing_role(monkeypatch, messages, expected):
    captured: dict = {}
    client, transport = _transport(monkeypatch, captured)

    async def run():
        async for _ in transport.stream(
            messages = messages,
            tools = None,
            tool_choice = "auto",
            cancel_event = threading.Event(),
        ):
            pass
        await client.close()

    _drive(run())
    assert captured["body"].get("continue_final_message", False) is expected
    assert ("add_generation_prompt" in captured["body"]) is expected
