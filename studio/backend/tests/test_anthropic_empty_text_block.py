# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""An image-only user turn must not carry an empty text block to Anthropic.

The Messages API 400s with "messages: text content blocks must be
non-empty", so `{"type":"text","text":""}` has to be dropped before the
wire the way the Gemini translator already drops it.
"""

import asyncio
import json

import httpx

from core.inference import external_provider as ep_mod
from core.inference.external_provider import ExternalProviderClient


def _drive(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def _capture(monkeypatch, messages) -> dict:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            content = b"event: message_stop\n" b'data: {"type": "message_stop"}\n\n',
            headers = {"content-type": "text/event-stream"},
        )

    monkeypatch.setattr(
        ep_mod,
        "_http_client",
        httpx.AsyncClient(transport = httpx.MockTransport(handler)),
    )

    async def run():
        client = ExternalProviderClient(
            provider_type = "anthropic",
            base_url = "https://api.anthropic.com/v1",
            api_key = "sk-ant-test",
        )
        async for _ in client._stream_anthropic(
            messages = messages,
            model = "claude-opus-4-7",
            temperature = 0.7,
            top_p = 0.95,
            max_tokens = 32,
            enable_prompt_caching = False,
        ):
            pass
        await client.close()

    _drive(run())
    return captured


_IMAGE_DATA_URI = "data:image/png;base64,aGVsbG8="


def test_empty_text_block_dropped_from_image_only_turn(monkeypatch):
    captured = _capture(
        monkeypatch,
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": ""},
                    {"type": "image_url", "image_url": {"url": _IMAGE_DATA_URI}},
                ],
            }
        ],
    )
    parts = captured["body"]["messages"][0]["content"]
    assert not [p for p in parts if p.get("type") == "text"], parts
    assert parts == [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": "aGVsbG8=",
            },
        }
    ]


def test_captioned_image_keeps_its_text_block(monkeypatch):
    captured = _capture(
        monkeypatch,
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is this?"},
                    {"type": "image_url", "image_url": {"url": _IMAGE_DATA_URI}},
                ],
            }
        ],
    )
    parts = captured["body"]["messages"][0]["content"]
    assert parts[0] == {"type": "text", "text": "what is this?"}
    assert parts[1]["type"] == "image"


def test_text_only_empty_block_drops_the_whole_message(monkeypatch):
    # Nothing usable survives, and an empty content array 400s with
    # "at least one block is required".
    captured = _capture(
        monkeypatch,
        [
            {"role": "user", "content": [{"type": "text", "text": ""}]},
            {"role": "user", "content": "but THIS one is fine"},
        ],
    )
    assert captured["body"]["messages"] == [{"role": "user", "content": "but THIS one is fine"}]
