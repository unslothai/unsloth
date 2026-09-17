# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import json

import httpx

from core.inference import external_provider as ep_mod
from core.inference.external_provider import ExternalProviderClient


def _drive(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def _capture(
    monkeypatch,
    messages,
    caching = False,
) -> dict:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            content = b'event: message_stop\ndata: {"type": "message_stop"}\n\n',
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
            enable_prompt_caching = caching,
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
    # Nothing usable survives, and an empty content array 400s too.
    captured = _capture(
        monkeypatch,
        [
            {"role": "user", "content": [{"type": "text", "text": ""}]},
            {"role": "user", "content": "but THIS one is fine"},
        ],
    )
    assert captured["body"]["messages"] == [{"role": "user", "content": "but THIS one is fine"}]


def test_whitespace_only_caption_dropped_from_image_turn(monkeypatch):
    captured = _capture(
        monkeypatch,
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "\n"},
                    {"type": "image_url", "image_url": {"url": _IMAGE_DATA_URI}},
                ],
            }
        ],
    )
    parts = captured["body"]["messages"][0]["content"]
    assert [p["type"] for p in parts] == ["image"], parts


def test_caption_keeps_its_own_surrounding_whitespace(monkeypatch):
    # Only the DECISION uses strip(); a real caption goes out verbatim.
    captured = _capture(
        monkeypatch,
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "  what is this?  "},
                    {"type": "image_url", "image_url": {"url": _IMAGE_DATA_URI}},
                ],
            }
        ],
    )
    parts = captured["body"]["messages"][0]["content"]
    assert parts[0] == {"type": "text", "text": "  what is this?  "}


def test_empty_string_content_message_dropped(monkeypatch):
    captured = _capture(
        monkeypatch,
        [
            {"role": "user", "content": ""},
            {"role": "user", "content": "   "},
            {"role": "user", "content": "but THIS one is fine"},
        ],
    )
    assert captured["body"]["messages"] == [{"role": "user", "content": "but THIS one is fine"}]


def test_dropping_the_last_message_moves_the_cache_breakpoint(monkeypatch):
    # cache_control on an empty text block is rejected separately.
    captured = _capture(
        monkeypatch,
        [
            {"role": "user", "content": "keep me"},
            {"role": "user", "content": [{"type": "text", "text": ""}]},
        ],
        caching = True,
    )
    messages = captured["body"]["messages"]
    assert len(messages) == 1
    last_block = messages[-1]["content"][-1]
    assert last_block["type"] == "text"
    assert last_block["text"] == "keep me"
    assert last_block.get("cache_control") is not None


def test_cached_image_only_turn_marks_the_image(monkeypatch):
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
        caching = True,
    )
    parts = captured["body"]["messages"][0]["content"]
    assert [p["type"] for p in parts] == ["image"], parts
    assert parts[-1].get("cache_control") is not None


def test_whitespace_only_assistant_text_beside_a_tool_call(monkeypatch):
    # The whitespace block 400s and takes the tool_use down with it.
    captured = _capture(
        monkeypatch,
        [
            {"role": "user", "content": "q"},
            {
                "role": "assistant",
                "content": "  \n ",
                "tool_calls": [
                    {
                        "id": "toolu_1",
                        "type": "function",
                        "function": {"name": "f", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "toolu_1", "content": "42"},
        ],
    )
    blocks = captured["body"]["messages"][1]["content"]
    assert [b["type"] for b in blocks] == ["tool_use"], blocks


def test_real_assistant_text_beside_a_tool_call_survives(monkeypatch):
    captured = _capture(
        monkeypatch,
        [
            {"role": "user", "content": "q"},
            {
                "role": "assistant",
                "content": "calling f",
                "tool_calls": [
                    {
                        "id": "toolu_1",
                        "type": "function",
                        "function": {"name": "f", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "toolu_1", "content": "42"},
        ],
    )
    blocks = captured["body"]["messages"][1]["content"]
    assert blocks[0] == {"type": "text", "text": "calling f"}
    assert blocks[1]["type"] == "tool_use"


def test_missing_text_key_does_not_raise(monkeypatch):
    # `part["text"]` used to KeyError, taking down the whole request.
    captured = _capture(
        monkeypatch,
        [
            {
                "role": "user",
                "content": [
                    {"type": "text"},
                    {"type": "image_url", "image_url": {"url": _IMAGE_DATA_URI}},
                ],
            }
        ],
    )
    parts = captured["body"]["messages"][0]["content"]
    assert [p["type"] for p in parts] == ["image"], parts
