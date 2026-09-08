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


# Anthropic rejects a whitespace-only text block as well as an empty one, with
# "text content blocks must contain non-whitespace text". Studio reaches that shape
# on its own: collectTextParts joins with "\n", so a turn carrying two empty caption
# parts serialises to "\n" rather than "".


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
    # The string shape of the same defect. The runtime substitutes an empty text
    # part for a stored message with no content, which serialises to "" -- and
    # Anthropic reads a plain string as one text block.
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
    # With caching on, the tail breakpoint lands on the LAST surviving message. If
    # the empty turn survived, cache_control would sit on an empty text block, which
    # Anthropic rejects separately ("cache_control cannot be set for empty text
    # blocks").
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


def test_missing_text_key_does_not_raise(monkeypatch):
    # `part["text"]` used to KeyError here, taking the whole request down rather
    # than dropping one unusable block.
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
