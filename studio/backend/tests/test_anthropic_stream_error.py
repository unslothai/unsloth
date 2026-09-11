# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import httpx
import pytest

from .test_anthropic_thinking_translation import (
    _anthropic_sse,
    _collect,
    _drive,
    _make_client,
    _mock_http_client,
    _payloads_from_lines,
)

_MESSAGE_START = {
    "type": "message_start",
    "message": {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "content": [],
        "usage": {"input_tokens": 5, "output_tokens": 1},
    },
}
_TEXT = [
    {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
    {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "text_delta", "text": "The three causes are"},
    },
]
_THINKING = [
    {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "thinking", "thinking": "", "signature": ""},
    },
    {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "thinking_delta", "thinking": "Let me plan"},
    },
]


def _error(error_type, message):
    return {"type": "error", "error": {"type": error_type, "message": message}}


@pytest.mark.parametrize(
    ("events", "error", "content", "expected"),
    [
        (
            _TEXT,
            _error("overloaded_error", "Overloaded"),
            "The three causes are",
            {"message": "Overloaded (overloaded_error)", "code": "529"},
        ),
        (
            [],
            _error("overloaded_error", "Overloaded"),
            "",
            {"message": "Overloaded (overloaded_error)", "code": "529"},
        ),
        (
            _THINKING,
            _error("overloaded_error", "Overloaded"),
            "<think>Let me plan</think>",
            {"message": "Overloaded (overloaded_error)", "code": "529"},
        ),
        (
            _TEXT,
            _error("api_error", "Internal server error"),
            "The three causes are",
            {"message": "Internal server error (api_error)", "code": "502"},
        ),
    ],
)
def test_midstream_error_event_ends_stream_with_error(
    monkeypatch, events, error, content, expected
):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content = _anthropic_sse([_MESSAGE_START, *events, error]),
            headers = {"content-type": "text/event-stream"},
        )

    _mock_http_client(monkeypatch, handler)

    async def run():
        client = _make_client()
        lines = await _collect(
            client._stream_anthropic(
                messages = [{"role": "user", "content": "hi"}],
                model = "claude-opus-4-6",
                temperature = 0.7,
                top_p = 0.95,
                max_tokens = 4096,
            )
        )
        await client.close()
        return lines

    payloads = _payloads_from_lines(_drive(run()))

    assert payloads[-1:] == [
        {"error": {**expected, "type": "provider_error", "provider": "anthropic"}}
    ]
    assert "[DONE]" not in payloads
    combined = "".join(p["choices"][0]["delta"].get("content", "") for p in payloads[:-1])
    assert combined == content
