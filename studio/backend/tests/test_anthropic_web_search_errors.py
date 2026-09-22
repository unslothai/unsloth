# SPDX-License-Identifier: AGPL-3.0-only

import asyncio
import json

import httpx
import pytest

from core.inference import external_provider as ep_mod
from .test_anthropic_web_fetch import _anthropic_sse, _make_client, _tool_events


@pytest.mark.parametrize(
    "content, expected",
    [
        ({"type": "web_search_tool_result_error", "error_code": code}, f"Error: {code}")
        for code in (
            "too_many_requests",
            "invalid_tool_input",
            "max_uses_exceeded",
            "query_too_long",
            "request_too_large",
            "unavailable",
        )
    ]
    + [
        ([], "(search complete)"),
        (
            [{"type": "web_search_result", "url": "https://example.com", "title": "Example"}],
            "Title: Example\nURL: https://example.com",
        ),
    ],
)
def test_native_search_result_preserves_outcome_and_answer(monkeypatch, content, expected):
    events = [
        {"type": "message_start", "message": {"usage": {}}},
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "server_tool_use", "id": "search-1", "name": "web_search"},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": '{"query":"example"}'},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {
                "type": "web_search_tool_result",
                "tool_use_id": "search-1",
                "content": content,
            },
        },
        {"type": "content_block_stop", "index": 1},
        {"type": "content_block_start", "index": 2, "content_block": {"type": "text", "text": ""}},
        {
            "type": "content_block_delta",
            "index": 2,
            "delta": {"type": "text_delta", "text": "The answer continues."},
        },
        {"type": "content_block_stop", "index": 2},
        {"type": "message_stop"},
    ]

    async def run():
        def handler(request):
            body = json.loads(request.content)
            assert any(t["name"] == "web_search" and t["max_uses"] == 5 for t in body["tools"])
            return httpx.Response(200, content = _anthropic_sse(events))

        async with httpx.AsyncClient(transport = httpx.MockTransport(handler)) as transport:
            monkeypatch.setattr(ep_mod, "_http_client", transport)
            return [
                line
                async for line in _make_client()._stream_anthropic(
                    messages = [{"role": "user", "content": "Search for example"}],
                    model = "claude-sonnet-4-5",
                    temperature = 0.7,
                    top_p = 0.95,
                    max_tokens = 1024,
                    enabled_tools = ["web_search"],
                )
            ]

    lines = asyncio.run(run())
    start, end = _tool_events(lines)
    assert start["arguments"] == {"query": "example", "_server_tool": True}
    assert end == {"type": "tool_end", "tool_call_id": "search-1", "result": expected}
    assert any("The answer continues." in line for line in lines)
    assert lines[-1] == "data: [DONE]"
