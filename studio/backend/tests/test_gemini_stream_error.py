# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import json

import httpx
import pytest

from core import research_runs

from .test_gemini_provider import _drive, _make_gemini_client, _mock_http, _parse_chunks

_TEXT = {
    "candidates": [
        {"content": {"parts": [{"text": "The three causes are"}], "role": "model"}, "index": 0}
    ],
    "modelVersion": "gemini-2.5-flash",
}


def _frame(event):
    return f"data: {json.dumps(event)}\r\n\r\n"


def _bare_error(code, message, status):
    return (
        json.dumps({"error": {"code": code, "message": message, "status": status}}, indent = 2) + "\n"
    )


_OVERLOADED = _bare_error(503, "The model is overloaded. Please try again later.", "UNAVAILABLE")
_OVERLOADED_SHOWN = {
    "message": "The model is overloaded. Please try again later. (UNAVAILABLE)",
    "code": "503",
}


def _stream_lines(monkeypatch, body, **kwargs):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content = body.encode("utf-8"),
            headers = {"content-type": "text/event-stream"},
        )

    _mock_http(monkeypatch, handler)

    async def run():
        client = _make_gemini_client()
        lines = [
            line
            async for line in client.stream_chat_completion(
                messages = [{"role": "user", "content": "hi"}],
                model = "gemini-2.5-flash",
                temperature = 0.7,
                top_p = 0.95,
                max_tokens = 64,
                **kwargs,
            )
        ]
        await client.close()
        return lines

    return _drive(run())


@pytest.mark.parametrize(
    ("body", "content", "expected"),
    [
        (
            _frame(_TEXT) + _OVERLOADED,
            "The three causes are",
            _OVERLOADED_SHOWN,
        ),
        (
            _OVERLOADED,
            "",
            _OVERLOADED_SHOWN,
        ),
        (
            _frame(_TEXT) + _bare_error(500, "An internal error has occurred.", "INTERNAL"),
            "The three causes are",
            {"message": "An internal error has occurred. (INTERNAL)", "code": "500"},
        ),
        (
            _frame(_TEXT)
            + _frame({"error": {"code": 503, "message": "Overloaded", "status": "UNAVAILABLE"}}),
            "The three causes are",
            {"message": "Overloaded (UNAVAILABLE)", "code": "503"},
        ),
    ],
    ids = [
        "bare-503-after-text",
        "bare-503-before-text",
        "bare-500-after-text",
        "data-503-after-text",
    ],
)
def test_midstream_error_ends_stream_with_error(monkeypatch, body, content, expected):
    lines = _stream_lines(monkeypatch, body)
    payloads = _parse_chunks(lines)

    assert payloads[-1:] == [
        {"error": {**expected, "type": "provider_error", "provider": "gemini"}}
    ]
    assert "data: [DONE]" not in lines
    combined = "".join(p["choices"][0]["delta"].get("content", "") for p in payloads[:-1])
    assert combined == content


def test_midstream_rate_limit_is_retried_by_research(monkeypatch):
    lines = _stream_lines(
        monkeypatch,
        _bare_error(429, "Resource has been exhausted.", "RESOURCE_EXHAUSTED"),
    )

    assert _parse_chunks(lines)[-1]["error"]["code"] == "429"
    assert research_runs._stream_rate_limit_delay(lines[0]) == 0.0


def test_midstream_error_marks_web_search_aborted(monkeypatch):
    lines = _stream_lines(monkeypatch, _frame(_TEXT) + _OVERLOADED, enabled_tools = ["web_search"])
    tool_end = [
        p["_toolEvent"]
        for p in _parse_chunks(lines)
        if p.get("_toolEvent", {}).get("type") == "tool_end"
    ]

    assert tool_end == [
        {
            "type": "tool_end",
            "tool_call_id": "gemini_web_search",
            "result": "(search aborted: The model is overloaded. Please try again later.)",
        }
    ]
