# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Unit tests for Anthropic's server-side `web_fetch_20250910` tool
translation in `_stream_anthropic`.

Covers:
- Request body: when ``enabled_tools=["web_fetch"]``, the outbound
  ``tools`` array carries ``{"type":"web_fetch_20250910",
  "name":"web_fetch", "max_uses":5}``. No beta header is required.
- Combined request: ``enabled_tools=["web_search","web_fetch",
  "code_execution"]`` sends all three tool entries.
- Disabled by default: with ``enabled_tools=["web_search"]`` (or None),
  the body does NOT carry a web_fetch entry.
- SSE translation (success): a `web_fetch` server_tool_use streaming
  ``{"url": "..."}`` followed by a `web_fetch_tool_result` block with
  a document source emits one ``tool_start`` and one ``tool_end``
  `_toolEvent`. The ``tool_start.arguments.url`` matches the fetched
  URL and the ``tool_end.result`` carries the Title / URL / snippet
  prefix the source-pill renderer expects.
- SSE translation (error): a `web_fetch_tool_error` with
  ``error_code="url_not_accessible"`` renders as ``"Error:
  url_not_accessible"`` in the tool_end result.
"""

import asyncio
import json

import httpx

from core.inference import external_provider as ep_mod
from core.inference.external_provider import ExternalProviderClient


def _drive(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


async def _collect(agen):
    out = []
    async for line in agen:
        out.append(line)
    return out


def _mock_http_client(monkeypatch, handler):
    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(ep_mod, "_http_client", httpx.AsyncClient(transport = transport))


def _make_client() -> ExternalProviderClient:
    return ExternalProviderClient(
        provider_type = "anthropic",
        base_url = "https://api.anthropic.com/v1",
        api_key = "sk-ant-test",
    )


def _anthropic_sse(events: list[dict]) -> bytes:
    chunks: list[str] = []
    for event in events:
        chunks.append(f"event: {event['type']}")
        chunks.append(f"data: {json.dumps(event)}")
        chunks.append("")
    return ("\n".join(chunks) + "\n").encode("utf-8")


def _tool_events(lines: list[str]) -> list[dict]:
    out: list[dict] = []
    for line in lines:
        if not line.startswith("data:"):
            continue
        raw = line[len("data:") :].strip()
        if not raw or raw == "[DONE]":
            continue
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and "_toolEvent" in parsed:
            out.append(parsed["_toolEvent"])
    return out


# ── request body ────────────────────────────────────────────────────


def test_web_fetch_tool_appended_to_request_body(monkeypatch):
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode("utf-8"))
        captured["headers"] = dict(request.headers)
        return httpx.Response(
            200,
            content = _anthropic_sse([{"type": "message_stop"}]),
            headers = {"content-type": "text/event-stream"},
        )

    _mock_http_client(monkeypatch, handler)

    async def run():
        client = _make_client()
        async for _ in client._stream_anthropic(
            messages = [{"role": "user", "content": "Fetch https://example.com/article"}],
            model = "claude-opus-4-7",
            temperature = 0.7,
            top_p = 0.95,
            max_tokens = 1024,
            enabled_tools = ["web_fetch"],
        ):
            pass
        await client.close()

    _drive(run())

    body = captured["body"]
    tools = body.get("tools") or []
    assert {
        "type": "web_fetch_20250910",
        "name": "web_fetch",
        "max_uses": 5,
    } in tools
    # web_fetch is GA; no beta header is required.
    assert "web-fetch" not in captured["headers"].get("anthropic-beta", "")


def test_web_fetch_combined_with_web_search_and_code_execution(monkeypatch):
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode("utf-8"))
        captured["headers"] = dict(request.headers)
        return httpx.Response(
            200,
            content = _anthropic_sse([{"type": "message_stop"}]),
            headers = {"content-type": "text/event-stream"},
        )

    _mock_http_client(monkeypatch, handler)

    async def run():
        client = _make_client()
        async for _ in client._stream_anthropic(
            messages = [{"role": "user", "content": "research this"}],
            model = "claude-opus-4-7",
            temperature = 0.7,
            top_p = 0.95,
            max_tokens = 1024,
            enabled_tools = ["web_search", "web_fetch", "code_execution"],
        ):
            pass
        await client.close()

    _drive(run())

    tools = captured["body"].get("tools") or []
    tool_types = [t.get("type") for t in tools]
    # After PR 5679's per-model tool version dispatch landed,
    # claude-opus-4-7 routes web_search to the _20260209 variant and
    # code_execution to _20260120. web_fetch still hardcodes
    # _20250910 today; see follow-up to thread it through
    # _anthropic_web_fetch_version.
    assert "web_search_20260209" in tool_types, tool_types
    assert "web_fetch_20250910" in tool_types, tool_types
    assert "code_execution_20260120" in tool_types, tool_types
    # Code-execution still adds its beta flag; web_fetch must not
    # have accidentally stripped it.
    assert "code-execution-2025-08-25" in captured["headers"].get("anthropic-beta", "")


def test_no_web_fetch_tool_when_pill_off(monkeypatch):
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            content = _anthropic_sse([{"type": "message_stop"}]),
            headers = {"content-type": "text/event-stream"},
        )

    _mock_http_client(monkeypatch, handler)

    async def run():
        client = _make_client()
        async for _ in client._stream_anthropic(
            messages = [{"role": "user", "content": "hi"}],
            model = "claude-opus-4-7",
            temperature = 0.7,
            top_p = 0.95,
            max_tokens = 64,
            enabled_tools = ["web_search"],
        ):
            pass
        await client.close()

    _drive(run())

    tools = captured["body"].get("tools") or []
    assert all(t.get("type") != "web_fetch_20250910" for t in tools)


# ── SSE translation ─────────────────────────────────────────────────


def test_web_fetch_success_emits_tool_start_and_end(monkeypatch):
    sse_events = [
        {"type": "message_start", "message": {"usage": {}}},
        # The model decides to fetch.
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "server_tool_use",
                "id": "srvtoolu_wf1",
                "name": "web_fetch",
            },
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {
                "type": "input_json_delta",
                "partial_json": '{"url": "https://example.com/article"}',
            },
        },
        {"type": "content_block_stop", "index": 0},
        # Anthropic returns the fetched document inline.
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {
                "type": "web_fetch_tool_result",
                "tool_use_id": "srvtoolu_wf1",
                "content": {
                    "type": "web_fetch_result",
                    "url": "https://example.com/article",
                    "retrieved_at": "2026-05-21T12:00:00Z",
                    "content": {
                        "type": "document",
                        "source": {
                            "type": "text",
                            "media_type": "text/plain",
                            "data": "Article body text begins here.",
                        },
                        "title": "Example Article",
                    },
                },
            },
        },
        {"type": "content_block_stop", "index": 1},
        {"type": "message_stop"},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content = _anthropic_sse(sse_events),
            headers = {"content-type": "text/event-stream"},
        )

    _mock_http_client(monkeypatch, handler)

    async def run():
        client = _make_client()
        return await _collect(
            client._stream_anthropic(
                messages = [
                    {"role": "user", "content": "Fetch https://example.com/article"}
                ],
                model = "claude-opus-4-7",
                temperature = 0.7,
                top_p = 0.95,
                max_tokens = 1024,
                enabled_tools = ["web_fetch"],
            )
        )

    lines = _drive(run())
    events = _tool_events(lines)
    assert len(events) == 2, f"expected 1 start + 1 end, got {events}"
    start, end = events
    assert start["type"] == "tool_start"
    assert start["tool_name"] == "web_fetch"
    assert start["tool_call_id"] == "srvtoolu_wf1"
    assert start["arguments"] == {"url": "https://example.com/article"}
    assert end["type"] == "tool_end"
    assert end["tool_call_id"] == "srvtoolu_wf1"
    # The source pill uses Title / URL / snippet as parseSourcesFromResult expects.
    assert "Title: Example Article" in end["result"]
    assert "URL: https://example.com/article" in end["result"]
    assert "Snippet: Article body text begins here." in end["result"]


def test_web_fetch_error_renders_error_code(monkeypatch):
    sse_events = [
        {"type": "message_start", "message": {"usage": {}}},
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "server_tool_use",
                "id": "srvtoolu_wf2",
                "name": "web_fetch",
            },
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {
                "type": "input_json_delta",
                "partial_json": '{"url": "https://example.com/404"}',
            },
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {
                "type": "web_fetch_tool_result",
                "tool_use_id": "srvtoolu_wf2",
                "content": {
                    "type": "web_fetch_tool_error",
                    "error_code": "url_not_accessible",
                },
            },
        },
        {"type": "content_block_stop", "index": 1},
        {"type": "message_stop"},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content = _anthropic_sse(sse_events),
            headers = {"content-type": "text/event-stream"},
        )

    _mock_http_client(monkeypatch, handler)

    async def run():
        client = _make_client()
        return await _collect(
            client._stream_anthropic(
                messages = [{"role": "user", "content": "fetch 404"}],
                model = "claude-opus-4-7",
                temperature = 0.7,
                top_p = 0.95,
                max_tokens = 1024,
                enabled_tools = ["web_fetch"],
            )
        )

    lines = _drive(run())
    events = _tool_events(lines)
    assert len(events) == 2
    end = events[1]
    assert end["type"] == "tool_end"
    assert end["result"] == "Error: url_not_accessible"


# ── pause_turn must not emit a truncating finish_reason ─────────────


def _finish_reasons(lines: list[str]) -> list:
    """Return the non-null finish_reason fields from every
    chat.completion.chunk.

    Content delta chunks carry ``finish_reason: None`` -- those are
    not finish reasons, they are mid-stream deltas. We only care about
    the chunks that actually announce the end of the assistant turn
    (the post-PR refusal path emits a user-visible content notice
    before the mapped content_filter chunk, which is exactly the
    case this helper used to ignore on the older streams)."""
    out: list = []
    for line in lines:
        if not line.startswith("data:"):
            continue
        raw = line[len("data:") :].strip()
        if not raw or raw == "[DONE]":
            continue
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if parsed.get("object") != "chat.completion.chunk":
            continue
        for choice in parsed.get("choices") or []:
            reason = choice.get("finish_reason")
            if reason is not None:
                out.append(reason)
    return out


def test_pause_turn_does_not_emit_finish_reason_chunk(monkeypatch):
    # `pause_turn` is what Anthropic emits when a long server-tool turn
    # (typically web_search / web_fetch) pauses and will resume on the
    # next request. Treating it as finish_reason="stop" makes the
    # OpenAI-formatted client truncate the rendered assistant message.
    # The adapter must skip the chunk so the stream ends cleanly with
    # [DONE] and no terminal finish_reason.
    sse_events = [
        {"type": "message_start", "message": {"usage": {}}},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "pause_turn"},
            "usage": {"input_tokens": 100, "output_tokens": 10},
        },
        {"type": "message_stop"},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content = _anthropic_sse(sse_events),
            headers = {"content-type": "text/event-stream"},
        )

    _mock_http_client(monkeypatch, handler)

    async def run():
        client = _make_client()
        return await _collect(
            client._stream_anthropic(
                messages = [{"role": "user", "content": "Search and read."}],
                model = "claude-opus-4-7",
                temperature = 0.7,
                top_p = 0.95,
                max_tokens = 1024,
                enabled_tools = ["web_search", "web_fetch"],
            )
        )

    lines = _drive(run())
    # No finish_reason chunk for pause_turn -- the only completion
    # signal is the [DONE] line.
    assert _finish_reasons(lines) == [], lines
    assert any(line.strip() == "data: [DONE]" for line in lines), lines


def test_end_turn_still_emits_stop_finish_reason(monkeypatch):
    # Sanity: the pause_turn -> None mapping must not regress normal
    # end_turn handling.
    sse_events = [
        {"type": "message_start", "message": {"usage": {}}},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"input_tokens": 100, "output_tokens": 10},
        },
        {"type": "message_stop"},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content = _anthropic_sse(sse_events),
            headers = {"content-type": "text/event-stream"},
        )

    _mock_http_client(monkeypatch, handler)

    async def run():
        client = _make_client()
        return await _collect(
            client._stream_anthropic(
                messages = [{"role": "user", "content": "hi"}],
                model = "claude-opus-4-7",
                temperature = 0.7,
                top_p = 0.95,
                max_tokens = 1024,
            )
        )

    lines = _drive(run())
    assert _finish_reasons(lines) == ["stop"], lines


def test_refusal_maps_to_content_filter(monkeypatch):
    sse_events = [
        {"type": "message_start", "message": {"usage": {}}},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "refusal"},
            "usage": {"input_tokens": 100, "output_tokens": 0},
        },
        {"type": "message_stop"},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content = _anthropic_sse(sse_events),
            headers = {"content-type": "text/event-stream"},
        )

    _mock_http_client(monkeypatch, handler)

    async def run():
        client = _make_client()
        return await _collect(
            client._stream_anthropic(
                messages = [{"role": "user", "content": "hi"}],
                model = "claude-opus-4-7",
                temperature = 0.7,
                top_p = 0.95,
                max_tokens = 1024,
            )
        )

    lines = _drive(run())
    assert _finish_reasons(lines) == ["content_filter"], lines


def test_web_fetch_titleless_document_falls_back_to_url(monkeypatch):
    # Anthropic may omit `document.title` on pages where the HTML
    # provides nothing usable. Without a fallback the formatter would
    # emit `URL: ...\nSnippet: ...` only, and the frontend's
    # parseSourcesFromResult skips entries that lack a `Title:` line,
    # so the source pill silently disappears. Verify the formatter
    # mirrors the web_search behaviour and falls back to the URL.
    sse_events = [
        {"type": "message_start", "message": {"usage": {}}},
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "server_tool_use",
                "id": "srvtoolu_wf3",
                "name": "web_fetch",
            },
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {
                "type": "input_json_delta",
                "partial_json": '{"url": "https://example.com/raw"}',
            },
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {
                "type": "web_fetch_tool_result",
                "tool_use_id": "srvtoolu_wf3",
                "content": {
                    "type": "web_fetch_result",
                    "url": "https://example.com/raw",
                    "retrieved_at": "2026-05-21T12:00:00Z",
                    "content": {
                        "type": "document",
                        "source": {
                            "type": "text",
                            "media_type": "text/plain",
                            "data": "Raw body without an HTML title tag.",
                        },
                        # No `title` field on the document.
                    },
                },
            },
        },
        {"type": "content_block_stop", "index": 1},
        {"type": "message_stop"},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content = _anthropic_sse(sse_events),
            headers = {"content-type": "text/event-stream"},
        )

    _mock_http_client(monkeypatch, handler)

    async def run():
        client = _make_client()
        return await _collect(
            client._stream_anthropic(
                messages = [{"role": "user", "content": "fetch raw"}],
                model = "claude-opus-4-7",
                temperature = 0.7,
                top_p = 0.95,
                max_tokens = 1024,
                enabled_tools = ["web_fetch"],
            )
        )

    lines = _drive(run())
    events = _tool_events(lines)
    assert len(events) == 2
    end = events[1]
    assert end["type"] == "tool_end"
    # Title must be present so parseSourcesFromResult emits a pill.
    assert "Title: https://example.com/raw" in end["result"]
    assert "URL: https://example.com/raw" in end["result"]
    assert "Snippet: Raw body without an HTML title tag." in end["result"]
