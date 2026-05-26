# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression tests for OpenAI Responses tool-result rendering.

Two bug classes covered:

1. web_search: every call's tool_end carried `result: ""`, leaving every
   card except the last one empty in the chat thread. Now each call seeds
   its own `Searching: <query>` text so the cards always render content;
   the last call is still overwritten at response.completed with the
   aggregated citation list (consumed by the source-pill extractor).

2. shell_call (code_execution): when OpenAI bundles the output on the
   shell_call item's `response.output_item.done` event instead of as a
   separate `shell_call_output`, the previous handler never emitted
   tool_end and the card stayed in "running" with no output. The fallback
   now emits tool_end from the bundled output. A final flush at
   response.completed catches shell_call ids that received neither.
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


def _make_client(base_url: str = "https://api.openai.com/v1") -> ExternalProviderClient:
    return ExternalProviderClient(
        provider_type = "openai",
        base_url = base_url,
        api_key = "sk-test",
    )


def _openai_sse(events: list[dict]) -> bytes:
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


def _drive_stream(sse_events, enabled_tools, monkeypatch):
    def handler(request):
        return httpx.Response(
            200,
            content = _openai_sse(sse_events),
            headers = {"content-type": "text/event-stream"},
        )

    _mock_http_client(monkeypatch, handler)

    async def run():
        client = _make_client()
        return await _collect(
            client._stream_openai_responses(
                messages = [{"role": "user", "content": "x"}],
                model = "gpt-5.5",
                temperature = 0.7,
                top_p = 0.95,
                max_tokens = 4096,
                enable_thinking = None,
                reasoning_effort = None,
                enabled_tools = enabled_tools,
            )
        )

    return _drive(run())


# ── web_search per-card result ─────────────────────────────────────────


def test_web_search_each_call_carries_its_own_query_as_result(monkeypatch):
    """Three search calls, no citations. Each card must render with its
    own `Searching: <query>` text; none should be empty."""
    sse_events = [
        {
            "type": "response.output_item.done",
            "item": {
                "type": "web_search_call",
                "id": "ws_1",
                "action": {"query": "popular animals 2026"},
            },
        },
        {
            "type": "response.output_item.done",
            "item": {
                "type": "web_search_call",
                "id": "ws_2",
                "action": {"query": "most loved animals poll"},
            },
        },
        {
            "type": "response.output_item.done",
            "item": {
                "type": "web_search_call",
                "id": "ws_3",
                "action": {"query": "tiger ranking"},
            },
        },
        {"type": "response.completed", "response": {}},
    ]
    lines = _drive_stream(sse_events, ["web_search"], monkeypatch)
    events = _tool_events(lines)
    ends = [e for e in events if e["type"] == "tool_end"]
    by_id = {e["tool_call_id"]: e for e in ends}
    assert by_id["ws_1"]["result"] == "Searching: popular animals 2026"
    assert by_id["ws_2"]["result"] == "Searching: most loved animals poll"
    assert by_id["ws_3"]["result"] == "Searching: tiger ranking"


def test_web_search_last_call_overwritten_with_citations(monkeypatch):
    """Pin the existing behaviour: the last call still gets the
    aggregated citation list (the source-pill extractor depends on this).
    Earlier calls keep their per-call `Searching:` text."""
    sse_events = [
        {
            "type": "response.output_item.done",
            "item": {
                "type": "web_search_call",
                "id": "ws_1",
                "action": {"query": "first query"},
            },
        },
        {
            "type": "response.output_item.done",
            "item": {
                "type": "web_search_call",
                "id": "ws_2",
                "action": {"query": "second query"},
            },
        },
        {
            "type": "response.output_text.annotation.added",
            "annotation": {
                "type": "url_citation",
                "url": "https://example.com/a",
                "title": "Example A",
            },
        },
        {"type": "response.completed", "response": {}},
    ]
    lines = _drive_stream(sse_events, ["web_search"], monkeypatch)
    events = _tool_events(lines)
    ends = [e for e in events if e["type"] == "tool_end"]
    by_id: dict = {}
    # Keep the LAST tool_end per id (the citation overwrite for ws_2).
    for e in ends:
        by_id[e["tool_call_id"]] = e
    # First call keeps its own query.
    assert by_id["ws_1"]["result"] == "Searching: first query"
    # Last call gets overwritten with the citation block.
    assert "Title: Example A" in by_id["ws_2"]["result"]
    assert "URL: https://example.com/a" in by_id["ws_2"]["result"]


def test_web_search_empty_query_falls_back_to_empty_result(monkeypatch):
    """Defensive: if the action carries no query, do not write a junk
    `Searching:` placeholder; leave the result empty so the existing
    last-call overwrite path is unchanged."""
    sse_events = [
        {
            "type": "response.output_item.done",
            "item": {
                "type": "web_search_call",
                "id": "ws_only",
                "action": {},
            },
        },
        {"type": "response.completed", "response": {}},
    ]
    lines = _drive_stream(sse_events, ["web_search"], monkeypatch)
    events = _tool_events(lines)
    ends = [e for e in events if e["type"] == "tool_end"]
    assert len(ends) == 1
    assert ends[0]["result"] == ""


# ── shell_call output fallbacks ────────────────────────────────────────


def test_shell_call_emits_tool_end_when_output_bundled_on_done(monkeypatch):
    """Some Responses streams ship the output array embedded on the
    shell_call item's done event instead of as a separate
    shell_call_output. The fallback must emit tool_end from that bundled
    output so the card never stays in "running"."""
    sse_events = [
        {
            "type": "response.output_item.done",
            "item": {
                "type": "shell_call",
                "id": "scall_bundled",
                "action": {"commands": ["echo hi"]},
                "output": [
                    {
                        "stdout": "hi\n",
                        "stderr": "",
                        "outcome": {"type": "exit", "exit_code": 0},
                    }
                ],
            },
        },
        {"type": "response.completed", "response": {}},
    ]
    lines = _drive_stream(sse_events, ["code_execution"], monkeypatch)
    events = _tool_events(lines)
    starts = [e for e in events if e["type"] == "tool_start"]
    ends = [e for e in events if e["type"] == "tool_end"]
    assert len(starts) == 1
    assert starts[0]["tool_call_id"] == "scall_bundled"
    assert len(ends) == 1
    assert ends[0]["tool_call_id"] == "scall_bundled"
    assert "hi" in ends[0]["result"]


def test_shell_call_bundled_then_separate_output_does_not_double_emit(monkeypatch):
    """If the bundled output ALREADY emitted tool_end and a separate
    shell_call_output event arrives afterwards (some streams do both),
    the second one is skipped so the card is not re-completed."""
    sse_events = [
        {
            "type": "response.output_item.done",
            "item": {
                "type": "shell_call",
                "id": "scall_both",
                "action": {"commands": ["echo bundle"]},
                "output": [
                    {
                        "stdout": "bundle\n",
                        "stderr": "",
                        "outcome": {"type": "exit", "exit_code": 0},
                    }
                ],
            },
        },
        {
            "type": "response.output_item.done",
            "item": {
                "type": "shell_call_output",
                "id": "scout_both",
                "call_id": "scall_both",
                "output": [
                    {
                        "stdout": "should not double-emit\n",
                        "stderr": "",
                        "outcome": {"type": "exit", "exit_code": 0},
                    }
                ],
            },
        },
        {"type": "response.completed", "response": {}},
    ]
    lines = _drive_stream(sse_events, ["code_execution"], monkeypatch)
    events = _tool_events(lines)
    ends = [e for e in events if e["type"] == "tool_end"]
    assert len(ends) == 1
    assert ends[0]["tool_call_id"] == "scall_both"
    assert "bundle" in ends[0]["result"]
    assert "should not double-emit" not in ends[0]["result"]


def test_shell_call_final_flush_on_completed_when_no_output_event(monkeypatch):
    """shell_call gets tool_start but no shell_call_output arrives and
    the done event has no bundled output either. The final flush at
    response.completed must emit tool_end so the card finalises."""
    sse_events = [
        {
            "type": "response.output_item.added",
            "item": {
                "type": "shell_call",
                "id": "scall_orphan",
                "action": {"commands": ["true"]},
            },
        },
        {
            "type": "response.output_item.done",
            "item": {
                "type": "shell_call",
                "id": "scall_orphan",
                "action": {"commands": ["true"]},
                "status": "completed",
            },
        },
        {"type": "response.completed", "response": {}},
    ]
    lines = _drive_stream(sse_events, ["code_execution"], monkeypatch)
    events = _tool_events(lines)
    ends = [e for e in events if e["type"] == "tool_end"]
    assert any(e["tool_call_id"] == "scall_orphan" for e in ends)
