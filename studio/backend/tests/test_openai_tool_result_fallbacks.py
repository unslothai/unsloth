# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression tests for OpenAI Responses tool-result rendering.

Two bug classes: empty web_search cards (per-card result seeded with
"Searching: <query>") and orphan shell_call cards (bundled-output
fallback + final flush at response.completed / response.incomplete).
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
    """Each card carries its own `Searching: <query>` text; no empties."""
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
    """Last call gets the aggregated citations; earlier calls keep their
    per-call `Searching:` text."""
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
    # Keep the LAST tool_end per id (citation overwrite for ws_2).
    for e in ends:
        by_id[e["tool_call_id"]] = e
    # First call keeps its own query.
    assert by_id["ws_1"]["result"] == "Searching: first query"
    # Last call overwritten with the citation block.
    assert "Title: Example A" in by_id["ws_2"]["result"]
    assert "URL: https://example.com/a" in by_id["ws_2"]["result"]


def test_web_search_empty_query_falls_back_to_empty_result(monkeypatch):
    """No query -> empty result (no `Searching:` placeholder)."""
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
    """Output bundled on the shell_call done event emits tool_end."""
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
    """Separate shell_call_output after bundled-output is a no-op."""
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
    """Orphan shell_call finalises via the response.completed flush."""
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


def test_shell_call_flushed_on_response_incomplete_truncation(monkeypatch):
    """Truncated streams (response.incomplete) also flush orphan calls."""
    sse_events = [
        {
            "type": "response.output_item.added",
            "item": {
                "type": "shell_call",
                "id": "scall_truncated",
                "action": {"commands": ["long_running"]},
            },
        },
        {
            "type": "response.output_item.done",
            "item": {
                "type": "shell_call",
                "id": "scall_truncated",
                "action": {"commands": ["long_running"]},
                "status": "in_progress",
            },
        },
        {
            "type": "response.incomplete",
            "response": {
                "incomplete_details": {"reason": "max_output_tokens"},
            },
        },
    ]
    lines = _drive_stream(sse_events, ["code_execution"], monkeypatch)
    events = _tool_events(lines)
    ends = [e for e in events if e["type"] == "tool_end"]
    assert any(e["tool_call_id"] == "scall_truncated" for e in ends)


def test_shell_call_incomplete_does_not_double_emit(monkeypatch):
    """response.incomplete is idempotent against already-finalised calls."""
    sse_events = [
        {
            "type": "response.output_item.done",
            "item": {
                "type": "shell_call",
                "id": "scall_done",
                "action": {"commands": ["echo done"]},
                "output": [
                    {
                        "stdout": "done\n",
                        "stderr": "",
                        "outcome": {"type": "exit", "exit_code": 0},
                    }
                ],
            },
        },
        {
            "type": "response.incomplete",
            "response": {
                "incomplete_details": {"reason": "max_output_tokens"},
            },
        },
    ]
    lines = _drive_stream(sse_events, ["code_execution"], monkeypatch)
    events = _tool_events(lines)
    ends = [e for e in events if e["type"] == "tool_end"]
    assert len(ends) == 1
    assert ends[0]["tool_call_id"] == "scall_done"
    assert "done" in ends[0]["result"]
