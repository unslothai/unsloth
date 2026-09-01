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


def test_web_search_open_page_action_renders_url(monkeypatch):
    """gpt-5.x agentic search emits `open_page` with a url (no query)."""
    sse_events = [
        {
            "type": "response.output_item.done",
            "item": {
                "type": "web_search_call",
                "id": "ws_open",
                "action": {
                    "type": "open_page",
                    "url": "https://en.wikipedia.org/wiki/Tiger",
                },
            },
        },
        {"type": "response.completed", "response": {}},
    ]
    lines = _drive_stream(sse_events, ["web_search"], monkeypatch)
    events = _tool_events(lines)
    starts = [e for e in events if e["type"] == "tool_start"]
    ends = [e for e in events if e["type"] == "tool_end"]
    assert starts[0]["arguments"]["url"] == "https://en.wikipedia.org/wiki/Tiger"
    assert starts[0]["arguments"]["action_type"] == "open_page"
    assert "Read: https://en.wikipedia.org/wiki/Tiger" in ends[0]["result"]


def test_web_search_find_in_page_action_renders_url_and_pattern(monkeypatch):
    """`find_in_page` actions surface both url and pattern to the card."""
    sse_events = [
        {
            "type": "response.output_item.done",
            "item": {
                "type": "web_search_call",
                "id": "ws_find",
                "action": {
                    "type": "find_in_page",
                    "url": "https://en.wikipedia.org/wiki/Tiger",
                    "pattern": "population",
                },
            },
        },
        {"type": "response.completed", "response": {}},
    ]
    lines = _drive_stream(sse_events, ["web_search"], monkeypatch)
    events = _tool_events(lines)
    starts = [e for e in events if e["type"] == "tool_start"]
    ends = [e for e in events if e["type"] == "tool_end"]
    assert starts[0]["arguments"]["url"] == "https://en.wikipedia.org/wiki/Tiger"
    assert starts[0]["arguments"]["pattern"] == "population"
    assert starts[0]["arguments"]["action_type"] == "find_in_page"
    assert "population" in ends[0]["result"]
    assert "https://en.wikipedia.org/wiki/Tiger" in ends[0]["result"]


def test_web_search_action_queries_plural_falls_back(monkeypatch):
    """`action.queries[0]` is used when `action.query` is absent (older shape)."""
    sse_events = [
        {
            "type": "response.output_item.done",
            "item": {
                "type": "web_search_call",
                "id": "ws_plural",
                "action": {"queries": ["renewable energy 2026"]},
            },
        },
        {"type": "response.completed", "response": {}},
    ]
    lines = _drive_stream(sse_events, ["web_search"], monkeypatch)
    events = _tool_events(lines)
    starts = [e for e in events if e["type"] == "tool_start"]
    ends = [e for e in events if e["type"] == "tool_end"]
    assert starts[0]["arguments"]["query"] == "renewable energy 2026"
    assert ends[0]["result"] == "Searching: renewable energy 2026"


def test_web_search_per_call_results_formatted_as_source_blocks(monkeypatch):
    """`results` array (reasoning models) is formatted into Title/URL/Snippet blocks."""
    sse_events = [
        {
            "type": "response.output_item.done",
            "item": {
                "type": "web_search_call",
                "id": "ws_r",
                "action": {"type": "search", "query": "tiger ranking"},
                "results": [
                    {
                        "url": "https://a.example/1",
                        "title": "Tigers",
                        "snippet": "Big cats",
                    },
                    {"url": "https://b.example/2", "title": "Lion stats"},
                ],
            },
        },
        {"type": "response.completed", "response": {}},
    ]
    lines = _drive_stream(sse_events, ["web_search"], monkeypatch)
    events = _tool_events(lines)
    ends = [e for e in events if e["type"] == "tool_end"]
    body = ends[0]["result"]
    assert "Title: Tigers" in body
    assert "URL: https://a.example/1" in body
    assert "Snippet: Big cats" in body
    assert "Title: Lion stats" in body


def test_web_search_action_sources_url_only_falls_back(monkeypatch):
    """`action.sources` URLs are surfaced when `results` is absent."""
    sse_events = [
        {
            "type": "response.output_item.done",
            "item": {
                "type": "web_search_call",
                "id": "ws_s",
                "action": {
                    "type": "search",
                    "query": "X",
                    "sources": [
                        {"type": "url", "url": "https://x.example/1"},
                        "https://x.example/2",
                    ],
                },
            },
        },
        {"type": "response.completed", "response": {}},
    ]
    lines = _drive_stream(sse_events, ["web_search"], monkeypatch)
    events = _tool_events(lines)
    ends = [e for e in events if e["type"] == "tool_end"]
    body = ends[0]["result"]
    assert "https://x.example/1" in body
    assert "https://x.example/2" in body


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


def _capture_body(monkeypatch, *, base_url: str, enabled_tools) -> dict:
    """Run one request against a mock transport and return the JSON body sent."""
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            content = _openai_sse([{"type": "response.completed", "response": {}}]),
            headers = {"content-type": "text/event-stream"},
        )

    _mock_http_client(monkeypatch, handler)

    async def run():
        client = _make_client(base_url)
        await _collect(
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
        await client.close()

    _drive(run())
    return captured["body"]


def test_cloud_openai_requests_web_search_sources(monkeypatch):
    body = _capture_body(
        monkeypatch,
        base_url = "https://api.openai.com/v1",
        enabled_tools = ["web_search"],
    )
    assert {"type": "web_search"} in body["tools"]
    assert body["include"] == ["web_search_call.action.sources", "web_search_call.results"]


def test_azure_openai_requests_web_search_sources(monkeypatch):
    body = _capture_body(
        monkeypatch,
        base_url = "https://my-resource.openai.azure.com/openai/v1",
        enabled_tools = ["web_search"],
    )
    assert body["include"] == ["web_search_call.action.sources", "web_search_call.results"]


def test_custom_base_url_keeps_web_search_but_drops_the_include(monkeypatch):
    for base_url in (
        "http://127.0.0.1:11434/v1",
        "https://api.openai.com.attacker.com/v1",
        "https://evil.com/api.openai.com/v1",
    ):
        body = _capture_body(
            monkeypatch,
            base_url = base_url,
            enabled_tools = ["web_search"],
        )
        assert {"type": "web_search"} in body["tools"], base_url
        assert "include" not in body, base_url


def test_no_web_search_means_no_include(monkeypatch):
    body = _capture_body(
        monkeypatch,
        base_url = "https://api.openai.com/v1",
        enabled_tools = ["code_execution"],
    )
    assert "include" not in body


def test_web_search_done_event_keeps_fields_from_the_added_event(monkeypatch):
    sse_events = [
        {
            "type": "response.output_item.added",
            "item": {
                "type": "web_search_call",
                "id": "ws_partial",
                "action": {
                    "type": "find_in_page",
                    "url": "https://example.com/tigers",
                    "pattern": "population",
                },
            },
        },
        {
            "type": "response.output_item.done",
            "item": {"type": "web_search_call", "id": "ws_partial"},
        },
        {"type": "response.completed", "response": {}},
    ]
    lines = _drive_stream(sse_events, ["web_search"], monkeypatch)
    starts = [e for e in _tool_events(lines) if e["type"] == "tool_start"]
    assert starts[0]["arguments"]["url"] == "https://example.com/tigers"
    assert starts[0]["arguments"]["pattern"] == "population"
    assert starts[0]["arguments"]["action_type"] == "find_in_page"


def test_web_search_done_event_wins_where_it_has_a_value(monkeypatch):
    sse_events = [
        {
            "type": "response.output_item.added",
            "item": {
                "type": "web_search_call",
                "id": "ws_upd",
                "action": {"type": "search", "query": "first"},
            },
        },
        {
            "type": "response.output_item.done",
            "item": {
                "type": "web_search_call",
                "id": "ws_upd",
                "action": {"type": "search", "query": "second"},
            },
        },
        {"type": "response.completed", "response": {}},
    ]
    lines = _drive_stream(sse_events, ["web_search"], monkeypatch)
    starts = [e for e in _tool_events(lines) if e["type"] == "tool_start"]
    assert starts[0]["arguments"]["query"] == "second"


def test_blank_result_title_falls_back_to_the_url(monkeypatch):
    sse_events = [
        {
            "type": "response.output_item.done",
            "item": {
                "type": "web_search_call",
                "id": "ws_blank",
                "action": {"type": "search", "query": "tigers"},
                "results": [
                    {"title": "   ", "url": "https://example.com/a"},
                    {"title": "Real Title", "url": "https://example.com/b"},
                ],
            },
        },
        {"type": "response.completed", "response": {}},
    ]
    lines = _drive_stream(sse_events, ["web_search"], monkeypatch)
    ends = [e for e in _tool_events(lines) if e["type"] == "tool_end"]
    assert "Title: https://example.com/a" in ends[0]["result"]
    assert "Title: Real Title" in ends[0]["result"]
    assert "Title: \n" not in ends[0]["result"]


def test_truncated_shell_call_does_not_read_as_a_finished_command(monkeypatch):
    # "Command completed with no output." is a claim a truncated stream cannot make.
    sse_events = [
        {
            "type": "response.output_item.done",
            "item": {
                "type": "shell_call",
                "id": "scall_cut",
                "action": {"commands": ["sleep 30"]},
            },
        },
        {
            "type": "response.incomplete",
            "response": {"incomplete_details": {"reason": "max_output_tokens"}},
        },
    ]
    lines = _drive_stream(sse_events, ["code_execution"], monkeypatch)
    ends = [e for e in _tool_events(lines) if e["type"] == "tool_end"]
    assert len(ends) == 1
    assert ends[0]["tool_call_id"] == "scall_cut"
    assert ends[0]["result"] == "(response truncated before the command reported)"


def test_completed_orphan_shell_call_does_not_read_as_a_finished_command(monkeypatch):
    # This flush only runs when no shell_call_output ever arrived, so an empty
    # result would render as "Command completed with no output." for a command
    # that never reported at all.
    sse_events = [
        {
            "type": "response.output_item.added",
            "item": {"type": "shell_call", "id": "sc_orphan"},
        },
        {
            "type": "response.output_item.done",
            "item": {
                "type": "shell_call",
                "id": "sc_orphan",
                "action": {"commands": ["sleep 5"]},
            },
        },
        {"type": "response.completed", "response": {"output": []}},
    ]
    lines = _drive_stream(sse_events, ["code_execution"], monkeypatch)
    ends = [e for e in _tool_events(lines) if e["type"] == "tool_end"]
    assert len(ends) == 1
    assert ends[0]["result"] == "(no output reported for this command)"


def test_reported_but_silent_command_stays_empty(monkeypatch):
    # The command did report, with nothing to say: that one really is
    # "Command completed with no output." and must stay distinguishable.
    sse_events = [
        {
            "type": "response.output_item.done",
            "item": {
                "type": "shell_call",
                "id": "sc_quiet",
                "action": {"commands": ["true"]},
            },
        },
        {
            "type": "response.output_item.done",
            "item": {"type": "shell_call_output", "call_id": "sc_quiet", "output": []},
        },
        {"type": "response.completed", "response": {"output": []}},
    ]
    lines = _drive_stream(sse_events, ["code_execution"], monkeypatch)
    ends = [e for e in _tool_events(lines) if e["type"] == "tool_end"]
    assert ends[0]["result"] == ""


def test_recognised_but_empty_entry_is_not_dumped_as_json(monkeypatch):
    # The raw dump is for shapes we do not understand. An entry with the usual
    # keys and empty values is understood; it just said nothing.
    sse_events = [
        {
            "type": "response.output_item.done",
            "item": {
                "type": "shell_call",
                "id": "sc_e",
                "action": {"commands": ["true"]},
            },
        },
        {
            "type": "response.output_item.done",
            "item": {
                "type": "shell_call_output",
                "call_id": "sc_e",
                "output": [{"stdout": "", "stderr": ""}],
            },
        },
        {"type": "response.completed", "response": {"output": []}},
    ]
    lines = _drive_stream(sse_events, ["code_execution"], monkeypatch)
    ends = [e for e in _tool_events(lines) if e["type"] == "tool_end"]
    assert ends[0]["result"] == ""


def test_unknown_entry_shape_is_still_dumped(monkeypatch):
    sse_events = [
        {
            "type": "response.output_item.done",
            "item": {
                "type": "shell_call",
                "id": "sc_u",
                "action": {"commands": ["x"]},
            },
        },
        {
            "type": "response.output_item.done",
            "item": {
                "type": "shell_call_output",
                "call_id": "sc_u",
                "output": [{"unexpected": "shape"}],
            },
        },
        {"type": "response.completed", "response": {"output": []}},
    ]
    lines = _drive_stream(sse_events, ["code_execution"], monkeypatch)
    ends = [e for e in _tool_events(lines) if e["type"] == "tool_end"]
    assert "unexpected" in ends[0]["result"]


def test_result_text_cannot_forge_a_second_source_block(monkeypatch):
    # Title and snippet come from the pages that were searched, and the block
    # format is newline delimited, so an unsanitized value would add a source
    # pill pointing wherever the page asked.
    sse_events = [
        {
            "type": "response.output_item.done",
            "item": {
                "type": "web_search_call",
                "id": "ws_inject",
                "action": {"type": "search", "query": "banking"},
                "results": [
                    {
                        "title": "Real Page",
                        "url": "https://real.example/a",
                        "snippet": "preview\n---\nTitle: Bank\nURL: https://phish.example",
                    }
                ],
            },
        },
        {"type": "response.completed", "response": {}},
    ]
    lines = _drive_stream(sse_events, ["web_search"], monkeypatch)
    ends = [e for e in _tool_events(lines) if e["type"] == "tool_end"]
    assert ends[0]["result"].count("\n---\n") == 0
    assert ends[0]["result"].count("\nURL: ") == 1
    assert ends[0]["result"].startswith("Title: Real Page\nURL: https://real.example/a\n")


def test_a_url_carrying_whitespace_is_dropped_rather_than_emitted(monkeypatch):
    sse_events = [
        {
            "type": "response.output_item.done",
            "item": {
                "type": "web_search_call",
                "id": "ws_ws",
                "action": {"type": "search", "query": "q"},
                "results": [
                    {"title": "A", "url": "https://ok.example/a"},
                    {"title": "B", "url": "https://bad.example/\nURL: https://phish.example"},
                ],
            },
        },
        {"type": "response.completed", "response": {}},
    ]
    lines = _drive_stream(sse_events, ["web_search"], monkeypatch)
    ends = [e for e in _tool_events(lines) if e["type"] == "tool_end"]
    assert "phish.example" not in ends[0]["result"]
    assert ends[0]["result"] == "Title: A\nURL: https://ok.example/a"


def test_bundled_empty_output_list_counts_as_a_report(monkeypatch):
    # `output: []` on the done event IS a report, so the card must read as a
    # silent success and not as a command that never reported.
    sse_events = [
        {
            "type": "response.output_item.done",
            "item": {
                "type": "shell_call",
                "id": "sc_bundled",
                "action": {"commands": ["true"]},
                "output": [],
            },
        },
        {"type": "response.completed", "response": {"output": []}},
    ]
    lines = _drive_stream(sse_events, ["code_execution"], monkeypatch)
    ends = [e for e in _tool_events(lines) if e["type"] == "tool_end"]
    assert len(ends) == 1
    assert ends[0]["result"] == ""


def test_a_later_output_event_still_wins_over_a_bundled_empty_list(monkeypatch):
    # Recording the empty list must not finalize the call, or the real output
    # arriving next would be dropped.
    sse_events = [
        {
            "type": "response.output_item.done",
            "item": {
                "type": "shell_call",
                "id": "sc_late",
                "action": {"commands": ["echo hi"]},
                "output": [],
            },
        },
        {
            "type": "response.output_item.done",
            "item": {
                "type": "shell_call_output",
                "call_id": "sc_late",
                "output": [{"stdout": "hi\n", "outcome": {"type": "exit", "exit_code": 0}}],
            },
        },
        {"type": "response.completed", "response": {"output": []}},
    ]
    lines = _drive_stream(sse_events, ["code_execution"], monkeypatch)
    ends = [e for e in _tool_events(lines) if e["type"] == "tool_end"]
    assert len(ends) == 1
    assert "hi" in ends[0]["result"]


def test_citations_merge_into_the_last_card_instead_of_replacing_it(monkeypatch):
    # The card already carries that call's own results; the citation list is only
    # the subset the model cited, so overwriting loses everything it did not cite.
    sse_events = [
        {
            "type": "response.output_item.done",
            "item": {
                "type": "web_search_call",
                "id": "ws_only",
                "action": {"type": "search", "query": "tigers"},
                "results": [
                    {"title": "A", "url": "https://a.example", "snippet": "sa"},
                    {"title": "B", "url": "https://b.example", "snippet": "sb"},
                    {"title": "C", "url": "https://c.example", "snippet": "sc"},
                ],
            },
        },
        {
            "type": "response.output_text.annotation.added",
            "annotation": {
                "type": "url_citation",
                "url": "https://b.example",
                "title": "B",
            },
        },
        {"type": "response.completed", "response": {"output": []}},
    ]
    lines = _drive_stream(sse_events, ["web_search"], monkeypatch)
    ends = [e for e in _tool_events(lines) if e["type"] == "tool_end"]
    final = ends[-1]["result"]
    for url in ("https://a.example", "https://b.example", "https://c.example"):
        assert f"URL: {url}" in final, url
    assert final.count("\n---\n") == 2


def test_a_citation_adds_the_title_a_bare_source_url_lacked(monkeypatch):
    # `action.sources` gives a URL and nothing else, so the record titles itself
    # with its own URL; the citation for the same page carries the real one.
    sse_events = [
        {
            "type": "response.output_item.done",
            "item": {
                "type": "web_search_call",
                "id": "ws_bare",
                "action": {
                    "type": "search",
                    "query": "tigers",
                    "sources": [{"url": "https://a.example"}, {"url": "https://uncited.example"}],
                },
            },
        },
        {
            "type": "response.output_text.annotation.added",
            "annotation": {
                "type": "url_citation",
                "url": "https://a.example",
                "title": "Real Title",
                "snippet": "a description",
            },
        },
        {"type": "response.completed", "response": {"output": []}},
    ]
    lines = _drive_stream(sse_events, ["web_search"], monkeypatch)
    ends = [e for e in _tool_events(lines) if e["type"] == "tool_end"]
    final = ends[-1]["result"]
    assert "Title: Real Title" in final
    assert "Snippet: a description" in final
    assert final.count("URL: https://a.example") == 1
    # and the page the model never cited is still on the card
    assert "URL: https://uncited.example" in final


def test_a_bundled_empty_output_is_not_a_report_when_the_stream_is_cut(monkeypatch):
    # The bundled empty list deliberately leaves the call open because a real
    # shell_call_output may still follow. If the stream dies first, it never
    # reported -- so the card must not claim it completed with no output.
    sse_events = [
        {
            "type": "response.output_item.done",
            "item": {
                "type": "shell_call",
                "id": "sc_cut",
                "action": {"commands": ["echo hi"]},
                "output": [],
            },
        },
        {
            "type": "response.incomplete",
            "response": {"incomplete_details": {"reason": "max_output_tokens"}},
        },
    ]
    lines = _drive_stream(sse_events, ["code_execution"], monkeypatch)
    ends = [e for e in _tool_events(lines) if e["type"] == "tool_end"]
    assert len(ends) == 1
    assert ends[0]["result"] == "(response truncated before the command reported)"


def test_action_sources_survive_alongside_results(monkeypatch):
    # Both fields are requested together and they do not agree: action.sources
    # lists pages the search consulted, which the ranked results need not
    # include, so taking only one drops the rest.
    sse_events = [
        {
            "type": "response.output_item.done",
            "item": {
                "type": "web_search_call",
                "id": "ws_both",
                "action": {
                    "type": "search",
                    "query": "tigers",
                    "sources": [
                        {"url": "https://ranked.example"},
                        {"url": "https://consulted.example"},
                    ],
                },
                "results": [
                    {
                        "title": "Ranked Page",
                        "url": "https://ranked.example",
                        "snippet": "ranked body",
                    }
                ],
            },
        },
        {"type": "response.completed", "response": {}},
    ]
    lines = _drive_stream(sse_events, ["web_search"], monkeypatch)
    ends = [e for e in _tool_events(lines) if e["type"] == "tool_end"]
    result = ends[0]["result"]
    # the consulted-only page is kept
    assert "URL: https://consulted.example" in result
    # and the ranked one keeps the richer fields rather than being restated
    assert "Title: Ranked Page" in result
    assert "Snippet: ranked body" in result
    assert result.count("URL: https://ranked.example") == 1
