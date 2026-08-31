# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Two accumulation rules the external loop has to get right per provider.

1. Streamed tool-call names arrive in two dialects. llama-server re-sends the
   whole name as it grows, OpenAI sends fragments that continue it. Handling
   only one produces ``webweb_search`` or ``_search``, and either way the name
   fails the enabled-tool check and the call silently never runs.

2. Usage. The loop withholds the provider's usage chunks and emits one summed
   chunk at the end, so a usage block riding on a chunk that also carries a
   choice has to be stripped rather than relayed: the totals already include it,
   and a client that sums chunks would count the turn twice.
"""

from __future__ import annotations

import json
import threading

import pytest

from core.inference import studio_tool_loop as loop_mod
from core.inference.studio_tool_loop import (
    ToolLoopPolicy,
    ToolLoopRun,
    stream_with_studio_tools,
)


_DONE = "data: [DONE]"


def _tool(name: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    }


WEB = _tool("web_search")


def _name_fragment(
    index: int,
    fragment: str,
    call_id: str = "c1",
) -> str:
    """One delta carrying part of a tool call's name."""
    return "data: " + json.dumps(
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": index,
                                "id": call_id,
                                "type": "function",
                                "function": {"name": fragment},
                            }
                        ]
                    },
                }
            ]
        }
    )


def _arguments(
    index: int,
    chunk: str,
    call_id: str = "c1",
) -> str:
    return "data: " + json.dumps(
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": index,
                                "id": call_id,
                                "function": {"arguments": chunk},
                            }
                        ]
                    },
                }
            ]
        }
    )


def _finish(reason: str = "tool_calls") -> str:
    return "data: " + json.dumps({"choices": [{"index": 0, "delta": {}, "finish_reason": reason}]})


class FakeTransport:
    def __init__(
        self,
        turns,
        *,
        heals = False,
        max_turns = 20,
    ):
        self.turns = [list(turn) for turn in turns]
        self.heals_text_tool_calls = heals
        self.requests: list[dict] = []
        self.max_turns = max_turns

    def stream(self, *, messages, tools, tool_choice, cancel_event):
        self.requests.append({"messages": [dict(m) for m in messages]})
        assert len(self.requests) <= self.max_turns, "loop never terminated"
        lines = self.turns.pop(0) if self.turns else [_DONE]

        async def _gen():
            for line in lines:
                yield line

        return _gen()


@pytest.fixture
def executed(monkeypatch):
    calls: list[dict] = []

    def _execute(name, arguments, **kwargs):
        calls.append({"name": name, "arguments": arguments})
        return f"RESULT<{name}>"

    monkeypatch.setattr(loop_mod, "execute_tool", _execute)
    monkeypatch.setattr(loop_mod, "build_rag_autoinject", lambda *a, **k: None)
    monkeypatch.setattr(loop_mod, "is_high_risk_tool_call", lambda name, args: False)
    return calls


def _run(transport, **policy_kwargs):
    import asyncio

    fields = {
        "tools": [WEB],
        "max_calls": 25,
        "timeout": 300,
        "permission_mode": "off",
        "confirm_calls": False,
        "bypass_permissions": False,
        "rag_scope": None,
    }
    fields.update(policy_kwargs)

    async def _collect():
        out: list[str] = []
        agen = stream_with_studio_tools(
            transport,
            run = ToolLoopRun(
                messages = [{"role": "user", "content": "hi"}],
                session_id = "s1",
                thread_id = "t1",
                model = "asked-for-model",
            ),
            policy = ToolLoopPolicy(**fields),
            cancel_event = threading.Event(),
        )
        async for line in agen:
            out.append(line)
        return out

    return asyncio.new_event_loop().run_until_complete(_collect())


# ── 1. the two streamed-name dialects ────────────────────────────────


def test_a_cumulative_name_is_not_doubled(executed):
    """llama-server resends the whole name: "web" then "web_search"."""
    transport = FakeTransport(
        [
            [
                _name_fragment(0, "web"),
                _name_fragment(0, "web_search"),
                _arguments(0, '{"query": "x"}'),
                _finish(),
            ],
            [_DONE],
        ]
    )
    _run(transport)
    assert [call["name"] for call in executed] == ["web_search"]


def test_an_incremental_name_is_joined(executed):
    """OpenAI sends fragments: "web" then "_search".

    Assignment would leave "_search", which is not a selected tool, so the call
    is refused and the user sees nothing run.
    """
    transport = FakeTransport(
        [
            [
                _name_fragment(0, "web"),
                _name_fragment(0, "_search"),
                _arguments(0, '{"query": "x"}'),
                _finish(),
            ],
            [_DONE],
        ]
    )
    _run(transport)
    assert [call["name"] for call in executed] == ["web_search"]


def test_a_single_whole_name_still_runs(executed):
    """The common case: one delta carrying the entire name."""
    transport = FakeTransport(
        [
            [
                _name_fragment(0, "web_search"),
                _arguments(0, '{"query": "x"}'),
                _finish(),
            ],
            [_DONE],
        ]
    )
    _run(transport)
    assert [call["name"] for call in executed] == ["web_search"]


def test_a_name_arriving_one_character_at_a_time_is_joined(executed):
    """The degenerate incremental case, which must still reassemble."""
    transport = FakeTransport(
        [
            [_name_fragment(0, char) for char in "web_search"]
            + [_arguments(0, '{"query": "x"}'), _finish()],
            [_DONE],
        ]
    )
    _run(transport)
    assert [call["name"] for call in executed] == ["web_search"]


# ── 2. usage is counted once ─────────────────────────────────────────


def _usage_chunks(lines: list[str]) -> list[dict]:
    found = []
    for line in lines:
        if not line.startswith("data:"):
            continue
        raw = line[len("data:") :].strip()
        if not raw or raw == "[DONE]":
            continue
        try:
            payload = json.loads(raw)
        except ValueError:
            continue
        if isinstance(payload, dict) and payload.get("usage"):
            found.append(payload)
    return found


def test_usage_riding_on_a_content_chunk_is_counted_once(executed):
    """A chunk carrying both a choice and usage must keep only the choice.

    The loop's summed chunk already includes those tokens, so relaying them here
    makes a client that adds up chunks report the turn twice.
    """
    content_with_usage = "data: " + json.dumps(
        {
            "choices": [{"index": 0, "delta": {"content": "hello"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
    )
    transport = FakeTransport([[content_with_usage, _finish("stop")], [_DONE]])
    out = _run(transport)

    # The content survived.
    assert any('"hello"' in line for line in out)
    # Exactly one usage report, and it is the loop's own summed chunk.
    reports = _usage_chunks(out)
    assert len(reports) == 1, reports
    assert reports[0]["choices"] == []
    assert reports[0]["usage"]["total_tokens"] == 15


def test_usage_totals_still_sum_across_turns(executed):
    """Stripping the relayed copy must not stop it being counted."""
    turn_one = "data: " + json.dumps(
        {
            "choices": [{"index": 0, "delta": {"content": "a"}}],
            "usage": {"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5},
        }
    )
    transport = FakeTransport(
        [
            [
                _name_fragment(0, "web_search"),
                _arguments(0, '{"query": "x"}'),
                _finish(),
            ],
            [turn_one, _finish("stop")],
            [_DONE],
        ]
    )
    out = _run(transport)
    reports = _usage_chunks(out)
    assert len(reports) == 1
    assert reports[0]["usage"]["total_tokens"] == 5
