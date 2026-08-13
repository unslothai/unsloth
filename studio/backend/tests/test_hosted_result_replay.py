# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A tool the provider ran must survive into the next turn's prompt.

Hosted and local tools coexist in one turn: Gemini can return a code-execution
result while asking for a local ``web_search``, and OpenAI can generate an image
before requesting one. The hosted output reaches the client as its own
``_toolEvent`` frame, but the loop rebuilds the assistant message from the text
and tool calls it saw, so that output was absent from the conversation replayed
on the follow-up request. The model then answered from the local results alone,
having lost what it had just produced.

Studio's own tool events are written with a top-level ``type`` and never appear
as ``_toolEvent``, so the two sides stay distinguishable and local results are
not replayed twice.
"""

from __future__ import annotations

import asyncio
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

WEB = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
}


def _hosted_event(payload: dict) -> str:
    """A provider-side tool frame, shaped as external_provider emits it."""
    return "data: " + json.dumps(
        {
            "id": "chatcmpl-x",
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {}, "finish_reason": None}],
            "_toolEvent": payload,
        }
    )


def _call_line(call_id: str = "c1") -> str:
    return "data: " + json.dumps(
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": call_id,
                                "type": "function",
                                "function": {
                                    "name": "web_search",
                                    "arguments": '{"query": "x"}',
                                },
                            }
                        ]
                    },
                }
            ]
        }
    )


def _text(content: str) -> str:
    return "data: " + json.dumps({"choices": [{"index": 0, "delta": {"content": content}}]})


def _finish(reason: str = "tool_calls") -> str:
    return "data: " + json.dumps({"choices": [{"index": 0, "delta": {}, "finish_reason": reason}]})


class FakeTransport:
    heals_text_tool_calls = False
    # The OAI-compat transport sanitizes at ingress, so the loop must not strip
    # the provider's own _toolEvent frames on the way through.
    sanitizes_provider_frames = True

    def __init__(self, turns, *, max_turns = 20):
        self.turns = [list(turn) for turn in turns]
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
    calls: list[str] = []

    def _execute(name, arguments, **kwargs):
        calls.append(name)
        return f"LOCAL<{name}>"

    monkeypatch.setattr(loop_mod, "execute_tool", _execute)
    monkeypatch.setattr(loop_mod, "build_rag_autoinject", lambda *a, **k: None)
    monkeypatch.setattr(loop_mod, "is_high_risk_tool_call", lambda name, args: False)
    return calls


def _run(transport):
    async def _collect():
        out: list[str] = []
        agen = stream_with_studio_tools(
            transport,
            run = ToolLoopRun(
                messages = [{"role": "user", "content": "hi"}],
                session_id = "s1",
                thread_id = "t1",
            ),
            policy = ToolLoopPolicy(
                tools = [WEB],
                max_calls = 25,
                timeout = 300,
                permission_mode = "off",
                confirm_calls = False,
                bypass_permissions = False,
                rag_scope = None,
            ),
            cancel_event = threading.Event(),
        )
        async for line in agen:
            out.append(line)
        return out

    return asyncio.run(asyncio.wait_for(_collect(), timeout = 30))


def _replayed(transport) -> str:
    """Everything the second request carried, as one searchable blob."""
    assert len(transport.requests) > 1, "the loop never made a follow-up request"
    return json.dumps(transport.requests[1]["messages"])


# ── the gap ──────────────────────────────────────────────────────────


def test_a_hosted_result_reaches_the_follow_up_request(executed):
    """The provider searched, then asked us to run a tool. Both must survive."""
    transport = FakeTransport(
        [
            [
                _hosted_event(
                    {
                        "type": "tool_start",
                        "tool_name": "web_search",
                        "tool_call_id": "hosted-1",
                        "arguments": {},
                    }
                ),
                _hosted_event(
                    {
                        "type": "tool_end",
                        "tool_call_id": "hosted-1",
                        "tool_name": "web_search",
                        "result": "Title: Unsloth\nURL: https://unsloth.ai",
                    }
                ),
                _text("Let me also compute that."),
                _call_line(),
                _finish(),
            ],
            [_text("done"), _finish("stop")],
            [_DONE],
        ]
    )
    _run(transport)
    replayed = _replayed(transport)
    assert "https://unsloth.ai" in replayed, "the hosted result was dropped"
    assert "LOCAL<web_search>" in replayed, "the local result was dropped"


def test_the_hosted_result_still_reaches_the_client(executed):
    """Capturing it for the replay must not stop it rendering."""
    transport = FakeTransport(
        [
            [
                _hosted_event(
                    {
                        "type": "tool_end",
                        "tool_call_id": "hosted-1",
                        "tool_name": "web_search",
                        "result": "hosted output",
                    }
                ),
                _call_line(),
                _finish(),
            ],
            [_finish("stop")],
            [_DONE],
        ]
    )
    out = _run(transport)
    assert any("hosted output" in line and "_toolEvent" in line for line in out)


def test_the_models_own_prose_is_kept_alongside(executed):
    transport = FakeTransport(
        [
            [
                _text("Searching now."),
                _hosted_event(
                    {
                        "type": "tool_end",
                        "tool_call_id": "hosted-1",
                        "tool_name": "web_search",
                        "result": "hosted output",
                    }
                ),
                _call_line(),
                _finish(),
            ],
            [_finish("stop")],
            [_DONE],
        ]
    )
    _run(transport)
    replayed = _replayed(transport)
    assert "Searching now." in replayed
    assert "hosted output" in replayed


# ── what must not be replayed ────────────────────────────────────────


def test_a_repeated_end_event_is_recorded_once(executed):
    transport = FakeTransport(
        [
            [
                _hosted_event(
                    {
                        "type": "tool_end",
                        "tool_call_id": "hosted-1",
                        "tool_name": "web_search",
                        "result": "once only",
                    }
                ),
                _hosted_event(
                    {
                        "type": "tool_end",
                        "tool_call_id": "hosted-1",
                        "tool_name": "web_search",
                        "result": "once only",
                    }
                ),
                _call_line(),
                _finish(),
            ],
            [_finish("stop")],
            [_DONE],
        ]
    )
    _run(transport)
    assert _replayed(transport).count("once only") == 1


def test_a_start_event_alone_adds_nothing(executed):
    """A hosted tool that never reported a result has nothing to replay."""
    transport = FakeTransport(
        [
            [
                _hosted_event(
                    {
                        "type": "tool_start",
                        "tool_name": "web_search",
                        "tool_call_id": "hosted-1",
                        "arguments": {},
                    }
                ),
                _text("hello"),
                _call_line(),
                _finish(),
            ],
            [_finish("stop")],
            [_DONE],
        ]
    )
    _run(transport)
    replayed = _replayed(transport)
    assert "hello" in replayed
    assert "web_search result" not in replayed


@pytest.mark.parametrize("result", ["", "   ", None, 42, {"a": 1}])
def test_a_malformed_result_is_ignored(executed, result):
    """A provider is not trusted to send a usable string here."""
    transport = FakeTransport(
        [
            [
                _hosted_event(
                    {
                        "type": "tool_end",
                        "tool_call_id": "hosted-1",
                        "tool_name": "web_search",
                        "result": result,
                    }
                ),
                _text("hello"),
                _call_line(),
                _finish(),
            ],
            [_finish("stop")],
            [_DONE],
        ]
    )
    _run(transport)
    assert "[web_search result]" not in _replayed(transport)


def test_an_event_without_a_call_id_is_ignored(executed):
    transport = FakeTransport(
        [
            [
                _hosted_event(
                    {"type": "tool_end", "tool_name": "web_search", "result": "orphan"}
                ),
                _text("hello"),
                _call_line(),
                _finish(),
            ],
            [_finish("stop")],
            [_DONE],
        ]
    )
    _run(transport)
    assert "orphan" not in _replayed(transport)


def test_a_frontend_image_sentinel_is_not_replayed(executed):
    """__IMAGES__ carries a full data URI for the card, not for the model.

    Replaying it verbatim would put megabytes of base64 into the next request as
    assistant text, costing context and tokens for something the model cannot
    read. Local results already go through the same stripper.
    """
    huge = "data:image/png;base64," + ("A" * 20000)
    transport = FakeTransport(
        [
            [
                _hosted_event(
                    {
                        "type": "tool_end",
                        "tool_call_id": "hosted-1",
                        "tool_name": "code_execution",
                        "result": '4\n__IMAGES__:["' + huge + '"]',
                    }
                ),
                _call_line(),
                _finish(),
            ],
            [_finish("stop")],
            [_DONE],
        ]
    )
    _run(transport)
    replayed = _replayed(transport)
    assert "4" in replayed
    assert "__IMAGES__" not in replayed
    assert "AAAA" not in replayed


def test_the_start_events_operation_labels_the_result(executed):
    """The tool_end producers generally omit tool_name.

    For Gemini code execution the code that ran is only in the start event, so a
    result recorded alone replays as an unlabelled value the model cannot
    interpret.
    """
    transport = FakeTransport(
        [
            [
                _hosted_event(
                    {
                        "type": "tool_start",
                        "tool_name": "code_execution",
                        "tool_call_id": "hosted-1",
                        "arguments": {"language": "python", "code": "print(2 + 2)"},
                    }
                ),
                _hosted_event(
                    {"type": "tool_end", "tool_call_id": "hosted-1", "result": "4"},
                ),
                _call_line(),
                _finish(),
            ],
            [_finish("stop")],
            [_DONE],
        ]
    )
    _run(transport)
    replayed = _replayed(transport)
    assert "code_execution" in replayed, "the tool name came only from the start event"
    assert "print(2 + 2)" in replayed, "the operation that produced the result was lost"
    assert "4" in replayed


def test_a_generated_image_is_noted_without_its_bytes(executed):
    """image_generation reports an empty result and carries the picture apart.

    Requiring non-empty text dropped it entirely, but replaying the base64 is
    the same mistake as the sentinel, so record only that it happened.
    """
    transport = FakeTransport(
        [
            [
                _hosted_event(
                    {
                        "type": "tool_start",
                        "tool_name": "image_generation",
                        "tool_call_id": "hosted-1",
                        "arguments": {},
                    }
                ),
                _hosted_event(
                    {
                        "type": "tool_end",
                        "tool_call_id": "hosted-1",
                        "result": "",
                        "image_b64": "B" * 5000,
                        "image_mime": "image/png",
                    }
                ),
                _call_line(),
                _finish(),
            ],
            [_finish("stop")],
            [_DONE],
        ]
    )
    _run(transport)
    replayed = _replayed(transport)
    assert "image_generation" in replayed
    assert "produced an image" in replayed
    assert "BBBB" not in replayed, "the base64 must not reach the next request"


def test_a_turn_with_no_hosted_tool_replays_exactly_as_before(executed):
    """The common case must be untouched by any of this."""
    transport = FakeTransport(
        [[_text("plain answer"), _call_line(), _finish()], [_finish("stop")], [_DONE]]
    )
    _run(transport)
    replayed = _replayed(transport)
    assert "plain answer" in replayed
    assert "result]" not in replayed
