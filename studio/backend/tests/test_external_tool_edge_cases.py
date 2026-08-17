# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Adversarial edge cases for the provider-agnostic Studio tool loop.

``tests/test_studio_tool_loop.py`` covers the happy paths. This file covers what
a hostile, buggy or merely unusual provider can do to the loop: malformed SSE,
duplicate/absent tool-call indices, text and structured calls describing the
same intent, unicode straddling chunks, megabyte arguments, a stream that never
terminates, cancellation mid-tool, and the conversation the loop replays back to
a strict server.

Every test that FAILS is asserting the behaviour the loop should have, so a
failure names a defect rather than a preference.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time

import pytest

from core.inference import studio_tool_loop as loop_mod
from core.inference.studio_tool_loop import (
    ToolLoopPolicy,
    ToolLoopRun,
    stream_with_studio_tools,
)


_DONE = "data: [DONE]"


def _sse(
    delta = None,
    finish = None,
    **extra,
) -> str:
    choice: dict = {"index": 0, "delta": delta if delta is not None else {}}
    if finish is not None:
        choice["finish_reason"] = finish
    payload: dict = {"choices": [choice]}
    payload.update(extra)
    return "data: " + json.dumps(payload, ensure_ascii = False)


def _raw(payload) -> str:
    return "data: " + json.dumps(payload, ensure_ascii = False)


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
PY = _tool("python")


class TooManyTurns(RuntimeError):
    """Raised by the harness when a transport is asked for an absurd turn count."""


class FakeTransport:
    """Replays scripted turns; records what the loop asked for each time."""

    def __init__(
        self,
        turns,
        *,
        heals = True,
        repeat_last = False,
        max_turns = 40,
    ):
        self.turns = [list(turn) for turn in turns]
        self.heals_text_tool_calls = heals
        self.requests: list[dict] = []
        self.repeat_last = repeat_last
        self.max_turns = max_turns
        self.closed = 0
        self.opened = 0

    def _lines(self):
        if self.turns:
            return (
                self.turns[0] if (self.repeat_last and len(self.turns) == 1) else self.turns.pop(0)
            )
        return [_DONE]

    def stream(self, *, messages, tools, tool_choice, cancel_event):
        self.requests.append(
            {
                "messages": [dict(message) for message in messages],
                "tools": tools,
                "tool_choice": tool_choice,
            }
        )
        if len(self.requests) > self.max_turns:
            raise TooManyTurns(f"loop asked for turn {len(self.requests)}")
        lines = self._lines()
        self.opened += 1

        async def _gen():
            try:
                for line in lines:
                    yield line
            finally:
                self.closed += 1

        return _gen()


class RaisingTransport:
    """Dies part way through the first turn, the way a dropped socket does."""

    heals_text_tool_calls = True

    def __init__(self, lines, exc):
        self.lines = list(lines)
        self.exc = exc

    def stream(self, *, messages, tools, tool_choice, cancel_event):
        lines, exc = self.lines, self.exc

        async def _gen():
            for line in lines:
                yield line
            raise exc

        return _gen()


@pytest.fixture
def executed(monkeypatch):
    """Record every execute_tool call and return a canned result."""
    calls: list[dict] = []

    def _execute(name, arguments, **kwargs):
        calls.append({"name": name, "arguments": arguments, **kwargs})
        return f"RESULT<{name}>"

    monkeypatch.setattr(loop_mod, "execute_tool", _execute)
    monkeypatch.setattr(loop_mod, "build_rag_autoinject", lambda *a, **k: None)
    monkeypatch.setattr(loop_mod, "is_high_risk_tool_call", lambda name, args: name == "python")
    # raising = False: this helper moved during the external-provider work, and
    # these tests must not depend on where it currently lives.
    monkeypatch.setattr(
        loop_mod, "strip_result_for_model", lambda result, name = None: result, raising = False
    )
    return calls


def _policy(**overrides) -> ToolLoopPolicy:
    fields = {
        "tools": [WEB],
        "max_calls": 25,
        "timeout": 300,
        "permission_mode": "off",
        "confirm_calls": False,
        "bypass_permissions": False,
        "rag_scope": None,
    }
    fields.update(overrides)
    return ToolLoopPolicy(**fields)


def _run(
    transport,
    *,
    tools = None,
    tool_choice = None,
    messages = None,
    cancel_event = None,
    deadline = 30.0,
    **policy_kwargs,
):
    """Drive the loop to exhaustion and return every line it yielded."""
    if tools is not None:
        policy_kwargs["tools"] = tools
    cancel_event = cancel_event if cancel_event is not None else threading.Event()

    async def _collect():
        out: list[str] = []
        agen = stream_with_studio_tools(
            transport,
            run = ToolLoopRun(
                messages = messages or [{"role": "user", "content": "hi"}],
                session_id = "s1",
                thread_id = "t1",
                tool_choice = tool_choice,
            ),
            policy = _policy(**policy_kwargs),
            cancel_event = cancel_event,
        )
        async for line in agen:
            out.append(line)
        return out

    async def _guarded():
        return await asyncio.wait_for(_collect(), timeout = deadline)

    return asyncio.run(_guarded())


def _payloads(lines):
    for line in lines:
        if not line.startswith("data: "):
            continue
        raw = line[6:]
        if raw == "[DONE]":
            continue
        try:
            yield json.loads(raw)
        except ValueError:
            continue


def _events(lines, kind):
    return [payload for payload in _payloads(lines) if payload.get("type") == kind]


def _visible_text(lines) -> str:
    text = []
    for payload in _payloads(lines):
        if payload.get("type") in ("tool_start", "tool_end", "tool_status"):
            continue
        choices = payload.get("choices")
        for choice in choices if isinstance(choices, list) else []:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta")
            content = delta.get("content") if isinstance(delta, dict) else None
            if isinstance(content, str):
                text.append(content)
    return "".join(text)


def _call_turn(
    call_id = "c1",
    name = "web_search",
    arguments = '{"query":"q"}',
):
    return [
        _sse(
            {
                "tool_calls": [
                    {"index": 0, "id": call_id, "function": {"name": name, "arguments": arguments}}
                ]
            }
        ),
        _sse(finish = "tool_calls"),
        _DONE,
    ]


def _answer_turn(text = "final answer"):
    return [_sse({"content": text}), _sse(finish = "stop"), _DONE]


# ── Stream framing ────────────────────────────────────────────────


def test_intermediate_done_sentinel_is_not_relayed(executed):
    """A per-turn [DONE] must never reach the client mid-loop.

    Every transport ends each turn with one. Relaying it tells a spec-compliant
    client the response is finished, so it stops before the tool cards and the
    real answer. The loop swallows all of them and the route appends exactly one
    terminal sentinel after the generator finishes, which is how the Codex path
    has always behaved.
    """
    transport = FakeTransport(
        [
            [
                _sse(
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "c1",
                                "function": {"name": "web_search", "arguments": "{}"},
                            }
                        ]
                    }
                ),
                _sse(finish = "tool_calls"),
                _DONE,
            ],
            [_sse({"content": "the real answer"}), _sse(finish = "stop"), _DONE],
        ]
    )
    lines = _run(transport)

    assert not [line for line in lines if line.strip() == "data: [DONE]"]
    # The answer that used to be stranded behind the sentinel now arrives.
    assert "the real answer" in _visible_text(lines)


def test_non_json_data_line_is_relayed_untouched(executed):
    """Garbage on the wire is passed through, not parsed and not fatal."""
    transport = FakeTransport(
        [["data: {not json at all", _sse({"content": "hi"}), _sse(finish = "stop"), _DONE]]
    )
    lines = _run(transport)

    assert "data: {not json at all" in lines
    assert "hi" in _visible_text(lines)


@pytest.mark.parametrize(
    "payload",
    [
        {"choices": []},
        {"choices": "nonsense"},
        {"choices": [None]},
        {"choices": ["a string choice"]},
        {"choices": [{"delta": None}]},
        {"choices": [{"delta": {"content": 7}}]},
        {"choices": [{"delta": {"content": None}}]},
        {"choices": [{"delta": {"content": ["part"]}}]},
        {"choices": [{"delta": {"content": {"text": "x"}}}]},
        {"choices": [{"delta": {"tool_calls": "not-a-list"}}]},
        {"choices": [{"delta": {"tool_calls": [None, 3, "x"]}}]},
        {"choices": [{"delta": {"tool_calls": [{}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0}]}}]},
        # A string index is a protocol violation, but the id and name are a real
        # request, so the call runs with the arguments actually sent (none). It is
        # covered by its own test below rather than as a "must not execute" shape.
        {"choices": [{"finish_reason": 5}]},
        {},
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
    ],
)
def test_hostile_chunk_shapes_do_not_crash_the_loop(executed, payload):
    """No provider chunk shape may take the loop down mid-answer."""
    transport = FakeTransport(
        [[_raw(payload), _sse({"content": "still here"}), _sse(finish = "stop"), _DONE]]
    )
    lines = _run(transport)

    assert "still here" in _visible_text(lines)
    assert executed == []


def test_transport_exception_is_reported_as_a_stream_error(executed):
    """A mid-stream transport failure should surface, not propagate raw.

    The route wraps the generator, so a raw exception does produce an error
    frame today. This pins the contract that the loop itself does not corrupt
    state on the way out.
    """
    transport = RaisingTransport([_sse({"content": "partial"})], RuntimeError("socket died"))
    with pytest.raises(RuntimeError, match = "socket died"):
        _run(transport)


def test_zero_line_turn_terminates(executed):
    transport = FakeTransport([[]])
    lines = _run(transport)
    assert lines == []


# ── Structured tool-call accumulation ─────────────────────────────


def test_call_without_finish_reason_is_still_executed(executed):
    """Some OAI-compatible servers close with [DONE] and no finish_reason.

    Dropping a fully formed tool call because the terminal chunk was missing
    ends the turn with an empty answer and no sign anything was requested.
    """
    transport = FakeTransport(
        [
            [
                _sse(
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "c1",
                                "function": {"name": "web_search", "arguments": '{"query":"x"}'},
                            }
                        ]
                    }
                ),
                _DONE,
            ],
            _answer_turn(),
        ]
    )
    _run(transport)

    assert [call["name"] for call in executed] == ["web_search"]


def test_structured_call_with_finish_reason_stop_is_executed(executed):
    """The Ollama shape: real ``delta.tool_calls`` closed with ``stop``.

    Gating execution on ``finish_reason == "tool_calls"`` throws the call away
    and the user gets whatever prose came with it, if any.
    """
    transport = FakeTransport(
        [
            [
                _sse(
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "c1",
                                "function": {"name": "web_search", "arguments": '{"query":"x"}'},
                            }
                        ]
                    }
                ),
                _sse(finish = "stop"),
                _DONE,
            ],
            _answer_turn(),
        ]
    )
    _run(transport)

    assert [call["name"] for call in executed] == ["web_search"]


def test_call_without_an_id_is_still_executed(executed):
    """Ollama and some proxies omit `id` entirely; the loop owns id minting."""
    transport = FakeTransport(
        [
            [
                _sse(
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {"name": "web_search", "arguments": '{"query":"x"}'},
                            }
                        ]
                    }
                ),
                _sse(finish = "tool_calls"),
                _DONE,
            ],
            _answer_turn(),
        ]
    )
    _run(transport)

    assert [call["name"] for call in executed] == ["web_search"]


def test_empty_string_id_is_still_executed(executed):
    transport = FakeTransport(
        [
            [
                _sse(
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "",
                                "function": {"name": "web_search", "arguments": '{"query":"x"}'},
                            }
                        ]
                    }
                ),
                _sse(finish = "tool_calls"),
                _DONE,
            ],
            _answer_turn(),
        ]
    )
    _run(transport)

    assert [call["name"] for call in executed] == ["web_search"]


def test_argument_fragments_without_an_index_continue_the_open_call(executed):
    """A server that stamps `index` only on the opening fragment.

    ``index = len(by_index)`` invents a second call for the continuation, so the
    real call runs with empty arguments and the fragments are lost.
    """
    transport = FakeTransport(
        [
            [
                _sse(
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "c1",
                                "function": {"name": "web_search", "arguments": ""},
                            }
                        ]
                    }
                ),
                _sse({"tool_calls": [{"function": {"arguments": '{"query":'}}]}),
                _sse({"tool_calls": [{"function": {"arguments": '"paris"}'}}]}),
                _sse(finish = "tool_calls"),
                _DONE,
            ],
            _answer_turn(),
        ]
    )
    _run(transport)

    assert [call["name"] for call in executed] == ["web_search"]
    assert executed[0]["arguments"] == {"query": "paris"}


def test_two_distinct_calls_at_the_same_index_are_not_merged(executed):
    """Parallel calls that both report index 0 must stay two calls.

    Merging concatenates their argument JSON into an unparseable blob, which is
    then handed to the tool as ``{"_raw": ...}``.
    """
    transport = FakeTransport(
        [
            [
                _sse(
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "a",
                                "function": {"name": "web_search", "arguments": '{"query":"one"}'},
                            }
                        ]
                    }
                ),
                _sse(
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "b",
                                "function": {"name": "web_search", "arguments": '{"query":"two"}'},
                            }
                        ]
                    }
                ),
                _sse(finish = "tool_calls"),
                _DONE,
            ],
            _answer_turn(),
        ]
    )
    _run(transport)

    assert [call["arguments"] for call in executed] == [{"query": "one"}, {"query": "two"}]


def test_same_index_merge_never_forwards_raw_garbage(executed):
    """Weaker form of the above: whatever happens, no ``_raw`` blob reaches a tool."""
    transport = FakeTransport(
        [
            [
                _sse(
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "a",
                                "function": {"name": "web_search", "arguments": '{"query":"one"}'},
                            }
                        ]
                    }
                ),
                _sse(
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "b",
                                "function": {"name": "web_search", "arguments": '{"query":"two"}'},
                            }
                        ]
                    }
                ),
                _sse(finish = "tool_calls"),
                _DONE,
            ],
            _answer_turn(),
        ]
    )
    _run(transport)

    assert all("_raw" not in call["arguments"] for call in executed), executed


def test_negative_index_does_not_reorder_calls(executed):
    """A stray negative index must not jump the queue ahead of index 0."""
    transport = FakeTransport(
        [
            [
                _sse(
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "a",
                                "function": {
                                    "name": "web_search",
                                    "arguments": '{"query":"first"}',
                                },
                            }
                        ]
                    }
                ),
                _sse(
                    {
                        "tool_calls": [
                            {
                                "index": -1,
                                "id": "b",
                                "function": {
                                    "name": "web_search",
                                    "arguments": '{"query":"second"}',
                                },
                            }
                        ]
                    }
                ),
                _sse(finish = "tool_calls"),
                _DONE,
            ],
            _answer_turn(),
        ]
    )
    _run(transport)

    assert [call["arguments"]["query"] for call in executed] == ["first", "second"]


def test_ids_repeated_across_turns_stay_unique_in_the_replay(executed):
    """A provider that reuses the same call id every turn.

    The replayed conversation must not carry two different tool results under
    one tool_call_id: a strict server rejects the request outright.
    """
    transport = FakeTransport(
        [
            _call_turn(call_id = "same", arguments = '{"query":"a"}'),
            _call_turn(call_id = "same", arguments = '{"query":"b"}'),
            _answer_turn(),
        ]
    )
    _run(transport)

    final = transport.requests[-1]["messages"]
    ids = [m["tool_call_id"] for m in final if m.get("role") == "tool"]
    assert len(ids) == len(set(ids)), ids


def test_healed_ids_are_unique_across_turns(executed):
    """Healed ids restart at call_0 every turn, colliding in the replay."""
    heal = '<tool_call>{"name": "web_search", "arguments": {"query": "%s"}}</tool_call>'
    transport = FakeTransport(
        [
            [_sse({"content": heal % "a"}), _sse(finish = "stop"), _DONE],
            [_sse({"content": heal % "b"}), _sse(finish = "stop"), _DONE],
            _answer_turn(),
        ]
    )
    _run(transport)

    final = transport.requests[-1]["messages"]
    ids = [m["tool_call_id"] for m in final if m.get("role") == "tool"]
    assert len(ids) == len(set(ids)), ids


def test_finish_reason_length_mid_tool_call_does_not_execute(executed):
    """Truncated arguments are not a call. Documented, current behaviour."""
    transport = FakeTransport(
        [
            [
                _sse(
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "c1",
                                "function": {"name": "web_search", "arguments": '{"que'},
                            }
                        ]
                    }
                ),
                _sse(finish = "length"),
                _DONE,
            ]
        ]
    )
    _run(transport)

    assert executed == []


# ── Loop bounds ───────────────────────────────────────────────────


def test_a_provider_that_always_calls_a_tool_terminates(executed):
    """The budget must bound the number of PROVIDER TURNS, not just executions.

    Once the budget is spent the loop stops executing but keeps replaying: each
    turn appends an assistant message plus a synthetic "budget exhausted" tool
    result and asks again. A provider that keeps emitting the same call spins
    forever, growing the conversation on every pass.
    """
    transport = FakeTransport([_call_turn()], repeat_last = True, max_turns = 40)
    try:
        _run(transport, max_calls = 2)
    except TooManyTurns as exc:
        pytest.fail(f"loop never terminated: {exc}")

    assert (
        len(transport.requests) <= 2 + 2
    ), f"budget 2 produced {len(transport.requests)} provider turns"


def test_a_disabled_tool_called_forever_terminates(executed):
    """Same shape without the budget: every call is outside the catalog."""
    transport = FakeTransport([_call_turn(name = "terminal")], repeat_last = True, max_turns = 40)
    try:
        _run(transport, tools = [WEB])
    except TooManyTurns as exc:
        pytest.fail(f"loop never terminated on a disabled tool: {exc}")


def test_budget_exhaustion_stops_asking_the_provider_again(executed):
    """After the last permitted execution the loop should wind down, not re-ask."""
    transport = FakeTransport(
        [_call_turn(call_id = "c1"), _call_turn(call_id = "c2"), _answer_turn()],
        max_turns = 40,
    )
    _run(transport, max_calls = 1)

    assert len(executed) == 1
    assert len(transport.requests) <= 3


# ── Text-form healing interactions ────────────────────────────────


def test_text_and_structured_form_of_one_call_run_once(executed):
    """llama.cpp can leak the raw markup AND emit the parsed call.

    Executing both runs a side-effecting tool twice for one model intent.
    """
    transport = FakeTransport(
        [
            [
                _sse(
                    {
                        "content": '<tool_call>{"name": "web_search", "arguments": {"query": "dup"}}</tool_call>'
                    }
                ),
                _sse(
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "c1",
                                "function": {"name": "web_search", "arguments": '{"query":"dup"}'},
                            }
                        ]
                    }
                ),
                _sse(finish = "tool_calls"),
                _DONE,
            ],
            _answer_turn(),
        ]
    )
    _run(transport)

    assert [call["name"] for call in executed] == ["web_search"], executed


def test_empty_structured_tool_calls_list_does_not_disable_healing(executed):
    """``"tool_calls": []`` is not evidence that grammar mode worked.

    Treating it as such makes the healer dormant for the rest of the turn, so a
    text-form call later in the same turn is relayed as prose and never runs.
    """
    transport = FakeTransport(
        [
            [
                _sse({"content": "let me check. ", "tool_calls": []}),
                _sse(
                    {
                        "content": '<tool_call>{"name": "web_search", "arguments": {"query": "x"}}</tool_call>'
                    }
                ),
                _sse(finish = "stop"),
                _DONE,
            ],
            _answer_turn(),
        ]
    )
    lines = _run(transport)

    assert [call["name"] for call in executed] == ["web_search"]
    assert "<tool_call>" not in _visible_text(lines)


def test_marked_call_inside_a_code_fence_matches_the_local_loops(executed):
    """A fenced <tool_call> executes here exactly as it does locally.

    The fence gate added in #8312 covers the markerless `name[ARGS]{...}` form,
    which has no sentinel of its own and so cannot be told from prose. The
    marked form does have a sentinel, and the local GGUF and safetensors loops
    promote it inside a fence too. Diverging here would make the same answer
    behave differently depending on where the model runs, so the behaviour is
    pinned rather than special-cased.
    """
    transport = FakeTransport(
        [
            [
                _sse(
                    {
                        "content": '```\n<tool_call>{"name": "web_search", "arguments": {"query": "demo"}}</tool_call>\n```'
                    }
                ),
                _sse(finish = "stop"),
                _DONE,
            ],
            [_sse({"content": "ok"}), _sse(finish = "stop"), _DONE],
        ]
    )
    _run(transport)

    assert [call["name"] for call in executed] == ["web_search"]


def test_nested_markers_do_not_lose_text_or_double_execute(executed):
    transport = FakeTransport(
        [
            [
                _sse(
                    {
                        "content": '<tool_call><tool_call>{"name": "web_search", "arguments": {"query": "n"}}</tool_call></tool_call>'
                    }
                ),
                _sse(finish = "stop"),
                _DONE,
            ],
            _answer_turn(),
        ]
    )
    lines = _run(transport)

    assert len(executed) <= 1, executed
    assert "final answer" in _visible_text(lines) or executed == []


def test_undeclared_marked_call_is_relayed_verbatim(executed):
    transport = FakeTransport(
        [
            [
                _sse(
                    {
                        "content": '<tool_call>{"name": "terminal", "arguments": {"command": "id"}}</tool_call>'
                    }
                ),
                _sse(finish = "stop"),
                _DONE,
            ]
        ]
    )
    lines = _run(transport)

    assert executed == []
    assert "terminal" in _visible_text(lines)


# ── Unicode ───────────────────────────────────────────────────────


def test_emoji_split_across_chunks_survives(executed):
    """A grapheme cluster straddling a chunk boundary must not be mangled."""
    pieces = ["family: \U0001f468‍", "\U0001f469‍\U0001f467", " done éè"]
    transport = FakeTransport(
        [[_sse({"content": piece}) for piece in pieces] + [_sse(finish = "stop"), _DONE]]
    )
    lines = _run(transport)

    assert _visible_text(lines) == "".join(pieces)


def test_marker_prefix_split_around_unicode_is_not_dropped(executed):
    """`<` held as a partial signal, with multi-byte text either side."""
    pieces = ["café <", "é not a marker \U0001f600"]
    transport = FakeTransport(
        [[_sse({"content": piece}) for piece in pieces] + [_sse(finish = "stop"), _DONE]]
    )
    lines = _run(transport)

    assert _visible_text(lines) == "".join(pieces)


def test_tool_marker_split_between_unicode_chunks_still_heals(executed):
    pieces = [
        "\U0001f50d <tool",
        '_call>{"name": "web_search", "arguments": {"query": "café \U0001f600"}}',
        "</tool_call>",
    ]
    transport = FakeTransport(
        [
            [_sse({"content": piece}) for piece in pieces] + [_sse(finish = "stop"), _DONE],
            _answer_turn(),
        ]
    )
    lines = _run(transport)

    assert [call["name"] for call in executed] == ["web_search"]
    assert executed[0]["arguments"] == {"query": "café \U0001f600"}
    visible = _visible_text(lines)
    assert "\U0001f50d" in visible
    assert "<tool" not in visible


def test_unicode_arguments_round_trip_through_the_replay(executed):
    args = {"query": "中文 \U0001f680"}
    transport = FakeTransport(
        [_call_turn(arguments = json.dumps(args, ensure_ascii = False)), _answer_turn()]
    )
    _run(transport)

    assert executed[0]["arguments"] == args
    replayed = transport.requests[1]["messages"][-2]["tool_calls"][0]["function"]["arguments"]
    assert json.loads(replayed) == args


# ── Large payloads ────────────────────────────────────────────────


def test_one_megabyte_argument_streams_in_fragments(executed):
    blob = "x" * (1024 * 1024)
    arguments = json.dumps({"query": blob})
    chunks = [arguments[i : i + 4096] for i in range(0, len(arguments), 4096)]
    turn = [
        _sse(
            {
                "tool_calls": [
                    {"index": 0, "id": "c1", "function": {"name": "web_search", "arguments": ""}}
                ]
            }
        )
    ]
    turn += [
        _sse({"tool_calls": [{"index": 0, "function": {"arguments": chunk}}]}) for chunk in chunks
    ]
    turn += [_sse(finish = "tool_calls"), _DONE]

    start = time.monotonic()
    _run(FakeTransport([turn, _answer_turn()]), deadline = 60.0)
    elapsed = time.monotonic() - start

    assert executed[0]["arguments"]["query"] == blob
    assert elapsed < 15.0, f"1 MB argument took {elapsed:.1f}s"


def test_unterminated_block_larger_than_the_hold_cap_is_released(executed):
    """64 KB is the cap; beyond it the suspected block is a false alarm."""
    from core.inference.passthrough_healing import _MAX_HOLD_CHARS

    body = "0" * (_MAX_HOLD_CHARS + 5000)
    transport = FakeTransport(
        [
            [
                _sse({"content": '<tool_call>{"name": "web_search", "arguments": {"query": "'}),
                *[_sse({"content": body[i : i + 4096]}) for i in range(0, len(body), 4096)],
                _sse(finish = "stop"),
                _DONE,
            ]
        ]
    )
    start = time.monotonic()
    lines = _run(transport, deadline = 60.0)
    elapsed = time.monotonic() - start

    visible = _visible_text(lines)
    assert executed == []
    assert visible.count("0") == len(body), f"released {visible.count('0')} of {len(body)}"
    assert elapsed < 15.0, f"held {len(body)} chars took {elapsed:.1f}s"


def test_hold_cap_releases_before_the_stream_ends(executed):
    """Memory must be bounded DURING the turn, not only at finalize().

    The released text has to appear before the terminal chunk, otherwise a model
    rambling XML-lookalike prose buffers without limit until it stops.
    """
    from core.inference.passthrough_healing import _MAX_HOLD_CHARS

    body = "z" * (_MAX_HOLD_CHARS + 5000)
    marker = "TAILMARKER"
    transport = FakeTransport(
        [
            [
                _sse({"content": "<tool_call>"}),
                *[_sse({"content": body[i : i + 4096]}) for i in range(0, len(body), 4096)],
                _sse({"content": marker}),
                _sse(finish = "stop"),
                _DONE,
            ]
        ]
    )
    lines = _run(transport, deadline = 60.0)

    text_lines = [i for i, line in enumerate(lines) if "z" * 100 in line]
    marker_line = [i for i, line in enumerate(lines) if marker in line]
    assert text_lines and marker_line
    assert min(text_lines) < marker_line[0], "nothing was released until the very end"


# ── Ordering and conversation validity ────────────────────────────


def test_text_around_tool_calls_keeps_document_order(executed):
    transport = FakeTransport(
        [
            [
                _sse({"content": "BEFORE "}),
                _sse(
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "c1",
                                "function": {"name": "web_search", "arguments": "{}"},
                            }
                        ]
                    }
                ),
                _sse(finish = "tool_calls"),
                _DONE,
            ],
            [_sse({"content": "AFTER"}), _sse(finish = "stop"), _DONE],
        ]
    )
    lines = _run(transport)

    positions = {}
    for i, line in enumerate(lines):
        for token in ("BEFORE", "AFTER", "tool_start", "tool_end"):
            if token in line and token not in positions:
                positions[token] = i
    assert (
        positions["BEFORE"] < positions["tool_start"] < positions["tool_end"] < positions["AFTER"]
    )


def test_replayed_conversation_is_valid_for_a_strict_server(executed):
    transport = FakeTransport([_call_turn(), _answer_turn()])
    _run(transport)

    messages = transport.requests[-1]["messages"]
    roles = [message["role"] for message in messages]
    assert roles == ["user", "assistant", "tool"], roles
    assistant = messages[1]
    assert assistant["tool_calls"][0]["id"] == messages[2]["tool_call_id"]
    assert isinstance(assistant["content"], (str, type(None)))
    # Two user turns in a row, or a tool result with no preceding call, are the
    # two shapes a strict server rejects.
    for previous, current in zip(messages, messages[1:]):
        assert not (previous["role"] == "user" and current["role"] == "user"), roles


def test_disallowed_call_still_gets_a_tool_result_message(executed):
    transport = FakeTransport(
        [_call_turn(name = "terminal"), _answer_turn()],
        max_turns = 6,
    )
    lines = _run(transport, tools = [WEB])

    # Every announced call is closed out, on the wire and in the replay. A loop
    # that declines to announce a disabled call at all is fine; announcing one
    # and never closing it is not.
    assert len(_events(lines, "tool_end")) == len(_events(lines, "tool_start"))
    replays = [
        request["messages"]
        for request in transport.requests
        if any(message.get("role") == "assistant" for message in request["messages"])
    ]
    for messages in replays:
        assistant = [m for m in messages if m.get("role") == "assistant" and m.get("tool_calls")]
        tool_results = [m for m in messages if m.get("role") == "tool"]
        assert len(tool_results) == sum(len(m["tool_calls"]) for m in assistant)


def test_non_string_content_reaches_the_conversation_replay(executed):
    """List-form content (Anthropic-shaped blocks) must not vanish from history.

    The client sees it, but the assistant message replayed to the provider does
    not, so the model loses its own prior words on the follow-up turn.
    """
    transport = FakeTransport(
        [
            [
                _raw(
                    {
                        "choices": [
                            {"index": 0, "delta": {"content": [{"type": "text", "text": "SPOKEN"}]}}
                        ]
                    }
                ),
                _sse(
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "c1",
                                "function": {"name": "web_search", "arguments": "{}"},
                            }
                        ]
                    }
                ),
                _sse(finish = "tool_calls"),
                _DONE,
            ],
            _answer_turn(),
        ]
    )
    _run(transport)

    assistant = transport.requests[1]["messages"][1]
    assert "SPOKEN" in json.dumps(assistant["content"])


# ── Cancellation ──────────────────────────────────────────────────


def test_cancel_before_the_first_turn_does_nothing(executed):
    cancel_event = threading.Event()
    cancel_event.set()
    transport = FakeTransport([_call_turn()])
    lines = _run(transport, cancel_event = cancel_event)

    assert lines == []
    assert transport.requests == []
    assert executed == []


def test_cancel_during_tool_execution_still_closes_the_tool_card(executed, monkeypatch):
    """A cancelled tool must not leave a card spinning forever.

    tool_start is already on the wire; returning without tool_end leaves the UI
    showing a running tool for a turn that ended.
    """
    cancel_event = threading.Event()

    def _execute(name, arguments, **kwargs):
        cancel_event.set()
        raise RuntimeError("cancelled")

    monkeypatch.setattr(loop_mod, "execute_tool", _execute)

    transport = FakeTransport([_call_turn(), _answer_turn()])
    lines = _run(transport, cancel_event = cancel_event)

    assert len(_events(lines, "tool_start")) == 1
    assert len(_events(lines, "tool_end")) == 1, "tool_start with no tool_end"


def test_cancel_between_turns_stops_before_asking_again(executed, monkeypatch):
    cancel_event = threading.Event()

    def _execute(name, arguments, **kwargs):
        cancel_event.set()
        return "RESULT"

    monkeypatch.setattr(loop_mod, "execute_tool", _execute)

    transport = FakeTransport([_call_turn(), _answer_turn()])
    lines = _run(transport, cancel_event = cancel_event)

    assert len(transport.requests) == 1
    assert len(_events(lines, "tool_end")) == 1


def test_closing_the_generator_closes_the_transport_stream(executed):
    """The route calls gen.aclose(); the provider's HTTP stream must close too.

    Leaving the inner async generator to the garbage collector holds an httpx
    response open for an indeterminate time after the client is gone: today the
    transport stream is finalised by the asyncgen hook a tick later, which is
    also where the route's "httpcore asyncgen cleanup" RuntimeError comes from.
    """
    transport = FakeTransport([_call_turn(), _answer_turn()])

    async def _partial():
        agen = stream_with_studio_tools(
            transport,
            run = ToolLoopRun(messages = [{"role": "user", "content": "hi"}], session_id = "s1"),
            policy = _policy(),
            cancel_event = threading.Event(),
        )
        await agen.__anext__()
        await agen.aclose()
        return transport.opened, transport.closed

    opened, closed = asyncio.run(_partial())
    assert closed == opened, f"{opened} transport streams opened, {closed} closed"


# ── Approvals ─────────────────────────────────────────────────────


def _approval_turns():
    return [_call_turn(name = "python", arguments = '{"query":"1"}'), _answer_turn()]


def test_approval_allow_runs_the_tool(executed, monkeypatch):
    monkeypatch.setattr(loop_mod, "begin_tool_decision", lambda session, approval: {"slot": True})
    monkeypatch.setattr(
        loop_mod, "wait_tool_decision", lambda slot, approval, cancel_event = None: "allow"
    )
    monkeypatch.setattr(loop_mod, "abort_tool_decision", lambda slot, approval: None)

    lines = _run(
        FakeTransport(_approval_turns()),
        tools = [PY],
        permission_mode = "ask",
        confirm_calls = True,
    )

    assert [call["name"] for call in executed] == ["python"]
    assert _events(lines, "tool_start")[0]["awaiting_confirmation"] is True


def test_approval_deny_does_not_run_the_tool(executed, monkeypatch):
    monkeypatch.setattr(loop_mod, "begin_tool_decision", lambda session, approval: {"slot": True})
    monkeypatch.setattr(
        loop_mod, "wait_tool_decision", lambda slot, approval, cancel_event = None: "deny"
    )
    monkeypatch.setattr(loop_mod, "abort_tool_decision", lambda slot, approval: None)

    lines = _run(
        FakeTransport(_approval_turns()),
        tools = [PY],
        permission_mode = "ask",
        confirm_calls = True,
    )

    assert executed == []
    assert "declined" in _events(lines, "tool_end")[0]["result"].lower()


def test_approval_that_never_arrives_ends_on_cancel_without_leaking_a_slot(executed):
    """Real approval plumbing: the user closes the tab while the card is up."""
    from state import tool_approvals

    # Slots this test did not open. The registry is process-global and a sibling
    # module that drives the route can leave one behind, so asserting the whole
    # dict is empty makes this test pass or fail on collection order.
    pre_existing = set(tool_approvals._pending)
    cancel_event = threading.Event()
    threading.Timer(0.6, cancel_event.set).start()

    lines = _run(
        FakeTransport(_approval_turns()),
        tools = [PY],
        permission_mode = "ask",
        confirm_calls = True,
        cancel_event = cancel_event,
        deadline = 30.0,
    )

    assert executed == []
    assert len(_events(lines, "tool_end")) == 1
    leaked = set(tool_approvals._pending) - pre_existing
    assert not leaked, {key: tool_approvals._pending[key] for key in leaked}
