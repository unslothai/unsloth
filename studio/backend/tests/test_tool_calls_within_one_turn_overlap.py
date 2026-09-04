# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""One turn's tool calls run at the same time, and still answer in order.

WHAT IT COST TO RUN THEM ONE AT A TIME

A model that asks for three web searches in one turn was getting them sequentially, so
the turn cost the sum of the three rather than the longest of them. Nothing about the
format asks for that: parallel tool calls are parallel precisely because the model has
declared them independent, and every provider that emits them expects them to overlap.

WHAT MUST NOT CHANGE

Overlapping the WAITING is the whole change. Everything downstream of a tool returning is
order sensitive and stays sequential:

  * the transcript. `tool_messages` must be in call order or the provider sees results
    attached to the wrong calls, and OpenAI, Anthropic and Gemini all reject that history.
  * the SSE. A card that fills in before the card above it opened reads as the wrong tool
    answering.
  * the call budget. `max_calls` counts calls, and launching four when one remains would
    spend a budget the loop had already refused.
  * approvals. They are interactive and one at a time, so a round containing any gated
    call keeps the strict order rather than asking about a decision whose siblings have
    already run.

The overlap itself is measured with a barrier rather than a sleep: two tools that must
each see the other before either may return can only both return if they were running
together, and the test hangs on its own timeout instead of passing on a fast machine.
"""

from __future__ import annotations

import asyncio
import json
import sys
import threading
from pathlib import Path

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

import pytest  # noqa: E402

from core.inference import studio_tool_loop as loop_mod  # noqa: E402

# The scripted transport and the SSE readers, rather than a second copy of them: a fake
# that drifts from the one the rest of the loop is tested against would be testing a
# different loop. Same sys.path dance as test_memory_contract.py.
from test_studio_tool_loop import (  # noqa: E402
    WEB,
    PY,
    FakeTransport,
    _DONE,
    _events,
    _run,
    _sse,
)


def _two_calls(first = "alpha", second = "beta", tool = "web_search"):
    """One turn asking for two calls of the same tool, the shape providers emit."""
    return FakeTransport(
        [
            [
                _sse(
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_a",
                                "function": {
                                    "name": tool,
                                    "arguments": json.dumps({"query": first}),
                                },
                            },
                            {
                                "index": 1,
                                "id": "call_b",
                                "function": {
                                    "name": tool,
                                    "arguments": json.dumps({"query": second}),
                                },
                            },
                        ]
                    }
                ),
                _sse(finish = "tool_calls"),
                _DONE,
            ],
            [_sse({"content": "done"}), _sse(finish = "stop"), _DONE],
        ],
        heals = False,
    )


@pytest.fixture
def rendezvous(monkeypatch):
    """A tool that cannot return until another call of it has also started.

    This is the measurement. A sleep would pass on a machine that happens to be fast and
    a timing assertion would be flaky on one that is loaded; a barrier can only be cleared
    by genuine overlap, and the absence of overlap shows up as the timeout rather than as
    a number that drifted.
    """
    # Long enough that a loaded runner still meets it, short enough that the
    # serialised cases (where it can never be met) do not dominate the suite.
    barrier = threading.Barrier(2, timeout = 4)
    order: list[str] = []
    lock = threading.Lock()

    def _execute(name, arguments, **kwargs):
        query = (arguments or {}).get("query", "")
        with lock:
            order.append(query)
        try:
            barrier.wait()
        except threading.BrokenBarrierError:
            return f"ALONE<{query}>"
        return f"TOGETHER<{query}>"

    monkeypatch.setattr(loop_mod, "execute_tool", _execute)
    monkeypatch.setattr(loop_mod, "build_rag_autoinject", lambda *a, **k: None)
    monkeypatch.setattr(loop_mod, "is_high_risk_tool_call", lambda name, args: name == "python")
    return order


@pytest.fixture
def recorder(monkeypatch):
    """Records the calls without blocking, for the cases that are about order."""
    calls: list[dict] = []

    def _execute(name, arguments, **kwargs):
        calls.append({"name": name, "arguments": arguments})
        return f"RESULT<{(arguments or {}).get('query')}>"

    monkeypatch.setattr(loop_mod, "execute_tool", _execute)
    monkeypatch.setattr(loop_mod, "build_rag_autoinject", lambda *a, **k: None)
    monkeypatch.setattr(loop_mod, "is_high_risk_tool_call", lambda name, args: name == "python")
    return calls


class TestTheyActuallyOverlap:
    def test_two_calls_are_in_flight_at_once(self, rendezvous):
        """Neither tool can return until the other has started, so both must be running."""
        lines = _run(_two_calls())
        ends = _events(lines, "tool_end")
        assert len(ends) == 2
        assert all("TOGETHER" in (end.get("result") or "") for end in ends), (
            "a tool returned without ever meeting the other, so the calls were "
            f"serialised: {[end.get('result') for end in ends]}"
        )
        assert sorted(rendezvous) == ["alpha", "beta"]

    def test_the_switch_puts_them_back_in_single_file(self, rendezvous, monkeypatch):
        """The escape hatch has to actually reach the loop, not just exist.

        "Independent" is the model's claim, not a guarantee, so an install that has two
        tools writing the same file needs a way back to the old order. With the calls
        serialised the barrier can never be met, and each returns ALONE.
        """
        monkeypatch.setenv("UNSLOTH_PARALLEL_TOOL_CALLS", "0")
        lines = _run(_two_calls())
        ends = _events(lines, "tool_end")
        assert len(ends) == 2
        assert all("ALONE" in (end.get("result") or "") for end in ends)

    def test_a_single_call_is_untouched(self, rendezvous):
        """One call is not a batch, and must not wait for a partner that never comes."""
        transport = FakeTransport(
            [
                [
                    _sse(
                        {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_a",
                                    "function": {
                                        "name": "web_search",
                                        "arguments": '{"query":"solo"}',
                                    },
                                }
                            ]
                        }
                    ),
                    _sse(finish = "tool_calls"),
                    _DONE,
                ],
                [_sse({"content": "done"}), _sse(finish = "stop"), _DONE],
            ],
            heals = False,
        )
        ends = _events(_run(transport), "tool_end")
        assert len(ends) == 1
        assert "ALONE" in (ends[0].get("result") or "")


class TestOrderIsStillTheModelsOrder:
    def test_the_cards_open_and_close_in_call_order(self, recorder):
        lines = _run(_two_calls("alpha", "beta"))
        starts = [event.get("tool_call_id") for event in _events(lines, "tool_start")]
        ends = [event.get("tool_call_id") for event in _events(lines, "tool_end")]
        assert starts == ["call_a", "call_b"]
        assert ends == ["call_a", "call_b"], (
            "the second call answered first, so a user watching the stream saw one "
            "card fill in with another card's result"
        )

    def test_the_transcript_keeps_each_result_with_its_call(self, recorder):
        """The second request carries the round's history; that is what the provider reads."""
        transport = _two_calls("alpha", "beta")
        _run(transport)
        assert len(transport.requests) == 2
        messages = transport.requests[1]["messages"]
        tool_rows = [m for m in messages if m.get("role") == "tool"]
        assert [row.get("tool_call_id") for row in tool_rows] == ["call_a", "call_b"]
        assert [row.get("content") for row in tool_rows] == [
            "RESULT<alpha>",
            "RESULT<beta>",
        ]

    def test_the_assistant_row_lists_both_calls(self, recorder):
        transport = _two_calls("alpha", "beta")
        _run(transport)
        assistant = [
            m
            for m in transport.requests[1]["messages"]
            if m.get("role") == "assistant" and m.get("tool_calls")
        ]
        assert assistant, "the round's calls never reached the history"
        assert [c["id"] for c in assistant[-1]["tool_calls"]] == ["call_a", "call_b"]


class TestTheLimitsThatMustHold:
    def test_the_call_budget_counts_launches_not_finishes(self, recorder):
        """With one call left, a two-call round must still run exactly one.

        The budget check reads `remaining`, which is only decremented when a call
        settles. Launching both and then discovering the budget was spent would run a
        tool the loop had already refused, side effects and all.
        """
        _run(_two_calls("alpha", "beta"), max_calls = 1)
        assert len(recorder) == 1

    def test_a_gated_round_is_not_parallelised(self, rendezvous, monkeypatch):
        """Approvals are interactive and one at a time.

        `permission_mode="auto"` gates only high-risk tools, and the fixture marks
        `python` as one, so this round asks for confirmation and must serialise. It never
        reaches an approval prompt in this test: what is asserted is that the loop chose
        the sequential path, which the barrier reports as ALONE.
        """
        monkeypatch.setattr(loop_mod, "begin_tool_decision", lambda *a, **k: object())
        monkeypatch.setattr(loop_mod, "abort_tool_decision", lambda *a, **k: None)
        monkeypatch.setattr(loop_mod, "wait_tool_decision", lambda *a, **k: "allow")
        lines = _run(
            _two_calls(tool = "python"),
            tools = [WEB, PY],
            permission_mode = "auto",
            confirm_calls = True,
        )
        ends = _events(lines, "tool_end")
        assert ends, "the round produced no tool results at all"
        assert all("TOGETHER" not in (end.get("result") or "") for end in ends)

    def test_an_ordinary_round_still_overlaps_under_auto(self, rendezvous):
        """The gate is per round and must not be the mere presence of a permission mode.

        Under `auto` a round of low-risk reads asks nothing of the user, so serialising it
        would be paying the approval tax without an approval.
        """
        lines = _run(_two_calls(), permission_mode = "auto", confirm_calls = True)
        ends = _events(lines, "tool_end")
        assert len(ends) == 2
        assert all("TOGETHER" in (end.get("result") or "") for end in ends)


class TestCancellation:
    def test_a_cancelled_round_does_not_hang(self, monkeypatch):
        """A tool that ignores the cancel flag must not hold the answer open forever.

        The pump stops asking for events as soon as the flag is set, and the settle path
        joins the worker rather than closing a generator that is still executing.
        """
        started = threading.Event()

        def _execute(name, arguments, **kwargs):
            started.set()
            event = kwargs.get("cancel_event")
            if event is not None:
                event.wait(timeout = 5)
            return "late"

        monkeypatch.setattr(loop_mod, "execute_tool", _execute)
        monkeypatch.setattr(loop_mod, "build_rag_autoinject", lambda *a, **k: None)
        monkeypatch.setattr(
            loop_mod, "is_high_risk_tool_call", lambda name, args: name == "python"
        )

        transport = _two_calls()
        cancel_event = threading.Event()

        async def _collect():
            from core.inference.studio_tool_loop import (
                ToolLoopPolicy,
                ToolLoopRun,
                stream_with_studio_tools,
            )

            agen = stream_with_studio_tools(
                transport,
                run = ToolLoopRun(
                    messages = [{"role": "user", "content": "hi"}],
                    session_id = "s1",
                    thread_id = "t1",
                    tool_choice = None,
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
                cancel_event = cancel_event,
            )
            out = []
            async for line in agen:
                out.append(line)
                if started.is_set():
                    cancel_event.set()
            return out

        # The assertion is that this returns at all. A leaked pump or a generator closed
        # while its worker was still running shows up here as the timeout.
        async def _bounded():
            return await asyncio.wait_for(_collect(), timeout = 30)

        lines = asyncio.run(_bounded())
        assert lines, "the round produced nothing before it was cancelled"
