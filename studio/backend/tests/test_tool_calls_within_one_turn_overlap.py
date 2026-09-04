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


def _two_calls(
    first = "alpha",
    second = "beta",
    tool = "web_search",
):
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
        assert [row.get("content") for row in tool_rows] == ["RESULT<alpha>", "RESULT<beta>"]

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
        monkeypatch.setattr(loop_mod, "is_high_risk_tool_call", lambda name, args: name == "python")

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


# ── The local GGUF loop ───────────────────────────────────────────────────────────
#
# A separate implementation with the same requirement. It matters more for this work than
# the provider loop does, because it is the path that decodes into the shared KV cache the
# preemptor manages: a chat parked on three serial searches holds its cells for the sum of
# the three.
#
# Its overlap comes from a different mechanism. `stream_tool_execution` spawns the tool's
# worker inside its GENERATOR BODY, so building the generator starts nothing and the first
# next() is what puts the tool in flight. The round primes every call, which starts them
# all, and then reads their events back in order.


from test_llama_cpp_tool_loop import _done as _gguf_done  # noqa: E402
from test_llama_cpp_tool_loop import _make_backend  # noqa: E402
from test_llama_cpp_tool_loop import _sse as _gguf_sse  # noqa: E402


def _gguf_round(calls):
    """One assistant turn asking for `calls` = [(id, name, args-dict), ...]."""
    return [
        _gguf_sse(
            {
                "tool_calls": [
                    {
                        "index": i,
                        "id": cid,
                        "type": "function",
                        "function": {"name": name, "arguments": json.dumps(args)},
                    }
                    for i, (cid, name, args) in enumerate(calls)
                ]
            }
        ),
        _gguf_done(),
    ]


def _gguf_events(
    monkeypatch,
    calls,
    execute,
    *,
    tools = None,
    **kwargs,
):
    payloads: list[dict] = []
    backend = _make_backend(
        monkeypatch,
        [_gguf_round(calls), [_gguf_sse({"content": "Final answer."}), _gguf_done()]],
        payloads,
    )
    monkeypatch.setattr("core.inference.tools.execute_tool", execute)
    names = sorted({name for _cid, name, _args in calls})
    events = list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "go"}],
            tools = tools or [{"type": "function", "function": {"name": name}} for name in names],
            max_tool_iterations = 2,
            **kwargs,
        )
    )
    return events, payloads


class TestTheLocalGgufLoopOverlapsToo:
    def test_two_different_searches_run_together(self, monkeypatch):
        barrier = threading.Barrier(2, timeout = 4)

        def _execute(name, arguments, **_kwargs):
            try:
                barrier.wait()
            except threading.BrokenBarrierError:
                return f"ALONE<{arguments.get('query')}>"
            return f"TOGETHER<{arguments.get('query')}>"

        events, _payloads = _gguf_events(
            monkeypatch,
            [
                ("call_a", "web_search", {"query": "alpha"}),
                ("call_b", "web_search", {"query": "beta"}),
            ],
            _execute,
        )
        ends = [e for e in events if e.get("type") == "tool_end"]
        assert len(ends) == 2
        assert all(
            "TOGETHER" in (end.get("result") or "") for end in ends
        ), f"the round serialised: {[end.get('result') for end in ends]}"

    def test_the_results_still_arrive_in_call_order(self, monkeypatch):
        def _execute(name, arguments, **_kwargs):
            return f"RESULT<{arguments.get('query')}>"

        events, _payloads = _gguf_events(
            monkeypatch,
            [
                ("call_a", "web_search", {"query": "alpha"}),
                ("call_b", "web_search", {"query": "beta"}),
            ],
            _execute,
        )
        ends = [e for e in events if e.get("type") == "tool_end"]
        assert [e.get("tool_call_id") for e in ends] == ["call_a", "call_b"]
        assert [e.get("result") for e in ends] == ["RESULT<alpha>", "RESULT<beta>"]

    def test_the_switch_reaches_this_loop_as_well(self, monkeypatch):
        monkeypatch.setenv("UNSLOTH_PARALLEL_TOOL_CALLS", "0")
        barrier = threading.Barrier(2, timeout = 4)

        def _execute(name, arguments, **_kwargs):
            try:
                barrier.wait()
            except threading.BrokenBarrierError:
                return f"ALONE<{arguments.get('query')}>"
            return f"TOGETHER<{arguments.get('query')}>"

        events, _payloads = _gguf_events(
            monkeypatch,
            [
                ("call_a", "web_search", {"query": "alpha"}),
                ("call_b", "web_search", {"query": "beta"}),
            ],
            _execute,
        )
        ends = [e for e in events if e.get("type") == "tool_end"]
        assert len(ends) == 2
        assert all("ALONE" in (end.get("result") or "") for end in ends)

    def test_a_round_that_repeats_a_call_stays_sequential(self, monkeypatch):
        """The one dependency between a round's calls, and why it is checked up front.

        `prepare_call` turns the second identical call into a no-op because
        `record_result` put the first one's key in `_successful_keys`. Deciding that with
        both in flight would run the tool twice, so a round containing a repeat keeps the
        order it has always had, and the second call is suppressed exactly as before.
        """
        ran: list = []

        def _execute(name, arguments, **_kwargs):
            ran.append(arguments.get("query"))
            return "search-result"

        events, _payloads = _gguf_events(
            monkeypatch,
            [
                ("call_a", "web_search", {"query": "same"}),
                ("call_b", "web_search", {"query": "same"}),
            ],
            _execute,
        )
        assert ran == ["same"], "the duplicate ran, so the round was overlapped"
        # One card, not two: the suppressed call is an internal no-op and never opens one.
        assert [e.get("type") for e in events].count("tool_end") == 1

    def test_the_result_budget_is_divided_by_the_whole_batch(self, monkeypatch):
        """Sequentially call k divides by the calls still to run, because `_spent` has
        already grown by the results before it. Run together they all price against the
        same `_spent`, so each dividing by its own remainder would hand out
        B/N + B/(N-1) + ... , which is more than the batch has.
        """
        budgets: list = []

        def _execute(name, arguments, **kwargs):
            budgets.append(kwargs.get("result_budget_tokens"))
            return f"RESULT<{arguments.get('query')}>"

        _gguf_events(
            monkeypatch,
            [
                ("call_a", "web_search", {"query": "alpha"}),
                ("call_b", "web_search", {"query": "beta"}),
                ("call_c", "web_search", {"query": "gamma"}),
            ],
            _execute,
        )
        given = [b for b in budgets if isinstance(b, int)]
        if not given:
            pytest.skip("this build does not pass result_budget_tokens")
        # Not identical: each call still subtracts the ARGUMENTS of the calls after it,
        # which is a real cost and differs per position. What must not differ is the
        # divisor, and a wrong one shows up as a spread far larger than that: with three
        # calls, B/3 against B/1 is a factor of three, where the argument term moves the
        # figure by a few per cent.
        assert (
            max(given) <= min(given) * 1.2
        ), f"the calls were priced against different batch sizes: {given}"
        # And the batch as a whole must not be handed more than one call's worth of the
        # window three times over.
        assert sum(given) <= max(given) * 3.3


class TestTheRoundLevelLimitsThatAreReadWhileItIsPrepared:
    """Anything the loop reads per call and updates per RESULT is a hazard here.

    An overlapped round prepares every call before any of them finishes, so a counter that
    is read while preparing and written when settling is consulted at its starting value
    every time. Two of these were found by running it: the controller's duplicate ledger,
    which is why a repeated call keeps the round sequential, and the RAG search cap below.
    """

    def test_the_search_cap_is_not_exceeded_by_a_single_round(self):
        from core.inference.tool_call_parser import RAG_MAX_SEARCHES_PER_TURN
        assert RAG_MAX_SEARCHES_PER_TURN >= 1

    def test_the_cap_counts_launches_not_finishes(self, monkeypatch):
        """More searches in one turn than the cap allows, all distinct so the round overlaps.

        Counting them as they settle lets the whole round through, because every call read
        the counter before any of them had incremented it.
        """
        from core.inference.tool_call_parser import RAG_MAX_SEARCHES_PER_TURN, RAG_SEARCH_TOOLS

        tool = sorted(RAG_SEARCH_TOOLS)[0]
        ran: list = []

        def _execute(name, arguments, **_kwargs):
            ran.append(arguments.get("query"))
            return f"RESULT<{arguments.get('query')}>"

        n = RAG_MAX_SEARCHES_PER_TURN + 2
        _gguf_events(
            monkeypatch,
            [(f"call_{i}", tool, {"query": f"q{i}"}) for i in range(n)],
            _execute,
            tools = [{"type": "function", "function": {"name": tool}}],
        )
        assert len(ran) <= RAG_MAX_SEARCHES_PER_TURN, (
            f"{len(ran)} searches ran against a cap of {RAG_MAX_SEARCHES_PER_TURN}: the "
            "cap was read while the round was being prepared and written when it settled"
        )


class TestNothingNewSlipsIntoTheSameHazard:
    """A guard on the shape of the bug, not on any one instance of it.

    Three bugs in this change were the same mistake: state that the loop READS while
    preparing a call and WRITES when that call settles is read at its starting value by
    every call in an overlapped round. The duplicate ledger, the RAG search cap and the
    forced tool choice were each found separately, by running it.

    So pin the list. Both settle paths declare what they write, and adding a name to
    either is exactly the moment to ask whether the head reads it too. A test that fails
    on a NEW name is worth more than three tests for the three names already handled.
    """

    def _nonlocals(self, path, marker):
        import ast
        import pathlib

        source = pathlib.Path(path).read_text(encoding = "utf-8")
        tree = ast.parse(source)
        found: set = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == marker:
                for inner in ast.walk(node):
                    if isinstance(inner, ast.Nonlocal):
                        found.update(inner.names)
        return found

    def test_the_gguf_settle_path_writes_only_the_reviewed_names(self):
        import core.inference.llama_cpp as mod
        names = self._nonlocals(mod.__file__, "_settle_tool_call")
        assert names == {
            # counted at LAUNCH for an overlapped round; the write here is the sequential path
            "_kb_search_count",
            # spent at launch too, for the same reason
            "_forced_tool_call_pending",
            # read only after the round, so settling order is enough
            "_last_reprompt_text",
            "_turn_executed_real_tool",
            "_forced_choice_resolved",
            # rebound by compaction; every call was appended to it before any settled
            "assistant_msg",
        }, (
            f"the settle path now writes {sorted(names)}. If the loop reads any new one "
            "while preparing a call, an overlapped round reads it before this runs."
        )

    def test_the_provider_settle_path_writes_only_the_reviewed_names(self):
        import core.inference.studio_tool_loop as mod
        names = self._nonlocals(mod.__file__, "_settle_call")
        assert names == {
            # the launch site subtracts len(pending_calls) so the budget counts launches
            "remaining",
            # read only after the round
            "turn_executed_real_tool",
            "executed_any",
            "last_reprompt_text",
        }, (
            f"the settle path now writes {sorted(names)}. If the loop reads any new one "
            "while preparing a call, an overlapped round reads it before this runs."
        )


class TestTheReplayedTurnIsWhatTheModelSees:
    """`assistant_msg` is the third name read while preparing and written when settling.

    It is safe for a structural reason rather than a counted one: every call is APPENDED to
    it in the preparing pass, and the rebinding only happens in the settling pass, so the
    hazard its own comment describes -- "the next one in the batch appends its tool_call to
    this handle while its RESULT goes to conversation" -- cannot occur in an overlapped
    round. That is an argument, so here it is as a measurement instead.
    """

    def test_one_assistant_row_carries_every_call_and_its_own_result(self, monkeypatch):
        def _execute(name, arguments, **_kwargs):
            return f"RESULT<{arguments.get('query')}>"

        _events, payloads = _gguf_events(
            monkeypatch,
            [
                ("call_a", "web_search", {"query": "alpha"}),
                ("call_b", "web_search", {"query": "beta"}),
                ("call_c", "web_search", {"query": "gamma"}),
            ],
            _execute,
        )
        assert len(payloads) >= 2, "the round never produced a follow-up request"
        messages = payloads[1]["messages"]
        assistant = [m for m in messages if m.get("role") == "assistant" and m.get("tool_calls")]
        assert assistant, "the round's calls never reached the replayed history"
        assert [c["id"] for c in assistant[-1]["tool_calls"]] == ["call_a", "call_b", "call_c"]
        tool_rows = [m for m in messages if m.get("role") == "tool"]
        assert [row.get("tool_call_id") for row in tool_rows] == ["call_a", "call_b", "call_c"]
        assert [row.get("content") for row in tool_rows] == [
            "RESULT<alpha>",
            "RESULT<beta>",
            "RESULT<gamma>",
        ], "a result was attached to a call that did not produce it"
