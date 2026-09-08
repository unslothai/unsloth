# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""One turn's tool calls run at the same time, in both loops, and still answer in the
model's order under every limit that is read while the round is prepared."""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

import pytest  # noqa: E402

from core.inference import llama_cpp as llama_mod  # noqa: E402
from core.inference import studio_tool_loop as loop_mod  # noqa: E402
from core.inference import tool_loop_controller as controller_mod  # noqa: E402

from .preempt_fakes import executed, rendezvous  # noqa: E402, F401

# The scripted transport and the SSE readers, rather than a second copy: a fake that drifts
# from the one the rest of the loop is tested against would be testing a different loop.
from test_studio_tool_loop import (  # noqa: E402
    WEB,
    PY,
    FakeTransport,
    _DONE,
    _events,
    _run,
    _sse,
)
from test_llama_cpp_tool_loop import _done as _gguf_done  # noqa: E402
from test_llama_cpp_tool_loop import _make_backend  # noqa: E402
from test_llama_cpp_tool_loop import _sse as _gguf_sse  # noqa: E402

_CAP = loop_mod._MAX_PARALLEL_TOOL_CALLS_PER_ROUND
# `max_tool_calls_per_message` at its unlimited sentinel, where nothing above the launch
# refuses a call: the value the overlap report was filed against.
_UNLIMITED = 9999


def _provider_round(calls, tool = "web_search"):
    """A provider turn asking for `calls`, then a plain answer."""
    return FakeTransport(
        [
            [
                _sse(
                    {
                        "tool_calls": [
                            {
                                "index": index,
                                "id": call_id,
                                "function": {
                                    "name": tool,
                                    "arguments": json.dumps({"query": query}),
                                },
                            }
                            for index, (call_id, query) in enumerate(calls)
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


def _two_calls(
    first = "alpha",
    second = "beta",
    tool = "web_search",
):
    return _provider_round([("call_a", first), ("call_b", second)], tool = tool)


@pytest.fixture
def recorder(monkeypatch):
    calls: list[dict] = []

    def _execute(name, arguments, **kwargs):
        calls.append({"name": name, "arguments": arguments})
        return f"RESULT<{(arguments or {}).get('query')}>"

    monkeypatch.setattr(loop_mod, "execute_tool", _execute)
    monkeypatch.setattr(loop_mod, "build_rag_autoinject", lambda *a, **k: None)
    monkeypatch.setattr(loop_mod, "is_high_risk_tool_call", lambda name, args: name == "python")
    return calls


def _gguf_round(calls):
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


def _searches(count):
    return [(f"call_{i}", "web_search", {"query": f"q{i}"}) for i in range(count)]


def _meeting_tool(parties, timeout = 4):
    """Returns TOGETHER only if `parties` calls are in flight at the same moment."""
    barrier = threading.Barrier(parties, timeout = timeout)

    def _execute(name, arguments, **_kwargs):
        query = (arguments or {}).get("query")
        try:
            barrier.wait()
        except threading.BrokenBarrierError:
            return f"ALONE<{query}>"
        return f"TOGETHER<{query}>"

    return _execute


def _results(events):
    return [end.get("result") or "" for end in events]


# ------------------------------------------------------------------- the provider loop


class TestTheProviderLoopOverlaps:
    def test_two_calls_are_in_flight_at_once(self, rendezvous):
        ends = _events(_run(_two_calls()), "tool_end")
        assert len(ends) == 2
        assert all(
            "TOGETHER" in result for result in _results(ends)
        ), f"a tool returned without ever meeting the other: {_results(ends)}"
        assert sorted(rendezvous) == ["alpha", "beta"]

    def test_the_switch_puts_them_back_in_single_file(self, rendezvous, monkeypatch):
        monkeypatch.setenv("UNSLOTH_PARALLEL_TOOL_CALLS", "0")
        ends = _events(_run(_two_calls()), "tool_end")
        assert len(ends) == 2
        assert all("ALONE" in result for result in _results(ends))

    def test_a_gated_round_is_not_parallelised(self, rendezvous, monkeypatch):
        monkeypatch.setattr(loop_mod, "begin_tool_decision", lambda *a, **k: object())
        monkeypatch.setattr(loop_mod, "abort_tool_decision", lambda *a, **k: None)
        monkeypatch.setattr(loop_mod, "wait_tool_decision", lambda *a, **k: "allow")
        ends = _events(
            _run(
                _two_calls(tool = "python"),
                tools = [WEB, PY],
                permission_mode = "auto",
                confirm_calls = True,
            ),
            "tool_end",
        )
        assert ends, "the round produced no tool results at all"
        assert all("TOGETHER" not in result for result in _results(ends))


class TestAProviderRoundBoundsItsOverlap:
    """A round the model made arbitrarily long would otherwise start a worker thread and
    a pump task per call, all of their side effects at once, with nothing bounding it."""

    def test_the_two_loops_agree_on_the_cap(self):
        assert _CAP == llama_mod._MAX_PARALLEL_TOOL_CALLS_PER_ROUND

    def test_a_round_past_the_cap_runs_single_file_and_still_answers_in_order(self, rendezvous):
        transport = _provider_round([(f"call_{i}", f"q{i}") for i in range(_CAP + 1)])
        lines = _run(transport, tools = [WEB], max_calls = _UNLIMITED)
        ends = _events(lines, "tool_end")
        assert len(ends) == _CAP + 1, "every call still has to run"
        assert all("ALONE" in result for result in _results(ends)), _results(ends)
        assert [e.get("tool_call_id") for e in ends] == [f"call_{i}" for i in range(_CAP + 1)]
        tool_rows = [row for row in transport.requests[1]["messages"] if row.get("role") == "tool"]
        assert [row.get("tool_call_id") for row in tool_rows] == [
            f"call_{i}" for i in range(_CAP + 1)
        ], "the provider reads results by position, so a reordered history answers wrongly"


class TestOrderIsStillTheModelsOrder:
    def test_the_cards_and_the_replayed_transcript_follow_the_call_order(self, recorder):
        transport = _two_calls("alpha", "beta")
        lines = _run(transport)
        assert [e.get("tool_call_id") for e in _events(lines, "tool_start")] == [
            "call_a",
            "call_b",
        ]
        assert [e.get("tool_call_id") for e in _events(lines, "tool_end")] == [
            "call_a",
            "call_b",
        ], (
            "the second call answered first, so a user watching the stream saw one card "
            "fill in with another card's result"
        )
        messages = transport.requests[1]["messages"]
        assistant = [m for m in messages if m.get("role") == "assistant" and m.get("tool_calls")]
        assert assistant and [c["id"] for c in assistant[-1]["tool_calls"]] == ["call_a", "call_b"]
        tool_rows = [m for m in messages if m.get("role") == "tool"]
        assert [row.get("tool_call_id") for row in tool_rows] == ["call_a", "call_b"]
        assert [row.get("content") for row in tool_rows] == ["RESULT<alpha>", "RESULT<beta>"]


def _fast_sizing(monkeypatch):
    monkeypatch.setattr(
        llama_mod.LlamaCppBackend,
        "count_chat_tokens",
        lambda self, messages, *args, **kwargs: 8 * len(messages or []),
    )


class TestTheLocalGgufLoopOverlapsToo:
    def test_two_different_searches_run_together(self, monkeypatch):
        events, _payloads = _gguf_events(monkeypatch, _searches(2), _meeting_tool(2))
        ends = [e for e in events if e.get("type") == "tool_end"]
        assert len(ends) == 2
        assert all("TOGETHER" in result for result in _results(ends)), _results(ends)

    def test_the_results_arrive_in_call_order_carrying_their_own_answers(self, monkeypatch):
        def _execute(name, arguments, **_kwargs):
            return f"RESULT<{arguments.get('query')}>"

        events, payloads = _gguf_events(
            monkeypatch,
            [
                ("call_a", "web_search", {"query": "alpha"}),
                ("call_b", "web_search", {"query": "beta"}),
                ("call_c", "web_search", {"query": "gamma"}),
            ],
            _execute,
        )
        ends = [e for e in events if e.get("type") == "tool_end"]
        assert [e.get("tool_call_id") for e in ends] == ["call_a", "call_b", "call_c"]
        assert _results(ends) == ["RESULT<alpha>", "RESULT<beta>", "RESULT<gamma>"]
        # And one assistant row carries every call, each beside its own result.
        messages = payloads[1]["messages"]
        assistant = [m for m in messages if m.get("role") == "assistant" and m.get("tool_calls")]
        assert assistant and [c["id"] for c in assistant[-1]["tool_calls"]] == [
            "call_a",
            "call_b",
            "call_c",
        ]
        tool_rows = [m for m in messages if m.get("role") == "tool"]
        assert [row.get("content") for row in tool_rows] == [
            "RESULT<alpha>",
            "RESULT<beta>",
            "RESULT<gamma>",
        ], "a result was attached to a call that did not produce it"

    def test_a_round_that_repeats_a_call_stays_sequential(self, monkeypatch):
        """The one-shot gate reads state the settle writes, so its round may not overlap."""
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
        assert [e.get("type") for e in events].count("tool_end") == 1

    def test_the_search_cap_counts_launches_and_keeps_the_capped_calls_in_place(self, monkeypatch):
        from core.inference.tool_call_parser import RAG_MAX_SEARCHES_PER_TURN, RAG_SEARCH_TOOLS

        tool = sorted(RAG_SEARCH_TOOLS)[0]
        assert RAG_MAX_SEARCHES_PER_TURN >= 1
        ran: list = []

        def _execute(name, arguments, **_kwargs):
            ran.append(arguments.get("query"))
            return f"RESULT<{arguments.get('query')}>"

        n = RAG_MAX_SEARCHES_PER_TURN + 2
        events, _payloads = _gguf_events(
            monkeypatch,
            [(f"call_{i}", tool, {"query": f"q{i}"}) for i in range(n)],
            _execute,
            tools = [{"type": "function", "function": {"name": tool}}],
        )
        assert len(ran) <= RAG_MAX_SEARCHES_PER_TURN, (
            f"{len(ran)} searches ran against a cap of {RAG_MAX_SEARCHES_PER_TURN}: the "
            "cap was read while the round was being prepared and written when it settled"
        )
        ends = [e.get("tool_call_id") for e in events if e.get("type") == "tool_end"]
        assert ends == [
            f"call_{i}" for i in range(n)
        ], f"cards closed {ends}: the capped calls overtook the ones still searching"


class TestARoundIsPreparedWholeAndBounded:
    def test_no_driver_starts_until_the_last_call_is_attached(self, monkeypatch):
        _fast_sizing(monkeypatch)
        order: list = []
        real_attach = controller_mod.ToolCallDecision.as_assistant_tool_call
        real_start = llama_mod._start_tool_call

        def _attach(self_):
            order.append(("attach", self_.tool_name))
            return real_attach(self_)

        # *args: this stands in for a production helper whose parameters have changed
        # once already, and a stale stub here fails as a TypeError rather than as this
        # test's own claim.
        def _start(decision, *args, **kwargs):
            order.append(("start", decision.tool_name))
            return real_start(decision, *args, **kwargs)

        monkeypatch.setattr(controller_mod.ToolCallDecision, "as_assistant_tool_call", _attach)
        monkeypatch.setattr(llama_mod, "_start_tool_call", _start)

        def _execute(name, arguments, **_kwargs):
            return f"RESULT<{arguments.get('query')}>"

        events, _payloads = _gguf_events(monkeypatch, _searches(3), _execute)
        ends = [e for e in events if e.get("type") == "tool_end"]
        assert [e.get("tool_call_id") for e in ends] == ["call_0", "call_1", "call_2"]

        starts = [i for i, (kind, _name) in enumerate(order) if kind == "start"]
        attaches = [i for i, (kind, _name) in enumerate(order) if kind == "attach"]
        assert len(starts) == 3 and len(attaches) >= 3
        assert max(attaches) < min(
            starts
        ), f"a driver started before the round was attached: {order}"

    def test_nine_calls_run_single_file(self, monkeypatch):
        _fast_sizing(monkeypatch)
        lock = threading.Lock()
        in_flight = [0]
        peak = [0]

        def _execute(name, arguments, **_kwargs):
            with lock:
                in_flight[0] += 1
                peak[0] = max(peak[0], in_flight[0])
            try:
                threading.Event().wait(0.05)
            finally:
                with lock:
                    in_flight[0] -= 1
            return f"RESULT<{arguments.get('query')}>"

        events, _payloads = _gguf_events(monkeypatch, _searches(9), _execute)
        assert len([e for e in events if e.get("type") == "tool_end"]) == 9
        assert peak[0] == 1, f"a round past the cap overlapped: {peak[0]} in flight at once"

    def test_eight_calls_still_overlap(self, monkeypatch):
        _fast_sizing(monkeypatch)
        events, _payloads = _gguf_events(monkeypatch, _searches(8), _meeting_tool(8))
        ends = [e for e in events if e.get("type") == "tool_end"]
        assert len(ends) == 8
        assert all(
            "TOGETHER" in result for result in _results(ends)
        ), f"a round at the cap serialised: {_results(ends)}"
