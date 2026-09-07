# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A parallel round attaches every call before any driver starts, and stays bounded.

Starting a call's driver while the loop was still appending the later calls let a
compaction inside that tool rebuild the transcript and detach the assistant message the
loop was writing to; the later calls then went onto a dictionary the transcript no longer
held. And a round overlapped every structured call it was given, so a response carrying
dozens of them multiplied driver threads and side effects with nothing bounding it.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from core.inference import llama_cpp as llama_mod  # noqa: E402
from core.inference import tool_loop_controller as controller_mod  # noqa: E402

from test_tool_calls_within_one_turn_overlap import _gguf_events  # noqa: E402


def _fast_sizing(monkeypatch):
    """The fake backend has no llama-server to count against, and every call's sizing
    would otherwise wait out an HTTP failure; a flat estimate keeps the starts together so
    the barriers below measure the round, not the harness."""
    monkeypatch.setattr(
        llama_mod.LlamaCppBackend,
        "count_chat_tokens",
        lambda self, messages, *args, **kwargs: 8 * len(messages or []),
    )


class TestEveryCallIsAttachedBeforeAnyStarts:
    def test_no_driver_starts_until_the_last_call_is_attached(self, monkeypatch):
        _fast_sizing(monkeypatch)
        order: list = []
        real_attach = controller_mod.ToolCallDecision.as_assistant_tool_call
        real_start = llama_mod._start_tool_call

        def _attach(self_):
            order.append(("attach", self_.tool_name))
            return real_attach(self_)

        def _start(decision, stream, budget_cell, starved_cell):
            order.append(("start", decision.tool_name))
            return real_start(decision, stream, budget_cell, starved_cell)

        monkeypatch.setattr(controller_mod.ToolCallDecision, "as_assistant_tool_call", _attach)
        monkeypatch.setattr(llama_mod, "_start_tool_call", _start)

        def _execute(name, arguments, **_kwargs):
            return f"RESULT<{arguments.get('query')}>"

        events, _payloads = _gguf_events(
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

        starts = [i for i, (kind, _name) in enumerate(order) if kind == "start"]
        attaches = [i for i, (kind, _name) in enumerate(order) if kind == "attach"]
        assert len(starts) == 3 and len(attaches) >= 3
        assert max(attaches) < min(
            starts
        ), f"a driver started before the round was attached: {order}"


class TestARoundIsBounded:
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

        calls = [(f"call_{i}", "web_search", {"query": f"q{i}"}) for i in range(9)]
        events, _payloads = _gguf_events(monkeypatch, calls, _execute)
        ends = [e for e in events if e.get("type") == "tool_end"]
        assert len(ends) == 9
        assert peak[0] == 1, f"a round past the cap overlapped: {peak[0]} in flight at once"

    def test_eight_calls_still_overlap(self, monkeypatch):
        _fast_sizing(monkeypatch)
        barrier = threading.Barrier(8, timeout = 4)

        def _execute(name, arguments, **_kwargs):
            try:
                barrier.wait()
            except threading.BrokenBarrierError:
                return f"ALONE<{arguments.get('query')}>"
            return f"TOGETHER<{arguments.get('query')}>"

        calls = [(f"call_{i}", "web_search", {"query": f"q{i}"}) for i in range(8)]
        events, _payloads = _gguf_events(monkeypatch, calls, _execute)
        ends = [e for e in events if e.get("type") == "tool_end"]
        assert len(ends) == 8
        assert all(
            "TOGETHER" in (end.get("result") or "") for end in ends
        ), f"a round at the cap serialised: {[end.get('result') for end in ends]}"
