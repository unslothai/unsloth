# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""How many of one provider round's tool calls may be in flight at once.

Overlapping a round's calls is worth having: a turn that asks for three searches should
cost the longest of them and not the sum. What it cannot be is unbounded. Each overlapped
call is a `stream_tool_execution` worker with a pump task on top of it, and every one of
them starts its side effects at once, so a provider turn carrying dozens of distinct calls
-- prompt induced, or simply a model that fanned out -- multiplied threads and side
effects with nothing bounding it.

`max_tool_calls_per_message` does not bound it. At its unlimited value the budget check
above the launch never refuses a call, which is exactly the configuration this was
reported against, and the round's length is the model's choice rather than the user's.

The local GGUF loop already caps it: a round past `_MAX_PARALLEL_TOOL_CALLS_PER_ROUND`
runs single file, as every round did before overlapping existed. This is the same rule and
the same figure on the provider loop.

The bound is measured with a barrier rather than a sleep: two tools that must each see the
other before either may return can only both return if they were running together, so a
round that overlaps reports TOGETHER and one that does not reports ALONE. A machine that
is merely slow cannot turn one into the other.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)


from core.inference import studio_tool_loop as loop_mod  # noqa: E402

from .preempt_fakes import rendezvous  # noqa: E402, F401

# The scripted transport and the SSE readers the rest of the loop is tested against,
# rather than a second copy of them.
from test_studio_tool_loop import (  # noqa: E402
    WEB,
    FakeTransport,
    _DONE,
    _events,
    _run,
    _sse,
)


_CAP = loop_mod._MAX_PARALLEL_TOOL_CALLS_PER_ROUND

# The value the report was filed against: `max_tool_calls_per_message` at its unlimited
# sentinel, where nothing above the launch refuses a call.
_UNLIMITED = 9999


def _round_of(count: int) -> FakeTransport:
    """One turn asking for `count` DISTINCT calls, the shape a fanned-out model emits."""
    return FakeTransport(
        [
            [
                _sse(
                    {
                        "tool_calls": [
                            {
                                "index": index,
                                "id": f"call_{index}",
                                "function": {
                                    "name": "web_search",
                                    "arguments": json.dumps({"query": f"q{index}"}),
                                },
                            }
                            for index in range(count)
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


class TestTheCapIsTheGgufLoopsCap:
    def test_the_two_loops_agree(self):
        """One user-visible rule, so the two loops must not drift apart on it."""
        from core.inference.llama_cpp import _MAX_PARALLEL_TOOL_CALLS_PER_ROUND
        assert _CAP == _MAX_PARALLEL_TOOL_CALLS_PER_ROUND


class TestARoundPastTheCapRunsSingleFile:
    def test_nothing_overlaps(self, rendezvous):
        lines = _run(_round_of(_CAP + 1), tools = [WEB], max_calls = _UNLIMITED)
        ends = _events(lines, "tool_end")
        assert len(ends) == _CAP + 1, "every call still has to run"
        assert all("ALONE" in (end.get("result") or "") for end in ends), (
            "a round the model made arbitrarily long overlapped every one of its calls: "
            "a worker thread and a pump task each, all of their side effects at once, "
            f"with nothing bounding it. Results: {[end.get('result') for end in ends]}"
        )

    def test_the_calls_still_answer_in_order(self, rendezvous):
        transport = _round_of(_CAP + 1)
        lines = _run(transport, tools = [WEB], max_calls = _UNLIMITED)
        ends = [event.get("tool_call_id") for event in _events(lines, "tool_end")]
        assert ends == [f"call_{index}" for index in range(_CAP + 1)]
        tool_rows = [row for row in transport.requests[1]["messages"] if row.get("role") == "tool"]
        assert [row.get("tool_call_id") for row in tool_rows] == [
            f"call_{index}" for index in range(_CAP + 1)
        ], "the provider reads results by position, so a reordered history answers wrongly"


class TestARoundAtTheCapStillOverlaps:
    def test_the_bound_did_not_serialise_everything(self, rendezvous):
        """The cap must not become "never overlap": the gain this exists for is real."""
        lines = _run(_round_of(_CAP), tools = [WEB], max_calls = _UNLIMITED)
        ends = _events(lines, "tool_end")
        assert len(ends) == _CAP
        assert all(
            "TOGETHER" in (end.get("result") or "") for end in ends
        ), f"a round exactly at the cap was serialised: {[end.get('result') for end in ends]}"

    def test_two_calls_are_untouched(self, rendezvous):
        ends = _events(_run(_round_of(2), tools = [WEB], max_calls = _UNLIMITED), "tool_end")
        assert len(ends) == 2
        assert all("TOGETHER" in (end.get("result") or "") for end in ends)
