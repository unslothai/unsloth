# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A no-op in an overlapped round must not spend the call budget."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)



from .preempt_fakes import executed  # noqa: E402, F401
from test_studio_tool_loop import (  # noqa: E402
    WEB,
    FakeTransport,
    _DONE,
    _events,
    _run,
    _sse,
)


def _calls_turn(calls):
    return [
        _sse(
            {
                "tool_calls": [
                    {
                        "index": i,
                        "id": cid,
                        "function": {
                            "name": "web_search",
                            "arguments": json.dumps({"query": query}),
                        },
                    }
                    for i, (cid, query) in enumerate(calls)
                ]
            }
        ),
        _sse(finish = "tool_calls"),
        _DONE,
    ]


def _transport_with_a_repeat():
    return FakeTransport(
        [
            _calls_turn([("call_a", "alpha")]),
            _calls_turn([("call_a2", "alpha"), ("call_b", "beta"), ("call_c", "gamma")]),
            [_sse({"content": "done"}), _sse(finish = "stop"), _DONE],
        ],
        heals = False,
    )


class TestABudgetIsSpentByLaunchesOnly:
    def test_a_repeat_beside_two_new_calls_does_not_refuse_the_second(self, executed):
        lines = _run(_transport_with_a_repeat(), max_calls = 3, tools = [WEB])

        def _query(arguments):
            parsed = arguments if isinstance(arguments, dict) else json.loads(arguments)
            return parsed["query"]

        queries = [_query(call["arguments"]) for call in executed]
        # A set: an overlapped round's tools start together, so which of the two new
        # searches reaches `execute_tool` first is not a property this owns.
        assert sorted(queries) == ["alpha", "beta", "gamma"], (
            "the repeat is a no-op and spends nothing, so both new searches fit the "
            f"remaining budget of two; got {queries}"
        )

    def test_no_call_comes_back_as_budget_exhausted(self, executed):
        lines = _run(_transport_with_a_repeat(), max_calls = 3, tools = [WEB])
        results = [end.get("result") or "" for end in _events(lines, "tool_end")]
        assert not any(
            "budget" in result.lower() for result in results
        ), f"a call was refused for a budget that was not spent: {results}"

    def test_a_budget_that_really_is_spent_still_refuses(self, executed):
        _run(_transport_with_a_repeat(), max_calls = 1, tools = [WEB])
        assert len(executed) == 1
