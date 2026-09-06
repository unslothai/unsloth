# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A no-op in an overlapped round must not spend the call budget.

An overlapped round has to check the budget against what it has LAUNCHED, because a
launched call has not settled yet and only settling decrements ``remaining``. Counting
the whole pending list instead counts entries that will never spend anything: a
duplicate the controller turned into a no-op, a denied call, a call an earlier budget
check already refused. All three are held in the same list so their cards land in call
order, and all three are tagged ``"lines"`` rather than ``"call"``.

The consequence is a call the model asked for, that the budget could pay for, that was
never run and came back to the provider as ``budget exhausted``. In a sequential round
-- where the pending list is always empty -- the same round runs it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

import pytest  # noqa: E402

from core.inference import studio_tool_loop as loop_mod  # noqa: E402
from test_studio_tool_loop import (  # noqa: E402
    WEB,
    FakeTransport,
    _DONE,
    _events,
    _run,
    _sse,
)


@pytest.fixture
def executed(monkeypatch):
    """Record every execute_tool call. Same shape as the loop's own fixture."""
    calls: list[dict] = []

    def _execute(name, arguments, **kwargs):
        calls.append({"name": name, "arguments": arguments})
        return f"RESULT<{name}>"

    monkeypatch.setattr(loop_mod, "execute_tool", _execute)
    monkeypatch.setattr(loop_mod, "build_rag_autoinject", lambda *a, **k: None)
    monkeypatch.setattr(loop_mod, "is_high_risk_tool_call", lambda name, args: False)
    return calls


def _calls_turn(calls):
    """One assistant turn asking for `calls` = [(id, query), ...] of web_search."""
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
    """Round one runs `alpha`. Round two repeats it, then asks for two new searches.

    The repeat is a controller no-op -- `record_result` put its key in
    `_successful_keys` -- so it costs nothing. The round's three keys are still distinct
    from each other, which is what the overlap gate asks, so the round runs in parallel.
    """
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
        """Budget 3: one for round one, two left for round two's two real searches.

        Counting the repeat's placeholder against the budget left `remaining - 3 <= 0`
        at the third call and refused `gamma` for a slot nothing had taken.
        """
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
        assert not any("budget" in result.lower() for result in results), (
            f"a call was refused for a budget that was not spent: {results}"
        )

    def test_a_budget_that_really_is_spent_still_refuses(self, executed):
        """The guard the counting exists for, unchanged.

        With one call for the whole response, round one takes it and round two's real
        searches must not run: launching them would spend a budget the loop had already
        refused, side effects and all.
        """
        _run(_transport_with_a_repeat(), max_calls = 1, tools = [WEB])
        assert len(executed) == 1
