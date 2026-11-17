# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A tool result is priced from the token count the provider reports (#18185)."""

from __future__ import annotations

from types import SimpleNamespace

from core.inference.context_window import (
    _reply_floor,
    estimate_messages_tokens_conservative,
    tool_result_budget,
)
from core.inference.studio_tool_loop import _result_budget_tokens

_WINDOW = 128_000
_PROSE = "The quick brown fox jumps over the lazy dog while the committee reviewed numbers. "
_CALLS = [{"function": {"name": "search_conversation"}, "arguments": '{"query": "x"}'}]


def _turn(usage = None):
    return SimpleNamespace(text = [""], reasoning = [""], usage = usage)


def _thread(result_chars):
    """About 50K real tokens of prose, then one earlier tool exchange of ``result_chars``."""
    messages = [{"role": "system", "content": "You are helpful."}]
    for _ in range(10):
        messages.append({"role": "user", "content": _PROSE * (11_000 // len(_PROSE))})
        messages.append({"role": "assistant", "content": _PROSE * (11_000 // len(_PROSE))})
    if result_chars:
        messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "c0",
                        "type": "function",
                        "function": {"name": "web_search", "arguments": '{"query": "a"}'},
                    }
                ],
            }
        )
        messages.append(
            {"role": "tool", "tool_call_id": "c0", "content": "result. " * (result_chars // 8)}
        )
    return messages


def _budget(
    turn,
    thread,
    tool_messages = (),
    calls = _CALLS,
    index = 0,
    max_tokens = None,
):
    return _result_budget_tokens(
        _WINDOW, max_tokens, thread, [], turn, calls, list(tool_messages), index
    )


def test_estimate_alone_starves_a_half_full_window():
    assert _budget(_turn(), _thread(80_000)) == 0


def test_provider_count_leaves_the_room_the_window_has():
    usage = {"prompt_tokens": 70_000, "completion_tokens": 40, "total_tokens": 70_040}
    budget = _budget(_turn(usage), _thread(80_000))

    assert budget == tool_result_budget(_WINDOW, _reply_floor(_WINDOW), 70_040) // 2
    assert budget > 20_000


def test_context_tokens_wins_over_prompt_and_completion():
    usage = {"prompt_tokens": 500, "completion_tokens": 40, "context_tokens": 90_000}

    assert _budget(_turn(usage), _thread(0)) == tool_result_budget(_WINDOW, None, 90_000)


def test_results_already_in_this_batch_are_still_priced():
    usage = {"prompt_tokens": 60_000, "completion_tokens": 40}
    earlier = [{"role": "tool", "tool_call_id": "c0", "content": "x" * 20_000}]
    two = _CALLS * 2

    fresh = _budget(_turn(usage), _thread(0), calls = two, index = 0)
    after = _budget(_turn(usage), _thread(0), earlier, calls = two, index = 1)
    held = estimate_messages_tokens_conservative(earlier, dense_ascii = True)

    assert held > 0
    assert after < fresh * 2
    assert after == min(
        tool_result_budget(_WINDOW, None, 60_040 + held),
        tool_result_budget(_WINDOW, _reply_floor(_WINDOW), 60_040 + 2 * held) // 2,
    )


def test_a_count_past_the_window_still_gives_zero():
    usage = {"prompt_tokens": 127_000, "completion_tokens": 900}

    assert _budget(_turn(usage), _thread(0)) == 0


def test_completion_only_report_falls_back_to_the_estimate():
    only_completion = _turn({"completion_tokens": 40})

    assert _budget(only_completion, _thread(80_000)) == _budget(_turn(), _thread(80_000)) == 0


def test_non_dict_usage_falls_back_to_the_estimate():
    thread = _thread(20_000)

    assert _budget(_turn("garbage"), thread) == _budget(_turn(), thread)
