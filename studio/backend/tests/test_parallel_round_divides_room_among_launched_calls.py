# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""An overlapped round divides its result room among the calls that actually launch.

`prepare_call` turns a repeat of an earlier successful call into a no-op that stores no
result, but the round is decided from the raw call list, so a round of three whose first
entry repeats round one still ran overlapped -- and each of the two real calls was handed a
THIRD of the room instead of a half. Below `_MIN_USEFUL_RESULT_TOKENS` the result is
replaced by the notice that says it was cut, which the model reads as a failure and retries.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

import pytest  # noqa: E402

from core.inference import llama_cpp as llama_mod  # noqa: E402
from .test_llama_cpp_tool_loop import _done as _gguf_done  # noqa: E402
from .test_llama_cpp_tool_loop import _make_backend  # noqa: E402
from .test_llama_cpp_tool_loop import _sse as _gguf_sse  # noqa: E402
from .test_tool_calls_within_one_turn_overlap import _gguf_round  # noqa: E402

_ROOM = 3000


@pytest.fixture
def fixed_room(monkeypatch):
    """A known amount of result room, so the divisor the round used is readable."""
    monkeypatch.setattr(llama_mod, "tool_result_budget", lambda *_a, **_k: _ROOM)


def _budgets(monkeypatch, second_round):
    """Round one runs `alpha`; round two is `second_round`. Returns {query: budget}."""
    given: dict = {}

    def _execute(name, arguments, **kwargs):
        given[(arguments or {}).get("query")] = kwargs.get("result_budget_tokens")
        return f"RESULT<{(arguments or {}).get('query')}>"

    monkeypatch.setattr("core.inference.tools.execute_tool", _execute)
    backend = _make_backend(
        monkeypatch,
        [
            _gguf_round([("call_a", "web_search", {"query": "alpha"})]),
            _gguf_round(second_round),
            [_gguf_sse({"content": "Final answer."}), _gguf_done()],
        ],
        [],
    )
    list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "go"}],
            tools = [{"type": "function", "function": {"name": "web_search"}}],
            max_tool_iterations = 3,
        )
    )
    return given


class TestASuppressedCallHoldsNoRoom:
    def test_the_repeat_does_not_take_a_share(self, monkeypatch, fixed_room):
        given = _budgets(
            monkeypatch,
            [
                ("call_a2", "web_search", {"query": "alpha"}),
                ("call_b", "web_search", {"query": "beta"}),
                ("call_c", "web_search", {"query": "gamma"}),
            ],
        )
        assert given["beta"] == given["gamma"] == _ROOM // 2, (
            "the repeat of round one is a no-op that stores nothing, so the room is split "
            f"between the two real searches; got {given}"
        )

    def test_a_round_with_nothing_suppressed_is_unchanged(self, monkeypatch, fixed_room):
        given = _budgets(
            monkeypatch,
            [
                ("call_b", "web_search", {"query": "beta"}),
                ("call_c", "web_search", {"query": "gamma"}),
                ("call_d", "web_search", {"query": "delta"}),
            ],
        )
        assert (
            given["beta"] == given["gamma"] == given["delta"] == _ROOM // 3
        ), f"three real calls share the room three ways; got {given}"

    def test_the_only_real_call_of_the_round_gets_the_room(self, monkeypatch, fixed_room):
        given = _budgets(
            monkeypatch,
            [
                ("call_a2", "web_search", {"query": "alpha"}),
                ("call_b", "web_search", {"query": "beta"}),
            ],
        )
        assert (
            given["beta"] == _ROOM
        ), f"one launched call divides the room by one, not by two; got {given}"
