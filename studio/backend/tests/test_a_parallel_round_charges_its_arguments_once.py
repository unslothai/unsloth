# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""An overlapped round's arguments are charged against its results once, not twice."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

import pytest  # noqa: E402

from core.inference.llama_cpp import LlamaCppBackend  # noqa: E402
from .test_tool_calls_within_one_turn_overlap import _gguf_events  # noqa: E402


@pytest.fixture(autouse = True)
def parallel_opted_in(monkeypatch):
    """Overlap is opt-in where preemption is not armed; these cases test an overlapped round."""
    monkeypatch.setenv("UNSLOTH_PARALLEL_TOOL_CALLS", "1")


@pytest.fixture
def counted(monkeypatch):
    """A tokenizer that prices whatever it is handed, so the count is inspectable."""

    def _count(self, messages, *_a, **_k):
        return len(json.dumps(messages, default = str)) // 4

    monkeypatch.setattr(LlamaCppBackend, "count_chat_tokens", _count, raising = False)


def _budgets(monkeypatch, pad_chars = 900):
    """Run one three-call round and return {query: result_budget_tokens}."""
    given: dict = {}

    def _execute(name, arguments, **kwargs):
        given[(arguments or {}).get("query")] = kwargs.get("result_budget_tokens")
        return "ok"

    pad = "X" * pad_chars
    _gguf_events(
        monkeypatch,
        [
            ("call_a", "web_search", {"query": "alpha", "pad": pad}),
            ("call_b", "web_search", {"query": "beta", "pad": pad}),
            ("call_c", "web_search", {"query": "gamma", "pad": pad}),
        ],
        _execute,
    )
    return given


class TestEveryCallOfTheRoundIsPricedTheSame:
    def test_the_calls_share_the_round(self, monkeypatch, counted):
        given = _budgets(monkeypatch)

        assert sorted(given) == ["alpha", "beta", "gamma"]
        assert len(set(given.values())) == 1, (
            "the round's calls are all attached before any of them runs, so they price "
            f"against the same prompt and get the same share; got {given}"
        )

    def test_no_call_is_charged_the_others_arguments(self, monkeypatch, counted):
        given = _budgets(monkeypatch)
        assert given["alpha"] == given["gamma"], (
            "the first call was charged the arguments of the two behind it, which its own "
            f"prompt count already rendered; got {given}"
        )
