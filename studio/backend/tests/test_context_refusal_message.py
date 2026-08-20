# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the context-overflow message picking advice the user can act on.

llama-server reports one number for the whole prompt and advises shortening the
conversation. On a two-message thread whose single turn is oversized that advice is
useless, and on a tool result it is worse than useless: the user did not write it. These
tests pin the three wordings and the conditions under which each is chosen.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from core.inference import context_refusal  # noqa: E402
from routes.inference import (  # noqa: E402
    _accumulate_context_truncation,
    _context_truncated_sse_chunk,
    _friendly_error,
)

_SERVER_ERROR = "the request (7153 tokens) exceeds the available context size (5120 tokens)"


@pytest.fixture(autouse = True)
def _no_carried_refusal():
    """Each test starts with no diagnosis, and leaves none behind."""
    context_refusal.clear()
    yield
    context_refusal.clear()


def _refusal(
    *,
    irreducible: int,
    latest_turn: int,
    role: str = "user",
    context_length: int = 5120,
) -> dict:
    return {
        "fits": False,
        "dropped_messages": 0,
        "irreducible_tokens": irreducible,
        "latest_turn_tokens": latest_turn,
        "latest_turn_role": role,
        "context_length": context_length,
        "prompt_target": context_length - 1024,
    }


# ---------------------------------------------------------------- wording


def test_no_diagnosis_keeps_the_generic_advice():
    message = _friendly_error(ValueError(_SERVER_ERROR))
    assert "Message too long: 7153 tokens exceeds the 5120-token context window." in message
    assert "shorten the conversation" in message


def test_long_history_keeps_the_generic_advice():
    # A 40-turn thread: the newest turn is a small part of what could not be evicted, so
    # shortening the conversation is exactly the right thing to suggest.
    context_refusal.record_fit(_refusal(irreducible = 5000, latest_turn = 300))
    message = _friendly_error(ValueError(_SERVER_ERROR))
    assert "shorten the conversation" in message
    assert "does not fit on its own" not in message


def test_single_oversized_turn_says_shortening_will_not_help():
    context_refusal.record_fit(_refusal(irreducible = 5000, latest_turn = 4800))
    message = _friendly_error(ValueError(_SERVER_ERROR))
    assert "The message just sent does not fit on its own" in message
    assert "shortening the conversation will not help" in message
    assert "Increase the Context Length in Model settings" in message


def test_oversized_tool_result_names_the_tool():
    context_refusal.record_fit(
        _refusal(irreducible = 5000, latest_turn = 4800, role = "tool")
    )
    message = _friendly_error(ValueError(_SERVER_ERROR))
    assert "A tool returned more than this context window can hold" in message
    assert "smaller slice" in message
    # The user cannot shorten what a tool wrote, so it must never be suggested.
    assert "send it in smaller pieces" not in message


def test_function_role_is_treated_as_a_tool_result():
    context_refusal.record_fit(
        _refusal(irreducible = 5000, latest_turn = 4800, role = "function")
    )
    assert "A tool returned" in _friendly_error(ValueError(_SERVER_ERROR))


def test_every_wording_keeps_the_counts_and_the_client_markers():
    # `isContextLimitError` in chat-adapter.ts matches on these substrings, and the
    # numbers are the only concrete thing the user has to size the window by.
    for refusal in (
        None,
        _refusal(irreducible = 5000, latest_turn = 300),
        _refusal(irreducible = 5000, latest_turn = 4800),
        _refusal(irreducible = 5000, latest_turn = 4800, role = "tool"),
    ):
        context_refusal.clear()
        if refusal is not None:
            context_refusal.record_fit(refusal)
        message = _friendly_error(ValueError(_SERVER_ERROR))
        assert "Message too long" in message
        assert "context window" in message
        assert "Context Length" in message
        assert "7153" in message and "5120" in message


# ---------------------------------------------------------------- selection


@pytest.mark.parametrize(
    "latest_turn,dominates",
    [(3379, False), (3380, True), (5000, True)],
)
def test_the_turn_has_to_dominate_the_floor(latest_turn, dominates):
    # Two thirds of the irreducible prompt. Below it the system prompt and the older
    # turns are a real share of the problem and the generic advice still applies.
    context_refusal.record_fit(_refusal(irreducible = 5120, latest_turn = latest_turn))
    message = _friendly_error(ValueError(_SERVER_ERROR))
    assert ("does not fit on its own" in message) is dominates


def test_a_diagnosis_for_a_different_window_is_ignored():
    # A model reload between the fit and the error. The recorded shape describes a
    # window the server did not just refuse, so it must not narrate this one.
    context_refusal.record_fit(
        _refusal(irreducible = 5000, latest_turn = 4800, context_length = 8192)
    )
    assert "shorten the conversation" in _friendly_error(ValueError(_SERVER_ERROR))


def test_a_diagnosis_with_no_window_recorded_is_still_usable():
    refusal = _refusal(irreducible = 5000, latest_turn = 4800)
    refusal.pop("context_length")
    context_refusal.record_fit(refusal)
    assert "does not fit on its own" in _friendly_error(ValueError(_SERVER_ERROR))


@pytest.mark.parametrize("field", ["irreducible_tokens", "latest_turn_tokens"])
def test_a_diagnosis_missing_its_counts_falls_back(field):
    refusal = _refusal(irreducible = 5000, latest_turn = 4800)
    refusal[field] = 0
    context_refusal.record_fit(refusal)
    assert "shorten the conversation" in _friendly_error(ValueError(_SERVER_ERROR))


def test_unparsable_counts_do_not_raise():
    refusal = _refusal(irreducible = 5000, latest_turn = 4800)
    refusal["irreducible_tokens"] = "lots"
    context_refusal.record_fit(refusal)
    assert "shorten the conversation" in _friendly_error(ValueError(_SERVER_ERROR))


# ---------------------------------------------------------------- recording


def test_a_fit_that_succeeded_clears_an_earlier_refusal():
    # A tool loop refuses on one iteration and fits on the next. The stale refusal must
    # not be left behind to explain an error from the iteration that fitted.
    context_refusal.record_fit(_refusal(irreducible = 5000, latest_turn = 4800))
    context_refusal.record_fit({"fits": True, "dropped_messages": 4})
    assert context_refusal.latest_refusal() is None
    assert "shorten the conversation" in _friendly_error(ValueError(_SERVER_ERROR))


def test_non_dict_events_are_ignored():
    context_refusal.record_fit(_refusal(irreducible = 5000, latest_turn = 4800))
    for value in (None, "fits", 7, ["fits"]):
        context_refusal.record_fit(value)
    assert context_refusal.latest_refusal() is not None


def test_the_sse_chunk_records_the_refusal_it_forwards():
    refusal = _refusal(irreducible = 5000, latest_turn = 4800)
    line = _context_truncated_sse_chunk("cmpl-1", "model", refusal)
    assert "context_truncated" in line
    assert context_refusal.latest_refusal() == refusal


def test_the_sse_chunk_clears_on_a_fit_that_succeeded():
    context_refusal.record_fit(_refusal(irreducible = 5000, latest_turn = 4800))
    _context_truncated_sse_chunk("cmpl-1", "model", {"fits": True, "dropped_messages": 2})
    assert context_refusal.latest_refusal() is None


def test_the_drain_records_each_fit_not_the_running_total():
    # `_accumulate_context_truncation` sums `dropped_messages` across a tool loop. The
    # refusal must be the per-fit event, or a loop would report counts no fit produced.
    first = {"type": "context_truncated", "fits": True, "dropped_messages": 4}
    second = {"type": "context_truncated", **_refusal(irreducible = 5000, latest_turn = 4800)}
    combined = _accumulate_context_truncation(None, first)
    combined = _accumulate_context_truncation(combined, second)
    assert combined["dropped_messages"] == 4
    recorded = context_refusal.latest_refusal()
    assert recorded is not None
    assert recorded["dropped_messages"] == 0
    assert recorded["latest_turn_tokens"] == 4800


def test_the_recorded_diagnosis_is_a_copy():
    refusal = _refusal(irreducible = 5000, latest_turn = 4800)
    context_refusal.record_fit(refusal)
    refusal["latest_turn_tokens"] = 1
    assert context_refusal.latest_refusal()["latest_turn_tokens"] == 4800


def test_other_friendly_errors_are_untouched():
    assert _friendly_error(RuntimeError("unrelated")) == "An internal error occurred"
    assert "Lost connection" in _friendly_error(
        RuntimeError("Lost connection to llama-server")
    )
