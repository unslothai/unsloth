# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the context-overflow message picking advice the user can act on.

llama-server reports one number for the whole prompt and advises shortening the
conversation. On a two-message thread whose single turn is oversized that advice is
useless, and on a tool result it is worse than useless: the user did not write it. These
tests pin each wording and the conditions under which it is chosen, including the two
thresholds: whose turn is the bulk of the prompt, and whether that turn could have been
sent at all.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from core.inference import context_refusal  # noqa: E402
from core.inference.context_window import fit_rolling_context  # noqa: E402
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
    prompt_target: int = 4096,
) -> dict:
    return {
        "fits": False,
        "dropped_messages": 0,
        "irreducible_tokens": irreducible,
        "latest_turn_tokens": latest_turn,
        "latest_turn_role": role,
        "context_length": context_length,
        "prompt_target": prompt_target,
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
    context_refusal.record_fit(_refusal(irreducible = 5600, latest_turn = 5400))
    message = _friendly_error(ValueError(_SERVER_ERROR))
    assert "The message just sent does not fit on its own" in message
    assert "shortening the conversation will not help" in message
    assert "Increase the Context Length in Model settings" in message


def test_oversized_tool_result_names_the_tool():
    context_refusal.record_fit(_refusal(irreducible = 5600, latest_turn = 5400, role = "tool"))
    message = _friendly_error(ValueError(_SERVER_ERROR))
    assert "A tool returned more than this context window can hold" in message
    assert "smaller slice" in message
    # The user cannot shorten what a tool wrote, so it must never be suggested.
    assert "send it in smaller pieces" not in message


def test_function_role_is_treated_as_a_tool_result():
    context_refusal.record_fit(_refusal(irreducible = 5600, latest_turn = 5400, role = "function"))
    assert "A tool returned" in _friendly_error(ValueError(_SERVER_ERROR))


def test_an_oversized_assistant_prefill_does_not_ask_the_user_to_split_it():
    # Reachable through auto-continue, which resends the truncated reply as the final
    # assistant message. The user did not write it and cannot send it in pieces.
    context_refusal.record_fit(_refusal(irreducible = 5600, latest_turn = 5400, role = "assistant"))
    message = _friendly_error(ValueError(_SERVER_ERROR))
    assert "The reply being continued is already too long for this window" in message
    assert "start a new reply" in message
    assert "send it in smaller pieces" not in message


@pytest.mark.parametrize("role", ["system", "developer"])
def test_oversized_instructions_point_at_the_system_prompt(role):
    # System and developer turns survive eviction, so splitting one across messages
    # preserves the total and resolves nothing.
    context_refusal.record_fit(_refusal(irreducible = 5600, latest_turn = 5400, role = role))
    message = _friendly_error(ValueError(_SERVER_ERROR))
    assert "The system instructions do not fit on their own" in message
    assert "shorten the system prompt" in message
    assert "send it in smaller pieces" not in message


def test_a_dominating_assistant_prefill_hedges_the_same_way():
    context_refusal.record_fit(_refusal(irreducible = 5120, latest_turn = 3500, role = "assistant"))
    message = _friendly_error(ValueError(_SERVER_ERROR))
    assert "Most of this prompt is the reply being continued" in message
    assert "shortening the conversation will not help much" in message


@pytest.mark.parametrize("role", ["", "moderator"])
def test_an_unnameable_role_falls_back_to_the_generic_advice(role):
    # Better to give advice that is merely unspecific than advice aimed at the wrong turn.
    context_refusal.record_fit(_refusal(irreducible = 5600, latest_turn = 5400, role = role))
    assert "shorten the conversation" in _friendly_error(ValueError(_SERVER_ERROR))


def test_every_wording_keeps_the_counts_and_the_client_markers():
    # `isContextLimitError` in chat-adapter.ts matches on these substrings, and the
    # numbers are the only concrete thing the user has to size the window by.
    for refusal in (
        None,
        _refusal(irreducible = 5000, latest_turn = 300),
        _refusal(irreducible = 5000, latest_turn = 4800),
        _refusal(irreducible = 5000, latest_turn = 4800, role = "tool"),
        _refusal(irreducible = 5600, latest_turn = 5400, role = "tool"),
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
    "latest_turn,expected",
    [
        # Below two thirds of the irreducible prompt the older turns and the system
        # prompt are a real share of the problem, so the generic advice still applies.
        (3379, "shorten the conversation"),
        # Over that share and inside the 5120-token window: the turn is the bulk of the
        # prompt and would still have been served by itself, so say only that. Note 4097,
        # over the 4096-token PROMPT BUDGET: the fit refuses it, llama-server would not.
        (3380, "Most of this prompt is the message just sent"),
        (4097, "Most of this prompt is the message just sent"),
        (5119, "Most of this prompt is the message just sent"),
        # At the window: llama-server refuses on the prompt size alone, so it cannot be
        # sent at all, whatever else is in the window.
        (5120, "does not fit on its own"),
    ],
)
def test_dominating_the_floor_is_not_the_same_as_not_fitting(latest_turn, expected):
    context_refusal.record_fit(_refusal(irreducible = 5120, latest_turn = latest_turn))
    assert expected in _friendly_error(ValueError(_SERVER_ERROR))


def test_a_turn_that_merely_dominates_hedges_its_advice():
    # It is the bulk of the prompt, so trimming the rest buys little. "Will not help"
    # would be a claim the numbers do not support.
    context_refusal.record_fit(_refusal(irreducible = 5120, latest_turn = 3500))
    message = _friendly_error(ValueError(_SERVER_ERROR))
    assert "shortening the conversation will not help much" in message
    assert "send it in smaller pieces" in message


def test_a_dominating_tool_result_hedges_the_same_way():
    context_refusal.record_fit(_refusal(irreducible = 5120, latest_turn = 3500, role = "tool"))
    message = _friendly_error(ValueError(_SERVER_ERROR))
    assert "Most of this prompt is a single tool result" in message
    assert "shortening the conversation will not help much" in message
    assert "smaller slice" in message


@pytest.mark.parametrize("role", ["user", "tool", "assistant", "system"])
def test_a_turn_the_window_could_have_held_is_never_called_too_big(role):
    """The reply reservation is not part of what the window "can hold".

    `prompt_budget` hands the prompt the window minus room for the reply (up to a
    quarter of it), so a 5120-token window with Max Tokens 1024 refuses the prompt at
    4096. But llama-server admits a prompt on its size alone -- `n_tokens >= n_ctx`,
    nothing reserved -- so a 4800-token turn in that window IS servable by itself, and
    every hard wording here would be a false claim about it.
    """
    context_refusal.record_fit(
        _refusal(irreducible = 5000, latest_turn = 4800, role = role, prompt_target = 4096)
    )
    message = _friendly_error(ValueError(_SERVER_ERROR))
    assert "Most of this prompt is" in message
    for false_claim in (
        "does not fit on its own",
        "do not fit on their own",
        "more than this context window can hold",
        "already too long for this window",
    ):
        assert false_claim not in message


def test_a_recorded_prompt_budget_does_not_move_the_hard_boundary():
    # Two fits of the same turn in the same window, differing only in what Max Tokens
    # reserved. The window is what "can hold" means, so both read the same.
    with_budget = _refusal(irreducible = 5000, latest_turn = 4800, prompt_target = 4096)
    without_budget = dict(with_budget)
    without_budget.pop("prompt_target")
    context_refusal.record_fit(with_budget)
    first = _friendly_error(ValueError(_SERVER_ERROR))
    context_refusal.record_fit(without_budget)
    assert _friendly_error(ValueError(_SERVER_ERROR)) == first


def test_a_diagnosis_for_a_different_window_is_ignored():
    # A model reload between the fit and the error. The recorded shape describes a
    # window the server did not just refuse, so it must not narrate this one.
    context_refusal.record_fit(_refusal(irreducible = 5000, latest_turn = 4800, context_length = 8192))
    assert "shorten the conversation" in _friendly_error(ValueError(_SERVER_ERROR))


# ------------------------------------------------- the floor both counts stand on


def _tool_catalogue_counter(catalogue_tokens: int):
    """`count_chat_tokens(fitted, None, safe_tools)` in miniature.

    The real counter renders a PROMPT: llama-server's /apply-template writes the tool
    catalogue into the system turn of whatever messages it is handed, so the catalogue is
    a constant on top of any slice -- including the one-message slice the fit prices to
    find the newest turn.
    """

    def count(messages):
        body = sum(max(1, len(json.dumps(message, ensure_ascii = False)) // 4) for message in messages)
        return body + catalogue_tokens

    return count


def _thread(*, system_tokens: int, turn_tokens: int, role: str = "user", history_turns: int = 6):
    messages = [{"role": "system", "content": "s" * (system_tokens * 4)}]
    for index in range(history_turns):
        messages.append({"role": "user", "content": f"q{index} " + "x" * 1200})
        messages.append({"role": "assistant", "content": f"a{index} " + "y" * 1200})
    messages.append({"role": role, "content": "z" * (turn_tokens * 4)})
    return messages


def _refuse_and_explain(*, window: int, catalogue: int, system_tokens: int, turn_tokens: int, role: str = "user"):
    """Drive the real path: fit -> recorded diagnosis -> the message the user reads."""
    _, truncation = fit_rolling_context(
        _thread(system_tokens = system_tokens, turn_tokens = turn_tokens, role = role),
        context_length = window,
        max_tokens = None,
        count_tokens = _tool_catalogue_counter(catalogue),
    )
    assert truncation is not None and not truncation["fits"]
    _context_truncated_sse_chunk("cmpl-1", "model", truncation)
    return truncation, _friendly_error(
        ValueError(f"the request (9000 tokens) exceeds the available context size ({window} tokens)")
    )


def test_a_tool_catalogue_is_not_the_message_just_sent():
    """A 20-token "hi" beside a large MCP catalogue is not what the user should shorten.

    Both counts in the diagnosis price a whole prompt, so both carry the catalogue: the
    turn reads as 97% of the irreducible prompt while contributing 20 tokens of it. The
    remedy is fewer tools or a bigger window, and neither is "send it in smaller pieces".
    """
    truncation, message = _refuse_and_explain(
        window = 8192, catalogue = 6000, system_tokens = 200, turn_tokens = 20
    )
    # The raw counts really are that lopsided; the floor is what makes them so.
    assert truncation["latest_turn_tokens"] > 0.9 * truncation["irreducible_tokens"]
    assert truncation["shared_prompt_tokens"] == 6000
    assert "shorten the conversation" in message
    assert "message just sent" not in message


def test_a_catalogue_bigger_than_the_window_never_makes_a_tiny_turn_unsendable():
    # The false claim, not merely the unhelpful one: a catalogue over the window makes
    # the one-message count clear the window on its own, so the turn is reported as
    # impossible to send when it is twenty tokens.
    truncation, message = _refuse_and_explain(
        window = 4096, catalogue = 4200, system_tokens = 200, turn_tokens = 20
    )
    assert truncation["latest_turn_tokens"] > truncation["context_length"]
    assert "does not fit on its own" not in message
    assert "shorten the conversation" in message


@pytest.mark.parametrize(
    "turn_tokens,expected",
    [
        # Still the bulk of what is left once the catalogue is off both sides.
        (5000, "Most of this prompt is the message just sent"),
        # And still bigger than the window on its own, catalogue or no catalogue.
        (8300, "does not fit on its own"),
    ],
)
def test_a_catalogue_does_not_cost_a_turn_that_really_is_the_problem(turn_tokens, expected):
    _, message = _refuse_and_explain(
        window = 8192, catalogue = 1500, system_tokens = 200, turn_tokens = turn_tokens
    )
    assert expected in message


def test_a_tool_result_beside_a_catalogue_is_judged_on_its_own_size():
    # The same trap on the role the user cannot edit: a small tool result must not be
    # reported as a tool returning more than the window can hold.
    _, small = _refuse_and_explain(
        window = 8192, catalogue = 6000, system_tokens = 200, turn_tokens = 20, role = "tool"
    )
    assert "tool result" not in small and "shorten the conversation" in small
    _, large = _refuse_and_explain(
        window = 8192, catalogue = 1500, system_tokens = 200, turn_tokens = 5000, role = "tool"
    )
    assert "Most of this prompt is a single tool result" in large


def test_the_floor_is_never_all_of_either_count():
    # A counter that cannot price an empty prompt, or one whose floor is nonsense, must
    # not drive either side to zero and invent a ratio.
    context_refusal.record_fit(
        _refusal(irreducible = 5120, latest_turn = 5000) | {"shared_prompt_tokens": 99999}
    )
    assert "shorten the conversation" in _friendly_error(ValueError(_SERVER_ERROR))
    for bad in (None, "", -5, "junk"):
        context_refusal.record_fit(_refusal(irreducible = 5120, latest_turn = 3500) | {"shared_prompt_tokens": bad})
        assert "Most of this prompt is the message just sent" in _friendly_error(
            ValueError(_SERVER_ERROR)
        )


def test_an_unrenderable_turn_records_no_floor_to_subtract():
    """The estimate fallback prices the message's own JSON and no catalogue.

    Strict templates reject a lone tool result, which is exactly the shape a tool loop
    refuses on, so the fit falls back to the estimator for that turn. There is no shared
    floor inside that number, so none is recorded, and the count stays comparable to
    nothing -- which lands on the generic advice rather than a wrong blame.
    """

    def _rejects_a_lone_tool_result(messages):
        if len(messages) == 1 and messages[0].get("role") == "tool":
            raise RuntimeError("template rejected the message")
        return sum(max(1, len(json.dumps(m, ensure_ascii = False)) // 4) for m in messages) + 6000

    _, truncation = fit_rolling_context(
        _thread(system_tokens = 200, turn_tokens = 20, role = "tool"),
        context_length = 8192,
        max_tokens = None,
        count_tokens = _rejects_a_lone_tool_result,
    )
    assert truncation is not None and not truncation["fits"]
    assert truncation["shared_prompt_tokens"] == 0
    _context_truncated_sse_chunk("cmpl-1", "model", truncation)
    assert "shorten the conversation" in _friendly_error(
        ValueError("the request (9000 tokens) exceeds the available context size (8192 tokens)")
    )


def test_a_diagnosis_with_no_window_recorded_is_still_usable():
    # The server's own number stands in for the window it did not record.
    refusal = _refusal(irreducible = 5600, latest_turn = 5400)
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


# ---------------------------------------------------------------- worker threads


def _record_in_worker():
    context_refusal.record_fit(_refusal(irreducible = 5600, latest_turn = 5400))
    return "drained"


def _record_then_fail():
    _record_in_worker()
    raise ValueError(_SERVER_ERROR)


async def _drain_like_the_route(func):
    """The exact shape both non-streaming GGUF drains use: a task around a thread.

    Two context copies between the record and the read, which is what makes the slot
    necessary. Anything less than this shape does not test the thing that broke.
    """
    task = asyncio.create_task(asyncio.to_thread(func))
    return await asyncio.shield(task)


def test_without_a_slot_the_drain_loses_the_refusal():
    # The behaviour being worked around. `asyncio.create_task` and `asyncio.to_thread`
    # each copy the context, and a `.set()` in a copy never reaches the request.
    async def _run():
        await _drain_like_the_route(_record_in_worker)
        return context_refusal.latest_refusal()

    assert asyncio.run(_run()) is None


def test_a_slot_carries_the_refusal_back_through_task_and_thread():
    async def _run():
        context_refusal.open_slot()
        assert await _drain_like_the_route(_record_in_worker) == "drained"
        return context_refusal.latest_refusal()

    # Read inside the coroutine: `asyncio.run` gives it its own context copy, exactly as
    # a request task does, and that is the context `_friendly_error` will read from.
    assert asyncio.run(_run())["latest_turn_tokens"] == 5400


def test_a_slot_carries_the_refusal_back_when_the_drain_raises():
    # The path that matters: the drain diagnoses the refusal and then the request fails
    # with the oversize error that refusal explains, so there is no return value.
    async def _run():
        context_refusal.open_slot()
        with pytest.raises(ValueError):
            await _drain_like_the_route(_record_then_fail)
        return _friendly_error(ValueError(_SERVER_ERROR))

    assert "does not fit on its own" in asyncio.run(_run())


def test_a_drain_that_records_nothing_leaves_the_slot_empty():
    def _quiet():
        return 1

    async def _run():
        context_refusal.open_slot()
        await _drain_like_the_route(_quiet)
        return context_refusal.latest_refusal()

    assert asyncio.run(_run()) is None


def test_a_drain_that_fits_clears_an_earlier_refusal_through_the_slot():
    def _fits():
        context_refusal.record_fit({"fits": True, "dropped_messages": 3})

    async def _run():
        context_refusal.open_slot()
        context_refusal.record_fit(_refusal(irreducible = 5000, latest_turn = 4800))
        await _drain_like_the_route(_fits)
        return context_refusal.latest_refusal()

    assert asyncio.run(_run()) is None


def test_opening_a_slot_starts_empty():
    # Two requests on one connection: the second must not inherit the first's refusal.
    context_refusal.record_fit(_refusal(irreducible = 5000, latest_turn = 4800))
    context_refusal.open_slot()
    assert context_refusal.latest_refusal() is None


def test_both_non_streaming_gguf_drains_open_a_slot_first():
    # Guards the two call sites named in review: dropping the `open_slot` line would
    # silently restore the generic advice on both non-streaming paths.
    source = (Path(_BACKEND_DIR) / "routes" / "inference.py").read_text()
    for drain in ("_drain_gguf_tool_loop", "_drain_gguf_choices"):
        spawn = f"asyncio.create_task(asyncio.to_thread({drain}))"
        assert spawn in source
        preceding = source.split(spawn)[0].splitlines()[-4:]
        assert any("context_refusal.open_slot()" in line for line in preceding)


def test_other_friendly_errors_are_untouched():
    assert _friendly_error(RuntimeError("unrelated")) == "An internal error occurred"
    assert "Lost connection" in _friendly_error(RuntimeError("Lost connection to llama-server"))
