# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Overflow advice depends on whose turn is the bulk of the prompt and whether it could be sent."""

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
from core.inference.context_window import (  # noqa: E402
    estimate_messages_tokens,
    fit_rolling_context,
)
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


def test_no_diagnosis_keeps_the_generic_advice():
    message = _friendly_error(ValueError(_SERVER_ERROR))
    assert "Message too long: 7153 tokens exceeds the 5120-token context window." in message
    assert "shorten the conversation" in message


def test_long_history_keeps_the_generic_advice():
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
    assert "send it in smaller pieces" not in message


def test_function_role_is_treated_as_a_tool_result():
    context_refusal.record_fit(_refusal(irreducible = 5600, latest_turn = 5400, role = "function"))
    assert "A tool returned" in _friendly_error(ValueError(_SERVER_ERROR))


def test_an_oversized_assistant_prefill_does_not_ask_the_user_to_split_it():
    context_refusal.record_fit(_refusal(irreducible = 5600, latest_turn = 5400, role = "assistant"))
    message = _friendly_error(ValueError(_SERVER_ERROR))
    assert "The reply being continued is already too long for this window" in message
    assert "start a new reply" in message
    assert "send it in smaller pieces" not in message


@pytest.mark.parametrize("role", ["system", "developer"])
def test_oversized_instructions_point_at_the_system_prompt(role):
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
def test_an_unnameable_role_is_never_blamed(role):
    context_refusal.record_fit(_refusal(irreducible = 5600, latest_turn = 5400, role = role))
    message = _friendly_error(ValueError(_SERVER_ERROR))
    for named in (
        "the message just sent",
        "a single tool result",
        "the reply being continued",
        "the system instructions",
    ):
        assert named not in message
    assert "Even with every earlier turn dropped" in message


def test_every_wording_keeps_the_counts_and_the_client_markers():
    # isContextLimitError in chat-adapter.ts matches these substrings.
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


@pytest.mark.parametrize(
    "latest_turn,expected",
    [
        (3379, "Even with every earlier turn dropped"),
        (3380, "Most of this prompt is the message just sent"),
        (4097, "Most of this prompt is the message just sent"),
        (5119, "Most of this prompt is the message just sent"),
        (5120, "does not fit on its own"),
    ],
)
def test_dominating_the_floor_is_not_the_same_as_not_fitting(latest_turn, expected):
    context_refusal.record_fit(_refusal(irreducible = 5120, latest_turn = latest_turn))
    assert expected in _friendly_error(ValueError(_SERVER_ERROR))


def test_a_turn_that_merely_dominates_hedges_its_advice():
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
    """The reply reserve is not part of what the window can hold; llama-server admits on size alone."""
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
    with_budget = _refusal(irreducible = 5000, latest_turn = 4800, prompt_target = 4096)
    without_budget = dict(with_budget)
    without_budget.pop("prompt_target")
    context_refusal.record_fit(with_budget)
    first = _friendly_error(ValueError(_SERVER_ERROR))
    context_refusal.record_fit(without_budget)
    assert _friendly_error(ValueError(_SERVER_ERROR)) == first


def test_a_diagnosis_for_a_different_window_is_ignored():
    context_refusal.record_fit(_refusal(irreducible = 5000, latest_turn = 4800, context_length = 8192))
    assert "shorten the conversation" in _friendly_error(ValueError(_SERVER_ERROR))


def _tool_catalogue_counter(catalogue_tokens: int):
    """The tool catalogue is a constant on top of any message slice, as /apply-template renders it."""

    def count(messages):
        body = sum(
            max(1, len(json.dumps(message, ensure_ascii = False)) // 4) for message in messages
        )
        return body + catalogue_tokens

    return count


def _thread(
    *,
    system_tokens: int,
    turn_tokens: int,
    role: str = "user",
    history_turns: int = 6,
):
    messages = [{"role": "system", "content": "s" * (system_tokens * 4)}]
    for index in range(history_turns):
        messages.append({"role": "user", "content": f"q{index} " + "x" * 1200})
        messages.append({"role": "assistant", "content": f"a{index} " + "y" * 1200})
    messages.append({"role": role, "content": "z" * (turn_tokens * 4)})
    return messages


def _refuse_and_explain(
    *,
    window: int,
    catalogue: int,
    system_tokens: int,
    turn_tokens: int,
    role: str = "user",
    history_turns: int = 6,
):
    """Drive the real path: fit -> recorded diagnosis -> the message the user reads."""
    _, truncation = fit_rolling_context(
        _thread(
            system_tokens = system_tokens,
            turn_tokens = turn_tokens,
            role = role,
            history_turns = history_turns,
        ),
        context_length = window,
        max_tokens = None,
        count_tokens = _tool_catalogue_counter(catalogue),
    )
    assert truncation is not None and not truncation["fits"]
    _context_truncated_sse_chunk("cmpl-1", "model", truncation)
    return truncation, _friendly_error(
        ValueError(
            f"the request (9000 tokens) exceeds the available context size ({window} tokens)"
        )
    )


def test_a_tool_catalogue_is_not_the_message_just_sent():
    """The catalogue is priced into both counts, so a small turn beside it is not the thing to shorten."""
    truncation, message = _refuse_and_explain(
        window = 8192, catalogue = 6000, system_tokens = 200, turn_tokens = 20
    )
    assert truncation["latest_turn_tokens"] > 0.9 * truncation["irreducible_tokens"]
    assert truncation["shared_prompt_tokens"] == 6000
    assert "shorten the conversation" in message
    assert "message just sent" not in message


def test_a_catalogue_bigger_than_the_window_never_makes_a_tiny_turn_unsendable():
    truncation, message = _refuse_and_explain(
        window = 4096, catalogue = 4200, system_tokens = 200, turn_tokens = 20
    )
    assert truncation["latest_turn_tokens"] > truncation["context_length"]
    assert "does not fit on its own" not in message
    assert "message just sent" not in message
    assert truncation["irreducible_tokens"] >= truncation["context_length"]
    assert "shortening the conversation will not help" in message


def _servable_without_history(*, window: int, catalogue: int, system_tokens: int) -> bool:
    """A refused fit returns the original messages, so servable means the untrimmed prompt fits n_ctx."""
    messages = _thread(system_tokens = system_tokens, turn_tokens = 20, history_turns = 0)
    count = _tool_catalogue_counter(catalogue)
    sent, _ = fit_rolling_context(
        messages, context_length = window, max_tokens = None, count_tokens = count
    )
    return count(sent) < window


def test_a_two_message_thread_is_never_told_to_shorten_the_conversation():
    """The floor is the prompt itself: nothing is evictable, so shortening history cannot help."""
    truncation, message = _refuse_and_explain(
        window = 4096, catalogue = 0, system_tokens = 5000, turn_tokens = 20, history_turns = 0
    )
    assert truncation["irreducible_tokens"] >= truncation["context_length"]
    assert not _servable_without_history(window = 4096, catalogue = 0, system_tokens = 5000)
    assert "Even with every earlier turn dropped" in message
    assert "shortening the conversation will not help" in message
    assert "the system prompt and any tools that are enabled" in message
    assert "or shorten the conversation" not in message


def test_a_floor_under_the_window_keeps_the_advice_that_still_works():
    """Under the window the untrimmed prompt is served, so advising a shorter conversation still works."""
    truncation, message = _refuse_and_explain(
        window = 8192, catalogue = 6000, system_tokens = 200, turn_tokens = 20
    )
    assert truncation["irreducible_tokens"] < truncation["context_length"]
    assert _servable_without_history(window = 8192, catalogue = 6000, system_tokens = 200)
    assert "shorten the conversation" in message
    assert "will not help" not in message


def test_a_diagnosis_for_a_different_window_claims_nothing_about_the_floor():
    context_refusal.record_fit(_refusal(irreducible = 9000, latest_turn = 300, context_length = 8192))
    message = _friendly_error(ValueError(_SERVER_ERROR))
    assert "shorten the conversation" in message
    assert "Even with every earlier turn dropped" not in message


@pytest.mark.parametrize(
    "turn_tokens,expected",
    [
        (5000, "Most of this prompt is the message just sent"),
        (8300, "does not fit on its own"),
    ],
)
def test_a_catalogue_does_not_cost_a_turn_that_really_is_the_problem(turn_tokens, expected):
    _, message = _refuse_and_explain(
        window = 8192, catalogue = 1500, system_tokens = 200, turn_tokens = turn_tokens
    )
    assert expected in message


def test_a_tool_result_beside_a_catalogue_is_judged_on_its_own_size():
    _, small = _refuse_and_explain(
        window = 8192, catalogue = 6000, system_tokens = 200, turn_tokens = 20, role = "tool"
    )
    assert "tool result" not in small and "shorten the conversation" in small
    _, large = _refuse_and_explain(
        window = 8192, catalogue = 1500, system_tokens = 200, turn_tokens = 5000, role = "tool"
    )
    assert "Most of this prompt is a single tool result" in large


def test_the_floor_is_never_all_of_either_count():
    context_refusal.record_fit(
        _refusal(irreducible = 5120, latest_turn = 5000) | {"shared_prompt_tokens": 99999}
    )
    assert "Most of this prompt is" not in _friendly_error(ValueError(_SERVER_ERROR))
    for bad in (None, "", -5, "junk"):
        context_refusal.record_fit(
            _refusal(irreducible = 5120, latest_turn = 3500) | {"shared_prompt_tokens": bad}
        )
        assert "Most of this prompt is the message just sent" in _friendly_error(
            ValueError(_SERVER_ERROR)
        )


def test_an_unrenderable_turn_records_no_floor_to_subtract():
    """A turn the template cannot render is estimated, so no floor is recorded to subtract."""

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


def _gemma_style_counter(catalogue_tokens: int):
    """Gemma 4 renders a lone tool result as nothing, so a one-message slice equals the empty prompt."""

    def count(messages):
        total = catalogue_tokens
        for index, message in enumerate(messages):
            if message.get("role") == "tool":
                previous = messages[index - 1] if index else None
                anchored = bool(previous) and (
                    previous.get("role") == "tool"
                    or (previous.get("role") == "assistant" and previous.get("tool_calls"))
                )
                if not anchored:
                    continue
            total += max(1, len(json.dumps(message, ensure_ascii = False)) // 4)
        return total

    return count


def _tool_loop_thread(
    turn_tokens: int,
    system_tokens: int = 200,
    history_turns: int = 6,
):
    """A tool loop caught mid-flight: the result of the call just made is last."""
    messages = [{"role": "system", "content": "s" * (system_tokens * 4)}]
    for index in range(history_turns):
        messages.append({"role": "user", "content": f"q{index} " + "x" * 1200})
        messages.append({"role": "assistant", "content": f"a{index} " + "y" * 1200})
    messages.append({"role": "user", "content": "read the file"})
    messages.append(
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": {"path": "big.txt"}},
                }
            ],
        }
    )
    messages.append(
        {
            "role": "tool",
            "tool_call_id": "c1",
            "name": "read_file",
            "content": "z" * (turn_tokens * 4),
        }
    )
    return messages


@pytest.mark.parametrize(
    "turn_tokens,system_tokens,expected",
    [
        (5000, 200, "Most of this prompt is a single tool result"),
        (20, 5000, "shorten the conversation"),
    ],
)
def test_a_turn_the_template_renders_as_nothing_is_not_counted_as_the_floor(
    turn_tokens, system_tokens, expected
):
    """A turn rendered as nothing is priced by difference against the prompt, not recorded as the floor."""
    _, truncation = fit_rolling_context(
        _tool_loop_thread(turn_tokens, system_tokens = system_tokens),
        context_length = 8192,
        max_tokens = None,
        count_tokens = _gemma_style_counter(1500),
    )
    assert truncation is not None and not truncation["fits"]
    assert truncation["latest_turn_tokens"] != 1500
    assert truncation["shared_prompt_tokens"] == 1500
    assert truncation["latest_turn_exact"] is True
    contribution = truncation["latest_turn_tokens"] - truncation["shared_prompt_tokens"]
    assert turn_tokens <= contribution <= turn_tokens + 100
    _context_truncated_sse_chunk("cmpl-1", "model", truncation)
    assert expected in _friendly_error(
        ValueError("the request (9000 tokens) exceeds the available context size (8192 tokens)")
    )


@pytest.mark.parametrize(
    "role,hard,soft",
    [
        ("user", "The message just sent does not fit on its own", "the message just sent"),
        ("tool", "A tool returned more than this context window can hold", "a single tool result"),
        (
            "assistant",
            "The reply being continued is already too long for this window",
            "the reply being continued",
        ),
        ("system", "The system instructions do not fit on their own", "the system instructions"),
    ],
)
def test_an_estimated_turn_names_no_turn_at_all(role, hard, soft):
    """An estimated turn is never blamed, since an estimate cannot be weighed against a tokenizer count."""
    estimated = _refusal(irreducible = 5120, latest_turn = 5400, role = role) | {
        "latest_turn_exact": False
    }
    context_refusal.record_fit(estimated)
    message = _friendly_error(ValueError(_SERVER_ERROR))
    assert hard not in message
    assert f"Most of this prompt is {soft}" not in message
    assert "Even with every earlier turn dropped" in message
    assert "the system prompt and any tools that are enabled" in message


def test_a_measured_turn_still_gets_the_hard_wording():
    context_refusal.record_fit(
        _refusal(irreducible = 5120, latest_turn = 5400, role = "tool") | {"latest_turn_exact": True}
    )
    assert "A tool returned more than this context window can hold" in _friendly_error(
        ValueError(_SERVER_ERROR)
    )


def test_a_payload_without_the_flag_is_read_as_a_count():
    refusal = _refusal(irreducible = 5120, latest_turn = 5400, role = "tool")
    refusal.pop("latest_turn_exact", None)
    context_refusal.record_fit(refusal)
    assert "A tool returned more than this context window can hold" in _friendly_error(
        ValueError(_SERVER_ERROR)
    )


def test_a_sparse_tool_result_is_blamed_for_no_more_than_it_rendered():
    """A sparse tool result is blamed for no more than it renders; JSON length overstates whitespace."""

    def count(messages):
        total = 0
        for index, message in enumerate(messages):
            text = json.dumps(message, ensure_ascii = False)
            if message.get("role") == "tool":
                previous = messages[index - 1] if index else None
                if not (
                    previous and previous.get("role") == "assistant" and previous.get("tool_calls")
                ):
                    continue
                # Calibrated: 32,876 chars of escaped JSON vs 838 real tokens, ~39 chars a token.
                total += max(1, len(text) // 39)
            else:
                total += max(1, len(text) // 4)
        return total

    thread = _tool_loop_thread(20, system_tokens = 2000, history_turns = 0)
    thread[-1]["content"] = ("\n" * 40 + "\t" * 40) * 205
    _, truncation = fit_rolling_context(
        thread, context_length = 2048, max_tokens = 512, count_tokens = count
    )
    assert truncation is not None and not truncation["fits"]
    assert estimate_messages_tokens(thread[-1:]) >= 0.66 * truncation["irreducible_tokens"]
    assert truncation["latest_turn_exact"] is True
    contribution = truncation["latest_turn_tokens"] - truncation["shared_prompt_tokens"]
    assert contribution == count(thread) - count(thread[:-1])
    assert contribution < 0.4 * truncation["irreducible_tokens"]
    _context_truncated_sse_chunk("cmpl-1", "model", truncation)
    message = _friendly_error(
        ValueError("the request (2899 tokens) exceeds the available context size (2048 tokens)")
    )
    assert "A tool returned more than this context window can hold" not in message
    assert "Most of this prompt is a single tool result" not in message
    assert "Even with every earlier turn dropped" in message
    assert "the system prompt and any tools that are enabled" in message


def test_a_diagnosis_with_no_window_recorded_is_still_usable():
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


def test_a_fit_that_succeeded_clears_an_earlier_refusal():
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
    # _accumulate_context_truncation sums dropped_messages, so the refusal must be per-fit.
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


def _record_in_worker():
    context_refusal.record_fit(_refusal(irreducible = 5600, latest_turn = 5400))
    return "drained"


def _record_then_fail():
    _record_in_worker()
    raise ValueError(_SERVER_ERROR)


async def _drain_like_the_route(func):
    """A task around a thread copies context twice between record and read, so the slot is needed."""
    task = asyncio.create_task(asyncio.to_thread(func))
    return await asyncio.shield(task)


def test_without_a_slot_the_drain_loses_the_refusal():
    # Both copy the context, and a .set() in a copy never reaches the request.
    async def _run():
        await _drain_like_the_route(_record_in_worker)
        return context_refusal.latest_refusal()

    assert asyncio.run(_run()) is None


def test_a_slot_carries_the_refusal_back_through_task_and_thread():
    async def _run():
        context_refusal.open_slot()
        assert await _drain_like_the_route(_record_in_worker) == "drained"
        return context_refusal.latest_refusal()

    # asyncio.run gives its own context copy, as a request task does.
    assert asyncio.run(_run())["latest_turn_tokens"] == 5400


def test_a_slot_carries_the_refusal_back_when_the_drain_raises():
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
    context_refusal.record_fit(_refusal(irreducible = 5000, latest_turn = 4800))
    context_refusal.open_slot()
    assert context_refusal.latest_refusal() is None


def test_both_non_streaming_gguf_drains_open_a_slot_first():
    # Dropping either open_slot would silently restore the generic advice.
    source = (Path(_BACKEND_DIR) / "routes" / "inference.py").read_text(encoding = "utf-8")
    for drain in ("_drain_gguf_tool_loop", "_drain_gguf_choices"):
        spawn = f"asyncio.create_task(asyncio.to_thread({drain}))"
        assert spawn in source
        preceding = source.split(spawn)[0].splitlines()[-4:]
        assert any("context_refusal.open_slot()" in line for line in preceding)


def _respawn_refit_then_refused():
    """The refit runs inside the generator's thread; nothing outside has recorded the refusal yet."""
    yield "the first tokens, before llama-server died"
    context_refusal.record_fit(_refusal(irreducible = 5600, latest_turn = 5400, role = "tool"))
    raise ValueError(_SERVER_ERROR)


async def _stream_like_the_tool_route(*, with_slot: bool):
    """The message is built in this generator's own except, where the slot has to be visible."""
    sentinel = object()
    if with_slot:
        context_refusal.open_slot()
    gen = _respawn_refit_then_refused()
    try:
        while True:
            next_task = asyncio.create_task(asyncio.to_thread(next, gen, sentinel))
            event = await asyncio.shield(next_task)
            if event is sentinel:
                break
            yield event
    except ValueError as exc:
        yield _friendly_error(exc)


def _drive(*, with_slot: bool) -> str:
    async def _run():
        async def _consume():
            return [chunk async for chunk in _stream_like_the_tool_route(with_slot = with_slot)]

        return await asyncio.create_task(_consume())

    return asyncio.run(_run())[-1]


def test_a_streaming_tool_loop_without_a_slot_loses_the_respawn_refusal():
    message = _drive(with_slot = False)
    assert "shorten the conversation" in message, message
    assert "tool" not in message, message


def test_a_streaming_tool_loop_with_a_slot_keeps_the_respawn_refusal():
    message = _drive(with_slot = True)
    assert "A tool returned more than this context window can hold" in message, message
    assert "ask for a smaller slice of the file or page" in message, message


def test_both_streaming_tool_loops_open_a_slot_first():
    """Only the tool loops need a slot, as the tool generator owns the respawn refit."""
    source = (Path(_BACKEND_DIR) / "routes" / "inference.py").read_text(encoding = "utf-8")
    loops = (
        ("async def gguf_tool_stream():", "gen = gguf_generate_with_tools()"),
        ("async def _anthropic_tool_stream(", "gen = run_gen()"),
    )
    for header, spawn in loops:
        body = source.split(header, 1)[1]
        assert spawn in body, header
        assert "context_refusal.open_slot()" in body.split(spawn, 1)[0], header


def test_other_friendly_errors_are_untouched():
    assert _friendly_error(RuntimeError("unrelated")) == "An internal error occurred"
    assert "Lost connection" in _friendly_error(RuntimeError("Lost connection to llama-server"))
