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


# ---------------------------------------------------------------- wording


def test_no_diagnosis_keeps_the_generic_advice():
    message = _friendly_error(ValueError(_SERVER_ERROR))
    assert "Message too long: 7153 tokens exceeds the 5120-token context window." in message
    assert "shorten the conversation" in message


def test_long_history_keeps_the_generic_advice():
    # A long thread: the newest turn is a small part of what could not be evicted.
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
    # The user cannot shorten what a tool wrote.
    assert "send it in smaller pieces" not in message


def test_function_role_is_treated_as_a_tool_result():
    context_refusal.record_fit(_refusal(irreducible = 5600, latest_turn = 5400, role = "function"))
    assert "A tool returned" in _friendly_error(ValueError(_SERVER_ERROR))


def test_an_oversized_assistant_prefill_does_not_ask_the_user_to_split_it():
    # Auto-continue resends the truncated reply, which the user did not write.
    context_refusal.record_fit(_refusal(irreducible = 5600, latest_turn = 5400, role = "assistant"))
    message = _friendly_error(ValueError(_SERVER_ERROR))
    assert "The reply being continued is already too long for this window" in message
    assert "start a new reply" in message
    assert "send it in smaller pieces" not in message


@pytest.mark.parametrize("role", ["system", "developer"])
def test_oversized_instructions_point_at_the_system_prompt(role):
    # These survive eviction, so splitting one preserves the total and resolves nothing.
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
    # Unspecific advice beats advice aimed at the wrong turn.
    context_refusal.record_fit(_refusal(irreducible = 5600, latest_turn = 5400, role = role))
    message = _friendly_error(ValueError(_SERVER_ERROR))
    for named in (
        "the message just sent",
        "a single tool result",
        "the reply being continued",
        "the system instructions",
    ):
        assert named not in message
    # Still says what IS known: this floor is over the window, so no shorter conversation
    # reaches the server either.
    assert "Even with every earlier turn dropped" in message


def test_every_wording_keeps_the_counts_and_the_client_markers():
    # `isContextLimitError` in chat-adapter.ts matches these substrings, and the numbers
    # are all the user has to size the window by.
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
        # Below two thirds, the rest of the prompt is a real share of the problem, so no
        # turn is named. This floor stands at the window, hence the fixed-overhead
        # wording rather than the generic one.
        (3379, "Even with every earlier turn dropped"),
        # Over that share but inside the window: the bulk of the prompt, yet servable by
        # itself. Note 4097, over the 4096 PROMPT BUDGET the fit (not the server) refuses.
        (3380, "Most of this prompt is the message just sent"),
        (4097, "Most of this prompt is the message just sent"),
        (5119, "Most of this prompt is the message just sent"),
        # At the window: llama-server refuses on prompt size alone, so it cannot be sent.
        (5120, "does not fit on its own"),
    ],
)
def test_dominating_the_floor_is_not_the_same_as_not_fitting(latest_turn, expected):
    context_refusal.record_fit(_refusal(irreducible = 5120, latest_turn = latest_turn))
    assert expected in _friendly_error(ValueError(_SERVER_ERROR))


def test_a_turn_that_merely_dominates_hedges_its_advice():
    # Trimming the rest buys little, but "will not help" would overstate the numbers.
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
    # Same turn and window, differing only in what Max Tokens reserved. "Can hold" means
    # the window, so both read the same.
    with_budget = _refusal(irreducible = 5000, latest_turn = 4800, prompt_target = 4096)
    without_budget = dict(with_budget)
    without_budget.pop("prompt_target")
    context_refusal.record_fit(with_budget)
    first = _friendly_error(ValueError(_SERVER_ERROR))
    context_refusal.record_fit(without_budget)
    assert _friendly_error(ValueError(_SERVER_ERROR)) == first


def test_a_diagnosis_for_a_different_window_is_ignored():
    # A model reload between the fit and the error: that shape describes another window.
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
    """A 20-token "hi" beside a large MCP catalogue is not what the user should shorten.

    Both counts in the diagnosis price a whole prompt, so both carry the catalogue: the
    turn reads as 97% of the irreducible prompt while contributing 20 tokens of it. The
    remedy is fewer tools or a bigger window, and neither is "send it in smaller pieces".
    """
    truncation, message = _refuse_and_explain(
        window = 8192, catalogue = 6000, system_tokens = 200, turn_tokens = 20
    )
    # The raw counts really are that lopsided; the floor is why.
    assert truncation["latest_turn_tokens"] > 0.9 * truncation["irreducible_tokens"]
    assert truncation["shared_prompt_tokens"] == 6000
    assert "shorten the conversation" in message
    assert "message just sent" not in message


def test_a_catalogue_bigger_than_the_window_never_makes_a_tiny_turn_unsendable():
    # The false claim, not just the unhelpful one: a catalogue over the window pushes the
    # one-message count past it, reporting a twenty-token turn as unsendable.
    truncation, message = _refuse_and_explain(
        window = 4096, catalogue = 4200, system_tokens = 200, turn_tokens = 20
    )
    assert truncation["latest_turn_tokens"] > truncation["context_length"]
    assert "does not fit on its own" not in message
    assert "message just sent" not in message
    # And a catalogue over the window puts the FLOOR over it too, so the honest advice is
    # the fixed-overhead one, not a shorter conversation.
    assert truncation["irreducible_tokens"] >= truncation["context_length"]
    assert "shortening the conversation will not help" in message


def _servable_without_history(*, window: int, catalogue: int, system_tokens: int) -> bool:
    """Would the same request go through with the conversation shortened to nothing?

    The one claim the generic advice makes. A refused fit hands the ORIGINAL messages on
    (dropping turns off a doomed request loses them for nothing), and llama-server admits
    a prompt on size alone, so "served" is the untrimmed prompt landing under `n_ctx`.
    """
    messages = _thread(system_tokens = system_tokens, turn_tokens = 20, history_turns = 0)
    count = _tool_catalogue_counter(catalogue)
    sent, _ = fit_rolling_context(
        messages, context_length = window, max_tokens = None, count_tokens = count
    )
    return count(sent) < window


def test_a_two_message_thread_is_never_told_to_shorten_the_conversation():
    """The case this module exists for, on the branch that names no turn.

    A system prompt over the window with a twenty-token "hi" after it: eviction has
    nothing to take (the primitive protects system turns and the newest user turn), so
    the floor IS the prompt. "Shorten the conversation" names an action that cannot
    work, and measurably does not: with the history at zero the request is still refused.
    """
    truncation, message = _refuse_and_explain(
        window = 4096, catalogue = 0, system_tokens = 5000, turn_tokens = 20, history_turns = 0
    )
    assert truncation["irreducible_tokens"] >= truncation["context_length"]
    assert not _servable_without_history(window = 4096, catalogue = 0, system_tokens = 5000)
    assert "Even with every earlier turn dropped" in message
    assert "shortening the conversation will not help" in message
    assert "the system prompt and any tools that are enabled" in message
    # Never the advice llama-server itself gives, which is the whole point of the rewrite.
    assert "or shorten the conversation" not in message


def test_a_floor_under_the_window_keeps_the_advice_that_still_works():
    """The other side of the same line, and why it is drawn at the window.

    A catalogue that fits leaves room the conversation is standing in: the fit refuses at
    `prompt_target`, but the untrimmed prompt is served whenever it lands under `n_ctx`,
    so trimming history really does clear this one. Advising against it would be the new
    false claim.
    """
    truncation, message = _refuse_and_explain(
        window = 8192, catalogue = 6000, system_tokens = 200, turn_tokens = 20
    )
    assert truncation["irreducible_tokens"] < truncation["context_length"]
    assert _servable_without_history(window = 8192, catalogue = 6000, system_tokens = 200)
    assert "shorten the conversation" in message
    assert "will not help" not in message


def test_a_diagnosis_for_a_different_window_claims_nothing_about_the_floor():
    # A reload between the fit and the error: that floor was measured elsewhere, so the
    # "cannot be shortened" claim has no evidence behind it either.
    context_refusal.record_fit(_refusal(irreducible = 9000, latest_turn = 300, context_length = 8192))
    message = _friendly_error(ValueError(_SERVER_ERROR))
    assert "shorten the conversation" in message
    assert "Even with every earlier turn dropped" not in message


@pytest.mark.parametrize(
    "turn_tokens,expected",
    [
        # Still the bulk of what is left once the catalogue is off both sides.
        (5000, "Most of this prompt is the message just sent"),
        # And still bigger than the window on its own.
        (8300, "does not fit on its own"),
    ],
)
def test_a_catalogue_does_not_cost_a_turn_that_really_is_the_problem(turn_tokens, expected):
    _, message = _refuse_and_explain(
        window = 8192, catalogue = 1500, system_tokens = 200, turn_tokens = turn_tokens
    )
    assert expected in message


def test_a_tool_result_beside_a_catalogue_is_judged_on_its_own_size():
    # The same trap on a role the user cannot edit.
    _, small = _refuse_and_explain(
        window = 8192, catalogue = 6000, system_tokens = 200, turn_tokens = 20, role = "tool"
    )
    assert "tool result" not in small and "shorten the conversation" in small
    _, large = _refuse_and_explain(
        window = 8192, catalogue = 1500, system_tokens = 200, turn_tokens = 5000, role = "tool"
    )
    assert "Most of this prompt is a single tool result" in large


def test_the_floor_is_never_all_of_either_count():
    # A nonsense floor must not drive either side to zero and invent a ratio.
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


def _gemma_style_counter(catalogue_tokens: int):
    """A counter that renders a lone tool result as nothing, as Gemma 4 does.

    Both bundled Gemma-4 templates skip `role: tool` in the message loop and emit the
    result only while scanning forward from the assistant tool call that asked for it, so
    a one-message slice renders byte-for-byte the same prompt as an empty one.
    """

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
        # Big enough to be the problem: the tool-specific advice must survive.
        (5000, 200, "Most of this prompt is a single tool result"),
        # And a small result beside instructions that do not fit is still not the result's
        # fault: the estimate must not invent a blame the count never earned.
        (20, 5000, "shorten the conversation"),
    ],
)
def test_a_turn_the_template_renders_as_nothing_is_not_counted_as_the_floor(
    turn_tokens, system_tokens, expected
):
    """A count no bigger than the empty prompt measured framing, not the turn.

    Gemma 4 renders a lone tool result as nothing, so the one-message slice succeeds and
    returns the floor exactly. Recording that as an exact turn size makes the turn worth
    ~0 once the floor comes off both sides, and a 5,000-token tool result reads as the
    conversation's fault. The remedy is to price the turn by DIFFERENCE against the prompt
    that was measured, which is still a tokenizer count of exactly its contribution.
    """
    _, truncation = fit_rolling_context(
        _tool_loop_thread(turn_tokens, system_tokens = system_tokens),
        context_length = 8192,
        max_tokens = None,
        count_tokens = _gemma_style_counter(1500),
    )
    assert truncation is not None and not truncation["fits"]
    # Not the floor reported as the turn: a number that moves with the result's size
    # instead of pinning to the 1,500-token catalogue.
    assert truncation["latest_turn_tokens"] != 1500
    # Counted by difference, so the floor IS recorded and comes off both sides, and what
    # is left is the turn's own contribution rather than a four-chars-a-token guess.
    assert truncation["shared_prompt_tokens"] == 1500
    assert truncation["latest_turn_exact"] is True
    # The payload plus its envelope (`tool_call_id`, `name`), and nothing else: 5,018 for
    # the big result and 38 for the small one, against a 1,500-token catalogue that used
    # to be the whole number.
    contribution = truncation["latest_turn_tokens"] - truncation["shared_prompt_tokens"]
    assert turn_tokens <= contribution <= turn_tokens + 100
    _context_truncated_sse_chunk("cmpl-1", "model", truncation)
    assert expected in _friendly_error(
        ValueError("the request (9000 tokens) exceeds the available context size (8192 tokens)")
    )


# ------------------------------------------------- an estimate is not a measurement


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
    """Neither wording, because an estimate cannot be weighed against a count.

    The fallback `latest_turn_tokens` is `len(json.dumps(message)) // 4` while
    `irreducible_tokens` is a tokenizer count of the rendered prompt, so the dominance
    ratio compares a guess with a truth. Text that tokenises sparsely blows through the
    guess: on the bundled gemma-4 template with a real Gemma tokenizer, 16,400 characters
    of newlines estimate 8,207 tokens against 557 rendered, 14.8x. That alone clears the
    ratio against an 8,629-token prompt the turn is 6.5% of, beside a system prompt that
    is 93% of it -- and the softer wording is then a false attribution, not a hedge. It is
    not correctable either: escaped JSON runs the other way, 0.86x.

    The producer prices such a turn by difference now, so this flag is only ever False
    when nothing could be counted, and there the generic advice is the honest answer.
    """
    estimated = _refusal(irreducible = 5120, latest_turn = 5400, role = role) | {
        "latest_turn_exact": False
    }
    context_refusal.record_fit(estimated)
    message = _friendly_error(ValueError(_SERVER_ERROR))
    assert hard not in message
    assert f"Most of this prompt is {soft}" not in message
    # No turn named, so the advice is whichever generic branch fits. This refusal is
    # irreducible at its window, so it is the one that says shortening cannot work and
    # names the levers instead of a role.
    assert "Even with every earlier turn dropped" in message
    assert "the system prompt and any tools that are enabled" in message


def test_a_measured_turn_still_gets_the_hard_wording():
    # The gate is provenance, not size: a counted turn over the window is unchanged.
    context_refusal.record_fit(
        _refusal(irreducible = 5120, latest_turn = 5400, role = "tool") | {"latest_turn_exact": True}
    )
    assert "A tool returned more than this context window can hold" in _friendly_error(
        ValueError(_SERVER_ERROR)
    )


def test_a_payload_without_the_flag_is_read_as_a_count():
    # Absent means a producer that predates the flag, and every one of those counted.
    refusal = _refusal(irreducible = 5120, latest_turn = 5400, role = "tool")
    refusal.pop("latest_turn_exact", None)
    context_refusal.record_fit(refusal)
    assert "A tool returned more than this context window can hold" in _friendly_error(
        ValueError(_SERVER_ERROR)
    )


def test_a_sparse_tool_result_is_blamed_for_no_more_than_it_rendered():
    """End to end on the Gemma shape, with a counter that tokenises whitespace runs.

    A real tokenizer merges long runs of whitespace into single tokens, so the JSON-length
    estimate the fit used to be forced onto for a lone `role: tool` message can clear the
    window while the rendered turn costs a fraction of it. Measured on the bundled
    gemma-4 template with a real Gemma tokenizer: 16,400 characters of newlines estimate
    8,207 tokens and render 557, 14.8x, and a lone tool message renders to exactly the
    empty prompt.

    The band this pins is the one no estimate can survive. Here the turn is 29% of the
    prompt -- too small to blame, too large for the estimate's error to cancel out of a
    ratio -- so the refusal is real, an oversized system prompt is what the request died
    of, and the tool result is neither what could not be sent nor the bulk of the prompt.
    """

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
                # Calibrated to the measurement above: 32,876 characters of escaped
                # JSON for this payload against 838 real tokens, so ~39 chars a token.
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
    # The estimate this replaced really would have blamed the turn: 8,218 against a
    # 2,899-token prompt clears the 0.66 share several times over.
    assert estimate_messages_tokens(thread[-1:]) >= 0.66 * truncation["irreducible_tokens"]
    # What it really contributed is 842 of 2,899, 29%, and it is a count, not a guess.
    assert truncation["latest_turn_exact"] is True
    contribution = truncation["latest_turn_tokens"] - truncation["shared_prompt_tokens"]
    assert contribution == count(thread) - count(thread[:-1])
    assert contribution < 0.4 * truncation["irreducible_tokens"]
    _context_truncated_sse_chunk("cmpl-1", "model", truncation)
    message = _friendly_error(
        ValueError("the request (2899 tokens) exceeds the available context size (2048 tokens)")
    )
    # Neither wording blames the tool result, and the advice names the parts eviction
    # never touches, which is where the 2,000-token system prompt actually is.
    assert "A tool returned more than this context window can hold" not in message
    assert "Most of this prompt is a single tool result" not in message
    assert "Even with every earlier turn dropped" in message
    assert "the system prompt and any tools that are enabled" in message


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
    # A tool loop refuses on one iteration and fits on the next: no stale refusal.
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
    # `_accumulate_context_truncation` sums `dropped_messages` across a tool loop, so the
    # refusal must be the per-fit event or it reports counts no fit produced.
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
    # The behaviour worked around: both copy the context, and a `.set()` in a copy never
    # reaches the request.
    async def _run():
        await _drain_like_the_route(_record_in_worker)
        return context_refusal.latest_refusal()

    assert asyncio.run(_run()) is None


def test_a_slot_carries_the_refusal_back_through_task_and_thread():
    async def _run():
        context_refusal.open_slot()
        assert await _drain_like_the_route(_record_in_worker) == "drained"
        return context_refusal.latest_refusal()

    # Read inside the coroutine: `asyncio.run` gives it its own context copy, as a request
    # task does, and that is where `_friendly_error` reads from.
    assert asyncio.run(_run())["latest_turn_tokens"] == 5400


def test_a_slot_carries_the_refusal_back_when_the_drain_raises():
    # The path that matters: the drain diagnoses, then raises the error that refusal
    # explains, so there is no return value to carry it back.
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
    # Dropping either `open_slot` would silently restore the generic advice.
    source = (Path(_BACKEND_DIR) / "routes" / "inference.py").read_text(encoding = "utf-8")
    for drain in ("_drain_gguf_tool_loop", "_drain_gguf_choices"):
        spawn = f"asyncio.create_task(asyncio.to_thread({drain}))"
        assert spawn in source
        preceding = source.split(spawn)[0].splitlines()[-4:]
        assert any("context_refusal.open_slot()" in line for line in preceding)


# ---------------------------------------------------------- streaming tool loops


def _respawn_refit_then_refused():
    """A tool generator that fits, respawns, refits into a refusal, then is refused.

    The refit runs inside the generator, i.e. inside the worker thread the stream loop
    drives it from, and the prompt that FIT emitted no `context_truncated` event, so
    nothing recorded out in the stream's own context first.
    """
    yield "the first tokens, before llama-server died"
    context_refusal.record_fit(_refusal(irreducible = 5600, latest_turn = 5400, role = "tool"))
    raise ValueError(_SERVER_ERROR)


async def _stream_like_the_tool_route(*, with_slot: bool):
    """The shape both streaming tool loops use: a task around a thread, per event.

    The message is built in this generator's own `except`, which is where the slot has
    to be visible; anything less than this shape does not test the thing that broke.
    """
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

        # Iterated from a task of its own, as a streaming response body is.
        return await asyncio.create_task(_consume())

    return asyncio.run(_run())[-1]


def test_a_streaming_tool_loop_without_a_slot_loses_the_respawn_refusal():
    # The regression: two context copies between the refit and the message, and no slot
    # in the stream's own context because the prompt that fit recorded nothing there.
    message = _drive(with_slot = False)
    assert "shorten the conversation" in message, message
    assert "tool" not in message, message


def test_a_streaming_tool_loop_with_a_slot_keeps_the_respawn_refusal():
    message = _drive(with_slot = True)
    assert "A tool returned more than this context window can hold" in message, message
    assert "ask for a smaller slice of the file or page" in message, message


def test_both_streaming_tool_loops_open_a_slot_first():
    """Only the loops that drive the tool generator: it owns the respawn refit.

    The no-tool streams reach `generate_chat_completion`, which has no refit callback,
    and their own fit is recorded out in the stream where the message is built.
    """
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
