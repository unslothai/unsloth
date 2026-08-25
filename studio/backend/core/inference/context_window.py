# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Message-aware rolling context helpers for local chat inference."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable
from typing import Any, Optional

_OMITTED_TOOL_EXCHANGE = "[Earlier tool exchange omitted from the rolling context window.]"

# How far BELOW the prompt budget a compaction trims, as a fraction of that budget.
# Trimming to exactly the budget puts the next turn over it again, so the boundary creeps
# forward every turn: llama-server's prefix cache dies each time and there is no discrete
# compaction event to report. Taking a chunk out in one go buys a stretch of turns with a
# fixed head, at the cost of the headroom itself, hence a minority of the budget.
_COMPACTION_HEADROOM_RATIO = max(
    0.0, min(0.9, float(os.environ.get("ROLLING_COMPACTION_HEADROOM_RATIO", "0.25")))
)


def estimate_message_tokens(message: dict) -> int:
    try:
        return max(1, len(json.dumps(message, ensure_ascii = False)) // 4)
    except Exception:
        return 1


def estimate_messages_tokens(messages: list[dict]) -> int:
    return sum(estimate_message_tokens(message) for message in messages)


def estimate_messages_tokens_dense(messages: list[dict]) -> int:
    """The same estimate, but honest about text that tokenises densely.

    Four characters per token is about right for English and badly wrong for CJK and
    emoji, which run closer to one token per character. Measured on an 81-message CJK
    chat: 1295 estimated against 2737 real, a 2.1x undercount, which a caller sizing a
    search budget spends as room it does not have. `_conversation_search_tokens` already
    charges non-ASCII a token each when pricing the result; this is the same rule applied
    to the spend, which is the half that was missing.

    Deliberately NOT the default. `truncate_oldest_messages` evicts on the flat estimate,
    and making eviction pessimistic would drop history a request could have kept. Only a
    caller deciding how much room is LEFT wants this, where erring high is the safe side.
    """
    total = 0
    for message in messages:
        try:
            text = json.dumps(message, ensure_ascii = False)
        except Exception:
            total += 1
            continue
        dense = sum(1 for char in text if ord(char) > 127)
        total += max(1, dense + (len(text) - dense) // 4)
    return total


# ASCII that runs this far without a break is not prose. Measured with Qwen3 over 16-20k
# character samples, in characters per token: base64 1.35, hex 1.13, minified JSON 2.75,
# against English prose at 3.27 and Python source at 4.38. The estimate above charges all
# of them the English four, so a pasted blob is priced at a third of what it costs, and a
# caller sizing the room LEFT then hands out room that is already occupied.
#
# The rule is per RUN rather than per message, so a blob pasted into a sentence is priced
# as a blob: a run this long is the thing itself. 64 characters rather than a rounder
# number because base64 is conventionally wrapped at 76, and at a threshold of 80 a
# wrapped blob scores as ordinary prose. Measured on the samples above, runs of 64+
# non-space characters cover 100% of an unwrapped blob, 98.6% of a wrapped one, 0% of the
# Python source and 23.5% of a README -- and that 23.5% is its URLs and table rules, which
# tokenise near this rate themselves rather than at the prose rate.
_DENSE_RUN_CHARS = 64
# Two characters per token, not one. It stays BELOW the measured cost of every sample, so
# no turn is priced above what it really costs: over-pricing a turn spends the very room
# this budget exists to hand out, and a result cut to pay for room that was never occupied
# is the same waste from the other side.
_DENSE_RUN_CHARS_PER_TOKEN = 2
_DENSE_RUN_RE = re.compile(r"\S{%d,}" % _DENSE_RUN_CHARS)


def estimate_messages_tokens_conservative(
    messages: list[dict], *, dense_ascii: bool = False
) -> int:
    """`estimate_messages_tokens_dense`, with unbroken ASCII runs charged as the blobs
    they are.

    For the caller with no tokenizer and no rolling fit behind it: the native loop prices
    what a thread has already spent, and if that number is low the result it then admits
    is what pushes the next prompt past the window, with nothing downstream to recover.
    Non-ASCII keeps its token per character, ordinary ASCII keeps the English four, and
    only the long unbroken runs are charged at `_DENSE_RUN_CHARS_PER_TOKEN`.

    ``dense_ascii`` charges ALL of a message's ASCII at that rate, for the messages that
    are dense whether or not they hold an unbroken run: a tool result is `hexdump`, `ls
    -l` or a stack trace as often as it is a blob, and those are space-separated. It
    applies to the ASCII only, so a CJK result is not charged twice for being a result.
    """
    total = 0
    for message in messages:
        try:
            text = json.dumps(message, ensure_ascii = False)
        except Exception:
            total += 1
            continue
        wide = sum(1 for char in text if ord(char) > 127)
        if dense_ascii:
            total += max(1, wide + (len(text) - wide) // _DENSE_RUN_CHARS_PER_TOKEN)
            continue
        # ASCII only: a run of CJK is already charged a token a character above, and
        # charging it here as well would price it twice.
        runs = sum(
            sum(1 for char in match.group(0) if ord(char) <= 127)
            for match in _DENSE_RUN_RE.finditer(text)
        )
        plain = len(text) - wide - runs
        total += max(1, wide + runs // _DENSE_RUN_CHARS_PER_TOKEN + plain // 4)
    return total


def group_turns(messages: list[dict]) -> list[list[dict]]:
    """Split messages into the turn groups the rolling window evicts as single units.

    Each assistant tool call starts its own group holding its tool results, so long agent
    runs evict old exchanges without orphaning results. Exposed so callers that act on
    evicted turns use the same unit the evictor does.
    """
    groups: list[list[dict]] = []
    for message in messages:
        starts_tool_exchange = message.get("role") == "assistant" and bool(
            message.get("tool_calls")
        )
        follows_instruction = bool(groups and groups[-1][0].get("role") in ("system", "developer"))
        if (
            message.get("role") in ("system", "developer", "user")
            or starts_tool_exchange
            or follows_instruction
        ):
            groups.append([message])
        elif not groups:
            groups.append([message])
        else:
            groups[-1].append(message)
    return groups


def evicted_messages(before: list[dict], after: list[dict]) -> list[dict]:
    """Messages present in ``before`` and absent from ``after``, in their original order.

    Identity, not equality: the truncation helpers reuse the same dict objects, and a
    chat can contain two byte-identical turns ("continue" twice) that equality collapses.
    """
    kept = {id(message) for message in after}
    return [message for message in before if id(message) not in kept]


def truncate_oldest_messages(
    messages: list[dict],
    keep_ratio: float,
    *,
    protected_message_ids: Optional[set[int]] = None,
    min_dropped: int = 0,
) -> tuple[list[dict], int]:
    """Drop complete oldest turns while preserving system messages and the latest turn.

    ``min_dropped`` keeps evicting past the point where the prompt fits, so a thread can
    re-apply the boundary it already compacted to instead of one that slides every turn.
    """
    if not messages or (keep_ratio >= 1.0 and min_dropped <= 0):
        return messages, 0

    groups = group_turns(messages)

    if len(groups) <= 1:
        return messages, 0

    estimates = {id(message): estimate_message_tokens(message) for message in messages}
    current_estimate = sum(estimates.values())
    target_estimate = int(current_estimate * max(0.0, keep_ratio))
    dropped = 0
    protected_ids = protected_message_ids or set()
    latest_user_group = next(
        (
            index
            for index in range(len(groups) - 1, -1, -1)
            if any(message.get("role") == "user" for message in groups[index])
        ),
        None,
    )
    protected_groups = {
        index
        for index, group in enumerate(groups)
        if index == len(groups) - 1
        or index == latest_user_group
        or any(message.get("role") in ("system", "developer") for message in group)
        or any(id(message) in protected_ids for message in group)
    }
    eviction_units: list[list[int]] = []
    index = 0
    while index < len(groups):
        if index in protected_groups:
            index += 1
            continue
        unit = [index]
        starts_user_turn = groups[index][0].get("role") == "user"
        next_index = index + 1
        if starts_user_turn:
            while next_index < len(groups) and groups[next_index][0].get("role") not in (
                "system",
                "developer",
                "user",
            ):
                unit.append(next_index)
                next_index += 1
        if not any(group_index in protected_groups for group_index in unit):
            eviction_units.append(unit)
        index = next_index if starts_user_turn else index + 1

    dropped_groups: set[int] = set()
    for unit in eviction_units:
        if current_estimate <= target_estimate and dropped >= min_dropped:
            break
        dropped_groups.update(unit)
        for group_index in unit:
            group = groups[group_index]
            dropped += len(group)
            current_estimate -= sum(estimates[id(message)] for message in group)

    if dropped == 0:
        return messages, 0

    kept: list[dict] = []
    for index, group in enumerate(groups):
        if index not in dropped_groups:
            if kept and kept[-1].get("role") == "user" and group and group[0].get("role") == "user":
                # Strict chat templates reject adjacent user turns, which an internal
                # tool re-prompt after an evicted exchange would produce.
                kept.append({"role": "assistant", "content": _OMITTED_TOOL_EXCHANGE})
            kept.extend(group)
    return kept, dropped


def messages_have_media(messages: list[dict]) -> bool:
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            # `input_video` is llama.cpp's own part type (written by `_inject_video_part`);
            # missing it here would let a video prompt take the rolling preflight, whose
            # `/apply-template` count omits the sampled video tokens.
            if part.get("type") in (
                "image_url",
                "input_audio",
                "audio",
                "input_image",
                "input_video",
            ):
                return True
    return False


def prompt_budget(context_length: int, max_tokens: Optional[int]) -> int:
    """Tokens available to the PROMPT, once room for the reply is set aside.

    Exported so the recall sizing and the client's over-long-request explanation share
    this formula rather than each keeping a copy that drifts from the actual fit.
    """
    if context_length <= 1:
        return context_length
    requested = max_tokens if max_tokens is not None and max_tokens > 0 else context_length
    return context_length - min(requested, max(1, context_length // 4))


# Cap a tool-loop retrieval without reducing the allowance of smaller results.
_RETRIEVAL_BUDGET_SHARE = 0.5

# The least reply room a rescued prompt must leave, as a fraction of the window. Small on
# purpose: missing the reserve is survivable, so this only rules out the stub-answer end.
_RESCUE_REPLY_FLOOR_DIVISOR = 16


def retrieval_budget(
    context_length: int,
    max_tokens: Optional[int],
    prompt_tokens: int,
    *,
    reply_returns: bool = False,
) -> int:
    """Return the prompt room available to one retrieval.

    In a tool loop the retrieved exchange and the reply are protected on the next fit, so
    one retrieval may use at most half the prompt budget; single-shot uses the remainder.
    Kept beside ``prompt_budget`` so every backend sizing ``search_conversation`` shares
    one policy rather than quietly disagreeing.
    """
    target = prompt_budget(context_length, max_tokens)
    room = max(0, target - int(prompt_tokens or 0))
    if reply_returns:
        room = min(room, int(target * _RETRIEVAL_BUDGET_SHARE))
    return room


# Headroom against the tokenizer disagreeing with the estimate that sized a result. A
# result is measured in characters and spent in tokens, so the conversion is the thing
# that can be wrong; 1% of the budget absorbs that without noticeably shortening output.
_TOOL_RESULT_BUDGET_BUFFER = 0.99

# What a truncated result costs BESIDES its body: the notice naming the cut, the spill
# path and the command that resumes it. Measured at 60-85 tokens, held clear of that.
# Reserved rather than ignored because the body is sized and the notice appended after,
# so a budget spent entirely on the body overshoots by the notice every time, and at zero
# room, where the notice IS the message, by the whole of it.
#
# Charged by `tools._truncate`, at the point it knows the result really is being cut, and
# NOT subtracted from the room below. A result that fits carries no notice, so holding
# this back up front would cut a result that would have fitted whole and then spend more
# than it saved: 200 tokens of room, a 100-token result, and reserving first leaves 72 for
# the body and appends ~70 tokens of notice explaining the 28 that were dropped.
_RESULT_NOTICE_RESERVE = 128


def tool_result_budget(
    context_length: int,
    max_tokens: Optional[int],
    prompt_tokens: int,
    *,
    buffer: float = _TOOL_RESULT_BUDGET_BUFFER,
) -> int:
    """Tokens a tool result may add without pushing the next prompt past its budget.

    The cap before this was a share of the WINDOW, so it never fell as the conversation
    filled: one result could take half a small context however little was left, and the
    next fit cannot recover because it protects the newest turn, the very result that does
    not fit. This prices the same result against the room actually remaining.

    Against ``prompt_budget`` rather than ``context_length``: a result sized to fill the
    physical window leaves the prompt fitting and nothing to answer in. Room for the reply
    is not room for the result. The figure covers the whole tool message, notice included:
    see `_RESULT_NOTICE_RESERVE`, which the truncation charges against this room when it
    turns out to need one. A caller with several tool calls to serve divides what comes
    back between them.
    """
    target = prompt_budget(context_length, max_tokens)
    return max(0, int(target * buffer) - int(prompt_tokens or 0))


def _latest_turn_count(
    messages: list[dict], count_tokens: Callable[[list[dict]], int]
) -> tuple[int, bool]:
    """`(tokens, exact)` for the newest message, estimated if the template refuses it.

    A tool loop can reach the does-not-fit diagnosis with a tool result last, which strict
    templates reject on its own; letting that raise would abort the fit and tell the user
    nothing. An approximate number beats a diagnosis that never arrives -- but the caller
    has to know which it got, because only the counted one carries the prompt's floor.
    """
    if not messages:
        return 0, False
    try:
        return int(count_tokens(messages[-1:])), True
    except Exception:
        return int(estimate_messages_tokens(messages[-1:])), False


def _shared_prompt_tokens(count_tokens: Callable[[list[dict]], int]) -> int:
    """What a rendered prompt costs before any message at all.

    The counter prices a whole PROMPT, not a bag of messages: the template's own wrapper
    and, on a request that advertises tools, the entire tool catalogue rendered into the
    system turn. So that constant sits inside every count the fit takes, including the
    one-message slice above -- on the GGUF tool loop the counters pass `safe_tools` to
    `count_chat_tokens`, which puts the schemas in the prompt it measures, and a couple
    of MCP servers is thousands of tokens of them.

    Measured on the empty prompt rather than estimated: this number is subtracted from
    the two counts that decide which part of the prompt gets named, and a subtraction
    that is only roughly right moves the blame instead of removing it. One extra count,
    taken only on the branch that has already decided to refuse the request.
    """
    try:
        return max(0, int(count_tokens([])))
    except Exception:
        # No measurable floor; zero leaves the diagnosis as it was before this existed.
        return 0


def _marginal_turn_count(
    fitted: list[dict], count_tokens: Callable[[list[dict]], int], irreducible_tokens: int
) -> Optional[int]:
    """What the newest turn ADDED to the prompt that was actually measured.

    The one-message slice is unusable when the template renders the newest message as
    nothing on its own, but the difference against the same prompt WITHOUT it is a real
    tokenizer count of exactly the turn's contribution, with the floor cancelled on both
    sides of the subtraction. So the diagnosis keeps a counted turn where it used to fall
    back to a guess.

    `irreducible_tokens` must be the count of `fitted` itself, which is what both callers
    pass. One extra count, on a branch that has already decided to refuse the request,
    next to the one `_shared_prompt_tokens` takes there. None when the counter cannot
    price the prefix, which is the caller's cue to fall back to the estimate.
    """
    try:
        return int(irreducible_tokens) - int(count_tokens(list(fitted)[:-1]))
    except Exception:
        return None


def turn_diagnosis(
    messages: list[dict],
    count_tokens: Callable[[list[dict]], int],
    *,
    irreducible_tokens: int,
    fitted: Optional[list[dict]] = None,
) -> dict[str, Any]:
    """The fields that say WHICH part of a refused prompt is which.

    `shared_prompt_tokens` is the floor both `latest_turn_tokens` and
    `irreducible_tokens` carry, so a consumer comparing the two can take it off both
    sides and compare what the turn actually contributed against what the rest of the
    conversation did. Zero when the turn was estimated rather than counted: that estimate
    prices the message's own JSON and no catalogue, so it has no floor to remove.

    `latest_turn_exact` says which of the two `latest_turn_tokens` is. It is False only
    on the fallback below, where nothing could be counted at all, and a consumer must
    then not compare it against `irreducible_tokens`: that number is a tokenizer count of
    the rendered prompt while the estimate is four characters to a token over the
    message's JSON, and the two do not share units. Measured on the bundled gemma-4
    template with a real Gemma tokenizer, 16,400 characters of newlines estimate 8,207
    tokens against 557 rendered, 14.8x, which is enough for the estimate alone to clear
    the dominance ratio against a prompt the turn is 6% of.

    `fitted` is the message list `irreducible_tokens` was counted over. Given it, a turn
    the template refuses to render alone is still priced by difference rather than
    guessed.
    """
    if not messages:
        # Vacuously exact: nothing was estimated, and a zero count is ignored anyway.
        return {
            "latest_turn_tokens": 0,
            "latest_turn_role": "",
            "shared_prompt_tokens": 0,
            "latest_turn_exact": True,
        }
    latest, exact = _latest_turn_count(messages, count_tokens)
    shared = _shared_prompt_tokens(count_tokens) if exact else 0
    if exact and latest <= shared:
        # Counted, and yet no bigger than the empty prompt: the template rendered the turn
        # as nothing. Both bundled Gemma-4 templates do exactly that to a `role: tool`
        # message on its own -- `{%- if message['role'] != 'tool' -%}` skips it, and the
        # result is emitted only while scanning forward from the assistant tool call that
        # asked for it, which a one-message slice does not contain. So the number is the
        # floor and nothing else, and subtracting the floor from it would leave the turn
        # contributing ~0 and blame the conversation.
        #
        # Price it by DIFFERENCE against the prompt that was measured instead: the slice
        # is unrenderable, the contribution is not. Reported floor-inclusive, so the
        # consumer's existing subtraction of `shared` leaves exactly the marginal and the
        # ratio still compares two tokenizer counts.
        marginal = _marginal_turn_count(
            fitted if fitted is not None else messages, count_tokens, irreducible_tokens
        )
        if marginal is not None and marginal > 0:
            latest = marginal + shared
        else:
            # Nothing countable left: same remedy as a template that refuses the slice
            # outright -- price the message's own JSON, and record no floor, because that
            # estimate does not carry one.
            latest = int(estimate_messages_tokens(messages[-1:]))
            shared = 0
            # What is REPORTED is now the estimate, and that is what the flag describes.
            # Leaving it True would be the worse half of the bug.
            exact = False
    # Never all of either side: a floor at or above them would leave no ratio to compare.
    shared = max(0, min(shared, latest - 1, int(irreducible_tokens) - 1))
    return {
        "latest_turn_tokens": latest,
        # Whose message it is: often a tool result the user cannot shorten.
        "latest_turn_role": str(messages[-1].get("role") or ""),
        "shared_prompt_tokens": shared,
        # Whether the number above is a token count or a four-characters-a-token guess.
        "latest_turn_exact": bool(exact),
    }


def fit_rolling_context(
    messages: list[dict],
    *,
    context_length: int,
    max_tokens: Optional[int],
    count_tokens: Callable[[list[dict]], int],
    protected_message_ids: Optional[set[int]] = None,
    reserve_tokens: int = 0,
    sticky_dropped: int = 0,
    keeps_boundary: bool = False,
) -> tuple[list[dict], Optional[dict[str, Any]]]:
    """Fit a chat into its real context by dropping oldest complete turns.

    The exact tokenizer/template count decides whether trimming is needed; the cheap
    estimator only picks candidate turns. The current turn is never clipped, so an
    irreducibly large request still reaches llama-server's context-length error.

    ``reserve_tokens`` leaves room for what the caller adds back after fitting (recalled
    turns). It deliberately does not affect whether to trim at all, so a chat that fits
    today is never evicted just because the reserve would not fit alongside it.

    ``sticky_dropped`` is the boundary this thread last compacted to, in messages,
    re-applied before anything else and moved only if what is left still does not fit.
    Without it the fit is stateless (the client re-sends the whole transcript each turn)
    so the boundary slides every reply; with it plus ``_COMPACTION_HEADROOM_RATIO`` of
    slack, compaction is an occasional event the prefix cache can survive.

    Acceptance is still checked against the untightened ``prompt_target``: falling short
    of the headroom is not a failure to fit.
    """
    if context_length <= 1:
        return messages, None

    prompt_target = prompt_budget(context_length, max_tokens)
    fitted = list(messages)
    initial_tokens = count_tokens(fitted)
    current_tokens = initial_tokens
    dropped_total = 0

    # Phase one: put the boundary back where this thread already had it, so a compacted
    # thread stops compacting further every turn. Gated on the prompt not already
    # fitting: a saved boundary describes the branch it was measured on, and after a
    # rollback it would evict most of a chat that comfortably fits.
    if sticky_dropped > 0 and initial_tokens > prompt_target:
        candidate, dropped = truncate_oldest_messages(
            fitted,
            1.0,
            protected_message_ids = protected_message_ids,
            min_dropped = sticky_dropped,
        )
        if dropped:
            fitted = candidate
            dropped_total = dropped
            current_tokens = count_tokens(fitted)

    # Phase two, only if what is left still does not fit: move the boundary, taking a
    # chunk out rather than skimming to the brim so it can stay put for a while.
    trim_target = prompt_target
    if current_tokens > prompt_target:
        # Summed, not max()'d: the reserve is spent immediately on recalled passages, so
        # counting it as headroom would hand back room that is already taken.
        #
        # And only for a caller that can put the boundary back next request. The headroom
        # buys quiet turns between compactions by cutting deeper than needed, which is a
        # bargain only if the deeper cut is remembered. An incognito chat, an API request
        # with no persisted thread, or a request whose turns are not saved gets neither
        # the boundary nor a recall of what went, so there it is simply 25% less history
        # than plain eviction would have kept.
        headroom = int(prompt_target * _COMPACTION_HEADROOM_RATIO) if keeps_boundary else 0
        trim_target = max(1, prompt_target - reserve_tokens - headroom)

    while current_tokens > trim_target:
        keep_ratio = min(0.95, trim_target / max(1, current_tokens))
        candidate, dropped = truncate_oldest_messages(
            fitted,
            keep_ratio,
            protected_message_ids = protected_message_ids,
        )
        if dropped == 0:
            break
        fitted = candidate
        dropped_total += dropped
        current_tokens = count_tokens(fitted)

    if current_tokens > prompt_target:
        # Missing the prompt target only loses reserved reply room, so keep the original
        # if it fits the physical window, else an eviction that does. Strictly under on
        # both sides: llama-server refuses at `n_ctx` exactly. But not under by one token,
        # which would lose the history AND answer in one token, worse than the refusal it
        # replaces. Capped by the reserve, so the floor demands no room the target did not.
        reply_floor = min(
            max(1, context_length // _RESCUE_REPLY_FLOOR_DIVISOR),
            context_length - prompt_target,
        )
        rescued = (
            dropped_total > 0
            and initial_tokens >= context_length
            and current_tokens + reply_floor <= context_length
        )
        return (fitted if rescued else messages), {
            "fits": False,
            "dropped_messages": dropped_total if rescued else 0,
            "prompt_tokens_before": initial_tokens,
            "prompt_tokens_after": current_tokens if rescued else initial_tokens,
            # Floor for the conversation, and how much of it is the message just sent:
            # together they say whether the chat or the single message is the problem.
            "irreducible_tokens": current_tokens,
            # `fitted` is what `current_tokens` prices, so the turn can be counted by
            # difference against it rather than estimated.
            **turn_diagnosis(
                messages, count_tokens, irreducible_tokens = current_tokens, fitted = fitted
            ),
            "context_length": context_length,
            "prompt_target": prompt_target,
        }
    if dropped_total == 0:
        return messages, None
    return fitted, {
        "dropped_messages": dropped_total,
        "prompt_tokens_before": initial_tokens,
        "prompt_tokens_after": current_tokens,
        "context_length": context_length,
        "fits": True,
    }
