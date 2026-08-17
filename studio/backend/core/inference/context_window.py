# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Message-aware rolling context helpers for local chat inference."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from typing import Any, Optional

_OMITTED_TOOL_EXCHANGE = "[Earlier tool exchange omitted from the rolling context window.]"

# How far BELOW the prompt budget a compaction trims, as a fraction of that budget.
#
# Trimming to exactly the budget looks efficient and behaves badly. The client re-sends
# the whole saved transcript on every request, so the fit runs from scratch each time,
# and a prompt trimmed to the brim is over it again as soon as the next turn is
# appended. The eviction boundary then creeps forward on nearly every turn, which costs
# twice: llama-server's prefix cache is invalidated each time the head moves, so the
# whole prompt is reprocessed, and there is no such thing as a discrete "compaction
# event" to tell the user about -- every reply has compacted a little more.
#
# Taking a chunk out in one go instead buys a stretch of turns whose eviction boundary
# does not move at all: same head, so the prefix cache holds, and one thing to report
# rather than a running commentary. The cost is the headroom itself, which is context
# that could have held conversation, so it is deliberately a minority of the budget.
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


def group_turns(messages: list[dict]) -> list[list[dict]]:
    """Split messages into the turn groups the rolling window evicts as single units.

    Normal user/assistant turns stay together. Each assistant tool call starts a
    separate group containing its tool results, so long agent runs can evict old
    exchanges without orphaning results or losing the task that initiated them.

    Exposed so callers that need to do something with the evicted turns operate on the
    same unit the evictor does, rather than on loose messages.
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

    Identity, not equality: the truncation helpers reuse the very same dict objects in
    their output, and a conversation can legitimately contain two byte-identical turns
    ("continue" twice), which an equality diff would collapse.
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

    ``min_dropped`` keeps evicting past the point where the prompt fits, until at least
    that many messages are gone. It is how a thread re-applies the boundary it already
    compacted to, rather than recomputing one that slides forward a little every turn.
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
                # Strict chat templates reject adjacent user turns. This occurs when
                # an internal tool re-prompt follows an evicted exchange.
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
            if part.get("type") in ("image_url", "input_audio", "audio", "input_image"):
                return True
    return False


def prompt_budget(context_length: int, max_tokens: Optional[int]) -> int:
    """Tokens available to the PROMPT, once room for the reply is set aside.

    Exported because two other things need the same number and must not re-derive it:
    the caller sizing a forced recall against the room a fit actually obtained, and the
    client explaining which part of an over-long request does not fit. A second copy of
    this formula would drift from the fit it is supposed to describe.
    """
    if context_length <= 1:
        return context_length
    requested = max_tokens if max_tokens is not None and max_tokens > 0 else context_length
    return context_length - min(requested, max(1, context_length // 4))


def _latest_turn_tokens(messages: list[dict], count_tokens: Callable[[list[dict]], int]) -> int:
    """Tokens in the newest message, counted without handing the template an orphan.

    The diagnosis this feeds is produced when a request cannot be made to fit, and a tool
    loop reaches that point with a tool result last. On its own that slice is not a
    conversation: templates that require a tool result to follow its assistant tool call
    refuse to render it, and the exception would escape the fit entirely -- the caller
    falls back to the untrimmed request and the client is told nothing at all, which is
    the opposite of what this branch exists for. A number that is approximate is worth
    more here than a diagnosis that never arrives.
    """
    if not messages:
        return 0
    try:
        return count_tokens(messages[-1:])
    except Exception:
        return estimate_messages_tokens(messages[-1:])


def fit_rolling_context(
    messages: list[dict],
    *,
    context_length: int,
    max_tokens: Optional[int],
    count_tokens: Callable[[list[dict]], int],
    protected_message_ids: Optional[set[int]] = None,
    reserve_tokens: int = 0,
    sticky_dropped: int = 0,
) -> tuple[list[dict], Optional[dict[str, Any]]]:
    """Fit a chat into its real context by dropping oldest complete turns.

    The exact tokenizer/template count decides whether trimming is needed. The
    inexpensive estimator only chooses candidate turns; exact recounts verify the
    result. The current turn is never clipped, so an irreducibly large request still
    reaches llama-server's normal context-length error.

    ``reserve_tokens`` leaves room for something the caller intends to add back after
    fitting (recalled earlier turns). It deliberately does NOT participate in the
    decision of whether to trim at all: a conversation that already fits is returned
    untouched even when the reserve would not fit alongside it. Charging the reserve up
    front would make chats start evicting turns that comfortably fit today, which is a
    silent regression in the common case for the benefit of the rare one.

    ``sticky_dropped`` is the boundary this thread last compacted to, in messages. The
    fit re-applies it before deciding anything, and only moves it when what is left still
    does not fit. Without it the fit is stateless: the client re-sends the whole saved
    transcript every request, so "keep the newest N tokens" slides forward a turn or two
    at a time and every single reply has compacted a little more than the last. With it,
    plus ``_COMPACTION_HEADROOM_RATIO`` of slack taken out when the boundary does move,
    compaction becomes an occasional event with quiet turns in between, which is both what
    a user can be told about and what lets llama-server's prefix cache survive a turn.

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

    # Phase one: put the boundary back where this thread already had it. Cheap, and it
    # is what makes a compacted thread stop compacting further on every turn.
    #
    # Gated on the prompt not already fitting, exactly as the reserve and the headroom
    # are. A saved boundary describes the branch it was measured on, and rolling back to
    # an early message leaves one that is far too aggressive for the conversation now in
    # front of us; applying it anyway would evict most of a chat that comfortably fits
    # and report a compaction that did not need to happen.
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

    # Phase two, only if what is left still does not fit: move the boundary, and take a
    # chunk out rather than skimming to the brim, so it can stay put for a while.
    #
    # The reserve and the headroom deliberately play no part in the decision to trim at
    # all, so a conversation that fits today is never evicted to satisfy either.
    trim_target = prompt_target
    if current_tokens > prompt_target:
        # Summed, not max()'d: the reserve is spent immediately on recalled passages,
        # so counting it as headroom would hand back room that is already taken and
        # the next turn would compact again.
        headroom = int(prompt_target * _COMPACTION_HEADROOM_RATIO)
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
        # Evicted everything evictable and it still does not fit. The ORIGINAL messages
        # are returned, not the partial eviction: the request is going to be refused
        # either way, and silently dropping turns off a doomed request would lose them
        # from the model's view with nothing to show for it.
        #
        # The diagnosis is worth returning even so. Without it the only thing the user
        # is told is llama-server's own error, which reports the size of the WHOLE
        # conversation and advises shortening it -- advice that cannot possibly work,
        # because what is left after maximal eviction is the system prompt and the
        # latest turn, and one of those is the thing that does not fit.
        #
        # Every consumer already gates on `fits`, so this is inert everywhere that
        # treats a truncation as a compaction.
        return messages, {
            "fits": False,
            "dropped_messages": 0,
            "prompt_tokens_before": initial_tokens,
            "prompt_tokens_after": initial_tokens,
            # What the conversation cannot be reduced below, and how much of that is
            # the message just sent. Between them these say whether the conversation
            # or the single message is the problem.
            "irreducible_tokens": current_tokens,
            "latest_turn_tokens": _latest_turn_tokens(messages, count_tokens),
            # ...and WHOSE message that is. A tool loop refits with the tool result
            # appended, so the last message is often a tool result rather than anything
            # the user wrote: telling them to shorten it names something they did not
            # write and cannot edit, while their own question may be one line.
            "latest_turn_role": str(messages[-1].get("role") or "") if messages else "",
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
