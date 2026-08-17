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
                # Strict chat templates reject adjacent user turns, which happens when an
                # internal tool re-prompt follows an evicted exchange.
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

    Exported so the recall sizing and the client's over-long-request explanation share
    this formula rather than each keeping a copy that drifts from the actual fit.
    """
    if context_length <= 1:
        return context_length
    requested = max_tokens if max_tokens is not None and max_tokens > 0 else context_length
    return context_length - min(requested, max(1, context_length // 4))


def _latest_turn_tokens(messages: list[dict], count_tokens: Callable[[list[dict]], int]) -> int:
    """Tokens in the newest message, estimated if the template refuses to render it.

    A tool loop can reach the does-not-fit diagnosis with a tool result last, which strict
    templates reject on its own; letting that raise would abort the fit and tell the user
    nothing. An approximate number beats a diagnosis that never arrives.
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
        # Evicted everything evictable and it still does not fit. Return the ORIGINAL
        # messages, not the partial eviction: the request is refused either way, so
        # dropping turns off a doomed request loses them for nothing. The diagnosis is
        # still worth returning; llama-server's own error reports the size of the WHOLE
        # conversation and advises shortening it, which cannot work when what is left is
        # the system prompt plus the latest turn. Consumers all gate on `fits`.
        return messages, {
            "fits": False,
            "dropped_messages": 0,
            "prompt_tokens_before": initial_tokens,
            "prompt_tokens_after": initial_tokens,
            # Floor for the conversation, and how much of it is the message just sent:
            # together they say whether the chat or the single message is the problem.
            "irreducible_tokens": current_tokens,
            "latest_turn_tokens": _latest_turn_tokens(messages, count_tokens),
            # ...and whose message that is. In a tool loop the last message is often a
            # tool result, which the user did not write and cannot shorten.
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
