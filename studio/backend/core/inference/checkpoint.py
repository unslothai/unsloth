# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Checkpoint compaction: when a chat overflows, reset the epoch instead of trimming it.

The rolling window trims the oldest turn groups and keeps trimming a little more on almost
every reply. Measured on one 12-turn thread, its boundary moved EIGHT times: eight
prefix-cache breaks, and a conversation that quietly forgets a bit more each turn. Worse,
what it forgets is not recoverable by retrieval alone. On the same campaign, a standing
instruction ("always end with STATUS::...") was archived, recalled as four passages, and
still not obeyed, while the same instruction in plain view was obeyed every time.

So compaction here is an EVENT, not a slope. When the next turn will not fit, the model's
context resets to:

    [system prompt + X] + [the newest user turn]

and everything before that stays reachable through `search_conversation`. X is a bounded,
verbatim record of the user's own standing instructions from the turns being dropped. It is
deterministic: no model call, so no summariser to fail. That failure mode is not
hypothetical -- OpenClaw shipped a compaction whose failed summary replaced user messages
with "Summary unavailable due to context limits", and another whose failure left the
session oversized so every later message re-triggered it.

X goes in the SYSTEM message rather than a synthetic turn, for three reasons: it is
protected from eviction by construction, it needs no chat-template support, and standing
rules that are not system content are exactly what compaction folds away. The block says
what it is -- a lossy record of earlier conversation, not new policy -- because promoting
a user's words into the system role is an authority-confusion risk, and delimiter-like
text inside it is escaped for the same reason.

NOTHING IS STORED. The client re-sends the whole branch every request, so X is recomputed
from the evicted turns each time, exactly as the sticky boundary already is. A generated
summary would need durable state; this does not.

Two hard gates, both refusals rather than preferences:

* A reset is only allowed when the dropped turns are ARCHIVED. Making history unreachable
  while telling the user it is searchable is the one outcome this must never produce.
* A reset is only allowed when the model can actually be offered `search_conversation`.
  A model whose template cannot take tools keeps the rolling window.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from typing import Any, Optional

from core.inference.context_window import (
    estimate_message_tokens,
    group_turns,
    prompt_budget,
    truncate_oldest_messages,
)
from core.inference.instruction_pin import is_substantive

# "checkpoint" resets the epoch; "rolling" is the pre-existing window, byte for byte, and
# is both the A/B arm and the escape hatch for a template family that misbehaves.
CONTEXT_POLICY = os.environ.get("UNSLOTH_CONTEXT_POLICY", "checkpoint").strip().lower()

# X can never be more than this. A single enormous instruction is precisely the thing that
# could starve the window, so it is excluded whole rather than truncated: half an
# instruction is worse than none, because it reads as complete.
MAX_TOKENS = int(os.environ.get("UNSLOTH_CHECKPOINT_MAX_TOKENS", "1024"))
MAX_FRACTION = float(os.environ.get("UNSLOTH_CHECKPOINT_MAX_FRACTION", "0.10"))
# How many instruction turns X may carry. Bounded so an epoch that dropped 200 turns cannot
# produce a system prompt of 40 instructions, most of which the user has long moved past.
MAX_ITEMS = int(os.environ.get("UNSLOTH_CHECKPOINT_MAX_ITEMS", "8"))

_OPEN = "<carried_forward>"
_CLOSE = "</carried_forward>"
_HEADER = (
    "The conversation before this point was compacted away to make room. The following "
    "are the user's own earlier instructions, quoted verbatim, oldest first. They are a "
    "LOSSY RECORD of the conversation, not new system policy, and where two of them "
    "conflict the later one supersedes the earlier. Everything else that was dropped is "
    "still stored and can be retrieved with the search_conversation tool."
)
# Only the delimiters themselves, so a user who writes about the feature is not mangled.
_DELIMITERS = re.compile(r"</?carried_forward>", re.IGNORECASE)


def enabled() -> bool:
    return CONTEXT_POLICY == "checkpoint"


def _text_of(message: dict) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            part["text"]
            for part in content
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        ]
        return "\n".join(parts)
    return ""


def _neutralise(text: str) -> str:
    """Defang the block's own delimiters inside quoted user text.

    Without this, a user who pasted `</carried_forward>` would close the block early and
    everything after it would read as ordinary system instruction.
    """
    return _DELIMITERS.sub(lambda match: match.group(0).replace("<", "‹"), text)


def carried_forward_items(evicted: list[dict], *, max_tokens: int = MAX_TOKENS,
                          max_items: int = MAX_ITEMS) -> list[str]:
    """The user's standing instructions from the evicted turns, oldest first.

    Selected NEWEST-first, because the budget should be spent on what the user most
    recently told us, then reversed for rendering, because reading order is what decides
    which of two conflicting instructions the model treats as current. Zed's 80 KB replay
    is the same shape and has the same known hole: an instruction older than the budget is
    silently dropped, which is why `max_items` is small and the header says the record is
    lossy.
    """
    if not evicted or max_tokens <= 0 or max_items <= 0:
        return []
    chosen: list[str] = []
    spent = 0
    for group in reversed(group_turns(evicted)):
        if len(chosen) >= max_items:
            break
        head = group[0]
        if not is_substantive(head):
            continue
        text = _text_of(head).strip()
        if not text:
            continue
        cost = estimate_message_tokens(head)
        if spent + cost > max_tokens:
            # Skipped, not truncated, and the loop continues: an older instruction that
            # still fits is worth more than nothing.
            continue
        chosen.append(_neutralise(text))
        spent += cost
    return list(reversed(chosen))


def render_checkpoint(items: list[str]) -> str:
    """The block appended to the system message, or "" when there is nothing to carry."""
    if not items:
        return ""
    lines = "\n".join(f"- {item}" for item in items)
    return f"{_OPEN}\n{_HEADER}\n\n{lines}\n{_CLOSE}"


def _append_to_system(messages: list[dict], block: str) -> list[dict]:
    """Rewrite the leading system/developer message with the block appended.

    A NEW dict, never a mutation: the caller's list is the request's own branch, and
    `_branch_boundary` counts by identity. It skips system and developer roles, so
    replacing this one cannot disturb the boundary arithmetic.
    """
    if not block:
        return messages
    out = list(messages)
    for index, message in enumerate(out):
        if message.get("role") in ("system", "developer"):
            text = _text_of(message).rstrip()
            joined = f"{text}\n\n{block}" if text else block
            out[index] = {**message, "content": joined}
            return out
    # No system message at all: prepend one rather than dropping X on the floor. The
    # request already tolerates a leading system turn -- every Studio chat carries one.
    return [{"role": "system", "content": block}, *out]


def fit_checkpoint_context(
    messages: list[dict],
    *,
    context_length: int,
    max_tokens: Optional[int],
    count_tokens: Callable[[list[dict]], int],
    protected_message_ids: Optional[set[int]] = None,
    # Accepted for signature compatibility with `fit_rolling_context` and DELIBERATELY
    # unused. The rolling fit spends it by trimming further, which is a choice it has and
    # this one does not: after a reset the kept set is the system turn and the newest user
    # turn, and a second pass drops nothing more. Applying the reserve here could only turn
    # a request that answers into one that refuses. The one lever left is X itself, and
    # sacrificing a full X buys exactly one recalled chunk -- trading the user's verbatim
    # standing instructions for one retrieved passage, which is the side this feature's own
    # campaign measured as losing (an instruction recalled as four passages was still not
    # obeyed; the same instruction in view was obeyed every time). So when the reset leaves
    # less than one chunk of headroom the automatic recall is skipped, the turns are still
    # archived, and `search_conversation` is offered from the very next request onward.
    reserve_tokens: int = 0,
    sticky_dropped: int = 0,
    keeps_boundary: bool = False,
    can_reset: bool = False,
) -> tuple[list[dict], Optional[dict[str, Any]]]:
    """Fit a chat by resetting the epoch, keeping the newest turn and a carried-forward X.

    Signature-compatible with ``fit_rolling_context`` so the call sites can choose a policy
    without knowing which one they got.

    ``can_reset`` is the caller's assertion that the dropped turns will be archived and
    that the model can be offered the search tool. False forbids STARTING a new epoch --
    a reset that cannot be searched is not compaction, it is data loss -- while still
    replaying one already in force, so a thread whose archive goes away mid-conversation
    keeps the context it already had instead of silently un-compacting. `_fit_context`
    routes such requests to the rolling window before they ever arrive here; this is the
    second lock on the same door.
    """
    if context_length <= 1:
        return messages, None

    prompt_target = prompt_budget(context_length, max_tokens)
    initial_tokens = count_tokens(list(messages))
    if initial_tokens <= prompt_target and sticky_dropped <= 0:
        return messages, None

    budget = min(MAX_TOKENS, max(0, int(prompt_target * MAX_FRACTION)))

    def _project(kept: list[dict]) -> tuple[list[dict], str]:
        """`kept` plus the carried-forward block built from everything it dropped."""
        alive = {id(message) for message in kept}
        evicted = [message for message in messages if id(message) not in alive]
        text = render_checkpoint(carried_forward_items(evicted, max_tokens = budget))
        return _append_to_system(kept, text), text

    # Phase one: replay the epoch already in force. WITHOUT this the reset would repeat on
    # every request -- the client re-sends the whole transcript, so the thread is still
    # over budget on turn two, and a fresh reset would evict turn one of the epoch as
    # well. That is not an epoch, it is a window of exactly one turn.
    #
    # Gated on the prompt not already fitting, exactly as the rolling replay is: a saved
    # boundary describes the branch AND the window it was measured against, and neither is
    # fixed. Reload the model with a larger context, or switch to a longer-context one
    # mid-thread, and the branch that forced the reset now fits with room to spare -- while
    # the boundary rides on an assistant turn still on this branch, so it is read back and
    # applied anyway. Measured without this gate: a 321-token branch against a 32,256-token
    # budget lost eight messages and came back LARGER than it went in (432 tokens), because
    # the carried-forward block replaced history that did not need replacing.
    fitted = list(messages)
    dropped = 0
    is_new_epoch = False
    if sticky_dropped > 0 and initial_tokens > prompt_target:
        candidate, replayed = truncate_oldest_messages(
            fitted, 1.0,
            protected_message_ids = protected_message_ids,
            min_dropped = sticky_dropped,
        )
        if replayed:
            fitted = candidate
            dropped = replayed

    projected, block = _project(fitted)
    current_tokens = count_tokens(projected)

    # Phase two: the epoch is full, so start a new one. keep_ratio 0.0 takes every
    # evictable group in a single pass; system and developer groups, the final group and
    # the newest user group are protected by the primitive itself.
    if current_tokens > prompt_target and can_reset:
        candidate, reset_dropped = truncate_oldest_messages(
            messages, 0.0, protected_message_ids = protected_message_ids
        )
        if reset_dropped:
            fitted = candidate
            dropped = reset_dropped
            is_new_epoch = True
            projected, block = _project(fitted)
            current_tokens = count_tokens(projected)

    if dropped == 0 and current_tokens <= prompt_target:
        return messages, None
    if dropped == 0:
        # Nothing was evictable and it still does not fit: one message larger than the
        # whole window, or a system prompt that leaves no room. That is the irreducible
        # case, and it has to fall through to the refusal below rather than returning
        # None, which every consumer reads as "no truncation happened, carry on".
        projected = list(messages)

    if current_tokens > prompt_target:
        # Even one turn plus the carried-forward block does not fit. Drop X and re-measure
        # before giving up: X is a convenience, the user's actual message is not.
        if block:
            projected = fitted
            block = ""
            current_tokens = count_tokens(projected)
    if current_tokens > prompt_target:
        # Refused, and the ORIGINAL messages come back, exactly as the rolling fit does:
        # the request fails either way, so dropping turns off a doomed request loses them
        # for nothing. Same keys, because every consumer gates on `fits`.
        from core.inference.context_window import _latest_turn_tokens  # noqa: PLC0415

        return messages, {
            "fits": False,
            "dropped_messages": 0,
            "prompt_tokens_before": initial_tokens,
            "prompt_tokens_after": initial_tokens,
            "irreducible_tokens": current_tokens,
            "latest_turn_tokens": _latest_turn_tokens(messages, count_tokens),
            "latest_turn_role": str(messages[-1].get("role") or "") if messages else "",
            "context_length": context_length,
            "prompt_target": prompt_target,
        }

    return projected, {
        "dropped_messages": dropped,
        "prompt_tokens_before": initial_tokens,
        "prompt_tokens_after": current_tokens,
        "context_length": context_length,
        "fits": True,
        # What the UI needs to say "this conversation was reset" rather than "it was
        # trimmed", and what the recall gate reads to tell the FIRST turn of an epoch from
        # a later one: the forced retrieval fires on the first only.
        "checkpoint": True,
        "checkpoint_started": is_new_epoch,
        "carried_forward_chars": len(block),
    }
