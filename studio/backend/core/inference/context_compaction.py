# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
#
# SlidingWindowCompact is adapted from forge
# (https://github.com/antoinezambelli/forge), Copyright (c) 2025-2026
# Antoine Zambelli, used under the MIT License.

"""Context-window compaction for OpenAI-style chat messages.

Long-running Studio Chat sessions can outgrow a model's context window.
This module ships a strategy that trims older messages from the prompt
sent to the model while preserving the persisted transcript shown in
the UI. The strategy returns a NEW list; ``messages`` is never mutated.

Invariants preserved by all strategies:

1. The system message (when present at index 0) is never dropped.
2. The first user message (when present) is never dropped: it carries
   the task prompt the rest of the conversation references.
3. Tool-call <-> tool-result pair linkage stays valid. An assistant
   message that carries ``tool_calls`` and the matching tool-role
   messages are kept or dropped as a unit. Dropping one side leaves
   the OpenAI chat template invalid and llama-server returns 400.
4. Multimodal turns (any message whose ``content`` is a list of parts)
   are treated as non-droppable. There is no tested compacted-media
   representation today; the strategies leave such turns intact.

Only one strategy is shipped here. ``TieredCompact`` from forge depends
on per-message metadata tags Studio's flat message dicts do not carry;
it will land in a follow-up once the message model gets the tags.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


# Char-to-token heuristic. Conservative on the high side so the
# compactor triggers earlier rather than later. Tokenizer-aware
# estimates may land in a follow-up.
_CHARS_PER_TOKEN = 4


def estimate_tokens(messages: list[dict]) -> int:
    """Rough token count for ``messages``. Uses a 4-char-per-token
    char-count heuristic on the visible ``content`` (str) and on
    serialized ``tool_calls`` arguments. Multimodal parts (list-typed
    content) contribute only their text parts. Studio's ``compaction``
    content parts (Anthropic round-trip state) also contribute their
    ``content`` string so a multi-KB compaction summary does not
    estimate as zero and slip past the threshold.
    """
    total_chars = 0
    for m in messages:
        content = m.get("content")
        if isinstance(content, str):
            total_chars += len(content)
        elif isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                t = part.get("text")
                if isinstance(t, str):
                    total_chars += len(t)
                # Studio's compaction content part: {"type":"compaction",
                # "content": "<summary>"}. Count the summary string.
                if part.get("type") == "compaction":
                    summary = part.get("content")
                    if isinstance(summary, str):
                        total_chars += len(summary)
        tcs = m.get("tool_calls")
        if isinstance(tcs, list):
            for tc in tcs:
                # Defensive: pre-pydantic OpenAI payloads occasionally
                # carry malformed entries (string, None) before
                # validation. Skip non-dict items instead of raising
                # AttributeError mid-compaction.
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function")
                args = (fn or {}).get("arguments") if isinstance(fn, dict) else None
                if isinstance(args, str):
                    total_chars += len(args)
    # Ceil-style division: floor (`// 4`) would systematically
    # underestimate non-multiple-of-4 lengths, letting just-over-budget
    # prompts appear under threshold and bypass compaction — the exact
    # failure this module exists to prevent. Round up so the heuristic
    # stays on the conservative side described above.
    return -(-total_chars // _CHARS_PER_TOKEN)


# Content-part types that carry no media payload and shouldn't anchor the
# message. ``compaction`` is Studio's Anthropic round-trip state -- pinning
# it would keep the very thing we're trying to compact away.
_TEXT_ONLY_PART_TYPES = {"text", "compaction"}


def _is_multimodal(msg: dict) -> bool:
    content = msg.get("content")
    if not isinstance(content, list):
        return False
    for part in content:
        # Unknown shapes (raw strings, ints, None) keep the conservative
        # "treat as multimodal" stance -- there's no test rendering for
        # them either.
        if not isinstance(part, dict):
            return True
        if part.get("type") not in _TEXT_ONLY_PART_TYPES:
            return True
    return False


def _assistant_tool_call_ids(msg: dict) -> set[str]:
    """Return the set of ``id`` values from an assistant message's
    ``tool_calls``. Empty set when the message has no tool calls.
    Mirrors ``estimate_tokens``: skip non-dict entries so malformed
    pre-pydantic inputs (a string or ``None`` in the list) don't crash
    ``_pair_linked_indices`` mid-compaction.
    """
    out: set[str] = set()
    tcs = msg.get("tool_calls")
    if isinstance(tcs, list):
        for tc in tcs:
            if not isinstance(tc, dict):
                continue
            tcid = tc.get("id")
            if isinstance(tcid, str) and tcid:
                out.add(tcid)
    return out


def _pair_linked_indices(messages: list[dict]) -> dict[int, set[int]]:
    """Map an assistant-message index to the indices of its tool-role
    follow-ups (matching ``tool_call_id``). Used so the compactor drops
    or keeps an assistant+tool group as a unit.
    """
    out: dict[int, set[int]] = {}
    # Walk in order so the next tool-role messages after an assistant
    # call are the natural matches. A tool message is matched to the
    # most recent prior assistant whose ``tool_calls`` contain that id.
    # ANY non-tool boundary (user, system, OR another assistant) ends
    # the pending window: per the OpenAI chat schema tool messages
    # must follow their assistant directly. A later assistant turn
    # arriving before the matching tool means that tool is malformed
    # input -- treating the late tool as still paired would let the
    # compactor drop a stale assistant + a later-turn tool together.
    pending_ids: dict[str, int] = {}
    for i, m in enumerate(messages):
        role = m.get("role")
        if role == "assistant":
            pending_ids.clear()
            ids = _assistant_tool_call_ids(m)
            out.setdefault(i, set())
            for tid in ids:
                pending_ids[tid] = i
        elif role == "tool":
            tcid = m.get("tool_call_id")
            if isinstance(tcid, str) and tcid in pending_ids:
                out.setdefault(pending_ids[tcid], set()).add(i)
        else:
            pending_ids.clear()
    return out


class CompactStrategy(ABC):
    """Interface for context-compaction strategies."""

    @abstractmethod
    def compact(self, messages: list[dict], budget_tokens: int) -> list[dict]:
        """Return a (possibly shorter) list of messages within
        ``budget_tokens``. Returns ``messages`` unchanged when no
        compaction is needed or possible.
        """
        ...


class NoCompact(CompactStrategy):
    """Passthrough strategy. Returns ``messages`` unchanged."""

    def compact(self, messages: list[dict], budget_tokens: int) -> list[dict]:
        return list(messages)


class SlidingWindowCompact(CompactStrategy):
    """Keep the system message, the first user message, and the last
    ``keep_recent`` non-droppable turns. Multimodal turns are never
    dropped (no compacted-media representation today). Assistant
    messages with ``tool_calls`` are grouped with their matching
    tool-role responses and treated as one unit.

    The strategy is a no-op when the estimated token count is already
    within ``budget_tokens`` or when there is nothing left to drop.
    """

    def __init__(self, keep_recent: int = 2, compact_threshold: float = 0.85) -> None:
        if keep_recent < 0:
            raise ValueError("keep_recent must be >= 0")
        if not (0.0 < compact_threshold <= 1.0):
            raise ValueError("compact_threshold must be in (0, 1]")
        self.keep_recent = keep_recent
        self.compact_threshold = compact_threshold

    def compact(self, messages: list[dict], budget_tokens: int) -> list[dict]:
        if budget_tokens <= 0 or not messages:
            return list(messages)

        threshold = int(budget_tokens * self.compact_threshold)
        if estimate_tokens(messages) <= threshold:
            return list(messages)

        # Anchor indices that may never be dropped.
        anchor_idx: set[int] = set()
        # System message at index 0.
        if messages and messages[0].get("role") == "system":
            anchor_idx.add(0)
        # First user message (the task prompt).
        for i, m in enumerate(messages):
            if m.get("role") == "user":
                anchor_idx.add(i)
                break
        # Multimodal turns.
        for i, m in enumerate(messages):
            if _is_multimodal(m):
                anchor_idx.add(i)

        # Tool-call / tool-result grouping. An assistant tool-call
        # message and its matching tool-role responses get the same
        # group id so we keep or drop them as a unit.
        pair_map = _pair_linked_indices(messages)
        # If any member of a valid asst+tools group is anchored,
        # anchor the whole group. Without this an anchored multimodal
        # tool whose assistant is droppable (or an anchored multimodal
        # assistant whose tools are droppable) would survive alone,
        # leaving the chat template invalid (OpenAI 400s on dangling
        # tool_calls / orphan tool messages).
        for asst_idx, tool_idxs in pair_map.items():
            if not tool_idxs:
                continue
            pair_idxs = {asst_idx, *tool_idxs}
            if pair_idxs & anchor_idx:
                anchor_idx |= pair_idxs
        group_id: list[int] = list(range(len(messages)))
        next_g = len(messages)
        for asst_idx, tool_idxs in pair_map.items():
            if not tool_idxs:
                continue
            g = next_g
            next_g += 1
            group_id[asst_idx] = g
            for ti in tool_idxs:
                group_id[ti] = g

        # The "recent window" is the last ``keep_recent`` distinct
        # groups encountered scanning from the end. ``keep_recent == 0``
        # must collect ZERO groups so the caller can drop everything
        # outside the anchor set (system + first user + multimodal).
        # The pre-fix loop tested the limit AFTER appending and so
        # always preserved at least one group even when keep_recent
        # was 0; flip the check to BEFORE appending so the bound holds.
        recent_groups: list[int] = []
        seen_groups: set[int] = set()
        for i in range(len(messages) - 1, -1, -1):
            if len(recent_groups) >= self.keep_recent:
                break
            g = group_id[i]
            if g in seen_groups:
                continue
            seen_groups.add(g)
            recent_groups.append(g)
        recent_groups_set = set(recent_groups)

        # Decide drop set: every index whose group is NOT in the recent
        # window AND that is not an anchor.
        droppable: list[int] = []
        for i in range(len(messages)):
            if i in anchor_idx:
                continue
            if group_id[i] in recent_groups_set:
                continue
            droppable.append(i)

        # Drop oldest-first until under threshold or nothing left.
        dropped: set[int] = set()
        for i in droppable:
            kept = [m for j, m in enumerate(messages) if j not in dropped and j != i]
            if estimate_tokens(kept) <= threshold:
                dropped.add(i)
                break
            dropped.add(i)

        # When dropping an assistant-with-tool-calls message we must
        # also drop the matching tool-role messages (and vice versa)
        # so the chat template stays valid. Iterate pair_map once.
        # Anchor indices stay regardless: dragging an anchored
        # multimodal assistant or first-user message into the drop
        # set just because its tool-pair partner was dropped would
        # violate the structural invariant the anchor set exists to
        # enforce, and llama-server would 400 on the resulting
        # template (a tool message whose tool_call_id has no
        # surviving assistant tool_calls entry).
        # Assistants whose tool_calls all got orphaned are repaired in
        # the final pass below: we keep the multimodal content but
        # strip the dangling tool_calls so OpenAI does not 400 on
        # "assistant message with tool_calls must be followed by tool
        # messages".
        rewrite_strip_tool_calls: set[int] = set()
        for asst_idx, tool_idxs in pair_map.items():
            if asst_idx in dropped:
                dropped.update(t for t in tool_idxs if t not in anchor_idx)
            elif tool_idxs and tool_idxs <= dropped:
                if asst_idx not in anchor_idx:
                    dropped.add(asst_idx)
                else:
                    rewrite_strip_tool_calls.add(asst_idx)

        # Final invariant sweep, two passes so the asst-strip decision
        # sees the post-orphan-drop tool set:
        #   pass 1: drop tools whose tcid has no matching assistant
        #     ``tool_calls`` earlier in the kept output (anchored tools
        #     stay; same leak-rather-than-violate-anchor rule);
        #   pass 2: recompute responded_ids from the surviving tools and
        #     mark assts whose tool_calls aren't all responded for strip.
        # Computing responded_ids before pass 1 would let a stale tcid
        # of a just-dropped orphan tool satisfy `ids <= responded_ids`,
        # leaving the asst with dangling tool_calls (OpenAI 400).
        seen_ids: set[str] = set()
        for i, m in enumerate(messages):
            if i in dropped:
                continue
            if m.get("role") == "assistant":
                seen_ids |= _assistant_tool_call_ids(m)
            elif m.get("role") == "tool" and i not in anchor_idx:
                tcid = m.get("tool_call_id")
                if isinstance(tcid, str) and tcid and tcid not in seen_ids:
                    dropped.add(i)
        responded_ids: set[str] = set()
        for i, m in enumerate(messages):
            if i in dropped:
                continue
            if m.get("role") == "tool":
                tcid = m.get("tool_call_id")
                if isinstance(tcid, str) and tcid:
                    responded_ids.add(tcid)
        for i, m in enumerate(messages):
            if i in dropped:
                continue
            if m.get("role") == "assistant":
                ids = _assistant_tool_call_ids(m)
                if ids and not (ids <= responded_ids):
                    rewrite_strip_tool_calls.add(i)

        out: list[dict] = []
        for i, m in enumerate(messages):
            if i in dropped:
                continue
            if i in rewrite_strip_tool_calls:
                # Keep the message (multimodal content stays) but strip
                # tool_calls entries with no surviving tool follow-up.
                kept_tcs = [
                    tc
                    for tc in (m.get("tool_calls") or [])
                    if isinstance(tc, dict)
                    and isinstance(tc.get("id"), str)
                    and tc["id"] in responded_ids
                ]
                copy = dict(m)
                if kept_tcs:
                    copy["tool_calls"] = kept_tcs
                else:
                    copy.pop("tool_calls", None)
                out.append(copy)
            else:
                out.append(m)
        return out


_STRATEGIES: dict[str, CompactStrategy] = {
    "none": NoCompact(),
    "sliding": SlidingWindowCompact(),
}


def get_strategy(name: str) -> CompactStrategy:
    """Return the compaction strategy registered under ``name``.

    Falls back to ``NoCompact`` for unknown names so a misconfigured
    request degrades to no-op rather than raising.
    """
    return _STRATEGIES.get(name, _STRATEGIES["none"])
