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


# Conservative char-to-token ratio so compaction fires early.
_CHARS_PER_TOKEN = 4


def estimate_tokens(messages: list[dict]) -> int:
    """Rough token count via 4-char-per-token heuristic.

    Counts visible content, multimodal text parts, ``compaction``
    summary text, and serialized tool_calls arguments.
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
                # Count compaction summary text.
                if part.get("type") == "compaction":
                    summary = part.get("content")
                    if isinstance(summary, str):
                        total_chars += len(summary)
        tcs = m.get("tool_calls")
        if isinstance(tcs, list):
            for tc in tcs:
                # Skip malformed pre-pydantic entries.
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function")
                args = (fn or {}).get("arguments") if isinstance(fn, dict) else None
                if isinstance(args, str):
                    total_chars += len(args)
    # Ceil-divide: floor would let just-over-budget prompts bypass compaction.
    return -(-total_chars // _CHARS_PER_TOKEN)


# Text-only parts (don't anchor); ``compaction`` is Anthropic round-trip
# state so pinning it would defeat compaction.
_TEXT_ONLY_PART_TYPES = {"text", "compaction"}


def _is_multimodal(msg: dict) -> bool:
    content = msg.get("content")
    if not isinstance(content, list):
        return False
    for part in content:
        # Unknown shape => conservatively treat as multimodal.
        if not isinstance(part, dict):
            return True
        if part.get("type") not in _TEXT_ONLY_PART_TYPES:
            return True
    return False


def _assistant_tool_call_ids(msg: dict) -> set[str]:
    """Return tool_call ids on an assistant message; skip malformed entries."""
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
    """Map asst index -> indices of its tool-role follow-ups so the
    compactor drops or keeps the group as a unit.
    """
    out: dict[int, set[int]] = {}
    # OpenAI schema: tool messages must follow their asst directly.
    # ANY non-tool boundary (user, system, another asst) clears the
    # pending window so a stale asst can't snap to a later-turn tool.
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
    """Keep system, first user, multimodal, and last ``keep_recent``
    groups. Asst+tools form one group. No-op when within budget.
    """

    def __init__(
        self,
        keep_recent: int = 2,
        compact_threshold: float = 0.85,
    ) -> None:
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

        # Group asst tool-call with its tool responses so they drop together.
        pair_map = _pair_linked_indices(messages)
        # If any pair member is anchored, anchor the whole group; otherwise
        # an anchored multimodal asst/tool could orphan its partner.
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

        # Last ``keep_recent`` distinct groups from the end. The limit
        # check runs BEFORE appending so ``keep_recent == 0`` keeps zero.
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

        # Droppable = non-anchor indices outside the recent window.
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

        # Drop the partner when one side of a pair is dropped, except
        # anchored asst whose tools all died -- keep their content but
        # strip the dangling tool_calls in the final pass.
        rewrite_strip_tool_calls: set[int] = set()
        for asst_idx, tool_idxs in pair_map.items():
            if asst_idx in dropped:
                dropped.update(t for t in tool_idxs if t not in anchor_idx)
            elif tool_idxs and tool_idxs <= dropped:
                if asst_idx not in anchor_idx:
                    dropped.add(asst_idx)
                else:
                    rewrite_strip_tool_calls.add(asst_idx)

        # Two-pass invariant sweep:
        #   1) drop tools whose tcid has no prior surviving asst
        #      (anchored ones too -- pair-validity beats anchor rule);
        #   2) recompute responded_ids from survivors, then mark assts
        #      with unanswered tool_calls for strip.
        # Order matters: a stale orphan tcid in responded_ids would let
        # the asst keep dangling tool_calls (OpenAI 400).
        seen_ids: set[str] = set()
        for i, m in enumerate(messages):
            if i in dropped:
                continue
            if m.get("role") == "assistant":
                seen_ids |= _assistant_tool_call_ids(m)
            elif m.get("role") == "tool":
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
                # Keep content but strip dangling tool_calls.
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
    """Return strategy by name; unknown names fall back to ``NoCompact``."""
    return _STRATEGIES.get(name, _STRATEGIES["none"])
