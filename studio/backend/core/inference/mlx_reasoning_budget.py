# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""llama-server's --reasoning-budget for MLX: after ``budget`` reasoning tokens, force the budget
message and the reasoning closer, then let the reply continue."""

from __future__ import annotations

from typing import Optional

from core.inference.chat_template_helpers import (
    _ATEM_REASONING_RECIPIENT,
    prompt_opens_reasoning_channel,
)

THINK_MARKERS = ("<think>", "</think>")


def _encode(tokenizer, text: str) -> list[int]:
    if not text:
        return []
    try:
        return list(tokenizer.encode(text, add_special_tokens = False))
    except TypeError:
        return list(tokenizer.encode(text))


def _ends_with(history: list[int], ids: list[int]) -> bool:
    return bool(ids) and history[-len(ids) :] == ids


def make_reasoning_budget_processor(
    tokenizer, budget: int, message: str, markers: Optional[tuple[str, ...]], prompt: Optional[str]
):
    """A logits processor, or None when no budget applies. ``markers`` are the template's native
    reasoning channel; without one the model reasons in ``<think>`` tags."""
    if budget is None or budget < 0:
        return None
    if markers and markers[0] == _ATEM_REASONING_RECIPIENT:
        # Recipient-addressed blocks end by framing, not a closing tag there is to force.
        return None
    opener, closer = tuple(markers[:2]) if markers and len(markers) >= 2 else THINK_MARKERS
    open_ids, close_ids = _encode(tokenizer, opener), _encode(tokenizer, closer)
    if not open_ids or not close_ids:
        return None
    forced = _encode(tokenizer, message) + close_ids
    state = {
        "seen": None,
        "history": [],
        "inside": prompt_opens_reasoning_channel(prompt, (opener, closer)),
        "spent": 0,
        "forcing": None,
        "done": False,
    }

    def _advance(token_id: int) -> None:
        history = state["history"]
        history.append(token_id)
        if state["forcing"] is not None:
            state["forcing"] += 1
            if state["forcing"] >= len(forced):
                state.update(forcing = None, inside = False, done = True)
        elif state["inside"]:
            if _ends_with(history, close_ids):
                state.update(inside = False, done = True)
            else:
                state["spent"] += 1
        elif not state["done"] and _ends_with(history, open_ids):
            state.update(inside = True, spent = 0)

    def _processor(tokens, logits):
        seen = state["seen"]
        if seen is not None:
            for token_id in tokens[seen:].tolist():
                _advance(int(token_id))
        state["seen"] = int(tokens.shape[0])
        if state["forcing"] is None and state["inside"] and state["spent"] >= budget:
            state["forcing"] = 0
        if state["forcing"] is None:
            return logits
        import mlx.core as mx

        wanted = forced[state["forcing"]]
        keep = mx.arange(logits.shape[-1]) == wanted
        return mx.where(keep, logits, mx.array(-float("inf"), dtype = logits.dtype))

    return _processor
