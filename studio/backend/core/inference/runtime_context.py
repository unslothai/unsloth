# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Runtime context length helpers shared by inference backends."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from itertools import chain
from typing import Any, Optional

#: Longest context a load may ask for, and the ceiling a resolved window is held to.
#: LoadRequest.max_seq_length bounds requests by this; a backend that reads a wider
#: window from the model reports it as native but does not serve past it.
MAX_REQUESTABLE_CONTEXT = 1048576

#: Budget for an unset limit when there is no free context to size it from.
UNSET_GENERATION_BUDGET = 2048


def _field(source: Any, name: str) -> Any:
    """Key or attribute, since mlx.nn.Module is a dict; a raiser is absent, never a failed load."""
    try:
        if isinstance(source, Mapping) and name in source:
            return source[name]
        return getattr(source, name, None)
    except Exception:
        return None


def _attached_window(model: Any) -> Any:
    """What Unsloth attached: getattr only, so a Mapping model's parameters cannot pose as it."""
    try:
        return getattr(model, "max_seq_length", None)
    except Exception:
        return None


def _declared_context_lengths(model: Any) -> Iterator[Any]:
    """Declared windows, best first. Yields, so an outer 0 / "n/a" cannot shadow a real one."""
    holders = (
        model,
        # config / _config: the spread _mlx_config_field walks in mlx_inference.py.
        _field(model, "config"),
        _field(model, "_config"),
        _field(model, "args"),
    )
    for holder in holders:
        if holder is None:
            continue
        for source in (holder, _field(holder, "text_config")):
            if source is None:
                continue
            value = _field(source, "max_position_embeddings")
            if value is not None:
                yield value


def runtime_context_length(model: Any, fallback: Optional[int] = None) -> Optional[int]:
    """Return the effective context length a loaded model runs with."""
    # Lazy: a transformers load always has a requested length, so it never reads the config.
    candidates = chain(
        (_attached_window(model), fallback),
        _declared_context_lengths(model),
    )
    for value in candidates:
        if isinstance(value, bool):
            continue
        try:
            value_int = int(value)
        # OverflowError: json.loads turns a bare Infinity into float("inf"), which int() rejects.
        except (TypeError, ValueError, OverflowError):
            continue
        if value_int > 0:
            return value_int
    return None


def generation_budget_for_window(
    window: Optional[int], prompt_length: int, max_new_tokens: Optional[int]
) -> Optional[int]:
    """Resolve a generation budget, where ``None`` means the caller set no limit.

    An unset limit becomes the context the prompt leaves free. An explicit one is returned
    untouched, so asking for more than fits still gets the backend's overflow error.
    """
    if max_new_tokens is not None:
        return max_new_tokens
    if not window:
        return UNSET_GENERATION_BUDGET
    free = int(window) - int(prompt_length)
    # No room left: take the default and let the backend's overflow check decide. A floor
    # of 1 would pass that check on a model loaded narrower than its checkpoint.
    return free if free > 0 else UNSET_GENERATION_BUDGET


def generation_budget_within_context(
    model: Any, prompt_length: int, max_new_tokens: Optional[int]
) -> Optional[int]:
    """``generation_budget_for_window`` against the window a loaded model declares."""
    return generation_budget_for_window(
        runtime_context_length(model), prompt_length, max_new_tokens
    )
