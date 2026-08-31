# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Runtime context length helpers shared by inference backends."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional


def _field(source: Any, name: str) -> Any:
    # mlx.nn.Module IS a dict, but its keys are parameters and its config is an attribute.
    if isinstance(source, Mapping) and name in source:
        return source[name]
    return getattr(source, name, None)


def _declared_context_length(model: Any) -> Any:
    """The window the model's own config declares."""
    for holder in (model, _field(model, "config"), _field(model, "args")):
        if holder is None:
            continue
        for source in (holder, _field(holder, "text_config")):
            if source is None:
                continue
            value = _field(source, "max_position_embeddings")
            if value is not None:
                return value
    return None


def runtime_context_length(model: Any, fallback: Optional[int] = None) -> Optional[int]:
    """Return the effective context length a loaded model runs with."""
    for value in (
        getattr(model, "max_seq_length", None),
        fallback,
        _declared_context_length(model),
    ):
        if isinstance(value, bool):
            continue
        try:
            value_int = int(value)
        except (TypeError, ValueError):
            continue
        if value_int > 0:
            return value_int
    return None
