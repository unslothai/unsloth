# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Runtime context length helpers shared by inference backends."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from itertools import chain
from typing import Any, Optional


def _field(source: Any, name: str) -> Any:
    """Read *name* off *source* however it is exposed, or None.

    mlx.nn.Module IS a dict, but its keys are parameters and its config is an
    attribute, so both shapes have to be tried. Anything a lookup raises is an
    absent field here: this runs on every load, against wrapper objects nobody
    controls (PEFT, accelerate, a config behind a property that needs a file),
    and a resolver that cannot answer must not be able to fail the load.
    """
    try:
        if isinstance(source, Mapping) and name in source:
            return source[name]
        return getattr(source, name, None)
    except Exception:
        return None


def _attached_window(model: Any) -> Any:
    """The window Unsloth attached to the model, if any.

    Plain getattr, so a Mapping model's parameters are not mistaken for it, but
    guarded for the same reason _field is: nothing here may fail a load.
    """
    try:
        return getattr(model, "max_seq_length", None)
    except Exception:
        return None


def _declared_context_lengths(model: Any) -> Iterator[Any]:
    """Every window the model's own config declares, best source first.

    Yields rather than returning the first hit so the caller validates each in
    turn: a wrapper whose outer metadata carries a placeholder (0, "n/a") would
    otherwise shadow the real window sitting on the config below it.
    """
    holders = (
        model,
        # config / _config is the same spread _mlx_config_field walks in
        # mlx_inference.py -- a checkpoint's config is a dict on some models and
        # an object on others, under either name.
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
    # Lazy on purpose: the declared window is only consulted once an attached or
    # requested length has failed, so a transformers load -- which always has a
    # requested length -- never touches the model's config at all.
    candidates = chain(
        (_attached_window(model), fallback),
        _declared_context_lengths(model),
    )
    for value in candidates:
        if isinstance(value, bool):
            continue
        try:
            value_int = int(value)
        # OverflowError because json.loads turns a bare Infinity in a hand-edited
        # config.json into float("inf"), and int() refuses that one alone.
        except (TypeError, ValueError, OverflowError):
            continue
        if value_int > 0:
            return value_int
    return None
