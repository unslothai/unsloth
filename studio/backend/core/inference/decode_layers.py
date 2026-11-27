# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Wrap and unwrap a layer on an instance method slot (e.g. VAE decode): unwrap restores the exact
prior entry only when outermost; else it is a pass-through skipped via __wrapped__."""

from __future__ import annotations

import functools
from typing import Any, Callable, Optional

LAYERS_ATTR = "_unsloth_decode_layers"
_MISSING = object()


class Layer:
    def __init__(self, owner: Any, attr: str, name: str, inner: Callable, original: Any) -> None:
        self.owner = owner
        self.attr = attr
        self.name = name
        self.inner = inner
        self.original = original  # the owner's own instance-dict entry before wrapping, or _MISSING
        self.active = True
        self.fn: Optional[Callable] = None

    @property
    def had_own(self) -> bool:
        return self.original is not _MISSING


def wrap(owner: Any, attr: str, name: str, make: Callable[[Callable, "Layer"], Callable]) -> Layer:
    """Install ``make(inner, layer)`` as the outermost ``attr`` layer of ``owner``."""
    original = owner.__dict__.get(attr, _MISSING)
    inner = getattr(owner, attr)
    layer = Layer(owner, attr, name, inner, original)
    fn = make(inner, layer)

    def dispatch(*args: Any, **kwargs: Any) -> Any:
        if not layer.active:
            return layer.inner(*args, **kwargs)
        return fn(*args, **kwargs)

    functools.update_wrapper(dispatch, fn)
    dispatch.__wrapped__ = inner
    dispatch._unsloth_layer = layer
    layer.fn = dispatch
    setattr(owner, attr, dispatch)
    owner.__dict__.setdefault(LAYERS_ATTR, []).append(layer)
    return layer


def find(owner: Any, attr: str, name: str) -> Optional[Layer]:
    for layer in reversed(getattr(owner, "__dict__", {}).get(LAYERS_ATTR, ()) or ()):
        if layer.attr == attr and layer.name == name and layer.active:
            return layer
    return None


def unwrap(layer: Optional[Layer]) -> bool:
    """True if outermost and slot holds its prior entry, else False (pass-through or removed)."""
    if layer is None or not layer.active:
        return False
    owner, attr = layer.owner, layer.attr
    layer.active = False
    layers = owner.__dict__.get(LAYERS_ATTR, [])
    if layer in layers:
        layers.remove(layer)
    if not layers:
        owner.__dict__.pop(LAYERS_ATTR, None)
    current = owner.__dict__.get(attr, _MISSING)
    if current is layer.fn:
        original = layer.original
        for _ in range(16):
            inner_layer = getattr(original, "_unsloth_layer", None)
            if inner_layer is None or inner_layer.active or original is not inner_layer.fn:
                break
            original = inner_layer.original
        if original is not _MISSING:
            owner.__dict__[attr] = original
        else:
            owner.__dict__.pop(attr, None)
        return True
    fn = None if current is _MISSING else current
    for _ in range(16):
        if fn is None:
            break
        if getattr(fn, "__wrapped__", None) is layer.fn:
            try:
                fn.__wrapped__ = layer.inner
            except Exception:  # noqa: BLE001 - read-only callable: the pass-through still holds
                pass
            break
        fn = next_layer(fn)
    return False


def next_layer(fn: Any) -> Any:
    """The decode a wrapper calls through: its mutable slot when set (eager outer wrappers keep what
    they wrap rewritable), else __wrapped__; all walkers use it so no slot is skipped."""
    slot = getattr(fn, "_unsloth_decode_slot", None)
    if slot is not None:
        inner = getattr(slot, "__dict__", {}).get("decode")
        if inner is not None:
            return inner
    return getattr(fn, "__wrapped__", None)


__all__ = ["LAYERS_ATTR", "Layer", "find", "next_layer", "unwrap", "wrap"]
