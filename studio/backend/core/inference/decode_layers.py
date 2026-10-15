# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Install and remove one wrapper layer on an object's method slot (a VAE's ``decode`` / ``_decode``), exactly.

``wrap`` records what the slot held on that object (an instance-dict entry, possibly an instance-bound override, or
nothing, so the class method showed through) and puts the layer in the instance dict. ``unwrap`` puts back exactly
that when the layer is still outermost. When later wrappers captured it (they hold it in their closures), the layer
cannot be cut out: it turns into a pass-through to what it wrapped, and the ``__wrapped__`` chain skips it.
"""

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
        self.inner = inner  # what the layer calls through to
        self.original = original  # the owner's own instance-dict entry before wrapping, or _MISSING
        self.active = True
        self.fn: Optional[Callable] = None

    @property
    def had_own(self) -> bool:
        return self.original is not _MISSING


def wrap(owner: Any, attr: str, name: str, make: Callable[[Callable, "Layer"], Callable]) -> Layer:
    """Install ``make(inner, layer)`` as the outermost ``attr`` layer of ``owner``. The installed callable copies the
    made function's attributes (markers such as ``_unsloth_compiled_decode``) and passes straight through to ``inner``
    once the layer is removed."""
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
    """Remove ``layer``. True when it was outermost and the slot holds exactly what it held before wrapping; False
    when later wrappers sit over it (it is now a pass-through) or it was already removed."""
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
        # A saved entry that is itself a removed (pass-through) layer collapses to what that layer had replaced.
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
    # Buried: later wrappers captured this layer. Point the chain past it for anyone walking ``__wrapped__``.
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
        fn = getattr(fn, "__wrapped__", None)
    return False


__all__ = ["LAYERS_ATTR", "Layer", "find", "unwrap", "wrap"]
