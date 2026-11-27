# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tensor helpers of ``diffusers.hooks.group_offloading`` across diffusers versions.

diffusers <= 0.42 moves torchao subclasses specially (``_is_torchao_tensor``, ``_swap_torchao_tensor``,
``_restore_torchao_tensor``, ``_record_stream_torchao_tensor``) and every other tensor through ``.data``. Since
huggingface/diffusers#15383 it moves any traceable wrapper subclass instead (``is_traceable_wrapper_subclass``,
``_swap_wrapper_tensor``, ``_record_stream_wrapper_tensor``; a restore swaps in a rebuild of the host copy). Studio
code that mirrors or wraps those moves resolves them here, so it treats a tensor the way the installed diffusers
does. The torchao names win when present, so a diffusers that has them behaves exactly as before.
"""

from typing import Any, Callable, Optional

LEGACY_IS = "_is_torchao_tensor"
LEGACY_SWAP = "_swap_torchao_tensor"
LEGACY_RESTORE = "_restore_torchao_tensor"
LEGACY_RECORD = "_record_stream_torchao_tensor"
WRAPPER_SWAP = "_swap_wrapper_tensor"
WRAPPER_RECORD = "_record_stream_wrapper_tensor"
WRAPPER_APPLY = "_apply_to_inner_tensors"


def group_offloading() -> Any:
    """``diffusers.hooks.group_offloading``, or None without group offloading."""
    try:
        from diffusers.hooks import group_offloading as go
    except Exception:  # noqa: BLE001 - no diffusers / no group offloading
        return None
    return go


def _resolve(go: Any) -> Any:
    return group_offloading() if go is None else go


def is_legacy(go: Any = None) -> bool:
    """diffusers with the torchao-only helpers (<= 0.42)."""
    go = _resolve(go)
    return go is not None and callable(getattr(go, LEGACY_IS, None))


def is_generic(go: Any = None) -> bool:
    """diffusers that moves every traceable wrapper subclass itself (huggingface/diffusers#15383)."""
    go = _resolve(go)
    return go is not None and not is_legacy(go) and callable(getattr(go, WRAPPER_SWAP, None))


def _traceable_wrapper(go: Any) -> Optional[Callable[[Any], bool]]:
    fn = getattr(go, "is_traceable_wrapper_subclass", None)
    if callable(fn):
        return fn
    try:
        from torch.utils._python_dispatch import is_traceable_wrapper_subclass
    except Exception:  # noqa: BLE001
        return None
    return is_traceable_wrapper_subclass


def is_wrapper_fn(go: Any = None) -> Optional[Callable[[Any], bool]]:
    """The predicate diffusers uses to pick the swap path over ``.data``; None when it has neither."""
    go = _resolve(go)
    if go is None:
        return None
    legacy = getattr(go, LEGACY_IS, None)
    if callable(legacy):
        return legacy
    if is_generic(go):
        return _traceable_wrapper(go)
    return None


def _never(tensor: Any) -> bool:
    return False


def is_wrapper_or_false(go: Any = None) -> Callable[[Any], bool]:
    """``is_wrapper_fn`` with an always-False fallback (the old ``getattr(go, ..., lambda t: False)``)."""
    return is_wrapper_fn(go) or _never


def swap_name(go: Any = None) -> Optional[str]:
    """Module attribute diffusers' own onload / offload call to swap a wrapper weight (wrap this one)."""
    go = _resolve(go)
    if go is None:
        return None
    if is_legacy(go):
        return LEGACY_SWAP if callable(getattr(go, LEGACY_SWAP, None)) else None
    return WRAPPER_SWAP if is_generic(go) else None


def swap_fn(go: Any = None) -> Optional[Callable[[Any, Any], None]]:
    """The current swap helper (a Studio wrapper around it included), or None."""
    go = _resolve(go)
    name = swap_name(go)
    return getattr(go, name) if name else None


def swap(
    param: Any,
    source: Any,
    go: Any = None,
) -> None:
    """Swap ``source`` into wrapper ``param`` via diffusers' helper; ``source`` gets ``param``'s old contents."""
    fn = swap_fn(go)
    if fn is None:
        raise AttributeError("diffusers group offloading has no wrapper-tensor swap helper")
    fn(param, source)


def wrapper_swap_supported(go: Any = None) -> bool:
    """Whether group offloading moves wrapper weights whole (torchao on 0.40+, any wrapper after #15383)."""
    go = _resolve(go)
    return callable(is_wrapper_fn(go)) and callable(swap_fn(go))


def restore_name(go: Any = None) -> Optional[str]:
    """Module attribute diffusers calls to point a wrapper back at its host copy; None after #15383 (inlined)."""
    go = _resolve(go)
    if go is not None and is_legacy(go) and callable(getattr(go, LEGACY_RESTORE, None)):
        return LEGACY_RESTORE
    return None


def record_name(go: Any = None) -> Optional[str]:
    """Module attribute diffusers calls to record a stream on a wrapper's inner tensors."""
    go = _resolve(go)
    if go is None:
        return None
    if is_legacy(go):
        return LEGACY_RECORD if callable(getattr(go, LEGACY_RECORD, None)) else None
    if is_generic(go) and callable(getattr(go, WRAPPER_RECORD, None)):
        return WRAPPER_RECORD
    return None


def restore_fn(go: Any = None) -> Optional[Callable[[Any, Any], None]]:
    """Point wrapper ``param`` back at host copy ``source`` without mutating ``source``, as diffusers' offload does."""
    go = _resolve(go)
    name = restore_name(go)
    if name:
        return getattr(go, name)
    if not is_generic(go):
        return None
    apply = getattr(go, WRAPPER_APPLY, None)

    def _restore(param: Any, source: Any) -> None:
        # diffusers main: swap in a fresh wrapper over the host copy's inner tensors (the copy itself stays put)
        if callable(apply):
            rebuilt = apply(source, lambda t: t)
        else:
            names, ctx = source.__tensor_flatten__()
            rebuilt = type(source).__tensor_unflatten__(
                {n: getattr(source, n) for n in names}, ctx, source.shape, source.stride()
            )
        swap(param, rebuilt, go)

    return _restore


def restore(
    param: Any,
    source: Any,
    go: Any = None,
) -> None:
    fn = restore_fn(go)
    if fn is None:
        raise AttributeError("diffusers group offloading has no wrapper-tensor restore")
    fn(param, source)


def record_stream_fn(go: Any = None) -> Optional[Callable[[Any, Any], None]]:
    go = _resolve(go)
    name = record_name(go)
    return getattr(go, name) if name else None


def record_stream(
    param: Any,
    stream: Any,
    go: Any = None,
) -> None:
    fn = record_stream_fn(go)
    if fn is None:
        raise AttributeError(
            "diffusers group offloading has no wrapper-tensor record_stream helper"
        )
    fn(param, stream)
