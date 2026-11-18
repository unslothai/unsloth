# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Prefix KV clone is copied out after prefill: Inductor makes it a view that pins the K/V buffer."""

from __future__ import annotations

import contextlib
import inspect
from typing import Any

_HOOK_ATTR = "_unsloth_prefix_kv_compaction"


def _takes_prefix_kv(module: Any) -> bool:
    try:
        params = inspect.signature(module.forward).parameters
    except Exception:  # noqa: BLE001 - no inspectable forward: nothing to hook
        return False
    return "kv_cache" in params and "kv_cache_mode" in params


def compact_prefix_kv_cache(cache: Any) -> int:
    """Copy out cached tensors that view a larger buffer; returns bytes released."""
    import torch

    released = 0
    try:
        groups = [v for v in vars(cache).values() if isinstance(v, (list, tuple))]
    except TypeError:
        return 0
    for group in groups:
        for layer in group:
            slots = getattr(layer, "__dict__", None)
            if not slots:
                continue
            for name, tensor in list(slots.items()):
                if not isinstance(tensor, torch.Tensor) or tensor.device.type == "meta":
                    continue
                used = tensor.numel() * tensor.element_size()
                held = tensor.untyped_storage().nbytes()
                if held > used:
                    setattr(layer, name, tensor.clone(memory_format = torch.contiguous_format))
                    released += held - used
    return released


def _returned_cache(output: Any) -> Any:
    # FLUX.2 builds the cache in the extract forward and returns it.
    cache = getattr(output, "kv_cache", None)
    if cache is None and isinstance(output, tuple):
        cache = next((o for o in output[1:] if o is not None and not hasattr(o, "shape")), None)
    return cache


_SEEN_ATTR = "_unsloth_prefix_kv_seen"
_TRACK_HOOK_ATTR = "_unsloth_prefix_kv_track"


def track_prefix_kv(transformer: Any) -> bool:
    """Weakly record the denoiser's prefix KV caches: Qwen-Image-2.1 keeps them as ``__call__`` locals through
    VAE decode (~2 GiB per guidance branch at 1 MP)."""
    if transformer is None or getattr(transformer, _TRACK_HOOK_ATTR, None) is not None:
        return False
    if not _takes_prefix_kv(transformer):
        return False
    import weakref

    seen: list = []

    def _record(_module, _args, kwargs):
        if kwargs.get("kv_cache_mode") != "extract":
            return None
        cache = kwargs.get("kv_cache")
        if cache is None:
            return None
        with contextlib.suppress(TypeError):
            if not any(ref() is cache for ref in seen):
                seen[:] = [ref for ref in seen if ref() is not None]
                seen.append(weakref.ref(cache))
        return None

    try:
        handle = transformer.register_forward_pre_hook(_record, with_kwargs = True)
    except Exception:  # noqa: BLE001 - a memory saving only
        return False
    setattr(transformer, _SEEN_ATTR, seen)
    setattr(transformer, _TRACK_HOOK_ATTR, handle)
    return True


def release_prefix_kv(transformer: Any) -> int:
    """Set tracked caches' per-layer tensors to None; returns bytes freed. Only safe after the last denoiser call."""
    import torch

    seen = getattr(transformer, _SEEN_ATTR, None) if transformer is not None else None
    if not seen:
        return 0
    released = 0
    caches = [ref() for ref in seen]
    seen.clear()
    for cache in caches:
        if cache is None:
            continue
        try:
            groups = [v for v in vars(cache).values() if isinstance(v, (list, tuple))]
        except TypeError:
            continue
        for group in groups:
            for layer in group:
                slots = getattr(layer, "__dict__", None)
                if not slots:
                    continue
                for name, tensor in list(slots.items()):
                    if isinstance(tensor, torch.Tensor):
                        released += tensor.untyped_storage().nbytes()
                        setattr(layer, name, None)
    return released


def install_prefix_kv_compaction(transformer: Any, logger: Any = None) -> bool:
    if getattr(transformer, _HOOK_ATTR, None) is not None or not _takes_prefix_kv(transformer):
        return False

    def _after_prefill(_module, _args, kwargs, _output):
        if kwargs.get("kv_cache_mode") != "extract":
            return None
        cache = kwargs.get("kv_cache")
        if cache is None:
            cache = _returned_cache(_output)
        if cache is not None:
            compact_prefix_kv_cache(cache)
        return None

    try:
        handle = transformer.register_forward_hook(_after_prefill, with_kwargs = True)
    except Exception as exc:  # noqa: BLE001 - a memory saving only
        if logger is not None:
            logger.warning("diffusion.prefix_kv: compaction hook unavailable (%s)", exc)
        return False
    setattr(transformer, _HOOK_ATTR, handle)
    return True
