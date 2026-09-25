# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Keep a compiled denoiser's prefix KV cache at its real size.

Qwen-Image-2.1 (and FLUX.2 klein KV, Wan-Animate-2) store each block's prefix keys and values on
the prefill step as ``key[:, prefix].clone()``. Under a regional compile inductor lowers that clone
to a view of the block's freshly allocated full-sequence K/V buffer, so the cache pins every
block's whole prefill K/V for the rest of the render: 2 GiB at 1024x1024 on Qwen-Image-2.1, where
the prefix it needs is a few MB. A forward hook copies the cached tensors out once, after the
prefill call, which frees those buffers. The values are unchanged.
"""

from __future__ import annotations

import inspect
from typing import Any

_HOOK_ATTR = "_unsloth_prefix_kv_compaction"


def _takes_prefix_kv(module: Any) -> bool:
    try:
        params = inspect.signature(module.forward).parameters
    except (TypeError, ValueError):
        return False
    return "kv_cache" in params and "kv_cache_mode" in params


def compact_prefix_kv_cache(cache: Any) -> int:
    """Replace every cached tensor that is a view into a larger buffer with its own copy. Returns
    the bytes of the buffers that are no longer referenced from the cache."""
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


def install_prefix_kv_compaction(transformer: Any, logger: Any = None) -> bool:
    """Hook ``transformer`` so its prefix KV cache is compacted after the prefill forward. No-op for
    a denoiser without a prefix KV cache, and idempotent."""
    if getattr(transformer, _HOOK_ATTR, None) is not None or not _takes_prefix_kv(transformer):
        return False

    def _after_prefill(_module, _args, kwargs, _output):
        if kwargs.get("kv_cache_mode") != "extract":
            return None
        cache = kwargs.get("kv_cache")
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
