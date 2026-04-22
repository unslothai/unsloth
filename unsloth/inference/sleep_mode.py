# SPDX-License-Identifier: GNU Affero General Public License v3.0
# Copyright 2023-present the Unsloth team. All rights reserved.

"""Sleep-mode helpers for the flex inference backend.

When ``UNSLOTH_VLLM_STANDBY=1`` is set and vLLM's ``CuMemAllocator`` is
importable, :class:`~unsloth.inference.flex_engine.FlexEngine` routes its
heavy GPU allocations (the inference deep-copies, the PEFT wrapper, and
per-layer :class:`~unsloth.inference.flex_paged_attention.PagedKVCache`
buffers) through cuMem-backed pools. ``sleep(level=1)`` then offloads the
``weights`` pool to pinned CPU memory and discards the ``kv_cache`` pool
outright; ``wake_up`` re-maps the handles at the same virtual addresses
so previously captured CUDA graphs and compiled artifacts stay valid.

If vLLM is not installed, :func:`_get_cumem_allocator` returns ``None``
and the helpers fall back to :func:`contextlib.nullcontext`, which keeps
the flex engine working without sleep support.
"""

from __future__ import annotations

import contextlib
import os
import warnings
from typing import Any, Optional


_WARNED_NO_VLLM = False


def _get_cumem_allocator() -> Optional[Any]:
    """Return ``vllm.device_allocator.cumem.CuMemAllocator.get_instance()``
    or ``None`` if vLLM is not importable.

    Emits a single warning on the first failed import so users who opt
    into sleep mode with ``UNSLOTH_VLLM_STANDBY=1`` see a clear message
    about the soft dependency on vLLM."""
    global _WARNED_NO_VLLM
    try:
        from vllm.device_allocator.cumem import CuMemAllocator
    except Exception as e:
        if not _WARNED_NO_VLLM:
            warnings.warn(
                "FlexEngine sleep mode requires vLLM's CuMemAllocator "
                f"(import failed: {e}). Sleep / wake_up will be no-ops. "
                "Install vLLM to enable level-1 sleep mode on the flex "
                "backend.",
                RuntimeWarning,
                stacklevel = 2,
            )
            _WARNED_NO_VLLM = True
        return None
    return CuMemAllocator.get_instance()


def sleep_mode_enabled() -> bool:
    """``True`` iff ``UNSLOTH_VLLM_STANDBY=1`` is set AND vLLM is
    available. Evaluated at :class:`FlexEngine.__init__` time so the
    choice of allocator is stable for the engine's lifetime."""
    if os.environ.get("UNSLOTH_VLLM_STANDBY", "0") != "1":
        return False
    return _get_cumem_allocator() is not None


def _pool(allocator: Optional[Any], tag: str):
    """Return a ``use_memory_pool(tag=...)`` context manager if the
    allocator is available, else ``nullcontext`` so callers can wrap
    allocation sites unconditionally."""
    if allocator is None:
        return contextlib.nullcontext()
    return allocator.use_memory_pool(tag = tag)


def weight_pool(allocator: Optional[Any]):
    """Context manager for ``tag="weights"`` allocations (offloaded to
    pinned CPU on sleep, restored via ``cudaMemcpy`` on wake)."""
    return _pool(allocator, "weights")


def kv_cache_pool(allocator: Optional[Any]):
    """Context manager for ``tag="kv_cache"`` allocations (discarded on
    sleep, re-mapped with zeros on wake)."""
    return _pool(allocator, "kv_cache")


def describe_sleep_state(engine) -> dict:
    """Return a small dict summarising the flex engine's sleep-mode
    state plus a torch CUDA memory snapshot. Useful from bench scripts
    and tests."""
    import torch

    allocator = getattr(engine, "_cumem_allocator", None)
    state: dict = {
        "sleep_mode_enabled": bool(getattr(engine, "_sleep_mode_enabled", False)),
        "cumem_allocator": type(allocator).__name__ if allocator is not None else None,
    }
    if torch.cuda.is_available():
        state["allocated_gb"] = round(torch.cuda.memory_allocated() / 1e9, 3)
        state["reserved_gb"] = round(torch.cuda.memory_reserved() / 1e9, 3)
    if allocator is not None and hasattr(allocator, "get_current_usage"):
        try:
            state["cumem_current_usage"] = allocator.get_current_usage()
        except Exception:
            pass
    return state


__all__ = [
    "_get_cumem_allocator",
    "sleep_mode_enabled",
    "weight_pool",
    "kv_cache_pool",
    "describe_sleep_state",
]
