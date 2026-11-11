# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Auto context for an NPU load: the largest window whose KV cache fits the RAM the weights leave,
as GGUF's Auto does on host RAM. The NPU has no memory of its own."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from utils import host_memory

if TYPE_CHECKING:
    from core.inference.flm_files import FlmModelFiles
    from core.inference.npu_backend import NpuModel

logger = logging.getLogger(__name__)

_MIB = 1024 * 1024
_GIB = 1024 * _MIB
# Floor and fallback: the old fixed default, and GGUF's _FIT_MIN_CTX.
DEFAULT_CONTEXT_LENGTH = 8192
# GGUF's Auto ceiling on unified memory (_APPLE_AUTO_CTX).
AUTO_CONTEXT_CEILING = 65536
# GGUF's _HOST_RAM_HEADROOM_MIB.
HOST_RAM_HEADROOM_MIB = 2048
CONTEXT_STEP = 1024
# FastFlowLM keeps K and V in bf16, for full-attention layers only.
_KV_BYTES = 2
# flm serve's --prefix-slots and --prefix-cache-mb defaults (FastFlowLM 1.0.12); lemond sets neither.
# Each slot saves up to a context's worth of KV rows in host RAM.
_PREFIX_SLOTS = 8
_PREFIX_CACHE_MIB = 4096
# Recurrent layers in hybrid models (Qwen3.5/3.6 linear attention, LFM2 conv) keep no KV.
_NO_KV_LAYER_TYPES = frozenset({"linear_attention", "conv", "mamba", "recurrent"})


@dataclass(frozen = True)
class ContextPlan:
    context_length: int
    # Largest context free RAM holds (never below the floor), or None when it could not be priced.
    fits: Optional[int]


def _available_mib() -> Optional[int]:
    return host_memory.usable_host_ram_mib()


def _positive(config: dict, key: str) -> Optional[int]:
    value = config.get(key)
    return value if isinstance(value, int) and value > 0 else None


def kv_bytes_per_token(config_path: Optional[Path]) -> Optional[int]:
    """KV bytes one token costs, from FastFlowLM's config.json; None when it can't be read."""
    if config_path is None:
        return None
    try:
        config: Any = json.loads(config_path.read_text(encoding = "utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(config, dict):
        return None
    if isinstance(config.get("text_config"), dict):
        config = config["text_config"]
    layers = _positive(config, "num_hidden_layers")
    heads = _positive(config, "num_attention_heads")
    kv_heads = _positive(config, "num_key_value_heads") or heads
    head_dim = _positive(config, "head_dim")
    hidden = _positive(config, "hidden_size")
    if head_dim is None and hidden and heads:
        head_dim = hidden // heads
    if not (layers and kv_heads and head_dim):
        return None
    types = config.get("layer_types")
    interval = _positive(config, "full_attention_interval")
    if isinstance(types, list) and len(types) == layers:
        layers = sum(1 for kind in types if kind not in _NO_KV_LAYER_TYPES)
    elif interval:
        layers //= interval
    return layers * 2 * kv_heads * head_dim * _KV_BYTES or None


def _cost(ctx: int, kv: int) -> int:
    rows = ctx * kv
    return rows + min(_PREFIX_CACHE_MIB * _MIB, _PREFIX_SLOTS * rows)


def _largest_fitting(budget: int, kv: int, ceiling: int) -> int:
    """Largest multiple of CONTEXT_STEP up to ceiling whose cost fits budget; 0 if none."""
    lo, hi, best = 1, ceiling // CONTEXT_STEP, 0
    while lo <= hi:
        mid = (lo + hi) // 2
        if _cost(mid * CONTEXT_STEP, kv) <= budget:
            best, lo = mid * CONTEXT_STEP, mid + 1
        else:
            hi = mid - 1
    return best


def plan_context(
    model: NpuModel,
    files: Optional[FlmModelFiles],
    requested: Optional[int],
    previous: Optional[NpuModel] = None,
) -> ContextPlan:
    """The context to load ``model`` with. A positive ``requested`` is honored up to the model's
    limit; otherwise Auto. ``previous`` is the resident NPU model the load replaces."""
    limit = model.max_context_length
    requested = requested if requested and requested > 0 else None
    floor = min(DEFAULT_CONTEXT_LENGTH, limit or DEFAULT_CONTEXT_LENGTH)
    fits = None
    kv = kv_bytes_per_token(files.folder / "config.json" if files is not None else None)
    weights = max(files.total if files is not None else 0, int((model.size_gb or 0) * _GIB))
    available = _available_mib() if kv and weights else None
    if available is not None:
        reclaimed = int((previous.size_gb or 0) * _GIB) if previous is not None else 0
        budget = (available - HOST_RAM_HEADROOM_MIB) * _MIB + reclaimed - weights
        ceiling = limit or max(AUTO_CONTEXT_CEILING, requested or 0)
        fits = max(_largest_fitting(budget, kv, ceiling), floor)
    if requested is not None:
        ctx = min(requested, limit) if limit else requested
        if fits is not None and ctx > fits:
            logger.warning(
                "Context %d for %s is above the %d tokens free RAM is estimated to hold; "
                "FastFlowLM may fail to load it",
                ctx,
                model.id,
                fits,
            )
        return ContextPlan(ctx, fits)
    if fits is None:
        return ContextPlan(floor, None)
    return ContextPlan(min(fits, AUTO_CONTEXT_CEILING), fits)
