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

from core.inference import npu_memory
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
    # Largest token window in budget, including zero; None when it could not be priced.
    fits: Optional[int]


def _available_mib() -> Optional[int]:
    return host_memory.usable_host_ram_mib()


def _npu_window_gb() -> Optional[float]:
    return npu_memory.npu_memory_window_gb(npu_memory.system_ram_gb())


def _npu_window_bytes(weights: int) -> Optional[int]:
    """What the NPU memory window leaves after the weights and load margin (half of RAM on
    Windows, as the catalog's fit warning prices it); None when RAM is unknown."""
    window = _npu_window_gb()
    needed = npu_memory.npu_memory_needed_gb(weights / _GIB)
    if window is None or needed is None:
        return None
    return int((window - needed) * _GIB)


def _positive(config: dict, key: str) -> Optional[int]:
    value = config.get(key)
    return value if isinstance(value, int) and value > 0 else None


@dataclass(frozen = True)
class KvLayout:
    full_row_bytes: int
    sliding_row_bytes: int
    sliding_window: int
    minimum_capacity: int = 512

    def bytes_for(self, tokens: int) -> int:
        return (
            tokens * self.full_row_bytes + min(tokens, self.sliding_window) * self.sliding_row_bytes
        )


def _kv_layout(config_path: Optional[Path]) -> Optional[KvLayout]:
    """Read the cache layout from the installed FastFlowLM checkpoint."""
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
    window = _positive(config, "sliding_window") or 0
    pattern = _positive(config, "sliding_window_pattern")
    indices = config.get("full_attn_idxs")
    sliding = 0
    if isinstance(types, list) and len(types) == layers:
        # Gemma 4's trailing shared layers reuse buffers from the preceding layers.
        shared = _positive(config, "num_kv_shared_layers") or 0
        cached_types = types[: layers - shared] if shared < layers else types
        sliding = cached_types.count("sliding_attention") if window else 0
        layers = sum(kind not in _NO_KV_LAYER_TYPES for kind in cached_types) - sliding
    elif window and pattern:
        # Gemma 3 uses full attention on every sixth layer (one-based).
        full = layers // pattern
        sliding, layers = layers - full, full
    elif isinstance(indices, list) and all(type(i) is int and 0 <= i < layers for i in indices):
        # LFM2 lists the attention layers explicitly; the others are convolutions.
        layers = len(set(indices))
    elif interval:
        layers //= interval
    if not (layers or sliding):
        return None
    global_heads = _positive(config, "num_global_key_value_heads") or kv_heads
    global_dim = _positive(config, "global_head_dim") or head_dim
    return KvLayout(
        full_row_bytes = layers * 2 * global_heads * global_dim * _KV_BYTES,
        sliding_row_bytes = sliding * 2 * kv_heads * head_dim * _KV_BYTES,
        sliding_window = window,
        minimum_capacity = 4096 if config.get("model_type") == "gemma4_text" else 512,
    )


def _cost(ctx: int, kv: KvLayout) -> int:
    # FLM 1.0.12 rounds the active allocation up; a saved prefix only holds the rows used.
    capacity = max(kv.minimum_capacity, 1 << (ctx - 1).bit_length())
    active = kv.bytes_for(capacity)
    prefix = min(_PREFIX_CACHE_MIB * _MIB, _PREFIX_SLOTS * kv.bytes_for(ctx))
    return active + prefix


def _largest_fitting(budget: int, kv: KvLayout, ceiling: int) -> int:
    """Price exact token counts so an arbitrary pin and the fit estimate agree."""
    lo, hi = 0, ceiling
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _cost(mid, kv) <= budget:
            lo = mid
        else:
            hi = mid - 1
    return lo


def plan_context(
    model: NpuModel, files: Optional[FlmModelFiles], requested: Optional[int]
) -> ContextPlan:
    """The context to load ``model`` with. A positive ``requested`` is honored up to the model's
    limit; otherwise Auto. Sample RAM after releasing the previous model."""
    limit = model.max_context_length
    requested = requested if requested and requested > 0 else None
    floor = min(DEFAULT_CONTEXT_LENGTH, limit or DEFAULT_CONTEXT_LENGTH)
    fits = None
    kv = _kv_layout(files.folder / "config.json" if files is not None else None)
    weights = max(files.total if files is not None else 0, int((model.size_gb or 0) * _GIB))
    available = _available_mib() if kv and weights else None
    if available is not None:
        budget = (available - HOST_RAM_HEADROOM_MIB) * _MIB - weights
        window = _npu_window_bytes(weights)
        if window is not None:
            budget = min(budget, window)
        ceiling = limit or max(AUTO_CONTEXT_CEILING, requested or 0)
        fits = _largest_fitting(budget, kv, ceiling)
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
    auto = min(fits, AUTO_CONTEXT_CEILING) // CONTEXT_STEP * CONTEXT_STEP
    return ContextPlan(max(floor, auto), fits)
