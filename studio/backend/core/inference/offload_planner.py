# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Placement planner: spill weights with ``-ot``, never the KV cache.

llama.cpp's ``--fit on`` spills whole layers through ``n_gpu_layers``, and a
layer's KV cache is allocated on ``model.dev_layer(il)`` (llama-kv-cache.cpp),
so spilling a layer drags its cache to host RAM with it. Measured at 128K on one
B200, that is the expensive direction by a wide margin:

    weights spilled, cache resident   71.63 t/s
    cache spilled, weights resident    3.24 t/s

``-ot`` overrides tensor buffer types WITHOUT touching layer assignment, so the
cache stays put: measured ``offloaded 66/66 layers to GPU`` with the whole cache
on CUDA0 even when every block tensor was forced to the host.

This module is pure arithmetic over a :class:`ModelLayout`. It performs no IO and
reads no globals, so the whole decision table is testable directly.
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Callable, Collection, Mapping, Optional, Sequence, Union

from core.inference.offload_cost_model import (
    Access,
    HostProfile,
    Placement,
    TensorGroup,
    generation_penalty_ms,
    rank,
)
from core.inference.offload_layout import (
    FFN_SPILL_CLASSES,
    LM_HEAD_PATTERN,
    BlockLayout,
    ModelLayout,
    SpillClass,
    spill_pattern_for,
    spill_pattern_for_class,
)

GIB = 1024**3
MIB = 1024**2

# Probe ceiling when nothing bounds the search.
_MAX_CTX_SEARCH = 1 << 24

_NOMINAL_BPW: dict[str, float] = {
    "Q2_K": 2.5625,
    "Q3_K": 3.4375,
    "Q4_K": 4.5,
    "Q5_K": 5.5,
    "Q6_K": 6.5625,
    "Q8_0": 8.5,
    "Q4_0": 4.5,
    "Q4_1": 5.0,
    "Q5_0": 5.5,
    "Q5_1": 6.0,
    "IQ2_XXS": 2.0625,
    "IQ2_XS": 2.3125,
    "IQ2_S": 2.5,
    "IQ3_XXS": 3.0625,
    "IQ3_S": 3.4375,
    "IQ4_NL": 4.5,
    "IQ4_XS": 4.25,
    "F16": 16.0,
    "BF16": 16.0,
    "F32": 32.0,
}

_LADDER_BPW_THRESHOLD = 1.40


def moe_down_up_bpw_ratio(layout) -> Optional[float]:
    """How much denser ``ffn_down`` is than ``ffn_up``, in bits per weight."""
    if not getattr(layout, "is_moe", False):
        return None
    downs, ups = [], []
    for b in getattr(layout, "blocks", ()):
        d = _NOMINAL_BPW.get((b.ffn_down_type or "").upper())
        u = _NOMINAL_BPW.get((b.ffn_up_type or "").upper())
        if d and u:
            downs.append(d)
            ups.append(u)
    if not downs:
        return None
    mean_down = sum(downs) / len(downs)
    mean_up = sum(ups) / len(ups)
    return (mean_down / mean_up) if mean_up else None


class ContextPolicy(Enum):
    """Whether the planner may shrink a context the user asked for.

    llama.cpp's fitter shrinks context before spilling anything, and on
    throughput grounds that is right: a resident smaller context beats a spilled
    larger one. But context is a user-visible feature, not a free variable, so
    quietly trading it away is not a safe default.
    """

    NEVER_REDUCE = "never"
    # Shrink if that avoids spilling entirely.
    PREFER_RESIDENT = "prefer_resident"
    # Shrink only when no rung of the ladder fits.
    FIT_ONLY = "fit_only"


class SpillOrder(Enum):
    """Which blocks to spill when only some are needed."""

    LARGEST_FIRST = "largest_first"
    FRONT_FIRST = "front_first"
    BACK_FIRST = "back_first"


class FfnGranularity(Enum):
    """How small a piece of one block's FFN the planner may move."""

    WHOLE = "whole"
    BOUNDARY = "boundary"
    ALL = "all"


@dataclass(frozen = True)
class PlanOptions:
    overhead_bytes_per_device: int = (3 * GIB) // 2
    overhead_bytes_per_token: int = 23961
    overhead_free_ctx: int = 32768
    extra_resident_bytes: int = 0
    pipeline_overhead_bytes: int = 0
    # Host RAM this planner refuses to spend, so a spill does not push the box into swap.
    host_ram_headroom_bytes: int = 2 * GIB
    # True when this launch reads per_layer_embd_bytes lazily from the mapping
    # (llama-model-loader.cpp lazy_read::add). False keeps the full host charge.
    ple_read_lazily: bool = False
    context_policy: ContextPolicy = ContextPolicy.NEVER_REDUCE
    min_ctx: int = 4096
    spill_order: SpillOrder = SpillOrder.BACK_FIRST
    # BOUNDARY is llama.cpp's behaviour (fit.cpp:490, LAYER_FRACTION_MOE past il0).
    ffn_granularity: FfnGranularity = FfnGranularity.BOUNDARY
    granularity_from_quant: bool = False
    # llama.cpp's order (common/fit.cpp:407-440); unmeasured by us.
    ffn_rung_order: tuple[SpillClass, ...] = FFN_SPILL_CLASSES
    allow_lm_head_spill: bool = True
    allow_attention_spill: bool = False
    allow_kv_host_fallback: bool = False
    # Allow a PARTIAL multi-device spill; relies on _device_slots matching llama-model.cpp:1425-1457.
    trust_device_row_model: bool = False
    host: HostProfile = field(default_factory = HostProfile)
    # q8_0 measured slower; only four matched K/V pairs compile without GGML_CUDA_FA_ALL_QUANTS.
    allow_kv_quant: bool = False
    kv_quant_type: str = "q8_0"
    cache_quantised: bool = False
    prompt_cache_unbounded: bool = False
    # -nkvo: the whole cache lives on the host (llama-kv-cache.cpp:210-219).
    kv_on_host: bool = False
    workload_prompt_tokens: int = 2048
    workload_generated_tokens: int = 256
    n_ubatch: int = 512
    min_penalty_reduction: float = 0.10
    require_cost_win: bool = False
    # False when --fit on cannot re-place this load (fit.cpp aborts on multi-device + tensor_split).
    rank_against_fitter: bool = True

    mmproj_bytes: int = 0
    mmproj_movable: bool = False
    n_parallel: int = 1
    min_parallel: int = 1
    kv_bytes_floor_by_parallel: Mapping[int, int] = field(default_factory = dict)
    kv_swa_bytes_floor: int = 0
    kv_bytes_at: Optional[Callable[[int, int], int]] = None
    kv_swa_bytes_at: Optional[Callable[[int, int], int]] = None
    kv_layer_weights_at: Optional[Callable[[int, int], Sequence[int]]] = None
    overhead_bytes_at: Optional[Callable[[int], int]] = None
    n_ubatch_by_parallel: Mapping[int, int] = field(default_factory = dict)
    draft_bytes: int = 0
    draft_droppable: bool = False
    draft_drop_penalty_frac: float = 0.05
    cache_ram_default_mib: int = 8192
    # The one operating point where -ot measured WORSE than llama.cpp's own layerwise fit.
    moe_long_prompt_ctx: int = 32768
    kv_unified: bool = False
    ctx_step: int = 1024
    # The -c a declined plan launches with and its per-device VRAM; 0 = the plan's own context.
    fitter_n_ctx: int = 0
    fitter_vram_bytes_per_device: tuple[int, ...] = ()
    fitter_outside_layout_bytes: Optional[int] = None
    fitter_drops_draft: bool = False
    lazy_read_bytes: int = 0


@dataclass(frozen = True)
class Plan:
    """False means emit nothing new: the planner abstained or the load needs no help; llama.cpp defaults then apply, always safe"""

    changed: bool = False
    n_ctx: int = 0
    ot_patterns: tuple[str, ...] = field(default_factory = tuple)
    load_mode_none: bool = False
    cache_type_k: Optional[str] = None
    cache_type_v: Optional[str] = None
    spilled_blocks: tuple[int, ...] = field(default_factory = tuple)
    spilled_lm_head: bool = False
    kv_spilled_to_host: bool = False
    insufficient: bool = False
    vram_bytes: int = 0
    host_bytes: int = 0
    predicted_gen_penalty_ms: float = 0.0
    # Cost gate sides, ms per request: this placement vs llama.cpp's fitter.
    predicted_request_ms: float = 0.0
    predicted_fit_request_ms: float = 0.0
    reason: str = ""
    n_parallel: int = 0
    mmproj_to_host: bool = False
    draft_dropped: bool = False
    cache_ram_mib: int = -1
    declined_by_gate: bool = False
    ple_charged_to_host: bool = True
    # A measured decline, not context-dependent: FIT_ONLY must not retry smaller.
    veto: bool = False
    priced: bool = False
    # Per-device layer rows emitted as -ts, since --fit off would re-derive the split from the
    # child's own free VRAM; () for one device or an unmodellable split.
    device_layer_counts: tuple[int, ...] = field(default_factory = tuple)

    @property
    def spills_anything(self) -> bool:
        return bool(self.spilled_blocks) or self.spilled_lm_head

    @property
    def reshapes_launch(self) -> bool:
        """A rung above the first weight spill fired, so the launch changes with no pattern."""
        return self.n_parallel > 0 or self.mmproj_to_host or self.draft_dropped


def _device_reserve(opts: PlanOptions, n_ctx: int) -> int:
    """Bytes to leave free on EVERY device: the flat term plus the context-linear one."""
    over = max(0, n_ctx - max(0, opts.overhead_free_ctx))
    flat = opts.overhead_bytes_per_device
    if opts.overhead_bytes_at is not None:
        flat = int(opts.overhead_bytes_at(max(0, n_ctx)))
    return max(0, flat) + over * max(0, opts.overhead_bytes_per_token)


def _usable_vram(
    vram_bytes_per_device: Sequence[int],
    opts: PlanOptions,
    n_ctx: int,
    *,
    outside_layout_bytes: Optional[int] = None,
) -> int:
    """Total creditable VRAM: every device pays the fixed per-device overhead, the split pays for
    each device AFTER the first, then the pool pays once for whatever sits on a card outside the
    layout.
    """
    reserve = _device_reserve(opts, n_ctx)
    pooled = sum(max(0, v - reserve) for v in vram_bytes_per_device)
    split = max(0, len(vram_bytes_per_device) - 1) * max(0, opts.pipeline_overhead_bytes)
    outside = (
        _outside_layout_bytes(opts)
        if outside_layout_bytes is None
        else max(0, outside_layout_bytes)
    )
    return pooled - split - outside


@dataclass(frozen = True)
class _Knobs:
    """What rungs 0 to 2 have given up so far. Immutable; a rung returns a new one."""

    n_parallel: int
    mmproj_to_host: bool = False
    draft_dropped: bool = False


def _outside_layout_bytes(opts: PlanOptions, knobs: Optional[_Knobs] = None) -> int:
    """Device bytes the layout cannot see: the caller's scalar, the projector unless rung 0 moved
    it, and the draft unless rung 2 dropped it."""
    total = max(0, opts.extra_resident_bytes)
    if not (knobs and knobs.mmproj_to_host):
        total += max(0, opts.mmproj_bytes)
    if not (knobs and knobs.draft_dropped):
        total += max(0, opts.draft_bytes)
    return total


def _measured_cache_at(layout: ModelLayout, opts: PlanOptions, n_ctx: int, n_parallel: int) -> int:
    """``opts.kv_bytes_at`` less the recurrent state charged separately."""
    assert opts.kv_bytes_at is not None
    slots = max(1, n_parallel)
    total = max(0, int(opts.kv_bytes_at(n_ctx, slots)))
    return max(0, total - max(0, layout.recurrent_bytes) * slots)


def _kv_floor_at(
    layout: ModelLayout,
    opts: PlanOptions,
    kv_bytes_floor: int,
    requested_ctx: int,
    n_ctx: int,
    n_parallel: int,
) -> Optional[int]:
    """The caller's cache floor re-priced for ``(n_ctx, n_parallel)``; ``None`` when it cannot be."""
    at = max(1, opts.n_parallel)
    asked = max(1, n_parallel)
    # A unified cache's windowed half shrinks with the slot count; only the linear fallback cannot
    # tell the halves apart.
    exact = opts.kv_bytes_at is not None or asked in opts.kv_bytes_floor_by_parallel
    want = asked if (exact or not opts.kv_unified) else at
    if opts.kv_bytes_at is not None:
        return _measured_cache_at(layout, opts, n_ctx, want)
    base = max(0, kv_bytes_floor)
    if want != at:
        mapped = opts.kv_bytes_floor_by_parallel.get(want)
        if mapped is not None:
            base = max(0, int(mapped))
        elif layout.has_swa:
            return None
        else:
            base = base * want // at
    if requested_ctx > 0 and n_ctx != requested_ctx and base > 0 and not layout.has_swa:
        # A hybrid's fixed state does not shrink with context.
        state = min(base, max(0, layout.recurrent_bytes) * want)
        base = state + (base - state) * n_ctx // requested_ctx
    return base


@dataclass(frozen = True)
class SpillUnit:
    """One block's worth of one rung: the smallest thing the ladder can move."""

    index: int
    cls: Optional[SpillClass]
    nbytes: int


def _units_of(blocks: Sequence[BlockLayout], cls: Optional[SpillClass]) -> list[SpillUnit]:
    if cls is None:
        return [
            SpillUnit(b.index, None, b.spillable_bytes) for b in blocks if b.spillable_bytes > 0
        ]
    return [SpillUnit(b.index, cls, b.class_bytes(cls)) for b in blocks if b.class_bytes(cls) > 0]


def _select_units(
    units: Sequence[SpillUnit],
    deficit: int,
    order: SpillOrder,
    *,
    cost_ranked: bool = False,
) -> tuple[list[SpillUnit], int]:
    """The MINIMAL set of ``units`` freeing at least ``deficit``, and what it frees."""

    def walk(how: SpillOrder) -> tuple[list[SpillUnit], int]:
        remaining = list(units)
        if how is SpillOrder.FRONT_FIRST:
            remaining.sort(key = lambda u: u.index)
        elif how is SpillOrder.BACK_FIRST:
            remaining.sort(key = lambda u: -u.index)
        else:
            remaining.sort(key = lambda u: -u.nbytes)

        chosen: list[SpillUnit] = []
        freed = 0
        while freed < deficit and remaining:
            if how is SpillOrder.LARGEST_FIRST:
                residual = deficit - freed
                # Prefer the SMALLEST unit that closes the gap, so the last pick cannot overshoot.
                covering = [u for u in remaining if u.nbytes >= residual]
                pick = min(covering, key = lambda u: u.nbytes) if covering else remaining[0]
            else:
                pick = remaining[0]
            remaining.remove(pick)
            chosen.append(pick)
            freed += pick.nbytes
        return chosen, freed

    chosen, freed = walk(order)
    if order is not SpillOrder.BACK_FIRST or deficit <= 0:
        return chosen, freed
    minimal, least = walk(SpillOrder.LARGEST_FIRST)
    if least < freed and (cost_ranked or freed > 2 * least):
        return minimal, least
    return chosen, freed


def _ordered_rungs(opts: PlanOptions) -> list[SpillClass]:
    """``ffn_rung_order`` as the ladder walks it: known classes, first occurrence only."""
    seen: list[SpillClass] = []
    for cls in opts.ffn_rung_order:
        if cls in FFN_SPILL_CLASSES and cls not in seen:
            seen.append(cls)
    return seen


def _grade_the_boundary_block(
    layout: ModelLayout, opts: PlanOptions, taken: list[SpillUnit], freed: int, deficit: int
) -> tuple[list[SpillUnit], int]:
    """Trim the LAST whole block taken down to the rungs actually needed."""
    if _effective_granularity(layout, opts) is not FfnGranularity.BOUNDARY:
        return taken, freed
    coarse = [u for u in taken if u.cls is None]
    if not coarse:
        return taken, freed
    by_index = {b.index: b for b in layout.blocks}
    boundary = coarse[-1]
    if taken[-1] is not boundary:
        return taken, freed
    block = by_index.get(boundary.index)
    if block is None or not block.graded:
        return taken, freed

    without = freed - boundary.nbytes
    rungs = _ordered_rungs(opts)
    kept: list[SpillUnit] = []
    running = without
    for cls in rungs:
        if running >= deficit:
            break
        nbytes = block.class_bytes(cls)
        if nbytes <= 0:
            continue
        kept.append(SpillUnit(block.index, cls, nbytes))
        running += nbytes
    if running < deficit or len(kept) >= len(rungs):
        # Graded rungs cannot do better: keep the coarse unit.
        return taken, freed
    return [u for u in taken if u is not boundary] + kept, running


def _spills_every_block(layout: ModelLayout, units: Sequence[SpillUnit]) -> bool:
    """Whether ``units`` takes every block's FFN and, for a shared/dense FFN or attention rung it
    touches, that rung off every block holding it: the only spill whose per-device share does not
    depend on where llama.cpp cuts the layer ranges."""
    moved = _moved_by_index(units)
    if not all(moved.get(b.index, 0) >= b.spillable_bytes for b in layout.blocks):
        return False
    touched: dict[SpillClass, set[int]] = {}
    for unit in units:
        if unit.cls in (SpillClass.DENSE_FFN, SpillClass.ATTENTION):
            touched.setdefault(unit.cls, set()).add(unit.index)
    return all(
        indices >= {u.index for u in _units_of(layout.blocks, cls)}
        for cls, indices in touched.items()
    )


def _moved_by_index(units: Sequence[SpillUnit]) -> dict[int, int]:
    """Bytes this spill takes off each block."""
    moved: dict[int, int] = {}
    for unit in units:
        moved[unit.index] = moved.get(unit.index, 0) + unit.nbytes
    return moved


_RUNG_NAMES: dict[Optional[SpillClass], str] = {
    None: "the FFN",
    SpillClass.FFN_DOWN: "the ffn_down",
    SpillClass.FFN_UP: "the ffn_up/gate_up",
    SpillClass.FFN_GATE: "the ffn_gate",
    SpillClass.DENSE_FFN: "the shared/dense FFN",
    SpillClass.ATTENTION: "the attention projections",
}


def _rung_description(units: Sequence[SpillUnit], layout: ModelLayout) -> str:
    """ "the ffn_down of every block plus the ffn_up/gate_up of 5 of 40 blocks"."""
    default_total = len([b for b in layout.blocks if b.spillable_bytes > 0]) or len(layout.blocks)
    seen: list[Optional[SpillClass]] = []
    counts: dict[Optional[SpillClass], set[int]] = {}
    for unit in units:
        if unit.cls not in counts:
            counts[unit.cls] = set()
            seen.append(unit.cls)
        counts[unit.cls].add(unit.index)
    parts = []
    for cls in seen:
        # Per class: a MoE's leading dense blocks carry shared/dense FFN but no experts.
        total = (len(_units_of(layout.blocks, cls)) if cls is not None else 0) or default_total
        n = len(counts[cls])
        where = "every block" if n >= total else f"{n} of {total} blocks"
        parts.append(f"{_RUNG_NAMES[cls]} of {where}")
    return " plus ".join(parts) if parts else "nothing"


def _effective_granularity(layout: ModelLayout, opts: PlanOptions) -> FfnGranularity:
    """The granularity actually used, after the optional quant-type rule."""
    if not opts.granularity_from_quant:
        return opts.ffn_granularity
    ratio = moe_down_up_bpw_ratio(layout)
    if ratio is None:
        return opts.ffn_granularity
    return FfnGranularity.ALL if ratio >= _LADDER_BPW_THRESHOLD else FfnGranularity.BOUNDARY


def _rung_classes(layout: ModelLayout, opts: PlanOptions) -> tuple[Optional[SpillClass], ...]:
    """The per-block rungs, cheapest first, for this layout and these options."""
    spillable = [b for b in layout.blocks if b.spillable_bytes > 0]
    graded = bool(spillable) and all(b.graded for b in spillable)
    if _effective_granularity(layout, opts) is FfnGranularity.ALL and graded:
        rungs: list[Optional[SpillClass]] = list(_ordered_rungs(opts))
    else:
        rungs = [None]
    if any(b.dense_ffn_bytes for b in layout.blocks):
        rungs.append(SpillClass.DENSE_FFN)
    return tuple(rungs)


def _ffn_group(layout: ModelLayout, spilled: int) -> TensorGroup:
    """The spillable FFN bytes as one group, charged the way they are read."""
    if layout.is_moe and layout.n_expert and layout.n_expert_used:
        return TensorGroup(
            "experts",
            spilled,
            Access.SCATTERED,
            activation_fraction = layout.n_expert_used / layout.n_expert,
        )
    return TensorGroup("ffn", spilled, Access.CONTIGUOUS)


def _spill_placement(
    layout: ModelLayout,
    units: Sequence[SpillUnit],
    spill_lm_head: bool,
    kv_host_bytes: int = 0,
) -> Placement:
    """This plan's spill, as something the cost model can price."""
    routed = sum(u.nbytes for u in units if u.cls is None or u.cls in FFN_SPILL_CLASSES)
    dense_ffn = sum(u.nbytes for u in units if u.cls is SpillClass.DENSE_FFN)
    attention = sum(u.nbytes for u in units if u.cls is SpillClass.ATTENTION)

    groups: list[TensorGroup] = []
    if routed:
        groups.append(_ffn_group(layout, routed))
    if dense_ffn:
        groups.append(TensorGroup("dense_ffn", dense_ffn, Access.CONTIGUOUS))
    if attention:
        groups.append(TensorGroup("attention", attention, Access.CONTIGUOUS))
    if spill_lm_head and layout.lm_head_bytes:
        groups.append(TensorGroup("lm_head", layout.lm_head_bytes, Access.SINGLE_MATVEC))
    return Placement(host_groups = groups, kv_host_bytes = max(0, kv_host_bytes))


def _spill_penalty_ms(
    layout: ModelLayout,
    units: Sequence[SpillUnit],
    spill_lm_head: bool,
    host: HostProfile,
    kv_host_bytes: int = 0,
) -> float:
    """Predicted extra ms per generated token for this spill, on this host."""
    placement = _spill_placement(layout, units, spill_lm_head, kv_host_bytes)
    if not placement.host_groups and not placement.kv_host_bytes:
        return 0.0
    return generation_penalty_ms(placement, host)


def _fit_boundary_overflow(block: BlockLayout, deficit: int) -> Optional[int]:
    """FFN bytes ``common/fit.cpp`` overflows off its boundary layer to cover ``deficit``, or None
    when even the whole FFN of it does not.
    """
    if deficit <= 0:
        return None
    down = block.class_bytes(SpillClass.FFN_DOWN)
    for nbytes in (down, down + block.class_bytes(SpillClass.FFN_GATE), block.spillable_bytes):
        if nbytes > 0 and nbytes >= deficit:
            return nbytes
    return None


def _swa_floor_at(opts: PlanOptions, n_ctx: int, n_seq: int) -> int:
    """The window-bound half of the cache at ``(n_ctx, n_seq)``; 0 when unknown.

    The scalar was measured at the caller's slot count and a window is stored per stream
    (``swa * slots + ubatch`` cells under a unified cache, ``swa + ubatch`` per stream otherwise),
    so after rung 1 it is stale in the direction that makes the fitter look dearer:
    ``min(stale, kv_total)`` reads most of a smaller cache as fully live.
    """
    slots = max(1, n_seq)
    if opts.kv_swa_bytes_at is not None:
        return max(0, int(opts.kv_swa_bytes_at(n_ctx, slots)))
    base = max(0, opts.kv_swa_bytes_floor)
    at = max(1, opts.n_parallel)
    if base <= 0 or slots == at:
        return base
    return base * slots // at


def _recurrent_rows(layout: ModelLayout, weights: Sequence[int]) -> set[int]:
    """The rows ``weights`` gives no cache, less any the file says hold no recurrent state either
    (nemotron_h's MLP-only rows, which would otherwise take a share of the Mamba state)."""
    rows = {i for i, w in enumerate(weights) if w == 0}
    return rows & set(layout.recurrent_rows) if layout.recurrent_rows else rows


def _fit_fallback_placement(
    layout: ModelLayout,
    opts: PlanOptions,
    budget: int,
    n_ctx: int,
    *,
    quantised: bool,
    kv_bytes_floor: int,
    kv_on_host: bool,
    n_seq: int = 1,
    kv_layer_weights: Sequence[int] = (),
) -> Optional[Placement]:
    """What llama.cpp's own fitter would place here, priced the same way.

    The output tensor rides the layer list at ``n_layer_all`` and is the LAST row to leave, not the
    first: llama.cpp keeps rows ``[i_gpu_start, i_gpu_start + act_gpu_layers)`` with
    ``i_gpu_start = max(n_layer_all + 1 - n_gpu_layers, 0)`` (llama-model.cpp:1467-1492), so
    lm_head is resident in every placement this loop can return.

    ``opts.fitter_n_ctx``, when set and not ``n_ctx``, is the context the declined launch really
    runs, so the arm is re-priced there with the original projector, draft and reserve on the card.
    ``opts.fitter_outside_layout_bytes`` replaces those outside-layout bytes with the ones that
    launch really holds (its draft at its own context, or none).
    """
    reprice_context = 0 < opts.fitter_n_ctx != n_ctx
    if reprice_context or opts.fitter_outside_layout_bytes is not None:
        if not opts.fitter_vram_bytes_per_device:
            return None
        if reprice_context:
            if opts.kv_bytes_at is None:
                return None
            n_ctx = opts.fitter_n_ctx
            kv_bytes_floor = _measured_cache_at(layout, opts, n_ctx, max(1, n_seq))
            if opts.kv_layer_weights_at is not None:
                kv_layer_weights = opts.kv_layer_weights_at(n_ctx, max(1, n_seq))
            elif layout.has_swa and kv_layer_weights:
                return None
        budget = _usable_vram(
            opts.fitter_vram_bytes_per_device,
            opts,
            n_ctx,
            outside_layout_bytes = opts.fitter_outside_layout_bytes,
        )
    blocks = list(layout.blocks)
    if not blocks:
        return None
    # Both arms size ONE cache (the product charges q4_0 1.78x).
    trust = opts.kv_bytes_at is not None
    resident = all_resident_bytes(
        layout,
        n_ctx,
        kv_quantised = quantised,
        kv_bytes_floor = kv_bytes_floor,
        kv_on_host = kv_on_host,
        n_seq = max(1, n_seq),
        trust_floor = trust,
    )
    if resident <= budget:
        return Placement(host_groups = [])
    kv_total = (
        0
        if kv_on_host
        else cache_bytes(
            layout,
            n_ctx,
            kv_quantised = quantised,
            kv_bytes_floor = kv_bytes_floor,
            trust_floor = trust,
        )
    )
    weights = [max(0, int(w)) for w in kv_layer_weights]
    if len(weights) != layout.n_layers or not any(weights):
        weights = []
    if not weights and not (layout.has_swa or layout.has_mla):
        # No caller vector: use the file's cache rows, not an even spread over recurrent rows.
        weights = list(layout.kv_heads_by_layer)
        if len(weights) != layout.n_layers or not any(weights):
            weights = []
    state_total = 0 if kv_on_host else layout.recurrent_bytes * max(1, n_seq)
    recurrent_rows = {b.index for b in blocks} & _recurrent_rows(layout, weights)
    if state_total > 0 and recurrent_rows:
        recurrent_prefix: list[int] = []
        seen_rows = 0
        for block in blocks:
            seen_rows += 1 if block.index in recurrent_rows else 0
            recurrent_prefix.append(state_total * seen_rows // len(recurrent_rows))
    else:
        recurrent_prefix = [
            int(state_total * moved / len(blocks)) for moved in range(1, len(blocks) + 1)
        ]

    def recurrent_moved(moved: int) -> int:
        """State bytes the first ``moved`` blocks carry to the host."""
        if moved <= 0:
            return 0
        return recurrent_prefix[min(moved, len(recurrent_prefix)) - 1]

    shares: list[int] = []
    running = 0
    for block in blocks:
        running += weights[block.index] if weights else 1
        shares.append(running)
    total_share = shares[-1]

    def kv_freed(total: int, moved: int) -> int:
        """``total`` bytes of cache carried off by the first ``moved`` blocks."""
        if moved <= 0 or total_share <= 0:
            return 0
        return int(total * shares[min(moved, len(shares)) - 1] / total_share)

    slot_window = n_ctx if opts.kv_unified else n_ctx // max(1, n_seq)
    swa_floor = _swa_floor_at(opts, n_ctx, n_seq)
    live_tokens = min(
        max(1, slot_window), max(1, opts.workload_prompt_tokens + opts.workload_generated_tokens)
    )
    reserved_product = cache_bytes(layout, n_ctx, kv_quantised = quantised)
    floor_scale = (kv_total / reserved_product) if reserved_product > 0 else 1.0
    if kv_on_host:
        kv_live_total = 0
    elif layout.has_swa and opts.kv_bytes_at is not None:
        # Populated cells only: --swa-full's reservation (size_swa = size_base) is not decode traffic.
        kv_live_total = min(kv_total, _measured_cache_at(layout, opts, live_tokens, 1))
    elif layout.has_swa and kv_bytes_floor > 0 and swa_floor > 0:
        swa_part = min(swa_floor, kv_total)
        # Without --kv-unified each slot owns a stream; a request reads only its own.
        swa_live = swa_part if opts.kv_unified else swa_part // max(1, n_seq)
        kv_live_total = swa_live + int((kv_total - swa_part) * live_tokens / max(1, n_ctx))
    elif layout.has_swa and kv_bytes_floor > 0:
        kv_live_total = kv_total
    else:
        kv_live_total = int(cache_bytes(layout, live_tokens, kv_quantised = quantised) * floor_scale)

    live_weights: list[int] = []
    if (
        not kv_on_host
        and layout.has_swa
        and weights
        and opts.kv_bytes_at is not None
        and opts.kv_layer_weights_at is not None
    ):
        live_weights = [max(0, int(w)) for w in opts.kv_layer_weights_at(live_tokens, 1)]
        if len(live_weights) != layout.n_layers or not any(live_weights):
            live_weights = []
    live_shares: list[int] = []
    running = 0
    for block in blocks:
        running += live_weights[block.index] if live_weights else 0
        live_shares.append(running)

    def kv_live_freed(moved: int) -> int:
        """Live cache bytes, the decode traffic, the first ``moved`` blocks carry to the host."""
        if not live_weights:
            return kv_freed(kv_live_total, moved)
        if moved <= 0 or live_shares[-1] <= 0:
            return 0
        return int(kv_live_total * live_shares[min(moved, len(live_shares)) - 1] / live_shares[-1])

    if layout.is_moe:
        host_experts = 0
        for block in reversed(blocks):
            overflow = _fit_boundary_overflow(block, resident - host_experts - budget)
            if overflow is not None:
                return Placement(host_groups = [_ffn_group(layout, host_experts + overflow)])
            host_experts += block.spillable_bytes
        host_layers = 0
        for moved, block in enumerate(blocks, start = 1):
            host_layers += block.resident_bytes
            host_recurrent = recurrent_moved(moved)
            freed = host_experts + host_layers + kv_freed(kv_total, moved) + host_recurrent
            if resident - freed <= budget:
                groups = [_ffn_group(layout, host_experts)]
                if host_layers:
                    groups.append(TensorGroup("layers", host_layers, Access.CONTIGUOUS))
                if host_recurrent:
                    # Reserved per slot, read per request (llama-memory-recurrent.cpp).
                    groups.append(
                        TensorGroup(
                            "recurrent (moved layers)",
                            host_recurrent // max(1, n_seq),
                            Access.CONTIGUOUS,
                        )
                    )
                return Placement(host_groups = groups, kv_host_bytes = kv_live_freed(moved))
        return None

    # Dense: the fitter only lowers n_gpu_layers (fit.cpp:551-559), so the host takes the LEADING
    # blocks (llama-model.cpp:1479-1484); a graded dense boundary is planner-only (fit.cpp:730-733).
    def dense_placement(moved: int, spilled_ffn: int, attention: int) -> Placement:
        """``moved`` whole layers off the device, plus ``spilled_ffn`` FFN bytes."""
        groups: list[TensorGroup] = []
        if spilled_ffn:
            groups.append(_ffn_group(layout, spilled_ffn))
        if attention:
            groups.append(TensorGroup("layers", attention, Access.CONTIGUOUS))
        host_recurrent = recurrent_moved(moved)
        if host_recurrent:
            groups.append(
                TensorGroup(
                    "recurrent (moved layers)",
                    host_recurrent // max(1, n_seq),
                    Access.CONTIGUOUS,
                )
            )
        live_kv = kv_live_freed(moved)
        # Charged at Access.KV_CACHE's 20.1x: batch-1 attention over a host cache is a strided GEMV.
        return Placement(host_groups = groups, kv_host_bytes = live_kv)

    host_weights = 0
    host_spillable = 0
    for moved, block in enumerate(blocks, start = 1):
        host_weights += block.spillable_bytes + block.resident_bytes
        host_spillable += block.spillable_bytes
        # lm_head stays on the device for any n_gpu_layers >= 1.
        freed = host_weights + kv_freed(kv_total, moved) + recurrent_moved(moved)
        if resident - freed <= budget:
            return dense_placement(moved, host_spillable, host_weights - host_spillable)
    return None


def _kv_elem_bytes(quantised: bool) -> int:
    return 1 if quantised else 2


def cache_bytes(
    layout: ModelLayout,
    n_ctx: int,
    *,
    kv_quantised: bool = False,
    kv_bytes_floor: int = 0,
    trust_floor: bool = False,
) -> int:
    """Attention cache to reserve, never below a caller-supplied measurement."""
    naive = layout.kv_bytes(n_ctx, _kv_elem_bytes(kv_quantised))
    floor = max(0, kv_bytes_floor)
    if trust_floor:
        return floor
    if layout.has_swa and floor:
        return floor
    if layout.has_mla and floor:
        return floor
    return max(naive, floor)


def resident_floor_bytes(
    layout: ModelLayout,
    n_ctx: int,
    *,
    kv_quantised: bool = False,
    kv_bytes_floor: int = 0,
    kv_on_host: bool = False,
    n_seq: int = 1,
    trust_floor: bool = False,
) -> int:
    """VRAM needed with EVERY spillable tensor already on the host."""
    if kv_on_host:
        # Both caches follow the same scalar, so neither is VRAM here.
        return layout.block_resident_bytes + layout.lm_head_bytes + layout.other_resident_bytes
    return (
        layout.block_resident_bytes
        + layout.lm_head_bytes
        + layout.other_resident_bytes
        + layout.recurrent_bytes * max(1, n_seq)
        + cache_bytes(
            layout,
            n_ctx,
            kv_quantised = kv_quantised,
            kv_bytes_floor = kv_bytes_floor,
            trust_floor = trust_floor,
        )
    )


def all_resident_bytes(
    layout: ModelLayout,
    n_ctx: int,
    *,
    kv_quantised: bool = False,
    kv_bytes_floor: int = 0,
    kv_on_host: bool = False,
    n_seq: int = 1,
    trust_floor: bool = False,
) -> int:
    """VRAM needed with nothing spilled. token_embd is excluded: it is never
    GPU-resident (llama-model.cpp pins dev_input to the CPU unconditionally)."""
    return (
        resident_floor_bytes(
            layout,
            n_ctx,
            kv_quantised = kv_quantised,
            kv_bytes_floor = kv_bytes_floor,
            kv_on_host = kv_on_host,
            n_seq = n_seq,
            trust_floor = trust_floor,
        )
        + layout.spillable_bytes
    )


def max_context_for(
    layout: ModelLayout,
    vram_bytes_per_device: Sequence[int],
    *,
    spill_all_ffn: bool = False,
    spill_lm_head: bool = False,
    kv_quantised: bool = False,
    opts: Optional[PlanOptions] = None,
    kv_bytes_floor: int = 0,
    floor_ctx: int = 0,
    n_seq: int = 1,
    kv_on_host: bool = False,
    outside_layout_bytes: Optional[int] = None,
    ctx_cap: int = 0,
) -> int:
    """Largest context whose cache fits, rounded down to 256 as CUDA wants.

    Bounded by the training window unless ``ctx_cap`` names an explicitly requested context:
    llama-server serves a ``-c`` above the window, so the ladder must be able to stop there.
    """
    opts = opts or PlanOptions()
    ctx_top = int(ctx_cap) if ctx_cap and ctx_cap > 0 else int(layout.n_ctx_train or 0)
    if not layout.complete or layout.kv_bytes_per_token_f16 <= 0:
        return 0
    fixed = (
        layout.block_resident_bytes
        + layout.other_resident_bytes
        + (0 if kv_on_host else layout.recurrent_bytes * max(1, n_seq))
        + (0 if spill_lm_head else layout.lm_head_bytes)
        + (0 if spill_all_ffn else layout.spillable_bytes)
        - (sum(b.dense_ffn_bytes for b in layout.blocks) if spill_all_ffn else 0)
    )
    per_token = layout.kv_bytes_per_token_f16 * _kv_elem_bytes(kv_quantised) // 2
    if per_token <= 0:
        return 0
    floor = max(0, kv_bytes_floor)
    floor_at = max(0, floor_ctx)

    def cache_at(ctx: int) -> int:
        if kv_on_host:
            return 0
        if opts.kv_bytes_at is not None:
            return _measured_cache_at(layout, opts, ctx, n_seq)
        naive = per_token * ctx
        if floor <= 0:
            return naive
        if layout.has_swa:
            # Flat: one scalar cannot say which layers hold the full-context half.
            return floor
        scaled = floor * ctx // floor_at if floor_at > 0 else floor
        if layout.has_mla:
            return scaled
        return max(naive, scaled)

    def usable(ctx: int) -> int:
        return _usable_vram(
            vram_bytes_per_device, opts, ctx, outside_layout_bytes = outside_layout_bytes
        )

    def fits(ctx: int) -> bool:
        return fixed + cache_at(ctx) <= usable(ctx)

    top = usable(0) - fixed
    if top <= 0:
        return 0
    priced = opts.kv_bytes_at is not None or (layout.has_mla and floor > 0)
    if kv_on_host or (layout.has_swa and floor > 0) or priced:
        slack = top - cache_at(1)
        if slack < 0:
            return 0
        per_token_reserve = max(0, opts.overhead_bytes_per_token)
        if per_token_reserve > 0:
            hi = max(0, opts.overhead_free_ctx) + slack // per_token_reserve
        elif ctx_top:
            hi = ctx_top
        elif priced:
            hi = 256
            while hi < _MAX_CTX_SEARCH and cache_at(hi) <= top:
                hi *= 2
        else:
            hi = top // per_token
        hi = hi // 256 * 256
    else:
        hi = (top // per_token) // 256 * 256
    if ctx_top:
        hi = min(hi, ctx_top // 256 * 256)
    if hi <= 0 or not fits(256):
        return 0
    lo = 256
    while lo < hi:
        mid = ((lo + hi + 256) // 512) * 256  # upper median, so lo always advances
        if fits(mid):
            lo = mid
        else:
            hi = mid - 256
    return lo


def plan_placement(
    layout: ModelLayout,
    vram_bytes_per_device: Sequence[int],
    host_ram_bytes: Optional[int],
    requested_ctx: int,
    *,
    opts: Optional[PlanOptions] = None,
    kv_bytes_floor: int = 0,
    split_weights_per_device: Sequence[int] = (),
    kv_layer_weights: Sequence[int] = (),
) -> Plan:
    """Decide the placement for one launch, and record the split it was decided on.

    Everything below is :func:`_plan_placement`; this only attaches the per-device layer counts the
    plan was budgeted against, so the launch can pin them with ``-ts`` instead of letting the child
    guess the split again.
    """
    plan = _plan_placement(
        layout,
        vram_bytes_per_device,
        host_ram_bytes,
        requested_ctx,
        opts = opts,
        kv_bytes_floor = kv_bytes_floor,
        split_weights_per_device = split_weights_per_device,
        kv_layer_weights = kv_layer_weights,
    )
    # Only a plan that will be emitted; an abstention keeps llama.cpp's placement.
    if not plan.priced or plan.insufficient or plan.declined_by_gate:
        return plan
    counts = _modelled_device_layers(layout, vram_bytes_per_device, split_weights_per_device)
    return plan if not counts else replace(plan, device_layer_counts = counts)


def _modelled_device_layers(
    layout: ModelLayout,
    vram_bytes_per_device: Sequence[int],
    split_weights_per_device: Sequence[int] = (),
) -> tuple[int, ...]:
    """Rows per device under llama.cpp's own split, or () for one device.

    The rows are the ``n_layers + 1`` :func:`_device_slots` hands out, the last being the output
    row, which goes to the last device (llama-model.cpp:1517). Handed back as ``-ts``, integer
    counts per device the way ``common/fit.cpp:555`` writes them (``tensor_split[id] =
    ngl_per_device[id].n_layer``), and they reproduce this assignment exactly: llama.cpp
    prefix-sums the shares and normalises, so a device whose count is ``c`` owns ``c`` rows.
    """
    if len(vram_bytes_per_device) <= 1:
        return ()
    n_slots = layout.n_layers + 1
    if n_slots <= 1:
        return ()
    try:
        slots = _device_slots(n_slots, split_weights_per_device or vram_bytes_per_device)
    except ValueError:
        # llama.cpp errors on an all-zero split.
        return ()
    return tuple(len(rows) for rows in slots)


def _plan_placement(
    layout: ModelLayout,
    vram_bytes_per_device: Sequence[int],
    host_ram_bytes: Optional[int],
    requested_ctx: int,
    *,
    opts: Optional[PlanOptions] = None,
    kv_bytes_floor: int = 0,
    split_weights_per_device: Sequence[int] = (),
    kv_layer_weights: Sequence[int] = (),
) -> Plan:
    """Rungs ordered in ms per token, not percent (percent losses use different baselines); split_weights_per_device is raw free VRAM, not the budget"""
    opts = opts or PlanOptions()

    if not layout.complete or not vram_bytes_per_device:
        return Plan(reason = "layout or device inventory incomplete, leaving llama.cpp defaults")
    if opts.host.unified_memory:
        return Plan(reason = "unified memory host, spilling frees no device memory")
    # The reserve has a context-linear term, so settle n_ctx first.
    n_ctx = requested_ctx if requested_ctx > 0 else layout.n_ctx_train
    if n_ctx <= 0:
        return Plan(reason = "no usable context length")

    if opts.kv_bytes_at is not None:
        kv_bytes_floor = (
            _kv_floor_at(layout, opts, kv_bytes_floor, n_ctx, n_ctx, max(1, opts.n_parallel)) or 0
        )

    may_shrink = opts.context_policy in (ContextPolicy.FIT_ONLY, ContextPolicy.PREFER_RESIDENT)
    budget = _usable_vram(vram_bytes_per_device, opts, n_ctx)
    if budget <= 0 and not may_shrink:
        return Plan(reason = "no creditable VRAM after per-device overhead and reserved allocations")
    if layout.has_swa and kv_bytes_floor <= 0:
        return Plan(
            n_ctx = n_ctx,
            reason = (
                "sliding-window attention and no measured cache size: the layout's "
                "cache estimate charges every layer the full context and would "
                "invent a deficit, so this is left to llama.cpp's own fitter"
            ),
        )

    resident_quantised = opts.cache_quantised
    if (
        opts.context_policy is ContextPolicy.PREFER_RESIDENT
        and all_resident_bytes(
            layout,
            n_ctx,
            kv_quantised = resident_quantised,
            kv_bytes_floor = kv_bytes_floor,
            kv_on_host = opts.kv_on_host,
            n_seq = max(1, opts.n_parallel),
            trust_floor = opts.kv_bytes_at is not None,
        )
        > budget
    ):
        shrunk = max_context_for(
            layout,
            vram_bytes_per_device,
            kv_quantised = resident_quantised,
            opts = opts,
            kv_bytes_floor = kv_bytes_floor,
            floor_ctx = n_ctx,
            n_seq = max(1, opts.n_parallel),
            kv_on_host = opts.kv_on_host,
        )
        if shrunk >= opts.min_ctx:
            # Same slot count the feasibility charged recurrent state for.
            resident_ctx = min(shrunk, n_ctx)
            resident_knobs = _Knobs(n_parallel = max(1, opts.n_parallel))
            resident_floor = _kv_floor_at(
                layout, opts, kv_bytes_floor, n_ctx, resident_ctx, resident_knobs.n_parallel
            )
            # The default split can still overflow one card: same per-device test as _plan_at.
            uneven = (
                _per_device_shortfall(
                    layout,
                    opts,
                    resident_ctx,
                    {},
                    False,
                    vram_bytes_per_device,
                    quantised = resident_quantised,
                    kv_bytes_floor = resident_floor if resident_floor is not None else 0,
                    split_weights_per_device = split_weights_per_device or vram_bytes_per_device,
                    kv_layer_weights = kv_layer_weights,
                    extra_on_device0 = _outside_layout_bytes(opts, resident_knobs),
                    n_seq = resident_knobs.n_parallel,
                )
                if len(vram_bytes_per_device) > 1
                else None
            )
            if uneven is None:
                return _finish(
                    layout,
                    opts,
                    resident_ctx,
                    [],
                    False,
                    host_ram_bytes,
                    quantised = resident_quantised,
                    kv_bytes_floor = resident_floor if resident_floor is not None else 0,
                    knobs = resident_knobs,
                    kv_layer_weights = kv_layer_weights,
                    requested_ctx = n_ctx,
                    reason = (
                        f"shrank context {n_ctx} -> {min(shrunk, n_ctx)} to keep every tensor "
                        "resident, which outruns a larger spilled context"
                    ),
                )

    declined: Optional[Plan] = None
    unplaced: Optional[Plan] = None
    for quantised in _kv_modes(opts):
        plan = _plan_at(
            layout,
            opts,
            n_ctx,
            host_ram_bytes,
            quantised,
            kv_bytes_floor,
            vram_bytes_per_device,
            split_weights_per_device or vram_bytes_per_device,
            kv_layer_weights,
            requested_ctx = n_ctx,
        )
        if plan is None:
            continue
        if _unplaced(plan):
            unplaced = unplaced or plan
            continue
        if not (plan.declined_by_gate and may_shrink and not plan.veto):
            return plan
        declined = declined or plan

    # The MoE long-prompt veto is read at the reached context, so a cut-context walk could launch
    # every expert on the host; at or above the threshold answer --fit on instead.
    requested_slot_ctx = n_ctx if opts.kv_unified else n_ctx // max(1, opts.n_parallel)
    if (
        may_shrink
        and opts.require_cost_win
        and opts.rank_against_fitter
        and layout.is_moe
        and opts.moe_long_prompt_ctx > 0
        and requested_slot_ctx >= opts.moe_long_prompt_ctx
    ):
        return Plan(
            n_ctx = n_ctx,
            declined_by_gate = True,
            veto = True,
            reason = (
                f"MoE at {requested_slot_ctx} tokens per slot does not place at the context "
                "asked for, and -ot measured 0.94 to 0.97x of llama.cpp's own layerwise fit at "
                "a 32K prompt; a smaller context is not searched, it is left to --fit on"
            ),
        )

    if may_shrink and len(vram_bytes_per_device) > 1 and layout.has_swa:
        may_shrink = False
        if declined is None and unplaced is None:
            return Plan(
                reason = (
                    "the load does not fit at the requested context and a windowed cache "
                    "split across devices cannot be re-priced at a smaller one; leaving "
                    "llama.cpp's own fitter to place it"
                )
            )
    if may_shrink:
        step = max(256, opts.ctx_step // 256 * 256)
        # Must assume every rung above the first weight spill, to be a real upper bound.
        relieved = _Knobs(
            n_parallel = max(1, min(opts.min_parallel, opts.n_parallel)),
            mmproj_to_host = bool(opts.mmproj_movable),
            draft_dropped = bool(opts.draft_droppable),
        )
        relieved_floor = _kv_floor_at(
            layout, opts, kv_bytes_floor, n_ctx, n_ctx, relieved.n_parallel
        )
        for quantised in _kv_modes(opts):
            hi = max_context_for(
                layout,
                vram_bytes_per_device,
                spill_all_ffn = True,
                spill_lm_head = opts.allow_lm_head_spill,
                kv_quantised = quantised,
                opts = opts,
                kv_bytes_floor = kv_bytes_floor if relieved_floor is None else relieved_floor,
                floor_ctx = n_ctx,
                n_seq = relieved.n_parallel,
                kv_on_host = opts.kv_on_host,
                outside_layout_bytes = _outside_layout_bytes(opts, relieved),
                ctx_cap = requested_ctx if requested_ctx > 0 else 0,
            )
            top = min(hi, n_ctx) // 256 * 256
            hi = top
            if (declined is not None or unplaced is not None) and hi >= n_ctx:
                hi = (n_ctx - step) // 256 * 256
            rungs: list[int] = []
            ctx = hi
            while ctx >= opts.min_ctx:
                rungs.append(ctx)
                ctx -= step
            if (
                opts.min_ctx <= top
                and opts.min_ctx < n_ctx
                and (not rungs or rungs[-1] > opts.min_ctx)
            ):
                rungs.append(opts.min_ctx)
            for ctx in rungs:
                plan = _plan_at(
                    layout,
                    opts,
                    ctx,
                    host_ram_bytes,
                    quantised,
                    kv_bytes_floor,
                    vram_bytes_per_device,
                    split_weights_per_device or vram_bytes_per_device,
                    kv_layer_weights,
                    requested_ctx = n_ctx,
                )
                if plan is not None and plan.veto:
                    return replace(plan, n_ctx = n_ctx)
                if plan is not None and _unplaced(plan):
                    unplaced = unplaced or replace(plan, n_ctx = n_ctx)
                    continue
                if plan is not None and not plan.declined_by_gate:
                    return plan

    if declined is not None:
        return declined
    if unplaced is not None:
        return unplaced

    if budget <= 0:
        return Plan(reason = "no creditable VRAM after per-device overhead and reserved allocations")

    floor = resident_floor_bytes(
        layout,
        n_ctx,
        kv_bytes_floor = kv_bytes_floor,
        kv_on_host = opts.kv_on_host,
        trust_floor = opts.kv_bytes_at is not None,
    )
    return Plan(
        changed = False,
        n_ctx = n_ctx,
        insufficient = True,
        vram_bytes = floor,
        reason = (
            f"even with every spillable tensor on the host the load needs "
            f"{floor / GIB:.2f} GiB of VRAM against {budget / GIB:.2f} GiB usable; "
            "keeping mmap so llama.cpp can page rather than be OOM-killed. "
            "A smaller quant or a shorter context is the fix, not more offload"
        ),
    )


def _unplaced(plan: Plan) -> bool:
    """``_plan_at``'s multi-device abstain: no per-device fit shown, nothing priced or gated."""
    return (
        not plan.priced and not plan.changed and not plan.declined_by_gate and not plan.insufficient
    )


def _kv_modes(opts: PlanOptions) -> tuple[bool, ...]:
    """f16 first, then q8_0 only if the caller opted in; a cache already quantised at launch is
    priced as quantised and never as f16."""
    if opts.cache_quantised:
        return (True,)
    return (False, True) if opts.allow_kv_quant else (False,)


def _device_slots(n_slots: int, split_weights: Sequence[float]) -> list[list[int]]:
    """Which of the ``n_slots`` layer rows land on which device.

    Mirrors llama.cpp's default tensor split exactly: free VRAM per device
    (llama-model.cpp:1420-1433), prefix-summed and normalised (:1439-1447), then ``upper_bound``
    on the normalised row index (:1457).
    """

    def f32(value: float) -> float:
        return struct.unpack("=f", struct.pack("=f", value))[0]

    weights = [max(0, v) for v in split_weights]
    total = sum(weights)
    if total <= 0:
        # An all-zero split errors in llama.cpp.
        raise ValueError("split weights are all zero for the selected devices")
    cumulative: list[float] = []
    running = f32(0.0)
    for w in weights:
        running = f32(running + f32(w))
        cumulative.append(running)
    cumulative = [f32(value / running) for value in cumulative]
    slots: list[list[int]] = [[] for _ in weights]
    for row in range(n_slots):
        fraction = f32(f32(row) / f32(n_slots))
        # std::upper_bound: first cumulative strictly greater than fraction.
        device = next((i for i, c in enumerate(cumulative) if c > fraction), len(weights) - 1)
        slots[device].append(row)
    return slots


def _fixed_device_reserve(opts: PlanOptions, n_ctx: int, device: int) -> int:
    """What a device must keep free before a weight lands on it: the per-device reserve (flat plus
    context term) and, on every device after the first, the pipeline buffers a layer split
    allocates there."""
    reserve = _device_reserve(opts, n_ctx)
    if device > 0:
        reserve += max(0, opts.pipeline_overhead_bytes)
    return reserve


def _per_device_usage(
    layout: ModelLayout,
    opts: PlanOptions,
    n_ctx: int,
    spilled: Union[Mapping[int, int], Collection[int]],
    spill_lm_head: bool,
    vram_bytes_per_device: Sequence[int],
    *,
    quantised: bool,
    kv_bytes_floor: int,
    split_weights_per_device: Sequence[float] = (),
    kv_layer_weights: Sequence[int] = (),
    extra_on_device0: Optional[int] = None,
    n_seq: int = 1,
) -> tuple[Optional[str], list[int], list[list[int]]]:
    """Bytes each device would hold under this spill, and the rows it owns."""
    if len(vram_bytes_per_device) <= 1:
        return None, [], []
    unequal_heads = len({h for h in layout.kv_heads_by_layer if h > 0}) > 1
    uneven_cache = (
        layout.recurrent_bytes > 0
        or layout.n_attention_layers != layout.n_layers
        or layout.has_swa
        or unequal_heads
    )
    weights = [max(0, int(w)) for w in kv_layer_weights]
    if len(weights) != layout.n_layers or not any(weights):
        weights = []
    if uneven_cache and not weights:
        if layout.recurrent_bytes > 0:
            return "the recurrent state's per-layer split is not visible in the layout", [], []
        if layout.n_attention_layers != layout.n_layers:
            return (
                f"only {layout.n_attention_layers} of {layout.n_layers} layers hold a cache "
                "and the layout does not say which",
                [],
                [],
            )
        if not layout.has_swa:
            # OpenELM: 3 to 6 KV heads a row; with FA off V is padded to the widest row anyway.
            return (
                "the per-layer KV head counts differ and no per-layer vector was supplied to "
                "say which device holds the wider caches",
                [],
                [],
            )
        return (
            "the cache is per-layer uneven (sliding-window attention) and no per-layer "
            "vector was supplied to say which layers are full-context",
            [],
            [],
        )
    if layout.has_excluded_blocks:
        return "the GGUF carries trailing blocks that shift llama.cpp's row count", [], []

    n_slots = layout.n_layers + 1
    if n_slots <= 1:
        return None, [], []
    cache = (
        0
        if opts.kv_on_host
        else cache_bytes(
            layout,
            n_ctx,
            kv_quantised = quantised,
            kv_bytes_floor = kv_bytes_floor,
            trust_floor = opts.kv_bytes_at is not None,
        )
    )
    total_weight = sum(weights)
    if weights and total_weight > 0:
        kv_by_layer = [(cache * w + total_weight - 1) // total_weight for w in weights]
    else:
        per = (cache + layout.n_layers - 1) // layout.n_layers if layout.n_layers else 0
        kv_by_layer = [per] * layout.n_layers
    if layout.recurrent_bytes > 0 and not opts.kv_on_host:
        recurrent_rows = sorted(_recurrent_rows(layout, weights))
        if not recurrent_rows:
            return "the recurrent state's per-layer split is not visible in the layout", [], []
        state = layout.recurrent_bytes * max(1, n_seq)
        per_row = (state + len(recurrent_rows) - 1) // len(recurrent_rows)
        for row in recurrent_rows:
            kv_by_layer[row] += per_row
    by_index = {b.index: b for b in layout.blocks}
    output_row_bytes = layout.other_resident_bytes + (0 if spill_lm_head else layout.lm_head_bytes)

    def moved_off(block: BlockLayout) -> int:
        if isinstance(spilled, Mapping):
            return max(0, spilled.get(block.index, 0))
        return block.spillable_bytes if block.index in spilled else 0

    try:
        slots = _device_slots(n_slots, split_weights_per_device or vram_bytes_per_device)
    except ValueError as exc:
        return str(exc), [], []
    usage: list[int] = []
    for device, rows in enumerate(slots):
        used = 0
        for row in rows:
            if row == n_slots - 1:
                used += output_row_bytes
                continue
            block = by_index.get(row)
            if block is None:
                continue
            used += max(0, block.resident_bytes + block.spillable_bytes - moved_off(block))
            if row < len(kv_by_layer):
                used += kv_by_layer[row]
        if device == 0:
            used += (
                _outside_layout_bytes(opts)
                if extra_on_device0 is None
                else max(0, extra_on_device0)
            )
        usage.append(used)
    return None, usage, slots


def _per_device_shortfall(
    layout: ModelLayout,
    opts: PlanOptions,
    n_ctx: int,
    spilled: Union[Mapping[int, int], Collection[int]],
    spill_lm_head: bool,
    vram_bytes_per_device: Sequence[int],
    *,
    quantised: bool,
    kv_bytes_floor: int,
    split_weights_per_device: Sequence[float] = (),
    kv_layer_weights: Sequence[int] = (),
    extra_on_device0: Optional[int] = None,
    n_seq: int = 1,
) -> Optional[str]:
    """``None`` when every device provably fits, else why it cannot be shown to.

    A per-device shortfall is a hard throw (llama-model.cpp:1731) and ``--fit off`` means
    common/fit.cpp never runs to catch it.
    """
    error, usage, slots = _per_device_usage(
        layout,
        opts,
        n_ctx,
        spilled,
        spill_lm_head,
        vram_bytes_per_device,
        quantised = quantised,
        kv_bytes_floor = kv_bytes_floor,
        split_weights_per_device = split_weights_per_device,
        kv_layer_weights = kv_layer_weights,
        extra_on_device0 = extra_on_device0,
        n_seq = n_seq,
    )
    if error is not None:
        return error
    for device, (used, rows) in enumerate(zip(usage, slots)):
        reserve = _fixed_device_reserve(opts, n_ctx, device)
        raw_vram = max(0, vram_bytes_per_device[device])
        if used + reserve > raw_vram:
            headroom = max(0, raw_vram - reserve)
            return (
                f"device {device} would still hold {used / GIB:.2f} GiB of its "
                f"{len(rows)}-row share against {headroom / GIB:.2f} GiB usable"
            )
    return None


def _select_units_per_device(
    layout: ModelLayout,
    opts: PlanOptions,
    n_ctx: int,
    vram_bytes_per_device: Sequence[int],
    *,
    quantised: bool,
    kv_bytes_floor: int,
    spill_lm_head: bool = False,
    split_weights_per_device: Sequence[float] = (),
    kv_layer_weights: Sequence[int] = (),
    extra_on_device0: Optional[int] = None,
    n_seq: int = 1,
) -> Optional[list[SpillUnit]]:
    """A spill chosen device by device, or ``None`` when one cannot be shown to fit."""
    error, usage, slots = _per_device_usage(
        layout,
        opts,
        n_ctx,
        {},
        spill_lm_head,
        vram_bytes_per_device,
        quantised = quantised,
        kv_bytes_floor = kv_bytes_floor,
        split_weights_per_device = split_weights_per_device,
        kv_layer_weights = kv_layer_weights,
        extra_on_device0 = extra_on_device0,
        n_seq = n_seq,
    )
    if error is not None:
        return None
    by_index = {block.index: block for block in layout.blocks}
    chosen: list[SpillUnit] = []
    for device, (used, rows) in enumerate(zip(usage, slots)):
        deficit = (
            used
            + _fixed_device_reserve(opts, n_ctx, device)
            - max(0, vram_bytes_per_device[device])
        )
        if deficit <= 0:
            continue
        local_blocks = [by_index[row] for row in rows if row in by_index]
        freed = 0
        taken: list[SpillUnit] = []
        for cls in _rung_classes(layout, opts):
            if freed >= deficit:
                break
            picked, got = _select_units(
                _units_of(local_blocks, cls),
                deficit - freed,
                opts.spill_order,
                cost_ranked = opts.require_cost_win,
            )
            taken.extend(picked)
            freed += got
        if freed < deficit:
            return None
        taken, freed = _grade_the_boundary_block(layout, opts, taken, freed, deficit)
        chosen.extend(taken)
    if (
        _per_device_shortfall(
            layout,
            opts,
            n_ctx,
            _moved_by_index(chosen),
            spill_lm_head,
            vram_bytes_per_device,
            quantised = quantised,
            kv_bytes_floor = kv_bytes_floor,
            split_weights_per_device = split_weights_per_device,
            kv_layer_weights = kv_layer_weights,
            extra_on_device0 = extra_on_device0,
            n_seq = n_seq,
        )
        is not None
    ):
        return None
    return chosen


def _knob_description(knobs: _Knobs, opts: PlanOptions) -> str:
    parts = []
    if knobs.mmproj_to_host:
        parts.append("moving the vision projector to the host")
    if knobs.n_parallel < max(1, opts.n_parallel):
        parts.append(f"serving {knobs.n_parallel} slot(s) instead of {opts.n_parallel}")
    if knobs.draft_dropped:
        parts.append("dropping the speculative draft")
    return ", ".join(parts)


def _plan_at(
    layout: ModelLayout,
    opts: PlanOptions,
    n_ctx: int,
    host_ram_bytes: Optional[int],
    quantised: bool,
    kv_bytes_floor: int = 0,
    vram_bytes_per_device: Sequence[int] = (),
    split_weights_per_device: Sequence[int] = (),
    kv_layer_weights: Sequence[int] = (),
    *,
    requested_ctx: int = 0,
) -> Optional[Plan]:
    """One pass of the ladder at a fixed context and cache dtype."""
    n_devices = len(vram_bytes_per_device)
    floor_ctx = requested_ctx or n_ctx

    def price(k: _Knobs) -> Optional[tuple[int, int, int]]:
        """(needed, budget, cache floor) at these knobs, or None if unpriceable."""
        floor = _kv_floor_at(layout, opts, kv_bytes_floor, floor_ctx, n_ctx, k.n_parallel)
        if floor is None:
            return None
        needed = all_resident_bytes(
            layout,
            n_ctx,
            kv_quantised = quantised,
            kv_bytes_floor = floor,
            kv_on_host = opts.kv_on_host,
            n_seq = k.n_parallel,
            trust_floor = opts.kv_bytes_at is not None,
        )
        budget = _usable_vram(
            vram_bytes_per_device,
            opts,
            n_ctx,
            outside_layout_bytes = _outside_layout_bytes(opts, k),
        )
        return needed, budget, floor

    def shortfall(k: _Knobs, kv_floor: int) -> Optional[str]:
        """Why the split cannot be shown to fit at these knobs with nothing spilled."""
        return _per_device_shortfall(
            layout,
            opts,
            n_ctx,
            {},
            False,
            vram_bytes_per_device,
            quantised = quantised,
            kv_bytes_floor = kv_floor,
            split_weights_per_device = split_weights_per_device,
            kv_layer_weights = kv_layer_weights,
            extra_on_device0 = _outside_layout_bytes(opts, k),
            n_seq = k.n_parallel,
        )

    knobs = _Knobs(n_parallel = max(1, opts.n_parallel))
    priced = price(knobs)
    assert priced is not None  # the caller's own slot count is always priceable
    needed, budget, floor = priced
    fallback_floor = floor

    if needed > budget and opts.mmproj_movable and opts.mmproj_bytes > 0:
        knobs = _Knobs(knobs.n_parallel, True, knobs.draft_dropped)
        needed, budget, floor = price(knobs)  # type: ignore[misc]
    slots_repriceable_per_device = not (n_devices > 1 and layout.has_swa)
    while (
        needed > budget
        and slots_repriceable_per_device
        and knobs.n_parallel > max(1, opts.min_parallel)
    ):
        cand = _Knobs(knobs.n_parallel - 1, knobs.mmproj_to_host, knobs.draft_dropped)
        got = price(cand)
        if got is None or got[0] >= needed:
            break
        knobs, (needed, budget, floor) = cand, got
    if needed > budget and opts.draft_droppable and opts.draft_bytes > 0:
        knobs = _Knobs(knobs.n_parallel, knobs.mmproj_to_host, True)
        needed, budget, floor = price(knobs)  # type: ignore[misc]

    if needed <= budget:
        if n_devices > 1:
            uneven = shortfall(knobs, floor)
            if uneven is not None:

                def take(cand: _Knobs) -> bool:
                    """Adopt ``cand`` if it changes the arithmetic, and re-check."""
                    nonlocal knobs, needed, budget, floor, uneven
                    got = price(cand)
                    if got is None or (got[0] >= needed and got[1] <= budget):
                        return False
                    knobs, (needed, budget, floor) = cand, got
                    uneven = shortfall(knobs, floor)
                    return True

                if uneven is not None and opts.mmproj_movable and opts.mmproj_bytes > 0:
                    take(_Knobs(knobs.n_parallel, True, knobs.draft_dropped))
                while (
                    uneven is not None
                    and slots_repriceable_per_device
                    and knobs.n_parallel > max(1, opts.min_parallel)
                ):
                    step = _Knobs(knobs.n_parallel - 1, knobs.mmproj_to_host, knobs.draft_dropped)
                    if not take(step):
                        break
                if uneven is not None and opts.draft_droppable and opts.draft_bytes > 0:
                    take(_Knobs(knobs.n_parallel, knobs.mmproj_to_host, True))
            if uneven is not None:
                gave_up = _knob_description(knobs, opts)
                after = f" after {gave_up}" if gave_up else ""
                per_device = _select_units_per_device(
                    layout,
                    opts,
                    n_ctx,
                    vram_bytes_per_device,
                    quantised = quantised,
                    kv_bytes_floor = floor,
                    split_weights_per_device = split_weights_per_device,
                    kv_layer_weights = kv_layer_weights,
                    extra_on_device0 = _outside_layout_bytes(opts, knobs),
                    n_seq = knobs.n_parallel,
                )
                if not per_device:
                    return Plan(
                        n_ctx = n_ctx,
                        reason = (
                            f"the pooled budget fits{after}, but {uneven}, and no rung "
                            "covers it device by device; leaving llama.cpp's own fitter "
                            "to place it"
                        ),
                    )
                moved_gib = sum(u.nbytes for u in per_device) / GIB
                return _finish(
                    layout,
                    opts,
                    n_ctx,
                    per_device,
                    False,
                    host_ram_bytes,
                    quantised = quantised,
                    kv_bytes_floor = floor,
                    budget = budget,
                    knobs = knobs,
                    kv_layer_weights = kv_layer_weights,
                    requested_ctx = requested_ctx,
                    fallback_kv_bytes_floor = fallback_floor,
                    reason = (
                        f"the pooled budget fits{after}, but {uneven}; spilled "
                        f"{_rung_description(per_device, layout)} ({moved_gib:.2f} GiB) "
                        "device by device to cover it"
                    ),
                )
        gave_up = _knob_description(knobs, opts)
        return _finish(
            layout,
            opts,
            n_ctx,
            [],
            False,
            host_ram_bytes,
            quantised = quantised,
            kv_bytes_floor = floor,
            budget = budget,
            knobs = knobs,
            kv_layer_weights = kv_layer_weights,
            requested_ctx = requested_ctx,
            fallback_kv_bytes_floor = fallback_floor,
            reason = (
                f"the whole load fits in VRAM ({needed / GIB:.2f} of "
                f"{budget / GIB:.2f} GiB usable"
                + (f" after {gave_up}" if gave_up else "")
                + "), so nothing is spilled"
            ),
        )

    kv_bytes_floor = floor
    deficit = needed - budget
    spillable = [b for b in layout.blocks if b.spillable_bytes > 0]

    per_device_kw = dict(
        quantised = quantised,
        kv_bytes_floor = kv_bytes_floor,
        split_weights_per_device = split_weights_per_device,
        kv_layer_weights = kv_layer_weights,
        extra_on_device0 = _outside_layout_bytes(opts, knobs),
        n_seq = knobs.n_parallel,
    )

    def head_rescue() -> Optional[Plan]:
        """The output device may be short by more than its FFN rows can give, since the output row
        holds lm_head, its own rung. Re-select per device with the head off the card; None when
        that does not fit either."""
        if not (n_devices > 1 and opts.allow_lm_head_spill and layout.lm_head_bytes):
            return None
        with_head = _select_units_per_device(
            layout, opts, n_ctx, vram_bytes_per_device, spill_lm_head = True, **per_device_kw
        )
        if with_head is None:
            return None
        head_freed = sum(u.nbytes for u in with_head) + layout.lm_head_bytes
        if needed - head_freed > budget:
            return None
        return _finish(
            layout,
            opts,
            n_ctx,
            with_head,
            True,
            host_ram_bytes,
            quantised = quantised,
            kv_bytes_floor = kv_bytes_floor,
            budget = budget,
            knobs = knobs,
            kv_layer_weights = kv_layer_weights,
            requested_ctx = requested_ctx,
            fallback_kv_bytes_floor = fallback_floor,
            reason = (
                "spilled the output head after its device could not cover "
                "the local shortfall with FFN blocks alone"
            ),
        )

    def attempt(
        units: list[SpillUnit],
        spill_lm_head: bool,
        kv_host: bool,
        what: str,
        reason: str,
        per_device_selected: bool = False,
    ) -> Plan:
        """Check ``units`` device by device, then either finish or say why not."""
        moved = _moved_by_index(units)
        full = _spills_every_block(layout, units)
        if (
            n_devices > 1
            and not full
            and not per_device_selected
            and not opts.trust_device_row_model
        ):
            # Pooled is not per-device for a PARTIAL spill: the split is fixed before -ot
            # (llama-model.cpp:1425-1457), and -ot only swaps buffer types (llama-model-loader.cpp:1177-1203).
            return Plan(
                n_ctx = n_ctx,
                reason = (
                    f"a partial spill ({_rung_description(units, layout)}) across "
                    f"{n_devices} devices cannot be checked against a pooled budget, "
                    "because llama.cpp assigns contiguous layer ranges per device and "
                    "-ot does not move a layer; leaving llama.cpp's own fitter to place it"
                ),
            )
        uneven = _per_device_shortfall(
            layout,
            opts,
            n_ctx,
            moved,
            spill_lm_head,
            vram_bytes_per_device,
            quantised = quantised,
            kv_bytes_floor = kv_bytes_floor,
            split_weights_per_device = split_weights_per_device,
            kv_layer_weights = kv_layer_weights,
            extra_on_device0 = _outside_layout_bytes(opts, knobs),
            n_seq = knobs.n_parallel,
        )
        if uneven is not None:
            rescued = head_rescue() if not spill_lm_head and not kv_host else None
            if rescued is not None:
                return rescued
            return Plan(
                n_ctx = n_ctx,
                reason = (
                    f"{what} still does not fit device by device: {uneven}; "
                    "leaving llama.cpp's own fitter to place it"
                ),
            )
        return _finish(
            layout,
            opts,
            n_ctx,
            units,
            spill_lm_head,
            host_ram_bytes,
            quantised = quantised,
            kv_bytes_floor = kv_bytes_floor,
            budget = budget,
            kv_on_host_rung = kv_host,
            knobs = knobs,
            kv_layer_weights = kv_layer_weights,
            requested_ctx = requested_ctx,
            fallback_kv_bytes_floor = fallback_floor,
            reason = reason,
        )

    taken: list[SpillUnit] = []
    freed = 0
    for cls in _rung_classes(layout, opts):
        if freed >= deficit:
            break
        chosen, got = _select_units(
            _units_of(layout.blocks, cls),
            deficit - freed,
            opts.spill_order,
            cost_ranked = opts.require_cost_win,
        )
        taken.extend(chosen)
        freed += got

    if freed >= deficit:
        # Graded first: trimming the boundary block can turn an every-block spill partial.
        graded, graded_freed = _grade_the_boundary_block(layout, opts, taken, freed, deficit)
        per_device_selected = False
        if (
            n_devices > 1
            and not _spills_every_block(layout, graded)
            and not opts.trust_device_row_model
        ):
            per_device = _select_units_per_device(
                layout, opts, n_ctx, vram_bytes_per_device, **per_device_kw
            )
            if per_device is not None and needed - sum(u.nbytes for u in per_device) <= budget:
                taken = per_device
                freed = sum(u.nbytes for u in per_device)
                per_device_selected = True
            else:
                rescued = head_rescue()
                if rescued is not None:
                    return rescued
        else:
            taken, freed = graded, graded_freed
            # Every block moved and one card is still short: its own rows can cover that before
            # the output head is given up.
            if (
                n_devices > 1
                and not opts.trust_device_row_model
                and _per_device_shortfall(
                    layout,
                    opts,
                    n_ctx,
                    _moved_by_index(taken),
                    False,
                    vram_bytes_per_device,
                    **per_device_kw,
                )
                is not None
            ):
                per_device = _select_units_per_device(
                    layout, opts, n_ctx, vram_bytes_per_device, **per_device_kw
                )
                if per_device is not None and needed - sum(u.nbytes for u in per_device) <= budget:
                    taken = per_device
                    freed = sum(u.nbytes for u in per_device)
                    per_device_selected = True
        described = _rung_description(taken, layout)
        return attempt(
            taken,
            False,
            False,
            f"spilling {described}",
            (
                f"spilled {described} ({freed / GIB:.2f} GiB) to cover a "
                f"{deficit / GIB:.2f} GiB deficit"
                + (" device by device" if per_device_selected else "")
                + ", keeping the KV cache resident"
            ),
            per_device_selected = per_device_selected,
        )

    # lm_head is next: FFN offload already made generation host-bandwidth-bound.
    if opts.allow_lm_head_spill and layout.lm_head_bytes:
        if freed + layout.lm_head_bytes >= deficit:
            return attempt(
                taken,
                True,
                False,
                "spilling every block and lm_head",
                (
                    f"spilled every block's FFN ({freed / GIB:.2f} GiB) plus lm_head "
                    f"({layout.lm_head_bytes / GIB:.2f} GiB) to cover a "
                    f"{deficit / GIB:.2f} GiB deficit"
                ),
            )
        freed += layout.lm_head_bytes

    if opts.allow_attention_spill:
        attn, got = _select_units(
            _units_of(layout.blocks, SpillClass.ATTENTION),
            deficit - freed,
            opts.spill_order,
            cost_ranked = opts.require_cost_win,
        )
        if got >= deficit - freed:
            units = taken + attn
            return attempt(
                units,
                opts.allow_lm_head_spill and bool(layout.lm_head_bytes),
                False,
                "spilling every block, lm_head and attention weights",
                (
                    f"spilled {_rung_description(units, layout)} plus lm_head to cover a "
                    f"{deficit / GIB:.2f} GiB deficit; attention weights on the host is the "
                    "second-worst rung there is"
                ),
            )
        freed += got
        taken = taken + attn

    if opts.allow_kv_host_fallback and not opts.kv_on_host:
        cache = cache_bytes(layout, n_ctx, kv_quantised = quantised, kv_bytes_floor = kv_bytes_floor)
        if freed + cache + layout.recurrent_bytes >= deficit:
            return attempt(
                taken,
                opts.allow_lm_head_spill and bool(layout.lm_head_bytes),
                True,
                "spilling everything including the KV cache",
                (
                    f"spilled everything the ladder has, including the {cache / GIB:.2f} GiB "
                    f"KV cache, to cover a {deficit / GIB:.2f} GiB deficit; this is the last "
                    "rung and it measured ~1.03 t/s"
                ),
            )
    return None


def _host_ram_refusal(
    opts: PlanOptions, n_ctx: int, host_bytes: int, host_ram_bytes: Optional[int]
) -> Optional[Plan]:
    """The one hard host-RAM admission check, shared by every plan that moves bytes onto the host."""
    if host_ram_bytes is None:
        return None
    spendable = max(0, host_ram_bytes - opts.host_ram_headroom_bytes)
    if host_bytes <= spendable:
        return None
    return Plan(
        n_ctx = n_ctx,
        declined_by_gate = True,
        reason = (
            f"the plan needs {host_bytes / GIB:.2f} GiB of host RAM and only "
            f"{spendable / GIB:.2f} GiB is spendable, so it would page from disk "
            "without even the mmap that makes that survivable; left to --fit on"
        ),
    )


def _fit_arm_slots(
    opts: PlanOptions, kv_bytes_floor: int, fallback_kv_bytes_floor: Optional[int]
) -> tuple[int, int]:
    """(slots, cache floor) of the launch a declined plan leaves to ``--fit on``.

    Declining keeps the caller's own --parallel, so the fitter places that many slots and the
    cache they allocate; pricing it at the count rung 1 lowered to hands it memory it never gets.
    """
    floor = kv_bytes_floor if fallback_kv_bytes_floor is None else fallback_kv_bytes_floor
    return max(1, opts.n_parallel), max(0, int(floor))


def _score_arms(
    plan: Placement,
    fallback: Placement,
    opts: PlanOptions,
    n_prompt: int,
    plan_slots: int,
    fit_slots: int,
) -> tuple[float, float]:
    """Request cost of each arm at the micro-batch its own launch runs with."""

    def score(placement: Placement, slots: int) -> float:
        return rank(
            [placement],
            opts.host,
            n_generated = opts.workload_generated_tokens,
            n_prompt = n_prompt,
            n_ubatch = opts.n_ubatch_by_parallel.get(slots) or opts.n_ubatch,
        )[0][1]

    return score(plan, plan_slots), score(fallback, fit_slots)


def _cost_gate(
    layout: ModelLayout,
    opts: PlanOptions,
    n_ctx: int,
    units: Sequence[SpillUnit],
    spill_lm_head: bool,
    budget: int,
    *,
    quantised: bool,
    kv_bytes_floor: int,
    host_bytes: int = 0,
    host_ram_bytes: Optional[int] = None,
    knobs: Optional[_Knobs] = None,
    kv_layer_weights: Sequence[int] = (),
    fallback_kv_bytes_floor: Optional[int] = None,
) -> tuple[Optional[Plan], float, float]:
    """An abstaining Plan when ``--fit on`` is as good as this spill, else None."""
    plan = _spill_placement(layout, units, spill_lm_head)
    if not plan.host_groups:
        return None, 0.0, 0.0

    # Measured unambiguously worse than the fitter: refuse before comparing.
    refused = _host_ram_refusal(opts, n_ctx, host_bytes, host_ram_bytes)
    if refused is not None:
        return refused, 0.0, 0.0
    if opts.prompt_cache_unbounded:
        # --cache-ram -1 is unbounded, so host residency can never be proved.
        return (
            Plan(
                n_ctx = n_ctx,
                declined_by_gate = True,
                reason = (
                    "the prompt cache is unbounded (--cache-ram -1), so the spill "
                    "would run mapped and the cost model, measured with host weights "
                    "resident, cannot price it; left to --fit on"
                ),
            ),
            0.0,
            0.0,
        )
    if not opts.rank_against_fitter:
        return None, 0.0, 0.0
    n_slots = max(1, knobs.n_parallel if knobs is not None else opts.n_parallel)
    per_slot_ctx = n_ctx if opts.kv_unified else n_ctx // n_slots
    if layout.is_moe and opts.moe_long_prompt_ctx > 0 and per_slot_ctx >= opts.moe_long_prompt_ctx:
        # MEASURED: -ot on an MoE loses to the layerwise fit at a 32K prompt while winning at 2K.
        return (
            Plan(
                n_ctx = n_ctx,
                declined_by_gate = True,
                veto = True,
                reason = (
                    f"MoE at {per_slot_ctx} tokens per slot: -ot measured 0.94 to 0.97x of "
                    "llama.cpp's own layerwise fit at a 32K prompt (5 cells, 2 models, 3 "
                    "hosts), so it is left to --fit on at the context asked for"
                ),
            ),
            0.0,
            0.0,
        )

    kept_on_card = _outside_layout_bytes(opts) - _outside_layout_bytes(opts, knobs)
    fit_slots, fit_floor = _fit_arm_slots(opts, kv_bytes_floor, fallback_kv_bytes_floor)
    fallback = _fit_fallback_placement(
        layout,
        opts,
        budget - max(0, kept_on_card),
        n_ctx,
        quantised = quantised,
        kv_bytes_floor = fit_floor,
        kv_on_host = opts.kv_on_host,
        n_seq = fit_slots,
        kv_layer_weights = kv_layer_weights,
    )
    if fallback is None:
        # Nothing to compare: fit.cpp stops after step 3 instead of failing, which is not modelled.
        return (
            Plan(
                n_ctx = n_ctx,
                declined_by_gate = True,
                reason = (
                    "llama.cpp's own placement for this load could not be modelled, so "
                    "the spill has nothing to be ranked against; left to --fit on"
                ),
            ),
            0.0,
            0.0,
        )

    window = n_ctx if opts.kv_unified else n_ctx // n_slots
    n_prompt = min(max(1, opts.workload_prompt_tokens), max(1, window))
    plan_ms, fit_ms = _score_arms(plan, fallback, opts, n_prompt, n_slots, fit_slots)
    if (
        knobs is not None
        and knobs.draft_dropped
        and opts.draft_drop_penalty_frac > 0
        and not opts.fitter_drops_draft
    ):
        plan_ms *= 1.0 + opts.draft_drop_penalty_frac
    if plan_ms <= fit_ms * (1.0 - opts.min_penalty_reduction):
        return None, plan_ms, fit_ms
    return (
        Plan(
            n_ctx = n_ctx,
            declined_by_gate = True,
            predicted_request_ms = plan_ms,
            predicted_fit_request_ms = fit_ms,
            reason = (
                f"planning this load is not worth it: the spill costs "
                f"{plan_ms:.0f} ms against {fit_ms:.0f} ms for llama.cpp's own fit "
                f"over {opts.workload_prompt_tokens} prompt and "
                f"{opts.workload_generated_tokens} generated tokens, so it is left to --fit on"
            ),
        ),
        plan_ms,
        fit_ms,
    )


def _knob_only_gate(
    layout: ModelLayout,
    opts: PlanOptions,
    n_ctx: int,
    budget: int,
    *,
    quantised: bool,
    kv_bytes_floor: int,
    knobs: Optional[_Knobs],
    kv_layer_weights: Sequence[int] = (),
    fallback_kv_bytes_floor: Optional[int] = None,
) -> tuple[Optional[Plan], float, float]:
    """The same ranking for a plan that spills nothing but gave a knob up."""
    if not opts.rank_against_fitter:
        return None, 0.0, 0.0
    n_slots = max(1, knobs.n_parallel if knobs is not None else opts.n_parallel)
    kept_on_card = _outside_layout_bytes(opts) - _outside_layout_bytes(opts, knobs)
    fit_slots, fit_floor = _fit_arm_slots(opts, kv_bytes_floor, fallback_kv_bytes_floor)
    fallback = _fit_fallback_placement(
        layout,
        opts,
        budget - max(0, kept_on_card),
        n_ctx,
        quantised = quantised,
        kv_bytes_floor = fit_floor,
        kv_on_host = opts.kv_on_host,
        n_seq = fit_slots,
        kv_layer_weights = kv_layer_weights,
    )
    if fallback is None:
        return (
            Plan(
                n_ctx = n_ctx,
                declined_by_gate = True,
                reason = (
                    "llama.cpp's own placement for this load could not be modelled, so "
                    "giving up the knob has nothing to be ranked against; left to --fit on"
                ),
            ),
            0.0,
            0.0,
        )
    plan = Placement(host_groups = [])
    window = n_ctx if opts.kv_unified else n_ctx // n_slots
    n_prompt = min(max(1, opts.workload_prompt_tokens), max(1, window))
    plan_ms, fit_ms = _score_arms(plan, fallback, opts, n_prompt, n_slots, fit_slots)
    if (
        knobs is not None
        and knobs.draft_dropped
        and opts.draft_drop_penalty_frac > 0
        and not opts.fitter_drops_draft
    ):
        plan_ms += opts.draft_drop_penalty_frac * fit_ms
    if plan_ms <= fit_ms * (1.0 - opts.min_penalty_reduction):
        return None, plan_ms, fit_ms
    gave_up = _knob_description(knobs, opts) if knobs is not None else "this plan"
    return (
        Plan(
            n_ctx = n_ctx,
            declined_by_gate = True,
            predicted_request_ms = plan_ms,
            predicted_fit_request_ms = fit_ms,
            reason = (
                f"planning this load is not worth it: {gave_up} "
                f"costs {plan_ms:.0f} ms against {fit_ms:.0f} ms for llama.cpp's own fit "
                f"over {opts.workload_prompt_tokens} prompt and "
                f"{opts.workload_generated_tokens} generated tokens, so it is left to --fit on"
            ),
        ),
        plan_ms,
        fit_ms,
    )


def _patterns_for(layout: ModelLayout, units: Sequence[SpillUnit]) -> list[str]:
    """One ``-ot`` pattern per rung, each naming only the blocks that rung moved."""
    by_class: dict[Optional[SpillClass], set[int]] = {}
    order: list[Optional[SpillClass]] = []
    for unit in units:
        if unit.cls not in by_class:
            by_class[unit.cls] = set()
            order.append(unit.cls)
        by_class[unit.cls].add(unit.index)

    patterns: list[str] = []
    for cls in order:
        indices = by_class[cls]
        candidates = {u.index for u in _units_of(layout.blocks, cls)}
        every = indices >= candidates and not layout.has_excluded_blocks
        listed = None if every else sorted(indices)
        if cls is None:
            patterns.append(spill_pattern_for(layout, listed))
        else:
            patterns.append(spill_pattern_for_class(layout, cls, listed))
    return patterns


def _finish(
    layout: ModelLayout,
    opts: PlanOptions,
    n_ctx: int,
    units: Sequence[SpillUnit],
    spill_lm_head: bool,
    host_ram_bytes: Optional[int],
    *,
    quantised: bool = False,
    kv_bytes_floor: int = 0,
    budget: Optional[int] = None,
    kv_on_host_rung: bool = False,
    knobs: Optional[_Knobs] = None,
    kv_layer_weights: Sequence[int] = (),
    requested_ctx: int = 0,
    reason: str = "",
    fallback_kv_bytes_floor: Optional[int] = None,
) -> Plan:
    """Assemble patterns, decide the load mode, and account for both sides."""
    kv_on_host = opts.kv_on_host or kv_on_host_rung
    spilled_weight_bytes = sum(u.nbytes for u in units) + (
        layout.lm_head_bytes if spill_lm_head else 0
    )
    mmproj_host_bytes = opts.mmproj_bytes if (knobs is not None and knobs.mmproj_to_host) else 0
    kv_host_bytes = (
        cache_bytes(
            layout,
            n_ctx,
            kv_quantised = quantised,
            kv_bytes_floor = kv_bytes_floor,
            trust_floor = opts.kv_bytes_at is not None,
        )
        + layout.recurrent_bytes
        * max(1, knobs.n_parallel if knobs is not None else opts.n_parallel)
        if kv_on_host
        else 0
    )
    ple_lazy_bytes = (
        layout.per_layer_embd_bytes
        if opts.ple_read_lazily
        else min(max(0, int(opts.lazy_read_bytes or 0)), layout.per_layer_embd_bytes)
    )
    host_side = (
        layout.token_embd_bytes
        + layout.per_layer_embd_bytes
        - ple_lazy_bytes
        + spilled_weight_bytes
        + mmproj_host_bytes
        + kv_host_bytes
    )

    if mmproj_host_bytes or (opts.require_cost_win and kv_host_bytes):
        refused = _host_ram_refusal(opts, n_ctx, host_side, host_ram_bytes)
        if refused is not None:
            return refused

    gave_up_a_knob = knobs is not None and (
        knobs.mmproj_to_host or knobs.draft_dropped or knobs.n_parallel < max(1, opts.n_parallel)
    )
    plan_ms = fit_ms = 0.0
    if opts.require_cost_win and budget is not None and (units or spill_lm_head):
        declined, plan_ms, fit_ms = _cost_gate(
            layout,
            opts,
            n_ctx,
            units,
            spill_lm_head,
            budget,
            quantised = quantised,
            kv_bytes_floor = kv_bytes_floor,
            host_bytes = host_side,
            host_ram_bytes = host_ram_bytes,
            knobs = knobs,
            kv_layer_weights = kv_layer_weights,
            fallback_kv_bytes_floor = fallback_kv_bytes_floor,
        )
        if declined is not None:
            return declined
    elif opts.require_cost_win and budget is not None and gave_up_a_knob:
        declined, plan_ms, fit_ms = _knob_only_gate(
            layout,
            opts,
            n_ctx,
            budget,
            quantised = quantised,
            kv_bytes_floor = kv_bytes_floor,
            knobs = knobs,
            kv_layer_weights = kv_layer_weights,
            fallback_kv_bytes_floor = fallback_kv_bytes_floor,
        )
        if declined is not None:
            return declined

    patterns = _patterns_for(layout, units)
    if spill_lm_head:
        patterns.append(LM_HEAD_PATTERN)
    indices = sorted({u.index for u in units})

    spilled_bytes = spilled_weight_bytes
    # token_embd is always host-resident; -nkvo cache included, lazy PLE not.
    host_bytes = host_side
    vram_bytes = (
        all_resident_bytes(
            layout,
            n_ctx,
            kv_quantised = quantised,
            kv_bytes_floor = kv_bytes_floor,
            kv_on_host = kv_on_host,
            n_seq = knobs.n_parallel if knobs is not None else 1,
            trust_floor = opts.kv_bytes_at is not None,
        )
        - spilled_bytes
    )

    # mmap costs 2-4.6x on host weight reads: drop it only when RAM holds the host side. Lazy PLE
    # stays out either way (init_mappings maps whenever lazy.any()).
    if host_ram_bytes is None or opts.prompt_cache_unbounded:
        load_mode_none = False
    else:
        load_mode_none = host_bytes <= max(0, host_ram_bytes - opts.host_ram_headroom_bytes)

    cache_ram_mib = -1
    if load_mode_none and host_ram_bytes is not None and opts.cache_ram_default_mib > 0:
        spendable = max(0, host_ram_bytes - opts.host_ram_headroom_bytes - host_bytes)
        clamped = min(opts.cache_ram_default_mib, spendable // MIB)
        if clamped < opts.cache_ram_default_mib:
            cache_ram_mib = int(clamped)

    n_parallel = 0
    mmproj_to_host = draft_dropped = False
    if knobs is not None:
        if knobs.n_parallel < max(1, opts.n_parallel):
            n_parallel = knobs.n_parallel
        mmproj_to_host = knobs.mmproj_to_host
        draft_dropped = knobs.draft_dropped
    cache_type = opts.kv_quant_type if (quantised and not opts.cache_quantised) else None
    changed = (
        bool(patterns)
        or load_mode_none
        or cache_type is not None
        or n_parallel > 0
        or mmproj_to_host
        or draft_dropped
        or (0 < n_ctx < requested_ctx)
    )
    return Plan(
        changed = changed,
        priced = True,
        n_ctx = n_ctx,
        n_parallel = n_parallel,
        mmproj_to_host = mmproj_to_host,
        draft_dropped = draft_dropped,
        cache_ram_mib = cache_ram_mib,
        ot_patterns = tuple(patterns),
        load_mode_none = load_mode_none,
        cache_type_k = cache_type,
        cache_type_v = cache_type,
        spilled_blocks = tuple(indices),
        spilled_lm_head = spill_lm_head,
        vram_bytes = vram_bytes,
        host_bytes = host_bytes,
        ple_charged_to_host = not ple_lazy_bytes,
        kv_spilled_to_host = kv_on_host_rung,
        predicted_gen_penalty_ms = _spill_penalty_ms(
            layout,
            units,
            spill_lm_head,
            opts.host,
            kv_host_bytes = (
                cache_bytes(layout, n_ctx, kv_quantised = quantised, kv_bytes_floor = kv_bytes_floor)
                if kv_on_host_rung
                else 0
            ),
        ),
        predicted_request_ms = plan_ms,
        predicted_fit_request_ms = fit_ms,
        reason = reason,
    )


def plan_to_args(plan: Plan) -> list[str]:
    """The launch flags for ``plan``. Empty when it changes nothing."""
    args: list[str] = []
    for pattern in plan.ot_patterns:
        args.extend(["-ot", f"{pattern}=CPU"])
    if plan.kv_spilled_to_host:
        args.append("--no-kv-offload")
    if plan.load_mode_none:
        args.extend(["--load-mode", "none"])
    if plan.cache_type_k and plan.cache_type_v:
        args.extend(["--cache-type-k", plan.cache_type_k])
        args.extend(["--cache-type-v", plan.cache_type_v])
    if plan.mmproj_to_host:
        args.append("--no-mmproj-offload")
    if plan.n_parallel > 0:
        args.extend(["--parallel", str(plan.n_parallel)])
    if plan.cache_ram_mib >= 0:
        args.extend(["--cache-ram", str(plan.cache_ram_mib)])
    # --fit off: without the budgeted split llama.cpp sizes rows from the child's free VRAM
    # (llama-model.cpp:1462-1477).
    if len(plan.device_layer_counts) > 1:
        args.extend(["--tensor-split", ",".join(str(n) for n in plan.device_layer_counts)])
    return args


_SMART_OFFLOAD_ON = ("1", "true", "yes", "on", "enabled")


def smart_offload_enabled(env: Optional[Mapping[str, str]] = None) -> bool:
    """Whether the launch path may plan a spill. OFF unless explicitly enabled.

    This was briefly opt-OUT, on 118 paired runs across T4, L4, RTX PRO 6000,
    A100, B200 and a gfx1151 APU. Every one of those hosts is a large one, and
    that turned out to be the whole of the calibration set: #9861 measured 76
    paired cells on a 6-core desktop and the planner was slower in 40 of the 43
    it planned, by up to 8x on generation.

    The mechanism is not the host size alone. The planner #9861 measured only
    ever called ``generation_penalty_ms``, so prefill was not priced at all --
    which is why prefill was slower in 43 of 43 planned cells, without one
    exception. Both gates now price prefill PLUS generation (``_cost_gate`` and
    ``_knob_only_gate`` score each arm with ``rank`` through ``_score_arms``),
    and ``generation_penalty_ms`` only fills the informational
    ``Plan.predicted_gen_penalty_ms``. It stays behind the flag until that
    gate is calibrated on the small desktop hosts #9861 measured.

    Off does not mean the load is unplaced: every path that would have consulted
    the planner falls through to ``--fit on``, which is what the same report
    measured at 0.93x to 1.16x across all 33 cells where the planner declined.

    An UNRECOGNISED value disables, same as before, and now agrees with the
    default rather than reversing it.
    """
    raw = (os.environ if env is None else env).get("UNSLOTH_SMART_OFFLOAD")
    if raw is None:
        return False
    return str(raw).strip().lower() in _SMART_OFFLOAD_ON
