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
from dataclasses import dataclass, field
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

# Ceiling for the context probe in :func:`max_context_for` when no reserve slope and
# no training window bound the search. Far above any window llama.cpp serves.
_MAX_CTX_SEARCH = 1 << 24

# Nominal ggml bits-per-weight, by quant type: a type's block size divided by its bytes per
# block.
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

# Above this ratio the every-block ladder beat the whole-FFN planner on every MoE
# measured; below it, it lost or went flat.
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
    # Mean over blocks: a mixed quant gives different types to different blocks, and
    # the decision is about the model, not about one layer.
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
    # The part of that reserve that is NOT flat, in KiB per token of TOTAL context, not of
    # prompt length.
    overhead_bytes_per_token: int = 23961
    # Below this the term is zero, so the flat reserve is unchanged and no existing placement
    # moves.
    overhead_free_ctx: int = 32768
    extra_resident_bytes: int = 0
    pipeline_overhead_bytes: int = 0
    # Host RAM this planner refuses to spend, so a spill does not push the box into swap.
    host_ram_headroom_bytes: int = 2 * GIB
    # True when THIS launch will read layout.per_layer_embd_bytes from the mapping rather than hold it resident, which
    # takes it out of the mmap branch's host RAM. The seam decides it: the arch, the build's --lazy-mode and the
    # resolved mode all have to agree (llama-model-loader.cpp:llama_model_loader::lazy_read::add). Default False keeps
    # the old full charge for every caller that does not price it.
    ple_read_lazily: bool = False
    context_policy: ContextPolicy = ContextPolicy.NEVER_REDUCE
    min_ctx: int = 4096
    spill_order: SpillOrder = SpillOrder.BACK_FIRST
    # How finely a block's FFN may be broken up. BOUNDARY is what llama.cpp does
    # (``fit.cpp:490`` applies its graded fraction to the boundary layer ``il0`` and
    # ``LAYER_FRACTION_MOE`` to every layer past it).
    ffn_granularity: FfnGranularity = FfnGranularity.BOUNDARY
    # Pick the granularity from the model's quant types instead of taking ``ffn_granularity`` as
    # given.
    granularity_from_quant: bool = False
    # Which matrix goes first. llama.cpp's order (common/fit.cpp:407-440), and UNMEASURED by us.
    ffn_rung_order: tuple[SpillClass, ...] = FFN_SPILL_CLASSES
    allow_lm_head_spill: bool = True
    allow_attention_spill: bool = False
    allow_kv_host_fallback: bool = False
    # Let a PARTIAL spill through the multi-device check instead of abstaining. Lifting it rests
    # on one unvalidated thing: ``_device_slots`` reproducing llama.cpp's row assignment (free
    # memory per device at llama-model.cpp:1425-1433, prefix-summed at :1439-1447, upper_bound
    # on the normalised layer index at :1457).
    trust_device_row_model: bool = False
    host: HostProfile = field(default_factory = HostProfile)
    # q8_0 measured slower generation, and without GGML_CUDA_FA_ALL_QUANTS only four
    # MATCHED K/V combinations are compiled (a mismatched pair falls to CPU and stalls).
    allow_kv_quant: bool = False
    kv_quant_type: str = "q8_0"
    # The launch's cache is ALREADY quantised (its element is under two bytes), so the first,
    # and normally only, mode is priced as such and ``kv_quant_type`` names the type in force.
    cache_quantised: bool = False
    # The user set ``--cache-ram -1``, so no figure charged for the cache is a ceiling.
    prompt_cache_unbounded: bool = False
    # The caller passed -nkvo, so llama.cpp puts the WHOLE cache on the host: offload is one
    # scalar and the buffer type falls back to the CPU one for every layer
    # (llama-kv-cache.cpp:210-219), same branch in the recurrent and DSV4 caches.
    kv_on_host: bool = False
    workload_prompt_tokens: int = 2048
    workload_generated_tokens: int = 256
    n_ubatch: int = 512
    # How much of llama.cpp's own predicted penalty a plan must remove before it is worth
    # deviating from ``--fit on`` at all.
    min_penalty_reduction: float = 0.10
    require_cost_win: bool = False

    # Every default below means "not supplied", so a call that sets none of them is
    # byte-identical to the planner before they existed.
    mmproj_bytes: int = 0
    mmproj_movable: bool = False
    n_parallel: int = 1
    min_parallel: int = 1
    kv_bytes_floor_by_parallel: Mapping[int, int] = field(default_factory = dict)
    kv_bytes_at: Optional[Callable[[int, int], int]] = None
    overhead_bytes_at: Optional[Callable[[int], int]] = None
    n_ubatch_by_parallel: Mapping[int, int] = field(default_factory = dict)
    draft_bytes: int = 0
    draft_droppable: bool = False
    draft_drop_penalty_frac: float = 0.05
    cache_ram_default_mib: int = 8192
    # MoE at a long prompt is the one operating point where -ot measured WORSE than llama.cpp's
    # own layerwise fit.
    moe_long_prompt_ctx: int = 32768
    kv_unified: bool = False
    ctx_step: int = 1024


@dataclass(frozen = True)
class Plan:
    """What to launch with, and why."""

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
    # Predicted extra ms per generated token versus fully resident, on the host this was
    # planned for. 0.0 when nothing is spilled.
    predicted_gen_penalty_ms: float = 0.0
    # The two sides of the cost gate, in ms for a whole request of ``PlanOptions``'s workload
    # shape: this placement, and what llama.cpp's own fitter would have cost.
    predicted_request_ms: float = 0.0
    predicted_fit_request_ms: float = 0.0
    reason: str = ""
    n_parallel: int = 0
    mmproj_to_host: bool = False
    draft_dropped: bool = False
    cache_ram_mib: int = -1
    declined_by_gate: bool = False
    # Whether host_bytes above includes the per-layer embeddings, so the branch this plan was sized on is on the record
    # rather than inferable only from load_mode_none.
    ple_charged_to_host: bool = True
    # The decline is a MEASUREMENT rather than a comparison, and it does not move with the
    # context, so FIT_ONLY must not retry smaller.
    veto: bool = False
    priced: bool = False

    @property
    def spills_anything(self) -> bool:
        return bool(self.spilled_blocks) or self.spilled_lm_head

    @property
    def reshapes_launch(self) -> bool:
        """A rung above the first weight spill fired, so the launch changes even
        when no pattern is emitted."""
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
    """Device bytes the layout cannot see: the caller's scalar, the projector unless
    rung 0 moved it, and the draft unless rung 2 dropped it."""
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
    # One unified cache serves every slot: only the recurrent state (charged per
    # slot by the resident sizes) follows the count, the attention cache does not.
    want = at if opts.kv_unified else max(1, n_parallel)
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
        # A hybrid's floor is part fixed state, and the state does not shrink with the context:
        # scaling the whole scalar prices away memory llama.cpp still allocates and the load
        # then OOMs.
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
                # Prefer the SMALLEST unit that closes the gap: the last pick must
                # not overshoot by a whole large one.
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
    # The last coarse unit taken IS the boundary: selection appends in the order
    # it closed the gap, so everything before it was still short of the deficit.
    boundary = coarse[-1]
    if taken[-1] is not boundary:
        # A LATER rung closed the gap: the coarse rung ran out and dense or shared-FFN units
        # followed it.
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
        # Either the graded rungs cannot cover what the whole block covered, or they add
        # up to it anyway. Keep the coarse unit: one pattern, one split fewer.
        return taken, freed
    return [u for u in taken if u is not boundary] + kept, running


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
    total = len([b for b in layout.blocks if b.spillable_bytes > 0]) or len(layout.blocks)
    seen: list[Optional[SpillClass]] = []
    counts: dict[Optional[SpillClass], set[int]] = {}
    for unit in units:
        if unit.cls not in counts:
            counts[unit.cls] = set()
            seen.append(unit.cls)
        counts[unit.cls].add(unit.index)
    parts = []
    for cls in seen:
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
        # Shared experts and dense FFN: read in full every token, unlike the
        # routed experts above, so no activation fraction applies.
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
    down = block.class_bytes(SpillClass.FFN_DOWN)
    for nbytes in (down, down + block.class_bytes(SpillClass.FFN_GATE), block.spillable_bytes):
        if nbytes > 0 and nbytes >= deficit:
            return nbytes
    return None


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
    """What llama.cpp's own fitter would place here, priced the same way. The output tensor rides
    the layer list at ``n_layer_all`` and is the LAST row to leave, not the first: llama.cpp
    keeps rows ``[i_gpu_start, i_gpu_start + act_gpu_layers)`` with ``i_gpu_start =
    max(n_layer_all + 1 - n_gpu_layers, 0)`` (llama-model.cpp:1467-1492), so lm_head is resident
    in every placement this loop can return.
    """
    blocks = list(layout.blocks)
    if not blocks:
        return None
    # Both arms size ONE cache: the planner's arm trusts the exact size, so the
    # fitter must too (the product charges a q4_0 cache 1.78x its real bytes).
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
    # Running weight over the blocks the fitter walks, so a cache total can be
    # apportioned over any PREFIX of them. Uniform when the caller cannot say.
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

    # The cache is RESERVED at n_ctx but only the live prefix is ever read, and reading is what
    # costs.
    slot_window = n_ctx if opts.kv_unified else n_ctx // max(1, n_seq)
    live_tokens = min(
        max(1, slot_window), max(1, opts.workload_prompt_tokens + opts.workload_generated_tokens)
    )
    # Scaled by whatever correction ``kv_bytes_floor`` applied to the RESERVED size, so both
    # sides of this function describe one cache.
    reserved_product = cache_bytes(layout, n_ctx, kv_quantised = quantised)
    floor_scale = (kv_total / reserved_product) if reserved_product > 0 else 1.0
    if kv_on_host:
        kv_live_total = 0
    elif layout.has_swa and kv_bytes_floor > 0:
        # The measured floor of a windowed cache is context-FLAT once the window is saturated,
        # and the layout does not say how it splits between windowed and full-context layers.
        kv_live_total = kv_total
    else:
        kv_live_total = int(cache_bytes(layout, live_tokens, kv_quantised = quantised) * floor_scale)

    if layout.is_moe:
        # MEASURED, not assumed. On an MoE model ``--fit on`` keeps EVERY layer on the
        # device and moves only the trailing layers' expert tensors, through the same kind
        # of tensor override the planner emits (fit.cpp:434-440), so the cache stays
        # resident and it moves close to the MINIMUM it needs, same as the planner.
        host_experts = 0
        for block in reversed(blocks):
            host_experts += block.spillable_bytes
            if resident - host_experts <= budget:
                return Placement(host_groups = [_ffn_group(layout, host_experts)])
        # Every expert on the host and still short. common/fit.cpp does not fail here: it
        # lowers n_gpu_layers and moves whole LEADING layers with their cache share.
        recurrent_per_layer = (
            0.0 if kv_on_host else layout.recurrent_bytes * max(1, n_seq) / len(blocks)
        )
        host_layers = 0
        for moved, block in enumerate(blocks, start = 1):
            host_layers += block.resident_bytes
            host_recurrent = int(recurrent_per_layer * moved)
            freed = host_experts + host_layers + kv_freed(kv_total, moved) + host_recurrent
            if resident - freed <= budget:
                groups = [_ffn_group(layout, host_experts)]
                if host_layers:
                    groups.append(TensorGroup("layers", host_layers, Access.CONTIGUOUS))
                if host_recurrent:
                    groups.append(
                        TensorGroup("recurrent (moved layers)", host_recurrent, Access.CONTIGUOUS)
                    )
                return Placement(host_groups = groups, kv_host_bytes = kv_freed(kv_live_total, moved))
        return None

    # Dense, where the whole-layer model IS what happens: the fitter only lowers
    # n_gpu_layers (common/fit.cpp:551-559) and ``i_gpu_start`` sends every row BELOW it
    # to the CPU (llama-model.cpp:1479-1484), so the host takes the LEADING blocks.
    recurrent_per_layer = (
        0.0 if kv_on_host else layout.recurrent_bytes * max(1, n_seq) / len(blocks)
    )

    def dense_placement(moved: int, spilled_ffn: int, attention: int) -> Placement:
        """``moved`` whole layers off the device, plus ``spilled_ffn`` FFN bytes."""
        groups: list[TensorGroup] = []
        if spilled_ffn:
            groups.append(_ffn_group(layout, spilled_ffn))
        if attention:
            groups.append(TensorGroup("layers", attention, Access.CONTIGUOUS))
        host_recurrent = int(recurrent_per_layer * moved)
        if host_recurrent:
            # CONTIGUOUS, not the cache rate: this is a small fixed-size conv and SSM state read
            # straight through by the scan, not attention over a growing prefix.
            groups.append(
                TensorGroup("recurrent (moved layers)", host_recurrent, Access.CONTIGUOUS)
            )
        live_kv = kv_freed(kv_live_total, moved)
        # ``kv_host_bytes``, so this is charged at ``Access.KV_CACHE``'s calibrated 20.1x
        # and not at the contiguous weight rate: a host layer's cache is a host buffer
        # (llama-kv-cache.cpp:214-225), but batch-1 attention over it is a strided GEMV,
        # not the contiguous quantised GEMM 1.00x was measured on.
        return Placement(host_groups = groups, kv_host_bytes = live_kv)

    host_weights = 0
    host_spillable = 0
    for moved, block in enumerate(blocks, start = 1):
        # fit.cpp's step 4 first: rather than lower ngl again it keeps this layer on the
        # device and overrides part of its FFN, so its cache and state stay resident.
        short = resident - (
            host_weights + kv_freed(kv_total, moved - 1) + int(recurrent_per_layer * (moved - 1))
        )
        overflow = _fit_boundary_overflow(block, short - budget)
        if overflow is not None:
            return dense_placement(
                moved - 1, host_spillable + overflow, host_weights - host_spillable
            )
        host_weights += block.spillable_bytes + block.resident_bytes
        host_spillable += block.spillable_bytes
        # lm_head is NOT in here: the output row stays on the device for any n_gpu_layers
        # >= 1, so charging it both freed and billed bytes llama.cpp never moves.
        freed = host_weights + kv_freed(kv_total, moved) + int(recurrent_per_layer * moved)
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
        # ``PlanOptions.kv_bytes_at`` priced THIS context and slot count, so there is nothing
        # left for the product to correct at any architecture.
        return floor
    if layout.has_swa and floor:
        # Sliding-window attention breaks the product in the UP direction by construction:
        # a window-sized cache, narrower heads, and one cache shared across 20 layers.
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
) -> int:
    """Largest context whose cache fits, rounded down to 256 as CUDA wants."""
    opts = opts or PlanOptions()
    if not layout.complete or layout.kv_bytes_per_token_f16 <= 0:
        return 0
    fixed = (
        layout.block_resident_bytes
        + layout.other_resident_bytes
        + (0 if kv_on_host else layout.recurrent_bytes * max(1, n_seq))
        + (0 if spill_lm_head else layout.lm_head_bytes)
        + (0 if spill_all_ffn else layout.spillable_bytes)
        # Shared and dense FFN inside a MoE block sit in resident_bytes and are a
        # rung of their own, so "every FFN spilled" has to release them too.
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
            # Flat for the WHOLE floor, and only because one scalar cannot say which layers hold
            # the full-context half. ``kv_bytes_at`` splits it.
            return floor
        scaled = floor * ctx // floor_at if floor_at > 0 else floor
        if layout.has_mla:
            # ``cache_bytes`` trusts the floor outright on MLA, so taking the max here refused
            # every MLA model that needed a shrink at a context this budget holds.
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
        elif layout.n_ctx_train:
            hi = layout.n_ctx_train
        elif priced:
            # No reserve slope and no window: double until the priced cache alone is
            # over budget, since the product would cut the search off far below it.
            hi = 256
            while hi < _MAX_CTX_SEARCH and cache_at(hi) <= top:
                hi *= 2
        else:
            hi = top // per_token
        hi = hi // 256 * 256
    else:
        hi = (top // per_token) // 256 * 256
    if layout.n_ctx_train:
        hi = min(hi, layout.n_ctx_train // 256 * 256)
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
    """Decide the placement for one launch.

    ``split_weights_per_device`` is the RAW free VRAM llama.cpp will size its row
    ranges from, in the same device order as ``vram_bytes_per_device``. It is a
    different quantity from the budget by construction -- the budget subtracts a
    per-card reserve -- so the two must not be conflated when modelling the
    split. Empty falls back to the budget, which is right whenever the caller has
    applied no per-card adjustment at all.

    ``kv_layer_weights`` is each layer's RELATIVE cache size, scaled to the total
    the planner already trusts: it PLACES the cache, never re-sizes it. Empty
    means the caller cannot say, and the per-device check then abstains.

    ``kv_bytes_floor`` is an attention-cache size the caller has already computed
    byte-accurately for this launch. The planner never reserves less than it; see
    :func:`cache_bytes` for why the layout's own f16 product is not enough on its
    own. 0 (the default) keeps the pure-layout arithmetic.

    Ladder, cheapest first, measured on a dense 27B at 128K:
      rung 0  nothing spilled                    75.37 t/s
      rung 1  FFN to host                        13.63 t/s
      rung 2  FFN + lm_head                      11.39 t/s
      never   -ngl or --no-kv-offload            ~1.03 t/s

    The order is confirmed by the cost model rather than assumed, and is stated
    in TIME. Ranking on percentage loss is wrong: lm_head reads "43% alone, 16%
    on top of FFN", which looks sub-additive, while the same 0.97 GiB costs
    10.206 ms/token alone and 14.428 on top -- 41% MORE, not less. Percentages
    of different baselines are not commensurable; milliseconds are.
    """
    opts = opts or PlanOptions()

    if not layout.complete or not vram_bytes_per_device:
        return Plan(reason = "layout or device inventory incomplete, leaving llama.cpp defaults")
    if opts.host.unified_memory:
        return Plan(reason = "unified memory host, spilling frees no device memory")
    # Settle the context first: the per-device reserve has a context-linear term, so the budget
    # is a function of n_ctx and cannot be computed above it.
    n_ctx = requested_ctx if requested_ctx > 0 else layout.n_ctx_train
    if n_ctx <= 0:
        return Plan(reason = "no usable context length")

    if opts.kv_bytes_at is not None:
        kv_bytes_floor = (
            _kv_floor_at(layout, opts, kv_bytes_floor, n_ctx, n_ctx, max(1, opts.n_parallel)) or 0
        )

    # A policy that may shrink prices the budget again at every candidate context: the
    # reserve is context-linear, so a card asked at 131072 can be left with nothing.
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

    # PREFER_RESIDENT gets its say before the ladder: a smaller fully resident context outruns a
    # larger spilled one.
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
            # The feasibility above charged one recurrent state per slot, so the plan has to be
            # assembled at the same count or vram_bytes under-reports by (slots - 1) states.
            resident_ctx = min(shrunk, n_ctx)
            resident_knobs = _Knobs(n_parallel = max(1, opts.n_parallel))
            resident_floor = _kv_floor_at(
                layout, opts, kv_bytes_floor, n_ctx, resident_ctx, resident_knobs.n_parallel
            )
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
        if not (plan.declined_by_gate and may_shrink and not plan.veto):
            return plan
        declined = declined or plan

    if may_shrink and len(vram_bytes_per_device) > 1 and layout.has_swa:
        may_shrink = False
        if declined is None:
            return Plan(
                reason = (
                    "the load does not fit at the requested context and a windowed cache "
                    "split across devices cannot be re-priced at a smaller one; leaving "
                    "llama.cpp's own fitter to place it"
                )
            )
    if may_shrink:
        step = max(256, opts.ctx_step // 256 * 256)
        # The bound has to assume every rung above the first weight spill is applied, or it is
        # not an upper bound for what _plan_at will retry, and the ladder then never looks.
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
            )
            top = min(hi, n_ctx) // 256 * 256
            hi = top
            if declined is not None and hi >= n_ctx:
                hi = (n_ctx - step) // 256 * 256
            rungs: list[int] = []
            ctx = hi
            while ctx >= opts.min_ctx:
                rungs.append(ctx)
                ctx -= step
            # The lattice steps down from the top and lands on min_ctx only by coincidence, so
            # the minimum is the last rung whenever it is feasible and not the refused request.
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
                if plan is not None and not plan.declined_by_gate:
                    return plan

    if declined is not None:
        return declined

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


def _kv_modes(opts: PlanOptions) -> tuple[bool, ...]:
    """f16 first, then q8_0 only if the caller opted in; a cache already quantised
    at launch is priced as quantised and never as f16."""
    if opts.cache_quantised:
        return (True,)
    return (False, True) if opts.allow_kv_quant else (False,)


def _device_slots(n_slots: int, split_weights: Sequence[float]) -> list[list[int]]:
    """Which of the ``n_slots`` layer rows land on which device. Mirrors llama.cpp's default tensor
    split exactly: free VRAM per device (llama-model.cpp:1420-1433), prefix-summed and
    normalised (:1439-1447), then ``upper_bound`` on the normalised row index (:1457).
    """

    def f32(value: float) -> float:
        return struct.unpack("=f", struct.pack("=f", value))[0]

    weights = [max(0, v) for v in split_weights]
    total = sum(weights)
    if total <= 0:
        # llama.cpp prefix-sums the split and divides by the total, so an all-zero one is not a
        # placement it produces: it errors.
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
    """What a device must keep free before a single weight lands on it: the
    per-device reserve (flat plus context term) and, on every device after the
    first, the pipeline buffers a layer split allocates there."""
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
    # These three shapes are only a problem when the cache has to be spread evenly for
    # want of anything better. A vector removes that guess; without one they abstain.
    uneven_cache = (
        layout.recurrent_bytes > 0 or layout.n_attention_layers != layout.n_layers or layout.has_swa
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
    # Scaled without under-booking the caller's total (ceiling, not floor: a
    # per-device shortfall is a hard throw). Uniform when unsupplied.
    total_weight = sum(weights)
    if weights and total_weight > 0:
        kv_by_layer = [(cache * w + total_weight - 1) // total_weight for w in weights]
    else:
        per = (cache + layout.n_layers - 1) // layout.n_layers if layout.n_layers else 0
        kv_by_layer = [per] * layout.n_layers
    # The recurrent state is one copy per slot on the rows that hold no cache, and the pooled
    # fit charges it in full.
    if layout.recurrent_bytes > 0 and not opts.kv_on_host:
        recurrent_rows = [i for i, w in enumerate(weights) if w == 0]
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
        # Everything outside the layout sits on the main device, which is
        # devices[0] once -sm none has already pruned the list.
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
    """``None`` when every device provably fits, else why it cannot be shown to. A per-device
    shortfall is a hard throw (llama-model.cpp:1731) and ``--fit off`` means common/fit.cpp
    never runs to catch it.
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
        # The same grading the pooled ladder applies, per device: each device's last whole
        # block is its own boundary, and left whole it is host traffic nothing needed.
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
            # A pooled fit is not a per-device fit, and a plan that spills nothing is still
            # emitted as ``-ngl -1 --fit off`` whenever it reshapes the launch.
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
        """The output device may be short by more than its FFN rows can give:
        the output row holds lm_head, which is its own rung. Re-select per
        device with the head off the card; None when that does not fit either."""
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
        full = all(moved.get(b.index, 0) >= b.spillable_bytes for b in spillable)
        if (
            n_devices > 1
            and not full
            and not per_device_selected
            and not opts.trust_device_row_model
        ):
            # A pooled budget is not a per-device fit test for a PARTIAL spill. llama.cpp fixes
            # the split before any override exists (llama-model.cpp:1425-1457), and -ot only
            # swaps a tensor's buffer type (llama-model-loader.cpp:1177-1203), leaving
            # dev_layer(il) untouched (llama-model.cpp:1467-1474).
            return Plan(
                n_ctx = n_ctx,
                reason = (
                    f"a partial spill ({len(moved)} of {len(spillable)} blocks) across "
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
            reason = reason,
        )

    # Walk the ladder: the MINIMAL set of units from the cheapest rung, stepping down
    # only when that rung is exhausted.
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
        moved = _moved_by_index(taken)
        full = all(moved.get(b.index, 0) >= b.spillable_bytes for b in spillable)
        per_device_selected = False
        if n_devices > 1 and not full and not opts.trust_device_row_model:
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
            # Otherwise attempt() abstains: the pooled pick is partial and cannot
            # be shown to fit device by device.
        else:
            taken, freed = _grade_the_boundary_block(layout, opts, taken, freed, deficit)
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

    # Every spillable weight is on the host and it is still short: lm_head is the next
    # rung, and it costs less here because FFN offload already made generation
    # host-bandwidth-bound.
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
) -> tuple[Optional[Plan], float, float]:
    """An abstaining Plan when ``--fit on`` is as good as this spill, else None."""
    plan = _spill_placement(layout, units, spill_lm_head)
    if not plan.host_groups:
        return None, 0.0, 0.0

    # A spill the host cannot hold in RAM is the one configuration measured to be
    # unambiguously worse than the fitter, so it is refused before any comparison.
    refused = _host_ram_refusal(opts, n_ctx, host_bytes, host_ram_bytes)
    if refused is not None:
        return refused, 0.0, 0.0
    if opts.prompt_cache_unbounded:
        # --cache-ram -1 bounds the prompt cache by nothing, so the host side can never be
        # proved resident and an accepted spill launches under mmap.
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
    n_slots = max(1, knobs.n_parallel if knobs is not None else opts.n_parallel)
    # One shared stream under --kv-unified, so the window a single request may fill is
    # the whole n_ctx however many slots are served; N private windows without it.
    per_slot_ctx = n_ctx if opts.kv_unified else n_ctx // n_slots
    if layout.is_moe and opts.moe_long_prompt_ctx > 0 and per_slot_ctx >= opts.moe_long_prompt_ctx:
        # MEASURED, and the cost model cannot see it: -ot on an MoE loses to llama.cpp's
        # layerwise fit at a 32K prompt while winning at 2K.
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

    fallback = _fit_fallback_placement(
        layout,
        opts,
        budget,
        n_ctx,
        quantised = quantised,
        kv_bytes_floor = kv_bytes_floor,
        kv_on_host = opts.kv_on_host,
        n_seq = n_slots,
        kv_layer_weights = kv_layer_weights,
    )
    if fallback is None:
        # Nothing to COMPARE to, which is not the same as "the fitter cannot place this":
        # common/fit.cpp stops after step 3 instead of failing, and that is not modelled here.
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

    # The workload is a request, and a request does not get longer because the server takes
    # fewer of them at once: rung 1 lowering the slot count leaves it alone.
    window = n_ctx if opts.kv_unified else n_ctx // n_slots
    n_prompt = min(max(1, opts.workload_prompt_tokens), max(1, window))
    scored = rank(
        [plan, fallback],
        opts.host,
        n_generated = opts.workload_generated_tokens,
        n_prompt = n_prompt,
        n_ubatch = opts.n_ubatch_by_parallel.get(n_slots) or opts.n_ubatch,
    )
    plan_ms = _score_of(plan, scored)
    fit_ms = _score_of(fallback, scored)
    if knobs is not None and knobs.draft_dropped and opts.draft_drop_penalty_frac > 0:
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
) -> tuple[Optional[Plan], float, float]:
    """The same ranking for a plan that spills nothing but gave a knob up."""
    n_slots = max(1, knobs.n_parallel if knobs is not None else opts.n_parallel)
    kept_on_card = _outside_layout_bytes(opts) - _outside_layout_bytes(opts, knobs)
    fallback = _fit_fallback_placement(
        layout,
        opts,
        budget - max(0, kept_on_card),
        n_ctx,
        quantised = quantised,
        kv_bytes_floor = kv_bytes_floor,
        kv_on_host = opts.kv_on_host,
        n_seq = n_slots,
        kv_layer_weights = kv_layer_weights,
    )
    if fallback is None:
        return None, 0.0, 0.0
    plan = Placement(host_groups = [])
    window = n_ctx if opts.kv_unified else n_ctx // n_slots
    scored = rank(
        [plan, fallback],
        opts.host,
        n_generated = opts.workload_generated_tokens,
        n_prompt = min(max(1, opts.workload_prompt_tokens), max(1, window)),
        n_ubatch = opts.n_ubatch_by_parallel.get(n_slots) or opts.n_ubatch,
    )
    plan_ms = _score_of(plan, scored)
    fit_ms = _score_of(fallback, scored)
    if knobs is not None and knobs.draft_dropped and opts.draft_drop_penalty_frac > 0:
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
        # "Every" means every block that HAS this rung: comparing attention against the
        # FFN-spillable set would collapse a partial spill to an unbounded pattern.
        candidates = {u.index for u in _units_of(layout.blocks, cls)}
        every = indices >= candidates and not layout.has_excluded_blocks
        listed = None if every else sorted(indices)
        if cls is None:
            patterns.append(spill_pattern_for(layout, listed))
        else:
            patterns.append(spill_pattern_for_class(layout, cls, listed))
    return patterns


def _score_of(placement: Placement, scored: Sequence[tuple[Placement, float]]) -> float:
    """``rank`` sorts, so the order it returns is not the order it was given."""
    for candidate, score in scored:
        if candidate is placement:
            return score
    raise KeyError("placement was not ranked")


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
) -> Plan:
    """Assemble patterns, decide the load mode, and account for both sides."""
    # The bottom rung moved the cache, so from here on this load behaves exactly
    # like one the caller had passed -nkvo for.
    kv_on_host = opts.kv_on_host or kv_on_host_rung
    spilled_weight_bytes = sum(u.nbytes for u in units) + (
        layout.lm_head_bytes if spill_lm_head else 0
    )
    # Rung 0 did not make the projector disappear, it moved it: clip.cpp allocates it in
    # a CPU backend buffer, so those bytes are HOST RAM for the life of the server.
    mmproj_host_bytes = opts.mmproj_bytes if (knobs is not None and knobs.mmproj_to_host) else 0
    # -nkvo puts the cache and the recurrent state in host RAM for the life of the server, so
    # they are part of the host side every decision below spends.
    kv_host_bytes = (
        cache_bytes(layout, n_ctx, kv_quantised = quantised, kv_bytes_floor = kv_bytes_floor)
        + layout.recurrent_bytes
        * max(1, knobs.n_parallel if knobs is not None else opts.n_parallel)
        if kv_on_host
        else 0
    )
    # Lazily-read per-layer embeddings are paged out of the mapping, so under mmap they are page cache the OS can
    # evict, not a resident cost: charging 26.82 GiB of Qwen3.8-Flash-Next's PLE flipped this plan to pageable and
    # refused it on machines that had the room.
    ple_lazy_bytes = layout.per_layer_embd_bytes if opts.ple_read_lazily else 0
    host_side = (
        layout.token_embd_bytes
        - ple_lazy_bytes
        + spilled_weight_bytes
        + mmproj_host_bytes
        + kv_host_bytes
    )

    # A projector alone can close the deficit, and then nothing below scores the plan: the cost
    # gate is skipped, and with it the only refusal that keeps a host side out of swap.
    if mmproj_host_bytes:
        refused = _host_ram_refusal(opts, n_ctx, host_side, host_ram_bytes)
        if refused is not None:
            return refused

    # A plan can give something up without moving a weight, and those were never ranked:
    # the fallback arm of a no-spill plan is the launch as the caller typed it.
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
        )
        if declined is not None:
            return declined

    patterns = _patterns_for(layout, units)
    if spill_lm_head:
        patterns.append(LM_HEAD_PATTERN)
    indices = sorted({u.index for u in units})

    spilled_bytes = spilled_weight_bytes
    # token_embd is host-resident on every launch, so it is host RAM this plan has to be
    # able to pay for even when nothing is spilled; the -nkvo cache is in there too, and
    # a lazily-read PLE is not, until the load mode below asks for it.
    host_bytes = host_side
    vram_bytes = (
        all_resident_bytes(
            layout,
            n_ctx,
            kv_quantised = quantised,
            kv_bytes_floor = kv_bytes_floor,
            kv_on_host = kv_on_host,
            n_seq = knobs.n_parallel if knobs is not None else 1,
        )
        - spilled_bytes
    )

    # mmap costs 2 to 4.6x on host-resident weight reads, so turn it off -- but only
    # when host RAM holds the host side; otherwise mmap keeps an over-commit pageable.
    # "none" is the branch that has to pay for the PLE: this plan asks for no mapping, so the tensor it was excused
    # from above has to fit in RAM before that flag can be emitted.
    if host_ram_bytes is None or opts.prompt_cache_unbounded:
        load_mode_none = False
    else:
        load_mode_none = host_bytes + ple_lazy_bytes <= max(
            0, host_ram_bytes - opts.host_ram_headroom_bytes
        )
    if load_mode_none:
        host_bytes += ple_lazy_bytes

    # The prompt cache is host RAM llama-server takes on top of the spill, 8 GiB by default, and
    # the cheapest thing in the system to give up.
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
        # A resident fit found by the context ladder emits no pattern and no knob, and is
        # still a different launch: the planner proved a context the fallback had capped.
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
        # Matched pairs only: an unmatched K/V combination is not compiled
        # without GGML_CUDA_FA_ALL_QUANTS and silently falls back to CPU.
        cache_type_k = cache_type,
        cache_type_v = cache_type,
        spilled_blocks = tuple(indices),
        spilled_lm_head = spill_lm_head,
        vram_bytes = vram_bytes,
        host_bytes = host_bytes,
        ple_charged_to_host = not ple_lazy_bytes or load_mode_none,
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
    return args


_SMART_OFFLOAD_ON = ("1", "true", "yes", "on", "enabled")


def smart_offload_enabled(env: Optional[Mapping[str, str]] = None) -> bool:
    """Whether the launch path may plan a spill. OFF unless explicitly enabled.

    This was briefly opt-OUT, on 118 paired runs across T4, L4, RTX PRO 6000,
    A100, B200 and a gfx1151 APU. Every one of those hosts is a large one, and
    that turned out to be the whole of the calibration set: #9861 measured 76
    paired cells on a 6-core desktop and the planner was slower in 40 of the 43
    it planned, by up to 8x on generation.

    The mechanism is not the host size alone. ``rank`` in offload_cost_model
    scores a placement as prefill PLUS generation, but the planner only ever
    calls ``generation_penalty_ms``, so prefill is not priced at all -- which is
    why #9861 measured prefill slower in 43 of 43 planned cells, without one
    exception. A gate that does not count half the request cannot be trusted to
    fire by default, so it goes back behind the flag until it does.

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
