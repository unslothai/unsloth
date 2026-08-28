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
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Mapping, Optional, Sequence

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
    """Which blocks to spill when only some are needed.

    UNMEASURED: every -ot measurement so far spilled all blocks or none, so the
    ordering is justified by byte-minimality alone, not by benchmark. Contiguous
    runs may schedule better (adjacent host blocks can merge into one graph
    split), which would favour FRONT/BACK over LARGEST. Hence configurable.
    """

    # Best-fit-decreasing: fewest blocks AND least overshoot. Overshoot is real
    # bandwidth -- a 209 MiB block for a 50 MiB deficit wastes 159 MiB per token.
    LARGEST_FIRST = "largest_first"
    FRONT_FIRST = "front_first"
    BACK_FIRST = "back_first"


class FfnGranularity(Enum):
    """How small a piece of one block's FFN the planner may move.

    See ``PlanOptions.ffn_granularity`` for the measurements that put the
    default at BOUNDARY rather than at either extreme.
    """

    # Whole FFN or nothing, per block. What the planner did before the ladder.
    WHOLE = "whole"
    # Whole FFNs, except the last block taken, which is graded. One extra graph
    # split bought at most one sub-FFN matrix of overshoot.
    BOUNDARY = "boundary"
    # Every block graded. Lowest overshoot, and measured 5 to 12% SLOWER than
    # WHOLE because of the split count. Kept so the comparison is reproducible.
    ALL = "all"


@dataclass(frozen = True)
class PlanOptions:
    # Compute buffer + CUDA context + scratch, charged on every device.
    #
    # 1 GiB was too thin and failed CONSISTENTLY: the planner fills to
    # ``budget - overhead_bytes_per_device``, leaving exactly this much free
    # whatever the budget is, so the dense 27B at depth 32768 died identically at
    # 6, 7, 8 and 10 GiB with
    #
    #   ggml_backend_cuda_buffer_type_alloc_buffer: allocating 594.16 MiB
    #   on device 0: cudaMalloc failed: out of memory
    #
    # The child needs the PREFILL compute buffer (594 MiB measured) plus its own
    # CUDA primary context, which took the rest of the old 1 GiB. Not benchmark
    # fragmentation: 16, 64 and 1024 MiB hog blocks all reproduced the identical
    # 594.16 MiB failure. 1.5 GiB covers the measured 1.07 GiB with margin -- a
    # measured floor, not a fitted curve, since the steady-state compute buffer is
    # flat in context (493 to 509 MiB from depth 4096 to 32768) but the prefill
    # graph's reservation is not. Erring high costs some spill (linear at
    # 5.544 ms/GiB), erring low costs the whole load.
    overhead_bytes_per_device: int = (3 * GIB) // 2
    # GPU-resident bytes NOT in the layout (a vision projector, an MTP draft
    # reserve), charged once against the pooled budget: the layout only knows the
    # target GGUF's tensor table. Subtracting from the budget also reaches
    # max_context_for. 0 keeps the pure-layout behaviour.
    extra_resident_bytes: int = 0
    # The fixed per-device cost of a LAYER SPLIT, charged once for every device
    # after the first. Separate from overhead_bytes_per_device because it is not
    # per-device in the same sense: the first device's share is already folded
    # into the compute buffer above, which is why every other site in
    # llama_cpp.py applies it as ``max(0, n_gpus - 1) * ...`` and skips it
    # entirely at k=1. Folded into the flat per-device term instead, it withheld
    # a GiB of a single card that nothing was ever going to allocate, which is
    # deficit the planner then spilled real blocks to cover.
    pipeline_overhead_bytes: int = 0
    # Host RAM this planner refuses to spend, so a spill does not push the box
    # into swap.
    host_ram_headroom_bytes: int = 2 * GIB
    context_policy: ContextPolicy = ContextPolicy.NEVER_REDUCE
    min_ctx: int = 4096
    spill_order: SpillOrder = SpillOrder.LARGEST_FIRST
    # How finely a block's FFN may be broken up. MEASURED, and the answer was
    # not the obvious one.
    #
    # The appeal of splitting every block's FFN into its three matrices is that
    # the planner stops overshooting the deficit by up to two thirds of a block.
    # Overshoot is real -- every byte past the deficit is re-read from the host
    # on every token that touches it -- and it is the shape of #9861's two worst
    # cells, an 8B with 11008 MiB free where 22 and 29 of 36 whole FFNs moved for
    # a much smaller deficit and landed at 0.19x and 0.21x of ``--fit on``.
    #
    # But a partly-spilled block costs a GRAPH SPLIT that a wholly-spilled one
    # does not: with ffn_down on the host and ffn_up/ffn_gate on the device, the
    # decode graph crosses the backend boundary twice per block instead of once,
    # and every crossing is a synchronisation. Measured on Qwen3.6-35B-A3B Q4,
    # 4 concurrent slots, ggml's own ``graph splits ... (with bs=1)``:
    #
    #     forced VRAM   whole-FFN        every-block ladder
    #     20 GiB        4 blocks,  10    11 blocks,  24   -> 0.88x generation
    #     18 GiB        9 blocks,  20    22 blocks,  46   -> 0.95x
    #     16 GiB       13 blocks,  28    34 blocks,  70   -> 0.92x
    #     14 GiB       17 blocks,  36    40 blocks,  82   -> 0.93x
    #
    # It cost 5 to 12% of generation to save 0.02 to 0.36 GiB of overshoot. And
    # it is not the ORDER of blocks: taking a trailing contiguous run instead of
    # the largest-first scatter produced the identical split count and the same
    # throughput, which rules out the contiguity hypothesis ``SpillOrder``
    # raises.
    #
    # So the default is BOUNDARY: whole FFNs for every block but the last one
    # taken, and the graded rungs only on that one. Overshoot then falls to at
    # most one sub-FFN matrix while exactly ONE extra split is paid. This is also
    # what llama.cpp does -- ``fit.cpp:490`` applies its graded fraction to the
    # boundary layer ``il0`` and ``LAYER_FRACTION_MOE`` to every layer past it --
    # which reads much less like an implementation detail once the split cost is
    # on the table.
    ffn_granularity: FfnGranularity = FfnGranularity.BOUNDARY
    # Which matrix goes first. llama.cpp's order (common/fit.cpp:407-440), and
    # UNMEASURED by us -- see :class:`SpillClass`. Exposed so the benchmark can
    # try the permutations rather than inheriting an assumption forever.
    ffn_rung_order: tuple[SpillClass, ...] = FFN_SPILL_CLASSES
    allow_lm_head_spill: bool = True
    # The two rungs below lm_head, and both are OFF.
    #
    # Not because they cannot be expressed -- ``-ot`` moves an attention
    # projection as happily as an expert, and ``-nkvo`` moves the cache -- but
    # because the module's own measurements put them off the bottom of the scale:
    #
    #     nothing spilled                       75.37 t/s
    #     FFN to host                           13.63 t/s
    #     FFN + lm_head                         11.39 t/s
    #     -ngl / --no-kv-offload                ~1.03 t/s
    #
    # A load that needs them is a load ``--fit on`` should place, and falling
    # through to ``--fit on`` is what happens when no rung fits. Spilling
    # attention weights is worse than the table suggests, too: the cache stays on
    # the device while the attention op moves to the CPU backend, so every token
    # drags the live cache back across the link -- the 20.1x regime
    # ``Access.KV_CACHE`` was calibrated on, now paid in the other direction.
    #
    # They exist because the ladder should be COMPLETE and the benchmark should
    # be able to reach the bottom of it, not because anything should turn them on.
    allow_attention_spill: bool = False
    allow_kv_host_fallback: bool = False
    # Let a PARTIAL spill through the multi-device check instead of abstaining.
    #
    # The abstain predates the ladder and was cheap then: the whole-FFN planner
    # usually took every block anyway, so "full spill only" cost little. The
    # ladder inverts that -- covering a deficit with one rung of a few blocks is
    # the entire point -- so on more than one device it now abstains almost
    # always, and a multi-GPU user gets ``--fit on`` whatever the planner could
    # have done.
    #
    # Lifting it rests on one unvalidated thing: ``_device_slots`` reproduces
    # llama.cpp's row assignment (free memory per device at
    # llama-model.cpp:1425-1433, prefix-summed at :1439-1447, upper_bound on the
    # normalised layer index at :1457). If that model is right then
    # ``_per_device_shortfall`` already does the per-device arithmetic correctly
    # for a partial spill, because it subtracts MOVED BYTES per row. If it is
    # wrong, a card is over and llama-model.cpp:1731 throws.
    #
    # Nobody has checked it against a real split, so it stays off and the
    # benchmark turns it on: a 2xT4 Kaggle kernel loading the resulting placement
    # is the evidence that would justify making it the default.
    trust_device_row_model: bool = False
    # What the host brings to bear on spilled weights. Spilled generation runs on
    # the CPU backend -- ggml only moves an op to the GPU at batch >= 32
    # (ggml-cuda.cu, op_offload_min_batch_size) and decode is batch 1 -- so the
    # penalty scales with core count: 2.42 / 5.83 / 11.82 / 14.94 t/s at
    # 4 / 16 / 64 / 192 threads.
    host: HostProfile = field(default_factory = HostProfile)
    # q8_0 measured 35% slower generation, and without GGML_CUDA_FA_ALL_QUANTS
    # only four MATCHED K/V combinations are compiled (a mismatched pair falls
    # to CPU and stalls). Off by default; matched pairs only when enabled.
    allow_kv_quant: bool = False
    kv_quant_type: str = "q8_0"
    # The caller passed -nkvo (or a false LLAMA_ARG_KV_OFFLOAD), so llama.cpp puts
    # the WHOLE cache on the host: offload is one scalar and the buffer type falls
    # back to the CPU one for every layer (llama-kv-cache.cpp:210-219), same branch
    # in the recurrent and DSV4 caches. The cache and the recurrent state move out
    # of the VRAM footprint and into the host one; charging them to VRAM anyway
    # would spill FFN blocks for a deficit the child never has.
    kv_on_host: bool = False
    # The request shape the plan is SCORED at. ``rank`` refuses to bake one in,
    # for good reason: a long prompt with a short reply and a short prompt with a
    # long reply rank placements differently, and the planner spent its whole
    # life scoring only the second. Prefill-weighted by default because that is
    # the shape of a chat turn carrying any context at all, and the shape #9861
    # measured (a ~2.3K token prompt, 128 generated).
    workload_prompt_tokens: int = 2048
    workload_generated_tokens: int = 256
    n_ubatch: int = 512
    # How much of llama.cpp's own predicted penalty a plan must remove before it
    # is worth deviating from ``--fit on`` at all. Not 0: the two outcomes are
    # not symmetric. #9861 measured all 33 cells where the planner declined at
    # 0.93x to 1.16x -- abstaining is nearly free -- while planning a cell it
    # should not have cost up to 8x. A near-tie is therefore not worth taking,
    # and this is the margin that says so.
    min_penalty_reduction: float = 0.10
    # Whether that comparison may VETO a plan. Default off, so this module keeps
    # answering the question it always answered -- "what placement covers the
    # deficit" -- and every existing caller and test still gets that answer. The
    # launch seam in llama_cpp.py turns it on, because it is the only caller that
    # has an alternative: when the planner declines, it emits ``--fit on``. A
    # caller asking the planner what it CAN do is not asking whether it should.
    require_cost_win: bool = False


@dataclass(frozen = True)
class Plan:
    """What to launch with, and why."""

    # False means "emit nothing new": either the planner abstained or the load
    # needs no help. Always safe, since llama.cpp's own defaults then apply.
    changed: bool = False
    n_ctx: int = 0
    ot_patterns: tuple[str, ...] = field(default_factory = tuple)
    load_mode_none: bool = False
    cache_type_k: Optional[str] = None
    cache_type_v: Optional[str] = None
    spilled_blocks: tuple[int, ...] = field(default_factory = tuple)
    spilled_lm_head: bool = False
    # The bottom rung fired and the plan is asking for -nkvo. Only reachable with
    # ``allow_kv_host_fallback``, which is off: see PlanOptions for why.
    kv_spilled_to_host: bool = False
    # No rung fits. mmap has to stay, because it is the only thing that makes an
    # over-commit pageable rather than OOM-killed.
    insufficient: bool = False
    vram_bytes: int = 0
    host_bytes: int = 0
    # Predicted extra ms per generated token versus fully resident, on the host
    # this was planned for. 0.0 when nothing is spilled. Reported so callers can
    # surface the real cost instead of implying a spill is free.
    predicted_gen_penalty_ms: float = 0.0
    # The two sides of the cost gate, in ms for a whole request of
    # ``PlanOptions``'s workload shape: this placement, and what llama.cpp's own
    # fitter would have cost instead. Both 0.0 when the gate did not run (no
    # spill to weigh, or a caller who is not choosing). Reported rather than
    # only logged, because "the planner declined" is not an answer anyone can
    # check without the numbers it declined on.
    predicted_request_ms: float = 0.0
    predicted_fit_request_ms: float = 0.0
    reason: str = ""

    @property
    def spills_anything(self) -> bool:
        return bool(self.spilled_blocks) or self.spilled_lm_head


def _usable_vram(vram_bytes_per_device: Sequence[int], opts: PlanOptions) -> int:
    """Total creditable VRAM: every device pays the fixed per-device overhead,
    the split pays for each device AFTER the first, then the pool pays once for
    whatever sits on a card outside the layout."""
    pooled = sum(max(0, v - opts.overhead_bytes_per_device) for v in vram_bytes_per_device)
    split = max(0, len(vram_bytes_per_device) - 1) * max(0, opts.pipeline_overhead_bytes)
    return pooled - split - max(0, opts.extra_resident_bytes)


@dataclass(frozen = True)
class SpillUnit:
    """One block's worth of one rung: the smallest thing the ladder can move.

    ``cls`` of ``None`` means the whole of that block's spillable FFN, which is
    the only unit the planner had before the ladder existed and is still what it
    uses for any layout whose per-rung breakdown does not add up (see
    :meth:`BlockLayout.graded`). Keeping the coarse unit expressible is what lets
    every pre-ladder caller and test get bit-identical answers.
    """

    index: int
    cls: Optional[SpillClass]
    nbytes: int


def _units_of(blocks: Sequence[BlockLayout], cls: Optional[SpillClass]) -> list[SpillUnit]:
    if cls is None:
        return [SpillUnit(b.index, None, b.spillable_bytes) for b in blocks if b.spillable_bytes > 0]
    return [SpillUnit(b.index, cls, b.class_bytes(cls)) for b in blocks if b.class_bytes(cls) > 0]


def _select_units(
    units: Sequence[SpillUnit], deficit: int, order: SpillOrder
) -> tuple[list[SpillUnit], int]:
    """The MINIMAL set of ``units`` freeing at least ``deficit``, and what it frees.

    "Minimal" in the two senses that matter and in this priority: fewest units,
    then least overshoot. Overshoot is not free -- every byte moved beyond the
    deficit is a byte read back across the host link on every token that touches
    it -- which is the whole reason the ladder was split finer than a block.
    """
    remaining = list(units)
    if order is SpillOrder.FRONT_FIRST:
        remaining.sort(key = lambda u: u.index)
    elif order is SpillOrder.BACK_FIRST:
        remaining.sort(key = lambda u: -u.index)
    else:
        remaining.sort(key = lambda u: -u.nbytes)

    chosen: list[SpillUnit] = []
    freed = 0
    while freed < deficit and remaining:
        if order is SpillOrder.LARGEST_FIRST:
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


def _grade_the_boundary_block(
    layout: ModelLayout,
    opts: PlanOptions,
    taken: list[SpillUnit],
    freed: int,
    deficit: int,
) -> tuple[list[SpillUnit], int]:
    """Trim the LAST whole block taken down to the rungs actually needed.

    A whole-FFN selection stops at the first block that covers the deficit, so
    the last one is almost always mostly overshoot -- on average half a block,
    and up to a whole one. This gives that block, and only that block, the
    graded treatment: keep the cheapest rungs of it until the gap closes, and
    leave the rest resident.

    Cost is exactly one extra graph split, because exactly one block ends up
    with part of its FFN on each side. Spending that split on every block
    instead measured 5 to 12% slower (see ``PlanOptions.ffn_granularity``), and
    spending it on none leaves up to a whole block of overshoot on the table.

    A no-op when the layout cannot be graded, when the mode is not BOUNDARY, or
    when dropping the block's smallest rung would reopen the deficit -- in which
    case the whole block was needed and there is nothing to trim.
    """
    if opts.ffn_granularity is not FfnGranularity.BOUNDARY:
        return taken, freed
    coarse = [u for u in taken if u.cls is None]
    if not coarse:
        return taken, freed
    by_index = {b.index: b for b in layout.blocks}
    # The last coarse unit taken IS the boundary: selection appends in the order
    # it closed the gap, so everything before it was still short of the deficit.
    boundary = coarse[-1]
    block = by_index.get(boundary.index)
    if block is None or not block.graded:
        return taken, freed

    without = freed - boundary.nbytes
    rungs = [cls for cls in opts.ffn_rung_order if cls in FFN_SPILL_CLASSES]
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
        # Either the graded rungs cannot cover what the whole block covered, or
        # they add up to the whole block anyway. Keep the coarse unit: it says
        # the same thing in one pattern and costs one split fewer.
        return taken, freed
    return [u for u in taken if u is not boundary] + kept, running


def _moved_by_index(units: Sequence[SpillUnit]) -> dict[int, int]:
    """Bytes this spill takes off each block. The per-device check needs BYTES,
    not a set of indices: a rung can move part of a block, and crediting the
    whole block would leave a card over on a load the pool says fits."""
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
    """"the ffn_down of every block plus the ffn_up/gate_up of 5 of 40 blocks".

    Spelled out per rung because "spilled 40 blocks" is the sentence that made
    #9861 hard to read: it says nothing about how much of each block moved, which
    after the ladder is most of what the decision is.
    """
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


def _rung_classes(layout: ModelLayout, opts: PlanOptions) -> tuple[Optional[SpillClass], ...]:
    """The per-block rungs, cheapest first, for this layout and these options.

    Falls back to the single coarse whole-FFN rung when the layout cannot be
    graded, so an architecture whose tensor names this module does not recognise
    degrades to the pre-ladder behaviour instead of to spilling nothing.
    """
    spillable = [b for b in layout.blocks if b.spillable_bytes > 0]
    graded = bool(spillable) and all(b.graded for b in spillable)
    if opts.ffn_granularity is FfnGranularity.ALL and graded:
        rungs: list[Optional[SpillClass]] = [
            cls for cls in opts.ffn_rung_order if cls in FFN_SPILL_CLASSES
        ]
    else:
        # WHOLE, BOUNDARY, and any ungraded layout all walk the coarse rung here.
        # BOUNDARY's graded piece is not a rung at all -- it applies to the ONE
        # block that closes the gap, and is handled in _plan_at where that block
        # is known.
        rungs = [None]
    # Shared experts and dense FFN inside a MoE model. On a dense model this
    # class is empty by construction, so the rung is a no-op rather than a
    # duplicate of the three above.
    if any(b.dense_ffn_bytes for b in layout.blocks):
        rungs.append(SpillClass.DENSE_FFN)
    return tuple(rungs)


def _ffn_group(layout: ModelLayout, spilled: int) -> TensorGroup:
    """The spillable FFN bytes as one group, charged the way they are read.

    MoE experts are charged their ROUTED fraction, since only ``n_expert_used``
    of ``n_expert`` are touched per generated token, which is why MoE tolerates
    spilling far better than a fully activated dense FFN.
    """
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
    """This plan's spill, as something the cost model can price.

    ``-ot`` moves the named weights and nothing else, so the attention cache
    stays on the device unless a rung explicitly moved it: ``kv_host_bytes`` is 0
    for every rung above the last. That is the whole of the planner's claimed
    advantage over a layer fitter, and pricing it is how
    :func:`_fit_fallback_placement` gets to argue back.

    Rungs are grouped by how they are READ, not by which rung they came from.
    The three expert matrices are one routed group however many of them moved:
    ``MULTI_GROUP_CONTENTION`` would otherwise charge a contention penalty for
    splitting one tensor set across three names, which is an artefact of the
    ladder's bookkeeping rather than anything the host does.
    """
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
    """Predicted extra ms per generated token for this spill, on this host.

    Spilled weights are read by the CPU backend, not streamed to the GPU: ggml
    only migrates an op at batch >= 32 and decode is batch 1, so the cost tracks
    host cores.
    """
    placement = _spill_placement(layout, units, spill_lm_head, kv_host_bytes)
    if not placement.host_groups and not placement.kv_host_bytes:
        return 0.0
    return generation_penalty_ms(placement, host)


def _fit_fallback_placement(
    layout: ModelLayout,
    opts: PlanOptions,
    budget: int,
    n_ctx: int,
    *,
    quantised: bool,
    kv_bytes_floor: int,
    kv_on_host: bool,
) -> Optional[Placement]:
    """What llama.cpp's own fitter would place here, priced the same way.

    This is the arm the planner is really competing against. When the planner
    abstains the launch path emits ``--fit on`` and no ``-ngl``, and fitting
    lowers the offloaded layer count until the load fits -- moving WHOLE layers,
    attention and FFN together, and dragging each moved layer's share of the
    attention cache to the host with it (``-ngl`` drags the cache off, ``-ot``
    does not). The output tensor rides the layer list at ``n_layer_all``, so it
    is the first thing to leave once the count drops below the layer count.

    Modelling it matters because the fallback is CHEAPER than a spill whenever
    the model nearly fitted anyway: few layers move, the cache mostly stays, and
    the planner's own placement has bought very little for its host round trip.
    #9861's two worst cells are exactly that shape -- an 8B with 11 GiB free,
    where the planner spilled 22 and 29 of 36 blocks and landed at 0.19x and
    0.21x against a fitter that had barely anything to move.

    ``None`` when even moving every layer does not fit, i.e. there is no viable
    fallback to lose to and the planner should say whatever it has to say.
    """
    blocks = list(layout.blocks)
    if not blocks:
        return None
    resident = all_resident_bytes(
        layout,
        n_ctx,
        kv_quantised = quantised,
        kv_bytes_floor = kv_bytes_floor,
        kv_on_host = kv_on_host,
    )
    # The cache follows the layer, so a layer moved to host takes its share with
    # it. Per-layer rather than per-attention-layer: SWA already makes the
    # planner abstain upstream, so the shares are uniform by the time we get here.
    kv_total = (
        0
        if kv_on_host
        else cache_bytes(layout, n_ctx, kv_quantised = quantised, kv_bytes_floor = kv_bytes_floor)
    )
    kv_per_layer = kv_total / len(blocks)

    # The cache is RESERVED at n_ctx but only the live prefix is ever read, and
    # reading is what costs. Pricing the reservation charges a 32K allocation for
    # a 2K conversation, which is the single biggest thumb on the scale in favour
    # of spilling: it makes any placement that moves cache look catastrophic and
    # the planner's own -ot placement look free by comparison. The reporter on
    # #9861 flagged the same gap from the measurement side -- 32768 allocated,
    # about 2.2K ever live. Feasibility above still uses the full reservation,
    # because llama.cpp really does allocate it.
    live_tokens = min(n_ctx, max(1, opts.workload_prompt_tokens + opts.workload_generated_tokens))
    kv_live_per_layer = (
        cache_bytes(layout, live_tokens, kv_quantised = quantised) / len(blocks)
        if not kv_on_host
        else 0.0
    )

    if layout.is_moe:
        # MEASURED, not assumed. On an MoE model ``--fit on`` keeps EVERY layer
        # on the device (n_layer=41/41 on both a 12 GiB L4 and a 16 GiB A100) and
        # moves only the trailing layers' expert tensors, through the same kind
        # of tensor override the planner emits:
        #
        #   blk.<il>.ffn_(up|down|gate_up|gate)_(ch|)exps   (fit.cpp:434-440)
        #
        # so the cache stays resident and no attention weight moves. That is the
        # planner's own strategy, which is why declining costs so little on this
        # architecture: both arms measured 33.65 against 34.77 t/s on generation,
        # a 1.03x tie.
        #
        # It moves close to the MINIMUM it needs, same as the planner. Measured
        # from the fitter's own trace: n_part of 14, 23 and 31 partial layers on
        # a 16 GiB A100, a 12 GiB L4 and an 8 GiB T4, against 13, 22 and 31
        # blocks for the planner on the same cards. Near-identical placements.
        #
        # An earlier revision of this claimed the fitter moved EVERY expert,
        # from a reading of CPU_Mapped in the benchmark. That was wrong:
        # CPU_Mapped reports host-resident bytes only when mmap is off, and the
        # fitter arm keeps mmap, so the figure was the size of the mapped FILE.
        # The tell was that it came back byte-identical (20763.72 MiB) on three
        # different cards -- a file size, not a decision.
        host_experts = 0
        for block in reversed(blocks):
            host_experts += block.spillable_bytes
            if resident - host_experts <= budget:
                return Placement(host_groups = [_ffn_group(layout, host_experts)])
        return None

    # Dense, where the whole-layer model IS what happens: measured n_part=0 with
    # n_layer 54 of 65 and 38 of 65, no overrides at all, and the cache off the
    # GPU with it. llama.cpp keeps the LAST n_gpu_layers on the device, so the
    # host takes the leading ones; walking from the end is equivalent here
    # because only the count enters the cost.
    host_weights = 0
    host_spillable = 0
    for moved, block in enumerate(reversed(blocks), start = 1):
        host_weights += block.spillable_bytes + block.resident_bytes
        host_spillable += block.spillable_bytes
        freed = host_weights + int(kv_per_layer * moved) + layout.lm_head_bytes
        if resident - freed <= budget:
            groups: list[TensorGroup] = []
            if host_spillable:
                groups.append(_ffn_group(layout, host_spillable))
            attention = host_weights - host_spillable
            if attention:
                # Attention, norms and routers: dense, read in full every token.
                groups.append(TensorGroup("layers", attention, Access.CONTIGUOUS))
            if layout.lm_head_bytes:
                groups.append(TensorGroup("lm_head", layout.lm_head_bytes, Access.SINGLE_MATVEC))
            live_kv = int(kv_live_per_layer * moved)
            if live_kv:
                # NOT ``kv_host_bytes``. That field carries the 20.1x rate, which
                # was calibrated on ``--no-kv-offload``: cache on the host while
                # attention still runs on the GPU, so every token drags the whole
                # thing back across the link. A layer the fitter moved is not in
                # that regime -- its attention runs on the CPU backend, next to
                # its own cache -- so it reads at host speed like any other host
                # tensor. Charging it 20.1x instead made one moved layer score
                # worse than two gigabytes of moved weights, and no plan could
                # ever lose to the fitter.
                groups.append(TensorGroup("kv (moved layers)", live_kv, Access.CONTIGUOUS))
            return Placement(host_groups = groups)
    return None


def _kv_elem_bytes(quantised: bool) -> int:
    return 1 if quantised else 2


def cache_bytes(
    layout: ModelLayout,
    n_ctx: int,
    *,
    kv_quantised: bool = False,
    kv_bytes_floor: int = 0,
) -> int:
    """Attention cache to reserve, never below a caller-supplied measurement.

    ``layout.kv_bytes`` is a plain f16 GQA product: heads times key+value width
    times context. It has no cache-dtype, SWA, MLA, unified-stream, slot-padding
    or flash-attention-padding term, so against a caller that has priced the real
    cache it can land either side. Over is harmless -- the plan just reserves
    more. UNDER is the dangerous direction: the deficit comes out too small, too
    few blocks are spilled, and the launch path follows that with ``--fit off``,
    so the server OOMs on a cache the caller had already sized correctly. MLA is
    the worst case (a compressed K-only latent that this product models as a full
    K+V pair), and it is exactly the huge-MoE shape this planner exists for.

    Taking the maximum keeps the planner conservative in both directions without
    a tolerance to tune. The floor is a measurement at the REQUESTED context, so
    where a shrink rung re-prices at a smaller context it over-reserves; that is
    the safe direction and at worst gives up a rung.
    """
    naive = layout.kv_bytes(n_ctx, _kv_elem_bytes(kv_quantised))
    floor = max(0, kv_bytes_floor)
    if layout.has_swa and floor:
        # Sliding-window attention breaks the product in the UP direction, badly
        # and by construction: most layers keep a window-sized cache rather than
        # a full-context one, and on gemma4 they also use narrower heads
        # (key_length_swa 256 against key_length 512) and share one cache across
        # 20 layers. The product models none of that.
        #
        # MEASURED on gemma-4-E2B-it UD-Q4_K_XL at n_ctx 9216: the product says
        # 0.615 GiB, llama.cpp allocated 48 MiB. Taking the max let a 13x
        # over-estimate override a real measurement, which is deficit the planner
        # then spills real blocks to cover.
        #
        # So a supplied measurement wins here. It is a measurement of the cache
        # the caller is about to allocate, which is strictly better evidence than
        # a product with no SWA term -- and the max is kept everywhere else,
        # where the product's failure mode is UNDER-counting (MLA) and the
        # measurement is the thing that might be short.
        return floor
    return max(naive, floor)


def resident_floor_bytes(
    layout: ModelLayout,
    n_ctx: int,
    *,
    kv_quantised: bool = False,
    kv_bytes_floor: int = 0,
    kv_on_host: bool = False,
) -> int:
    """VRAM needed with EVERY spillable tensor already on the host.

    Attention weights, norms, routers, shared experts, the recurrent state, the
    cache and lm_head. Below this, ``-ot`` has nothing left to give and only a
    smaller quant or less context can help.
    """
    if kv_on_host:
        # Both caches follow the same scalar, so neither is VRAM here.
        return layout.block_resident_bytes + layout.lm_head_bytes + layout.other_resident_bytes
    return (
        layout.block_resident_bytes
        + layout.lm_head_bytes
        + layout.other_resident_bytes
        + layout.recurrent_bytes
        + cache_bytes(layout, n_ctx, kv_quantised = kv_quantised, kv_bytes_floor = kv_bytes_floor)
    )


def all_resident_bytes(
    layout: ModelLayout,
    n_ctx: int,
    *,
    kv_quantised: bool = False,
    kv_bytes_floor: int = 0,
    kv_on_host: bool = False,
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
) -> int:
    """Largest context whose cache fits, rounded down to 256 as CUDA wants."""
    opts = opts or PlanOptions()
    if not layout.complete or layout.kv_bytes_per_token_f16 <= 0:
        return 0
    fixed = (
        layout.block_resident_bytes
        + layout.other_resident_bytes
        + layout.recurrent_bytes
        + (0 if spill_lm_head else layout.lm_head_bytes)
        + (0 if spill_all_ffn else layout.spillable_bytes)
    )
    free = _usable_vram(vram_bytes_per_device, opts) - fixed
    if free <= 0:
        return 0
    per_token = layout.kv_bytes_per_token_f16 * _kv_elem_bytes(kv_quantised) // 2
    if per_token <= 0:
        return 0
    ctx = (free // per_token) // 256 * 256
    if layout.n_ctx_train:
        ctx = min(ctx, layout.n_ctx_train)
    return max(0, ctx)


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
        # One pool: "spilling" renames bytes on the same chips and frees nothing.
        # Metal also keeps mmap zero copy (buffer_from_host_ptr), so the no-mmap
        # rule inverts there too.
        return Plan(reason = "unified memory host, spilling frees no device memory")
    budget = _usable_vram(vram_bytes_per_device, opts)
    if budget <= 0:
        return Plan(reason = "no creditable VRAM after per-device overhead and reserved allocations")

    n_ctx = requested_ctx if requested_ctx > 0 else layout.n_ctx_train
    if layout.n_ctx_train:
        n_ctx = min(n_ctx, layout.n_ctx_train)
    if n_ctx <= 0:
        return Plan(reason = "no usable context length")
    if layout.has_swa and kv_bytes_floor <= 0:
        # Sliding-window attention, and nobody measured the cache. The layout's
        # own product has no SWA term -- it charges every attention layer the
        # full context at the full head width -- so it over-counts by whatever
        # the window and the narrower SWA heads save, and the planner would spill
        # real blocks to cover a deficit that is mostly arithmetic.
        #
        # MEASURED on gemma-4-E2B-it UD-Q4_K_XL, 4.15 GiB free: the product says
        # a 0.615 GiB cache, llama.cpp allocated 48 MiB. Together with a second
        # over-count since fixed, that invented a 0.85 GiB deficit on a model
        # that FIT, the planner spilled the FFN of 32 of 35 blocks, and
        # generation fell from 447.8 to 187.7 t/s -- 0.42x, against a ``--fit on``
        # that had nothing to do and measured the same as on a card half again
        # as large.
        #
        # The module already abstains on SWA for a multi-device split, on the
        # grounds that it does not know WHICH layers hold the full-context cache.
        # The same ignorance makes the TOTAL wrong, so it abstains here too. A
        # caller that has priced the real cache passes kv_bytes_floor and this
        # does not fire; the launch seam in llama_cpp.py does exactly that.
        return Plan(
            n_ctx = n_ctx,
            reason = (
                "sliding-window attention and no measured cache size: the layout's "
                "cache estimate charges every layer the full context and would "
                "invent a deficit, so this is left to llama.cpp's own fitter"
            ),
        )

    # PREFER_RESIDENT gets its say before the ladder: a smaller fully resident
    # context outruns a larger spilled one, when the caller allows it to move.
    if (
        opts.context_policy is ContextPolicy.PREFER_RESIDENT
        and all_resident_bytes(
            layout, n_ctx, kv_bytes_floor = kv_bytes_floor, kv_on_host = opts.kv_on_host
        )
        > budget
    ):
        shrunk = max_context_for(layout, vram_bytes_per_device, opts = opts)
        if shrunk >= opts.min_ctx:
            return _finish(
                layout,
                opts,
                min(shrunk, n_ctx),
                [],
                False,
                host_ram_bytes,
                reason = (
                    f"shrank context {n_ctx} -> {min(shrunk, n_ctx)} to keep every tensor "
                    "resident, which outruns a larger spilled context"
                ),
            )

    for quantised in _kv_modes(opts):
        plan = _plan_at(
            layout,
            opts,
            n_ctx,
            budget,
            host_ram_bytes,
            quantised,
            kv_bytes_floor,
            vram_bytes_per_device,
            split_weights_per_device or vram_bytes_per_device,
            kv_layer_weights,
        )
        if plan is not None:
            return plan

    # Nothing fit at the requested context. Only now may FIT_ONLY shrink it.
    if opts.context_policy in (ContextPolicy.FIT_ONLY, ContextPolicy.PREFER_RESIDENT):
        for quantised in _kv_modes(opts):
            shrunk = max_context_for(
                layout,
                vram_bytes_per_device,
                spill_all_ffn = True,
                spill_lm_head = opts.allow_lm_head_spill,
                kv_quantised = quantised,
                opts = opts,
            )
            shrunk = min(shrunk, n_ctx)
            if shrunk >= opts.min_ctx:
                plan = _plan_at(
                    layout,
                    opts,
                    shrunk,
                    budget,
                    host_ram_bytes,
                    quantised,
                    kv_bytes_floor,
                    vram_bytes_per_device,
                    split_weights_per_device or vram_bytes_per_device,
                    kv_layer_weights,
                )
                if plan is not None:
                    return plan

    floor = resident_floor_bytes(
        layout, n_ctx, kv_bytes_floor = kv_bytes_floor, kv_on_host = opts.kv_on_host
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
    """f16 first, then q8_0 only if the caller opted in."""
    return (False, True) if opts.allow_kv_quant else (False,)


def _device_slots(n_slots: int, split_weights: Sequence[int]) -> list[list[int]]:
    """Which of the ``n_slots`` layer rows land on which device.

    Mirrors llama.cpp's default tensor split exactly: free VRAM per device
    (llama-model.cpp:1420-1433), prefix-summed and normalised (:1439-1447), then
    ``upper_bound`` on the normalised row index (:1457). Row ``n_layer_all`` is
    the output row (:1467). With every layer offloaded ``i_gpu_start`` is 0 and
    ``act_gpu_layers`` is ``n_layer_all + 1``, which is ``n_slots`` here.
    """
    weights = [max(0, v) for v in split_weights]
    total = sum(weights)
    if total <= 0:
        return [list(range(n_slots))] + [[] for _ in weights[1:]]
    cumulative: list[float] = []
    running = 0.0
    for w in weights:
        running += w
        cumulative.append(running / total)
    slots: list[list[int]] = [[] for _ in weights]
    for row in range(n_slots):
        fraction = row / n_slots
        # std::upper_bound: first cumulative strictly greater than fraction.
        device = next((i for i, c in enumerate(cumulative) if c > fraction), len(weights) - 1)
        slots[device].append(row)
    return slots


def _per_device_shortfall(
    layout: ModelLayout,
    opts: PlanOptions,
    n_ctx: int,
    spilled_bytes_by_index: Mapping[int, int],
    spill_lm_head: bool,
    vram_bytes_per_device: Sequence[int],
    *,
    quantised: bool,
    kv_bytes_floor: int,
    split_weights_per_device: Sequence[int] = (),
    kv_layer_weights: Sequence[int] = (),
) -> Optional[str]:
    """``None`` when every device provably fits, else why it cannot be shown to.

    A pooled budget is not a per-device fit test, and it does not become one just
    because every spillable block was taken. llama.cpp hands out CONTIGUOUS ROW
    RANGES sized by free memory, so a device's share of the ROWS is proportional
    to its free VRAM while its share of the BYTES is not: what stays resident
    differs row by row (a block with a shared expert keeps more than a plain
    dense one), and the budget subtracts a FIXED per-device overhead, which
    already breaks proportionality on mixed cards -- 24 GiB and 8 GiB split the
    rows 75/25 but the budgets 77.6/22.4, so the small card is over on a load the
    pool says fits. A per-device shortfall is a hard throw (llama-model.cpp:1731)
    and ``--fit off`` means common/fit.cpp never runs to catch it.
    """
    if len(vram_bytes_per_device) <= 1:
        return None
    # These three shapes -- recurrent hybrid, n_attention_layers short of
    # n_layers, sliding window -- are only a problem when the cache has to be
    # spread evenly for want of anything better. A vector removes that guess;
    # without one they still abstain.
    uneven_cache = (
        layout.recurrent_bytes > 0 or layout.n_attention_layers != layout.n_layers or layout.has_swa
    )
    weights = [max(0, int(w)) for w in kv_layer_weights]
    if len(weights) != layout.n_layers or not any(weights):
        weights = []
    if uneven_cache and not weights:
        if layout.recurrent_bytes > 0:
            return "the recurrent state's per-layer split is not visible in the layout"
        if layout.n_attention_layers != layout.n_layers:
            return (
                f"only {layout.n_attention_layers} of {layout.n_layers} layers hold a cache "
                "and the layout does not say which"
            )
        return (
            "the cache is per-layer uneven (sliding-window attention) and no per-layer "
            "vector was supplied to say which layers are full-context"
        )
    if layout.has_excluded_blocks:
        return "the GGUF carries trailing blocks that shift llama.cpp's row count"

    n_slots = layout.n_layers + 1
    if n_slots <= 1:
        return None
    cache = (
        0
        if opts.kv_on_host
        else cache_bytes(layout, n_ctx, kv_quantised = quantised, kv_bytes_floor = kv_bytes_floor)
    )
    # Scaled to the total the caller already priced. Uniform when unsupplied.
    total_weight = sum(weights)
    if weights and total_weight > 0:
        kv_by_layer = [cache * w // total_weight for w in weights]
    else:
        per = cache // layout.n_layers if layout.n_layers else 0
        kv_by_layer = [per] * layout.n_layers
    by_index = {b.index: b for b in layout.blocks}
    output_row_bytes = layout.other_resident_bytes + (0 if spill_lm_head else layout.lm_head_bytes)

    slots = _device_slots(n_slots, split_weights_per_device or vram_bytes_per_device)
    for device, rows in enumerate(slots):
        used = 0
        for row in rows:
            if row == n_slots - 1:
                used += output_row_bytes
                continue
            block = by_index.get(row)
            if block is None:
                continue
            # Whatever this rung did NOT take off the block is still on the card.
            # Subtracting BYTES rather than skipping a whole block is what keeps
            # this honest once a rung can move a third of an FFN: crediting the
            # whole block for a partial move is the optimistic direction, and a
            # per-device shortfall is a hard throw rather than a slow load.
            moved = max(0, spilled_bytes_by_index.get(row, 0))
            used += max(0, block.resident_bytes + block.spillable_bytes - moved)
            if row < len(kv_by_layer):
                used += kv_by_layer[row]
        # Everything outside the layout sits on the main device, which is
        # devices[0] once -sm none has already pruned the list.
        if device == 0:
            used += max(0, opts.extra_resident_bytes)
        headroom = max(0, vram_bytes_per_device[device] - opts.overhead_bytes_per_device)
        if used > headroom:
            return (
                f"device {device} would still hold {used / GIB:.2f} GiB of its "
                f"{len(rows)}-row share against {headroom / GIB:.2f} GiB usable"
            )
    return None


def _plan_at(
    layout: ModelLayout,
    opts: PlanOptions,
    n_ctx: int,
    budget: int,
    host_ram_bytes: Optional[int],
    quantised: bool,
    kv_bytes_floor: int = 0,
    vram_bytes_per_device: Sequence[int] = (),
    split_weights_per_device: Sequence[int] = (),
    kv_layer_weights: Sequence[int] = (),
) -> Optional[Plan]:
    """One pass of the ladder at a fixed context and cache dtype."""
    needed = all_resident_bytes(
        layout,
        n_ctx,
        kv_quantised = quantised,
        kv_bytes_floor = kv_bytes_floor,
        kv_on_host = opts.kv_on_host,
    )
    if needed <= budget:
        return _finish(
            layout,
            opts,
            n_ctx,
            [],
            False,
            host_ram_bytes,
            quantised = quantised,
            kv_bytes_floor = kv_bytes_floor,
            budget = budget,
            reason = (
                f"the whole load fits in VRAM ({needed / GIB:.2f} of "
                f"{budget / GIB:.2f} GiB usable), so nothing is spilled"
            ),
        )

    n_devices = len(vram_bytes_per_device)
    deficit = needed - budget
    spillable = [b for b in layout.blocks if b.spillable_bytes > 0]

    def attempt(
        units: list[SpillUnit],
        spill_lm_head: bool,
        kv_host: bool,
        what: str,
        reason: str,
    ) -> Plan:
        """Check ``units`` device by device, then either finish or say why not."""
        moved = _moved_by_index(units)
        full = all(moved.get(b.index, 0) >= b.spillable_bytes for b in spillable)
        if n_devices > 1 and not full and not opts.trust_device_row_model:
            # A pooled budget is not a per-device fit test for a PARTIAL spill.
            # llama.cpp fixes the split before any override exists -- free memory
            # per device at llama-model.cpp:1425-1433, prefix-summed at :1439-1447,
            # then upper_bound on the normalised LAYER INDEX at :1457, so each
            # device owns a contiguous index range -- and -ot only swaps a tensor's
            # buffer type in llama_model_loader::create_tensor
            # (llama-model-loader.cpp:1177-1203), leaving dev_layer(il) untouched
            # (llama-model.cpp:1467-1474). Nothing rebalances afterwards and with
            # --fit off common/fit.cpp never runs, so a subset of indices sitting
            # in one device's range relieves only that device: the aggregate
            # deficit is covered while a single card is still over, and a
            # per-device shortfall is a hard throw (llama-model.cpp:1731-1733).
            # Which rows the chosen indices land on is exactly what makes it
            # uneven, so no arithmetic rescues it. Abstain: --fit on is per-device
            # aware (common/fit.cpp:646-651, :687, :705). A FULL spill IS
            # checkable, and is checked below rather than assumed.
            #
            # The ladder makes partial spills the COMMON case rather than the
            # exception -- covering a deficit with one rung of a few blocks is the
            # point of it -- so on multiple devices this abstains far more often
            # than the whole-FFN planner did. That is the honest answer until the
            # per-device row arithmetic can be checked against a real split.
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
        )
        if uneven is not None:
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
            reason = reason,
        )

    # Walk the ladder: take the MINIMAL set of units from the cheapest rung, and
    # step down only when that rung is exhausted. Because a rung is reached only
    # once every rung above it has been taken in full, what comes out is exactly
    # "all of rungs 1..k-1, plus the fewest units of rung k that close the gap" --
    # which is the minimal-blocks rule applied at every level of the ladder rather
    # than only at the whole-FFN one.
    taken: list[SpillUnit] = []
    freed = 0
    for cls in _rung_classes(layout, opts):
        if freed >= deficit:
            break
        chosen, got = _select_units(
            _units_of(layout.blocks, cls), deficit - freed, opts.spill_order
        )
        taken.extend(chosen)
        freed += got

    if freed >= deficit:
        taken, freed = _grade_the_boundary_block(layout, opts, taken, freed, deficit)
        described = _rung_description(taken, layout)
        return attempt(
            taken,
            False,
            False,
            f"spilling {described}",
            (
                f"spilled {described} ({freed / GIB:.2f} GiB) to cover a "
                f"{deficit / GIB:.2f} GiB deficit, keeping the KV cache resident"
            ),
        )

    # Every spillable weight is on the host and it is still short: lm_head is the
    # next rung. It costs 16% here against 43% if taken first, because FFN offload
    # has already made generation host-bandwidth-bound.
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

    # The bottom two rungs, both off by default. See PlanOptions for the measured
    # reason: a load that needs these is a load --fit on should place.
    if opts.allow_attention_spill:
        attn, got = _select_units(
            _units_of(layout.blocks, SpillClass.ATTENTION), deficit - freed, opts.spill_order
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
        cache = cache_bytes(
            layout, n_ctx, kv_quantised = quantised, kv_bytes_floor = kv_bytes_floor
        )
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
) -> tuple[Optional[Plan], float, float]:
    """An abstaining Plan when ``--fit on`` is as good as this spill, else None.

    Both arms are scored with ``rank``, so PREFILL is counted. That is the half
    the planner never priced: it called ``generation_penalty_ms`` alone, and
    #9861 duly measured prefill slower in 43 of 43 planned cells, an exceptionless
    result that only a structural cause explains.

    Prefill is where a spill is worst and a fitter is best. Spilled FFN bytes are
    streamed once per ubatch at FULL size -- a 512-token ubatch selects
    essentially every expert, so MoE sparsity buys nothing -- while the fitter's
    moved layers are simply not on the critical path for the resident ones.
    """
    plan = _spill_placement(layout, units, spill_lm_head)
    if not plan.host_groups:
        return None, 0.0, 0.0

    # A spill the host cannot hold in RAM is the one configuration measured to be
    # unambiguously worse than letting llama.cpp fit the model, so it is refused
    # before any cost comparison rather than scored.
    #
    # The cost model cannot see this. It prices host bytes at host bandwidth,
    # which is right only while they are IN host memory; past that they are read
    # from disk, and the planner also loses ``--load-mode none`` (mmap can page,
    # a no-mmap load cannot), which is where most of its measured advantage came
    # from in the first place. On a 12.67 GiB box the planner ran at 0.31x and
    # 0.23x of the fitter on generation, and its dense placement failed to load
    # at all -- twice, reproducibly, with
    #
    #   ... preferred buffer type CUDA0, using CUDA_Host instead   then   Killed
    #
    # while ``--fit on`` completed both times on the same host.
    if host_ram_bytes is not None:
        spendable = max(0, host_ram_bytes - opts.host_ram_headroom_bytes)
        if host_bytes > spendable:
            return (
                Plan(
                    n_ctx = n_ctx,
                    reason = (
                        f"the spill needs {host_bytes / GIB:.2f} GiB of host RAM and only "
                        f"{spendable / GIB:.2f} GiB is spendable, so it would page from disk "
                        "without even the mmap that makes that survivable; left to --fit on"
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
    )
    if fallback is None:
        # Nothing to lose to: the fitter cannot place this load either, so the
        # spill is the only thing standing between the caller and a failed launch.
        return None, 0.0, 0.0

    scored = rank(
        [plan, fallback],
        opts.host,
        n_generated = opts.workload_generated_tokens,
        n_prompt = opts.workload_prompt_tokens,
        n_ubatch = opts.n_ubatch,
    )
    plan_ms = _score_of(plan, scored)
    fit_ms = _score_of(fallback, scored)
    if plan_ms <= fit_ms * (1.0 - opts.min_penalty_reduction):
        return None, plan_ms, fit_ms
    return (
        Plan(
            n_ctx = n_ctx,
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


def _patterns_for(layout: ModelLayout, units: Sequence[SpillUnit]) -> list[str]:
    """One ``-ot`` pattern per rung, each naming only the blocks that rung moved.

    Per rung rather than per block because llama.cpp walks the override list for
    every tensor it creates: one pattern covering 40 blocks costs one regex, 40
    patterns cost 40. And the block list is collapsed to an unbounded ``\\d+``
    only when a rung really did take every spillable block AND the GGUF carries
    no trailing blocks the layout dropped -- the unbounded form would otherwise
    also match the nextn/MTP blocks, whose ``ffn_*_exps`` load the moment a draft
    is engaged, moving bytes that neither ``host_bytes`` nor the deficit counted.
    """
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
        # "Every" means every block that HAS this rung, which is not the same set
        # for every rung: attention is only reached once the FFN rungs are
        # exhausted, so comparing it against the FFN-spillable set would collapse
        # a partial attention spill to an unbounded pattern and move the lot.
        candidates = {u.index for u in _units_of(layout.blocks, cls)}
        every = indices >= candidates and not layout.has_excluded_blocks
        listed = None if every else sorted(indices)
        if cls is None:
            patterns.append(spill_pattern_for(layout, listed))
        else:
            patterns.append(spill_pattern_for_class(layout, cls, listed))
    return patterns


def _score_of(placement: Placement, scored: Sequence[tuple[Placement, float]]) -> float:
    """``rank`` sorts, so the order it returns is not the order it was given.

    Matched on identity rather than equality: two placements can compare equal
    (an empty spill and an empty fallback both hold no groups) and picking the
    wrong one would silently compare a candidate against itself.
    """
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
    reason: str = "",
) -> Plan:
    """Assemble patterns, decide the load mode, and account for both sides.

    Also the one gate: every plan that spills anything is scored against what
    llama.cpp's own fitter would have done with the same budget, and dropped if
    it does not win by a margin. Feasibility was the only test before this, so a
    spill that COULD cover the deficit was always taken, however badly it paid.
    """
    # The bottom rung moved the cache, so from here on this load behaves exactly
    # like one the caller had passed -nkvo for.
    kv_on_host = opts.kv_on_host or kv_on_host_rung
    spilled_weight_bytes = sum(u.nbytes for u in units) + (
        layout.lm_head_bytes if spill_lm_head else 0
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
            host_bytes = layout.token_embd_bytes + spilled_weight_bytes,
            host_ram_bytes = host_ram_bytes,
        )
        if declined is not None:
            return declined

    patterns = _patterns_for(layout, units)
    if spill_lm_head:
        patterns.append(LM_HEAD_PATTERN)
    indices = sorted({u.index for u in units})

    spilled_bytes = spilled_weight_bytes
    # token_embd is host-resident on every launch, so it is host RAM this plan
    # has to be able to pay for even when nothing is spilled.
    host_bytes = layout.token_embd_bytes + spilled_bytes
    if kv_on_host:
        # -nkvo moved the cache and the recurrent state out of VRAM, not out of
        # existence: they are host RAM now, and the mmap decision below has to see
        # them or it answers against a footprint short by the whole cache.
        host_bytes += (
            cache_bytes(layout, n_ctx, kv_quantised = quantised, kv_bytes_floor = kv_bytes_floor)
            + layout.recurrent_bytes
        )
    vram_bytes = (
        all_resident_bytes(
            layout,
            n_ctx,
            kv_quantised = quantised,
            kv_bytes_floor = kv_bytes_floor,
            kv_on_host = kv_on_host,
        )
        - spilled_bytes
    )

    # mmap costs 2 to 4.6x on host-resident weight reads, so turn it off -- but only
    # when host RAM holds the host side; otherwise mmap keeps an over-commit pageable.
    if host_ram_bytes is None:
        load_mode_none = False
    else:
        load_mode_none = host_bytes <= max(0, host_ram_bytes - opts.host_ram_headroom_bytes)

    cache_type = opts.kv_quant_type if quantised else None
    changed = bool(patterns) or load_mode_none or cache_type is not None
    return Plan(
        changed = changed,
        n_ctx = n_ctx,
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
