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
    LM_HEAD_PATTERN,
    BlockLayout,
    ModelLayout,
    spill_pattern_for,
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
    allow_lm_head_spill: bool = True
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


def _select_blocks(
    blocks: Iterable[BlockLayout], deficit: int, order: SpillOrder
) -> tuple[list[BlockLayout], int]:
    """Blocks to spill to free at least ``deficit``, and what they actually free."""
    remaining = [b for b in blocks if b.spillable_bytes > 0]
    if order is SpillOrder.FRONT_FIRST:
        remaining.sort(key = lambda b: b.index)
    elif order is SpillOrder.BACK_FIRST:
        remaining.sort(key = lambda b: -b.index)
    else:
        remaining.sort(key = lambda b: -b.spillable_bytes)

    chosen: list[BlockLayout] = []
    freed = 0
    while freed < deficit and remaining:
        if order is SpillOrder.LARGEST_FIRST:
            residual = deficit - freed
            # Prefer the SMALLEST block that closes the gap: the last pick must
            # not overshoot by a whole large block.
            covering = [b for b in remaining if b.spillable_bytes >= residual]
            pick = min(covering, key = lambda b: b.spillable_bytes) if covering else remaining[0]
        else:
            pick = remaining[0]
        remaining.remove(pick)
        chosen.append(pick)
        freed += pick.spillable_bytes
    return chosen, freed


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
    layout: ModelLayout, chosen: Sequence[BlockLayout], spill_lm_head: bool
) -> Placement:
    """This plan's spill, as something the cost model can price.

    ``-ot`` moves the named weights and nothing else, so the attention cache
    stays on the device: ``kv_host_bytes`` is 0 here by construction. That is the
    whole of the planner's claimed advantage over a layer fitter, and pricing it
    is how :func:`_fit_fallback_placement` gets to argue back.
    """
    groups: list[TensorGroup] = []
    spilled = sum(b.spillable_bytes for b in chosen)
    if spilled:
        groups.append(_ffn_group(layout, spilled))
    if spill_lm_head and layout.lm_head_bytes:
        groups.append(TensorGroup("lm_head", layout.lm_head_bytes, Access.SINGLE_MATVEC))
    return Placement(host_groups = groups)


def _spill_penalty_ms(
    layout: ModelLayout, chosen: Sequence[BlockLayout], spill_lm_head: bool, host: HostProfile
) -> float:
    """Predicted extra ms per generated token for this spill, on this host.

    Spilled weights are read by the CPU backend, not streamed to the GPU: ggml
    only migrates an op at batch >= 32 and decode is batch 1, so the cost tracks
    host cores.
    """
    placement = _spill_placement(layout, chosen, spill_lm_head)
    if not placement.host_groups:
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
    return max(layout.kv_bytes(n_ctx, _kv_elem_bytes(kv_quantised)), max(0, kv_bytes_floor))


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
    spilled_indices: set[int],
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
            used += block.resident_bytes
            if row < len(kv_by_layer):
                used += kv_by_layer[row]
            if row not in spilled_indices:
                used += block.spillable_bytes
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
    chosen, freed = _select_blocks(layout.blocks, deficit, opts.spill_order)
    if freed >= deficit:
        spillable = [b for b in layout.blocks if b.spillable_bytes > 0]
        if n_devices > 1 and len(chosen) < len(spillable):
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
            return Plan(
                n_ctx = n_ctx,
                reason = (
                    f"a partial spill ({len(chosen)} of {len(spillable)} blocks) across "
                    f"{n_devices} devices cannot be checked against a pooled budget, "
                    "because llama.cpp assigns contiguous layer ranges per device and "
                    "-ot does not move a layer; leaving llama.cpp's own fitter to place it"
                ),
            )
        uneven = _per_device_shortfall(
            layout,
            opts,
            n_ctx,
            {b.index for b in chosen},
            False,
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
                    f"spilling every block still does not fit device by device: {uneven}; "
                    "leaving llama.cpp's own fitter to place it"
                ),
            )
        return _finish(
            layout,
            opts,
            n_ctx,
            chosen,
            False,
            host_ram_bytes,
            quantised = quantised,
            kv_bytes_floor = kv_bytes_floor,
            budget = budget,
            reason = (
                f"spilled the FFN of {len(chosen)} of {len(layout.blocks)} blocks "
                f"({freed / GIB:.2f} GiB) to cover a {deficit / GIB:.2f} GiB deficit, "
                "keeping the KV cache resident"
            ),
        )

    # Every block spilled and still short: lm_head is the last rung. It costs
    # 16% here against 43% if taken first, because FFN offload has already made
    # generation host-bandwidth-bound.
    if opts.allow_lm_head_spill and layout.lm_head_bytes:
        if freed + layout.lm_head_bytes >= deficit:
            uneven = _per_device_shortfall(
                layout,
                opts,
                n_ctx,
                {b.index for b in chosen},
                True,
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
                        "spilling every block and lm_head still does not fit device by "
                        f"device: {uneven}; leaving llama.cpp's own fitter to place it"
                    ),
                )
            return _finish(
                layout,
                opts,
                n_ctx,
                chosen,
                True,
                host_ram_bytes,
                quantised = quantised,
                kv_bytes_floor = kv_bytes_floor,
                budget = budget,
                reason = (
                    f"spilled every block's FFN ({freed / GIB:.2f} GiB) plus lm_head "
                    f"({layout.lm_head_bytes / GIB:.2f} GiB) to cover a "
                    f"{deficit / GIB:.2f} GiB deficit"
                ),
            )
    return None


def _cost_gate(
    layout: ModelLayout,
    opts: PlanOptions,
    n_ctx: int,
    chosen: Sequence[BlockLayout],
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
    plan = _spill_placement(layout, chosen, spill_lm_head)
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
    chosen: list[BlockLayout],
    spill_lm_head: bool,
    host_ram_bytes: Optional[int],
    *,
    quantised: bool = False,
    kv_bytes_floor: int = 0,
    budget: Optional[int] = None,
    reason: str = "",
) -> Plan:
    """Assemble patterns, decide the load mode, and account for both sides.

    Also the one gate: every plan that spills anything is scored against what
    llama.cpp's own fitter would have done with the same budget, and dropped if
    it does not win by a margin. Feasibility was the only test before this, so a
    spill that COULD cover the deficit was always taken, however badly it paid.
    """
    plan_ms = fit_ms = 0.0
    if opts.require_cost_win and budget is not None and (chosen or spill_lm_head):
        declined, plan_ms, fit_ms = _cost_gate(
            layout,
            opts,
            n_ctx,
            chosen,
            spill_lm_head,
            budget,
            quantised = quantised,
            kv_bytes_floor = kv_bytes_floor,
            host_bytes = layout.token_embd_bytes
            + sum(b.spillable_bytes for b in chosen)
            + (layout.lm_head_bytes if spill_lm_head else 0),
            host_ram_bytes = host_ram_bytes,
        )
        if declined is not None:
            return declined

    patterns: list[str] = []
    indices = sorted(b.index for b in chosen)
    if indices:
        # One global pattern when every spillable block is going -- shorter, and
        # the form the benchmarks used. NOT when the GGUF carries blocks the layout
        # dropped: the unbounded \d+ would also match the trailing nextn/MTP
        # blocks, whose ffn_*_exps load the moment a draft is engaged. That moves
        # bytes neither host_bytes nor the deficit counted (so the mmap decision is
        # made on an undercount) and drags the draft FFN onto the CPU backend.
        spillable = [b.index for b in layout.blocks if b.spillable_bytes > 0]
        all_of_them = set(indices) == set(spillable) and not layout.has_excluded_blocks
        patterns.append(spill_pattern_for(layout, None if all_of_them else indices))
    if spill_lm_head:
        patterns.append(LM_HEAD_PATTERN)

    spilled_bytes = sum(b.spillable_bytes for b in chosen) + (
        layout.lm_head_bytes if spill_lm_head else 0
    )
    # token_embd is host-resident on every launch, so it is host RAM this plan
    # has to be able to pay for even when nothing is spilled.
    host_bytes = layout.token_embd_bytes + spilled_bytes
    if opts.kv_on_host:
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
            kv_on_host = opts.kv_on_host,
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
        predicted_gen_penalty_ms = _spill_penalty_ms(layout, chosen, spill_lm_head, opts.host),
        predicted_request_ms = plan_ms,
        predicted_fit_request_ms = fit_ms,
        reason = reason,
    )


def plan_to_args(plan: Plan) -> list[str]:
    """The launch flags for ``plan``. Empty when it changes nothing."""
    args: list[str] = []
    for pattern in plan.ot_patterns:
        args.extend(["-ot", f"{pattern}=CPU"])
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
