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
    reason: str = ""

    @property
    def spills_anything(self) -> bool:
        return bool(self.spilled_blocks) or self.spilled_lm_head


def _usable_vram(vram_bytes_per_device: Sequence[int], opts: PlanOptions) -> int:
    """Total creditable VRAM: every device pays the fixed per-device overhead,
    then the pool pays once for whatever sits on a card outside the layout."""
    pooled = sum(max(0, v - opts.overhead_bytes_per_device) for v in vram_bytes_per_device)
    return pooled - max(0, opts.extra_resident_bytes)


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


def _spill_penalty_ms(
    layout: ModelLayout, chosen: Sequence[BlockLayout], spill_lm_head: bool, host: HostProfile
) -> float:
    """Predicted extra ms per generated token for this spill, on this host.

    Spilled weights are read by the CPU backend, not streamed to the GPU: ggml
    only migrates an op at batch >= 32 and decode is batch 1, so the cost tracks
    host cores. MoE experts are charged their ROUTED fraction, since only
    ``n_expert_used`` of ``n_expert`` are touched per token, which is why MoE
    tolerates spilling far better than a fully activated dense FFN.
    """
    groups: list[TensorGroup] = []
    spilled = sum(b.spillable_bytes for b in chosen)
    if spilled:
        if layout.is_moe and layout.n_expert and layout.n_expert_used:
            groups.append(
                TensorGroup(
                    "experts",
                    spilled,
                    Access.SCATTERED,
                    activation_fraction = layout.n_expert_used / layout.n_expert,
                )
            )
        else:
            groups.append(TensorGroup("ffn", spilled, Access.CONTIGUOUS))
    if spill_lm_head and layout.lm_head_bytes:
        groups.append(TensorGroup("lm_head", layout.lm_head_bytes, Access.SINGLE_MATVEC))
    if not groups:
        return 0.0
    return generation_penalty_ms(Placement(host_groups = groups), host)


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
) -> int:
    """VRAM needed with EVERY spillable tensor already on the host.

    Attention weights, norms, routers, shared experts, the recurrent state, the
    cache and lm_head. Below this, ``-ot`` has nothing left to give and only a
    smaller quant or less context can help.
    """
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
) -> int:
    """VRAM needed with nothing spilled. token_embd is excluded: it is never
    GPU-resident (llama-model.cpp pins dev_input to the CPU unconditionally)."""
    return (
        resident_floor_bytes(
            layout, n_ctx, kv_quantised = kv_quantised, kv_bytes_floor = kv_bytes_floor
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
) -> Plan:
    """Decide the placement for one launch.

    ``split_weights_per_device`` is the RAW free VRAM llama.cpp will size its row
    ranges from, in the same device order as ``vram_bytes_per_device``. It is a
    different quantity from the budget by construction -- the budget subtracts a
    per-card reserve -- so the two must not be conflated when modelling the
    split. Empty falls back to the budget, which is right whenever the caller has
    applied no per-card adjustment at all.

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
        and all_resident_bytes(layout, n_ctx, kv_bytes_floor = kv_bytes_floor) > budget
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
                )
                if plan is not None:
                    return plan

    floor = resident_floor_bytes(layout, n_ctx, kv_bytes_floor = kv_bytes_floor)
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
    # These three make the per-row byte split unknowable from the layout, so
    # there is nothing to validate against and the honest answer is to abstain.
    if layout.recurrent_bytes > 0:
        return "the recurrent state's per-layer split is not visible in the layout"
    if layout.n_attention_layers != layout.n_layers:
        return (
            f"only {layout.n_attention_layers} of {layout.n_layers} layers hold a cache "
            "and the layout does not say which"
        )
    if layout.has_swa:
        # The check above passes (every layer IS attention), but a window layer's
        # cache is a fraction of a full-context one (Gemma3 interleaves 5:1) and
        # the layout does not say which rows are which. An even spread under-books
        # whichever card drew the full-context rows.
        return "the cache is per-layer uneven (sliding-window attention) and the layout does not say which layers are full-context"
    if layout.has_excluded_blocks:
        return "the GGUF carries trailing blocks that shift llama.cpp's row count"

    n_slots = layout.n_layers + 1
    if n_slots <= 1:
        return None
    cache = cache_bytes(layout, n_ctx, kv_quantised = quantised, kv_bytes_floor = kv_bytes_floor)
    kv_per_layer = cache // layout.n_layers if layout.n_layers else 0
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
            used += block.resident_bytes + kv_per_layer
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
) -> Optional[Plan]:
    """One pass of the ladder at a fixed context and cache dtype."""
    needed = all_resident_bytes(
        layout, n_ctx, kv_quantised = quantised, kv_bytes_floor = kv_bytes_floor
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
                reason = (
                    f"spilled every block's FFN ({freed / GIB:.2f} GiB) plus lm_head "
                    f"({layout.lm_head_bytes / GIB:.2f} GiB) to cover a "
                    f"{deficit / GIB:.2f} GiB deficit"
                ),
            )
    return None


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
    reason: str = "",
) -> Plan:
    """Assemble patterns, decide the load mode, and account for both sides."""
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
    vram_bytes = (
        all_resident_bytes(layout, n_ctx, kv_quantised = quantised, kv_bytes_floor = kv_bytes_floor)
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


def smart_offload_enabled(env: Optional[Mapping[str, str]] = None) -> bool:
    """Whether the launch path may emit a spill plan.

    Off by default. The planner changes how a model is placed, and the failure
    mode of getting it wrong is a load that OOMs where llama.cpp's own default
    would have paged, so it opts in rather than out until it has run on more
    than one machine.
    """
    raw = (os.environ if env is None else env).get("UNSLOTH_SMART_OFFLOAD", "")
    return str(raw).strip().lower() in ("1", "true", "yes", "on", "enabled")
