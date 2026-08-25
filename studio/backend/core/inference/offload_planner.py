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
    # bandwidth: taking a 209 MiB block to cover a 50 MiB deficit wastes 159 MiB
    # of host traffic on every token.
    LARGEST_FIRST = "largest_first"
    FRONT_FIRST = "front_first"
    BACK_FIRST = "back_first"


@dataclass(frozen = True)
class PlanOptions:
    # Compute buffer + CUDA context + scratch, charged on every device.
    overhead_bytes_per_device: int = GIB
    # Host RAM this planner refuses to spend, so a spill does not push the box
    # into swap.
    host_ram_headroom_bytes: int = 2 * GIB
    context_policy: ContextPolicy = ContextPolicy.NEVER_REDUCE
    min_ctx: int = 4096
    spill_order: SpillOrder = SpillOrder.LARGEST_FIRST
    allow_lm_head_spill: bool = True
    # What the host can bring to bear on spilled weights. This is NOT a detail:
    # spilled generation runs on the CPU backend, because ggml only moves an op
    # to the GPU at batch >= 32 (ggml-cuda.cu, op_offload_min_batch_size) and
    # decode is batch 1. So the penalty scales with core count -- measured
    # 2.42 / 5.83 / 11.82 / 14.94 t/s at 4 / 16 / 64 / 192 threads. Defaulting
    # to the reference host would hand a desktop user server-shaped advice.
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
    # Predicted extra milliseconds per generated token versus fully resident,
    # on the host this was planned for. 0.0 when nothing is spilled. Reported so
    # callers can surface the real cost instead of implying a spill is free, and
    # so a plan on a small host is visibly expensive rather than quietly so.
    predicted_gen_penalty_ms: float = 0.0
    reason: str = ""

    @property
    def spills_anything(self) -> bool:
        return bool(self.spilled_blocks) or self.spilled_lm_head


def _usable_vram(vram_bytes_per_device: Sequence[int], opts: PlanOptions) -> int:
    """Total creditable VRAM: every device pays the fixed per-device overhead."""
    return sum(max(0, v - opts.overhead_bytes_per_device) for v in vram_bytes_per_device)


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
            # Prefer the SMALLEST block that still closes the gap: that is what
            # keeps the last pick from overshooting by a whole large block.
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


def resident_floor_bytes(
    layout: ModelLayout,
    n_ctx: int,
    *,
    kv_quantised: bool = False,
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
        + layout.kv_bytes(n_ctx, _kv_elem_bytes(kv_quantised))
    )


def all_resident_bytes(
    layout: ModelLayout,
    n_ctx: int,
    *,
    kv_quantised: bool = False,
) -> int:
    """VRAM needed with nothing spilled. token_embd is excluded: it is never
    GPU-resident (llama-model.cpp pins dev_input to the CPU unconditionally)."""
    return resident_floor_bytes(layout, n_ctx, kv_quantised = kv_quantised) + layout.spillable_bytes


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
) -> Plan:
    """Decide the placement for one launch.

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
        # One pool: "spilling" moves bytes between two names for the same chips,
        # so it frees nothing and the trade this planner exists to make does not
        # exist. Metal additionally keeps mmap zero copy (buffer_from_host_ptr),
        # so the no-mmap rule inverts there too.
        return Plan(reason = "unified memory host, spilling frees no device memory")
    budget = _usable_vram(vram_bytes_per_device, opts)
    if budget <= 0:
        return Plan(reason = "no creditable VRAM after per-device overhead")

    n_ctx = requested_ctx if requested_ctx > 0 else layout.n_ctx_train
    if layout.n_ctx_train:
        n_ctx = min(n_ctx, layout.n_ctx_train)
    if n_ctx <= 0:
        return Plan(reason = "no usable context length")

    # PREFER_RESIDENT gets its say before the ladder: a smaller fully resident
    # context is faster than a larger spilled one, when the caller has said the
    # context may move.
    if (
        opts.context_policy is ContextPolicy.PREFER_RESIDENT
        and all_resident_bytes(layout, n_ctx) > budget
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
        plan = _plan_at(layout, opts, n_ctx, budget, host_ram_bytes, quantised)
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
                plan = _plan_at(layout, opts, shrunk, budget, host_ram_bytes, quantised)
                if plan is not None:
                    return plan

    floor = resident_floor_bytes(layout, n_ctx)
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


def _plan_at(
    layout: ModelLayout,
    opts: PlanOptions,
    n_ctx: int,
    budget: int,
    host_ram_bytes: Optional[int],
    quantised: bool,
) -> Optional[Plan]:
    """One pass of the ladder at a fixed context and cache dtype."""
    needed = all_resident_bytes(layout, n_ctx, kv_quantised = quantised)
    if needed <= budget:
        return _finish(
            layout,
            opts,
            n_ctx,
            [],
            False,
            host_ram_bytes,
            quantised = quantised,
            reason = (
                f"the whole load fits in VRAM ({needed / GIB:.2f} of "
                f"{budget / GIB:.2f} GiB usable), so nothing is spilled"
            ),
        )

    deficit = needed - budget
    chosen, freed = _select_blocks(layout.blocks, deficit, opts.spill_order)
    if freed >= deficit:
        return _finish(
            layout,
            opts,
            n_ctx,
            chosen,
            False,
            host_ram_bytes,
            quantised = quantised,
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
            return _finish(
                layout,
                opts,
                n_ctx,
                chosen,
                True,
                host_ram_bytes,
                quantised = quantised,
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
    reason: str = "",
) -> Plan:
    """Assemble patterns, decide the load mode, and account for both sides."""
    patterns: list[str] = []
    indices = sorted(b.index for b in chosen)
    if indices:
        # One global pattern when every spillable block is going, which is both
        # shorter and exactly the form the benchmarks used.
        spillable = [b.index for b in layout.blocks if b.spillable_bytes > 0]
        all_of_them = set(indices) == set(spillable)
        patterns.append(spill_pattern_for(layout, None if all_of_them else indices))
    if spill_lm_head:
        patterns.append(LM_HEAD_PATTERN)

    spilled_bytes = sum(b.spillable_bytes for b in chosen) + (
        layout.lm_head_bytes if spill_lm_head else 0
    )
    # token_embd is host-resident on every launch, so it is host RAM this plan
    # has to be able to pay for even when nothing is spilled.
    host_bytes = layout.token_embd_bytes + spilled_bytes
    vram_bytes = all_resident_bytes(layout, n_ctx, kv_quantised = quantised) - spilled_bytes

    # mmap costs 2 to 4.6x on host-resident weight reads, so turn it off -- but
    # only when host RAM can really hold the host side. Where it cannot, mmap is
    # the only thing that keeps an over-commit pageable.
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
