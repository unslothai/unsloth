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

# Nominal ggml bits-per-weight, by quant type. Standard block layouts: a type's
# block size divided by its bytes per block.
#
# Used ONLY by :func:`moe_down_up_bpw_ratio`, which decides a granularity, never
# to size anything. A byte count always comes from the tensor table.
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

# Above this ratio the every-block ladder beat the whole-FFN planner on every
# MoE measured; below it, it lost or went flat. See
# ``PlanOptions.granularity_from_quant`` for the cells and the caveats.
_LADDER_BPW_THRESHOLD = 1.40


def moe_down_up_bpw_ratio(layout) -> Optional[float]:
    """How much denser ``ffn_down`` is than ``ffn_up``, in bits per weight.

    Returns None when the layout is not a graded MoE or the quant types were not
    recovered, which is the "no opinion" answer every caller must handle: the
    types come from the GGUF tensor table and an architecture whose tails match
    no class pattern leaves them empty.

    Computed from the TYPE rather than from bytes over elements because the
    layout carries types and not element counts. Checked against the five
    model-quants measured end to end, where nominal and measured bits-per-weight
    agree to within 0.025 and, critically, put every model on the same side of
    :data:`_LADDER_BPW_THRESHOLD`:

        gemma-26B Q2   measured 1.937   nominal 1.946
        gemma-26B Q3   measured 1.494   nominal 1.469
        Qwen35B   Q2   measured 1.349   nominal 1.324
        gemma-26B Q4   measured 1.341   nominal 1.333
        Qwen35B   Q4   measured 1.231   nominal 1.222

    Note the middle pair SWAPS order between the two columns. That is a real
    limitation and it is why the threshold sits at 1.40 rather than hard against
    either point: the rule separates the group, and does not resolve two models
    0.008 apart.
    """
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
    # Mean over blocks: a mixed quant gives different types to different blocks,
    # and the decision is about the model, not about one layer.
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
    """Which blocks to spill when only some are needed.

    MEASURED, over 41 spilled cells that ran both LARGEST_FIRST and BACK_FIRST
    (non-spilled controls excluded, where the two orders are trivially equal):

        median contig / largest-first   1.007
        MoE (n=29)                      1.007
        dense (n=12)                    1.009
        better by more than 3%          4 of 41
        WORSE by more than 3%           0 of 41
        range                           0.991 to 1.079

    The useful part is the SHAPE rather than the median. Contiguous never costs
    more than 3% and occasionally pays up to 1.079, on both MoE and dense, across
    five hosts. So the case for BACK_FIRST is an ASYMMETRY argument -- it is
    close to free and sometimes wins -- and not an effect-size one.

    One cell shows 1.22x (gemma-31B Q2 at 16 GiB, replicated on two runs). That
    is an outlier and not the effect: the same model one VRAM level down measures
    exactly 1.000. It is recorded here because it was briefly mistaken for the
    effect size, which is the error this docstring now exists to prevent.

    The default is BACK_FIRST. Re-scored over 98 cells that ran both orders, five
    hosts, the median is 1.0073, better by more than 3% in 10 and worse in 0,
    worst cell 0.977. That is the same one-sided shape on 2.4x the sample, and it
    is enough to swap a byte-minimality guarantee for it: the guarantee bought a
    few MiB of overshoot, the order buys throughput and never costs it.
    LARGEST_FIRST stays selectable for the byte-minimal answer and for the
    benchmark's controls.

    An earlier version of this docstring said UNMEASURED, which was true when
    every -ot run spilled all blocks or none. It is no longer true.
    """

    # Best-fit-decreasing: fewest blocks AND least overshoot. Overshoot is real
    # bandwidth -- a 209 MiB block for a 50 MiB deficit wastes 159 MiB per token.
    # No longer the default, see above.
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
    # The part of that reserve that is NOT flat. MEASURED by sweeping device headroom with -ot at
    # fixed context and budget until the load flips from failing to serving: gemma-4-26B-A4B Q4
    # needs ~1184 MiB at n_ctx 33792, (1786.3, 1932.5] at ~66600 and more than 1958.1 at 132096,
    # so the slope is in [18.8, 23.4] KiB per token of TOTAL context. Total context, not prompt
    # length: three runs held n_ctx to 1.2% while moving pp 4x and parallel 4x the other way and
    # their brackets still overlapped. Budget independent: the 12 and 16 GiB ladders agree at
    # every context. And invisible in the buffer report, where the cache read 1200.0 MiB and the
    # compute buffer 366.0 MiB across a 14x span of context.
    # 23.4 KiB per token, the TOP of the measured bracket, so that starting the term later
    # (below) costs no margin at 66560 or 132096 against the curve that was validated there.
    overhead_bytes_per_token: int = 23961
    # Below this the term is zero, so the flat reserve is unchanged and no existing placement
    # moves. The measured requirement is FLAT up to here (<= 1184 MiB at n_ctx 9216 and ~1184
    # at 33792), so starting the slope at 16384 charged 343 MiB at a 32K context for nothing:
    # on a 27B at 32K that produced a 0.2 GiB deficit, a spill, and a measured loss where
    # llama.cpp's own fitter, which targets a flat 1 GiB, had slack and moved nothing.
    overhead_free_ctx: int = 32768
    # Both reserve terms withhold DEVICE memory on a discrete card. On a unified-memory part
    # (Strix Halo, DGX Spark) the binding constraint is host MemAvailable and the planner
    # abstains before either term is read, so neither has a meaning there.
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
    spill_order: SpillOrder = SpillOrder.BACK_FIRST
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
    # Pick the granularity from the model's quant types instead of taking
    # ``ffn_granularity`` as given. OFF, and it should stay off until the
    # evidence is better than it is.
    #
    # WHAT IT ENCODES. The every-block ladder spills ``ffn_down`` and keeps
    # ``ffn_up``/``gate_up`` resident. It wins when the tensor it MOVES is much
    # denser in bits per weight than the one it KEEPS, because it then frees
    # more VRAM per unit of host work. Ratio of down bpw to up bpw against the
    # measured ladder-over-coarse generation ratio, six model-quants, two
    # families, five hosts:
    #
    #     1.93  gemma-26B Q2   1.27-1.33x   WIN
    #     1.49  gemma-26B Q3   1.34x        WIN
    #     1.35  Qwen35B   Q2   0.98-1.02x   flat
    #     1.34  gemma-26B Q4   0.93-0.96x   LOSS
    #     1.28  Qwen35B   Q6   0.954, 0.994 LOSS
    #     1.23  Qwen35B   Q4   0.88-0.95x   LOSS
    #
    # Monotonic, no inversion, and it holds inside each family taken alone, so
    # it is not a family effect wearing a ratio as a disguise.
    #
    # WHY IT IS OFF. Six points with a threshold located only to within
    # 1.35-1.49 is a candidate, not a default.
    #
    # MoE ONLY, and the reason is that DENSE IS HETEROGENEOUS rather than
    # uniformly one way. Scored over 26 dense cells, the ladder beats the coarse
    # planner by more than 3% in exactly two, and this threshold gets half the
    # dense models wrong:
    #
    #     ratio  model         rule says   measured
    #     1.34   gemma-31B Q2  BOUNDARY    WIN 1.069-1.444   rule WRONG
    #     1.31   gemma-31B Q3  BOUNDARY    WIN 1.136-1.40    rule WRONG
    #     1.22   gemma-31B Q4  BOUNDARY    flat 1.000        rule right
    #     1.33   gemma-E2B Q4  BOUNDARY    flat 0.975-1.008  rule right
    #
    # and Qwen3.8-27B, also dense, LOSES in all four of its spilled cells
    # (0.905, 0.953, 0.961, 0.971). So gemma-31B wins where the ratio says it
    # should not while two other dense families behave as the ratio predicts,
    # which means dense carries a model-specific factor this rule does not
    # capture. Abstaining is the honest response to that, not a placeholder.
    #
    # WHAT WOULD FALSIFY IT. An MoE landing in the untested 1.35-1.49 gap that
    # comes out on the wrong side, or any MoE above 1.5 that loses.
    granularity_from_quant: bool = False
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
    # The launch's cache is ALREADY quantised (its element is under two bytes),
    # so the first, and normally only, mode is priced as such and ``kv_quant_type``
    # names the type in force. Handed over only as a byte floor, the f16 product
    # overrode the smaller measured cache and spilled to cover a deficit that was
    # arithmetic: a model whose weights plus q8 cache are fully resident read as
    # several GiB over budget.
    cache_quantised: bool = False
    # The user set ``--cache-ram -1``: llama-server keeps every prompt it can and
    # bounds the cache by nothing, so no figure charged for it is a ceiling. The
    # RAM proof that picks ``--load-mode none`` abstains (mmap keeps the spill
    # pageable when the cache grows into it) and the clamp, which would rewrite
    # the user's value, is never derived.
    prompt_cache_unbounded: bool = False
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

    # ---- rungs 0 to 2: things a launch can give up BEFORE it spills a weight. ----
    # Every default below means "not supplied", so a call that sets none of them is
    # byte-identical to the planner before they existed. The owner's priority
    # order for what a launch KEEPS is: generation speed, context, the prompt
    # cache, MTP / speculative draft, --parallel, mmproj resident. The ladder gives
    # those up in reverse, and every one of the three below is free per token,
    # which is why they sit above the first FFN rung.
    #
    # Vision projector bytes on the device, SEPARATE from ``extra_resident_bytes``
    # (a caller that sets this must stop folding the projector into that scalar, or
    # it is charged twice). ``mmproj_movable`` is True only when the caller would
    # accept ``--no-mmproj-offload``; the projector runs once per IMAGE, not per
    # token, so moving it costs nothing on decode.
    mmproj_bytes: int = 0
    mmproj_movable: bool = False
    # Slot count the cache floor was priced at, and the fewest slots the caller will
    # serve. Rung 1 steps the count down ONE slot at a time, re-pricing the cache at
    # each step, because the cache is what a slot costs: ~300 MiB per slot flat in
    # context on a sliding-window model, and linear in slots times per-sequence
    # context otherwise, which at n_ctx 132096 is ~6 GiB across four slots.
    n_parallel: int = 1
    min_parallel: int = 1
    # slots -> cache bytes the caller priced at the requested context for that slot
    # count. Authoritative where present; see ``_kv_floor_at`` for the fallback and
    # the one shape (sliding window, no map) where there is none and rung 1 is
    # skipped.
    kv_bytes_floor_by_parallel: Mapping[int, int] = field(default_factory = dict)
    # (n_ctx, n_parallel) -> the KV cache bytes llama.cpp allocates for this launch,
    # fixed recurrent state included; the seam passes its own estimator. Authoritative
    # when set: no scalar floor can be re-priced for a hybrid (fixed + growing), a
    # windowed cache (flat + linear) or an MLA latent at once. None keeps the
    # scaling rules in ``_kv_floor_at``.
    kv_bytes_at: Optional[Callable[[int, int], int]] = None
    # The micro-batch the launch normalises at each slot count, keyed like the
    # floor map. The emitted batch floor is max(slots, 2), so a first-class
    # batch of 1 launches at micro-batch 4 with four slots and 2 with one; the
    # gate scores a candidate at the batch ITS slot count launches, or a plan
    # rung 1 reduced is priced at a prefill stream twice its real size. Empty
    # means n_ubatch at every count.
    n_ubatch_by_parallel: Mapping[int, int] = field(default_factory = dict)
    # The MTP nextn block plus its cache, or a separate draft model plus its cache,
    # on the device. Charged unless rung 2 drops it. ``draft_drop_penalty_frac`` is
    # the generation cost of losing the draft, as a fraction of the plan's request
    # time. MEASURED on Qwen3.6-35B-A3B Q4 with its MTP head, fully resident on an
    # A100, --spec-type draft-mtp against none, 4 cells (2 budgets x 3 and 5
    # repeats): +2.7 / +14.3 / +7.6 / +2.3 percent generation, prefill -9 percent
    # every time. The +14.3 is a single 3-repeat reading the 5-repeat re-run of the
    # same cell did not reproduce (+2.3); the other three centre on ~5 percent.
    # Charged at 0.05, so the gate sees dropping the draft as roughly the cost of
    # spilling half a GiB (7 to 12 percent per GiB measured), and rung 2 stays
    # ahead of the weight rungs as the owner's ordering says, but no longer free.
    draft_bytes: int = 0
    draft_droppable: bool = False
    draft_drop_penalty_frac: float = 0.05
    # llama-server's default prompt-cache bound (--cache-ram, MiB). Host RAM, so it
    # competes with the plan only through the --load-mode none footprint; the plan
    # clamps it to what the host has left and reports the clamp only when it binds.
    cache_ram_default_mib: int = 8192
    # MoE at a long prompt is the one operating point where -ot measured WORSE than
    # llama.cpp's own layerwise fit: 0.94 to 0.97x at PP 32768 on 5 cells, 2 models,
    # 3 hosts, against 1.05 to 1.08x at PP 2048. The cost model cannot see it (the
    # MoE fallback spills through the same -ot mechanism, so both arms carry the
    # same per-byte prefill penalty and ``rank`` can only report a near-tie), so it
    # is a hard gate on the per-slot context. 0 disables it.
    moe_long_prompt_ctx: int = 32768
    # Whether the launch shares ONE cache across slots (``--kv-unified``). It
    # decides what "per slot" above means: llama-server sets n_ctx_slot = n_ctx
    # under a unified cache and n_ctx / n_parallel without one, so a single
    # request can consume the WHOLE window when this is set. Studio appends
    # ``--kv-unified`` on every launch with more than one slot that the binary
    # supports it on (llama_cpp.py, `n_parallel > 1 and caps["supports_kv_unified"]`),
    # and drops to one slot when it does not, so dividing unconditionally
    # understated the real prompt window by the slot count on exactly the
    # multi-slot loads Studio actually starts. Defaults False, which is the
    # divided behaviour every existing caller already gets.
    kv_unified: bool = False
    # Step of the descending context ladder a FIT_ONLY shrink walks. Feasibility is
    # monotone in context; the cost gate's acceptance is not, so a binary search can
    # skip the largest accepted context. 256-aligned.
    ctx_step: int = 1024


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
    # Rungs 0 to 2. ``n_parallel`` is 0 when the caller's slot count stands and the
    # reduced count when rung 1 fired (emit ``--parallel N``). ``mmproj_to_host``
    # asks for ``--no-mmproj-offload``. ``draft_dropped`` is a decision only: the
    # caller owns the speculative flags, since only it knows which kind of draft
    # it priced. ``cache_ram_mib`` is -1 when llama-server's default stands and
    # the clamped value only when the clamp BINDS.
    n_parallel: int = 0
    mmproj_to_host: bool = False
    draft_dropped: bool = False
    cache_ram_mib: int = -1
    # The cost gate said no. Distinct from "cannot be checked" (per-device, multi
    # device) and from "does not fit": a declined load is feasible as planned and
    # a smaller context may make it worth taking, which is what FIT_ONLY tries.
    declined_by_gate: bool = False
    # The decline is a MEASUREMENT rather than a comparison, and it does not move
    # with the context. A plain gate decline invites FIT_ONLY to try a smaller one;
    # a veto does not, because the same spill at 31744 tokens per slot is the same
    # trade that measured worse at 32768. The fall-through is llama.cpp's own fit at
    # the context the caller asked for, not a shorter context the planner picked.
    veto: bool = False
    # The planner priced this launch at ``n_ctx`` and it fits as described. False
    # on every abstain, which also carries a reason and may carry an n_ctx, so a
    # caller cannot otherwise tell "fits at 32768, nothing to move" from "could
    # not price this at all".
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
    """Bytes to leave free on EVERY device: the flat term plus the context-linear one.

    Charged per device like the flat term it extends. Whether the context-linear part is really
    per device or shared across a layer split is UNMEASURED, every calibration run being single
    GPU, so this over-reserves on multi-GPU rather than under-reserving. Over-reserving spills
    more, at a measured 7 to 12% of generation per surplus GiB; under-reserving loses the load.
    """
    over = max(0, n_ctx - max(0, opts.overhead_free_ctx))
    return max(0, opts.overhead_bytes_per_device) + over * max(0, opts.overhead_bytes_per_token)


def _usable_vram(
    vram_bytes_per_device: Sequence[int],
    opts: PlanOptions,
    n_ctx: int,
    *,
    outside_layout_bytes: Optional[int] = None,
) -> int:
    """Total creditable VRAM: every device pays the fixed per-device overhead,
    the split pays for each device AFTER the first, then the pool pays once for
    whatever sits on a card outside the layout.

    ``n_ctx`` is required rather than defaulted: the reserve is context dependent and a caller
    that silently got the flat term would under-reserve at long context and fail the load.
    ``outside_layout_bytes`` is what rungs 0 to 2 have left on the card; ``None`` charges the
    full amount, which is what every call that predates those rungs meant.
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
    """``opts.kv_bytes_at`` less the recurrent state charged separately.

    The callable answers with the WHOLE memory llama.cpp allocates for the launch, the
    fixed state included. Every sizing site here adds ``recurrent_bytes`` once per slot
    of its own, so the state comes out here rather than being counted twice. A layout
    that does not know its state leaves the callable's figure intact, which is the case
    the scaling rules below get wrong.
    """
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
    """The caller's cache floor re-priced for ``(n_ctx, n_parallel)``; ``None`` when it cannot be.

    ``opts.kv_bytes_at`` answers this exactly and is taken whenever it is set, because no
    rule over one scalar can be right for a windowed, latent and fixed-state cache at once.

    Without it the floor is a measurement at the requested context and the caller's slot
    count, and moving either axis needs a rule:

    - a different slot count takes the caller's ``kv_bytes_floor_by_parallel`` entry when it
      has one. Without one, a non-SWA cache scales linearly in slots (each slot is a
      per-sequence cache), and ``cache_bytes`` still takes the max against the layout's own
      product, which is what keeps the linear guess from under-reserving. A sliding-window
      cache has NO product to fall back on -- the layout's product has no window term, the
      reason ``plan_placement`` abstains on SWA without a floor at all -- so without the map
      the answer is ``None`` and the rung that asked is skipped rather than guessed;
    - a different context scales a non-SWA floor linearly in context and leaves an SWA floor
      flat, since a windowed cache is capped per slot by the window (measured 300 MiB per
      slot from n_ctx 9216 to 132096 on gemma-4-26B), which is the safe direction for both.
    """
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
        # A hybrid's floor is part fixed state, and the state does not shrink with the
        # context: scaling the whole scalar prices away memory llama.cpp still allocates
        # (a 5 GiB floor of 4 GiB state re-priced from 32768 to 8192 read 1.25 GiB
        # against a true 4.25, and the load then OOMs). Only the layout's own state can
        # be named here, so a floor the caller already stripped it from over-reserves by
        # that much on a shrink, which is the safe direction. ``kv_bytes_at`` is the way
        # to say it exactly.
        state = min(base, max(0, layout.recurrent_bytes) * want)
        base = state + (base - state) * n_ctx // requested_ctx
    return base


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
    """The MINIMAL set of ``units`` freeing at least ``deficit``, and what it frees.

    "Minimal" in the two senses that matter and in this priority: fewest units,
    then least overshoot. Overshoot is not free -- every byte moved beyond the
    deficit is a byte read back across the host link on every token that touches
    it -- which is the whole reason the ladder was split finer than a block.

    BACK_FIRST, the default, is a POSITION rule and so was blind to size: on a
    layout whose last block carries an 8 GiB FFN, a 100 MiB deficit took 4 GiB
    of it while a 128 MiB block sat one row down. The 98-cell measurement behind
    the order ran on layouts whose blocks are within a factor of two of each
    other, where the two picks are near enough the same bytes, so it says
    nothing about that case. The byte-minimal walk is therefore built alongside
    the tail and taken when either

      - the tail moves MORE THAN TWICE the bytes the minimal walk does. That is
        the rule, and it is a ratio rather than an overshoot bound so that it
        cannot fire on the sizes the measurement covers: it leaves every cell
        behind the 1.0073 median alone and only catches the size accident.
      - the caller is ranking on cost, where the smaller pick simply wins. Both
        candidates are one rung, so one access class and one rate: ``rank`` is
        monotone in bytes there and the byte comparison IS the ranking.

    Ties keep the tail, which is what preserves the measured asymmetry result.
    FRONT_FIRST is left as the benchmark's control that it is.
    """

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
    """``ffn_rung_order`` as the ladder walks it: known classes, first occurrence
    only. Each rung pass re-reads its units from the layout, so a class listed
    twice would be counted twice toward the deficit while ``_patterns_for``
    emits it once, and the plan would claim a fit the override does not free."""
    seen: list[SpillClass] = []
    for cls in opts.ffn_rung_order:
        if cls in FFN_SPILL_CLASSES and cls not in seen:
            seen.append(cls)
    return seen


def _grade_the_boundary_block(
    layout: ModelLayout, opts: PlanOptions, taken: list[SpillUnit], freed: int, deficit: int
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
        # A LATER rung closed the gap: the coarse rung ran out and dense or
        # shared-FFN units followed it. Trimming the last coarse block against a
        # deficit those units already cover keeps cheap expert bytes resident so
        # that dearer dense bytes can stay spilled, which reverses the ladder.
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
    """ "the ffn_down of every block plus the ffn_up/gate_up of 5 of 40 blocks".

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


def _effective_granularity(layout: ModelLayout, opts: PlanOptions) -> FfnGranularity:
    """The granularity actually used, after the optional quant-type rule.

    Resolved in ONE place because two sites consume it -- rung selection and the
    boundary grading -- and a rule applied at one but not the other would emit a
    plan whose rungs and whose boundary block disagree about what mode it is in.

    Returns ``opts.ffn_granularity`` unchanged unless
    ``granularity_from_quant`` is set AND the layout is a MoE whose quant types
    were recovered. Dense models keep the shipped default even with the flag on:
    the two dense models measured both win with the ladder at ratios that would
    put them below the threshold, so the rule is known not to describe them.
    """
    if not opts.granularity_from_quant:
        return opts.ffn_granularity
    ratio = moe_down_up_bpw_ratio(layout)
    if ratio is None:
        return opts.ffn_granularity
    return FfnGranularity.ALL if ratio >= _LADDER_BPW_THRESHOLD else FfnGranularity.BOUNDARY


def _rung_classes(layout: ModelLayout, opts: PlanOptions) -> tuple[Optional[SpillClass], ...]:
    """The per-block rungs, cheapest first, for this layout and these options.

    Falls back to the single coarse whole-FFN rung when the layout cannot be
    graded, so an architecture whose tensor names this module does not recognise
    degrades to the pre-ladder behaviour instead of to spilling nothing.
    """
    spillable = [b for b in layout.blocks if b.spillable_bytes > 0]
    graded = bool(spillable) and all(b.graded for b in spillable)
    if _effective_granularity(layout, opts) is FfnGranularity.ALL and graded:
        rungs: list[Optional[SpillClass]] = list(_ordered_rungs(opts))
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


def _fit_boundary_overflow(block: BlockLayout, deficit: int) -> Optional[int]:
    """FFN bytes ``common/fit.cpp`` overflows off its boundary layer to cover
    ``deficit``, or None when even the whole FFN of it does not.

    Step 4 of the fitter keeps ONE more layer on the device and overrides part of
    that layer's FFN to the host instead of lowering ``n_gpu_layers`` again, so
    the layer's attention, cache and recurrent state stay resident. It tries
    ``LAYER_FRACTION_UP`` (``ffn_(gate|gate_up|down)``, keeping ffn_up), narrows
    to ``_GATE`` (``ffn_down`` alone) when that still fits and widens to ``_ATTN``
    (the whole FFN) when it does not. The three nest, so the smallest that covers
    is what that trial order arrives at, and that is what this walks. A layout
    with no per-rung breakdown has only the whole-FFN candidate, which is the
    ``_ATTN`` one.
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
    """What llama.cpp's own fitter would place here, priced the same way.

    ``n_seq`` is the slot count the child serves: the recurrent state is one copy
    per sequence, resident and moved alike, so the fitter modelled at one copy
    reaches the budget at the wrong layer count on a multi-slot hybrid.

    ``kv_layer_weights`` is each layer's relative cache size, the same vector the
    per-device check takes. Without it the cache was spread evenly over every
    block, which charges a recurrent or MLP-only row a share it does not hold and
    so over-charges the fitter on exactly the hybrid and iSWA layouts where the
    gate has been wrong.

    This is the arm the planner is really competing against. When the planner
    abstains the launch path emits ``--fit on`` and no ``-ngl``, and fitting
    lowers the offloaded layer count until the load fits -- moving WHOLE layers,
    attention and FFN together, and dragging each moved layer's share of the
    attention cache to the host with it (``-ngl`` drags the cache off, ``-ot``
    does not). The output tensor rides the layer list at ``n_layer_all``, and
    that row is the LAST to leave, not the first: llama.cpp keeps rows
    ``[i_gpu_start, i_gpu_start + act_gpu_layers)`` with
    ``i_gpu_start = max(n_layer_all + 1 - n_gpu_layers, 0)``
    (llama-model.cpp:1467-1492), so row ``n_layer_all`` -- and therefore
    ``lm_head`` -- stays on the device for every ``n_gpu_layers >= 1``. Every
    placement this loop can return is a partial fit with at least one layer
    resident, so lm_head is resident in all of them.

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
    # The cache follows the layer, so a layer moved to host takes its share with
    # it. Per-layer rather than per-attention-layer: SWA already makes the
    # planner abstain upstream, so the shares are uniform by the time we get here.
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

    # The cache is RESERVED at n_ctx but only the live prefix is ever read, and
    # reading is what costs. Pricing the reservation charges a 32K allocation for
    # a 2K conversation, which is the single biggest thumb on the scale in favour
    # of spilling: it makes any placement that moves cache look catastrophic and
    # the planner's own -ot placement look free by comparison. The reporter on
    # #9861 flagged the same gap from the measurement side -- 32768 allocated,
    # about 2.2K ever live. Feasibility above still uses the full reservation,
    # because llama.cpp really does allocate it.
    # A request lives in ONE slot's window: the whole context under a unified
    # cache, n_ctx / slots without one (the same bound the cost gate puts on the
    # prompt). Capping at the total context priced live cache a slot cannot hold
    # once the prompt already filled its window, and only the fitter's moved-cache
    # arm carries that term, so the over-charge bought spills the fitter beats.
    slot_window = n_ctx if opts.kv_unified else n_ctx // max(1, n_seq)
    live_tokens = min(
        max(1, slot_window), max(1, opts.workload_prompt_tokens + opts.workload_generated_tokens)
    )
    # Scaled by whatever correction ``kv_bytes_floor`` applied to the RESERVED
    # size, so both sides of this function describe one cache. The floor is the
    # caller's byte-accurate measurement at the requested context, and where it
    # exceeds the layout's plain f16 GQA product it is carrying a real term the
    # product has no expression for -- cache dtype, SWA's second cache, MLA's
    # compressed latent, slot padding, flash-attention padding (see
    # :func:`cache_bytes`). Those terms scale with the live prefix exactly as
    # they scale with the reservation, so reading the live size off the bare
    # product while the freed size uses the floor made the fitter FREE a
    # correctly sized cache and be CHARGED for an undersized one. On MLA -- the
    # huge-MoE shape this planner exists for, where the product models a K-only
    # latent as a full K+V pair -- the two differ by more than 2x.
    reserved_product = cache_bytes(layout, n_ctx, kv_quantised = quantised)
    floor_scale = (kv_total / reserved_product) if reserved_product > 0 else 1.0
    if kv_on_host:
        kv_live_total = 0
    elif layout.has_swa and kv_bytes_floor > 0:
        # The measured floor of a windowed cache is context-FLAT once the window
        # is saturated (``cache_bytes`` returns it unscaled for that reason), and
        # the layout does not say how it splits between the windowed layers and
        # the few full-context ones. Scaling all of it by the live fraction
        # charged the fitter a fraction of a cache it reads in full -- 4x short
        # on a gemma-shaped 5.5 GiB cache at a 2K prompt -- and declined spills
        # the hardware wins. The whole floor as live over-charges only the
        # full-context share, the direction a gate that has been wrong the other
        # way can afford.
        kv_live_total = kv_total
    else:
        kv_live_total = int(cache_bytes(layout, live_tokens, kv_quantised = quantised) * floor_scale)

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
        # Every expert on the host and still short. common/fit.cpp does not fail
        # here: with the experts gone it lowers n_gpu_layers like the dense fitter
        # below, moving whole LEADING layers with their attention, dense FFN,
        # cache share and recurrent state (fit.cpp: the `hp_nex == 0 ||
        # global_surplus_cpu_moe <= 0` branch sets ngl and returns). Answering
        # None here let the planner's lm_head rung through with no comparison at
        # all, and a large vocabulary head can cost more than the few layers the
        # fitter moves instead.
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

    # Dense, where the whole-layer model IS what happens: measured n_part=0 with
    # n_layer 54 of 65 and 38 of 65, no overrides at all, and the cache off the
    # GPU with it. llama.cpp keeps the LAST n_gpu_layers on the device -- the
    # fitter only lowers n_gpu_layers (common/fit.cpp:551-559) and
    # ``i_gpu_start = max(n_layer_all + 1 - n_gpu_layers, 0)`` sends every row
    # BELOW it to the CPU (llama-model.cpp:1479-1484) -- so the host takes the
    # LEADING blocks and this loop walks them in ascending index order.
    #
    # It used to walk from the end, on the grounds that only the COUNT enters the
    # cost. That is true only for a layout whose blocks are all the same size, and
    # the byte terms below are per block: a dense recurrent/hybrid interleaves
    # attention and SSM blocks of materially different sizes, so a prefix and a
    # suffix of the same length are different numbers of bytes. The loop could
    # therefore stop at the wrong layer count and price the wrong host weights for
    # the Falcon-H1 / Nemotron-H shapes this branch now serves, in either
    # direction. The MoE branch above keeps ``reversed`` deliberately: there the
    # fitter keeps every layer and overrides the TRAILING ones' experts
    # (common/fit.cpp:563-577), which is the opposite end.
    #
    # The recurrent state follows its layer exactly as the cache does: for layer
    # ``i`` llama.cpp takes ``ggml_backend_dev_buffer_type(model.dev_layer(i))``
    # when offload is on and the CPU buffer otherwise
    # (llama-memory-recurrent.cpp:85-89, the same branch llama-kv-cache.cpp:214-225
    # uses for the attention cache), which is what ``ModelLayout.recurrent_bytes``
    # already documents. So a layer the fitter moves takes its share of the state
    # to the host, and the fallback has to free it and then pay for it.
    #
    # It did neither. ``all_resident_bytes`` counts the whole state in
    # ``resident`` and this loop never freed any of it, so on a single-GPU dense
    # hybrid -- which is not caught by the uneven-cache abstain, that one returns
    # early on ``len(vram_bytes_per_device) <= 1`` -- the fallback had to move
    # more layers than the fitter would to reach the same budget, and was scored
    # on that heavier placement. Worst case it ran out of layers and returned
    # ``None``, which the gate reads as "the fitter cannot place this either" and
    # takes the spill WITHOUT any comparison at all. Reachable on any dense
    # Mamba-hybrid GGUF that carries ``ssm.*`` and ``full_attention_interval``
    # (Falcon-H1, Nemotron-H) on one card.
    #
    # Spread over every block, not over the recurrent layers alone, because the
    # layout does not say which layers are recurrent -- the same approximation
    # ``kv_per_layer`` already makes for a cache that likewise lives on only some
    # of them.
    #
    # Zero under ``-nkvo``, on the same terms as ``kv_live_per_layer`` above.
    # ``llama_memory_hybrid``'s constructor hands the SAME ``offload`` flag to the
    # attention cache and to the recurrent memory, and that flag is
    # ``cparams.offload_kqv`` where ``llama_model::create_memory`` builds the
    # hybrid, so ``--no-kv-offload`` puts the whole recurrent state on the host
    # before any layer moves: ``llama_memory_recurrent``'s ctor takes
    # ``ggml_backend_cpu_buffer_type()`` unless ``offload``. Read at llama.cpp
    # ``90c26fc``; the symbols are the citation, not the line numbers, which have
    # already drifted twice in a week on the sibling PR.
    # ``resident_floor_bytes`` already says
    # so and leaves ``recurrent_bytes`` out of ``resident`` on that branch, so
    # freeing a per-layer share here frees bytes that were never counted -- the
    # modeled fitter reached the budget two or three layers early on a
    # Nemotron-H-shaped hybrid -- and then billed a host recurrent group that
    # BOTH placements pay, which is state common to the two arms and cancels.
    recurrent_per_layer = (
        0.0 if kv_on_host else layout.recurrent_bytes * max(1, n_seq) / len(blocks)
    )

    def dense_placement(moved: int, spilled_ffn: int, attention: int) -> Placement:
        """``moved`` whole layers off the device, plus ``spilled_ffn`` FFN bytes."""
        groups: list[TensorGroup] = []
        if spilled_ffn:
            groups.append(_ffn_group(layout, spilled_ffn))
        if attention:
            # Attention, norms and routers: dense, read in full every token.
            groups.append(TensorGroup("layers", attention, Access.CONTIGUOUS))
        host_recurrent = int(recurrent_per_layer * moved)
        if host_recurrent:
            # CONTIGUOUS, not the cache rate: this is a small fixed-size conv
            # and SSM state read straight through by the scan, not attention
            # over a prefix that grows with the conversation. It is also
            # context independent, so unlike the cache there is no reserved
            # versus live distinction to make.
            groups.append(
                TensorGroup("recurrent (moved layers)", host_recurrent, Access.CONTIGUOUS)
            )
        live_kv = kv_freed(kv_live_total, moved)
        # ``kv_host_bytes``, so this is charged at ``Access.KV_CACHE``'s
        # calibrated 20.1x and not at the contiguous weight rate. This is the
        # ONLY term that can see the planner's measured dense advantage, and
        # an earlier revision charged it at 1.00x on the reasoning that a
        # moved layer's attention runs on the CPU backend next to its own
        # cache, so it never crosses the link. The residency claim is right
        # (llama-kv-cache.cpp:214-225 gives layer ``il``'s cache the buffer
        # type of ``model.dev_layer(il)``, so a host layer's cache is a host
        # buffer), but the RATE that follows from it is not: 1.00x is
        # ``REFERENCE_CONTIGUOUS_MS_PER_GIB``, measured on Q4 dense FFN, a
        # contiguous quantised GEMM. Batch-1 attention over an f16 cache is a
        # strided GEMV that parallelises over heads (4 KV heads here) rather
        # than over rows, so it cannot reach that rate, and nothing in the
        # model measures 1.00x for it. Charging it 1.00x made the model
        # contradict every dense measurement we have: over the bench13 cells
        # that reach this gate the fitter scored CHEAPER on 5 of 5 dense
        # cells the hardware says the planner WINS, by 1.08x to 1.53x on
        # generation, which would make ``-ot`` a MoE-only feature.
        #
        # 20.1x is the model's only calibrated constant for "attention cache
        # read from host RAM", and it is the best-anchored one in the file
        # (dense 119.7 against MoE 121.2 ms/GiB, agreeing to 1.3%). It is an
        # UPPER bound for this regime, because that anchor is
        # ``--no-kv-offload``, where the layer's weights stay on the GPU so
        # the attention op stays there too and drags the cache over PCIe
        # every token -- ggml only runs an op where its sources live for
        # sources in a WEIGHTS buffer, and the cache is not one, with
        # FLASH_ATTN_EXT excluded from that rule outright
        # (ggml-backend.cpp:952-981). The true rate for CPU-side attention
        # over a host cache is between 1.0x and 20.1x and is unmeasured.
        #
        # Taking the upper bound is the conservative end for a gate that has
        # already been wrong in the other direction, and it still
        # UNDER-predicts: it scores those five at 1.12x to 1.19x against a
        # measured penalty ratio of 1.43x to 2.44x where a resident baseline
        # exists to compute one (L4, Qwen3.8-27B Q4, 12/14/16 GiB). It
        # also cannot rescue a cell it should not by much, because the term
        # scales with how many layers the fitter has to move: where the load
        # nearly fitted and the fitter moves almost nothing -- the #9861
        # shape this gate exists to stop -- it contributes almost nothing.
        # The two worst #9861 cells (Qwen3-8B Q4 near 11 GiB, measured 0.19x
        # and 0.21x) decline at every thread count and context tried.
        #
        # It is NOT clean on the measured set, and that is stated here rather
        # than in a commit message nobody reads at this line. Scored over the
        # bench13 cells that reach this gate: 5 correct accepts, 0 false
        # declines, and 2 FALSE ACCEPTS -- both gemma-4-E2B Q4 at 2 GiB,
        # which is also the one cell that disagrees with itself across hosts
        # (0.968 and 0.878 for the same placement). The threshold is not what
        # is wrong there: E2B scores 1.465, HIGHER than every correct accept
        # (1.12 to 1.19), so the ordering is wrong on that cell and no margin
        # separates it. Left as a known miss rather than tuned away, because
        # fitting a constant to two readings that disagree by 10% would be
        # fitting noise.
        return Placement(host_groups = groups, kv_host_bytes = live_kv)

    host_weights = 0
    host_spillable = 0
    for moved, block in enumerate(blocks, start = 1):
        # fit.cpp's step 4 first: rather than lower ngl again it keeps this layer
        # on the device and overrides part of its FFN to the host, so the layer's
        # attention, cache and recurrent state stay resident with it. Tried before
        # the whole-layer move because that is the order the fitter settles them
        # in, and it is the difference between a boundary layer that was needed in
        # part and one charged in full.
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
        # lm_head is NOT in here: the output row stays on the device for any
        # n_gpu_layers >= 1, so the fitter never frees it on a partial fit and
        # never pays for it on the host either. Charging it did both -- the
        # fallback appeared to fit a layer or two early AND was billed a host
        # lm_head llama.cpp would not move, which inflated its score and let
        # spills through this gate that the real fitter beats.
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
    """Attention cache to reserve, never below a caller-supplied measurement.

    ``layout.kv_bytes`` is a plain f16 GQA product: heads times key+value width
    times context. It has no cache-dtype, SWA, MLA, unified-stream, slot-padding
    or flash-attention-padding term, so against a caller that has priced the real
    cache it can land either side. Over is harmless -- the plan just reserves
    more. UNDER is the dangerous direction: the deficit comes out too small, too
    few blocks are spilled, and the launch path follows that with ``--fit off``,
    so the server OOMs on a cache the caller had already sized correctly.

    Taking the maximum keeps the planner conservative in both directions without
    a tolerance to tune, on the plain GQA shapes where the product is at worst a
    little short. Two shapes break the product in the UP direction by so much
    that the maximum would throw the measurement away, and there the supplied
    floor wins outright: sliding-window attention (below) and MLA, where the
    cache is one compressed K-only latent per token that the per-head K+V product
    models as a full pair per head, about 70x over on DeepSeek-V3 at f16. Under
    the maximum a fully resident MLA load read as several GiB over budget and was
    shrunk or spilled for nothing; the launch seam's estimator prices the latent
    exactly (_estimate_kv_cache_bytes handles kv_lora_rank). The floor is a
    measurement at the REQUESTED context, so where a shrink rung re-prices at a
    smaller context it over-reserves; that is the safe direction and at worst
    gives up a rung.
    """
    naive = layout.kv_bytes(n_ctx, _kv_elem_bytes(kv_quantised))
    floor = max(0, kv_bytes_floor)
    if trust_floor:
        # ``PlanOptions.kv_bytes_at`` priced THIS context and slot count, so there is
        # nothing left for the product to correct at any architecture. Keeping the max
        # here would put a plain GQA hybrid back on the product it over-counts by 3.6x.
        return floor
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
        # where the product's failure mode is under-counting and the measurement
        # is the thing that might be short.
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
    """VRAM needed with EVERY spillable tensor already on the host.

    Attention weights, norms, routers, shared experts, the recurrent state, the
    cache and lm_head. Below this, ``-ot`` has nothing left to give and only a
    smaller quant or less context can help.

    ``n_seq`` is the slot count: the recurrent state is one copy PER SEQUENCE
    (offload_layout.py documents it as such), so a caller that knows its slot
    count charges it that many times. 1 keeps the old arithmetic.
    """
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
    """Largest context whose cache fits, rounded down to 256 as CUDA wants.

    ``outside_layout_bytes`` is what sits on the card outside the layout after the
    rungs the caller intends to apply; ``None`` charges the projector and the draft
    in full, as every call that predates those rungs meant.

    ``kv_bytes_floor`` is the caller's measured cache at ``floor_ctx``; without it
    the layout's f16 product sizes the cache, which is the product the rest of
    this module refuses to trust on a sliding-window or MLA model. With it, a
    non-SWA floor scales linearly in context and an SWA floor is flat, and the
    larger of floor and product is charged, exactly as ``cache_bytes`` does --
    except on MLA, where the floor wins outright on both sides.

    ``opts.kv_bytes_at`` replaces all of that: it prices each candidate context
    directly, and the search is then bounded by the training window and the
    reserve rather than by a per-token product that does not describe the cache.

    The reserve is a function of the context and the context is what is being
    solved for, so this is a search rather than a division: the predicate
    "fixed bytes plus the cache at ctx fit under the reserve at ctx" is monotone
    in ctx, and a binary search over 256-multiples finds its largest true value.
    """
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
            # -nkvo puts both caches in host RAM, so no context charges VRAM here.
            return 0
        if opts.kv_bytes_at is not None:
            return _measured_cache_at(layout, opts, ctx, n_seq)
        naive = per_token * ctx
        if floor <= 0:
            return naive
        if layout.has_swa:
            # Flat for the WHOLE floor, and only because one scalar cannot say which
            # layers hold the full-context half; those layers really do grow, so a
            # shrink here frees nothing the caller can see. ``kv_bytes_at`` splits it.
            return floor
        scaled = floor * ctx // floor_at if floor_at > 0 else floor
        if layout.has_mla:
            # ``cache_bytes`` trusts the floor outright on MLA -- the per-head product
            # models a K-only latent as a full K+V pair, ~70x over -- so taking the max
            # here made the two disagree, and every MLA model that needed a shrink was
            # refused at a context this budget holds.
            return scaled
        return max(naive, scaled)

    def usable(ctx: int) -> int:
        return _usable_vram(
            vram_bytes_per_device, opts, ctx, outside_layout_bytes = outside_layout_bytes
        )

    def fits(ctx: int) -> bool:
        return fixed + cache_at(ctx) <= usable(ctx)

    # Upper bound: the answer with no reserve growth at all, which nothing can exceed.
    top = usable(0) - fixed
    if top <= 0:
        return 0
    # ``top // per_token`` bounds the answer only while the cache really grows with
    # the context. It does NOT when ``cache_at`` above is flat: -nkvo holds both
    # caches in host RAM (llama.cpp allocates the KV buffers on the GPU unless
    # --no-kv-offload is given), and a windowed cache is capped by the window
    # (llama_kv_cache_unified_iswa sizes the SWA half at n_ctx_swa, not n_ctx), which
    # is why this file charges a measured SWA floor context-flat in the first place.
    # Bounding a flat cache by the naive product cuts the search off far below the
    # real answer -- the measured example in this module has that product 13x the
    # real SWA cache -- so the only context-linear term left, the reserve, sets the
    # bound instead.
    # Priced, not products: a caller callable and an MLA floor both describe a cache
    # the per-token product is not a bound for, so they take the branch below too.
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
            # Nothing grows with the context at all; only training length caps it.
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
        # One pool: "spilling" renames bytes on the same chips and frees nothing.
        # Metal also keeps mmap zero copy (buffer_from_host_ptr), so the no-mmap
        # rule inverts there too.
        return Plan(reason = "unified memory host, spilling frees no device memory")
    # Settle the context first: the per-device reserve has a context-linear term, so the budget
    # is a function of n_ctx and cannot be computed above it.
    # An EXPLICIT request is priced as asked, above the training window included:
    # llama-server serves a -c above n_ctx_train (with a warning, and RoPE scaling
    # where the user set it), so a plan clamped to the window prices a cache the
    # child does not run at, and the seam then either rewrote an explicit -c the
    # launch promised to honour or, with the context in the extras, launched
    # --fit off with a spill sized for the smaller cache. Only the default (no
    # request) reads the window; the FIT_ONLY ladder still tops out at it.
    n_ctx = requested_ctx if requested_ctx > 0 else layout.n_ctx_train
    if n_ctx <= 0:
        return Plan(reason = "no usable context length")

    if opts.kv_bytes_at is not None:
        # One cache size for the whole function. A no-op without the callable, since
        # ``_kv_floor_at`` at the requested context and the caller's own slot count
        # returns the floor it was handed.
        kv_bytes_floor = (
            _kv_floor_at(layout, opts, kv_bytes_floor, n_ctx, n_ctx, max(1, opts.n_parallel)) or 0
        )

    # A policy that may shrink prices the budget again at every candidate context, so
    # this is not its answer: the reserve has a context-linear term, and an 8 GiB card
    # asked at 131072 can be left with nothing while 32768 fits resident. Returning here
    # meant the ladder that exists for exactly that case never ran. NEVER_REDUCE has no
    # candidate but the context it was asked for, so it still stops here.
    may_shrink = opts.context_policy in (ContextPolicy.FIT_ONLY, ContextPolicy.PREFER_RESIDENT)
    budget = _usable_vram(vram_bytes_per_device, opts, n_ctx)
    if budget <= 0 and not may_shrink:
        return Plan(reason = "no creditable VRAM after per-device overhead and reserved allocations")
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
    # This branch runs ahead of _kv_modes, so the cache the child already carries
    # has to be priced here as it is there: an f16 product over a q8 floor shrank
    # a context that fit fully resident with the cache type the child runs.
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
            # The feasibility above charged one recurrent state per slot; the plan
            # has to be assembled at the same count, and at the floor re-priced for
            # the context it settled on, or vram_bytes under-reports a hybrid by
            # (slots - 1) recurrent states.
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
        # The gate refused a FEASIBLE plan. Context is the last thing to give up,
        # but it is the only lever left, so keep the refusal and try below. A veto
        # is not that: it is a measurement the context does not move, and shrinking
        # under it walked down to the first per-slot context below the long-prompt
        # point and took the SAME spill there.
        declined = declined or plan

    # Nothing fit at the requested context, or the gate refused it. Only now may
    # FIT_ONLY shrink, and it walks DOWN from the largest feasible context in
    # ctx_step increments so the first context the gate accepts is the largest
    # one: feasibility is monotone in context, acceptance is not (a smaller
    # deficit changes the spill set on both arms), so a binary search on the
    # gate could land on a smaller accepted context than exists.
    if may_shrink:
        step = max(256, opts.ctx_step // 256 * 256)
        # The bound has to assume every rung above the first weight spill is
        # applied, or it is not an upper bound for what _plan_at will retry: with
        # the projector and the draft still charged and the cache at the full slot
        # count it can sit below a context that fits once rung 0 fires, or at zero,
        # and the ladder then never looks. Over-estimating only costs steps.
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
                # The requested context was feasible and refused; start below it.
                hi = (n_ctx - step) // 256 * 256
            rungs: list[int] = []
            ctx = hi
            while ctx >= opts.min_ctx:
                rungs.append(ctx)
                ctx -= step
            # The lattice steps down from the top and lands on min_ctx only by
            # coincidence: a refused 8960 minus a 1024 step is 7936, below an 8192
            # minimum, and the ladder never asked about 8192 at all. The minimum
            # is the last rung whenever it is feasible and not the refused request.
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
        # The ladder ran and no context fit either. The requested one is still the one
        # the reserve leaves nothing of, and reporting a floor against a zero budget
        # would only describe the context nobody can have.
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
    """Which of the ``n_slots`` layer rows land on which device.

    Mirrors llama.cpp's default tensor split exactly: free VRAM per device
    (llama-model.cpp:1420-1433), prefix-summed and normalised (:1439-1447), then
    ``upper_bound`` on the normalised row index (:1457). Row ``n_layer_all`` is
    the output row (:1467). With every layer offloaded ``i_gpu_start`` is 0 and
    ``act_gpu_layers`` is ``n_layer_all + 1``, which is ``n_slots`` here.

    The arithmetic is done in float32 because llama.cpp's is: ``tensor_split`` is
    a ``float`` array and ``il / n_layer`` is computed in single precision, so at
    353 rows over a 39407:12114 split the boundary row moves by one against a
    double-precision transcription, and that one row can carry a 2 GiB block.
    """

    def f32(value: float) -> float:
        return struct.unpack("=f", struct.pack("=f", value))[0]

    weights = [max(0, v) for v in split_weights]
    total = sum(weights)
    if total <= 0:
        # llama.cpp prefix-sums the split and divides by the total, so an all-zero
        # one is not a placement it produces: it errors. Answering "device 0 takes
        # every row" put the per-device check on a split that will never exist,
        # and a -ts truncated to its active prefix reaches this with all zeros.
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
    """Bytes each device would hold under this spill, and the rows it owns.

    ``(error, usage, slots)``: ``error`` says why the split cannot be modelled at
    all, in which case the other two are empty. ``spilled`` is either BYTES moved
    per block index (what a rung took, which may be part of a block) or a bare
    collection of indices, meaning the whole of each block's spillable FFN.
    ``extra_on_device0`` is what rungs 0 to 2 have left outside the layout on the
    main device; ``None`` charges the options' scalar as before.
    """
    if len(vram_bytes_per_device) <= 1:
        return None, [], []
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
    # The recurrent state is one copy per slot on the rows that hold no cache,
    # and the pooled fit charges it in full. A vector with zero-weight rows says
    # which rows those are; spread the state over them, ceiling so no device is
    # under-booked. Without any such row the state cannot be placed, and the
    # abstain that applied before a vector was supplied applies still.
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
        # A split the planner cannot model is an abstain, not a guess.
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
            # Whatever this rung did NOT take off the block is still on the card.
            # Subtracting BYTES rather than skipping a whole block is what keeps
            # this honest once a rung can move a third of an FFN: crediting the
            # whole block for a partial move is the optimistic direction, and a
            # per-device shortfall is a hard throw rather than a slow load.
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

    The reserve is charged against the RAW device size, not folded into a
    clamped headroom: a secondary device that owns no rows still has to hold its
    pipeline buffers, and ``max(0, vram - reserve)`` would wave that through.
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
    """A spill chosen device by device, or ``None`` when one cannot be shown to fit.

    The pooled ladder picks the fewest units that close the POOLED deficit, but
    -ot does not move a layer between devices, so a pick that happens to sit in
    one device's row range relieves only that device. This walks the same ladder
    per device instead: each device's own deficit (its rows' bytes plus its
    fixed reserves, against its raw size) is closed with units drawn from ITS
    rows, cheapest rung first, and the result is re-checked as a whole. A device
    that cannot close its deficit from its own rows fails the selection, which
    the caller turns into the lm_head rung or an abstain.
    """
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
        # The same grading the pooled ladder applies, per device: each device's
        # last whole block is its own boundary, and left whole it is up to a full
        # block of host traffic per device that nothing needed.
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
    """One pass of the ladder at a fixed context and cache dtype.

    The budget is computed HERE, not by the caller: it is a function of the
    context (the per-device reserve has a context term) and of what rungs 0 to 2
    have taken off the card, so it changes inside this function.

    Rungs 0 to 2 come first because each is free per token: the projector runs
    once per image, a slot is a cache the caller is not using, and the draft is
    priced by ``draft_drop_penalty_frac``. Each stops the moment the load fits.
    Only when all three are exhausted does the first weight leave the device.
    """
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

    # Rung 0: the projector.
    if needed > budget and opts.mmproj_movable and opts.mmproj_bytes > 0:
        knobs = _Knobs(knobs.n_parallel, True, knobs.draft_dropped)
        needed, budget, floor = price(knobs)  # type: ignore[misc]
    # Rung 1: one slot at a time. A step the floor cannot be re-priced for ends it.
    # Not across a layer split on a windowed cache: the per-layer cache weights
    # were measured at the caller's slot count, and under iSWA the windowed
    # layers grow with the slots while the full-attention ones do not, so the
    # per-device check would split a re-priced total by ratios that no longer
    # hold and could pass a card that then fails allocation under --fit off.
    #
    # A step that leaves ``needed`` where it was (a flat floor map, or a unified
    # cache with no recurrent state) buys concurrency away for nothing: it is
    # undone and the rung ends there.
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
    # Rung 2: the draft.
    if needed > budget and opts.draft_droppable and opts.draft_bytes > 0:
        knobs = _Knobs(knobs.n_parallel, knobs.mmproj_to_host, True)
        needed, budget, floor = price(knobs)  # type: ignore[misc]

    if needed <= budget:
        if n_devices > 1:
            # A pooled fit is not a per-device fit, and a plan that spills nothing
            # is still emitted as ``-ngl -1 --fit off`` whenever it reshapes the
            # launch (llama_cpp.py:_spill_plan_flags_for): a knob it gave up, a
            # context the ladder shrank, or a context the seam restores above the
            # one Auto capped to, which this function cannot tell from a plain
            # fit. All of those take llama.cpp's own per-device fitter out of the
            # loop, and a card that is still over then throws on load ("unable to
            # allocate %s buffer" in llama_model_base::load_tensors) rather than
            # loading slowly. Spilling plans already run this check; run it for
            # every fit across a split.
            uneven = shortfall(knobs, floor)
            if uneven is not None:
                # One card is over on a load the pool fits: the lopsided pair the
                # row split lands too many bytes on. Handing it straight back gave
                # up on it before trying anything, so walk rungs 0 to 2 here as
                # well -- each relieves a device without moving a weight -- and
                # then let the per-device selector move that card's own rows.

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
                # Rungs spent and the card still over: spill from ITS rows. This is
                # the one partial multi-device pick whose row arithmetic is known,
                # and _select_units_per_device re-checks every device itself.
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
        """Check ``units`` device by device, then either finish or say why not.

        ``per_device_selected`` marks units that _select_units_per_device chose
        from each device's own rows, which is the one partial multi-device spill
        whose row arithmetic IS known; every other partial pick abstains below.
        """
        moved = _moved_by_index(units)
        full = all(moved.get(b.index, 0) >= b.spillable_bytes for b in spillable)
        if (
            n_devices > 1
            and not full
            and not per_device_selected
            and not opts.trust_device_row_model
        ):
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
            # The pooled pick relieves whichever devices its rows happen to sit
            # on. Re-select per device from each device's own rows; the pooled
            # deficit must still be covered, since the budget is what the cost
            # gate and the seam were priced against.
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
    """The one hard host-RAM admission check, shared by every plan that moves
    bytes onto the host.

    A spilled weight can at least page through mmap; a projector pinned to the
    CPU cannot, since clip.cpp allocates it in a CPU backend buffer of its own,
    so both are refused at the same line rather than only the one the cost gate
    happens to score.
    """
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
    refused = _host_ram_refusal(opts, n_ctx, host_bytes, host_ram_bytes)
    if refused is not None:
        return refused, 0.0, 0.0
    if opts.prompt_cache_unbounded:
        # --cache-ram -1 bounds the prompt cache by nothing, so the host side can
        # never be proved resident and an accepted spill launches under mmap
        # (_finish keeps it pageable). Every host-side number the ranking below
        # rests on was measured with the weights UNMAPPED, and mapped reads run 2
        # to 4.6x slower, so a spill that wins here can lose on the launch it
        # gets. There is no pageable cost model to score it with; decline.
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
    # One shared stream under --kv-unified, so the window a single request may
    # fill is the whole n_ctx however many slots are served; N private windows
    # of n_ctx / N without it.
    per_slot_ctx = n_ctx if opts.kv_unified else n_ctx // n_slots
    if layout.is_moe and opts.moe_long_prompt_ctx > 0 and per_slot_ctx >= opts.moe_long_prompt_ctx:
        # MEASURED, and the cost model cannot see it: -ot on an MoE loses to
        # llama.cpp's layerwise fit at a 32K prompt (0.94 to 0.97x on 5 cells, 2
        # models, 3 hosts: gemma-4-26B-A4B on A100 and G4, Qwen3.6-35B-A3B on
        # A100 and L4) while winning 1.05 to 1.08x at 2K. Both arms keep the
        # cache resident on MoE, so there is no KV advantage to grow with the
        # prompt, and the planner's placement only costs graph splits at the
        # prefill-bound end. The fallback below is priced through the same -ot
        # mechanism, so rank() can only ever call this a near-tie.
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
        # Nothing to COMPARE to, which is not the same as "the fitter cannot place
        # this load": on a MoE where moving every expert is still short,
        # common/fit.cpp does not fail, it simply stops after step 3 with fewer
        # dense-only layers on the device (fit.cpp: `if (hp_nex == 0 ||
        # global_surplus_cpu_moe <= 0) { set_ngl_tensor_split_tbo(...); return; }`),
        # and that placement is not modelled here.
        #
        # So this is a DECLINE, not an accept. Passing the spill through unranked
        # took it on exactly the layouts whose arithmetic is least trustworthy --
        # the ones this loop cannot walk -- and the gate exists because an unranked
        # spill measured up to 8x slower than the fit it replaced. llama.cpp places
        # what this cannot price.
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

    # The workload is a request, and a request does not get longer because the
    # server takes fewer of them at once: rung 1 lowering the slot count leaves
    # it alone. (It was scaled by the old/new slot ratio here, which at four
    # slots to one quadrupled the prompt, charged prefill once per micro-batch
    # of a request that never launches, and swung the verdict.) What a slot
    # count does bound is the window a slot can serve, the whole context under
    # a unified cache and n_ctx / slots without one, so the prompt is capped
    # there and nowhere else.
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
        # Rung 2 gave up the draft to make room for this spill; the fitter keeps
        # it. Charge the plan what that costs, so a spill that only fits because
        # the draft went is scored against the draft's own speed.
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
    """The same ranking for a plan that spills nothing but gave a knob up.

    ``_cost_gate`` prices a SPILL, so a plan that only pins the projector, drops
    the draft or serves fewer slots was never ranked at all and
    ``draft_drop_penalty_frac`` applied to no comparison. The arm to rank it
    against is the launch as the caller typed it, fitted by llama.cpp: the same
    fallback the spill path uses, at the budget the knobs have NOT relieved,
    since the fitter keeps the projector and the draft on the card.

    ``rank`` prices deviation from a fully resident launch rather than a request
    time, so a plan that moves nothing scores 0 and there is no absolute figure
    for the draft's fraction to take a share of. The fallback's own cost is the
    only request-scale number here, so the fraction is charged against it: the
    draft is worth giving up unless keeping it is worth more than the whole of
    what the alternative costs. At the shipped 0.05 that accepts every knob-only
    plan the fitter has real work to do on, which is the measured answer; what
    changes is that the two figures are now computed and reported instead of
    left at zero.
    """
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
        # Nothing to compare to, exactly as in _cost_gate.
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
    knobs: Optional[_Knobs] = None,
    kv_layer_weights: Sequence[int] = (),
    requested_ctx: int = 0,
    reason: str = "",
) -> Plan:
    """Assemble patterns, decide the load mode, and account for both sides.

    ``requested_ctx`` is the context the caller asked for; a plan at a shorter one
    is a change in its own right even when it spills nothing, since the launch
    has to carry the shorter ``-c``.

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
    # Rung 0 did not make the projector disappear, it moved it: --no-mmproj-offload
    # clears mmproj_use_gpu and clip.cpp then allocates the projector in a CPU
    # backend buffer, so those bytes are HOST RAM for the life of the server. They
    # were charged to the card before the rung fired and to nothing after it, which
    # is exactly the term the two host-RAM decisions below spend: the hard refusal
    # that keeps a spill out of swap, and the --cache-ram clamp. A 1 GiB projector
    # under a 2 GiB headroom is enough to turn a refusal into a --load-mode none
    # that cannot be paged.
    #
    # The whole of ``mmproj_bytes``, file and runtime surcharge together: on the
    # host the projector needs its weights AND the buffers its graph runs in, and
    # the surcharge is the caller's measured allowance for the second
    # (_MMPROJ_VRAM_SAFETY, ~1.3x runtime over file size).
    mmproj_host_bytes = opts.mmproj_bytes if (knobs is not None and knobs.mmproj_to_host) else 0
    # -nkvo puts the cache and the recurrent state in host RAM for the life of the
    # server, so they are part of the host side every decision below spends: the
    # refusal that keeps a spill out of swap, the mmap decision and the --cache-ram
    # clamp. The refusal saw only the weights, so a spill that fits the host with
    # the cache left out was admitted onto a box the cache had already filled.
    # One recurrent state per slot, the same count the floor was measured at.
    kv_host_bytes = (
        cache_bytes(layout, n_ctx, kv_quantised = quantised, kv_bytes_floor = kv_bytes_floor)
        + layout.recurrent_bytes
        * max(1, knobs.n_parallel if knobs is not None else opts.n_parallel)
        if kv_on_host
        else 0
    )
    host_side = layout.token_embd_bytes + spilled_weight_bytes + mmproj_host_bytes + kv_host_bytes

    # A projector alone can close the deficit, and then nothing below scores the
    # plan: ``units`` and ``spill_lm_head`` are both empty, the cost gate is
    # skipped, and with it the only refusal that keeps a host side out of swap.
    # The projector on the host is resident, not pageable (clip.cpp allocates it
    # in a CPU backend buffer; mmap covers the model file, not that), so a host
    # that cannot hold it gets the same refusal a weight spill would.
    if mmproj_host_bytes:
        refused = _host_ram_refusal(opts, n_ctx, host_side, host_ram_bytes)
        if refused is not None:
            return refused

    # A plan can give something up without moving a weight, and those were never
    # ranked: the fallback arm of a no-spill plan is the launch as the caller
    # typed it, which is as priceable as any other.
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
    # token_embd is host-resident on every launch, so it is host RAM this plan
    # has to be able to pay for even when nothing is spilled; the -nkvo cache is
    # in there too, on the terms above.
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
    if host_ram_bytes is None or opts.prompt_cache_unbounded:
        load_mode_none = False
    else:
        load_mode_none = host_bytes <= max(0, host_ram_bytes - opts.host_ram_headroom_bytes)

    # The prompt cache is host RAM llama-server takes on top of the spill, 8 GiB by
    # default, and it is the cheapest thing in the system to give up: losing a
    # prefix hit costs one re-prefill, mis-sizing host RAM costs the load. Bound it
    # to what is left under the headroom once the plan's own host side is paid,
    # and only under --load-mode none, where the host side is resident rather than
    # pageable. Reported only when the bound is BELOW the default.
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
    # A type the launch already carries is not a change the plan makes.
    cache_type = opts.kv_quant_type if (quantised and not opts.cache_quantised) else None
    changed = (
        bool(patterns)
        or load_mode_none
        or cache_type is not None
        or n_parallel > 0
        or mmproj_to_host
        or draft_dropped
        # A resident fit found by the context ladder emits no pattern and no knob,
        # and is still a different launch from the one the caller priced: the
        # planner proved a context the fallback path had already capped below.
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
    # draft_dropped has no flag here: the caller owns the speculative flags.
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
