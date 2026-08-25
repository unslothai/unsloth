# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Relative throughput of a candidate offload placement.

The planner needs to RANK placements, not predict absolute t/s. So this models
the marginal cost of moving a tensor group off the GPU, in milliseconds per
token, and leaves the resident baseline as a free parameter the caller supplies
or ignores.

Two regimes, measured separately because they have different mechanisms:

``generation``
    One token at a time. Host-resident weights are computed by the CPU backend,
    NOT streamed to the GPU. Measured directly: with the dense FFN spilled,
    generation scales with thread count (2.42 / 5.83 / 11.82 t/s at 4 / 16 / 64
    threads) while a fully resident run does not move at all (87.30 vs 87.31 at
    4 vs 64). So the cost scales with the HOST's compute and memory bandwidth,
    not with the PCIe generation.

``prefill``
    A whole ubatch at once. ggml's scheduler sends the large GEMM to the GPU and
    copies the weights in, so this regime IS link-bound, and it reads the FULL
    tensor rather than only the activated slice.

The distinction matters most for MoE: generation touches ``n_used / n_expert``
of the expert weights per token, but a 512-token prefill ubatch touches
essentially all of them. MoE therefore tolerates spilling far better than a
dense model during generation and slightly WORSE during prefill.

Calibration is in ``_RATE_RATIOS`` and comes from the measurements in
``plans/impl_planner_cost.md``. Rates are expressed as multiples of the
contiguous-weight rate, because ratios transfer across machines much better than
absolute milliseconds do.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

GIB = float(1024**3)

# Reference host (192 cores, one B200), dense FFN spilled in 16/32/48/65-block
# steps: least-squares fit gives 5.544 ms per GiB, intercept -0.10 ms. The
# near-zero intercept means partial spilling is LINEAR, so spill the minimum.
REFERENCE_CONTIGUOUS_MS_PER_GIB = 5.544

# Reference host thread count the rate above was measured at.
REFERENCE_HOST_THREADS = 192

# Generation penalty vs threads, ms per GiB = a / threads + b. Least squares over
# 39.81 / 15.86 / 7.248 / 5.498 ms per GiB at 4 / 16 / 64 / 192 threads (full
# dense FFN spilled) gives a = 138.4, b = 5.57, every point within 14%. Amdahl,
# not a power law: ``a`` is the CPU backend's parallel quantised matmul, ``b`` the
# floor (host bandwidth, graph-split sync, non-overlapping GPU work), so 4x the
# cores buys much less than 4x the speed.
_THREAD_PARALLEL_MS_PER_GIB = 138.4
_THREAD_SERIAL_MS_PER_GIB = 5.57

# The fit above varies threads on ONE machine; across DIFFERENT machines the
# slope is steeper (12 cloud vCPUs are ~6 physical cores on one memory
# controller). Least squares over dense Q4 on 12 and 48 vCPU hosts (24.21 and
# 6.82 ms per GiB) gives a = 278.2, b = 1.03. Neither fit dominates -- at 192
# threads cross-machine predicts 2.48 vs 5.498 measured, at 12 one-machine
# predicts 17.1 vs 24.21 -- so take whichever is MORE expensive: optimism quotes
# a spill that runs several times slower than promised.
_CROSS_HOST_PARALLEL_MS_PER_GIB = 278.2
_CROSS_HOST_SERIAL_MS_PER_GIB = 1.03

# Residual model/measured over 70 runs on T4, L4, A100 and RTX PRO 6000 (2, 8,
# 12, 48 vCPU), Qwen3.8-27B and Qwen3.6-35B-A3B at Q2_K_XL / Q4_K_XL / Q8_K_XL:
# medians 1.09 (48 cpu), 0.59 (12), 0.24 (8), 0.24 (2).
#
# One case stays materially under-priced: DENSE Q2_K on a very small host. Cost
# per spilled GiB tracks dequant complexity, not byte count -- at 12 vCPU Q2_K
# 84.17, Q4_K 24.21, Q8 19.96 ms per GiB (Q8 cheapest despite being largest),
# compressing to 8.70 / 6.82 / 6.29 at 48 vCPU. A quant multiplier would itself
# move with core count (3.5x at 12 vCPU, 1.3x at 48), so fitting it from two core
# counts would be overfitting. Left unmodelled deliberately.


class Access(str, Enum):
    """How a tensor group is read, which sets its cost per byte."""

    #: Dense FFN: large contiguous matmuls, best case for the CPU backend.
    CONTIGUOUS = "contiguous"
    #: lm_head: one tall matvec, usually higher-bit. Parallelises worse than 64
    #: independent FFN blocks and dequantises more bytes per output element.
    SINGLE_MATVEC = "single_matvec"
    #: MoE experts: 8-of-256 scattered small matmuls per layer per token.
    SCATTERED = "scattered"
    #: The attention cache. By far the worst thing to move off the device.
    KV_CACHE = "kv_cache"


# Cost per activated byte, relative to CONTIGUOUS. Derived in the plan doc:
#   lm_head 10.51 / 5.955 = 1.77;  MoE expert 14.80 / 5.955 = 2.49
#   KV cache 119.7 / 5.955 = 20.1 (20.4 from 121.2 on a structurally different
#   model -- agreeing to 1.3%, the strongest single calibration point here)
_RATE_RATIOS: dict[Access, float] = {
    Access.CONTIGUOUS: 1.00,
    Access.SINGLE_MATVEC: 1.77,
    Access.SCATTERED: 2.49,
    Access.KV_CACHE: 20.1,
}

# Prefill streaming bandwidth, GiB/s. Bracketed by the two measurements (dense
# 49.4, MoE 66.9); that spread is why this is a coarse constant, not a curve.
PREFILL_STREAM_GIB_S = 55.0

# Spilling several groups costs MORE than the sum of each alone -- contention,
# not amortisation. Measured 6% for the dense FFN + lm_head pair. The "costs are
# sub-additive" reading compares throughput percentages instead of times.
MULTI_GROUP_CONTENTION = 0.06


@dataclass(frozen = True)
class TensorGroup:
    """A set of weights the planner can place as a unit."""

    name: str
    bytes_total: int
    access: Access
    #: Fraction of ``bytes_total`` read per token during GENERATION. 1.0 for
    #: dense weights; ``n_expert_used / n_expert`` for MoE experts.
    activation_fraction: float = 1.0

    @property
    def activated_bytes(self) -> int:
        return int(self.bytes_total * self.activation_fraction)


@dataclass(frozen = True)
class HostProfile:
    """What the host can bring to bear on CPU-resident weights.

    ``threads`` is the honest knob. Generation cost scales with it (measured
    2.42 / 5.83 / 11.82 t/s at 4 / 16 / 64), with diminishing returns as the
    memory subsystem saturates, which ``thread_scaling_exponent`` captures.
    """

    threads: int = REFERENCE_HOST_THREADS
    #: Set for unified-memory hosts (Apple Silicon, AMD APU, Vulkan iGPU) where
    #: "spilling" moves nothing, because the two pools are one pool.
    unified_memory: bool = False

    @property
    def generation_slowdown(self) -> float:
        """How much slower than the reference host this one is per host byte."""
        if self.threads <= 0:
            return 1.0

        def rate(threads: float) -> float:
            """ms per GiB, taking whichever fit is more expensive at this size.

            The two fits cross: the one-machine sweep wins at large thread
            counts, the cross-machine one at small. Taking the max keeps the
            reference host's own validated numbers while refusing to quote a
            small host the throughput of a server.
            """
            return max(
                _THREAD_PARALLEL_MS_PER_GIB / threads + _THREAD_SERIAL_MS_PER_GIB,
                _CROSS_HOST_PARALLEL_MS_PER_GIB / threads + _CROSS_HOST_SERIAL_MS_PER_GIB,
            )

        return rate(float(self.threads)) / rate(float(REFERENCE_HOST_THREADS))


@dataclass
class Placement:
    """A candidate: which groups sit in host memory, and where the cache is."""

    host_groups: list[TensorGroup] = field(default_factory = list)
    #: Bytes of attention cache forced to host RAM. Only ``--no-kv-offload``
    #: puts anything here; ``-ot`` leaves it resident, ``-ngl`` drags it off.
    kv_host_bytes: int = 0


def generation_penalty_ms(placement: Placement, host: HostProfile | None = None) -> float:
    """Extra milliseconds per generated token, versus everything resident.

    Additive in TIME across groups, plus a contention term when more than one
    group is spilled. Returns 0.0 for a fully resident placement, and for any
    placement on a unified-memory host, where moving a tensor between "VRAM"
    and "RAM" does not change which chips hold it.
    """
    host = host or HostProfile()
    if host.unified_memory:
        return 0.0

    groups = list(placement.host_groups)
    per_group = [
        (g.activated_bytes / GIB) * REFERENCE_CONTIGUOUS_MS_PER_GIB * _RATE_RATIOS[g.access]
        for g in groups
    ]
    if placement.kv_host_bytes > 0:
        per_group.append(
            (placement.kv_host_bytes / GIB)
            * REFERENCE_CONTIGUOUS_MS_PER_GIB
            * _RATE_RATIOS[Access.KV_CACHE]
        )

    total = sum(per_group)
    if len([c for c in per_group if c > 0.0]) > 1:
        total *= 1.0 + MULTI_GROUP_CONTENTION
    return total * host.generation_slowdown


def prefill_penalty_ms_per_token(
    placement: Placement,
    n_ubatch: int = 512,
    host: HostProfile | None = None,
) -> float:
    """Extra milliseconds per PROMPT token during prefill.

    Uses FULL bytes, not activated bytes: a 512-token ubatch selects
    essentially every expert at least once, so sparsity buys nothing here. This
    is the term that makes MoE's prefill penalty WORSE than a dense model's even
    though its generation penalty is much better.

    The weights are copied once per ubatch and reused across every token in it,
    so the per-token cost falls as ``n_ubatch`` rises -- the amortisation that
    makes prefill so much cheaper than generation per byte moved.

    ``host`` is accepted and used only for ``unified_memory``: this regime runs
    on the GPU with the weights copied in, so it is bound by the link and NOT by
    host cores. That asymmetry against generation is the point, so callers pass
    the same profile to both and let each use what applies.
    """
    host = host or HostProfile()
    if host.unified_memory:
        return 0.0
    # kv_host_bytes counts here too: leaving it out made a cache-offloaded
    # placement prefill FOR FREE while generation_penalty_ms charged it.
    # UNCALIBRATED lower bound -- counts the cache as streamed once per ubatch
    # like the weights, though attention re-reads it and the read grows with
    # depth. No production path builds kv_host_bytes > 0; this only stops rank()
    # calling a cache spill free.
    host_bytes = sum(g.bytes_total for g in placement.host_groups) + placement.kv_host_bytes
    if host_bytes <= 0 or n_ubatch <= 0:
        return 0.0
    per_ubatch_ms = (host_bytes / GIB) / PREFILL_STREAM_GIB_S * 1000.0
    return per_ubatch_ms / float(n_ubatch)


def rank(
    candidates: list[Placement],
    host: HostProfile | None = None,
    n_generated: int = 1,
    n_prompt: int = 0,
    n_ubatch: int = 512,
) -> list[tuple[Placement, float]]:
    """Sort placements cheapest first for a workload of a given shape.

    Both terms are milliseconds for the WHOLE request, so they are commensurate:
    ``n_prompt`` tokens of prefill plus ``n_generated`` tokens of decode. A long
    prompt with a short reply and a short prompt with a long reply are genuinely
    different questions and can rank placements differently, so the caller
    states the shape rather than accepting a baked-in answer.

    Defaults to pure single-token generation, which is where placements differ
    most.
    """
    scored = [
        (
            c,
            n_generated * generation_penalty_ms(c, host)
            + n_prompt * prefill_penalty_ms_per_token(c, n_ubatch, host),
        )
        for c in candidates
    ]
    return sorted(scored, key = lambda pair: pair[1])
