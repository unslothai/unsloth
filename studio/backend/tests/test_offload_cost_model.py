# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The cost model against every measurement that produced it.

Anchors are real llama-bench runs on one B200 with a 192-core host, llama.cpp
b10360-era build, at 128K context unless stated. Each ``MEASURED_*`` figure is a
throughput in t/s converted to milliseconds per token, because time is the
quantity that composes and throughput is not.
"""

from __future__ import annotations

import pytest

from core.inference.offload_cost_model import (
    REFERENCE_CONTIGUOUS_MS_PER_GIB,
    Access,
    HostProfile,
    Placement,
    TensorGroup,
    generation_penalty_ms,
    prefill_penalty_ms_per_token,
    rank,
)

GIB = float(1024**3)


def ms(t_per_s: float) -> float:
    return 1000.0 / t_per_s


# ---------------------------------------------------------------- the anchors

# Qwen3.8-27B UD-Q4_K_XL, dense, 128K.
DENSE_BASE = ms(75.37)
DENSE_LM_HEAD = ms(42.60)
DENSE_FFN = ms(13.63)
DENSE_BOTH = ms(11.39)
DENSE_KV_HOST = ms(1.03)

# Qwen3.6-35B-A3B UD-Q4_K_XL, MoE, 128K.
MOE_BASE = ms(182.0)
MOE_EXPERTS = ms(71.63)
MOE_KV_HOST = ms(3.24)

DENSE_FFN_G = TensorGroup("ffn", int(10.092 * GIB), Access.CONTIGUOUS)
DENSE_LM_G = TensorGroup("lm_head", int(0.9713 * GIB), Access.SINGLE_MATVEC)
# 16 attention layers x 4 kv heads x (256+256) x 2 bytes x 131072 tokens.
DENSE_KV_BYTES = 16 * 4 * 512 * 2 * 131072

# 256 experts, 8 used per token: the sparsity that makes MoE cheap to spill.
MOE_EXPERT_G = TensorGroup(
    "experts", int(18.320 * GIB), Access.SCATTERED, activation_fraction = 8 / 256
)
# 10 attention layers x 2 kv heads x (256+256) x 2 bytes x 131072 tokens.
MOE_KV_BYTES = 10 * 2 * 512 * 2 * 131072


def rel_err(predicted: float, measured: float) -> float:
    return abs(predicted - measured) / measured


# One free constant plus three per-access ratios. It under-predicts every 128K
# anchor by a near-identical ~7%: the base rate is calibrated on the cleaner
# depth-0 partial-spill sweep while these anchors sit at 128K where attention
# contends. A uniform offset cannot change an ordering, which is all the planner
# asks of it.
ANCHOR_TOL = 0.10


@pytest.mark.parametrize(
    "label,placement,measured_delta",
    [
        ("lm_head only", Placement([DENSE_LM_G]), DENSE_LM_HEAD - DENSE_BASE),
        ("ffn only", Placement([DENSE_FFN_G]), DENSE_FFN - DENSE_BASE),
        ("ffn + lm_head", Placement([DENSE_FFN_G, DENSE_LM_G]), DENSE_BOTH - DENSE_BASE),
        (
            "kv to host",
            Placement([], kv_host_bytes = DENSE_KV_BYTES),
            DENSE_KV_HOST - DENSE_BASE,
        ),
        ("moe experts", Placement([MOE_EXPERT_G]), MOE_EXPERTS - MOE_BASE),
        (
            "moe kv to host",
            Placement([], kv_host_bytes = MOE_KV_BYTES),
            MOE_KV_HOST - MOE_BASE,
        ),
    ],
)
def test_the_model_reproduces_every_measured_anchor(label, placement, measured_delta):
    assert rel_err(generation_penalty_ms(placement), measured_delta) < ANCHOR_TOL, label


def test_the_kv_ratio_transfers_across_two_unrelated_models():
    """The strongest calibration point, and the only genuinely predictive one.

    The KV rate was derived from the dense model alone. Applied unchanged to a
    MoE model with a different layer count, head count and cache size, it lands
    within the same tolerance. Two structurally unrelated models agreeing on one
    constant is what makes "never move the cache" a rule rather than a datum.
    """
    dense_rate = (DENSE_KV_HOST - DENSE_BASE) / (DENSE_KV_BYTES / GIB)
    moe_rate = (MOE_KV_HOST - MOE_BASE) / (MOE_KV_BYTES / GIB)
    assert rel_err(moe_rate, dense_rate) < 0.03


# ------------------------------------------------- the orderings that matter


def test_the_cache_is_the_worst_byte_to_move_by_an_order_of_magnitude():
    """20x per byte against contiguous weights. This is the whole reason the
    planner uses -ot (which leaves the cache resident) instead of -ngl."""
    one_gib_weights = Placement([TensorGroup("w", int(GIB), Access.CONTIGUOUS)])
    one_gib_cache = Placement([], kv_host_bytes = int(GIB))
    assert generation_penalty_ms(one_gib_cache) > 15 * generation_penalty_ms(one_gib_weights)


def test_ffn_is_cheaper_per_byte_than_lm_head():
    """Which is why FFN is spilled first, even though lm_head is the smaller
    tensor and looks like the tidier thing to move."""
    per_gib = {
        access: generation_penalty_ms(Placement([TensorGroup("x", int(GIB), access)]))
        for access in (Access.CONTIGUOUS, Access.SINGLE_MATVEC, Access.SCATTERED)
    }
    assert per_gib[Access.CONTIGUOUS] < per_gib[Access.SINGLE_MATVEC]
    assert per_gib[Access.SINGLE_MATVEC] < per_gib[Access.SCATTERED]


def test_spilling_two_groups_costs_MORE_than_the_sum_not_less():
    """The correction that matters most.

    Read as throughput percentages, lm_head "costs 43% alone but only 16% on top
    of FFN", which reads as a discount. In time it is the opposite: the same
    0.97 GiB adds 10.2 ms alone and 14.4 ms once FFN is already spilled. Ranking
    on percentages would pick the wrong placement.
    """
    alone = generation_penalty_ms(Placement([DENSE_LM_G]))
    ffn_only = generation_penalty_ms(Placement([DENSE_FFN_G]))
    both = generation_penalty_ms(Placement([DENSE_FFN_G, DENSE_LM_G]))
    assert both > ffn_only + alone
    marginal = both - ffn_only
    assert marginal > alone


def test_the_measured_marginal_cost_of_lm_head_really_does_rise():
    """Same claim, against the raw numbers rather than the model."""
    alone = DENSE_LM_HEAD - DENSE_BASE
    marginal_on_top_of_ffn = DENSE_BOTH - DENSE_FFN
    assert marginal_on_top_of_ffn > alone
    assert 1.3 < marginal_on_top_of_ffn / alone < 1.5


# ---------------------------------------------------------- partial spilling


@pytest.mark.parametrize(
    "gib,measured_delta",
    [
        # depth 0, so these are compared among themselves, not to the 128K set.
        (2.184, ms(43.67) - ms(87.30)),
        (4.610, ms(26.62) - ms(87.30)),
        (7.053, ms(19.71) - ms(87.30)),
        (10.092, ms(14.94) - ms(87.30)),
    ],
)
def test_partial_spilling_is_linear(gib, measured_delta):
    """A least-squares fit over these four points gives 5.544 ms/GiB with a
    -0.10 ms intercept, so there is no per-split fixed cost worth modelling and
    the planner may spill exactly the minimum that fits."""
    predicted = generation_penalty_ms(
        Placement([TensorGroup("part", int(gib * GIB), Access.CONTIGUOUS)])
    )
    assert rel_err(predicted, measured_delta) < 0.10


def test_spilling_less_always_costs_less():
    """Monotonicity. Without it the planner could prefer a larger spill."""
    costs = [
        generation_penalty_ms(Placement([TensorGroup("p", int(gib * GIB), Access.CONTIGUOUS)]))
        for gib in (1.0, 3.0, 5.0, 10.0)
    ]
    assert costs == sorted(costs)


# ------------------------------------------------------ prefill vs generation


def test_moe_wins_at_generation_and_loses_at_prefill():
    """The crossover the two regimes produce, and the reason they are modelled
    apart. Generation reads 8/256 of the experts; a 512-token prefill ubatch
    reads all of them, so the sparsity that makes MoE cheap to spill during
    generation buys nothing during prefill."""
    moe = Placement([MOE_EXPERT_G])
    dense = Placement([DENSE_FFN_G])
    assert generation_penalty_ms(moe) < generation_penalty_ms(dense)
    assert prefill_penalty_ms_per_token(moe) > prefill_penalty_ms_per_token(dense)


def test_the_measured_penalties_show_that_same_crossover():
    """Against the raw anchors: MoE is hurt less on generation, more on prefill."""
    moe_gen = MOE_EXPERTS / MOE_BASE
    dense_gen = DENSE_FFN / DENSE_BASE
    assert moe_gen < dense_gen  # 2.54x vs 5.39x
    moe_pp, dense_pp = 5522.0 / 1397.0, 2095.0 / 1141.0
    assert moe_pp > dense_pp  # 3.95x vs 1.84x


# ----------------------------------------------------------------- the hosts


def test_a_smaller_host_makes_every_spill_worse():
    """Generation cost tracks host threads, measured 2.42 / 5.83 / 11.82 t/s at
    4 / 16 / 64 with the FFN spilled, against a flat 87.30 resident. A desktop
    is not a small version of this box; it is a different recommendation.

    This used to under-warn by about 22% at 16 threads, predicting 2.26x against
    a measured 2.885x. That gap was the one-machine fit being applied across
    machines. The cross-host floor closes it: the prediction is now 2.93x, a
    little OVER the measured ratio rather than well under it.

    Over is the side to be on. Under-warning quotes a spill that then runs
    several times slower than promised, which is the same direction as every
    real defect this planner has had; over-warning costs some throughput a user
    could have had. Held to within 10% so "conservative" cannot drift into
    "useless".
    """
    big = generation_penalty_ms(Placement([DENSE_FFN_G]), HostProfile(threads = 192))
    small = generation_penalty_ms(Placement([DENSE_FFN_G]), HostProfile(threads = 16))
    assert small > 2 * big
    measured_ratio = (ms(5.83) - ms(87.30)) / (ms(14.94) - ms(87.30))
    assert small / big >= measured_ratio  # no longer under-warns
    assert small / big < measured_ratio * 1.1


def test_thread_scaling_matches_the_measured_sweep():
    """Predicted ratio between 16 and 64 threads against the measured 11.82/5.83."""
    at16 = generation_penalty_ms(Placement([DENSE_FFN_G]), HostProfile(threads = 16))
    at64 = generation_penalty_ms(Placement([DENSE_FFN_G]), HostProfile(threads = 64))
    measured_ratio = 11.82 / 5.83
    assert rel_err(at16 / at64, measured_ratio) < 0.20


def test_prefill_ignores_host_threads_while_generation_does_not():
    """The asymmetry between the two regimes, asserted as a contrast rather than
    by comparing a call to itself.

    Prefill clears ggml's op-offload batch threshold (32, ggml-cuda.cu:5465) so
    the op moves to the GPU and the weights are copied in: link-bound, cores
    irrelevant. Generation at batch 1 stays below it and runs on the CPU
    backend: core-bound.
    """
    p = Placement([DENSE_FFN_G])
    big, small = HostProfile(threads = 192), HostProfile(threads = 8)
    assert prefill_penalty_ms_per_token(p, host = small) == prefill_penalty_ms_per_token(p, host = big)
    assert generation_penalty_ms(p, small) > 2 * generation_penalty_ms(p, big)


def test_prefill_amortises_over_the_ubatch():
    """Weights are copied once per ubatch and reused by every token in it, which
    is why prefill is so much cheaper per byte moved than generation."""
    p = Placement([DENSE_FFN_G])
    assert prefill_penalty_ms_per_token(p, n_ubatch = 512) == pytest.approx(
        prefill_penalty_ms_per_token(p, n_ubatch = 256) / 2.0
    )
    # And the measured per-token prefill penalty is far below the generation one.
    assert prefill_penalty_ms_per_token(p) < generation_penalty_ms(p) / 100.0


def test_unified_memory_hosts_gain_nothing_from_spilling():
    """Apple Silicon, AMD APUs and Vulkan iGPUs report host RAM as VRAM. Moving
    a tensor between the two does not change which chips hold it, so the planner
    must not pay a penalty for it -- nor claim it freed anything."""
    unified = HostProfile(unified_memory = True)
    assert generation_penalty_ms(Placement([DENSE_FFN_G, DENSE_LM_G]), unified) == 0.0
    assert generation_penalty_ms(Placement([], kv_host_bytes = DENSE_KV_BYTES), unified) == 0.0


# ------------------------------------------------------------------ ranking


def test_ranking_puts_the_measured_best_placement_first():
    """The ladder, rediscovered rather than hard-coded: resident, then FFN, then
    FFN plus lm_head, and the cache last by a wide margin."""
    resident = Placement([])
    ffn = Placement([DENSE_FFN_G])
    ffn_lm = Placement([DENSE_FFN_G, DENSE_LM_G])
    kv = Placement([], kv_host_bytes = DENSE_KV_BYTES)
    order = [p for p, _ in rank([kv, ffn_lm, ffn, resident])]
    assert order == [resident, ffn, ffn_lm, kv]


def test_a_prefill_heavy_mix_can_reorder_dense_against_moe():
    """A caller that weights prefill is answering a different question, and the
    model must let it, rather than baking in the generation answer."""
    moe, dense = Placement([MOE_EXPERT_G]), Placement([DENSE_FFN_G])
    gen_first = [p for p, _ in rank([dense, moe], n_generated = 1, n_prompt = 0)]
    pp_first = [p for p, _ in rank([dense, moe], n_generated = 0, n_prompt = 4096)]
    assert gen_first[0] is moe
    assert pp_first[0] is dense


def test_the_cross_host_floor_matches_the_measured_cloud_hosts():
    """The floor is fitted to real cloud VMs, so hold it to them.

    Measured dense Q4_K_XL, ms per GiB of spilled weights, over 70 runs on
    T4 / L4 / A100 / RTX PRO 6000: 24.21 at 12 vCPU, 6.82 at 48, and 5.498 at
    the 192-thread reference. Before the floor the 12 vCPU case was predicted at
    17.1, i.e. 0.59 of the truth.
    """
    rate = lambda t: (  # noqa: E731 - one expression, reads better inline
        HostProfile(threads = t).generation_slowdown * REFERENCE_CONTIGUOUS_MS_PER_GIB
    )
    for threads, measured in ((12, 24.21), (48, 6.82), (192, 5.498)):
        ratio = rate(threads) / measured
        assert 0.85 <= ratio <= 1.15, (threads, rate(threads), measured, ratio)
    # Monotone in cores, and a tiny host is charged much more than a big one.
    assert rate(2) > rate(8) > rate(12) > rate(48) > rate(192)
    assert rate(2) > 10 * rate(192)


def test_a_host_cache_is_not_free_during_prefill():
    """generation_penalty_ms charges kv_host_bytes and prefill did not, so a
    cache-offloaded placement prefilled for free and TIED with a fully resident
    one at n_generated = 0. The asymmetry was the bug: the cache crosses the same
    link as the weights."""
    resident = Placement()
    kv_host = Placement(kv_host_bytes = int(4 * GIB))

    assert prefill_penalty_ms_per_token(resident) == 0.0
    assert prefill_penalty_ms_per_token(kv_host) > 0.0

    ordered = rank([kv_host, resident], n_generated = 0, n_prompt = 4096)
    assert ordered[0][0] is resident, "resident must win a pure-prefill ranking"
    assert ordered[0][1] < ordered[1][1], "and it must not be a tie"

    # Still zero where moving bytes between two names for one pool is free.
    assert prefill_penalty_ms_per_token(kv_host, host = HostProfile(unified_memory = True)) == 0.0
