# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The in-backward optimizer step (optimizers/auto_fit.py) on FSDP2 fully_shard models: 2 gloo ranks on CPU, a tiny
fp32 Llama with one fully_shard group per decoder layer plus the root group. After 5 AdamW steps the weights match
FSDP2's standard step and a single-process reference that applies the same clipping rule per FSDP2 group, for every
clip mode, with gradient accumulation (no_sync, or reducing every micro-batch behind a sync gate), with bfloat16
compute and float32 reduction, and when the optimizer was built before the model was sharded (accelerate's order).
Each group's gradients are freed at its last hook; without the hooks (compiled autograd) step() steps them all."""

import multiprocessing
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))

from _rendezvous_port import free_port  # noqa: E402

import _auto_fit_fsdp2_worker as worker  # noqa: E402

pytestmark = pytest.mark.skipif(
    not (torch.distributed.is_available() and torch.distributed.is_gloo_available()),
    reason = "needs torch.distributed with gloo",
)

TOL = 1e-5
STEPS = worker.STEPS

# name: (gradient_accumulation_steps, clip mode or None, micro-batch sync, bf16 compute, arm)
CASES = {
    "adamw": (1, None, "no_sync", False, "in_backward"),
    "estimate": (1, "estimate", "no_sync", False, "in_backward"),
    "exact": (1, "exact", "no_sync", False, "in_backward"),
    "lagged": (1, "lagged", "no_sync", False, "in_backward"),
    "per_tensor": (1, "per_tensor", "no_sync", False, "in_backward"),
    "ga2_no_sync": (2, "estimate", "no_sync", False, "in_backward"),
    "ga2_sync_gate": (2, "estimate", "every", False, "in_backward"),
    "ga2_exact": (2, "exact", "no_sync", False, "in_backward"),
    "accelerate_order": (1, "estimate", "no_sync", False, "in_backward_swap"),
    "bf16_reduce_fp32": (1, None, "no_sync", True, "in_backward"),
    "bf16_reduce_fp32_exact": (1, "exact", "no_sync", True, "in_backward"),
}


@pytest.fixture(scope = "module")
def results():
    pytest.importorskip("transformers")
    try:
        from torch.distributed.fsdp import fully_shard  # noqa: F401
    except ImportError:
        pytest.skip("torch without torch.distributed.fsdp.fully_shard")
    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    port = free_port()
    procs = [ctx.Process(target = worker.run_rank, args = (r, port, queue, CASES)) for r in range(2)]
    for p in procs:
        p.start()
    out = {}
    for _ in procs:
        rank, result = queue.get(timeout = 240)
        out[rank] = result
    for p in procs:
        p.join(timeout = 60)
    skip = next((r["skip"] for r in out.values() if "skip" in r), None)
    if skip:
        pytest.skip(skip)
    return out


def _case(results, rank, case):
    r = results[rank][case]
    assert "error" not in r, f"rank {rank}:\n{r['error']}"
    return r


@pytest.mark.parametrize("case", list(CASES))
def test_weights_match_standard_fsdp2_and_single_process(results, case):
    ga, clip, sync, mp, arm = CASES[case]
    for rank in (0, 1):
        r = _case(results, rank, case)
        if mp:
            # bfloat16 compute, float32 reduction: bitwise FSDP2's standard step without clipping; with it, the
            # clip coefficient is rounded differently (float32 norms here, float64 in the harness) and bfloat16
            # forwards amplify that, so the losses and norms are compared instead.
            if clip is None:
                assert r["inb_vs_std"] == 0.0, r
            assert r["loss_vs_std"] < 1e-3 and r["norm_vs_std"] < 2e-2, r
            continue
        assert r["moved"] > 1e-3, r  # the weights trained, so equality is not trivial
        if clip in (None, "exact"):
            assert (
                r["std_vs_ref"] < TOL
            ), r  # the harness: FSDP2's own step (exact global clip) equals the reference
            assert r["inb_vs_std"] < TOL, r
        assert r["inb_vs_ref"] < TOL, r
        assert (
            r["norm_diff"] < 1e-5
        ), r  # total_norm() = the full model's gradient norm on every rank
        assert r["loss_diff"] < 1e-5, r
        # Measuring passes: every step for exact, the first for estimate / lagged; none under gradient accumulation
        # (those steps are kept for optimizer.step() instead).
        want = (
            0
            if ga > 1
            else (STEPS if clip == "exact" else (1 if clip in ("estimate", "lagged") else 0))
        )
        assert r["inb_stats"]["measured_steps"] == want, r


def test_clip_modes_differ(results):
    """The clip modes really clip differently here (so matching the per-mode reference is not trivial)."""
    for rank in (0, 1):
        ref = {
            c: _case(results, rank, c)["inb_vs_std"] for c in ("estimate", "lagged", "per_tensor")
        }
        assert min(ref.values()) > 1e-4, ref  # each differs from exact global clipping


@pytest.mark.parametrize("case", list(CASES))
def test_steps_in_backward_and_frees_each_group(results, case):
    for rank in (0, 1):
        r = _case(results, rank, case)
        inb, std = r["inb_stats"], r["std_stats"]
        assert len(inb["groups"]) >= 3, inb["groups"]  # decoder layer groups + the root group
        ga, clip = CASES[case][0], CASES[case][1]
        deferred_steps = (
            (STEPS if clip == "exact" else 1) if ga > 1 and clip not in (None, "per_tensor") else 0
        )
        # every weight stepped from a hook, except the steps gradient accumulation keeps for optimizer.step()
        assert inb["late_steps"] == deferred_steps * inb["total_params"], inb
        kept = deferred_steps if clip == "exact" else min(deferred_steps, 1)
        # every gradient freed by the end of backward, except on the steps kept for optimizer.step()
        assert inb["live_after_backward"] == [inb["total_params"]] * kept + [0] * (
            STEPS - kept
        ), inb
        assert all(n == std["total_params"] for n in std["live_after_backward"]), std
        if kept:
            continue  # a kept step holds every gradient until optimizer.step(), as gradient accumulation does
        if CASES[case][2] == "every" and CASES[case][0] > 1:
            # Reduced every micro-batch: the sharded gradients of the earlier micro-batches stay until the last one,
            # so all of them are alive at once (no saving over the standard step).
            assert inb["peak_live"] == inb["total_params"], inb
        else:
            # At any hook at most one group's gradients are alive (the largest group is the root group here).
            assert inb["peak_live"] <= max(inb["groups"]), inb
            assert inb["peak_live"] < inb["total_params"], inb


def test_hooks_not_run_falls_back_to_step(results):
    """FSDP2 skips the hooks under compiled autograd: step() then steps every gradient, same weights."""
    for rank in (0, 1):
        r = _case(results, rank, "adamw")
        assert r["unarmed_vs_ref"] < TOL, r
        assert r["unarmed_late_steps"] == STEPS * r["inb_stats"]["total_params"], r
        assert r["unarmed_vs_ref"] < TOL, r


def test_planner_reads_which_fsdp2_groups_stay_whole(results):
    for rank in (0, 1):
        got = results[rank]["_gathered"]
        assert tuple(got["got"]) == tuple(got["want"]), got


def test_ranks_agree_on_the_memory_budget(results):
    assert results[0]["_min_over_ranks"] == results[1]["_min_over_ranks"] == 1000
