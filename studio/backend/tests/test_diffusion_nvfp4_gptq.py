# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit tests for the NVFP4 build-time calibration (``diffusion_nvfp4_gptq.py``)."""

from __future__ import annotations

import pytest
import torch

from core.inference import diffusion_nvfp4_gptq as gq


def _spd(
    k: int,
    *,
    seed: int = 0,
    device = "cpu",
) -> torch.Tensor:
    """A random symmetric positive definite matrix, as an activation second moment is."""
    generator = torch.Generator(device = device).manual_seed(seed)
    a = torch.randn(4 * k, k, generator = generator, device = device)
    return (a.transpose(0, 1) @ a) / (4 * k) + torch.eye(k, device = device) * 1e-2


def _hessian_error(weight, candidate, hessian) -> float:
    return gq.hessian_weighted_error(weight.float() - candidate.float(), hessian)




def test_the_correction_lands_on_the_same_grid_as_round_to_nearest():
    """GPTQ's output must be exactly representable in NVFP4 or the packing undoes the correction."""
    weight = torch.randn(64, 128, dtype = torch.bfloat16) * 0.05
    hessian = _spd(128, seed = 1)
    corrected = gq.gptq_quantize_to_nvfp4(weight, hessian)
    assert corrected.dtype == weight.dtype and corrected.shape == weight.shape
    assert torch.equal(gq.rtn_quantize_to_nvfp4(corrected), corrected)


def test_the_correction_lowers_the_hessian_weighted_error_against_round_to_nearest():
    weight = torch.randn(64, 256, dtype = torch.bfloat16) * 0.05
    hessian = _spd(256, seed = 2)
    rtn = gq.rtn_quantize_to_nvfp4(weight)
    corrected = gq.gptq_quantize_to_nvfp4(weight, hessian)
    assert _hessian_error(weight, corrected, hessian) < _hessian_error(weight, rtn, hessian)


def test_the_damping_escalates_and_then_raises_rather_than_falling_back_to_rtn():
    """A degenerate Hessian raises rather than silently shipping that layer at round-to-nearest."""
    weight = torch.randn(8, 32, dtype = torch.bfloat16) * 0.05
    calls: list = []
    real = gq.gptq_quantize_to_nvfp4

    def _flaky(
        w,
        h,
        block = 16,
        damp = 0.01,
    ):
        calls.append(damp)
        if damp < 0.1:
            raise torch._C._LinAlgError("cholesky: the factorization could not be completed")
        return real(w, h, block = block, damp = damp)

    gq.gptq_quantize_to_nvfp4 = _flaky
    try:
        corrected, damp = gq.gptq_correct(weight, _spd(32, seed = 3))
        assert damp == 0.1 and calls == [0.01, 0.05, 0.1]
        assert corrected.shape == weight.shape

        calls.clear()

        def _always_fails(
            w,
            h,
            block = 16,
            damp = 0.01,
        ):
            calls.append(damp)
            raise torch._C._LinAlgError("cholesky: the factorization could not be completed")

        gq.gptq_quantize_to_nvfp4 = _always_fails
        with pytest.raises(gq.GPTQFailure) as excinfo:
            gq.gptq_correct(weight, _spd(32, seed = 3))
        assert calls == list(gq.DAMP_LADDER)
        assert "round-to-nearest" in str(excinfo.value)
    finally:
        gq.gptq_quantize_to_nvfp4 = real




def test_a_full_schedule_samples_the_named_steps_and_a_short_one_the_same_places():
    assert gq.sampled_steps(38, (0, 12, 25, 37)) == (0, 12, 25, 37)
    assert gq.sampled_steps(50, (0, 12, 25, 37)) == (0, 12, 25, 37)
    assert gq.sampled_steps(8, (0, 12, 25, 37)) == (0, 2, 4, 6)
    assert gq.sampled_steps(4, (0, 12, 25, 37)) == (0, 1, 2, 3)
    assert gq.sampled_steps(1, (0, 12, 25, 37)) == (0,)
    assert gq.sampled_steps(0, (0, 12, 25, 37)) == (0,)




def _two_linears():
    torch.manual_seed(0)
    return {
        "blocks.0.attention.to_q": torch.nn.Linear(16, 8, bias = False),
        "blocks.1.attention.to_q": torch.nn.Linear(16, 8, bias = False),
    }


def test_the_hessians_only_accumulate_on_the_armed_steps():
    layers = _two_linears()
    acc = gq.HessianAccumulator(layers).attach()
    try:
        x = torch.randn(4, 16)
        layers["blocks.0.attention.to_q"](x)  # not armed: nothing recorded
        assert acc.samples["blocks.0.attention.to_q"] == 0
        assert acc.unseen() == sorted(layers)

        acc.arm_first((0, 2))
        layers["blocks.0.attention.to_q"](x)
        assert acc.samples["blocks.0.attention.to_q"] == 4
        want = x.transpose(0, 1).float() @ x.float()
        assert torch.allclose(acc.hessians["blocks.0.attention.to_q"], want, atol = 1e-4)

        callback = acc.step_callback((0, 2))
        assert callback(None, 0, None, {"latents": 1}) == {"latents": 1}
        assert acc.active is False  # step 1 is not sampled
        layers["blocks.0.attention.to_q"](x)
        assert acc.samples["blocks.0.attention.to_q"] == 4
        callback(None, 1, None, {})
        assert acc.active is True  # step 2 is
        layers["blocks.0.attention.to_q"](x)
        assert acc.samples["blocks.0.attention.to_q"] == 8
        assert torch.allclose(
            acc.normalised()["blocks.0.attention.to_q"],
            acc.hessians["blocks.0.attention.to_q"] / 8,
        )
        assert acc.unseen() == ["blocks.1.attention.to_q"]
    finally:
        acc.detach()
    layers["blocks.0.attention.to_q"](torch.randn(4, 16))
    assert acc.samples["blocks.0.attention.to_q"] == 8  # detached: the hook is gone


def test_a_layer_set_that_does_not_fit_the_budget_is_refused_before_it_allocates():
    layers = {"big": torch.nn.Linear(4096, 8, bias = False)}
    acc = gq.HessianAccumulator(layers, budget_bytes = 1024)
    assert acc.planned_bytes() == 4 * 4096 * 4096
    refusal = acc.budget_refusal()
    assert refusal is not None and "big" in refusal and "budget" in refusal
    with pytest.raises(ValueError):
        acc.attach()
    assert acc.hessians == {}




def test_only_the_layers_that_measure_better_take_their_correction():
    scores = {
        "a": {"err_rtn": 1.0, "err_gptq": 0.5, "ratio": 0.5, "improved": True},
        "b": {"err_rtn": 1.0, "err_gptq": 1.2, "ratio": 1.2, "improved": False},
        "c": {"err_rtn": 1.0, "err_gptq": 3.0, "ratio": 3.0, "improved": False},
    }
    plan = gq.plan_corrections(scores)
    assert plan["apply"] == ["a"]
    assert plan["counts"] == {"applied": 1, "applied_regressed": 0, "skipped_no_gain": 2}
    relaxed = gq.plan_corrections(scores, max_regressions = 1)
    assert relaxed["apply"] == ["a", "b"]
    assert relaxed["counts"] == {"applied": 2, "applied_regressed": 1, "skipped_no_gain": 1}




def test_the_baked_scale_is_the_running_max_over_every_forward():
    layers = _two_linears()
    acc = gq.ActivationAmaxAccumulator(layers).attach()
    try:
        small = torch.full((2, 16), 0.5)
        big = torch.full((2, 16), 4.0)
        layers["blocks.0.attention.to_q"](big)
        layers["blocks.0.attention.to_q"](small)  # a later, smaller step must not lower it
        assert float(acc.amax["blocks.0.attention.to_q"]) == pytest.approx(4.0)
        assert acc.unseen() == ["blocks.1.attention.to_q"]
        scales = acc.global_scales()
        assert scales["blocks.0.attention.to_q"] == pytest.approx(6.0 * 448.0 / 4.0)
        assert "blocks.1.attention.to_q" not in scales
    finally:
        acc.detach()


def test_a_non_finite_forward_never_becomes_the_baked_scale():
    """A non-finite activation must never reach the baked scale: that is the black-frame latch."""
    layers = _two_linears()
    acc = gq.ActivationAmaxAccumulator(layers).attach()
    try:
        layers["blocks.0.attention.to_q"](torch.full((2, 16), 2.0))
        layers["blocks.0.attention.to_q"](torch.full((2, 16), float("inf")))
        assert float(acc.amax["blocks.0.attention.to_q"]) == pytest.approx(2.0)
        assert acc.global_scales()["blocks.0.attention.to_q"] == pytest.approx(6.0 * 448.0 / 2.0)
    finally:
        acc.detach()




@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a CUDA device")
def test_cuda_gptq_beats_rtn_on_a_512_square_layer_through_torchaos_quantiser():
    pytest.importorskip("torchao.prototype.mx_formats")
    torch.manual_seed(3407)
    weight = (torch.randn(512, 512, device = "cuda") * 0.05).to(torch.bfloat16)
    hessian = _spd(512, seed = 7, device = "cuda")
    corrected, damp = gq.gptq_correct(weight, hessian)
    assert damp in gq.DAMP_LADDER
    score = gq.score_correction(weight, corrected, hessian)
    assert score["err_gptq"] < score["err_rtn"]
    assert score["improved"] is True
    assert 0.0 < score["ratio"] < 1.0
