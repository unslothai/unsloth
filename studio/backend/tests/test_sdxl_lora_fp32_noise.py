# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""SDXL LoRA: the noisy latents and the v-target are computed in fp32, then cast (diffusers #15300)."""

from __future__ import annotations

import ast
import inspect

import pytest
import torch

diffusers = pytest.importorskip("diffusers")

from core.training import diffusion_lora_trainer as trainer

SDXL_SCHEDULER = dict(
    num_train_timesteps = 1000,
    beta_start = 0.00085,
    beta_end = 0.012,
    beta_schedule = "scaled_linear",
)


def _reference(scheduler, latents, noise, timesteps):
    acp = scheduler.alphas_cumprod.to(torch.float64)[timesteps].view(-1, 1, 1, 1)
    x0, eps = latents.double(), noise.double()
    noisy = acp.sqrt() * x0 + (1 - acp).sqrt() * eps
    velocity = acp.sqrt() * eps - (1 - acp).sqrt() * x0
    return noisy, velocity


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_noise_and_velocity_match_fp64_at_ends_and_middle(dtype):
    scheduler = diffusers.DDPMScheduler(prediction_type = "v_prediction", **SDXL_SCHEDULER)
    gen = torch.Generator().manual_seed(0)
    timesteps = torch.tensor([0, 1, 500, 998, 999])
    latents = torch.randn(len(timesteps), 4, 8, 8, generator = gen).to(dtype)
    noise = torch.randn(len(timesteps), 4, 8, 8, generator = gen).to(dtype)
    ref_noisy, ref_vel = _reference(scheduler, latents, noise, timesteps)

    noisy, velocity = trainer._sdxl_noisy_and_velocity(scheduler, latents, noise, timesteps, True)
    assert noisy.dtype == dtype and velocity.dtype == dtype
    tol = torch.finfo(dtype).eps * 2
    for got, ref in ((noisy, ref_noisy), (velocity, ref_vel)):
        err = ((got.double() - ref).abs() / ref.abs().clamp_min(1e-2)).amax()
        assert err <= tol, (dtype, float(err))

    assert not torch.equal(noisy[0], latents[0])

    eps_noisy, eps_vel = trainer._sdxl_noisy_and_velocity(
        scheduler, latents, noise, timesteps, False
    )
    assert torch.equal(eps_noisy, noisy) and eps_vel is None


def test_stock_bf16_scheduler_drops_the_noise_at_t0():
    """Documents the defect the helper avoids: the scheduler in bf16 returns the clean latents at t=0."""
    scheduler = diffusers.DDPMScheduler(**SDXL_SCHEDULER)
    latents = torch.randn(1, 4, 8, 8).bfloat16()
    noise = torch.randn(1, 4, 8, 8).bfloat16()
    t = torch.tensor([0])
    noisy, _ = _reference(scheduler, latents, noise, t)
    stock = scheduler.add_noise(latents, noise, t)
    assert (stock.double() - noisy).abs().amax() > (
        trainer._sdxl_noisy_and_velocity(scheduler, latents, noise, t, False)[0].double() - noisy
    ).abs().amax()


def test_training_loop_does_not_call_the_scheduler_in_weight_dtype():
    tree = ast.parse(inspect.getsource(trainer.run_diffusion_lora_training))
    direct = [
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in ("add_noise", "get_velocity")
    ]
    assert direct == []
    helper_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_sdxl_noisy_and_velocity"
    ]
    assert len(helper_calls) == 1
