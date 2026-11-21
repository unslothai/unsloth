# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""CPU tests: the static step skip on a U-Net pipeline (SDXL: ``pipe.unet``, CFG batched into one call per step)."""

from __future__ import annotations

import types

import pytest

from core.inference import diffusion_cache as dcache
from core.inference import diffusion_step_skip as ss

torch = pytest.importorskip("torch")

SDXL = "stabilityai/stable-diffusion-xl-base-1.0"


class _UNet(torch.nn.Module):
    """UNet2DConditionModel's call shape: positional (sample, timestep), tuple out with return_dict=False."""

    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(
        self,
        sample,
        timestep,
        encoder_hidden_states = None,
        return_dict = True,
    ):
        self.calls += 1
        return (sample * 0 + float(timestep.reshape(-1)[0]),)


def _sdxl_pipe(unet = None):
    return types.SimpleNamespace(unet = unet if unet is not None else _UNet(), vae = object())


def _render(pipe, steps):
    ss.reset_static_step_skip(pipe, steps, step_signal = True)
    outs = []
    for i in range(steps):
        # SDXL batches CFG: one U-Net call per step on a doubled batch.
        outs.append(
            pipe.unet(
                torch.zeros(2, 4, 8, 8), torch.tensor([1000.0 * (1 - i / steps)]), return_dict = False
            )
        )
        ss.mark_step_end(pipe)
    return outs


def test_unet_pipeline_installs_the_layer():
    pipe = _sdxl_pipe()
    assert (
        ss.install_static_step_skip(pipe, settings = {"mode": "taylor1", "every": 2})
        == dcache.TC_STATIC
    )
    assert ss._find(pipe) is pipe.unet.__dict__["forward"]
    assert ss.uninstall_static_step_skip(pipe)
    assert "forward" not in pipe.unet.__dict__ and ss._find(pipe) is None


def test_a_transformer_still_wins_over_a_unet():
    dit = _UNet()
    pipe = types.SimpleNamespace(transformer = dit, unet = _UNet())
    assert ss.install_static_step_skip(pipe, settings = {"mode": "taylor1"}) == dcache.TC_STATIC
    assert "forward" in dit.__dict__ and "forward" not in pipe.unet.__dict__


def test_no_denoiser_declines():
    assert ss.install_static_step_skip(types.SimpleNamespace(vae = object())) is None


def test_sdxl_auto_row_skips_every_other_step_at_25_on_max():
    plan = dcache.auto_static_skip_plan((SDXL,), "max", 25, env = {})
    assert plan == {"every": 2, "min_steps": 25}
    # Default tier never skips automatically; the distilled turbo sibling (1 step) has no row.
    assert dcache.auto_static_skip_plan((SDXL,), "default", 25, env = {}) is None
    assert dcache.auto_static_skip_plan(("stabilityai/sdxl-turbo",), "max", 1, env = {}) is None
    pipe = _sdxl_pipe()
    assert (
        ss.install_static_step_skip(pipe, settings = ss.auto_static_settings(plan, env = {}))
        == dcache.TC_STATIC
    )
    _render(pipe, 25)
    want = ss.static_schedule(25, every = 2)
    assert pipe.unet.calls == sum(want) == 16
    assert ss.static_skip_stats(pipe)["stats"] == {"calls": 25, "computed": 16, "skipped": 9}
    # Fewer steps than measured compute every step.
    pipe.unet.calls = 0
    _render(pipe, 20)
    assert pipe.unet.calls == 20


def test_skip_survives_diffusers_model_offload_reattach():
    """Studio installs the skip before placement; model offload then wraps it, and diffusers removes and re-adds
    its hooks around every call (maybe_free_model_hooks -> enable_model_cpu_offload). The layer must stay in the
    call path through that cycle."""
    hooks = pytest.importorskip("accelerate.hooks")
    pipe = _sdxl_pipe()
    assert (
        ss.install_static_step_skip(pipe, settings = {"mode": "taylor1", "every": 2})
        == dcache.TC_STATIC
    )
    for _ in range(2):
        hooks.add_hook_to_module(pipe.unet, hooks.ModelHook())
        pipe.unet.calls = 0
        _render(pipe, 25)
        assert pipe.unet.calls == 16, "the skip layer fell out of the hooked call path"
        hooks.remove_hook_from_module(pipe.unet)
        assert pipe.unet.__dict__.get("forward") is ss._find(pipe)
