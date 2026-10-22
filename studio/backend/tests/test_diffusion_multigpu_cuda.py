# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""CFG-parallel on two real CUDA cards. On one device ``.to()`` aliases storage, so the CPU tests in
test_diffusion_multigpu.py cannot see a replica that diverged from its source; these can. Skipped below two cards."""

from __future__ import annotations

import pytest
import torch

from core.inference import diffusion_multigpu as mg

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason = "needs two CUDA cards"
)


class _Dit(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(4, 4)
        self.register_buffer("scale", torch.ones(4))

    def forward(
        self,
        hidden_states,
        timestep,
        encoder_hidden_states,
        return_dict = False,
    ):
        return ((self.proj(hidden_states) * self.scale) * timestep + encoder_hidden_states.sum(),)


def _render(
    dit,
    latents,
    cond,
    neg,
    steps,
    between = None,
):
    x = latents.clone()
    for i in range(steps):
        t = torch.tensor(1.0 - i / steps, device = x.device)
        a = dit(hidden_states = x, timestep = t, encoder_hidden_states = cond)[0]
        if between is not None:
            between(i, x)
        b = dit(hidden_states = x, timestep = t, encoder_hidden_states = neg)[0]
        x = x - 0.1 * (b + 4.0 * (a - b))
    return x


def _setup():
    torch.manual_seed(0)
    dit = _Dit().cuda(0)
    latents = torch.randn(2, 4, device = "cuda:0")
    cond = torch.randn(3, 4, device = "cuda:0")
    neg = torch.randn(3, 4, device = "cuda:0")
    return dit, latents, cond, neg


def _arm(dit):
    router = mg.CfgParallelRouter(dit, primary = "cuda:0", secondary = "cuda:1")
    dit.forward = router
    return router


def test_an_input_edited_after_the_prefetch_reaches_an_unpredicted_replica_call():
    dit, latents, cond, neg = _setup()

    def edit(step, x):
        if (
            step == 1
        ):  # the first step of a render runs unpredicted (no template yet): only the prefetch was made
            x.add_(1.0)

    with torch.no_grad():
        reference = [_render(dit, latents, cond, neg, 4), _render(dit, latents, cond, neg, 4, edit)]
        router = _arm(dit)
        first = _render(dit, latents, cond, neg, 4)
        router.end_render()
        second = _render(dit, latents, cond, neg, 4, edit)
    torch.cuda.synchronize()
    assert router.stats["replica_calls"] > 0 and router.replica is not None
    assert torch.equal(first, reference[0])
    assert torch.equal(second, reference[1])


def test_an_edited_buffer_rebuilds_the_replica():
    dit, latents, cond, neg = _setup()
    with torch.no_grad():
        router = _arm(dit)
        _render(dit, latents, cond, neg, 3)
        router.end_render()
        assert router.replica is not None
        dit.scale.mul_(3.0)
        dit.forward = dit.__class__.forward.__get__(dit)
        reference = _render(dit, latents, cond, neg, 4)
        dit.forward = router
        out = _render(dit, latents, cond, neg, 4)
    torch.cuda.synchronize()
    assert router.stats["replica_rebuilds"] >= 1
    assert torch.equal(out, reference)


def _graphed_render(dit, latents, cond, neg, steps, renders):
    outs = []
    for _ in range(renders):
        x = latents.clone()
        for i in range(steps):
            t = torch.tensor(1.0 - i / steps, device = x.device)
            a = dit(hidden_states = x, timestep = t, encoder_hidden_states = cond, return_dict = False)[0]
            b = dit(hidden_states = x, timestep = t, encoder_hidden_states = neg, return_dict = False)[0]
            x = x - 0.1 * (b + 4.0 * (a - b))
        outs.append(x)
        router = dit.__dict__.get("_unsloth_cfg_router")
        if router is not None:
            router.end_render()
    torch.cuda.synchronize()
    return outs


def test_default_tier_graph_under_the_router_replays_and_stays_exact():
    from core.inference import diffusion_cuda_graph as cg

    dit, latents, cond, neg = _setup()
    with torch.no_grad():
        single = cg.GraphedForward(dit).enable()
        reference = _graphed_render(dit, latents, cond, neg, 8, 3)
        assert single.stats["captures"] >= 1 and single.stats["replays"] > 0
        single.disable()
        router = _arm(dit)
        dit._unsloth_cfg_router = router
        # the deferred speed profile arms its graph after the router
        graph = cg.GraphedForward(dit).enable()
        assert dit.__dict__["forward"] is router and router.inner is graph
        out = _graphed_render(dit, latents, cond, neg, 8, 3)
    for a, b in zip(out, reference):
        assert torch.equal(a, b)
    assert graph.stats["captures"] >= 1 and graph.stats["replays"] > 0
    assert router.stats["steps_parallel"] > 0 and router.disabled_reason is None
    assert str(next(router.replica.parameters()).device) == "cuda:1"
    captures = graph.stats["captures"]
    with torch.no_grad():
        _graphed_render(dit, latents, cond * 2, neg, 8, 1)
    assert graph.stats["captures"] == captures  # a follow-up prompt records nothing new
    graph.disable()


def test_zimage_pair_runs_on_two_cards_and_stays_exact():
    zmod = pytest.importorskip("diffusers.models.transformers.transformer_z_image")
    from core.inference import diffusion_zimage_cfg_split as zs

    torch.manual_seed(0)
    model = (
        zmod.ZImageTransformer2DModel(
            all_patch_size = (2,),
            all_f_patch_size = (1,),
            in_channels = 16,
            dim = 32,
            n_layers = 2,
            n_refiner_layers = 1,
            n_heads = 2,
            n_kv_heads = 2,
            norm_eps = 1e-5,
            qk_norm = True,
            cap_feat_dim = 16,
            rope_theta = 256.0,
            t_scale = 1000.0,
            axes_dims = [8, 4, 4],
            axes_lens = [512, 64, 64],
        )
        .eval()
        .cuda(0)
    )

    class Pipe:
        transformer = model

    pipe = Pipe()
    g = torch.Generator().manual_seed(7)
    latent0 = torch.randn(16, 1, 8, 8, generator = g).cuda(0)
    cond = torch.randn(9, 16, generator = g).cuda(0)
    neg = torch.randn(40, 16, generator = g).cuda(0)

    def steps():
        latent = latent0.clone()
        with torch.no_grad(), zs.render(pipe, 1):
            for i in range(4):
                t = torch.tensor([1.0 - i / 4], device = "cuda:0").repeat(2)
                out = model([latent.clone(), latent.clone()], t, [cond, neg], return_dict = False)[0]
                latent = latent - 0.1 * (out[1] + 3.0 * (out[0] - out[1]))
        torch.cuda.synchronize()
        return latent

    try:
        zs.install_zimage_cfg_split(pipe, offload_policy = "none")
        reference = steps()
        router = _arm(model)
        model._unsloth_cfg_router = router
        out = steps()
    finally:
        zs.uninstall_zimage_cfg_split(pipe)
    assert torch.equal(out, reference)
    assert router.stats["pairs_parallel"] == 4 and router.disabled_reason is None
