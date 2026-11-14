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


def _int8_dit(load_under_inference_mode):
    pytest.importorskip("torchao")
    from torchao.quantization import quantize_

    from core.inference.diffusion_transformer_quant import TQ_INT8, _make_quant_config

    torch.manual_seed(0)

    class Dit(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a = torch.nn.Linear(128, 128)
            self.b = torch.nn.Linear(128, 128)

        def forward(
            self,
            hidden_states,
            timestep,
            encoder_hidden_states,
            return_dict = False,
        ):
            h = self.b(torch.nn.functional.gelu(self.a(hidden_states)))
            return (h * timestep + encoder_hidden_states.sum(),)

    ctx = torch.inference_mode() if load_under_inference_mode else torch.no_grad()
    with ctx:
        dit = Dit().to("cuda:0", torch.bfloat16).eval()
        quantize_(dit, _make_quant_config(TQ_INT8))
    return dit


@pytest.mark.parametrize("load_under_inference_mode", [False, True])
def test_int8_replica_builds_inside_inference_mode_on_the_second_card(load_under_inference_mode):
    # Inside inference_mode: outside-made weights raised "Cannot set version_counter"; inside-made stayed on cuda:0.
    dit = _int8_dit(load_under_inference_mode)
    latents = torch.randn(4, 128, device = "cuda:0", dtype = torch.bfloat16)
    cond = torch.randn(3, 128, device = "cuda:0", dtype = torch.bfloat16)
    neg = torch.randn(3, 128, device = "cuda:0", dtype = torch.bfloat16)
    with torch.inference_mode():
        reference = _render(dit, latents, cond, neg, 4)
        router = _arm(dit)
        out = _render(dit, latents, cond, neg, 4)
    torch.cuda.synchronize()
    assert router.disabled_reason is None, router.disabled_reason
    assert router.stats["steps_parallel"] > 0
    assert {str(t.device) for t in router.replica.parameters()} == {"cuda:1"}
    assert torch.equal(out, reference)


def test_an_in_place_edit_of_inference_weights_rebuilds_the_replica():
    # Inference tensors carry no version counter: the fingerprint alone kept a replica holding the old values.
    with torch.inference_mode():
        dit, latents, cond, neg = _setup()
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
    assert router.stats["replica_rebuilds"] == 1
    assert torch.equal(out, reference)


class _PrefixKVDit(torch.nn.Module):
    """Qwen-Image-2.1 prefix KV cache shape: a list of per-layer K/V holders."""

    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(4, 4)

    def forward(
        self,
        hidden_states,
        timestep,
        encoder_hidden_states,
        kv_cache = None,
        kv_cache_mode = None,
        return_dict = False,
    ):
        key = self.proj(hidden_states)
        for layer in kv_cache.layer_caches:
            if kv_cache_mode == "extract":
                layer.k = self.proj(encoder_hidden_states).clone()
            key = torch.cat([layer.k, key], dim = 0)[-hidden_states.shape[0] :] + layer.k.sum(0)
        return (key * timestep,)


def _kv_cache(layers = 3):
    from types import SimpleNamespace
    class Cache:
        def __init__(self):
            self.layer_caches = [SimpleNamespace(k = None) for _ in range(layers)]

    return Cache()


def _kv_render(dit, latents, cond, neg, steps):
    x = latents.clone()
    caches = (_kv_cache(), _kv_cache())
    for i in range(steps):
        t = torch.tensor(1.0 - i / steps, device = x.device)
        mode = "extract" if i == 0 else "cached"
        a = dit(
            hidden_states = x,
            timestep = t,
            encoder_hidden_states = cond,
            kv_cache = caches[0],
            kv_cache_mode = mode,
        )[0]
        b = dit(
            hidden_states = x,
            timestep = t,
            encoder_hidden_states = neg,
            kv_cache = caches[1],
            kv_cache_mode = mode,
        )[0]
        x = x - 0.1 * (b + 4.0 * (a - b))
    return x


def test_prefix_kv_cache_branch_runs_on_the_second_card_and_stays_exact():
    torch.manual_seed(0)
    dit = _PrefixKVDit().cuda(0)
    latents = torch.randn(2, 4, device = "cuda:0")
    cond = torch.randn(3, 4, device = "cuda:0")
    neg = torch.randn(3, 4, device = "cuda:0")
    with torch.no_grad():
        reference = [_kv_render(dit, latents, cond, neg, 5) for _ in range(2)]
        router = _arm(dit)
        outs = []
        for _ in range(2):
            outs.append(_kv_render(dit, latents, cond, neg, 5))
            router.end_render()
    torch.cuda.synchronize()
    assert router.disabled_reason is None, router.disabled_reason
    assert router.stats["steps_parallel"] >= 6 and router.stats["kv_cache_copies"] == 2
    assert all(torch.equal(o, r) for o, r in zip(outs, reference))


def test_a_step_graph_on_the_second_card_records_its_forward():
    # Regression: cuda:1 captured on torch's cuda:0 default capture stream, so replays returned the recording's output.
    from core.inference import diffusion_cuda_graph as cg

    def graphed_steps(card):
        torch.manual_seed(0)
        dit = _Dit().cuda(card)
        handle = cg.GraphedForward(dit).enable()
        outs, refs = [], []
        with torch.cuda.device(card), torch.inference_mode():
            for value in (1.0, 2.0, 3.0):
                x = torch.full((2, 4), value, device = f"cuda:{card}")
                t = torch.tensor(value, device = f"cuda:{card}")
                e = torch.randn(3, 4, device = f"cuda:{card}")
                refs.append(dit.__class__.forward(dit, x, t, e)[0].clone())
                outs.append(
                    handle(hidden_states = x, timestep = t, encoder_hidden_states = e, return_dict = False)[
                        0
                    ]
                )
            torch.cuda.synchronize()
        return handle, outs, refs

    first, outs0, refs0 = graphed_steps(0)
    second, outs1, refs1 = graphed_steps(1)
    for handle in (first, second):
        assert handle.stats["captures"] == 1 and handle.stats["replays"] == 3, handle.stats
    for out, ref in zip(outs0 + outs1, refs0 + refs1):
        assert torch.equal(out.cpu(), ref.cpu())
