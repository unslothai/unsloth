# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""CFG batching (diffusion_cfg_batch): both guidance branches through one offloaded forward, CPU fakes."""

from __future__ import annotations

import types

import pytest
import torch

from core.inference import diffusion_cfg_batch as cb


class Flux2Transformer2DModel(torch.nn.Module):
    """Stateless per-sample forward in the FLUX.2-klein call shape (hidden, timestep, text, ids)."""

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.linspace(0.5, 1.5, 8))
        self.calls = []

    def forward(
        self,
        hidden_states,
        timestep,
        encoder_hidden_states,
        txt_ids,
        img_ids,
        guidance = None,
        joint_attention_kwargs = None,
        return_dict = False,
    ):
        self.calls.append(int(hidden_states.shape[0]))
        text = encoder_hidden_states.mean(dim = 1, keepdim = True)
        out = (
            hidden_states * self.w
            + text
            + timestep.view(-1, 1, 1)
            + txt_ids.float().mean()
            + img_ids.float().mean()
        )
        return (out,)


class QwenImageTransformer2DModel(torch.nn.Module):
    """Qwen-Image call shape: a per-prompt-length text with a mask, and a per-sample ``img_shapes`` list (as the
    zero_cond_t path reads it)."""

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.full((8,), 0.75))

    def forward(
        self,
        hidden_states,
        timestep,
        encoder_hidden_states,
        encoder_hidden_states_mask = None,
        img_shapes = None,
        guidance = None,
        attention_kwargs = None,
        return_dict = False,
    ):
        assert len(img_shapes) == hidden_states.shape[0]
        if encoder_hidden_states_mask is None:
            encoder_hidden_states_mask = torch.ones(encoder_hidden_states.shape[:2])
        m = encoder_hidden_states_mask.float().unsqueeze(-1)
        text = (encoder_hidden_states * m).sum(1, keepdim = True) / m.sum(1, keepdim = True)
        return (hidden_states * self.w + text + timestep.view(-1, 1, 1),)


for cls in (Flux2Transformer2DModel, QwenImageTransformer2DModel):
    cls.__module__ = "diffusers.models.fake"


def _offloaded(module):
    # what the decline reads as offload machinery: a diffusers group-offloading registry on a submodule
    sub = torch.nn.Linear(1, 1)
    module.add_module("offloaded_group", sub)
    sub._diffusers_hook = types.SimpleNamespace(hooks = {"group_offloading": object()})
    return module


class _Pipe:
    def __init__(self, transformer):
        self.transformer = transformer


def _flux_loop(
    model,
    steps = 6,
    *,
    neg = None,
    guided = None,
    between = None,
):
    """The diffusers FLUX.2-klein loop: per step one latent object, cond then uncond call, CFG combine."""
    torch.manual_seed(0)
    latents = torch.randn(1, 4, 8)
    pos = torch.randn(1, 5, 8)
    neg = torch.randn(1, 5, 8) if neg is None else neg
    txt_ids = torch.zeros(1, 5, 4)
    img_ids = torch.ones(1, 4, 4)
    outs = []
    for i in range(steps):
        t = torch.full((1,), 1.0 - i / steps)
        latent_model_input = latents.clone()
        cond = model(
            hidden_states = latent_model_input,
            timestep = t / 1000,
            encoder_hidden_states = pos,
            txt_ids = txt_ids,
            img_ids = img_ids,
            return_dict = False,
        )[0]
        pred = cond
        if between is not None:
            between(i, neg)
        if guided is None or guided(i):
            uncond = model(
                hidden_states = latent_model_input,
                timestep = t / 1000,
                encoder_hidden_states = neg,
                txt_ids = txt_ids,
                img_ids = img_ids,
                return_dict = False,
            )[0]
            pred = uncond + 4.0 * (cond - uncond)
        latents = latents - 0.1 * pred
        outs.append(pred)
    return latents, outs


def _armed(
    model,
    monkeypatch,
    mode = None,
):
    if mode is not None:
        monkeypatch.setenv(cb.CFG_BATCH_ENV, mode)
    pipe = _Pipe(_offloaded(model))
    assert cb.install_cfg_batch(pipe, offload_policy = "group") is None
    return pipe, cb.router_for(pipe)


def test_batched_steps_match_two_calls_and_engage(monkeypatch):
    ref, _ = _flux_loop(_offloaded(Flux2Transformer2DModel()))
    model = Flux2Transformer2DModel()
    pipe, router = _armed(model, monkeypatch)
    got, _ = _flux_loop(model)
    torch.testing.assert_close(got, ref)
    assert router.stats["batched_steps"] == 4 and router.stats["hits"] == 4
    assert model.calls[-2:] == [2, 2]
    assert cb.cfg_batch_status(pipe)["batched_steps"] == 4
    cb.end_cfg_batch_render(pipe)
    assert cb.cfg_batch_status(pipe)["plain_pair_s"] is not None or not torch.cuda.is_available()
    assert cb.cfg_batch_status(pipe)["shape_off"] is None


def test_wan_expanded_timesteps_serve_the_second_branch(monkeypatch):
    """WanPipeline passes ``timestep = t.expand(batch)`` per call: a fresh stride-0 tensor each branch. Equal bytes
    must count as equal; a byte view of the stride-0 tensor raised, so every step missed and three misses turned
    batching off."""

    def loop(model):
        torch.manual_seed(0)
        latents = torch.randn(1, 4, 8)
        pos, neg = torch.randn(1, 5, 8), torch.randn(1, 5, 8)
        txt_ids, img_ids = torch.zeros(1, 5, 4), torch.ones(1, 4, 4)
        for t in torch.tensor([999, 941, 883, 825, 767, 709]):
            branch = {"txt_ids": txt_ids, "img_ids": img_ids, "return_dict": False}
            c = model(hidden_states = latents, timestep = t.expand(latents.shape[0]), encoder_hidden_states = pos, **branch)[0]  # fmt: skip
            u = model(hidden_states = latents, timestep = t.expand(latents.shape[0]), encoder_hidden_states = neg, **branch)[0]  # fmt: skip
            latents = latents - 0.1 * (u + 4.0 * (c - u))
        return latents

    ref = loop(_offloaded(Flux2Transformer2DModel()))
    model = Flux2Transformer2DModel()
    _pipe, router = _armed(model, monkeypatch)
    torch.testing.assert_close(loop(model), ref)
    assert router.stats["hits"] == 4 and router.stats["misses"] == 0, router.describe()
    assert router.shape.off is None


def test_qwen_unequal_prompt_lengths_and_img_shapes_batch(monkeypatch):
    def loop(model):
        torch.manual_seed(1)
        latents = torch.randn(1, 4, 8)
        pos, neg = torch.randn(1, 7, 8), torch.randn(1, 3, 8)
        pos_mask, neg_mask = torch.ones(1, 7, dtype = torch.long), None
        img_shapes = [[(1, 2, 2)]]
        for i in range(6):
            t = torch.full((1,), 1.0 - i / 6)
            c = model(
                hidden_states = latents,
                timestep = t / 1000,
                encoder_hidden_states = pos,
                encoder_hidden_states_mask = pos_mask,
                img_shapes = img_shapes,
                return_dict = False,
            )[0]
            u = model(
                hidden_states = latents,
                timestep = t / 1000,
                encoder_hidden_states = neg,
                encoder_hidden_states_mask = neg_mask,
                img_shapes = img_shapes,
                return_dict = False,
            )[0]
            latents = latents - 0.1 * (u + 4.0 * (c - u))
        return latents

    ref = loop(_offloaded(QwenImageTransformer2DModel()))
    model = QwenImageTransformer2DModel()
    _pipe, router = _armed(model, monkeypatch)
    torch.testing.assert_close(loop(model), ref)
    assert router.stats["batched_steps"] == 4, router.describe()


def test_a_changed_second_branch_reruns_plain_and_stops_after_three_misses(monkeypatch):
    # uncond prompt rewritten in place: the doubled call must miss, rerun plain, and match the two-call result
    def between(i, neg):
        if i >= 2:
            neg.add_(0.5)

    ref, _ = _flux_loop(
        _offloaded(Flux2Transformer2DModel()), steps = 8, neg = torch.zeros(1, 5, 8), between = between
    )
    model = Flux2Transformer2DModel()
    _pipe, router = _armed(model, monkeypatch)
    got, _ = _flux_loop(model, steps = 8, neg = torch.zeros(1, 5, 8), between = between)
    torch.testing.assert_close(got, ref, rtol = 0, atol = 0)
    assert router.stats["misses"] == 3 and router.stats["hits"] == 0
    assert router.shape.off is not None and "3 consecutive misses" in router.shape.off


def test_guidance_ending_wastes_at_most_one_doubled_call(monkeypatch):
    model = Flux2Transformer2DModel()
    _pipe, router = _armed(model, monkeypatch)
    ref, _ = _flux_loop(_offloaded(Flux2Transformer2DModel()), steps = 8, guided = lambda i: i < 5)
    got, _ = _flux_loop(model, steps = 8, guided = lambda i: i < 5)
    torch.testing.assert_close(got, ref)
    assert router.stats["misses"] == 1  # step 6's doubled call found no second branch
    assert model.calls[-2:] == [1, 1]


def test_a_third_call_per_step_turns_batching_off(monkeypatch):
    model = Flux2Transformer2DModel()
    _pipe, router = _armed(model, monkeypatch)
    x = torch.randn(1, 4, 8)
    kw = dict(
        timestep = torch.ones(1),
        encoder_hidden_states = torch.randn(1, 5, 8),
        txt_ids = torch.zeros(1, 5, 4),
        img_ids = torch.ones(1, 4, 4),
    )
    for _ in range(3):
        model(hidden_states = x, **kw)
    assert router.disabled == "more than two denoiser calls per step"


def test_a_displaced_router_passes_calls_through(monkeypatch):
    model = Flux2Transformer2DModel()
    pipe, router = _armed(model, monkeypatch)
    model.__dict__["forward"] = lambda *a, **k: router(*a, **k)  # an outer layer took the slot
    _flux_loop(model)
    assert router.stats["batched_steps"] == 0 and router.describe()["displaced"]


def test_install_uninstall_restores_the_forward_slot(monkeypatch):
    model = _offloaded(Flux2Transformer2DModel())
    wrapper = model.forward
    pipe = _Pipe(model)
    assert cb.install_cfg_batch(pipe, offload_policy = "group") is None
    assert isinstance(model.__dict__["forward"], cb.CfgBatchRouter)
    cb.uninstall_cfg_batch(pipe)
    assert "forward" not in model.__dict__ and model.forward == wrapper
    cb.end_cfg_batch_render(pipe)  # no router: nothing to do
    assert cb.cfg_batch_status(pipe) is None


def test_switches_are_independent_of_multi_gpu(monkeypatch):
    pipe = _Pipe(_offloaded(Flux2Transformer2DModel()))
    monkeypatch.setenv("UNSLOTH_DIFFUSION_MULTIGPU", "0")
    assert cb.cfg_batch_decline(pipe, offload_policy = "group") is None
    monkeypatch.setenv(cb.CFG_BATCH_ENV, "0")
    assert cb.cfg_batch_decline(pipe, offload_policy = "group") == f"{cb.CFG_BATCH_ENV}=0"


def test_declines(monkeypatch):
    resident = _Pipe(Flux2Transformer2DModel())
    assert "resident" in cb.cfg_batch_decline(resident, offload_policy = "none")

    class ZImageTransformer2DModel(Flux2Transformer2DModel):
        pass

    ZImageTransformer2DModel.__module__ = "diffusers.models.fake"
    assert "not on the stateless-forward list" in cb.cfg_batch_decline(
        _Pipe(_offloaded(ZImageTransformer2DModel())), offload_policy = "group"
    )

    class ComfyNVFP4DynamicLinear(torch.nn.Linear):
        pass

    dyn = _offloaded(Flux2Transformer2DModel())
    dyn.add_module("proj", ComfyNVFP4DynamicLinear(2, 2))
    assert "per-call activation scaling" in cb.cfg_batch_decline(_Pipe(dyn), offload_policy = "group")

    class NVFP4FlashInferLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("a_gsf", torch.ones(()))

    frozen = _offloaded(Flux2Transformer2DModel())
    frozen.add_module("proj", NVFP4FlashInferLinear())
    assert cb.cfg_batch_decline(_Pipe(frozen), offload_policy = "group") is None

    two = _Pipe(_offloaded(Flux2Transformer2DModel()))
    two.transformer_2 = torch.nn.Linear(1, 1)
    assert cb.cfg_batch_decline(two, offload_policy = "group") == "two-expert denoiser"
    graphs = _Pipe(_offloaded(Flux2Transformer2DModel()))
    graphs._unsloth_cuda_graphs = (object(),)
    assert "CUDA graphs" in cb.cfg_batch_decline(graphs, offload_policy = "group")
    cached = _offloaded(Flux2Transformer2DModel())
    cached._diffusers_hook = types.SimpleNamespace(hooks = {"first_block_cache": object()})
    assert "first_block_cache" in cb.cfg_batch_decline(_Pipe(cached), offload_policy = "group")


def test_sync_waits_on_the_latents_card(monkeypatch):
    seen = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device = None: seen.append(device))
    router = cb.CfgBatchRouter(Flux2Transformer2DModel(), reason = "test")
    latent = types.SimpleNamespace(device = torch.device("cuda", 3))
    assert router._sync(latent)
    assert seen == [torch.device("cuda", 3)]


def test_merge_pads_text_and_joins_per_sample_lists():
    a = (
        (),
        {
            "hidden_states": torch.zeros(1, 2),
            "encoder_hidden_states": torch.ones(1, 3, 2),
            "encoder_hidden_states_mask": torch.ones(1, 3),
            "img_shapes": [[(1, 2, 2)]],
            "scale": 1.0,
        },
    )
    b = (
        (),
        {
            "hidden_states": torch.zeros(1, 2),
            "encoder_hidden_states": torch.ones(1, 5, 2),
            "encoder_hidden_states_mask": None,
            "img_shapes": [[(1, 2, 2)]],
            "scale": 1.0,
        },
    )
    _args, kw = cb.merge_branches(a, b, 1)
    assert kw["encoder_hidden_states"].shape == (2, 5, 2)
    assert kw["encoder_hidden_states_mask"].tolist() == [[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]]
    assert kw["img_shapes"] == [[(1, 2, 2)], [(1, 2, 2)]]
    b[1]["scale"] = 2.0
    with pytest.raises(cb._NotBatchable):
        cb.merge_branches(a, b, 1)


def test_an_oom_preparing_the_doubled_call_reruns_plain_and_clears_next_render(monkeypatch):
    ref, _ = _flux_loop(_offloaded(Flux2Transformer2DModel()))
    model = Flux2Transformer2DModel()
    pipe, router = _armed(model, monkeypatch)
    real = cb.merge_branches

    def oom(*a, **k):
        raise torch.cuda.OutOfMemoryError("CUDA out of memory")

    monkeypatch.setattr(cb, "merge_branches", oom)
    got, _ = _flux_loop(model)
    torch.testing.assert_close(got, ref, rtol = 0, atol = 0)  # the render ran plain, it did not abort
    assert router.stats["batched_steps"] == 0 and "out of memory" in router.shape.off
    cb.end_cfg_batch_render(pipe)
    monkeypatch.setattr(cb, "merge_branches", real)
    got, _ = _flux_loop(model)
    torch.testing.assert_close(got, ref)
    assert router.stats["batched_steps"] > 0  # the transient verdict did not outlive its render


def test_a_step_cache_hooked_at_request_time_keeps_the_render_plain(monkeypatch):
    ref, _ = _flux_loop(_offloaded(Flux2Transformer2DModel()))
    model = Flux2Transformer2DModel()
    pipe, router = _armed(model, monkeypatch)
    cache = torch.nn.Linear(1, 1)
    model.add_module("fbcache_block", cache)
    cache._diffusers_hook = types.SimpleNamespace(hooks = {"fbc_leader_block_hook": object()})
    got, _ = _flux_loop(model)
    torch.testing.assert_close(got, ref, rtol = 0, atol = 0)
    assert router.stats["batched_steps"] == 0 and "fbc_leader_block_hook" in router.render_off
    del model._modules["fbcache_block"]
    cb.end_cfg_batch_render(pipe)
    _flux_loop(model)
    assert router.stats["batched_steps"] > 0


def test_a_new_request_shape_starts_without_the_previous_shapes_template(monkeypatch):
    model = Flux2Transformer2DModel()
    _pipe, router = _armed(model, monkeypatch)
    _flux_loop(model)
    first = router.last_shape
    assert router.template is not None
    router.prev_latent = None  # a new chunk: a different latent object
    big = torch.randn(2, 4, 8)
    router._new_step((), {"hidden_states": big}, big)
    assert router.last_shape is not first and router.template is None
