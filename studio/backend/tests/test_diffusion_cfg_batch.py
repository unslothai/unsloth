# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""CFG batching (diffusion_cfg_batch): both guidance branches through one forward (offloaded or resident), CPU fakes."""

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
    device = "cpu",
):
    """The diffusers FLUX.2-klein loop: per step one latent object, cond then uncond call, CFG combine."""
    torch.manual_seed(0)
    latents = torch.randn(1, 4, 8).to(device)
    pos = torch.randn(1, 5, 8).to(device)
    neg = (torch.randn(1, 5, 8) if neg is None else neg).to(device)
    txt_ids = torch.zeros(1, 5, 4, device = device)
    img_ids = torch.ones(1, 4, 4, device = device)
    outs = []
    for i in range(steps):
        t = torch.full((1,), 1.0 - i / steps, device = device)
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
    assert cb.cfg_batch_status(pipe)["plain_pair_s"] is None
    assert cb.cfg_batch_status(pipe)["shape_off"] is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a CUDA device timer")
def test_cuda_latents_time_the_step_two_pair_and_keep_the_image(monkeypatch):
    ref, _ = _flux_loop(_offloaded(Flux2Transformer2DModel()).cuda(), device = "cuda")
    model = Flux2Transformer2DModel().cuda()
    pipe, router = _armed(model, monkeypatch)
    got, _ = _flux_loop(model, device = "cuda")
    torch.testing.assert_close(got, ref)
    status = cb.cfg_batch_status(pipe)
    assert status["plain_pair_s"] is not None and status["batched_pair_s"] is not None
    if status["shape_off"] is None:
        assert router.stats["batched_steps"] == 4 and model.calls[-2:] == [2, 2]
    else:
        assert "measured no gain" in status["shape_off"]
        assert router.stats["batched_steps"] == 0 and model.calls[-2:] == [1, 1]
    cb.end_cfg_batch_render(pipe)


def test_wan_expanded_timesteps_serve_the_second_branch(monkeypatch):
    """WanPipeline's stride-0 ``t.expand(batch)`` timestep must compare equal, not disable batching."""

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
        if i >= 3:
            neg.add_(0.5)

    ref, _ = _flux_loop(
        _offloaded(Flux2Transformer2DModel()), steps = 8, neg = torch.zeros(1, 5, 8), between = between
    )
    model = Flux2Transformer2DModel()
    _pipe, router = _armed(model, monkeypatch)
    got, _ = _flux_loop(model, steps = 8, neg = torch.zeros(1, 5, 8), between = between)
    torch.testing.assert_close(got, ref, rtol = 0, atol = 0)
    assert router.stats["misses"] == 3 and router.stats["hits"] == 1
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


def test_a_resident_denoiser_is_armed_by_default(monkeypatch):
    resident = _Pipe(Flux2Transformer2DModel())
    assert cb.cfg_batch_decline(resident, offload_policy = "none") is None
    assert cb.install_cfg_batch(resident, offload_policy = "none") is None
    assert cb.router_for(resident).reason == "resident denoiser"
    assert "model offload" in cb.cfg_batch_decline(
        _Pipe(Flux2Transformer2DModel()), offload_policy = "model"
    )
    monkeypatch.setenv(cb.CFG_BATCH_ENV, "0")
    assert cb.cfg_batch_decline(_Pipe(Flux2Transformer2DModel()), offload_policy = "none") == (
        f"{cb.CFG_BATCH_ENV}=0"
    )


def test_a_resident_denoiser_off_cuda_stays_plain_unless_forced(monkeypatch):
    """CPU / MPS loads have no device timer, so the trial cannot price the doubled call: the shape stays plain and the
    render matches two plain calls exactly. ``force`` still batches."""
    model = Flux2Transformer2DModel()
    pipe = _Pipe(model)
    assert cb.install_cfg_batch(pipe, offload_policy = "none") is None
    router = cb.router_for(pipe)
    ref, _ = _flux_loop(Flux2Transformer2DModel())
    for _ in range(2):
        got, _ = _flux_loop(model)
        cb.end_cfg_batch_render(pipe)
        torch.testing.assert_close(got, ref, rtol = 0, atol = 0)
    assert set(model.calls) == {1}
    assert router.stats["batched_steps"] == 0
    assert "no device timer on cpu" in cb.cfg_batch_status(pipe)["shape_off"]

    monkeypatch.setenv(cb.CFG_BATCH_ENV, "force")
    forced = Flux2Transformer2DModel()
    fpipe = _Pipe(forced)
    assert cb.install_cfg_batch(fpipe, offload_policy = "none") is None
    _flux_loop(forced)
    assert 2 in forced.calls and cb.router_for(fpipe).stats["batched_steps"] > 0


class _Clock:
    """A fake device timer: each forward costs ``cost(batch)`` seconds of the router's perf_counter."""

    def __init__(self, cost):
        self.now = 0.0
        self.cost = cost

    def __call__(self):
        return self.now


def _timed(monkeypatch, cost):
    clock = _Clock(cost)
    monkeypatch.setattr(cb.time, "perf_counter", clock)
    monkeypatch.setattr(cb, "_RESIDENT_MAX_PAIR_S", 1e9)
    monkeypatch.setattr(cb.CfgBatchRouter, "_sync", lambda self, latent: True)

    def forward(self, *a, **k):
        clock.now += clock.cost(int(k["hidden_states"].shape[0]))
        return Flux2Transformer2DModel.forward(self, *a, **k)

    timed = type("Flux2Transformer2DModel", (Flux2Transformer2DModel,), {"forward": forward})
    timed.__module__ = "diffusers.models.fake"
    return timed()


@pytest.mark.parametrize("batched_cost, wins", [(1.2, True), (2.0, False)])
def test_the_timing_trial_never_mixes_paths_and_repeats_bit_identically(
    monkeypatch, batched_cost, wins
):
    model = _timed(monkeypatch, lambda b: 1.0 if b == 1 else batched_cost)
    pipe = _Pipe(model)
    assert cb.install_cfg_batch(pipe, offload_policy = "none") is None
    router = cb.router_for(pipe)
    first, _ = _flux_loop(model)
    cb.end_cfg_batch_render(pipe)
    calls_after_first = list(model.calls)
    second, _ = _flux_loop(model)
    cb.end_cfg_batch_render(pipe)
    torch.testing.assert_close(first, second, rtol = 0, atol = 0)
    assert calls_after_first[:10] == [1, 1, 1, 1, 1, 1, 2, 1, 1, 2]
    render2 = model.calls[len(calls_after_first) :]
    if wins:
        assert render2 == [1, 1, 1, 1] + [2] * 4 and router.last_shape.off is None
    else:
        assert render2 == [1] * 12 and "measured no gain" in router.last_shape.off
    plain = Flux2Transformer2DModel()
    ref_plain, _ = _flux_loop(plain)
    if not wins:
        torch.testing.assert_close(second, ref_plain, rtol = 0, atol = 0)
    else:
        torch.testing.assert_close(second, ref_plain)


def test_a_shape_that_stays_plain_records_no_prediction(monkeypatch):
    model = _timed(monkeypatch, lambda b: 1.0 if b == 1 else 2.0)
    pipe = _Pipe(model)
    assert cb.install_cfg_batch(pipe, offload_policy = "none") is None
    router = cb.router_for(pipe)
    _flux_loop(model)
    cb.end_cfg_batch_render(pipe)
    assert "measured no gain" in router.last_shape.off
    recorded = []
    monkeypatch.setattr(router, "_record_template", lambda *a, **k: recorded.append(1))
    _flux_loop(model)
    cb.end_cfg_batch_render(pipe)
    assert recorded == [] and model.calls[-12:] == [1] * 12


def _quiet_flux():
    """The fake FLUX.2 transformer without its call log (a list append Dynamo would have to trace)."""

    def forward(
        self,
        hidden_states,
        timestep,
        encoder_hidden_states,
        txt_ids,
        img_ids,
        return_dict = False,
        **_,
    ):
        text = encoder_hidden_states.mean(dim = 1, keepdim = True)
        return (
            hidden_states * self.w
            + text
            + timestep.view(-1, 1, 1)
            + txt_ids.mean()
            + img_ids.mean(),
        )

    cls = type("Flux2Transformer2DModel", (Flux2Transformer2DModel,), {"forward": forward})
    cls.__module__ = "diffusers.models.fake"
    return cls()


def test_compiled_denoisers_are_seen_in_every_form():
    plain = Flux2Transformer2DModel()
    assert cb.compiled_denoiser(plain) is None
    marked = Flux2Transformer2DModel()
    marked._unsloth_regional_compile_kwargs = {"dynamic": False}
    assert cb.compiled_denoiser(marked) == "the denoiser is compiled"
    block = Flux2Transformer2DModel()
    block.add_module("blocks", torch.nn.ModuleList([torch.nn.Linear(1, 1)]))
    block.blocks[0].compile()
    assert "blocks.0 is compiled" == cb.compiled_denoiser(block)
    # #14528: the compile moved below a group-offload hook, the block's _compiled_call_impl cleared
    hooked = Flux2Transformer2DModel()
    hooked.add_module("blocks", torch.nn.ModuleList([torch.nn.Linear(1, 1)]))
    ref = types.SimpleNamespace(forward = torch.compile(hooked.blocks[0].forward))
    hooked.blocks[0]._diffusers_hook = types.SimpleNamespace(
        hooks = {"group_offloading": object()}, _fn_refs = [ref]
    )
    assert "below its hooks" in cb.compiled_denoiser(hooked)
    assert (
        cb.cfg_batch_decline(_Pipe(marked), offload_policy = "none")
        == "the denoiser is compiled and resident"
    )
    assert cb.cfg_batch_decline(_Pipe(_offloaded(marked)), offload_policy = "group") is None


def test_a_compile_on_a_resident_denoiser_after_install_keeps_every_render_plain(monkeypatch):
    model = Flux2Transformer2DModel()
    pipe = _Pipe(model)
    assert cb.install_cfg_batch(pipe, offload_policy = "none") is None
    model._unsloth_regional_compile_kwargs = {
        "dynamic": False
    }  # the deferred profile compiled the blocks
    _flux_loop(model)
    cb.end_cfg_batch_render(pipe)
    _flux_loop(model)
    assert model.calls == [1] * 24
    assert cb.cfg_batch_status(pipe)["render_off"] == "the denoiser is compiled and resident"


def test_an_offloaded_shape_judged_before_a_compile_stays_plain_after_it(monkeypatch):
    model = Flux2Transformer2DModel()
    pipe, router = _armed(model, monkeypatch)
    _flux_loop(model)  # eager render: trial on step 1, steps 2-6 batched
    cb.end_cfg_batch_render(pipe)
    assert router.stats["batched_steps"] == 4
    model._unsloth_regional_compile_kwargs = {
        "dynamic": False
    }  # compiled below the offload hooks afterwards
    del model.calls[:]
    _flux_loop(model)
    assert model.calls == [1] * 12  # no doubled call reaches the new compile on a follow-up prompt
    assert "compiled after this shape's trial" in router.last_shape.off


def test_an_offloaded_compiled_denoiser_batches_from_its_trial_render(monkeypatch):
    model = Flux2Transformer2DModel()
    model._unsloth_regional_compile_kwargs = {"dynamic": False}
    pipe, router = _armed(model, monkeypatch)
    for _ in range(2):
        _flux_loop(model)
        cb.end_cfg_batch_render(pipe)
    assert router.stats["batched_steps"] == 8 and router.disabled is None
    assert router.last_shape.compiled_at_trial and router.last_shape.trial_render == 0


def test_a_doubled_call_compiling_on_a_follow_up_render_ends_batching_after_that_render(
    monkeypatch,
):
    model = Flux2Transformer2DModel()
    pipe, router = _armed(model, monkeypatch)
    _flux_loop(model)
    cb.end_cfg_batch_render(pipe)
    graphs = iter(range(1000))
    monkeypatch.setattr(
        cb, "_graph_count", lambda: next(graphs)
    )  # every doubled call "compiles" from now on
    del model.calls[:]
    _flux_loop(model)
    assert model.calls == [1, 1, 1, 1] + [2] * 4  # render 2 stays on its one path
    cb.end_cfg_batch_render(pipe)
    assert "after its warm-up render" in router.disabled
    del model.calls[:]
    _flux_loop(model)
    assert model.calls == [1] * 12


def test_a_doubled_call_that_compiles_a_graph_turns_batching_off(monkeypatch):
    model = _timed(monkeypatch, lambda b: 1.0 if b == 1 else 1.2)
    pipe = _Pipe(model)
    assert cb.install_cfg_batch(pipe, offload_policy = "none") is None
    graphs = iter(range(100))
    monkeypatch.setattr(cb, "_graph_count", lambda: next(graphs))
    _flux_loop(model)
    router = cb.router_for(pipe)
    assert "compiled a new graph" in router.disabled
    assert (
        model.calls[:4] == [1, 1, 1, 1] and model.calls[-8:] == [1] * 8
    )  # nothing batched after the trial


def test_a_router_traced_by_dynamo_passes_straight_through():
    model = _quiet_flux()
    pipe = _Pipe(model)
    assert cb.install_cfg_batch(pipe, offload_policy = "none") is None
    torch._dynamo.reset()
    compiled = torch.compile(model, backend = "eager", fullgraph = True)
    kw = dict(
        timestep = torch.ones(1),
        encoder_hidden_states = torch.randn(1, 5, 8),
        txt_ids = torch.zeros(1, 5, 4),
        img_ids = torch.ones(1, 4, 4),
    )
    for batch in (1, 2):
        x = torch.randn(batch, 4, 8)
        out = compiled(hidden_states = x, **{**kw, "timestep": torch.ones(batch)})[0]
        torch.testing.assert_close(
            out,
            model.forward.orig_forward(hidden_states = x, **{**kw, "timestep": torch.ones(batch)})[0],
        )
    assert (
        cb.router_for(pipe).stats["plain_calls"] == 0
    )  # the traced calls never ran the router's own logic


@pytest.mark.parametrize("pair_s, gated", [(0.3, True), (0.1, False)])
def test_a_compute_bound_resident_pair_skips_the_trial(monkeypatch, pair_s, gated):
    model = _timed(monkeypatch, lambda b: pair_s / 2 if b == 1 else pair_s)
    monkeypatch.setattr(cb, "_RESIDENT_MAX_PAIR_S", 0.25)
    pipe = _Pipe(model)
    assert cb.install_cfg_batch(pipe, offload_policy = "none") is None
    _flux_loop(model)
    off = cb.router_for(pipe).last_shape.off
    if gated:
        assert "compute-bound" in off and model.calls == [1] * 12
    else:
        assert "measured no gain" in off and 2 in model.calls  # the trial ran its doubled calls


def test_an_offloaded_pair_is_never_gated_as_compute_bound(monkeypatch):
    model = _timed(monkeypatch, lambda b: 5.0 if b == 1 else 6.0)
    monkeypatch.setattr(cb, "_RESIDENT_MAX_PAIR_S", 0.25)
    pipe = _Pipe(_offloaded(model))
    assert cb.install_cfg_batch(pipe, offload_policy = "group") is None
    _flux_loop(model)
    assert (
        cb.router_for(pipe).last_shape.off is None and model.calls[-1] == 2
    )  # an upload saved per step wins


def test_one_slow_plain_sample_does_not_decide_the_trial(monkeypatch):
    seen = {"n": 0}

    def cost(b):
        seen["n"] += 1
        if seen["n"] == 5:  # the trial's first plain call (after steps 1 and 2)
            return 50.0
        return 1.0 if b == 1 else 2.0

    model = _timed(monkeypatch, cost)
    pipe = _Pipe(model)
    assert cb.install_cfg_batch(pipe, offload_policy = "none") is None
    _flux_loop(model)
    cb.end_cfg_batch_render(pipe)
    shape = cb.router_for(pipe).last_shape
    assert shape.off is not None and "measured no gain" in shape.off
    assert shape.plain_pair_s == 2.0 and min(shape.batched_s) == 2.0


def test_declines(monkeypatch):
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


def test_the_trial_prices_the_batched_steps_preparation_and_verification(monkeypatch):
    model = _timed(monkeypatch, lambda b: 1.0 if b == 1 else 1.2)
    clock = cb.time.perf_counter
    real_matches = cb._matches

    def slow_matches(snapshots, call):
        clock.now += 0.7
        return real_matches(snapshots, call)

    monkeypatch.setattr(cb, "_matches", slow_matches)
    pipe = _Pipe(model)
    assert cb.install_cfg_batch(pipe, offload_policy = "none") is None
    _flux_loop(model)
    shape = cb.router_for(pipe).last_shape
    assert shape.off is not None and "measured no gain" in shape.off
    assert min(shape.batched_s) == pytest.approx(1.9)


@pytest.mark.parametrize("offload_policy", ["none", "group"])
def test_an_explicit_speed_off_request_keeps_one_branch_per_call(monkeypatch, offload_policy):
    def pipe():
        model = Flux2Transformer2DModel()
        return _Pipe(_offloaded(model) if offload_policy == "group" else model)

    off = pipe()
    reason = cb.install_cfg_batch(off, offload_policy = offload_policy, speed_mode = "off")
    assert reason is not None and "speed_mode=off" in reason and cb.router_for(off) is None
    assert "forward" not in off.transformer.__dict__
    assert "speed_mode=off" in cb.cfg_batch_decline(
        pipe(), offload_policy = offload_policy, speed_mode = " OFF "
    )
    for speed in (None, "default", "max"):  # unset keeps the default-on lever
        assert cb.cfg_batch_decline(pipe(), offload_policy = offload_policy, speed_mode = speed) is None
    monkeypatch.setenv(cb.CFG_BATCH_ENV, "force")
    assert cb.cfg_batch_decline(pipe(), offload_policy = offload_policy, speed_mode = "off") is None


def test_both_backends_pass_the_requested_speed_mode():
    import inspect

    from core.inference import diffusion, video
    for module in (diffusion, video):
        src = inspect.getsource(module)
        calls = [c for c in src.split("install_cfg_batch(")[1:] if not c.startswith("pipe: Any")]
        assert calls and all(
            "speed_mode = speed_mode" in c.split(")")[0] for c in calls
        ), module.__name__


_TRIAL = [
    1,
    1,
    2,
    1,
    1,
    2,
]  # the discarded trial after steps 1-2 (two plain pairs, two doubled calls)


def _switchable(
    monkeypatch,
    tmp_path,
    card = "FakeGPU",
):
    """A timed fake whose doubled-call cost the test moves (timing noise around the gate), on a named card, with
    the verdict store in ``tmp_path``. ``raising = False``: main has no store, so the behaviour is what fails there."""
    cost = {"batched": 1.2}
    model = _timed(monkeypatch, lambda b: 1.0 if b == 1 else cost["batched"])
    monkeypatch.setenv(cb.CFG_BATCH_ENV, "auto")
    monkeypatch.setenv("UNSLOTH_DIFFUSION_CFG_BATCH_VERDICTS_FILE", str(tmp_path / "verdicts.json"))
    monkeypatch.setattr(cb, "_card_name", lambda latent: card, raising = False)
    return model, cost


def _render(pipe, model):
    start = len(model.calls)
    _flux_loop(model)
    cb.end_cfg_batch_render(pipe)
    return model.calls[start:]


def _fresh_load(model_cls):
    """A new load of the same model (another process: only the verdict store survives)."""
    model = model_cls()
    pipe = _Pipe(model)
    assert cb.install_cfg_batch(pipe, offload_policy = "none") is None
    return pipe, model


@pytest.mark.parametrize("first_wins", [True, False])
def test_a_borderline_shape_takes_one_path_in_every_process(monkeypatch, tmp_path, first_wins):
    model, cost = _switchable(monkeypatch, tmp_path)
    cost["batched"] = (
        1.8 if first_wins else 1.9
    )  # 0.90 vs 0.95 of the plain pair: either side of the 0.92 gate
    pipe = _Pipe(model)
    assert cb.install_cfg_batch(pipe, offload_policy = "none") is None
    first = _render(pipe, model)
    assert first[4:10] == _TRIAL
    first_path = first[:4] + first[10:]
    # the next process times the same shape on the other side of the gate (noise): it must not re-decide
    cost["batched"] = 1.9 if first_wins else 1.8
    for _ in range(2):
        pipe2, model2 = _fresh_load(type(model))
        later = _render(pipe2, model2)
        assert later == first_path  # no trial, the stored path
        assert cb.router_for(pipe2).last_shape.verdict == ("batch" if first_wins else "plain")
    assert first_path == ([1] * 4 + [2] * 4 if first_wins else [1] * 12)


def test_memory_pressure_never_re_times_a_judged_shape(monkeypatch, tmp_path):
    model, cost = _switchable(monkeypatch, tmp_path)
    monkeypatch.setenv("UNSLOTH_DIFFUSION_CFG_BATCH_VERDICTS", "0")  # the in-load latch alone
    pipe = _Pipe(model)
    assert cb.install_cfg_batch(pipe, offload_policy = "none") is None
    router = cb.router_for(pipe)
    assert _render(pipe, model)[10:] == [2] * 4
    real = router.orig_forward
    state = {"oom": True}

    def oom_once(*a, **k):
        if state["oom"] and int(k["hidden_states"].shape[0]) == 2:
            state["oom"] = False
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        return real(*a, **k)

    router.orig_forward = oom_once
    assert _render(pipe, model) == [1] * 12  # the OOM render fell back to plain
    router.orig_forward = real
    cost["batched"] = 1.9  # a re-timing now would turn the shape plain for good
    assert _render(pipe, model) == [1] * 4 + [2] * 4
    assert router.last_shape.off is None and router.last_shape.verdict == "batch"


def test_no_room_mid_render_keeps_the_rest_of_that_render_plain(monkeypatch, tmp_path):
    model, _ = _switchable(monkeypatch, tmp_path)
    free = iter([True, True, True, False, True, True] + [True] * 20)
    monkeypatch.setattr(cb.CfgBatchRouter, "_memory_ok", lambda self, latent, shape: next(free))
    pipe = _Pipe(model)
    assert cb.install_cfg_batch(pipe, offload_policy = "none") is None
    calls = _render(pipe, model)
    # trial gate, verdict gate, step 3 batched, step 4 no room: steps 4-6 plain (not batched again at step 5)
    assert calls[10:] == [2] + [1] * 6
    assert _render(pipe, model) == [1] * 4 + [2] * 4  # the next render batches again, untimed


def test_the_verdict_store_is_keyed_on_the_card_and_skipped_off_cuda(monkeypatch, tmp_path):
    model, cost = _switchable(monkeypatch, tmp_path)
    pipe = _Pipe(model)
    assert cb.install_cfg_batch(pipe, offload_policy = "none") is None
    _render(pipe, model)
    assert "verdict" in (tmp_path / "verdicts.json").read_text()
    cost["batched"] = 1.9
    monkeypatch.setattr(cb, "_card_name", lambda latent: "OtherGPU")
    pipe2, model2 = _fresh_load(type(model))
    assert _render(pipe2, model2)[4:10] == _TRIAL  # another card is priced on its own
    assert "measured no gain" in cb.router_for(pipe2).last_shape.off
    monkeypatch.setattr(cb, "_card_name", lambda latent: None)  # off CUDA: nothing timed is stored
    before = (tmp_path / "verdicts.json").read_text()
    pipe3, model3 = _fresh_load(type(model))
    assert _render(pipe3, model3)[4:10] == _TRIAL
    assert (tmp_path / "verdicts.json").read_text() == before


def test_usable_free_does_not_count_graph_held_cache(monkeypatch):
    import core.inference.diffusion_memory as dm

    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device = None: (1000 * 2**20, 0))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device = None: 600 * 2**20)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device = None: 0)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(dm, "_user_budget_reserve_mib", lambda total_mib: 0)
    monkeypatch.setattr(dm, "_graph_held_cache_bytes", lambda device = None: 500 * 2**20)
    monkeypatch.setattr(cb, "_torch", lambda: torch)
    assert cb._usable_free_mib(torch.device("cuda", 0)) == 1100


class _QuantDen(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = {"num_layers": 2, "hidden": 64}
        torch.manual_seed(0)
        self.a = torch.nn.Linear(64, 64, dtype = torch.bfloat16)
        self.b = torch.nn.Linear(64, 64, dtype = torch.bfloat16)


def _torchao_loads():
    import copy

    tq = pytest.importorskip("torchao.quantization")
    base = _QuantDen()
    loads = {"bf16": base}
    int8 = copy.deepcopy(base)
    tq.quantize_(int8, tq.Int8DynamicActivationInt8WeightConfig())
    loads["int8"] = int8
    try:
        from torchao.float8.inference import Float8MMConfig
        for fast in (True, False):
            fp8 = copy.deepcopy(base)
            tq.quantize_(
                fp8,
                tq.Float8DynamicActivationFloat8WeightConfig(
                    granularity = tq.PerRow(), mm_config = Float8MMConfig(use_fast_accum = fast)
                ),
            )
            loads[f"fp8_fast{int(fast)}"] = fp8
    except Exception:  # noqa: BLE001
        pass
    assert {str(m.a.weight.dtype) for m in loads.values()} == {"torch.bfloat16"}
    return loads


def test_the_verdict_key_tells_bf16_from_torchao_int8_and_fp8_loads():
    loads = _torchao_loads()
    keys = {name: cb._load_identity(m, "none", None) for name, m in loads.items()}
    assert None not in keys.values()
    assert len(set(keys.values())) == len(keys), keys


def test_the_verdict_key_includes_the_engaged_transformer_quant():
    model = _QuantDen()
    keys = {q: cb._load_identity(model, "none", None, q) for q in (None, "int8", "fp8", "nvfp4")}
    assert len(set(keys.values())) == len(keys)


def test_the_same_load_gives_a_stable_verdict_key():
    loads = _torchao_loads()
    import copy

    for name, model in loads.items():
        again = copy.deepcopy(model)
        assert cb._load_identity(model, "none", "default", "int8") == cb._load_identity(
            again, "none", "default", "int8"
        ), name
    assert cb._load_identity(_QuantDen(), "none", None) == cb._load_identity(
        _QuantDen(), "none", None
    )


def test_install_keys_the_router_on_the_engaged_quant(monkeypatch):
    monkeypatch.setenv(cb.CFG_BATCH_ENV, "auto")
    ids = set()
    for quant in (None, "int8"):
        pipe = _Pipe(_quiet_flux())
        assert cb.install_cfg_batch(pipe, offload_policy = "none", transformer_quant = quant) is None
        ids.add(cb.router_for(pipe).identity)
    assert len(ids) == 2 and None not in ids


def test_the_verdict_key_follows_the_live_attention_backend(monkeypatch):
    monkeypatch.setenv(cb.CFG_BATCH_ENV, "auto")
    monkeypatch.setattr(cb, "_card_name", lambda latent: "card")
    pipe = _Pipe(_quiet_flux())
    assert cb.install_cfg_batch(pipe, offload_policy = "none") is None
    router = cb.router_for(pipe)
    shape = cb._Shape("s")
    keys = set()
    for backend in (None, "native", "flash"):
        pipe.transformer._unsloth_attention_backend = backend
        keys.add(router._verdict_key(shape, None))
    assert len(keys) == 3 and None not in keys


def test_a_version_1_verdict_store_is_not_read(monkeypatch, tmp_path):
    import json

    from core.inference import diffusion_measured_peaks as peaks

    path = tmp_path / "verdicts.json"
    monkeypatch.setenv("UNSLOTH_DIFFUSION_CFG_BATCH_VERDICTS_FILE", str(path))
    entry = {"k": {"verdict": "plain", "reason": "old key"}}
    path.write_text(json.dumps({"version": 1, "runtime": peaks.runtime_stamp(), "entries": entry}))
    assert cb.stored_verdict("k") is None
    path.write_text(
        json.dumps(
            {"version": cb._VERDICTS_VERSION, "runtime": peaks.runtime_stamp(), "entries": entry}
        )
    )
    assert cb.stored_verdict("k") == entry["k"]
