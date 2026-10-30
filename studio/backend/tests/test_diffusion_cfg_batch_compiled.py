# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""CFG batching on a compiled + resident Wan denoiser: one doubled call per guided step from step 2, CPU fakes."""

from __future__ import annotations

import pytest
import torch

from core.inference import diffusion_cfg_batch as cb


class WanTransformer3DModel(torch.nn.Module):
    """Stateless per-sample forward in the diffusers Wan call shape (hidden, expanded timestep, text)."""

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.linspace(0.5, 1.5, 8))
        self.calls = []

    def forward(
        self,
        hidden_states,
        timestep,
        encoder_hidden_states,
        encoder_hidden_states_image = None,
        return_dict = True,
        attention_kwargs = None,
    ):
        self.calls.append(int(hidden_states.shape[0]))
        text = encoder_hidden_states.mean(dim = 1, keepdim = True)
        return (hidden_states * self.w + text + timestep.view(-1, 1, 1).float() / 1000,)


class Flux2Transformer2DModel(WanTransformer3DModel):
    pass


for _cls in (WanTransformer3DModel, Flux2Transformer2DModel):
    _cls.__module__ = "diffusers.models.fake"


class _Pipe:
    def __init__(self, transformer):
        self.transformer = transformer


def _compiled(model):
    model._unsloth_regional_compile_kwargs = {"fullgraph": True, "dynamic": True}
    return model


def _wan_loop(
    model,
    steps = 6,
    seed = 0,
    neg_seed = 1,
):
    """WanPipeline's loop: one latent object per step, cond then uncond with ``t.expand(batch)``, CFG combine."""
    torch.manual_seed(seed)
    latents = torch.randn(1, 4, 8)
    pos = torch.randn(1, 5, 8)
    neg = torch.randn(1, 5, 8, generator = torch.Generator().manual_seed(neg_seed))
    for t in torch.tensor([999, 941, 883, 825, 767, 709])[:steps]:
        latent_model_input = latents
        timestep = t.expand(latents.shape[0])
        c = model(hidden_states = latent_model_input, timestep = timestep, encoder_hidden_states = pos, return_dict = False)[0]  # fmt: skip
        u = model(hidden_states = latent_model_input, timestep = timestep, encoder_hidden_states = neg, return_dict = False)[0]  # fmt: skip
        latents = latents - 0.1 * (u + 5.0 * (c - u))
    return latents


@pytest.fixture
def on_cuda(monkeypatch):
    # the path is CUDA-only; the fakes run on CPU
    monkeypatch.setattr(cb, "_cuda_latent", lambda latent: True)


def _install(model):
    pipe = _Pipe(model)
    assert cb.install_cfg_batch(pipe, offload_policy = "none") is None
    return pipe, cb.router_for(pipe)


def test_a_compiled_resident_wan_batches_from_step_two_on_every_render(on_cuda):
    ref = _wan_loop(WanTransformer3DModel())
    model = _compiled(WanTransformer3DModel())
    pipe, router = _install(model)
    assert "compiled resident" in router.reason
    for _ in range(3):
        del model.calls[:]
        torch.testing.assert_close(_wan_loop(model), ref, rtol = 0, atol = 0)
        assert (
            model.calls == [1, 1] + [2] * 5
        )  # step 1 plain (records the template), steps 2-6 doubled
        cb.end_cfg_batch_render(pipe)
    status = cb.cfg_batch_status(pipe)
    assert status["compiled_resident"] and status["render_off"] is None and router.disabled is None
    assert status["batched_steps"] == 15 and status["hits"] == 15 and status["misses"] == 0
    assert (
        status["plain_pair_s"] is None and status["batched_pair_s"] is None
    )  # no timing trial ran


def test_a_new_prompt_negative_and_seed_keep_the_one_path(on_cuda):
    model = _compiled(WanTransformer3DModel())
    pipe, router = _install(model)
    _wan_loop(model)
    cb.end_cfg_batch_render(pipe)
    ref = _wan_loop(WanTransformer3DModel(), seed = 7, neg_seed = 9)
    del model.calls[:]
    torch.testing.assert_close(_wan_loop(model, seed = 7, neg_seed = 9), ref, rtol = 0, atol = 0)
    assert model.calls == [1, 1] + [2] * 5 and router.stats["misses"] == 0


def test_the_compiled_resident_kill_switch_keeps_one_branch_per_call(monkeypatch, on_cuda):
    monkeypatch.setenv(cb.COMPILED_RESIDENT_ENV, "0")
    model = _compiled(WanTransformer3DModel())
    pipe = _Pipe(model)
    assert (
        cb.install_cfg_batch(pipe, offload_policy = "none") == "the denoiser is compiled and resident"
    )
    _wan_loop(model)
    assert model.calls == [1] * 12


def test_the_global_switch_still_turns_everything_off(monkeypatch, on_cuda):
    monkeypatch.setenv(cb.CFG_BATCH_ENV, "0")
    assert cb.install_cfg_batch(
        _Pipe(_compiled(WanTransformer3DModel())), offload_policy = "none"
    ) == (f"{cb.CFG_BATCH_ENV}=0")


def test_other_compiled_resident_classes_stay_declined(on_cuda):
    model = _compiled(Flux2Transformer2DModel())
    assert (
        cb.install_cfg_batch(_Pipe(model), offload_policy = "none")
        == "the denoiser is compiled and resident"
    )


def test_off_cuda_a_compiled_resident_wan_stays_plain():
    model = _compiled(WanTransformer3DModel())
    pipe, router = _install(model)
    _wan_loop(model)
    assert model.calls == [1] * 12
    assert "CUDA only" in router.last_shape.off


def test_a_doubled_call_compiling_on_a_follow_up_render_ends_batching_after_it(
    monkeypatch, on_cuda
):
    model = _compiled(WanTransformer3DModel())
    pipe, router = _install(model)
    _wan_loop(model)  # render 1 compiles the batch 2 graph: expected warm-up
    cb.end_cfg_batch_render(pipe)
    graphs = iter(range(1000))
    monkeypatch.setattr(
        cb, "_graph_count", lambda: next(graphs)
    )  # every doubled call "compiles" from now on
    del model.calls[:]
    _wan_loop(model)
    assert model.calls == [1, 1] + [2] * 5  # the render keeps its one path
    cb.end_cfg_batch_render(pipe)
    assert "after its warm-up render" in router.disabled
    del model.calls[:]
    _wan_loop(model)
    assert model.calls == [1] * 12


def test_the_first_render_doubled_compile_is_warm_up(monkeypatch, on_cuda):
    graphs = iter(range(1000))
    monkeypatch.setattr(cb, "_graph_count", lambda: next(graphs))
    model = _compiled(WanTransformer3DModel())
    pipe, router = _install(model)
    _wan_loop(model)
    cb.end_cfg_batch_render(pipe)
    assert router.disabled is None and router.pending_disable is None


def test_no_room_for_the_doubled_call_keeps_that_render_plain(monkeypatch, on_cuda):
    model = _compiled(WanTransformer3DModel())
    pipe, router = _install(model)
    monkeypatch.setattr(router, "_memory_ok", lambda latent, shape: False)
    _wan_loop(model)
    assert model.calls == [1] * 12 and "no room" in router.last_shape.off
    cb.end_cfg_batch_render(pipe)
    monkeypatch.setattr(router, "_memory_ok", lambda latent, shape: True)
    del model.calls[:]
    _wan_loop(model)
    assert model.calls == [1, 1] + [2] * 5  # memory pressure is per render


def test_an_oom_on_the_doubled_call_reruns_the_branch_plain(monkeypatch, on_cuda):
    ref = _wan_loop(WanTransformer3DModel())
    model = _compiled(WanTransformer3DModel())
    pipe, router = _install(model)
    real = router.orig_forward

    def oom_on_doubled(*args, **kwargs):
        if kwargs["hidden_states"].shape[0] == 2:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        return real(*args, **kwargs)

    router.orig_forward = oom_on_doubled
    torch.testing.assert_close(_wan_loop(model), ref, rtol = 0, atol = 0)
    assert router.stats["batched_steps"] == 0 and "out of memory" in router.last_shape.off
    cb.end_cfg_batch_render(pipe)
    router.orig_forward = real
    del model.calls[:]
    torch.testing.assert_close(_wan_loop(model), ref, rtol = 0, atol = 0)
    assert model.calls == [1, 1] + [2] * 5


def test_an_offloaded_compiled_wan_keeps_the_timed_path(monkeypatch):
    model = _compiled(WanTransformer3DModel())
    sub = torch.nn.Linear(1, 1)
    model.add_module("offloaded_group", sub)
    import types

    sub._diffusers_hook = types.SimpleNamespace(hooks = {"group_offloading": object()})
    pipe = _Pipe(model)
    assert cb.install_cfg_batch(pipe, offload_policy = "group") is None
    router = cb.router_for(pipe)
    assert router.offloaded and "compiled resident" not in router.reason
    _wan_loop(model)
    assert not router.compiled_resident
