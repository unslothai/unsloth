# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""CFG-parallel for U-Nets (SDXL / SD): one batch-2B call cut into two batch-B halves, the second on the replica."""

import pytest

torch = pytest.importorskip("torch")

from core.inference import diffusion_multigpu as mg  # noqa: E402


class UNet2DConditionModel(torch.nn.Module):
    """Named like diffusers' class: the router keys on the class name."""

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 4, 3, padding = 1)
        self.text = torch.nn.Linear(8, 4)
        self.add = torch.nn.Linear(6, 4)
        self.calls = []

    def forward(
        self,
        sample,
        timestep,
        encoder_hidden_states,
        added_cond_kwargs = None,
        cross_attention_kwargs = None,
        return_dict = True,
        **extra,
    ):
        self.calls.append(int(sample.shape[0]))
        h = self.conv(sample) * timestep
        h = h + self.text(encoder_hidden_states).mean(1)[:, :, None, None]
        if added_cond_kwargs:
            h = h + self.add(added_cond_kwargs["time_ids"])[:, :, None, None]
        for key, value in extra.items():
            if isinstance(value, torch.Tensor):
                h = h + value.sum()
        return (h,) if not return_dict else type("Out", (), {"sample": h})()


def _router(unet):
    replicas = []

    def factory():
        r = mg.build_replica(unet, torch.device("cpu"))
        replicas.append(r)
        return r

    router = mg.UnetCfgParallelRouter(unet, primary = "cpu", secondary = "cpu", replica_factory = factory)
    unet.forward = router
    return router, replicas


def _sdxl_loop(
    unet,
    latents,
    steps,
    batch = 1,
    cfg = True,
    **extra,
):
    g = torch.Generator().manual_seed(3)
    cond = torch.randn(batch, 5, 8, generator = g)
    uncond = torch.randn(batch, 5, 8, generator = g)
    ids = torch.randn(batch, 6, generator = g)
    for i in range(steps):
        t = torch.tensor(1.0 - i / steps)
        if cfg:
            x = torch.cat([latents] * 2)
            emb = torch.cat([uncond, cond])
            tid = torch.cat([ids, ids])
        else:
            x, emb, tid = latents, cond, ids
        out = unet(
            x,
            t,
            encoder_hidden_states = emb,
            added_cond_kwargs = {"time_ids": tid},
            return_dict = False,
            **extra,
        )[0]
        if cfg:
            u, c = out.chunk(2)
            out = u + 5.0 * (c - u)
        latents = latents - 0.1 * out
    return latents


def test_unet_cfg_parallel_matches_one_card_and_runs_halves_on_the_replica():
    torch.manual_seed(0)
    unet = UNet2DConditionModel()
    lat = torch.randn(1, 4, 8, 8)
    ref = _sdxl_loop(unet, lat.clone(), steps = 4)
    assert unet.calls == [2, 2, 2, 2]
    unet.calls.clear()
    router, replicas = _router(unet)
    out = _sdxl_loop(unet, lat.clone(), steps = 4)
    torch.testing.assert_close(out, ref, rtol = 1e-5, atol = 1e-6)
    assert len(replicas) == 1 and replicas[0] is not unet
    # the source runs one half per step; the replica the other
    assert unet.calls == [1, 1, 1, 1]
    assert replicas[0].calls == [1, 1, 1, 1]
    assert router.stats["pairs_parallel"] == 4 and router.stats["replica_calls"] == 4
    assert router.describe()["disabled_reason"] is None


def test_unet_batch_two_per_branch_is_cut_in_half():
    unet = UNet2DConditionModel()
    lat = torch.randn(2, 4, 8, 8)
    ref = _sdxl_loop(unet, lat.clone(), steps = 2, batch = 2)
    unet.calls.clear()
    router, replicas = _router(unet)
    out = _sdxl_loop(unet, lat.clone(), steps = 2, batch = 2)
    torch.testing.assert_close(out, ref, rtol = 1e-5, atol = 1e-6)
    assert unet.calls == [2, 2] and replicas[0].calls == [2, 2]


def test_unet_odd_batch_stays_whole_on_the_source():
    unet = UNet2DConditionModel()
    router, replicas = _router(unet)
    _sdxl_loop(unet, torch.randn(1, 4, 8, 8), steps = 3, cfg = False)
    assert unet.calls == [1, 1, 1] and replicas == [] and router.replica is None
    assert router.stats["replica_calls"] == 0


def test_unet_unknown_tensor_keyword_stays_whole_on_the_source():
    unet = UNet2DConditionModel()
    router, replicas = _router(unet)
    _sdxl_loop(unet, torch.randn(1, 4, 8, 8), steps = 2, mystery = torch.ones(2, 3))
    assert unet.calls == [2, 2] and replicas == []


def test_unet_replica_failure_recomputes_on_the_source():
    unet = UNet2DConditionModel()
    lat = torch.randn(1, 4, 8, 8)
    ref = _sdxl_loop(unet, lat.clone(), steps = 3)

    def broken():
        raise RuntimeError("no room")

    router = mg.UnetCfgParallelRouter(unet, primary = "cpu", secondary = "cpu", replica_factory = broken)
    unet.forward = router
    out = _sdxl_loop(unet, lat.clone(), steps = 3)
    torch.testing.assert_close(out, ref, rtol = 1e-5, atol = 1e-6)
    assert "replica unavailable" in (router.disabled_reason or "")


def test_unet_is_a_cfg_parallel_candidate(monkeypatch):
    class Pipe:
        transformer = None
        unet = UNet2DConditionModel()

    sel = mg.MultiGpuSelection(primary = 0, secondaries = (1,))
    # resident on the CPU here: past the U-Net gate, refused only for not being on a CUDA card
    assert (
        mg.cfg_parallel_decline(Pipe(), offload_policy = "none", selection = sel)
        == "denoiser is not on a CUDA card"
    )
    assert mg._cfg_denoiser(Pipe()) is Pipe.unet
    monkeypatch.setenv(mg.UNET_CFG_PARALLEL_ENV, "0")
    assert mg._cfg_denoiser(Pipe()) is None
    assert mg.UNET_CFG_PARALLEL_ENV in mg.cfg_parallel_decline(
        Pipe(), offload_policy = "none", selection = sel
    )


def test_real_diffusers_unet_split_matches_batched():
    diffusers = pytest.importorskip("diffusers")
    torch.manual_seed(0)
    unet = diffusers.UNet2DConditionModel(
        sample_size = 8,
        in_channels = 4,
        out_channels = 4,
        layers_per_block = 1,
        block_out_channels = (8, 16),
        down_block_types = ("DownBlock2D", "CrossAttnDownBlock2D"),
        up_block_types = ("CrossAttnUpBlock2D", "UpBlock2D"),
        cross_attention_dim = 8,
        attention_head_dim = 2,
        norm_num_groups = 4,
        addition_embed_type = "text_time",
        addition_time_embed_dim = 4,
        projection_class_embeddings_input_dim = 8 + 6 * 4,
    ).eval()
    x = torch.randn(2, 4, 8, 8)
    emb = torch.randn(2, 5, 8)
    added = {"text_embeds": torch.randn(2, 8), "time_ids": torch.randn(2, 6)}
    t = torch.tensor(500.0)
    with torch.no_grad():
        ref = unet(x, t, encoder_hidden_states = emb, added_cond_kwargs = added, return_dict = False)[0]
        router = mg.UnetCfgParallelRouter(
            unet,
            primary = "cpu",
            secondary = "cpu",
            replica_factory = lambda: mg.build_replica(unet, torch.device("cpu")),
        )
        unet.forward = router
        out = unet(x, t, encoder_hidden_states = emb, added_cond_kwargs = added, return_dict = False)[0]
        out2 = unet(x, t, encoder_hidden_states = emb, added_cond_kwargs = added).sample
    torch.testing.assert_close(out, ref, rtol = 1e-4, atol = 1e-5)
    torch.testing.assert_close(out2, ref, rtol = 1e-4, atol = 1e-5)
    assert router.stats["pairs_parallel"] == 2


def test_unet_with_batch_chunking_processors_is_not_a_candidate():
    class PAGCFGIdentitySelfAttnProcessor2_0:
        pass

    class Pipe:
        transformer = None
        unet = UNet2DConditionModel()

    Pipe.unet.attn_processors = {"down.attn1.processor": PAGCFGIdentitySelfAttnProcessor2_0()}
    assert mg._cfg_denoiser(Pipe()) is None


def test_unet_compiled_whole_after_install_runs_whole_on_the_source():
    unet = UNet2DConditionModel()
    router, replicas = _router(unet)
    # a deferred whole-module compile landed after the router: nn.Module.__call__ now goes through it
    unet._compiled_call_impl = lambda *a, **k: unet.forward(*a, **k)
    _sdxl_loop(unet, torch.randn(1, 4, 8, 8), steps = 2)
    assert unet.calls == [2, 2] and replicas == []


class _Dit(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(4, 4)

    def forward(
        self,
        hidden_states,
        timestep = None,
        encoder_hidden_states = None,
        return_dict = False,
    ):
        return (self.proj(hidden_states),)


def _automatic_install(
    monkeypatch,
    used_mib,
    names = ("Tesla T4", "Tesla T4"),
    env = None,
):
    class Pipe:
        transformer = _Dit()

    probes = {
        o: mg.DeviceProbe(ordinal = o, name = names[o], total_mib = 15360, free_mib = 15360 - used_mib[o])
        for o in (0, 1)
    }
    if env is not None:
        monkeypatch.setenv(mg.AUTO_CFG_PARALLEL_ENV, env)
    monkeypatch.setattr(mg, "_module_device", lambda m: torch.device("cuda", 0))
    monkeypatch.setattr(
        mg, "probe_devices", lambda ordinals, **kw: {o: probes[o] for o in ordinals}
    )
    monkeypatch.setattr(mg, "_hooks_present", lambda m: None)
    monkeypatch.setattr(mg, "is_layer_split", lambda p: False, raising = False)
    pipe = Pipe()
    try:
        with mg.selection_scope(mg.MultiGpuSelection(primary = 0, secondaries = (1,), automatic = True)):
            return mg.install_cfg_parallel(pipe, offload_policy = "none")
    finally:
        mg.uninstall_cfg_parallel(pipe)


def test_automatic_arms_cfg_parallel_on_an_idle_card_of_the_same_model(monkeypatch):
    assert _automatic_install(monkeypatch, used_mib = (9000, 3)) is None


def test_automatic_keeps_a_busy_second_card_free(monkeypatch):
    reason = _automatic_install(monkeypatch, used_mib = (9000, 4000))
    assert reason is not None and "no idle card" in reason


def test_automatic_never_borrows_a_different_card_model(monkeypatch):
    reason = _automatic_install(monkeypatch, used_mib = (9000, 3), names = ("Tesla T4", "Tesla P4"))
    assert reason is not None and "different model" in reason


def test_automatic_cfg_parallel_kill_switch(monkeypatch):
    reason = _automatic_install(monkeypatch, used_mib = (9000, 3), env = "0")
    assert reason is not None and "automatic GPU placement" in reason


def test_automatic_selection_records_compile_arguments_for_a_replica(monkeypatch):
    calls = []
    monkeypatch.setattr(mg, "install_compile_recorder", lambda: calls.append(1) or True)
    with mg.selection_scope(mg.MultiGpuSelection(primary = 0, secondaries = (1,), automatic = True)):
        pass
    assert calls == [1]
    monkeypatch.setenv(mg.AUTO_CFG_PARALLEL_ENV, "0")
    with mg.selection_scope(mg.MultiGpuSelection(primary = 0, secondaries = (1,), automatic = True)):
        pass
    assert calls == [1]


def test_automatic_replica_is_not_built_on_a_card_that_got_busy(monkeypatch):
    unet = UNet2DConditionModel()
    router, replicas = _router(unet)
    router.automatic_idle_only = True
    monkeypatch.setattr(router, "_secondary_free_total_mib", lambda: (9000, 15360))
    lat = torch.randn(1, 4, 8, 8)
    _sdxl_loop(unet, lat, steps = 2)
    assert replicas == [] and "no longer idle" in (router.disabled_reason or "")
    assert unet.calls[-1] == 2  # whole on the source from then on


def test_automatic_replica_is_not_built_on_a_card_another_job_keeps_busy(monkeypatch):
    # memory still free (under 1 GiB used), but another process computes on it: the build-time recheck refuses too
    unet = UNet2DConditionModel()
    router, replicas = _router(unet)
    router.automatic_idle_only = True
    monkeypatch.setattr(router, "_secondary_free_total_mib", lambda: (15000, 15360))
    # read at render time, outside the load's selection scope (where _util_pct answers None)
    monkeypatch.setattr(mg, "_smi_util_pct", lambda _o: 80.0)
    assert mg._SELECTION.get() is None
    lat = torch.randn(1, 4, 8, 8)
    _sdxl_loop(unet, lat, steps = 2)
    assert replicas == [] and "no longer idle" in (router.disabled_reason or "")
    assert "80.0% busy" in router.disabled_reason


def test_group_offloaded_denoiser_runs_cfg_parallel_from_its_onload_card(monkeypatch):
    from core.inference import diffusion_memory as dm

    class Group:
        onload_device = torch.device("cuda", 0)

    class Pipe:
        transformer = _Dit()

    monkeypatch.setattr(dm, "_offload_groups", lambda module: [Group()])
    monkeypatch.setattr(mg, "_hooks_present", lambda m: None)
    sel = mg.MultiGpuSelection(primary = 0, secondaries = (1,))
    # parameters in host memory between calls, groups load onto cuda:0: a CFG-parallel candidate
    assert mg._execution_device(Pipe.transformer) == torch.device("cuda", 0)
    assert mg.cfg_parallel_decline(Pipe(), offload_policy = "group", selection = sel) is None
    monkeypatch.setattr(dm, "_offload_groups", lambda module: [])
    assert (
        mg.cfg_parallel_decline(Pipe(), offload_policy = "group", selection = sel)
        == "denoiser is not on a CUDA card"
    )


def test_group_offloaded_onto_bare_cuda_names_the_current_card(monkeypatch):
    from core.inference import diffusion_memory as dm

    class Group:
        onload_device = torch.device("cuda")

    monkeypatch.setattr(dm, "_offload_groups", lambda module: [Group()])
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    assert mg._execution_device(_Dit()) == torch.device("cuda", 0)
