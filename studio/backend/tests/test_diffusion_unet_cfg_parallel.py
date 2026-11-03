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


def _sdxl_loop(unet, latents, steps, batch = 1, cfg = True, **extra):
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
            x, t, encoder_hidden_states = emb, added_cond_kwargs = {"time_ids": tid}, return_dict = False, **extra
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
    assert mg.cfg_parallel_decline(Pipe(), offload_policy = "none", selection = sel) == "denoiser is not on a CUDA card"
    assert mg._cfg_denoiser(Pipe()) is Pipe.unet
    monkeypatch.setenv(mg.UNET_CFG_PARALLEL_ENV, "0")
    assert mg._cfg_denoiser(Pipe()) is None
    assert mg.UNET_CFG_PARALLEL_ENV in mg.cfg_parallel_decline(Pipe(), offload_policy = "none", selection = sel)


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
            unet, primary = "cpu", secondary = "cpu",
            replica_factory = lambda: mg.build_replica(unet, torch.device("cpu")),
        )
        unet.forward = router
        out = unet(x, t, encoder_hidden_states = emb, added_cond_kwargs = added, return_dict = False)[0]
        out2 = unet(x, t, encoder_hidden_states = emb, added_cond_kwargs = added).sample
    torch.testing.assert_close(out, ref, rtol = 1e-4, atol = 1e-5)
    torch.testing.assert_close(out2, ref, rtol = 1e-4, atol = 1e-5)
    assert router.stats["pairs_parallel"] == 2
