# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""AdaLN precompute against real diffusers FLUX.1 / Qwen-Image transformers (tiny random configs): the denoiser
outputs stay bit-identical while the modulation weights live on the host and are read once per request."""

from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")
diffusers = pytest.importorskip("diffusers")

from core.inference import diffusion_adaln_precompute as ap  # noqa: E402

DTYPES = [torch.float32, torch.bfloat16]


def _flux(guidance: bool = True):
    torch.manual_seed(0)
    m = diffusers.FluxTransformer2DModel(
        patch_size = 1,
        in_channels = 16,
        num_layers = 2,
        num_single_layers = 2,
        attention_head_dim = 16,
        num_attention_heads = 2,
        joint_attention_dim = 32,
        pooled_projection_dim = 24,
        guidance_embeds = guidance,
        axes_dims_rope = (4, 6, 6),
    )
    return m.eval()


def _qwen(zero_cond_t: bool = False):
    torch.manual_seed(0)
    m = diffusers.QwenImageTransformer2DModel(
        patch_size = 2,
        in_channels = 16,
        out_channels = 4,
        num_layers = 2,
        attention_head_dim = 16,
        num_attention_heads = 2,
        joint_attention_dim = 32,
        axes_dims_rope = (4, 6, 6),
        zero_cond_t = zero_cond_t,
    )
    return m.eval()


class _Pipe:
    def __init__(self, timesteps):
        self.scheduler = types.SimpleNamespace(timesteps = timesteps)


_PIPES: list = []


def _keep(p):
    _PIPES.append(p)
    return p


# the transformer holds its pipe weakly, as Studio's pipe outlives every render


def _pipe(steps: int):
    p = _Pipe(torch.linspace(1000.0, 30.0, steps))
    _PIPES.append(p)
    return p


def _flux_calls(
    dtype,
    batch = 1,
    guidance = 3.5,
    pooled_seed = 1,
    steps = 4,
):
    g = torch.Generator().manual_seed(pooled_seed)
    hs = torch.randn(batch, 16, 16, generator = g).to(dtype)
    ehs = torch.randn(batch, 8, 32, generator = g).to(dtype)
    pooled = torch.randn(batch, 24, generator = g).to(dtype)
    img_ids = torch.zeros(16, 3)
    img_ids[:, 1] = torch.arange(16) // 4
    img_ids[:, 2] = torch.arange(16) % 4
    txt_ids = torch.zeros(8, 3)
    sched = torch.linspace(1000.0, 30.0, steps)
    calls = []
    for t in sched:
        ts = t.expand(batch).to(dtype)
        kw = dict(
            hidden_states = hs,
            encoder_hidden_states = ehs,
            pooled_projections = pooled,
            timestep = ts / 1000,
            img_ids = img_ids,
            txt_ids = txt_ids,
            return_dict = False,
        )
        if guidance is not None:
            kw["guidance"] = torch.full([batch], guidance).to(dtype)
        calls.append(kw)
    return calls


def _qwen_calls(
    dtype,
    batch = 1,
    steps = 4,
    zero_cond_t = False,
):
    g = torch.Generator().manual_seed(2)
    n_img = 16 if not zero_cond_t else 32
    hs = torch.randn(batch, n_img, 16, generator = g).to(dtype)
    ehs = torch.randn(batch, 6, 32, generator = g).to(dtype)
    mask = torch.ones(batch, 6, dtype = torch.long)
    shapes = [[(1, 4, 4)] if not zero_cond_t else [(1, 4, 4), (1, 4, 4)]] * batch
    sched = torch.linspace(1000.0, 30.0, steps)
    calls = []
    for t in sched:
        ts = t.expand(batch).to(dtype)
        calls.append(
            dict(
                hidden_states = hs,
                encoder_hidden_states = ehs,
                encoder_hidden_states_mask = mask,
                timestep = ts / 1000,
                img_shapes = shapes,
                return_dict = False,
            )
        )
    return calls


def _run(model, calls):
    with torch.no_grad():
        return [model(**kw)[0].clone() for kw in calls]


def _counts():
    return dict(ap.COUNTS)


@pytest.fixture(autouse = True)
def _verify(monkeypatch):
    monkeypatch.setenv(ap.VERIFY_ENV, "1")
    yield


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("batch", [1, 2])
def test_flux_bit_identical_one_weight_pass(dtype, batch):
    model = _flux().to(dtype)
    calls = _flux_calls(dtype, batch = batch)
    ref = _run(model, calls)
    mod_params = sum(
        p.numel() for n, p in model.named_parameters() if "norm" in n and "linear" in n
    )
    total = sum(p.numel() for p in model.parameters())
    before = _counts()
    assert ap.install(model, _pipe(len(calls)), requested = ap.MODE_ON)
    after_total = sum(p.numel() for p in model.parameters())
    # every modulation Linear left the module tree (the parameters a placement / offloader would move)
    assert total - after_total == mod_params
    assert not any("norm1.linear" in n or "norm.linear" in n for n, _ in model.named_parameters())
    out = _run(model, calls)
    c = _counts()
    for a, b in zip(ref, out):
        assert torch.equal(a, b)
    assert (
        c["weight_passes"] - before["weight_passes"] == 1
    )  # every step predicted from the scheduler
    assert c["served"] - before["served"] == len(calls)
    assert c["verify_checks"] > before["verify_checks"]
    # a second request with the same inputs re-reads no weights
    out2 = _run(model, calls)
    assert _counts()["weight_passes"] == c["weight_passes"]
    assert all(torch.equal(a, b) for a, b in zip(ref, out2))


@pytest.mark.parametrize("dtype", DTYPES)
def test_flux_cfg_branches_and_guidance_change(dtype):
    model = _flux().to(dtype)
    cond = _flux_calls(dtype, pooled_seed = 1)
    uncond = _flux_calls(dtype, pooled_seed = 7)
    other_g = _flux_calls(dtype, pooled_seed = 1, guidance = 5.0)
    interleaved = [kw for pair in zip(cond, uncond) for kw in pair] + other_g
    ref = _run(model, interleaved)
    assert ap.install(model, _pipe(4), requested = ap.MODE_ON)
    out = _run(model, interleaved)
    for a, b in zip(ref, out):
        assert torch.equal(a, b)
    # each pooled / guidance set has its own rows: 3 requests' worth
    assert ap.summary(model)["cached_rows"] == 12


def test_flux_schnell_no_guidance():
    model = _flux(guidance = False)
    calls = _flux_calls(torch.float32, guidance = None)
    ref = _run(model, calls)
    assert ap.install(model, _pipe(len(calls)), requested = ap.MODE_ON)
    assert all(torch.equal(a, b) for a, b in zip(ref, _run(model, calls)))


def test_unpredicted_timestep_computes_lazily():
    model = _flux()
    calls = _flux_calls(torch.float32)
    ref = _run(model, calls)
    # a scheduler grid that never matches: every step is its own lazy pass, still exact
    assert ap.install(model, _keep(_Pipe(torch.linspace(999.0, 31.0, 9))), requested = ap.MODE_ON)
    before = _counts()["weight_passes"]
    assert all(torch.equal(a, b) for a, b in zip(ref, _run(model, calls)))
    assert _counts()["weight_passes"] - before == len(calls)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("zero_cond_t", [False, True])
def test_qwen_image_bit_identical(dtype, zero_cond_t):
    model = _qwen(zero_cond_t = zero_cond_t).to(dtype)
    calls = _qwen_calls(dtype, zero_cond_t = zero_cond_t)
    ref = _run(model, calls)
    assert ap.install(model, _pipe(len(calls)), requested = ap.MODE_ON)
    assert not any("img_mod" in n or "txt_mod" in n for n, _ in model.named_parameters())
    before = _counts()["weight_passes"]
    out = _run(model, calls)
    assert all(torch.equal(a, b) for a, b in zip(ref, out))
    assert _counts()["weight_passes"] - before == 1


def test_uninstall_restores_stock_modules():
    model = _flux()
    calls = _flux_calls(torch.float32)
    ref = _run(model, calls)
    n_params = sum(p.numel() for p in model.parameters())
    assert ap.install(model, _pipe(4), requested = ap.MODE_ON)
    assert ap.uninstall(model, reason = "test")
    assert not ap.installed(model)
    assert sum(p.numel() for p in model.parameters()) == n_params
    assert isinstance(model.transformer_blocks[0].norm1.linear, torch.nn.Linear)
    assert all(torch.equal(a, b) for a, b in zip(ref, _run(model, calls)))


def test_modes_and_declines(monkeypatch):
    model = _flux()
    monkeypatch.setenv(ap.ENV, "0")
    assert ap.decline_reason(model, offloaded = True) is not None
    monkeypatch.setenv(ap.ENV, "auto")
    assert ap.decline_reason(model, offloaded = False) is not None
    assert ap.decline_reason(model, offloaded = True) is None
    monkeypatch.setenv(ap.ENV, "1")
    assert ap.decline_reason(model, offloaded = False) is None
    # shared-modulation families are not installed
    assert ap.adapter_for(torch.nn.Linear(2, 2)) is None


def test_compiled_blocks_bit_identical():
    model = _flux()
    calls = _flux_calls(torch.float32)
    ref = _run(model, calls)
    assert ap.install(model, _pipe(len(calls)), requested = ap.MODE_ON)
    import os

    os.environ.pop(ap.VERIFY_ENV, None)
    for blk in list(model.transformer_blocks) + list(model.single_transformer_blocks):
        blk.compile(backend = "aot_eager", fullgraph = True)
    out = _run(model, calls)
    assert all(torch.allclose(a, b, atol = 1e-6, rtol = 0) for a, b in zip(ref, out))
