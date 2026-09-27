# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The Qwen-Image VAE single-frame 2D path: same image as the stock causal-3D walk, stock path for video/tiles."""

import copy
import types

import pytest

torch = pytest.importorskip("torch")
autoencoders = pytest.importorskip("diffusers.models.autoencoders.autoencoder_kl_qwenimage")

from core.inference import diffusion_vae_single_frame as sf  # noqa: E402


def _tiny_vae():
    torch.manual_seed(0)
    vae = autoencoders.AutoencoderKLQwenImage(
        base_dim = 16,
        z_dim = 4,
        dim_mult = [1, 2, 2],
        num_res_blocks = 1,
        temperal_downsample = [False, True],
        attn_scales = [],
    )
    return vae.float().eval()


@pytest.fixture(autouse = True)
def _no_kill_switch(monkeypatch):
    monkeypatch.delenv(sf.SINGLE_FRAME_ENV, raising = False)


def test_single_frame_decode_and_encode_match_the_stock_walk():
    stock = _tiny_vae()
    fast = copy.deepcopy(stock)
    assert sf.install(fast) is True
    z = torch.randn(1, 4, 1, 16, 16)
    x = torch.rand(1, 3, 1, 64, 64) * 2 - 1
    with torch.no_grad():
        want = stock.decode(z).sample
        got = fast.decode(z).sample
        assert got.shape == want.shape
        assert torch.allclose(got, want, atol = 1e-5, rtol = 1e-4)
        e_want = stock.encode(x).latent_dist.mean
        e_got = fast.encode(x).latent_dist.mean
        assert torch.allclose(e_got, e_want, atol = 1e-5, rtol = 1e-4)


def test_multi_frame_and_tiled_calls_take_the_stock_path(monkeypatch):
    stock = _tiny_vae()
    fast = copy.deepcopy(stock)
    sf.install(fast)
    z = torch.randn(1, 4, 3, 16, 16)
    with torch.no_grad():
        # Bit-identical: the first chunk of the walk also reaches every conv with no cache and one frame.
        assert torch.equal(fast.decode(z).sample, stock.decode(z).sample)
        x = torch.rand(1, 3, 5, 64, 64) * 2 - 1
        assert torch.equal(fast.encode(x).latent_dist.mean, stock.encode(x).latent_dist.mean)
    for vae in (stock, fast):
        vae.enable_tiling(tile_sample_min_height = 32, tile_sample_min_width = 32)
    z1 = torch.randn(1, 4, 1, 16, 16)
    with torch.no_grad():
        assert torch.equal(fast.decode(z1).sample, stock.decode(z1).sample)
    calls = []
    real = fast._unsloth_stock_decode
    fast._unsloth_stock_decode = lambda *a, **k: calls.append(1) or real(*a, **k)
    fast.enable_tiling(tile_sample_min_height = 32, tile_sample_min_width = 32)
    with torch.no_grad():
        fast.decode(torch.randn(1, 4, 1, 16, 16))
    assert calls == [1]


def test_kill_switch_and_foreign_classes(monkeypatch):
    assert sf.install(None) is False
    assert sf.install(types.SimpleNamespace()) is False
    monkeypatch.setenv(sf.SINGLE_FRAME_ENV, "0")
    assert sf.install(_tiny_vae()) is False


def test_install_is_idempotent_and_marks_the_vae():
    vae = _tiny_vae()
    assert sf.install(vae) is True
    first = vae._decode
    assert sf.install(vae) is True
    assert vae._decode is first and vae._unsloth_single_frame is True


@pytest.mark.parametrize("single_frame", ["1", "0"])
@pytest.mark.parametrize("compile_env", [None, "1"])
def test_compile_cache_key_matches_the_later_vae_compile(monkeypatch, single_frame, compile_env):
    """The loader keys the compile bundle on vae_decode_compile_allowed() BEFORE apply_speed_optims installs the
    single-frame path; the key must still equal what that later pass compiles."""
    from core.inference import diffusion_speed as ds

    monkeypatch.setenv(sf.SINGLE_FRAME_ENV, single_frame)
    if compile_env is None:
        monkeypatch.delenv(ds.COMPILE_VAE_ENV, raising = False)
    else:
        monkeypatch.setenv(ds.COMPILE_VAE_ENV, compile_env)
    monkeypatch.setattr(torch, "compile", lambda fn, **kw: fn)
    monkeypatch.setattr(ds, "compile_eligible", lambda *a, **k: True)
    monkeypatch.setattr(ds, "_compile_repeated_blocks", lambda *a, **k: True)
    monkeypatch.setattr(ds, "_install_inductor_backports", lambda *a, **k: False)
    target = types.SimpleNamespace(
        device = "cpu", dtype = torch.bfloat16, supports_default_torch_compile = True
    )
    family = types.SimpleNamespace(supports_torch_compile = True)
    for mode in (ds.SPEED_MAX, ds.SPEED_DEFAULT):
        vae = _tiny_vae()
        pipe = types.SimpleNamespace(vae = vae, transformer = types.SimpleNamespace())
        key = ds.vae_decode_compile_allowed(pipe, mode)
        applied = ds.apply_speed_optims(pipe, target, is_gguf = False, family = family, speed_mode = mode)
        assert applied["compiled_vae_decode"] is key, (mode, single_frame, compile_env)
        # A second load of the same weights asks again with the path already armed.
        assert ds.vae_decode_compile_allowed(pipe, mode) is key
