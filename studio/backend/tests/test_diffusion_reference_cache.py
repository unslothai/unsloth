# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Reused condition latents equal a fresh encode; every input that changes them encodes again."""

from __future__ import annotations

import pytest

from core.inference import diffusion_reference_cache as rc

torch = pytest.importorskip("torch")
diffusers = pytest.importorskip("diffusers")
q21 = pytest.importorskip("diffusers.pipelines.qwenimage21.pipeline_qwenimage21")


@pytest.fixture(autouse = True)
def _env(monkeypatch):
    monkeypatch.delenv(rc.CACHE_ENV, raising = False)
    for k in rc.STATS:
        rc.STATS[k] = 0


def _vae(seed = 0):
    torch.manual_seed(seed)
    vae = diffusers.AutoencoderKLQwenImage21(
        base_dim = 8,
        decoder_base_dim = 8,
        z_dim = 4,
        dim_mult = [1, 2, 4, 4, 4],
        num_res_blocks = 1,
        attn_scales = [],
        temperal_downsample = [False, True, True, True],
        in_channels = 4,
        out_channels = 4,
        latents_mean = [0.1, -0.2, 0.3, 0.0],
        latents_std = [1.5, 0.5, 2.0, 1.0],
    ).eval()
    return vae


def _pipe(vae = None):
    cls = q21.QwenImage21Pipeline
    pipe = cls.__new__(cls)
    pipe.__dict__.update(vae = vae or _vae(), latent_channels = 4, vae_scale_factor = 16)
    return pipe


def _spy(vae):
    calls = []
    real = vae.encode

    def encode(*a, **k):
        calls.append(1)
        return real(*a, **k)

    vae.__dict__["encode"] = encode
    return calls


def _images(seed = 1, n = 2):
    g = torch.Generator().manual_seed(seed)
    return [torch.rand(1, 4, 1, 64, 48, generator = g) * 2 - 1 for _ in range(n)]


def _prepare(pipe, images, seed):
    g = torch.Generator().manual_seed(seed)
    with torch.inference_mode():
        lat, img = pipe.prepare_latents(images, 1, 4, 64, 64, torch.float32, torch.device("cpu"), g)
    return lat, img, g.get_state()


def test_reuse_is_exact_and_skips_the_encode():
    images = _images()
    stock = _pipe()
    ref = [_prepare(stock, images, s) for s in (3, 4, 3)]
    pipe = _pipe()
    calls = _spy(pipe.vae)
    assert rc.install(pipe) and rc.install(pipe)
    got = [_prepare(pipe, images, s) for s in (3, 4, 3)]
    for (rl, ri, rg), (gl, gi, gg) in zip(ref, got):
        assert torch.equal(ri, gi) and torch.equal(rl, gl) and torch.equal(rg, gg)
    assert len(calls) == len(images)
    assert rc.STATS["misses"] == 2 and rc.STATS["hits"] == 4
    rc.uninstall(pipe)
    _prepare(pipe, images, 3)
    assert len(calls) == 2 * len(images)


def test_every_changed_input_encodes_again():
    images = _images(n = 1)
    pipe = _pipe()
    calls = _spy(pipe.vae)
    rc.install(pipe)
    _prepare(pipe, images, 0)
    other = [images[0].clone()]
    other[0][0, 0, 0, 0, 0] += 0.25
    _prepare(pipe, other, 0)
    assert len(calls) == 2
    pipe.vae.use_tiling = True
    _, tiled, _ = _prepare(pipe, images, 0)  # tiling changes what encode returns
    assert len(calls) == 3
    pipe.vae.use_tiling = False
    _prepare(pipe, images, 0)
    assert len(calls) == 3
    pipe.vae.to(torch.float64)
    with torch.inference_mode():
        pipe.prepare_latents(images, 1, 4, 64, 64, torch.float64, torch.device("cpu"), None)
    assert len(calls) == 4  # a cast VAE (and float64 pixels)
    pipe.vae.to(torch.float32)
    fresh = _vae(seed = 5)
    fresh_calls = _spy(fresh)
    pipe.__dict__["vae"] = fresh
    _, img, _ = _prepare(pipe, images, 0)
    assert len(fresh_calls) == 1  # another VAE object: the cache starts over
    _, ref, _ = _prepare(_pipe(_vae(seed = 5)), images, 0)
    assert torch.equal(img, ref)


def test_kill_switch(monkeypatch):
    pipe = _pipe()
    calls = _spy(pipe.vae)
    monkeypatch.setenv(rc.CACHE_ENV, "0")
    assert not rc.install(pipe)
    monkeypatch.delenv(rc.CACHE_ENV)
    assert rc.install(pipe)
    monkeypatch.setenv(rc.CACHE_ENV, "0")
    _prepare(pipe, _images(n = 1), 0)
    _prepare(pipe, _images(n = 1), 0)
    assert len(calls) == 2 and rc.STATS["bypassed"] == 2


def test_a_sampled_encode_is_never_cached(monkeypatch):
    def sampled(self, image, generator):
        return q21.retrieve_latents(
            self.vae.encode(image), generator = generator, sample_mode = "sample"
        )

    monkeypatch.setattr(q21.QwenImage21Pipeline, "_encode_vae_image", sampled)
    assert not rc.install(_pipe())


def test_other_pipelines_are_left_alone():
    class Other:
        def _encode_vae_image(self, image, generator):
            return image

    assert not rc.install(Other())


def test_the_cache_is_bounded(monkeypatch):
    monkeypatch.setattr(rc, "_MAX_ENTRIES", 2)
    pipe = _pipe()
    calls = _spy(pipe.vae)
    rc.install(pipe)
    sets = [_images(seed = s, n = 1) for s in range(3)]
    for images in sets:
        _prepare(pipe, images, 0)
    _prepare(pipe, sets[0], 0)  # LRU-evicted
    _prepare(pipe, sets[2], 0)
    assert len(calls) == 4
    assert len(getattr(pipe, rc._MARK).entries) == 2


def _encode(
    pipe,
    image,
    autocast = False,
):
    with torch.inference_mode(), torch.autocast("cpu", dtype = torch.bfloat16, enabled = autocast):
        return pipe._encode_vae_image(image, None), pipe._encode_vae_image.__wrapped__(image, None)


def test_autocast_is_part_of_the_key():
    pipe = _pipe()
    image = _images(n = 1)[0]
    rc.install(pipe)
    _encode(pipe, image)
    got, want = _encode(pipe, image, autocast = True)
    assert got.dtype == want.dtype == torch.bfloat16 and torch.equal(got, want)
    got, want = _encode(pipe, image)
    assert got.dtype == torch.float32 and torch.equal(got, want)
    assert rc.STATS["misses"] == 2 and rc.STATS["hits"] == 1


def test_reloaded_weights_and_config_encode_again():
    pipe = _pipe()
    image = _images(n = 1)[0]
    rc.install(pipe)
    _encode(pipe, image)
    pipe.vae.to(torch.float64).to(torch.float32)  # an offload-style move keeps the entry
    _encode(pipe, image)
    assert rc.STATS["hits"] == 1
    pipe.vae.load_state_dict(_vae(seed = 9).state_dict())  # same object, new weights
    got, want = _encode(pipe, image)
    assert torch.equal(got, want)
    pipe.vae.register_to_config(latents_mean = [0.5, 0.5, 0.5, 0.5])
    got, want = _encode(pipe, image)
    assert torch.equal(got, want)
    assert rc.STATS["misses"] == 3 and rc.STATS["hits"] == 1


def test_assigned_weights_encode_again():
    pipe = _pipe()
    image = _images(n = 1)[0]
    rc.install(pipe)
    _encode(pipe, image)
    before = rc._weights_revision(pipe.vae)
    pipe.vae.load_state_dict(_vae(seed = 9).state_dict(), assign = True)
    assert rc._weights_revision(pipe.vae) == before  # the revision alone cannot see the swap
    got, want = _encode(pipe, image)
    assert torch.equal(got, want)
    pipe.vae.decoder.load_state_dict(_vae(seed = 11).decoder.state_dict(), assign = True)
    got, want = _encode(pipe, image)
    assert torch.equal(got, want)
    _encode(pipe, image)
    assert rc.STATS["misses"] == 3 and rc.STATS["hits"] == 1


def test_a_vae_that_takes_no_hooks_is_never_cached(monkeypatch):
    pipe = _pipe()
    image = _images(n = 1)[0]

    def _refuse(*a, **k):
        raise RuntimeError("no hooks")

    monkeypatch.setattr(torch.nn.Module, "register_load_state_dict_post_hook", _refuse)
    rc.install(pipe)
    got, want = _encode(pipe, image)
    _encode(pipe, image)
    assert torch.equal(got, want) and rc.STATS["hits"] == 0 and rc.STATS["bypassed"] == 2
