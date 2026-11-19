# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The batched tiled decode writes frames and tiles into one preallocated output instead of ``torch.cat``: bit-identical
to diffusers' stock ``tiled_decode`` (CPU, tiny random Wan 2.1 / Wan 2.2 / Qwen-Image VAEs), and no output-sized cat."""

import inspect
import os

import pytest

torch = pytest.importorskip("torch")
diffusers = pytest.importorskip("diffusers")

from core.inference import diffusion_vae_fused as F  # noqa: E402


def _wan21():
    torch.manual_seed(0)
    return diffusers.AutoencoderKLWan(
        base_dim = 8, z_dim = 4, dim_mult = [1, 2, 2, 2], num_res_blocks = 1
    ).eval()


def _wan22():
    torch.manual_seed(0)
    return diffusers.AutoencoderKLWan(
        base_dim = 8, decoder_base_dim = 8, z_dim = 8, dim_mult = [1, 2, 2, 2], num_res_blocks = 1, in_channels = 12,
        out_channels = 12, is_residual = True, patch_size = 2, scale_factor_spatial = 16,
        latents_mean = [0.0] * 8, latents_std = [1.0] * 8,
    ).eval()  # fmt: skip


def _qwen():
    torch.manual_seed(0)
    return diffusers.AutoencoderKLQwenImage(
        base_dim = 8, z_dim = 4, dim_mult = [1, 2, 2, 2], num_res_blocks = 1
    ).eval()


# (factory, latent channels, tile px, stride px, latent (H, W)): odd tile counts, short edge tiles on both axes
_CASES = {
    "wan21": (_wan21, 4, 32, 24, (7, 10)),
    "wan22_patch2": (_wan22, 8, 64, 48, (7, 10)),
    "qwen": (_qwen, 4, 32, 24, (10, 7)),
}


def _stock_decode(vae, z):
    return type(vae).tiled_decode(vae, z, return_dict = False)[0]


@pytest.mark.parametrize("name", sorted(_CASES))
@pytest.mark.parametrize("frames", [1, 3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_batched_tiled_decode_bit_identical_to_stock(monkeypatch, name, frames, dtype):
    # one tile per decoder call: a batched conv may round differently from stock's one-tile calls on CPU
    batch = "1"
    make, zc, tile, stride, (h, w) = _CASES[name]
    vae = make().to(dtype)
    vae.enable_tiling(tile_sample_min_height = tile, tile_sample_min_width = tile, tile_sample_stride_height = stride,
                      tile_sample_stride_width = stride)  # fmt: skip
    monkeypatch.setenv(F.TILE_BATCH_ENV, batch)
    z = torch.randn(1, zc, frames, h, w, generator = torch.Generator().manual_seed(1)).to(dtype)
    clamp = F._stock_tiled_decode_clamps(type(vae))
    assert clamp is not None
    with torch.no_grad():
        ref = _stock_decode(vae, z)
        got = F._wan_batched_tiled_decode(vae, z, return_dict = False, clamp = clamp)[0]
    assert got.shape == ref.shape and got.dtype == ref.dtype
    assert torch.equal(got, ref)


def test_no_output_sized_cat_on_the_tiled_decode_path(monkeypatch):
    """main concatenated every tile's frames, then every row, then the rows; now only the latent tiles (dim 0) are."""
    make, zc, tile, stride, (h, w) = _CASES["wan22_patch2"]
    vae = make()
    vae.enable_tiling(tile_sample_min_height = tile, tile_sample_min_width = tile, tile_sample_stride_height = stride,
                      tile_sample_stride_width = stride)  # fmt: skip
    monkeypatch.setenv(F.TILE_BATCH_ENV, "2")
    real_cat = torch.cat
    ours = os.path.abspath(F.__file__)
    seen = []

    def counting_cat(tensors, *args, **kwargs):
        caller = inspect.currentframe().f_back
        dim = kwargs.get("dim", args[0] if args else 0)
        if os.path.abspath(caller.f_code.co_filename) == ours and dim != 0:
            seen.append((caller.f_code.co_name, dim))
        return real_cat(tensors, *args, **kwargs)

    monkeypatch.setattr(torch, "cat", counting_cat)
    z = torch.randn(1, zc, 4, h, w, generator = torch.Generator().manual_seed(2))
    with torch.no_grad():
        F._wan_batched_tiled_decode(vae, z, return_dict = False, clamp = True)
    assert seen == []


@pytest.mark.parametrize(
    "sizes", [[1, 4, 4, 4], [1, 4], [3], [2, 2, 2], [1, 4, 3, 4], [1, 4, 4, 5], [1, 3, 4, 4]]
)
@pytest.mark.parametrize("channels_last", [False, True])
def test_frame_sink_matches_cat(sizes, channels_last):
    g = torch.Generator().manual_seed(3)
    parts = [torch.randn(2, 3, t, 5, 7, generator = g) for t in sizes]
    if channels_last:
        parts = [p.contiguous(memory_format = torch.channels_last_3d) for p in parts]
    sink = F._FrameSink(len(parts))
    for p in parts:
        sink.add(p)
    got = sink.result()
    ref = torch.cat(parts, dim = 2)
    assert got.shape == ref.shape and torch.equal(got, ref)
    if all(t == sizes[1] for t in sizes[1:]) and len(sizes) > 2:
        # preallocated: the same layout as the cat would have produced
        assert got.is_contiguous(memory_format = torch.channels_last_3d) == ref.is_contiguous(
            memory_format = torch.channels_last_3d
        )
        assert got.is_contiguous() == ref.is_contiguous()


def _old_front_pad(x, norm, act, front, cache, zero_pad, tail):
    """main's reference path of rms_norm_act for a causal front (cat + pad + pad + channels-last)."""
    import torch.nn.functional as tf

    y = F.rms_norm_reference(x, norm, act, None)
    n_cache = 0
    if cache is not None:
        cache = cache[:, :, -front:].to(y.dtype)
        n_cache = cache.shape[2]
        y = torch.cat([cache, y], dim = 2)
    y = tf.pad(y, (0, 0, 0, 0, front - n_cache, 0))
    tail_t = y[:, :, -tail:].clone(memory_format = torch.channels_last_3d) if tail else None
    if zero_pad is not None:
        y = tf.pad(y, (zero_pad[1], zero_pad[1], zero_pad[0], zero_pad[0], 0, 0))
    y = y.contiguous(memory_format = torch.channels_last_3d)
    return (y, tail_t) if tail else y


@pytest.mark.parametrize("cache_frames", [None, 1, 2, 3])
@pytest.mark.parametrize("zero_pad", [None, (1, 1), (1, 2)])
@pytest.mark.parametrize("tail", [0, 2])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_rms_norm_act_front_buffer_matches_cat_and_pads(cache_frames, zero_pad, tail, dtype):
    from diffusers.models.autoencoders.autoencoder_kl_wan import WanRMS_norm

    torch.manual_seed(4)
    norm = WanRMS_norm(6, images = False)
    norm.gamma.data.uniform_(0.5, 1.5)
    x = torch.randn(1, 6, 3, 5, 7).to(dtype)
    cache = None if cache_frames is None else torch.randn(1, 6, cache_frames, 5, 7)
    ref = _old_front_pad(x, norm, True, 2, cache, zero_pad, tail)
    got = F.rms_norm_act(x, norm, True, front = 2, cache = cache, zero_pad = zero_pad, tail = tail)
    ref, got = (ref, got) if tail else ((ref,), (got,))
    for r, g in zip(ref, got):
        assert r.shape == g.shape and r.dtype == g.dtype and torch.equal(r, g)
        assert g.is_contiguous(memory_format = torch.channels_last_3d)
