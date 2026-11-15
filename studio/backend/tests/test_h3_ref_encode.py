# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""CPU tests for the chunked MiniMax-H3 reference-video condition on a tiny random H3 VAE."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
encoders = pytest.importorskip("diffusers.modular_pipelines.minimax_h3.encoders")
vae_mod = pytest.importorskip("diffusers.models.autoencoders.autoencoder_kl_minimax_h3")

from core.inference import video_minimax_h3_ref_encode as ref_encode  # noqa: E402

_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


def _vae():
    torch.manual_seed(0)
    vae = vae_mod.AutoencoderKLMiniMaxH3(
        latent_channels = 4,
        block_out_channels = [8, 8, 16, 16, 16, 16],
        layers_per_block = 1,
        norm_num_groups = 4,
        decoder_num_layers = 1,
        decoder_num_attention_heads = 2,
        decoder_attention_head_dim = 8,
        latents_mean = (0.1, -0.2, 0.3, 0.0),
        latents_std = (1.5, 0.5, 2.0, 1.0),
    ).eval()
    return vae


def _pixels(
    frames,
    height = 32,
    width = 48,
    seed = 0,
):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (1, 3, frames, height, width), generator = g, dtype = torch.uint8)


@pytest.fixture
def shim(monkeypatch):
    monkeypatch.delenv(ref_encode.H3_CHUNKED_REF_ENCODE_ENV, raising = False)
    original = encoders.encode_vae_condition
    if ref_encode._sha(original) not in ref_encode._KNOWN_SOURCES["encode_vae_condition"]:
        pytest.skip("diffusers H3 encoder source this shim does not mirror")
    monkeypatch.setattr(encoders, "encode_vae_condition", original)
    assert ref_encode.install_h3_chunked_reference_encode()
    assert ref_encode.install_h3_chunked_reference_encode()
    installed = encoders.encode_vae_condition
    assert installed is not original and installed.__wrapped__ is original
    return original, installed


@pytest.mark.parametrize("frames", [22, 34, 39, 56])
def test_chunked_reference_matches_stock_bit_for_bit(shim, frames):
    original, installed = shim
    vae = _vae()
    pixels = _pixels(frames)
    with torch.no_grad():
        stock = original(vae, pixels, _MEAN, _STD, 42)
        calls = {"n": 0}
        clip = vae._encode_clip

        def counted(x):
            calls["n"] += 1
            assert x.shape[2] == vae.config.clip_length and x.is_contiguous()
            return clip(x)

        vae._encode_clip = counted
        chunked = installed(vae, pixels, _MEAN, _STD, 42)
    assert calls["n"] == -(-frames // vae.config.clip_length)
    assert stock.dtype == chunked.dtype and stock.shape == chunked.shape
    assert torch.equal(stock, chunked)


def test_spatial_tiles_inside_each_chunk_match(shim):
    original, installed = shim
    vae = _vae()
    vae.enable_tiling(
        tile_sample_min_height = 32,
        tile_sample_min_width = 32,
        tile_sample_min_overlap_height = 16,
        tile_sample_min_overlap_width = 16,
    )
    pixels = _pixels(22, height = 48, width = 64, seed = 3)
    with torch.no_grad():
        assert torch.equal(
            original(vae, pixels, _MEAN, _STD, 7), installed(vae, pixels, _MEAN, _STD, 7)
        )


def test_single_frames_short_clips_and_the_switch_keep_stock(shim, monkeypatch):
    original, installed = shim
    vae = _vae()
    seen = []
    monkeypatch.setattr(ref_encode, "_chunked_moments", lambda *a: seen.append(a) or 1 / 0)
    with torch.no_grad():
        for frames in (1, 17):
            p = _pixels(frames)
            assert torch.equal(
                installed(vae, p, _MEAN, _STD, 42), original(vae, p, _MEAN, _STD, 42)
            )
        p = _pixels(22)
        monkeypatch.setenv(ref_encode.H3_CHUNKED_REF_ENCODE_ENV, "0")
        assert torch.equal(installed(vae, p, _MEAN, _STD, 42), original(vae, p, _MEAN, _STD, 42))
        monkeypatch.delenv(ref_encode.H3_CHUNKED_REF_ENCODE_ENV)
        # float pixels or an instance-level encode wrapper are not what the mirror covers
        assert torch.equal(
            installed(vae, p.float(), _MEAN, _STD, 42), original(vae, p.float(), _MEAN, _STD, 42)
        )
        vae.encode = vae.encode
        assert torch.equal(installed(vae, p, _MEAN, _STD, 42), original(vae, p, _MEAN, _STD, 42))
    assert seen == []


def test_unknown_source_is_not_shimmed(monkeypatch):
    monkeypatch.delenv(ref_encode.H3_CHUNKED_REF_ENCODE_ENV, raising = False)

    def encode_vae_condition(
        vae,
        pixels,
        pixel_mean,
        pixel_std,
        encode_seed = 42,
    ):
        return None

    monkeypatch.setattr(encoders, "encode_vae_condition", encode_vae_condition)
    assert not ref_encode.install_h3_chunked_reference_encode()
    assert encoders.encode_vae_condition is encode_vae_condition
