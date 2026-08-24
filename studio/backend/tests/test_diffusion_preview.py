# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit tests for the latent preview renderer (``diffusion_preview.py``).

Runs on CPU against a stub VAE whose decoder is an exact linear map, so the fitted
projection has a known answer and the shape handling (dense, packed, video, junk) is
exercised without any model weights."""

from __future__ import annotations

import base64

import pytest
import torch

from core.inference import diffusion_preview as preview


class _Config:
    def __init__(
        self,
        channels,
        scaling = 1.0,
        shift = 0.0,
    ):
        self.latent_channels = channels
        self.scaling_factor = scaling
        self.shift_factor = shift


class _LinearVae(torch.nn.Module):
    """A VAE whose decode is a fixed per-channel linear map to RGB, upscaled 8x."""

    def __init__(
        self,
        channels = 4,
        scaling = 1.0,
        shift = 0.0,
    ):
        super().__init__()
        self.config = _Config(channels, scaling, shift)
        torch.manual_seed(7)
        self.weight = torch.nn.Parameter(torch.randn(channels, 3) * 0.3)

    def decode(self, latents):
        rgb = (latents.permute(0, 2, 3, 1) @ self.weight).permute(0, 3, 1, 2)
        sample = torch.nn.functional.interpolate(rgb, scale_factor = 8, mode = "nearest")
        return type("DecoderOutput", (), {"sample": sample})()


class _BrokenVae(_LinearVae):
    def decode(self, latents):
        raise RuntimeError("no decoder here")


@pytest.fixture(autouse = True)
def _clear_cache():
    preview.reset()
    yield
    preview.reset()


def _decode_data_url(url):
    assert url.startswith("data:image/jpeg;base64,")
    return base64.b64decode(url.split(",", 1)[1])


def test_projection_recovers_a_linear_decoder():
    vae = _LinearVae()
    fitted = preview.projection(vae, torch)
    assert fitted is not None
    assert fitted.shape == (5, 3)
    assert torch.allclose(fitted[:4], vae.weight, atol = 1e-3)
    assert torch.allclose(fitted[4], torch.zeros(3), atol = 1e-3)


def test_projection_is_cached_per_vae():
    vae = _LinearVae()
    assert preview.projection(vae, torch) is preview.projection(vae, torch)


def test_dense_latents_render_a_jpeg():
    vae = _LinearVae()
    url = preview.render(torch.randn(1, 4, 32, 32), vae, 512, 512, torch)
    assert _decode_data_url(url).startswith(b"\xff\xd8")


def test_packed_latents_render_a_jpeg():
    vae = _LinearVae()
    url = preview.render(torch.randn(1, 32 * 32, 16), vae, 512, 512, torch)
    assert _decode_data_url(url).startswith(b"\xff\xd8")


def test_packed_latents_survive_a_non_square_request():
    vae = _LinearVae()
    assert preview.render(torch.randn(1, 24 * 32, 16), vae, 512, 384, torch) is not None


def test_video_latents_preview_their_first_frame():
    vae = _LinearVae()
    assert preview.render(torch.randn(1, 4, 5, 32, 32), vae, 512, 512, torch) is not None


def test_the_preview_matches_the_request_aspect_ratio():
    from PIL import Image
    import io

    vae = _LinearVae()
    url = preview.render(torch.randn(1, 4, 24, 32), vae, 512, 384, torch)
    image = Image.open(io.BytesIO(_decode_data_url(url)))
    assert image.size == (preview.MAX_EDGE_PX, preview.MAX_EDGE_PX * 384 // 512)


def test_a_broken_decoder_disables_previews_without_raising():
    assert preview.render(torch.randn(1, 4, 32, 32), _BrokenVae(), 512, 512, torch) is None


def test_absent_latents_render_nothing():
    assert preview.render(None, _LinearVae(), 512, 512, torch) is None


def test_a_channel_count_the_vae_disowns_renders_nothing():
    assert preview.render(torch.randn(1, 7, 32, 32), _LinearVae(), 512, 512, torch) is None


def test_an_unpackable_sequence_renders_nothing():
    assert preview.render(torch.randn(1, 32 * 32, 15), _LinearVae(), 512, 512, torch) is None


def test_scaling_and_shift_are_undone_before_projecting():
    plain = _LinearVae()
    scaled = _LinearVae(scaling = 4.0, shift = 0.5)
    latents = torch.randn(1, 4, 16, 16)
    assert preview.render(latents, plain, 512, 512, torch) != preview.render(
        latents * 4.0, scaled, 512, 512, torch
    )
    assert preview.render(latents, plain, 512, 512, torch) == preview.render(
        (latents - 0.5) * 4.0, scaled, 512, 512, torch
    )


def test_previews_can_be_disabled_by_env(monkeypatch):
    assert preview.previews_enabled()
    monkeypatch.setenv(preview.PREVIEWS_ENV, "0")
    assert not preview.previews_enabled()
