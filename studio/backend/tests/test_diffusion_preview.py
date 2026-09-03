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


class _PerChannelVae(_LinearVae):
    """A VAE that normalizes each latent channel, as Qwen-Image and Krea-2 do.

    Those configs carry no ``scaling_factor`` at all, so reading only the scalar pair left
    their latents un-denormalized and their previews miscoloured."""

    def __init__(self, channels = 4):
        super().__init__(channels = channels)
        del self.config.scaling_factor
        del self.config.shift_factor
        self.config.latents_mean = [0.10, -0.20, 0.30, -0.40][:channels]
        self.config.latents_std = [2.0, 0.5, 1.5, 0.8][:channels]


class _BatchNormVae(_LinearVae):
    """FLUX.2 shape: normalization lives in the VAE's BatchNorm, over patchified latents."""

    def __init__(self):
        super().__init__()
        self.bn = torch.nn.BatchNorm2d(4)


def test_the_projection_solve_survives_a_backend_without_lstsq():
    """The real reproduction: fit a VAE that lives on MPS.

    ``torch.linalg.lstsq`` has no MPS kernel, ``projection()`` swallows the
    NotImplementedError, and None is cached -- so every Mac generation silently got no
    preview after paying for the decode. Solving on CPU is what makes this pass."""
    if not torch.backends.mps.is_available():
        # Hides no changed-path failure: the bug IS a missing MPS kernel, so only a host
        # with MPS reproduces it, and every other test here covers the same solve on CPU.
        pytest.skip(reason = "no MPS backend on this host, which is what this test needs")
    vae = _LinearVae().to("mps")
    fitted = preview.projection(vae, torch)
    assert fitted is not None, "the projection was not fitted on an MPS VAE"
    assert fitted.device.type == "cpu"
    url = preview.render(torch.randn(1, 4, 32, 32, device = "mps"), vae, 512, 512, torch)
    assert url is not None and _decode_data_url(url).startswith(b"\xff\xd8")


def test_per_channel_statistics_are_undone_before_projecting():
    vae = _PerChannelVae()
    std = torch.tensor(vae.config.latents_std)
    mean = torch.tensor(vae.config.latents_mean)
    latents = torch.randn(1, 4, 16, 16)
    # The denoiser-side value whose decoder-space counterpart is exactly `latents`.
    normalized = (latents - mean.view(1, 4, 1, 1)) / std.view(1, 4, 1, 1)
    plain = _LinearVae()
    assert preview.render(normalized, vae, 512, 512, torch) == preview.render(
        latents, plain, 512, 512, torch
    )


def test_ignoring_per_channel_statistics_would_change_the_preview():
    # Guards the fix itself: if the scalar path were still taken for these VAEs the frame
    # above would equal the un-denormalized one, and the assertion there could not fail.
    vae = _PerChannelVae()
    latents = torch.randn(1, 4, 16, 16)
    plain = _LinearVae()
    assert preview.render(latents, vae, 512, 512, torch) != preview.render(
        latents, plain, 512, 512, torch
    )


def test_a_batchnorm_normalized_vae_renders_nothing():
    # FLUX.2's statistics are over patchified latents, whose channel count no longer
    # matches the config: no preview beats a miscoloured one.
    assert preview.render(torch.randn(1, 4, 16, 16), _BatchNormVae(), 512, 512, torch) is None


class _QwenShapedVae(_LinearVae):
    """Qwen-Image / Krea-2 shape: channels named z_dim, and a 5-D [B,C,T,H,W] decoder."""

    def __init__(self, channels = 4):
        super().__init__(channels = channels)
        del self.config.latent_channels
        del self.config.scaling_factor
        del self.config.shift_factor
        self.config.z_dim = channels
        self.config.latents_mean = [0.10, -0.20, 0.30, -0.40][:channels]
        self.config.latents_std = [2.0, 0.5, 1.5, 0.8][:channels]

    def decode(self, latents):
        if latents.ndim != 5:
            raise ValueError("this decoder takes [B, C, T, H, W]")
        out = super().decode(latents[:, :, 0])
        return type("DecoderOutput", (), {"sample": out.sample.unsqueeze(2)})()


class _UpcastVae(_LinearVae):
    """An SDXL-shaped VAE: loaded in float16, but decoded in float32 by the pipeline."""

    def __init__(self):
        super().__init__()
        self.config.force_upcast = True
        self.half()

    def decode(self, latents):
        if latents.dtype == torch.float16:
            raise RuntimeError("decoding this VAE in float16 overflows")
        return super().decode(latents)


def test_a_vae_that_names_its_channels_z_dim_still_previews():
    # latent_channels is absent on these configs, so reading only that name left channels at
    # 0 and projection() cached None -- Qwen-Image and Krea-2 never previewed at all.
    vae = _QwenShapedVae()
    assert preview._latent_channels(vae) == 4
    assert preview.projection(vae, torch) is not None


def test_a_five_dimensional_decoder_renders_a_jpeg():
    vae = _QwenShapedVae()
    url = preview.render(torch.randn(1, 4, 32, 32), vae, 512, 512, torch)
    assert url is not None and _decode_data_url(url).startswith(b"\xff\xd8")


def test_a_force_upcast_vae_is_fitted_in_float32_and_left_as_it_was():
    vae = _UpcastVae()
    assert preview.projection(vae, torch) is not None
    # The fit must not leave the VAE upcast behind it.
    assert next(vae.parameters()).dtype == torch.float16
