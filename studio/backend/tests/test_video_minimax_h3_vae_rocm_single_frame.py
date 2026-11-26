# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""ROCm single-frame H3 encoder lowering (causal Conv3d -> Conv2d of the last tap). CPU only; ROCm is mocked."""

from __future__ import annotations

import pytest
import torch

from core.inference import video_minimax_h3_vae as H

H3 = pytest.importorskip("diffusers.models.autoencoders.autoencoder_kl_minimax_h3")
CausalConv = H3.MiniMaxH3VideoCausalConv3d


class AutoencoderKLMiniMaxH3(torch.nn.Module):
    """Stand-in carrying the class name the lever keys on and an encoder of causal convs."""

    def __init__(self, convs):
        super().__init__()
        self.encoder = torch.nn.Sequential(*convs)


def _convs():
    torch.manual_seed(0)
    return [
        CausalConv(3, 8, (3, 3, 3), stride = 1, spatial_padding = 1, temporal_padding = 2),
        CausalConv(8, 8, (3, 3, 3), stride = (1, 2, 2), spatial_padding = 1, temporal_padding = 2),
        CausalConv(8, 8, (2, 1, 1), stride = (2, 1, 1), spatial_padding = 0, temporal_padding = 1),
        CausalConv(8, 4, (1, 1, 1), stride = 1, spatial_padding = 0, temporal_padding = 0),
        CausalConv(4, 4, (1, 3, 3), stride = 1, spatial_padding = 1, temporal_padding = 0),  # kt 1: stock
    ]


@pytest.fixture
def rocm(monkeypatch):
    monkeypatch.delenv(H.H3_ROCM_SINGLE_FRAME_ENV, raising = False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(H, "is_rocm_torch", lambda _torch: True)


@pytest.mark.parametrize("idx", range(5))
def test_lowered_conv_matches_stock_in_float64(idx):
    conv = _convs()[idx].double()
    x = torch.randn(1, conv.in_channels, 1, 18, 22, dtype = torch.float64)
    want = conv(x)
    got = H._single_frame_conv_forward(conv)(x)
    assert got.shape == want.shape
    torch.testing.assert_close(got, want, rtol = 1e-12, atol = 1e-12)


def test_multi_frame_input_runs_stock():
    conv = _convs()[0].double()
    x = torch.randn(1, 3, 5, 12, 12, dtype = torch.float64)
    assert torch.equal(H._single_frame_conv_forward(conv)(x), conv(x))


def test_install_wraps_every_encoder_conv_on_rocm(rocm):
    vae = AutoencoderKLMiniMaxH3(_convs()).double()
    x = torch.randn(1, 3, 1, 16, 16, dtype = torch.float64)
    want = vae.encoder(x)
    assert H._install_rocm_single_frame_encoder(vae) == 5
    torch.testing.assert_close(vae.encoder(x), want, rtol = 1e-12, atol = 1e-12)
    assert H._install_rocm_single_frame_encoder(vae) == 0  # already wrapped


def test_apply_reports_the_lever_on_rocm(rocm):
    vae = AutoencoderKLMiniMaxH3(_convs())
    assert H.apply_h3_vae_speedups(vae, speed_mode = "off", consumer_gpu = True) == (
        H.LEVER_ROCM_SINGLE_FRAME_ENCODER,
    )


def test_kill_switch(rocm, monkeypatch):
    monkeypatch.setenv(H.H3_ROCM_SINGLE_FRAME_ENV, "0")
    assert H._install_rocm_single_frame_encoder(AutoencoderKLMiniMaxH3(_convs())) == 0


def test_not_engaged_off_rocm(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(H, "is_rocm_torch", lambda _torch: False)
    assert H._install_rocm_single_frame_encoder(AutoencoderKLMiniMaxH3(_convs())) == 0


def test_other_vae_classes_untouched(rocm):
    class OtherVAE(AutoencoderKLMiniMaxH3):
        pass

    assert H._install_rocm_single_frame_encoder(OtherVAE(_convs())) == 0
