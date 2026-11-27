# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MPS single-frame H3 encoder lowering (causal Conv3d -> Conv2d of the last tap). CPU only; MPS is mocked."""

from __future__ import annotations

import pytest
import torch

from core.inference import video_minimax_h3_vae as H

H3 = pytest.importorskip("diffusers.models.autoencoders.autoencoder_kl_minimax_h3")
CausalConv = H3.MiniMaxH3VideoCausalConv3d


class AutoencoderKLMiniMaxH3(torch.nn.Module):
    def __init__(self, convs):
        super().__init__()
        self.encoder = torch.nn.Sequential(*convs)


def _convs():
    torch.manual_seed(0)
    return [
        CausalConv(3, 8, (3, 3, 3), stride = 1, spatial_padding = 1, temporal_padding = 2),
        CausalConv(8, 8, (3, 3, 3), stride = (1, 2, 2), spatial_padding = 0, temporal_padding = 2),
        CausalConv(8, 4, (1, 1, 1), stride = 1, spatial_padding = 0, temporal_padding = 0),
    ]


@pytest.fixture
def mps(monkeypatch):
    monkeypatch.delenv(H.H3_MPS_SINGLE_FRAME_ENV, raising = False)
    monkeypatch.delenv(H.H3_ROCM_SINGLE_FRAME_ENV, raising = False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)


@pytest.mark.parametrize("speed_mode", ["default", "eager", "max"])
def test_apply_engages_the_lever_on_mps(mps, speed_mode):
    vae = AutoencoderKLMiniMaxH3(_convs()).double()
    x = torch.randn(1, 3, 1, 16, 16, dtype = torch.float64)
    want = vae.encoder(x)
    assert H.apply_h3_vae_speedups(vae, speed_mode = speed_mode, consumer_gpu = True) == (
        H.LEVER_MPS_SINGLE_FRAME_ENCODER,
    )
    assert vae._unsloth_h3_vae_levers == (H.LEVER_MPS_SINGLE_FRAME_ENCODER,)
    assert H.h3_vae_lever_calls(vae) == {H.LEVER_MPS_SINGLE_FRAME_ENCODER: 0}
    torch.testing.assert_close(vae.encoder(x), want, rtol = 1e-12, atol = 1e-12)
    # the two kt 3 convs take the Conv2d, the 1x1x1 one stays stock
    assert H.h3_vae_lever_calls(vae) == {H.LEVER_MPS_SINGLE_FRAME_ENCODER: 2}
    assert [c._unsloth_rocm_single_frame_calls for c in vae.encoder] == [1, 1, 0]
    vae.encoder(torch.randn(1, 3, 4, 16, 16, dtype = torch.float64))
    assert H.h3_vae_lever_calls(vae) == {H.LEVER_MPS_SINGLE_FRAME_ENCODER: 2}


def test_speed_mode_off_keeps_stock_on_mps(mps):
    vae = AutoencoderKLMiniMaxH3(_convs())
    assert H.apply_h3_vae_speedups(vae, speed_mode = "off", consumer_gpu = True) == ()
    assert all("forward" not in c.__dict__ for c in vae.encoder)
    vae.encoder(torch.randn(1, 3, 1, 16, 16))
    assert H.h3_vae_lever_calls(vae) == {}


def test_mps_kill_switch(mps, monkeypatch):
    monkeypatch.setenv(H.H3_MPS_SINGLE_FRAME_ENV, "0")
    vae = AutoencoderKLMiniMaxH3(_convs())
    assert H.apply_h3_vae_speedups(vae, speed_mode = "default", consumer_gpu = True) == ()
    assert all("forward" not in c.__dict__ for c in vae.encoder)


def test_rocm_kill_switch_does_not_turn_mps_off(mps, monkeypatch):
    monkeypatch.setenv(H.H3_ROCM_SINGLE_FRAME_ENV, "0")
    assert H._install_rocm_single_frame_encoder(AutoencoderKLMiniMaxH3(_convs())) == 3


def test_cpu_only_host_untouched(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    vae = AutoencoderKLMiniMaxH3(_convs())
    assert H.apply_h3_vae_speedups(vae, speed_mode = "default", consumer_gpu = True) == ()


def test_nvidia_cuda_host_does_not_take_the_mps_lever(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(H, "is_rocm_torch", lambda _torch: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert H._install_rocm_single_frame_encoder(AutoencoderKLMiniMaxH3(_convs())) == 0


def test_rocm_still_reports_its_own_lever(monkeypatch):
    monkeypatch.delenv(H.H3_ROCM_SINGLE_FRAME_ENV, raising = False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(H, "is_rocm_torch", lambda _torch: True)
    vae = AutoencoderKLMiniMaxH3(_convs())
    assert H.apply_h3_vae_speedups(vae, speed_mode = "default", consumer_gpu = True) == (
        H.LEVER_ROCM_SINGLE_FRAME_ENCODER,
    )
