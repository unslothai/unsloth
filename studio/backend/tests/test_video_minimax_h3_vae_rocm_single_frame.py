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
        CausalConv(4, 4, (1, 3, 3), stride = 1, spatial_padding = 1, temporal_padding = 0),
    ]


@pytest.fixture
def rocm(monkeypatch):
    monkeypatch.delenv(H.H3_ROCM_SINGLE_FRAME_ENV, raising = False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(H, "is_rocm_torch", lambda _torch: True)


_LOWERED = (True, True, True, False, False)


@pytest.mark.parametrize("idx", range(5))
def test_lowered_conv_matches_stock_in_float64(idx, monkeypatch):
    conv = _convs()[idx].double()
    x = torch.randn(1, conv.in_channels, 1, 18, 22, dtype = torch.float64)
    want = conv(x)
    conv2d_calls = []
    real_conv2d = torch.nn.functional.conv2d
    monkeypatch.setattr(
        torch.nn.functional,
        "conv2d",
        lambda *a, **k: conv2d_calls.append(1) or real_conv2d(*a, **k),
    )
    calls = {}
    got = H._single_frame_conv_forward(conv, calls)(x)
    assert got.shape == want.shape
    torch.testing.assert_close(got, want, rtol = 1e-12, atol = 1e-12)
    lowered = _LOWERED[idx]
    assert len(conv2d_calls) == int(lowered)
    assert conv._unsloth_rocm_single_frame_calls == int(lowered)
    assert calls.get(H.LEVER_ROCM_SINGLE_FRAME_ENCODER, 0) == int(lowered)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(spatial_padding = 1, temporal_padding = 0),
        dict(spatial_padding = 1, temporal_padding = 1),
        dict(spatial_padding = 1, temporal_padding = 2, dilation = (2, 1, 1)),
    ],
)
def test_non_causal_or_dilated_conv_runs_stock(kwargs):
    torch.manual_seed(0)
    dilation = kwargs.pop("dilation", None)
    conv = CausalConv(3, 4, (3, 3, 3), stride = 1, **kwargs).double()
    if dilation is not None:
        conv.dilation = dilation
    calls = {}
    x = torch.randn(1, 3, 1, 10, 10, dtype = torch.float64)
    fwd = H._single_frame_conv_forward(conv, calls)
    try:
        want = conv(x)
    except RuntimeError:  # stock rejects the shape: the wrapper must reject it the same way
        with pytest.raises(RuntimeError):
            fwd(x)
    else:
        assert torch.equal(fwd(x), want)
    assert conv._unsloth_rocm_single_frame_calls == 0
    assert calls == {}


def test_multi_frame_input_runs_stock():
    conv = _convs()[0].double()
    x = torch.randn(1, 3, 5, 12, 12, dtype = torch.float64)
    calls = {}
    assert torch.equal(H._single_frame_conv_forward(conv, calls)(x), conv(x))
    assert conv._unsloth_rocm_single_frame_calls == 0 and calls == {}


def test_install_wraps_every_encoder_conv_on_rocm(rocm):
    vae = AutoencoderKLMiniMaxH3(_convs()).double()
    x = torch.randn(1, 3, 1, 16, 16, dtype = torch.float64)
    want = vae.encoder(x)
    assert H._install_rocm_single_frame_encoder(vae) == 5
    assert H.h3_vae_lever_calls(vae) == {H.LEVER_ROCM_SINGLE_FRAME_ENCODER: 0}
    torch.testing.assert_close(vae.encoder(x), want, rtol = 1e-12, atol = 1e-12)
    assert H.h3_vae_lever_calls(vae) == {H.LEVER_ROCM_SINGLE_FRAME_ENCODER: 3}
    assert [c._unsloth_rocm_single_frame_calls for c in vae.encoder] == [1, 1, 1, 0, 0]
    vae.encoder(torch.randn(1, 3, 4, 16, 16, dtype = torch.float64))
    assert H.h3_vae_lever_calls(vae) == {H.LEVER_ROCM_SINGLE_FRAME_ENCODER: 3}
    vae.encoder(torch.randn(1, 3, 2, 16, 16, dtype = torch.float64))
    # T 2 -> 1 after the temporal stride 2 conv: the later convs see T = 1, but are kt 1 and stay stock
    assert H.h3_vae_lever_calls(vae) == {H.LEVER_ROCM_SINGLE_FRAME_ENCODER: 3}
    assert H._install_rocm_single_frame_encoder(vae) == 0


def test_t1_conv_inside_a_multi_frame_encode_takes_the_lowering(rocm):
    torch.manual_seed(0)
    convs = [
        CausalConv(3, 4, (2, 1, 1), stride = (2, 1, 1), spatial_padding = 0, temporal_padding = 1),
        CausalConv(4, 4, (3, 3, 3), stride = 1, spatial_padding = 1, temporal_padding = 2),
    ]
    vae = AutoencoderKLMiniMaxH3(convs).double()
    x = torch.randn(1, 3, 2, 8, 8, dtype = torch.float64)
    want = vae.encoder(x)
    assert H._install_rocm_single_frame_encoder(vae) == 2
    torch.testing.assert_close(vae.encoder(x), want, rtol = 1e-12, atol = 1e-12)
    assert [c._unsloth_rocm_single_frame_calls for c in vae.encoder] == [0, 1]


@pytest.mark.parametrize("speed_mode", ["default", "eager", "max"])
def test_apply_reports_the_lever_on_rocm(rocm, speed_mode):
    vae = AutoencoderKLMiniMaxH3(_convs())
    assert H.apply_h3_vae_speedups(vae, speed_mode = speed_mode, consumer_gpu = True) == (
        H.LEVER_ROCM_SINGLE_FRAME_ENCODER,
    )
    assert vae._unsloth_h3_vae_levers == (H.LEVER_ROCM_SINGLE_FRAME_ENCODER,)
    vae.encoder(torch.randn(1, 3, 1, 16, 16))
    assert H.h3_vae_lever_calls(vae) == {H.LEVER_ROCM_SINGLE_FRAME_ENCODER: 3}


def test_speed_mode_off_keeps_stock_like_the_cuda_fused_encoder(rocm):
    assert H.LEVER_FUSED_ENCODER not in H.plan_h3_vae_levers("off", workflow = "fl2va")
    vae = AutoencoderKLMiniMaxH3(_convs())
    assert H.apply_h3_vae_speedups(vae, speed_mode = "off", consumer_gpu = True) == ()
    assert all("forward" not in c.__dict__ for c in vae.encoder)
    vae.encoder(torch.randn(1, 3, 1, 16, 16))
    assert H.h3_vae_lever_calls(vae) == {}


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
