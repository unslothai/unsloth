# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""channels_last on 3D-conv VAEs (Wan, Qwen-Image, HunyuanVideo-1.5, LTX-2, MiniMax-H3): relaid per weight rank under
the fused VAE passes, skipped (and said so) without them, never left half converted; the Wan half decode kill switch;
the A14B untiled-decode estimate."""

from __future__ import annotations

import logging
import types

import pytest

torch = pytest.importorskip("torch")

from core.inference import diffusion_speed as ds_mod  # noqa: E402
from core.inference import video_vae_untiled as U  # noqa: E402


class _Fake3dVAE(torch.nn.Module):
    """A Conv2d registered before the first Conv3d, as in the Wan / Qwen-Image resample blocks: Module.to(channels_last)
    converted it and then raised on the Conv3d."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3), torch.nn.Conv3d(4, 4, 3))
        self.post_quant_conv = torch.nn.Conv3d(4, 4, 1)
        self.decoder = torch.nn.Sequential(
            torch.nn.Conv2d(4, 4, 3), torch.nn.Conv3d(4, 4, 3), torch.nn.GroupNorm(2, 4)
        )


def _layouts(vae):
    return {name: (p.stride(), p.data_ptr()) for name, p in vae.named_parameters()}


def test_fused_3d_vae_gets_channels_last_decode_weights_per_rank_without_a_failure_warning(caplog):
    vae = _Fake3dVAE()
    encoder_before = {n: p.stride() for n, p in vae.encoder.named_parameters()}
    with caplog.at_level(logging.INFO, logger = "t3d"):
        assert ds_mod._vae_channels_last(
            types.SimpleNamespace(vae = vae), logging.getLogger("t3d"), fused = True
        )
    assert vae.decoder[0].weight.is_contiguous(memory_format = torch.channels_last)
    assert not vae.decoder[0].weight.is_contiguous()
    assert vae.decoder[1].weight.is_contiguous(memory_format = torch.channels_last_3d)
    assert not vae.decoder[1].weight.is_contiguous()
    assert vae.post_quant_conv.weight.is_contiguous(memory_format = torch.channels_last_3d)
    # the encoder keeps its layout (channels_last encoder weights encoded slower)
    assert {n: p.stride() for n, p in vae.encoder.named_parameters()} == encoder_before
    assert vae.decoder[2].weight.is_contiguous()  # GroupNorm untouched
    assert "failed" not in caplog.text
    assert "channels_last(_3d) decode weights" in caplog.text


def test_unfused_3d_vae_is_skipped_untouched_and_logged(caplog):
    vae = _Fake3dVAE()
    before = _layouts(vae)
    with caplog.at_level(logging.INFO, logger = "t3d"):
        assert not ds_mod._vae_channels_last(
            types.SimpleNamespace(vae = vae), logging.getLogger("t3d")
        )
    assert _layouts(vae) == before  # nothing half converted
    assert "channels_last skipped" in caplog.text and "failed" not in caplog.text


def test_env_switch_keeps_the_3d_vae_contiguous(monkeypatch):
    monkeypatch.setenv(ds_mod.VAE_CHANNELS_LAST_3D_ENV, "0")
    vae = _Fake3dVAE()
    before = _layouts(vae)
    assert not ds_mod._vae_channels_last(types.SimpleNamespace(vae = vae), None, fused = True)
    assert _layouts(vae) == before


def test_2d_vae_still_takes_the_module_wide_channels_last():
    vae = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3), torch.nn.Conv2d(4, 4, 3))
    assert ds_mod._vae_channels_last(types.SimpleNamespace(vae = vae), None)
    assert all(m.weight.is_contiguous(memory_format = torch.channels_last) for m in vae)


@pytest.mark.parametrize(
    "name", ["AutoencoderKLWan", "AutoencoderKLQwenImage", "AutoencoderKLHunyuanVideo15"]
)
def test_real_3d_vae_classes_take_the_3d_path(name):
    diffusers = pytest.importorskip("diffusers")
    cls = getattr(diffusers, name, None)
    if cls is None:
        pytest.skip(f"{name} not in this diffusers")
    kw = {
        "AutoencoderKLWan": dict(
            base_dim = 8, z_dim = 4, dim_mult = [1, 1], num_res_blocks = 1, temperal_downsample = [False]
        ),
        "AutoencoderKLQwenImage": dict(
            base_dim = 8, z_dim = 4, dim_mult = [1, 1], num_res_blocks = 1, temperal_downsample = [False]
        ),
        "AutoencoderKLHunyuanVideo15": dict(
            block_out_channels = (8, 8),
            latent_channels = 4,
            layers_per_block = 1,
            spatial_compression_ratio = 2,
            temporal_compression_ratio = 1,
        ),
    }[name]
    try:
        vae = cls(**kw)
    except Exception as exc:  # noqa: BLE001 - constructor signature drift across diffusers releases
        pytest.skip(f"cannot build a tiny {name}: {exc}")
    assert ds_mod._has_conv3d(vae)
    assert ds_mod._vae_channels_last(types.SimpleNamespace(vae = vae), None, fused = True)
    convs = [m for m in vae.decoder.modules() if isinstance(m, torch.nn.Conv3d)]
    assert convs and all(
        m.weight.is_contiguous(memory_format = torch.channels_last_3d) for m in convs
    )
    encoder = [m for m in vae.encoder.modules() if isinstance(m, torch.nn.Conv3d)]
    assert all(m.weight.is_contiguous() for m in encoder)


class _FakeWanVAE(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = torch.nn.Conv3d(3, 4, 3)
        self.post_quant_conv = torch.nn.Conv3d(4, 4, 1)
        self.decoder = torch.nn.Sequential(torch.nn.Conv3d(4, 4, 3))

    def decode(
        self,
        z,
        return_dict = True,
    ):
        return (z,)


def test_video_vae_half_kill_switch_keeps_fp32(monkeypatch):
    monkeypatch.setattr(ds_mod, "_fp16_compile_capable", lambda target: True)
    target = types.SimpleNamespace(
        device = "cuda", backend = "cuda", dtype = torch.bfloat16, ordinal = None
    )
    family = types.SimpleNamespace(vae_force_fp32 = True)
    monkeypatch.setenv(ds_mod.VIDEO_VAE_HALF_ENV, "0")
    pipe = types.SimpleNamespace(vae = _FakeWanVAE())
    assert ds_mod._video_vae_half_decode(pipe, target, family, None) is False
    assert (
        all(p.dtype == torch.float32 for p in pipe.vae.parameters())
        and "decode" not in pipe.vae.__dict__
    )
    monkeypatch.delenv(ds_mod.VIDEO_VAE_HALF_ENV)
    assert ds_mod._video_vae_half_decode(pipe, target, family, None) is True
    assert all(p.dtype == torch.float16 for p in pipe.vae.decoder.parameters())


A14B = "wan2.2-t2v-a14b"


def test_a14b_untiled_estimate_covers_the_measured_peak():
    # B200, Wan2.1 VAE, fp16 + fused + channels_last_3d, untiled: allocator peak over the pre-decode baseline (MiB).
    measured = {
        (21, 90, 160): 8989.3,
        (21, 60, 104): 3891.4,
        (11, 90, 160): 8764.9,
        (31, 90, 160): 9195.6,
        (41, 60, 104): 4078.1,
    }
    for (t, h, w), mib in measured.items():
        need = U.untiled_decode_bytes(A14B, (1, 16, t, h, w))
        assert need is not None and mib * 2**20 <= need <= 1.25 * mib * 2**20
    # fp32 peaked at 17960 MiB at 1280x720x81: the itemsize scaling still covers it
    assert U.untiled_decode_bytes(A14B, (1, 16, 21, 90, 160), itemsize = 4) >= 17960.5 * 2**20


def test_a14b_resident_vae_decodes_untiled_when_it_fits(monkeypatch):
    class _Tiled:
        def __init__(self) -> None:
            self.decoder = torch.nn.Conv3d(1, 1, 1).to(torch.float16)
            self.use_tiling = True
            self.calls: list = []

        def decode(
            self,
            z,
            return_dict = False,
        ):
            self.calls.append(self.use_tiling)
            return ("tiled" if self.use_tiling else "untiled",)

    vae = _Tiled()
    monkeypatch.setattr(U, "_free_bytes", lambda device: 100 * 2**30)
    assert U.install_untiled_decode(types.SimpleNamespace(vae = vae), A14B)
    assert vae.decode(torch.zeros(1, 16, 21, 90, 160)) == ("untiled",)
    monkeypatch.setattr(U, "_free_bytes", lambda device: 8 * 2**30)
    assert vae.decode(torch.zeros(1, 16, 21, 90, 160)) == ("tiled",)
