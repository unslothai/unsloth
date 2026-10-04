# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""ROCm RDNA3+ decodes the fp32-pinned Wan VAE in bf16 (weights cast by default, autocast on request); every other
device keeps the fp32 decode."""

from __future__ import annotations

import contextlib
import types

import pytest
import torch

from core.inference import diffusion_device as dd
from core.inference.diffusion_device import DiffusionDeviceTarget, install_rocm_vae_bf16_decode


def _target(
    backend = "rocm",
    vendor = "amd",
    device = "cuda",
):
    return DiffusionDeviceTarget(
        device = device,
        dtype = torch.bfloat16,
        backend = backend,
        vendor = vendor,
        supports_model_cpu_offload = True,
        supports_default_torch_compile = backend != "rocm",
        supports_pinned_transfer = True,
    )


class _FakeVAE:
    def __init__(self, dtype = torch.float32):
        self.dtype = dtype
        self.calls = []
        self.weight = torch.ones(2, dtype = dtype)

    def decode(
        self,
        z,
        return_dict = False,
    ):
        self.calls.append(_AUTOCAST[-1] if _AUTOCAST else None)
        dtype = torch.bfloat16 if _AUTOCAST else torch.float32
        return (z.to(dtype) * 2,)


_AUTOCAST: list = []


@pytest.fixture
def gpu(monkeypatch):
    state = {"arch": "gfx1151", "bf16": True}

    @contextlib.contextmanager
    def fake_autocast(device_type, dtype):
        _AUTOCAST.append((device_type, dtype))
        try:
            yield
        finally:
            _AUTOCAST.pop()

    monkeypatch.setattr(torch, "autocast", fake_autocast)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda i: types.SimpleNamespace(gcnArchName = state["arch"]),
    )
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda *a, **k: state["bf16"])
    monkeypatch.delenv(dd.VAE_BF16_DECODE_ENV, raising = False)
    return state


def _pipe(dtype = torch.float32):
    return types.SimpleNamespace(vae = _FakeVAE(dtype))


@pytest.mark.parametrize("arch", ["gfx1151", "gfx1100", "gfx1201", "gfx1151:sramecc-:xnack-"])
def test_rocm_rdna3_plus_decodes_under_bf16_autocast_and_returns_fp32(gpu, arch, monkeypatch):
    monkeypatch.setenv(dd.VAE_BF16_DECODE_ENV, "autocast")
    gpu["arch"] = arch
    pipe = _pipe()
    original = pipe.vae.decode
    assert install_rocm_vae_bf16_decode(pipe, _target()) == "autocast"
    out = pipe.vae.decode(torch.ones(3), return_dict = False)
    assert pipe.vae.calls == [("cuda", torch.bfloat16)]
    assert out[0].dtype == torch.float32 and torch.equal(out[0], torch.full((3,), 2.0))
    assert (
        pipe.vae.dtype is torch.float32 and pipe.vae.weight.dtype == torch.float32
    )  # weights untouched
    assert pipe.vae.decode.__wrapped__ == original


def test_nvidia_is_untouched(gpu):
    pipe = _pipe()
    decode = pipe.vae.decode
    assert install_rocm_vae_bf16_decode(pipe, _target(backend = "cuda", vendor = "nvidia")) is None
    assert pipe.vae.decode == decode
    assert pipe.vae.decode(torch.ones(1))[0].dtype == torch.float32 and pipe.vae.calls == [None]


@pytest.mark.parametrize("arch", ["gfx1030", "gfx90a", "gfx942", ""])
def test_other_rocm_arches_keep_fp32(gpu, arch):
    gpu["arch"] = arch
    pipe = _pipe()
    decode = pipe.vae.decode
    assert install_rocm_vae_bf16_decode(pipe, _target()) is None
    assert pipe.vae.decode == decode


@pytest.mark.parametrize("device", ["mps", "cpu", "xpu"])
def test_non_cuda_devices_untouched(gpu, device):
    pipe = _pipe()
    assert install_rocm_vae_bf16_decode(pipe, _target(device = device)) is None


def test_kill_switch(gpu, monkeypatch):
    monkeypatch.setenv(dd.VAE_BF16_DECODE_ENV, "0")
    pipe = _pipe()
    assert install_rocm_vae_bf16_decode(pipe, _target()) is None


def test_force_allows_nvidia_with_bf16(gpu, monkeypatch):
    monkeypatch.setenv(dd.VAE_BF16_DECODE_ENV, "1")
    assert (
        install_rocm_vae_bf16_decode(_pipe(), _target(backend = "cuda", vendor = "nvidia"))
        == "autocast"
    )  # no decoder module to cast: autocast
    gpu["bf16"] = False
    assert install_rocm_vae_bf16_decode(_pipe(), _target(backend = "cuda", vendor = "nvidia")) is None


def test_only_fp32_vaes_and_only_once(gpu):
    assert install_rocm_vae_bf16_decode(_pipe(torch.bfloat16), _target()) is None
    pipe = _pipe()
    assert (
        install_rocm_vae_bf16_decode(pipe, _target()) == "autocast"
    )  # no decoder module to cast: autocast
    assert install_rocm_vae_bf16_decode(pipe, _target()) is None  # no double wrap
    pipe.vae.decode(torch.ones(1))
    assert pipe.vae.calls == [("cuda", torch.bfloat16)]


def test_decoder_output_object_is_widened(gpu):
    class Out:
        def __init__(self, sample):
            self.sample = sample

    vae = _FakeVAE()
    vae.decode = lambda z, return_dict = True: Out(z.to(torch.bfloat16))
    pipe = types.SimpleNamespace(vae = vae)
    assert (
        install_rocm_vae_bf16_decode(pipe, _target()) == "autocast"
    )  # no decoder module to cast: autocast
    assert pipe.vae.decode(torch.ones(2)).sample.dtype == torch.float32


def test_video_backend_gates_on_fp32_families_and_speed_tier():
    from pathlib import Path

    src = (Path(dd.__file__).with_name("video.py")).read_text(encoding = "utf-8")
    i = src.index("install_rocm_vae_bf16_decode(pipe, target")
    window = src[i - 400 : i]
    assert (
        'getattr(fam, "vae_force_fp32", False)' in window
        and "effective_speed != SPEED_OFF" in window
    )


@pytest.fixture
def tiny_wan_vae():
    # Built before the ``gpu`` fixture fakes the device properties diffusers reads on import.
    diffusers = pytest.importorskip("diffusers")
    torch.manual_seed(0)
    return diffusers.AutoencoderKLWan(
        base_dim = 8, z_dim = 4, dim_mult = [1, 2], num_res_blocks = 1, temperal_downsample = [True]
    ).eval()


@pytest.mark.parametrize("gate", [None, "auto", "weights", "1"])
def test_default_casts_only_the_decode_half_and_matches_fp32(tiny_wan_vae, gpu, monkeypatch, gate):
    if gate is not None:
        monkeypatch.setenv(dd.VAE_BF16_DECODE_ENV, gate)
    vae = tiny_wan_vae
    z = torch.randn(1, 4, 2, 4, 4)
    with torch.no_grad():
        ref = vae.decode(z, return_dict = False)[0]
    pipe = types.SimpleNamespace(vae = vae)
    assert install_rocm_vae_bf16_decode(pipe, _target()) == "weights"
    assert vae._unsloth_bf16_decode_mode == "weights"
    assert all(p.dtype == torch.bfloat16 for p in vae.decoder.parameters())
    assert all(p.dtype == torch.bfloat16 for p in vae.post_quant_conv.parameters())
    assert all(p.dtype == torch.float32 for p in vae.encoder.parameters())
    assert (
        vae.dtype == torch.float32
    )  # pipelines keep handing it fp32 latents; the encode stays fp32
    with torch.no_grad():
        out = vae.decode(z, return_dict = False)[0]
        assert vae.encode(torch.randn(1, 3, 1, 8, 8)).latent_dist.mean.dtype == torch.float32
    assert out.dtype == torch.float32 and out.shape == ref.shape
    assert (out - ref).abs().max().item() < 0.05


def test_decoder_reached_without_vae_decode_still_gets_bf16_inputs(tiny_wan_vae, gpu):
    vae = tiny_wan_vae
    pipe = types.SimpleNamespace(vae = vae)
    assert install_rocm_vae_bf16_decode(pipe, _target()) == "weights"
    with torch.no_grad():
        x = vae.post_quant_conv(
            torch.randn(1, 4, 1, 4, 4)
        )  # fp32 input, bf16 weights: the pre-hook casts
    assert x.dtype == torch.bfloat16


def test_weights_mode_with_tiling_matches_fp32(tiny_wan_vae, gpu):
    vae = tiny_wan_vae
    vae.enable_tiling(
        tile_sample_min_height = 16,
        tile_sample_min_width = 16,
        tile_sample_stride_height = 8,
        tile_sample_stride_width = 8,
    )
    z = torch.randn(1, 4, 2, 4, 4)
    with torch.no_grad():
        ref = vae.decode(z, return_dict = False)[0]
    assert install_rocm_vae_bf16_decode(types.SimpleNamespace(vae = vae), _target()) == "weights"
    with torch.no_grad():
        out = vae.decode(z, return_dict = False)[0]
    assert out.dtype == torch.float32 and (out - ref).abs().max().item() < 0.05


def test_autocast_mode_keeps_every_weight_fp32(tiny_wan_vae, gpu, monkeypatch):
    monkeypatch.setenv(dd.VAE_BF16_DECODE_ENV, "autocast")
    vae = tiny_wan_vae
    assert install_rocm_vae_bf16_decode(types.SimpleNamespace(vae = vae), _target()) == "autocast"
    assert all(p.dtype == torch.float32 for p in vae.parameters())


def test_nvidia_real_vae_untouched(tiny_wan_vae, gpu):
    vae = tiny_wan_vae
    assert (
        install_rocm_vae_bf16_decode(
            types.SimpleNamespace(vae = vae), _target(backend = "cuda", vendor = "nvidia")
        )
        is None
    )
    assert all(p.dtype == torch.float32 for p in vae.parameters())


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason = "needs a real CUDA device for the NVIDIA fp16 decode"
)
def test_forced_bf16_leaves_nvidia_fp16_decode_alone(monkeypatch):
    # Stacked, the fp16 wrapper's non-finite fallback recasts the decoder to fp32 under bf16 input hooks: crash.
    diffusers = pytest.importorskip("diffusers")
    from core.inference import diffusion_speed as ds

    monkeypatch.setenv(dd.VAE_BF16_DECODE_ENV, "1")
    torch.manual_seed(0)
    vae = (
        diffusers.AutoencoderKLWan(
            base_dim = 8, z_dim = 4, dim_mult = [1, 2], num_res_blocks = 1, temperal_downsample = [True]
        )
        .eval()
        .cuda()
    )
    pipe = types.SimpleNamespace(vae = vae)
    target = dd.diffusion_device_target_from_torch_device("cuda", torch.bfloat16)
    if target.backend != "cuda" or not ds._video_vae_half_decode(
        pipe, target, types.SimpleNamespace(vae_force_fp32 = True), None
    ):
        pytest.skip("this device has no NVIDIA fp16 VAE decode")
    assert install_rocm_vae_bf16_decode(pipe, target) is None
    with torch.no_grad():
        vae.decode(torch.full((1, 4, 2, 4, 4), float("nan"), device = "cuda"), return_dict = False)
        out = vae.decode(torch.randn(1, 4, 2, 4, 4, device = "cuda"), return_dict = False)[0]
    assert out.dtype == torch.float32 and bool(torch.isfinite(out).all())
