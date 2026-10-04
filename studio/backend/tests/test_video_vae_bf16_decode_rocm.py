# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""ROCm RDNA3+ decodes the fp32-pinned Wan VAE under bf16 autocast; every other device keeps the fp32 decode."""

from __future__ import annotations

import contextlib
import types

import pytest
import torch

from core.inference import diffusion_device as dd
from core.inference.diffusion_device import DiffusionDeviceTarget, install_rocm_vae_bf16_decode


def _target(backend = "rocm", vendor = "amd", device = "cuda"):
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

    def decode(self, z, return_dict = False):
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
        torch.cuda, "get_device_properties", lambda i: types.SimpleNamespace(gcnArchName = state["arch"])
    )
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda *a, **k: state["bf16"])
    monkeypatch.delenv(dd.VAE_BF16_DECODE_ENV, raising = False)
    return state


def _pipe(dtype = torch.float32):
    return types.SimpleNamespace(vae = _FakeVAE(dtype))


@pytest.mark.parametrize("arch", ["gfx1151", "gfx1100", "gfx1201", "gfx1151:sramecc-:xnack-"])
def test_rocm_rdna3_plus_decodes_under_bf16_autocast_and_returns_fp32(gpu, arch):
    gpu["arch"] = arch
    pipe = _pipe()
    original = pipe.vae.decode
    assert install_rocm_vae_bf16_decode(pipe, _target()) is True
    out = pipe.vae.decode(torch.ones(3), return_dict = False)
    assert pipe.vae.calls == [("cuda", torch.bfloat16)]
    assert out[0].dtype == torch.float32 and torch.equal(out[0], torch.full((3,), 2.0))
    assert pipe.vae.dtype is torch.float32 and pipe.vae.weight.dtype == torch.float32  # weights untouched
    assert pipe.vae.decode.__wrapped__ == original


def test_nvidia_is_untouched(gpu):
    pipe = _pipe()
    decode = pipe.vae.decode
    assert install_rocm_vae_bf16_decode(pipe, _target(backend = "cuda", vendor = "nvidia")) is False
    assert pipe.vae.decode == decode
    assert pipe.vae.decode(torch.ones(1))[0].dtype == torch.float32 and pipe.vae.calls == [None]


@pytest.mark.parametrize("arch", ["gfx1030", "gfx90a", "gfx942", ""])
def test_other_rocm_arches_keep_fp32(gpu, arch):
    gpu["arch"] = arch
    pipe = _pipe()
    decode = pipe.vae.decode
    assert install_rocm_vae_bf16_decode(pipe, _target()) is False
    assert pipe.vae.decode == decode


@pytest.mark.parametrize("device", ["mps", "cpu", "xpu"])
def test_non_cuda_devices_untouched(gpu, device):
    pipe = _pipe()
    assert install_rocm_vae_bf16_decode(pipe, _target(device = device)) is False


def test_kill_switch(gpu, monkeypatch):
    monkeypatch.setenv(dd.VAE_BF16_DECODE_ENV, "0")
    pipe = _pipe()
    assert install_rocm_vae_bf16_decode(pipe, _target()) is False


def test_force_allows_nvidia_with_bf16(gpu, monkeypatch):
    monkeypatch.setenv(dd.VAE_BF16_DECODE_ENV, "1")
    assert install_rocm_vae_bf16_decode(_pipe(), _target(backend = "cuda", vendor = "nvidia")) is True
    gpu["bf16"] = False
    assert install_rocm_vae_bf16_decode(_pipe(), _target(backend = "cuda", vendor = "nvidia")) is False


def test_only_fp32_vaes_and_only_once(gpu):
    assert install_rocm_vae_bf16_decode(_pipe(torch.bfloat16), _target()) is False
    pipe = _pipe()
    assert install_rocm_vae_bf16_decode(pipe, _target()) is True
    assert install_rocm_vae_bf16_decode(pipe, _target()) is False  # no double wrap
    pipe.vae.decode(torch.ones(1))
    assert pipe.vae.calls == [("cuda", torch.bfloat16)]


def test_decoder_output_object_is_widened(gpu):
    class Out:
        def __init__(self, sample):
            self.sample = sample

    vae = _FakeVAE()
    vae.decode = lambda z, return_dict = True: Out(z.to(torch.bfloat16))
    pipe = types.SimpleNamespace(vae = vae)
    assert install_rocm_vae_bf16_decode(pipe, _target()) is True
    assert pipe.vae.decode(torch.ones(2)).sample.dtype == torch.float32


def test_video_backend_gates_on_fp32_families_and_speed_tier():
    from pathlib import Path

    src = (Path(dd.__file__).with_name("video.py")).read_text(encoding = "utf-8")
    i = src.index("install_rocm_vae_bf16_decode(pipe, target")
    window = src[i - 400 : i]
    assert 'getattr(fam, "vae_force_fp32", False)' in window and "effective_speed != SPEED_OFF" in window
