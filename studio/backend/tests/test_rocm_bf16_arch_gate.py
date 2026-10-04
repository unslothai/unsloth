# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""ROCm bf16 is gated by gfx arch: torch.cuda.is_bf16_supported() is True on every HIP build, so
RDNA2-and-older / Vega cards (no bf16 MFMA / WMMA / dot) must resolve float16, while CDNA (gfx908+),
RDNA3+, gfx1151 and every NVIDIA card keep their previous pick."""

from __future__ import annotations

import sys
import types

import pytest

import core.inference.diffusion_device as dd
import core.inference.native_audio as na
import core.training.diffusion_train_common as tc

BF16, FP16, FP32 = "bf16", "fp16", "fp32"


def _fake_torch(*, hip, arch = "", capability = (8, 0)):
    torch = types.ModuleType("torch")
    torch.bfloat16, torch.float16, torch.float32 = BF16, FP16, FP32
    torch.version = types.SimpleNamespace(hip = hip)
    props = types.SimpleNamespace(gcnArchName = arch, major = capability[0], minor = capability[1])

    def _is_bf16_supported(including_emulation = True):
        # torch's real body: any HIP build answers True before looking at the device.
        if hip:
            return True
        return capability[0] >= 8 or including_emulation

    torch.cuda = types.SimpleNamespace(
        is_available = lambda: True,
        is_bf16_supported = _is_bf16_supported,
        get_device_capability = lambda device = None: capability,
        get_device_properties = lambda device = None: props,
        current_device = lambda: 0,
        device_count = lambda: 1,
        mem_get_info = lambda index = None: (0, 0),
        set_device = lambda index: None,
    )
    torch.backends = types.SimpleNamespace(mps = types.SimpleNamespace(is_available = lambda: False))
    return torch


def _install(monkeypatch, torch, *, is_rocm):
    monkeypatch.setitem(sys.modules, "torch", torch)

    class _DT:
        CUDA, XPU, MLX, CPU = "cuda", "xpu", "mlx", "cpu"

    fake_uh = types.ModuleType("utils.hardware")
    fake_uh.DeviceType = _DT
    fake_uh.get_device = lambda: "cuda"
    fake_uh.hardware = types.SimpleNamespace(IS_ROCM = is_rocm)
    monkeypatch.setitem(sys.modules, "utils.hardware", fake_uh)
    monkeypatch.delenv("UNSLOTH_STUDIO_ROCM_BF16", raising = False)


AMD_CASES = [
    ("gfx1030", FP16),  # RX 6800/6900, RDNA2
    ("gfx1031", FP16),  # RX 6700, RDNA2
    ("gfx1010", FP16),  # RX 5700, RDNA1
    ("gfx906:sramecc+:xnack-", FP16),  # MI50 / Radeon VII, suffix stripped
    ("gfx900:xnack-", FP16),  # Vega 10
    ("gfx803", FP16),  # Polaris
    ("gfx908:sramecc+:xnack-", BF16),  # MI100, bf16 MFMA
    ("gfx90a:sramecc+:xnack-", BF16),  # MI200
    ("gfx942:sramecc+:xnack-", BF16),  # MI300
    ("gfx950", BF16),  # MI350
    ("gfx1100", BF16),  # RDNA3
    ("gfx1151", BF16),  # Strix Halo
    ("gfx1201", BF16),  # RDNA4
    ("", BF16),  # no arch reported: torch's answer stands (previous behaviour)
]


@pytest.mark.parametrize("arch,expected", AMD_CASES)
def test_diffusion_rocm_dtype_by_arch(monkeypatch, arch, expected):
    torch = _fake_torch(hip = "6.4", arch = arch, capability = (9, 0))
    _install(monkeypatch, torch, is_rocm = True)
    t = dd.resolve_diffusion_device_target()
    assert (t.backend, t.dtype) == ("rocm", expected)


@pytest.mark.parametrize("arch,expected", AMD_CASES)
def test_native_audio_rocm_dtype_by_arch(monkeypatch, arch, expected):
    torch = _fake_torch(hip = "6.4", arch = arch, capability = (9, 0))
    _install(monkeypatch, torch, is_rocm = True)
    backend = types.SimpleNamespace(device = "cuda")
    assert na.NativeAudioBackend._dtype(backend) == expected


@pytest.mark.parametrize("arch,expected", AMD_CASES)
def test_training_native_bf16_by_arch(monkeypatch, arch, expected):
    torch = _fake_torch(hip = "6.4", arch = arch, capability = (9, 0))
    _install(monkeypatch, torch, is_rocm = True)
    assert tc.native_bf16_supported() is (expected == BF16)


@pytest.mark.parametrize("capability,expected", [((7, 5), FP16), ((8, 0), BF16), ((9, 0), BF16)])
def test_nvidia_unchanged(monkeypatch, capability, expected):
    # NVIDIA reports no gcnArchName; a stray one must not matter since the arch gate is ROCm-only.
    torch = _fake_torch(hip = None, arch = "gfx1030", capability = capability)
    _install(monkeypatch, torch, is_rocm = False)
    assert dd.resolve_diffusion_device_target().dtype == expected
    assert na.NativeAudioBackend._dtype(types.SimpleNamespace(device = "cuda")) == expected
    assert tc.native_bf16_supported() is (expected == BF16)


def test_env_override(monkeypatch):
    torch = _fake_torch(hip = "6.4", arch = "gfx1030")
    _install(monkeypatch, torch, is_rocm = True)
    monkeypatch.setenv("UNSLOTH_STUDIO_ROCM_BF16", "1")
    assert dd.resolve_diffusion_device_target().dtype == BF16
    torch_cdna = _fake_torch(hip = "6.4", arch = "gfx942")
    _install(monkeypatch, torch_cdna, is_rocm = True)
    monkeypatch.setenv("UNSLOTH_STUDIO_ROCM_BF16", "0")
    assert dd.resolve_diffusion_device_target().dtype == FP16


def test_selected_ordinal_arch_is_read(monkeypatch):
    # Mixed box: card 0 is gfx1100, the selected card 1 is gfx1030 -> fp16 for card 1.
    from core.inference.rocm_bf16 import rocm_bf16_supported

    torch = _fake_torch(hip = "6.4")
    archs = {0: "gfx1100", 1: "gfx1030"}
    torch.cuda.get_device_properties = lambda i = None: types.SimpleNamespace(gcnArchName = archs[i])
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.delenv("UNSLOTH_STUDIO_ROCM_BF16", raising = False)
    assert rocm_bf16_supported(torch, 1) is False
    assert rocm_bf16_supported(torch, 0) is True
    assert rocm_bf16_supported(torch) is True


@pytest.mark.parametrize("arch,expected", [("gfx1030", FP16), ("gfx906", FP16), ("gfx90a", BF16), ("gfx1151", BF16)])
def test_laya_precision_by_arch(monkeypatch, arch, expected):
    import core.systemone.laya_runtime as lr

    torch = _fake_torch(hip = "6.4", arch = arch)
    _install(monkeypatch, torch, is_rocm = True)
    monkeypatch.delenv("UNSLOTH_SYSTEMONE_FP32", raising = False)
    device = types.SimpleNamespace(type = "cuda", index = 0)
    assert lr._precision(device, fp16_checkpoint = False) == (expected, expected)
