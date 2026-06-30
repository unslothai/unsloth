# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Hermetic, CPU-only tests for the diffusion device/dtype resolver.

`torch` is stubbed via a fake module so no GPU/torch is needed, and
`utils.hardware` is either stubbed (studio-layer path) or forced to fail
(torch-probe fallback path). Both paths are asserted.
"""

from __future__ import annotations

import sys
import types

from core.inference import diffusion_device as dd


# ── Fakes ─────────────────────────────────────────────────────────────


class _FakeDtype:
    def __init__(self, name: str) -> None:
        self.name = name

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _FakeDtype) and other.name == self.name

    def __hash__(self) -> int:
        return hash(self.name)

    def __repr__(self) -> str:  # str(dtype) -> "torch.bfloat16"
        return f"torch.{self.name}"


BF16 = _FakeDtype("bfloat16")
FP16 = _FakeDtype("float16")
FP32 = _FakeDtype("float32")


class _FiniteResult:
    def __init__(self, finite: bool) -> None:
        self._finite = finite

    def all(self) -> "_FiniteResult":
        return self

    def item(self) -> bool:
        return self._finite


class _FakeTensor:
    def __init__(self, finite: bool = True) -> None:
        self._finite = finite

    def __add__(self, other: object) -> "_FakeTensor":
        return self

    def float(self) -> "_FakeTensor":
        return self


def _make_torch(
    *,
    cuda_available: bool = False,
    capability = (8, 0),
    capability_raises: bool = False,
    bf16_supported: bool = False,
    hip = None,
    mps_available: bool = False,
    mps_probe: str = "pass",  # "pass" | "raise" | "nonfinite"
    xpu_available = None,  # None -> no xpu attr; True/False -> present
    xpu_bf16: bool = False,
) -> types.ModuleType:
    torch = types.ModuleType("torch")
    torch.bfloat16 = BF16
    torch.float16 = FP16
    torch.float32 = FP32
    torch.version = types.SimpleNamespace(hip = hip)

    def _get_cap():
        if capability_raises:
            raise RuntimeError("no capability")
        return capability

    torch.cuda = types.SimpleNamespace(
        is_available = lambda: cuda_available,
        get_device_capability = _get_cap,
        is_bf16_supported = lambda: bf16_supported,
    )

    mps_ns = types.SimpleNamespace(is_available = lambda: mps_available)
    torch.backends = types.SimpleNamespace(mps = mps_ns)

    def _ones(*_a, **_k):
        if mps_probe == "raise":
            raise RuntimeError("bf16 unsupported on this MPS")
        return _FakeTensor(finite = (mps_probe == "pass"))

    torch.ones = _ones
    torch.isfinite = lambda t: _FiniteResult(getattr(t, "_finite", True))

    if xpu_available is not None:
        torch.xpu = types.SimpleNamespace(
            is_available = lambda: xpu_available,
            is_bf16_supported = lambda: xpu_bf16,
        )
    return torch


def _install(
    monkeypatch,
    torch,
    *,
    studio_device = None,
    is_rocm = False,
    hardware_fails = False,
):
    """Install the fake torch and either a fake or failing `utils.hardware`."""
    monkeypatch.setitem(sys.modules, "torch", torch)
    if hardware_fails:
        # Force `from utils.hardware import ...` to raise -> torch-probe fallback.
        monkeypatch.setitem(sys.modules, "utils.hardware", None)
        return

    class _DT:
        CUDA = "cuda"
        XPU = "xpu"
        MLX = "mlx"
        CPU = "cpu"

    fake_uh = types.ModuleType("utils.hardware")
    fake_uh.DeviceType = _DT
    fake_uh.get_device = lambda: studio_device
    fake_uh.hardware = types.SimpleNamespace(IS_ROCM = is_rocm)
    monkeypatch.setitem(sys.modules, "utils.hardware", fake_uh)


# ── Studio-layer path ─────────────────────────────────────────────────


def test_cuda_ampere_bf16(monkeypatch):
    torch = _make_torch(cuda_available = True, capability = (8, 0))
    _install(monkeypatch, torch, studio_device = "cuda")
    t = dd.resolve_diffusion_device_target()
    assert (t.device, t.dtype, t.backend, t.vendor) == ("cuda", BF16, "cuda", "nvidia")
    assert (
        t.supports_model_cpu_offload
        and t.supports_default_torch_compile
        and t.supports_pinned_transfer
    )


def test_cuda_pre_ampere_fp16(monkeypatch):
    torch = _make_torch(cuda_available = True, capability = (7, 5), bf16_supported = True)
    _install(monkeypatch, torch, studio_device = "cuda")
    t = dd.resolve_diffusion_device_target()
    # is_bf16_supported() is True (emulated) but capability < 8 -> fp16.
    assert t.dtype == FP16 and t.backend == "cuda"


def test_cuda_capability_raises_falls_back_fp16(monkeypatch):
    torch = _make_torch(cuda_available = True, capability_raises = True)
    _install(monkeypatch, torch, studio_device = "cuda")
    t = dd.resolve_diffusion_device_target()
    assert t.dtype == FP16 and t.device == "cuda"


def test_cuda_studio_says_cuda_but_unavailable_is_cpu(monkeypatch):
    torch = _make_torch(cuda_available = False)
    _install(monkeypatch, torch, studio_device = "cuda")
    t = dd.resolve_diffusion_device_target()
    assert t.device == "cpu" and t.dtype == FP32


def test_rocm_target(monkeypatch):
    torch = _make_torch(cuda_available = True, bf16_supported = True)
    _install(monkeypatch, torch, studio_device = "cuda", is_rocm = True)
    t = dd.resolve_diffusion_device_target()
    assert (t.device, t.backend, t.vendor) == ("cuda", "rocm", "amd")
    assert t.dtype == BF16
    assert t.supports_default_torch_compile is False  # ROCm disables default compile


def test_rocm_without_bf16_uses_fp16(monkeypatch):
    torch = _make_torch(cuda_available = True, bf16_supported = False)
    _install(monkeypatch, torch, studio_device = "cuda", is_rocm = True)
    t = dd.resolve_diffusion_device_target()
    assert t.dtype == FP16 and t.backend == "rocm"


def test_xpu_bf16(monkeypatch):
    torch = _make_torch(xpu_available = True, xpu_bf16 = True)
    _install(monkeypatch, torch, studio_device = "xpu")
    t = dd.resolve_diffusion_device_target()
    assert (t.device, t.backend, t.vendor, t.dtype) == ("xpu", "xpu", "intel", BF16)
    assert (
        t.supports_model_cpu_offload
        and not t.supports_default_torch_compile
        and not t.supports_pinned_transfer
    )


def test_xpu_without_bf16_fp16(monkeypatch):
    torch = _make_torch(xpu_available = True, xpu_bf16 = False)
    _install(monkeypatch, torch, studio_device = "xpu")
    t = dd.resolve_diffusion_device_target()
    assert t.device == "xpu" and t.dtype == FP16


def test_mps_probe_pass_bf16(monkeypatch):
    torch = _make_torch(mps_available = True, mps_probe = "pass")
    _install(monkeypatch, torch, studio_device = "mlx")
    t = dd.resolve_diffusion_device_target()
    assert (t.device, t.backend, t.vendor, t.dtype) == ("mps", "mps", "apple", BF16)
    assert not t.supports_model_cpu_offload


def test_mps_probe_raises_uses_fp32_not_fp16(monkeypatch):
    torch = _make_torch(mps_available = True, mps_probe = "raise")
    _install(monkeypatch, torch, studio_device = "mlx")
    t = dd.resolve_diffusion_device_target()
    assert t.device == "mps" and t.dtype == FP32  # strict: never silent fp16


def test_mps_probe_nonfinite_uses_fp32(monkeypatch):
    torch = _make_torch(mps_available = True, mps_probe = "nonfinite")
    _install(monkeypatch, torch, studio_device = "mlx")
    t = dd.resolve_diffusion_device_target()
    assert t.device == "mps" and t.dtype == FP32


def test_studio_cpu_on_apple_prefers_mps(monkeypatch):
    torch = _make_torch(mps_available = True, mps_probe = "pass")
    _install(monkeypatch, torch, studio_device = "cpu")  # Studio reports CPU (no mlx pkg)
    t = dd.resolve_diffusion_device_target()
    assert t.device == "mps" and t.dtype == BF16


def test_cpu_when_nothing_available(monkeypatch):
    torch = _make_torch(mps_available = False)
    _install(monkeypatch, torch, studio_device = "cpu")
    t = dd.resolve_diffusion_device_target()
    assert (t.device, t.backend, t.vendor, t.dtype) == ("cpu", "cpu", None, FP32)
    assert not any(
        (t.supports_model_cpu_offload, t.supports_default_torch_compile, t.supports_pinned_transfer)
    )


# ── torch-probe fallback path (utils.hardware import fails) ────────────


def test_fallback_cuda(monkeypatch):
    torch = _make_torch(cuda_available = True, capability = (9, 0))
    _install(monkeypatch, torch, hardware_fails = True)
    t = dd.resolve_diffusion_device_target()
    assert t.device == "cuda" and t.dtype == BF16 and t.backend == "cuda"


def test_fallback_rocm_via_torch_hip(monkeypatch):
    torch = _make_torch(cuda_available = True, bf16_supported = True, hip = "6.2")
    _install(monkeypatch, torch, hardware_fails = True)
    t = dd.resolve_diffusion_device_target()
    assert t.backend == "rocm" and t.vendor == "amd"


def test_fallback_xpu(monkeypatch):
    torch = _make_torch(cuda_available = False, xpu_available = True, xpu_bf16 = True)
    _install(monkeypatch, torch, hardware_fails = True)
    t = dd.resolve_diffusion_device_target()
    assert t.device == "xpu" and t.dtype == BF16


def test_fallback_mps(monkeypatch):
    torch = _make_torch(cuda_available = False, mps_available = True, mps_probe = "pass")
    _install(monkeypatch, torch, hardware_fails = True)
    t = dd.resolve_diffusion_device_target()
    assert t.device == "mps" and t.dtype == BF16


def test_fallback_cpu(monkeypatch):
    torch = _make_torch(cuda_available = False, mps_available = False)
    _install(monkeypatch, torch, hardware_fails = True)
    t = dd.resolve_diffusion_device_target()
    assert t.device == "cpu" and t.dtype == FP32


