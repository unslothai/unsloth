# SPDX-License-Identifier: AGPL-3.0-only
"""
Comprehensive hardware dispatch matrix for Studio.

Drives every supported hardware profile from a single test host by
spoofing platform / torch.cuda / torch.xpu / sys.modules['mlx'] so we
can exercise the CUDA, ROCm, XPU, MLX, and CPU dispatch paths
deterministically without real hardware.

Profiles checked:

    nvidia_cuda          Linux x86_64 + torch.cuda.is_available()=True,
                         torch.version.hip=None
    amd_rocm             Linux x86_64 + torch.cuda.is_available()=True,
                         torch.version.hip="6.1"   (PyTorch ROCm aliases
                         torch.cuda.* over HIP)
    intel_xpu            Linux x86_64 + torch.cuda off, torch.xpu.is_available()=True
    apple_silicon_mlx    Darwin arm64 + cuda off + xpu off + mlx importable
    apple_silicon_no_mlx Darwin arm64 + everything off (no mlx pkg)
    linux_arm64_with_mlx Linux arm64 + mlx importable -- gate must NOT activate
                         (canary against accidental Linux-arm64 hijack)
    cpu_only             Linux x86_64 + nothing -- pure CPU fallback

For each profile we assert three contracts:

  1. ``unsloth._IS_MLX`` (re-evaluated under the spoof).
  2. ``utils.hardware.detect_hardware()`` ``DeviceType`` and ``IS_ROCM``.
  3. ``utils.hardware.is_apple_silicon()``.

Add a row to ``PROFILES`` to extend coverage; tests parametrize over it
automatically. No real hardware required.
"""

from __future__ import annotations

import importlib
import importlib.machinery
import importlib.util
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
STUDIO_BACKEND = REPO_ROOT / "studio" / "backend"


# ---------------------------------------------------------------------------
# Profile definition
# ---------------------------------------------------------------------------


@dataclass
class HardwareProfile:
    name: str
    system: str  # platform.system() value
    machine: str  # platform.machine() value
    cuda_available: bool  # torch.cuda.is_available() value
    hip_version: Optional[
        str
    ]  # torch.version.hip; None for NVIDIA, "6.1" etc. for ROCm
    xpu_available: bool  # torch.xpu.is_available() value
    has_mlx: bool  # whether to inject a fake mlx into sys.modules
    mps_available: bool  # torch.backends.mps.is_available() value

    expect_is_mlx: bool  # unsloth._IS_MLX
    expect_device_type: (
        str  # Studio DeviceType (uppercased name: "CUDA"/"XPU"/"MLX"/"CPU")
    )
    expect_is_rocm: bool  # Studio IS_ROCM
    expect_apple_silicon: bool  # Studio is_apple_silicon()
    extra_notes: str = ""


PROFILES = [
    HardwareProfile(
        name = "nvidia_cuda",
        system = "Linux",
        machine = "x86_64",
        cuda_available = True,
        hip_version = None,
        xpu_available = False,
        has_mlx = False,
        mps_available = False,
        expect_is_mlx = False,
        expect_device_type = "CUDA",
        expect_is_rocm = False,
        expect_apple_silicon = False,
    ),
    HardwareProfile(
        name = "amd_rocm",
        system = "Linux",
        machine = "x86_64",
        cuda_available = True,
        hip_version = "6.1",
        xpu_available = False,
        has_mlx = False,
        mps_available = False,
        expect_is_mlx = False,
        expect_device_type = "CUDA",
        expect_is_rocm = True,
        expect_apple_silicon = False,
        extra_notes = "PyTorch ROCm reuses torch.cuda.* over HIP; "
        "Studio still uses DeviceType.CUDA but flips IS_ROCM=True.",
    ),
    HardwareProfile(
        name = "intel_xpu",
        system = "Linux",
        machine = "x86_64",
        cuda_available = False,
        hip_version = None,
        xpu_available = True,
        has_mlx = False,
        mps_available = False,
        expect_is_mlx = False,
        expect_device_type = "XPU",
        expect_is_rocm = False,
        expect_apple_silicon = False,
    ),
    HardwareProfile(
        name = "apple_silicon_mlx",
        system = "Darwin",
        machine = "arm64",
        cuda_available = False,
        hip_version = None,
        xpu_available = False,
        has_mlx = True,
        mps_available = True,
        expect_is_mlx = True,
        expect_device_type = "MLX",
        expect_is_rocm = False,
        expect_apple_silicon = True,
    ),
    HardwareProfile(
        name = "apple_silicon_no_mlx",
        system = "Darwin",
        machine = "arm64",
        cuda_available = False,
        hip_version = None,
        xpu_available = False,
        has_mlx = False,
        mps_available = True,
        expect_is_mlx = False,
        expect_device_type = "CPU",
        expect_is_rocm = False,
        expect_apple_silicon = True,
        extra_notes = "Mac without mlx falls through to CPU (chat-only).",
    ),
    HardwareProfile(
        name = "linux_arm64_with_mlx",
        system = "Linux",
        machine = "arm64",
        cuda_available = False,
        hip_version = None,
        xpu_available = False,
        has_mlx = True,
        mps_available = False,
        expect_is_mlx = False,
        expect_device_type = "CPU",
        expect_is_rocm = False,
        expect_apple_silicon = False,
        extra_notes = "Canary: Linux ARM64 with mlx package installed must NOT "
        "trigger MLX dispatch; the system check is what guards it.",
    ),
    HardwareProfile(
        name = "cpu_only",
        system = "Linux",
        machine = "x86_64",
        cuda_available = False,
        hip_version = None,
        xpu_available = False,
        has_mlx = False,
        mps_available = False,
        expect_is_mlx = False,
        expect_device_type = "CPU",
        expect_is_rocm = False,
        expect_apple_silicon = False,
    ),
]

PROFILE_IDS = [p.name for p in PROFILES]


# ---------------------------------------------------------------------------
# Spoofing helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def spoof_hardware(monkeypatch):
    """Return a function that applies a HardwareProfile to the live process.

    Idempotent: each call re-applies the profile. Cleanup happens
    automatically when the test exits via monkeypatch.
    """

    def _apply(profile: HardwareProfile) -> None:
        import platform
        import torch

        # platform spoof (used by both the unsloth gate and Studio's helpers)
        monkeypatch.setattr(platform, "system", lambda: profile.system)
        monkeypatch.setattr(platform, "machine", lambda: profile.machine)

        # torch.cuda.is_available
        monkeypatch.setattr(torch.cuda, "is_available", lambda: profile.cuda_available)
        # detect_hardware reads torch.cuda.get_device_properties(0).name when
        # cuda_available is True. On a CPU CI runner that triggers _cuda_init
        # and crashes with "No CUDA GPUs are available". Stub it so the
        # dispatch path under test runs end-to-end.
        if profile.cuda_available:
            stub_props = types.SimpleNamespace(
                name = "Stub GPU" if not profile.hip_version else "Stub AMD GPU",
            )
            monkeypatch.setattr(
                torch.cuda,
                "get_device_properties",
                lambda i = 0: stub_props,
                raising = False,
            )

        # torch.version.hip — None on NVIDIA, "6.1" etc. on ROCm
        torch_version = torch.version
        monkeypatch.setattr(torch_version, "hip", profile.hip_version, raising = False)

        # torch.xpu.is_available + get_device_name -- detect_hardware reads both.
        # Real torch.xpu.get_device_name requires the XPU-compiled torch build,
        # so always stub it under the spoof to keep tests hardware-agnostic.
        if hasattr(torch, "xpu"):
            monkeypatch.setattr(
                torch.xpu, "is_available", lambda: profile.xpu_available
            )
            monkeypatch.setattr(
                torch.xpu,
                "get_device_name",
                lambda i = 0: "Intel XPU (stub)",
                raising = False,
            )
        elif profile.xpu_available:
            xpu_stub = types.SimpleNamespace(
                is_available = lambda: True,
                get_device_name = lambda i = 0: "Intel XPU (stub)",
            )
            monkeypatch.setattr(torch, "xpu", xpu_stub, raising = False)

        # torch.backends.mps.is_available
        if hasattr(torch.backends, "mps"):
            monkeypatch.setattr(
                torch.backends.mps, "is_available", lambda: profile.mps_available
            )

        # mlx + mlx.core in sys.modules
        if profile.has_mlx:
            fake_mlx = types.ModuleType("mlx")
            fake_mlx.__spec__ = importlib.machinery.ModuleSpec("mlx", loader = None)
            fake_mlx.__path__ = []
            fake_mlx_core = types.ModuleType("mlx.core")
            fake_mlx.core = fake_mlx_core
            monkeypatch.setitem(sys.modules, "mlx", fake_mlx)
            monkeypatch.setitem(sys.modules, "mlx.core", fake_mlx_core)
        else:
            monkeypatch.delitem(sys.modules, "mlx", raising = False)
            monkeypatch.delitem(sys.modules, "mlx.core", raising = False)
            real_find_spec = importlib.util.find_spec

            def _no_mlx(name, *args, **kwargs):
                if name == "mlx":
                    return None
                return real_find_spec(name, *args, **kwargs)

            monkeypatch.setattr(importlib.util, "find_spec", _no_mlx)

    return _apply


def _evaluate_unsloth_is_mlx_gate() -> bool:
    """Re-evaluate the exact expression from unsloth/__init__.py:20-24."""
    import importlib.util
    import platform

    return (
        platform.system() == "Darwin"
        and platform.machine() == "arm64"
        and importlib.util.find_spec("mlx") is not None
    )


def _import_studio_hardware_module():
    """Lazy-load Studio's hardware module under the bare-imports layout."""
    if str(STUDIO_BACKEND) not in sys.path:
        sys.path.insert(0, str(STUDIO_BACKEND))
    # Force a fresh import so detect_hardware re-runs under the current spoofs.
    sys.modules.pop("utils.hardware.hardware", None)
    sys.modules.pop("utils.hardware", None)
    from utils.hardware import hardware as hw  # type: ignore

    return hw


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("profile", PROFILES, ids = PROFILE_IDS)
def test_unsloth_is_mlx_gate_matches_profile(profile, spoof_hardware):
    """The _IS_MLX expression in unsloth/__init__.py flips correctly per profile."""
    spoof_hardware(profile)
    actual = _evaluate_unsloth_is_mlx_gate()
    assert actual is profile.expect_is_mlx, (
        f"profile {profile.name}: expected _IS_MLX={profile.expect_is_mlx}, "
        f"got {actual}. {profile.extra_notes}"
    )


@pytest.mark.parametrize("profile", PROFILES, ids = PROFILE_IDS)
def test_studio_detect_hardware_matches_profile(profile, spoof_hardware):
    """Studio's detect_hardware() routes to the right DeviceType per profile."""
    spoof_hardware(profile)
    hw = _import_studio_hardware_module()
    detected = hw.detect_hardware()
    expected = getattr(hw.DeviceType, profile.expect_device_type)
    assert detected == expected, (
        f"profile {profile.name}: expected {profile.expect_device_type}, "
        f"got {detected!r}. {profile.extra_notes}"
    )
    assert hw.IS_ROCM is profile.expect_is_rocm, (
        f"profile {profile.name}: expected IS_ROCM={profile.expect_is_rocm}, "
        f"got {hw.IS_ROCM}"
    )


@pytest.mark.parametrize("profile", PROFILES, ids = PROFILE_IDS)
def test_studio_is_apple_silicon_matches_profile(profile, spoof_hardware):
    """Studio's is_apple_silicon() helper agrees with platform spoof."""
    spoof_hardware(profile)
    hw = _import_studio_hardware_module()
    assert hw.is_apple_silicon() is profile.expect_apple_silicon, (
        f"profile {profile.name}: expected is_apple_silicon={profile.expect_apple_silicon}, "
        f"got {hw.is_apple_silicon()}"
    )


# ---------------------------------------------------------------------------
# Negative-space tests: catch regressions where the dispatch order changes.
# ---------------------------------------------------------------------------


def test_cuda_takes_priority_over_mlx_when_both_available(spoof_hardware):
    """If both CUDA and MLX are available, Studio MUST pick CUDA. This is the
    canary that protects every existing GPU user from being silently routed
    to MLX after future refactors.
    """
    profile = HardwareProfile(
        name = "cuda_plus_mlx",
        system = "Darwin",
        machine = "arm64",
        cuda_available = True,
        hip_version = None,
        xpu_available = False,
        has_mlx = True,
        mps_available = True,
        expect_is_mlx = True,
        expect_device_type = "CUDA",
        expect_is_rocm = False,
        expect_apple_silicon = True,
    )
    spoof_hardware(profile)
    hw = _import_studio_hardware_module()
    assert hw.detect_hardware() == hw.DeviceType.CUDA


def test_xpu_takes_priority_over_mlx_when_both_available(spoof_hardware):
    """XPU is selected over MLX in the dispatch order."""
    profile = HardwareProfile(
        name = "xpu_plus_mlx",
        system = "Darwin",
        machine = "arm64",
        cuda_available = False,
        hip_version = None,
        xpu_available = True,
        has_mlx = True,
        mps_available = True,
        expect_is_mlx = True,
        expect_device_type = "XPU",
        expect_is_rocm = False,
        expect_apple_silicon = True,
    )
    spoof_hardware(profile)
    hw = _import_studio_hardware_module()
    assert hw.detect_hardware() == hw.DeviceType.XPU
