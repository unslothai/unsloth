# SPDX-License-Identifier: AGPL-3.0-only
"""Unsloth hardware dispatch matrix: spoofs platform/torch/mlx per PROFILES to exercise CUDA/ROCm/XPU/MLX/CPU paths without real hardware."""

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


@dataclass
class HardwareProfile:
    name: str
    system: str  # platform.system() value
    machine: str  # platform.machine() value
    cuda_available: bool  # torch.cuda.is_available() value
    hip_version: Optional[str]  # torch.version.hip; None for NVIDIA, "6.1" etc. for ROCm
    xpu_available: bool  # torch.xpu.is_available() value
    has_mlx: bool  # whether to inject a fake mlx into sys.modules
    mps_available: bool  # torch.backends.mps.is_available() value

    expect_is_mlx: bool  # unsloth._IS_MLX
    expect_device_type: str  # Unsloth DeviceType (uppercased name: "CUDA"/"XPU"/"MLX"/"CPU")
    expect_is_rocm: bool  # Unsloth IS_ROCM
    expect_apple_silicon: bool  # Unsloth is_apple_silicon()
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
        "Unsloth still uses DeviceType.CUDA but flips IS_ROCM=True.",
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


@pytest.fixture
def spoof_hardware(monkeypatch):
    """Return a function that applies a HardwareProfile to the live process; monkeypatch cleans up on exit."""

    def _apply(profile: HardwareProfile) -> None:
        import platform
        import torch

        # platform spoof (used by both the unsloth gate and Unsloth's helpers)
        monkeypatch.setattr(platform, "system", lambda: profile.system)
        monkeypatch.setattr(platform, "machine", lambda: profile.machine)

        monkeypatch.setattr(torch.cuda, "is_available", lambda: profile.cuda_available)
        # Stub get_device_properties: detect_hardware reads .name, which crashes on a CPU CI runner.
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

        # torch.version.hip: None on NVIDIA, "6.1" etc. on ROCm
        torch_version = torch.version
        monkeypatch.setattr(torch_version, "hip", profile.hip_version, raising = False)

        # Stub torch.xpu.* always; real get_device_name needs the XPU torch build.
        if hasattr(torch, "xpu"):
            monkeypatch.setattr(torch.xpu, "is_available", lambda: profile.xpu_available)
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
            monkeypatch.setattr(torch.backends.mps, "is_available", lambda: profile.mps_available)

        # mlx + mlx.core in sys.modules
        if profile.has_mlx:
            fake_mlx = types.ModuleType("mlx")
            fake_mlx.__spec__ = importlib.machinery.ModuleSpec("mlx", loader = None)
            fake_mlx.__path__ = []
            fake_mlx_core = types.ModuleType("mlx.core")
            fake_mlx.core = fake_mlx_core
            monkeypatch.setitem(sys.modules, "mlx", fake_mlx)
            monkeypatch.setitem(sys.modules, "mlx.core", fake_mlx_core)
            # detect_hardware gates MLX on the full stack via utils.mlx_repair (it
            # imports mlx_lm/mlx_vlm and checks dist versions), which faking only
            # mlx.core cannot satisfy. An mlx profile means a complete, healthy stack,
            # so model that here; the internals are covered by test_mlx_repair.py.
            # Both entry points, because the gate asks for the blocker LIST: one
            # measurement decides the verdict and explains it. Stubbing only
            # mlx_stack_available() runs the real check against a Linux runner with no
            # MLX distributions, so the Apple Silicon profile detects CPU.
            if str(STUDIO_BACKEND) not in sys.path:
                sys.path.insert(0, str(STUDIO_BACKEND))
            import utils.mlx_repair as _mlx_repair  # type: ignore

            monkeypatch.setattr(_mlx_repair, "mlx_stack_available", lambda: True)
            monkeypatch.setattr(_mlx_repair, "mlx_stack_blockers", lambda: [])
        else:
            # Drop cached mlx and patch find_spec so the unsloth gate sees mlx as absent.
            monkeypatch.delitem(sys.modules, "mlx", raising = False)
            monkeypatch.delitem(sys.modules, "mlx.core", raising = False)
            real_find_spec = importlib.util.find_spec

            def _no_mlx(name, *args, **kwargs):
                if name == "mlx" or name.startswith("mlx."):
                    return None
                return real_find_spec(name, *args, **kwargs)

            monkeypatch.setattr(importlib.util, "find_spec", _no_mlx)

            # Unsloth's _has_mlx() does `import mlx.core`, not find_spec; block it
            # with a meta_path finder that raises ImportError for mlx.*.
            class _BlockMLXFinder:
                def find_spec(
                    self_inner,
                    name,
                    path = None,
                    target = None,
                ):
                    if name == "mlx" or name.startswith("mlx."):
                        raise ImportError(
                            f"mlx import blocked by spoof_hardware (profile={profile.name})"
                        )
                    return None

            blocker = _BlockMLXFinder()
            # New list so monkeypatch fully restores on teardown.
            monkeypatch.setattr(
                sys,
                "meta_path",
                [blocker, *sys.meta_path],
            )

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
    """Lazy-load Unsloth's hardware module under the bare-imports layout."""
    if str(STUDIO_BACKEND) not in sys.path:
        sys.path.insert(0, str(STUDIO_BACKEND))
    # Fresh import so detect_hardware re-runs under the current spoofs.
    sys.modules.pop("utils.hardware.hardware", None)
    sys.modules.pop("utils.hardware", None)
    from utils.hardware import hardware as hw  # type: ignore

    return hw


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
    """Unsloth's detect_hardware() routes to the right DeviceType per profile."""
    spoof_hardware(profile)
    hw = _import_studio_hardware_module()
    detected = hw.detect_hardware()
    expected = getattr(hw.DeviceType, profile.expect_device_type)
    assert detected == expected, (
        f"profile {profile.name}: expected {profile.expect_device_type}, "
        f"got {detected!r}. {profile.extra_notes}"
    )
    assert (
        hw.IS_ROCM is profile.expect_is_rocm
    ), f"profile {profile.name}: expected IS_ROCM={profile.expect_is_rocm}, got {hw.IS_ROCM}"


@pytest.mark.parametrize("profile", PROFILES, ids = PROFILE_IDS)
def test_studio_is_apple_silicon_matches_profile(profile, spoof_hardware):
    """Unsloth's is_apple_silicon() helper agrees with platform spoof."""
    spoof_hardware(profile)
    hw = _import_studio_hardware_module()
    assert hw.is_apple_silicon() is profile.expect_apple_silicon, (
        f"profile {profile.name}: expected is_apple_silicon={profile.expect_apple_silicon}, "
        f"got {hw.is_apple_silicon()}"
    )


# Negative-space tests: catch regressions where the dispatch order changes.


def test_cuda_takes_priority_over_mlx_when_both_available(spoof_hardware):
    """CUDA wins over MLX when both available: canary against GPU users being routed to MLX after refactors."""
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


# Unsloth's placement, against the loader's opt-in device map.
#
# unsloth's loader upgrades a "sequential" device_map to the "unsloth" planning sentinel
# when UNSLOTH_AUTO_DEVICE_MAP=1. Unsloth does not pass the sentinel and never sets that
# variable, but an operator can set it process-wide, and Unsloth's "sequential" is not a
# default it forgot to change: it is get_device_map() saying "one device". These pin the
# two facts that keep that safe on every profile above -- Unsloth's multi-GPU answer is
# "balanced", which is never upgraded, and its single-GPU answer is reached only inside a
# worker that has already narrowed the visible devices to the selection.


def _loader_device_map_helpers():
    """The two loader functions, rebuilt over a fabricated torch.

    ast rather than an import: `unsloth.models.loader_utils` pulls in the whole CUDA
    import chain, which is exactly what the spoofs in this file are pretending about.
    """
    import ast as _ast

    source = (REPO_ROOT / "unsloth" / "models" / "loader_utils.py").read_text(encoding = "utf-8")

    class _Cuda:
        def __init__(self, count):
            self._count = count

        def device_count(self):
            return self._count

        def mem_get_info(self, index):
            return (8 * 2**30, 16 * 2**30)

    def build(visible_devices):
        import os as _os

        namespace = {
            "os": _os,
            "torch": types.SimpleNamespace(cuda = _Cuda(visible_devices)),
            "DEVICE_TYPE_TORCH": "cuda",
            "is_distributed": lambda: False,
        }
        for node in _ast.parse(source).body:
            if isinstance(node, _ast.FunctionDef) and node.name in (
                "requested_device_map",
                "resolve_unsloth_device_map",
                "_as_bytes",
            ):
                exec(_ast.get_source_segment(source, node), namespace)
            elif isinstance(node, _ast.ClassDef) and node.name == "_DefaultDeviceMap":
                exec(_ast.get_source_segment(source, node), namespace)
            elif isinstance(node, _ast.Assign) and getattr(node.targets[0], "id", None) in (
                "UNSLOTH_DEVICE_MAP",
                "UNSLOTH_BALANCED_DEVICE_MAP",
                "_PLANNED_DEVICE_MAPS",
                "DEFAULT_DEVICE_MAP",
                "_SIZE_UNITS",
            ):
                exec(_ast.get_source_segment(source, node), namespace)
        # No planner installed: the fallback is what a decline looks like from here.
        sys.modules.pop("unsloth_zoo.device_map_planner", None)
        sys.modules["unsloth_zoo.device_map_planner"] = types.ModuleType(
            "unsloth_zoo.device_map_planner"
        )
        return namespace

    return build


@pytest.mark.parametrize("profile", PROFILES, ids = PROFILE_IDS)
@pytest.mark.parametrize("gpu_ids", [None, [], [0], [0, 1], [2, 3, 5]], ids = repr)
@pytest.mark.parametrize("opt_in", ["unset", "0", "1"])
def test_studio_placement_survives_the_loader_opt_in(
    profile, gpu_ids, opt_in, spoof_hardware, monkeypatch
):
    """Whatever Unsloth decided, the loader hands transformers a map of the same shape.

    "sequential" and "balanced" survive untouched. "unsloth_balanced" is a request to
    plan, so the loader may answer with a plan or, when it declines -- as it does here,
    with no planner installed -- with the sharding map that name declines to. What it
    must never do is turn a multi-GPU ask into "sequential", which fills cuda:0 first.
    """
    spoof_hardware(profile)
    hw = _import_studio_hardware_module()

    if gpu_ids and hw.get_device() not in (hw.DeviceType.CUDA, hw.DeviceType.XPU):
        pytest.skip(f"{profile.name} does not take an explicit gpu_ids")

    device_map = hw.get_device_map(gpu_ids)
    assert device_map in ("balanced", "sequential", "unsloth_balanced")

    if opt_in == "unset":
        monkeypatch.delenv("UNSLOTH_AUTO_DEVICE_MAP", raising = False)
    else:
        monkeypatch.setenv("UNSLOTH_AUTO_DEVICE_MAP", opt_in)

    # The worker narrows CUDA_VISIBLE_DEVICES to the selection before torch initialises,
    # so the loader counts the selected devices, not the machine's.
    visible = len(gpu_ids) if gpu_ids else 1
    loader = _loader_device_map_helpers()(visible)

    resolved = loader["resolve_unsloth_device_map"](
        loader["requested_device_map"](device_map), "unsloth/Qwen3-0.6B"
    )
    # No planner module is installed, so a planned name always reaches its fallback.
    expected = loader["_PLANNED_DEVICE_MAPS"].get(device_map, device_map)
    assert resolved == expected, (
        f"profile {profile.name}, gpu_ids={gpu_ids}, UNSLOTH_AUTO_DEVICE_MAP={opt_in}: "
        f"Unsloth asked for {device_map!r} and the loader produced {resolved!r}"
    )


@pytest.mark.parametrize("profile", PROFILES, ids = PROFILE_IDS)
def test_studio_never_speaks_the_planning_sentinel(profile, spoof_hardware):
    """get_device_map is the only thing that names a placement for Unsloth's loads, and
    the plain "unsloth" sentinel is not one of its answers on any backend.

    That name declines to "sequential", which gives cuda:0 its whole free budget and so
    puts a model that fits there on one card. Every path the planner vetoes -- a full
    finetune, an `auto_model` with no `_model_mapping`, a Falcon-H1 checkpoint missing
    the mamba exclusions -- ends in that fallback, so a multi-GPU ask has to use the
    name whose fallback still shards.
    """
    spoof_hardware(profile)
    hw = _import_studio_hardware_module()
    answers = {hw.get_device_map(None), hw.get_device_map([])}
    if hw.get_device() in (hw.DeviceType.CUDA, hw.DeviceType.XPU):
        answers |= {hw.get_device_map([0]), hw.get_device_map([0, 1])}
    assert "unsloth" not in answers
    assert answers <= {"balanced", "sequential", "unsloth_balanced"}
