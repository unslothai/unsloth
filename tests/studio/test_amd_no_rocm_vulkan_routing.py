# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""Guards Vulkan routing for Linux AMD GPUs with no usable ROCm.

History: every AMD branch in the selector is gated on has_rocm and the Vulkan branch
required has_intel_gpu, so an AMD GPU without working ROCm was indistinguishable from a
headless CPU-only box and fell all the way through to the CPU llama.cpp bundle. ROCm does
not support most AMD parts -- APUs least of all -- and plenty of distros ship Mesa/RADV
without ROCm ever being installed, so that population is large and was getting a CPU-only
binary on a GPU host.

Why widening this is safe: the Vulkan bundle is a SUPERSET of the CPU bundle. It ships
the same libggml-cpu-*.so variants alongside libggml-vulkan.so, and with no usable Vulkan
device it enumerates zero devices and runs on the CPU backend -- verified by running
`--list-devices` with the Vulkan ICD removed, which lists nothing and exits 0. A host that
turns out to have no working Vulkan therefore lands exactly where it does today.

Measured on a Steam Deck (Van Gogh gfx1033, Qwen2.5-0.5B Q4_0, llama-bench, pp128/tg64):

    Vulkan (RADV)   1444.80 pp / 112.81 tg
    CPU              753.08 pp /  49.76 tg
    ROCm (gfx103X prebuilt via HSA_OVERRIDE_GFX_VERSION=10.3.0)
                     802.06 pp /  17.51 tg

The contract:
  * AMD GPU + no usable ROCm     -> Vulkan bundle (was: CPU)
  * AMD GPU + working ROCm       -> unchanged; detect_host() only probes DRM vendor ids
                                    when ROCm is absent, so the flag stays False there
                                    and the ROCm branches keep the host
  * Any host + PHYSICAL NVIDIA   -> never Vulkan (it ignores CUDA_VISIBLE_DEVICES and
                                    would enumerate a deliberately hidden card)
  * No GPU at all                -> unchanged (CPU bundle)
  * force_cpu                    -> clears the flag, so CPU still wins
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from studio import install_llama_prebuilt as ilp  # noqa: E402


def _host(**overrides):
    """A Linux x86_64 host with no NVIDIA and no ROCm, tweaked per test."""
    base = dict(
        system = "Linux",
        machine = "x86_64",
        is_windows = False,
        is_linux = True,
        is_macos = False,
        is_x86_64 = True,
        is_arm64 = False,
        nvidia_smi = None,
        driver_cuda_version = None,
        compute_caps = [],
        visible_cuda_devices = None,
        has_physical_nvidia = False,
        has_usable_nvidia = False,
        has_rocm = False,
        has_intel_gpu = False,
        has_amd_gpu_without_rocm = False,
    )
    base.update(overrides)
    fields = {f.name for f in dataclasses.fields(ilp.HostInfo)}
    return ilp.HostInfo(**{k: v for k, v in base.items() if k in fields})


def _patch_drm(monkeypatch, tmp_path, vendor_ids):
    """Fake /sys/class/drm/card*/device/vendor for the given vendor ids."""
    files = []
    for index, vendor in enumerate(vendor_ids):
        card = tmp_path / f"card{index}" / "device"
        card.mkdir(parents = True)
        (card / "vendor").write_text(vendor + "\n")
        files.append(str(card / "vendor"))
    monkeypatch.setattr(ilp.glob, "glob", lambda _pat: files)


_HOST_GPU_TOOLS = frozenset(
    {"nvidia-smi", "rocminfo", "amd-smi", "rocm_agent_enumerator", "hipconfig"}
)


def _patch_no_nvidia_no_rocm(monkeypatch):
    """Hide the REAL host's NVIDIA/ROCm from detect_host().

    The DRM vendor pass is gated on `not has_usable_nvidia and not has_rocm`, so on any
    machine with a GPU the fake sysfs above is never read and every case below collapses
    to False. Stub the two probes that see through to the host: nvidia-smi/rocminfo on
    PATH, and the /proc/driver/nvidia/gpus fallback.
    """
    _which = ilp.shutil.which
    monkeypatch.setattr(
        ilp.shutil,
        "which",
        lambda name, *a, **k: None if name in _HOST_GPU_TOOLS else _which(name, *a, **k),
    )
    _isdir = ilp.os.path.isdir
    monkeypatch.setattr(
        ilp.os.path,
        "isdir",
        lambda p: False if "nvidia" in str(p) else _isdir(p),
    )
    _exists = ilp.os.path.exists
    monkeypatch.setattr(
        ilp.os.path,
        "exists",
        lambda p: False if ("rocm" in str(p) or "kfd" in str(p)) else _exists(p),
    )


def test_hostinfo_exposes_the_flag():
    assert "has_amd_gpu_without_rocm" in {f.name for f in dataclasses.fields(ilp.HostInfo)}
    assert _host().has_amd_gpu_without_rocm is False


@pytest.mark.parametrize(
    "vendor_ids, expect_amd, expect_intel",
    [
        (["0x1002"], True, False),  # AMD only
        (["0x8086"], False, True),  # Intel only
        (["0x8086", "0x1002"], True, True),  # Intel iGPU + AMD dGPU: both seen
        (["0x1002", "0x8086"], True, True),  # order must not matter (no early break)
        (["0x10de"], False, False),  # NVIDIA vendor id is not ours to claim
        ([], False, False),  # headless
    ],
)
def test_drm_vendor_detection(monkeypatch, tmp_path, vendor_ids, expect_amd, expect_intel):
    """Both vendors come from one sysfs pass; an AMD dGPU next to an Intel iGPU must not
    be lost to an early break."""
    _patch_drm(monkeypatch, tmp_path, vendor_ids)
    _patch_no_nvidia_no_rocm(monkeypatch)
    monkeypatch.setattr(ilp.platform, "system", lambda: "Linux")
    host = ilp.detect_host()
    assert host.has_amd_gpu_without_rocm is expect_amd
    assert host.has_intel_gpu is expect_intel


@pytest.mark.parametrize(
    "env, rocm_tool_installed, expect_amd",
    [
        ({}, False, True),  # the target host: driver-only AMD, no mask
        ({}, True, True),  # ROCm tools present but the probe found nothing: still AMD
        # A masked-out AMD GPU on a ROCm host leaves the rocminfo probe empty and looks
        # exactly like a driver-only box. Vulkan honours no HIP mask, so routing there
        # would hand llama.cpp the card the caller hid.
        ({"ROCR_VISIBLE_DEVICES": ""}, True, False),
        ({"HIP_VISIBLE_DEVICES": "-1"}, True, False),
        ({"CUDA_VISIBLE_DEVICES": ""}, True, False),
        # ...but a bare exported CUDA_VISIBLE_DEVICES on a host with no ROCm installed at
        # all cannot be hiding a ROCm device, and must not cost the Steam Deck this branch
        # exists for its Vulkan bundle.
        ({"CUDA_VISIBLE_DEVICES": "0"}, False, True),
        ({"ROCR_VISIBLE_DEVICES": ""}, False, True),
    ],
)
def test_amd_visibility_mask_suppresses_auto_vulkan(
    monkeypatch, tmp_path, env, rocm_tool_installed, expect_amd
):
    _patch_drm(monkeypatch, tmp_path, ["0x1002"])
    _patch_no_nvidia_no_rocm(monkeypatch)
    monkeypatch.setattr(ilp.platform, "system", lambda: "Linux")
    for _var in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        monkeypatch.delenv(_var, raising = False)
    for _var, _val in env.items():
        monkeypatch.setenv(_var, _val)
    monkeypatch.setattr(ilp, "_rocm_probe_tool_present", lambda: rocm_tool_installed)
    assert ilp.detect_host().has_amd_gpu_without_rocm is expect_amd


def test_intel_detection_survives_an_amd_mask(monkeypatch, tmp_path):
    """The masks address HIP devices; an Intel iGPU keeps its own Vulkan route."""
    _patch_drm(monkeypatch, tmp_path, ["0x8086", "0x1002"])
    _patch_no_nvidia_no_rocm(monkeypatch)
    monkeypatch.setattr(ilp.platform, "system", lambda: "Linux")
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "")
    monkeypatch.setattr(ilp, "_rocm_probe_tool_present", lambda: True)
    host = ilp.detect_host()
    assert host.has_intel_gpu is True
    assert host.has_amd_gpu_without_rocm is False


def test_force_cpu_clears_the_flag():
    """--force-cpu must not leak out through the new Vulkan arm."""
    forced = ilp._apply_host_overrides(_host(has_amd_gpu_without_rocm = True), force_cpu = True)
    assert (
        forced.has_amd_gpu_without_rocm is False
    ), "force_cpu left the flag set; CPU would be bypassed"
    assert forced.has_intel_gpu is False
    assert forced.has_rocm is False


@pytest.mark.parametrize(
    "host_kwargs, expect_vulkan_eligible",
    [
        (dict(has_amd_gpu_without_rocm = True), True),  # the widening
        (dict(has_intel_gpu = True), True),  # unchanged
        (dict(has_amd_gpu_without_rocm = True, has_physical_nvidia = True), False),  # hidden CUDA card
        (dict(), False),  # headless / CPU-only
    ],
)
def test_vulkan_eligibility_gate(host_kwargs, expect_vulkan_eligible):
    """Mirrors the guard used by the Linux x86_64 dispatch branches."""
    host = _host(**host_kwargs)
    eligible = (
        host.has_intel_gpu or host.has_amd_gpu_without_rocm
    ) and not host.has_physical_nvidia
    assert eligible is expect_vulkan_eligible


def test_rocm_host_keeps_the_rocm_branch():
    """detect_host() only probes DRM vendor ids when ROCm is absent, so a working-ROCm
    host keeps the flag False and stays on the ROCm branches."""
    assert _host(has_rocm = True, rocm_gfx_target = "gfx1100").has_amd_gpu_without_rocm is False


def test_vendor_probe_is_gated_off_rocm_in_source():
    """Belt and braces on the above, without a monkeypatch that could pass vacuously:
    assert in the source that the DRM vendor scan sits inside the
    `not has_usable_nvidia and not has_rocm` guard. If it ever moves out, a working-ROCm
    host would start setting the flag and could be diverted off the ROCm branches."""
    src = Path(ilp.__file__).read_text(encoding = "utf-8")
    guard = "if not has_usable_nvidia and not has_rocm:"
    assert guard in src
    after_guard = src.index(guard)
    scan = src.index('_vendor_id == "0x1002"')
    assert scan > after_guard, "AMD vendor scan is no longer inside the no-ROCm guard"
    # ...and that nothing re-enables it later outside the guard.
    assert src.count('_vendor_id == "0x1002"') == 1
