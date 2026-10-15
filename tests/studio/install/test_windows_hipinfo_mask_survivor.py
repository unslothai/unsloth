# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""Windows hipinfo is a HIP application, so HIP_VISIBLE_DEVICES / CUDA_VISIBLE_DEVICES have
already filtered and renumbered what it lists. Indexing that list with the mask again picked
the wrong card: three AMD GPUs with HIP_VISIBLE_DEVICES=1,2 list two devices, and index 1
named physical GPU 2 instead of GPU 1. amd-smi is not filtered and keeps the mask index."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = PACKAGE_ROOT / "studio" / "install_llama_prebuilt.py"
MODULE_NAME = "studio_install_llama_prebuilt"
if MODULE_NAME in sys.modules:
    m = sys.modules[MODULE_NAME]
else:
    SPEC = importlib.util.spec_from_file_location(MODULE_NAME, MODULE_PATH)
    assert SPEC is not None and SPEC.loader is not None
    m = importlib.util.module_from_spec(SPEC)
    sys.modules[MODULE_NAME] = m
    SPEC.loader.exec_module(m)

PHYSICAL = ["gfx1036", "gfx1201", "gfx1100"]


def hipinfo_listing(archs):
    return "".join(
        f"device#                           {i}\nName: AMD GPU\ngcnArchName: {arch}\n"
        for i, arch in enumerate(archs)
    )


def amd_smi_listing(archs):
    return "".join(f"GPU: {i}\n    TARGET_GRAPHICS_VERSION: {arch}\n" for i, arch in enumerate(archs))


def detect(monkeypatch, probe, stdout, env):
    monkeypatch.setattr(m.platform, "system", lambda: "Windows")
    monkeypatch.setattr(m.platform, "machine", lambda: "AMD64")
    for var in (
        "CUDA_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "HIP_PATH",
        "ROCM_PATH",
        "UNSLOTH_ENABLE_AMD_SMI",
        "UNSLOTH_ROCM_GFX_ARCH",
    ):
        monkeypatch.delenv(var, raising = False)
    for var, value in env.items():
        monkeypatch.setenv(var, value)
    if probe == "amd-smi":
        monkeypatch.setenv("UNSLOTH_ENABLE_AMD_SMI", "1")
    monkeypatch.setattr(m.shutil, "which", lambda name: name if name == probe else None)
    monkeypatch.setattr(m, "nvidia_library_inventory", lambda *a, **k: None)

    def fake_run_capture(command, **kwargs):
        if command[0] == probe:
            return SimpleNamespace(returncode = 0, stdout = stdout, stderr = "")
        return SimpleNamespace(returncode = 1, stdout = "", stderr = "")

    monkeypatch.setattr(m, "run_capture", fake_run_capture)
    return m.detect_host()


@pytest.mark.parametrize(
    "env",
    [{"HIP_VISIBLE_DEVICES": "1,2"}, {"CUDA_VISIBLE_DEVICES": "1,2"}],
)
def test_hipinfo_under_a_mask_picks_the_first_listed_device(monkeypatch, env):
    host = detect(monkeypatch, "hipinfo", hipinfo_listing(PHYSICAL[1:]), env)
    assert host.has_rocm
    assert host.rocm_gfx_targets == ["gfx1201", "gfx1100"]
    assert host.rocm_gfx_target == "gfx1201"
    # setup.ps1 forwards the same card, so the probe and the forward now agree.
    routed = m._apply_host_overrides(host, override_rocm_gfx = "gfx1201")
    assert m._active_rocm_gfx_target(routed) == "gfx1201"
    # A mask still blocks the automatic Vulkan route.
    assert m._should_auto_vulkan_for_amd_windows(routed) is False
    # amd-smi lists every physical device, so the mask index still applies there.
    unfiltered = detect(monkeypatch, "amd-smi", amd_smi_listing(PHYSICAL), env)
    assert unfiltered.rocm_gfx_target == "gfx1201"
