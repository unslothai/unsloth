# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Validates that the installer correctly resolves lemonade ROCm prebuilt assets.

Hits the real lemonade GitHub API with a faked HostInfo so no AMD GPU is needed.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

_studio = Path(__file__).resolve().parent.parent.parent
if str(_studio) not in sys.path:
    sys.path.insert(0, str(_studio))

_mod = importlib.import_module("install_llama_prebuilt")
HostInfo = _mod.HostInfo
resolve_lemonade_rocm_choice = getattr(_mod, "resolve_lemonade_rocm_choice", None)
_LEMONADE_GFX_FAMILIES = getattr(_mod, "_LEMONADE_GFX_FAMILIES", None)

if resolve_lemonade_rocm_choice is None or _LEMONADE_GFX_FAMILIES is None:
    pytest.skip("PR symbols not present - check branch", allow_module_level=True)


def _make_rocm_host(gfx_target: str, *, windows: bool = False) -> HostInfo:
    return HostInfo(
        system="Windows" if windows else "Linux",
        machine="amd64" if windows else "x86_64",
        is_windows=windows,
        is_linux=not windows,
        is_macos=False,
        is_x86_64=True,
        is_arm64=False,
        nvidia_smi=None,
        driver_cuda_version=None,
        compute_caps=[],
        visible_cuda_devices=None,
        has_physical_nvidia=False,
        has_usable_nvidia=False,
        has_rocm=True,
        rocm_gfx_target=gfx_target,
    )


def _lookup_family(gfx: str) -> str | None:
    for prefix, family in _LEMONADE_GFX_FAMILIES:
        if gfx.startswith(prefix):
            return family
    return None


# ---------------------------------------------------------------------------
# GPU family mapping
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("gfx,expected_family", [
    ("gfx1151", "gfx1151"),
    ("gfx1150", "gfx1150"),
    ("gfx1201", "gfx120X"),
    ("gfx1200", "gfx120X"),
    ("gfx1100", "gfx110X"),
    ("gfx1030", "gfx103X"),
])
def test_gpu_family_mapping(gfx, expected_family):
    assert _lookup_family(gfx) == expected_family


def test_unknown_gpu_not_in_families():
    assert _lookup_family("gfx999") is None


# ---------------------------------------------------------------------------
# Asset resolution - hits real lemonade GitHub API
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("gfx,os_prefix,windows", [
    ("gfx1151", "ubuntu", False),
    ("gfx1150", "ubuntu", False),
    ("gfx1201", "ubuntu", False),
    ("gfx1100", "ubuntu", False),
    ("gfx1030", "ubuntu", False),
    ("gfx1151", "windows", True),
    ("gfx1100", "windows", True),
])
def test_asset_resolves_for_known_gpu(gfx, os_prefix, windows):
    host = _make_rocm_host(gfx, windows=windows)
    result = resolve_lemonade_rocm_choice(host, os_prefix, "default", llama_tag="latest")
    assert result is not None, f"Installer will NOT fetch lemonade binary for {gfx} ({os_prefix})"
    assert _lookup_family(gfx) in result.name
    assert result.url.startswith("https://github.com/lemonade-sdk/llamacpp-rocm")


def test_unknown_gpu_falls_through_to_upstream():
    host = _make_rocm_host("gfx999")
    result = resolve_lemonade_rocm_choice(host, "ubuntu", "default", llama_tag="latest")
    assert result is None
