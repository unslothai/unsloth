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
    pytest.skip("PR symbols not present - check branch", allow_module_level = True)


def _make_rocm_host(gfx_target: str, *, windows: bool = False) -> HostInfo:
    return HostInfo(
        system = "Windows" if windows else "Linux",
        machine = "amd64" if windows else "x86_64",
        is_windows = windows,
        is_linux = not windows,
        is_macos = False,
        is_x86_64 = True,
        is_arm64 = False,
        nvidia_smi = None,
        driver_cuda_version = None,
        compute_caps = [],
        visible_cuda_devices = None,
        has_physical_nvidia = False,
        has_usable_nvidia = False,
        has_rocm = True,
        rocm_gfx_target = gfx_target,
    )


def _lookup_family(gfx: str) -> str | None:
    for prefix, family in _LEMONADE_GFX_FAMILIES:
        if gfx.startswith(prefix):
            return family
    return None


# ---------------------------------------------------------------------------
# GPU family mapping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "gfx,expected_family",
    [
        ("gfx1151", "gfx1151"),
        ("gfx1150", "gfx1150"),
        ("gfx1201", "gfx120X"),
        ("gfx1200", "gfx120X"),
        ("gfx1100", "gfx110X"),
        ("gfx1030", "gfx103X"),
    ],
)
def test_gpu_family_mapping(gfx, expected_family):
    assert _lookup_family(gfx) == expected_family


def test_unknown_gpu_not_in_families():
    assert _lookup_family("gfx999") is None


# ---------------------------------------------------------------------------
# Asset resolution - hits real lemonade GitHub API
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "gfx,os_prefix,windows",
    [
        ("gfx1151", "ubuntu", False),
        ("gfx1150", "ubuntu", False),
        ("gfx1201", "ubuntu", False),
        ("gfx1100", "ubuntu", False),
        ("gfx1030", "ubuntu", False),
        ("gfx1151", "windows", True),
        ("gfx1100", "windows", True),
    ],
)
def test_asset_resolves_for_known_gpu(gfx, os_prefix, windows):
    host = _make_rocm_host(gfx, windows = windows)
    result = resolve_lemonade_rocm_choice(
        host, os_prefix, "default", llama_tag = "latest"
    )
    assert (
        result is not None
    ), f"Installer will NOT fetch lemonade binary for {gfx} ({os_prefix})"
    assert _lookup_family(gfx) in result.name
    assert result.url.startswith("https://github.com/lemonade-sdk/llamacpp-rocm")


def test_unknown_gpu_falls_through_to_upstream():
    host = _make_rocm_host("gfx999")
    result = resolve_lemonade_rocm_choice(host, "ubuntu", "default", llama_tag = "latest")
    assert result is None


# ---------------------------------------------------------------------------
# Simple-policy dispatcher must plan a lemonade ROCm attempt for AMD-only hosts.
# This is the path setup.sh actually invokes (via --simple-policy), so the
# lemonade integration is useless if it isn't wired in here.
# ---------------------------------------------------------------------------

direct_linux_release_plan = getattr(_mod, "direct_linux_release_plan", None)
direct_upstream_release_plan = getattr(_mod, "direct_upstream_release_plan", None)


def _stub_unsloth_release(release_tag: str = "b9022") -> dict:
    # Minimal payload that parse_direct_linux_release_bundle accepts. It
    # requires at least one `app-{label}-linux-x64*.tar.gz` asset for the
    # bundle to be recognised; we ship a bare CPU one so the planner has a
    # baseline non-ROCm attempt to fall through to.
    asset_name = f"app-{release_tag}-linux-x64.tar.gz"
    return {
        "tag_name": release_tag,
        "name": release_tag,
        "assets": [
            {
                "name": asset_name,
                "browser_download_url": f"https://example.invalid/{asset_name}",
            },
        ],
    }


@pytest.mark.skipif(
    direct_linux_release_plan is None,
    reason = "simple-policy dispatcher not present on this branch",
)
def test_simple_policy_plans_lemonade_for_rocm_host():
    host = _make_rocm_host("gfx1151")
    plan = direct_linux_release_plan(
        _stub_unsloth_release(),
        host,
        "unslothai/llama.cpp",
        "latest",
    )
    assert plan is not None, "ROCm host should not be skipped by simple-policy planner"
    kinds = [a.install_kind for a in plan.attempts]
    assert (
        "linux-rocm" in kinds
    ), f"simple-policy planner did not include a lemonade ROCm attempt; got {kinds}"
    rocm_attempt = next(a for a in plan.attempts if a.install_kind == "linux-rocm")
    assert rocm_attempt.source_label == "lemonade"
    assert "gfx1151" in rocm_attempt.name


@pytest.mark.skipif(
    direct_upstream_release_plan is None,
    reason = "simple-policy dispatcher not present on this branch",
)
def test_simple_policy_plans_lemonade_for_windows_hip_host():
    host = _make_rocm_host("gfx1151", windows = True)
    release = {
        "tag_name": "b9022",
        "name": "b9022",
        "assets": [],
    }
    plan = direct_upstream_release_plan(release, host, "ggml-org/llama.cpp", "latest")
    assert plan is not None, "Windows ROCm host should plan a lemonade HIP attempt"
    kinds = [a.install_kind for a in plan.attempts]
    assert (
        "windows-hip" in kinds
    ), f"simple-policy planner did not include a lemonade HIP attempt; got {kinds}"
