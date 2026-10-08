# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""#12842: the CUDA prebuilts are toolkit 12.8+ builds whose device code ggml compresses with
-compress-mode=size, which a driver older than CUDA 12.4 cannot load ("device kernel image is
invalid" on every kernel). Neither selector may hand a cuda12 bundle to such a driver, and a
kept cuda12 install must stop counting as covering it."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

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


# Trimmed from the b11443-mix-d65395f llama-prebuilt-manifest.json, as published.
TAG = "b11443-mix-d65395f"
_PROFILES = {
    "cuda12-legacy": ("cuda12", "legacy", ["50", "52", "60", "61"], 5, "12.8"),
    "cuda12-older": ("cuda12", "older", ["70", "75", "80", "86", "89"], 10, "12.8"),
    "cuda12-portable": (
        "cuda12",
        "portable",
        ["70", "75", "80", "86", "89", "90", "100", "103", "120"],
        30,
        "12.8",
    ),
    "cuda13-older": ("cuda13", "older", ["75", "80", "86", "89"], 40, "13.3"),
    "cuda13-portable": (
        "cuda13",
        "portable",
        ["75", "80", "86", "89", "90", "100", "103", "120"],
        60,
        "13.3",
    ),
}


def _manifest_artifacts(platform, install_kind, ext):
    rows = []
    for profile, (line, coverage, sms, rank, toolkit) in _PROFILES.items():
        rows.append(
            {
                "asset_name": f"app-{TAG}-{platform}-{profile}.{ext}",
                "install_kind": install_kind,
                "bundle_profile": profile,
                "runtime_line": line,
                "coverage_class": coverage,
                "supported_sms": sms,
                "min_sm": int(sms[0]),
                "max_sm": int(sms[-1]),
                "rank": rank,
                "toolkit_version": toolkit,
            }
        )
    return rows


def _release():
    raw = _manifest_artifacts("windows-x64", "windows-cuda", "zip") + _manifest_artifacts(
        "linux-x64", "linux-cuda", "tar.gz"
    )
    artifacts = [a for a in (m.parse_published_artifact(r) for r in raw) if a is not None]
    assert len(artifacts) == len(raw)
    return m.PublishedReleaseBundle(
        repo = "unslothai/llama.cpp",
        release_tag = TAG,
        upstream_tag = "b11443",
        assets = {a.asset_name: f"https://example.com/{a.asset_name}" for a in artifacts},
        artifacts = artifacts,
    )


RELEASE = _release()


def host(
    system,
    driver,
    caps = ("86",),
):
    return m.HostInfo(
        system = system,
        machine = "AMD64" if system == "Windows" else "x86_64",
        is_windows = system == "Windows",
        is_linux = system == "Linux",
        is_macos = False,
        is_x86_64 = True,
        is_arm64 = False,
        nvidia_smi = "nvidia-smi",
        driver_cuda_version = driver,
        compute_caps = list(caps),
        physical_compute_caps = list(caps),
        visible_cuda_devices = None,
        has_physical_nvidia = True,
        has_usable_nvidia = True,
    )


@pytest.fixture(autouse = True)
def _isolated(monkeypatch):
    for key in ("UNSLOTH_LLAMA_CPP_BACKEND", "CUDA_VISIBLE_DEVICES"):
        monkeypatch.delenv(key, raising = False)
    monkeypatch.setattr(
        m, "detect_torch_cuda_runtime_preference", lambda *a, **k: m.CudaRuntimePreference(None, [])
    )
    monkeypatch.setattr(m, "detected_windows_runtime_lines", lambda: ([], {}))
    # Both runtimes on disk, so only the driver decides on Linux.
    monkeypatch.setattr(
        m,
        "detected_linux_runtime_lines",
        lambda: (["cuda13", "cuda12"], {"cuda13": ["/usr/lib"], "cuda12": ["/usr/lib"]}),
    )


def windows_profiles(driver, caps = ("86",)):
    return [
        a.bundle_profile
        for a in m.published_windows_cuda_attempts(host("Windows", driver, caps), RELEASE, None)
    ]


def linux_profiles(driver, caps = ("86",)):
    selection = m.linux_cuda_choice_from_release(host("Linux", driver, caps), RELEASE)
    return [] if selection is None else [a.bundle_profile for a in selection.attempts]


@pytest.mark.parametrize("driver", [(12, 0), (12, 2), (12, 3)])
class TestADriverBelowCuda124GetsNoCudaPrebuilt:
    def test_windows(self, driver):
        assert m.compatible_windows_runtime_lines(host("Windows", driver)) == []
        assert windows_profiles(driver) == []
        # Pascal included: the legacy bundle is a 12.8 build too.
        assert windows_profiles(driver, caps = ("61",)) == []

    def test_linux(self, driver):
        assert m.compatible_linux_runtime_lines(host("Linux", driver)) == []
        assert linux_profiles(driver) == []

    def test_the_selection_says_why(self, driver, capsys):
        m.published_windows_cuda_attempts(host("Windows", driver), RELEASE, None)
        m.linux_cuda_choice_from_release(host("Linux", driver), RELEASE)
        err = capsys.readouterr().err
        assert err.count(f"NVIDIA driver reports CUDA {driver[0]}.{driver[1]}") == 2
        assert "CUDA 12.4+" in err

    def test_a_kept_cuda12_install_no_longer_covers_the_host(self, driver):
        marker = {"backend": "cuda", "runtime_line": "cuda12", "supported_sms": ["86"]}
        assert not m._kept_install_covers_host(marker, host("Windows", driver))
        assert not m._kept_install_covers_host(marker, host("Linux", driver))


@pytest.mark.parametrize("driver", [(12, 4), (12, 8)])
class TestACuda12DriverFromTheFloorStillGetsCuda12:
    def test_windows(self, driver):
        assert windows_profiles(driver) == ["cuda12-older", "cuda12-portable"]
        assert windows_profiles(driver, caps = ("61",)) == ["cuda12-legacy"]

    def test_linux(self, driver):
        assert linux_profiles(driver) == ["cuda12-older", "cuda12-portable"]

    def test_a_kept_cuda12_install_still_covers_the_host(self, driver):
        marker = {"backend": "cuda", "runtime_line": "cuda12", "supported_sms": ["86"]}
        assert m._kept_install_covers_host(marker, host("Windows", driver))
        assert m._kept_install_covers_host(marker, host("Linux", driver))


def test_a_cuda13_driver_keeps_cuda13_first_then_cuda12():
    order = ["cuda13-older", "cuda13-portable", "cuda12-older", "cuda12-portable"]
    assert windows_profiles((13, 0)) == order
    assert linux_profiles((13, 0)) == order


def test_an_unknown_driver_is_not_gated_by_the_floor():
    assert m.driver_below_cuda_prebuilt_floor(host("Windows", None)) is False
