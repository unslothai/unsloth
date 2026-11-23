# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""A CUDA_VISIBLE_DEVICES mask that hides a card of another SM must not narrow the llama.cpp
CUDA bundle to the visible cards: the install outlives the mask. Torch and setup.sh already
target every physical GPU. When no bundle covers them all, the visible-only choice stands."""

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


def host(
    system,
    caps,
    physical,
    visible = "1",
    driver = (12, 8),
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
        physical_compute_caps = list(physical),
        visible_cuda_devices = visible,
        has_physical_nvidia = True,
        has_usable_nvidia = True,
    )


def artifact(
    name,
    kind,
    line,
    profile,
    sms,
    coverage = "targeted",
    rank = 100,
):
    caps = [str(s) for s in sms]
    return m.PublishedLlamaArtifact(
        asset_name = name,
        install_kind = kind,
        runtime_line = line,
        coverage_class = coverage,
        supported_sms = caps,
        min_sm = int(caps[0]),
        max_sm = int(caps[-1]),
        bundle_profile = profile,
        rank = rank,
    )


def release(artifacts):
    return m.PublishedReleaseBundle(
        repo = "unslothai/llama.cpp",
        release_tag = "v1.0",
        upstream_tag = "b8508",
        assets = {a.asset_name: f"https://example.com/{a.asset_name}" for a in artifacts},
        artifacts = list(artifacts),
    )


LINUX_NEWER = artifact(
    "app-b8508-linux-x64-cuda12-newer.tar.gz", "linux-cuda", "cuda12", "cuda12-newer", [86, 89, 90]
)
LINUX_WIDE = artifact(
    "app-b8508-linux-x64-cuda12-wide.tar.gz",
    "linux-cuda",
    "cuda12",
    "cuda12-wide",
    [61, 70, 75, 80, 86, 89, 90],
    rank = 200,
)
WIN_NEWER = artifact(
    "llama-app-win-cuda12-newer-x64.zip", "windows-cuda", "cuda12", "cuda12-newer", [86, 89, 90]
)
WIN_WIDE = artifact(
    "llama-app-win-cuda12-wide-x64.zip",
    "windows-cuda",
    "cuda12",
    "cuda12-wide",
    [61, 70, 75, 80, 86, 89, 90],
    rank = 200,
)


@pytest.fixture(autouse = True)
def _isolated(monkeypatch):
    monkeypatch.setattr(m, "_UNCOVERED_CUDA_HOST_WARNINGS", set())
    monkeypatch.setattr(
        m, "detected_linux_runtime_lines", lambda: (["cuda12"], {"cuda12": ["/usr/lib"]})
    )
    monkeypatch.setattr(m, "detected_windows_runtime_lines", lambda: (["cuda12"], {}))


# GTX 1080 (sm_61) + RTX 4090 (sm_89), CUDA_VISIBLE_DEVICES=1 leaves only the 4090 visible.
MASKED_LINUX = host("Linux", ["89"], ["61", "89"])
MASKED_WINDOWS = host("Windows", ["89"], ["61", "89"])


class TestLinux:
    def test_the_bundle_covers_the_card_the_mask_hides(self):
        selection = m.linux_cuda_choice_from_release(
            MASKED_LINUX, release([LINUX_NEWER, LINUX_WIDE])
        )
        assert selection is not None
        # The visible-only pick stays as the fallback attempt.
        assert [a.bundle_profile for a in selection.attempts] == ["cuda12-wide", "cuda12-newer"]
        assert any("physical_sms=61,89" in line for line in selection.selection_log)

    def test_no_bundle_for_every_card_keeps_the_visible_choice_without_a_warning(self, monkeypatch):
        warnings = []
        monkeypatch.setattr(m, "log", warnings.append)
        selection = m.linux_cuda_choice_from_release(MASKED_LINUX, release([LINUX_NEWER]))
        assert selection is not None
        assert [a.bundle_profile for a in selection.attempts] == ["cuda12-newer"]
        assert any("selecting for the visible GPUs" in line for line in selection.selection_log)
        assert not any("no published CUDA bundle covers" in w for w in warnings)

    def test_the_published_attempts_path_takes_the_wide_bundle(self, monkeypatch):
        monkeypatch.setattr(
            m,
            "detect_torch_cuda_runtime_preference",
            lambda *a, **k: m.CudaRuntimePreference(None, []),
        )
        attempts = m._linux_published_attempts(MASKED_LINUX, release([LINUX_NEWER, LINUX_WIDE]))
        assert [a.bundle_profile for a in attempts] == ["cuda12-wide", "cuda12-newer"]


class TestWindows:
    def test_the_bundle_covers_the_card_the_mask_hides(self):
        attempts = m.published_windows_cuda_attempts(
            MASKED_WINDOWS, release([WIN_NEWER, WIN_WIDE]), None
        )
        assert [a.bundle_profile for a in attempts] == ["cuda12-wide", "cuda12-newer"]

    def test_no_bundle_for_every_card_keeps_the_visible_choice(self):
        attempts = m.published_windows_cuda_attempts(MASKED_WINDOWS, release([WIN_NEWER]), None)
        assert [a.bundle_profile for a in attempts] == ["cuda12-newer"]
        assert any("selecting for the visible GPUs" in line for line in attempts[0].selection_log)

    def test_unknown_coverage_is_not_a_widening(self):
        # A bundle without SM metadata cannot prove it covers the hidden card, so the visible pick stays.
        unknown = m.PublishedLlamaArtifact(
            asset_name = "llama-app-win-cuda12-unknown-x64.zip",
            install_kind = "windows-cuda",
            runtime_line = "cuda12",
            coverage_class = "targeted",
            supported_sms = [],
            min_sm = None,
            max_sm = None,
            bundle_profile = "cuda12-unknown",
            rank = 50,
        )
        attempts = m.published_windows_cuda_attempts(
            MASKED_WINDOWS, release([WIN_NEWER, unknown]), None
        )
        assert "cuda12-newer" in [a.bundle_profile for a in attempts]


class TestKeptInstallProfile:
    def test_a_masked_mixed_host_records_its_physical_caps(self):
        assert m.host_profile(MASKED_LINUX)["physical_compute_caps"] == ["61", "89"]

    @pytest.mark.parametrize("physical", [["89"], []])
    def test_other_hosts_keep_their_profile_keys(self, physical):
        assert "physical_compute_caps" not in m.host_profile(host("Linux", ["89"], physical))


@pytest.mark.parametrize("system", ["Linux", "Windows"])
def test_unknown_visible_caps_keep_the_unknown_gpu_safeguard(system):
    # An empty or unresolvable mask hides which card runs: no widening, main's safeguard decides.
    assert m._mask_widened_sms(host(system, [], ["61", "89"], visible = "")) is None
