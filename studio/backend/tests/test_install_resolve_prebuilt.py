# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""install_llama_prebuilt.py: the --resolve-prebuilt probe (plans against the fork
by default; --published-repo overrides).

These back the in-app update for source-build (markerless) installs: the backend
asks the installer whether an official prebuilt exists for this host without
downloading. Network and host detection are stubbed; no GPU or internet needed.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_studio = Path(__file__).resolve().parent.parent.parent
if str(_studio) not in sys.path:
    sys.path.insert(0, str(_studio))

ilp = importlib.import_module("install_llama_prebuilt")

if not hasattr(ilp, "resolve_simple_install_release_plans"):
    pytest.skip("PR symbols not present - check branch", allow_module_level = True)

FORK = ilp.DEFAULT_PUBLISHED_REPO  # unslothai/llama.cpp
UPSTREAM = ilp.UPSTREAM_REPO  # ggml-org/llama.cpp


def _host(**kw):
    base = dict(
        system = "Linux",
        machine = "x86_64",
        is_windows = False,
        is_linux = False,
        is_macos = False,
        is_x86_64 = False,
        is_arm64 = False,
        nvidia_smi = None,
        driver_cuda_version = None,
        compute_caps = [],
        visible_cuda_devices = None,
        has_physical_nvidia = False,
        has_usable_nvidia = False,
        has_rocm = False,
        rocm_gfx_target = None,
        macos_version = None,
    )
    base.update(kw)
    return ilp.HostInfo(**base)


def test_macos_upstream_pin_only_for_explicit_pre26_upstream():
    pre26 = _host(
        system = "Darwin",
        is_macos = True,
        is_arm64 = True,
        machine = "arm64",
        macos_version = (15, 5),
    )
    assert ilp.pinned_macos_release_tag(pre26, UPSTREAM) == "b9415"
    assert ilp.pinned_macos_release_tag(pre26, FORK) is None
    tahoe = _host(
        system = "Darwin",
        is_macos = True,
        is_arm64 = True,
        machine = "arm64",
        macos_version = (26, 0),
    )
    assert ilp.pinned_macos_release_tag(tahoe, UPSTREAM) is None


def _run_resolve(monkeypatch, capsys, plans_or_exc):
    monkeypatch.setattr(
        ilp,
        "detect_host",
        lambda: _host(system = "Darwin", is_macos = True, is_arm64 = True, machine = "arm64"),
    )

    def _resolver(tag, host, repo, published_release_tag):
        if isinstance(plans_or_exc, Exception):
            raise plans_or_exc
        return ("b9585", plans_or_exc)

    monkeypatch.setattr(ilp, "resolve_simple_install_release_plans", _resolver)
    monkeypatch.setattr(
        sys,
        "argv",
        ["install_llama_prebuilt.py", "--resolve-prebuilt", "latest", "--output-format", "json"],
    )
    rc = ilp.main()
    assert rc == ilp.EXIT_SUCCESS
    return json.loads(capsys.readouterr().out.strip().splitlines()[-1])


def test_resolve_prebuilt_available(monkeypatch, capsys):
    plan = SimpleNamespace(
        release_tag = "b9585",
        llama_tag = "b9585",
        attempts = [
            SimpleNamespace(name = "llama-b9585-bin-macos-arm64.tar.gz", install_kind = "macos-arm64")
        ],
    )
    out = _run_resolve(monkeypatch, capsys, [plan])
    assert out["prebuilt_available"] is True
    assert out["repo"] == FORK
    assert out["release_tag"] == "b9585"
    assert out["asset"] == "llama-b9585-bin-macos-arm64.tar.gz"
    assert out["install_kind"] == "macos-arm64"


def test_resolve_prebuilt_unavailable(monkeypatch, capsys):
    out = _run_resolve(monkeypatch, capsys, ilp.PrebuiltFallback("no macOS asset"))
    assert out["prebuilt_available"] is False
    assert out["repo"] == FORK


def _run_resolve_capture_host(monkeypatch, capsys):
    """Drive --resolve-prebuilt and return the host the resolver was handed."""
    seen = {}

    def _resolver(tag, host, repo, published_release_tag):
        seen["repo"] = repo
        seen["host"] = host
        raise ilp.PrebuiltFallback("no asset")

    monkeypatch.setattr(ilp, "resolve_simple_install_release_plans", _resolver)
    monkeypatch.setattr(
        sys,
        "argv",
        ["install_llama_prebuilt.py", "--resolve-prebuilt", "latest", "--output-format", "json"],
    )
    assert ilp.main() == ilp.EXIT_SUCCESS
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    return seen, out


def test_resolve_prebuilt_cpu_linux_routes_to_fork(monkeypatch, capsys):
    # CPU-only Linux host (no GPU): the dispatch routes to the fork, which now
    # ships the CPU prebuilt -- it no longer falls back to ggml-org upstream.
    monkeypatch.setattr(ilp, "detect_host", lambda: _host(is_linux = True, is_x86_64 = True))
    seen, out = _run_resolve_capture_host(monkeypatch, capsys)
    assert seen["repo"] == FORK
    assert out["repo"] == FORK


def test_resolve_prebuilt_rocm_sdk_only_host_still_offered_cpu(monkeypatch, capsys):
    # A CPU-only host that merely has ROCm/HIP SDK tools on PATH (no AMD GPU, so
    # detect_host leaves has_rocm False) is a valid CPU-prebuilt target. The probe
    # must NOT reclassify it as ROCm from tool presence alone and suppress the CPU
    # bundle -- that would deny the fork CPU prebuilt to a legitimate CPU source
    # build. The host is left CPU-only and resolves against the fork.
    monkeypatch.setattr(ilp, "detect_host", lambda: _host(is_linux = True, is_x86_64 = True))
    monkeypatch.setattr(
        ilp.shutil, "which", lambda tool: "/opt/rocm/bin/hipconfig" if tool == "hipconfig" else None
    )
    seen, out = _run_resolve_capture_host(monkeypatch, capsys)
    assert seen["repo"] == FORK
    assert seen["host"].has_rocm is False


# Blackwell floor is sm_100 (data-center B100/B200, B300/GB300), below consumer
# sm_120 -- 120 wrongly excluded data-center hosts from the prebuilt selection.


def _gpu_linux_host(caps):
    return _host(
        is_linux = True,
        is_x86_64 = True,
        has_physical_nvidia = True,
        has_usable_nvidia = True,
        driver_cuda_version = (13, 1),
        compute_caps = caps,
    )


def test_host_is_blackwell_includes_datacenter_parts():
    assert ilp._host_is_blackwell(_gpu_linux_host(["10.0"])) is True  # B200 sm_100
    assert ilp._host_is_blackwell(_gpu_linux_host(["10.3"])) is True  # B300 sm_103
    assert ilp._host_is_blackwell(_gpu_linux_host(["12.0"])) is True  # RTX 50 sm_120
    assert ilp._host_is_blackwell(_gpu_linux_host(["12.1"])) is True  # DGX Spark sm_121
    assert ilp._host_is_blackwell(_gpu_linux_host(["9.0"])) is False  # Hopper
    assert ilp._host_is_blackwell(_gpu_linux_host(["8.0"])) is False  # Ampere
    assert ilp._host_is_blackwell(_gpu_linux_host(["9.0", "10.0"])) is True  # highest cap wins


def _linux_cuda_artifact(runtime_line, supported_sms, min_sm, max_sm, profile):
    return ilp.PublishedLlamaArtifact(
        asset_name = f"app-b9739-linux-x64-{profile}.tar.gz",
        install_kind = "linux-cuda",
        runtime_line = runtime_line,
        coverage_class = "newer",
        supported_sms = supported_sms,
        min_sm = min_sm,
        max_sm = max_sm,
        bundle_profile = profile,
        rank = 50,
    )


def test_linux_blackwell_override_prefers_cuda13_for_datacenter(monkeypatch):
    # Both bundles cover sm_100 and torch reports cuda12, so coverage alone can't
    # decide -- only the sm_100 Blackwell floor lifts cuda13 to the front.
    cuda12 = _linux_cuda_artifact(
        "cuda12", ["86", "89", "90", "100", "120"], 86, 120, "cuda12-newer"
    )
    cuda13 = _linux_cuda_artifact(
        "cuda13", ["86", "89", "90", "100", "103", "120"], 86, 120, "cuda13-newer"
    )
    release = ilp.PublishedReleaseBundle(
        repo = FORK,
        release_tag = "b9739-mix",
        upstream_tag = "b9739",
        assets = {cuda12.asset_name: "https://x/cuda12", cuda13.asset_name: "https://x/cuda13"},
        artifacts = [cuda12, cuda13],
    )
    monkeypatch.setattr(
        ilp,
        "detected_linux_runtime_lines",
        lambda: (["cuda13", "cuda12"], {"cuda13": ["/usr/lib"], "cuda12": ["/usr/lib"]}),
    )

    selection = ilp.linux_cuda_choice_from_release(
        _gpu_linux_host(["10.0"]), release, preferred_runtime_line = "cuda12"
    )
    assert selection is not None
    assert selection.primary.runtime_line == "cuda13"
    assert selection.primary.bundle_profile == "cuda13-newer"


def test_drop_blackwell_incapable_windows_cuda_applies_to_datacenter():
    # B200 (sm_100) on Windows must drop the cuda-12.4 build and keep cuda13.
    host = _host(
        system = "Windows",
        is_windows = True,
        is_x86_64 = True,
        has_physical_nvidia = True,
        has_usable_nvidia = True,
        compute_caps = ["10.0"],
    )
    cuda124 = ilp.AssetChoice(
        repo = FORK,
        tag = "b9739",
        name = "llama-b9739-bin-win-cuda-12.4-x64.zip",
        url = "https://x/124",
        source_label = "published",
        install_kind = "windows-cuda",
    )
    cuda13 = ilp.AssetChoice(
        repo = FORK,
        tag = "b9739",
        name = "app-b9739-windows-x64-cuda13-newer.zip",
        url = "https://x/13",
        source_label = "published",
        install_kind = "windows-cuda",
        max_sm = 120,
    )
    kept = ilp._drop_blackwell_incapable_windows_cuda(host, [cuda124, cuda13])
    assert [a.name for a in kept] == [cuda13.name]


def test_blackwell_min_toolkit_is_sm_aware():
    # Family floor is 12.8; sm_103/sm_121 (no native target before 12.9) lift it.
    f = ilp._blackwell_min_toolkit_for_host
    assert f(_gpu_linux_host(["10.0"])) == (12, 8)  # B200
    assert f(_gpu_linux_host(["12.0"])) == (12, 8)  # RTX 50
    assert f(_gpu_linux_host(["10.3"])) == (12, 9)  # B300
    assert f(_gpu_linux_host(["12.1"])) == (12, 9)  # DGX Spark
    assert f(_gpu_linux_host(["10.0", "10.3"])) == (12, 9)  # max across SMs wins


def test_sm103_host_drops_cuda128_windows_build():
    # B300 (sm_103) needs cuda-12.9: a legacy win-cuda-12.8 build must be dropped.
    host = _host(
        system = "Windows",
        is_windows = True,
        is_x86_64 = True,
        has_physical_nvidia = True,
        has_usable_nvidia = True,
        compute_caps = ["10.3"],
    )
    cuda128 = ilp.AssetChoice(
        repo = FORK,
        tag = "b9739",
        name = "llama-b9739-bin-win-cuda-12.8-x64.zip",
        url = "https://x/128",
        source_label = "published",
        install_kind = "windows-cuda",
    )
    cuda129 = ilp.AssetChoice(
        repo = FORK,
        tag = "b9739",
        name = "llama-b9739-bin-win-cuda-12.9-x64.zip",
        url = "https://x/129",
        source_label = "published",
        install_kind = "windows-cuda",
    )
    kept = ilp._drop_blackwell_incapable_windows_cuda(host, [cuda128, cuda129])
    assert [a.name for a in kept] == [cuda129.name]
    # sm_100 stays on the 12.8 family floor and keeps the same 12.8 build.
    b200 = _host(
        system = "Windows",
        is_windows = True,
        is_x86_64 = True,
        has_physical_nvidia = True,
        has_usable_nvidia = True,
        compute_caps = ["10.0"],
    )
    kept_b200 = ilp._drop_blackwell_incapable_windows_cuda(b200, [cuda128, cuda129])
    assert [a.name for a in kept_b200] == [cuda128.name, cuda129.name]
