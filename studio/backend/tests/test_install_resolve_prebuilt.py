# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""install_llama_prebuilt.py: the --resolve-prebuilt probe (plans against the fork
by default; --published-repo overrides).

These back the in-app update for source-build (markerless) installs: the backend
asks the installer whether an official prebuilt exists for this host without
downloading. Network and host detection are stubbed; no GPU or internet needed. The one
exception is the windows-rocm floor guard, which reads the fork's published manifest
because nothing in-tree mirrors it, and skips when that release is unreachable.
"""

from __future__ import annotations

import dataclasses
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


@pytest.fixture(autouse = True)
def _no_ambient_hip_device_mask(monkeypatch):
    """These tests describe hosts through HostInfo, not through the environment.

    A mask inherited from the shell (ML boxes commonly export CUDA_VISIBLE_DEVICES) means
    the arch probe saw only part of the GPUs, which the Windows auto-Vulkan guard treats as
    an unknown physical inventory. Clear all three so a host is described by its fields
    alone; the tests that are about the mask set it explicitly."""
    for _env in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        monkeypatch.delenv(_env, raising = False)


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


def test_force_cpu_clears_all_gpu_attributes_including_intel():
    # --cpu-fallback is the "select the CPU prebuilt even when a GPU is present"
    # escape hatch. It must drop EVERY GPU attribute, including has_intel_gpu, or
    # the planner still prepends the Vulkan asset on an Intel-GPU host.
    host = _host(
        is_linux = True,
        is_x86_64 = True,
        has_usable_nvidia = True,
        has_physical_nvidia = True,
        has_rocm = True,
        rocm_gfx_target = "gfx1100",
        has_intel_gpu = True,
    )
    forced = ilp._apply_host_overrides(host, force_cpu = True)
    assert forced.has_usable_nvidia is False
    assert forced.has_physical_nvidia is False
    assert forced.has_rocm is False
    assert forced.rocm_gfx_target is None
    assert forced.has_intel_gpu is False


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


def _run_resolve(
    monkeypatch,
    capsys,
    plans_or_exc,
    *,
    host = None,
    extra_args = (),
):
    monkeypatch.setattr(
        ilp,
        "detect_host",
        lambda: host or _host(system = "Darwin", is_macos = True, is_arm64 = True, machine = "arm64"),
    )

    def _resolver(tag, host, repo, published_release_tag):
        if isinstance(plans_or_exc, Exception):
            raise plans_or_exc
        return ("b9585", plans_or_exc)

    monkeypatch.setattr(ilp, "resolve_simple_install_release_plans", _resolver)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "install_llama_prebuilt.py",
            "--resolve-prebuilt",
            "latest",
            "--output-format",
            "json",
            *extra_args,
        ],
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


def test_resolve_prebuilt_backend_flag_filters_cpu_fallback(monkeypatch, capsys):
    cpu_plan = ilp.InstallReleasePlan(
        requested_tag = "latest",
        llama_tag = "b9585",
        release_tag = "release",
        attempts = [
            ilp.AssetChoice(
                repo = FORK,
                tag = "release",
                name = "app-release-linux-x64-cpu",
                url = "https://example/cpu",
                source_label = "published",
                install_kind = "linux-cpu",
            )
        ],
        approved_checksums = SimpleNamespace(),
    )
    out = _run_resolve(
        monkeypatch,
        capsys,
        [cpu_plan],
        host = _host(is_linux = True, is_x86_64 = True),
        extra_args = ("--llama-backend", "vulkan"),
    )
    assert out["prebuilt_available"] is False


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


# _host_is_blackwell / _blackwell_min_toolkit_for_host are prebuilt_core
# re-exports; their value tables moved verbatim to
# tests/studio/install/test_prebuilt_core.py.


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


def _upstream_release(tag, asset_names):
    return {
        "tag_name": tag,
        "assets": [
            {"name": n, "browser_download_url": f"https://example/{n}"} for n in asset_names
        ],
    }


def test_direct_upstream_arm64_intel_prefers_vulkan():
    # Auto-detected Intel GPU on Linux arm64 -> Vulkan prebuilt first, CPU
    # second (mirrors the x86_64 branch; ggml-org ships the arm64 Vulkan asset).
    host = _host(is_linux = True, is_arm64 = True, machine = "aarch64", has_intel_gpu = True)
    rel = _upstream_release(
        "b9925",
        ["llama-b9925-bin-ubuntu-vulkan-arm64.tar.gz", "llama-b9925-bin-ubuntu-arm64.tar.gz"],
    )
    plan = ilp.direct_upstream_release_plan(rel, host, UPSTREAM, "latest")
    kinds = [a.install_kind for a in plan.attempts]
    assert kinds[0] == "linux-vulkan", kinds
    assert "linux-arm64" in kinds
    assert plan.attempts[0].name == "llama-b9925-bin-ubuntu-vulkan-arm64.tar.gz"


def test_direct_upstream_intel_with_hidden_nvidia_is_cpu_only():
    # A host with a physical NVIDIA hidden via CUDA_VISIBLE_DEVICES (physical
    # True, usable False) + an Intel iGPU must NOT get the Vulkan archive even
    # when planning directly against upstream: Vulkan ignores CUDA_VISIBLE_DEVICES
    # and could grab the reserved card. It falls through to the CPU asset.
    host = _host(
        is_linux = True,
        is_x86_64 = True,
        has_intel_gpu = True,
        has_physical_nvidia = True,
        has_usable_nvidia = False,
    )
    rel = _upstream_release(
        "b9925",
        ["llama-b9925-bin-ubuntu-vulkan-x64.tar.gz", "llama-b9925-bin-ubuntu-x64.tar.gz"],
    )
    plan = ilp.direct_upstream_release_plan(rel, host, UPSTREAM, "latest")
    assert [a.install_kind for a in plan.attempts] == ["linux-cpu"]


def test_direct_upstream_arm64_without_intel_is_cpu_only():
    host = _host(is_linux = True, is_arm64 = True, machine = "aarch64")
    rel = _upstream_release(
        "b9925",
        ["llama-b9925-bin-ubuntu-vulkan-arm64.tar.gz", "llama-b9925-bin-ubuntu-arm64.tar.gz"],
    )
    plan = ilp.direct_upstream_release_plan(rel, host, UPSTREAM, "latest")
    assert [a.install_kind for a in plan.attempts] == ["linux-arm64"]


def test_direct_upstream_x86_intel_prefers_vulkan():
    host = _host(is_linux = True, is_x86_64 = True, has_intel_gpu = True)
    rel = _upstream_release(
        "b9925",
        ["llama-b9925-bin-ubuntu-vulkan-x64.tar.gz", "llama-b9925-bin-ubuntu-x64.tar.gz"],
    )
    plan = ilp.direct_upstream_release_plan(rel, host, UPSTREAM, "latest")
    kinds = [a.install_kind for a in plan.attempts]
    assert kinds[0] == "linux-vulkan", kinds
    assert "linux-cpu" in kinds


def _published_vulkan_bundle(*install_kinds):
    profiles = {
        "linux-vulkan": "linux-vulkan-x64",
        "windows-vulkan": "windows-vulkan-x64",
        "linux-cpu": "linux-cpu-x64",
        "linux-arm64": "linux-cpu-arm64",
        "windows-cpu": "windows-cpu-x64",
    }
    artifacts = [
        ilp.PublishedLlamaArtifact(
            asset_name = f"app-release-{install_kind}",
            install_kind = install_kind,
            runtime_line = None,
            coverage_class = None,
            supported_sms = [],
            min_sm = None,
            max_sm = None,
            bundle_profile = profiles.get(install_kind),
            rank = 60,
        )
        for install_kind in install_kinds
    ]
    return ilp.PublishedReleaseBundle(
        repo = FORK,
        release_tag = "release",
        upstream_tag = "b9925",
        assets = {
            artifact.asset_name: f"https://example/{artifact.asset_name}" for artifact in artifacts
        },
        artifacts = artifacts,
    )


def test_fork_linux_intel_prefers_published_vulkan_bundle():
    bundle = _published_vulkan_bundle("linux-vulkan", "linux-cpu")
    attempts = ilp._linux_published_attempts(
        _host(is_linux = True, is_x86_64 = True, has_intel_gpu = True),
        bundle,
    )
    assert [attempt.install_kind for attempt in attempts] == ["linux-vulkan", "linux-cpu"]
    assert attempts[0].source_label == "published"
    assert ["llama-diffusion-gemma-visual-server"] in ilp.runtime_payload_health_groups(attempts[0])


def test_fork_linux_arm64_does_not_select_x64_vulkan_bundle():
    bundle = _published_vulkan_bundle("linux-vulkan", "linux-arm64")
    attempts = ilp._linux_published_attempts(
        _host(
            machine = "aarch64",
            is_linux = True,
            is_arm64 = True,
            has_intel_gpu = True,
        ),
        bundle,
    )

    assert [attempt.install_kind for attempt in attempts] == ["linux-arm64"]


def test_fork_windows_intel_prefers_published_vulkan_bundle():
    bundle = _published_vulkan_bundle("windows-vulkan", "windows-cpu")
    checksums = ilp.ApprovedReleaseChecksums(
        repo = FORK,
        release_tag = bundle.release_tag,
        upstream_tag = bundle.upstream_tag,
        artifacts = {
            artifact.asset_name: ilp.ApprovedArtifactHash(
                asset_name = artifact.asset_name,
                sha256 = "a" * 64,
                repo = FORK,
                kind = artifact.install_kind,
            )
            for artifact in bundle.artifacts
        },
    )
    attempts = ilp.resolve_release_asset_choice(
        _host(
            system = "Windows",
            is_windows = True,
            is_x86_64 = True,
            has_intel_gpu = True,
        ),
        bundle.upstream_tag,
        bundle,
        checksums,
    )
    assert [attempt.install_kind for attempt in attempts] == ["windows-vulkan", "windows-cpu"]
    assert attempts[0].source_label == "published"
    assert ["llama-diffusion-gemma-visual-server.exe"] in ilp.runtime_payload_health_groups(
        attempts[0]
    )


def test_linux_vulkan_health_glob_matches_bare_cpu_lib():
    # The widened glob must cover both arch-suffixed (x64) and bare (arm64) CPU
    # libs so a valid Vulkan install is not re-flagged unhealthy every check.
    choice = ilp.AssetChoice(
        repo = UPSTREAM,
        tag = "b9925",
        name = "llama-b9925-bin-ubuntu-vulkan-arm64.tar.gz",
        url = "https://example/x",
        source_label = "upstream",
        install_kind = "linux-vulkan",
    )
    groups = ilp.runtime_payload_health_groups(choice)
    assert ["libggml-cpu*.so*"] in groups
    assert ["libggml-cpu-*.so*"] not in groups


@pytest.mark.parametrize(
    "backend, legacy, expected_backend, expected",
    [
        (None, None, None, False),
        ("vulkan", None, "vulkan", True),
        (" VULKAN ", None, "vulkan", True),
        ("auto", "1", None, True),
        (None, "true", None, True),
        (None, "on", None, True),
        ("auto", None, None, False),
        ("cpu", "0", "cpu", False),
        ("hip", "1", "hip", False),
        ("rocm", "on", "hip", False),
        (" HIP ", "1", "hip", False),
        (" ROCM ", "1", "hip", False),
    ],
)
def test_force_vulkan_requested_accepts_public_selector_and_legacy_alias(
    monkeypatch, backend, legacy, expected_backend, expected
):
    # UNSLOTH_LLAMA_CPP_BACKEND is the public selector; a recognized non-vulkan
    # value is authoritative, so it opts out even against a stale legacy alias.
    # auto/unset/unknown leave the backend unpinned so selection stays automatic.
    for name, value in (
        ("UNSLOTH_LLAMA_CPP_BACKEND", backend),
        ("UNSLOTH_FORCE_VULKAN", legacy),
    ):
        if value is None:
            monkeypatch.delenv(name, raising = False)
        else:
            monkeypatch.setenv(name, value)
    assert ilp.llama_backend_from_env() == expected_backend
    assert ilp.force_vulkan_requested() is expected


def test_forced_vulkan_filters_cpu_fallback_before_validation():
    vulkan = ilp.AssetChoice(
        repo = UPSTREAM,
        tag = "b9925",
        name = "llama-b9925-bin-ubuntu-vulkan-x64.tar.gz",
        url = "https://example/vulkan",
        source_label = "upstream",
        install_kind = "linux-vulkan",
    )
    cpu = ilp.AssetChoice(
        repo = UPSTREAM,
        tag = "b9925",
        name = "llama-b9925-bin-ubuntu-x64.tar.gz",
        url = "https://example/cpu",
        source_label = "upstream",
        install_kind = "linux-cpu",
    )
    assert ilp._vulkan_only_attempts([vulkan, cpu]) == [vulkan]


def test_forced_vulkan_skips_cpu_only_release_plans():
    vulkan = ilp.AssetChoice(
        repo = UPSTREAM,
        tag = "b9924",
        name = "llama-b9924-bin-ubuntu-vulkan-x64.tar.gz",
        url = "https://example/vulkan",
        source_label = "upstream",
        install_kind = "linux-vulkan",
    )
    cpu = ilp.AssetChoice(
        repo = UPSTREAM,
        tag = "b9925",
        name = "llama-b9925-bin-ubuntu-x64.tar.gz",
        url = "https://example/cpu",
        source_label = "upstream",
        install_kind = "linux-cpu",
    )
    plans = [
        ilp.InstallReleasePlan("latest", "b9925", "b9925", [cpu], SimpleNamespace()),
        ilp.InstallReleasePlan("latest", "b9924", "b9924", [vulkan], SimpleNamespace()),
    ]

    filtered = ilp._vulkan_only_release_plans(plans)

    assert [plan.release_tag for plan in filtered] == ["b9924"]
    assert filtered[0].attempts == [vulkan]


def test_route_to_vulkan_prebuilt_auto_intel_keeps_fork_release():
    # The fork now publishes a verified Vulkan app bundle that includes the
    # DiffusionGemma runner, so keep its release namespace and pin.
    host = _host(is_linux = True, is_x86_64 = True, has_intel_gpu = True)
    routed, repo, tag, _persist = ilp._route_to_vulkan_prebuilt(
        host, FORK, "b9596-mix-abc", force_cpu = False
    )
    assert repo == FORK
    assert tag == "b9596-mix-abc"
    assert routed.has_intel_gpu is True


def test_route_to_vulkan_prebuilt_preserves_explicit_upstream_pin():
    # A pin set WITH an explicit upstream repo is already on upstream -> kept.
    host = _host(is_linux = True, is_x86_64 = True, has_intel_gpu = True)
    _routed, repo, tag, _persist = ilp._route_to_vulkan_prebuilt(
        host, UPSTREAM, "b9596", force_cpu = False
    )
    assert repo == UPSTREAM
    assert tag == "b9596"


def _linux_arm64_vulkan_host():
    return _host(
        system = "Linux",
        machine = "aarch64",
        is_linux = True,
        is_arm64 = True,
        has_intel_gpu = True,
    )


def test_route_to_vulkan_prebuilt_linux_arm64_falls_back_to_upstream():
    # The fork ships no ARM64 Vulkan bundle, so a forced-Vulkan Linux ARM64 host
    # must keep planning against upstream (which publishes
    # llama-<tag>-bin-ubuntu-vulkan-arm64.tar.gz). The fork pin is dropped because
    # the two repos use different tag namespaces.
    routed, repo, tag, persist = ilp._route_to_vulkan_prebuilt(
        _linux_arm64_vulkan_host(),
        FORK,
        "b9596-mix-abc",
        force_cpu = False,
        llama_backend = "vulkan",
    )

    assert repo == UPSTREAM
    assert tag == ""
    assert persist == "vulkan"
    assert routed.has_intel_gpu is True


def test_forced_vulkan_linux_arm64_still_resolves_a_vulkan_bundle():
    # End to end for the routing above: without it the fork planner yields only the
    # ARM64 CPU attempt, and the strict-Vulkan filter then leaves nothing to install.
    routed, repo, _tag, _persist = ilp._route_to_vulkan_prebuilt(
        _linux_arm64_vulkan_host(),
        FORK,
        "",
        force_cpu = False,
        llama_backend = "vulkan",
    )
    fork_attempts = ilp._linux_published_attempts(
        routed, _published_vulkan_bundle("linux-vulkan", "linux-arm64")
    )
    assert [attempt.install_kind for attempt in fork_attempts] == ["linux-arm64"]

    release = _upstream_release(
        "b9925",
        ["llama-b9925-bin-ubuntu-vulkan-arm64.tar.gz", "llama-b9925-bin-ubuntu-arm64.tar.gz"],
    )
    plan = ilp.direct_upstream_release_plan(release, routed, repo, "latest")
    filtered = ilp._vulkan_only_release_plans([plan])

    assert [attempt.name for attempt in filtered[0].attempts] == [
        "llama-b9925-bin-ubuntu-vulkan-arm64.tar.gz"
    ]
    assert filtered[0].attempts[0].repo == UPSTREAM


def test_route_to_vulkan_prebuilt_linux_arm64_keeps_an_explicit_repo_override():
    # A hand-picked --published-repo is the user's call; only the default fork
    # repo is rerouted.
    other = "someone/llama.cpp"
    _routed, repo, tag, _persist = ilp._route_to_vulkan_prebuilt(
        _linux_arm64_vulkan_host(),
        other,
        "b9596",
        force_cpu = False,
        llama_backend = "vulkan",
    )

    assert repo == other
    assert tag == "b9596"


def test_route_to_vulkan_prebuilt_linux_x86_64_keeps_the_fork_bundle():
    # Only ARM64 lacks a fork Vulkan bundle; x86_64 must stay on the fork release
    # so it keeps the DiffusionGemma visual server.
    _routed, repo, tag, _persist = ilp._route_to_vulkan_prebuilt(
        _host(is_linux = True, is_x86_64 = True, has_intel_gpu = True),
        FORK,
        "b9596-mix-abc",
        force_cpu = False,
        llama_backend = "vulkan",
    )

    assert repo == FORK
    assert tag == "b9596-mix-abc"


def test_route_to_vulkan_prebuilt_cpu_fallback_wins():
    # --cpu-fallback suppresses Vulkan routing even for an Intel host.
    host = _host(is_linux = True, is_x86_64 = True, has_intel_gpu = True)
    routed, repo, tag, _persist = ilp._route_to_vulkan_prebuilt(
        host, FORK, "b9596-mix-abc", force_cpu = True
    )
    assert repo == FORK
    assert tag == "b9596-mix-abc"
    assert routed is host


@pytest.mark.parametrize("via", ["env", "argument"])
@pytest.mark.parametrize("backend", ["hip", "rocm", "cpu"])
def test_non_vulkan_backend_suppresses_intel_auto_route(monkeypatch, backend, via):
    # The selector suppresses the Intel auto-route the same way whether it arrives
    # from the environment or from --llama-backend. Pinned to Linux ARM64, the one
    # host where the suppression shows up in a returned VALUE: without it the Intel
    # auto-route fires, falls through to the ARM64 fork -> upstream reroute and
    # hands back UPSTREAM with the fork pin dropped. On x86_64 the auto-route
    # rewrites neither host nor repo, so only a log line differs and the same case
    # there cannot fail.
    kwargs = {}
    if via == "env":
        monkeypatch.setenv("UNSLOTH_LLAMA_CPP_BACKEND", backend)
    else:
        kwargs["llama_backend"] = backend
    host = _linux_arm64_vulkan_host()

    routed, repo, tag, persist = ilp._route_to_vulkan_prebuilt(
        host, FORK, "b9596-mix-abc", force_cpu = False, **kwargs
    )

    assert routed is host
    assert repo == FORK, repo
    assert tag == "b9596-mix-abc", tag
    assert persist is None


@pytest.mark.parametrize("cpu_flag", ["--cpu-fallback", "--force-cpu"])
def test_resolve_prebuilt_cpu_fallback_overrides_intel_vulkan(monkeypatch, capsys, cpu_flag):
    """Either CPU flag via CLI must suppress Vulkan even on an Intel GPU host: both
    drop GPU detection (--force-cpu additionally persists, on the install path)."""
    monkeypatch.setattr(
        ilp,
        "detect_host",
        lambda: _host(is_linux = True, is_x86_64 = True, has_intel_gpu = True),
    )
    seen = {}

    def _resolver(tag, host, repo, published_release_tag):
        seen["host"] = host
        seen["repo"] = repo
        raise ilp.PrebuiltFallback("no asset")

    monkeypatch.setattr(ilp, "resolve_simple_install_release_plans", _resolver)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "install_llama_prebuilt.py",
            "--resolve-prebuilt",
            "latest",
            cpu_flag,
            "--output-format",
            "json",
        ],
    )
    assert ilp.main() == ilp.EXIT_SUCCESS
    # The CPU flag must suppress Intel GPU, route to fork (not upstream Vulkan)
    assert seen["host"].has_intel_gpu is False
    assert seen["repo"] == FORK


def test_cpu_backend_env_forces_cpu_in_resolver(monkeypatch, capsys):
    monkeypatch.setenv("UNSLOTH_LLAMA_CPP_BACKEND", "cpu")
    monkeypatch.setattr(
        ilp,
        "detect_host",
        lambda: _host(is_linux = True, is_x86_64 = True, has_intel_gpu = True),
    )
    seen, _ = _run_resolve_capture_host(monkeypatch, capsys)
    assert seen["host"].has_intel_gpu is False
    assert seen["repo"] == FORK


def test_cpu_backend_env_forces_cpu_in_direct_install(monkeypatch, tmp_path):
    monkeypatch.setenv("UNSLOTH_LLAMA_CPP_BACKEND", "cpu")
    monkeypatch.setattr(
        ilp,
        "detect_host",
        lambda: _host(is_linux = True, is_x86_64 = True, has_intel_gpu = True),
    )
    seen = {}

    def _route(
        host,
        repo,
        tag,
        *,
        force_cpu,
        llama_backend = None,
    ):
        seen["host"] = host
        seen["force_cpu"] = force_cpu
        raise RuntimeError("stop after routing")

    monkeypatch.setattr(ilp, "_route_to_vulkan_prebuilt", _route)
    with pytest.raises(RuntimeError, match = "stop after routing"):
        ilp.install_prebuilt(tmp_path, "latest", FORK, "")

    assert seen["host"].has_intel_gpu is False
    assert seen["force_cpu"] is True


@pytest.mark.parametrize(
    "flags, expect_force, expect_persist",
    [
        ([], False, False),
        # Automatic/transient last resort (arm64 GPU-build recovery): drops GPU but
        # does NOT persist, so a later update heals to a GPU bundle (#6097).
        (["--cpu-fallback"], True, False),
        # Deliberate CPU-only (UNSLOTH_LLAMA_CPP_BACKEND=cpu): drops GPU AND persists so
        # the updater re-asserts it and never revives the Intel iGPU crash (#7213).
        (["--force-cpu"], True, True),
        (["--cpu-fallback", "--force-cpu"], True, True),
    ],
)
def test_cli_cpu_flags_thread_force_and_persist(
    monkeypatch, tmp_path, flags, expect_force, expect_persist
):
    captured = {}
    monkeypatch.setattr(ilp, "install_prebuilt", lambda **kw: captured.update(kw))
    monkeypatch.setattr(
        sys,
        "argv",
        ["install_llama_prebuilt.py", "--install-dir", str(tmp_path / "llama.cpp"), *flags],
    )
    assert ilp.main() == ilp.EXIT_SUCCESS
    assert captured["force_cpu"] is expect_force
    assert captured["persist_force_cpu"] is expect_persist


@pytest.mark.parametrize(
    "existing, requested, expected",
    [
        # A deliberate --force-cpu on top of a naturally-installed CPU bundle (same
        # asset, install skipped) must still flip the marker to true (#7213).
        (False, True, True),
        (None, True, True),
        # No spurious writes when already in sync, and a released force syncs down.
        (True, True, True),
        (False, False, False),
        (True, False, False),
    ],
)
def test_sync_marker_force_cpu(tmp_path, existing, requested, expected):
    marker = {"tag": "b9585", "asset": "llama-b9585-bin-ubuntu-x64.tar.gz"}
    if existing is not None:
        marker["force_cpu"] = existing
    marker_path = tmp_path / "UNSLOTH_PREBUILT_INFO.json"
    marker_path.write_text(json.dumps(marker))
    ilp.sync_marker_force_cpu(tmp_path, requested)
    written = json.loads(marker_path.read_text())
    assert written["force_cpu"] is expected
    # Unrelated fields are preserved.
    assert written["asset"] == "llama-b9585-bin-ubuntu-x64.tar.gz"


def test_sync_marker_force_cpu_missing_marker_is_noop(tmp_path):
    # No marker (or unreadable) must not crash the reuse path.
    ilp.sync_marker_force_cpu(tmp_path, True)
    assert not (tmp_path / "UNSLOTH_PREBUILT_INFO.json").exists()


def test_route_to_vulkan_prebuilt_hidden_nvidia_not_rerouted():
    # A mixed NVIDIA+Intel host that hid NVIDIA (CUDA_VISIBLE_DEVICES=""/-1):
    # physical NVIDIA present but not usable. Must NOT auto-route to Vulkan, or
    # Vulkan (which ignores CUDA_VISIBLE_DEVICES) could grab the reserved GPU.
    host = _host(
        is_linux = True,
        is_x86_64 = True,
        has_intel_gpu = True,
        has_physical_nvidia = True,
        has_usable_nvidia = False,
    )
    _routed, repo, _tag, _persist = ilp._route_to_vulkan_prebuilt(host, FORK, "", force_cpu = False)
    assert repo == FORK


def test_route_to_vulkan_prebuilt_rocm_host_not_rerouted():
    # An Intel iGPU alongside a usable ROCm GPU stays on its ROCm/fork path.
    host = _host(is_linux = True, is_x86_64 = True, has_intel_gpu = True, has_rocm = True)
    _routed, repo, _tag, _persist = ilp._route_to_vulkan_prebuilt(host, FORK, "", force_cpu = False)
    assert repo == FORK


def test_route_to_vulkan_prebuilt_non_intel_unchanged():
    host = _host(is_linux = True, is_x86_64 = True)
    routed, repo, _tag, _persist = ilp._route_to_vulkan_prebuilt(host, FORK, "", force_cpu = False)
    assert repo == FORK
    assert routed is host


def test_resolve_prebuilt_intel_host_keeps_fork_manifest(monkeypatch, capsys):
    # The --resolve-prebuilt probe must agree with the install path: an
    # auto-detected Intel host resolves the fork's Vulkan app bundle.
    monkeypatch.setattr(
        ilp, "detect_host", lambda: _host(is_linux = True, is_x86_64 = True, has_intel_gpu = True)
    )
    seen, out = _run_resolve_capture_host(monkeypatch, capsys)
    assert seen["repo"] == FORK
    assert out["repo"] == FORK


# ---------------------------------------------------------------------------
# windows_intel_gpu_in_registry: the in-process Windows Intel probe. A fake
# winreg module stands in for the real registry so the walk runs anywhere.
# ---------------------------------------------------------------------------


class _FakeRegKey:
    def __init__(
        self,
        subkeys = None,
        values = None,
        denied = False,
    ):
        self.subkeys = subkeys or {}
        self.values = values or {}
        self.denied = denied

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeWinreg:
    HKEY_LOCAL_MACHINE = object()

    def __init__(self, root_key):
        self._root_key = root_key

    def OpenKey(self, parent, name):
        if parent is self.HKEY_LOCAL_MACHINE:
            # Pin the production constant: a typo'd class GUID must fail here,
            # not silently return the fake tree.
            if name != ilp._WINDOWS_DISPLAY_CLASS_KEY:
                raise FileNotFoundError(name)
            if self._root_key is None:
                raise FileNotFoundError(name)
            return self._root_key
        key = parent.subkeys.get(name)
        if key is None:
            # Real winreg raises OSError, never KeyError, for a missing key.
            raise FileNotFoundError(name)
        if key.denied:
            raise PermissionError(name)
        return key

    def QueryInfoKey(self, key):
        return (len(key.subkeys), len(key.values), 0)

    def EnumKey(self, key, index):
        return list(key.subkeys)[index]

    def QueryValueEx(self, key, value_name):
        if value_name not in key.values:
            raise FileNotFoundError(value_name)
        return (key.values[value_name], 1)


def _probe_with_display_class(monkeypatch, adapters):
    # The helper lazily does `import winreg`; plant the fake in sys.modules the
    # same way unsloth_cli/tests/test_start.py fakes it for _refresh_windows_path.
    monkeypatch.setitem(sys.modules, "winreg", _FakeWinreg(_FakeRegKey(subkeys = adapters)))
    return ilp.windows_intel_gpu_in_registry()


def test_windows_intel_registry_matches_vendor_id(monkeypatch):
    assert (
        _probe_with_display_class(
            monkeypatch,
            {
                "0000": _FakeRegKey(
                    values = {
                        "MatchingDeviceId": r"PCI\VEN_8086&DEV_56A0&SUBSYS_12345678",
                        "DriverDesc": "Intel(R) Arc(TM) A770 Graphics",
                    }
                ),
            },
        )
        is True
    )


def test_windows_intel_registry_matches_driver_desc_without_device_id(monkeypatch):
    assert (
        _probe_with_display_class(
            monkeypatch,
            {
                "0000": _FakeRegKey(values = {"DriverDesc": "Intel(R) UHD Graphics 630"}),
            },
        )
        is True
    )


def test_windows_intel_registry_ignores_non_intel_adapters(monkeypatch):
    assert (
        _probe_with_display_class(
            monkeypatch,
            {
                "0000": _FakeRegKey(
                    values = {
                        "MatchingDeviceId": r"PCI\VEN_10DE&DEV_2684",
                        "DriverDesc": "NVIDIA GeForce RTX 4090",
                    }
                ),
                "0001": _FakeRegKey(
                    values = {
                        "MatchingDeviceId": r"PCI\VEN_1002&DEV_744C",
                        "DriverDesc": "AMD Radeon RX 7900 XTX",
                    }
                ),
            },
        )
        is False
    )


def test_windows_intel_registry_skips_restricted_properties_subkey(monkeypatch):
    # The real class key carries an ACL-restricted "Properties" subkey and can
    # deny access to individual adapter keys; neither may abort the walk.
    assert (
        _probe_with_display_class(
            monkeypatch,
            {
                "Properties": _FakeRegKey(denied = True),
                "0000": _FakeRegKey(denied = True),
                "0001": _FakeRegKey(
                    values = {
                        "MatchingDeviceId": r"PCI\VEN_8086&DEV_56A0",
                    }
                ),
            },
        )
        is True
    )


def test_windows_intel_registry_missing_class_key_is_false(monkeypatch):
    monkeypatch.setitem(sys.modules, "winreg", _FakeWinreg(None))
    assert ilp.windows_intel_gpu_in_registry() is False


def _detect_windows_host(
    monkeypatch,
    winreg_fake,
    powershell_stdout = "",
):
    """Drive the real detect_host() as a GPU-less Windows host with a fake
    registry, recording every run_capture invocation. Pins the wiring the
    unit tests above cannot see: registry-first, CIM only on a registry miss."""
    monkeypatch.setitem(sys.modules, "winreg", winreg_fake)
    monkeypatch.setattr(ilp.platform, "system", lambda: "Windows")
    monkeypatch.setattr(ilp.platform, "machine", lambda: "AMD64")
    for _env in (
        "CUDA_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "HIP_PATH",
        "ROCM_PATH",
    ):
        monkeypatch.delenv(_env, raising = False)
    monkeypatch.setattr(
        ilp.shutil,
        "which",
        lambda name: "powershell" if name in ("powershell", "pwsh") else None,
    )
    captured = []

    def _fake_run_capture(command, **kwargs):
        captured.append(command[0])
        if command[0] == "powershell":
            return SimpleNamespace(returncode = 0, stdout = powershell_stdout, stderr = "")
        return SimpleNamespace(returncode = 1, stdout = "", stderr = "")

    monkeypatch.setattr(ilp, "run_capture", _fake_run_capture)
    return ilp.detect_host(), captured


def test_detect_host_registry_intel_skips_cim_probe(monkeypatch):
    winreg = _FakeWinreg(
        _FakeRegKey(
            subkeys = {
                "0000": _FakeRegKey(values = {"MatchingDeviceId": r"PCI\VEN_8086&DEV_56A0"}),
            }
        )
    )
    host, captured = _detect_windows_host(monkeypatch, winreg)
    assert host.has_intel_gpu is True
    assert "powershell" not in captured


def test_detect_host_cim_fallback_fires_on_registry_miss(monkeypatch):
    winreg = _FakeWinreg(
        _FakeRegKey(
            subkeys = {
                "0000": _FakeRegKey(values = {"MatchingDeviceId": r"PCI\VEN_10DE&DEV_2684"}),
            }
        )
    )
    host, captured = _detect_windows_host(
        monkeypatch, winreg, powershell_stdout = "Intel(R) Arc(TM) A770 Graphics"
    )
    assert host.has_intel_gpu is True
    assert "powershell" in captured


def test_windows_intel_registry_unexpected_error_is_false(monkeypatch):
    # The probe is advisory: even a non-OSError bug in the walk must return
    # False (deferring to the CIM fallback), never crash detect_host.
    class _ExplodingWinreg:
        HKEY_LOCAL_MACHINE = object()

        def OpenKey(self, parent, name):
            raise TypeError(name)

    monkeypatch.setitem(sys.modules, "winreg", _ExplodingWinreg())
    assert ilp.windows_intel_gpu_in_registry() is False


def test_detect_host_cim_rescues_exploding_registry(monkeypatch):
    class _ExplodingWinreg:
        HKEY_LOCAL_MACHINE = object()

        def OpenKey(self, parent, name):
            raise TypeError(name)

    host, captured = _detect_windows_host(
        monkeypatch, _ExplodingWinreg(), powershell_stdout = "Intel(R) Arc(TM) A770 Graphics"
    )
    assert host.has_intel_gpu is True
    assert "powershell" in captured


def _windows_amd_host(**overrides):
    defaults = dict(
        system = "Windows",
        machine = "amd64",
        is_windows = True,
        is_linux = False,
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
        has_intel_gpu = False,
    )
    defaults.update(overrides)
    return ilp.HostInfo(**defaults)


def test_route_to_vulkan_prebuilt_auto_fallback_for_legacy_amd_gfx():
    host = _windows_amd_host(rocm_gfx_target = "gfx803", rocm_gfx_targets = ["gfx803"])
    routed, repo, _tag, persist = ilp._route_to_vulkan_prebuilt(host, FORK, "pin", force_cpu = False)
    assert repo == FORK
    assert persist == "auto"
    assert routed.has_intel_gpu is True
    assert routed.has_rocm is False


def test_route_to_vulkan_prebuilt_keeps_hip_when_one_gpu_is_supported():
    host = _windows_amd_host(
        rocm_gfx_target = "gfx1201",
        rocm_gfx_targets = ["gfx1201", "gfx803"],
    )
    routed, repo, _tag, persist = ilp._route_to_vulkan_prebuilt(host, FORK, "pin", force_cpu = False)
    assert routed is host
    assert repo == FORK
    assert persist is None


def test_route_to_vulkan_prebuilt_auto_fallback_skips_hip_masked_hosts():
    # A HIP mask can hide a HIP-capable dGPU, but the Vulkan runtime honours none of them,
    # so auto-routing would let the installed backend grab the gfx1201 the user masked
    # off.
    host = _windows_amd_host(
        rocm_gfx_target = "gfx803",
        rocm_gfx_targets = ["gfx1201", "gfx803"],
    )
    routed, repo, _tag, persist = ilp._route_to_vulkan_prebuilt(host, FORK, "pin", force_cpu = False)
    assert repo == FORK
    assert persist is None
    assert routed is host


def test_route_to_vulkan_prebuilt_auto_fallback_when_no_amd_gpu_reaches_floor():
    # Every physical AMD device is below the floor, so no card can be exposed to HIP and
    # the #7357 auto-Vulkan fallback still fires.
    host = _windows_amd_host(
        rocm_gfx_target = "gfx900",
        rocm_gfx_targets = ["gfx803", "gfx900"],
    )
    routed, repo, _tag, persist = ilp._route_to_vulkan_prebuilt(host, FORK, "pin", force_cpu = False)
    assert repo == FORK
    assert persist == "auto"
    assert routed.has_rocm is False


@pytest.mark.parametrize(
    "mask_env", ["HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"]
)
def test_auto_vulkan_declines_when_a_hip_device_mask_filtered_the_probe(mask_env, monkeypatch):
    # hipinfo is a HIP application, so under a mask rocm_gfx_targets is the VISIBLE set and
    # a HIP-capable card can be hidden entirely. "No AMD GPU here reaches the floor" is then
    # unprovable, and Vulkan honours none of these masks, so the auto fallback must decline
    # rather than hand it the reserved card.
    monkeypatch.setenv(mask_env, "1")
    host = _windows_amd_host(rocm_gfx_target = "gfx803", rocm_gfx_targets = ["gfx803"])
    assert ilp._should_auto_vulkan_for_amd_windows(host, FORK) is False
    routed, repo, _tag, persist = ilp._route_to_vulkan_prebuilt(host, FORK, "pin", force_cpu = False)
    assert routed is host
    assert repo == FORK
    assert persist is None


@pytest.mark.parametrize("mask_value", ["", "  ", "-1"])
def test_auto_vulkan_declines_when_the_mask_hides_every_amd_gpu(mask_value, monkeypatch):
    # An all-hiding mask is the strongest form of the same signal, not an exemption:
    # detect_host() resolves no arch under it, but a forwarded --rocm-gfx still reconstructs
    # one (setup infers it from the display-adapter name, which no HIP mask touches), so
    # auto-routing would hand Vulkan every AMD GPU the user hid from HIP.
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", mask_value)
    host = _windows_amd_host(rocm_gfx_target = None, rocm_gfx_targets = [])
    host = ilp._apply_host_overrides(host, override_rocm_gfx = "gfx803")
    assert ilp._active_rocm_gfx_target(host) == "gfx803"
    assert ilp._should_auto_vulkan_for_amd_windows(host, FORK) is False
    routed, repo, _tag, persist = ilp._route_to_vulkan_prebuilt(host, FORK, "pin", force_cpu = False)
    assert routed is host
    assert repo == FORK
    assert persist is None


def test_hip_device_mask_check_is_presence_not_value(monkeypatch):
    # Presence is the whole test: any value means the HIP view is not the physical one, and
    # no value can be read as "the probe saw everything".
    assert ilp._hip_visible_device_mask_set() is False
    for value in ("", "  ", "-1", "0", "1", "0,1"):
        monkeypatch.setenv("HIP_VISIBLE_DEVICES", value)
        assert ilp._hip_visible_device_mask_set() is True, value
    monkeypatch.delenv("HIP_VISIBLE_DEVICES")
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "0")
    assert ilp._hip_visible_device_mask_set() is True
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    assert ilp._hip_visible_device_mask_set() is True


def test_masked_probe_suppression_does_not_touch_non_amd_auto_paths(monkeypatch):
    # The mask says nothing about an Intel iGPU, whose Vulkan auto path is unrelated.
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "1")
    host = _host(
        system = "Windows",
        is_windows = True,
        has_intel_gpu = True,
        has_rocm = False,
        has_physical_nvidia = False,
        has_usable_nvidia = False,
    )
    _routed, repo, _tag, _persist = ilp._route_to_vulkan_prebuilt(
        host, FORK, "pin", force_cpu = False
    )
    assert repo == FORK


def test_route_to_vulkan_prebuilt_hip_masked_host_still_honours_explicit_optin(monkeypatch):
    # The mask guard only suppresses the AUTOMATIC fallback; an explicit opt-in is the user
    # taking responsibility for the Vulkan device mask themselves.
    monkeypatch.delenv("UNSLOTH_FORCE_VULKAN", raising = False)
    host = _windows_amd_host(
        rocm_gfx_target = "gfx803",
        rocm_gfx_targets = ["gfx1201", "gfx803"],
    )
    _routed, repo, _tag, persist = ilp._route_to_vulkan_prebuilt(
        host, FORK, "pin", force_cpu = False, llama_backend = "vulkan"
    )
    assert repo == FORK
    assert persist == "vulkan"


def test_forced_vulkan_on_an_auto_routing_amd_host_persists_as_automatic(monkeypatch):
    # This box has no HIP-capable AMD GPU, so it routes to Vulkan either way and
    # the bundle is identical. Persisting it as automatic is what lets pre-#8050
    # installs (whose every update re-asserts --llama-backend vulkan) stay
    # eligible for the Vulkan CPU crash recovery.
    monkeypatch.delenv("UNSLOTH_FORCE_VULKAN", raising = False)
    host = _windows_amd_host(rocm_gfx_target = "gfx1034", rocm_gfx_targets = ["gfx1034"])
    _routed, _repo, _tag, persist = ilp._route_to_vulkan_prebuilt(
        host, UPSTREAM, "pin", force_cpu = False, llama_backend = "vulkan"
    )
    assert persist == "auto"
    # A host that can run HIP keeps the explicit choice explicit.
    supported = _windows_amd_host(rocm_gfx_target = "gfx1100", rocm_gfx_targets = ["gfx1100"])
    _routed, _repo, _tag, persist = ilp._route_to_vulkan_prebuilt(
        supported, UPSTREAM, "pin", force_cpu = False, llama_backend = "vulkan"
    )
    assert persist == "vulkan"


def test_auto_vulkan_is_repository_specific_for_fork_only_gfx():
    # gfx1034 is served only by the fork's gfx103X bundle: ggml-org's windows-hip radeon
    # build does not target it and direct_upstream_release_plan() offers win-hip then CPU
    # with no Vulkan branch, so the predicate must answer per repo.
    host = _windows_amd_host(rocm_gfx_target = "gfx1034", rocm_gfx_targets = ["gfx1034"])
    assert ilp._should_auto_vulkan_for_amd_windows(host, FORK) is False
    assert ilp._should_auto_vulkan_for_amd_windows(host, UPSTREAM) is True
    # An arch upstream really does build stays on HIP for both repos.
    supported = _windows_amd_host(rocm_gfx_target = "gfx1100", rocm_gfx_targets = ["gfx1100"])
    assert ilp._should_auto_vulkan_for_amd_windows(supported, FORK) is False
    assert ilp._should_auto_vulkan_for_amd_windows(supported, UPSTREAM) is False
    # A family label is a bundle name, not an arch: upstream builds every member but
    # gfx1034 / gfx1103, and the label cannot say which card this is, so it stays on HIP
    # rather than moving the covered members onto Vulkan.
    family = _windows_amd_host(rocm_gfx_target = "gfx110X", rocm_gfx_targets = ["gfx110X"])
    assert ilp._should_auto_vulkan_for_amd_windows(family, UPSTREAM) is False


@pytest.mark.parametrize(
    "repo", ["acme/llama.cpp-mirror", "GGML-ORG/llama.cpp", "unslothAI/llama.cpp"]
)
def test_fork_only_gfx_coverage_is_not_granted_to_other_repos(repo):
    # Only the fork is planned from a manifest: resolve_simple_install_release_plans()
    # compares == DEFAULT_PUBLISHED_REPO and sends everything else, mirrors and differently
    # cased spellings alike, to direct_upstream_release_plan(). Granting a fork-only arch
    # coverage there lands it on win-hip-radeon or CPU instead of Vulkan, so the predicate
    # must gate on the fork rather than exempt one name.
    host = _windows_amd_host(rocm_gfx_target = "gfx1034", rocm_gfx_targets = ["gfx1034"])
    assert ilp._should_auto_vulkan_for_amd_windows(host, repo) is True
    supported = _windows_amd_host(rocm_gfx_target = "gfx1100", rocm_gfx_targets = ["gfx1100"])
    assert ilp._should_auto_vulkan_for_amd_windows(supported, repo) is False


@pytest.mark.parametrize("repo", [None, ""])
def test_empty_published_repo_gets_fork_coverage(repo):
    # Negative control: the resolver defaults an empty repo to the fork, so the predicate
    # must too, or the default install path loses its fork-only archs.
    host = _windows_amd_host(rocm_gfx_target = "gfx1034", rocm_gfx_targets = ["gfx1034"])
    assert ilp._should_auto_vulkan_for_amd_windows(host, repo) is False


def test_upstream_windows_hip_targets_are_a_subset_of_the_combined_floor():
    # The floor must stay a superset, else auto-Vulkan steals a host upstream builds for.
    assert ilp.UPSTREAM_WINDOWS_HIP_GFX_TARGETS <= ilp.WINDOWS_HIP_PREBUILT_GFX_TARGETS
    # The fork-only extras are exactly the archs that must route to Vulkan upstream.
    assert ilp.WINDOWS_HIP_PREBUILT_GFX_TARGETS - ilp.UPSTREAM_WINDOWS_HIP_GFX_TARGETS == {
        "gfx908",
        "gfx90a",
        "gfx1034",
        "gfx1103",
    }


def test_route_to_vulkan_prebuilt_unknown_gfx_does_not_auto_fallback():
    host = _windows_amd_host(
        has_rocm = True,
        rocm_gfx_target = None,
        rocm_gfx_targets = [],
    )
    routed, repo, _tag, persist = ilp._route_to_vulkan_prebuilt(host, FORK, "pin", force_cpu = False)
    assert routed is host
    assert repo == FORK
    assert persist is None


def test_route_to_vulkan_prebuilt_family_gfx_token_keeps_rocm():
    host = _windows_amd_host(rocm_gfx_target = "gfx110X", rocm_gfx_targets = ["gfx110X"])
    routed, repo, _tag, persist = ilp._route_to_vulkan_prebuilt(host, FORK, "pin", force_cpu = False)
    assert routed is host
    assert repo == FORK
    assert persist is None


def test_route_to_vulkan_prebuilt_gfx1103_keeps_rocm():
    host = _windows_amd_host(rocm_gfx_target = "gfx1103", rocm_gfx_targets = ["gfx1103"])
    routed, repo, _tag, persist = ilp._route_to_vulkan_prebuilt(host, FORK, "pin", force_cpu = False)
    assert routed is host
    assert repo == FORK
    assert persist is None


def test_route_to_vulkan_prebuilt_gfx1034_keeps_rocm():
    # gfx1034 (RX 6500/6400-class) is covered by the fork's gfx103X bundle.
    host = _windows_amd_host(rocm_gfx_target = "gfx1034", rocm_gfx_targets = ["gfx1034"])
    routed, repo, _tag, persist = ilp._route_to_vulkan_prebuilt(host, FORK, "pin", force_cpu = False)
    assert routed is host
    assert repo == FORK
    assert persist is None


def test_route_to_vulkan_prebuilt_explicit_opt_in_on_mixed_amd(monkeypatch):
    monkeypatch.setenv("UNSLOTH_LLAMA_CPP_BACKEND", "vulkan")
    host = _windows_amd_host(
        rocm_gfx_target = "gfx1201",
        rocm_gfx_targets = ["gfx1201", "gfx803"],
    )
    routed, repo, _tag, persist = ilp._route_to_vulkan_prebuilt(host, FORK, "pin", force_cpu = False)
    assert repo == FORK
    assert persist == "vulkan"
    assert routed.has_rocm is False


def test_direct_upstream_windows_amd_legacy_gfx_routes_to_vulkan():
    host = _windows_amd_host(rocm_gfx_target = "gfx803", rocm_gfx_targets = ["gfx803"])
    routed, repo, _tag, persist = ilp._route_to_vulkan_prebuilt(host, FORK, "pin", force_cpu = False)
    rel = _upstream_release(
        "b9925",
        [
            "llama-b9925-bin-win-hip-radeon-x64.zip",
            "llama-b9925-bin-win-vulkan-x64.zip",
            "llama-b9925-bin-win-cpu-x64.zip",
        ],
    )
    plan = ilp.direct_upstream_release_plan(rel, routed, repo, "latest")
    assert persist == "auto"
    assert plan.attempts[0].install_kind == "windows-vulkan"


def test_hip_backend_env_suppresses_auto_vulkan_fallback_on_unsupported_gfx(monkeypatch):
    # An explicit HIP choice keeps HIP on an unsupported gfx target instead of
    # silently substituting Vulkan.
    monkeypatch.delenv("UNSLOTH_FORCE_VULKAN", raising = False)
    monkeypatch.setenv("UNSLOTH_LLAMA_CPP_BACKEND", "hip")
    host = _windows_amd_host(rocm_gfx_target = "gfx803", rocm_gfx_targets = ["gfx803"])
    routed, repo, _tag, persist = ilp._route_to_vulkan_prebuilt(host, FORK, "pin", force_cpu = False)
    assert routed is host
    assert repo == FORK
    assert persist is None


def test_route_to_vulkan_prebuilt_hidden_physical_nvidia_amd_not_rerouted():
    # Vulkan ignores CUDA_VISIBLE_DEVICES, so a CUDA-masked NVIDIA card next to a legacy
    # AMD gfx must not auto-route: Vulkan could grab the reserved NVIDIA GPU.
    host = _windows_amd_host(
        rocm_gfx_target = "gfx803",
        rocm_gfx_targets = ["gfx803"],
        has_physical_nvidia = True,
        has_usable_nvidia = False,
    )
    routed, repo, _tag, persist = ilp._route_to_vulkan_prebuilt(host, FORK, "pin", force_cpu = False)
    assert routed is host
    assert repo == FORK
    assert persist is None


def test_route_to_vulkan_prebuilt_explicit_opt_in_overrides_hidden_nvidia(monkeypatch):
    # The physical-NVIDIA guard only gates the AMD auto path; an explicit opt-in wins.
    monkeypatch.setenv("UNSLOTH_LLAMA_CPP_BACKEND", "vulkan")
    host = _windows_amd_host(
        rocm_gfx_target = "gfx803",
        rocm_gfx_targets = ["gfx803"],
        has_physical_nvidia = True,
        has_usable_nvidia = False,
    )
    routed, repo, _tag, persist = ilp._route_to_vulkan_prebuilt(host, FORK, "pin", force_cpu = False)
    assert repo == FORK
    assert persist == "vulkan"


# The gfx archs the fork's llama-prebuilt-manifest.json maps to a windows-rocm bundle.
# Static because parametrisation happens at import time and the routing tests below must
# stay offline; the guard further down re-derives it from the published manifest and fails
# on drift, so this is a checked mirror, not a second source of truth.
_FORK_WINDOWS_ROCM_GFX = (
    "gfx908",
    "gfx90a",
    "gfx1030",
    "gfx1031",
    "gfx1032",
    "gfx1034",
    "gfx1100",
    "gfx1101",
    "gfx1102",
    "gfx1103",
    "gfx1150",
    "gfx1151",
    "gfx1200",
    "gfx1201",
)


def _published_fork_windows_rocm_artifacts():
    """The fork's windows-rocm artifact records, read the way an install reads them.

    _download_host_resolved_release is the path a default fork install takes first: it
    resolves the latest release off the download host and hands llama-prebuilt-manifest.json
    to parse_published_release_bundle, so these are the very records
    published_rocm_choice_for_host later matches a host gfx against. No api.github.com call,
    hence no shared rate-limit bucket to exhaust.

    The manifest ships only as a release asset and nothing in-tree mirrors it, so this is
    the one honest source. Only OSError and the release-side PrebuiltFallback become a skip,
    so an offline run stays quiet while a manifest that fetches but no longer parses still
    fails loudly."""
    try:
        resolved = ilp._download_host_resolved_release(FORK)
    except OSError as exc:
        pytest.skip(f"{FORK} release manifest unreachable: {exc}")
    except ilp.PrebuiltFallback as exc:
        pytest.skip(f"{FORK} latest release was rejected before its manifest parsed: {exc}")
    if resolved is None:
        pytest.skip(f"{FORK} published no resolvable latest release")
    tag = resolved.bundle.release_tag
    artifacts = [
        artifact
        for artifact in resolved.bundle.artifacts
        if artifact.install_kind == "windows-rocm"
    ]
    assert artifacts, f"{FORK}@{tag} manifest listed no windows-rocm artifacts"
    return tag, artifacts


def test_windows_hip_gfx_floor_covers_every_fork_windows_rocm_bundle():
    # Derived from the published manifest, not a second literal: a gfx the fork builds but
    # the floor omits bypasses the fork manifest, downgrading a hash-approved windows-rocm
    # bundle to an unhashed upstream Vulkan build. A newly published arch must redden here.
    tag, artifacts = _published_fork_windows_rocm_artifacts()
    # published_rocm_choice_for_host serves a bundle on a concrete mapped_targets entry or on
    # the umbrella gfx_target itself, so both spellings must clear a floor. A gfx_target
    # absent from its own mapped_targets is the family label (gfx110X); one present in it is
    # a standalone bundle (gfx908) already counted as concrete.
    concrete = {target.lower() for artifact in artifacts for target in artifact.mapped_targets}
    labels = {
        artifact.gfx_target.lower()
        for artifact in artifacts
        if artifact.gfx_target and artifact.gfx_target.lower() not in concrete
    }
    unfloored = sorted(concrete - ilp.WINDOWS_HIP_PREBUILT_GFX_TARGETS)
    assert (
        not unfloored
    ), f"auto-Vulkan would steal windows-rocm archs published in {FORK}@{tag}: {unfloored}"
    unlabelled = sorted(labels - ilp.WINDOWS_ROCM_FAMILY_GFX_LABELS)
    assert not unlabelled, (
        f"update markers forward family labels {FORK}@{tag} publishes but "
        f"WINDOWS_ROCM_FAMILY_GFX_LABELS omits: {unlabelled}"
    )
    # Keep the import-time tuple the offline routing tests parametrise on an exact mirror.
    assert set(_FORK_WINDOWS_ROCM_GFX) == concrete, (
        f"_FORK_WINDOWS_ROCM_GFX drifted from {FORK}@{tag}: "
        f"gained {sorted(concrete - set(_FORK_WINDOWS_ROCM_GFX))}, "
        f"lost {sorted(set(_FORK_WINDOWS_ROCM_GFX) - concrete)}"
    )


@pytest.mark.parametrize("gfx", _FORK_WINDOWS_ROCM_GFX)
def test_route_to_vulkan_prebuilt_keeps_every_fork_windows_rocm_arch(gfx, monkeypatch):
    # No ambient opt-in: this asserts the AUTO path leaves covered archs alone.
    monkeypatch.delenv("UNSLOTH_LLAMA_CPP_BACKEND", raising = False)
    monkeypatch.delenv("UNSLOTH_FORCE_VULKAN", raising = False)
    host = _windows_amd_host(rocm_gfx_target = gfx, rocm_gfx_targets = [gfx])
    routed, repo, tag, persist = ilp._route_to_vulkan_prebuilt(host, FORK, "pin", force_cpu = False)
    assert routed is host
    assert (repo, tag) == (FORK, "pin")
    assert persist is None


def test_forwarded_gfx_does_not_undo_visible_device_auto_vulkan(monkeypatch):
    # Mixed-AMD Windows host: GPU 0 = gfx1100 (HIP prebuilt exists), GPU 1 = gfx1010 (none).
    # Under CUDA_VISIBLE_DEVICES=1 setup.ps1 still resolves GPU 0 and forwards gfx1100, but
    # detect_host() resolved the visible gfx1010, so folding the forward in must not
    # reinstate gfx1100 and install a HIP bundle the visible GPU cannot run.
    monkeypatch.delenv("UNSLOTH_LLAMA_CPP_BACKEND", raising = False)
    monkeypatch.delenv("UNSLOTH_FORCE_VULKAN", raising = False)
    monkeypatch.delenv("UNSLOTH_ROCM_GFX_ARCH", raising = False)
    host = _windows_amd_host(rocm_gfx_target = "gfx1010", rocm_gfx_targets = ["gfx1100", "gfx1010"])
    host = ilp._apply_host_overrides(host, override_rocm_gfx = "gfx1100")
    assert ilp._active_rocm_gfx_target(host) == "gfx1010"
    assert host.rocm_gfx_targets == ["gfx1100", "gfx1010"]
    # gfx1100 is masked off, not absent, and Vulkan does not honour the HIP mask, so the
    # automatic fallback stays off and the HIP / fork path is kept.
    assert ilp._should_auto_vulkan_for_amd_windows(host, FORK) is False
    _routed, repo, _tag, persist = ilp._route_to_vulkan_prebuilt(host, FORK, "pin", force_cpu = False)
    assert repo == FORK
    assert persist is None


def test_forwarded_gfx_absent_from_probe_keeps_the_physical_hip_card(monkeypatch):
    # Mixed-AMD Windows host: GPU 0 = gfx1100 (HIP prebuilt exists), GPU 1 = gfx803 (below
    # the floor). CUDA_VISIBLE_DEVICES=1 reserves the gfx1100, so detect_host() picks gfx803
    # as active but still reports both cards, and setup forwards a third arch the probe never
    # saw (a stale env var, or name inference reading the other card). That forward selects
    # the HIP target but must not delete the probe's inventory, or the floor check concludes
    # no AMD GPU here reaches HIP and auto-routes to Vulkan, which ignores the HIP mask and
    # enumerates the reserved gfx1100.
    monkeypatch.delenv("UNSLOTH_LLAMA_CPP_BACKEND", raising = False)
    monkeypatch.delenv("UNSLOTH_FORCE_VULKAN", raising = False)
    monkeypatch.delenv("UNSLOTH_ROCM_GFX_ARCH", raising = False)
    host = _windows_amd_host(rocm_gfx_target = "gfx803", rocm_gfx_targets = ["gfx1100", "gfx803"])
    host = ilp._apply_host_overrides(host, override_rocm_gfx = "gfx900")
    assert ilp._active_rocm_gfx_target(host) == "gfx900"
    assert host.rocm_gfx_targets == ["gfx1100", "gfx803", "gfx900"]
    assert ilp._should_auto_vulkan_for_amd_windows(host, FORK) is False
    _routed, repo, _tag, persist = ilp._route_to_vulkan_prebuilt(host, FORK, "pin", force_cpu = False)
    assert repo == FORK
    assert persist is None


def test_forwarded_gfx_absent_from_probe_keeps_a_single_probed_hip_card(monkeypatch):
    # Same rule on a single-GPU box: a stale below-floor forward over a probe-confirmed
    # gfx1100 must not auto-route that machine to Vulkan.
    monkeypatch.delenv("UNSLOTH_LLAMA_CPP_BACKEND", raising = False)
    monkeypatch.delenv("UNSLOTH_FORCE_VULKAN", raising = False)
    monkeypatch.delenv("UNSLOTH_ROCM_GFX_ARCH", raising = False)
    host = _windows_amd_host(rocm_gfx_target = "gfx1100", rocm_gfx_targets = ["gfx1100"])
    host = ilp._apply_host_overrides(host, override_rocm_gfx = "gfx803")
    assert ilp._active_rocm_gfx_target(host) == "gfx803"
    assert host.rocm_gfx_targets == ["gfx1100", "gfx803"]
    assert ilp._should_auto_vulkan_for_amd_windows(host, FORK) is False


def test_forwarded_gfx_absent_from_probe_still_allows_explicit_vulkan(monkeypatch):
    # The physical-inventory rule gates the AUTO path only; naming the backend wins.
    monkeypatch.setenv("UNSLOTH_LLAMA_CPP_BACKEND", "vulkan")
    monkeypatch.delenv("UNSLOTH_ROCM_GFX_ARCH", raising = False)
    host = _windows_amd_host(rocm_gfx_target = "gfx1100", rocm_gfx_targets = ["gfx1100"])
    host = ilp._apply_host_overrides(host, override_rocm_gfx = "gfx803")
    _routed, repo, _tag, persist = ilp._route_to_vulkan_prebuilt(host, FORK, "pin", force_cpu = False)
    assert repo == FORK
    assert persist == "vulkan"


def test_forwarded_gfx_on_unprobed_host_still_auto_vulkans(monkeypatch):
    # Negative control: a driver-only AMD host runs no successful probe (no hipinfo, amd-smi
    # suppressed), so --rocm-gfx is the ONLY source of the arch and there is no inventory to
    # preserve. This is the #7357 path the feature exists for; it must still reach Vulkan.
    monkeypatch.delenv("UNSLOTH_LLAMA_CPP_BACKEND", raising = False)
    monkeypatch.delenv("UNSLOTH_FORCE_VULKAN", raising = False)
    monkeypatch.delenv("UNSLOTH_ROCM_GFX_ARCH", raising = False)
    host = _windows_amd_host(rocm_gfx_target = None, rocm_gfx_targets = [])
    host = ilp._apply_host_overrides(host, override_rocm_gfx = "gfx803")
    assert host.rocm_gfx_targets == ["gfx803"]
    assert ilp._should_auto_vulkan_for_amd_windows(host, FORK) is True
    _routed, repo, _tag, persist = ilp._route_to_vulkan_prebuilt(host, FORK, "pin", force_cpu = False)
    assert repo == FORK
    assert persist == "auto"


def test_forwarded_gfx_still_fills_an_unprobed_arch(monkeypatch):
    # Negative control: on an amd-smi-only host detect_host() reports no arch, so the
    # forward is the only source and must still apply.
    monkeypatch.delenv("UNSLOTH_LLAMA_CPP_BACKEND", raising = False)
    monkeypatch.delenv("UNSLOTH_FORCE_VULKAN", raising = False)
    monkeypatch.delenv("UNSLOTH_ROCM_GFX_ARCH", raising = False)
    host = _windows_amd_host(rocm_gfx_target = None, rocm_gfx_targets = [])
    host = ilp._apply_host_overrides(host, override_rocm_gfx = "gfx1151")
    assert ilp._active_rocm_gfx_target(host) == "gfx1151"
    assert ilp._should_auto_vulkan_for_amd_windows(host) is False
    _routed, repo, _tag, persist = ilp._route_to_vulkan_prebuilt(host, FORK, "pin", force_cpu = False)
    assert repo == FORK
    assert persist is None


def test_llama_cpp_backend_cpu_opts_out_of_auto_vulkan(monkeypatch):
    # cpu is an explicit non-Vulkan choice, so it suppresses the auto-fallback.
    monkeypatch.setenv("UNSLOTH_LLAMA_CPP_BACKEND", "cpu")
    host = _windows_amd_host(rocm_gfx_target = "gfx803", rocm_gfx_targets = ["gfx803"])
    routed, repo, _tag, persist = ilp._route_to_vulkan_prebuilt(host, FORK, "pin", force_cpu = False)
    assert routed is host
    assert repo == FORK
    assert persist is None
    assert ilp.force_vulkan_requested() is False


def test_explicit_backend_beats_legacy_force_vulkan(monkeypatch):
    # A stale UNSLOTH_FORCE_VULKAN must not overrule the canonical CPU choice.
    monkeypatch.setenv("UNSLOTH_FORCE_VULKAN", "1")
    monkeypatch.setenv("UNSLOTH_LLAMA_CPP_BACKEND", "cpu")
    assert ilp.resolved_llama_backend() == "cpu"
    assert ilp.force_vulkan_requested() is False
    host = _windows_amd_host(rocm_gfx_target = "gfx1100", rocm_gfx_targets = ["gfx1100"])
    _routed, repo, _tag, persist = ilp._route_to_vulkan_prebuilt(host, FORK, "pin", force_cpu = False)
    assert repo == FORK
    assert persist is None


def test_unknown_llama_backend_value_falls_through_to_legacy_flag(monkeypatch):
    # An unrecognised value is ignored, not an error, so the legacy flag still works.
    monkeypatch.setenv("UNSLOTH_LLAMA_CPP_BACKEND", "banana")
    assert ilp.resolved_llama_backend() is None
    assert ilp.force_vulkan_requested() is False
    monkeypatch.setenv("UNSLOTH_FORCE_VULKAN", "1")
    assert ilp.force_vulkan_requested() is True


def test_llama_backend_flag_beats_conflicting_env(monkeypatch):
    # --llama-backend is the caller's explicit request and outranks the env.
    monkeypatch.setenv("UNSLOTH_LLAMA_CPP_BACKEND", "cpu")
    assert ilp.force_vulkan_requested("vulkan") is True
    host = _windows_amd_host(rocm_gfx_target = "gfx1100", rocm_gfx_targets = ["gfx1100"])
    _routed, repo, _tag, persist = ilp._route_to_vulkan_prebuilt(
        host, FORK, "pin", force_cpu = False, llama_backend = "vulkan"
    )
    assert repo == FORK
    assert persist == "vulkan"


def _windows_arm64_host(**overrides):
    defaults = dict(
        system = "Windows",
        machine = "ARM64",
        is_windows = True,
        is_linux = False,
        is_macos = False,
        is_x86_64 = False,
        is_arm64 = True,
        nvidia_smi = None,
        driver_cuda_version = None,
        compute_caps = [],
        visible_cuda_devices = None,
        has_physical_nvidia = False,
        has_usable_nvidia = False,
        has_rocm = False,
        has_intel_gpu = False,
    )
    defaults.update(overrides)
    return ilp.HostInfo(**defaults)


@pytest.mark.parametrize(
    "env, flag",
    [
        ({"UNSLOTH_LLAMA_CPP_BACKEND": "vulkan"}, None),
        ({"UNSLOTH_FORCE_VULKAN": "1"}, None),
        ({}, "vulkan"),
    ],
)
def test_vulkan_opt_in_keeps_windows_arm64_host_for_strict_rejection(monkeypatch, env, flag):
    # Windows arm64 has no compatible Vulkan bundle. Keep the real host shape so
    # strict filtering rejects the CPU plan instead of routing through x64.
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    host = _windows_arm64_host()
    routed, repo, tag, persist = ilp._route_to_vulkan_prebuilt(
        host, FORK, "pin", force_cpu = False, llama_backend = flag
    )
    assert routed is host
    assert (repo, tag) == (FORK, "pin")
    assert persist is None


def test_vulkan_opt_in_still_routes_on_windows_x64(monkeypatch):
    # Negative control for the arm64 guard: x64 keeps its Vulkan routing.
    monkeypatch.setenv("UNSLOTH_LLAMA_CPP_BACKEND", "vulkan")
    host = _windows_amd_host(rocm_gfx_target = "gfx1100", rocm_gfx_targets = ["gfx1100"])
    _routed, repo, _tag, persist = ilp._route_to_vulkan_prebuilt(host, FORK, "pin", force_cpu = False)
    assert repo == FORK
    assert persist == "vulkan"


def _choice(install_kind, name = "asset.zip"):
    return ilp.AssetChoice(
        repo = UPSTREAM,
        tag = "b9925",
        name = name,
        url = f"https://example/{name}",
        source_label = "upstream",
        install_kind = install_kind,
    )


@pytest.mark.parametrize("kind", ["windows-vulkan", "linux-vulkan"])
def test_persisted_llama_backend_keeps_vulkan_for_a_vulkan_bundle(kind):
    assert ilp.persisted_llama_backend("vulkan", _choice(kind)) == "vulkan"


@pytest.mark.parametrize("kind", ["windows-arm64", "windows-cpu", "linux-cpu", "windows-rocm"])
def test_persisted_llama_backend_drops_vulkan_for_a_non_vulkan_bundle(kind):
    # _plan_llama_phase re-asserts the marker's backend on every later update, so a Vulkan
    # request that fell through to CPU must not leave a marker claiming Vulkan.
    assert ilp.persisted_llama_backend("vulkan", _choice(kind)) is None


def test_persisted_llama_backend_passes_none_through():
    assert ilp.persisted_llama_backend(None, _choice("windows-vulkan")) is None


def test_persisted_llama_backend_keeps_automatic_vulkan_distinct():
    assert ilp.persisted_llama_backend("auto", _choice("windows-vulkan")) == "auto"


def test_marker_records_no_backend_when_vulkan_fell_back_to_cpu(tmp_path):
    # End to end over write_prebuilt_metadata: describe the CPU attempt that actually won,
    # so the next update re-detects instead of re-asserting Vulkan forever.
    checksums = ilp.ApprovedReleaseChecksums(
        repo = UPSTREAM,
        release_tag = "b9925",
        upstream_tag = "b9925",
        source_repo = UPSTREAM,
        source_repo_url = f"https://github.com/{UPSTREAM}",
    )
    cpu = _choice("windows-arm64", "llama-b9925-bin-win-cpu-arm64.zip")
    ilp.write_prebuilt_metadata(
        tmp_path,
        requested_tag = "latest",
        llama_tag = "b9925",
        release_tag = "b9925",
        choice = cpu,
        approved_checksums = checksums,
        prebuilt_fallback_used = False,
        llama_backend = "vulkan",
    )
    marker = json.loads((tmp_path / "UNSLOTH_PREBUILT_INFO.json").read_text())
    assert marker["asset"] == "llama-b9925-bin-win-cpu-arm64.zip"
    assert marker["llama_backend"] is None

    vulkan = _choice("windows-vulkan", "llama-b9925-bin-win-vulkan-x64.zip")
    ilp.write_prebuilt_metadata(
        tmp_path,
        requested_tag = "latest",
        llama_tag = "b9925",
        release_tag = "b9925",
        choice = vulkan,
        approved_checksums = checksums,
        prebuilt_fallback_used = False,
        llama_backend = "vulkan",
    )
    marker = json.loads((tmp_path / "UNSLOTH_PREBUILT_INFO.json").read_text())
    assert marker["llama_backend"] == "vulkan"


def test_automatic_amd_vulkan_marker_records_the_routing_gfx(tmp_path, monkeypatch):
    """An "auto" Vulkan marker must carry the arch that produced it.

    The Windows AMD hosts that route here have no arch probe of their own
    (setup.ps1 skips amd-smi without a HIP SDK and hipinfo is absent), so the
    forwarded --rocm-gfx is the only evidence. The Vulkan asset name has no gfx
    for rocm_install_args() to recover, so an update that re-detects the host
    would find no ROCm at all and drop the bundle to CPU.
    """
    monkeypatch.delenv("UNSLOTH_FORCE_VULKAN", raising = False)
    monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising = False)
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising = False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising = False)
    # A driver-only host: nothing probed, the arch arrived over --rocm-gfx.
    unprobed = ilp.HostInfo(
        system = "Windows",
        machine = "amd64",
        is_windows = True,
        is_linux = False,
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
    )
    host = ilp._apply_host_overrides(unprobed, override_rocm_gfx = "gfx1034")
    assert ilp._should_auto_vulkan_for_amd_windows(host, UPSTREAM) is True
    gfx = ilp._active_rocm_gfx_target(host)
    routed, _repo, _tag, persist = ilp._route_to_vulkan_prebuilt(
        host, UPSTREAM, "pin", force_cpu = False
    )
    assert persist == "auto"
    # The route drops the arch, so it must be read before it, not after.
    assert ilp._active_rocm_gfx_target(routed) is None
    assert gfx == "gfx1034"

    checksums = ilp.ApprovedReleaseChecksums(
        repo = UPSTREAM,
        release_tag = "b9925",
        upstream_tag = "b9925",
        source_repo = UPSTREAM,
        source_repo_url = f"https://github.com/{UPSTREAM}",
    )
    ilp.write_prebuilt_metadata(
        tmp_path,
        requested_tag = "latest",
        llama_tag = "b9925",
        release_tag = "b9925",
        choice = _choice("windows-vulkan", "llama-b9925-bin-win-vulkan-x64.zip"),
        approved_checksums = checksums,
        prebuilt_fallback_used = False,
        llama_backend = persist,
        rocm_gfx = gfx,
    )
    marker = json.loads((tmp_path / "UNSLOTH_PREBUILT_INFO.json").read_text())
    assert marker["llama_backend"] == "auto"
    assert marker["rocm_gfx"] == "gfx1034"

    # Replaying the marker's arch reaches the same route on a re-detected host.
    replayed = ilp._apply_host_overrides(unprobed, override_rocm_gfx = marker["rocm_gfx"])
    assert ilp._should_auto_vulkan_for_amd_windows(replayed, UPSTREAM) is True
    # Without it the update re-detects a ROCm-less host and leaves Vulkan behind.
    assert ilp._should_auto_vulkan_for_amd_windows(unprobed, UPSTREAM) is False
    assert ilp._route_to_vulkan_prebuilt(unprobed, UPSTREAM, "pin", force_cpu = False)[3] is None


def test_a_remembered_arch_never_outranks_a_live_probe(monkeypatch):
    """The updater replays the marker's arch, which is a stale probe result, not
    an operator override: a host whose GPU changed must keep what it reports."""
    monkeypatch.delenv("UNSLOTH_ROCM_GFX_ARCH", raising = False)
    unprobed = dataclasses.replace(
        _windows_amd_host(rocm_gfx_target = None, rocm_gfx_targets = []), has_rocm = False
    )
    monkeypatch.setenv("UNSLOTH_ROCM_GFX_REMEMBERED", "gfx1034")

    # No probe of its own: the remembered arch fills the gap.
    filled = ilp._apply_host_overrides(unprobed)
    assert ilp._active_rocm_gfx_target(filled) == "gfx1034"
    assert filled.has_rocm is True

    # A live probe wins, so a replaced card is not routed as the old one.
    probed = _windows_amd_host(rocm_gfx_target = "gfx1100", rocm_gfx_targets = ["gfx1100"])
    assert ilp._active_rocm_gfx_target(ilp._apply_host_overrides(probed)) == "gfx1100"


def test_reusing_an_install_still_records_the_routing_gfx(tmp_path):
    """The reuse paths skip write_prebuilt_metadata, so they must sync it too."""
    marker_path = tmp_path / "UNSLOTH_PREBUILT_INFO.json"
    marker_path.write_text(json.dumps({"llama_backend": "auto", "asset": "win-vulkan.zip"}))

    ilp.sync_marker_rocm_gfx(tmp_path, "gfx1034")
    assert json.loads(marker_path.read_text())["rocm_gfx"] == "gfx1034"

    # Nothing to record leaves the marker untouched.
    ilp.sync_marker_rocm_gfx(tmp_path, None)
    assert json.loads(marker_path.read_text())["rocm_gfx"] == "gfx1034"


def test_non_amd_installs_record_no_routing_gfx(tmp_path):
    """The key is absent unless an arch actually routed the install."""
    checksums = ilp.ApprovedReleaseChecksums(
        repo = UPSTREAM,
        release_tag = "b9925",
        upstream_tag = "b9925",
        source_repo = UPSTREAM,
        source_repo_url = f"https://github.com/{UPSTREAM}",
    )
    ilp.write_prebuilt_metadata(
        tmp_path,
        requested_tag = "latest",
        llama_tag = "b9925",
        release_tag = "b9925",
        choice = _choice("windows-vulkan", "llama-b9925-bin-win-vulkan-x64.zip"),
        approved_checksums = checksums,
        prebuilt_fallback_used = False,
        llama_backend = None,
    )
    marker = json.loads((tmp_path / "UNSLOTH_PREBUILT_INFO.json").read_text())
    assert "rocm_gfx" not in marker


# setup translates UNSLOTH_LLAMA_CPP_BACKEND=cpu into --force-cpu to pin the
# CPU-only bundle on a GPU host, which is what keeps Intel iGPU Vulkan crashes
# away (#7213). Vulkan is opt-in, so no trigger may outrank that flag on any host.
_SIM_PLATFORMS = {
    # WSL presents as Linux to this resolver, so it rides the Linux row.
    "Linux": dict(
        system = "Linux",
        is_windows = False,
        is_linux = True,
        is_macos = False,
        machine = "x86_64",
        is_x86_64 = True,
        is_arm64 = False,
    ),
    "Windows": dict(
        system = "Windows",
        is_windows = True,
        is_linux = False,
        is_macos = False,
        machine = "amd64",
        is_x86_64 = True,
        is_arm64 = False,
    ),
    "macOS": dict(
        system = "Darwin",
        is_windows = False,
        is_linux = False,
        is_macos = True,
        machine = "arm64",
        is_x86_64 = False,
        is_arm64 = True,
    ),
}
_SIM_GPUS = {
    "nvidia": dict(
        has_physical_nvidia = True,
        has_usable_nvidia = True,
        has_rocm = False,
        has_intel_gpu = False,
        nvidia_smi = "/usr/bin/nvidia-smi",
        driver_cuda_version = "12.4",
        compute_caps = ["8.9"],
    ),
    "amd": dict(
        has_physical_nvidia = False,
        has_usable_nvidia = False,
        has_rocm = True,
        has_intel_gpu = False,
        nvidia_smi = None,
        driver_cuda_version = None,
        compute_caps = [],
        rocm_gfx_target = "gfx803",
        rocm_gfx_targets = ["gfx803"],
    ),
    "intel": dict(
        has_physical_nvidia = False,
        has_usable_nvidia = False,
        has_rocm = False,
        has_intel_gpu = True,
        nvidia_smi = None,
        driver_cuda_version = None,
        compute_caps = [],
    ),
    "cpu_only": dict(
        has_physical_nvidia = False,
        has_usable_nvidia = False,
        has_rocm = False,
        has_intel_gpu = False,
        nvidia_smi = None,
        driver_cuda_version = None,
        compute_caps = [],
    ),
}


def _sim_host(platform_name, gpu_name):
    base = dict(visible_cuda_devices = None)
    base.update(_SIM_PLATFORMS[platform_name])
    base.update(_SIM_GPUS[gpu_name])
    return ilp.HostInfo(**base)


@pytest.mark.parametrize("platform_name", sorted(_SIM_PLATFORMS))
@pytest.mark.parametrize("gpu_name", sorted(_SIM_GPUS))
@pytest.mark.parametrize("backend_env", [None, "auto", "vulkan", "cpu"])
def test_forced_cpu_outranks_every_vulkan_trigger(
    monkeypatch, platform_name, gpu_name, backend_env
):
    """A deliberate CPU install stays CPU on every host, whatever asks for Vulkan."""
    monkeypatch.delenv("UNSLOTH_FORCE_VULKAN", raising = False)
    if backend_env is None:
        monkeypatch.delenv("UNSLOTH_LLAMA_CPP_BACKEND", raising = False)
    else:
        monkeypatch.setenv("UNSLOTH_LLAMA_CPP_BACKEND", backend_env)
    # The legacy switch too, so a stale one cannot smuggle Vulkan past --force-cpu.
    monkeypatch.setenv("UNSLOTH_FORCE_VULKAN", "1")

    repo, tag = "unslothai/llama.cpp-prebuilt", "latest"
    _, out_repo, _, persist = ilp._route_to_vulkan_prebuilt(
        _sim_host(platform_name, gpu_name),
        repo,
        tag,
        force_cpu = True,
        llama_backend = "vulkan",
    )
    assert out_repo == repo, (platform_name, gpu_name, backend_env)
    assert persist is None, (platform_name, gpu_name, backend_env)


def test_the_forced_cpu_guard_is_not_vacuous():
    """The same host DOES take Vulkan once the CPU pin is gone, or the check above
    would pass on a resolver that had stopped routing to Vulkan entirely."""
    repo, tag = "unslothai/llama.cpp-prebuilt", "latest"
    _, out_repo, _, persist = ilp._route_to_vulkan_prebuilt(
        _sim_host("Linux", "amd"),
        repo,
        tag,
        force_cpu = False,
        llama_backend = "vulkan",
    )
    assert out_repo != repo or persist == "vulkan"
