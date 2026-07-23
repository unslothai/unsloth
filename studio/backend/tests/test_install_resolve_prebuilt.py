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


def test_route_to_vulkan_prebuilt_auto_intel_goes_upstream_and_drops_fork_pin():
    # Routing fork -> upstream also drops the fork release pin, which is in a
    # different tag namespace and would make the upstream resolver miss.
    host = _host(is_linux = True, is_x86_64 = True, has_intel_gpu = True)
    routed, repo, tag, _persist = ilp._route_to_vulkan_prebuilt(host, FORK, "b9596-mix-abc", force_cpu = False)
    assert repo == UPSTREAM
    assert tag == ""
    assert routed.has_intel_gpu is True


def test_route_to_vulkan_prebuilt_preserves_explicit_upstream_pin():
    # A pin set WITH an explicit upstream repo is already on upstream -> kept.
    host = _host(is_linux = True, is_x86_64 = True, has_intel_gpu = True)
    _routed, repo, tag, _persist = ilp._route_to_vulkan_prebuilt(host, UPSTREAM, "b9596", force_cpu = False)
    assert repo == UPSTREAM
    assert tag == "b9596"


def test_route_to_vulkan_prebuilt_cpu_fallback_wins():
    # --cpu-fallback suppresses Vulkan routing even for an Intel host.
    host = _host(is_linux = True, is_x86_64 = True, has_intel_gpu = True)
    routed, repo, tag, _persist = ilp._route_to_vulkan_prebuilt(host, FORK, "b9596-mix-abc", force_cpu = True)
    assert repo == FORK
    assert tag == "b9596-mix-abc"
    assert routed is host


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


def test_resolve_prebuilt_intel_host_routes_to_upstream(monkeypatch, capsys):
    # The --resolve-prebuilt probe must agree with the install path: an
    # auto-detected Intel host resolves against upstream (Vulkan), not the fork.
    monkeypatch.setattr(
        ilp, "detect_host", lambda: _host(is_linux = True, is_x86_64 = True, has_intel_gpu = True)
    )
    seen, out = _run_resolve_capture_host(monkeypatch, capsys)
    assert seen["repo"] == UPSTREAM
    assert out["repo"] == UPSTREAM


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
    assert repo == UPSTREAM
    assert persist == "vulkan"
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


def test_route_to_vulkan_prebuilt_explicit_opt_in_on_mixed_amd(monkeypatch):
    monkeypatch.setenv("UNSLOTH_LLAMA_BACKEND", "vulkan")
    host = _windows_amd_host(
        rocm_gfx_target = "gfx1201",
        rocm_gfx_targets = ["gfx1201", "gfx803"],
    )
    routed, repo, _tag, persist = ilp._route_to_vulkan_prebuilt(host, FORK, "pin", force_cpu = False)
    assert repo == UPSTREAM
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
    assert persist == "vulkan"
    assert plan.attempts[0].install_kind == "windows-vulkan"


def test_llama_backend_env_requests_vulkan(monkeypatch):
    assert ilp.llama_backend_from_env() is None
    monkeypatch.setenv("UNSLOTH_LLAMA_BACKEND", "vulkan")
    assert ilp.llama_backend_from_env() == "vulkan"
    assert ilp.force_vulkan_requested() is True
