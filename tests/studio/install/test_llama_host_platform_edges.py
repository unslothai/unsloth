# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""llama.cpp prebuilt choice on three host edges, all I/O faked:
Windows AMD against upstream's win-rocm-<ver> rename, Apple Silicon under Rosetta, and Jetson."""

import builtins
import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = PACKAGE_ROOT / "studio" / "install_llama_prebuilt.py"
SPEC = importlib.util.spec_from_file_location("studio_install_llama_prebuilt_edges", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
ILP = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ILP
SPEC.loader.exec_module(ILP)

TAG = "b9900"


def _host(**overrides):
    system = overrides.pop("system", "Windows")
    machine = overrides.pop("machine", "AMD64")
    defaults = dict(
        system = system,
        machine = machine,
        is_linux = system == "Linux",
        is_windows = system == "Windows",
        is_macos = system == "Darwin",
        is_x86_64 = machine.lower() in {"x86_64", "amd64"},
        is_arm64 = machine.lower() in {"arm64", "aarch64"},
        nvidia_smi = None,
        driver_cuda_version = None,
        compute_caps = [],
        visible_cuda_devices = None,
        has_physical_nvidia = False,
        has_usable_nvidia = False,
    )
    defaults.update(overrides)
    return ILP.HostInfo(**defaults)


def _release(names):
    return {
        "tag_name": TAG,
        "assets": [
            {
                "name": n,
                "browser_download_url": f"https://example.com/{n}",
                "digest": "sha256:" + hashlib.sha256(n.encode()).hexdigest(),
            }
            for n in names
        ],
    }


ROCM = f"llama-{TAG}-bin-win-rocm-10.0-x64.zip"
ROCM_OLD = f"llama-{TAG}-bin-win-rocm-7.2-x64.zip"
HIP_RADEON = f"llama-{TAG}-bin-win-hip-radeon-x64.zip"
VULKAN = f"llama-{TAG}-bin-win-vulkan-x64.zip"
CPU = f"llama-{TAG}-bin-win-cpu-x64.zip"


@pytest.fixture(autouse = True)
def _no_hip_mask(monkeypatch):
    for var in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        monkeypatch.delenv(var, raising = False)


def _direct_names(names, host):
    plan = ILP.direct_upstream_release_plan(_release(names), host, ILP.UPSTREAM_REPO, "latest")
    return [a.name for a in plan.attempts]


def _single_name(names, host, monkeypatch):
    assets = {n: f"https://example.com/{n}" for n in names}
    monkeypatch.setattr(ILP, "github_release_assets", lambda repo, tag: assets)
    return ILP.resolve_upstream_asset_choice(host, TAG).name


@pytest.mark.parametrize(
    "names, direct, single",
    [
        # Current upstream: only the versioned ROCm bundle.
        ([ROCM, VULKAN, CPU], [ROCM, VULKAN, CPU], ROCM),
        ([ROCM, ROCM_OLD, VULKAN, CPU], [ROCM, ROCM_OLD, VULKAN, CPU], ROCM),
        # No HIP bundle at all: Vulkan before CPU.
        ([VULKAN, CPU], [VULKAN, CPU], VULKAN),
        # Older tags with the legacy name keep it first.
        ([HIP_RADEON, VULKAN, CPU], [HIP_RADEON, VULKAN, CPU], HIP_RADEON),
        ([HIP_RADEON, CPU], [HIP_RADEON, CPU], HIP_RADEON),
        ([CPU], [CPU], CPU),
    ],
)
def test_windows_rocm_host_accepts_win_rocm_assets(names, direct, single, monkeypatch):
    host = _host(has_rocm = True)
    assert _direct_names(names, host) == direct
    assert _single_name(names, host, monkeypatch) == single


def test_windows_rocm_bundle_keeps_the_hip_install_kind(monkeypatch):
    host = _host(has_rocm = True)
    plan = ILP.direct_upstream_release_plan(
        _release([ROCM, VULKAN, CPU]), host, ILP.UPSTREAM_REPO, "latest"
    )
    assert [a.install_kind for a in plan.attempts] == [
        "windows-hip",
        "windows-vulkan",
        "windows-cpu",
    ]


def test_windows_rocm_with_physical_nvidia_never_takes_vulkan(monkeypatch):
    # Vulkan ignores CUDA_VISIBLE_DEVICES; a hidden NVIDIA card must not be enumerated.
    host = _host(has_rocm = True, has_physical_nvidia = True, visible_cuda_devices = "-1")
    assert _direct_names([VULKAN, CPU], host) == [CPU]
    assert _single_name([VULKAN, CPU], host, monkeypatch) == CPU


@pytest.mark.parametrize(
    "trace",
    [
        dict(nvidia_smi = "C:/Windows/System32/nvidia-smi.exe"),
        dict(compute_caps = ["89"]),
        dict(physical_compute_caps = ["89"]),
    ],
)
def test_windows_rocm_following_torch_keeps_nvidia_out_of_vulkan(trace, monkeypatch):
    # Following ROCm torch clears has_physical_nvidia, but Vulkan would still enumerate the card.
    host = _host(has_rocm = True, **trace)
    assert _direct_names([ROCM, VULKAN, CPU], host) == [ROCM, CPU]
    assert _single_name([VULKAN, CPU], host, monkeypatch) == CPU


@pytest.mark.parametrize(
    "var", ["HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"]
)
def test_windows_rocm_hip_mask_never_falls_back_to_vulkan(var, monkeypatch):
    # Vulkan ignores HIP masks, so it could enumerate an AMD card the user hid.
    monkeypatch.setenv(var, "1")
    host = _host(has_rocm = True)
    assert _direct_names([ROCM, VULKAN, CPU], host) == [ROCM, CPU]
    assert _single_name([VULKAN, CPU], host, monkeypatch) == CPU


@pytest.mark.parametrize(
    "var", ["HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", None]
)
@pytest.mark.parametrize("other", ["has_intel_gpu", "has_amd_gpu_without_rocm"])
def test_windows_rocm_beside_another_gpu_keeps_the_vulkan_guard(var, other, monkeypatch):
    # An Intel iGPU or a non-ROCm AMD card must not bypass the ROCm mask / NVIDIA guard.
    if var:
        monkeypatch.setenv(var, "1")
        host = _host(has_rocm = True, **{other: True})
    else:
        host = _host(has_rocm = True, nvidia_smi = "C:/nvidia-smi.exe", **{other: True})
    assert _direct_names([VULKAN, CPU], host) == [CPU]
    assert _single_name([VULKAN, CPU], host, monkeypatch) == CPU


def test_windows_rocm_beside_an_intel_igpu_still_takes_vulkan_unmasked(monkeypatch):
    host = _host(has_rocm = True, has_intel_gpu = True)
    assert _single_name([VULKAN, CPU], host, monkeypatch) == VULKAN


@pytest.mark.parametrize(
    "flags, expected",
    [
        (dict(has_amd_gpu_without_rocm = True), [VULKAN, CPU]),
        (dict(has_intel_gpu = True), [VULKAN, CPU]),
        (dict(), [CPU]),
    ],
)
def test_windows_non_rocm_hosts_unchanged(flags, expected, monkeypatch):
    host = _host(**flags)
    names = [ROCM, HIP_RADEON, VULKAN, CPU]
    assert _direct_names(names, host) == expected
    assert _single_name(names, host, monkeypatch) == expected[0]


MAC_ARM = f"llama-{TAG}-bin-macos-arm64.tar.gz"
MAC_X64 = f"llama-{TAG}-bin-macos-x64.tar.gz"


def _fake_mac(monkeypatch, machine, hw_arm64):
    """Darwin with ``platform.machine()`` = machine and sysctl hw.optional.arm64 = hw_arm64.
    Returns the subprocess calls made: reading the key must spawn none."""
    calls = []
    real_run = subprocess.run

    def fake_run(cmd, *args, **kwargs):
        if list(cmd)[:1] == ["sysctl"]:
            calls.append(list(cmd))
        return real_run(cmd, *args, **kwargs)

    reads = []

    def fake_hw_arm64():
        reads.append(1)
        return hw_arm64

    monkeypatch.setattr(ILP.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(ILP.platform, "machine", lambda: machine)
    monkeypatch.setattr(ILP.platform, "mac_ver", lambda: ("14.5", ("", "", ""), machine))
    monkeypatch.setattr(ILP.shutil, "which", lambda *_a, **_k: None)
    monkeypatch.setattr(ILP.subprocess, "run", fake_run)
    monkeypatch.setattr(ILP, "_darwin_hw_optional_arm64", fake_hw_arm64)
    return calls, reads


@pytest.mark.parametrize(
    "machine, hw_arm64, expected_machine, expected_asset",
    [
        ("x86_64", True, "arm64", MAC_ARM),  # Rosetta
        ("x86_64", False, "x86_64", MAC_X64),  # Intel Mac, key absent or unreadable
        ("arm64", True, "arm64", MAC_ARM),  # native arm64
    ],
)
def test_macos_host_machine(machine, hw_arm64, expected_machine, expected_asset, monkeypatch):
    calls, _reads = _fake_mac(monkeypatch, machine, hw_arm64)
    host = ILP.detect_host()
    assert host.machine == expected_machine
    assert host.is_arm64 == (expected_machine == "arm64")
    assert host.is_x86_64 == (expected_machine == "x86_64")
    plan = ILP.direct_upstream_release_plan(
        _release([MAC_ARM, MAC_X64]), host, ILP.UPSTREAM_REPO, "latest"
    )
    assert [a.name for a in plan.attempts] == [expected_asset]
    cheap = ILP.platform_only_host()
    assert (cheap.machine, cheap.is_arm64, cheap.is_x86_64) == (
        host.machine,
        host.is_arm64,
        host.is_x86_64,
    )
    assert calls == []


def test_sysctl_is_not_read_off_macos(monkeypatch):
    calls, reads = _fake_mac(monkeypatch, "x86_64", True)
    monkeypatch.setattr(ILP.platform, "system", lambda: "Linux")
    assert ILP.platform_only_host().machine == "x86_64"
    assert calls == [] and reads == []


def test_the_rosetta_read_spawns_no_subprocess_and_is_cached(monkeypatch):
    # platform_only_host() runs on every desktop launch preflight: an Intel Mac must not
    # pay a sysctl subprocess (up to 5 s) per call for an answer that never changes.
    calls = []

    def refuse(cmd, *args, **kwargs):
        calls.append(list(cmd))
        raise AssertionError("no subprocess on the health path")

    monkeypatch.setattr(ILP.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(ILP.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(ILP.platform, "mac_ver", lambda: ("14.5", ("", "", ""), "x86_64"))
    monkeypatch.setattr(ILP.subprocess, "run", refuse)
    for _ in range(3):
        ILP.platform_only_host()
    assert calls == []
    # Off macOS the libc has no sysctlbyname: the read fails closed to x86_64.
    if sys.platform.startswith("linux"):
        ILP._darwin_hw_optional_arm64.cache_clear()
        assert ILP._darwin_hw_optional_arm64() is False


def _fake_tegra(
    monkeypatch,
    *,
    release_file = False,
    compatible = None,
):
    real_exists = os.path.exists
    real_open = builtins.open

    def fake_exists(path):
        if path == "/etc/nv_tegra_release":
            return release_file
        return real_exists(path)

    def fake_open(path, *args, **kwargs):
        if path == "/proc/device-tree/compatible":
            if compatible is None:
                raise FileNotFoundError(path)
            return io.BytesIO(compatible)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(ILP.os.path, "exists", fake_exists)
    monkeypatch.setattr(builtins, "open", fake_open)
    tegra = getattr(ILP, "_is_tegra", None)
    if tegra is not None:
        tegra.cache_clear()
        monkeypatch.setattr(ILP, "_TEGRA_FALLBACK_LOGGED", [])


@pytest.fixture(autouse = True)
def _clear_tegra_cache():
    yield
    tegra = getattr(ILP, "_is_tegra", None)
    if tegra is not None:
        tegra.cache_clear()


def _arm64_cuda_release():
    def art(name, line, sms, lo, hi):
        return ILP.PublishedLlamaArtifact(
            asset_name = name,
            install_kind = "linux-arm64-cuda",
            runtime_line = line,
            coverage_class = "portable",
            supported_sms = sms,
            min_sm = lo,
            max_sm = hi,
            bundle_profile = f"{line}-portable",
            rank = 100,
        )

    artifacts = [
        art("app-linux-arm64-cuda12-portable.tar.gz", "cuda12", ["87", "90"], 87, 90),
        art(
            "app-linux-arm64-cuda13-portable.tar.gz", "cuda13", ["87", "90", "110", "121"], 87, 121
        ),
    ]
    return ILP.PublishedReleaseBundle(
        repo = "unslothai/llama.cpp",
        release_tag = "v1.0",
        upstream_tag = TAG,
        source_repo = None,
        source_repo_url = None,
        source_ref_kind = None,
        requested_source_ref = None,
        resolved_source_ref = None,
        source_commit = None,
        source_commit_short = None,
        assets = {a.asset_name: f"https://example.com/{a.asset_name}" for a in artifacts},
        manifest_asset_name = "llama-prebuilt-manifest.json",
        artifacts = artifacts,
        selection_log = [],
    )


def _linux_attempts(monkeypatch, host, lines):
    monkeypatch.setattr(
        ILP, "detected_linux_runtime_lines", lambda: (list(lines), {l: ["/stub"] for l in lines})
    )
    monkeypatch.setattr(
        ILP,
        "detect_torch_cuda_runtime_preference",
        lambda host, **_: ILP.CudaRuntimePreference(runtime_line = None, selection_log = []),
    )
    return [a.name for a in ILP._linux_published_attempts(host, _arm64_cuda_release())]


def _arm_host(driver, caps):
    return _host(
        system = "Linux",
        machine = "aarch64",
        nvidia_smi = "/usr/bin/nvidia-smi",
        driver_cuda_version = driver,
        compute_caps = caps,
        physical_compute_caps = caps,
        has_physical_nvidia = True,
        has_usable_nvidia = True,
    )


@pytest.mark.parametrize(
    "tegra_fs",
    [dict(release_file = True), dict(compatible = b"nvidia,p3768-0000\x00nvidia,tegra234\x00")],
)
def test_jetson_cuda12_source_builds(tegra_fs, monkeypatch):
    _fake_tegra(monkeypatch, **tegra_fs)
    host = _arm_host((12, 6), ["87"])
    assert _linux_attempts(monkeypatch, host, ["cuda12"]) == []
    with pytest.raises(ILP.PrebuiltFallback):
        ILP.resolve_linux_cuda_choice(host, _arm64_cuda_release())


@pytest.mark.parametrize(
    "tegra_fs, driver, caps, lines, expected",
    [
        # GH200 / DGX Spark: server ARM64, not Tegra.
        (dict(), (12, 8), ["90"], ["cuda12"], ["app-linux-arm64-cuda12-portable.tar.gz"]),
        (dict(), (13, 0), ["121"], ["cuda13"], ["app-linux-arm64-cuda13-portable.tar.gz"]),
        (
            dict(compatible = b"nvidia,grace\x00"),
            (13, 0),
            ["90"],
            ["cuda13"],
            ["app-linux-arm64-cuda13-portable.tar.gz"],
        ),
        # Thor, JetPack 7: Tegra but CUDA 13 (SBSA).
        (
            dict(release_file = True),
            (13, 0),
            ["110"],
            ["cuda13"],
            ["app-linux-arm64-cuda13-portable.tar.gz"],
        ),
        # Tegra with an unknown driver version fails open.
        (dict(release_file = True), None, ["87"], ["cuda12"], None),
    ],
)
def test_non_jetson_or_cuda13_arm64_unchanged(tegra_fs, driver, caps, lines, expected, monkeypatch):
    _fake_tegra(monkeypatch)
    names_base = _linux_attempts(monkeypatch, _arm_host(driver, caps), lines)
    _fake_tegra(monkeypatch, **tegra_fs)
    names = _linux_attempts(monkeypatch, _arm_host(driver, caps), lines)
    assert names == names_base
    if expected is not None:
        assert names[:1] == expected


def _jetson_cuda12(monkeypatch):
    _fake_tegra(monkeypatch, release_file = True)
    return _arm_host((12, 6), ["87"])


CUDA12_MARKER = {"backend": "cuda", "runtime_line": "cuda12", "supported_sms": ["87", "90"]}


def test_jetson_rejects_an_existing_cuda_prebuilt(tmp_path, monkeypatch):
    # Installed before the Jetson check: an update must not keep or reuse it.
    monkeypatch.setattr(ILP, "compatible_linux_runtime_lines", lambda host: ["cuda12"])
    monkeypatch.setattr(ILP, "_existing_install_runs", lambda *_a: True)
    (tmp_path / "UNSLOTH_PREBUILT_INFO.json").write_text(json.dumps(CUDA12_MARKER))
    _fake_tegra(monkeypatch)
    server = _arm_host((12, 6), ["87"])
    assert ILP._kept_install_covers_host(CUDA12_MARKER, server)
    assert ILP.reusable_existing_install(tmp_path, server)
    host = _jetson_cuda12(monkeypatch)
    assert not ILP._kept_install_covers_host(CUDA12_MARKER, host)
    assert not ILP.reusable_existing_install(tmp_path, host)
    assert ILP._kept_install_covers_host({"backend": "cpu"}, host)


def test_jetson_stored_cuda_request_with_lazy_plans_falls_back(monkeypatch):
    # The fork's plans are lazy: their Jetson rejection must not surface after the try that converts it.
    def lazy_plans(*_a, **_k):
        def plans():
            raise ILP.PrebuiltFallback("no compatible Linux prebuilt asset was found")
            yield  # pragma: no cover

        return "latest", plans()

    monkeypatch.setattr(ILP, "resolve_simple_install_release_plans", lazy_plans)
    monkeypatch.setattr(ILP, "_rocm_torch_preferred", lambda: False)
    host = _jetson_cuda12(monkeypatch)
    with pytest.raises(ILP.BackendUnavailable):
        selection = ILP.select_backend_install(
            backend = "cuda",
            llama_tag = "latest",
            published_repo = "unslothai/llama.cpp",
            published_release_tag = "",
            host = host,
        )
        list(getattr(selection, "release_plans", []) or [])


def test_jetson_stored_cuda_request_falls_back_to_detection(monkeypatch):
    def no_plans(*_a, **_k):
        raise ILP.PrebuiltFallback("no compatible Linux prebuilt asset was found")

    monkeypatch.setattr(ILP, "resolve_simple_install_release_plans", no_plans)
    monkeypatch.setattr(ILP, "_rocm_torch_preferred", lambda: False)
    host = _jetson_cuda12(monkeypatch)
    with pytest.raises(ILP.BackendUnavailable):
        ILP.select_backend_install(
            backend = "cuda",
            llama_tag = "latest",
            published_repo = "unslothai/llama.cpp",
            published_release_tag = "",
            host = host,
        )
