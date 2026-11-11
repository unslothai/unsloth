# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A CUDA install with no cudart anywhere gets NVIDIA's runtime wheels; anything else gets none.
On Linux the wheels go in before selection, which only takes a CUDA bundle whose runtime is on disk."""

from __future__ import annotations

import json
import pathlib
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "studio"))

import install_llama_prebuilt as ip  # noqa: E402


def _install(tmp_path, asset, kind):
    # The fields the installer's marker has: it records the backend and the asset, no install kind.
    (tmp_path / "UNSLOTH_PREBUILT_INFO.json").write_text(
        json.dumps({"asset": asset, "backend": ip.backend_for_install_kind(kind)})
    )
    return tmp_path


def _host(
    system = "Linux",
    machine = "x86_64",
    driver = (13, 0),
    caps = ("86",),
    nvidia = True,
):
    return ip.HostInfo(
        system = system,
        machine = machine,
        is_windows = system == "Windows",
        is_linux = system == "Linux",
        is_macos = False,
        is_x86_64 = machine == "x86_64",
        is_arm64 = machine == "aarch64",
        nvidia_smi = "nvidia-smi" if nvidia else None,
        driver_cuda_version = driver if nvidia else None,
        compute_caps = list(caps) if nvidia else [],
        visible_cuda_devices = None,
        has_physical_nvidia = nvidia,
        has_usable_nvidia = nvidia,
    )


@pytest.fixture(autouse = True)
def _no_host_runtime(monkeypatch, tmp_path_factory):
    site = tmp_path_factory.mktemp("site")
    monkeypatch.setattr(ip.sysconfig, "get_paths", lambda: {"purelib": str(site)})
    monkeypatch.delenv("CUDA_PATH", raising = False)
    monkeypatch.setattr(ip, "detected_linux_runtime_lines", lambda: ([], {}))
    monkeypatch.setattr(ip, "windows_runtime_dirs", lambda: [])
    return site


@pytest.mark.parametrize(
    "asset, kind, wheels",
    [
        (
            "app-b1-windows-x64-cuda13-newer.zip",
            "windows-cuda",
            ("nvidia-cuda-runtime", "nvidia-cublas"),
        ),
        (
            "app-b1-windows-x64-cuda12-newer.zip",
            "windows-cuda",
            ("nvidia-cuda-runtime-cu12", "nvidia-cublas-cu12==12.9.2.10"),
        ),
        (
            "app-b1-linux-x64-cuda12-newer.tar.gz",
            "linux-cuda",
            ("nvidia-cuda-runtime-cu12", "nvidia-cublas-cu12"),
        ),
        ("app-b1-windows-x64-cpu.zip", "windows-cpu", ()),
        ("app-b1-windows-arm64-cuda13-portable.zip", "windows-arm64-cuda", ()),
    ],
)
def test_wheels_for_a_bare_install(tmp_path, asset, kind, wheels):
    assert ip._missing_cuda_runtime_wheels(_install(tmp_path, asset, kind)) == wheels


def test_bundled_cudart_needs_nothing(tmp_path):
    install = _install(tmp_path, "app-b1-windows-x64-cuda13-newer.zip", "windows-cuda")
    (install / "build" / "bin" / "Release").mkdir(parents = True)
    (install / "build" / "bin" / "Release" / "cudart64_13.dll").touch()
    (install / "build" / "bin" / "Release" / "cublas64_13.dll").touch()
    (install / "build" / "bin" / "Release" / "cublasLt64_13.dll").touch()
    assert ip._missing_cuda_runtime_wheels(install) == ()


def test_torch_runtime_needs_nothing(tmp_path, _no_host_runtime):
    (_no_host_runtime / "torch" / "lib").mkdir(parents = True)
    (_no_host_runtime / "torch" / "lib" / "cudart64_12.dll").touch()
    (_no_host_runtime / "torch" / "lib" / "cublas64_12.dll").touch()
    (_no_host_runtime / "torch" / "lib" / "cublasLt64_12.dll").touch()
    install = _install(tmp_path, "app-b1-windows-x64-cuda12-newer.zip", "windows-cuda")
    assert ip._missing_cuda_runtime_wheels(install) == ()


def test_other_major_does_not_count(tmp_path, _no_host_runtime):
    (_no_host_runtime / "torch" / "lib").mkdir(parents = True)
    (_no_host_runtime / "torch" / "lib" / "cudart64_12.dll").touch()
    install = _install(tmp_path, "app-b1-windows-x64-cuda13-newer.zip", "windows-cuda")
    assert ip._missing_cuda_runtime_wheels(install) == ("nvidia-cuda-runtime", "nvidia-cublas")


def test_the_marker_the_installer_writes_is_recognised(tmp_path):
    # Through the real writer: a hand-written marker with fields the installer never records
    # would pass here while every real install went unnoticed.
    choice = ip.AssetChoice(
        repo = "unslothai/llama.cpp",
        tag = "b1",
        name = "app-b1-windows-x64-cuda13-newer.zip",
        url = "https://example.invalid/app-b1-windows-x64-cuda13-newer.zip",
        source_label = "published",
        is_ready_bundle = True,
        install_kind = "windows-cuda",
        runtime_line = "cuda13",
        expected_sha256 = "a" * 64,
    )
    checksums = ip.ApprovedReleaseChecksums(
        repo = choice.repo,
        release_tag = "b1",
        upstream_tag = "b1",
        artifacts = {
            choice.name: ip.ApprovedArtifactHash(
                choice.name, "a" * 64, choice.repo, choice.install_kind
            )
        },
    )
    ip.write_prebuilt_metadata(
        tmp_path,
        requested_tag = "latest",
        llama_tag = "b1",
        release_tag = "b1",
        choice = choice,
        approved_checksums = checksums,
        prebuilt_fallback_used = False,
    )
    assert ip._missing_cuda_runtime_wheels(tmp_path) == ("nvidia-cuda-runtime", "nvidia-cublas")


def test_linux_runtime_selection_can_see_needs_nothing(tmp_path, monkeypatch):
    # Wherever selection found it (a versioned toolkit dir, Ollama's), not just the env.
    monkeypatch.setattr(ip, "detected_linux_runtime_lines", lambda: (["cuda12"], {}))
    install = _install(tmp_path, "app-b1-linux-x64-cuda12-newer.tar.gz", "linux-cuda")
    assert ip._missing_cuda_runtime_wheels(install) == ()


@pytest.mark.parametrize(
    "host, detected, wheels",
    [
        (_host(driver = (13, 0)), [], ("nvidia-cuda-runtime", "nvidia-cublas")),
        (_host(driver = (12, 8)), [], ("nvidia-cuda-runtime-cu12", "nvidia-cublas-cu12")),
        # CUDA 13 dropped pre-Turing GPUs.
        (
            _host(driver = (13, 0), caps = ("61",)),
            [],
            ("nvidia-cuda-runtime-cu12", "nvidia-cublas-cu12"),
        ),
        # A runtime the driver cannot run is no runtime.
        (_host(driver = (12, 8)), ["cuda13"], ("nvidia-cuda-runtime-cu12", "nvidia-cublas-cu12")),
        (_host(driver = (13, 0)), ["cuda12"], ()),
        (_host(driver = (13, 0)), ["cuda13", "cuda12"], ()),
        # A CUDA 13 runtime cannot serve a pre-Turing GPU either.
        (
            _host(driver = (13, 0), caps = ("61",)),
            ["cuda13"],
            ("nvidia-cuda-runtime-cu12", "nvidia-cublas-cu12"),
        ),
        (_host(driver = (13, 0), caps = ("61",)), ["cuda13", "cuda12"], ()),
        (_host(nvidia = False), [], ()),
        (_host(machine = "aarch64"), [], ()),
        (_host(system = "Windows"), [], ()),
    ],
)
def test_wheels_a_linux_host_needs_before_selection(monkeypatch, host, detected, wheels):
    monkeypatch.setattr(ip, "detected_linux_runtime_lines", lambda: (list(detected), {}))
    assert ip._linux_cuda_runtime_wheels_to_supply(host) == wheels


@pytest.mark.parametrize(
    "backend, installs", [(None, 1), ("auto", 1), ("cuda", 1), ("vulkan", 0), ("cpu", 0)]
)
def test_only_a_cuda_bound_request_supplies_the_runtime(monkeypatch, backend, installs):
    calls = []
    monkeypatch.setattr(ip, "_install_cuda_runtime_wheels", lambda wheels: calls.append(wheels))
    route = ip.BackendRoute(
        backend = backend,
        host = _host(),
        published_repo = "unslothai/llama.cpp",
        published_release_tag = "",
        persist_llama_backend = None,
        persist_rocm_gfx = None,
    )
    ip.ensure_linux_cuda_runtime_before_selection(route)
    assert len(calls) == installs


def test_a_failing_wheel_step_does_not_stop_the_install(monkeypatch):
    def boom(wheels):
        raise OSError("no network")

    monkeypatch.setattr(ip, "_install_cuda_runtime_wheels", boom)
    route = ip.BackendRoute(
        backend = "auto",
        host = _host(),
        published_repo = "unslothai/llama.cpp",
        published_release_tag = "",
        persist_llama_backend = None,
        persist_rocm_gfx = None,
    )
    ip.ensure_linux_cuda_runtime_before_selection(route)


def test_cudart_without_cublas_still_installs(tmp_path, monkeypatch):
    toolkit = tmp_path / "toolkit"
    (toolkit / "bin").mkdir(parents = True)
    (toolkit / "bin" / "cudart64_13.dll").touch()
    monkeypatch.setenv("CUDA_PATH", str(toolkit))
    install = tmp_path / "install"
    install.mkdir()
    _install(install, "app-b1-windows-x64-cuda13-newer.zip", "windows-cuda")
    assert ip._missing_cuda_runtime_wheels(install) == ("nvidia-cuda-runtime", "nvidia-cublas")


def test_windows_loader_dirs_count(tmp_path, monkeypatch):
    on_path = tmp_path / "on_path"
    on_path.mkdir()
    (on_path / "cudart64_13.dll").touch()
    (on_path / "cublas64_13.dll").touch()
    (on_path / "cublasLt64_13.dll").touch()
    monkeypatch.setattr(ip, "windows_runtime_dirs", lambda: [str(on_path)])
    install = tmp_path / "install"
    install.mkdir()
    _install(install, "app-b1-windows-x64-cuda13-newer.zip", "windows-cuda")
    assert ip._missing_cuda_runtime_wheels(install) == ()


def test_cublas_without_cublaslt_still_installs(tmp_path, monkeypatch):
    on_path = tmp_path / "on_path"
    on_path.mkdir()
    (on_path / "cudart64_13.dll").touch()
    (on_path / "cublas64_13.dll").touch()
    monkeypatch.setattr(ip, "windows_runtime_dirs", lambda: [str(on_path)])
    install = tmp_path / "install"
    install.mkdir()
    _install(install, "app-b1-windows-x64-cuda13-newer.zip", "windows-cuda")
    assert ip._missing_cuda_runtime_wheels(install) == ("nvidia-cuda-runtime", "nvidia-cublas")
