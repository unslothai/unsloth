# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Linux Intel SYCL prebuilt: selection, oneAPI runtime discovery, device gate."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = PACKAGE_ROOT / "studio" / "install_llama_prebuilt.py"
SPEC = importlib.util.spec_from_file_location("studio_install_llama_prebuilt_sycl", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
ILP = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ILP
SPEC.loader.exec_module(ILP)

from backend.utils.prebuilt.runtime_libs import ONEAPI_LIB_DIRS_FILENAME, oneapi_lib_dirs  # noqa: E402

SYCL8 = "app-b1-linux-x64-sycl8.tar.gz"
SYCL9 = "app-b1-linux-x64-sycl9.tar.gz"
SYCL_ANY = "app-b1-linux-x64-sycl.tar.gz"
VULKAN = "app-b1-linux-x64-vulkan.tar.gz"
CPU = "app-b1-linux-x64-cpu.tar.gz"


def _host(**overrides):
    machine = overrides.pop("machine", "x86_64")
    defaults = dict(
        system = "Linux",
        machine = machine,
        is_linux = True,
        is_windows = False,
        is_macos = False,
        is_x86_64 = machine == "x86_64",
        is_arm64 = machine == "aarch64",
        nvidia_smi = None,
        driver_cuda_version = None,
        compute_caps = [],
        visible_cuda_devices = None,
        has_physical_nvidia = False,
        has_usable_nvidia = False,
        has_intel_gpu = True,
    )
    defaults.update(overrides)
    return ILP.HostInfo(**defaults)


def _bundle(sycl = ((SYCL8, "sycl8"),)):
    def art(
        name,
        kind,
        line = None,
        profile = None,
    ):
        return ILP.PublishedLlamaArtifact(
            asset_name = name,
            install_kind = kind,
            runtime_line = line,
            coverage_class = None,
            supported_sms = [],
            min_sm = None,
            max_sm = None,
            bundle_profile = profile,
            rank = 60,
        )

    artifacts = [art(name, "linux-sycl", line) for name, line in sycl]
    artifacts += [
        art(VULKAN, "linux-vulkan", profile = "linux-vulkan-x64"),
        art(CPU, "linux-cpu", profile = "linux-cpu-x64"),
    ]
    return ILP.PublishedReleaseBundle(
        repo = "unslothai/llama.cpp",
        release_tag = "b1",
        upstream_tag = "b1",
        assets = {a.asset_name: f"https://example.com/{a.asset_name}" for a in artifacts},
        artifacts = artifacts,
    )


def _venv_with_libsycl(monkeypatch, tmp_path, *names):
    lib = tmp_path / "venv" / "lib"
    lib.mkdir(parents = True)
    for name in names:
        (lib / name).write_bytes(b"")
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "venv"))
    return lib


def _attempts(host, bundle):
    return [a.name for a in ILP._linux_published_attempts(host, bundle)]


def test_intel_host_with_venv_libsycl_tries_sycl_before_vulkan(monkeypatch, tmp_path):
    _venv_with_libsycl(monkeypatch, tmp_path, "libsycl.so.8")
    assert _attempts(_host(), _bundle()) == [SYCL8, VULKAN, CPU]


@pytest.mark.parametrize(
    "overrides",
    [
        {"has_amd_gpu_without_rocm": True},
        {"has_physical_nvidia": True},
        {"machine": "aarch64"},
        {"has_intel_gpu": False},
    ],
)
def test_sycl_is_not_tried_off_the_intel_x86_path(monkeypatch, tmp_path, overrides):
    _venv_with_libsycl(monkeypatch, tmp_path, "libsycl.so.8")
    assert not any("sycl" in name for name in _attempts(_host(**overrides), _bundle()))


def test_no_venv_libsycl_keeps_vulkan_first(monkeypatch, tmp_path):
    _venv_with_libsycl(monkeypatch, tmp_path)
    assert _attempts(_host(), _bundle()) == [VULKAN, CPU]


def test_runtime_line_follows_the_newest_venv_libsycl(monkeypatch, tmp_path):
    _venv_with_libsycl(monkeypatch, tmp_path, "libsycl.so.8", "libsycl.so.9")
    both = ((SYCL8, "sycl8"), (SYCL9, "sycl9"))
    assert _attempts(_host(), _bundle(both))[0] == SYCL9
    # No matching line: an unlabelled SYCL bundle still fits, a mislabelled one never does.
    assert _attempts(_host(), _bundle(((SYCL8, "sycl8"),)))[0] == VULKAN
    assert _attempts(_host(), _bundle(((SYCL8, "sycl8"), (SYCL_ANY, None))))[0] == SYCL_ANY


def test_recorded_oneapi_dirs_reach_the_binary_env(monkeypatch, tmp_path):
    bin_dir = tmp_path / "build" / "bin"
    bin_dir.mkdir(parents = True)
    present = tmp_path / "venv" / "lib"
    present.mkdir(parents = True)
    (bin_dir / ONEAPI_LIB_DIRS_FILENAME).write_text(
        f"{present}:{tmp_path / 'gone'}\n", encoding = "utf-8"
    )
    assert oneapi_lib_dirs(bin_dir) == [str(present)]
    assert oneapi_lib_dirs(tmp_path) == []
    env = ILP.binary_env(bin_dir / "llama-server", tmp_path, _host())
    assert str(present) in env["LD_LIBRARY_PATH"].split(":")


@pytest.mark.parametrize(
    "stdout, listed",
    [
        ("Available devices:\n  SYCL0: Intel(R) Arc(TM) Pro B60 Graphics (24480 MiB)\n", True),
        ("Available devices:\n", False),
    ],
)
def test_sycl_device_gate_reads_list_devices(monkeypatch, tmp_path, stdout, listed):
    monkeypatch.setattr(
        ILP.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a[0], 0, stdout = stdout, stderr = ""),
    )
    assert (
        ILP.bundle_lists_vulkan_device(
            tmp_path / "llama-server", tmp_path, _host(), ILP._LISTED_SYCL_DEVICE_RE
        )
        is listed
    )


def _install_sycl_bundle(monkeypatch, tmp_path):
    import hashlib
    import io
    import shutil
    import tarfile

    bundle = tmp_path / SYCL8
    with tarfile.open(bundle, "w:gz") as archive:
        for name in (
            "llama-server",
            "llama-quantize",
            "libllama-common.so",
            "libllama.so",
            "libggml.so",
            "libggml-base.so",
            "libggml-cpu-x64.so",
            "libmtmd.so",
            "libggml-sycl.so",
        ):
            payload = f"{name}\n".encode()
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
    monkeypatch.setattr(
        ILP, "download_file_verified", lambda url, target, **_: shutil.copy2(bundle, target)
    )
    choice = ILP.AssetChoice(
        repo = "unslothai/llama.cpp",
        tag = "b1",
        name = SYCL8,
        url = f"https://example.com/{SYCL8}",
        source_label = "published",
        install_kind = "linux-sycl",
        runtime_line = "sycl8",
        expected_sha256 = hashlib.sha256(bundle.read_bytes()).hexdigest(),
    )
    install = tmp_path / "install"
    work = tmp_path / "work"
    install.mkdir()
    work.mkdir()
    return ILP.install_from_archives(choice, _host(), install, work), install


def test_sycl_install_records_the_venv_oneapi_dirs(monkeypatch, tmp_path):
    lib = _venv_with_libsycl(monkeypatch, tmp_path, "libsycl.so.8")
    (_server, quantize), install = _install_sycl_bundle(monkeypatch, tmp_path)
    bin_dir = install / "build" / "bin"
    assert (bin_dir / "libggml-sycl.so").is_file()
    assert oneapi_lib_dirs(bin_dir) == [str(lib)]
    # llama-quantize is CPU work and ggml skips a backend it cannot dlopen: no wrapper needed.
    root_quantize = install / "llama-quantize"
    assert root_quantize.is_symlink() and root_quantize.resolve() == quantize.resolve()


def test_sycl_install_without_a_oneapi_runtime_falls_back(monkeypatch, tmp_path):
    _venv_with_libsycl(monkeypatch, tmp_path)
    with pytest.raises(ILP.PrebuiltFallback, match = "no oneAPI runtime"):
        _install_sycl_bundle(monkeypatch, tmp_path)
