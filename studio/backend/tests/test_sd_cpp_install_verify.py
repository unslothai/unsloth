# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A new sd.cpp bundle must start before replacing the installed one; Windows ROCm without the HIP SDK uses Vulkan."""

from __future__ import annotations

import io
import os
import re
import sys
import types
import zipfile
from pathlib import Path

import pytest

_STUDIO = Path(__file__).resolve().parents[2]
if str(_STUDIO) not in sys.path:
    sys.path.insert(0, str(_STUDIO))

import install_sd_cpp_prebuilt as sdmod  # noqa: E402

_CLI, _SERVER = sdmod._binary_names()
_WIN = sys.platform == "win32"


def _starting_binary() -> bytes:
    if _WIN:
        return (
            Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "where.exe"
        ).read_bytes()
    return b"#!/bin/sh\necho 'usage: sd-cli [options]'\n"


def _refused_binary() -> bytes:
    """An executable the loader refuses (Windows: python.exe with its pythonXY.dll import renamed)."""
    if _WIN:
        data = (Path(sys.base_prefix) / "python.exe").read_bytes()
        dll = f"python{sys.version_info[0]}{sys.version_info[1]}".encode()
        patched = re.sub(re.escape(dll) + rb"(?=\.dll)", b"z" * len(dll), data, flags = re.I)
        assert patched != data
        return patched
    return (
        b"#!/bin/sh\necho 'sd-cli: error while loading shared libraries: libhipblas.so.3: "
        b"cannot open shared object file: No such file or directory' >&2\nexit 127\n"
    )


def _bundle(payload: bytes, subdir: str = "") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name in (_CLI, _SERVER):
            zf.writestr(f"{subdir}{name}", payload)
        zf.writestr(f"{subdir}stable-diffusion.dll", b"payload")
    return buf.getvalue()


def _names():
    if _WIN:
        return {
            "rocm": "sd-master-0518b34-bin-win-rocm-10.1.0-x64.zip",
            "vulkan": "sd-master-0518b34-bin-win-vulkan-x64.zip",
        }
    return {
        "rocm": "sd-master-0518b34-bin-Linux-Ubuntu-24.04-x86_64-rocm-7.2.1.zip",
        "vulkan": "sd-master-0518b34-bin-Linux-Ubuntu-24.04-x86_64-vulkan.zip",
    }


def _stub_release(monkeypatch, zips: dict):
    names = _names()
    release = {
        "tag_name": "master-982-0518b34",
        "assets": [
            {"name": names[k], "browser_download_url": f"https://example.invalid/{names[k]}"}
            for k in zips
        ],
    }
    by_name = {names[k]: v for k, v in zips.items()}
    monkeypatch.setenv("UNSLOTH_SD_CPP_TAG", "")
    monkeypatch.setattr(sdmod, "_fetch_release", lambda *a, **k: release)
    monkeypatch.setattr(
        sdmod, "_download", lambda url, dest, **k: dest.write_bytes(by_name[url.rsplit("/", 1)[1]])
    )
    monkeypatch.setattr(sdmod.platform, "system", lambda: "Windows" if _WIN else "Linux")
    monkeypatch.setattr(sdmod.platform, "machine", lambda: "AMD64" if _WIN else "x86_64")


def _managed_root(tmp_path, monkeypatch) -> Path:
    root = tmp_path / "sd-home" / "stable-diffusion.cpp"
    root.mkdir(parents = True)
    (root / sdmod.OWNERSHIP_MARKER).touch()
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "sd-home" / "studio"))
    sdmod._INSTALLED_ACCELERATOR_MEMO.clear()
    return root


def _working_vulkan_install(root: Path) -> Path:
    """What the affected host had: the mirror's Vulkan build in its versioned subdirectory."""
    sub = root / "sd-master-982-0518b34-uaba3a1c-bin-win-vulkan-x64"
    sub.mkdir()
    for name in (_CLI, _SERVER):
        (sub / name).write_bytes(_starting_binary())
        if not _WIN:
            (sub / name).chmod(0o755)
    sdmod._write_install_record(root, accelerator = "vulkan", repo = sdmod.DEFAULT_REPO, tag = "t")
    return sub / _SERVER


def test_the_loader_refusal_is_read_from_the_exit_status_and_output():
    assert "not found" in sdmod.loader_refusal(0xC0000135)
    # Python reports the NTSTATUS signed on some builds.
    assert sdmod.loader_refusal(0xC0000135 - (1 << 32))
    assert sdmod.loader_refusal(0xC0000139)
    assert sdmod.loader_refusal(
        127, "sd-cli: error while loading shared libraries: libhipblas.so.3"
    )
    # A process that reached main is not refused, whatever it exits with.
    assert sdmod.loader_refusal(0) is None
    assert sdmod.loader_refusal(1, "unknown option --help") is None
    assert sdmod.loader_refusal(127) is None
    assert sdmod.loader_refusal(None) is None


def test_probe_reports_a_refused_binary_and_passes_a_starting_one(tmp_path):
    bad = tmp_path / "bad" / _CLI
    good = tmp_path / "good" / _CLI
    for path, data in ((bad, _refused_binary()), (good, _starting_binary())):
        path.parent.mkdir()
        path.write_bytes(data)
        if not _WIN:
            path.chmod(0o755)
    assert sdmod.probe_binary_starts(bad)
    assert sdmod.probe_binary_starts(good) is None


def test_a_bundle_that_cannot_start_never_replaces_the_working_one(tmp_path, monkeypatch):
    """Regression: a refused ROCm zip must not sweep the working Vulkan build."""
    root = _managed_root(tmp_path, monkeypatch)
    old_server = _working_vulkan_install(root)
    before = old_server.read_bytes()
    _stub_release(monkeypatch, {"rocm": _bundle(_refused_binary())})

    with pytest.raises(RuntimeError, match = "cannot start"):
        sdmod.install(install_dir = root, accelerator = "rocm")

    assert old_server.is_file() and old_server.read_bytes() == before
    assert (old_server.parent / _CLI).is_file()
    # Nothing of the refused bundle reached the tree, and its record still names the working build.
    assert not (root / _CLI).exists() and not (root / _SERVER).exists()
    assert sdmod.read_install_record(root).get("accelerator") == "vulkan"
    assert not list(root.parent.glob(".sd-cpp-verify-*"))


def test_the_refusal_names_what_was_asked_for(tmp_path, monkeypatch):
    root = _managed_root(tmp_path, monkeypatch)
    _stub_release(monkeypatch, {"rocm": _bundle(_refused_binary())})
    with pytest.raises(sdmod.UnrunnableBuildError) as info:
        sdmod.install(install_dir = root, accelerator = "rocm")
    assert info.value.accelerator == "rocm"
    assert info.value.asset == _names()["rocm"]


def test_a_bundle_that_starts_still_replaces_the_previous_one(tmp_path, monkeypatch):
    root = _managed_root(tmp_path, monkeypatch)
    old_server = _working_vulkan_install(root)
    _stub_release(monkeypatch, {"rocm": _bundle(_starting_binary())})

    cli = sdmod.install(install_dir = root, accelerator = "rocm")

    assert cli == root / _CLI
    assert not old_server.exists()
    assert sdmod.read_install_record(root).get("accelerator") == "rocm"
    assert not list(root.parent.glob(".sd-cpp-verify-*"))


def test_the_kill_switch_restores_the_unverified_install(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_SD_CPP_VERIFY_INSTALL", "0")
    root = _managed_root(tmp_path, monkeypatch)
    old_server = _working_vulkan_install(root)
    _stub_release(monkeypatch, {"rocm": _bundle(_refused_binary())})

    sdmod.install(install_dir = root, accelerator = "rocm")

    assert not old_server.exists()
    assert (root / _SERVER).is_file()


def test_cuda_bundles_are_not_probed(tmp_path):
    # They load the host / llama.cpp CUDA runtime the backend adds at spawn; a bare probe would refuse them.
    assert sdmod._should_verify("sd-master-x-bin-win-cuda12-x64.zip") is False
    assert sdmod._should_verify("sd-master-x-bin-win-rocm-10.1.0-x64.zip") is True


# --- backend: the Windows ROCm request and the install ladder ---------------------------------


@pytest.fixture
def fake_settings(monkeypatch):
    from storage import studio_db

    store: dict = {}
    monkeypatch.setattr(
        studio_db, "get_app_setting", lambda key, fallback = None: store.get(key, fallback)
    )
    monkeypatch.setattr(
        studio_db,
        "upsert_app_settings",
        lambda settings, **_k: store.update(settings) or dict(store),
    )
    return store


@pytest.fixture
def backend(monkeypatch, fake_settings):
    from core.inference import sd_cpp_backend as bk

    monkeypatch.setattr(bk, "_accelerator_runtime_failures", {}, raising = False)
    monkeypatch.setattr(
        bk,
        "_HOST_FINGERPRINT_MEMO",
        {"runtime": "7.1", "gpus": ["AMD Radeon 8060S"]},
        raising = False,
    )
    monkeypatch.setattr(bk, "_failed_accelerator_upgrades", set(), raising = False)
    monkeypatch.setattr(bk, "_failed_pin_upgrades", set(), raising = False)
    monkeypatch.delenv("UNSLOTH_DIFFUSION_SD_CPP_VULKAN_FALLBACK", raising = False)
    return bk


@pytest.fixture
def windows_host(monkeypatch, backend, tmp_path):
    """A Windows host whose loader search path is ``dirs``; add hipblas.dll to one to install the HIP SDK."""
    sdk = tmp_path / "hip-sdk-bin"
    system32 = tmp_path / "System32"
    sdk.mkdir()
    system32.mkdir()
    # The Adrenalin driver puts amdhip64 in System32; hipblas only comes with the HIP SDK.
    (system32 / "amdhip64_7.dll").write_bytes(b"")
    monkeypatch.delenv("UNSLOTH_SD_CPP_WINDOWS_ROCM", raising = False)
    monkeypatch.setattr(backend, "_on_windows", lambda: True, raising = False)
    monkeypatch.setattr(
        backend, "_windows_dll_search_dirs", lambda: [str(system32), str(sdk)], raising = False
    )
    return types.SimpleNamespace(sdk = sdk, system32 = system32)


def test_windows_rocm_without_the_hip_sdk_starts_at_vulkan(windows_host, backend):
    assert backend.preferred_accelerator("rocm") == "vulkan"


def test_windows_rocm_with_the_hip_sdk_on_path_keeps_rocm(windows_host, backend):
    (windows_host.sdk / "hipblas.dll").write_bytes(b"")
    assert backend.preferred_accelerator("rocm") == "rocm"


@pytest.mark.parametrize(
    "env,value,want",
    [
        ("UNSLOTH_SD_CPP_WINDOWS_ROCM", "1", "rocm"),
        ("UNSLOTH_SD_CPP_WINDOWS_ROCM", "0", "vulkan"),
        ("UNSLOTH_DIFFUSION_SD_CPP_VULKAN_FALLBACK", "0", "rocm"),
    ],
)
def test_windows_rocm_kill_switches(windows_host, backend, monkeypatch, env, value, want):
    monkeypatch.setenv(env, value)
    assert backend.preferred_accelerator("rocm") == want


def test_other_hosts_and_accelerators_are_untouched(backend, monkeypatch):
    monkeypatch.delenv("UNSLOTH_SD_CPP_WINDOWS_ROCM", raising = False)
    monkeypatch.setattr(backend, "_on_windows", lambda: False, raising = False)
    assert backend.preferred_accelerator("rocm") == "rocm"
    monkeypatch.setattr(backend, "_on_windows", lambda: True, raising = False)
    monkeypatch.setattr(backend, "_windows_dll_search_dirs", lambda: [], raising = False)
    assert backend.preferred_accelerator("cuda") == "cuda"
    assert backend.preferred_accelerator("vulkan") == "vulkan"


def test_an_unstartable_rocm_bundle_falls_to_the_vulkan_build(tmp_path, monkeypatch, backend):
    """A fresh install with a refused ROCm bundle ends on Vulkan and records the host."""
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "sd-home" / "studio"))
    sdmod._INSTALLED_ACCELERATOR_MEMO.clear()
    root = sdmod.default_install_dir()
    _stub_release(
        monkeypatch,
        {"rocm": _bundle(_refused_binary()), "vulkan": _bundle(_starting_binary(), "vk/")},
    )
    monkeypatch.setattr(backend, "_managed_tree_in_use", lambda: False)
    monkeypatch.setattr(backend, "is_managed_binary", lambda _b: True)

    server = backend.ensure_sd_server_binary(accelerator = "rocm")

    assert server is not None and Path(server) == root / "vk" / _SERVER
    assert sdmod.read_install_record(root).get("accelerator") == "vulkan"
    assert backend.accelerator_runtime_failed("rocm") is True
    assert backend.preferred_accelerator("rocm") == "vulkan"
    cli = backend.ensure_sd_cpp_binary(accelerator = backend.preferred_accelerator("rocm"))
    assert cli is not None and Path(cli) == root / "vk" / _CLI


def test_the_refusal_record_survives_a_fallback_from_another_release(
    tmp_path, monkeypatch, backend
):
    """The record applies even when the Vulkan fallback comes from a non-pinned release."""
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "sd-home" / "studio"))
    sdmod._INSTALLED_ACCELERATOR_MEMO.clear()
    root = sdmod.default_install_dir()
    _stub_release(
        monkeypatch,
        {"rocm": _bundle(_refused_binary()), "vulkan": _bundle(_starting_binary(), "vk/")},
    )
    monkeypatch.setattr(sdmod, "_pinned_tag", lambda: "master-982-0518b34-uaba3a1c")
    monkeypatch.setattr(backend, "_managed_tree_in_use", lambda: False)
    monkeypatch.setattr(backend, "is_managed_binary", lambda _b: True)

    server = backend.ensure_sd_server_binary(accelerator = "rocm")

    assert server is not None
    assert sdmod.read_install_record(root).get("tag") != "master-982-0518b34-uaba3a1c"
    assert backend.accelerator_runtime_failed("rocm") is True
    assert backend.preferred_accelerator("rocm") == "vulkan"


def _dll_dir(path: Path, *dlls: str) -> Path:
    path.mkdir(parents = True, exist_ok = True)
    for dll in dlls:
        (path / dll).write_bytes(b"")
    return path


@pytest.fixture
def hip_sdk(tmp_path, monkeypatch):
    """HIP SDK 7.2 as its MSI leaves it: HIP_PATH (with its trailing separator) set, bin dir not on PATH."""
    program_files = tmp_path / "Program Files"
    root = program_files / "AMD" / "ROCm" / "7.2"
    bindir = _dll_dir(root / "bin", "hipblas.dll", "rocblas.dll", "amdhip64_7.dll")
    monkeypatch.setenv("ProgramFiles", str(program_files))
    monkeypatch.setenv("HIP_PATH", str(root) + os.sep)
    monkeypatch.delenv("ROCM_PATH", raising = False)
    return types.SimpleNamespace(bin = bindir, program_files = program_files)


def _rocm_bundle(tmp_path: Path) -> Path:
    """The upstream win-rocm layout: amdhip64_7.dll beside sd-cli, hipblas.dll left to the SDK."""
    return _dll_dir(tmp_path / "bundle", _CLI, "stable-diffusion.dll", "amdhip64_7.dll") / _CLI


def test_the_hip_sdk_is_found_through_hip_path(hip_sdk):
    assert sdmod.windows_hip_sdk_bin_dirs() == [str(hip_sdk.bin)]


def test_without_hip_path_the_default_location_is_searched_newest_7x_first(hip_sdk, monkeypatch):
    monkeypatch.delenv("HIP_PATH")
    rocm = hip_sdk.program_files / "AMD" / "ROCm"
    older = _dll_dir(rocm / "7.1" / "bin", "hipblas.dll", "amdhip64_7.dll")
    # A 6.x hipblas.dll binds amdhip64_6.dll, not the bundle's HIP 7 runtime.
    _dll_dir(rocm / "6.4" / "bin", "hipblas.dll", "amdhip64_6.dll")
    assert sdmod.windows_hip_sdk_bin_dirs() == [str(hip_sdk.bin), str(older)]


def test_no_hip_sdk_finds_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("ProgramFiles", str(tmp_path / "Program Files"))
    monkeypatch.delenv("HIP_PATH", raising = False)
    monkeypatch.delenv("ROCM_PATH", raising = False)
    assert sdmod.windows_hip_sdk_bin_dirs() == []


def _search_path(env: dict, var: str) -> list[str]:
    return env[var].split(os.pathsep)


def test_the_rocm_bundle_child_gets_the_hip_sdk_after_the_callers_path(tmp_path, hip_sdk):
    from core.inference import sd_cpp_engine

    binary = _rocm_bundle(tmp_path)
    var = sd_cpp_engine._lib_path_var()
    env = sd_cpp_engine.runtime_env(str(binary), {var: "caller-dir"})
    assert _search_path(env, var) == [str(binary.parent.resolve()), "caller-dir", str(hip_sdk.bin)]
    probe = sdmod._probe_env(binary)
    assert _search_path(probe, var)[0] == str(binary.parent)
    assert _search_path(probe, var)[-1] == str(hip_sdk.bin)


def test_other_bundles_and_a_bundled_hipblas_get_no_hip_sdk(tmp_path, hip_sdk):
    from core.inference import sd_cpp_engine

    vulkan = _dll_dir(tmp_path / "vk", _CLI, "stable-diffusion.dll") / _CLI
    rocm_with_hipblas = _dll_dir(tmp_path / "full", _CLI, "amdhip64_7.dll", "hipblas.dll") / _CLI
    var = sd_cpp_engine._lib_path_var()
    for binary in (vulkan, rocm_with_hipblas):
        assert sdmod.hip_sdk_dirs_for_bundle(binary.parent) == []
        env = sd_cpp_engine.runtime_env(str(binary), {var: "caller-dir"})
        assert str(hip_sdk.bin) not in _search_path(env, var)


@pytest.fixture
def windows_path_only(tmp_path, monkeypatch, backend):
    system_root = tmp_path / "Windows"
    (system_root / "System32").mkdir(parents = True)
    monkeypatch.setenv("SystemRoot", str(system_root))
    monkeypatch.setenv("PATH", str(_dll_dir(tmp_path / "on-path")))
    monkeypatch.delenv("UNSLOTH_SD_CPP_WINDOWS_ROCM", raising = False)
    monkeypatch.setattr(backend, "_on_windows", lambda: True, raising = False)
    return backend


def test_windows_rocm_with_the_hip_sdk_installed_off_path_keeps_rocm(windows_path_only, hip_sdk):
    assert windows_path_only.windows_hip_runtime_missing_dlls() == []
    assert windows_path_only.preferred_accelerator("rocm") == "rocm"


def test_windows_rocm_without_any_hip_sdk_still_starts_at_vulkan(
    windows_path_only, tmp_path, monkeypatch
):
    monkeypatch.setenv("ProgramFiles", str(tmp_path / "Program Files"))
    monkeypatch.delenv("HIP_PATH", raising = False)
    monkeypatch.delenv("ROCM_PATH", raising = False)
    assert windows_path_only.windows_hip_runtime_missing_dlls() == ["hipblas.dll"]
    assert windows_path_only.preferred_accelerator("rocm") == "vulkan"
