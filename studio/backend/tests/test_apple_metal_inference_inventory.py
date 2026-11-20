# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""An Apple Silicon host detected as CPU (GGUF-only install, or MLX missing) still runs GGUFs on
Metal, so /api/system's inference inventory must list the unified pool instead of no GPU."""

import builtins
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from core.inference.llama_cpp import (
    _metal_device_is_paravirtual as _REAL_FULL_PROBE,
    _metal_device_looks_paravirtual_quick as _REAL_QUICK_PROBE,
)
from utils.hardware import DeviceType
from utils.hardware import hardware as hw

GiB = 1024**3


@pytest.fixture(autouse = True)
def _fresh_working_set_cache():
    hw._apple_metal_working_set_bytes = 0
    yield
    hw._apple_metal_working_set_bytes = 0


def _agx_must_not_be_read():
    raise AssertionError("system-wide AGX usage must not reach the model-switch budget")


def _metal_info(
    *,
    apple = True,
    device = DeviceType.CPU,
    offloads = True,
    working_set = lambda **_: int(10.67 * GiB),
    total = 16 * GiB,
):
    vm = SimpleNamespace(total = total, available = 6 * GiB)
    with (
        patch.object(hw, "is_apple_silicon", return_value = apple),
        patch.object(hw, "get_device", return_value = device),
        patch.object(hw, "_llama_cpp_offloads_to_metal", return_value = offloads),
        patch.object(hw, "_metal_recommended_working_set_bytes", working_set),
        patch.object(hw, "_read_apple_gpu_stats", _agx_must_not_be_read),
        patch("psutil.virtual_memory", return_value = vm),
    ):
        return hw.get_apple_metal_inference_gpu_info()


def test_cpu_classified_mac_lists_one_shared_metal_device():
    info = _metal_info()
    assert info["available"] is True
    assert info["backend"] == "metal"
    assert info["gguf_gpu_ids_supported"] is False
    (device,) = info["devices"]
    assert device["memory_total_gb"] == 16.0
    assert device["shared_memory"] is True
    assert device["shared_memory_host_backed_gb"] == 16.0
    assert device["metal_working_set_gb"] == pytest.approx(10.67)
    # AGX in-use is system-wide, not the resident model's share.
    assert "vram_used_gb" not in device


def test_no_metal_offload_means_no_metal_inventory():
    # A -DGGML_METAL=OFF build or a paravirtual GPU: llama.cpp runs on the CPU.
    assert _metal_info(offloads = False) is None


def test_the_polled_probe_is_told_not_to_import():
    seen = []

    def probe(**kwargs):
        seen.append(kwargs)
        return int(10.67 * GiB)

    _metal_info(working_set = probe)
    assert seen == [{"allow_import": False}]


def test_unknown_working_set_falls_back_to_total_ram_like_the_loader():
    (device,) = _metal_info(working_set = lambda **_: 0)["devices"]
    assert device["metal_working_set_gb"] == 16.0
    assert "vram_used_gb" not in device


def test_working_set_probe_failure_falls_back_to_total_ram():
    def boom(**_):
        raise RuntimeError("no metal")

    (device,) = _metal_info(working_set = boom)["devices"]
    assert device["metal_working_set_gb"] == 16.0


def test_a_read_working_set_is_kept():
    calls = []

    def probe(**_):
        calls.append(1)
        return int(10.67 * GiB)

    _metal_info(working_set = probe)
    _metal_info(working_set = probe)
    assert len(calls) == 1


def test_an_unread_working_set_is_retried_once_a_module_is_loaded():
    readings = iter([0, int(10.67 * GiB)])
    probe = lambda **_: next(readings)  # noqa: E731
    (first,) = _metal_info(working_set = probe)["devices"]
    (second,) = _metal_info(working_set = probe)["devices"]
    assert first["metal_working_set_gb"] == 16.0
    assert second["metal_working_set_gb"] == pytest.approx(10.67)


@pytest.mark.parametrize(
    "apple, device",
    [(False, DeviceType.CPU), (True, DeviceType.MLX), (False, DeviceType.CUDA)],
)
def test_other_hosts_keep_their_inventory(apple, device):
    # MLX already publishes the device through the training inventory; off Apple there is no Metal.
    assert _metal_info(apple = apple, device = device) is None


def test_unreadable_total_memory_reports_nothing():
    assert _metal_info(total = 0) is None


def _fake_mlx(rec_bytes):
    metal = SimpleNamespace(
        is_available = lambda: True,
        device_info = lambda: {"max_recommended_working_set_size": rec_bytes},
    )
    return SimpleNamespace(metal = metal, device_info = metal.device_info)


def _fake_torch(rec_bytes):
    return SimpleNamespace(
        backends = SimpleNamespace(mps = SimpleNamespace(is_available = lambda: True)),
        mps = SimpleNamespace(recommended_max_memory = lambda: rec_bytes),
    )


def test_probe_without_import_never_imports_mlx_or_torch():
    imported = []
    real_import = builtins.__import__

    def guard(name, *args, **kwargs):
        if name.split(".")[0] in {"mlx", "torch"}:
            imported.append(name)
            raise AssertionError(f"polled probe imported {name}")
        return real_import(name, *args, **kwargs)

    modules = {k: v for k, v in sys.modules.items() if k.split(".")[0] not in {"mlx", "torch"}}
    with patch.dict(sys.modules, modules, clear = True), patch.object(builtins, "__import__", guard):
        assert hw._metal_recommended_working_set_bytes(allow_import = False) == 0
    assert imported == []


def test_probe_without_import_reads_modules_already_loaded():
    with patch.dict(sys.modules, {"mlx.core": _fake_mlx(int(10.67 * GiB))}):
        assert hw._metal_recommended_working_set_bytes(allow_import = False) == int(10.67 * GiB)
    with patch.dict(sys.modules, {"mlx.core": None, "torch": _fake_torch(int(9 * GiB))}):
        assert hw._metal_recommended_working_set_bytes(allow_import = False) == 9 * GiB


# ── Does llama.cpp put layers on Metal? ──────────────────────────────────────


def _llama_install(tmp_path, libs, marker_backend = None):
    """A llama-server with the given ggml libraries beside it, and optionally a prebuilt marker."""
    bin_dir = tmp_path / "llama.cpp" / "build" / "bin"
    bin_dir.mkdir(parents = True)
    server = bin_dir / "llama-server"
    server.write_bytes(b"\xcf\xfa\xed\xfe")  # Mach-O magic, not a wrapper script
    for lib in libs:
        (bin_dir / lib).write_bytes(b"")
    if marker_backend is not None:
        import json

        (tmp_path / "llama.cpp" / "UNSLOTH_PREBUILT_INFO.json").write_text(
            json.dumps({"backend": marker_backend, "tag": "b0"})
        )
    return str(server)


@pytest.mark.parametrize(
    "libs, marker, expected",
    [
        # The prebuilt macOS bundle, as on a real M1.
        (["libggml-base.dylib", "libggml-cpu.dylib", "libggml-metal.dylib"], "metal", True),
        # setup.sh's -DGGML_METAL=OFF fallback after a failed Metal build: no libggml-metal.
        (["libggml-base.dylib", "libggml-cpu.dylib", "libggml-blas.dylib"], None, False),
        # The library set wins over a stale marker beside it.
        (["libggml-base.dylib", "libggml-cpu.dylib"], "metal", False),
        # Versioned names, as the prebuilt also ships them.
        (["libggml-base.0.dylib", "libggml-metal.0.20.1.dylib"], None, True),
        # Only a loadable library name counts: dyld never opens a renamed one.
        (["libggml-base.dylib", "libggml-cpu.dylib", "libggml-metal.dylib.disabled"], "metal", False),
        # A static build: only the marker can tell.
        ([], "metal", True),
        ([], None, False),
    ],
)
def test_build_has_metal_reads_the_installed_files(tmp_path, libs, marker, expected):
    from core.inference.llama_cpp import LlamaCppBackend
    from utils import llama_cpp_freshness

    llama_cpp_freshness._marker_cache.clear()
    assert LlamaCppBackend._build_has_metal(_llama_install(tmp_path, libs, marker)) is expected


def test_build_has_metal_follows_the_installer_wrapper(tmp_path):
    # ~/.unsloth/llama.cpp/llama-server is a wrapper script; the libraries sit beside the binary it runs.
    from core.inference.llama_cpp import LlamaCppBackend

    server = _llama_install(tmp_path, ["libggml-base.dylib", "libggml-metal.dylib"])
    wrapper = tmp_path / "llama.cpp" / "llama-server"
    wrapper.write_text('#!/bin/sh\nexec "$(dirname "$0")/build/bin/llama-server" "$@"\n')
    (tmp_path / "llama.cpp" / "libggml-base.dylib").write_bytes(b"")  # a CPU-only set at the top
    assert LlamaCppBackend._build_has_metal(str(wrapper)) is True
    assert server.endswith("build/bin/llama-server")


@pytest.mark.parametrize(
    "value, hides",
    [
        (None, False),
        ("1", False),
        ("2", False),
        ("3x", False),  # atoi reads the leading digits
        ("0", True),
        (" 0", True),
        ("", True),  # set but empty: atoi gives 0
        ("off", True),
        ("-1", True),
    ],
)
def test_metal_devices_env_follows_ggml_metal_atoi(value, hides):
    from core.inference.llama_cpp import _metal_devices_env_hides_all

    env = {} if value is None else {"GGML_METAL_DEVICES": value}
    assert _metal_devices_env_hides_all(env) is hides


@pytest.fixture
def _fresh_paravirtual_caches(monkeypatch):
    # conftest stubs both probes so a paravirtual macOS runner reads as bare metal; restore them.
    import core.inference.llama_cpp as llama_cpp

    monkeypatch.setattr(llama_cpp, "_metal_device_is_paravirtual", _REAL_FULL_PROBE)
    monkeypatch.setattr(llama_cpp, "_metal_device_looks_paravirtual_quick", _REAL_QUICK_PROBE)
    _REAL_FULL_PROBE.cache_clear()
    _REAL_QUICK_PROBE.cache_clear()
    yield llama_cpp
    _REAL_FULL_PROBE.cache_clear()
    _REAL_QUICK_PROBE.cache_clear()


def _hw_model(monkeypatch, llama_cpp, model, calls):
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.delenv("UNSLOTH_ALLOW_PARAVIRTUAL_METAL", raising = False)

    def run(cmd, *args, **kwargs):
        calls.append(cmd[0])
        return SimpleNamespace(stdout = model)

    monkeypatch.setattr(llama_cpp.subprocess, "run", run)


@pytest.mark.parametrize(
    "model, expected", [("VirtualMac2,1", True), ("VMM-x86_64", True), ("Macmini9,1", False)]
)
def test_polled_paravirtual_check_uses_hw_model_only(
    monkeypatch, _fresh_paravirtual_caches, model, expected
):
    llama_cpp = _fresh_paravirtual_caches
    calls = []
    _hw_model(monkeypatch, llama_cpp, model, calls)
    modules = {k: v for k, v in sys.modules.items() if k.split(".")[0] != "mlx"}
    with patch.dict(sys.modules, modules, clear = True):
        assert llama_cpp._metal_offload_forced_off_for_polling() is expected
        assert "mlx.core" not in sys.modules
    # sysctl only: system_profiler takes seconds, and it would run on every poll until cached.
    assert calls == ["sysctl"]


def test_polled_paravirtual_check_honours_the_override(monkeypatch, _fresh_paravirtual_caches):
    llama_cpp = _fresh_paravirtual_caches
    calls = []
    _hw_model(monkeypatch, llama_cpp, "VirtualMac2,1", calls)
    monkeypatch.setenv("UNSLOTH_ALLOW_PARAVIRTUAL_METAL", "1")
    assert llama_cpp._metal_offload_forced_off_for_polling() is False
    assert calls == []


def test_polled_paravirtual_check_prefers_the_loaders_full_answer(
    monkeypatch, _fresh_paravirtual_caches
):
    # A VM hidden from hw.model but caught by system_profiler during a load: the badge follows the load.
    llama_cpp = _fresh_paravirtual_caches
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.delenv("UNSLOTH_ALLOW_PARAVIRTUAL_METAL", raising = False)
    outputs = {"sysctl": "Macmini9,1", "system_profiler": "Chipset Model: Apple Paravirtual device"}
    monkeypatch.setattr(
        llama_cpp.subprocess, "run", lambda cmd, *a, **k: SimpleNamespace(stdout = outputs[cmd[0]])
    )
    modules = {k: v for k, v in sys.modules.items() if k.split(".")[0] != "mlx"}
    with patch.dict(sys.modules, modules, clear = True):
        assert llama_cpp._metal_offload_forced_off_for_polling() is False
        assert llama_cpp._metal_device_is_paravirtual() is True
        assert llama_cpp._metal_offload_forced_off_for_polling() is True


@pytest.mark.parametrize(
    "has_metal, paravirtual, metal_devices, expected",
    [
        (True, False, None, True),
        (False, False, None, False),
        (True, True, None, False),
        # llama-server inherits Studio's environment, and ggml-metal then registers no device.
        (True, False, "0", False),
        (True, False, "1", True),
    ],
)
def test_offload_needs_a_metal_build_on_a_real_gpu(
    monkeypatch, has_metal, paravirtual, metal_devices, expected
):
    import core.inference.llama_cpp as llama_cpp

    if metal_devices is None:
        monkeypatch.delenv("GGML_METAL_DEVICES", raising = False)
    else:
        monkeypatch.setenv("GGML_METAL_DEVICES", metal_devices)
    with (
        patch.object(llama_cpp.LlamaCppBackend, "_build_has_metal", return_value = has_metal),
        patch.object(llama_cpp, "_metal_offload_forced_off_for_polling", return_value = paravirtual),
    ):
        assert hw._llama_cpp_offloads_to_metal() is expected
