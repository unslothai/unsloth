# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A CPU-detected Apple Silicon host still runs GGUFs on Metal; /api/system must list the pool."""

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
        # Some CI runners ship no psutil.
        patch.dict(sys.modules, {"psutil": SimpleNamespace(virtual_memory = lambda: vm)}),
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
    assert "vram_used_gb" not in device


def test_no_metal_offload_means_no_metal_inventory():
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


@pytest.mark.parametrize(
    "apple, device",
    [(False, DeviceType.CPU), (True, DeviceType.MLX), (False, DeviceType.CUDA)],
)
def test_other_hosts_keep_their_inventory(apple, device):
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
    with (
        patch.dict(sys.modules, modules, clear = True),
        patch.object(builtins, "__import__", guard),
        patch.object(hw, "_metal_framework_working_set_bytes", return_value = 7 * GiB),
    ):
        assert hw._metal_recommended_working_set_bytes(allow_import = False) == 7 * GiB
    assert imported == []


def test_probe_without_import_reads_modules_already_loaded():
    with patch.dict(sys.modules, {"mlx.core": _fake_mlx(int(10.67 * GiB))}):
        assert hw._metal_recommended_working_set_bytes(allow_import = False) == int(10.67 * GiB)
    with patch.dict(sys.modules, {"mlx.core": None, "torch": _fake_torch(int(9 * GiB))}):
        assert hw._metal_recommended_working_set_bytes(allow_import = False) == 9 * GiB


def _llama_install(
    tmp_path,
    libs,
    marker_backend = None,
):
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
        (["libggml-base.dylib", "libggml-cpu.dylib", "libggml-metal.dylib"], "metal", True),
        # setup.sh's -DGGML_METAL=OFF fallback.
        (["libggml-base.dylib", "libggml-cpu.dylib", "libggml-blas.dylib"], None, False),
        # The library set wins over a stale marker.
        (["libggml-base.dylib", "libggml-cpu.dylib"], "metal", False),
        (["libggml-base.0.dylib", "libggml-metal.0.20.1.dylib"], None, True),
        # dyld never opens a renamed library.
        (
            ["libggml-base.dylib", "libggml-cpu.dylib", "libggml-metal.dylib.disabled"],
            "metal",
            False,
        ),
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
    from core.inference.llama_cpp import LlamaCppBackend

    server = _llama_install(tmp_path, ["libggml-base.dylib", "libggml-metal.dylib"])
    wrapper = tmp_path / "llama.cpp" / "llama-server"
    wrapper.write_text('#!/bin/sh\nexec "$(dirname "$0")/build/bin/llama-server" "$@"\n')
    (tmp_path / "llama.cpp" / "libggml-base.dylib").write_bytes(b"")  # a CPU-only set at the top
    assert LlamaCppBackend._build_has_metal(str(wrapper)) is True


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
        ("１", True),  # full-width 1: atoi reads only ASCII digits
        (" 1", True),  # no-break space: not C isspace, so atoi stops at 0
        ("\t1", False),  # ASCII whitespace is skipped, as atoi does
    ],
)
def test_metal_devices_env_follows_ggml_metal_atoi(value, hides):
    from core.inference.llama_cpp import _metal_devices_env_hides_all
    env = {} if value is None else {"GGML_METAL_DEVICES": value}
    assert _metal_devices_env_hides_all(env) is hides


@pytest.fixture
def _fresh_paravirtual_caches(monkeypatch):
    # conftest stubs both probes; restore the real ones.
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
    imported = []
    real_import = builtins.__import__

    def guard(name, *args, **kwargs):
        if name.split(".")[0] == "mlx":
            imported.append(name)
        return real_import(name, *args, **kwargs)

    # Blocked, not dropped: re-importing an already loaded mlx.core aborts the process.
    with (
        patch.dict(sys.modules, {"mlx": None, "mlx.core": None}),
        patch.object(builtins, "__import__", guard),
    ):
        assert llama_cpp._metal_offload_forced_off_for_polling() is expected
    assert imported == []
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
    llama_cpp = _fresh_paravirtual_caches
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.delenv("UNSLOTH_ALLOW_PARAVIRTUAL_METAL", raising = False)
    outputs = {"sysctl": "Macmini9,1", "system_profiler": "Chipset Model: Apple Paravirtual device"}
    monkeypatch.setattr(
        llama_cpp.subprocess, "run", lambda cmd, *a, **k: SimpleNamespace(stdout = outputs[cmd[0]])
    )
    # Blocked, not dropped: re-importing an already loaded mlx.core aborts the process.
    with patch.dict(sys.modules, {"mlx": None, "mlx.core": None}):
        assert llama_cpp._metal_offload_forced_off_for_polling() is False
        assert llama_cpp._metal_device_is_paravirtual() is True
        assert llama_cpp._metal_offload_forced_off_for_polling() is True


@pytest.mark.parametrize(
    "has_metal, paravirtual, metal_devices, expected",
    [
        (True, False, None, True),
        (False, False, None, False),
        (True, True, None, False),
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


@pytest.mark.parametrize("metal", [True, False])
def test_system_route_serves_the_metal_inventory_to_a_cpu_mac(monkeypatch, metal):
    import main
    import utils.hardware as hardware

    cpu = {"available": False, "backend": "cpu", "devices": [], "index_kind": "relative"}
    metal_info = {"available": True, "backend": "metal", "devices": [{"index": 0}]}
    monkeypatch.setattr(hardware, "get_backend_visible_gpu_info", lambda: cpu)
    monkeypatch.setattr(hardware, "get_visible_gpu_utilization", lambda: {**cpu})
    monkeypatch.setattr(hardware, "get_vulkan_inference_gpu_info", lambda: None)
    monkeypatch.setattr(hardware, "get_cross_vendor_inference_gpu_info", lambda: None)
    monkeypatch.setattr(hardware, "get_mixed_vulkan_inference_gpu_info", lambda info: None)
    monkeypatch.setattr(
        hardware, "get_apple_metal_inference_gpu_info", lambda: metal_info if metal else None
    )
    monkeypatch.setattr(main, "_system_gpu_cache", None)

    gpu, inference_gpu = main._get_cached_system_gpu_info(SimpleNamespace(debug = lambda *a: None))
    assert gpu["available"] is False
    assert inference_gpu["backend"] == ("metal" if metal else "cpu")
    assert inference_gpu["devices"] == (metal_info["devices"] if metal else [])
