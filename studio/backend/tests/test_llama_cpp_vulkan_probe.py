# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Vulkan free-VRAM reader regression tests on a synthetic probe output.

Covers the post-probe handling in
``LlamaCppBackend._get_gpu_free_memory_vulkan``:

  * integrated GPUs (probe reports is_igpu=1) leave a flat per-device host
    margin matching llama.cpp's --fit-target, so context auto-sizing can't
    over-commit shared RAM, and report total 0 (shared RAM is not a budget),
  * discrete GPUs (is_igpu=0) keep their free untouched and pass their real
    total through so the fit can reserve absolute headroom,
  * an inherited ``GGML_VK_VISIBLE_DEVICES`` is passed through to ggml unchanged
    (ggml applies it), not stripped or filtered in Python -- the probe reports
    ggml's compact ordinal, which load_model pins with ``--device Vulkan<i>``.

The ggml Vulkan library is never loaded: subprocess.run is mocked to emit
the tab-separated lines the real ``_vulkan_probe.py`` would print.
"""

from __future__ import annotations

import subprocess
import sys
import types as _types
from pathlib import Path
from unittest import mock

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

import importlib as _importlib  # noqa: E402


def _maybe_stub(name: str, builder):
    try:
        _importlib.import_module(name)
    except ImportError:
        sys.modules[name] = builder()


def _build_loggers_stub():
    m = _types.ModuleType("loggers")
    m.get_logger = lambda name: __import__("logging").getLogger(name)
    return m


_maybe_stub("loggers", _build_loggers_stub)
_maybe_stub("structlog", lambda: _types.ModuleType("structlog"))

from core.inference import llama_cpp as _llama_mod  # noqa: E402
from core.inference._vulkan_probe import _igpu_flags_and_names  # noqa: E402
from core.inference.llama_cpp import (  # noqa: E402
    LlamaCppBackend,
    _llama_lib_dir,
)

MIB = 1024 * 1024
GIB = 1024 * MIB


class _FakeCFunction:
    def __init__(self, result):
        self.result = result

    def __call__(self, *_args):
        return self.result


def test_missing_description_symbol_keeps_igpu_detection():
    base = _types.SimpleNamespace(
        ggml_backend_reg_dev_count = _FakeCFunction(1),
        ggml_backend_reg_dev_get = _FakeCFunction(1),
        ggml_backend_dev_type = _FakeCFunction(2),
        ggml_backend_dev_name = _FakeCFunction(b"Legacy Vulkan iGPU"),
    )
    lib = _types.SimpleNamespace(ggml_backend_vk_reg = _FakeCFunction(1))

    flags, names, type_known = _igpu_flags_and_names(base, lib, 1)

    assert flags == [True]
    assert names == ["Legacy Vulkan iGPU"]
    assert type_known == [True]


def _make_vulkan_install(tmp_path: Path) -> str:
    """A binary whose sibling dir holds the Vulkan ggml lib, so the
    reader's ``is_vulkan_backend`` sibling-file check passes."""
    bindir = tmp_path / "build" / "bin"
    bindir.mkdir(parents = True)
    binary = bindir / ("llama-server.exe" if sys.platform == "win32" else "llama-server")
    binary.write_bytes(b"stub")
    vulkan_lib = "ggml-vulkan.dll" if sys.platform == "win32" else "libggml-vulkan.so"
    (bindir / vulkan_lib).write_bytes(b"stub")
    return str(binary)


def _mock_probe(rows: list[str], captured_env: dict | None = None):
    """Patch subprocess.run so the _vulkan_probe.py call returns ``rows``
    (already tab-formatted), recording the env it was launched with."""
    real_run = subprocess.run

    def fake_run(cmd, *args, **kwargs):
        if isinstance(cmd, list) and any("_vulkan_probe" in str(c) for c in cmd):
            if captured_env is not None:
                captured_env.clear()
                captured_env.update(kwargs.get("env") or {})
            return subprocess.CompletedProcess(
                args = cmd, returncode = 0, stdout = "\n".join(rows), stderr = ""
            )
        return real_run(cmd, *args, **kwargs)

    return mock.patch("subprocess.run", side_effect = fake_run)


def _row(
    idx: int,
    free_bytes: int,
    is_igpu: int,
    total_bytes: int = 0,
    name: str | None = None,
    type_known: int | None = 1,
) -> str:
    row = f"{idx}\t{free_bytes}\t{is_igpu}\t{total_bytes}"
    if type_known is None:
        return f"{row}\t{name}" if name is not None else row
    return f"{row}\t{name or ''}\t{type_known}"


def _host_memory(monkeypatch, *, available_mib, total_mib):
    """Set deterministic host-memory bounds for iGPU tests."""
    monkeypatch.setattr(
        LlamaCppBackend, "_available_system_memory_mib", staticmethod(lambda: available_mib)
    )
    monkeypatch.setattr(
        LlamaCppBackend, "_total_system_memory_mib", staticmethod(lambda: total_mib)
    )


def test_integrated_gpu_leaves_host_margin(tmp_path, monkeypatch):
    binary = _make_vulkan_install(tmp_path)
    # iGPU with 30 GiB free; reserve a flat 1024 MiB (llama.cpp --fit-target).
    # total stays 0: shared system RAM is not a VRAM budget for the fit.
    _host_memory(monkeypatch, available_mib = 31 * 1024, total_mib = 32 * 1024)
    rows = [_row(0, 30 * GIB, is_igpu = 1, total_bytes = 32 * GIB)]
    with _mock_probe(rows):
        gpus = LlamaCppBackend._get_gpu_free_memory_vulkan(binary)
    assert gpus == [(0, 30 * 1024 - 1024, 0)], gpus


def test_integrated_gpu_host_bound_does_not_replace_planner_reading(tmp_path, monkeypatch):
    """Placement keeps Vulkan free while admission caps its host-backed share."""
    binary = _make_vulkan_install(tmp_path)
    _host_memory(monkeypatch, available_mib = 4000, total_mib = 31000)
    # 512 MiB of UMA plus a GTT heap half the size of RAM
    rows = [_row(0, 15900 * MIB, is_igpu = 1, total_bytes = 16012 * MIB)]
    with _mock_probe(rows):
        gpus = LlamaCppBackend._get_gpu_free_memory_vulkan(binary)
    assert gpus == [(0, 15900 - 1024, 0)], gpus
    assert LlamaCppBackend._igpu_backed_usable_mib(gpus[0][1]) == 4000 - 2048


def test_integrated_gpu_keeps_a_heap_the_os_cannot_see(tmp_path, monkeypatch):
    """Free memory above MemTotal is a conservative carve-out floor."""
    binary = _make_vulkan_install(tmp_path)
    _host_memory(monkeypatch, available_mib = 13312, total_mib = 32154)
    rows = [_row(0, 108782 * MIB, is_igpu = 1, total_bytes = 114507 * MIB)]
    with _mock_probe(rows):
        gpus = LlamaCppBackend._get_gpu_free_memory_vulkan(binary)
    assert gpus == [(0, 108782 - 1024, 0)], gpus
    expected = (108782 - 32154) + 13312 - 2048
    assert LlamaCppBackend._igpu_backed_usable_mib(gpus[0][1]) == expected


def test_a_carve_out_another_process_holds_is_not_credited(tmp_path, monkeypatch):
    """Use free, not total, to exclude carve-out memory held elsewhere."""
    binary = _make_vulkan_install(tmp_path)
    _host_memory(monkeypatch, available_mib = 4096, total_mib = 32768)
    # The free reading does not exceed MemTotal, so no carve-out is credited.
    rows = [_row(0, 22528 * MIB, is_igpu = 1, total_bytes = 114688 * MIB)]
    with _mock_probe(rows):
        gpus = LlamaCppBackend._get_gpu_free_memory_vulkan(binary)
    assert gpus == [(0, 22528 - 1024, 0)], gpus
    assert LlamaCppBackend._igpu_backed_usable_mib(gpus[0][1]) == 4096 - 2048


def test_wsl_credits_no_heap_beyond_its_own_memtotal(tmp_path, monkeypatch):
    """WSL MemTotal does not describe the adapter's host pool."""
    binary = _make_vulkan_install(tmp_path)
    _host_memory(monkeypatch, available_mib = 6000, total_mib = 8192)
    monkeypatch.setattr(_llama_mod, "_is_wsl", lambda: True)
    rows = [_row(0, 11000 * MIB, is_igpu = 1, total_bytes = 12000 * MIB)]
    with _mock_probe(rows):
        gpus = LlamaCppBackend._get_gpu_free_memory_vulkan(binary)
    assert gpus == [(0, 11000 - 1024, 0)], gpus
    assert LlamaCppBackend._igpu_backed_usable_mib(gpus[0][1]) == 6000 - 2048


def test_unreadable_host_memory_leaves_the_integrated_reading_alone(tmp_path, monkeypatch):
    """Missing host readings leave the Vulkan value unchanged."""
    binary = _make_vulkan_install(tmp_path)
    _host_memory(monkeypatch, available_mib = None, total_mib = None)
    rows = [_row(0, 30 * GIB, is_igpu = 1, total_bytes = 32 * GIB)]
    with _mock_probe(rows):
        gpus = LlamaCppBackend._get_gpu_free_memory_vulkan(binary)
    assert gpus == [(0, 30 * 1024 - 1024, 0)], gpus
    assert LlamaCppBackend._igpu_backed_usable_mib(gpus[0][1]) == 30 * 1024 - 1024


def test_discrete_gpu_free_is_untouched_and_total_passed_through(tmp_path):
    binary = _make_vulkan_install(tmp_path)
    # 6 GiB free on a partially occupied 24 GiB card: free is untouched and the
    # real total flows through so the fit reserves absolute headroom (CUDA/ROCm
    # parity) instead of the looser free*frac budget.
    rows = [_row(0, 6 * GIB, is_igpu = 0, total_bytes = 24 * GIB)]
    with _mock_probe(rows):
        gpus = LlamaCppBackend._get_gpu_free_memory_vulkan(binary)
    assert gpus == [(0, 6 * 1024, 24 * 1024)], gpus


def test_failed_device_type_lookup_keeps_the_snapshot_unknown(tmp_path):
    binary = _make_vulkan_install(tmp_path)
    rows = [_row(0, 6 * GIB, is_igpu = 0, total_bytes = 24 * GIB, type_known = 0)]
    with _mock_probe(rows):
        gpus = LlamaCppBackend._get_gpu_free_memory_vulkan(binary)
    assert gpus == [(0, 6 * 1024, 24 * 1024)]
    assert gpus.known_vulkan_igpus is None


def test_large_discrete_gpu_is_untouched(tmp_path):
    binary = _make_vulkan_install(tmp_path)
    # A 48 GiB discrete card stays untouched regardless of size; only the
    # iGPU flag triggers the host margin, never a VRAM/RAM ratio.
    rows = [_row(0, 47 * GIB, is_igpu = 0, total_bytes = 48 * GIB)]
    with _mock_probe(rows):
        gpus = LlamaCppBackend._get_gpu_free_memory_vulkan(binary)
    assert gpus == [(0, 47 * 1024, 48 * 1024)], gpus


def test_inherited_visible_devices_mask_is_passed_through_to_probe(tmp_path, monkeypatch):
    # The mask is NOT stripped or filtered in Python: ggml parses it in raw
    # physical-device space while this probe reports the compact post-filter
    # ordinal, so mixing spaces would be wrong. It is passed through unchanged
    # so ggml applies it to the same device list the launch will enumerate.
    binary = _make_vulkan_install(tmp_path)
    monkeypatch.setenv("GGML_VK_VISIBLE_DEVICES", "1")
    captured: dict = {}
    rows = [_row(0, 23 * GIB, is_igpu = 0, total_bytes = 24 * GIB)]
    with _mock_probe(rows, captured_env = captured):
        LlamaCppBackend._get_gpu_free_memory_vulkan(binary)
    assert captured.get("GGML_VK_VISIBLE_DEVICES") == "1", captured


def test_vulkan_pin_args_uses_device_names_not_env_mask():
    # Pin by compact device name via --device (the space the probe reports and
    # the registry names), never by writing a compact ordinal into the raw
    # GGML_VK_VISIBLE_DEVICES index space.
    assert LlamaCppBackend._vulkan_pin_args([0]) == ["--device", "Vulkan0"]
    assert LlamaCppBackend._vulkan_pin_args([1, 2]) == ["--device", "Vulkan1,Vulkan2"]
    assert LlamaCppBackend._vulkan_pin_args(None) == []
    assert LlamaCppBackend._vulkan_pin_args([]) == []


def test_vulkan_only_build_is_detected(tmp_path):
    binary = _make_vulkan_install(tmp_path)
    assert LlamaCppBackend._is_vulkan_backend(binary) is True


def test_multi_backend_build_is_not_vulkan_only(tmp_path):
    # A custom build that ships CUDA (or HIP) alongside Vulkan must NOT be
    # treated as Vulkan-only, or its CUDA GPU would be probed/pinned as a Vulkan
    # device; defer to the CUDA/HIP path instead.
    binary = _make_vulkan_install(tmp_path)
    cuda = "ggml-cuda.dll" if sys.platform == "win32" else "libggml-cuda.so"
    (_llama_lib_dir(binary) / cuda).write_bytes(b"stub")
    assert LlamaCppBackend._is_vulkan_backend(binary) is False


@pytest.mark.skipif(sys.platform == "win32", reason = "shell wrapper fallback is POSIX")
def test_shell_wrapper_entrypoint_resolves_to_real_lib_dir(tmp_path):
    # create_exec_entrypoint falls back to a #!/bin/sh wrapper at the install root
    # when it cannot symlink; _find_llama_server_binary returns that root entrypoint,
    # so _llama_lib_dir must follow the wrapper's exec target to build/bin -- else
    # _is_vulkan_backend misses libggml-vulkan.so and the Vulkan probe/pin silently
    # never engage on a valid Vulkan install.
    import os

    binary = _make_vulkan_install(tmp_path)  # tmp_path/build/bin/llama-server + vulkan lib
    bindir = Path(binary).parent
    wrapper = tmp_path / "llama-server"
    wrapper.write_text('#!/bin/sh\nexec "$(dirname "$0")/build/bin/llama-server" "$@"\n')
    os.chmod(wrapper, 0o755)
    assert _llama_lib_dir(str(wrapper)) == bindir
    assert LlamaCppBackend._is_vulkan_backend(str(wrapper)) is True


@pytest.mark.skipif(sys.platform == "win32", reason = "soname versioning is POSIX")
def test_versioned_only_vulkan_soname_is_probed(tmp_path):
    # Split-library install: only the versioned soname libggml-vulkan.so.0 exists
    # (no unversioned dev symlink). The reader must still classify Vulkan and run
    # the probe instead of returning [] and rejecting gpu_ids before launch (#7188).
    bindir = tmp_path / "build" / "bin"
    bindir.mkdir(parents = True)
    binary = bindir / "llama-server"
    binary.write_bytes(b"stub")
    (bindir / "libggml-vulkan.so.0").write_bytes(b"stub")
    rows = [_row(0, 23 * GIB, is_igpu = 0, total_bytes = 24 * GIB)]
    with _mock_probe(rows):
        gpus = LlamaCppBackend._get_gpu_free_memory_vulkan(str(binary))
    assert gpus == [(0, 23 * 1024, 24 * 1024)], gpus


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
