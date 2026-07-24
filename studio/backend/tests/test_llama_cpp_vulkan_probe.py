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
from core.inference.llama_cpp import (  # noqa: E402
    LlamaCppBackend,
    _llama_lib_dir,
    _vulkan_lib_filename,
)

MIB = 1024 * 1024
GIB = 1024 * MIB


def _make_vulkan_install(tmp_path: Path) -> str:
    """A binary whose sibling dir holds the Vulkan ggml lib, so the
    reader's ``is_vulkan_backend`` sibling-file check passes."""
    bindir = tmp_path / "build" / "bin"
    bindir.mkdir(parents = True)
    binary = bindir / ("llama-server.exe" if sys.platform == "win32" else "llama-server")
    binary.write_bytes(b"stub")
    (bindir / _vulkan_lib_filename()).write_bytes(b"stub")
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
) -> str:
    row = f"{idx}\t{free_bytes}\t{is_igpu}\t{total_bytes}"
    return f"{row}\t{name}" if name is not None else row


def test_integrated_gpu_leaves_host_margin(tmp_path):
    binary = _make_vulkan_install(tmp_path)
    # iGPU with 30 GiB free; reserve a flat 1024 MiB (llama.cpp --fit-target).
    # total stays 0: shared system RAM is not a VRAM budget for the fit.
    rows = [_row(0, 30 * GIB, is_igpu = 1, total_bytes = 32 * GIB)]
    with _mock_probe(rows):
        gpus = LlamaCppBackend._get_gpu_free_memory_vulkan(binary)
    assert gpus == [(0, 30 * 1024 - 1024, 0)], gpus


def test_discrete_gpu_free_is_untouched_and_total_passed_through(tmp_path):
    binary = _make_vulkan_install(tmp_path)
    # 6 GiB free on a partially occupied 24 GiB card: free is untouched and the
    # real total flows through so the fit reserves absolute headroom (CUDA/ROCm
    # parity) instead of the looser free*frac budget.
    rows = [_row(0, 6 * GIB, is_igpu = 0, total_bytes = 24 * GIB)]
    with _mock_probe(rows):
        gpus = LlamaCppBackend._get_gpu_free_memory_vulkan(binary)
    assert gpus == [(0, 6 * 1024, 24 * 1024)], gpus


def test_vulkan_device_info_exposes_names_and_pinnable_ordinals(tmp_path):
    binary = _make_vulkan_install(tmp_path)
    rows = [
        _row(0, 6 * GIB, is_igpu = 0, total_bytes = 24 * GIB, name = "Radeon W7900"),
        _row(1, 7 * GIB, is_igpu = 0, total_bytes = 8 * GIB, name = "Radeon W7500"),
    ]
    with _mock_probe(rows):
        devices = LlamaCppBackend._get_vulkan_gpu_info(binary)

    assert [(d["index"], d["index_kind"], d["name"]) for d in devices] == [
        (0, "vulkan", "Radeon W7900"),
        (1, "vulkan", "Radeon W7500"),
    ]
    assert devices[0]["memory_total_gb"] == 24
    assert devices[0]["vram_free_gb"] == 6
    assert devices[0]["vram_used_gb"] == 18


def test_vulkan_device_info_accepts_legacy_four_column_probe(tmp_path):
    binary = _make_vulkan_install(tmp_path)
    with _mock_probe([_row(2, 7 * GIB, is_igpu = 0, total_bytes = 8 * GIB)]):
        devices = LlamaCppBackend._get_vulkan_gpu_info(binary)
    assert devices[0]["name"] == "Vulkan2"


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
    (bindir / (_vulkan_lib_filename() + ".0")).write_bytes(b"stub")
    rows = [_row(0, 23 * GIB, is_igpu = 0, total_bytes = 24 * GIB)]
    with _mock_probe(rows):
        gpus = LlamaCppBackend._get_gpu_free_memory_vulkan(str(binary))
    assert gpus == [(0, 23 * 1024, 24 * 1024)], gpus


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
