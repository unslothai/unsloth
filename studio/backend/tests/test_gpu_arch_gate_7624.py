# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Regression tests for #7624: multi-GPU auto-selection on ROCm must not pick a
device the installed llama.cpp prebuilt has no kernels for.

Covers the two pieces of the fix: _installed_llama_gfx_archs (reads the
mapped_targets list recorded in UNSLOTH_PREBUILT_INFO.json at install time)
and the per-device arch gate inside _get_gpu_memory's torch fallback.
"""

import json
import subprocess
import sys
import types

import pytest

from core.inference.llama_cpp import LlamaCppBackend


def _write_info(tmp_path, payload):
    (tmp_path / "UNSLOTH_PREBUILT_INFO.json").write_text(json.dumps(payload), encoding = "utf-8")


class TestInstalledLlamaGfxArchs:
    def test_reads_mapped_targets(self, tmp_path, monkeypatch):
        monkeypatch.setenv("UNSLOTH_LLAMA_CPP_PATH", str(tmp_path))
        _write_info(tmp_path, {"mapped_targets": ["gfx1100", "GFX1101", "gfx1102:xnack-"]})
        archs = LlamaCppBackend._installed_llama_gfx_archs()
        assert archs == frozenset({"gfx1100", "gfx1101", "gfx1102"})

    def test_missing_file_is_unknown(self, tmp_path, monkeypatch):
        monkeypatch.setenv("UNSLOTH_LLAMA_CPP_PATH", str(tmp_path))
        monkeypatch.setattr("core.inference.llama_cpp.Path.home", lambda: tmp_path / "nohome")
        assert LlamaCppBackend._installed_llama_gfx_archs() is None

    def test_pre_7624_install_is_unknown(self, tmp_path, monkeypatch):
        # Older installs have no mapped_targets key: fail open.
        monkeypatch.setenv("UNSLOTH_LLAMA_CPP_PATH", str(tmp_path))
        _write_info(tmp_path, {"asset": "app-windows-x64-rocm-gfx110X.zip"})
        assert LlamaCppBackend._installed_llama_gfx_archs() is None

    def test_empty_targets_is_unknown(self, tmp_path, monkeypatch):
        # Non-ROCm bundles record []: fail open rather than drop every GPU.
        monkeypatch.setenv("UNSLOTH_LLAMA_CPP_PATH", str(tmp_path))
        _write_info(tmp_path, {"mapped_targets": []})
        assert LlamaCppBackend._installed_llama_gfx_archs() is None


class TestKernelImageInvalidMarker:
    def test_detects_rocm_arch_mismatch(self):
        tail = (
            "load_model: loading model 'x.gguf'\n"
            "E ROCm error: device kernel image is invalid\n"
            "E   current device: 0, in function ggml_cuda_kernel_launch"
        )
        assert LlamaCppBackend._kernel_image_invalid(tail)

    def test_ignores_other_crashes(self):
        assert not LlamaCppBackend._kernel_image_invalid("out of memory")
        assert not LlamaCppBackend._kernel_image_invalid("")
        assert not LlamaCppBackend._kernel_image_invalid(None)


class _FakeProps:
    def __init__(self, arch):
        self.gcnArchName = arch


def _fake_torch(archs, free_mib):
    torch = types.ModuleType("torch")
    torch.version = types.SimpleNamespace(hip = "7.1.0")
    torch.cuda = types.SimpleNamespace(
        is_available = lambda: True,
        device_count = lambda: len(archs),
        mem_get_info = lambda o: (free_mib[o] * 1024 * 1024, 32 * 1024**3),
        get_device_properties = lambda o: _FakeProps(archs[o]),
    )
    return torch


@pytest.fixture
def rocm_probe_env(monkeypatch):
    """Force _get_gpu_memory down the torch/ROCm fallback, hermetically."""

    def _no_nvidia_smi(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(subprocess, "run", _no_nvidia_smi)
    # No real llama.cpp install: skips the Vulkan probe branch.
    monkeypatch.setattr(LlamaCppBackend, "_find_llama_server_binary", staticmethod(lambda: None))
    for var in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        monkeypatch.delenv(var, raising = False)


class TestGpuArchGate:
    def test_igpu_dropped_when_arch_unsupported(self, tmp_path, monkeypatch, rocm_probe_env):
        # dGPU (gfx1101) + iGPU (gfx1036) whose shared-RAM "free memory"
        # outranks the dGPU's free VRAM -- the #7624 shape. Only the dGPU
        # may survive enumeration.
        monkeypatch.setenv("UNSLOTH_LLAMA_CPP_PATH", str(tmp_path))
        _write_info(tmp_path, {"mapped_targets": ["gfx1100", "gfx1101", "gfx1102", "gfx1103"]})
        monkeypatch.setitem(
            sys.modules, "torch", _fake_torch(["gfx1101", "gfx1036"], [12049, 12176])
        )
        assert LlamaCppBackend._get_gpu_free_memory() == [(0, 12049)]

    def test_unknown_coverage_keeps_all_devices(self, tmp_path, monkeypatch, rocm_probe_env):
        # No prebuilt info (source build): behavior unchanged.
        monkeypatch.setenv("UNSLOTH_LLAMA_CPP_PATH", str(tmp_path))
        monkeypatch.setattr("core.inference.llama_cpp.Path.home", lambda: tmp_path / "nohome")
        monkeypatch.setitem(
            sys.modules, "torch", _fake_torch(["gfx1101", "gfx1036"], [12049, 12176])
        )
        assert LlamaCppBackend._get_gpu_free_memory() == [(0, 12049), (1, 12176)]

    def test_unknown_device_arch_fails_open(self, tmp_path, monkeypatch, rocm_probe_env):
        # A device torch can't describe is kept, never silently dropped.
        monkeypatch.setenv("UNSLOTH_LLAMA_CPP_PATH", str(tmp_path))
        _write_info(tmp_path, {"mapped_targets": ["gfx1101"]})
        monkeypatch.setitem(sys.modules, "torch", _fake_torch(["gfx1101", ""], [12049, 12176]))
        assert LlamaCppBackend._get_gpu_free_memory() == [(0, 12049), (1, 12176)]
