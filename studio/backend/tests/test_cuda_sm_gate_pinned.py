# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The CUDA SM gate checks the GPUs a launch is PINNED to, not "any visible".

RTX 4090 (sm_89, PCI 0) + GTX 1060 (sm_61, PCI 1) under a bundle whose oldest
arch is sm_75: the 4090 passed the old any-visible check for a pick that also
put work on the 1060, and the child aborted ("no kernel image"). Auto placement
now leaves the 1060 out of the pool; an explicit pick naming it is refused up
front. Homogeneous and single-GPU hosts see no change."""

import json
import subprocess
import sys
import types

import pytest

from core.inference.llama_cpp import LlamaCppBackend

_CAPS = {0: 89, 1: 61}


def _binary(tmp_path, sms = ("75", "80", "86", "89", "90", "120")):
    (tmp_path / "UNSLOTH_PREBUILT_INFO.json").write_text(
        json.dumps({"supported_sms": list(sms)}), encoding = "utf-8"
    )
    return str(tmp_path / "build" / "bin" / "llama-server")


@pytest.fixture(autouse = True)
def _host(monkeypatch):
    monkeypatch.setattr(LlamaCppBackend, "_is_vulkan_backend", staticmethod(lambda b = None: False))
    monkeypatch.setattr(
        LlamaCppBackend, "_is_unsloth_managed_binary", staticmethod(lambda _b: True)
    )
    monkeypatch.setattr(
        LlamaCppBackend, "_get_gpu_memory_amd_smi", staticmethod(lambda *a, **k: [])
    )
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising = False)
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")


def _smi(monkeypatch, caps = _CAPS, free = {0: 20000, 1: 5800}):
    def _run(cmd, **_kwargs):
        query = " ".join(cmd)
        if "compute_cap" in query:
            out = "\n".join(f"{i}, {sm // 10}.{sm % 10}" for i, sm in caps.items())
        else:
            out = "\n".join(f"{i}, {free[i]}, 24564" for i in caps)
        return types.SimpleNamespace(returncode = 0, stdout = out + "\n")

    monkeypatch.setattr(subprocess, "run", _run)


class TestPinnedGate:
    def test_a_pin_including_the_old_card_is_refused(self, tmp_path, monkeypatch):
        _smi(monkeypatch)
        binary = _binary(tmp_path)
        error = LlamaCppBackend._cuda_sm_gate_error(binary, [0, 1])
        assert error is not None and "GPU 1 is sm_61" in error
        assert "GPU 0" not in error
        assert LlamaCppBackend._cuda_sm_gate_error(binary, [1]) is not None

    def test_a_pin_on_the_new_card_proceeds(self, tmp_path, monkeypatch):
        _smi(monkeypatch)
        assert LlamaCppBackend._cuda_sm_gate_error(_binary(tmp_path), [0]) is None

    def test_unpinned_keeps_the_any_visible_rule(self, tmp_path, monkeypatch):
        # --fit on owns placement there; refusing it would over-refuse a working 4090.
        _smi(monkeypatch)
        assert LlamaCppBackend._cuda_sm_gate_error(_binary(tmp_path)) is None
        _smi(monkeypatch, caps = {0: 61}, free = {0: 5800})
        assert LlamaCppBackend._cuda_sm_gate_error(_binary(tmp_path)) is not None

    def test_unknown_coverage_or_caps_fail_open(self, tmp_path, monkeypatch):
        _smi(monkeypatch)
        assert LlamaCppBackend._cuda_sm_gate_error(str(tmp_path / "llama-server"), [1]) is None
        monkeypatch.setattr(LlamaCppBackend, "_cuda_compute_caps", staticmethod(lambda: {}))
        assert LlamaCppBackend._cuda_sm_gate_error(_binary(tmp_path), [1]) is None


class TestAutoPool:
    def test_auto_pool_drops_the_card_the_build_cannot_run(self, tmp_path, monkeypatch):
        _smi(monkeypatch)
        binary = _binary(tmp_path)
        assert [g[0] for g in LlamaCppBackend._get_gpu_memory(binary, for_llama_server = True)] == [0]
        # Off the llama-server path (embeddings via torch) every card stays.
        assert [g[0] for g in LlamaCppBackend._get_gpu_memory(binary)] == [0, 1]

    def test_every_card_too_old_keeps_the_pool_for_the_gate(self, tmp_path, monkeypatch):
        _smi(monkeypatch, caps = {0: 61, 1: 61}, free = {0: 5800, 1: 5800})
        binary = _binary(tmp_path)
        assert [g[0] for g in LlamaCppBackend._get_gpu_memory(binary, for_llama_server = True)] == [0, 1]

    def test_homogeneous_and_ptx_forward_hosts_are_unchanged(self, tmp_path, monkeypatch):
        _smi(monkeypatch, caps = {0: 89, 1: 86}, free = {0: 20000, 1: 20000})
        binary = _binary(tmp_path)
        assert [g[0] for g in LlamaCppBackend._get_gpu_memory(binary, for_llama_server = True)] == [0, 1]
        # A legacy sm_50-61 bundle JITs forward onto both: nothing is dropped.
        (tmp_path / "legacy").mkdir()
        legacy = _binary(tmp_path / "legacy", sms = ("50", "61"))
        _smi(monkeypatch)
        assert [g[0] for g in LlamaCppBackend._get_gpu_memory(legacy, for_llama_server = True)] == [0, 1]
        assert LlamaCppBackend._cuda_sm_gate_error(legacy, [0, 1]) is None


def test_the_launch_gate_receives_the_pinned_ids():
    import re
    from pathlib import Path

    import core.inference.llama_cpp as mod

    src = Path(mod.__file__).read_text(encoding = "utf-8")
    assert re.search(
        r"self\._cuda_sm_gate_error\(\s*binary,\s*None if is_vulkan_backend else gpu_indices", src
    )
