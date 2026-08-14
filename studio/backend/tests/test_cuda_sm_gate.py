# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The CUDA SM gate: a prebuilt installed on one GPU (e.g. a cloud image baked
on a T4) must fail fast when run on a GPU its bundle has no kernels for,
instead of llama-server aborting on every launch attempt."""

import json
import subprocess
import types

import pytest

from core.inference.llama_cpp import LlamaCppBackend


def _binary_with_marker(tmp_path, payload):
    (tmp_path / "UNSLOTH_PREBUILT_INFO.json").write_text(json.dumps(payload), encoding = "utf-8")
    return str(tmp_path / "build" / "bin" / "llama-server")


class TestInstalledLlamaCudaSms:
    def test_reads_supported_sms(self, tmp_path):
        binary = _binary_with_marker(tmp_path, {"supported_sms": ["75", "80", 86, " 89 "]})
        assert LlamaCppBackend._installed_llama_cuda_sms(binary) == frozenset({75, 80, 86, 89})

    def test_no_marker_is_unknown(self, tmp_path):
        assert LlamaCppBackend._installed_llama_cuda_sms(str(tmp_path / "llama-server")) is None

    def test_no_binary_is_unknown(self, monkeypatch):
        monkeypatch.setattr(
            LlamaCppBackend, "_find_llama_server_binary", staticmethod(lambda: None)
        )
        assert LlamaCppBackend._installed_llama_cuda_sms() is None

    @pytest.mark.parametrize("sms", [None, [], ["gfx1100"], ["86", "abc"], "86"])
    def test_missing_or_malformed_is_unknown(self, tmp_path, sms):
        binary = _binary_with_marker(
            tmp_path, {"supported_sms": sms} if sms is not None else {"asset": "x.tar.gz"}
        )
        assert LlamaCppBackend._installed_llama_cuda_sms(binary) is None

    def test_unreadable_marker_is_unknown(self, tmp_path, monkeypatch):
        import utils.llama_cpp_freshness as freshness

        def _boom(_binary):
            raise OSError("marker read failed")

        monkeypatch.setattr(freshness, "read_install_marker", _boom)
        assert LlamaCppBackend._installed_llama_cuda_sms(str(tmp_path / "llama-server")) is None


def _fake_smi(monkeypatch, stdout, returncode = 0):
    def _run(cmd, **_kwargs):
        assert cmd[0] == "nvidia-smi"
        return types.SimpleNamespace(returncode = returncode, stdout = stdout)

    monkeypatch.setattr(subprocess, "run", _run)


class TestCudaComputeCaps:
    def test_parses_index_and_cap(self, monkeypatch):
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising = False)
        _fake_smi(monkeypatch, "0, 9.0\n1, 12.0\n")
        assert LlamaCppBackend._cuda_compute_caps() == {0: 90, 1: 120}

    def test_honors_visible_devices_mask(self, monkeypatch):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
        _fake_smi(monkeypatch, "0, 7.5\n1, 9.0\n")
        assert LlamaCppBackend._cuda_compute_caps() == {1: 90}

    def test_bad_lines_are_skipped(self, monkeypatch):
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising = False)
        _fake_smi(monkeypatch, "0, 9.0\nno-cap-line\n1, N/A\n")
        assert LlamaCppBackend._cuda_compute_caps() == {0: 90}

    def test_probe_failure_is_empty(self, monkeypatch):
        _fake_smi(monkeypatch, "", returncode = 1)
        assert LlamaCppBackend._cuda_compute_caps() == {}

        def _raise(*_args, **_kwargs):
            raise OSError("no nvidia-smi")

        monkeypatch.setattr(subprocess, "run", _raise)
        assert LlamaCppBackend._cuda_compute_caps() == {}


class TestCudaSmGateError:
    def _caps(self, monkeypatch, caps):
        monkeypatch.setattr(LlamaCppBackend, "_cuda_compute_caps", staticmethod(lambda: caps))

    def test_uncovered_gpu_errors_with_the_fix(self, tmp_path, monkeypatch):
        # The incident shape: a cuda13-older bundle (75-89) baked on a T4, run on an H100.
        self._caps(monkeypatch, {0: 90})
        binary = _binary_with_marker(tmp_path, {"supported_sms": ["75", "80", "86", "89"]})
        error = LlamaCppBackend._cuda_sm_gate_error(binary)
        assert error is not None
        assert "sm_75-sm_89" in error
        assert "GPU 0 is sm_90" in error
        assert "unsloth studio update" in error

    def test_covered_gpu_passes(self, tmp_path, monkeypatch):
        self._caps(monkeypatch, {0: 90})
        binary = _binary_with_marker(tmp_path, {"supported_sms": ["86", "89", "90", "120"]})
        assert LlamaCppBackend._cuda_sm_gate_error(binary) is None

    def test_any_covered_gpu_passes_a_mixed_host(self, tmp_path, monkeypatch):
        self._caps(monkeypatch, {0: 90, 1: 61})
        binary = _binary_with_marker(tmp_path, {"supported_sms": ["86", "89", "90"]})
        assert LlamaCppBackend._cuda_sm_gate_error(binary) is None

    def test_unknown_coverage_fails_open(self, tmp_path, monkeypatch):
        self._caps(monkeypatch, {0: 90})
        binary = _binary_with_marker(tmp_path, {"asset": "x.tar.gz"})
        assert LlamaCppBackend._cuda_sm_gate_error(binary) is None

    def test_unknown_caps_fail_open(self, tmp_path, monkeypatch):
        self._caps(monkeypatch, {})
        binary = _binary_with_marker(tmp_path, {"supported_sms": ["75", "80"]})
        assert LlamaCppBackend._cuda_sm_gate_error(binary) is None
