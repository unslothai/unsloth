# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The CUDA SM gate: a prebuilt installed on one GPU (e.g. a cloud image baked
on a T4) must fail fast when run on a GPU its bundle has no kernels for,
instead of llama-server aborting on every launch attempt."""

import json
import os
import struct
import subprocess
import types
from pathlib import Path
from unittest.mock import patch

import pytest

from core.inference.llama_cpp import GgufLoadIntent, LlamaCppBackend


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


def _fake_smi(
    monkeypatch,
    stdout,
    returncode = 0,
):
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
        monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        _fake_smi(monkeypatch, "0, 7.5\n1, 9.0\n")
        assert LlamaCppBackend._cuda_compute_caps() == {1: 90}

    def test_numeric_mask_with_non_physical_order_fails_open(self, monkeypatch):
        # CUDA ordinal 0 can name physical GPU 1 under FASTEST_FIRST, so filtering
        # nvidia-smi's physical row 0 would risk rejecting a compatible launch.
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
        monkeypatch.setenv("CUDA_DEVICE_ORDER", "FASTEST_FIRST")
        _fake_smi(monkeypatch, "0, 7.5\n1, 9.0\n")
        assert LlamaCppBackend._cuda_compute_caps() == {}

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

    def _managed(self, monkeypatch, managed):
        monkeypatch.setattr(
            LlamaCppBackend, "_is_unsloth_managed_binary", staticmethod(lambda _binary: managed)
        )

    def test_uncovered_gpu_errors_with_the_fix(self, tmp_path, monkeypatch):
        # The incident shape: a cuda13-older bundle (75-89) baked on a T4, run on an H100.
        self._caps(monkeypatch, {0: 90})
        self._managed(monkeypatch, True)
        binary = _binary_with_marker(tmp_path, {"supported_sms": ["75", "80", "86", "89"]})
        error = LlamaCppBackend._cuda_sm_gate_error(binary)
        assert error is not None
        assert "sm_75-sm_89" in error
        assert "GPU 0 is sm_90" in error
        assert "unsloth studio update" in error

    def test_a_custom_binary_is_told_to_rebuild_not_to_update(self, tmp_path, monkeypatch):
        # An extracted prebuilt tree reached through LLAMA_SERVER_PATH or PATH still
        # carries the marker, but the updater cannot replace it, so the update
        # instruction would send its owner back through this same gate.
        self._caps(monkeypatch, {0: 90})
        self._managed(monkeypatch, False)
        binary = _binary_with_marker(tmp_path, {"supported_sms": ["75", "80", "86", "89"]})
        error = LlamaCppBackend._cuda_sm_gate_error(binary)
        assert error is not None
        assert "GPU 0 is sm_90" in error
        assert "reinstall or rebuild that custom llama.cpp" in error
        assert "unsloth studio update" not in error

    def test_covered_gpu_passes(self, tmp_path, monkeypatch):
        self._caps(monkeypatch, {0: 90})
        binary = _binary_with_marker(tmp_path, {"supported_sms": ["86", "89", "90", "120"]})
        assert LlamaCppBackend._cuda_sm_gate_error(binary) is None

    def test_fastest_first_numeric_mask_does_not_reject_a_supported_gpu(
        self, tmp_path, monkeypatch
    ):
        # CUDA ordinal 0 selects the faster physical GPU 1 (sm_90), not the
        # nvidia-smi row 0 (sm_75), so the physical mapping is unknown here.
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
        monkeypatch.setenv("CUDA_DEVICE_ORDER", "FASTEST_FIRST")
        _fake_smi(monkeypatch, "0, 7.5\n1, 9.0\n")
        binary = _binary_with_marker(tmp_path, {"supported_sms": ["90"]})
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


def _gated_backend(
    tmp_path,
    monkeypatch,
    *,
    supported_sms = ("75", "80", "86", "89"),
):
    """A load on the incident host: the installed bundle covers sm_75-sm_89 and the
    only GPU is an sm_90 H100, so the gate wants to refuse. Everything below the
    placement decision is faked -- Popen never runs and health answers True."""
    install = tmp_path / "llama.cpp"
    (install / "build" / "bin").mkdir(parents = True)
    binary = _binary_with_marker(install, {"supported_sms": list(supported_sms)})
    Path(binary).write_text("", encoding = "utf-8")
    os.chmod(binary, 0o755)

    gguf = tmp_path / "model.gguf"

    def _string(value):
        data = value.encode()
        return struct.pack("<Q", len(data)) + data

    gguf.write_bytes(
        struct.pack("<IIQQ", 0x46554747, 3, 0, 1)
        + _string("general.architecture")
        + struct.pack("<I", 8)
        + _string("llama")
    )

    monkeypatch.setattr(LlamaCppBackend, "_cuda_compute_caps", staticmethod(lambda: {0: 90}))
    monkeypatch.setattr(
        LlamaCppBackend, "_is_vulkan_backend", staticmethod(lambda *_a, **_kw: False)
    )
    # Pinned so the refusal wording does not depend on where the runner puts tmp_path.
    monkeypatch.setattr(
        LlamaCppBackend, "_is_unsloth_managed_binary", staticmethod(lambda _binary: True)
    )
    backend = LlamaCppBackend()
    backend._get_gpu_memory = lambda _binary = None, **_kw: []
    backend._get_gpu_free_memory = lambda _binary = None, **_kw: []
    backend._read_gguf_metadata = lambda _path: None
    backend._can_estimate_kv = lambda: False
    backend._get_gguf_size_bytes = lambda _path: 1024
    backend._mmproj_vram_bytes = lambda _path: 0
    backend._resolve_launch_mmproj_path = lambda **_kw: None
    backend._apu_ram_shortfall_message = lambda *_a, **_kw: None
    backend._amd_apu_wants_unified_memory = lambda *_a, **_kw: False
    backend._find_llama_server_binary = lambda include_denied = False: binary
    backend._fit_off_retry_eligible = lambda *_a, **_kw: False
    backend.probe_server_capabilities = lambda _binary: {"found": True}
    backend._record_server_pid = lambda _pid: None
    backend._clear_server_pid = lambda: None
    backend._prepare_cpu_fallback_launch = lambda *_a, **_kw: None
    backend._detect_audio_type_strict = lambda: None
    backend._apply_detected_audio = lambda _detected: True
    backend._wait_for_health = lambda timeout: True
    backend._llama_server_env_for_binary = lambda _binary: {"PATH": os.environ.get("PATH", "")}
    return backend, gguf


def _drive_load(backend, gguf, **intent_kwargs):
    """Return (launches, error): a refusal has no launch to point at, so the
    exception is handed back rather than raised through the harness."""
    launches = []

    class _Process:
        pid = 123
        stdout = ()
        returncode = None

        def poll(self):
            return None

        def terminate(self):
            return None

        def wait(self, timeout = None):
            return 0

        def kill(self):
            return None

    def _popen(cmd, **kwargs):
        launches.append((list(cmd), dict(kwargs.get("env") or {})))
        return _Process()

    error = None
    with patch.object(subprocess, "Popen", side_effect = _popen):
        try:
            backend.load_model(
                GgufLoadIntent(
                    gguf_path = str(gguf),
                    model_identifier = "owner/model",
                    **intent_kwargs,
                )
            )
        except Exception as exc:
            error = exc
    return launches, error


class TestTheGateSparesADeliberateCpuOnlyLoad:
    """A manual zero-offload load launches with CUDA_VISIBLE_DEVICES=-1, so the
    child never initialises CUDA and the missing kernels cannot reach it. Gating
    it turned a load that works on the mismatched host into a hard refusal."""

    def test_manual_zero_offload_still_launches_on_cpu(self, tmp_path, monkeypatch):
        backend, gguf = _gated_backend(tmp_path, monkeypatch)
        launches, error = _drive_load(backend, gguf, gpu_memory_mode = "manual", gpu_layers = 0)
        assert error is None, f"the SM gate refused a CPU-only load: {error}"
        assert len(launches) == 1
        _cmd, env = launches[0]
        assert env.get("CUDA_VISIBLE_DEVICES") == "-1"

    def test_a_gpu_offload_request_is_still_refused(self, tmp_path, monkeypatch):
        backend, gguf = _gated_backend(tmp_path, monkeypatch)
        launches, error = _drive_load(backend, gguf, gpu_memory_mode = "auto")
        assert isinstance(error, RuntimeError)
        assert "unsloth studio update" in str(error)
        assert launches == []

    @pytest.mark.parametrize(
        "companion",
        [
            {"extra_args": ["--device", "CUDA0"]},
            {"extra_args": ["--model-draft", "/tmp/draft.gguf"]},
        ],
        ids = ["device_pin", "gpu_drafter"],
    )
    def test_a_gpu_companion_keeps_the_gate(self, tmp_path, monkeypatch, companion):
        # These keep the GPUs visible to the child, so the kernels are needed after
        # all and the exemption must not swallow them.
        backend, gguf = _gated_backend(tmp_path, monkeypatch)
        launches, error = _drive_load(
            backend, gguf, gpu_memory_mode = "manual", gpu_layers = 0, **companion
        )
        assert isinstance(error, RuntimeError)
        assert launches == []

    def test_the_uis_default_speculative_auto_still_launches_on_cpu(self, tmp_path, monkeypatch):
        # The chat store seeds speculativeType from readPersistedSpeculativeType(),
        # which defaults to "auto", so nearly every /load carries it. Auto resolves
        # to a drafterless ngram mode here, no --model-draft is emitted and the
        # launch masks the GPUs away -- deciding the gate from the requested mode
        # refused the ordinary CPU-only load this exemption exists for.
        backend, gguf = _gated_backend(tmp_path, monkeypatch)
        launches, error = _drive_load(
            backend,
            gguf,
            gpu_memory_mode = "manual",
            gpu_layers = 0,
            speculative_type = "auto",
        )
        assert error is None, f"the SM gate refused the default CPU-only load: {error}"
        assert len(launches) == 1
        _cmd, env = launches[0]
        assert env.get("CUDA_VISIBLE_DEVICES") == "-1"

    def test_a_cpu_pinned_drafter_still_launches_on_cpu(self, tmp_path, monkeypatch):
        # --spec-draft-ngl 0 keeps the drafter off the GPU, so the launch masks
        # them away like any other zero-offload load.
        backend, gguf = _gated_backend(tmp_path, monkeypatch)
        launches, error = _drive_load(
            backend,
            gguf,
            gpu_memory_mode = "manual",
            gpu_layers = 0,
            speculative_type = "auto",
            extra_args = ["--spec-draft-ngl", "0"],
        )
        assert error is None, f"the SM gate refused a CPU-pinned drafter: {error}"
        assert len(launches) == 1
        _cmd, env = launches[0]
        assert env.get("CUDA_VISIBLE_DEVICES") == "-1"

    def test_a_cpu_pinned_projector_still_launches_on_cpu(self, tmp_path, monkeypatch):
        # --no-mmproj-offload clears mmproj_use_gpu, so clip.cpp holds no VRAM and
        # the launch hides the GPUs; the gate must follow that mask, not the fact
        # that a projector was resolved.
        backend, gguf = _gated_backend(tmp_path, monkeypatch)
        mmproj = tmp_path / "mmproj.gguf"
        mmproj.write_bytes(b"")
        backend._resolve_launch_mmproj_path = lambda **_kw: str(mmproj)
        launches, error = _drive_load(
            backend,
            gguf,
            gpu_memory_mode = "manual",
            gpu_layers = 0,
            is_vision = True,
            extra_args = ["--no-mmproj-offload"],
        )
        assert error is None, f"the SM gate refused a CPU-pinned projector: {error}"
        assert len(launches) == 1
        cmd, env = launches[0]
        assert "--no-mmproj-offload" in cmd
        assert env.get("CUDA_VISIBLE_DEVICES") == "-1"

    def test_a_gpu_projector_keeps_the_gate(self, tmp_path, monkeypatch):
        # Without the CPU pin the projector is offloaded, the GPUs stay visible and
        # the child needs the kernels the bundle has no code for.
        backend, gguf = _gated_backend(tmp_path, monkeypatch)
        mmproj = tmp_path / "mmproj.gguf"
        mmproj.write_bytes(b"")
        backend._resolve_launch_mmproj_path = lambda **_kw: str(mmproj)
        launches, error = _drive_load(
            backend, gguf, gpu_memory_mode = "manual", gpu_layers = 0, is_vision = True
        )
        assert isinstance(error, RuntimeError)
        assert launches == []

    def test_an_auto_placement_device_none_keeps_the_gate(self, tmp_path, monkeypatch):
        # Auto placement writes no CPU mask, so the child still enumerates the
        # uncovered card. --device none only empties model->devices AFTER
        # ggml_cuda_init has run, and it is one typo ("cpu", which llama.cpp
        # rejects outright) away from a launch that dies anyway -- the exemption
        # tracks the mask, not the argv.
        backend, gguf = _gated_backend(tmp_path, monkeypatch)
        launches, error = _drive_load(
            backend, gguf, gpu_memory_mode = "auto", extra_args = ["--device", "none"]
        )
        assert isinstance(error, RuntimeError)
        assert launches == []
