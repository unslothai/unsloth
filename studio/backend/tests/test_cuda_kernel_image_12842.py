# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""#12842: a CUDA llama.cpp build whose kernels the driver cannot load.

A CUDA >= 12.8 build compresses its fatbin with -compress-mode, which drivers older
than CUDA 12.4 cannot decode, so every kernel load returns "device kernel image is
invalid". The crash used to walk the whole retry ladder (--fit on, one slot,
flash-attn off, no drafter), five doomed restarts, and end on "Check that the GGUF
file is valid and you have enough memory". Pinned here: the classification, the one
spawn, and that the ROCm (#7624) spelling is untouched.

Mock-based: no GPU, OS-neutral.
"""

from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from core.inference.llama_cpp import GgufLoadIntent, LlamaCppBackend  # noqa: E402

_REAL_POPEN = subprocess.Popen

# The tail from the issue (Windows A40, b11443-mix-d65395f cuda12 bundle).
_ISSUE_TAIL = """\
0.31.322.841 I sched_reserve:      CUDA0 compute buffer size = 13951.75 MiB
0.31.323.945 I cmn  common_init_: warming up the model with an empty run - please wait ... (--no-warmup to disable)
0.31.368.986 E CUDA error: device kernel image is invalid
0.31.369.005 E D:\\a\\llama.cpp\\llama.cpp\\src\\ggml\\src\\ggml-cuda\\ggml-cuda.cu:134: CUDA error
  current device: 0, in function ggml_cuda_kernel_can_use_pdl at D:\\a\\llama.cpp\\llama.cpp\\src\\ggml\\src\\ggml-cuda\\common.cuh:1234
  cudaFuncGetAttributes(&attr, kernel)"""

_NO_IMAGE_TAIL = """\
ggml_cuda_compute_forward: MUL_MAT failed
CUDA error: no kernel image is available for execution on the device
  current device: 0, in function ggml_cuda_compute_forward at ggml-cuda.cu:2367"""

_ROCM_TAIL = """\
ROCm error: device kernel image is invalid
  current device: 0, in function ggml_cuda_compute_forward at ggml-cuda.cu:2367"""

# 0xC0000409, what the issue's Windows child exited with; any non-zero code is a crash.
_WIN_FAIL_FAST = 3221226505


class TestClassification:
    def test_issue_tail_is_a_cuda_kernel_image_error(self):
        assert LlamaCppBackend._cuda_kernel_image_error(_ISSUE_TAIL) == (
            "device kernel image is invalid"
        )

    def test_no_kernel_image_spelling(self):
        assert LlamaCppBackend._cuda_kernel_image_error(_NO_IMAGE_TAIL) == (
            "no kernel image is available for execution"
        )

    @pytest.mark.parametrize(
        "output",
        [
            _ROCM_TAIL,
            "",
            None,
            "CUDA error: out of memory",
            "device kernel image is invalid",  # unprefixed: not a ggml CUDA error line
        ],
    )
    def test_other_output_does_not_match(self, output):
        assert LlamaCppBackend._cuda_kernel_image_error(output) is None

    def test_rocm_tail_still_matches_the_7624_marker(self):
        assert LlamaCppBackend._kernel_image_invalid(_ROCM_TAIL)

    def test_message_names_the_driver_not_memory(self):
        msg = LlamaCppBackend._classify_llama_start_failure(
            _ISSUE_TAIL, "/m/model.gguf", "model", _WIN_FAIL_FAST, None
        )
        assert "device kernel image is invalid" in msg
        assert "NVIDIA driver" in msg
        assert "R550" in msg
        assert "not out of memory" in msg
        assert "enough memory" not in msg

    def test_no_kernel_image_message_does_not_blame_the_driver(self):
        msg = LlamaCppBackend._classify_llama_start_failure(
            _NO_IMAGE_TAIL, "/m/model.gguf", "model", 134, None
        )
        assert "no kernels for this GPU" in msg
        assert "R550" not in msg

    def test_rocm_tail_keeps_its_old_classification(self):
        msg = LlamaCppBackend._classify_llama_start_failure(
            _ROCM_TAIL, "/m/model.gguf", "model", 134, None
        )
        assert "NVIDIA driver" not in msg

    @staticmethod
    def _marker(tmp_path, monkeypatch, marker):
        # The shape write_prebuilt_metadata produces: runtime_line, bundle_profile, host_profile.
        if marker is not None:
            (tmp_path / "UNSLOTH_PREBUILT_INFO.json").write_text(
                marker if isinstance(marker, str) else json.dumps(marker), encoding = "utf-8"
            )
        import utils.llama_cpp_update as update

        monkeypatch.setattr(update, "_llama_install_root", lambda _binary: tmp_path)
        return str(tmp_path / "llama-server")

    def test_driver_version_from_the_install_marker(self, tmp_path, monkeypatch):
        binary = self._marker(
            tmp_path,
            monkeypatch,
            {
                "runtime_line": "cuda12",
                "bundle_profile": "cuda12-older",
                "host_profile": {"driver_cuda_version": [12, 2]},
            },
        )
        assert LlamaCppBackend._cuda_install_driver_version(binary) == (12, 2)
        msg = LlamaCppBackend._cuda_kernel_image_message(
            "device kernel image is invalid", binary, "/logs/llama.log"
        )
        assert "supports CUDA 12.2" in msg and "R550" in msg
        assert msg.endswith("Full log: /logs/llama.log")

    def test_new_driver_blames_the_build_not_the_driver(self, tmp_path, monkeypatch):
        binary = self._marker(
            tmp_path, monkeypatch, {"host_profile": {"driver_cuda_version": [13, 0]}}
        )
        msg = LlamaCppBackend._cuda_kernel_image_message("device kernel image is invalid", binary)
        assert "supports CUDA 13.0" in msg
        assert "damaged or mismatched" in msg
        assert "R550" not in msg

    @pytest.mark.parametrize(
        "marker",
        [None, "not json", {"runtime_line": "cuda12"}, {"host_profile": {}}],
    )
    def test_driver_version_unknown(self, tmp_path, monkeypatch, marker):
        binary = self._marker(tmp_path, monkeypatch, marker)
        assert LlamaCppBackend._cuda_install_driver_version(binary) is None
        msg = LlamaCppBackend._cuda_kernel_image_message("device kernel image is invalid", binary)
        assert "most likely too old" in msg and "supports CUDA" not in msg


# ── The load path: one spawn, terminal error ─────────────────────────


def _write_gguf(path: Path, architecture: str = "llama") -> Path:
    def string(value: str) -> bytes:
        data = value.encode()
        return struct.pack("<Q", len(data)) + data

    metadata = string("general.architecture") + struct.pack("<I", 8) + string(architecture)
    path.write_bytes(struct.pack("<IIQQ", 0x46554747, 3, 0, 1) + metadata)
    return path


def _backend(tmp_path: Path, gpus):
    backend = LlamaCppBackend()
    rows = [(index, 40000, 46000) for index in gpus]
    backend._get_gpu_memory = lambda _binary = None, **_kw: list(rows)
    backend._get_gpu_free_memory = lambda _binary = None, **_kw: [(i, f) for i, f, _t in rows]
    backend._read_gguf_metadata = lambda _path: None
    backend._can_estimate_kv = lambda: False
    backend._get_gguf_size_bytes = lambda _path: 1024
    backend._mmproj_vram_bytes = lambda _path: 0
    backend._resolve_launch_mmproj_path = lambda **kwargs: None
    backend._apu_ram_shortfall_message = lambda *args, **kwargs: None
    backend._amd_apu_wants_unified_memory = lambda *args, **kwargs: False
    backend._find_llama_server_binary = lambda include_denied = False: "/fake/llama-server"
    backend._is_vulkan_backend = lambda _binary = None: False
    backend._detect_audio_type_strict = lambda: None
    backend._apply_detected_audio = lambda _detected: True

    def _crashed_health(timeout, **_kw):
        # The drain thread owns _stdout_lines; let it finish before the caller reads them.
        thread = getattr(backend, "_stdout_thread", None)
        if thread is not None:
            thread.join(5)
        return False

    backend._wait_for_health = _crashed_health
    return backend, _write_gguf(tmp_path / "model.gguf")


def _crash_every_spawn(
    backend,
    gguf,
    tail,
    rc,
    first_tail = None,
    n_parallel = 1,
):
    spawns: list[list[str]] = []

    def fake_popen(cmd, **kwargs):
        if not cmd or str(cmd[0]) != "/fake/llama-server":
            return _REAL_POPEN(cmd, **kwargs)
        spawns.append(list(cmd))
        output = first_tail if first_tail is not None and len(spawns) == 1 else tail
        return type(
            "Process",
            (),
            {
                "pid": 123,
                "stdout": [line + "\n" for line in output.splitlines()],
                "returncode": rc,
                "poll": lambda self: rc,
                "terminate": lambda self: None,
                "wait": lambda self, timeout = None: rc,
                "kill": lambda self: None,
            },
        )()

    with patch.object(subprocess, "Popen", side_effect = fake_popen):
        try:
            result = backend.load_model(
                GgufLoadIntent(gguf_path = str(gguf), model_identifier = "test", n_parallel = n_parallel)
            )
            error = None
        except RuntimeError as exc:
            result, error = None, str(exc)
    return spawns, result, error


class TestLoadStopsAtTheFirstCrash:
    @pytest.mark.parametrize("rc", [_WIN_FAIL_FAST, -6, 134])
    def test_single_cuda_gpu_spawns_once_and_names_the_driver(self, tmp_path, rc):
        backend, gguf = _backend(tmp_path, [0])
        spawns, result, error = _crash_every_spawn(backend, gguf, _ISSUE_TAIL, rc)
        assert error is not None, (result, len(spawns))
        assert "NVIDIA driver" in error
        assert "enough memory" not in error
        assert len(spawns) == 1, [" ".join(s[-6:]) for s in spawns]

    def test_no_kernel_image_also_stops(self, tmp_path):
        backend, gguf = _backend(tmp_path, [0])
        spawns, _result, error = _crash_every_spawn(backend, gguf, _NO_IMAGE_TAIL, 134)
        assert error is not None and "no kernels for this GPU" in error
        assert len(spawns) == 1

    def test_rocm_crash_is_not_short_circuited(self, tmp_path):
        # The ROCm spelling keeps whatever ladder it had; only the CUDA prefix stops it.
        backend, gguf = _backend(tmp_path, [0])
        spawns, _result, error = _crash_every_spawn(backend, gguf, _ROCM_TAIL, 134)
        assert error is None or "NVIDIA driver" not in error

    @pytest.mark.parametrize("rc", [_WIN_FAIL_FAST, -6])
    def test_a_recovery_rung_that_reaches_the_kernels_stops_too(self, tmp_path, rc):
        # The first launch fails for another reason (unified KV refused with four
        # slots); the one-slot retry is the first to reach warm-up and the CUDA error.
        # No flash-attn or drafter rung may follow it.
        backend, gguf = _backend(tmp_path, [0])
        probe = backend.probe_server_capabilities
        backend.probe_server_capabilities = lambda path, *a, **kw: {
            **probe(path, *a, **kw),
            "supports_kv_unified": True,
        }
        refused = "a unified KV cache is only supported with a single sequence"
        spawns, result, error = _crash_every_spawn(
            backend, gguf, _ISSUE_TAIL, rc, first_tail = refused, n_parallel = 4
        )
        assert error is not None and "NVIDIA driver" in error, (result, len(spawns))
        assert "4" in spawns[0][spawns[0].index("--parallel") + 1]
        assert len(spawns) == 2, [" ".join(s[-6:]) for s in spawns]
