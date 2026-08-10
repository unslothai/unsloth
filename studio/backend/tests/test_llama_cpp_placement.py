# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Focused integration tests for explicit GGUF GPU placement."""

from __future__ import annotations

import os
import struct
import subprocess
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)


def _stub_module(name: str, **attrs):
    if name in sys.modules:
        return
    try:
        __import__(name)
        return
    except Exception:
        module = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(module, key, value)
        sys.modules[name] = module


_stub_module("loggers", get_logger = lambda name: __import__("logging").getLogger(name))
_stub_module("structlog", get_logger = lambda *a, **k: __import__("logging").getLogger("stub"))
_stub_module(
    "jwt",
    decode = lambda *a, **k: {},
    ExpiredSignatureError = type("ExpiredSignatureError", (Exception,), {}),
    InvalidTokenError = type("InvalidTokenError", (Exception,), {}),
)
if "httpx" not in sys.modules:
    try:
        import httpx  # noqa: F401
    except Exception:
        module = types.ModuleType("httpx")
        for name in (
            "ConnectError",
            "TimeoutException",
            "ReadTimeout",
            "ReadError",
            "RemoteProtocolError",
            "CloseError",
        ):
            setattr(module, name, type(name, (Exception,), {}))
        module.Timeout = type("Timeout", (), {"__init__": lambda self, *a, **k: None})
        module.Client = type(
            "Client",
            (),
            {
                "__init__": lambda self, **kwargs: None,
                "__enter__": lambda self: self,
                "__exit__": lambda self, *args: None,
            },
        )
        sys.modules["httpx"] = module

from core.inference.llama_cpp import GgufLoadIntent, LlamaCppBackend

_REAL_POPEN = subprocess.Popen


def _write_gguf(path: Path, architecture: str = "llama") -> Path:
    def string(value: str) -> bytes:
        data = value.encode()
        return struct.pack("<Q", len(data)) + data

    metadata = string("general.architecture") + struct.pack("<I", 8) + string(architecture)
    path.write_bytes(struct.pack("<IIQQ", 0x46554747, 3, 0, 1) + metadata)
    return path


def _backend(tmp_path: Path, *, vulkan: bool, memory):
    backend = LlamaCppBackend()
    gguf = _write_gguf(tmp_path / "model.gguf")
    backend._get_gpu_memory = lambda _binary = None: list(memory)
    backend._get_gpu_free_memory = lambda _binary = None: [
        (index, free) for index, free, _total in memory
    ]
    backend._read_gguf_metadata = lambda _path: None
    backend._can_estimate_kv = lambda: False
    backend._get_gguf_size_bytes = lambda _path: 1024
    backend._mmproj_vram_bytes = lambda _path: 0
    backend._resolve_launch_mmproj_path = lambda **kwargs: None
    backend._apu_ram_shortfall_message = lambda *args, **kwargs: None
    backend._amd_apu_wants_unified_memory = lambda *args, **kwargs: False
    backend._find_llama_server_binary = lambda include_denied = False: "/fake/llama-server"
    backend._is_vulkan_backend = lambda _binary = None: vulkan
    backend._wait_for_health = lambda timeout: True
    backend._detect_audio_type_strict = lambda: None
    backend._apply_detected_audio = lambda _detected: True
    return backend, gguf


def _launch(backend, gguf, **load_kwargs):
    captured = {}

    def fake_popen(cmd, **kwargs):
        if not cmd or str(cmd[0]) != "/fake/llama-server":
            return _REAL_POPEN(cmd, **kwargs)
        captured["cmd"] = list(cmd)
        captured["env"] = kwargs.get("env") or dict(os.environ)
        return type(
            "Process",
            (),
            {
                "pid": 123,
                "stdout": (),
                "poll": lambda self: None,
                "terminate": lambda self: None,
                "wait": lambda self, timeout = None: 0,
                "kill": lambda self: None,
            },
        )()

    with patch.object(subprocess, "Popen", side_effect = fake_popen):
        assert backend.load_model(
            GgufLoadIntent(
                gguf_path = str(gguf),
                model_identifier = "test",
                **load_kwargs,
            )
        )
    return captured


def test_vulkan_selection_uses_ordinals_and_owns_device_flags(tmp_path):
    backend, gguf = _backend(
        tmp_path,
        vulkan = True,
        memory = [(0, 10_000, 16_000), (1, 8_000, 16_000)],
    )
    backend._select_gpus = lambda *args, **kwargs: ([1], False)

    result = _launch(
        backend,
        gguf,
        gpu_ids = [0, 1],
        extra_args = ["--device", "Vulkan0", "--main-gpu", "0", "--top-k", "5"],
    )

    cmd = result["cmd"]
    assert cmd[cmd.index("--device") + 1] == "Vulkan1"
    assert cmd.count("--device") == 1
    assert "--main-gpu" not in cmd
    assert cmd[cmd.index("--top-k") + 1] == "5"
    assert backend.requested_gpu_ids == [0, 1]
    assert backend.gpu_ids == [1]


@pytest.mark.parametrize(
    "gpu_ids,extra_args,expected_draft,user_device_survives",
    [
        (None, None, "Vulkan1", False),
        (None, ["--device", "Vulkan1", "-dev=Vulkan0"], "Vulkan0", True),
        ([1], ["--device", "Vulkan1", "-dev=Vulkan0"], "Vulkan1", False),
    ],
)
def test_vulkan_fit_and_mtp_drafter_follow_placement_owner(
    tmp_path, gpu_ids, extra_args, expected_draft, user_device_survives
):
    backend, gguf = _backend(
        tmp_path,
        vulkan = True,
        memory = [(0, 24_000, 0), (1, 8_000, 16_000)],
    )
    planned = []

    def fallback(_model_size, gpus, *args, **kwargs):
        planned.append(list(gpus))
        return None, True

    backend._select_gpus = fallback
    backend.probe_server_capabilities = lambda _binary = None: {
        "mtp_token": "draft-mtp",
        "spec_draft_n_max_flag": "--spec-draft-n-max",
    }
    backend._resolve_launch_mtp_path = lambda **_kwargs: "/fake/mtp.gguf"
    result = _launch(
        backend,
        gguf,
        mtp_draft_path = "/fake/mtp.gguf",
        speculative_type = "mtp",
        gpu_ids = gpu_ids,
        extra_args = extra_args,
    )

    assert planned
    assert all(gpus == [(1, 8_000)] for gpus in planned)
    cmd = result["cmd"]
    assert cmd[cmd.index("--device") + 1] == "Vulkan1"
    assert cmd[cmd.index("--spec-draft-device") + 1] == expected_draft
    assert ("-dev=Vulkan0" in cmd) is user_device_survives


@pytest.mark.parametrize("use_fit", [False, True])
def test_dspark_composed_argv_respects_placement_fit_decision(tmp_path, use_fit):
    backend, gguf = _backend(
        tmp_path,
        vulkan = False,
        memory = [(0, 24_000, 24_000)],
    )
    sidecar = tmp_path / "dspark-model-Q8_0.gguf"
    sidecar.write_bytes(b"draft")
    backend._select_gpus = lambda *args, **kwargs: ((None, True) if use_fit else ([0], False))
    backend.probe_server_capabilities = lambda _binary = None: {
        "supports_dspark": True,
        "spec_draft_n_max_flag": "--spec-draft-n-max",
    }

    result = _launch(
        backend,
        gguf,
        dspark_draft_path = str(sidecar),
        speculative_type = "dspark",
    )

    cmd = result["cmd"]
    assert cmd.count("--fit") == 1
    assert cmd[cmd.index("--fit") + 1] == ("on" if use_fit else "off")
    # DSpark engages under either placement: --fit on only means llama.cpp skips
    # the sidecar's memory reserve, it does not refuse to load it.
    assert cmd[cmd.index("--model-draft") + 1] == str(sidecar)
    assert cmd[cmd.index("--spec-type") + 1] == "draft-dspark"
    assert backend.spec_fallback_reason is None


def test_dspark_keeps_a_user_fit_flag(tmp_path):
    """A caller's --fit is theirs to set: the sidecar loads under either value,
    so Studio has no reason to rewrite it."""
    backend, gguf = _backend(tmp_path, vulkan = False, memory = [(0, 24_000, 24_000)])
    sidecar = tmp_path / "dspark-model-Q8_0.gguf"
    sidecar.write_bytes(b"draft")
    backend._select_gpus = lambda *args, **kwargs: ([0], False)
    backend.probe_server_capabilities = lambda _binary = None: {
        "supports_dspark": True,
        "spec_draft_n_max_flag": "--spec-draft-n-max",
    }

    result = _launch(
        backend,
        gguf,
        dspark_draft_path = str(sidecar),
        speculative_type = "dspark",
        extra_args = ["--fit", "on", "--top-k", "5"],
        gpu_ids = [0],
    )

    cmd = result["cmd"]
    assert cmd[len(cmd) - 1 - cmd[::-1].index("--fit") + 1] == "on"
    assert cmd[cmd.index("--top-k") + 1] == "5"
    assert cmd[cmd.index("--spec-type") + 1] == "draft-dspark"


def test_pass_through_dspark_loads_under_an_auto_fit_placement(tmp_path):
    """Manual + Auto layers emits --fit on and a user-owned --spec-type returns
    from _build_speculative_flags early. Nothing rewrites the placement: llama.cpp
    only skips the sidecar's memory reserve under fitting, it still loads it."""
    backend, gguf = _backend(tmp_path, vulkan = False, memory = [(0, 24_000, 24_000)])
    sidecar = tmp_path / "dspark-model-Q8_0.gguf"
    sidecar.write_bytes(b"draft")

    result = _launch(
        backend,
        gguf,
        gpu_memory_mode = "manual",
        gpu_layers = -1,
        extra_args = ["--spec-type", "draft-dspark", "--model-draft", str(sidecar)],
    )

    cmd = result["cmd"]
    assert cmd.count("--fit") == 1
    assert cmd[cmd.index("--fit") + 1] == "on"
    assert cmd[cmd.index("--spec-type") + 1] == "draft-dspark"


def test_cuda_selection_uses_visibility_and_removes_environment_placement(tmp_path, monkeypatch):
    monkeypatch.setenv("LLAMA_ARG_DEVICE", "CUDA0")
    monkeypatch.setenv("LLAMA_ARG_MAIN_GPU", "0")
    backend, gguf = _backend(
        tmp_path,
        vulkan = False,
        memory = [(0, 10_000, 16_000), (1, 8_000, 16_000)],
    )
    backend._select_gpus = lambda *args, **kwargs: ([1], False)

    result = _launch(backend, gguf, gpu_ids = [1])

    assert result["env"]["CUDA_VISIBLE_DEVICES"] == "1"
    assert "LLAMA_ARG_DEVICE" not in result["env"]
    assert "LLAMA_ARG_MAIN_GPU" not in result["env"]


def test_backend_detection_accepts_versioned_vulkan_soname(tmp_path):
    binary = tmp_path / "llama-server"
    binary.write_bytes(b"x")
    lib_dir = tmp_path / "lib"
    lib_dir.mkdir()
    prefix = "" if sys.platform == "win32" else "lib"
    extension = "dll" if sys.platform == "win32" else "so"
    (lib_dir / f"{prefix}ggml-vulkan.{extension}.0").write_bytes(b"x")

    with patch("core.inference.llama_cpp._llama_lib_dir", return_value = lib_dir):
        assert LlamaCppBackend._is_vulkan_backend(str(binary)) is True
        assert LlamaCppBackend._backend_lacks_gpu_lib(str(binary)) is False


def test_cpu_only_detection_requires_a_proven_split_library_layout(tmp_path):
    binary = tmp_path / "llama-server"
    binary.write_bytes(b"x")
    lib_dir = tmp_path / "lib"
    lib_dir.mkdir()
    prefix = "" if sys.platform == "win32" else "lib"
    extension = "dll" if sys.platform == "win32" else "so"
    (lib_dir / f"{prefix}ggml-cpu.{extension}").write_bytes(b"x")

    with patch("core.inference.llama_cpp._llama_lib_dir", return_value = lib_dir):
        assert LlamaCppBackend._backend_lacks_gpu_lib(str(binary)) is True

    (lib_dir / f"{prefix}ggml-vulkan.{extension}").write_bytes(b"x")
    with patch("core.inference.llama_cpp._llama_lib_dir", return_value = lib_dir):
        assert LlamaCppBackend._backend_lacks_gpu_lib(str(binary)) is False


def test_diffusion_does_not_reinterpret_vulkan_ordinals(tmp_path):
    gguf = _write_gguf(tmp_path / "diffusion.gguf", "diffusion-gemma")
    backend = LlamaCppBackend()
    backend._find_llama_server_binary = lambda include_denied = False: "/fake/llama-server"
    backend._is_vulkan_backend = lambda _binary = None: True
    backend._get_gpu_memory = lambda _binary = None: [(1, 8_000, 8_000)]
    backend._download_gguf = lambda **kwargs: str(gguf)
    backend._read_gguf_metadata = lambda _path: setattr(backend, "_is_diffusion", True)
    backend._start_diffusion_server = lambda **kwargs: pytest.fail(
        "Vulkan ordinal reached the CUDA diffusion runner"
    )

    with pytest.raises(ValueError, match = "no defined mapping"):
        backend.load_model(
            GgufLoadIntent(
                hf_repo = "renamed/model",
                hf_variant = "Q4_K_M",
                model_identifier = "renamed/model",
                speculative_type = "off",
                gpu_ids = [1],
            )
        )
