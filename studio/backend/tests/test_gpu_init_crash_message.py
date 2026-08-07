# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for GPU-init crash recovery and backend-specific advice."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import struct
import subprocess
import sys
import types as _types
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)


# Allow importing the module in a lightweight environment without fastapi.
class _LoggerStub:
    def bind(self, *args, **kwargs):
        return self

    def __getattr__(self, _name):
        return lambda *args, **kwargs: None


_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: _LoggerStub()
sys.modules.setdefault("loggers", _loggers_stub)
_structlog_stub = _types.ModuleType("structlog")
_structlog_stub.get_logger = lambda *a, **k: _LoggerStub()
sys.modules.setdefault("structlog", _structlog_stub)
if not hasattr(sys.modules["structlog"], "get_logger"):
    sys.modules["structlog"].get_logger = _structlog_stub.get_logger

from core.inference import llama_cpp  # noqa: E402
from core.inference.llama_cpp import GgufLoadIntent, LlamaCppBackend  # noqa: E402
from models.inference import InferenceStatusResponse, LoadRequest, LoadResponse  # noqa: E402


_RAW_MAIN_PLACEMENT_ARGS = (
    ("--tensor-split", "1,1"),
    ("-ts", "1,1"),
    ("--fit", "off"),
    ("-fit", "off"),
    ("--cpu-moe",),
    ("-cmoe",),
    ("--n-cpu-moe", "4"),
    ("-ncmoe", "4"),
)


def _managed_runtime(monkeypatch, tmp_path):
    install = tmp_path / "install" / "build" / "bin"
    install.mkdir(parents = True)
    binary = install / "llama-server"
    binary.write_bytes(b"binary")
    binary.chmod(0o755)
    (install / "libggml-base.so").write_bytes(b"base")
    (install / "libggml-cpu-haswell.so").write_bytes(b"cpu")
    (install / "libggml-vulkan.so").write_bytes(b"vulkan")
    monkeypatch.setattr(
        LlamaCppBackend,
        "_is_unsloth_managed_binary",
        staticmethod(lambda _binary: True),
    )
    monkeypatch.setattr(
        llama_cpp,
        "_swa_cache_path",
        lambda: tmp_path / "studio" / "swa_cache.json",
    )
    return binary


def _run_cpu_fallback_load(
    monkeypatch,
    tmp_path,
    *,
    returncodes,
    first_output = "",
    mmproj_from_argv = False,
    mmproj_env = None,
    cpu_fallback_available = True,
):
    def _gguf_string(value: str) -> bytes:
        encoded = value.encode()
        return struct.pack("<Q", len(encoded)) + encoded

    metadata = _gguf_string("general.architecture") + struct.pack("<I", 8) + _gguf_string("llama")
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(struct.pack("<IIQQ", 0x46554747, 3, 0, 1) + metadata)
    mmproj = tmp_path / "mmproj.gguf"
    mmproj.write_bytes(b"projector")

    backend = LlamaCppBackend()
    backend._get_gpu_memory = lambda _binary = None: []
    backend._get_gpu_free_memory = lambda _binary = None: []
    backend._read_gguf_metadata = lambda _path: None
    backend._can_estimate_kv = lambda: False
    backend._get_gguf_size_bytes = lambda _path: 1024
    backend._mmproj_vram_bytes = lambda _path: 0
    backend._resolve_launch_mmproj_path = lambda **_kwargs: (
        str(mmproj) if mmproj_from_argv else None
    )
    backend._apu_ram_shortfall_message = lambda *_args, **_kwargs: None
    backend._amd_apu_wants_unified_memory = lambda *_args, **_kwargs: False
    backend._find_llama_server_binary = lambda include_denied = False: "/fake/llama-server"
    monkeypatch.setattr(
        LlamaCppBackend,
        "_is_vulkan_backend",
        staticmethod(lambda _binary = None: True),
    )
    monkeypatch.setattr(
        LlamaCppBackend,
        "_vulkan_prebuilt_was_auto_selected",
        staticmethod(lambda _binary: True),
    )
    backend._fit_off_retry_eligible = lambda *_args, **_kwargs: False
    backend.probe_server_capabilities = lambda _binary: {
        "found": True,
        "supports_no_mmproj_offload": True,
    }
    backend._record_server_pid = lambda _pid: None
    backend._clear_server_pid = lambda: None
    backend._llama_server_env_for_binary = lambda _binary: {
        "PATH": os.environ.get("PATH", ""),
        **(mmproj_env or {}),
    }
    backend._is_projector_incompatibility = lambda output: "projector-incompatible" in output

    launches = []
    fallback_sources = []

    class _Process:
        pid = 123
        stdout = ()

        def __init__(self, returncode):
            self.returncode = returncode

        def poll(self):
            return self.returncode

        def terminate(self):
            return None

        def wait(self, timeout = None):
            return self.returncode

        def kill(self):
            return None

    def _popen(cmd, **kwargs):
        index = len(launches)
        launches.append((list(cmd), dict(kwargs["env"])))
        return _Process(returncodes[index])

    def _wait_for_health(timeout):
        if len(launches) == 1 and first_output:
            backend._stdout_lines = [first_output]
        return returncodes[len(launches) - 1] is None

    def _prepare_cpu_fallback(_binary, failed_cmd, _env, _server_caps):
        fallback_sources.append(list(failed_cmd))
        if not cpu_fallback_available:
            return None
        return ["/staged/llama-server", "--device", "none"], None

    backend._wait_for_health = _wait_for_health
    backend._prepare_cpu_fallback_launch = _prepare_cpu_fallback
    monkeypatch.setattr(subprocess, "Popen", _popen)

    loaded = backend.load_model(
        GgufLoadIntent(
            gguf_path = str(gguf),
            mmproj_path = str(mmproj) if mmproj_from_argv else None,
            model_identifier = "owner/model",
            is_vision = mmproj_from_argv,
        )
    )
    return backend, loaded, launches, fallback_sources


class TestGpuInitCrashMessage:
    def _message(self, monkeypatch, backend):
        monkeypatch.setattr(
            LlamaCppBackend,
            "_installed_ggml_backends",
            staticmethod(lambda *a: frozenset({backend}) if backend else frozenset()),
        )
        return LlamaCppBackend._gpu_init_crash_message("/opt/llama/llama-server")

    def test_vulkan_host_gets_no_rocm_advice(self, monkeypatch):
        message = self._message(monkeypatch, "vulkan")
        assert "ROCR_VISIBLE_DEVICES" not in message
        assert "CUDA_VISIBLE_DEVICES" not in message
        assert "Vulkan" in message
        assert "UNSLOTH_LLAMA_CPP_BACKEND=cpu" in message

    def test_rocm_host_keeps_the_original_advice(self, monkeypatch):
        message = self._message(monkeypatch, "hip")
        assert "ROCR_VISIBLE_DEVICES=0" in message

    def test_cuda_host_gets_cuda_advice(self, monkeypatch):
        message = self._message(monkeypatch, "cuda")
        assert "CUDA_VISIBLE_DEVICES" in message
        assert "ROCR_VISIBLE_DEVICES" not in message

    @pytest.mark.parametrize("backend", [None, "sycl"])
    def test_unknown_backend_gives_neutral_advice(self, monkeypatch, backend):
        # Metal's .dylib is not detected, so macOS also gets neutral advice.
        message = self._message(monkeypatch, backend)
        assert "ROCR_VISIBLE_DEVICES" not in message
        assert "CUDA_VISIBLE_DEVICES" not in message

    def test_every_variant_still_names_the_gpu_as_the_cause(self, monkeypatch):
        for backend in ("vulkan", "hip", "cuda", None):
            message = self._message(monkeypatch, backend)
            assert "GPU driver/runtime initialization crash" in message


class TestAutoVulkanCpuFallbackGate:
    def _managed_marker(self, monkeypatch, tmp_path, **values):
        marker = {
            "asset": "llama-b9999-bin-ubuntu-vulkan-x64.tar.gz",
            "force_cpu": False,
            **values,
        }
        (tmp_path / "UNSLOTH_PREBUILT_INFO.json").write_text(json.dumps(marker))
        import utils.llama_cpp_update as update

        monkeypatch.setattr(update, "_llama_install_root", lambda _binary: tmp_path)

    def test_managed_auto_selected_marker_is_eligible(self, monkeypatch, tmp_path):
        self._managed_marker(monkeypatch, tmp_path)
        monkeypatch.setattr(
            LlamaCppBackend, "_is_vulkan_backend", staticmethod(lambda _binary = None: True)
        )
        intent = GgufLoadIntent(model_identifier = "owner/model")
        assert LlamaCppBackend._auto_vulkan_cpu_fallback_eligible(
            "/managed/llama-server", intent, None, {}
        )

    def test_automatic_windows_amd_marker_is_eligible(self, monkeypatch, tmp_path):
        self._managed_marker(monkeypatch, tmp_path, llama_backend = "auto")
        monkeypatch.setattr(
            LlamaCppBackend, "_is_vulkan_backend", staticmethod(lambda _binary = None: True)
        )
        assert LlamaCppBackend._auto_vulkan_cpu_fallback_eligible(
            "/managed/llama-server",
            GgufLoadIntent(model_identifier = "m"),
            None,
            {},
        )

    @pytest.mark.parametrize(
        "name,value",
        [
            ("LLAMA_ARG_N_GPU_LAYERS", "20"),
            ("LLAMA_ARG_DEVICE", "Vulkan0"),
            ("LLAMA_ARG_MAIN_GPU", "1"),
            ("LLAMA_ARG_SPLIT_MODE", "tensor"),
            ("LLAMA_ARG_TENSOR_SPLIT", "1,1"),
            ("LLAMA_ARG_OVERRIDE_TENSOR", ".*=Vulkan0"),
            ("LLAMA_ARG_CPU_MOE", "true"),
            ("LLAMA_ARG_N_CPU_MOE", "4"),
            ("LLAMA_ARG_FIT", "off"),
            ("LLAMA_ARG_FIT_TARGET", "1024"),
            ("LLAMA_ARG_FIT_CTX", "4096"),
        ],
    )
    def test_inherited_main_placement_never_downgrades(self, monkeypatch, tmp_path, name, value):
        self._managed_marker(monkeypatch, tmp_path)
        monkeypatch.setattr(
            LlamaCppBackend, "_is_vulkan_backend", staticmethod(lambda _binary = None: True)
        )
        assert not LlamaCppBackend._auto_vulkan_cpu_fallback_eligible(
            "/managed/llama-server",
            GgufLoadIntent(model_identifier = "m"),
            None,
            {name: value},
        )

    @pytest.mark.parametrize(
        "intent, extras",
        [
            (GgufLoadIntent(model_identifier = "m", gpu_ids = (0,)), None),
            (GgufLoadIntent(model_identifier = "m", gpu_memory_mode = "manual"), None),
            (GgufLoadIntent(model_identifier = "m", tensor_parallel = True), None),
            (GgufLoadIntent(model_identifier = "m"), ["--device", "Vulkan0"]),
            (GgufLoadIntent(model_identifier = "m"), ["-ngl", "20"]),
            (GgufLoadIntent(model_identifier = "m"), ["-sm", "tensor"]),
        ],
    )
    def test_explicit_gpu_placement_never_downgrades(self, monkeypatch, tmp_path, intent, extras):
        self._managed_marker(monkeypatch, tmp_path)
        monkeypatch.setattr(
            LlamaCppBackend, "_is_vulkan_backend", staticmethod(lambda _binary = None: True)
        )
        assert not LlamaCppBackend._auto_vulkan_cpu_fallback_eligible(
            "/managed/llama-server", intent, extras
        )

    @pytest.mark.parametrize("extras", _RAW_MAIN_PLACEMENT_ARGS)
    def test_raw_main_placement_never_downgrades(self, monkeypatch, tmp_path, extras):
        self._managed_marker(monkeypatch, tmp_path)
        monkeypatch.setattr(
            LlamaCppBackend, "_is_vulkan_backend", staticmethod(lambda _binary = None: True)
        )
        assert not LlamaCppBackend._auto_vulkan_cpu_fallback_eligible(
            "/managed/llama-server",
            GgufLoadIntent(model_identifier = "m"),
            extras,
        )

    @pytest.mark.parametrize(
        "extras",
        [
            ("--spec-draft-ngl", "12"),
            ("-ngld", "12"),
            ("--gpu-layers-draft", "12"),
            ("--n-gpu-layers-draft", "12"),
            ("--mmproj-offload",),
            ("--no-mmproj-offload",),
        ],
    )
    def test_explicit_companion_placement_never_downgrades(self, monkeypatch, tmp_path, extras):
        self._managed_marker(monkeypatch, tmp_path)
        monkeypatch.setattr(
            LlamaCppBackend, "_is_vulkan_backend", staticmethod(lambda _binary = None: True)
        )
        assert not LlamaCppBackend._auto_vulkan_cpu_fallback_eligible(
            "/managed/llama-server",
            GgufLoadIntent(model_identifier = "m"),
            extras,
            {},
        )

    @pytest.mark.parametrize(
        "name,value",
        [
            ("LLAMA_ARG_N_GPU_LAYERS_DRAFT", "12"),
            ("LLAMA_ARG_MMPROJ_OFFLOAD", "1"),
            ("LLAMA_ARG_MMPROJ_OFFLOAD", "0"),
            ("GGML_BACKEND_PATH", "/custom/ggml/backends"),
        ],
    )
    def test_inherited_companion_placement_never_downgrades(
        self, monkeypatch, tmp_path, name, value
    ):
        self._managed_marker(monkeypatch, tmp_path)
        monkeypatch.setattr(
            LlamaCppBackend, "_is_vulkan_backend", staticmethod(lambda _binary = None: True)
        )
        assert not LlamaCppBackend._auto_vulkan_cpu_fallback_eligible(
            "/managed/llama-server",
            GgufLoadIntent(model_identifier = "m"),
            None,
            {name: value},
        )

    def test_persisted_explicit_vulkan_choice_never_downgrades(self, monkeypatch, tmp_path):
        self._managed_marker(monkeypatch, tmp_path, llama_backend = "vulkan")
        monkeypatch.setattr(
            LlamaCppBackend, "_is_vulkan_backend", staticmethod(lambda _binary = None: True)
        )
        assert not LlamaCppBackend._auto_vulkan_cpu_fallback_eligible(
            "/managed/llama-server", GgufLoadIntent(model_identifier = "m"), None
        )

    def test_custom_markerless_binary_never_downgrades(self, monkeypatch):
        import utils.llama_cpp_update as update

        monkeypatch.setattr(update, "_llama_install_root", lambda _binary: None)
        monkeypatch.setattr(
            LlamaCppBackend, "_is_vulkan_backend", staticmethod(lambda _binary = None: True)
        )
        assert not LlamaCppBackend._auto_vulkan_cpu_fallback_eligible(
            "/custom/llama-server", GgufLoadIntent(model_identifier = "m"), None
        )


class TestCpuIsolatedReplay:
    @pytest.mark.parametrize("name", ["LLAMA_ARG_MMPROJ", "LLAMA_ARG_MMPROJ_URL"])
    def test_env_projector_counts_as_a_vision_launch(self, name):
        assert LlamaCppBackend._launch_has_mmproj(
            ["llama-server", "-m", "model.gguf"],
            {name: "projector.gguf"},
        )

    @pytest.mark.parametrize(
        "platform,loader_path",
        [("linux", "LD_LIBRARY_PATH"), ("win32", "PATH")],
    )
    def test_prepared_launch_retargets_the_native_library_path(
        self, monkeypatch, platform, loader_path
    ):
        backend = LlamaCppBackend()
        monkeypatch.setattr(llama_cpp.sys, "platform", platform)
        monkeypatch.setattr(
            backend,
            "_cpu_isolated_replay",
            lambda _cmd, _env, _caps: ["original", "--device", "none"],
        )
        monkeypatch.setattr(backend, "_cpu_isolated_binary", lambda _binary: "/staged/server")
        monkeypatch.setattr(
            backend,
            "_llama_server_env_for_binary",
            lambda _binary: {loader_path: "/staged/libs"},
        )
        env = {"PATH": "/original/bin", "LD_LIBRARY_PATH": "/original/lib", "KEEP": "1"}

        prepared = backend._prepare_cpu_fallback_launch("/original/server", ["original"], env, {})

        assert prepared == (["/staged/server", "--device", "none"], None)
        assert env[loader_path] == "/staged/libs"
        assert env["KEEP"] == "1"

    def test_text_launch_removes_every_gpu_placement(self):
        env = {
            "LLAMA_ARG_N_GPU_LAYERS": "99",
            "LLAMA_ARG_DEVICE": "Vulkan0",
            "LLAMA_ARG_MAIN_GPU": "0",
            "LLAMA_ARG_SPLIT_MODE": "tensor",
            "LLAMA_ARG_TENSOR_SPLIT": "1,1",
            "LLAMA_ARG_OVERRIDE_TENSOR": ".*=Vulkan0",
            "LLAMA_ARG_CPU_MOE": "true",
            "LLAMA_ARG_N_CPU_MOE": "4",
            "LLAMA_ARG_FIT": "off",
            "LLAMA_ARG_FIT_TARGET": "1024",
            "LLAMA_ARG_FIT_CTX": "4096",
            "PATH": "/usr/bin",
        }
        replay = LlamaCppBackend._cpu_isolated_replay(
            [
                "llama-server",
                "-m",
                "model.gguf",
                "-ngl",
                "99",
                "--device",
                "Vulkan0",
                "-sm",
                "tensor",
                "-ot",
                ".*=Vulkan0",
            ],
            env,
            {"found": True},
        )
        assert replay is not None
        assert replay[-6:] == ["--gpu-layers", "0", "--fit", "off", "--device", "none"]
        assert "Vulkan0" not in replay
        assert not set(LlamaCppBackend._CPU_FALLBACK_MAIN_PLACEMENT_ENV_VARS) & env.keys()
        assert env["PATH"] == "/usr/bin"

    def test_vision_and_drafter_get_independent_cpu_pins(self):
        env = {}
        replay = LlamaCppBackend._cpu_isolated_replay(
            [
                "llama-server",
                "-m",
                "model.gguf",
                "--mmproj",
                "mmproj.gguf",
                "--model-draft",
                "draft.gguf",
            ],
            env,
            {
                "found": True,
                "supports_no_mmproj_offload": True,
                "spec_draft_ngl_flag": "--spec-draft-ngl",
                "mtp_probe_inconclusive": False,
            },
        )
        assert replay is not None
        assert "--mmproj" in replay
        assert "--no-mmproj-offload" in replay
        assert replay[replay.index("--spec-draft-ngl") + 1] == "0"
        assert replay[replay.index("--device-draft") + 1] == "none"

    def test_vision_replay_is_refused_when_projector_cannot_be_cpu_pinned(self):
        replay = LlamaCppBackend._cpu_isolated_replay(
            ["llama-server", "--mmproj", "mmproj.gguf"],
            {},
            {"found": True, "supports_no_mmproj_offload": False},
        )
        assert replay is None

    @pytest.mark.parametrize("name", ["LLAMA_ARG_MMPROJ", "LLAMA_ARG_MMPROJ_URL"])
    def test_env_projector_gets_a_cpu_pin(self, name):
        env = {name: "projector.gguf"}
        replay = LlamaCppBackend._cpu_isolated_replay(
            ["llama-server", "-m", "model.gguf"],
            env,
            {"found": True, "supports_no_mmproj_offload": True},
        )

        assert replay is not None
        assert "--no-mmproj-offload" in replay
        assert env[name] == "projector.gguf"

    def test_staged_runtime_excludes_the_gpu_backend(self, monkeypatch, tmp_path):
        binary = _managed_runtime(monkeypatch, tmp_path)
        backend = LlamaCppBackend()
        staged = backend._cpu_isolated_binary(str(binary))
        assert staged is not None
        staged_dir = Path(staged).parent
        assert (staged_dir / "libggml-cpu-haswell.so").is_file()
        assert not (staged_dir / "libggml-vulkan.so").exists()
        backend._cleanup_cpu_fallback_runtime()
        assert not staged_dir.exists()

    def test_terminal_cpu_replay_failure_removes_staged_runtime(self, monkeypatch, tmp_path):
        binary = _managed_runtime(monkeypatch, tmp_path)
        backend = LlamaCppBackend()
        staged = backend._cpu_isolated_binary(str(binary))
        assert staged is not None
        staged_dir = Path(staged).parent
        backend._cpu_fallback_reason = "vulkan_startup_crash"

        backend._cleanup_failed_cpu_fallback()

        assert backend._cpu_fallback_reason is None
        assert backend._cpu_fallback_runtime is None
        assert not staged_dir.exists()

    @pytest.mark.skipif(sys.platform == "win32", reason = "shell wrapper fallback is POSIX")
    def test_wrapper_based_runtime_stages_the_real_executable(self, monkeypatch, tmp_path):
        install = tmp_path / "install"
        bindir = install / "build" / "bin"
        bindir.mkdir(parents = True)
        binary = bindir / "llama-server"
        binary.write_text("#!/bin/sh\nprintf 'healthy\\n'\n")
        binary.chmod(0o755)
        (bindir / "libggml-base.so").write_bytes(b"base")
        (bindir / "libggml-cpu-haswell.so").write_bytes(b"cpu")
        (bindir / "libggml-vulkan.so").write_bytes(b"vulkan")
        wrapper = install / "llama-server"
        wrapper.write_text('#!/bin/sh\nexec "$(dirname "$0")/build/bin/llama-server" "$@"\n')
        wrapper.chmod(0o755)
        monkeypatch.setattr(
            LlamaCppBackend,
            "_is_unsloth_managed_binary",
            staticmethod(lambda _binary: True),
        )
        monkeypatch.setattr(
            llama_cpp,
            "_swa_cache_path",
            lambda: tmp_path / "studio" / "swa_cache.json",
        )

        backend = LlamaCppBackend()
        staged = backend._cpu_isolated_binary(str(wrapper))
        assert staged is not None
        staged_dir = Path(staged).parent
        assert Path(staged).read_bytes() == binary.read_bytes()
        assert not (staged_dir / "libggml-vulkan.so").exists()
        completed = subprocess.run(
            [staged],
            check = True,
            capture_output = True,
            text = True,
        )
        assert completed.stdout == "healthy\n"
        backend._cleanup_cpu_fallback_runtime()
        assert not staged_dir.exists()


def test_confirmed_projector_failure_replays_the_text_command_on_cpu(monkeypatch, tmp_path):
    backend, loaded, launches, fallback_sources = _run_cpu_fallback_load(
        monkeypatch,
        tmp_path,
        returncodes = [1, -11, None],
        first_output = "projector-incompatible",
        mmproj_from_argv = True,
    )

    assert loaded is True
    assert len(launches) == 3
    assert "--mmproj" in launches[0][0]
    assert "--mmproj" not in launches[1][0]
    assert len(fallback_sources) == 1
    assert "--mmproj" not in fallback_sources[0]
    assert backend._is_vision is False


def test_confirmed_projector_signal_retry_keeps_gpu_init_diagnosis(monkeypatch, tmp_path):
    with pytest.raises(RuntimeError, match = "GPU driver/runtime initialization crash"):
        _run_cpu_fallback_load(
            monkeypatch,
            tmp_path,
            returncodes = [1, -11],
            first_output = "projector-incompatible",
            mmproj_from_argv = True,
            cpu_fallback_available = False,
        )


@pytest.mark.parametrize("name", ["LLAMA_ARG_MMPROJ", "LLAMA_ARG_MMPROJ_URL"])
def test_env_projector_cpu_recovery_preserves_vision_state(monkeypatch, tmp_path, name):
    backend, loaded, launches, fallback_sources = _run_cpu_fallback_load(
        monkeypatch,
        tmp_path,
        returncodes = [-11, -11, None],
        mmproj_env = {name: "projector.gguf"},
    )

    assert loaded is True
    assert len(launches) == 3
    assert len(fallback_sources) == 1
    assert "--mmproj" not in fallback_sources[0]
    assert backend._is_vision is True


@pytest.mark.parametrize(
    "placement_env,extra_args",
    [
        ({"LLAMA_ARG_CPU_MOE": "true"}, None),
        ({"LLAMA_ARG_N_CPU_MOE": "4"}, None),
        ({"LLAMA_ARG_N_GPU_LAYERS": "20"}, None),
        ({"LLAMA_ARG_TENSOR_SPLIT": "1"}, None),
        ({"LLAMA_ARG_FIT": "off"}, None),
        ({"LLAMA_ARG_FIT_TARGET": "1024"}, None),
        ({"LLAMA_ARG_FIT_CTX": "4096"}, None),
        ({"LLAMA_ARG_DEVICE": "Vulkan0"}, None),
        ({"LLAMA_ARG_MAIN_GPU": "0"}, None),
        ({"LLAMA_ARG_SPLIT_MODE": "tensor"}, None),
        ({"LLAMA_ARG_SPLIT_MODE": "row"}, None),
        ({"LLAMA_ARG_OVERRIDE_TENSOR": ".*=Vulkan0"}, None),
        (
            {
                "LLAMA_ARG_SPLIT_MODE": "tensor",
                "LLAMA_ARG_TENSOR_SPLIT": "1",
            },
            None,
        ),
        *(({}, extras) for extras in _RAW_MAIN_PLACEMENT_ARGS),
    ],
)
def test_terminal_signal_with_explicit_child_placement_does_not_replay(
    monkeypatch, tmp_path, placement_env, extra_args
):
    def _gguf_string(value: str) -> bytes:
        encoded = value.encode()
        return struct.pack("<Q", len(encoded)) + encoded

    metadata = _gguf_string("general.architecture") + struct.pack("<I", 8) + _gguf_string("llama")
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(struct.pack("<IIQQ", 0x46554747, 3, 0, 1) + metadata)

    backend = LlamaCppBackend()
    backend._get_gpu_memory = lambda _binary = None: [(0, 8_000, 16_000)]
    backend._get_gpu_free_memory = lambda _binary = None: [(0, 8_000)]
    backend._read_gguf_metadata = lambda _path: None
    backend._can_estimate_kv = lambda: False
    backend._get_gguf_size_bytes = lambda _path: 1024
    backend._mmproj_vram_bytes = lambda _path: 0
    backend._resolve_launch_mmproj_path = lambda **_kwargs: None
    backend._apu_ram_shortfall_message = lambda *_args, **_kwargs: None
    backend._amd_apu_wants_unified_memory = lambda *_args, **_kwargs: False
    backend._find_llama_server_binary = lambda include_denied = False: "/fake/llama-server"
    backend._is_vulkan_backend = lambda _binary = None: True
    backend._vulkan_prebuilt_was_auto_selected = lambda _binary: True
    backend.probe_server_capabilities = lambda _binary: {"found": True}
    backend._wait_for_health = lambda timeout: False
    backend._record_server_pid = lambda _pid: None
    backend._clear_server_pid = lambda: None
    backend._llama_server_env_for_binary = lambda _binary: {
        "PATH": os.environ.get("PATH", ""),
        **placement_env,
    }
    isolated_replay_calls = []
    backend._cpu_isolated_binary = lambda _binary: isolated_replay_calls.append(_binary)

    launches = []

    class _SignalProcess:
        pid = 123
        returncode = -11
        stdout = ()

        def poll(self):
            return self.returncode

        def terminate(self):
            return None

        def wait(self, timeout = None):
            return self.returncode

        def kill(self):
            return None

    def _popen(cmd, **kwargs):
        launches.append((list(cmd), dict(kwargs["env"])))
        return _SignalProcess()

    monkeypatch.setattr(subprocess, "Popen", _popen)

    with pytest.raises(RuntimeError):
        backend.load_model(
            GgufLoadIntent(
                gguf_path = str(gguf),
                model_identifier = "owner/model",
                extra_args = extra_args,
            )
        )

    assert launches
    assert all(cmd[0] == "/fake/llama-server" for cmd, _env in launches)
    assert isolated_replay_calls == []


def test_vulkan_device_none_is_recorded_as_zero_vram(monkeypatch):
    monkeypatch.setattr(
        LlamaCppBackend, "_is_vulkan_backend", staticmethod(lambda _binary = None: True)
    )
    assert (
        LlamaCppBackend._zero_offload_gpu_flag(
            ["llama-server", "--gpu-layers", "0", "--device", "none"],
            [(0, 4096)],
            {},
        )
        is False
    )


@pytest.mark.parametrize("device", ["none", "cpu"])
def test_vulkan_cpu_device_is_zero_vram_when_probe_is_empty(monkeypatch, device):
    monkeypatch.setattr(
        LlamaCppBackend, "_is_vulkan_backend", staticmethod(lambda _binary = None: True)
    )
    assert (
        LlamaCppBackend._zero_offload_gpu_flag(
            ["llama-server", "--gpu-layers", "0", "--device", device],
            [],
            {},
        )
        is False
    )
    assert LlamaCppBackend._zero_offload_gpu_flag(["llama-server", "-ngl", "0"], [], {}) is None


def test_empty_probe_cpu_recovery_releases_chat_ownership(monkeypatch):
    route_path = Path(_BACKEND_DIR) / "routes" / "inference.py"
    spec = importlib.util.spec_from_file_location(
        "inference_route_for_empty_probe_cpu_recovery", route_path
    )
    route = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(route)

    backend = LlamaCppBackend()
    backend.matches_load_source = lambda _intent: False

    def _recover_on_cpu(*, intent):
        backend._gpu_memory_mode = "manual"
        backend._gpu_layers = 0
        backend._gpu_offload_active = backend._zero_offload_gpu_flag(
            ["llama-server", "--gpu-layers", "0", "--device", "none"],
            [],
            {},
        )
        backend._cpu_fallback_reason = "vulkan_startup_crash"
        return True

    backend.load_model = _recover_on_cpu
    unsloth_backend = SimpleNamespace(active_model_name = None)
    config = SimpleNamespace(
        identifier = "owner/model.gguf",
        display_name = "model.gguf",
        is_gguf = True,
        is_lora = False,
        is_vision = False,
        is_audio = False,
        is_local = True,
        gguf_hf_repo = None,
        gguf_file = "/models/model.gguf",
        gguf_mmproj_file = None,
        gguf_mtp_file = None,
        gguf_variant = None,
    )
    intent = GgufLoadIntent(
        model_identifier = config.identifier,
        gguf_path = config.gguf_file,
    )
    response = object()
    owner = [None]
    acquired = []
    released = []

    async def _inline_to_thread(func, /, *args, **kwargs):
        return func(*args, **kwargs)

    async def _prepare_load_placement(*_args, **_kwargs):
        return route._LoadPlacement(None, None, False, False)

    async def _wait_for_model_switch_idle(**_kwargs):
        return None

    def _acquire(requested, register = None):
        if register is not None:
            register()
        owner[0] = requested
        acquired.append(requested)

    def _release(requested):
        if owner[0] == requested:
            owner[0] = None
        released.append(requested)

    import core.inference.gpu_arbiter as arbiter
    import core.inference.llama_keepwarm as keepwarm

    monkeypatch.setattr(
        LlamaCppBackend, "_is_vulkan_backend", staticmethod(lambda _binary = None: True)
    )
    monkeypatch.setattr(route.asyncio, "to_thread", _inline_to_thread)
    monkeypatch.setattr(
        route,
        "_resolve_model_identifier_for_request",
        lambda *_args, **_kwargs: (config.identifier, config.identifier, False),
    )
    monkeypatch.setattr(route, "resolve_effective_chat_template_override", lambda **_kwargs: None)
    monkeypatch.setattr(route, "get_inference_backend", lambda: unsloth_backend)
    monkeypatch.setattr(route, "get_llama_cpp_backend", lambda: backend)
    monkeypatch.setattr(
        route,
        "ModelConfig",
        SimpleNamespace(from_identifier = lambda **_kwargs: config),
    )
    monkeypatch.setattr(route, "_hf_offline_if_unreachable_for", lambda *_args: nullcontext())
    monkeypatch.setattr(route, "_resolve_inherited_extra_args", lambda *_args: None)
    monkeypatch.setattr(route, "_prepare_load_placement", _prepare_load_placement)
    monkeypatch.setattr(route, "_resolve_gguf_load_intent", lambda *_args, **_kwargs: intent)
    monkeypatch.setattr(route, "_guard_chat_load_against_training", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(route, "_raise_if_sidecar_swap_in_progress", lambda: None)
    monkeypatch.setattr(route, "_wait_for_model_switch_idle", _wait_for_model_switch_idle)
    monkeypatch.setattr(route, "_close_load_event", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(route, "_gguf_load_response", lambda *_args, **_kwargs: response)
    monkeypatch.setattr(route.api_monitor, "record_lifecycle", lambda **_kwargs: object())
    monkeypatch.setattr(route, "_request_used_api_key", lambda _request: False)
    monkeypatch.setattr(keepwarm, "note_model_loaded", lambda _backend: None)
    monkeypatch.setattr(arbiter, "acquire_for", _acquire)
    monkeypatch.setattr(arbiter, "current_owner", lambda: owner[0])
    monkeypatch.setattr(arbiter, "release", _release)

    request = route.LoadRequest(model_path = config.identifier)
    fastapi_request = SimpleNamespace(
        app = SimpleNamespace(state = SimpleNamespace(llama_parallel_slots = 1))
    )
    result = asyncio.run(
        route._load_model_impl(request, fastapi_request, current_subject = "test-user")
    )

    assert result is response
    assert backend.holds_no_vram is True
    assert acquired == [arbiter.CHAT]
    assert released == [arbiter.CHAT]
    assert owner[0] is None


def test_duplicate_auto_request_matches_recovered_cpu_server(tmp_path):
    class _LiveProcess:
        def poll(self):
            return None

    backend = LlamaCppBackend()
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"GGUF")
    backend._process = _LiveProcess()
    backend._healthy = True
    backend._model_identifier = "owner/model"
    backend._gguf_path = str(gguf)
    backend._requested_n_ctx = 4096
    backend._requested_n_parallel = 1
    backend._requested_spec_mode = "auto"
    backend._gpu_memory_mode = "manual"
    backend._gpu_layers = 0
    backend._cpu_fallback_reason = "vulkan_startup_crash"
    assert backend.adopt_load_intent_if_matched(
        GgufLoadIntent(
            model_identifier = "owner/model",
            gguf_path = str(gguf),
            n_ctx = 4096,
            gpu_memory_mode = "auto",
        )
    )

    preserved = backend._preserve_cpu_fallback_intent(
        GgufLoadIntent(
            model_identifier = "owner/model",
            gguf_path = str(gguf),
            n_ctx = 8192,
            gpu_memory_mode = "manual",
            gpu_layers = 0,
        )
    )
    assert preserved.n_ctx == 8192
    assert preserved.cpu_fallback is True

    explicit_gpu = backend._preserve_cpu_fallback_intent(
        GgufLoadIntent(
            model_identifier = "owner/model",
            gguf_path = str(gguf),
            n_ctx = 8192,
            gpu_memory_mode = "manual",
            gpu_layers = 1,
        )
    )
    assert explicit_gpu.cpu_fallback is False


def test_cpu_fallback_request_keeps_the_replay_intent():
    route_path = Path(_BACKEND_DIR) / "routes" / "inference.py"
    spec = importlib.util.spec_from_file_location(
        "inference_route_for_cpu_fallback_rollback", route_path
    )
    route = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(route)
    request = LoadRequest(
        model_path = "owner/model",
        gpu_memory_mode = "manual",
        gpu_layers = 0,
        cpu_fallback = True,
    )
    intent = route._gguf_request_intent(
        GgufLoadIntent(model_identifier = "owner/model"),
        request,
        chat_template_override = None,
        extra_args = None,
        gpu_ids = None,
        n_parallel = 1,
    )
    assert intent.cpu_fallback is True


@pytest.mark.parametrize("model_cls", [LoadResponse, InferenceStatusResponse])
def test_success_response_reports_cpu_downgrade(model_cls):
    kwargs = {"cpu_fallback_reason": "vulkan_startup_crash"}
    if model_cls is LoadResponse:
        kwargs.update(status = "loaded", model = "owner/model", display_name = "model", inference = {})
    response = model_cls(**kwargs)
    assert response.model_dump()["cpu_fallback_reason"] == "vulkan_startup_crash"
