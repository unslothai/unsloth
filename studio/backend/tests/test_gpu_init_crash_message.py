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
    # _cpu_isolated_binary asks whether this is an install tree, which is not
    # the same question as whether the updater can replace the file: a
    # --with-llama-cpp-dir checkout is the active install while being the
    # user's to maintain. Staging a CPU copy only reads, so it uses this one.
    monkeypatch.setattr(
        LlamaCppBackend,
        "_is_llama_install_tree",
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
    extra_args = None,
    vulkan = True,
    cpu_fallback = False,
    resident_fallback = False,
    cancel_after = None,
    cancel_in_prepare = False,
    platform = None,
    sink = None,
):
    if platform is not None:
        monkeypatch.setattr(llama_cpp.sys, "platform", platform)

    def _gguf_string(value: str) -> bytes:
        encoded = value.encode()
        return struct.pack("<Q", len(encoded)) + encoded

    metadata = _gguf_string("general.architecture") + struct.pack("<I", 8) + _gguf_string("llama")
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(struct.pack("<IIQQ", 0x46554747, 3, 0, 1) + metadata)
    mmproj = tmp_path / "mmproj.gguf"
    mmproj.write_bytes(b"projector")

    backend = LlamaCppBackend()
    backend._get_gpu_memory = lambda _binary = None, **_kw: []
    backend._get_gpu_free_memory = lambda _binary = None, **_kw: []
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
        staticmethod(lambda _binary = None: vulkan),
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
        if cancel_after is not None and len(launches) >= cancel_after:
            backend._cancel_event.set()
        return returncodes[len(launches) - 1] is None

    cleanups = []

    def _prepare_cpu_fallback(_binary, failed_cmd, _env, _server_caps, **_kwargs):
        fallback_sources.append(list(failed_cmd))
        if cancel_in_prepare:
            # An /unload landing while the runtime is being staged.
            backend._cancel_event.set()
        # A callable decides per command, for builds that can only replay some.
        available = (
            cpu_fallback_available(failed_cmd)
            if callable(cpu_fallback_available)
            else cpu_fallback_available
        )
        if not available:
            return None
        return ["/staged/llama-server", "--device", "none"], None

    backend._wait_for_health = _wait_for_health
    backend._prepare_cpu_fallback_launch = _prepare_cpu_fallback
    backend._cleanup_cpu_fallback_runtime = lambda: cleanups.append(True)
    monkeypatch.setattr(subprocess, "Popen", _popen)
    # Lets a test that expects the load to raise still read what was spawned.
    if sink is not None:
        sink["launches"] = launches
        sink["fallback_sources"] = fallback_sources
        sink["cleanups"] = cleanups
    if resident_fallback:
        backend._cpu_fallback_reason = "vulkan_startup_crash"

    loaded = backend.load_model(
        GgufLoadIntent(
            gguf_path = str(gguf),
            mmproj_path = str(mmproj) if mmproj_from_argv else None,
            model_identifier = "owner/model",
            is_vision = mmproj_from_argv,
            extra_args = list(extra_args) if extra_args else None,
            cpu_fallback = cpu_fallback,
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


class TestPlatformMatrix:
    """Every OS and runtime flavour Unsloth ships, so the recovery cannot leak
    into a path that already worked."""

    # (platform, library prefix, library suffix, binary name)
    OSES = {
        "linux": ("linux", "lib", "so", "llama-server"),
        "windows": ("win32", "", "dll", "llama-server.exe"),
        "macos": ("darwin", "lib", "dylib", "llama-server"),
    }
    RUNTIMES = {
        "vulkan": ("base", "cpu", "vulkan"),
        "cuda": ("base", "cpu", "cuda"),
        "hip": ("base", "cpu", "hip"),
        "cpu": ("base", "cpu"),
        # A custom multi-backend build defers to CUDA/HIP, never to Vulkan.
        "vulkan+cuda": ("base", "cpu", "vulkan", "cuda"),
    }

    def _install(self, monkeypatch, tmp_path, os_key, runtime):
        platform, prefix, suffix, exe = self.OSES[os_key]
        bindir = tmp_path / "install" / "build" / "bin"
        bindir.mkdir(parents = True)
        binary = bindir / exe
        binary.write_bytes(b"llama-server")
        binary.chmod(0o755)
        for backend in self.RUNTIMES[runtime]:
            stem = "cpu-haswell" if backend == "cpu" else backend
            (bindir / f"{prefix}ggml-{stem}.{suffix}").write_bytes(backend.encode())
            if backend == "cpu":
                (bindir / f"{prefix}ggml-cpu.{suffix}").write_bytes(b"cpu")
        (bindir / f"{prefix}llama.{suffix}").write_bytes(b"llama")
        monkeypatch.setattr(llama_cpp.sys, "platform", platform)
        monkeypatch.setattr(
            LlamaCppBackend, "_is_unsloth_managed_binary", staticmethod(lambda _binary: True)
        )
        monkeypatch.setattr(
            LlamaCppBackend, "_is_llama_install_tree", staticmethod(lambda _binary: True)
        )
        monkeypatch.setattr(
            llama_cpp, "_swa_cache_path", lambda: tmp_path / "studio" / "swa_cache.json"
        )
        return binary

    @pytest.mark.parametrize("os_key", list(OSES))
    @pytest.mark.parametrize("runtime", list(RUNTIMES))
    def test_only_a_vulkan_only_build_is_ever_staged(self, monkeypatch, tmp_path, os_key, runtime):
        binary = self._install(monkeypatch, tmp_path, os_key, runtime)
        backend = LlamaCppBackend()
        backend._llama_server_env_for_binary = lambda _binary: {
            "PATH": "/staged",
            "LD_LIBRARY_PATH": "/staged",
        }

        prepared = backend._prepare_cpu_fallback_launch(
            str(binary), [str(binary), "-m", "m.gguf"], {}, {"found": True}
        )

        # Backend detection reads .so / .dll only, so a macOS bundle is never Vulkan.
        expected = os_key != "macos" and runtime == "vulkan"
        assert (prepared is not None) is expected
        if not expected:
            assert backend._cpu_fallback_runtime is None
            return
        staged = {p.name.lower() for p in Path(prepared[0][0]).parent.iterdir()}
        assert not any("ggml-vulkan" in n or "ggml-cuda" in n or "ggml-hip" in n for n in staged)
        assert any("ggml-cpu" in n for n in staged)
        assert any(n.startswith(("libllama", "llama.")) for n in staged)
        backend._cleanup_cpu_fallback_runtime()

    @pytest.mark.parametrize(
        "marker,eligible",
        [
            ({"llama_backend": None}, True),  # auto Intel, post-#7188
            ({"llama_backend": "auto"}, True),  # auto AMD, post-#8050
            ({}, True),  # pre-#7188, key absent
            ({"llama_backend": ""}, True),
            ({"llama_backend": "vulkan"}, False),  # explicit choice
            ({"llama_backend": "cuda"}, False),  # unknown/future value
            ({"llama_backend": None, "force_cpu": True}, False),
        ],
    )
    def test_marker_states_old_and_new(self, monkeypatch, tmp_path, marker, eligible):
        binary = self._install(monkeypatch, tmp_path, "linux", "vulkan")
        payload = {"asset": "llama-b1-bin-ubuntu-vulkan-x64.tar.gz", "force_cpu": False}
        payload.update(marker)
        (tmp_path / "UNSLOTH_PREBUILT_INFO.json").write_text(json.dumps(payload))
        import utils.llama_cpp_update as update

        monkeypatch.setattr(update, "_llama_install_root", lambda _binary: tmp_path)
        for name in ("UNSLOTH_FORCE_VULKAN", "UNSLOTH_LLAMA_CPP_BACKEND"):
            monkeypatch.delenv(name, raising = False)

        assert (
            LlamaCppBackend._auto_vulkan_cpu_fallback_eligible(
                str(binary), GgufLoadIntent(model_identifier = "owner/model"), None, {}
            )
            is eligible
        )

    @pytest.mark.parametrize("marker_text", ["not json", "", "[]", '{"llama_backend": 7}'])
    def test_a_corrupt_marker_is_ineligible_and_never_raises(
        self, monkeypatch, tmp_path, marker_text
    ):
        binary = self._install(monkeypatch, tmp_path, "linux", "vulkan")
        (tmp_path / "UNSLOTH_PREBUILT_INFO.json").write_text(marker_text)
        import utils.llama_cpp_update as update

        monkeypatch.setattr(update, "_llama_install_root", lambda _binary: tmp_path)

        assert (
            LlamaCppBackend._auto_vulkan_cpu_fallback_eligible(
                str(binary), GgufLoadIntent(model_identifier = "owner/model"), None, {}
            )
            is False
        )


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

    def test_auto_suppresses_a_stale_legacy_vulkan_flag(self, monkeypatch, tmp_path):
        # UNSLOTH_LLAMA_CPP_BACKEND=auto outranks UNSLOTH_FORCE_VULKAN everywhere
        # else, so setup detected this bundle rather than being told to install it.
        # Reading the legacy flag as a choice here would leave a crashing Vulkan
        # install with no automatic CPU replay.
        self._managed_marker(monkeypatch, tmp_path, llama_backend = "auto")
        monkeypatch.setattr(
            LlamaCppBackend, "_is_vulkan_backend", staticmethod(lambda _binary = None: True)
        )
        monkeypatch.setenv("UNSLOTH_LLAMA_CPP_BACKEND", "auto")
        monkeypatch.setenv("UNSLOTH_FORCE_VULKAN", "1")

        assert LlamaCppBackend._auto_vulkan_cpu_fallback_eligible(
            "/managed/llama-server", GgufLoadIntent(model_identifier = "m"), None, {}
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

    def test_auto_environment_override_does_not_make_a_custom_binary_managed(self, monkeypatch):
        import utils.llama_cpp_update as update

        monkeypatch.setenv("UNSLOTH_LLAMA_CPP_BACKEND", "auto")
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
            LlamaCppBackend, "_is_vulkan_backend", staticmethod(lambda _binary = None: True)
        )
        monkeypatch.setattr(
            backend,
            "_cpu_isolated_replay",
            lambda _cmd, _env, _caps, **_kwargs: ["original", "--device", "none"],
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

    def test_a_replay_request_on_a_non_vulkan_runtime_loads_normally(self, monkeypatch, tmp_path):
        """Switching the managed build must not turn a rollback into a failure."""
        backend, loaded, launches, fallback_sources = _run_cpu_fallback_load(
            monkeypatch,
            tmp_path,
            returncodes = [None],
            vulkan = False,
            cpu_fallback = True,
        )

        assert loaded is True
        assert fallback_sources == []
        assert backend._cpu_fallback_reason is None

    def test_the_windows_full_offload_thread_cap_is_undone(self, monkeypatch, tmp_path):
        """Two threads and PASSIVE OpenMP are tuned for a GPU doing the work, so
        a CPU replay that kept them would decode on two cores."""
        env = {"OMP_NUM_THREADS": "2", "OMP_WAIT_POLICY": "PASSIVE", "KEEP": "1"}
        cmd = ["llama-server", "-m", "m.gguf", "--threads", "2", "--jinja"]

        replay = LlamaCppBackend._cpu_isolated_replay(
            cmd, env, {"found": True}, drop_full_offload_threads = True
        )

        assert "--threads" not in replay
        assert "--jinja" in replay
        assert "OMP_NUM_THREADS" not in env
        assert "OMP_WAIT_POLICY" not in env
        assert env["KEEP"] == "1"

    def test_a_user_thread_choice_survives_the_cpu_replay(self, monkeypatch, tmp_path):
        """The caller only reports the cap as ours when the user set no override."""
        env = {"OMP_NUM_THREADS": "8"}
        cmd = ["llama-server", "-m", "m.gguf", "--threads", "8"]

        replay = LlamaCppBackend._cpu_isolated_replay(cmd, env, {"found": True})

        assert replay[replay.index("--threads") + 1] == "8"
        assert env["OMP_NUM_THREADS"] == "8"

    def test_non_vulkan_managed_runtime_is_never_staged(self, monkeypatch, tmp_path):
        binary = _managed_runtime(monkeypatch, tmp_path)
        monkeypatch.setattr(
            LlamaCppBackend, "_is_vulkan_backend", staticmethod(lambda _binary = None: False)
        )
        backend = LlamaCppBackend()

        prepared = backend._prepare_cpu_fallback_launch(
            str(binary), [str(binary), "-m", "model.gguf"], {}, {"found": True}
        )

        assert prepared is None
        assert backend._cpu_fallback_runtime is None

    def test_an_updated_install_restages_instead_of_reusing(self, monkeypatch, tmp_path):
        binary = _managed_runtime(monkeypatch, tmp_path)
        backend = LlamaCppBackend()
        first = backend._cpu_isolated_binary(str(binary))
        assert first is not None
        assert backend._cpu_isolated_binary(str(binary)) == first

        # An update swaps the tree in place, so the path is unchanged.
        binary.unlink()
        binary.write_bytes(b"rebuilt binary")
        binary.chmod(0o755)

        second = backend._cpu_isolated_binary(str(binary))
        assert second is not None and second != first
        assert Path(second).read_bytes() == b"rebuilt binary"
        assert not Path(first).parent.exists()

    def test_a_runtime_abandoned_by_a_dead_studio_is_swept(self, monkeypatch, tmp_path):
        binary = _managed_runtime(monkeypatch, tmp_path)
        runtime_root = tmp_path / "studio" / "runtime"
        runtime_root.mkdir(parents = True)
        dead = runtime_root / "llama-cpu-dead"
        dead.mkdir()
        # A pid no live process can hold, so the sweep must collect it.
        (dead / "UNSLOTH_OWNER_PID").write_text("0")
        legacy = runtime_root / "llama-cpu-legacy"
        legacy.mkdir()

        staged = LlamaCppBackend()._cpu_isolated_binary(str(binary))

        assert staged is not None
        assert not dead.exists()
        # No owner stamp means an older Unsloth wrote it; leave it alone.
        assert legacy.exists()

    def test_a_live_owner_keeps_its_runtime(self, monkeypatch, tmp_path):
        binary = _managed_runtime(monkeypatch, tmp_path)
        one = LlamaCppBackend()
        first = Path(one._cpu_isolated_binary(str(binary))).parent

        second = Path(LlamaCppBackend()._cpu_isolated_binary(str(binary))).parent

        assert first != second
        assert first.exists()
        assert (first / "UNSLOTH_OWNER_PID").read_text() == str(os.getpid())

    def test_a_failed_cleanup_is_retried_rather_than_orphaned(self, monkeypatch, tmp_path):
        binary = _managed_runtime(monkeypatch, tmp_path)
        backend = LlamaCppBackend()
        assert backend._cpu_isolated_binary(str(binary)) is not None
        tempdir = backend._cpu_fallback_runtime.tempdir
        staged_dir = Path(tempdir.name)
        attempts = []
        real_cleanup = tempdir.cleanup

        def _cleanup():
            attempts.append(True)
            if len(attempts) == 1:
                raise PermissionError("locked")
            real_cleanup()

        monkeypatch.setattr(tempdir, "cleanup", _cleanup)

        backend._cleanup_cpu_fallback_runtime()
        assert staged_dir.exists()
        assert backend._pending_cpu_fallback_cleanups == [tempdir]

        backend._cleanup_cpu_fallback_runtime()
        assert not staged_dir.exists()
        assert backend._pending_cpu_fallback_cleanups == []

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
            LlamaCppBackend,
            "_is_llama_install_tree",
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


def test_env_projector_cpu_recovery_keeps_audio_input(monkeypatch, tmp_path):
    """An audio projector handed over by env must not report as text-only."""
    projector = tmp_path / "env-mmproj.gguf"
    projector.write_bytes(b"projector")
    read = []
    import utils.models.gguf_metadata as gguf_metadata

    monkeypatch.setattr(
        gguf_metadata, "read_mmproj_audio_capability", lambda path: read.append(path) or True
    )
    monkeypatch.setenv("LLAMA_ARG_MMPROJ", str(projector))

    backend, loaded, launches, fallback_sources = _run_cpu_fallback_load(
        monkeypatch,
        tmp_path,
        returncodes = [-11, -11, None],
        mmproj_env = {"LLAMA_ARG_MMPROJ": str(projector)},
    )

    assert loaded is True
    assert read == [str(projector)]
    assert backend._is_vision is True
    assert backend._mmproj_has_audio is True


def test_inherited_split_mode_dropped_before_spawn_still_recovers(monkeypatch, tmp_path):
    """A stale env value this launch scrubs never reached the crashed child."""
    backend, loaded, launches, fallback_sources = _run_cpu_fallback_load(
        monkeypatch,
        tmp_path,
        returncodes = [-11, -11, None],
        mmproj_env = {"LLAMA_ARG_SPLIT_MODE": "tensor"},
    )

    assert loaded is True
    assert "LLAMA_ARG_SPLIT_MODE" not in launches[0][1]
    assert len(fallback_sources) == 1
    assert backend._cpu_fallback_reason == "vulkan_startup_crash"


def test_drafter_survives_the_cpu_replay(monkeypatch, tmp_path):
    """The speculative retry must not become the source of the CPU replay."""
    backend, loaded, launches, fallback_sources = _run_cpu_fallback_load(
        monkeypatch,
        tmp_path,
        returncodes = [-11, -11, -11, None],
        extra_args = ["--spec-type", "mtp"],
    )

    assert loaded is True
    assert len(fallback_sources) == 1
    assert "--spec-default" not in fallback_sources[0]
    assert backend._spec_fallback_reason is None


def test_drafter_replay_falls_back_to_the_stripped_command(monkeypatch, tmp_path):
    """A build that cannot CPU-pin a drafter still recovers, drafterless."""
    backend, loaded, launches, fallback_sources = _run_cpu_fallback_load(
        monkeypatch,
        tmp_path,
        returncodes = [-11, -11, -11, None],
        extra_args = ["--spec-type", "mtp"],
        cpu_fallback_available = lambda cmd: "--spec-default" in cmd,
    )

    assert loaded is True
    assert len(fallback_sources) == 2
    assert "--spec-default" not in fallback_sources[0]
    assert "--spec-default" in fallback_sources[1]
    assert backend._cpu_fallback_reason == "vulkan_startup_crash"


def test_a_drafter_that_cannot_start_anywhere_still_recovers(monkeypatch, tmp_path):
    """The drafter replay can be staged and still die of the drafter, so the
    drafterless argv must get its own attempt rather than the load aborting."""
    backend, loaded, launches, fallback_sources = _run_cpu_fallback_load(
        monkeypatch,
        tmp_path,
        # The speculative CPU replay (4th launch) dies too; the drafterless one wins.
        returncodes = [-11, -11, -11, 1, None],
        extra_args = ["--spec-type", "mtp"],
    )

    assert loaded is True
    assert len(fallback_sources) == 2
    assert "--spec-default" not in fallback_sources[0]
    assert "--spec-default" in fallback_sources[1]
    assert backend._cpu_fallback_reason == "vulkan_startup_crash"


def test_a_replay_request_needs_the_same_bar_as_the_crash_path(monkeypatch, tmp_path):
    """cpu_fallback is client-supplied, so an explicitly GPU-pinned request must
    not be silently staged onto CPU and reported as a Vulkan crash."""
    backend, loaded, launches, fallback_sources = _run_cpu_fallback_load(
        monkeypatch,
        tmp_path,
        returncodes = [None],
        cpu_fallback = True,
        extra_args = ["--device", "Vulkan0"],
    )

    assert loaded is True
    assert fallback_sources == []
    assert backend._cpu_fallback_reason is None


def test_a_cancelled_load_never_spawns_the_cpu_replay(monkeypatch, tmp_path):
    """An /unload racing the crash must not leave a CPU server behind."""
    sink = {}
    with pytest.raises(Exception):
        _run_cpu_fallback_load(
            monkeypatch,
            tmp_path,
            returncodes = [-11, -11, None],
            cancel_after = 1,
            sink = sink,
        )

    assert sink["fallback_sources"] == []


def test_the_exit_handler_removes_the_runtime_after_the_kill():
    """TemporaryDirectory's exit hook is registered later, so it runs first and
    cannot delete a runtime whose server is still holding the files open."""
    backend = LlamaCppBackend()
    order = []
    backend._kill_process = lambda: order.append("kill")
    backend._cleanup_cpu_fallback_runtime = lambda: order.append("clean")

    backend._cleanup()

    assert order == ["kill", "clean"]


def test_an_unload_during_staging_takes_the_runtime_back(monkeypatch, tmp_path):
    """Staging copies a whole runtime, so /unload can land after the first gate
    and find nothing registered to clean up."""
    sink = {}
    with pytest.raises(Exception):
        _run_cpu_fallback_load(
            monkeypatch,
            tmp_path,
            returncodes = [-11, -11, None],
            cancel_in_prepare = True,
            sink = sink,
        )

    assert len(sink["fallback_sources"]) == 1  # staged
    assert len(sink["launches"]) == 2  # never spawned
    assert sink["cleanups"]  # and handed back


def test_a_windows_ggml_assert_reaches_the_cpu_replay(monkeypatch, tmp_path):
    """MSVC turns GGML_ASSERT into a CRT abort (exit 3), not a signal."""
    _backend, loaded, launches, fallback_sources = _run_cpu_fallback_load(
        monkeypatch,
        tmp_path,
        returncodes = [3, None],
        platform = "win32",
    )

    assert loaded is True
    assert len(fallback_sources) == 1
    assert "--device" in launches[-1][0]


def test_a_posix_exit_three_is_not_a_crash(monkeypatch, tmp_path):
    """Exit 3 is an ordinary failure everywhere except the MSVC runtime."""
    with pytest.raises(Exception):
        _run_cpu_fallback_load(
            monkeypatch,
            tmp_path,
            returncodes = [3, None],
            platform = "linux",
        )


def test_an_accepted_replay_request_normalizes_the_placement(monkeypatch, tmp_path):
    """A client that sends only the flag must still end up on manual/0, or the
    session reports an Auto GPU placement for a CPU-only server."""
    backend, loaded, launches, fallback_sources = _run_cpu_fallback_load(
        monkeypatch,
        tmp_path,
        returncodes = [None],
        cpu_fallback = True,
    )

    assert loaded is True
    assert backend._cpu_fallback_reason == "vulkan_startup_crash"
    assert backend._gpu_memory_mode == "manual"
    assert backend._gpu_layers == 0
    assert backend._gpu_ids is None
    assert backend._last_load_intent.cpu_fallback is True


def test_an_ineligible_replay_request_drops_the_recovery_state(monkeypatch, tmp_path):
    """The kill phase keeps the staged recovery for a replay. A request that then
    fails the eligibility bar must not leave the next model wearing it."""
    backend, loaded, launches, fallback_sources = _run_cpu_fallback_load(
        monkeypatch,
        tmp_path,
        returncodes = [None],
        cpu_fallback = True,
        resident_fallback = True,
        extra_args = ["--device", "Vulkan0"],
    )

    assert loaded is True
    assert fallback_sources == []
    assert backend._cpu_fallback_reason is None
    assert backend._cpu_fallback_runtime is None
    assert backend._last_load_intent.cpu_fallback is False


def test_preserving_a_recovery_ignores_env_the_replay_strips(monkeypatch, tmp_path):
    """The replay pops every main-model placement var, so an inherited one must
    not stop the resident CPU intent from being carried into a reload."""
    monkeypatch.setenv("LLAMA_ARG_SPLIT_MODE", "tensor")
    backend = LlamaCppBackend()
    backend._cpu_fallback_reason = "vulkan_startup_crash"
    intent = GgufLoadIntent(
        model_identifier = "owner/model",
        gguf_path = str(tmp_path / "model.gguf"),
    )

    preserved = backend._preserve_cpu_fallback_intent(intent, source_matches = True)

    assert preserved.cpu_fallback is True
    assert preserved.gpu_memory_mode == "manual"
    assert preserved.gpu_layers == 0


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
    backend._get_gpu_memory = lambda _binary = None, **_kw: [(0, 8_000, 16_000)]
    backend._get_gpu_free_memory = lambda _binary = None, **_kw: [(0, 8_000)]
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

    # /load now hands the loader its scoped cancel event alongside the intent.
    def _recover_on_cpu(*, intent, load_cancel_event = None):
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


_HIP_ROCR_MISMATCH = (
    "llama-server: symbol lookup error: "
    "/home/t/.unsloth/llama.cpp/build/bin/libamdhip64.so.7: "
    "undefined symbol: hsa_amd_queue_create, version ROCR_1"
)
_VRAM_CRASH = "ggml_backend_hip_buffer_type_alloc_buffer: failed to allocate"


def _fit_mode(cmd):
    if "--fit" in cmd:
        return cmd[cmd.index("--fit") + 1]
    return None


def _run_full_offload_spawns(monkeypatch, tmp_path, *, outputs, returncodes):
    """Drive load_model with a model that fits on GPU (--fit off).

    Each spawn's stdout is ``outputs[i]`` and its exit is ``returncodes[i]``
    (None means healthy). The child env prepends /opt/rocm/lib unless
    ``use_system_rocm=False``, which is the retry under test.
    """

    def _gguf_string(value: str) -> bytes:
        encoded = value.encode()
        return struct.pack("<Q", len(encoded)) + encoded

    metadata = _gguf_string("general.architecture") + struct.pack("<I", 8) + _gguf_string("llama")
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(struct.pack("<IIQQ", 0x46554747, 3, 0, 1) + metadata)

    backend = LlamaCppBackend()
    backend._get_gpu_memory = lambda _binary = None, **_kw: [(0, 24 * 1024**3, 24 * 1024**3)]
    backend._get_gpu_free_memory = lambda _binary = None, **_kw: [(0, 24 * 1024**3)]
    backend._read_gguf_metadata = lambda _path: None
    backend._can_estimate_kv = lambda: False
    backend._get_gguf_size_bytes = lambda _path: 1024
    backend._mmproj_vram_bytes = lambda _path: 0
    backend._resolve_launch_mmproj_path = lambda **_kwargs: None
    backend._apu_ram_shortfall_message = lambda *_args, **_kwargs: None
    backend._amd_apu_wants_unified_memory = lambda *_args, **_kwargs: False
    backend._find_llama_server_binary = lambda include_denied = False: "/fake/llama-server"
    backend._select_gpus = lambda *_a, **_k: ([0], False)
    backend._host_torch_is_rocm = lambda: False
    monkeypatch.setattr(
        LlamaCppBackend, "_is_vulkan_backend", staticmethod(lambda _binary = None: False)
    )
    backend.probe_server_capabilities = lambda _binary: {"found": True}
    backend._record_server_pid = lambda _pid: None
    backend._clear_server_pid = lambda: None
    monkeypatch.setattr(
        llama_cpp, "_swa_cache_path", lambda: tmp_path / "studio" / "swa_cache.json"
    )

    def _env_for_binary(
        _binary,
        *,
        use_system_rocm = True,
        **_k,
    ):
        ld = "/opt/rocm/lib:/bundle/bin" if use_system_rocm else "/bundle/bin"
        return {"PATH": os.environ.get("PATH", ""), "LD_LIBRARY_PATH": ld}

    backend._llama_server_env_for_binary = _env_for_binary
    backend._prepare_cpu_fallback_launch = lambda *_a, **_kw: None
    # Class-level and set by a successful bundle-only retry, so give every run a
    # fresh one rather than leaking a correction into the next test.
    monkeypatch.setattr(LlamaCppBackend, "_bundle_only_rocm_dirs", {})

    launches = []

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

    _real_popen = subprocess.Popen

    def _popen(cmd, **kwargs):
        # Only the server is a launch. A host with the rocm_sdk wheel installed
        # shells out to offload-arch from inside load_model, which would land at
        # index 0 and shift every assertion below onto the wrong process.
        if not cmd or str(cmd[0]) != "/fake/llama-server":
            return _real_popen(cmd, **kwargs)
        idx = len(launches)
        launches.append((list(cmd), dict(kwargs["env"])))
        code = returncodes[idx] if idx < len(returncodes) else 1
        return _Process(code)

    def _wait_for_health(timeout = 600.0):
        idx = len(launches) - 1
        if 0 <= idx < len(outputs):
            backend._stdout_lines = [outputs[idx]]
        code = returncodes[idx] if 0 <= idx < len(returncodes) else 1
        return code is None

    backend._wait_for_health = _wait_for_health
    monkeypatch.setattr(subprocess, "Popen", _popen)
    loaded = False
    error = None
    try:
        loaded = backend.load_model(
            GgufLoadIntent(gguf_path = str(gguf), model_identifier = "owner/model")
        )
    except Exception as exc:
        error = exc
    return launches, loaded, error


class TestHipRocrRetryKeepsFitBudget:
    """The ROCm env correction has its own attempt, so a later VRAM crash
    after the bundle-only launch can still take the --fit on recovery.
    """

    def test_mix_then_vram_still_gets_fit_on(self, monkeypatch, tmp_path):
        launches, loaded, error = _run_full_offload_spawns(
            monkeypatch,
            tmp_path,
            outputs = [_HIP_ROCR_MISMATCH, _VRAM_CRASH, ""],
            returncodes = [127, 1, None],
        )
        assert error is None
        assert loaded
        assert len(launches) == 3
        cmd0, env0 = launches[0]
        cmd1, env1 = launches[1]
        cmd2, env2 = launches[2]
        assert env0["LD_LIBRARY_PATH"].startswith("/opt/rocm/lib")
        assert "/opt/rocm/lib" not in env1["LD_LIBRARY_PATH"].split(":")
        assert "/opt/rocm/lib" not in env2["LD_LIBRARY_PATH"].split(":")
        assert _fit_mode(cmd0) == "off"
        assert _fit_mode(cmd1) == "off"
        assert _fit_mode(cmd2) == "on"

    def test_mix_then_healthy_bundled_hip_does_not_fit_retry(self, monkeypatch, tmp_path):
        launches, loaded, error = _run_full_offload_spawns(
            monkeypatch,
            tmp_path,
            outputs = [_HIP_ROCR_MISMATCH, ""],
            returncodes = [127, None],
        )
        assert error is None
        assert loaded
        assert len(launches) == 2
        assert _fit_mode(launches[0][0]) == "off"
        assert _fit_mode(launches[1][0]) == "off"
        assert "/opt/rocm/lib" not in launches[1][1]["LD_LIBRARY_PATH"].split(":")
        # Proved on this host, so later children skip the prepend outright.
        assert LlamaCppBackend._prefers_bundle_only_rocm("/fake/llama-server")

    def test_a_later_launch_in_the_same_load_still_records_the_correction(
        self, monkeypatch, tmp_path
    ):
        # The correction edits the shared env, so it survives into the outer
        # recovery spawns (no-flash here). Those call _spawn_and_wait afresh, so
        # a per-call flag left the launch that actually came up healthy
        # unrecorded and the sidecar kept the prepend.
        launches, loaded, error = _run_full_offload_spawns(
            monkeypatch,
            tmp_path,
            outputs = [_HIP_ROCR_MISMATCH, "", "", ""],
            returncodes = [127, -11, -11, None],
        )
        assert error is None
        assert loaded
        assert len(launches) == 4
        assert "/opt/rocm/lib" not in launches[-1][1]["LD_LIBRARY_PATH"].split(":")
        assert LlamaCppBackend._prefers_bundle_only_rocm("/fake/llama-server")

    def test_a_mix_the_retry_did_not_fix_does_not_spend_the_fit_slot(self, monkeypatch, tmp_path):
        # Bundle-only did not help, so the symbol is still missing. --fit cannot
        # load a missing symbol: stop at two launches and report the mix.
        launches, loaded, error = _run_full_offload_spawns(
            monkeypatch,
            tmp_path,
            outputs = [_HIP_ROCR_MISMATCH, _HIP_ROCR_MISMATCH],
            returncodes = [127, 127],
        )
        assert not loaded
        assert len(launches) == 2
        assert _fit_mode(launches[1][0]) == "off"
        assert "HIP/ROCR" in str(error)
        # The retry did not fix it, so nothing was proved: do not latch.
        assert not LlamaCppBackend._prefers_bundle_only_rocm("/fake/llama-server")
