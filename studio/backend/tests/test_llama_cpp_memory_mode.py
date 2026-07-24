# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for GGUF memory placement mode and explicit GPU selection (#7164)."""

from __future__ import annotations

import os
import struct
import subprocess
import sys
import types as _types
from pathlib import Path
from unittest.mock import patch

_REAL_POPEN = subprocess.Popen

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)


def _install_stub_if_absent(name: str, build):
    """Install a fallback stub without shadowing an importable module."""
    if name in sys.modules:
        return
    try:
        __import__(name)
        return
    except Exception:
        sys.modules[name] = build()


def _build_loggers_stub():
    mod = _types.ModuleType("loggers")
    mod.get_logger = lambda name: __import__("logging").getLogger(name)
    return mod


def _build_structlog_stub():
    mod = _types.ModuleType("structlog")
    mod.get_logger = lambda *a, **k: __import__("logging").getLogger("stub")
    return mod


def _build_httpx_stub():
    mod = _types.ModuleType("httpx")
    for _exc_name in (
        "ConnectError",
        "TimeoutException",
        "ReadTimeout",
        "ReadError",
        "RemoteProtocolError",
        "CloseError",
    ):
        setattr(mod, _exc_name, type(_exc_name, (Exception,), {}))
    mod.Timeout = type("T", (), {"__init__": lambda s, *a, **k: None})
    mod.Client = type(
        "Client",
        (),
        {
            "__init__": lambda self, **kw: None,
            "__enter__": lambda self: self,
            "__exit__": lambda self, *a: None,
        },
    )
    return mod


def _build_jwt_stub():
    mod = _types.ModuleType("jwt")
    mod.decode = lambda *a, **k: {}
    mod.ExpiredSignatureError = type("ExpiredSignatureError", (Exception,), {})
    mod.InvalidTokenError = type("InvalidTokenError", (Exception,), {})
    return mod


_install_stub_if_absent("loggers", _build_loggers_stub)
_install_stub_if_absent("structlog", _build_structlog_stub)
_install_stub_if_absent("httpx", _build_httpx_stub)
_install_stub_if_absent("jwt", _build_jwt_stub)

import pytest

from core.inference.llama_cpp import LlamaCppBackend

_GGUF_MAGIC = 0x46554747
_VTYPE_STRING = 8


def _enc_string(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def _enc_kv_string(key: str, value: str) -> bytes:
    return _enc_string(key) + struct.pack("<I", _VTYPE_STRING) + _enc_string(value)


def _write_minimal_gguf(path: Path, *, arch: str = "llama") -> Path:
    body = _enc_kv_string("general.architecture", arch)
    header = struct.pack("<IIQQ", _GGUF_MAGIC, 3, 0, 1)
    path.write_bytes(header + body)
    return path


class _FakeProcess:
    """Minimal stand-in so is_loaded returns True."""

    def terminate(self):
        pass

    def wait(self, timeout = None):
        return 0

    def kill(self):
        pass

    def poll(self):
        return 0


def _loaded_backend(**overrides):
    backend = LlamaCppBackend()
    backend._process = _FakeProcess()
    backend._healthy = True
    backend._model_identifier = "owner/repo"
    backend._hf_variant = "Q4_K_M"
    backend._requested_n_ctx = 8192
    backend._cache_type_kv = None
    backend._speculative_type = None
    backend._requested_spec_mode = "auto"
    backend._chat_template_override = None
    backend._is_vision = False
    backend._extra_args = None
    backend._extra_args_source = None
    backend._gguf_path = None
    backend._gpu_ids = None
    backend._requested_memory_mode = None
    for key, value in overrides.items():
        setattr(backend, key, value)
    if "_gpu_ids" in overrides and "_requested_gpu_ids" not in overrides:
        backend._requested_gpu_ids = list(overrides["_gpu_ids"] or []) or None
    return backend


# ── _memory_mode_flags ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "mode,expected",
    [
        (None, []),
        ("default", []),
        ("DEFAULT", []),
        ("pinned", ["--mlock"]),
        ("PINNED", ["--mlock"]),
        ("resident", ["--no-mmap", "--mlock"]),
        ("RESIDENT", ["--no-mmap", "--mlock"]),
        ("", []),
    ],
)
def test_memory_mode_flags_maps_modes(mode, expected):
    assert LlamaCppBackend._memory_mode_flags(mode) == expected


@pytest.mark.parametrize(
    "mode,expected",
    [
        (None, []),
        ("default", []),
        ("pinned", ["--load-mode", "mlock"]),
        ("resident", ["--load-mode", "none"]),
    ],
)
def test_memory_mode_flags_use_unified_load_mode(mode, expected):
    assert LlamaCppBackend._memory_mode_flags(mode, supports_load_mode = True) == expected


# ── _already_in_target_state ─────────────────────────────────────────────────


def _base_target_state_kwargs(backend):
    return {
        "model_identifier": "owner/repo",
        "hf_variant": "Q4_K_M",
        "n_ctx": 8192,
        "cache_type_kv": None,
        "speculative_type": None,
        "chat_template_override": None,
        "extra_args": None,
        "is_vision": False,
        "gpu_ids": backend.gpu_ids,
        "memory_mode": backend.memory_mode,
    }


def test_already_in_target_state_matches_same_memory_mode():
    backend = _loaded_backend(_requested_memory_mode = "resident")
    kwargs = _base_target_state_kwargs(backend)
    assert backend._already_in_target_state(**kwargs) is True


def test_already_in_target_state_rejects_different_memory_mode():
    backend = _loaded_backend(_requested_memory_mode = "resident")
    kwargs = _base_target_state_kwargs(backend)
    kwargs["memory_mode"] = "pinned"
    assert backend._already_in_target_state(**kwargs) is False


def test_already_in_target_state_keeps_device_extras_without_gpu_ids():
    # Without gpu_ids, --device is not stripped, so a genuine extras change still reloads.
    backend = _loaded_backend(_gpu_ids = None, _extra_args = ["--flash-attn", "on"])
    kwargs = _base_target_state_kwargs(backend)
    kwargs["gpu_ids"] = None
    kwargs["extra_args"] = ["--flash-attn", "on", "--device", "CUDA0"]
    assert backend._already_in_target_state(**kwargs) is False


def test_already_in_target_state_strips_device_extras_under_gpu_ids():
    # load_model stores device-stripped extras when gpu_ids owns placement, so a
    # duplicate /load carrying a user --device must strip the same way before the
    # dedupe compare, else it needlessly restarts an already-correct server (#7188).
    backend = _loaded_backend(_gpu_ids = [0, 1], _extra_args = ["--flash-attn", "on"])
    kwargs = _base_target_state_kwargs(backend)
    kwargs["gpu_ids"] = [0, 1]
    kwargs["extra_args"] = ["--flash-attn", "on", "--device", "CUDA0"]
    assert backend._already_in_target_state(**kwargs) is True


# GPU selection launch behavior. Route-level validation lives in
# test_gpu_selection.py; these tests cover the backend's emitted placement.


def _fit_fallback_backend(
    tmp_path,
    gpu_memory,
    *,
    vulkan = False,
):
    """Backend stubbed like the --fit-fallback test but with a configurable probe."""
    gguf = tmp_path / "model.gguf"
    _write_minimal_gguf(gguf)

    backend = LlamaCppBackend()
    backend._get_gpu_memory = lambda _binary = None: list(gpu_memory)
    backend._read_gguf_metadata = lambda _p: None
    backend._can_estimate_kv = lambda: False
    backend._get_gguf_size_bytes = lambda _p: 1024
    backend._mmproj_vram_bytes = lambda _p: 0
    backend._resolve_launch_mmproj_path = lambda **k: None
    backend._apu_ram_shortfall_message = lambda *a, **k: None
    backend._amd_apu_wants_unified_memory = lambda *a, **k: False
    backend._find_llama_server_binary = lambda include_denied = False: "/fake/llama-server"
    backend._is_vulkan_backend = lambda _binary = None: vulkan
    backend._get_gpu_free_memory = lambda _binary = None: list(gpu_memory)
    backend._wait_for_health = lambda timeout: True
    backend._detect_audio_type_strict = lambda: None
    backend._apply_detected_audio = lambda _d: True
    return backend, gguf


def test_empty_probe_preserves_explicit_gpu_ids(tmp_path):
    """Keep a route-validated CUDA/ROCm pin when telemetry is unavailable."""
    from utils.hardware import DeviceType

    backend, gguf = _fit_fallback_backend(tmp_path, gpu_memory = [])  # empty probe

    captured = {}

    def _make_fake_popen(cmd, **kwargs):
        if not cmd or str(cmd[0]) != "/fake/llama-server":
            return _REAL_POPEN(cmd, **kwargs)

        class _FakePopen:
            pid = 12345

            def __init__(self, cmd, **kwargs):
                captured["env"] = kwargs.get("env") or dict(os.environ)
                captured["cmd"] = list(cmd)

            def poll(self):
                return None

        return _FakePopen(cmd, **kwargs)

    with (
        patch.object(subprocess, "Popen", side_effect = _make_fake_popen),
        patch("utils.hardware.get_device", return_value = DeviceType.CUDA),
        patch("utils.hardware.get_parent_visible_gpu_ids", return_value = [0, 1]),
    ):
        assert backend.load_model(
            gguf_path = str(gguf),
            model_identifier = "test",
            gpu_ids = [1],
        )

    assert captured["env"]["CUDA_VISIBLE_DEVICES"] == "1"
    cmd = captured["cmd"]
    assert "--fit" in cmd and cmd[cmd.index("--fit") + 1] == "on"


def test_torchless_vulkan_populated_probe_uses_identity_ordinals(tmp_path):
    """Use Vulkan ordinals directly when torch has no visible GPUs."""
    backend, gguf = _fit_fallback_backend(
        tmp_path, gpu_memory = [(0, 10000, 16000), (1, 8000, 16000)], vulkan = True
    )
    backend._select_gpus = lambda *a, **k: ([1], False)

    captured = {}

    def _make_fake_popen(cmd, **kwargs):
        if not cmd or str(cmd[0]) != "/fake/llama-server":
            return _REAL_POPEN(cmd, **kwargs)

        class _FakePopen:
            pid = 321

            def __init__(self, cmd, **kwargs):
                captured["cmd"] = list(cmd)

            def poll(self):
                return None

        return _FakePopen(cmd, **kwargs)

    with (
        patch.object(subprocess, "Popen", side_effect = _make_fake_popen),
        patch("utils.hardware.get_parent_visible_gpu_ids", return_value = []),
    ):
        assert backend.load_model(
            gguf_path = str(gguf),
            model_identifier = "test",
            gpu_ids = [0, 1],
        )
    cmd = captured["cmd"]
    assert "--device" in cmd and cmd[cmd.index("--device") + 1] == "Vulkan1"
    assert backend.gpu_ids == [1]
    assert backend.requested_gpu_ids == [0, 1]


def test_vulkan_fit_keeps_discrete_device_selected(tmp_path):
    """Keep the discrete Vulkan device pinned when fit adds CPU offload."""
    backend, gguf = _fit_fallback_backend(
        tmp_path,
        gpu_memory = [(0, 30000, 0), (1, 14000, 16000)],
        vulkan = True,
    )
    backend._select_gpus = lambda *a, **k: (None, True)

    captured = {}

    def _make_fake_popen(cmd, **kwargs):
        if not cmd or str(cmd[0]) != "/fake/llama-server":
            return _REAL_POPEN(cmd, **kwargs)

        class _FakePopen:
            pid = 322

            def __init__(self, cmd, **kwargs):
                captured["cmd"] = list(cmd)

            def poll(self):
                return None

        return _FakePopen(cmd, **kwargs)

    with (
        patch.object(subprocess, "Popen", side_effect = _make_fake_popen),
        patch("utils.hardware.get_parent_visible_gpu_ids", return_value = []),
    ):
        assert backend.load_model(gguf_path = str(gguf), model_identifier = "test")

    cmd = captured["cmd"]
    assert "--fit" in cmd and cmd[cmd.index("--fit") + 1] == "on"
    assert "--device" in cmd and cmd[cmd.index("--device") + 1] == "Vulkan1"


def test_vulkan_gpu_ids_strips_conflicting_user_device(tmp_path):
    """Keep only Unsloth's Vulkan device pin while preserving unrelated extras."""
    backend, gguf = _fit_fallback_backend(
        tmp_path, gpu_memory = [(0, 10000, 16000), (1, 8000, 16000)], vulkan = True
    )
    backend._select_gpus = lambda *a, **k: ([0], False)

    captured = {}

    def _make_fake_popen(cmd, **kwargs):
        if not cmd or str(cmd[0]) != "/fake/llama-server":
            return _REAL_POPEN(cmd, **kwargs)

        class _FakePopen:
            pid = 999

            def __init__(self, cmd, **kwargs):
                captured["cmd"] = list(cmd)

            def poll(self):
                return None

        return _FakePopen(cmd, **kwargs)

    with (
        patch.object(subprocess, "Popen", side_effect = _make_fake_popen),
        patch("utils.hardware.get_parent_visible_gpu_ids", return_value = [0, 1]),
    ):
        assert backend.load_model(
            gguf_path = str(gguf),
            model_identifier = "test",
            gpu_ids = [0],
            extra_args = ["--device", "Vulkan1", "--top-k", "5"],
        )

    cmd = captured["cmd"]
    assert "Vulkan1" not in cmd
    device_idxs = [i for i, tok in enumerate(cmd) if tok == "--device"]
    assert len(device_idxs) == 1
    assert cmd[device_idxs[0] + 1] == "Vulkan0"
    assert "--top-k" in cmd and cmd[cmd.index("--top-k") + 1] == "5"


def test_gpu_ids_preserved_on_fit_fallback(tmp_path):
    """When _select_gpus falls back to --fit on, still pin CUDA_VISIBLE_DEVICES."""
    gguf = tmp_path / "model.gguf"
    _write_minimal_gguf(gguf)

    backend = LlamaCppBackend()
    backend._get_gpu_memory = lambda _binary = None: [
        (0, 10000, 16000),
        (1, 8000, 16000),
        (2, 6000, 16000),
    ]
    backend._read_gguf_metadata = lambda _p: None
    backend._can_estimate_kv = lambda: False
    backend._get_gguf_size_bytes = lambda _p: 1024
    backend._mmproj_vram_bytes = lambda _p: 0
    backend._resolve_launch_mmproj_path = lambda **k: None
    backend._apu_ram_shortfall_message = lambda *a, **k: None
    backend._amd_apu_wants_unified_memory = lambda *a, **k: False
    backend._find_llama_server_binary = lambda include_denied = False: "/fake/llama-server"
    backend._select_gpus = lambda *a, **k: (None, True)
    backend._wait_for_health = lambda timeout: True
    backend._detect_audio_type_strict = lambda: None
    backend._apply_detected_audio = lambda _d: True

    captured_envs = []

    def _make_fake_popen(cmd, **kwargs):
        if not cmd or str(cmd[0]) != "/fake/llama-server":
            return _REAL_POPEN(cmd, **kwargs)

        class _FakePopen:
            pid = 12345

            def __init__(self, cmd, **kwargs):
                captured_envs.append(kwargs.get("env") or dict(os.environ))

            def poll(self):
                return None

        return _FakePopen(cmd, **kwargs)

    with patch.object(subprocess, "Popen", side_effect = _make_fake_popen):
        backend.load_model(
            gguf_path = str(gguf),
            model_identifier = "test",
            gpu_ids = [1, 2],
        )

    assert captured_envs, "llama-server was not spawned"
    assert captured_envs[-1]["CUDA_VISIBLE_DEVICES"] == "1,2"


@pytest.mark.parametrize("gpu_ids,scrubbed", [([1, 2], True), (None, False)])
def test_gpu_ids_scrubs_inherited_llama_arg_device(tmp_path, monkeypatch, gpu_ids, scrubbed):
    """Scrub inherited LLAMA_ARG_DEVICE only when gpu_ids owns placement."""
    monkeypatch.setenv("LLAMA_ARG_DEVICE", "CUDA3")

    gguf = tmp_path / "model.gguf"
    _write_minimal_gguf(gguf)

    backend = LlamaCppBackend()
    backend._get_gpu_memory = lambda _binary = None: [
        (0, 10000, 16000),
        (1, 8000, 16000),
        (2, 6000, 16000),
    ]
    backend._read_gguf_metadata = lambda _p: None
    backend._can_estimate_kv = lambda: False
    backend._get_gguf_size_bytes = lambda _p: 1024
    backend._mmproj_vram_bytes = lambda _p: 0
    backend._resolve_launch_mmproj_path = lambda **k: None
    backend._apu_ram_shortfall_message = lambda *a, **k: None
    backend._amd_apu_wants_unified_memory = lambda *a, **k: False
    backend._find_llama_server_binary = lambda include_denied = False: "/fake/llama-server"
    backend._select_gpus = lambda *a, **k: ((list(gpu_ids) if gpu_ids else [0]), False)
    backend._wait_for_health = lambda timeout: True
    backend._detect_audio_type_strict = lambda: None
    backend._apply_detected_audio = lambda _d: True

    captured_envs = []

    def _make_fake_popen(cmd, **kwargs):
        if not cmd or str(cmd[0]) != "/fake/llama-server":
            return _REAL_POPEN(cmd, **kwargs)

        class _FakePopen:
            pid = 12345

            def __init__(self, cmd, **kwargs):
                captured_envs.append(kwargs.get("env") or dict(os.environ))

            def poll(self):
                return None

        return _FakePopen(cmd, **kwargs)

    with patch.object(subprocess, "Popen", side_effect = _make_fake_popen):
        backend.load_model(
            gguf_path = str(gguf),
            model_identifier = "test",
            gpu_ids = gpu_ids,
        )

    assert captured_envs, "llama-server was not spawned"
    assert ("LLAMA_ARG_DEVICE" not in captured_envs[-1]) == scrubbed


@pytest.mark.parametrize(
    "mode,user_flag,winning",
    [
        ("resident", "--mmap", "--no-mmap"),
        ("pinned", "--no-mmap", None),
        ("default", "--mlock", None),
    ],
)
def test_memory_mode_strips_conflicting_extra_args(tmp_path, mode, user_flag, winning):
    """Strip user memory flags when a first-class mode owns placement."""
    gguf = tmp_path / "model.gguf"
    _write_minimal_gguf(gguf)
    backend = _mem_env_backend(gguf)

    captured_cmds = []

    def _make_fake_popen(cmd, **kwargs):
        if not cmd or str(cmd[0]) != "/fake/llama-server":
            return _REAL_POPEN(cmd, **kwargs)

        class _FakePopen:
            pid = 12345

            def __init__(self, cmd, **kwargs):
                captured_cmds.append(list(cmd))

            def poll(self):
                return None

        return _FakePopen(cmd, **kwargs)

    with patch.object(subprocess, "Popen", side_effect = _make_fake_popen):
        backend.load_model(
            gguf_path = str(gguf),
            model_identifier = "test",
            memory_mode = mode,
            extra_args = [user_flag],
        )

    assert captured_cmds, "llama-server was not spawned"
    cmd = captured_cmds[-1]
    assert user_flag not in cmd
    mmap_flags = [a for a in cmd if a in ("--mmap", "--no-mmap")]
    if winning is None:
        assert "--mmap" not in cmd  # user --mmap/--no-mmap fully stripped
    else:
        assert mmap_flags[-1] == winning


def test_vulkan_gpu_ids_used_as_direct_ordinals_not_remapped(tmp_path):
    """Do not remap Vulkan ordinals through the CUDA/HIP visibility mask."""
    gguf = tmp_path / "model.gguf"
    _write_minimal_gguf(gguf)

    backend = LlamaCppBackend()
    backend._is_vulkan_backend = lambda _binary = None: True
    backend._get_gpu_memory = lambda _binary = None: [(0, 10000, 16000), (1, 9000, 16000)]
    backend._get_gpu_free_memory = lambda _binary = None: [(0, 10000), (1, 9000)]
    backend._read_gguf_metadata = lambda _p: None
    backend._can_estimate_kv = lambda: False
    backend._get_gguf_size_bytes = lambda _p: 1024
    backend._mmproj_vram_bytes = lambda _p: 0
    backend._resolve_launch_mmproj_path = lambda **k: None
    backend._apu_ram_shortfall_message = lambda *a, **k: None
    backend._amd_apu_wants_unified_memory = lambda *a, **k: False
    backend._find_llama_server_binary = lambda include_denied = False: "/fake/llama-server"
    backend._select_gpus = lambda requested_total, gpus, **k: ([gpus[0][0]], False)
    backend._wait_for_health = lambda timeout: True
    backend._detect_audio_type_strict = lambda: None
    backend._apply_detected_audio = lambda _d: True

    captured_cmds = []

    def _make_fake_popen(cmd, **kwargs):
        if not cmd or str(cmd[0]) != "/fake/llama-server":
            return _REAL_POPEN(cmd, **kwargs)

        class _FakePopen:
            pid = 12345

            def __init__(self, cmd, **kwargs):
                captured_cmds.append(list(cmd))

            def poll(self):
                return None

        return _FakePopen(cmd, **kwargs)

    with (
        patch.object(
            subprocess,
            "Popen",
            side_effect = _make_fake_popen,
        ),
        patch(
            "utils.hardware.get_parent_visible_gpu_ids",
            return_value = [2, 3],
        ),
    ):
        backend.load_model(
            gguf_path = str(gguf),
            model_identifier = "test",
            gpu_ids = [1],
        )

    assert captured_cmds, "llama-server was not spawned"
    cmd = captured_cmds[-1]
    assert "--device" in cmd
    assert cmd[cmd.index("--device") + 1] == "Vulkan1"


def test_backend_lacks_gpu_lib_detection(tmp_path):
    """Only classify a proven CPU-only split-library layout."""
    ext = "dll" if sys.platform == "win32" else "so"
    pre = "" if sys.platform == "win32" else "lib"

    def _lib_dir_with(*names):
        d = tmp_path / ("libs_" + ("_".join(names) or "empty"))
        d.mkdir()
        for nm in names:
            (d / f"{pre}ggml-{nm}.{ext}").write_bytes(b"x")
        return d

    binary = str(tmp_path / "llama-server")
    (tmp_path / "llama-server").write_bytes(b"x")

    with patch("core.inference.llama_cpp._llama_lib_dir", return_value = _lib_dir_with("cpu")):
        assert LlamaCppBackend._backend_lacks_gpu_lib(binary) is True
    for gpu in ("cuda", "hip", "vulkan"):
        with patch(
            "core.inference.llama_cpp._llama_lib_dir", return_value = _lib_dir_with("cpu", gpu)
        ):
            assert LlamaCppBackend._backend_lacks_gpu_lib(binary) is False
    with patch("core.inference.llama_cpp._llama_lib_dir", return_value = _lib_dir_with()):
        assert LlamaCppBackend._backend_lacks_gpu_lib(binary) is False

    for gpu in ("cuda", "hip", "vulkan"):
        d = tmp_path / f"libsv_cpu_{gpu}"
        d.mkdir()
        (d / f"{pre}ggml-cpu.{ext}").write_bytes(b"x")
        (d / f"{pre}ggml-{gpu}.{ext}.0").write_bytes(b"x")
        with patch("core.inference.llama_cpp._llama_lib_dir", return_value = d):
            assert LlamaCppBackend._backend_lacks_gpu_lib(binary) is False
    d = tmp_path / "libsv_cpu_only"
    d.mkdir()
    (d / f"{pre}ggml-cpu.{ext}.0").write_bytes(b"x")
    with patch("core.inference.llama_cpp._llama_lib_dir", return_value = d):
        assert LlamaCppBackend._backend_lacks_gpu_lib(binary) is True


def test_is_vulkan_backend_matches_versioned_soname(tmp_path):
    """Detect versioned Vulkan sonames and reject mixed-backend layouts."""
    ext = "dll" if sys.platform == "win32" else "so"
    pre = "" if sys.platform == "win32" else "lib"
    binary = str(tmp_path / "llama-server")
    (tmp_path / "llama-server").write_bytes(b"x")
    counter = {"n": 0}

    def _dir(*files):
        counter["n"] += 1
        d = tmp_path / f"vkdir_{counter['n']}"
        d.mkdir()
        for f in files:
            (d / f).write_bytes(b"x")
        return d

    with patch(
        "core.inference.llama_cpp._llama_lib_dir", return_value = _dir(f"{pre}ggml-vulkan.{ext}.0")
    ):
        assert LlamaCppBackend._is_vulkan_backend(binary) is True
    with patch(
        "core.inference.llama_cpp._llama_lib_dir", return_value = _dir(f"{pre}ggml-vulkan.{ext}")
    ):
        assert LlamaCppBackend._is_vulkan_backend(binary) is True
    for sib in (f"{pre}ggml-cuda.{ext}", f"{pre}ggml-hip.{ext}.0"):
        with patch(
            "core.inference.llama_cpp._llama_lib_dir",
            return_value = _dir(f"{pre}ggml-vulkan.{ext}.0", sib),
        ):
            assert LlamaCppBackend._is_vulkan_backend(binary) is False
    with patch(
        "core.inference.llama_cpp._llama_lib_dir", return_value = _dir(f"{pre}ggml-cpu.{ext}")
    ):
        assert LlamaCppBackend._is_vulkan_backend(binary) is False


def test_explicit_gpu_ids_strips_stored_device_extra_args(tmp_path):
    """Do not persist a user --device overridden by gpu_ids."""
    backend, gguf = _fit_fallback_backend(tmp_path, gpu_memory = [(0, 10000, 16000)], vulkan = True)
    backend._select_gpus = lambda requested_total, gpus, **k: ([gpus[0][0]], False)

    def _make_fake_popen(cmd, **kwargs):
        if not cmd or str(cmd[0]) != "/fake/llama-server":
            return _REAL_POPEN(cmd, **kwargs)

        class _FakePopen:
            pid = 12345

            def __init__(self, cmd, **kwargs):
                pass

            def poll(self):
                return None

        return _FakePopen(cmd, **kwargs)

    with (
        patch.object(subprocess, "Popen", side_effect = _make_fake_popen),
        patch("utils.hardware.get_parent_visible_gpu_ids", return_value = []),
    ):
        backend.load_model(
            gguf_path = str(gguf),
            model_identifier = "test",
            gpu_ids = [0],
            extra_args = ["--device", "Vulkan3", "--top-k", "5"],
        )

    stored = backend._extra_args or []
    assert "--device" not in stored and "Vulkan3" not in stored
    assert "--top-k" in stored  # unrelated extras are preserved


def test_memory_mode_default_matches_none_in_target_state():
    """An explicit default request should not reload a load that omitted the field."""
    backend = _loaded_backend()
    kwargs = _base_target_state_kwargs(backend)
    kwargs["memory_mode"] = "default"
    assert backend._already_in_target_state(**kwargs) is True


def test_explicit_default_matches_child_with_authoritative_mem_env(monkeypatch):
    """A live environment override makes the UI/API mode irrelevant."""
    monkeypatch.setenv("LLAMA_ARG_MLOCK", "1")
    backend = _loaded_backend(_launched_with_inherited_mem_env = True)
    kwargs = _base_target_state_kwargs(backend)
    kwargs["memory_mode"] = "default"
    assert backend._already_in_target_state(**kwargs) is True


def test_removed_mem_env_reloads_child_that_inherited_it():
    """Removing an operator override requires a reload to drop its effect."""
    backend = _loaded_backend(_launched_with_inherited_mem_env = True)
    kwargs = _base_target_state_kwargs(backend)
    kwargs["memory_mode"] = None
    assert backend._already_in_target_state(**kwargs) is False


def test_explicit_default_matches_child_without_mem_env():
    backend = _loaded_backend(_launched_with_inherited_mem_env = False)
    kwargs = _base_target_state_kwargs(backend)
    kwargs["memory_mode"] = "default"
    assert backend._already_in_target_state(**kwargs) is True


def test_new_mem_env_reloads_child_to_apply_operator_override(monkeypatch):
    monkeypatch.setenv("LLAMA_ARG_MLOCK", "1")
    backend = _loaded_backend(_launched_with_inherited_mem_env = False)
    kwargs = _base_target_state_kwargs(backend)
    kwargs["memory_mode"] = None
    assert backend._already_in_target_state(**kwargs) is False


def test_memory_mode_pinned_does_not_match_none():
    backend = _loaded_backend()
    kwargs = _base_target_state_kwargs(backend)
    kwargs["memory_mode"] = "pinned"
    assert backend._already_in_target_state(**kwargs) is False


def test_load_response_and_status_round_trip_placement_fields():
    """Placement fields round-trip through load and status schemas."""
    from models.inference import InferenceStatusResponse, LoadResponse

    load_resp = LoadResponse(
        status = "loaded",
        model = "m",
        display_name = "m",
        is_gguf = True,
        inference = {},
        gpu_ids = [0, 1],
        host_memory_mode = "resident",
    )
    assert load_resp.gpu_ids == [0, 1]
    assert load_resp.host_memory_mode == "resident"

    status_resp = InferenceStatusResponse(
        is_gguf = True,
        gpu_ids = [0, 1],
        host_memory_mode = "pinned",
    )
    assert status_resp.gpu_ids == [0, 1]
    assert status_resp.host_memory_mode == "pinned"


@pytest.mark.parametrize(
    "mode,expected_requested,expected_canonical",
    [
        ("default", "default", None),
        ("DEFAULT", "default", None),
        ("pinned", "pinned", "pinned"),
        (None, None, None),
    ],
)
def test_requested_memory_mode_preserves_explicit_default(
    tmp_path, mode, expected_requested, expected_canonical
):
    """Preserve explicit default for status while canonicalizing it to None."""
    gguf = tmp_path / "model.gguf"
    _write_minimal_gguf(gguf)
    backend = _mem_env_backend(gguf)

    with patch.object(subprocess, "Popen"):
        assert backend.load_model(gguf_path = str(gguf), model_identifier = "t", memory_mode = mode)
    assert backend.requested_memory_mode == expected_requested
    assert backend.memory_mode == expected_canonical


# ── LLAMA_ARG_* host-memory env is authoritative ─────────────────────────────


def _mem_env_backend(gguf):
    backend = LlamaCppBackend()
    backend._get_gpu_memory = lambda _binary = None: [(0, 10000, 16000)]
    backend._read_gguf_metadata = lambda _p: None
    backend._can_estimate_kv = lambda: False
    backend._get_gguf_size_bytes = lambda _p: 1024
    backend._mmproj_vram_bytes = lambda _p: 0
    backend._resolve_launch_mmproj_path = lambda **k: None
    backend._apu_ram_shortfall_message = lambda *a, **k: None
    backend._amd_apu_wants_unified_memory = lambda *a, **k: False
    backend._find_llama_server_binary = lambda include_denied = False: "/fake/llama-server"
    backend._select_gpus = lambda *a, **k: ([0], False)
    backend._wait_for_health = lambda timeout: True
    backend._detect_audio_type_strict = lambda: None
    backend._apply_detected_audio = lambda _d: True
    return backend


@pytest.mark.parametrize("mode", ["default", "pinned", "resident", None])
def test_memory_mode_env_overrides_every_request(tmp_path, monkeypatch, mode):
    monkeypatch.setenv("LLAMA_ARG_MLOCK", "1")
    monkeypatch.setenv("LLAMA_ARG_NO_MMAP", "1")
    monkeypatch.setenv("LLAMA_ARG_MMAP", "true")
    monkeypatch.setenv("LLAMA_ARG_LOAD_MODE", "dio")
    monkeypatch.setenv("LLAMA_ARG_DIO", "1")

    gguf = tmp_path / "model.gguf"
    _write_minimal_gguf(gguf)
    backend = _mem_env_backend(gguf)

    captured_envs = []
    captured_cmds = []

    def _make_fake_popen(cmd, **kwargs):
        if not cmd or str(cmd[0]) != "/fake/llama-server":
            return _REAL_POPEN(cmd, **kwargs)

        class _FakePopen:
            pid = 12345

            def __init__(self, cmd, **kwargs):
                captured_cmds.append(list(cmd))
                captured_envs.append(kwargs.get("env") or {})

            def poll(self):
                return None

        return _FakePopen(cmd, **kwargs)

    with patch.object(subprocess, "Popen", side_effect = _make_fake_popen):
        backend.load_model(
            gguf_path = str(gguf),
            model_identifier = "test",
            memory_mode = mode,
            extra_args = ["--load-mode", "none", "--top-k", "20"],
        )

    assert captured_envs, "llama-server was not spawned"
    env = captured_envs[-1]
    for var in (
        "LLAMA_ARG_LOAD_MODE",
        "LLAMA_ARG_MLOCK",
        "LLAMA_ARG_NO_MMAP",
        "LLAMA_ARG_MMAP",
        "LLAMA_ARG_DIO",
    ):
        assert var in env
    cmd = captured_cmds[-1]
    assert "--load-mode" not in cmd
    assert "--mlock" not in cmd
    assert "--mmap" not in cmd
    assert "--no-mmap" not in cmd
    assert "--top-k" in cmd


# ── diffusion GGUF placement ─────────────────────────────────────────────────


def test_remote_diffusion_load_rejects_vulkan_ordinal_after_download(tmp_path):
    """Reject a Vulkan ordinal when a remote model proves to be diffusion."""
    gguf = tmp_path / "renamed.gguf"
    _write_minimal_gguf(gguf, arch = "diffusion-gemma")

    backend = LlamaCppBackend()
    backend._find_llama_server_binary = lambda include_denied = False: "/fake/llama-server"
    backend._is_vulkan_backend = lambda _binary = None: True
    backend._get_gpu_memory = lambda _binary = None: [(1, 8 * 1024**3, 8 * 1024**3)]
    backend._download_gguf = lambda **_kwargs: str(gguf)
    backend._read_gguf_metadata = lambda _path: setattr(backend, "_is_diffusion", True)
    backend._start_diffusion_server = lambda **_kwargs: pytest.fail(
        "Vulkan ordinal reached the CUDA diffusion runner"
    )

    with pytest.raises(ValueError, match = "no defined mapping"):
        backend.load_model(
            hf_repo = "renamed/model",
            hf_variant = "Q4_K_M",
            model_identifier = "renamed/model",
            speculative_type = "off",
            gpu_ids = [1],
        )


def test_confirmed_diffusion_allows_physical_gpu_id_on_vulkan_build(tmp_path):
    """Keep diffusion pins in CUDA physical-ID space on Vulkan builds."""
    gguf = tmp_path / "diffusion.gguf"
    _write_minimal_gguf(gguf, arch = "diffusion-gemma")

    backend = LlamaCppBackend()
    backend._find_llama_server_binary = lambda include_denied = False: "/fake/llama-server"
    backend._is_vulkan_backend = lambda _binary = None: True
    backend._get_gpu_memory = lambda _binary: [(0, 8 * 1024**3, 8 * 1024**3)]
    backend._read_gguf_metadata = lambda _path: setattr(backend, "_is_diffusion", True)
    captured = {}
    backend._start_diffusion_server = lambda **kwargs: captured.update(kwargs) or True

    assert backend.load_model(
        gguf_path = str(gguf),
        model_identifier = "diffusion/model",
        speculative_type = "off",
        gpu_ids = [1],
        gpu_ids_are_vulkan_ordinals = False,
    )
    assert captured["gpu_ids"] == [1]


def test_remote_diffusion_rejects_explicit_memory_mode_before_teardown(tmp_path):
    gguf = tmp_path / "renamed.gguf"
    _write_minimal_gguf(gguf, arch = "diffusion-gemma")

    backend = LlamaCppBackend()
    backend._find_llama_server_binary = lambda include_denied = False: "/fake/llama-server"
    backend._is_vulkan_backend = lambda _binary = None: False
    backend._download_gguf = lambda **_kwargs: str(gguf)
    backend._read_gguf_metadata = lambda _path: setattr(backend, "_is_diffusion", True)
    backend._start_diffusion_server = lambda **_kwargs: pytest.fail(
        "unsupported memory mode reached the diffusion runner"
    )

    with (
        patch.object(
            backend,
            "_kill_process",
            side_effect = AssertionError("invalid placement tore down the live model"),
        ),
        pytest.raises(ValueError, match = "host-memory modes are not supported"),
    ):
        backend.load_model(
            hf_repo = "renamed/model",
            hf_variant = "Q4_K_M",
            model_identifier = "renamed/model",
            speculative_type = "off",
            memory_mode = "resident",
        )


def test_remote_normal_cpu_only_pin_rejects_before_teardown(tmp_path):
    gguf = tmp_path / "renamed.gguf"
    _write_minimal_gguf(gguf, arch = "llama")

    backend = LlamaCppBackend()
    backend._find_llama_server_binary = lambda include_denied = False: "/fake/llama-server"
    backend._is_vulkan_backend = lambda _binary = None: False
    backend._backend_lacks_gpu_lib = lambda _binary = None: True
    backend._download_gguf = lambda **_kwargs: str(gguf)

    with (
        patch.object(
            backend,
            "_kill_process",
            side_effect = AssertionError("invalid CPU-only pin tore down the live model"),
        ),
        pytest.raises(ValueError, match = "CPU-only build"),
    ):
        backend.load_model(
            hf_repo = "renamed/model",
            hf_variant = "Q4_K_M",
            model_identifier = "renamed/model",
            speculative_type = "off",
            gpu_ids = [0],
            gpu_ids_are_vulkan_ordinals = False,
        )


def test_remote_diffusion_cpu_only_pin_reaches_runner(tmp_path):
    gguf = tmp_path / "renamed.gguf"
    _write_minimal_gguf(gguf, arch = "diffusion-gemma")

    backend = LlamaCppBackend()
    backend._find_llama_server_binary = lambda include_denied = False: "/fake/llama-server"
    backend._is_vulkan_backend = lambda _binary = None: False
    backend._backend_lacks_gpu_lib = lambda _binary = None: True
    backend._download_gguf = lambda **_kwargs: str(gguf)
    backend._read_gguf_metadata = lambda _path: setattr(backend, "_is_diffusion", True)
    captured = {}
    backend._start_diffusion_server = lambda **kwargs: captured.update(kwargs) or True

    assert backend.load_model(
        hf_repo = "renamed/model",
        hf_variant = "Q4_K_M",
        model_identifier = "renamed/model",
        speculative_type = "off",
        gpu_ids = [1],
        gpu_ids_are_vulkan_ordinals = False,
    )
    assert captured["gpu_ids"] == [1]


@pytest.mark.parametrize("mode", [None, "default", "DEFAULT", ""])
def test_diffusion_load_clears_stale_memory_mode(tmp_path, mode):
    """A diffusion load clears stale llama-server memory-mode state."""
    gguf = tmp_path / "diffusion.gguf"
    _write_minimal_gguf(gguf, arch = "diffusion-gemma")

    backend = LlamaCppBackend()
    backend._read_gguf_metadata = lambda _p: setattr(backend, "_is_diffusion", True)
    backend._find_llama_server_binary = lambda include_denied = False: "/fake/llama-server"
    backend._is_vulkan_backend = lambda _binary = None: False
    backend._start_diffusion_server = lambda **kw: True

    backend._requested_memory_mode = "resident"
    backend._launched_with_inherited_mem_env = True

    assert (
        backend.load_model(
            gguf_path = str(gguf),
            model_identifier = "d",
            memory_mode = mode,
        )
        is True
    )
    assert backend.memory_mode is None
    assert backend._requested_memory_mode is None
    assert backend._launched_with_inherited_mem_env is False


def test_local_chat_gguf_in_diffusion_path_not_prekilled(tmp_path):
    """A diffusion-like path cannot override a normal local GGUF header."""
    backend, gguf = _fit_fallback_backend(tmp_path, gpu_memory = [(0, 10000, 16000)])
    backend._select_gpus = lambda *a, **k: ([0], False)

    with patch.object(subprocess, "Popen"):
        assert (
            backend.load_model(
                gguf_path = str(gguf),
                model_identifier = "/models/diffusion/chat.gguf",
                gpu_ids = [0],
            )
            is True
        )
