# SPDX-License-Identifier: AGPL-3.0-only
"""Production-path contract tests for the GGUF KV-cache advisory."""

from __future__ import annotations

import inspect
import struct
import subprocess
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest

# Keep this focused module importable in the lightweight backend test runner.
_loggers = types.ModuleType("loggers")
_loggers.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers)
_structlog = types.ModuleType("structlog")
_structlog.get_logger = lambda *args, **kwargs: __import__("logging").getLogger("stub")
sys.modules.setdefault("structlog", _structlog)

from core.inference import llama_cpp


RISKY = ("q4_1", "q5_0", "q5_1", "iq4_nl")
SAFE = ("f16", "bf16", "q8_0", "q4_0", "f32", None)
BACKENDS = ("cuda", "hip")


@pytest.mark.parametrize("cache_type", RISKY)
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "mode,layers",
    [("auto", -1), ("manual", -1), ("manual", 1)],
)
def test_production_warning_path_warns(cache_type, backend, mode, layers):
    assert llama_cpp._kv_cache_gpu_fallback_warning(cache_type, mode, layers, backend) is not None


@pytest.mark.parametrize("cache_type", SAFE)
@pytest.mark.parametrize("backend", ("cuda", "hip", "vulkan", "metal", "cpu", "unknown", None))
@pytest.mark.parametrize(
    "mode,layers",
    [("auto", -1), ("manual", -1), ("manual", 0), ("manual", 1)],
)
def test_safe_types_and_non_cuda_hip_backends_are_silent(cache_type, backend, mode, layers):
    assert llama_cpp._kv_cache_gpu_fallback_warning(cache_type, mode, layers, backend) is None


@pytest.mark.parametrize("backend", ("cuda", "hip"))
def test_manual_zero_and_cpu_fallback_are_silent(backend):
    assert llama_cpp._kv_cache_gpu_fallback_warning("q4_1", "manual", 0, backend) is None
    assert llama_cpp._kv_cache_gpu_fallback_warning("q4_1", "manual", 0, None) is None


def test_load_model_uses_scalar_intent_and_preserves_command_path():
    source = inspect.getsource(llama_cpp.LlamaCppBackend.load_model)
    assert "_kv_cache_gpu_fallback_warning(" in source
    warning_start = source.index("gpu_fallback_warning = _kv_cache_gpu_fallback_warning(")
    warning_block = source[warning_start : warning_start + 650]
    assert "advisory_cache_type" in warning_block
    assert "_extras_cache is None" in source
    assert "not _cache_type_from_env" in source
    assert "\n                    cache_type_kv," not in warning_block
    assert "intent.gpu_memory_mode" in source
    assert "intent.gpu_layers" in source
    assert "intent.gpu_ids" not in source[source.find("_kv_cache_gpu_fallback_warning") :]
    assert '"--cache-type-k"' in source
    assert '"--cache-type-v"' in source


def test_warning_does_not_read_per_axis_extra_arguments_or_environment():
    source = inspect.getsource(llama_cpp._kv_cache_gpu_fallback_warning)
    assert "extra_args" not in source
    assert "environ" not in source


def _write_minimal_gguf(path: Path) -> str:
    key = b"general.architecture"
    value = b"llama"
    metadata = struct.pack("<Q", len(key)) + key
    metadata += struct.pack("<I", 8) + struct.pack("<Q", len(value)) + value
    path.write_bytes(struct.pack("<IIQQ", 0x46554747, 3, 0, 1) + metadata)
    return str(path)


@pytest.mark.parametrize(
    "extra_args,env_cache,expected_warning",
    [
        (None, None, ("q4_1", "auto", -1, "cuda")),
        (["--cache-type-k", "q8_0"], None, (None, "auto", -1, "cuda")),
        (None, "f32", None),
    ],
)
def test_production_load_path_applies_advisory_authority(
    monkeypatch, tmp_path, extra_args, env_cache, expected_warning
):
    backend = llama_cpp.LlamaCppBackend()
    gguf = _write_minimal_gguf(tmp_path / "model.gguf")
    backend._get_gpu_memory = lambda _binary = None, **_kw: [(0, 10_000, 16_000)]
    backend._get_gpu_free_memory = lambda _binary = None, **_kw: [(0, 10_000)]
    backend._read_gguf_metadata = lambda _path: None
    backend._can_estimate_kv = lambda: False
    backend._get_gguf_size_bytes = lambda _path: 1024
    backend._mmproj_vram_bytes = lambda _path: 0
    backend._resolve_launch_mmproj_path = lambda **_kwargs: None
    backend._apu_ram_shortfall_message = lambda *args, **kwargs: None
    backend._launch_host_shortfall_message = lambda *args, **kwargs: None
    backend._amd_apu_wants_unified_memory = lambda *args, **kwargs: False
    backend._find_llama_server_binary = lambda **_kwargs: "/fake/llama-server"
    backend._is_vulkan_backend = lambda _binary = None: False
    backend._installed_ggml_backends = lambda _binary = None: frozenset({"cuda"})
    backend._wait_for_health = lambda timeout: True
    backend._detect_audio_type_strict = lambda: None
    backend._apply_detected_audio = lambda _detected: True
    if env_cache is None:
        monkeypatch.delenv("LLAMA_ARG_CACHE_TYPE_K", raising = False)
    else:
        monkeypatch.setenv("LLAMA_ARG_CACHE_TYPE_K", env_cache)

    warnings = []
    monkeypatch.setattr(
        llama_cpp,
        "_kv_cache_gpu_fallback_warning",
        lambda *args: warnings.append(args) or None,
    )

    real_popen = subprocess.Popen

    def fake_popen(cmd, **kwargs):
        if not cmd or str(cmd[0]) != "/fake/llama-server":
            return real_popen(cmd, **kwargs)
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
        assert (
            backend.load_model(
                llama_cpp.GgufLoadIntent(
                    gguf_path = gguf,
                    model_identifier = "test",
                    cache_type_kv = None if env_cache is not None else "q4_1",
                    extra_args = extra_args,
                )
            )
            is True
        )

    if expected_warning is None:
        assert warnings == []
    else:
        assert warnings == [expected_warning]
