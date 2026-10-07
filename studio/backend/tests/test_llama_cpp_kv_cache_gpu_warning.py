# SPDX-License-Identifier: AGPL-3.0-only
"""Production-path contract tests for the GGUF KV-cache advisory."""

from __future__ import annotations

import struct
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from core.inference import llama_cpp


# q4_1 / q5_0 / q5_1 run on the GPU since ggml-org/llama.cpp#28079.
SAFE = ("f16", "bf16", "q8_0", "q4_0", "q4_1", "q5_0", "q5_1", "f32", None)
GPU = (frozenset({"cpu", "cuda"}), frozenset({"cpu", "hip"}))
NO_GPU = (frozenset(), frozenset({"cpu"}), frozenset({"cpu", "vulkan"}))


def _warning(cache_type, mode, layers, backends):
    return llama_cpp._kv_cache_gpu_fallback_warning(cache_type, mode, layers, lambda: backends)


@pytest.mark.parametrize("backends", GPU)
@pytest.mark.parametrize("mode,layers", [("auto", -1), ("manual", -1), ("manual", 1)])
def test_iq4_nl_warns_on_gpu_placement(backends, mode, layers):
    assert "iq4_nl" in _warning("iq4_nl", mode, layers, backends)


@pytest.mark.parametrize("cache_type", SAFE)
@pytest.mark.parametrize("mode,layers", [("auto", -1), ("manual", 0), ("manual", 1)])
def test_other_types_are_silent_and_skip_the_probe(cache_type, mode, layers):
    def probe():
        raise AssertionError("backend probe ran for a type that cannot warn")

    assert llama_cpp._kv_cache_gpu_fallback_warning(cache_type, mode, layers, probe) is None


@pytest.mark.parametrize("backends", NO_GPU)
def test_iq4_nl_is_silent_without_cuda_or_hip(backends):
    assert _warning("iq4_nl", "auto", -1, backends) is None


@pytest.mark.parametrize("backends", GPU)
def test_manual_zero_layers_is_silent(backends):
    assert _warning("iq4_nl", "manual", 0, backends) is None


def _write_minimal_gguf(path: Path) -> str:
    key = b"general.architecture"
    value = b"llama"
    metadata = struct.pack("<Q", len(key)) + key
    metadata += struct.pack("<I", 8) + struct.pack("<Q", len(value)) + value
    path.write_bytes(struct.pack("<IIQQ", 0x46554747, 3, 0, 1) + metadata)
    return str(path)


@pytest.mark.parametrize(
    "cache_type,extra_args,env_cache,expected_warning",
    [
        ("iq4_nl", None, None, ("iq4_nl", "auto", -1, True)),
        ("IQ4_NL", None, None, ("iq4_nl", "auto", -1, True)),
        ("q8_0", None, None, ("q8_0", "auto", -1, False)),
        ("iq4_nl", ["--cache-type-k", "q8_0"], None, None),
        (None, None, "f32", None),
    ],
)
def test_load_path_warns_only_for_its_own_cache_type(
    monkeypatch, tmp_path, cache_type, extra_args, env_cache, expected_warning
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
    backend._wait_for_health = lambda *_a, **_kw: True
    backend._detect_audio_type_strict = lambda: None
    backend._apply_detected_audio = lambda _detected: True
    if env_cache is None:
        monkeypatch.delenv("LLAMA_ARG_CACHE_TYPE_K", raising = False)
    else:
        monkeypatch.setenv("LLAMA_ARG_CACHE_TYPE_K", env_cache)

    warnings = []
    real_warning = llama_cpp._kv_cache_gpu_fallback_warning

    def record(cache_type_kv, mode, layers, installed_backends):
        result = real_warning(cache_type_kv, mode, layers, installed_backends)
        warnings.append((cache_type_kv, mode, layers, result is not None))
        return result

    monkeypatch.setattr(llama_cpp, "_kv_cache_gpu_fallback_warning", record)

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
                    cache_type_kv = cache_type,
                    extra_args = extra_args,
                )
            )
            is True
        )

    if expected_warning is None:
        assert warnings == []
    else:
        assert warnings == [expected_warning]
