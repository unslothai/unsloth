# SPDX-License-Identifier: AGPL-3.0-only
"""Production-path contract tests for the GGUF KV-cache advisory."""

from __future__ import annotations

import inspect
import sys
import types

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
    assert (
        llama_cpp._kv_cache_gpu_fallback_warning(cache_type, mode, layers, backend)
        is not None
    )


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
    warning_block = source[warning_start : warning_start + 260]
    assert "intent.cache_type_kv" in warning_block
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
