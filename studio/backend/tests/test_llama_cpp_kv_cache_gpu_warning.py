# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the KV-cache GPU-compat warning (#6272).

q4_1/q5_0/q5_1 KV cache types have no CUDA dequant kernel in mainline
llama.cpp, so a GPU-offloaded launch requesting one silently falls back to
the CPU reference path for KV ops, which can dominate decode latency.
``_kv_cache_gpu_fallback_warning`` hedges that with a warning at launch
time, using the ``gpus`` list already probed by ``load_model`` (no new
GPU probe). Pins: warns only for the risky types, only when a GPU is
present, and composes correctly against the real ``gpus``-shaped value
used at the call site.
"""

from __future__ import annotations

import sys
import types as _types
from pathlib import Path

import pytest

# Stub heavy / unavailable deps before importing the module under test.
# Same pattern as test_llama_cpp_max_context_threshold.py.

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

# loggers
_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)

# structlog
_structlog_stub = _types.ModuleType("structlog")
sys.modules.setdefault("structlog", _structlog_stub)

# httpx
_httpx_stub = _types.ModuleType("httpx")
for _exc_name in (
    "ConnectError",
    "TimeoutException",
    "ReadTimeout",
    "ReadError",
    "RemoteProtocolError",
    "CloseError",
):
    setattr(_httpx_stub, _exc_name, type(_exc_name, (Exception,), {}))


class _FakeTimeout:
    def __init__(self, *a, **kw):
        pass


_httpx_stub.Timeout = _FakeTimeout
_httpx_stub.Client = type(
    "Client",
    (),
    {
        "__init__": lambda self, **kw: None,
        "__enter__": lambda self: self,
        "__exit__": lambda self, *a: None,
    },
)
sys.modules.setdefault("httpx", _httpx_stub)

from core.inference.llama_cpp import (
    _GPU_UNACCELERATED_CACHE_TYPES,
    _kv_cache_gpu_fallback_warning,
)


class TestWarnsForRiskyTypeWithGpu:
    """Every GPU-unaccelerated cache type should warn when a GPU is present."""

    @pytest.mark.parametrize("cache_type", sorted(_GPU_UNACCELERATED_CACHE_TYPES))
    def test_warns(self, cache_type):
        msg = _kv_cache_gpu_fallback_warning(cache_type, True)
        assert msg is not None
        assert cache_type in msg
        assert "q8_0" in msg

    def test_case_insensitive(self):
        assert _kv_cache_gpu_fallback_warning("Q5_1", True) is not None


class TestSilentForSafeTypeWithGpu:
    """GPU-accelerated cache types must never warn, even with a GPU present."""

    @pytest.mark.parametrize("cache_type", ["f16", "bf16", "f32", "q8_0", "q4_0", "iq4_nl"])
    def test_silent(self, cache_type):
        assert _kv_cache_gpu_fallback_warning(cache_type, True) is None

    def test_silent_for_none(self):
        assert _kv_cache_gpu_fallback_warning(None, True) is None


class TestSilentForRiskyTypeWithoutGpu:
    """A CPU-only launch is on the CPU path regardless of cache type, so no
    GPU-fallback warning applies."""

    @pytest.mark.parametrize("cache_type", sorted(_GPU_UNACCELERATED_CACHE_TYPES))
    def test_silent(self, cache_type):
        assert _kv_cache_gpu_fallback_warning(cache_type, False) is None


class TestLogsViaExistingGpusVar:
    """The launch call site passes ``bool(gpus)`` where ``gpus`` is the list
    of ``(gpu_index, free_mib, total_mib)`` / ``(gpu_index, free_mib)``
    tuples already probed by ``load_model`` -- no new GPU probe. Exercise
    that exact conversion against realistic shapes."""

    def test_nonempty_gpus_list_warns(self):
        gpus = [(0, 24_000, 24_000)]
        assert _kv_cache_gpu_fallback_warning("q5_0", bool(gpus)) is not None

    def test_empty_gpus_list_is_silent(self):
        gpus: list[tuple[int, int, int]] = []
        assert _kv_cache_gpu_fallback_warning("q5_0", bool(gpus)) is None

    def test_multi_gpu_list_warns(self):
        gpus = [(0, 24_000), (1, 24_000)]
        assert _kv_cache_gpu_fallback_warning("q4_1", bool(gpus)) is not None
