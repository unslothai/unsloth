# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Deterministic compute-graph buffer estimate (``_estimate_compute_buffer_bytes``).

llama.cpp reserves a per-device compute buffer dominated by the vocab-width
output buffer plus a small activation scratch. It is context-independent and
scales with ``--parallel`` (concurrent serving slots), not with how the model is
split across GPUs. In tensor mode the buffer is materialized on every device.

These tests pin the estimate's shape and that it is a SAFE upper bound on the
allocations measured on real hardware (Qwen3.6-27B-MTP Q6_K, b9625):

    single-GPU serving slots  parallel 1/2/4/8 -> ~36 / 492 / 1388 / 3220 MiB
    tensor 2-GPU parallel 1   -> ~600 MiB/device  (vs the old flat 5120 reserve)

No GPU, subprocess, or GGUF I/O. Cross-platform.
"""

from __future__ import annotations

import sys
import types as _types
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)
_structlog_stub = _types.ModuleType("structlog")
_structlog_stub.get_logger = lambda *a, **k: __import__("logging").getLogger("stub")
sys.modules.setdefault("structlog", _structlog_stub)
_httpx_stub = _types.ModuleType("httpx")
for _exc in (
    "ConnectError",
    "TimeoutException",
    "ReadTimeout",
    "ReadError",
    "RemoteProtocolError",
    "CloseError",
):
    setattr(_httpx_stub, _exc, type(_exc, (Exception,), {}))
_httpx_stub.Timeout = type("T", (), {"__init__": lambda s, *a, **k: None})
_httpx_stub.Client = type(
    "C",
    (),
    {"__init__": lambda s, **kw: None, "__enter__": lambda s: s, "__exit__": lambda s, *a: None},
)
sys.modules.setdefault("httpx", _httpx_stub)

from core.inference.llama_cpp import LlamaCppBackend

MIB = 1024 * 1024


def _backend(vocab=248320, embd=5120):
    """Backend with just the dims the compute-buffer estimate reads."""
    b = LlamaCppBackend.__new__(LlamaCppBackend)
    b._vocab_size = vocab
    b._embedding_length = embd
    return b


# Measured ground truth (MiB) the estimate must upper-bound.
_PIPELINE_MEASURED = {1: 36, 2: 492, 4: 1388, 8: 3220}
_TENSOR_MEASURED_PER_DEVICE = 600


class TestSafeUpperBound:
    """The estimate must be >= every measured allocation (never under-reserve)."""

    @pytest.mark.parametrize("parallel,measured", sorted(_PIPELINE_MEASURED.items()))
    def test_pipeline_upper_bounds_measured(self, parallel, measured):
        est = _backend()._estimate_compute_buffer_bytes(n_parallel=parallel) / MIB
        assert est >= measured, f"under-reserved at parallel={parallel}: {est:.0f} < {measured}"

    @pytest.mark.parametrize("parallel,measured", sorted(_PIPELINE_MEASURED.items()))
    def test_pipeline_not_wildly_over(self, parallel, measured):
        # Stay within ~2x of measured so we don't waste context (the point of
        # replacing the flat reserve). parallel=1 is tiny in absolute terms.
        est = _backend()._estimate_compute_buffer_bytes(n_parallel=parallel) / MIB
        assert est <= max(measured * 2.0, 128)

    def test_tensor_upper_bounds_measured(self):
        est = _backend()._estimate_compute_buffer_bytes(
            n_parallel=1, per_device_tensor=True
        ) / MIB
        assert est >= _TENSOR_MEASURED_PER_DEVICE

    def test_tensor_far_below_old_flat_reserve(self):
        # The whole point: deterministic estimate << flat 5120 for this model.
        est = _backend()._estimate_compute_buffer_bytes(
            n_parallel=1, per_device_tensor=True
        ) / MIB
        assert est < LlamaCppBackend._TENSOR_PARALLEL_BUFFER_RESERVE_MIB


class TestScaling:
    def test_grows_with_serving_slots(self):
        b = _backend()
        vals = [b._estimate_compute_buffer_bytes(n_parallel=p) for p in (1, 2, 4, 8)]
        assert vals == sorted(vals) and vals[0] < vals[-1]

    def test_parallel_1_is_small(self):
        # Single-token decode: a few tens of MiB, not gigabytes.
        est = _backend()._estimate_compute_buffer_bytes(n_parallel=1) / MIB
        assert est < 128

    def test_tensor_exceeds_pipeline_at_same_parallel(self):
        b = _backend()
        pipe = b._estimate_compute_buffer_bytes(n_parallel=1)
        tens = b._estimate_compute_buffer_bytes(n_parallel=1, per_device_tensor=True)
        assert tens > pipe

    def test_scales_with_vocab(self):
        small = _backend(vocab=32000)._estimate_compute_buffer_bytes(n_parallel=4)
        big = _backend(vocab=256000)._estimate_compute_buffer_bytes(n_parallel=4)
        assert big > small

    def test_scales_with_ubatch(self):
        b = _backend()
        lo = b._estimate_compute_buffer_bytes(n_parallel=4, n_ubatch=256)
        hi = b._estimate_compute_buffer_bytes(n_parallel=4, n_ubatch=1024)
        assert hi > lo


class TestFallback:
    def test_zero_when_vocab_missing(self):
        assert _backend(vocab=None)._estimate_compute_buffer_bytes(n_parallel=4) == 0

    def test_zero_when_embd_missing(self):
        assert _backend(embd=None)._estimate_compute_buffer_bytes(n_parallel=4) == 0

    def test_zero_lets_tensor_plan_use_flat_fallback(self):
        # When dims are missing, _plan_tensor_parallel must fall back to the flat
        # reserve (defense-in-depth) rather than reserving 0 and OOMing.
        b = _backend(vocab=None, embd=None)
        b._n_layers = None  # can't estimate KV -> floors ctx, still returns a plan
        ec, mac, gi, ts = b._plan_tensor_parallel([(0, 48000), (1, 48000)], 8 * 1024**3, 8192)
        assert gi == [0, 1]  # both GPUs usable under the flat fallback


class TestParallel1Default:
    """At Studio's default --parallel 1 the buffer is negligible in pipeline."""

    def test_default_n_parallel(self):
        est = _backend()._estimate_compute_buffer_bytes() / MIB
        assert est < 128
