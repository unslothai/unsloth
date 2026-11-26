# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import types

import pytest
import torch

from unsloth.kernels import int4_packed


def _tinygemm_weight(N = 8, K = 128):
    qs = types.SimpleNamespace(
        layout = "tinygemm",
        shape = (N, K),
        group_size = 128,
        zero_point = None,
        scale = torch.ones(N, 1, dtype = torch.bfloat16),
        dtype = torch.bfloat16,
        _fast = None,
    )
    packed = torch.zeros(N * K // 8, dtype = torch.int32)
    return packed, qs


def _probe(monkeypatch, error):
    packed, qs = _tinygemm_weight()
    monkeypatch.setattr(int4_packed, "_FAST_CHECKED", {})
    monkeypatch.setattr(
        int4_packed, "int4_dequantize", lambda p, q, dtype: torch.zeros(q.shape, dtype = dtype)
    )

    def fail(call, x2):
        raise error

    monkeypatch.setattr(int4_packed, "_fast_call", fail)
    key = (qs.layout, packed.device, False, qs.dtype)
    return packed, qs, key


def test_oom_during_probe_propagates_and_is_not_cached(monkeypatch):
    packed, qs, key = _probe(monkeypatch, torch.cuda.OutOfMemoryError("CUDA out of memory"))
    with pytest.raises(torch.cuda.OutOfMemoryError):
        int4_packed._fast_args(packed, qs)
    assert key not in int4_packed._FAST_CHECKED
    assert qs._fast is None


def test_unknown_signature_still_caches_the_slow_path(monkeypatch):
    packed, qs, key = _probe(monkeypatch, RuntimeError("unknown op signature"))
    assert int4_packed._fast_args(packed, qs) is None
    assert int4_packed._FAST_CHECKED[key] is False
