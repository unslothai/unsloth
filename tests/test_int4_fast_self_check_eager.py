# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The int4 fused path's self-check (Generator draw + .item()) runs eagerly at load, never inside a trace or a
CUDA graph capture; until it has run, a traced / captured call takes the exact path and caches nothing."""

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


@pytest.fixture
def probe(monkeypatch):
    calls = []
    monkeypatch.setattr(int4_packed, "_FAST_CHECKED", {})
    monkeypatch.setattr(
        int4_packed, "int4_dequantize", lambda p, q, dtype: torch.zeros(q.shape, dtype = dtype)
    )

    def fast_call(call, x2):
        calls.append(x2.shape)
        return torch.zeros(x2.shape[0], 8, dtype = x2.dtype)

    monkeypatch.setattr(int4_packed, "_fast_call", fast_call)
    return calls


def test_no_self_check_while_compiling(monkeypatch, probe):
    packed, qs = _tinygemm_weight()
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    assert int4_packed._fast_args(packed, qs) is None
    assert probe == [] and int4_packed._FAST_CHECKED == {}
    # Nothing cached: the next eager call still runs the check and adopts the fused path.
    assert qs._fast is None


def test_no_self_check_while_capturing(monkeypatch, probe):
    packed, qs = _tinygemm_weight()
    monkeypatch.setattr(int4_packed, "_self_check_unsafe", lambda device: True)
    assert int4_packed._fast_args(packed, qs) is None
    assert probe == [] and qs._fast is None


def test_eager_self_check_runs_once_then_traced_calls_use_it(monkeypatch, probe):
    packed, qs = _tinygemm_weight()
    assert int4_packed.int4_fast_self_check(packed, qs) is True
    assert len(probe) == 1
    key = (qs.layout, packed.device, False, qs.dtype)
    assert int4_packed._FAST_CHECKED[key] is True
    # Another layer of the same kind: the cached verdict, no second probe.
    packed2, qs2 = _tinygemm_weight()
    assert int4_packed.int4_fast_self_check(packed2, qs2) is True
    assert len(probe) == 1
    # Once checked, a traced call builds the fused call without probing again.
    monkeypatch.setattr(int4_packed, "_self_check_unsafe", lambda device: True)
    assert int4_packed._fast_args(packed2, qs2) is not None
    assert len(probe) == 1


def test_finalize_runs_the_self_check_at_load(monkeypatch):
    from unsloth.models import compressed_tensors_int4 as ct

    seen = []
    monkeypatch.setattr(int4_packed, "int4_repack_", lambda packed, qs: None)
    monkeypatch.setattr(
        int4_packed, "int4_fast_self_check", lambda packed, qs: seen.append(qs.layout) or True
    )

    class FakeLinear(ct.Int4PackedLinear):
        def __init__(self):
            torch.nn.Module.__init__(self)

    module = FakeLinear()
    packed = torch.nn.Parameter(torch.zeros(4, dtype = torch.int32), requires_grad = False)
    module._parameters["weight_packed"] = packed
    qs = types.SimpleNamespace(layout = "marlin")
    monkeypatch.setattr(ct.Int4PackedLinear, "quant_state", property(lambda self: qs))
    model = torch.nn.Sequential(module)
    assert ct.finalize_int4_packed_linears(model) == 1
    assert seen == ["marlin"]
