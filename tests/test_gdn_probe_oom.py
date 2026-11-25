# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""An OOM during the GatedDeltaNet op self-check must not be cached as "op broken" for the process."""

import sys
import types

import pytest
import torch

from unsloth.models import _block_compile, _gdn_op


def _fake_fla(monkeypatch):
    chunk = types.ModuleType("fla.ops.gated_delta_rule.chunk")
    for name in (
        "chunk_gated_delta_rule_fwd",
        "chunk_gated_delta_rule_bwd",
        "l2norm_fwd",
        "l2norm_bwd",
    ):
        setattr(chunk, name, lambda *a, **k: None)
    for name in ("fla", "fla.ops", "fla.ops.gated_delta_rule"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "fla.ops.gated_delta_rule.chunk", chunk)


@pytest.mark.parametrize(
    "error, cached",
    [(torch.OutOfMemoryError("CUDA out of memory"), None), (RuntimeError("bad"), False)],
)
def test_fla_ops_self_check(monkeypatch, error, cached):
    _fake_fla(monkeypatch)
    monkeypatch.setattr(_gdn_op, "_FLA", None)

    def fail(*a):
        raise error

    monkeypatch.setattr(_gdn_op, "_self_check", fail)
    make_inputs = lambda: [torch.zeros(1)]
    if cached is None:
        with pytest.raises(torch.OutOfMemoryError):
            _gdn_op._fla_ops(make_inputs)
    else:
        assert _gdn_op._fla_ops(make_inputs) is False
    assert _gdn_op._FLA is cached


@pytest.mark.parametrize(
    "error, cached",
    [(torch.OutOfMemoryError("CUDA out of memory"), None), (RuntimeError("bad"), False)],
)
def test_prepare_gdn(monkeypatch, error, cached):
    monkeypatch.setattr(_block_compile, "_GDN_OP", [None])
    monkeypatch.setattr(_block_compile, "_route_sm_count", lambda: None)

    def fail(make_inputs):
        raise error

    monkeypatch.setattr(_gdn_op, "_fla_ops", fail)
    gdn = types.SimpleNamespace(num_v_heads = 2, head_k_dim = 4, head_v_dim = 4, num_k_heads = 2)
    layer = types.SimpleNamespace(linear_attn = gdn)
    hidden = torch.zeros(1, 3, 8)
    if cached is None:
        with pytest.raises(torch.OutOfMemoryError):
            _block_compile._prepare_gdn(layer, hidden)
    else:
        _block_compile._prepare_gdn(layer, hidden)
    assert _block_compile._GDN_OP[0] is cached
