# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The armed FLUX.1 GELU route (``convrot_gelu_act_quant``) traced whole by Dynamo: its row gate picks the with /
without attention prefix threshold, which torch 2.10 and 2.11 could not trace as a tuple indexed by a bool. The op
handle is a CPU stand-in, the layout check passes, so the gate is the only code under test."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))


@pytest.fixture()
def armed(monkeypatch):
    torch = pytest.importorskip("torch")
    from core.inference import diffusion_convrot_quant as cq

    calls = []

    def handle(prefix, gate, group, v2):
        calls.append(prefix is None)
        return gate * 2, gate.sum(-1)

    monkeypatch.setattr(cq, "_GELU_HANDLE", handle)
    monkeypatch.setattr(cq, "producer_layout_ok", lambda *a: True)
    monkeypatch.setitem(cq._GELU_DEVICE, None, True)
    # with the prefix from 8 rows, without it from 32: the index decides which rows engage
    monkeypatch.setitem(cq._GELU_DEVICE_MIN_ROWS, None, (8, 32))
    yield torch, cq, calls


def _route(cq):
    def f(prefix, gate):
        out = cq.convrot_gelu_act_quant(prefix, gate, 256, True)
        return gate + 1 if out is None else out[0]

    return f


@pytest.mark.parametrize("fullgraph", [True, False])
def test_the_armed_gate_traces_in_one_graph(armed, fullgraph):
    torch, cq, calls = armed
    from torch._dynamo.testing import CompileCounter

    torch._dynamo.reset()
    counter = CompileCounter()
    fn = torch.compile(_route(cq), backend = counter, fullgraph = fullgraph)
    gate, prefix = torch.randn(16, 512), torch.randn(16, 64)
    assert torch.equal(fn(prefix, gate), gate * 2)  # 16 rows >= 8: fused
    assert torch.equal(fn(None, gate), gate + 1)  # 16 rows < 32 without the prefix: stock
    assert counter.frame_count == 2  # one graph per prefix layout, no graph break
    assert calls == [False]
    torch._dynamo.reset()


def test_the_gate_reads_the_threshold_for_its_layout(armed):
    torch, cq, calls = armed
    gate = torch.randn(40, 512)
    assert cq.convrot_gelu_act_quant(None, gate, 256, True) is not None  # 40 >= 32
    assert cq.convrot_gelu_act_quant(None, gate[:16], 256, True) is None  # 16 < 32
    assert (
        cq.convrot_gelu_act_quant(torch.randn(16, 64), gate[:16], 256, True) is not None
    )  # 16 >= 8
    assert calls == [True, False]
