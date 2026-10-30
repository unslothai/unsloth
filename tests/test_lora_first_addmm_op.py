# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Traced fused LoRA order runs as one opaque op (unsloth::lora_first_addmm): same bits as the eager sequence,
no graph break, and the kill switch keeps the old order."""

import pytest
import torch
import unsloth  # noqa: F401
from unsloth.kernels import utils as KU

pytestmark = pytest.mark.skipif(not KU._HAS_LORA_FIRST_OP, reason = "needs torch.library.custom_op")


def _mk(
    M = 64,
    K = 96,
    N = 80,
    r = 8,
    dtype = torch.bfloat16,
):
    g = torch.Generator().manual_seed(0)
    X = torch.randn(M, K, generator = g).to(dtype)
    W = (torch.randn(N, K, generator = g) * 0.02).to(dtype)
    A = (torch.randn(r, K, generator = g) * 0.1).to(dtype)
    B = (torch.randn(N, r, generator = g) * 2e-3).to(dtype)
    return X, W, A, B


def test_op_matches_eager_sequence_bitwise():
    X, W, A, B = _mk()
    XA = X @ A.t()
    ref = torch.addmm(XA.new_zeros(1), XA, B.t(), beta = 0, alpha = 0.5)
    ref.addmm_(X, W.t())
    got = torch.ops.unsloth.lora_first_addmm(X, W.t(), XA, B.t(), 0.5)
    assert torch.equal(got, ref)


def test_opcheck():
    X, W, A, B = _mk(dtype = torch.float32)
    XA = X @ A.t()
    torch.library.opcheck(
        torch.ops.unsloth.lora_first_addmm.default,
        (X, W.t(), XA, B.t(), 0.5),
        test_utils = ("test_schema", "test_faketensor", "test_aot_dispatch_static"),
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_compiled_matmul_lora_uses_op_fullgraph(dtype):
    X, W, A, B = _mk(dtype = dtype)
    eager = KU.matmul_lora(X, W, None, A, B, 0.5)
    import torch._dynamo

    torch._dynamo.reset()
    seen = []
    orig = torch.ops.unsloth.lora_first_addmm

    def backend(gm, inputs):
        seen.extend(str(n.target) for n in gm.graph.nodes if n.op == "call_function")
        return gm.forward

    f = torch.compile(
        lambda X, W, A, B: KU.matmul_lora(X, W, None, A, B, 0.5), backend = backend, fullgraph = True
    )
    got = f(X, W, A, B)
    assert any("lora_first_addmm" in s for s in seen), seen
    assert torch.equal(got, eager)
    f2 = torch.compile(
        lambda X, W, A, B: KU.matmul_lora(X, W, None, A, B, 0.5),
        backend = "aot_eager",
        fullgraph = True,
    )
    assert torch.equal(f2(X, W, A, B), eager)


def test_kill_switch_keeps_old_order_compiled(monkeypatch):
    X, W, A, B = _mk()
    monkeypatch.setattr(KU, "LORA_FUSED_PASSES", False)
    seen = []

    def backend(gm, inputs):
        seen.extend(str(n.target) for n in gm.graph.nodes if n.op == "call_function")
        return gm.forward

    import torch._dynamo

    torch._dynamo.reset()
    torch.compile(
        lambda X, W, A, B: KU.matmul_lora(X, W, None, A, B, 0.5), backend = backend, fullgraph = True
    )(X, W, A, B)
    assert not any("lora_first_addmm" in s for s in seen)


def test_without_the_op_traced_falls_back_to_the_functional_order(monkeypatch):
    # Registration failed (or an older torch.library): the traced fused branch keeps the two functional addmms.
    X, W, A, B = _mk(dtype = torch.float32)
    eager = KU.matmul_lora(X, W, None, A, B, 0.5)
    monkeypatch.setattr(KU, "_HAS_LORA_FIRST_OP", False)
    seen = []

    def backend(gm, inputs):
        seen.extend(str(n.target) for n in gm.graph.nodes if n.op == "call_function")
        return gm.forward

    torch._dynamo.reset()
    got = torch.compile(
        lambda X, W, A, B: KU.matmul_lora(X, W, None, A, B, 0.5), backend = backend, fullgraph = True
    )(X, W, A, B)
    assert not any("lora_first_addmm" in s for s in seen)
    torch.testing.assert_close(got, eager, rtol = 1e-6, atol = 1e-6)
