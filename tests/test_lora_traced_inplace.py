# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""Traced LoRA backward and forward without copies: the accumulating addmm_ runs in place under
torch.compile (unsloth::addmm_), and the SwiGLU backward is plain torch math there, so Inductor
neither copies the accumulator nor the saved activations. Both must match eager."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("needs a CUDA GPU", allow_module_level = True)
pytest.importorskip("triton")

from unsloth.kernels import utils
from unsloth.kernels.swiglu import swiglu_DWf_DW_dfg_kernel


def _inputs(seed = 0):
    g = torch.Generator(device = "cuda").manual_seed(seed)
    d = dict(device = "cuda", dtype = torch.bfloat16, generator = g)
    return (
        torch.randn(64, 256, **d),
        torch.randn(512, 256, **d) / 16,
        torch.randn(8, 256, **d) / 16,
        torch.randn(512, 8, **d) / 16,
    )


def _lora_forward(X, W, A, B):
    out = X @ W.t()
    return utils.addmm_(out, X @ A.t(), B.t(), alpha = 2.0)


@pytest.mark.skipif(not utils._HAS_ADDMM_OP, reason = "torch.library.custom_op needs torch 2.4+")
def test_traced_addmm_is_the_in_place_op_and_matches_eager():
    torch._dynamo.reset()
    graphs = []

    def backend(gm, example_inputs):
        graphs.append(gm)
        return gm

    X, W, A, B = _inputs()
    with torch.no_grad():
        compiled = torch.compile(_lora_forward, backend = backend, fullgraph = True)(X, W, A, B)
        inductor = torch.compile(_lora_forward, fullgraph = True)(X, W, A, B)
        eager = _lora_forward(X, W, A, B)
    targets = [node.target for gm in graphs for node in gm.graph.nodes]
    assert torch.ops.unsloth.addmm_.default in targets or torch.ops.unsloth.addmm_ in targets
    assert torch.equal(compiled, eager) and torch.equal(inductor, eager)


def test_addmm_keeps_autograd_with_grad_enabled():
    # The custom op has no derivative; with grad enabled the plain addmm_ must be used.
    torch._dynamo.reset()
    X, W, A, B = _inputs(1)
    W.requires_grad_(True)
    out = torch.compile(_lora_forward, fullgraph = True)(X, W, A, B)
    out.float().sum().backward()
    assert W.grad is not None and torch.isfinite(W.grad).all()


# bf16 only where Unsloth trains in it (compute capability 8+); T4 trains in fp16.
DTYPES = [torch.float16] + ([torch.bfloat16] if torch.cuda.get_device_capability()[0] >= 8 else [])


@pytest.mark.parametrize("dtype", DTYPES, ids = str)
def test_traced_swiglu_backward_matches_the_kernel(dtype):
    torch._dynamo.reset()
    g = torch.Generator(device = "cuda").manual_seed(2)
    d = dict(device = "cuda", dtype = dtype, generator = g)
    DW, e, gate = (
        torch.randn(256, 1024, **d),
        torch.randn(256, 1024, **d),
        torch.randn(256, 1024, **d),
    )
    ref = swiglu_DWf_DW_dfg_kernel(DW.clone(), e.clone(), gate.clone())
    traced = torch.compile(swiglu_DWf_DW_dfg_kernel, fullgraph = True)(DW, e, gate)
    for name, a, b in zip(("h", "df", "de"), traced, ref):
        # Inductor keeps the intermediate products in fp32 where the kernel rounds them to dtype.
        err = ((a.float() - b.float()).abs().max() / b.float().abs().max()).item()
        assert err < 1e-2, (name, err)
