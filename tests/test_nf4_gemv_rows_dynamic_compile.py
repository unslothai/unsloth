# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Compiled multi-row NF4 GEMV with a dynamic decode batch. The rows-per-program count is a Triton constexpr
picked from the batch size; under torch.compile with a symbolic batch that pinned the graph to one batch size,
so mark_dynamic raised ConstraintViolationError and dynamic=True compiled a graph per batch size. A symbolic
batch now launches one row per program (the batch stays a runtime argument); a static one keeps the
dequantize-once tile."""

import pytest
import torch

pytestmark = pytest.mark.gpu

if not torch.cuda.is_available():
    pytest.skip("needs a CUDA GPU", allow_module_level = True)
pytest.importorskip("triton")
F = pytest.importorskip("bitsandbytes.functional")

from unsloth.kernels.nf4_gemv import gemv_nf4_rows

N, K = 1024, 2048


def _weights(dtype):
    g = torch.Generator(device = "cuda").manual_seed(0)
    W = torch.randn(N, K, dtype = dtype, device = "cuda", generator = g)
    q, s = F.quantize_4bit(W, quant_type = "nf4", compress_statistics = True)
    args = (
        q,
        s.absmax,
        s.state2.code,
        s.state2.absmax,
        s.offset,
        s.code,
        s.blocksize,
        s.state2.blocksize,
        s.shape,
    )
    return args, F.dequantize_4bit(q, s).float()


def _fn(args):
    def fn(X):
        out = torch.empty((X.shape[0], 1, N), dtype = X.dtype, device = X.device)
        return gemv_nf4_rows(X, *args, out)

    return fn


def _check(y, X, W):
    ref = X.float().view(-1, K) @ W.t()
    err = (y.float().view(-1, N) - ref).abs().max() / ref.abs().max()
    assert err.item() < 1.2e-2, err.item()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids = str)
def test_marked_dynamic_batch_compiles_once(dtype):
    from torch._dynamo.testing import CompileCounterWithBackend

    torch._dynamo.reset()
    args, W = _weights(dtype)
    counter = CompileCounterWithBackend("aot_eager")
    compiled = torch.compile(_fn(args), backend = counter, fullgraph = True)
    for m in (2, 3, 4, 2):
        X = torch.randn(m, 1, K, dtype = dtype, device = "cuda")
        torch._dynamo.mark_dynamic(X, 0, min = 2, max = 4)
        _check(compiled(X), X, W)
    assert counter.frame_count == 1, counter.frame_count


def test_dynamic_true_reuses_one_graph_across_batch_sizes():
    from torch._dynamo.testing import CompileCounterWithBackend

    torch._dynamo.reset()
    args, W = _weights(torch.bfloat16)
    counter = CompileCounterWithBackend("aot_eager")
    compiled = torch.compile(_fn(args), backend = counter, dynamic = True, fullgraph = True)
    for m in (2, 3, 4, 3, 2):
        X = torch.randn(m, 1, K, dtype = torch.bfloat16, device = "cuda")
        _check(compiled(X), X, W)
    assert counter.frame_count == 1, counter.frame_count


def test_static_batch_keeps_its_rows_per_program():
    # A static batch is already one graph per batch size: it keeps the dequantize-once tile, matching eager.
    torch._dynamo.reset()
    args, W = _weights(torch.bfloat16)
    fn = _fn(args)
    compiled = torch.compile(fn, backend = "aot_eager", dynamic = False, fullgraph = True)
    X = torch.randn(3, 1, K, dtype = torch.bfloat16, device = "cuda")
    y = compiled(X)
    _check(y, X, W)
    torch.testing.assert_close(y, fn(X), rtol = 0, atol = 0)
