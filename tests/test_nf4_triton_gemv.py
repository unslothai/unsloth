# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Batch-1 NF4 GEMV (unsloth/kernels/nf4_gemv.py) on a real GPU.

The Triton kernel reduces each quantization block before scaling it, so it is held to an fp32
reference with a tolerance; the bitsandbytes wrapper must stay byte-identical to the ctypes fallback (main's fast_gemv).
"""

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("needs a CUDA GPU", allow_module_level = True)
pytest.importorskip("triton")
F = pytest.importorskip("bitsandbytes.functional")

from unsloth.kernels.nf4_gemv import gemv_nf4, gemv_nf4_bnb, triton_gemv_supported

SHAPES = [(4096, 4096), (14336, 4096), (4096, 14336), (1024, 4096), (128256, 4096)]
DTYPES = [torch.float16, torch.bfloat16]


def _quant(n, k, dtype, nested = True, seed = 0):
    g = torch.Generator(device = "cuda").manual_seed(seed)
    W = torch.randn(n, k, dtype = dtype, device = "cuda", generator = g)
    return F.quantize_4bit(W, quant_type = "nf4", compress_statistics = nested)


def _args(q, s):
    if s.nested:
        return (
            q, s.absmax, s.state2.code, s.state2.absmax, s.offset, s.code,
            s.blocksize, s.state2.blocksize, s.shape, s.dtype,
        )
    return (q, s.absmax, None, None, None, s.code, s.blocksize, None, s.shape, s.dtype)


def _reference(X, q, s):
    n, k = s.shape
    return (X.float().view(1, k) @ F.dequantize_4bit(q, s).float().t()).view(1, 1, n)


def _rel_err(a, ref):
    return ((a.float() - ref).abs().max() / ref.abs().max()).item()


# bf16 output rounding alone is ~4e-3 relative; bitsandbytes lands at the same level.
TOL = {torch.float16: 2e-3, torch.bfloat16: 1.2e-2}


@pytest.mark.parametrize("nested", [True, False])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_triton_gemv_matches_fp32_reference(shape, dtype, nested):
    n, k = shape
    q, s = _quant(n, k, dtype, nested)
    X = torch.randn(1, 1, k, dtype = dtype, device = "cuda")
    ref = _reference(X, q, s)
    out = gemv_nf4(X, *_args(q, s))
    assert out.shape == (1, 1, n) and out.dtype == dtype
    assert _rel_err(out, ref) < TOL[dtype]
    # Never worse than bitsandbytes' own kernel by more than a rounding step.
    assert _rel_err(out, ref) <= _rel_err(gemv_nf4_bnb(X, *_args(q, s)), ref) + 4e-3


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES[:2])
def test_bnb_wrapper_is_byte_identical_to_the_ctypes_gemv(shape, dtype):
    from unsloth.kernels.utils import _fast_gemv_ctypes

    n, k = shape
    q, s = _quant(n, k, dtype)
    X = torch.randn(1, 1, k, dtype = dtype, device = "cuda")
    assert torch.equal(gemv_nf4_bnb(X, *_args(q, s)), _fast_gemv_ctypes(X, q, s))


def test_explicit_out_is_written_and_returned():
    q, s = _quant(1024, 4096, torch.bfloat16)
    X = torch.randn(1, 1, 4096, dtype = torch.bfloat16, device = "cuda")
    out = torch.empty(1, 1, 1024, dtype = torch.bfloat16, device = "cuda")
    got = gemv_nf4(X, *_args(q, s), out = out)
    assert got.data_ptr() == out.data_ptr()
    assert torch.equal(out, gemv_nf4(X, *_args(q, s)))


@pytest.mark.parametrize("impl", [gemv_nf4, gemv_nf4_bnb])
def test_side_stream_without_sync(impl):
    """Inputs produced on a side stream and consumed there, with no host sync in between."""
    q, s = _quant(4096, 4096, torch.bfloat16)

    def produce():
        # A slow producer: if the GEMV ran on another stream it would read X half written.
        X = torch.full((1, 1, 4096), 0.25, dtype = torch.bfloat16, device = "cuda")
        for _ in range(5):
            X = X + 0.05
        return X

    expected = impl(produce(), *_args(q, s))
    torch.cuda.synchronize()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    results = []
    with torch.cuda.stream(side):
        for _ in range(20):
            results.append(impl(produce(), *_args(q, s)))
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    for r in results:
        assert torch.equal(r, expected)


@pytest.mark.parametrize("impl", [gemv_nf4, gemv_nf4_bnb])
def test_cuda_graph_capture_and_replay(impl):
    q, s = _quant(4096, 4096, torch.bfloat16)
    A = _args(q, s)
    static_x = torch.randn(1, 1, 4096, dtype = torch.bfloat16, device = "cuda")
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            impl(static_x, *A)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_out = impl(static_x, *A)
    for seed in range(3):
        new_x = torch.randn(1, 1, 4096, dtype = torch.bfloat16, device = "cuda",
                            generator = torch.Generator(device = "cuda").manual_seed(seed))
        static_x.copy_(new_x)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(static_out, impl(new_x, *A))


@pytest.mark.parametrize("impl", [gemv_nf4, gemv_nf4_bnb])
def test_compiles_fullgraph_with_no_graph_breaks(impl):
    import torch._dynamo

    q, s = _quant(1024, 4096, torch.bfloat16)
    A = _args(q, s)
    X = torch.randn(1, 1, 4096, dtype = torch.bfloat16, device = "cuda")

    def step(x):
        return impl(x * 2.0, *A) + 1.0

    torch._dynamo.reset()
    explained = torch._dynamo.explain(step)(X)
    assert explained.graph_break_count == 0, explained.break_reasons
    compiled = torch.compile(step, fullgraph = True)
    assert torch.equal(compiled(X), step(X))


def test_support_predicate():
    assert triton_gemv_supported(4096, 64)
    assert not triton_gemv_supported(4100, 64)
    assert not triton_gemv_supported(4096, 48)
