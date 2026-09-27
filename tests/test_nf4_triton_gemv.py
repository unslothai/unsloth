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

"""Batch-1 NF4 GEMV kernel (unsloth/kernels/nf4_gemv.py). It reduces each quantization block
before scaling it, so it is held to an fp32 reference with a tolerance and must never be worse
than bitsandbytes' own GEMV. Streams, CUDA graphs and torch.compile through fast_gemv are covered
in test_bnb_integration_compile.py."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("needs a CUDA GPU", allow_module_level = True)
pytest.importorskip("triton")
F = pytest.importorskip("bitsandbytes.functional")

from unsloth.kernels.nf4_gemv import gemv_nf4
from unsloth.kernels.utils import _fast_gemv_ctypes

SHAPES = [(4096, 4096), (14336, 4096), (4096, 14336), (1024, 4096)]
DTYPES = [torch.float16, torch.bfloat16]


def _quant(
    n,
    k,
    dtype,
    nested = True,
    seed = 0,
):
    g = torch.Generator(device = "cuda").manual_seed(seed)
    W = torch.randn(n, k, dtype = dtype, device = "cuda", generator = g)
    return F.quantize_4bit(W, quant_type = "nf4", compress_statistics = nested)


def _args(q, s):
    if s.nested:
        return (
            q,
            s.absmax,
            s.state2.code,
            s.state2.absmax,
            s.offset,
            s.code,
            s.blocksize,
            s.state2.blocksize,
            s.shape,
            s.dtype,
        )
    return (q, s.absmax, None, None, None, s.code, s.blocksize, None, s.shape, s.dtype)


def _reference(X, q, s):
    n, k = s.shape
    return (X.float().view(1, k) @ F.dequantize_4bit(q, s).float().t()).view(1, 1, n)


def _rel_err(a, ref):
    return ((a.float() - ref).abs().max() / ref.abs().max()).item()


# bf16 output rounding alone is about 4e-3 relative; bitsandbytes lands at the same level.
TOL = {torch.float16: 2e-3, torch.bfloat16: 1.2e-2}


@pytest.mark.parametrize("nested", [True, False], ids = ["nested", "flat"])
@pytest.mark.parametrize("dtype", DTYPES, ids = str)
@pytest.mark.parametrize("shape", SHAPES, ids = lambda s: f"{s[0]}x{s[1]}")
def test_matches_fp32_reference_and_bitsandbytes(shape, dtype, nested):
    n, k = shape
    q, s = _quant(n, k, dtype, nested)
    X = torch.randn(1, 1, k, dtype = dtype, device = "cuda")
    ref = _reference(X, q, s)
    out = gemv_nf4(X, *_args(q, s))
    assert out.shape == (1, 1, n) and out.dtype == dtype
    assert _rel_err(out, ref) < TOL[dtype]
    if nested:  # the ctypes GEMV only takes double quantized states
        assert _rel_err(out, ref) <= _rel_err(_fast_gemv_ctypes(X, q, s), ref) + 4e-3


def test_vocab_sized_weight():
    q, s = _quant(128256, 4096, torch.bfloat16)
    X = torch.randn(1, 1, 4096, dtype = torch.bfloat16, device = "cuda")
    assert _rel_err(gemv_nf4(X, *_args(q, s)), _reference(X, q, s)) < TOL[torch.bfloat16]


def test_explicit_out_is_written_and_returned():
    q, s = _quant(1024, 4096, torch.bfloat16)
    X = torch.randn(1, 1, 4096, dtype = torch.bfloat16, device = "cuda")
    out = torch.empty(1, 1, 1024, dtype = torch.bfloat16, device = "cuda")
    assert gemv_nf4(X, *_args(q, s), out = out).data_ptr() == out.data_ptr()
    assert torch.equal(out, gemv_nf4(X, *_args(q, s)))
