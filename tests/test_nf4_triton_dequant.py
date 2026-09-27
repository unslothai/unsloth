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

"""The fused Triton NF4 dequant must be byte-identical to bitsandbytes and to Unsloth's
ctypes fast_dequantize, on any stream, under CUDA-graph replay and under torch.compile."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("needs a CUDA device", allow_module_level = True)

bnb_functional = pytest.importorskip("bitsandbytes.functional")

from unsloth.kernels import nf4 as nf4_mod
from unsloth.kernels.nf4 import dequantize_nf4

_INT_VIEW = {torch.float16: torch.int16, torch.bfloat16: torch.int16, torch.float32: torch.int32}


def _bytes_equal(a, b):
    assert a.shape == b.shape and a.dtype == b.dtype
    iv = _INT_VIEW[a.dtype]
    return torch.equal(a.contiguous().view(iv), b.contiguous().view(iv))


def _quantize(shape, dtype, blocksize, nested, seed = 0):
    g = torch.Generator(device = "cuda").manual_seed(seed)
    W = torch.randn(shape, dtype = dtype, device = "cuda", generator = g)
    return bnb_functional.quantize_4bit(
        W, blocksize = blocksize, quant_type = "nf4", compress_statistics = nested
    )


def _ours(q, s, out = None):
    if s.nested:
        return dequantize_nf4(
            q,
            s.absmax,
            s.state2.code,
            s.state2.absmax,
            s.offset,
            s.blocksize,
            s.state2.blocksize,
            s.shape,
            s.dtype,
            out = out,
        )
    return dequantize_nf4(q, s.absmax, None, None, None, s.blocksize, 0, s.shape, s.dtype, out = out)


def _free_gib():
    free, _ = torch.cuda.mem_get_info()
    return free / 2**30


SHAPES = [(1, 4096), (4096, 1), (17, 33), (4096, 4096), (14336, 4096), (4096, 14336), (128256, 4096)]


@pytest.mark.parametrize("shape", SHAPES, ids = lambda s: f"{s[0]}x{s[1]}")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32], ids = str)
@pytest.mark.parametrize("blocksize", [64, 128])
@pytest.mark.parametrize("nested", [True, False], ids = ["nested", "flat"])
def test_bytes_match_bitsandbytes(shape, dtype, blocksize, nested):
    if shape[0] * shape[1] >= 128256 * 4096 and _free_gib() < 20:
        pytest.skip("not enough free GPU memory for the vocab-sized weight")
    q, s = _quantize(shape, dtype, blocksize, nested)
    assert _bytes_equal(_ours(q, s), bnb_functional.dequantize_4bit(q, s))


@pytest.mark.parametrize("shape", [(17, 33), (4096, 4096), (14336, 4096), (1, 4096)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids = str)
def test_bytes_match_unsloth_fast_dequantize(shape, dtype):
    from unsloth.kernels.utils import fast_dequantize

    q, s = _quantize(shape, dtype, 64, True)
    ref = fast_dequantize(q, s)
    ours = _ours(q, s)
    # fast_dequantize returns a transposed view for 1-row packed weights; compare the data.
    if ref.shape != ours.shape:
        ref = ref.t()
    assert _bytes_equal(ours, ref)


def test_explicit_out_is_written_and_returned():
    q, s = _quantize((256, 512), torch.bfloat16, 64, True)
    out = torch.full(s.shape, 7.0, dtype = s.dtype, device = "cuda")
    res = _ours(q, s, out = out)
    assert res.data_ptr() == out.data_ptr()
    assert _bytes_equal(out, bnb_functional.dequantize_4bit(q, s))


def test_float_offset_is_accepted():
    q, s = _quantize((128, 256), torch.bfloat16, 64, True)
    res = dequantize_nf4(
        q,
        s.absmax,
        s.state2.code,
        s.state2.absmax,
        float(s.offset),
        s.blocksize,
        s.state2.blocksize,
        s.shape,
        s.dtype,
    )
    assert _bytes_equal(res, bnb_functional.dequantize_4bit(q, s))


def test_rejects_unsupported_dtype_and_blocksize():
    q, s = _quantize((64, 64), torch.bfloat16, 64, True)
    with pytest.raises(TypeError):
        dequantize_nf4(q, s.absmax, s.state2.code, s.state2.absmax, s.offset, 64, 256, s.shape, torch.int8)
    with pytest.raises(ValueError):
        dequantize_nf4(q, s.absmax, s.state2.code, s.state2.absmax, s.offset, 48, 256, s.shape, s.dtype)


def test_fma_contraction_would_be_caught():
    """Control: a contracted fma(code2, absmax2, offset) must differ somewhere, or the bytes
    test above could not tell a fused kernel from a correct one."""
    q, s = _quantize((4096, 4096), torch.float32, 64, True)
    code2, absmax2 = s.state2.code, s.state2.absmax
    blk2 = torch.arange(s.absmax.numel(), device = "cuda") // s.state2.blocksize
    c = code2[s.absmax.long()].double()
    a2 = absmax2[blk2].double()
    two_roundings = (code2[s.absmax.long()] * absmax2[blk2]) + s.offset
    one_rounding = (c * a2 + s.offset.double()).float()
    assert not torch.equal(two_roundings, one_rounding)


@pytest.mark.skipif(not nf4_mod._HAS_MUL_RN, reason = "only meaningful where mul_rn is used")
def test_fp_fusion_disabled_fallback_is_also_exact(monkeypatch):
    """The HIP route (no mul_rn, fp fusion disabled at launch) exercised on CUDA."""
    monkeypatch.setattr(nf4_mod, "_HAS_MUL_RN", False)
    for dtype in (torch.float32, torch.bfloat16):
        q, s = _quantize((4096, 4096), dtype, 64, True, seed = 3)
        assert _bytes_equal(_ours(q, s), bnb_functional.dequantize_4bit(q, s))


def test_side_stream_without_sync():
    ref_q, ref_s = _quantize((4096, 4096), torch.bfloat16, 64, True, seed = 1)
    expected = bnb_functional.dequantize_4bit(ref_q, ref_s)
    torch.cuda.synchronize()
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        # Inputs produced on the side stream, consumed on it, no host sync in between.
        q = ref_q.clone()
        absmax = ref_s.absmax.clone()
        res = dequantize_nf4(
            q,
            absmax,
            ref_s.state2.code,
            ref_s.state2.absmax,
            ref_s.offset,
            ref_s.blocksize,
            ref_s.state2.blocksize,
            ref_s.shape,
            ref_s.dtype,
        )
        checksum = res.float().sum()
    torch.cuda.current_stream().wait_stream(side)
    assert _bytes_equal(res, expected)
    assert torch.isfinite(checksum)


def test_cuda_graph_replay_sees_new_weight_bytes():
    q, s = _quantize((2048, 1024), torch.bfloat16, 64, True, seed = 5)
    q2, s2 = _quantize((2048, 1024), torch.bfloat16, 64, True, seed = 6)
    static_q = q.clone()
    static_absmax = s.absmax.clone()
    args = (s.state2.code, s.state2.absmax, s.offset, s.blocksize, s.state2.blocksize, s.shape, s.dtype)

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):  # warm up (compiles the kernel) outside capture
            dequantize_nf4(static_q, static_absmax, *args)
    torch.cuda.current_stream().wait_stream(side)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_out = dequantize_nf4(static_q, static_absmax, *args)
    graph.replay()
    torch.cuda.synchronize()
    assert _bytes_equal(static_out, bnb_functional.dequantize_4bit(q, s))

    # New weight and absmax bytes, same second-level state: replay must pick them up.
    s2_same_state2 = bnb_functional.QuantState(
        absmax = s2.absmax,
        shape = s.shape,
        code = s.code,
        blocksize = s.blocksize,
        quant_type = "nf4",
        dtype = s.dtype,
        offset = s.offset,
        state2 = s.state2,
    )
    static_q.copy_(q2)
    static_absmax.copy_(s2.absmax)
    graph.replay()
    torch.cuda.synchronize()
    assert _bytes_equal(static_out, bnb_functional.dequantize_4bit(q2, s2_same_state2))


def test_torch_compile_fullgraph_no_graph_breaks():
    import torch._dynamo

    q, s = _quantize((1024, 2048), torch.bfloat16, 64, True, seed = 7)
    code2, absmax2, offset = s.state2.code, s.state2.absmax, s.offset
    shape = list(s.shape)

    def f(q, absmax):
        w = dequantize_nf4(q, absmax, code2, absmax2, offset, 64, s.state2.blocksize, shape, s.dtype)
        return w

    torch._dynamo.reset()
    explanation = torch._dynamo.explain(f)(q, s.absmax)
    assert explanation.graph_break_count == 0, explanation.break_reasons
    assert explanation.graph_count == 1

    compiled = torch.compile(f, fullgraph = True)
    assert _bytes_equal(compiled(q, s.absmax), bnb_functional.dequantize_4bit(q, s))


def test_torch_compile_out_variant():
    q, s = _quantize((256, 512), torch.float16, 64, True, seed = 8)
    code2, absmax2, offset = s.state2.code, s.state2.absmax, s.offset

    def f(q, absmax, out):
        dequantize_nf4(q, absmax, code2, absmax2, offset, 64, s.state2.blocksize, s.shape, s.dtype, out = out)
        return out * 1

    torch._dynamo.reset()
    out = torch.empty(s.shape, dtype = s.dtype, device = "cuda")
    res = torch.compile(f, fullgraph = True)(q, s.absmax, out)
    ref = bnb_functional.dequantize_4bit(q, s)
    assert _bytes_equal(out, ref) and _bytes_equal(res, ref)


def test_torch_compile_with_cold_lut_cache_does_not_cache_a_fake_tensor(monkeypatch):
    """The first call ever may be under torch.compile: the LUT created while tracing is fake
    and must not be cached, or the next trace mixes fake modes."""
    monkeypatch.setattr(nf4_mod, "_LUTS", {})
    q, s = _quantize((512, 512), torch.bfloat16, 64, True, seed = 9)
    code2, absmax2, offset = s.state2.code, s.state2.absmax, s.offset

    def f(q, absmax):
        return dequantize_nf4(q, absmax, code2, absmax2, offset, 64, s.state2.blocksize, list(s.shape), s.dtype)

    torch._dynamo.reset()
    res = torch.compile(f, fullgraph = True)(q, s.absmax)
    assert _bytes_equal(res, bnb_functional.dequantize_4bit(q, s))
    assert all(not isinstance(t, torch._subclasses.FakeTensor) for t in nf4_mod._LUTS.values())
    torch._dynamo.reset()
    res2 = torch.compile(f, fullgraph = True)(q, s.absmax)
    assert _bytes_equal(res2, res)
