# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""Rowwise FP8 scaled_mm route: fp16 range before torch 2.14, and no row-count guard for calls that always
dequantize when compiled. CPU tensors; torch._scaled_mm and the dequant GEMM run for real."""

import pytest
import torch
import torch._dynamo
from torch._subclasses.fake_tensor import FakeTensorMode

from unsloth.kernels import fp8 as F8

LARGE = (14336, 4096)  # Llama-3.1-8B MLP: 112 MiB in bf16, spills L4's 48 MiB L2


def _cpu_scaled_mm_fp16():
    try:
        a = torch.ones(16, 16).to(torch.float8_e4m3fn)
        ones = torch.ones(16, 1)
        torch._scaled_mm(
            a,
            a.t(),
            scale_a = ones,
            scale_b = ones.t(),
            bias = torch.zeros(16, dtype = torch.float16),
            out_dtype = torch.float16,
        )
        return True
    except Exception:
        return False


def _fp16_mode(monkeypatch, mode):
    # "bias": torch 2.10-2.13 CUDA (zero fp16 bias selects the fp16 epilogue). "dequant": torch < 2.10 or ROCm
    # before 2.14 (rowwise writes bf16 only). "native": torch 2.14+ (fp16 output as asked).
    monkeypatch.setattr(F8, "_SCALED_MM_FP16_OUT_BROKEN", mode != "native")
    monkeypatch.setattr(F8, "_SCALED_MM_FP16_ZERO_BIAS", mode == "bias")
    monkeypatch.setattr(F8, "_SCALED_MM_FP16_DEQUANT_ONLY", mode == "dequant")
    monkeypatch.setattr(F8, "_SCALED_MM_FP16_ZERO_BIAS_CACHE", {})


@pytest.mark.parametrize("mode", ["bias", "dequant", "native"])
def test_fp16_scaled_mm_out_dtype_and_bias_follow_the_fp16_mode(monkeypatch, mode):
    _fp16_mode(monkeypatch, mode)
    x = torch.empty(4, 32, dtype = torch.float16)
    w = torch.empty(48, 32, dtype = torch.float8_e4m3fn)
    assert F8._scaled_mm_out_dtype(torch.float16) is (
        torch.bfloat16 if mode == "dequant" else torch.float16
    )
    assert F8._scaled_mm_dtype_supported(x) is (mode != "dequant")
    bias = F8._scaled_mm_fp16_bias(x, w)
    if mode == "bias":
        assert bias.dtype == torch.float16 and bias.shape == (48,) and not bias.any()
        # Cached per (rows, device, stream): no allocation + fill per eager call.
        assert F8._scaled_mm_fp16_bias(x, w) is bias
        assert (
            F8._scaled_mm_fp16_bias(x, torch.empty(64, 32, dtype = torch.float8_e4m3fn)) is not bias
        )
        assert len(F8._SCALED_MM_FP16_ZERO_BIAS_CACHE) == 2
    else:
        assert bias is None
    for dtype in (torch.bfloat16, torch.float32):
        assert F8._scaled_mm_out_dtype(dtype) is dtype
        assert F8._scaled_mm_fp16_bias(x.to(dtype), w) is None


def test_fp16_zero_bias_cache_is_bounded_and_skips_fake_tensors(monkeypatch):
    _fp16_mode(monkeypatch, "bias")
    monkeypatch.setattr(F8, "_SCALED_MM_FP16_ZERO_BIAS_CACHE_MAX", 4)
    x = torch.empty(4, 32, dtype = torch.float16)
    for n in range(16, 16 * 11, 16):
        F8._scaled_mm_fp16_bias(x, torch.empty(n, 32, dtype = torch.float8_e4m3fn))
        assert len(F8._SCALED_MM_FP16_ZERO_BIAS_CACHE) <= 4
    monkeypatch.setattr(F8, "_SCALED_MM_FP16_ZERO_BIAS_CACHE", {})
    with FakeTensorMode():
        fx = torch.empty(4, 32, dtype = torch.float16)
        fw = torch.empty(48, 32, dtype = torch.float8_e4m3fn)
        assert F8._scaled_mm_fp16_bias(fx, fw) is not None
    assert F8._SCALED_MM_FP16_ZERO_BIAS_CACHE == {}
    # Inference mode does not leak an inference tensor into later autograd-enabled calls.
    with torch.inference_mode():
        bias = F8._scaled_mm_fp16_bias(x, torch.empty(48, 32, dtype = torch.float8_e4m3fn))
    assert not bias.is_inference()


def test_fp16_flags_follow_the_installed_torch():
    from torch.torch_version import TorchVersion

    assert F8._SCALED_MM_FP16_BIAS_OUT is (
        torch.__version__ >= "2.10.0" and torch.version.hip is None
    )
    broken = F8._SCALED_MM_FP16_OUT_BROKEN
    assert F8._SCALED_MM_FP16_ZERO_BIAS is (broken and F8._SCALED_MM_FP16_BIAS_OUT)
    assert F8._SCALED_MM_FP16_DEQUANT_ONLY is (broken and not F8._SCALED_MM_FP16_BIAS_OUT)
    # 2.9 CUTLASS rowwise accepts only float32 / bf16 bias and bf16 output.
    assert not TorchVersion("2.9.1+cu128") >= "2.10.0"
    assert TorchVersion("2.10.0+cu128") >= "2.10.0"


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("mode", ["bias", "dequant"])
@pytest.mark.parametrize("rows", [80, 300])
def test_fp16_output_near_the_fp16_max_stays_finite(monkeypatch, compiled, mode, rows):
    # All-ones activations and weights with row scale 65504 / K: every output is exactly 65504, which a bf16
    # output rounds to 65536 and the cast back to fp16 turns into inf. Over 64 rows (the scaled_mm row range)
    # "bias" keeps fp16 through the epilogue; "dequant" (torch < 2.10) must not call _scaled_mm at all.
    if mode == "bias" and not _cpu_scaled_mm_fp16():
        pytest.skip("needs CPU torch._scaled_mm with fp16 output")
    _fp16_mode(monkeypatch, mode)
    calls = []
    real_scaled_mm = torch._scaled_mm

    def counting_scaled_mm(*args, **kwargs):
        calls.append(kwargs.get("out_dtype"))
        return real_scaled_mm(*args, **kwargs)

    monkeypatch.setattr(torch, "_scaled_mm", counting_scaled_mm)
    N, K = 64, 32
    x = torch.ones(rows, K, dtype = torch.float16)
    weight = torch.ones(N, K).to(torch.float8_e4m3fn)
    scales = torch.full((N, 1), 65504.0 / K)
    fn = lambda x: F8.FbgemmFp8Linear_matmul.apply(x, weight, scales, None, "scaled_mm")
    if compiled:
        torch._dynamo.reset()
        fn = torch.compile(fn, backend = "aot_eager", fullgraph = True)
    out = fn(x)
    assert out.dtype == torch.float16
    assert torch.isfinite(out).all()
    assert (out == 65504).all()
    if mode == "dequant":
        assert calls == []
    else:
        assert calls and all(d is torch.float16 for d in calls)


@pytest.mark.parametrize("rows", [17, 32, 64, 65, 128, 4096])
@pytest.mark.parametrize("mode", ["bias", "dequant", "native"])
def test_fp16_routing_decision(monkeypatch, mode, rows):
    # The routing decision, CPU only: where rowwise writes bf16 only, fp16 dequantizes at every row count, in the
    # eager direct route and in the autograd function alike. bf16 is untouched.
    monkeypatch.setattr(torch.version, "hip", None, raising = False)
    monkeypatch.setattr(F8, "_fp8_rowwise_backend", lambda device: "scaled_mm")
    monkeypatch.setattr(F8, "_FP8_ROWWISE_DIRECT", True)
    monkeypatch.setattr(F8, "_FP8_DEQUANT_MAX_ROWS_ENV", None)
    monkeypatch.setattr(F8, "_FP8_DEQUANT_MAX_ROWS", 64)
    monkeypatch.setattr(F8, "_fp8_l2_cache_bytes", lambda device: 48 * 1024 * 1024)
    _fp16_mode(monkeypatch, mode)
    calls = []

    def fake_scaled_mm(a, b, **kwargs):
        calls.append(("scaled_mm", kwargs["out_dtype"]))
        return torch.empty(a.shape[0], b.shape[1], device = a.device, dtype = kwargs["out_dtype"])

    def fake_dequant(w, s, dtype):
        calls.append(("dequant", dtype))
        return torch.empty(w.shape, device = w.device, dtype = dtype)

    monkeypatch.setattr(torch, "_scaled_mm", fake_scaled_mm)
    monkeypatch.setattr(F8, "weight_dequant", fake_dequant)
    N, K = LARGE
    scaled_rows = rows > 16  # the L2 gate: this weight spills 48 MiB, so 17+ rows take scaled_mm
    for dtype in (torch.float16, torch.bfloat16):
        expected = (
            "scaled_mm"
            if scaled_rows and (dtype == torch.bfloat16 or mode != "dequant")
            else "dequant"
        )
        calls.clear()
        with FakeTensorMode(), torch.no_grad():
            X = torch.empty(1, rows, K, device = "cuda", dtype = dtype)
            w = torch.empty(N, K, device = "cuda", dtype = torch.float8_e4m3fn)
            s = torch.empty(N, 1, device = "cuda", dtype = torch.float32)
            assert F8._fp8_rowwise_direct_route(X, w, s) == expected, (dtype, rows)
            y = F8.FbgemmFp8Linear_matmul.apply(X, w, s, None, "scaled_mm")
        assert y.shape == (1, rows, N) and y.dtype == dtype
        assert calls == [
            (expected, dtype if expected == "dequant" else F8._scaled_mm_out_dtype(dtype))
        ]
    assert F8._SCALED_MM_FP16_ZERO_BIAS_CACHE == {}


def _compile_count(
    fn,
    rows_seq,
    K,
    dtype,
    mark = None,
):
    from torch._dynamo.backends.debugging import aot_eager

    graphs = []

    def backend(gm, example_inputs):
        graphs.append(gm)
        return aot_eager(gm, example_inputs)

    torch._dynamo.reset()
    compiled = torch.compile(fn, backend = backend, dynamic = mark is None)
    for rows in rows_seq:
        x = torch.randn(rows, K, dtype = dtype)
        if mark is not None:
            torch._dynamo.mark_dynamic(x, 0, min = mark[0], max = mark[1])
        compiled(x)
    return len(graphs)


@pytest.mark.parametrize("mark", [None, (2, 256)])
def test_compiled_float32_dequant_adds_no_row_guard(monkeypatch, mark):
    # torch < 2.14 without float32 rowwise output: every row count dequantizes, so the compiled forward must not
    # branch on the row count (one graph under dynamic shapes, and a strict mark_dynamic range must hold).
    monkeypatch.setattr(F8, "_SCALED_MM_FP16_OUT_BROKEN", True)
    monkeypatch.setattr(
        F8, "_scaled_mm_float32_out", torch._dynamo.assume_constant_result(lambda device: False)
    )
    N, K = 64, 32
    weight = (torch.randn(N, K) * 0.5).to(torch.float8_e4m3fn)
    scales = torch.rand(N, 1) + 0.5
    fn = lambda x: F8.FbgemmFp8Linear_matmul.apply(x, weight, scales, None, "scaled_mm")
    assert _compile_count(fn, [8, 128, 64, 256], K, torch.float32, mark) == 1


def test_compiled_float32_with_float32_output_still_routes_by_rows(monkeypatch):
    # Where float32 output works (Hopper cuBLAS) the row threshold is the intended route boundary.
    monkeypatch.setattr(F8, "_SCALED_MM_FP16_OUT_BROKEN", True)
    monkeypatch.setattr(
        F8, "_scaled_mm_float32_out", torch._dynamo.assume_constant_result(lambda device: True)
    )
    N, K = 64, 32
    weight = (torch.randn(N, K) * 0.5).to(torch.float8_e4m3fn)
    scales = torch.rand(N, 1) + 0.5
    fn = lambda x: F8.FbgemmFp8Linear_matmul.apply(x, weight, scales, None, "scaled_mm")
    assert _compile_count(fn, [8, 128, 64, 256], K, torch.float32) == 2


@pytest.mark.parametrize("mark", [None, (2, 256)])
def test_compiled_fp16_dequant_only_adds_no_row_guard(monkeypatch, mark):
    # torch < 2.10: fp16 dequantizes at every row count, so like float32 above it gets one graph.
    _fp16_mode(monkeypatch, "dequant")
    N, K = 64, 32
    weight = (torch.randn(N, K) * 0.5).to(torch.float8_e4m3fn)
    scales = torch.rand(N, 1) + 0.5
    fn = lambda x: F8.FbgemmFp8Linear_matmul.apply(x, weight, scales, None, "scaled_mm")
    assert _compile_count(fn, [8, 128, 64, 256], K, torch.float16, mark) == 1


@pytest.mark.skipif(not _cpu_scaled_mm_fp16(), reason = "needs CPU torch._scaled_mm with fp16 output")
def test_compiled_fp16_zero_bias_adds_no_graphs_or_breaks(monkeypatch):
    # The zero bias is built in the graph when traced (the eager cache is never consulted): fp16 compiles exactly
    # like bf16 (one row guard), with no graph break, and leaves the eager cache empty.
    from torch._dynamo.utils import counters

    _fp16_mode(monkeypatch, "bias")
    N, K = 64, 32
    weight = (torch.randn(N, K) * 0.5).to(torch.float8_e4m3fn)
    scales = torch.rand(N, 1) + 0.5
    fn = lambda x: F8.FbgemmFp8Linear_matmul.apply(x, weight, scales, None, "scaled_mm")
    counters.clear()
    fp16 = _compile_count(fn, [8, 128, 64, 256, 300], K, torch.float16)
    breaks = sum(counters["graph_break"].values())
    bf16 = _compile_count(fn, [8, 128, 64, 256, 300], K, torch.bfloat16)
    assert fp16 == bf16 == 2
    assert breaks == 0
    assert F8._SCALED_MM_FP16_ZERO_BIAS_CACHE == {}


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a CUDA device")
def test_fp16_zero_bias_is_not_shared_across_cuda_streams(monkeypatch):
    # The zeros fill is asynchronous on the stream that made it: another stream must get its own bias.
    _fp16_mode(monkeypatch, "bias")
    x = torch.empty(4, 32, dtype = torch.float16, device = "cuda")
    w = torch.empty(48, 32, dtype = torch.float8_e4m3fn, device = "cuda")
    bias = F8._scaled_mm_fp16_bias(x, w)
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        other = F8._scaled_mm_fp16_bias(x, w)
        assert F8._scaled_mm_fp16_bias(x, w) is other
    assert other is not bias
    assert F8._scaled_mm_fp16_bias(x, w) is bias
