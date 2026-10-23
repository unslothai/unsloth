# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""FP8 block dequant to fp16: the integer-rounding path (ROCm default, UNSLOTH_FP8_EXACT_F16_ROUND=1
elsewhere) must match the torch loop bit for bit."""

import numpy as np
import pytest
import torch

from unsloth.kernels import fp8

CUDA = torch.cuda.is_available()
HIP = torch.version.hip is not None
needs_cuda = pytest.mark.skipif(not CUDA, reason = "needs a GPU")
BLOCK = 16


def _data(device):
    # Every e4m3fn byte per 16x16 block; scales 1e-4..1e-2 (rounding ties) and 2^-40..2^10 (subnormal / overflow).
    g = torch.Generator(device = "cpu").manual_seed(0)
    rows, cols = 256, 128
    w = torch.arange(256, dtype = torch.uint8).view(16, 16).repeat(rows, cols)
    s = torch.rand(rows, cols, generator = g, dtype = torch.float32) * 1e-2 + 1e-4
    wide = torch.rand(rows // 4, cols, generator = g, dtype = torch.float64) + 1.0
    wide = wide * 2.0 ** torch.randint(-40, 11, (rows // 4, cols), generator = g).double()
    s[: rows // 4] = wide.float()
    return w.view(torch.float8_e4m3fn).to(device), s.to(device)


def _same_bits(a, b):
    nan = torch.isnan(a)
    assert torch.equal(nan, torch.isnan(b))
    ai = a.masked_fill(nan, 0).view(torch.int16)
    bi = b.masked_fill(nan, 0).view(torch.int16)
    bad = int((ai != bi).sum())
    assert bad == 0, f"{bad} of {ai.numel()} differ"


def test_data_has_double_rounding_ties():
    # Non-vacuity: some products round differently from exact -> fp16 than from fp32 -> fp16.
    w, s = _data("cpu")
    p = (w.double() * s.double().repeat_interleave(BLOCK, 0).repeat_interleave(BLOCK, 1)).numpy()
    once, twice = (
        p.astype(np.float16).view(np.int16),
        p.astype(np.float32).astype(np.float16).view(np.int16),
    )
    assert int((once != twice).sum()) > 50


def _rounding_inputs():
    # Every fp32 exponent x mantissas at the fp16 rounding point (halves, +-1, odd / even), both signs.
    low = torch.tensor(
        [0, 1, 0xFFF, 0x1000, 0x1001, 0x1FFF, 0x2000, 0x3000, 0x7FFFFF, 0x7FE000, 0x7FF000],
        dtype = torch.int64,
    )
    exps = torch.arange(256, dtype = torch.int64)
    bits = (exps[:, None] << 23 | low[None, :]).flatten()
    g = torch.Generator(device = "cpu").manual_seed(0)
    bits = torch.cat([bits, torch.randint(0, 1 << 31, (1 << 16,), generator = g)])
    bits = torch.cat([bits, bits | (1 << 31)])
    return bits.to(torch.int32).view(torch.float32)


@needs_cuda
def test_f32_to_f16_rne_matches_torch_cast():
    import triton
    import triton.language as tl

    @triton.jit
    def _round(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        tl.store(y_ptr + offs, fp8._f32_to_f16_rne(tl.load(x_ptr + offs, mask = mask)), mask = mask)

    x = _rounding_inputs().cuda()
    y = torch.empty_like(x, dtype = torch.float16)
    _round[(triton.cdiv(x.numel(), 1024),)](x, y, x.numel(), BLOCK = 1024, enable_fp_fusion = True)
    _same_bits(y, x.half())


@needs_cuda
@pytest.mark.parametrize("exact", ["1", "0"])
@pytest.mark.parametrize("kernel", ["bytes", "fp8e4nv"])
def test_dequant_kernels_match_torch_loop_fp16(monkeypatch, kernel, exact):
    if exact == "0" and HIP:
        pytest.skip("the plain fp16 cast is the one ROCm Triton fuses into v_fma_mixlo_f16")
    monkeypatch.setenv("UNSLOTH_FP8_EXACT_F16_ROUND", exact)
    w, s = _data("cuda")
    ref = fp8._torch_blockwise_dequant(w, s, [BLOCK, BLOCK], torch.float16)
    if kernel == "bytes":
        out = fp8._e4m3fn_bytes_dequant(w, s, BLOCK, BLOCK, torch.float16)
    else:
        fp8._fp8_device_lacks_kernel.cache_clear()
        if fp8._fp8_kernel_unsupported(w):
            pytest.skip("no fp8e4nv on this device")
        out = fp8.weight_dequant_block(w, s, block_size = BLOCK, dtype = torch.float16)
    _same_bits(out, ref)


@needs_cuda
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_other_dtypes_untouched(monkeypatch, dtype):
    monkeypatch.setenv("UNSLOTH_FP8_EXACT_F16_ROUND", "1")
    assert fp8._fp8_exact_f16_round(dtype) is False
    w, s = _data("cuda")
    _same_bits(
        fp8._e4m3fn_bytes_dequant(w, s, BLOCK, BLOCK, dtype),
        fp8._torch_blockwise_dequant(w, s, [BLOCK, BLOCK], dtype),
    )


def test_exact_round_is_the_rocm_fp16_default(monkeypatch):
    monkeypatch.delenv("UNSLOTH_FP8_EXACT_F16_ROUND", raising = False)
    monkeypatch.setattr(fp8, "_FP8_EXACT_F16_ROUND", True)
    assert fp8._fp8_exact_f16_round(torch.float16) is True
    assert fp8._fp8_exact_f16_round(torch.bfloat16) is False
    monkeypatch.setattr(fp8, "_FP8_EXACT_F16_ROUND", False)
    assert fp8._fp8_exact_f16_round(torch.float16) is False
    monkeypatch.setenv("UNSLOTH_FP8_EXACT_F16_ROUND", "1")
    assert fp8._fp8_exact_f16_round(torch.float16) is True
    monkeypatch.setattr(fp8, "_FP8_EXACT_F16_ROUND", True)
    monkeypatch.setenv("UNSLOTH_FP8_EXACT_F16_ROUND", "0")
    assert fp8._fp8_exact_f16_round(torch.float16) is False


def test_hip_codegen_has_no_fma_mix():
    # Offline: compile both kernels for gfx1151; the exact path must not round through an f16-output
    # mix op (v_fma_mixlo/hi_f16 with a +0 addend). v_fma_mix_f32 (f16 operand, -0 addend) is a plain fmul.
    triton = pytest.importorskip("triton")
    try:
        import triton.backends.amd  # noqa: F401
        from triton.backends.compiler import GPUTarget
        from triton.compiler import ASTSource
    except Exception:
        pytest.skip("Triton built without the AMD backend")
    target = GPUTarget("hip", "gfx1151", 32)
    cases = {
        "weight_dequant_kernel": (
            fp8.weight_dequant_kernel,
            {"x_ptr": "*fp8e4nv", "s_ptr": "*fp32", "y_ptr": "*fp16", "M": "i32", "N": "i32"},
            {"BLOCK_SIZE": 128},
        ),
        "_e4m3fn_bytes_dequant_kernel": (
            fp8._e4m3fn_bytes_dequant_kernel,
            {
                "x_ptr": "*u8",
                "s_ptr": "*fp32",
                "y_ptr": "*fp16",
                "M": "i32",
                "N": "i32",
                "stride_xm": "i32",
                "stride_xn": "i32",
                "stride_sm": "i32",
                "stride_sn": "i32",
            },
            {
                "BLOCK_M_Q": 128,
                "BLOCK_N_Q": 128,
                "BLOCK_R": 8,
                "BLOCK_C": 512,
                "LONG_INDEXING": False,
            },
        ),
    }
    for name, (fn, sig, consts) in cases.items():
        found = {}
        for exact in (False, True):
            c = {**consts, "EXACT_F16": exact}
            try:
                k = triton.compile(
                    ASTSource(
                        fn = fn, signature = {**sig, **{x: "constexpr" for x in c}}, constexprs = c
                    ),
                    target = target,
                )
            except Exception as e:  # noqa: BLE001
                pytest.skip(f"offline HIP compile unavailable: {type(e).__name__}: {e}"[:200])
            asm = k.asm["amdgcn"]
            found[exact] = "v_fma_mixlo_f16" in asm or "v_fma_mixhi_f16" in asm
        assert found[True] is False, name
