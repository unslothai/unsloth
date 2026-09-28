# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""The fused Triton NF4 dequant kernel must be byte-identical to bitsandbytes. Streams, CUDA
graphs and torch.compile through fast_dequantize are covered in test_bnb_integration_compile.py."""

import hashlib
import os

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("needs a CUDA device", allow_module_level = True)

pytest.importorskip("triton")
bnb_functional = pytest.importorskip("bitsandbytes.functional")

from unsloth.kernels import nf4 as nf4_mod
from unsloth.kernels import nf4_gemv as nf4_gemv_mod
from unsloth.kernels.nf4 import dequantize_nf4

_INT_VIEW = {torch.float16: torch.int16, torch.bfloat16: torch.int16, torch.float32: torch.int32}

# nf4_dequantization_lut in bitsandbytes csrc/kernels.cu.
CSRC_NF4 = (
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
)


def _bytes_equal(a, b):
    assert a.shape == b.shape and a.dtype == b.dtype
    iv = _INT_VIEW[a.dtype]
    return torch.equal(a.contiguous().view(iv), b.contiguous().view(iv))


def _quantize(
    shape,
    dtype,
    blocksize = 64,
    nested = True,
    seed = 0,
):
    g = torch.Generator(device = "cuda").manual_seed(seed)
    W = torch.randn(shape, dtype = dtype, device = "cuda", generator = g)
    return bnb_functional.quantize_4bit(
        W, blocksize = blocksize, quant_type = "nf4", compress_statistics = nested
    )


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
    return (q, s.absmax, None, None, None, s.code, s.blocksize, 0, s.shape, s.dtype)


SHAPES = [(1, 4096), (4096, 1), (17, 33), (4096, 4096), (14336, 4096), (4096, 14336)]


@pytest.mark.parametrize("shape", SHAPES, ids = lambda s: f"{s[0]}x{s[1]}")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32], ids = str)
@pytest.mark.parametrize("blocksize", [64, 128])
@pytest.mark.parametrize("nested", [True, False], ids = ["nested", "flat"])
def test_bytes_match_bitsandbytes(shape, dtype, blocksize, nested):
    q, s = _quantize(shape, dtype, blocksize, nested)
    assert _bytes_equal(dequantize_nf4(*_args(q, s)), bnb_functional.dequantize_4bit(q, s))


def test_vocab_sized_weight_matches_bitsandbytes():
    if torch.cuda.mem_get_info()[0] < 8 * 2**30:
        pytest.skip("not enough free GPU memory for the vocab-sized weight")
    q, s = _quantize((128256, 4096), torch.bfloat16)
    assert _bytes_equal(dequantize_nf4(*_args(q, s)), bnb_functional.dequantize_4bit(q, s))


def test_quant_state_code_is_the_kernel_table():
    """fast_dequantize passes quant_state.code as the table; bitsandbytes' own kernel uses the
    hardcoded csrc table, so the two must be the same fp32 values."""
    for dtype in (torch.float16, torch.bfloat16):
        _, s = _quantize((64, 64), dtype)
        csrc = torch.tensor(CSRC_NF4, dtype = torch.float32)
        assert torch.equal(s.code.cpu().view(torch.int32), csrc.view(torch.int32))


def test_explicit_out_is_written_and_returned():
    q, s = _quantize((256, 512), torch.bfloat16)
    out = torch.full(s.shape, 7.0, dtype = s.dtype, device = "cuda")
    assert dequantize_nf4(*_args(q, s), out = out).data_ptr() == out.data_ptr()
    assert _bytes_equal(out, bnb_functional.dequantize_4bit(q, s))


def test_compiled_out_variant():
    q, s = _quantize((256, 512), torch.float16, seed = 8)

    def f(q, absmax, out):
        args = list(_args(q, s))
        args[1] = absmax
        dequantize_nf4(*args, out = out)
        return out * 1

    torch._dynamo.reset()
    out = torch.empty(s.shape, dtype = s.dtype, device = "cuda")
    res = torch.compile(f, fullgraph = True)(q, s.absmax, out)
    ref = bnb_functional.dequantize_4bit(q, s)
    assert _bytes_equal(out, ref) and _bytes_equal(res, ref)


def test_rejects_unsupported_dtype_and_blocksize():
    q, s = _quantize((64, 64), torch.bfloat16)
    args = list(_args(q, s))
    with pytest.raises(TypeError):
        dequantize_nf4(*args[:-1], torch.int8)
    args[6] = 48
    with pytest.raises(ValueError):
        dequantize_nf4(*args)


def test_fma_contraction_would_be_caught():
    """Control: a contracted fma(code2, absmax2, offset) differs from the two-rounding result
    somewhere, so the bytes test above can tell a fused kernel from a correct one."""
    _, s = _quantize((4096, 4096), torch.float32)
    code2, absmax2 = s.state2.code, s.state2.absmax
    blk2 = torch.arange(s.absmax.numel(), device = "cuda") // s.state2.blocksize
    two_roundings = (code2[s.absmax.long()] * absmax2[blk2]) + s.offset
    one_rounding = (
        code2[s.absmax.long()].double() * absmax2[blk2].double() + s.offset.double()
    ).float()
    assert not torch.equal(two_roundings, one_rounding)


@pytest.mark.skipif(not nf4_mod._HAS_MUL_RN, reason = "already the no-mul_rn route")
def test_hip_route_without_mul_rn_is_also_exact(monkeypatch):
    """The HIP route (plain multiply, fp fusion disabled at launch) exercised on CUDA."""
    monkeypatch.setattr(nf4_mod, "_HAS_MUL_RN", False)
    for dtype in (torch.float32, torch.bfloat16):
        q, s = _quantize((4096, 4096), dtype, seed = 3)
        assert _bytes_equal(dequantize_nf4(*_args(q, s)), bnb_functional.dequantize_4bit(q, s))


@pytest.mark.parametrize("fp_fusion", [False, True], ids = ["fusion_off", "fusion_on"])
def test_exact_with_fp_fusion_on_or_off(fp_fusion):
    """Eager launches with fp fusion off; torch.compile re-emits the kernel with it on. Both must
    keep code2 * absmax2 and + offset as two roundings (fp32 output shows a 1 ulp scale error)."""
    if fp_fusion and not nf4_mod._HAS_MUL_RN:
        # Without libdevice.mul_rn (HIP, the interpreter) the kernel never runs with fusion on.
        pytest.skip("no mul_rn: this route never launches with fp fusion on")
    for dtype in (torch.float32, torch.bfloat16):
        q, s = _quantize((4096, 4096), dtype, seed = 3)
        out = torch.empty(4096, 4096, dtype = dtype, device = "cuda")
        nf4_mod._launch(
            nf4_mod._nf4_dequant_kernel,
            q,
            s.absmax,
            s.state2.code,
            s.state2.absmax,
            s.offset,
            s.code,
            s.blocksize,
            s.state2.blocksize,
            out,
            fp_fusion = fp_fusion,
        )
        assert _bytes_equal(out, bnb_functional.dequantize_4bit(q, s))


@pytest.mark.skipif(not nf4_mod._HAS_MUL_RN, reason = "the plain multiply keeps subnormals")
@pytest.mark.parametrize("nested", [True, False], ids = ["nested", "flat"])
def test_subnormal_products_flush_like_bitsandbytes(nested):
    """bitsandbytes' kernels flush subnormal products to zero, in the nested absmax decode and in
    code * absmax; the Triton kernel must too."""
    q, s = _quantize((256, 1024), torch.float32, nested = nested, seed = 5)
    if nested:
        s.state2.absmax.mul_(1e-36)
        s.offset.zero_()
    else:
        s.absmax.fill_(2e-38)
    ref = bnb_functional.dequantize_4bit(q, s)
    assert ref.abs().max() > 0 and (ref == 0).float().mean() > 0.1
    assert _bytes_equal(dequantize_nf4(*_args(q, s)), ref)


@pytest.mark.parametrize("storage", [torch.bfloat16, torch.float16, torch.float32], ids = str)
def test_float_packed_storage_is_read_as_bytes(storage):
    """FSDP-QLoRA packs the 4bit weight into float storage (quant_storage); it is still bytes."""
    g = torch.Generator(device = "cuda").manual_seed(4)
    W = torch.randn(512, 256, dtype = torch.bfloat16, device = "cuda", generator = g)
    q, s = bnb_functional.quantize_4bit(
        W, quant_type = "nf4", compress_statistics = True, quant_storage = storage
    )
    assert q.dtype == storage
    assert _bytes_equal(dequantize_nf4(*_args(q, s)), bnb_functional.dequantize_4bit(q, s))


@pytest.mark.parametrize("lut_mode", [0, 1, 2])
@pytest.mark.parametrize("words", [False, True])
@pytest.mark.parametrize("evict", [False, True])
@pytest.mark.parametrize("shape", [(17, 33), (5, 4096), (1024, 4096)], ids = str)
def test_every_launch_config_is_exact(shape, words, evict, lut_mode, monkeypatch):
    # The per-GPU launch table only picks speed; every knob combination must stay bit-exact.
    for dtype in (torch.float16, torch.bfloat16):
        for blocksize in (64, 128):
            q, s = _quantize(shape, dtype, blocksize)
            ref = bnb_functional.dequantize_4bit(q, s)
            for target in (256, 2048):
                monkeypatch.setattr(
                    nf4_mod, "_CONFIG_OVERRIDE", (target, 4, words, evict, lut_mode)
                )
                assert _bytes_equal(dequantize_nf4(*_args(q, s)), ref), (dtype, blocksize, target)


def test_compile_cache_key_covers_the_kernel_source():
    """Inductor's FX graph cache keys a triton_op call without the Triton source behind it: with a
    warm cache, an edited kernel came back as the previous kernel's code. The kernel files' hashes
    in cache_key_tag make an edited kernel a cache miss."""
    if not hasattr(getattr(torch.compiler, "config", None), "cache_key_tag"):
        pytest.skip("torch has no compile cache key tag")
    tags = torch.compiler.config.cache_key_tag.split(",")
    for mod in (nf4_mod, nf4_gemv_mod):
        with open(mod.__file__, "rb") as file:
            digest = hashlib.sha256(file.read()).hexdigest()[:16]
        assert f"unsloth/{os.path.basename(mod.__file__)}:{digest}" in tags
