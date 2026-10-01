# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

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

from unsloth.kernels import nf4_gemv
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


@pytest.fixture(params = [None, "narrow", "bytes", "words"], ids = ["auto", "narrow", "bytes", "words"])
def kernel(request, monkeypatch):
    # Both kernel forms, whichever this Triton version and GPU would pick on its own, and the
    # T4 / L4 large-weight config on this GPU.
    if request.param == "narrow":
        capability = torch.cuda.get_device_capability()
        monkeypatch.setattr(nf4_gemv, "_NARROW_CAPS", (capability,))
        nf4_gemv._gemv_config.cache_clear()
        request.addfinalizer(nf4_gemv._gemv_config.cache_clear)
    else:
        monkeypatch.setattr(nf4_gemv, "_FORCE_KERNEL", request.param)
    return request.param


@pytest.mark.parametrize("nested", [True, False], ids = ["nested", "flat"])
@pytest.mark.parametrize("dtype", DTYPES, ids = str)
@pytest.mark.parametrize("shape", SHAPES, ids = lambda s: f"{s[0]}x{s[1]}")
def test_matches_fp32_reference_and_bitsandbytes(shape, dtype, nested, kernel):
    n, k = shape
    q, s = _quant(n, k, dtype, nested)
    X = torch.randn(1, 1, k, dtype = dtype, device = "cuda")
    ref = _reference(X, q, s)
    out = gemv_nf4(X, *_args(q, s))
    assert out.shape == (1, 1, n) and out.dtype == dtype
    assert _rel_err(out, ref) < TOL[dtype]
    if nested:  # the ctypes GEMV only takes double quantized states
        assert _rel_err(out, ref) <= _rel_err(_fast_gemv_ctypes(X, q, s), ref) + 4e-3


def test_unaligned_weight_view_uses_the_byte_kernel(monkeypatch):
    # The words kernel reads int32; a weight view starting off a 4 byte boundary must not reach it.
    monkeypatch.setattr(nf4_gemv, "_FORCE_KERNEL", "words")
    q, s = _quant(1024, 4096, torch.float16)
    buf = torch.empty(q.numel() + 2, dtype = torch.uint8, device = "cuda")
    buf[2:].copy_(q.view(-1))
    Wv = buf[2:].view(q.shape)
    assert not nf4_gemv._word_aligned(Wv)
    X = torch.randn(1, 1, 4096, dtype = torch.float16, device = "cuda")
    assert _rel_err(gemv_nf4(X, Wv, *_args(q, s)[1:]), _reference(X, q, s)) < TOL[torch.float16]


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


def _utils_with_kernels():
    from unsloth.kernels import utils
    if not (utils._USE_NF4_KERNELS and utils._TRITON_GEMV_EAGER):
        pytest.skip("eager decode does not take the Triton GEMV here")
    return utils


@pytest.mark.parametrize("nested", [True, False], ids = ["nested", "flat"])
def test_decode_plan_reruns_match_the_kernel(nested):
    """Eager decode repeats the launch planned on each weight's first call. With weights
    interleaved as in a decoder, every rerun equals a fresh gemv_nf4 launch bit for bit."""
    from unsloth.kernels import triton_launch

    utils = _utils_with_kernels()
    weights = [
        _quant(n, k, torch.bfloat16, nested = nested, seed = i)
        for i, (n, k) in enumerate([(512, 256), (256, 1024), (768, 512)])
    ]
    for step in range(3):
        for i, (q, s) in enumerate(weights):
            X = torch.randn(1, 1, s.shape[1], dtype = torch.bfloat16, device = "cuda")
            got = utils.fast_gemv(X, q, s)
            want = gemv_nf4(X, *_args(q, s))
            assert torch.equal(got, want), (step, i)
    if triton_launch._ENABLED:
        assert all(getattr(s, "_unsloth_gemv_plan", None) for _, s in weights)


def test_decode_plan_is_rebuilt_or_bypassed_when_inputs_change():
    """New absmax (QuantState.to), another activation dtype, or an activation that is not 16 byte
    aligned would not fit the planned launch: each must still give the kernel's own result."""
    import copy
    import pickle

    utils = _utils_with_kernels()
    q, s = _quant(512, 256, torch.bfloat16, nested = False)
    X = torch.randn(1, 1, 256, dtype = torch.bfloat16, device = "cuda")
    for _ in range(2):
        utils.fast_gemv(X, q, s)
    s.absmax = s.absmax * 2
    assert torch.equal(utils.fast_gemv(X, q, s), gemv_nf4(X, *_args(q, s)))
    X16 = X.half()
    assert torch.equal(utils.fast_gemv(X16, q, s), gemv_nf4(X16, *_args(q, s)))
    base = torch.randn(1, 1, 257, dtype = torch.bfloat16, device = "cuda")
    Xm = base[:, :, 1:]  # 2 bytes past a 16 byte boundary
    assert torch.equal(utils.fast_gemv(Xm, q, s), gemv_nf4(Xm.contiguous(), *_args(q, s)))
    assert getattr(copy.deepcopy(s), "_unsloth_gemv_plan", None) is None
    assert getattr(pickle.loads(pickle.dumps(s)), "_unsloth_gemv_plan", None) is None
