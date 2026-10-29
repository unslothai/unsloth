# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""NF4 GEMV reduces before scaling, so it is checked against fp32 and must not lose to bitsandbytes."""

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
    # Auto pick, each forced kernel form, and "narrow": the T4 / L4 large-weight config on this GPU.
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
    # The words kernel reads int32; a view off a 4 byte boundary must not reach it.
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


def _linear4bit(
    n,
    k,
    seed = 0,
    dtype = torch.bfloat16,
):
    bnb = pytest.importorskip("bitsandbytes")
    torch.manual_seed(seed)
    proj = bnb.nn.Linear4bit(k, n, bias = False, compute_dtype = dtype, quant_type = "nf4")
    W = torch.randn(n, k, dtype = dtype)
    proj.weight = bnb.nn.Params4bit(W, requires_grad = False, quant_type = "nf4")
    return proj.to("cuda")


@pytest.mark.parametrize("nested", [True, False], ids = ["nested", "flat"])
@pytest.mark.parametrize("bsz", [2, 3, 8])
@pytest.mark.parametrize("shape", [(1024, 4096), (4096, 14336), (1000, 192)], ids = str)
def test_rows_kernel_is_the_batch_1_kernel_per_row(shape, bsz, nested, kernel):
    """Strided out, as fast_swiglu_inference passes up[:, :, :hd]."""
    from unsloth.kernels.nf4_gemv import gemv_nf4_rows

    q, s = _quant(*shape, torch.bfloat16, nested = nested)
    a = _args(q, s)
    X = torch.randn(bsz, 1, shape[1], dtype = torch.bfloat16, device = "cuda")
    wide = torch.zeros(bsz, 1, shape[0] + 64, dtype = torch.bfloat16, device = "cuda")
    gemv_nf4_rows(X, *a[:-1], wide[:, :, : shape[0]])
    for i in range(bsz):
        assert torch.equal(wide[i, :, : shape[0]], gemv_nf4(X[i : i + 1].contiguous(), *a)[0]), i
    assert not wide[:, :, shape[0] :].any()


@pytest.mark.parametrize("bsz", [2, 3, 4])
def test_small_decode_batches_skip_the_dequantize(bsz, monkeypatch):
    utils = _utils_with_kernels()
    proj = _linear4bit(1024, 4096)
    X = torch.randn(bsz, 1, 4096, dtype = torch.bfloat16, device = "cuda")
    ref = (
        X.float().view(bsz, -1)
        @ F.dequantize_4bit(proj.weight.data, proj.weight.quant_state).float().t()
    )
    monkeypatch.setattr(utils, "_gemv_rows_cap", lambda device: 4)
    monkeypatch.setattr(utils, "_BNB_GEMM_4BIT", True)
    monkeypatch.setattr(utils, "_bnb_gemm_4bit_device_ok", lambda device: True)
    monkeypatch.setattr(utils, "fast_dequantize", lambda *a, **k: pytest.fail("dequantized"))
    first = utils.fast_linear_forward(proj, X)
    assert _rel_err(first.view(bsz, -1), ref) < TOL[torch.bfloat16]
    out = torch.empty(bsz, 1, 1024, dtype = torch.bfloat16, device = "cuda")
    for _ in range(3):
        assert torch.equal(utils.fast_linear_forward(proj, X, out = out), first)
    import copy

    assert getattr(copy.deepcopy(proj.weight.quant_state), "_unsloth_gemv_rows", None) is None


def test_planned_rows_skip_a_strided_out(monkeypatch):
    utils = _utils_with_kernels()
    monkeypatch.setattr(utils, "_gemv_rows_cap", lambda device: 4)
    proj = _linear4bit(1024, 4096)
    X = torch.randn(2, 1, 4096, dtype = torch.bfloat16, device = "cuda")
    ref = utils.fast_linear_forward(proj, X)
    wide = torch.zeros(2, 1, 2048, dtype = torch.bfloat16, device = "cuda")
    utils.fast_linear_forward(proj, X, out = wide[:, :, :1024])
    strided = wide[:, :, ::2]
    assert strided.stride(0) == 2048 and strided.stride(-1) == 2
    assert (
        _rel_err(utils.fast_linear_forward(proj, X, out = strided).float(), ref.float())
        < TOL[torch.bfloat16]
    )


def test_one_row_into_a_taller_out_writes_only_that_row():
    q, s = _quant(1024, 4096, torch.bfloat16, nested = True)
    X = torch.randn(1, 1, 4096, dtype = torch.bfloat16, device = "cuda")
    tall = torch.zeros(4, 1, 1024, dtype = torch.bfloat16, device = "cuda")
    gemv_nf4(X, *_args(q, s), out = tall)
    torch.cuda.synchronize()
    assert torch.equal(tall[:1], gemv_nf4(X, *_args(q, s)))
    assert not tall[1:].any()


def test_planned_rows_are_reused_without_an_out(monkeypatch):
    # fast_geglu_inference calls the gate and up projections with out = None.
    utils = _utils_with_kernels()
    from unsloth.kernels import triton_launch

    if not triton_launch._ENABLED:
        pytest.skip("plans are kept only with the direct launcher (off on HIP)")
    monkeypatch.setattr(utils, "_gemv_rows_cap", lambda device: 4)
    proj = _linear4bit(1024, 4096)
    W, W_quant = utils.get_lora_parameters_bias(proj)[:2]
    X = torch.randn(3, 1, 4096, dtype = torch.bfloat16, device = "cuda")
    first = utils._gemv_rows(X, W, W_quant, None)
    monkeypatch.setattr(utils, "_unpack_quant_state", lambda *a: pytest.fail("replanned"))
    again = utils._gemv_rows(X, W, W_quant, None)
    assert again.data_ptr() != first.data_ptr() and torch.equal(again, first)


def test_rows_gemv_rejects_an_out_it_cannot_fill(monkeypatch):
    utils = _utils_with_kernels()
    monkeypatch.setattr(utils, "_gemv_rows_cap", lambda device: 4)
    proj = _linear4bit(1024, 4096)
    W, W_quant = utils.get_lora_parameters_bias(proj)[:2]
    X = torch.randn(2, 1, 4096, dtype = torch.bfloat16, device = "cuda")
    good = torch.empty(2, 1, 1024, dtype = torch.bfloat16, device = "cuda")
    assert utils._gemv_rows(X, W, W_quant, good) is good
    for bad in (
        torch.empty(2, 1, 1023, dtype = torch.bfloat16, device = "cuda"),
        torch.empty(1, 1, 1024, dtype = torch.bfloat16, device = "cuda"),
        torch.empty(2, 0, 1024, dtype = torch.bfloat16, device = "cuda"),
        torch.empty(2, 2, 1024, dtype = torch.bfloat16, device = "cuda"),
        torch.empty(2048, dtype = torch.bfloat16, device = "cuda").as_strided(
            (2, 1, 1024), (512, 512, 1)
        ),
    ):
        assert utils._gemv_rows(X, W, W_quant, bad) is None
    assert utils._gemv_rows(X, W, W_quant, torch.empty(2, 1, 1024, dtype = torch.bfloat16)) is None
    for wrong_k in (4095, 4097):
        assert (
            utils._gemv_rows(
                torch.randn(2, 1, wrong_k, dtype = torch.bfloat16, device = "cuda"), W, W_quant, None
            )
            is None
        )


def test_rows_gemv_needs_the_weight_on_the_activation_device(monkeypatch):
    utils = _utils_with_kernels()
    monkeypatch.setattr(utils, "_gemv_rows_cap", lambda device: 4)
    proj = _linear4bit(1024, 4096)
    W, W_quant = utils.get_lora_parameters_bias(proj)[:2]
    X = torch.randn(2, 1, 4096, dtype = torch.bfloat16, device = "cuda")
    # Triton rejects a CPU pointer but not another GPU's: the guard must stop the launch.
    for name in ("gemv_nf4_rows", "gemv_nf4_rows_planned"):
        monkeypatch.setattr(utils, name, lambda *a: pytest.fail("launched"))
    assert utils._gemv_rows(X, W.data.cpu(), W_quant, None) is None
    absmax = W_quant.absmax
    monkeypatch.setattr(W_quant, "absmax", absmax.cpu())
    assert utils._gemv_rows(X, W, W_quant, None) is None


def test_empty_decode_batch_keeps_the_matmul_path(monkeypatch):
    utils = _utils_with_kernels()
    monkeypatch.setattr(utils, "_gemv_rows_cap", lambda device: 4)
    proj = _linear4bit(1024, 4096)
    monkeypatch.setattr(utils, "_gemv_rows", lambda *a: pytest.fail("rows GEMV for 0 rows"))
    X = torch.randn(0, 1, 4096, dtype = torch.bfloat16, device = "cuda")
    assert utils.fast_linear_forward(proj, X).shape == (0, 1, 1024)


def test_larger_decode_batches_still_dequantize(monkeypatch):
    utils = _utils_with_kernels()
    proj = _linear4bit(1024, 4096)
    X = torch.randn(utils._GEMV_MAX_ROWS + 1, 1, 4096, dtype = torch.bfloat16, device = "cuda")
    calls = []
    real = utils.fast_dequantize
    monkeypatch.setattr(utils, "fast_dequantize", lambda *a, **k: calls.append(1) or real(*a, **k))
    ref = (
        X.float().view(X.shape[0], -1)
        @ F.dequantize_4bit(proj.weight.data, proj.weight.quant_state).float().t()
    )
    assert (
        _rel_err(utils.fast_linear_forward(proj, X).view(X.shape[0], -1), ref) < TOL[torch.bfloat16]
    )
    assert calls


@pytest.mark.parametrize(
    "arch, env, wins",
    [
        ((7, 5), None, {2: True, 3: True, 4: True}),
        ((8, 9), None, {2: True, 4: True}),
        ((12, 0), None, {2: True, 4: True}),
        ((8, 0), None, {2: True, 3: False, 4: False}),
        ((9, 0), None, {2: False, 4: False}),
        ((10, 0), None, {2: False, 4: False}),
        ((10, 0), "3", {2: True, 3: True}),
        ((7, 5), "1", {2: False}),
    ],
)
def test_rows_or_dequantize_is_fixed_per_architecture(monkeypatch, arch, env, wins):
    utils = _utils_with_kernels()
    monkeypatch.setattr(utils, "_GEMV_ROWS_CAP", {})
    monkeypatch.setattr(utils, "_GEMV_ROWS_BNB_FROM", {})
    if env is None:
        monkeypatch.delenv("UNSLOTH_GEMV_MAX_ROWS", raising = False)
    else:
        monkeypatch.setenv("UNSLOTH_GEMV_MAX_ROWS", env)
        monkeypatch.setattr(utils, "_GEMV_MAX_ROWS", int(env))
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: arch)
    proj = _linear4bit(1024, 4096)
    W, W_quant = utils.get_lora_parameters_bias(proj)[:2]
    for _ in range(2):
        for bsz, win in wins.items():
            X = torch.randn(bsz, 1, 4096, dtype = torch.bfloat16, device = "cuda")
            assert (utils._gemv_rows(X, W, W_quant, None) is not None) is win


def test_rows_above_the_cap_skip_straight_to_dequantize(monkeypatch):
    # A100 batch 4 decoded 10% slower when a strided X was unpacked and copied before the cap said no.
    utils = _utils_with_kernels()
    monkeypatch.setattr(utils, "_gemv_rows_cap", lambda device: 2)
    monkeypatch.setattr(utils, "_unpack_quant_state", lambda *a: pytest.fail("unpacked"))
    proj = _linear4bit(1024, 4096)
    W, W_quant = utils.get_lora_parameters_bias(proj)[:2]
    X = torch.randn(4, 1, 8192, dtype = torch.bfloat16, device = "cuda")[:, :, :4096]
    assert not X.is_contiguous()
    assert utils._gemv_rows(X, W, W_quant, None) is None


def test_rocm_never_takes_the_rows_gemv_by_default(monkeypatch):
    utils = _utils_with_kernels()
    monkeypatch.setattr(utils, "_GEMV_ROWS_CAP", {})
    monkeypatch.setattr(utils, "_GEMV_ROWS_BNB_FROM", {})
    monkeypatch.delenv("UNSLOTH_GEMV_MAX_ROWS", raising = False)
    monkeypatch.setattr(torch.version, "hip", "6.4")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (12, 0))
    proj = _linear4bit(1024, 4096)
    W, W_quant = utils.get_lora_parameters_bias(proj)[:2]
    X = torch.randn(2, 1, 4096, dtype = torch.bfloat16, device = "cuda")
    assert utils._gemv_rows(X, W, W_quant, None) is None


def test_small_tiles_keep_the_table_load():
    gather = 2 if nf4_gemv._HAS_TL_GATHER else 0
    assert nf4_gemv._lut_mode(64) == 0
    assert nf4_gemv._lut_mode(256) == 0
    assert nf4_gemv._lut_mode(512) == gather
    assert nf4_gemv._lut_mode(2048) == gather


@pytest.mark.parametrize("rows", [1, 2, 4])
@pytest.mark.parametrize("dtype", DTYPES, ids = str)
@pytest.mark.parametrize("shape", [(1000, 192), (2048, 256), (2048, 512), (1536, 8960)], ids = str)
def test_small_and_odd_k_tiles_match_fp32(shape, dtype, rows, kernel):
    """K <= 256 (table load), K = 512 (gather) and an odd K = 8960 (masked tail)."""
    from unsloth.kernels.nf4_gemv import gemv_nf4_rows

    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("no bf16")
    n, k = shape
    q, s = _quant(n, k, dtype)
    a = _args(q, s)
    X = torch.randn(rows, 1, k, dtype = dtype, device = "cuda")
    ref = X.float().view(rows, k) @ F.dequantize_4bit(q, s).float().t()
    if rows == 1:
        out = gemv_nf4(X, *a).view(1, n)
    else:
        out = torch.empty(rows, 1, n, dtype = dtype, device = "cuda")
        gemv_nf4_rows(X, *a[:-1], out)
    assert _rel_err(out.view(rows, n), ref) < TOL[dtype]


def _bnb_gemm_utils():
    from unsloth.kernels import utils
    if not hasattr(torch.ops.bitsandbytes, "gemm_4bit") or torch.version.hip:
        pytest.skip("bitsandbytes has no fused gemm_4bit here")
    return utils


@pytest.mark.parametrize("bsz", [4, 5, 8, 16])
def test_decode_rows_above_the_gemv_cap_use_the_bnb_gemm(bsz, monkeypatch):
    utils = _bnb_gemm_utils()
    proj = _linear4bit(1024, 4096, dtype = torch.float16)
    X = torch.randn(bsz, 1, 4096, dtype = torch.float16, device = "cuda")
    ref = (
        X.float().view(bsz, -1)
        @ F.dequantize_4bit(proj.weight.data, proj.weight.quant_state).float().t()
    )
    monkeypatch.setattr(utils, "_gemv_rows_cap", lambda device: 4)
    # T4: batch 4 is within the GEMV cap but handed to bitsandbytes.
    monkeypatch.setattr(utils, "_GEMV_ROWS_BNB_FROM", {X.device.index: 4})
    monkeypatch.setattr(utils, "_BNB_GEMM_4BIT", True)
    monkeypatch.setattr(utils, "_bnb_gemm_4bit_device_ok", lambda device: True)
    monkeypatch.setattr(utils, "fast_dequantize", lambda *a, **k: pytest.fail("dequantized"))
    calls = []
    real = utils.bnb.matmul_4bit
    monkeypatch.setattr(utils.bnb, "matmul_4bit", lambda *a, **k: calls.append(1) or real(*a, **k))
    with torch.no_grad():
        first = utils.fast_linear_forward(proj, X)
        assert first.shape == (bsz, 1, 1024)
        assert _rel_err(first.view(bsz, -1), ref) < TOL[torch.float16]
        out = torch.empty(bsz, 1, 1024, dtype = torch.float16, device = "cuda")
        assert utils.fast_linear_forward(proj, X, out = out) is out
        assert torch.equal(out, first)
    assert len(calls) == 2


@pytest.mark.parametrize(
    "case",
    [
        "switch_off",
        "other_arch",
        "too_many_rows",
        "grad",
        "bf16",
        "weight_elsewhere",
        "unaligned_k",
    ],
)
def test_the_bnb_gemm_is_skipped_where_it_does_not_apply(case, monkeypatch):
    utils = _bnb_gemm_utils()
    bsz = utils._BNB_GEMM_4BIT_MAX_ROWS + 1 if case == "too_many_rows" else 8
    dtype = torch.bfloat16 if case == "bf16" else torch.float16
    proj = _linear4bit(1024, 4096, dtype = dtype)
    X = torch.randn(bsz, 1, 4096, dtype = dtype, device = "cuda")
    monkeypatch.setattr(utils, "_gemv_rows_cap", lambda device: 4)
    monkeypatch.setattr(utils, "_BNB_GEMM_4BIT", case != "switch_off")
    if case == "other_arch":
        monkeypatch.setattr(utils, "_BNB_GEMM_4BIT_ARCHS", set())
        monkeypatch.setattr(utils, "_BNB_GEMM_4BIT_DEVICE", {})
    else:
        monkeypatch.setattr(utils, "_bnb_gemm_4bit_device_ok", lambda device: True)
    if case == "unaligned_k":
        # K not a multiple of the 64 blocksize.
        proj = _linear4bit(1024, 4128, dtype = dtype)
        X = torch.randn(bsz, 1, 4128, dtype = dtype, device = "cuda")
    if case == "weight_elsewhere":
        assert not utils._bnb_gemm_4bit_ok(
            X.view(bsz, -1), proj.weight.data.cpu(), proj.weight.quant_state
        )
        return
    calls = []
    real = utils.fast_dequantize
    monkeypatch.setattr(utils, "fast_dequantize", lambda *a, **k: calls.append(1) or real(*a, **k))
    with torch.set_grad_enabled(case == "grad"):
        utils.fast_linear_forward(proj, X)
    assert calls, case


@pytest.mark.parametrize(
    "case", ["eligible", "switch_off", "grad", "bf16", "compiling", "env_override", "other_arch"]
)
def test_t4_batch_4_leaves_the_gemv_only_for_a_bnb_gemm_that_runs(case, monkeypatch):
    """T4 batch 4 takes the bitsandbytes gemm_4bit; wherever that branch would not run, it keeps the
    GEMV instead of falling to dequantize."""
    utils = _utils_with_kernels()
    _bnb_gemm_utils()
    monkeypatch.setattr(utils, "_GEMV_ROWS_CAP", {})
    monkeypatch.setattr(utils, "_GEMV_ROWS_BNB_FROM", {})
    monkeypatch.setattr(utils, "_GEMV_MAX_ROWS", 4)
    monkeypatch.setattr(torch.version, "hip", None)
    arch = (8, 0) if case == "other_arch" else (7, 5)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: arch)
    if case == "env_override":
        monkeypatch.setenv("UNSLOTH_GEMV_MAX_ROWS", "4")
    else:
        monkeypatch.delenv("UNSLOTH_GEMV_MAX_ROWS", raising = False)
    monkeypatch.setattr(utils, "_BNB_GEMM_4BIT", case != "switch_off")
    monkeypatch.setattr(utils, "_bnb_gemm_4bit_device_ok", lambda device: True)
    if case == "compiling":
        monkeypatch.setattr(utils, "_is_compiling", lambda: True)
    dtype = torch.bfloat16 if case == "bf16" else torch.float16
    proj = _linear4bit(1024, 4096, dtype = dtype)
    W, W_quant = utils.get_lora_parameters_bias(proj)[:2]
    with torch.set_grad_enabled(case == "grad"):
        rows = {
            bsz: utils._gemv_rows(
                torch.randn(bsz, 1, 4096, dtype = dtype, device = "cuda"), W, W_quant, None
            )
            is not None
            for bsz in (2, 3, 4)
        }
    if case == "other_arch":
        assert rows == {2: True, 3: False, 4: False}
    else:
        assert rows == {2: True, 3: True, 4: case != "eligible"}, case


@pytest.mark.parametrize("env", [None, "0", "1"])
def test_the_bnb_gemm_runs_only_on_listed_archs_unless_forced(env, monkeypatch):
    utils = _bnb_gemm_utils()
    monkeypatch.setattr(utils, "_BNB_GEMM_4BIT", None)
    monkeypatch.setattr(utils, "_BNB_GEMM_4BIT_DEVICE", {})
    monkeypatch.setattr(utils, "_BNB_GEMM_4BIT_DTYPES", utils._BNB_GEMM_4BIT_DTYPES)
    if env is None:
        monkeypatch.delenv("UNSLOTH_BNB_GEMM_4BIT", raising = False)
    else:
        monkeypatch.setenv("UNSLOTH_BNB_GEMM_4BIT", env)
    proj = _linear4bit(1024, 4096, dtype = torch.float16)
    X = torch.randn(8, 4096, dtype = torch.float16, device = "cuda")
    listed = [
        torch.cuda.get_device_capability(i) in utils._BNB_GEMM_4BIT_ARCHS
        for i in range(torch.cuda.device_count())
    ]
    with torch.no_grad():
        ok = utils._bnb_gemm_4bit_ok(X, proj.weight.data, proj.weight.quant_state)
    # Decided once for the process: off when no visible GPU is listed, so the decode loop skips the check.
    assert utils._BNB_GEMM_4BIT == (env != "0" and (env == "1" or any(listed)))
    assert ok == (env != "0" and (env == "1" or listed[X.device.index]))
    assert (torch.bfloat16 in utils._BNB_GEMM_4BIT_DTYPES) == (env == "1")


@pytest.mark.parametrize(
    "hip, triton_version, env, eager",
    [
        (None, (3, 2), None, True),
        (None, (3, 6), None, True),
        (None, (3, 7), None, True),
        (None, (3, 6), "0", False),
        (None, (3, 6), "1", True),
        ("7.1", (3, 4), None, False),
        ("7.1", (3, 5), None, False),
        ("7.1", (3, 6), None, True),
        ("7.1", (3, 7), None, True),
        ("7.1", (3, 6), "0", False),
        ("7.1", (3, 5), "1", True),
        ("7.1", (3, 5), "yes", False),
    ],
)
def test_eager_gemv_gate_per_platform(monkeypatch, hip, triton_version, env, eager):
    """CUDA always; HIP from Triton 3.6; UNSLOTH_TRITON_GEMV_EAGER=0 / =1 overrides, others ignored."""
    monkeypatch.setattr(torch.version, "hip", hip)
    monkeypatch.setattr(nf4_gemv, "_TRITON_VERSION", triton_version)
    if env is None:
        monkeypatch.delenv("UNSLOTH_TRITON_GEMV_EAGER", raising = False)
    else:
        monkeypatch.setenv("UNSLOTH_TRITON_GEMV_EAGER", env)
    assert nf4_gemv.triton_gemv_eager() is eager


@pytest.mark.parametrize("env", [None, "0"], ids = ["default", "switch_off"])
def test_rocm_batch_1_decode_takes_the_triton_gemv(monkeypatch, env):
    """A faked ROCm torch with Triton 3.6 (Triton's own launch path, as on HIP): batch-1 decode
    takes the Triton GEMV unless switched off, and decode rows still dequantize."""
    from unsloth.kernels import triton_launch, utils

    if not utils._USE_NF4_KERNELS:
        pytest.skip("NF4 kernels unavailable or disabled")
    monkeypatch.setattr(torch.version, "hip", "7.1")
    monkeypatch.setattr(nf4_gemv, "_TRITON_VERSION", (3, 6))
    monkeypatch.setattr(triton_launch, "_ENABLED", False)
    if env is None:
        monkeypatch.delenv("UNSLOTH_TRITON_GEMV_EAGER", raising = False)
    else:
        monkeypatch.setenv("UNSLOTH_TRITON_GEMV_EAGER", env)
    monkeypatch.delenv("UNSLOTH_GEMV_MAX_ROWS", raising = False)
    monkeypatch.setattr(utils, "_TRITON_GEMV_EAGER", nf4_gemv.triton_gemv_eager())
    monkeypatch.setattr(utils, "_GEMV_ROWS_CAP", {})
    monkeypatch.setattr(utils, "_GEMV_ROWS_BNB_FROM", {})
    calls = []
    for name in ("_fast_gemv_ctypes", "gemv_nf4_planned", "gemv_nf4"):
        real = getattr(utils, name)
        monkeypatch.setattr(
            utils, name, lambda *a, _r = real, _n = name, **k: calls.append(_n) or _r(*a, **k)
        )
    proj = _linear4bit(1024, 4096)
    q, s = proj.weight.data, proj.weight.quant_state
    X = torch.randn(1, 1, 4096, dtype = torch.bfloat16, device = "cuda")
    triton_out = gemv_nf4(X, *_args(q, s))
    bnb_out = _fast_gemv_ctypes(X, q, s)
    for _ in range(3):
        calls.clear()
        got = utils.fast_linear_forward(proj, X)
        if env is None:
            assert calls == ["gemv_nf4_planned"] and torch.equal(got, triton_out)
        else:
            assert calls == ["_fast_gemv_ctypes"] and torch.equal(got, bnb_out)
    assert getattr(s, "_unsloth_gemv_plan", None) is None  # no direct relaunch on HIP
    X2 = torch.randn(2, 1, 4096, dtype = torch.bfloat16, device = "cuda")
    assert utils._gemv_rows(X2, q, s, None) is None


@pytest.mark.parametrize("nested", [True, False], ids = ["nested", "flat"])
@pytest.mark.parametrize("dtype", DTYPES, ids = str)
@pytest.mark.parametrize("shape", SHAPES + [(12288, 4096), (256, 256)], ids = str)
def test_eager_decode_route_matches_dequantize_matmul(shape, dtype, nested):
    """The route eager batch-1 decode takes on this GPU (on ROCm with Triton 3.6+, the Triton
    GEMV) against dequantize + fp32 matmul, on the Llama 3 8B / Qwen3 8B decode shapes. Reruns
    of a weight's planned launch must give the same result."""
    from unsloth.kernels import utils

    if torch.version.hip and nf4_gemv._TRITON_VERSION >= nf4_gemv._HIP_EAGER_MIN_TRITON:
        assert utils._USE_NF4_KERNELS and utils._TRITON_GEMV_EAGER, "ROCm decode kept bitsandbytes"
    if not nested and not (utils._USE_NF4_KERNELS and utils._TRITON_GEMV_EAGER):
        pytest.skip("the ctypes GEMV only takes double quantized states")
    n, k = shape
    q, s = _quant(n, k, dtype, nested)
    X = torch.randn(1, 1, k, dtype = dtype, device = "cuda")
    ref = (X.float().view(1, k) @ utils.fast_dequantize(q, s).float().t()).view(1, 1, n)
    got = utils.fast_gemv(X, q, s)
    assert got.shape == (1, 1, n) and got.dtype == dtype
    assert _rel_err(got, ref) < TOL[dtype]
    for _ in range(2):
        assert torch.equal(utils.fast_gemv(X, q, s), got)
