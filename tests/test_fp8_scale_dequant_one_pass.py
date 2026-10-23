# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""Per-tensor / rowwise e4m3fn dequant on CUDA runs as one Triton pass with the torch expression's bits."""

import os

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None, reason = "NVIDIA CUDA only"
)


@pytest.fixture(scope = "module")
def fp8():
    os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")
    from unsloth.kernels import fp8
    return fp8


@pytest.fixture
def F(fp8, monkeypatch):
    # Small test matrices: take the kernel at any size.
    monkeypatch.setattr(fp8, "_SCALE_DEQUANT_MIN_NUMEL", 0)
    return fp8


def _reference(x, s, dtype):
    # main before the one-pass kernel: fp16 multiplies by the fp32 scale, other dtypes natively.
    if dtype == torch.float16:
        return (x.to(torch.float32) * s.to(torch.float32)).to(dtype)
    return x.to(dtype) * s.to(dtype)


def _weight(
    m,
    n,
    scale_dtype = torch.float32,
    seed = 0,
):
    g = torch.Generator(device = "cuda").manual_seed(seed)
    w = torch.randn(m, n, device = "cuda", generator = g)
    w = w * torch.logspace(-6, 2, m, device = "cuda")[:, None]
    s = (w.abs().amax(1, keepdim = True) / 448).to(scale_dtype)
    q = (w / s.float()).to(torch.float8_e4m3fn)
    # Every byte, including NaN, both zeros and subnormals, in the first rows.
    codes = torch.arange(256, device = "cuda", dtype = torch.int32).to(torch.uint8)
    q.view(torch.uint8)[: (256 + n - 1) // n].view(-1)[:256] = codes
    return q, s


def _assert_same_bits(out, ref):
    # Bit patterns, so signed zeros and subnormals count; NaN payloads may differ.
    nan = ref.isnan()
    assert nan.any() and torch.equal(out.isnan(), nan)
    bits = torch.int32 if ref.dtype == torch.float32 else torch.int16
    assert torch.equal(out.view(bits)[~nan], ref.view(bits)[~nan])


def _count_launches(F, monkeypatch):
    calls = []
    real = F._scale_dequant_one_pass
    monkeypatch.setattr(F, "_scale_dequant_one_pass", lambda *a: calls.append(1) or real(*a))
    return calls


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("scale_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("layout", ["row", "row_t", "tensor", "tensor_t", "view_t"])
def test_one_pass_matches_torch_expression(F, monkeypatch, dtype, scale_dtype, layout):
    q, s = _weight(96, 160, scale_dtype)
    if layout == "row":
        x, xs, ref = q, s, _reference(q, s, dtype)
    elif layout == "row_t":
        # weight_dequant(W.t(), s): scale per column of the view.
        x, xs, ref = q.t(), s, _reference(q, s, dtype).t()
    elif layout == "tensor":
        t = s[17:18]
        x, xs, ref = q, t, _reference(q, t.view(1, 1), dtype)
    elif layout == "tensor_t":
        t = s[17:18].view(1)
        x, xs, ref = q.t(), t, _reference(q.t(), t.view(1, 1), dtype)
    else:
        # Transposed view whose rows carry the scales.
        qt = q.t().contiguous().t()
        x, xs, ref = qt, s, _reference(qt, s, dtype)
    calls = _count_launches(F, monkeypatch)
    out = F.weight_dequant(x, xs, dtype)
    assert calls, "one-pass kernel not used"
    assert out.dtype == dtype and out.shape == ref.shape and out.stride() == ref.stride()
    _assert_same_bits(out, ref)


def test_fp16_extreme_scales_round_once(F, monkeypatch):
    calls = _count_launches(F, monkeypatch)
    q = torch.full((4, 8), 448.0, device = "cuda").to(torch.float8_e4m3fn)
    for scale in (1e-8, 1.234567e-4, 1e5):
        s = torch.full((4, 1), scale, device = "cuda")
        out = F.weight_dequant(q, s, torch.float16)
        assert torch.equal(out, (q.double() * s.double()).half())
    assert len(calls) == 3


def test_other_inputs_keep_the_torch_path(F, monkeypatch):
    calls = _count_launches(F, monkeypatch)
    q, s = _weight(32, 48)
    F.weight_dequant(q, s, torch.float64)
    F.weight_dequant(q.to(torch.float8_e5m2), s, torch.bfloat16)
    F.weight_dequant(q, s.double(), torch.bfloat16)
    F.weight_dequant(q, s.clone().requires_grad_(), torch.bfloat16)
    assert not calls


def test_trainable_fp8_input_keeps_autograd(F, monkeypatch):
    calls = _count_launches(F, monkeypatch)
    q, s = _weight(32, 48)
    x = q.detach().requires_grad_()
    out = F.weight_dequant(x, s, torch.bfloat16)
    assert out.requires_grad and not calls


def test_strided_transposed_view_keeps_the_torch_path(F, monkeypatch):
    calls = _count_launches(F, monkeypatch)
    q, s = _weight(96, 160)
    # 48 rows of a transposed (160, 96) weight: the transpose of q[:, 95:143] is not dense.
    x = q.t()[95:143]
    assert F._is_transposed_view(x) and not x.t().is_contiguous()
    sx = torch.rand(48, 1, device = "cuda")
    _assert_same_bits(F._scale_mul(x, sx, torch.float16), _reference(x, sx, torch.float16))
    assert not calls


def test_compiled_dequant_keeps_the_fused_expression(F, monkeypatch):
    calls = _count_launches(F, monkeypatch)
    q, s = _weight(64, 128)
    torch._dynamo.reset()
    fn = torch.compile(lambda x, y: F.weight_dequant(x, y, torch.float16), fullgraph = True)
    _assert_same_bits(fn(q, s), _reference(q, s, torch.float16))
    assert not calls


def test_small_weights_keep_the_torch_path(fp8, monkeypatch):
    calls = _count_launches(fp8, monkeypatch)
    q, s = _weight(2048, 2048)
    fp8.weight_dequant(q, s, torch.bfloat16)
    assert not calls
    q, s = _weight(2049, 2048)
    _assert_same_bits(fp8.weight_dequant(q, s, torch.bfloat16), _reference(q, s, torch.bfloat16))
    assert len(calls) == 1
