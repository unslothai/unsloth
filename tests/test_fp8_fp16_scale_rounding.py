# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""Per-tensor / rowwise FP8 dequant to fp16 rounds once: an fp16 scale underflows or overflows."""

import os

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")


@pytest.fixture(scope = "module")
def F():
    os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")
    from unsloth.kernels import fp8
    return fp8


def _q(value, shape = (2, 4)):
    return torch.full(shape, value).to(torch.float8_e4m3fn)


def _oracle(q, s, dtype):
    return (q.double() * s.double()).to(dtype)


@pytest.mark.parametrize("layout", ["tensor", "row", "row_t"])
@pytest.mark.parametrize("value, scale", [(448.0, 1e-8), (0.001953125, 1e5), (448.0, 1.234567e-4)])
def test_fp16_scalar_and_row_scales_round_once(F, layout, value, scale):
    q = _q(value)
    if layout == "tensor":
        s = torch.tensor(scale, dtype = torch.float32)
        x, ref_s = q, s
    elif layout == "row":
        s = torch.full((2, 1), scale, dtype = torch.float32)
        x, ref_s = q, s
    else:
        s = torch.full((2, 1), scale, dtype = torch.float32)
        x, ref_s = q.t(), s.view(1, 2)
    out = F.weight_dequant(x, s, torch.float16)
    ref = _oracle(x, ref_s, torch.float16)
    assert out.dtype == torch.float16
    assert torch.isfinite(ref).all() and (ref != 0).all()
    assert torch.equal(out, ref)


def test_fp16_rowwise_small_amax_error(F):
    torch.manual_seed(0)
    w = torch.randn(64, 256) * 0.001 / 4
    w[:, 0] = 0.001
    s = (w.abs().amax(1, keepdim = True) / 448).float()
    q = (w / s).to(torch.float8_e4m3fn)
    ref = q.double() * s.double()
    out = F.weight_dequant(q, s, torch.float16)
    assert torch.equal(out, ref.half())


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("layout", ["tensor", "row", "row_t"])
def test_bf16_fp32_unchanged(F, dtype, layout):
    torch.manual_seed(1)
    w = torch.randn(32, 48) * torch.linspace(1e-4, 1.0, 32)[:, None]
    s = (w.abs().amax(1, keepdim = True) / 448).float()
    q = (w / s).to(torch.float8_e4m3fn)
    if layout == "tensor":
        s = s.amax().reshape(())
        expected = q.to(dtype) * s.view(1, 1).to(dtype)
        out = F.weight_dequant(q, s, dtype)
    elif layout == "row":
        expected = q.to(dtype) * s.to(dtype)
        out = F.weight_dequant(q, s, dtype)
    else:
        expected = (q.to(dtype) * s.to(dtype)).t()
        out = F.weight_dequant(q.t().contiguous(), s, dtype)
    assert out.dtype == dtype
    assert torch.equal(out, expected)
