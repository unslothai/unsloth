# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
# LoRA backward passes W.t(); for a square W only the strides tell the scales' axis apart.
import os

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs CUDA")


@pytest.fixture(scope = "module")
def F():
    os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")
    from unsloth.kernels import fp8
    return fp8


def _rowwise(n):
    w = torch.randn(n, n, device = "cuda") * torch.linspace(0.01, 1.0, n, device = "cuda")[:, None]
    s = (w.abs().amax(1, keepdim = True) / 448).float()
    q = (w / s).to(torch.float8_e4m3fn)
    return q, s, q.float() * s


def _block(n, bs):
    torch.manual_seed(0)
    p = -(-n // bs)
    s = torch.rand(p, p, device = "cuda") * 4 + 0.1
    full = s.repeat_interleave(bs, 0)[:n].repeat_interleave(bs, 1)[:, :n]
    q = (torch.randn(n, n, device = "cuda") * 0.02 / full).to(torch.float8_e4m3fn)
    s.block_size = [bs, bs]
    return q, s, q.float() * full


@pytest.mark.parametrize("kind", ["rowwise", "block128", "block64"])
def test_transposed_view_dequantizes_stored_layout(F, kind):
    q, s, ref = (
        _rowwise(256) if kind == "rowwise" else _block(256, 128 if kind == "block128" else 64)
    )
    torch.testing.assert_close(F.weight_dequant(q, s, torch.float32), ref, rtol = 1e-2, atol = 1e-4)
    torch.testing.assert_close(
        F.weight_dequant(q.t(), s, torch.float32), ref.t(), rtol = 1e-2, atol = 1e-4
    )


@pytest.mark.parametrize("kind", ["rowwise", "block128"])
def test_fp8_linear_on_transposed_view(F, kind):
    q, s, ref = _rowwise(256) if kind == "rowwise" else _block(256, 128)
    X = torch.randn(64, 256, device = "cuda", dtype = torch.bfloat16)
    y = F.fp8_linear(X, q.t(), s)
    expect = X.float() @ ref
    assert ((y.float() - expect).norm() / expect.norm()) < 0.05
