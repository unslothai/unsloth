# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Transposed int8 weight reaches cuBLASLt uncopied (row-major copy: illegal access on L4, M=4384 K=7168 N=5376)."""

from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")

from core.inference import diffusion_torchao_patches as patches  # noqa: E402


def _patched(monkeypatch, env = None):
    seen = []

    def out_dtype(op, dtype, a, b):
        seen.append(b)
        return torch.empty(a.shape[0], b.shape[1], dtype = dtype, device = a.device)

    def original(input, mat2):  # pragma: no cover - never called
        raise AssertionError

    module = types.SimpleNamespace(out_dtype = out_dtype, dynamo_is_compiling = lambda: False)
    if env is None:
        monkeypatch.delenv(patches._TORCHAO_INT_MM_TRANSPOSED_ENV, raising = False)
    else:
        monkeypatch.setenv(patches._TORCHAO_INT_MM_TRANSPOSED_ENV, env)
    return patches._make_safe_int_mm(module, original), seen


def _operands(
    m = 64,
    k = 128,
    n = 96,
):
    x = torch.empty(m, k, dtype = torch.int8, device = "meta")
    w = torch.empty(n, k, dtype = torch.int8, device = "meta")  # torchao qdata [N, K]
    return x, w


def test_transposed_weight_reaches_the_gemm_uncopied(monkeypatch):
    fn, seen = _patched(monkeypatch)
    x, w = _operands()
    fn(x, w.t())
    assert seen[-1].stride() == (1, 128), "the column-major weight was copied row-major"


def test_switch_off_restores_the_row_major_copy(monkeypatch):
    fn, seen = _patched(monkeypatch, env = "0")
    x, w = _operands()
    fn(x, w.t())
    assert seen[-1].is_contiguous()


def test_other_non_contiguous_weights_are_still_copied(monkeypatch):
    fn, seen = _patched(monkeypatch)
    x, _ = _operands()
    sliced = torch.empty(128, 192, dtype = torch.int8, device = "meta")[
        :, ::2
    ]  # neither layout: must be copied
    fn(x, sliced)
    assert seen[-1].is_contiguous()
    row_major = torch.empty(128, 96, dtype = torch.int8, device = "meta")
    fn(x, row_major)
    assert seen[-1] is row_major


def test_rocm_keeps_the_row_major_copy(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", "6.4", raising = False)
    fn, seen = _patched(monkeypatch)
    x, w = _operands()
    fn(x, w.t())
    assert seen[-1].is_contiguous()


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a CUDA GPU (cuBLASLt int8)")
@pytest.mark.parametrize(
    "m,k,n", [(4384, 7168, 5376), (4128, 5376, 7168), (17, 64, 32), (1024, 14336, 5376)]
)
def test_cuda_transposed_int_mm_is_exact(monkeypatch, m, k, n):
    import importlib

    intmm = None
    for name in patches._TORCHAO_INTMM_MODULES:
        try:
            candidate = importlib.import_module(name)
        except ImportError:
            continue
        if callable(getattr(candidate, "out_dtype", None)) and callable(
            getattr(candidate, "safe_int_mm", None)
        ):
            intmm = candidate
            break
    if intmm is None:
        pytest.skip("torchao has no safe_int_mm here")
    monkeypatch.delenv(patches._TORCHAO_INT_MM_TRANSPOSED_ENV, raising = False)
    fn = patches._make_safe_int_mm(
        intmm, getattr(intmm.safe_int_mm, "__unsloth_original__", intmm.safe_int_mm)
    )
    g = torch.Generator(device = "cuda").manual_seed(0)
    x = torch.randint(-127, 128, (m, k), dtype = torch.int8, device = "cuda", generator = g)
    w = torch.randint(-127, 128, (n, k), dtype = torch.int8, device = "cuda", generator = g)
    got = fn(x, w.t())
    torch.cuda.synchronize()
    ref = (x.double() @ w.double().t()).to(torch.int64)
    assert torch.equal(got.to(torch.int64), ref)
