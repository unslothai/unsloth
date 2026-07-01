# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Numerical + lifecycle tests for the shared eager speedup patches.

Builds the REAL diffusers 0.38 modules, captures the stock output, installs the patches,
and asserts the patched output matches within tolerance (fp32 on CPU always; bf16 on CUDA
when available). Also checks install/uninstall reversibility + idempotency, the
signature-guard no-op, and that a patched block compiles ``fullgraph=True`` (no graph break).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")

import torch.nn as nn  # noqa: E402

from core.inference import diffusion_eager_patches as ep  # noqa: E402
from diffusers.models.normalization import (  # noqa: E402
    AdaLayerNormContinuous,
    AdaLayerNormZero,
    AdaLayerNormZeroSingle,
    RMSNorm,
)

B, S, D, COND = 2, 16, 64, 32


@pytest.fixture(autouse = True)
def _clean_patches():
    ep.uninstall_patches()
    yield
    ep.uninstall_patches()


def _devices_dtypes():
    cases = [("cpu", torch.float32)]
    if torch.cuda.is_available():
        cases.append(("cuda", torch.bfloat16))
    return cases


def _build(cls, device, dtype):
    torch.manual_seed(0)
    if cls is RMSNorm:
        m = RMSNorm(D, eps = 1e-6, elementwise_affine = True)
    elif cls is AdaLayerNormContinuous:
        m = AdaLayerNormContinuous(
            D, COND, elementwise_affine = False, eps = 1e-6, norm_type = "layer_norm"
        )
    elif cls is AdaLayerNormZero:
        m = AdaLayerNormZero(D, num_embeddings = None, norm_type = "layer_norm")
    elif cls is AdaLayerNormZeroSingle:
        m = AdaLayerNormZeroSingle(D, norm_type = "layer_norm")
    return m.to(device = device, dtype = dtype).eval()


def _inputs(cls, device, dtype):
    torch.manual_seed(1)
    x = torch.randn(B, S, D, device = device, dtype = dtype)
    if cls is RMSNorm:
        return (x,)
    if cls is AdaLayerNormContinuous:
        return (x, torch.randn(B, COND, device = device, dtype = dtype))
    # AdaLayerNormZero / Single take the conditioning emb of width D
    return (x, torch.randn(B, D, device = device, dtype = dtype))


def _call(cls, m, args):
    if cls is AdaLayerNormZero:
        return m(args[0], emb = args[1])
    return m(*args)


def _first(out):
    return out[0] if isinstance(out, tuple) else out


@pytest.mark.parametrize(
    "cls", [RMSNorm, AdaLayerNormContinuous, AdaLayerNormZero, AdaLayerNormZeroSingle]
)
@pytest.mark.parametrize("device,dtype", _devices_dtypes())
def test_patched_matches_original(cls, device, dtype):
    m = _build(cls, device, dtype)
    args = _inputs(cls, device, dtype)

    with torch.inference_mode():
        ref = _first(_call(cls, m, args)).clone()

    assert ep.install_compile_safe_patches() >= 1
    with torch.inference_mode():
        got = _first(_call(cls, m, args))

    # The fused ops are FMA-based (addcmul) / fused (F.rms_norm): within ~1 ULP of the
    # stock mul+add (and more accurate, single rounding), NOT bit-identical in fp32.
    atol, rtol = (1e-5, 1e-4) if dtype == torch.float32 else (8e-3, 8e-3)
    torch.testing.assert_close(got, ref, atol = atol, rtol = rtol)


def test_rmsnorm_mixed_dtype_falls_back():
    """fp32 activations into a bf16-weight RMSNorm: diffusers reduces variance in fp32 from
    the original tensor, so the fused path must FALL BACK (identical output, not divergent)."""
    m = RMSNorm(D, eps = 1e-6, elementwise_affine = True).to(torch.bfloat16).eval()
    x = torch.randn(B, S, D, dtype = torch.float32)
    with torch.inference_mode():
        ref = m(x).clone()
    ep.install_compile_safe_patches()
    with torch.inference_mode():
        got = m(x)
    torch.testing.assert_close(got, ref, atol = 0.0, rtol = 0.0)  # exact fallback


def test_rmsnorm_tuple_dim_falls_back():
    """diffusers RMSNorm always reduces the LAST dim even for a tuple `dim`; F.rms_norm
    would reduce all of them, so a multi-dim `dim` must FALL BACK to the original."""
    m = RMSNorm((2, D), eps = 1e-6, elementwise_affine = True).eval()
    x = torch.randn(B, 2, D)
    with torch.inference_mode():
        ref = m(x).clone()
    ep.install_compile_safe_patches()
    with torch.inference_mode():
        got = m(x)
    torch.testing.assert_close(got, ref, atol = 0.0, rtol = 0.0)  # exact fallback


def test_install_idempotent_and_reversible():
    rms = RMSNorm(D, eps = 1e-6)
    orig = RMSNorm.forward
    n1 = ep.install_compile_safe_patches()
    n2 = ep.install_compile_safe_patches()  # second call is a no-op
    assert n1 >= 1 and n2 == n1
    assert RMSNorm.forward is not orig
    assert ep.is_installed()
    ep.uninstall_patches()
    assert RMSNorm.forward is orig  # exact restore
    assert not ep.is_installed()
    ep.uninstall_patches()  # idempotent uninstall
    del rms


def test_kill_switch_disables_patches(monkeypatch):
    monkeypatch.setenv("UNSLOTH_DIFFUSION_EAGER_PATCHES", "0")
    orig = RMSNorm.forward
    assert ep.install_compile_safe_patches() == 0  # no-op
    assert not ep.is_installed()
    assert RMSNorm.forward is orig  # untouched


def test_signature_guard_skips_changed_class(monkeypatch):
    """A diffusers class whose forward signature differs must be left untouched."""

    class WeirdRMS(nn.Module):
        def forward(self, x, extra):  # not (self, hidden_states)
            return x

    orig = WeirdRMS.forward
    monkeypatch.setattr(ep, "_RMSNorm", WeirdRMS)
    monkeypatch.setattr(ep, "_AdaLayerNormContinuous", None)
    monkeypatch.setattr(ep, "_AdaLayerNormZero", None)
    monkeypatch.setattr(ep, "_AdaLayerNormZeroSingle", None)
    applied = ep.install_compile_safe_patches()
    assert applied == 0  # nothing matched -> nothing patched
    assert WeirdRMS.forward is orig  # left untouched


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "compile graph-break check needs CUDA")
def test_no_graph_break_under_fullgraph():
    ep.install_compile_safe_patches()

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.rms = RMSNorm(D, eps = 1e-6)
            self.ada = AdaLayerNormContinuous(
                D, COND, elementwise_affine = False, norm_type = "layer_norm"
            )

        def forward(self, x, cond):
            return self.ada(self.rms(x), cond)

    m = Block().to("cuda", torch.bfloat16).eval()
    x = torch.randn(B, S, D, device = "cuda", dtype = torch.bfloat16)
    cond = torch.randn(B, COND, device = "cuda", dtype = torch.bfloat16)
    compiled = torch.compile(m, fullgraph = True)  # raises if a graph break occurs
    with torch.inference_mode():
        out = compiled(x, cond)
    assert out.shape == (B, S, D)
