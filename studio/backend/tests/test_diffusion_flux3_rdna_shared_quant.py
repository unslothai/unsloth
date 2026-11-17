# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""FLUX.3 explicit int8 on ROCm RDNA: native W8A8 Linears share one rotation + act quant per block
(``diffusion_int8_gemm.shared_rotq_linears`` under ``diffusion_flux_convrot_fused``), bit-identical to each Linear's own
forward, and the explicit request picks W8A8 for the measured family only. CPU: the RDNA kernels are replaced by their
eager references (``reference_rotq`` / ``reference_mm``), counted per call."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import core.inference.diffusion_int8_gemm as gemm
import core.inference.diffusion_native_quant as nq
import core.inference.diffusion_native_w8a8_rdna as rdna
import core.inference.diffusion_transformer_quant as tq


@pytest.fixture()
def rdna_cpu(monkeypatch):
    """Native int8 Linears engage the RDNA path on CPU; the kernels are the eager references, counted."""
    calls = {"rotq": 0, "mm": 0}

    def rotq(x2, group):
        calls["rotq"] += 1
        return rdna.reference_rotq(x2, group)

    def mm(xq, xs, wq, ws, bias, out_dtype):
        calls["mm"] += 1
        return rdna.reference_mm(xq, xs, wq, ws, bias, out_dtype)

    monkeypatch.setattr(
        rdna, "_STATE", {"devices": {}, "on": {None: True}, "broken": None, "rotq": rotq, "mm": mm}
    )
    # A ROCm host probes its card here and would install the real ops over the counted references.
    monkeypatch.setattr(rdna, "prepare", lambda device = None: True)
    cls = nq.native_linear_class()
    monkeypatch.setattr(cls, "_rdna_on", lambda self, x: bool(self._rdna))
    return cls, calls


def _native(cls, lin):
    layer = cls(lin, nq.NATIVE_INT8, act_int8 = True, rot_group = 256)
    layer._rdna = True
    return layer


def test_shared_rotq_is_bit_identical_to_each_linear_with_one_quant(rdna_cpu):
    cls, calls = rdna_cpu
    torch.manual_seed(0)
    linears = tuple(
        _native(cls, torch.nn.Linear(512, n, bias = b))
        for n, b in ((512, True), (512, True), (256, False), (1024, True))
    )
    assert all(gemm.rdna_native_rotated(m) for m in linears)
    x = torch.randn(2, 40, 512)
    with torch.no_grad():
        ref = [m(x) for m in linears]
        assert calls == {"rotq": 4, "mm": 4}
        calls.update(rotq = 0, mm = 0)
        out = gemm.shared_rotq_linears(x, linears)
    assert out is not None and calls == {"rotq": 1, "mm": 4}
    for a, b in zip(ref, out):
        assert a.shape == b.shape and torch.equal(a, b)


def test_shared_rotq_declines_what_it_cannot_cover(rdna_cpu):
    cls, calls = rdna_cpu
    torch.manual_seed(0)
    a, b = _native(cls, torch.nn.Linear(512, 512)), _native(cls, torch.nn.Linear(512, 512))
    x = torch.randn(1, 8, 512)
    b._rdna = False  # off the RDNA kernels (probe failed / shape unsupported)
    assert gemm.shared_rotq_linears(x, (a, b)) is None
    b._rdna = True
    b.register_forward_pre_hook(lambda *_a: None)  # a hook the shared path would bypass
    assert gemm.shared_rotq_linears(x, (a, b)) is None
    c = _native(cls, torch.nn.Linear(768, 512))  # another input width
    assert gemm.shared_rotq_linears(x, (a, c)) is None
    weight_only = cls(torch.nn.Linear(512, 512), nq.NATIVE_INT8)
    assert not gemm.rdna_native_rotated(weight_only)
    assert calls["rotq"] == 0


@pytest.fixture()
def flux3(monkeypatch):
    pytest.importorskip("diffusers")
    f3 = pytest.importorskip("diffusers.models.transformers.transformer_flux3")
    from core.inference import diffusion_flux_convrot_fused as fused

    fused.uninstall()
    monkeypatch.delenv(fused.FLUX_CONVROT_FUSED_ENV, raising = False)
    yield f3, fused
    fused.uninstall()


def _native_block(f3, cls, kind):
    dim, heads, hd = 512, 4, 128
    torch.manual_seed(0)
    block = (
        f3.Flux3ModeTransformerBlock(dim, heads, hd)
        if kind == "mode"
        else f3.Flux3SingleTransformerBlock(dim, heads, hd)
    ).eval()
    attn, ff = block.attn, block.ff
    for owner, name in (
        (attn, "to_q"),
        (attn, "to_k"),
        (attn, "to_v"),
        (ff, "linear_in"),
        (ff, "linear_out"),
    ):
        setattr(owner, name, _native(cls, getattr(owner, name)))
    attn.to_out[0] = _native(cls, attn.to_out[0])
    return block


def _rope(seq, dim):
    g = torch.Generator().manual_seed(1)
    ang = torch.rand(seq, dim, generator = g) * 3
    return ang.cos(), ang.sin()


@pytest.mark.parametrize("kind", ["mode", "single"])
@pytest.mark.parametrize("mode", ["eager", "compiled"])
def test_flux3_block_on_native_rdna_shares_the_quant_bit_identically(rdna_cpu, flux3, kind, mode):
    cls, calls = rdna_cpu
    f3, fused = flux3
    block = _native_block(f3, cls, kind)
    dim = 512
    hs, enc = torch.randn(1, 12, dim), torch.randn(1, 5, dim)
    mod, cmod = torch.randn(1, 1, 3 * dim) * 0.1, torch.randn(1, 1, 3 * dim) * 0.1
    if kind == "mode":
        rope = _rope(12, 128)
        call = lambda b: b(hs, mod, rope)  # noqa: E731
    else:
        rope = _rope(17, 128)
        call = lambda b: b(hs, enc, mod, cmod, rope)  # noqa: E731
    with torch.no_grad():
        ref = call(block)
        assert calls["rotq"] == 6  # Q, K, V, MLP in, attention out, MLP out: one quant each
        assert fused._install_patches()["f3_mode" if kind == "mode" else "f3_single"]  # noqa: SLF001
        calls.update(rotq = 0, mm = 0)
        if mode == "compiled":
            torch._dynamo.reset()
            out = call(torch.compile(block, backend = "eager", fullgraph = True))
            torch._dynamo.reset()
        else:
            out = call(block)
    assert calls == {"rotq": 3, "mm": 6}  # Q / K / V / MLP in share one quant
    ref, out = (ref,) if torch.is_tensor(ref) else ref, (out,) if torch.is_tensor(out) else out
    assert all(torch.equal(a, b) for a, b in zip(ref, out)) and len(ref) == len(out)


def test_install_counts_native_rdna_linears_and_arms_nothing(rdna_cpu, flux3, monkeypatch):
    cls, _calls = rdna_cpu
    f3, fused = flux3
    from core.inference import diffusion_int8_fused as i8f

    def no_arming(*_a, **_k):
        raise AssertionError("no ConvRot-swapped Linear: nothing to arm")

    monkeypatch.setattr(i8f, "run_on_first_call", no_arming)

    class Flux3Transformer2DModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.block = _native_block(f3, cls, "mode")

    out = fused.install(Flux3Transformer2DModel())
    assert out["rotated"] == 6 and out["rdna_native"] == 6
    assert out["patches"].get("f3_mode") and fused.is_installed()


@pytest.fixture
def rocm(monkeypatch):
    monkeypatch.setattr(tq, "is_stubbed", lambda name: False)
    monkeypatch.setattr(tq, "torch_is_rocm", lambda: True)
    monkeypatch.delenv(nq.NATIVE_INT8_ACT_ENV, raising = False)
    monkeypatch.setattr(tq._sys, "platform", "linux")
    monkeypatch.setattr(tq, "rocm_rdna_w8a8_ready", lambda *_a: True)


def _target(device = "cuda", dtype = None):
    return types.SimpleNamespace(device = device, dtype = torch.bfloat16 if dtype is None else dtype)


def test_explicit_w8a8_ships_for_flux3_klein_only():
    assert set(tq._FAMILY_RDNA_EXPLICIT_W8A8) == {"flux.3-klein"}
    # AUTO keeps bf16 for it
    assert "flux.3-klein" not in tq._FAMILY_RDNA_AUTO_W8A8


def test_explicit_w8a8_reason_follows_the_explicit_compile_gates(rocm, monkeypatch):
    assert tq.rdna_explicit_w8a8_reason(_target(), "flux.3-klein")
    assert tq.rdna_explicit_w8a8_reason(_target(), "FLUX.3-Klein ")
    assert tq.rdna_explicit_w8a8_reason(_target(), "qwen-image-2.1") is None
    assert tq.rdna_explicit_w8a8_reason(_target(dtype = torch.float16), "flux.3-klein") is None
    assert tq.rdna_explicit_w8a8_reason(_target("cpu"), "flux.3-klein") is None
    monkeypatch.setenv(nq.NATIVE_INT8_ACT_ENV, "0")
    assert tq.rdna_explicit_w8a8_reason(_target(), "flux.3-klein") is None
    monkeypatch.delenv(nq.NATIVE_INT8_ACT_ENV)
    monkeypatch.setattr(tq, "rocm_rdna_w8a8_ready", lambda *_a: False)
    assert tq.rdna_explicit_w8a8_reason(_target(), "flux.3-klein") is None
    monkeypatch.setattr(tq, "rocm_rdna_w8a8_ready", lambda *_a: True)
    monkeypatch.setattr(tq._sys, "platform", "win32")
    assert tq.rdna_explicit_w8a8_reason(_target(), "flux.3-klein") is None


def test_explicit_w8a8_reason_is_rocm_only(monkeypatch):
    monkeypatch.setattr(tq, "is_stubbed", lambda name: False)
    monkeypatch.setattr(tq, "torch_is_rocm", lambda: False)
    monkeypatch.setattr(tq._sys, "platform", "linux")
    monkeypatch.setattr(tq, "rocm_rdna_w8a8_ready", lambda *_a: True)
    assert tq.rdna_explicit_w8a8_reason(_target(), "flux.3-klein") is None


def test_load_path_asks_for_the_explicit_w8a8_table():
    """The act_int8 decision at load reads ``rdna_explicit_w8a8_reason`` for an explicit int8 request."""
    src = (_BACKEND / "core" / "inference" / "diffusion.py").read_text(encoding = "utf-8")
    i = src.index('"act_int8": native_scheme == TQ_INT8')
    block = src[i : src.index("if native_scheme is not None", i)]
    assert "rdna_explicit_int8 is not None" in block and "rdna_explicit_w8a8_reason(" in block
