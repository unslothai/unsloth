# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""FLUX.3 Q / K / V as one int8 GEMM (diffusion_int8_gemm.merge_payloads + diffusion_flux_convrot_fused): the
weights become row slices of one buffer (no second copy), the merged GEMM's columns equal the three GEMMs bit for
bit, and anything that is not the merged plain swapped Linear runs the three GEMMs. CPU, fake int8 weights."""

import types

import pytest

torch = pytest.importorskip("torch")

from core.inference import diffusion_flux_convrot_fused as F  # noqa: E402
from core.inference import diffusion_int8_gemm as G  # noqa: E402


class _W:
    """A plain per-row int8 weight's payload (torchao ``Int8Tensor`` stand-in)."""

    def __init__(self, qdata, scale):
        self.qdata, self.scale = qdata, scale


class _Lin(torch.nn.Module):
    def __init__(
        self,
        n,
        k,
        seed,
        fused = True,
    ):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.__dict__["weight"] = _W(
            torch.randint(-127, 128, (n, k), generator = g, dtype = torch.int8),
            (torch.rand(n, 1, generator = g) * 1e-3 + 1e-5).to(torch.bfloat16),
        )
        self.bias = None
        self.in_features, self.out_features = k, n
        self.__dict__[G._REC] = ("v2", 256, True, self.weight, fused)
        self.__dict__[G._MARK] = None
        self.__dict__["forward"] = types.MethodType(G._linear_forward, self)


class _Attn(torch.nn.Module):
    def __init__(
        self,
        n = 48,
        k = 64,
    ):
        super().__init__()
        self.to_q, self.to_k, self.to_v = _Lin(n, k, 1), _Lin(n, k, 2), _Lin(n, k, 3)


def _ref_gemm(xq, wq, xs, ws, bias):
    y = (xq.to(torch.int64) @ wq.to(torch.int64).t()).to(torch.float32)
    y = (y * xs.reshape(-1, 1).float() * ws.reshape(1, -1).float()).to(torch.bfloat16)
    return y if bias is None else (y + bias).to(torch.bfloat16)


def _gemms(
    attn,
    merged,
    extra = (),
):
    linears = (attn.to_q, attn.to_k, attn.to_v) + extra
    recs = [G._swapped_rec(m) for m in linears]
    parts = [G._payload(r[0], r[3]) for r in recs]
    g = torch.Generator().manual_seed(9)
    xq = torch.randint(-127, 128, (2 * 37, linears[0].in_features), generator = g, dtype = torch.int8)
    xs = torch.rand(2 * 37, 1, generator = g) * 1e-2
    return G._shared_gemms(xq, xs, (2, 37), xq.device, linears, recs, parts, merged)


def test_merge_shares_one_buffer_and_matches_three_gemms(monkeypatch):
    monkeypatch.setattr(G, "_OP_HANDLE", _ref_gemm)
    attn = _Attn()
    want = _gemms(attn, None, (_Lin(96, 64, 4),))
    before = [
        (m.weight.qdata.clone(), m.weight.scale.clone()) for m in (attn.to_q, attn.to_k, attn.to_v)
    ]
    merged = G.merge_payloads((attn.to_q, attn.to_k, attn.to_v))
    assert merged is not None
    wq, ws, n, fused = merged
    assert (tuple(wq.shape), tuple(ws.shape), n, fused) == ((144, 64), (144,), 3, True)
    for i, (m, (q0, s0)) in enumerate(zip((attn.to_q, attn.to_k, attn.to_v), before)):
        # the weights are row slices of the merged buffer: one copy in memory, values unchanged
        assert m.weight.qdata.untyped_storage().data_ptr() == wq.untyped_storage().data_ptr()
        assert m.weight.qdata.data_ptr() == wq[48 * i].data_ptr()
        assert m.weight.scale.untyped_storage().data_ptr() == ws.untyped_storage().data_ptr()
        assert torch.equal(m.weight.qdata, q0) and torch.equal(m.weight.scale, s0)
        assert G._swapped_rec(m) is not None
    extra = _Lin(96, 64, 4)
    got = _gemms(attn, merged, (extra,))
    assert len(got) == 4
    for a, b in zip(want, got):
        assert a.shape == b.shape and torch.equal(a, b)
    # an in-place LoRA merge writes through the slices into the merged rows
    attn.to_k.weight.qdata.zero_()
    assert not wq[48:96].any()


def test_merge_undone_when_another_holder_keeps_a_payload():
    attn = _Attn()
    held = attn.to_k.weight.qdata
    orig = [(m.weight.qdata, m.weight.scale) for m in (attn.to_q, attn.to_k, attn.to_v)]
    assert G.merge_payloads((attn.to_q, attn.to_k, attn.to_v)) is None
    for m, (q, s) in zip((attn.to_q, attn.to_k, attn.to_v), orig):
        assert m.weight.qdata is q and m.weight.scale is s
    del held


@pytest.mark.parametrize("why", ["bias", "mixed_path", "not_swapped", "hooked"])
def test_merge_refuses_what_three_gemms_must_serve(why):
    attn = _Attn()
    if why == "bias":
        attn.to_v.bias = torch.nn.Parameter(torch.zeros(48, dtype = torch.bfloat16))
    elif why == "mixed_path":
        rec = attn.to_v.__dict__[G._REC]
        attn.to_v.__dict__[G._REC] = rec[:4] + (False,)
    elif why == "not_swapped":
        del attn.to_v.__dict__[G._REC]
    else:
        attn.to_v.register_forward_pre_hook(lambda m, a: None)
    assert G.merge_payloads((attn.to_q, attn.to_k, attn.to_v)) is None


def test_gate_falls_back_when_a_projection_is_replaced(monkeypatch):
    monkeypatch.setattr(G, "_OP_HANDLE", _ref_gemm)
    attn = _Attn()
    merged = G.merge_payloads((attn.to_q, attn.to_k, attn.to_v))
    attn.__dict__[F._QKV_ATTR] = (attn.to_q, attn.to_k, attn.to_v, merged)
    assert F._merged_qkv(attn) is merged

    class _Lora(torch.nn.Module):  # a PEFT wrapper around the swapped Linear
        def __init__(self, base):
            super().__init__()
            self.base_layer = base
            self.scaling = {"a": 1.0}

    attn.to_q = _Lora(attn.to_q)
    assert F._merged_qkv(attn) is None
    # a stale merge (rows on another card) runs the three GEMMs too
    attn.to_q = attn.to_q.base_layer
    stale = (merged[0].to("meta"),) + merged[1:]
    want = _gemms(attn, None)
    got = _gemms(attn, stale)
    for a, b in zip(want, got):
        assert torch.equal(a, b)


def test_merge_qkv_needs_the_switch_and_a_resident_flux3(monkeypatch):
    class Flux3Transformer2DModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList([_Attn(), _Attn()])

    monkeypatch.setitem(F._CLS, "Flux3Attention", _Attn)
    from core.inference import diffusion_int8_fused as I

    t = Flux3Transformer2DModel()
    monkeypatch.setattr(I, "resident_cuda_cards", lambda m: ())
    assert F.merge_qkv(t) == 0 and F.fused_qkv_count(t) == 0
    monkeypatch.setattr(I, "resident_cuda_cards", lambda m: (0,))
    monkeypatch.setenv(F.FLUX3_FUSED_QKV_ENV, "0")
    assert F.merge_qkv(t) == 0 and F.fused_qkv_count(t) == 0
    monkeypatch.delenv(F.FLUX3_FUSED_QKV_ENV)
    assert F.merge_qkv(t) == 2 and F.fused_qkv_count(t) == 2
    assert F.merge_qkv(t) == 2  # idempotent


def test_int8_attention_route_keeps_three_gemms(monkeypatch):
    """The INT8 attention kernel's quantisers read Q / K / V as the projections wrote them: on that route the merged
    rows stay unused (L4: +2.3% per step merged); torch attention takes the merged GEMM."""
    monkeypatch.setattr(G, "_OP_HANDLE", _ref_gemm)
    attn = _Attn()
    merged = G.merge_payloads((attn.to_q, attn.to_k, attn.to_v))
    attn.__dict__[F._QKV_ATTR] = (attn.to_q, attn.to_k, attn.to_v, merged)
    seen = []
    real = G.shared_rotq_linears

    def spy(
        x,
        linears,
        merged = None,
    ):
        seen.append(merged)
        return real(x, linears, merged = merged) if merged is not None else real(x, linears)

    monkeypatch.setattr(G, "shared_rotq_linears", spy)
    x = torch.zeros(2, 37, 64)

    class _Int8Proc:
        _unsloth_int8 = True

    attn.processor = _Int8Proc()
    F._shared_qkv(attn, x)
    attn.processor = types.SimpleNamespace(_unsloth_int8 = False)
    F._shared_qkv(attn, x)
    del attn.processor
    F._shared_qkv(attn, x)
    assert seen == [None, merged, merged]
