# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MOSS-Image on the shared speed routes: its in-tree blocks take the INT8 attention dispatch route
(``diffusion_int8_attention_dispatch``) and, on a ConvRot int8 checkpoint, run q / k / v on one shared rotation + act
quant (``diffusion_moss_fused``). CPU: the card is faked and the INT8 op is replaced by a recorder."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest import mock

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

torch = pytest.importorskip("torch")
diffusers = pytest.importorskip("diffusers")
pytest.importorskip("transformers")
if not hasattr(diffusers, "AutoencoderKLFlux2"):
    pytest.skip("diffusers without AutoencoderKLFlux2", allow_module_level = True)
attention_dispatch = pytest.importorskip("diffusers.models.attention_dispatch")

from core.inference import diffusion_int8_attention as I  # noqa: E402
from core.inference import diffusion_int8_attention_dispatch as D  # noqa: E402
from core.inference import diffusion_moss_fused as fused  # noqa: E402
from core.inference import diffusion_moss_image as moss  # noqa: E402


@pytest.fixture(autouse = True)
def _clean(monkeypatch):
    monkeypatch.delenv(I.INT8_ATTENTION_ENV, raising = False)
    monkeypatch.delenv(I.IMPL_ENV, raising = False)
    monkeypatch.delenv(D.DISPATCH_ENV, raising = False)
    monkeypatch.delenv(fused.MOSS_SHARED_QKV_ENV, raising = False)
    monkeypatch.delenv(fused.MOSS_FUSED_ENV, raising = False)
    monkeypatch.delenv(fused.MOSS_SITU_FUSED_ENV, raising = False)
    stats = dict(I.STATS)
    D.ROUTE_STATS.clear()
    D._DECISIONS.clear()
    monkeypatch.setattr(D, "MIN_TOKENS", 0)
    yield
    I.STATS.clear()
    I.STATS.update(stats)
    D.ROUTE_STATS.clear()


@pytest.fixture
def card(monkeypatch):
    """CPU tensors count as the route's card; the op records its calls and returns the native result."""
    calls = []
    monkeypatch.setattr(D, "_on_card", lambda x, route: True)

    def impl(q, k, v):
        calls.append((tuple(q.shape), tuple(k.shape)))
        return I.native_attention(q, k, v)

    monkeypatch.setattr(I, "int8_attention_impl", impl)
    I.int8_attention_op()
    return calls


def _fake_cuda(monkeypatch):
    from core.inference import rocm_bf16

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (8, 9))
    monkeypatch.setattr(rocm_bf16, "is_rocm_torch", lambda *a, **k: False)
    monkeypatch.setattr(I, "implementation", lambda index, head_dim: "cuda")
    monkeypatch.setattr(D, "_race_wins", lambda *a, **k: False)
    monkeypatch.setattr(I, "verdict", lambda *a, **k: {"impl": "cuda", "ok": True, "min_tokens": 0})


def _tiny(dtype = torch.bfloat16):
    torch.manual_seed(0)
    return (
        moss.MOSSImageTransformer(dim = 256, ffn_dim = 512, num_heads = 2, num_layers = 3, text_dim = 64)
        .eval()
        .to(dtype)
    )


def _inputs(dtype = torch.bfloat16):
    g = torch.Generator().manual_seed(1)
    return (
        torch.randn(1, 128, 4, 4, generator = g).to(dtype),
        torch.tensor([500.0]),
        torch.randn(1, 7, 64, generator = g).to(dtype),
    )


class _Pipe:
    def __init__(self, transformer):
        self.transformer = transformer


_TARGET = types.SimpleNamespace(device = "cuda", dtype = torch.bfloat16)


def _compiling():
    return mock.patch.object(torch.compiler, "is_compiling", lambda: True), mock.patch.object(
        moss, "_length_dynamic", lambda *a: None
    )


# ----------------------------------------------------------------------------------------------- INT8 attention route


def test_the_blocks_read_the_dispatch_from_the_module_globals():
    """The route rewires a module's ``dispatch_attention_fn``; the blocks must look it up per call, not hold a copy."""
    assert moss.dispatch_attention_fn is not None
    assert not hasattr(moss, "_DISPATCH")
    assert D.install_dispatch_route() >= 1
    assert moss.dispatch_attention_fn is D.routed_dispatch_attention_fn
    for code in (moss._attention.__code__, moss._attention_heads_first.__code__):
        assert "dispatch_attention_fn" in code.co_names


def test_every_moss_attention_module_is_routable():
    dit = _tiny()
    mods = D.routable_modules(dit)
    # 2 text refiner blocks + 3 joint blocks, head dim from the module
    assert len(mods) == 5
    assert {I._head_dim(m) for m in mods} == {128}


def test_moss_routes_its_attention_calls_and_keeps_the_output(card, monkeypatch):
    _fake_cuda(monkeypatch)
    monkeypatch.setenv(I.INT8_ATTENTION_ENV, "1")
    dit = _tiny()
    dit.set_attention_backend("native")
    x, t, e = _inputs()
    with torch.no_grad():
        ref = dit(x, t, e, return_dict = False)[0]
    pipe = _Pipe(dit)
    assert D.install(pipe, _TARGET) == 5 and D.routed_count(pipe) == 5
    with torch.no_grad():
        out = dit(x, t, e, return_dict = False)[0]
    # 2 text blocks on the 7 text tokens, 3 joint blocks on 16 image + 7 text tokens
    assert sorted(card) == sorted(
        [((1, 7, 2, 128), (1, 7, 2, 128))] * 2 + [((1, 23, 2, 128), (1, 23, 2, 128))] * 3
    )
    # the recorder returns native attention: same numerics as before the route
    assert torch.equal(out, ref)
    assert D.remove(pipe) == 5 and D.routed_count(pipe) == 0


def _quantize_one(dit):
    """Stand-in for an INT8 checkpoint: one block Linear with int8 storage (install never runs a forward)."""
    block = fused._blocks(dit)[0]
    w = block.proj["o"].weight
    block.proj["o"].weight = torch.nn.Parameter(
        torch.zeros(w.shape, dtype = torch.int8), requires_grad = False
    )
    return dit


def test_moss_races_the_kernel_with_the_switch_unset_on_quantized_weights(card, monkeypatch):
    """Unset switch: MOSS-Image is an AUTO_DITS family raced only with quantized weights (its INT8-ConvRot default);
    a bf16 load keeps its attention, speed off stays native, the switch 0 turns it off."""
    _fake_cuda(monkeypatch)
    assert "MOSSImageTransformer" in D.AUTO_DITS and "MOSSImageTransformer" in D.AUTO_QUANTIZED_ONLY
    bf16 = _Pipe(_tiny())
    assert not D.quantized_weights(bf16.transformer)
    assert D.install(bf16, _TARGET) == 0
    pipe = _Pipe(_quantize_one(_tiny()))
    assert D.quantized_weights(pipe.transformer)
    assert D.install(pipe, _TARGET, speed_active = False) == 0
    assert D.install(pipe, _TARGET) == 5
    monkeypatch.setenv(I.INT8_ATTENTION_ENV, "0")
    assert D.install(pipe, _TARGET) == 0 and D.routed_count(pipe) == 0


def test_a_request_by_name_routes_a_bf16_moss(card, monkeypatch):
    _fake_cuda(monkeypatch)
    pipe = _Pipe(_tiny())
    assert D.install(pipe, _TARGET, requested = I.BACKEND) == 5


# ----------------------------------------------------------------------------------------------- shared q / k / v quant


def test_the_shared_quant_serves_q_k_v_in_one_call():
    dit = _tiny(torch.float32)
    x, t, e = _inputs(torch.float32)
    seen = []

    def shared(y, linears):
        seen.append(len(linears))
        return [lin(y) for lin in linears]

    with torch.no_grad():
        ref = dit(x, t, e, return_dict = False)[0]
        for block in fused._blocks(dit):
            block.__dict__[moss.SHARED_QKV_ATTR] = shared
        a, b = _compiling()
        with a, b:
            out = dit(x, t, e, return_dict = False)[0]
        # eager never reads it: bit-identical to the reference port
        eager = dit(x, t, e, return_dict = False)[0]
    assert seen == [3, 3, 3]
    assert torch.equal(out, ref) and torch.equal(eager, ref) and len(seen) == 3


def test_a_declined_shared_quant_runs_the_three_projections():
    dit = _tiny(torch.float32)
    x, t, e = _inputs(torch.float32)
    with torch.no_grad():
        ref = dit(x, t, e, return_dict = False)[0]
        for block in fused._blocks(dit):
            block.__dict__[moss.SHARED_QKV_ATTR] = lambda y, linears: None
        a, b = _compiling()
        with a, b:
            out = dit(x, t, e, return_dict = False)[0]
    assert torch.equal(out, ref)


def _rotate_qkv(dit, which = ("q", "k", "v")):
    from core.inference.diffusion_convrot import _install_rotation
    for block in fused._blocks(dit):
        for name in which:
            _install_rotation(block.proj[name], 256)


def test_install_arms_the_shared_quant_on_rotated_blocks_only(monkeypatch):
    from core.inference import diffusion_int8_fused

    monkeypatch.setattr(
        diffusion_int8_fused, "resident_cuda_device", lambda root: torch.device("cuda", 0)
    )
    dit = _tiny(torch.float32)
    assert fused.install_modules(dit)["shared_qkv"] == 0
    _rotate_qkv(dit)
    result = fused.install_modules(dit)
    assert result["shared_qkv"] == 3
    from core.inference.diffusion_int8_gemm import shared_rotq_linears

    assert all(b.__dict__[moss.SHARED_QKV_ATTR] is shared_rotq_linears for b in fused._blocks(dit))
    monkeypatch.setenv(fused.MOSS_SHARED_QKV_ENV, "0")
    assert fused.install_modules(dit)["shared_qkv"] == 0
    assert all(moss.SHARED_QKV_ATTR not in b.__dict__ for b in fused._blocks(dit))


def test_a_partly_rotated_block_keeps_its_own_quants(monkeypatch):
    from core.inference import diffusion_int8_fused

    monkeypatch.setattr(
        diffusion_int8_fused, "resident_cuda_device", lambda root: torch.device("cuda", 0)
    )
    dit = _tiny(torch.float32)
    _rotate_qkv(dit, ("q", "k"))
    assert fused.install_modules(dit)["shared_qkv"] == 0


def test_shared_rotq_declines_unswapped_linears():
    """The per-call check behind the attribute: a Linear the int8 GEMM swap never took is not covered."""
    from core.inference.diffusion_int8_gemm import shared_rotq_linears

    lins = tuple(torch.nn.Linear(256, 256, bias = False) for _ in range(3))
    assert shared_rotq_linears(torch.randn(32, 256), lins) is None


# ----------------------------------------------------------------------------------------------- SiTU-GLU down projection


def test_the_situ_down_hook_replaces_the_stock_product_and_projection():
    dit = _tiny(torch.float32)
    x, t, e = _inputs(torch.float32)
    seen = []

    def situ_down(down, gate, value):
        seen.append((down.in_features, tuple(gate.shape)))
        return down(moss._situ(value, gate))

    with torch.no_grad():
        ref = dit(x, t, e, return_dict = False)[0]
        for block in fused._blocks(dit):
            block.__dict__[moss.SITU_DOWN_ATTR] = situ_down
        a, b = _compiling()
        with a, b:
            out = dit(x, t, e, return_dict = False)[0]
        eager = dit(x, t, e, return_dict = False)[0]
        # declined (None): the stock product and projection
        for block in fused._blocks(dit):
            block.__dict__[moss.SITU_DOWN_ATTR] = lambda down, gate, value: None
        with a, b:
            declined = dit(x, t, e, return_dict = False)[0]
    # gate / value are the [B, S, ffn] halves of ffn_up_gate's output
    assert seen == [(512, (1, 23, 512))] * 3
    assert torch.equal(out, ref) and torch.equal(eager, ref) and torch.equal(declined, ref)


def test_install_arms_the_situ_down_on_rotated_down_projections(monkeypatch):
    from core.inference import diffusion_int8_fused
    from core.inference.diffusion_int8_gemm import situ_linear

    monkeypatch.setattr(
        diffusion_int8_fused, "resident_cuda_device", lambda root: torch.device("cuda", 0)
    )
    armed = []
    monkeypatch.setattr(
        fused, "_arm_situ_cards", lambda root, logger = None: armed.append(root) or [0]
    )
    dit = _tiny(torch.float32)
    assert fused.install_modules(dit)["situ_down"] == 0 and armed == []
    _rotate_qkv(dit, ("ffn_down",))
    assert fused.install_modules(dit)["situ_down"] == 3 and armed == [dit]
    assert all(b.__dict__[moss.SITU_DOWN_ATTR] is situ_linear for b in fused._blocks(dit))
    monkeypatch.setenv(fused.MOSS_SITU_FUSED_ENV, "0")
    assert fused.install_modules(dit)["situ_down"] == 0
    assert all(moss.SITU_DOWN_ATTR not in b.__dict__ for b in fused._blocks(dit))


def test_situ_linear_declines_unswapped_linears():
    from core.inference.diffusion_int8_gemm import situ_linear

    down = torch.nn.Linear(256, 64, bias = False)
    g = torch.randn(32, 256)
    assert situ_linear(down, g, g) is None


def test_situ_reference_matches_the_port_in_fp32():
    from core.inference.diffusion_convrot_quant import producer_reference, situ_reference

    g = torch.Generator().manual_seed(3)
    gate = (torch.randn(8, 256, generator = g) * 8).to(torch.bfloat16)
    value = (torch.randn(8, 256, generator = g) * 60).to(torch.bfloat16)
    want = moss._situ(value.float(), gate.float()).to(torch.bfloat16)
    assert torch.equal(situ_reference(gate, value), want)
    assert torch.equal(producer_reference(None, gate, value, 1), want)
    # act 0 keeps the SwiGLU product
    assert torch.equal(
        producer_reference(None, gate, value), torch.nn.functional.silu(gate) * value
    )


def test_the_situ_entry_point_declines_an_unarmed_card():
    from core.inference import diffusion_convrot_quant as Q

    g = torch.zeros(32, 256, dtype = torch.bfloat16)
    assert Q.convrot_situ_act_quant(g, g, 256, True) is None
    assert Q.convrot_situ_act_quant(g, g, 128, True) is None


def test_the_situ_producer_arms_only_on_measured_cards(monkeypatch):
    from core.inference import diffusion_convrot_quant as Q

    monkeypatch.delenv(Q.SITU_FUSED_ENV, raising = False)
    assert Q.situ_arch_ok((12, 0)) and Q.situ_arch_ok((8, 9))
    for cap in ((8, 0), (8, 6), (7, 5), (9, 0), (10, 0), None):
        assert not Q.situ_arch_ok(cap)
    monkeypatch.setenv(Q.SITU_FUSED_ENV, "1")
    assert Q.situ_arch_ok((8, 0))
    # an unmeasured card is not probed and stays unarmed, even with rotq_h16 armed on it
    monkeypatch.delenv(Q.SITU_FUSED_ENV, raising = False)
    probed = []
    monkeypatch.setitem(Q._H16_DEVICE, 7, True)
    monkeypatch.setattr(Q, "_device_cap", lambda index: (8, 0))
    monkeypatch.setattr(Q, "_situ_probe", lambda index: probed.append(index) or True)
    monkeypatch.setattr(Q, "_SITU_DEVICE", {})
    monkeypatch.setattr(Q, "_SITU_OFF", set())
    assert Q._arm_situ(7) is False
    assert probed == [] and 7 in Q._SITU_OFF and not Q.situ_armed(7)
