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
