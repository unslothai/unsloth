# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""INT8 on fp16-only NVIDIA cards (Turing, T4): the torchao-free int8 linear instead of a refusal. Capability and
ROCm are monkeypatched, so this runs without a GPU."""

from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")

import core.inference.diffusion_native_quant as nq
import core.inference.diffusion_transformer_quant as tq

# Literal, not tq.FP16_CARD_INT8_ENV: the behavioural tests below must also run against a tree without the name.
KILL_SWITCH = "UNSLOTH_DIFFUSION_FP16_CARD_INT8"


def _target(dtype, device = "cuda"):
    return types.SimpleNamespace(device = device, dtype = dtype)


@pytest.fixture
def nvidia(monkeypatch):
    monkeypatch.setattr(tq, "torch_is_rocm", lambda: False)
    monkeypatch.setattr(tq, "is_stubbed", lambda name: False)
    monkeypatch.delenv(KILL_SWITCH, raising = False)


def _cap(monkeypatch, cap):
    monkeypatch.setattr(tq, "_capability", lambda ordinal = None: cap)


class _Toy(torch.nn.Module):
    def __init__(self, dim = 128):
        super().__init__()
        self.to_q = torch.nn.Linear(dim, dim)
        self.ff = torch.nn.Linear(dim, 2 * dim)
        self.out = torch.nn.Linear(2 * dim, dim)

    def forward(self, x):
        return self.out(torch.nn.functional.gelu(self.ff(self.to_q(x))))


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_turing_fp16_and_fp32_pipelines_take_int8(nvidia, monkeypatch, dtype):
    _cap(monkeypatch, (7, 5))
    target = _target(dtype)
    assert tq.fp16_card_int8_host(target)
    assert tq.fp16_card_int8_scheme(target, "int8") == "int8"
    assert tq.fp16_card_int8_scheme(target, " INT8 ") == "int8"


@pytest.mark.parametrize("scheme", [None, "auto", "none", "fp8", "mxfp8"])
def test_only_an_explicit_int8_request_is_taken(nvidia, monkeypatch, scheme):
    _cap(monkeypatch, (7, 5))
    assert tq.fp16_card_int8_scheme(_target(torch.float16), scheme) is None


@pytest.mark.parametrize(
    "cap", [(7, 0), (6, 1), (8, 0), (8, 6), (8, 9), (9, 0), (10, 0), (12, 0), None]
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_every_other_nvidia_card_is_unchanged(nvidia, monkeypatch, cap, dtype):
    # sm70 has no int8 tensor cores; Ampere+ runs bf16 through torchao; None = no CUDA answer.
    _cap(monkeypatch, cap)
    target = _target(dtype)
    assert not tq.fp16_card_int8_host(target)
    assert tq.fp16_card_int8_scheme(target, "int8") is None


def test_a_bf16_pipeline_on_turing_is_unchanged(nvidia, monkeypatch):
    _cap(monkeypatch, (7, 5))
    assert not tq.fp16_card_int8_host(_target(torch.bfloat16))


@pytest.mark.parametrize("device", ["cpu", "mps", "xpu"])
def test_non_cuda_devices_are_unchanged(nvidia, monkeypatch, device):
    _cap(monkeypatch, (7, 5))
    assert not tq.fp16_card_int8_host(_target(torch.float16, device = device))


def test_rocm_and_the_torchao_stub_keep_their_own_path(monkeypatch):
    _cap(monkeypatch, (7, 5))
    monkeypatch.setattr(tq, "is_stubbed", lambda name: False)
    monkeypatch.setattr(tq, "torch_is_rocm", lambda: True)
    assert not tq.fp16_card_int8_host(_target(torch.float16))
    monkeypatch.setattr(tq, "torch_is_rocm", lambda: False)
    monkeypatch.setattr(tq, "is_stubbed", lambda name: name == "torchao")
    assert not tq.fp16_card_int8_host(_target(torch.float16))


@pytest.mark.parametrize("raw", ["0", "false", "off", "no"])
def test_kill_switch_restores_the_refusal(nvidia, monkeypatch, raw):
    _cap(monkeypatch, (7, 5))
    monkeypatch.setenv(KILL_SWITCH, raw)
    assert not tq.fp16_card_int8_host(_target(torch.float16))
    assert tq.fp16_card_int8_scheme(_target(torch.float16), "int8") is None


def test_the_family_deny_list_still_applies(nvidia, monkeypatch):
    _cap(monkeypatch, (7, 5))
    monkeypatch.setitem(tq._FAMILY_SCHEME_DENY, "toy-family", frozenset({tq.TQ_INT8}))
    assert tq.fp16_card_int8_scheme(_target(torch.float16), "int8", family = "toy-family") is None


def test_the_shared_native_route_is_not_widened(nvidia, monkeypatch):
    # The single-file int8 loader and video read native_quant_scheme: Turing must not change their answer.
    _cap(monkeypatch, (7, 5))
    for dtype in (torch.float16, torch.float32):
        assert not tq.native_quant_host(_target(dtype))
        for offload in (False, True):
            assert tq.native_quant_scheme(_target(dtype), "int8", offload = offload) is None


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_quantize_transformer_builds_native_int8_linears_in_the_pipeline_dtype(
    nvidia, monkeypatch, dtype
):
    _cap(monkeypatch, (7, 5))
    torch.manual_seed(0)
    model = _Toy().to(dtype)
    x = torch.randn(4, 128, dtype = dtype)
    with torch.no_grad():
        ref = model(x).float()
    got = tq.quantize_transformer(
        types.SimpleNamespace(transformer = model), _target(dtype), mode = "int8", min_features = 64
    )
    assert got == "int8"
    for name in ("to_q", "ff", "out"):
        layer = getattr(model, name)
        assert nq.is_native_linear(layer), name
        assert layer.compute_dtype == dtype
        assert layer.weight_q.dtype == torch.int8
    with torch.no_grad():
        y = model(x)
    assert y.dtype == dtype and torch.isfinite(y).all()
    rel = ((y.float() - ref).norm() / ref.norm()).item()
    assert rel < 0.05, rel


def test_mixed_dtype_linears_keep_their_deliberate_fp32_layers(nvidia, monkeypatch):
    _cap(monkeypatch, (7, 5))
    model = _Toy().to(torch.float16)
    model.out.float()
    got = tq.quantize_transformer(
        types.SimpleNamespace(transformer = model),
        _target(torch.float16),
        mode = "int8",
        min_features = 64,
    )
    assert got == "int8"
    assert nq.is_native_linear(model.to_q) and nq.is_native_linear(model.ff)
    assert isinstance(model.out, torch.nn.Linear) and model.out.weight.dtype == torch.float32


def test_quantize_transformer_on_ampere_fp16_still_declines(nvidia, monkeypatch):
    _cap(monkeypatch, (8, 6))
    model = _Toy().to(torch.float16)
    got = tq.quantize_transformer(
        types.SimpleNamespace(transformer = model),
        _target(torch.float16),
        mode = "int8",
        min_features = 64,
    )
    assert got is None
    assert isinstance(model.to_q, torch.nn.Linear) and not nq.is_native_linear(model.to_q)


def _image_family(name):
    from core.inference.diffusion_families import _FAMILIES
    return next(f for f in _FAMILIES if f.name == name)


def _image_gate(
    monkeypatch,
    fam,
    pinned,
    dtype,
    *,
    memory_mode = None,
):
    import core.inference.diffusion as d
    monkeypatch.setattr(d, "effective_te_quant", lambda *a, **k: None)
    d.DiffusionBackend._assert_precision_for_target(
        None,
        fam,
        _target(dtype),
        model_kind = "pipeline",
        pinned = pinned,
        te_mode = None,
        memory_mode = memory_mode,
        cpu_offload = False,
    )


@pytest.mark.parametrize(
    "family,dtype",
    [
        ("qwen-image-2.1", torch.float16),
        ("flux.3-klein", torch.float32),
        ("flux.3-klein", torch.float16),
    ],
)
@pytest.mark.parametrize("memory_mode", [None, "low_vram"])
def test_image_gate_admits_int8_on_turing(nvidia, monkeypatch, family, dtype, memory_mode):
    _cap(monkeypatch, (7, 5))
    _image_gate(monkeypatch, _image_family(family), "int8", dtype, memory_mode = memory_mode)


@pytest.mark.parametrize("pinned", ["fp8", "mxfp8"])
def test_image_gate_still_refuses_other_schemes_on_turing(nvidia, monkeypatch, pinned):
    _cap(monkeypatch, (7, 5))
    with pytest.raises(RuntimeError, match = "transformer_quant"):
        _image_gate(monkeypatch, _image_family("qwen-image-2.1"), pinned, torch.float16)


def test_image_gate_kill_switch_refuses_int8_on_turing(nvidia, monkeypatch):
    _cap(monkeypatch, (7, 5))
    monkeypatch.setenv(KILL_SWITCH, "0")
    with pytest.raises(RuntimeError, match = "needs a CUDA GPU in bf16"):
        _image_gate(monkeypatch, _image_family("qwen-image-2.1"), "int8", torch.float16)


@pytest.mark.parametrize("cap", [(7, 0), (8, 6)])
def test_image_gate_on_other_fp16_cards_is_unchanged(nvidia, monkeypatch, cap):
    _cap(monkeypatch, cap)
    with pytest.raises(RuntimeError, match = "needs a CUDA GPU in bf16"):
        _image_gate(monkeypatch, _image_family("qwen-image-2.1"), "int8", torch.float16)


def test_dense_blocker_accepts_float16_only_when_asked():
    pipe = types.SimpleNamespace(transformer = _Toy().to(torch.float16))
    assert "float16" in (tq.dense_quant_blocker(pipe) or "")
    assert tq.dense_quant_blocker(pipe, allow_float16 = True) is None
    pipe.transformer._unsloth_runtime_quant = "int8"
    assert "already quantised" in (tq.dense_quant_blocker(pipe, allow_float16 = True) or "")


def test_dense_blocker_still_refuses_fp8_storage_with_float16_allowed():
    model = _Toy().to(torch.float16)
    model.to_q.weight.data = model.to_q.weight.data.to(torch.float8_e4m3fn)
    pipe = types.SimpleNamespace(transformer = model)
    assert "float8_e4m3fn" in (tq.dense_quant_blocker(pipe, allow_float16 = True) or "")


def test_small_host_int8_linears_arm_w8a8_on_request(monkeypatch):
    import core.inference.diffusion_small_host as sh

    monkeypatch.delenv(sh.INT8_ACT_ENV, raising = False)
    monkeypatch.setattr(sh, "_INT8_MIN_ELEMENTS", 1)
    model = _Toy().to(torch.float32)
    sh.quantize_int8_weight_(model, compute_dtype = torch.float32, work_device = "cpu", act_int8 = False)
    layers = [m for m in model.modules() if isinstance(m, sh.int8_linear_class())]
    assert layers and not any(m.act_int8 for m in layers)
    assert sh.enable_int8_act_(model) == len(layers)
    assert all(m.act_int8 for m in layers)
    with torch.no_grad():
        assert torch.isfinite(model(torch.randn(32, 128))).all()


def test_small_host_w8a8_kill_switch(monkeypatch):
    import core.inference.diffusion_small_host as sh

    monkeypatch.setattr(sh, "_INT8_MIN_ELEMENTS", 1)
    model = _Toy().to(torch.float32)
    sh.quantize_int8_weight_(model, compute_dtype = torch.float32, work_device = "cpu")
    assert any(isinstance(m, sh.int8_linear_class()) for m in model.modules())
    monkeypatch.setenv(sh.INT8_ACT_ENV, "0")
    assert sh.enable_int8_act_(model) == 0
    assert not any(getattr(m, "act_int8", False) for m in model.modules())


def test_the_kill_switch_name_matches_the_module():
    assert tq.FP16_CARD_INT8_ENV == KILL_SWITCH


# Explicit int8 on Turing (W8A8) vs #17079's checkpoint int8 storage under offload (weight-only): same fp16 target.
STORAGE_SWITCH = "UNSLOTH_DIFFUSION_INT8_STORAGE"
ACT_SWITCH = "UNSLOTH_NATIVE_INT8_ACT"


def _quantised_layers(dtype, target):
    model = _Toy().to(dtype)
    got = tq.quantize_transformer(
        types.SimpleNamespace(transformer = model), target, mode = "int8", min_features = 64
    )
    assert got == "int8"
    return [m for m in model.modules() if nq.is_native_linear(m)]


def test_explicit_int8_on_a_t4_fp16_target_builds_w8a8_although_it_is_a_storage_host(
    nvidia, monkeypatch
):
    _cap(monkeypatch, (7, 5))
    monkeypatch.delenv(STORAGE_SWITCH, raising = False)
    monkeypatch.delenv(ACT_SWITCH, raising = False)
    target = _target(torch.float16)
    assert tq.int8_storage_host(target)
    layers = _quantised_layers(torch.float16, target)
    assert layers and all(m.act_int8 for m in layers)
    # A caller that passes the storage answer (native_int8_act) still gets the W8A8 build.
    model = _Toy().to(torch.float16)
    tq.quantize_transformer(
        types.SimpleNamespace(transformer = model),
        target,
        mode = "int8",
        min_features = 64,
        act_int8 = tq.native_int8_act(target),
    )
    assert all(m.act_int8 for m in model.modules() if nq.is_native_linear(m))


def test_explicit_int8_w8a8_honours_the_native_act_kill_switch(nvidia, monkeypatch):
    _cap(monkeypatch, (7, 5))
    monkeypatch.setenv(ACT_SWITCH, "0")
    layers = _quantised_layers(torch.float16, _target(torch.float16))
    assert layers and not any(m.act_int8 for m in layers)


@pytest.mark.parametrize("offload", [False, True])
def test_the_storage_scheme_on_a_t4_keeps_its_merged_answers(nvidia, monkeypatch, offload):
    import core.inference.diffusion_comfy_quant as cq

    _cap(monkeypatch, (7, 5))
    monkeypatch.delenv(STORAGE_SWITCH, raising = False)
    target = _target(torch.float16)
    # Storage stays weight-only: native_int8_act is False whatever the Turing switch says.
    for raw in (None, "0"):
        if raw is None:
            monkeypatch.delenv(KILL_SWITCH, raising = False)
        else:
            monkeypatch.setenv(KILL_SWITCH, raw)
        assert tq.native_int8_act(target) is False
        assert tq.int8_storage_scheme(target, offload = offload) == ("int8" if offload else None)
        assert cq.comfy_int8_backend(target, "qwen-image-2.1", offload = offload) == (
            "native" if offload else None
        )


@pytest.mark.parametrize("cap,device", [((7, 0), "cuda"), ((6, 1), "cuda"), (None, "xpu")])
def test_sm70_and_xpu_keep_storage_and_never_get_w8a8(nvidia, monkeypatch, cap, device):
    _cap(monkeypatch, cap)
    target = _target(torch.float16, device = device)
    assert tq.int8_storage_host(target)
    assert not tq.fp16_card_int8_act(target)
    assert tq.fp16_card_int8_scheme(target, "int8") is None
    assert tq.native_int8_act(target) is False


def test_storage_kill_switch_does_not_touch_the_turing_w8a8_build(nvidia, monkeypatch):
    _cap(monkeypatch, (7, 5))
    monkeypatch.setenv(STORAGE_SWITCH, "0")
    target = _target(torch.float16)
    assert not tq.int8_storage_host(target)
    assert tq.int8_storage_scheme(target, offload = True) is None
    layers = _quantised_layers(torch.float16, target)
    assert layers and all(m.act_int8 for m in layers)
