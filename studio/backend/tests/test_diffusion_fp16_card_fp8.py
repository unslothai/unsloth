# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Weight-only fp8 on fp16-only NVIDIA cards (Turing); capability is monkeypatched, no GPU needed."""

from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")
if not hasattr(torch, "float8_e4m3fn"):
    pytest.skip("torch without float8 dtypes", allow_module_level = True)

import core.inference.diffusion_native_quant as nq  # noqa: E402
import core.inference.diffusion_transformer_quant as tq  # noqa: E402

# Literals, not tq.*_ENV: the behavioural tests must also run against a tree without the names.
FP8_KILL_SWITCH = "UNSLOTH_DIFFUSION_FP16_CARD_FP8"
INT8_KILL_SWITCH = "UNSLOTH_DIFFUSION_FP16_CARD_INT8"


def _target(dtype, device = "cuda"):
    return types.SimpleNamespace(device = device, dtype = dtype)


@pytest.fixture
def nvidia(monkeypatch):
    monkeypatch.setattr(tq, "torch_is_rocm", lambda: False)
    monkeypatch.setattr(tq, "is_stubbed", lambda name: False)
    monkeypatch.delenv(FP8_KILL_SWITCH, raising = False)
    monkeypatch.delenv(INT8_KILL_SWITCH, raising = False)


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
def test_turing_fp16_and_fp32_pipelines_take_fp8(nvidia, monkeypatch, dtype):
    _cap(monkeypatch, (7, 5))
    target = _target(dtype)
    assert tq.fp16_card_fp8_scheme(target, "fp8") == "fp8"
    assert tq.fp16_card_native_scheme(target, " FP8 ") == "fp8"
    assert tq.fp16_card_native_scheme(target, "int8") == "int8"
    assert tq.fp16_card_native_host(target, "fp8") and tq.fp16_card_native_host(target, "int8")


@pytest.mark.parametrize("scheme", [None, "auto", "none", "int8", "mxfp8"])
def test_only_an_explicit_fp8_request_is_taken(nvidia, monkeypatch, scheme):
    _cap(monkeypatch, (7, 5))
    assert tq.fp16_card_fp8_scheme(_target(torch.float16), scheme) is None
    if scheme not in ("int8",):
        assert tq.fp16_card_native_scheme(_target(torch.float16), scheme) is None


@pytest.mark.parametrize(
    "cap", [None, (6, 1), (7, 0), (8, 0), (8, 6), (8, 9), (9, 0), (10, 0), (12, 0)]
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_every_other_nvidia_card_is_unchanged(nvidia, monkeypatch, cap, dtype):
    _cap(monkeypatch, cap)
    assert tq.fp16_card_fp8_scheme(_target(dtype), "fp8") is None
    assert not tq.fp16_card_native_host(_target(dtype), "fp8")


@pytest.mark.parametrize("device", ["cpu", "mps", "xpu"])
def test_bf16_turing_and_non_cuda_are_unchanged(nvidia, monkeypatch, device):
    _cap(monkeypatch, (7, 5))
    assert tq.fp16_card_fp8_scheme(_target(torch.bfloat16), "fp8") is None
    assert tq.fp16_card_fp8_scheme(_target(torch.float16, device), "fp8") is None


@pytest.mark.parametrize("raw", ["0", "false", "off", "no"])
def test_kill_switch_restores_the_refusal(nvidia, monkeypatch, raw):
    _cap(monkeypatch, (7, 5))
    monkeypatch.setenv(FP8_KILL_SWITCH, raw)
    assert tq.fp16_card_fp8_scheme(_target(torch.float32), "fp8") is None
    assert not tq.fp16_card_native_host(_target(torch.float32), "fp8")
    assert tq.fp16_card_native_scheme(_target(torch.float32), "int8") == "int8"


def test_the_int8_kill_switch_also_closes_fp8(nvidia, monkeypatch):
    _cap(monkeypatch, (7, 5))
    monkeypatch.setenv(INT8_KILL_SWITCH, "0")
    assert tq.fp16_card_fp8_scheme(_target(torch.float32), "fp8") is None


def test_the_family_deny_list_still_applies(nvidia, monkeypatch):
    _cap(monkeypatch, (7, 5))
    monkeypatch.setitem(tq._FAMILY_SCHEME_DENY, "toy-family", ("fp8",))
    assert tq.fp16_card_fp8_scheme(_target(torch.float16), "fp8", family = "toy-family") is None


def test_the_shared_native_route_is_not_widened(nvidia, monkeypatch):
    _cap(monkeypatch, (7, 5))
    assert tq.native_quant_scheme(_target(torch.float16), "fp8") is None
    assert tq.select_transformer_quant_scheme(_target(torch.float16), "fp8") is None


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_quantize_transformer_builds_weight_only_fp8_in_the_pipeline_dtype(
    nvidia, monkeypatch, dtype
):
    _cap(monkeypatch, (7, 5))
    torch.manual_seed(0)
    model = _Toy().to(dtype)
    x = torch.randn(4, 128, dtype = dtype)
    with torch.no_grad():
        ref = model(x).float()
    got = tq.quantize_transformer(
        types.SimpleNamespace(transformer = model), _target(dtype), mode = "fp8", min_features = 64
    )
    assert got == "fp8"
    for name in ("to_q", "ff", "out"):
        layer = getattr(model, name)
        assert nq.is_native_linear(layer), name
        assert layer.compute_dtype == dtype
        assert layer.scheme == "fp8" and not layer.act_int8
        assert layer.weight_q.element_size() == 1
    with torch.no_grad():
        y = model(x)
    assert y.dtype == dtype and torch.isfinite(y).all()
    rel = ((y.float() - ref).norm() / ref.norm()).item()
    assert rel < 0.05, rel


def test_quantize_transformer_on_ampere_fp16_still_declines(nvidia, monkeypatch):
    _cap(monkeypatch, (8, 6))
    model = _Toy().to(torch.float16)
    got = tq.quantize_transformer(
        types.SimpleNamespace(transformer = model),
        _target(torch.float16),
        mode = "fp8",
        min_features = 64,
    )
    assert got is None
    assert not nq.is_native_linear(model.to_q)


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
def test_image_gate_admits_fp8_on_turing(nvidia, monkeypatch, family, dtype, memory_mode):
    _cap(monkeypatch, (7, 5))
    _image_gate(monkeypatch, _image_family(family), "fp8", dtype, memory_mode = memory_mode)


def test_image_gate_kill_switch_refuses_fp8_on_turing(nvidia, monkeypatch):
    _cap(monkeypatch, (7, 5))
    monkeypatch.setenv(FP8_KILL_SWITCH, "0")
    with pytest.raises(RuntimeError, match = "needs a CUDA GPU in bf16"):
        _image_gate(monkeypatch, _image_family("flux.3-klein"), "fp8", torch.float32)


@pytest.mark.parametrize("cap", [(7, 0), (8, 6)])
def test_image_gate_on_other_fp16_cards_is_unchanged(nvidia, monkeypatch, cap):
    _cap(monkeypatch, cap)
    with pytest.raises(RuntimeError, match = "needs a CUDA GPU in bf16"):
        _image_gate(monkeypatch, _image_family("flux.3-klein"), "fp8", torch.float16)


def test_the_kill_switch_name_matches_the_module():
    assert tq.FP16_CARD_FP8_ENV == FP8_KILL_SWITCH
