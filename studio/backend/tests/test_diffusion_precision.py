# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit tests for text-encoder quantisation (``diffusion_precision.py``).

Hermetic: torch + the diffusers / torchao casters are stubbed via ``sys.modules`` so
gating and the apply path run without a GPU, real diffusers, or real torchao.
"""

from __future__ import annotations

import sys
import types

import pytest

import core.inference.diffusion_precision as dp
from core.inference.diffusion_precision import (
    TE_QUANT_FP8,
    TE_QUANT_FP8_DYNAMIC,
    TE_QUANT_INT8,
    TE_QUANT_NVFP4,
    _cast_int8_selective,
    _keep_bf16_block_fqns,
    normalize_te_quant,
    quantize_text_encoders,
    te_quant_supported,
)


def _target(
    *,
    device = "cuda",
    dtype = "bfloat16",
    cc = (10, 0),
):
    return types.SimpleNamespace(device = device, dtype = dtype, _cc = cc)


def _stub_torch(
    monkeypatch,
    *,
    with_fp8 = True,
    cc = (10, 0),
):
    torch = types.ModuleType("torch")
    torch.bfloat16 = "bfloat16"
    torch.float16 = "float16"
    if with_fp8:
        torch.float8_e4m3fn = "float8_e4m3fn"
    # _cast_fp8 skips nn.Embedding tables (skip_modules_classes) to keep prompt
    # tokens full precision, and _keep_bf16_block_fqns walks for nn.ModuleList block
    # stacks, so the stub torch must expose both.
    torch.nn = types.SimpleNamespace(
        Embedding = type("Embedding", (), {}),
        ModuleList = type("ModuleList", (list,), {}),
    )
    torch.cuda = types.SimpleNamespace(get_device_capability = lambda *a: cc)
    monkeypatch.setitem(sys.modules, "torch", torch)
    return torch


def _stub_casters(monkeypatch, recorder):
    # diffusers fp8 layerwise casting
    hooks = types.ModuleType("diffusers.hooks")
    casting = types.ModuleType("diffusers.hooks.layerwise_casting")
    casting.DEFAULT_SKIP_MODULES_PATTERN = ("norm",)
    hooks.apply_layerwise_casting = lambda module, **kw: recorder.append(("fp8", module))
    monkeypatch.setitem(sys.modules, "diffusers.hooks", hooks)
    monkeypatch.setitem(sys.modules, "diffusers.hooks.layerwise_casting", casting)
    # torchao nvfp4
    tq = types.ModuleType("torchao.quantization")
    tq.quantize_ = lambda module, config: recorder.append(("nvfp4", module))
    mx = types.ModuleType("torchao.prototype.mx_formats")
    mx.NVFP4WeightOnlyConfig = lambda: "nvfp4cfg"
    monkeypatch.setitem(sys.modules, "torchao.quantization", tq)
    monkeypatch.setitem(sys.modules, "torchao.prototype.mx_formats", mx)


# ── normalisation ─────────────────────────────────────────────────────────────


def test_normalize_te_quant():
    assert normalize_te_quant(None) is None
    assert normalize_te_quant("") is None
    assert normalize_te_quant("none") is None
    assert normalize_te_quant("FP8") == TE_QUANT_FP8
    assert normalize_te_quant("NVFP4") == TE_QUANT_NVFP4
    assert normalize_te_quant("int8") == TE_QUANT_INT8
    # Hyphens fold to underscores so "fp8-dynamic" is accepted.
    assert normalize_te_quant("FP8-Dynamic") == TE_QUANT_FP8_DYNAMIC
    with pytest.raises(ValueError):
        normalize_te_quant("int2")


# ── gating ────────────────────────────────────────────────────────────────────


def test_fp8_supported_requires_cuda_bf16_and_fp8(monkeypatch):
    _stub_torch(monkeypatch, with_fp8 = True)
    assert te_quant_supported(_target(), TE_QUANT_FP8) is True
    assert te_quant_supported(_target(device = "cpu"), TE_QUANT_FP8) is False
    assert te_quant_supported(_target(dtype = "float16"), TE_QUANT_FP8) is False


def test_nvfp4_supported_requires_blackwell(monkeypatch):
    _stub_torch(monkeypatch, cc = (10, 0))
    assert te_quant_supported(_target(), TE_QUANT_NVFP4) is True
    # Hopper (cc 9.0) has no NVFP4 tensor cores.
    _stub_torch(monkeypatch, cc = (9, 0))
    assert te_quant_supported(_target(), TE_QUANT_NVFP4) is False


def test_int8_supported_requires_sm80(monkeypatch):
    # int8 tensor cores (torch._int_mm) need Ampere sm_80+.
    _stub_torch(monkeypatch, cc = (8, 0))
    assert te_quant_supported(_target(), TE_QUANT_INT8) is True
    _stub_torch(monkeypatch, cc = (7, 5))
    assert te_quant_supported(_target(), TE_QUANT_INT8) is False
    # Still needs CUDA + bf16 like every mode.
    _stub_torch(monkeypatch, cc = (8, 0))
    assert te_quant_supported(_target(device = "cpu"), TE_QUANT_INT8) is False


def test_fp8_dynamic_supported_requires_sm89_and_fp8(monkeypatch):
    # Compute fp8 (torch._scaled_mm) needs fp8-GEMM silicon: Ada sm_89+ / Hopper / Blackwell.
    _stub_torch(monkeypatch, cc = (8, 9))
    assert te_quant_supported(_target(), TE_QUANT_FP8_DYNAMIC) is True
    _stub_torch(monkeypatch, cc = (9, 0))
    assert te_quant_supported(_target(), TE_QUANT_FP8_DYNAMIC) is True
    # Ampere (8.0) has int8 but not fp8 GEMM.
    _stub_torch(monkeypatch, cc = (8, 0))
    assert te_quant_supported(_target(), TE_QUANT_FP8_DYNAMIC) is False
    # No fp8 dtype at all -> unsupported regardless of arch.
    _stub_torch(monkeypatch, with_fp8 = False, cc = (9, 0))
    assert te_quant_supported(_target(), TE_QUANT_FP8_DYNAMIC) is False


# ── apply ─────────────────────────────────────────────────────────────────────


def test_quantize_disabled_returns_none(monkeypatch):
    _stub_torch(monkeypatch)
    pipe = types.SimpleNamespace(text_encoder = object())
    assert quantize_text_encoders(pipe, _target(), mode = None) is None
    assert quantize_text_encoders(pipe, _target(), mode = "none") is None


def test_quantize_fp8_casts_all_encoders(monkeypatch):
    _stub_torch(monkeypatch)
    recorder: list = []
    _stub_casters(monkeypatch, recorder)
    te1, te3 = object(), object()
    pipe = types.SimpleNamespace(text_encoder = te1, text_encoder_2 = None, text_encoder_3 = te3)
    mode = quantize_text_encoders(pipe, _target(), mode = "fp8")
    assert mode == TE_QUANT_FP8
    assert recorder == [("fp8", te1), ("fp8", te3)]


def test_quantize_nvfp4_uses_torchao(monkeypatch):
    _stub_torch(monkeypatch, cc = (10, 0))
    recorder: list = []
    _stub_casters(monkeypatch, recorder)
    te = object()
    pipe = types.SimpleNamespace(text_encoder = te)
    mode = quantize_text_encoders(pipe, _target(), mode = "nvfp4")
    assert mode == TE_QUANT_NVFP4
    assert recorder == [("nvfp4", te)]


def test_quantize_nvfp4_unsupported_on_hopper_is_noop(monkeypatch):
    _stub_torch(monkeypatch, cc = (9, 0))
    recorder: list = []
    _stub_casters(monkeypatch, recorder)
    pipe = types.SimpleNamespace(text_encoder = object())
    assert quantize_text_encoders(pipe, _target(cc = (9, 0)), mode = "nvfp4") is None
    assert recorder == []


def test_quantize_tolerates_caster_failure(monkeypatch):
    _stub_torch(monkeypatch)
    hooks = types.ModuleType("diffusers.hooks")
    casting = types.ModuleType("diffusers.hooks.layerwise_casting")
    casting.DEFAULT_SKIP_MODULES_PATTERN = ("norm",)

    def _boom(module, **kwargs):
        raise RuntimeError("fp8 unsupported for this layer")

    hooks.apply_layerwise_casting = _boom
    monkeypatch.setitem(sys.modules, "diffusers.hooks", hooks)
    monkeypatch.setitem(sys.modules, "diffusers.hooks.layerwise_casting", casting)
    pipe = types.SimpleNamespace(text_encoder = object())
    # The only encoder fails to cast -> nothing applied -> None.
    assert quantize_text_encoders(pipe, _target(), mode = "fp8") is None


# ── int8 (selective) + fp8_dynamic routing ─────────────────────────────────────


def test_quantize_int8_uses_family_keep_bf16_schedule(monkeypatch):
    # int8 for a family with a measured schedule routes to the selective caster with
    # that family's (skip_first, skip_last); qwen-image keeps first+last 6 blocks bf16.
    _stub_torch(monkeypatch, cc = (10, 0))
    calls: list = []
    monkeypatch.setattr(
        dp, "_cast_int8_selective", lambda enc, tgt, first, last: calls.append((enc, first, last))
    )
    te = object()
    pipe = types.SimpleNamespace(text_encoder = te)
    mode = quantize_text_encoders(pipe, _target(), mode = "int8", family = "qwen-image")
    assert mode == TE_QUANT_INT8
    assert calls == [(te, 6, 6)]


def test_quantize_int8_unknown_family_falls_back_to_fp8(monkeypatch):
    # A family without an int8 keep-bf16 schedule falls back to layerwise fp8 (logged),
    # never silently running full int8 that would degrade the encoder.
    _stub_torch(monkeypatch, cc = (10, 0))
    int8_calls: list = []
    fp8_calls: list = []
    monkeypatch.setattr(dp, "_cast_int8_selective", lambda *a: int8_calls.append(a))
    monkeypatch.setattr(dp, "_cast_fp8", lambda enc, tgt: fp8_calls.append(enc))
    te = object()
    pipe = types.SimpleNamespace(text_encoder = te)
    mode = quantize_text_encoders(pipe, _target(), mode = "int8", family = "wan-umt5")
    assert mode == TE_QUANT_FP8
    assert int8_calls == [] and fp8_calls == [te]


def test_quantize_fp8_dynamic_uses_compute_caster(monkeypatch):
    # fp8_dynamic routes to the torchao per-row compute caster (not the layerwise one)
    # and needs no per-family schedule.
    _stub_torch(monkeypatch, cc = (9, 0))
    calls: list = []
    monkeypatch.setattr(dp, "_cast_fp8_dynamic", lambda enc, tgt: calls.append(enc))
    te = object()
    pipe = types.SimpleNamespace(text_encoder = te)
    mode = quantize_text_encoders(pipe, _target(), mode = "fp8_dynamic")
    assert mode == TE_QUANT_FP8_DYNAMIC
    assert calls == [te]


def test_quantize_int8_unsupported_hw_is_noop(monkeypatch):
    # int8 on pre-Ampere silicon (no int8 tensor cores) applies nothing.
    _stub_torch(monkeypatch, cc = (7, 5))
    monkeypatch.setattr(dp, "_cast_int8_selective", lambda *a: pytest.fail("must not cast"))
    pipe = types.SimpleNamespace(text_encoder = object())
    assert quantize_text_encoders(pipe, _target(), mode = "int8", family = "qwen-image") is None


# ── block selection + real int8 filter closure ─────────────────────────────────


def test_keep_bf16_block_fqns_selects_first_and_last(monkeypatch):
    torch = _stub_torch(monkeypatch)
    module_list = torch.nn.ModuleList
    layers = module_list([object() for _ in range(10)])
    # A short stack (<= skip_first + skip_last) contributes nothing (keeping it all would
    # leave no interior to quantise).
    short = module_list([object() for _ in range(4)])
    enc = types.SimpleNamespace()
    enc.named_modules = lambda: [("", enc), ("model.layers", layers), ("aux.blocks", short)]
    keep = _keep_bf16_block_fqns(enc, 3, 2)
    assert keep == {
        "model.layers.0",
        "model.layers.1",
        "model.layers.2",
        "model.layers.8",
        "model.layers.9",
    }


def _stub_transformer_quant(monkeypatch, captured):
    # Reuse the committed factory's names but record what the int8 caster hands quantize_().
    dtq = types.ModuleType("core.inference.diffusion_transformer_quant")
    dtq.TQ_INT8 = "int8"
    dtq.TQ_FP8 = "fp8"
    dtq.DEFAULT_MIN_LINEAR_FEATURES = 512
    dtq._make_quant_config = lambda scheme, *a, **k: f"cfg:{scheme}"
    dtq.exclude_tokens_for_scheme = lambda scheme: ("modulation",)

    def _make_filter_fn(min_features, exclude_name_tokens = ()):
        def _f(module, fqn = ""):
            return not any(tok in fqn for tok in exclude_name_tokens)

        return _f

    dtq.make_filter_fn = _make_filter_fn
    monkeypatch.setitem(sys.modules, "core.inference.diffusion_transformer_quant", dtq)

    tq = types.ModuleType("torchao.quantization")

    def _quantize_(module, config, filter_fn = None):
        captured["config"] = config
        captured["filter_fn"] = filter_fn

    tq.quantize_ = _quantize_
    monkeypatch.setitem(sys.modules, "torchao.quantization", tq)


def test_int8_filter_keeps_blocks_and_towers_dense(monkeypatch):
    # The real selective closure: interior Linears quantise, but the kept first blocks,
    # the vision tower, lm_head, and the encoder's fp32-kept modules (T5 "wo") stay bf16.
    torch = _stub_torch(monkeypatch)
    captured: dict = {}
    _stub_transformer_quant(monkeypatch, captured)
    layers = torch.nn.ModuleList([object() for _ in range(8)])
    enc = types.SimpleNamespace(_keep_in_fp32_modules = ["wo"])
    enc.named_modules = lambda: [("model.layers", layers)]

    _cast_int8_selective(enc, _target(), 3, 0)
    assert captured["config"] == "cfg:int8"
    ff = captured["filter_fn"]
    # Kept first-3 decoder blocks stay bf16.
    assert ff(object(), "model.layers.0.self_attn.q_proj") is False
    assert ff(object(), "model.layers.2.mlp.gate_proj") is False
    # An interior block is quantised.
    assert ff(object(), "model.layers.5.self_attn.q_proj") is True
    # Vision tower / lm_head / T5 wo are excluded by the shared token filter.
    assert ff(object(), "visual.blocks.0.attn.qkv") is False
    assert ff(object(), "lm_head") is False
    assert ff(object(), "model.decoder.wo") is False
