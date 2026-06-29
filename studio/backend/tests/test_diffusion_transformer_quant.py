# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit tests for transformer quantisation (``diffusion_transformer_quant.py``).

Hermetic: torch + torchao are stubbed via ``sys.modules``, and the per-scheme smoke
probe (``_scheme_supported`` / ``_smoke_probe``) is monkeypatched where the test cares
about the selection ladder rather than the GPU probe, so everything runs CPU-only.
"""

from __future__ import annotations

import sys
import types

import pytest

import core.inference.diffusion_transformer_quant as tq
from core.inference.diffusion_transformer_quant import (
    TQ_FP8,
    TQ_INT8,
    TQ_MXFP8,
    TQ_NVFP4,
    dense_transformer_supported,
    make_filter_fn,
    normalize_transformer_quant,
    quantize_transformer,
    select_transformer_quant_scheme,
)


def _target(*, device = "cuda", dtype = "bfloat16"):
    return types.SimpleNamespace(device = device, dtype = dtype)


def _stub_torch(
    monkeypatch,
    *,
    cc = (10, 0),
    with_fp8 = True,
    cuda_available = True,
    device_name = "NVIDIA B200",
):
    torch = types.ModuleType("torch")
    torch.bfloat16 = "bfloat16"
    torch.float16 = "float16"
    if with_fp8:
        torch.float8_e4m3fn = "float8_e4m3fn"
    torch.cuda = types.SimpleNamespace(
        is_available = lambda: cuda_available,
        get_device_capability = lambda *a: cc,
        # data-center name by default so the ladder tests get the data-center order;
        # consumer tests pass a GeForce name (or monkeypatch _is_consumer_gpu).
        get_device_name = lambda *a: device_name,
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    return torch


# ── normalisation ─────────────────────────────────────────────────────────────


def test_normalize_transformer_quant():
    assert normalize_transformer_quant(None) is None
    assert normalize_transformer_quant("") is None
    assert normalize_transformer_quant("none") is None
    assert normalize_transformer_quant("off") is None
    assert normalize_transformer_quant("AUTO") == "auto"
    assert normalize_transformer_quant("INT8") == TQ_INT8
    assert normalize_transformer_quant("fp8") == TQ_FP8
    with pytest.raises(ValueError):
        normalize_transformer_quant("int2")


# ── dense-source gate ───────────────────────────────────────────────────────────


def test_dense_transformer_supported_requires_cuda_bf16(monkeypatch):
    _stub_torch(monkeypatch)
    assert dense_transformer_supported(_target()) is True
    assert dense_transformer_supported(_target(device = "cpu")) is False
    assert dense_transformer_supported(_target(dtype = "float16")) is False


# ── scheme selection ladder ─────────────────────────────────────────────────────


def _allow(monkeypatch, allowed):
    """Force ``_scheme_supported`` to accept only ``allowed`` (simulates smoke results)."""
    monkeypatch.setattr(tq, "_scheme_supported", lambda scheme, device: scheme in allowed)


def test_auto_blackwell_prefers_fp8_then_falls_back(monkeypatch):
    _stub_torch(monkeypatch, cc = (10, 0))
    # Even with every scheme available, auto picks fp8 on Blackwell: measured on a B200
    # (torch 2.11 + torchao CUTLASS FP4), fp8 is both faster and more accurate than nvfp4
    # for the DiT's shapes -- nvfp4's FP4 GEMM only wins on very large GEMMs, not here.
    _allow(monkeypatch, {TQ_NVFP4, TQ_MXFP8, TQ_FP8, TQ_INT8})
    assert select_transformer_quant_scheme(_target(), "auto") == TQ_FP8
    # fp8 unavailable: nvfp4 is the next pick (above mxfp8 / int8).
    _allow(monkeypatch, {TQ_NVFP4, TQ_MXFP8, TQ_INT8})
    assert select_transformer_quant_scheme(_target(), "auto") == TQ_NVFP4
    # Only mxfp8 + int8 left -> mxfp8 (still above int8).
    _allow(monkeypatch, {TQ_MXFP8, TQ_INT8})
    assert select_transformer_quant_scheme(_target(), "auto") == TQ_MXFP8
    # Only int8 usable -> int8.
    _allow(monkeypatch, {TQ_INT8})
    assert select_transformer_quant_scheme(_target(), "auto") == TQ_INT8


def test_auto_consumer_blackwell_prefers_int8(monkeypatch):
    # Consumer Blackwell (RTX 50xx): fp8 FP32-accumulate is throughput-halved while int8 is
    # full-rate, so auto prefers int8 even though fp8 is available (the data-center default).
    _stub_torch(monkeypatch, cc = (12, 0), device_name = "NVIDIA GeForce RTX 5090")
    _allow(monkeypatch, {TQ_NVFP4, TQ_MXFP8, TQ_FP8, TQ_INT8})
    assert select_transformer_quant_scheme(_target(), "auto") == TQ_INT8
    # int8 unavailable -> falls back to the rest of the tier (fp8 next).
    _allow(monkeypatch, {TQ_NVFP4, TQ_MXFP8, TQ_FP8})
    assert select_transformer_quant_scheme(_target(), "auto") == TQ_FP8


def test_auto_consumer_ada_prefers_int8(monkeypatch):
    # Consumer Ada (RTX 4090): int8 runs ~2x fp8's nerfed FP32-accumulate rate.
    _stub_torch(monkeypatch, cc = (8, 9), device_name = "NVIDIA GeForce RTX 4090")
    _allow(monkeypatch, {TQ_FP8, TQ_INT8})
    assert select_transformer_quant_scheme(_target(), "auto") == TQ_INT8


def test_auto_workstation_unknown_prefers_int8(monkeypatch):
    # Unknown / workstation name -> treated as consumer (the safe default) -> int8 first.
    _stub_torch(monkeypatch, cc = (8, 9), device_name = "NVIDIA RTX 6000 Ada Generation")
    _allow(monkeypatch, {TQ_FP8, TQ_INT8})
    assert select_transformer_quant_scheme(_target(), "auto") == TQ_INT8


def test_auto_ada_hopper_prefers_fp8(monkeypatch):
    # Data-center Ada (L40S) / Hopper (H100): not nerfed -> fp8 first.
    _stub_torch(monkeypatch, cc = (8, 9), device_name = "NVIDIA L40S")
    _allow(monkeypatch, {TQ_NVFP4, TQ_MXFP8, TQ_FP8, TQ_INT8})
    assert select_transformer_quant_scheme(_target(), "auto") == TQ_FP8
    _stub_torch(monkeypatch, cc = (9, 0), device_name = "NVIDIA H100 80GB HBM3")  # Hopper
    assert select_transformer_quant_scheme(_target(), "auto") == TQ_FP8


def test_auto_ampere_prefers_int8(monkeypatch):
    _stub_torch(monkeypatch, cc = (8, 0))
    _allow(monkeypatch, {TQ_FP8, TQ_INT8})  # fp8 cores absent on Ampere -> int8 only in ladder
    assert select_transformer_quant_scheme(_target(), "auto") == TQ_INT8
    _stub_torch(monkeypatch, cc = (8, 6))
    assert select_transformer_quant_scheme(_target(), "auto") == TQ_INT8


def test_auto_pre_ampere_unsupported(monkeypatch):
    _stub_torch(monkeypatch, cc = (7, 5))  # Turing: below the int8-dynamic floor
    _allow(monkeypatch, {TQ_INT8, TQ_FP8})
    assert select_transformer_quant_scheme(_target(), "auto") is None


def test_explicit_scheme_honored_or_none(monkeypatch):
    _stub_torch(monkeypatch, cc = (8, 0))
    _allow(monkeypatch, {TQ_INT8})
    assert select_transformer_quant_scheme(_target(), "int8") == TQ_INT8
    # Explicit unsupported scheme is NOT silently downgraded -> None (-> GGUF fallback).
    assert select_transformer_quant_scheme(_target(), "fp8") is None
    assert select_transformer_quant_scheme(_target(), "nvfp4") is None


def test_select_none_when_disabled_or_non_cuda(monkeypatch):
    _stub_torch(monkeypatch)
    _allow(monkeypatch, {TQ_INT8, TQ_FP8, TQ_NVFP4})
    assert select_transformer_quant_scheme(_target(), None) is None
    assert select_transformer_quant_scheme(_target(device = "cpu"), "auto") is None


# ── _scheme_supported / _smoke_probe ────────────────────────────────────────────


def test_scheme_supported_shortcircuits(monkeypatch):
    # No CUDA -> False without running the smoke probe.
    _stub_torch(monkeypatch, cuda_available = False)
    monkeypatch.setattr(tq, "_smoke_probe", lambda *a: pytest.fail("probe should not run"))
    assert tq._scheme_supported(TQ_INT8, "cuda") is False
    # fp8 requested but the fp8 dtype is missing -> False before the probe.
    _stub_torch(monkeypatch, with_fp8 = False)
    monkeypatch.setattr(tq, "_smoke_probe", lambda *a: pytest.fail("probe should not run"))
    assert tq._scheme_supported(TQ_FP8, "cuda") is False


def test_smoke_probe_caches_and_tolerates_failure(monkeypatch):
    tq._SMOKE_CACHE.clear()
    calls = {"n": 0}

    class _Lin:
        def __init__(self, *a, **k):
            pass

        def to(self, **k):
            return self

    torch = types.ModuleType("torch")
    torch.bfloat16 = "bfloat16"
    torch.nn = types.SimpleNamespace(Linear = _Lin)
    torch.randn = lambda *a, **k: object()
    torch.no_grad = lambda: __import__("contextlib").nullcontext()
    torch.cuda = types.SimpleNamespace(is_available = lambda: True, synchronize = lambda: None)
    monkeypatch.setitem(sys.modules, "torch", torch)

    tqz = types.ModuleType("torchao.quantization")

    def _quantize_ok(
        module,
        config,
        filter_fn = None,
    ):
        calls["n"] += 1

    tqz.quantize_ = _quantize_ok
    tqz.Int8DynamicActivationInt8WeightConfig = lambda: "int8cfg"
    tqz.Float8DynamicActivationFloat8WeightConfig = lambda: "fp8cfg"
    monkeypatch.setitem(sys.modules, "torchao.quantization", tqz)
    # _Lin is callable? No -> the forward lin(x) would fail. Make instances callable.
    _Lin.__call__ = lambda self, x: x

    assert tq._smoke_probe(TQ_INT8, "cuda") is True
    assert tq._smoke_probe(TQ_INT8, "cuda") is True  # cached, no second quantize_
    assert calls["n"] == 1

    # A scheme whose quantize_ raises -> probe False (and cached).
    tq._SMOKE_CACHE.clear()

    def _quantize_boom(
        module,
        config,
        filter_fn = None,
    ):
        raise RuntimeError("kernel unavailable")

    tqz.quantize_ = _quantize_boom
    assert tq._smoke_probe(TQ_FP8, "cuda") is False


# ── consumer-vs-datacenter detection (fp8 fast-accumulate gate) ──────────────────


def _stub_device_name(monkeypatch, name):
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(get_device_name = lambda device = None: name)
    monkeypatch.setitem(sys.modules, "torch", torch)


@pytest.mark.parametrize(
    "name",
    [
        "NVIDIA GeForce RTX 5090",
        "NVIDIA GeForce RTX 4090",
        "NVIDIA RTX A4000",  # workstation: A4000 token, NOT the data-center A40
        "NVIDIA RTX 6000 Ada Generation",
        "NVIDIA Some Future Card 9000",  # unknown -> default consumer (fast accum is free on DC)
    ],
)
def test_is_consumer_gpu_true(monkeypatch, name):
    _stub_device_name(monkeypatch, name)
    assert tq._is_consumer_gpu() is True


@pytest.mark.parametrize(
    "name",
    [
        "NVIDIA B200",
        "NVIDIA H100 80GB HBM3",
        "NVIDIA A100-SXM4-80GB",
        "NVIDIA A40",  # data-center Ampere (distinct token from RTX A4000)
        "NVIDIA L40S",
        "NVIDIA L4",
        "Tesla V100-SXM2-16GB",
    ],
)
def test_is_consumer_gpu_false_for_datacenter(monkeypatch, name):
    _stub_device_name(monkeypatch, name)
    assert tq._is_consumer_gpu() is False


def test_is_consumer_gpu_defaults_true_on_probe_failure(monkeypatch):
    # No torch / no device name available -> assume consumer (safe: fast accum is free
    # on data center and a win on consumer).
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace()  # no get_device_name
    monkeypatch.setitem(sys.modules, "torch", torch)
    assert tq._is_consumer_gpu() is True


# ── filter ──────────────────────────────────────────────────────────────────────


def test_make_filter_fn(monkeypatch):
    class _Lin:
        def __init__(self, i, o):
            self.in_features, self.out_features = i, o

    torch = types.ModuleType("torch")
    torch.nn = types.SimpleNamespace(Linear = _Lin)
    monkeypatch.setitem(sys.modules, "torch", torch)

    keep = make_filter_fn(512)
    assert keep(_Lin(1024, 4096), "blocks.0.attn.to_q") is True
    assert keep(_Lin(256, 4096), "time_proj") is False  # small in_features -> skip
    assert keep(_Lin(4096, 256), "out_proj") is False  # small out_features -> skip
    assert keep(object(), "not_linear") is False  # non-Linear -> skip
    assert keep(types.SimpleNamespace(), "no_attrs") is False


def test_make_filter_fn_int8_excludes_modulation_and_embedders(monkeypatch):
    # The int8 path skips the large M=1 AdaLN modulation / conditioning-embedder projections
    # (they crash torch._int_mm's M>16), while keeping the attention / FFN compute layers and
    # the sequence embedders. fp8 (no exclusion) keeps everything.
    from core.inference.diffusion_transformer_quant import _INT8_EXCLUDE_NAME_TOKENS

    class _Lin:
        def __init__(self, i, o):
            self.in_features, self.out_features = i, o

    torch = types.ModuleType("torch")
    torch.nn = types.SimpleNamespace(Linear = _Lin)
    monkeypatch.setitem(sys.modules, "torch", torch)

    keep = make_filter_fn(512, exclude_name_tokens = _INT8_EXCLUDE_NAME_TOKENS)
    big = lambda: _Lin(3072, 18432)  # noqa: E731 — large enough to pass min_features
    # Excluded (M=1 modulation / conditioning embedders), despite large features:
    for fqn in (
        "transformer_blocks.0.norm1.linear",
        "transformer_blocks.0.norm1_context.linear",
        "single_transformer_blocks.0.norm.linear",
        "norm_out.linear",
        "transformer_blocks.0.img_mod.1",
        "transformer_blocks.0.txt_mod.1",
        "double_stream_modulation_img.linear",
        "time_text_embed.timestep_embedder.linear_2",
        "time_text_embed.guidance_embedder.linear_2",
        "time_guidance_embed.timestep_embedder.linear_2",
    ):
        assert keep(big(), fqn) is False, fqn
    # Kept (M=seq compute layers + sequence embedders), NOT matched by the modulation tokens:
    for fqn in (
        "transformer_blocks.0.attn.to_q",
        "transformer_blocks.0.ff.net.0.proj",
        "single_transformer_blocks.0.proj_mlp",
        "single_transformer_blocks.0.attn.to_qkv_mlp_proj",
        "context_embedder",  # "context" contains "text" -> must NOT be excluded
        "txt_in",
    ):
        assert keep(big(), fqn) is True, fqn
    # Without the exclusion (fp8 path), the modulation layer is kept.
    assert make_filter_fn(512)(big(), "transformer_blocks.0.norm1.linear") is True
    # A None / empty fqn must not crash the exclusion check (defensive against the callback
    # passing no name); with no name nothing matches the exclusion tokens -> kept.
    assert keep(big(), None) is True
    assert keep(big(), "") is True


def test_exclude_tokens_for_scheme_shared_by_runtime_and_builder():
    # The runtime quantiser and the offline prequant builder must apply the SAME int8
    # exclusion, or an int8 prequant artifact quantises the M=1 modulation/embedder linears
    # and reintroduces the torch._int_mm crash. int8 gets the exclusion; others get none.
    from core.inference.diffusion_transformer_quant import (
        _INT8_EXCLUDE_NAME_TOKENS,
        exclude_tokens_for_scheme,
    )

    assert exclude_tokens_for_scheme(TQ_INT8) == _INT8_EXCLUDE_NAME_TOKENS
    for scheme in (TQ_FP8, TQ_NVFP4, TQ_MXFP8):
        assert exclude_tokens_for_scheme(scheme) == ()


# ── apply ───────────────────────────────────────────────────────────────────────


def test_resolve_fast_accum(monkeypatch):
    # None auto-detects by GPU class; an explicit bool forces it.
    monkeypatch.setattr(tq, "_is_consumer_gpu", lambda *a: True)
    assert tq._resolve_fast_accum(None) is True
    monkeypatch.setattr(tq, "_is_consumer_gpu", lambda *a: False)
    assert tq._resolve_fast_accum(None) is False
    assert tq._resolve_fast_accum(True) is True  # forced on (e.g. on a data-center card)
    assert tq._resolve_fast_accum(False) is False  # forced off (e.g. on a consumer card)


def test_quantize_transformer_applies_and_marks(monkeypatch):
    monkeypatch.setattr(tq, "select_transformer_quant_scheme", lambda target, mode: TQ_FP8)
    seen: dict = {}

    def _mk(scheme, fast_accum = None):
        seen["scheme"], seen["fast_accum"] = scheme, fast_accum
        return f"{scheme}cfg"

    monkeypatch.setattr(tq, "_make_quant_config", _mk)
    recorder: list = []
    tqz = types.ModuleType("torchao.quantization")
    tqz.quantize_ = lambda module, config, filter_fn = None: recorder.append(
        (module, config, filter_fn)
    )
    monkeypatch.setitem(sys.modules, "torchao.quantization", tqz)

    transformer = types.SimpleNamespace()
    pipe = types.SimpleNamespace(transformer = transformer)
    assert quantize_transformer(pipe, _target(), mode = "fp8", fast_accum = False) == TQ_FP8
    assert len(recorder) == 1 and recorder[0][0] is transformer and recorder[0][1] == "fp8cfg"
    assert callable(recorder[0][2])  # a filter_fn was passed
    assert transformer._unsloth_runtime_quant == TQ_FP8  # diagnostic marker set
    assert seen["fast_accum"] is False  # the override is forwarded into the config


def test_quantize_transformer_none_when_unsupported(monkeypatch):
    monkeypatch.setattr(tq, "select_transformer_quant_scheme", lambda target, mode: None)
    pipe = types.SimpleNamespace(transformer = types.SimpleNamespace())
    assert quantize_transformer(pipe, _target(), mode = "auto") is None


def test_quantize_transformer_tolerates_failure(monkeypatch):
    monkeypatch.setattr(tq, "select_transformer_quant_scheme", lambda target, mode: TQ_INT8)
    monkeypatch.setattr(tq, "_make_quant_config", lambda scheme: "cfg")
    tqz = types.ModuleType("torchao.quantization")

    def _boom(
        module,
        config,
        filter_fn = None,
    ):
        raise RuntimeError("partial quant failure")

    tqz.quantize_ = _boom
    monkeypatch.setitem(sys.modules, "torchao.quantization", tqz)
    pipe = types.SimpleNamespace(transformer = types.SimpleNamespace())
    # A quantise failure returns None (caller falls back to GGUF), never raises.
    assert quantize_transformer(pipe, _target(), mode = "int8") is None
