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
    TE_QUANT_AUTO,
    TE_QUANT_FP8,
    TE_QUANT_FP8_DYNAMIC,
    TE_QUANT_INT8,
    TE_QUANT_NVFP4,
    _cast_int8_selective,
    _cast_nvfp4,
    _keep_bf16_block_fqns,
    normalize_te_quant,
    quantize_text_encoders,
    select_te_quant_scheme,
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
    # torchao nvfp4 -- quantize_ now receives the vision-tower exclusion filter_fn; accept + ignore.
    tq = types.ModuleType("torchao.quantization")
    tq.quantize_ = lambda module, config, filter_fn = None: recorder.append(("nvfp4", module))
    mx = types.ModuleType("torchao.prototype.mx_formats")
    mx.NVFP4WeightOnlyConfig = lambda: "nvfp4cfg"
    monkeypatch.setitem(sys.modules, "torchao.quantization", tq)
    monkeypatch.setitem(sys.modules, "torchao.prototype.mx_formats", mx)
    # _cast_nvfp4 / _cast_fp8_dynamic pull the shared linear filter from the transformer-quant module.
    dtq = types.ModuleType("core.inference.diffusion_transformer_quant")
    dtq.DEFAULT_MIN_LINEAR_FEATURES = 512
    dtq.make_filter_fn = lambda min_features, exclude = (), *, require_bf16 = False: (
        lambda module, fqn = "": True
    )
    # The explicit-torchao path now runs the same kernel smoke test the auto ladder uses; pass it
    # by default so these caster tests exercise the cast, not a broken-kernel fallback.
    dtq._smoke_probe = lambda tq, device: True
    monkeypatch.setitem(sys.modules, "core.inference.diffusion_transformer_quant", dtq)
    # nvfp4 TE probes its own weight-only kernel (not the dynamic _smoke_probe); pass it too.
    monkeypatch.setattr(dp, "_te_nvfp4_weightonly_probe", lambda device: True)


# ── normalisation ─────────────────────────────────────────────────────────────


def test_normalize_te_quant():
    assert normalize_te_quant(None) is None
    assert normalize_te_quant("") is None
    assert normalize_te_quant("none") is None
    # "off" disables (like the transformer's normalize) -> dense.
    assert normalize_te_quant("off") is None
    # "auto" passes through for select_te_quant_scheme to resolve.
    assert normalize_te_quant("AUTO") == TE_QUANT_AUTO
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
    monkeypatch.setattr(dp, "_te_scheme_probe", lambda scheme, device: True)
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
    monkeypatch.setattr(dp, "_te_scheme_probe", lambda scheme, device: True)
    calls: list = []
    monkeypatch.setattr(dp, "_cast_fp8_dynamic", lambda enc, tgt: calls.append(enc))
    te = object()
    pipe = types.SimpleNamespace(text_encoder = te)
    mode = quantize_text_encoders(pipe, _target(), mode = "fp8_dynamic")
    assert mode == TE_QUANT_FP8_DYNAMIC
    assert calls == [te]


def test_quantize_explicit_torchao_probes_kernel(monkeypatch):
    # An EXPLICIT torchao TE mode (int8 / fp8_dynamic / nvfp4) clears the capability gate but must
    # still run the real GEMM smoke test the auto ladder uses: on a build where quantize_ wraps the
    # encoder yet the kernel is broken, report dense (None) instead of crashing on the first forward.
    _stub_torch(monkeypatch, cc = (10, 0))
    monkeypatch.setattr(dp, "_te_scheme_probe", lambda scheme, device: False)
    monkeypatch.setattr(
        dp, "_cast_fp8_dynamic", lambda *a: pytest.fail("must not cast on probe fail")
    )
    monkeypatch.setattr(dp, "_cast_nvfp4", lambda *a: pytest.fail("must not cast on probe fail"))
    monkeypatch.setattr(
        dp, "_cast_int8_selective", lambda *a: pytest.fail("must not cast on probe fail")
    )
    pipe = types.SimpleNamespace(text_encoder = object())
    assert quantize_text_encoders(pipe, _target(), mode = "fp8_dynamic") is None
    assert quantize_text_encoders(pipe, _target(), mode = "nvfp4") is None
    assert quantize_text_encoders(pipe, _target(), mode = "int8", family = "qwen-image") is None


def test_te_scheme_probe_bypasses_layerwise_fp8():
    # Layerwise fp8 has no torchao GEMM (not in _TE_SMOKE_SCHEME), so the probe is a no-op (True)
    # for it and never vetoes it -- this is why the explicit-torchao veto above leaves plain fp8
    # casting untouched. The torchao schemes DO carry a smoke scheme.
    assert dp._te_scheme_probe(TE_QUANT_FP8, "cuda") is True
    assert TE_QUANT_FP8 not in dp._TE_SMOKE_SCHEME
    for scheme in (TE_QUANT_FP8_DYNAMIC, TE_QUANT_INT8, TE_QUANT_NVFP4):
        assert scheme in dp._TE_SMOKE_SCHEME


def test_te_scheme_probe_nvfp4_uses_weightonly_kernel(monkeypatch):
    # nvfp4 TE casts weight-only (_cast_nvfp4 -> NVFP4WeightOnlyConfig), a different torchao kernel
    # from the transformer's dynamic-activation NVFP4 probe. On a build where the dynamic FP4 GEMM
    # is unavailable but weight-only FP4 works, the nvfp4 TE probe must consult its own weight-only
    # probe, not the transformer dynamic one, or an explicit request would falsely stay dense.
    dp._TE_NVFP4_PROBE_CACHE.clear()
    dtq = types.ModuleType("core.inference.diffusion_transformer_quant")
    dtq._smoke_probe = lambda scheme, device: False  # every dynamic-activation GEMM "unavailable"
    monkeypatch.setitem(sys.modules, "core.inference.diffusion_transformer_quant", dtq)
    monkeypatch.setattr(dp, "_te_nvfp4_weightonly_probe", lambda device: True)
    # nvfp4 follows its own weight-only probe (True), not the transformer dynamic probe (False).
    assert dp._te_scheme_probe(TE_QUANT_NVFP4, "cuda") is True
    # int8 / fp8_dynamic still follow the (dynamic) transformer probe -> False here.
    assert dp._te_scheme_probe(TE_QUANT_INT8, "cuda") is False
    assert dp._te_scheme_probe(TE_QUANT_FP8_DYNAMIC, "cuda") is False


def test_quantize_explicit_denied_scheme_stays_dense(monkeypatch):
    # _TE_FAMILY_SCHEME_DENY's contract: a denied scheme is refused even when requested
    # explicitly (mirroring the VAE module), gating the FINAL concrete mode so an
    # int8 -> fp8 fallback is re-checked too.
    _stub_torch(monkeypatch, cc = (10, 0))
    recorder: list = []
    _stub_casters(monkeypatch, recorder)
    monkeypatch.setitem(dp._TE_FAMILY_SCHEME_DENY, "z-image", frozenset({TE_QUANT_FP8_DYNAMIC}))
    pipe = types.SimpleNamespace(text_encoder = object())
    assert quantize_text_encoders(pipe, _target(), mode = "fp8_dynamic", family = "z-image") is None
    assert recorder == []  # denied before any cast


def test_quantize_int8_unsupported_hw_is_noop(monkeypatch):
    # int8 on pre-Ampere silicon (no int8 tensor cores) applies nothing.
    _stub_torch(monkeypatch, cc = (7, 5))
    monkeypatch.setattr(dp, "_cast_int8_selective", lambda *a: pytest.fail("must not cast"))
    pipe = types.SimpleNamespace(text_encoder = object())
    assert quantize_text_encoders(pipe, _target(), mode = "int8", family = "qwen-image") is None


def test_quantize_te_skips_torchao_modes_under_offload(monkeypatch):
    # The torchao modes (int8-with-schedule / fp8_dynamic / nvfp4) produce tensor subclasses that
    # reject Module.to(), which an offload hook uses, so they must be skipped under offload (the DiT
    # path skips torchao quant for the same reason). Hardware supports every mode here, so a None
    # result proves the offload skip, not a capability gate; the casters fail if wrongly invoked.
    _stub_torch(monkeypatch, cc = (10, 0))
    monkeypatch.setattr(
        dp, "_cast_fp8_dynamic", lambda *a: pytest.fail("torchao caster must not run")
    )
    monkeypatch.setattr(dp, "_cast_nvfp4", lambda *a: pytest.fail("torchao caster must not run"))
    monkeypatch.setattr(
        dp, "_cast_int8_selective", lambda *a: pytest.fail("torchao caster must not run")
    )
    pipe = types.SimpleNamespace(text_encoder = object())
    assert quantize_text_encoders(pipe, _target(), mode = "fp8_dynamic", offload_active = True) is None
    assert quantize_text_encoders(pipe, _target(), mode = "nvfp4", offload_active = True) is None
    assert (
        quantize_text_encoders(
            pipe, _target(), mode = "int8", family = "qwen-image", offload_active = True
        )
        is None
    )
    # Layerwise fp8 is not torchao and streams fine under offload, so it still engages.
    fp8_calls: list = []
    monkeypatch.setattr(dp, "_cast_fp8", lambda enc, tgt: fp8_calls.append(enc))
    assert quantize_text_encoders(pipe, _target(), mode = "fp8", offload_active = True) == TE_QUANT_FP8
    assert len(fp8_calls) == 1


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

    def _make_filter_fn(
        min_features,
        exclude_name_tokens = (),
        *,
        require_bf16 = False,
    ):
        def _f(module, fqn = ""):
            return not any(tok in fqn for tok in exclude_name_tokens)

        return _f

    dtq.make_filter_fn = _make_filter_fn
    monkeypatch.setitem(sys.modules, "core.inference.diffusion_transformer_quant", dtq)

    tq = types.ModuleType("torchao.quantization")

    def _quantize_(
        module,
        config,
        filter_fn = None,
    ):
        captured["config"] = config
        captured["filter_fn"] = filter_fn

    tq.quantize_ = _quantize_
    monkeypatch.setitem(sys.modules, "torchao.quantization", tq)
    # _cast_nvfp4 builds its config from here.
    mx = types.ModuleType("torchao.prototype.mx_formats")
    mx.NVFP4WeightOnlyConfig = lambda: "nvfp4cfg"
    monkeypatch.setitem(sys.modules, "torchao.prototype.mx_formats", mx)


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


def test_nvfp4_filter_keeps_vision_tower_dense(monkeypatch):
    # Weight-only NVFP4 on a text encoder must exclude the VLM vision tower / lm_head / T5 "wo"
    # like the int8 / fp8 torchao TE modes -- 4-bit-ing a qwen-image(-edit) Qwen2.5-VL image tower
    # degrades the edit/image conditioning the sibling schemes deliberately protect. Before the fix
    # _cast_nvfp4 quantised every nn.Linear (no filter_fn), so the tower was silently 4-bit.
    _stub_torch(monkeypatch)
    captured: dict = {}
    _stub_transformer_quant(monkeypatch, captured)
    enc = types.SimpleNamespace(_keep_in_fp32_modules = ["wo"])

    _cast_nvfp4(enc, _target())

    assert captured["config"] == "nvfp4cfg"
    ff = captured["filter_fn"]
    assert ff is not None  # a filter is passed now, not None (which quantised everything)
    # Vision tower / lm_head / T5 wo stay bf16; an interior projection still quantises.
    assert ff(object(), "visual.blocks.0.attn.qkv") is False
    assert ff(object(), "vision_tower.encoder.layers.0.mlp.fc1") is False
    assert ff(object(), "lm_head") is False
    assert ff(object(), "model.decoder.wo") is False
    assert ff(object(), "model.layers.5.self_attn.q_proj") is True


# ── auto ladder (select_te_quant_scheme) ────────────────────────────────────────


def _stub_tq_select(
    monkeypatch,
    *,
    cc,
    consumer = False,
    smoke = True,
):
    """Stub the transformer module's shared helpers that select_te_quant_scheme imports:
    capability, GPU class, and the kernel smoke probe (bool or a (tq, dev) predicate)."""
    dtq = types.ModuleType("core.inference.diffusion_transformer_quant")
    dtq._capability = lambda: cc
    dtq._is_consumer_gpu = lambda device = None: consumer
    dtq._smoke_probe = smoke if callable(smoke) else (lambda tq, dev: smoke)
    monkeypatch.setitem(sys.modules, "core.inference.diffusion_transformer_quant", dtq)
    return dtq


def _allow_te(monkeypatch, allowed):
    """Force te_quant_supported to accept only ``allowed`` (simulates the hardware gate)."""
    monkeypatch.setattr(dp, "te_quant_supported", lambda target, mode: mode in allowed)


def test_select_te_auto_datacenter_prefers_fp8_dynamic(monkeypatch):
    # Data-center fp8-GEMM silicon: fp8_dynamic (compute fp8) leads the ladder.
    _stub_tq_select(monkeypatch, cc = (10, 0), consumer = False)
    _allow_te(monkeypatch, {TE_QUANT_FP8_DYNAMIC, TE_QUANT_INT8, TE_QUANT_FP8})
    assert select_te_quant_scheme(_target(), "auto", family = "qwen-image") == TE_QUANT_FP8_DYNAMIC


def test_select_te_auto_falls_through_to_int8_then_fp8(monkeypatch):
    _stub_tq_select(monkeypatch, cc = (10, 0))
    # fp8_dynamic unavailable -> int8 (family has a keep-bf16 schedule).
    _allow_te(monkeypatch, {TE_QUANT_INT8, TE_QUANT_FP8})
    assert select_te_quant_scheme(_target(), "auto", family = "qwen-image") == TE_QUANT_INT8
    # A family with NO int8 schedule skips int8 -> layerwise fp8.
    assert select_te_quant_scheme(_target(), "auto", family = "z-image") == TE_QUANT_FP8


def test_select_te_auto_consumer_prefers_int8(monkeypatch):
    # Consumer GDDR halves fp8 FP32-accumulate but runs int8 full-rate -> int8 first.
    _stub_tq_select(monkeypatch, cc = (10, 0), consumer = True)
    _allow_te(monkeypatch, {TE_QUANT_FP8_DYNAMIC, TE_QUANT_INT8, TE_QUANT_FP8})
    assert select_te_quant_scheme(_target(), "auto", family = "qwen-image") == TE_QUANT_INT8


def test_select_te_auto_offload_uses_layerwise_fp8(monkeypatch):
    # Under offload the torchao modes (reject Module.to()) are skipped -> layerwise fp8.
    _stub_tq_select(monkeypatch, cc = (10, 0))
    _allow_te(monkeypatch, {TE_QUANT_FP8_DYNAMIC, TE_QUANT_INT8, TE_QUANT_FP8})
    assert (
        select_te_quant_scheme(_target(), "auto", family = "qwen-image", offload_active = True)
        == TE_QUANT_FP8
    )


def test_select_te_auto_ampere_uses_int8(monkeypatch):
    # Ampere sm_80 has no fp8 GEMM; the tier is (int8, fp8).
    _stub_tq_select(monkeypatch, cc = (8, 0))
    _allow_te(monkeypatch, {TE_QUANT_INT8, TE_QUANT_FP8})
    assert select_te_quant_scheme(_target(), "auto", family = "qwen-image") == TE_QUANT_INT8


def test_select_te_auto_family_deny_skips_scheme(monkeypatch):
    _stub_tq_select(monkeypatch, cc = (10, 0))
    _allow_te(monkeypatch, {TE_QUANT_FP8_DYNAMIC, TE_QUANT_INT8, TE_QUANT_FP8})
    monkeypatch.setattr(
        dp, "_TE_FAMILY_SCHEME_DENY", {"qwen-image": frozenset({TE_QUANT_FP8_DYNAMIC})}
    )
    # fp8_dynamic denied for this family -> falls to int8.
    assert select_te_quant_scheme(_target(), "auto", family = "qwen-image") == TE_QUANT_INT8


def test_select_te_auto_smoke_failure_skips_scheme(monkeypatch):
    # fp8_dynamic is hardware-supported but its kernel smoke-probe fails -> skip to int8.
    _stub_tq_select(monkeypatch, cc = (10, 0), smoke = lambda tq, dev: tq != "fp8")
    _allow_te(monkeypatch, {TE_QUANT_FP8_DYNAMIC, TE_QUANT_INT8, TE_QUANT_FP8})
    assert select_te_quant_scheme(_target(), "auto", family = "qwen-image") == TE_QUANT_INT8


def test_select_te_auto_pre_ampere_and_no_cuda_are_none(monkeypatch):
    _stub_tq_select(monkeypatch, cc = (7, 5))
    _allow_te(monkeypatch, {TE_QUANT_FP8})
    assert select_te_quant_scheme(_target(), "auto", family = "qwen-image") is None
    _stub_tq_select(monkeypatch, cc = None)
    assert select_te_quant_scheme(_target(), "auto", family = "qwen-image") is None


def test_select_te_explicit_scheme_passes_through(monkeypatch):
    # An explicit request is returned as-is (quantize_text_encoders re-gates it); no ladder walk,
    # so no transformer-module stub is needed.
    assert select_te_quant_scheme(_target(), "fp8") == TE_QUANT_FP8
    assert select_te_quant_scheme(_target(), "int8") == TE_QUANT_INT8
    assert select_te_quant_scheme(_target(), None) is None
    assert select_te_quant_scheme(_target(), "none") is None


def test_quantize_text_encoders_auto_resolves_and_applies(monkeypatch):
    # End-to-end: mode="auto" resolves via the ladder then applies the resolved caster.
    _stub_tq_select(monkeypatch, cc = (10, 0))
    _allow_te(monkeypatch, {TE_QUANT_FP8_DYNAMIC, TE_QUANT_INT8, TE_QUANT_FP8})
    calls: list = []
    monkeypatch.setattr(dp, "_cast_fp8_dynamic", lambda enc, tgt: calls.append(enc))
    te = object()
    pipe = types.SimpleNamespace(text_encoder = te)
    mode = quantize_text_encoders(pipe, _target(), mode = "auto", family = "qwen-image")
    assert mode == TE_QUANT_FP8_DYNAMIC
    assert calls == [te]


def test_select_te_auto_resolves_dense_for_hunyuanvideo15(monkeypatch):
    # HunyuanVideo-1.5 (both repacks): TE quant perturbs the conditioning and the video
    # trajectory amplifies it chaotically (measured LPIPS 0.236 vs bit-exact from TE
    # fp8_dynamic ALONE, vs 0.052 for the rest of the stack) at zero speed win, so the
    # AUTO default keeps the encoder dense on ANY hardware.
    _stub_tq_select(monkeypatch, cc = (10, 0), consumer = False)
    _allow_te(monkeypatch, {TE_QUANT_FP8_DYNAMIC, TE_QUANT_INT8, TE_QUANT_FP8})
    assert select_te_quant_scheme(_target(), "auto", family = "hunyuanvideo-1.5") is None
    assert select_te_quant_scheme(_target(), "auto", family = "HunyuanVideo-1.5-720p") is None
    # Other families keep the normal ladder on the same stubbed hardware.
    assert select_te_quant_scheme(_target(), "auto", family = "qwen-image") == TE_QUANT_FP8_DYNAMIC


def test_select_te_auto_resolves_dense_for_wan_a14b_but_not_wan_5b(monkeypatch):
    # Wan2.2-A14B: TE fp8_dynamic alone costs pairwise LPIPS 0.1195 vs the dense-TE
    # stack for a 1.03x once-per-generation encode (146.7 -> 142.7 s e2e), so AUTO
    # keeps the encoder dense. Wan2.2-TI2V-5B shares the UMT5 encoder but measured
    # in-bar (0.0396 pairwise) at a real 1.09x on its far faster DiT, so it keeps the
    # normal ladder.
    _stub_tq_select(monkeypatch, cc = (10, 0), consumer = False)
    _allow_te(monkeypatch, {TE_QUANT_FP8_DYNAMIC, TE_QUANT_INT8, TE_QUANT_FP8})
    assert select_te_quant_scheme(_target(), "auto", family = "wan2.2-t2v-a14b") is None
    assert select_te_quant_scheme(_target(), "auto", family = "Wan2.2-T2V-A14B") is None
    assert (
        select_te_quant_scheme(_target(), "auto", family = "wan2.2-ti2v-5b") == TE_QUANT_FP8_DYNAMIC
    )
    # The auto-dense table steers only the DEFAULT; an explicit request stays verbatim.
    assert (
        select_te_quant_scheme(_target(), "fp8_dynamic", family = "wan2.2-t2v-a14b")
        == TE_QUANT_FP8_DYNAMIC
    )


def test_select_te_explicit_scheme_still_honored_for_hunyuanvideo15(monkeypatch):
    # The auto-dense table steers only the DEFAULT; an explicit request stays verbatim
    # (select returns it as-is; quantize_text_encoders re-gates hardware support).
    _stub_tq_select(monkeypatch, cc = (10, 0), consumer = False)
    _allow_te(monkeypatch, {TE_QUANT_FP8_DYNAMIC})
    assert (
        select_te_quant_scheme(_target(), "fp8_dynamic", family = "hunyuanvideo-1.5-720p")
        == TE_QUANT_FP8_DYNAMIC
    )


def test_select_te_auto_ltx2_denies_fp8_dynamic_falls_to_layerwise_fp8(monkeypatch):
    # LTX-2's Gemma3-27B encoder BLACK-FRAMES the whole clip under torchao per-row
    # compute fp8 (measured pairwise vs the dense encoder: mean luma 137.9 -> 0.0,
    # LPIPS 0.78), while layerwise fp8 is near-lossless (0.0043) at the same shrink --
    # so the family deny drops fp8_dynamic and auto falls through (int8 has no ltx-2
    # keep-bf16 schedule) to layerwise fp8.
    _stub_tq_select(monkeypatch, cc = (10, 0), consumer = False)
    _allow_te(monkeypatch, {TE_QUANT_FP8_DYNAMIC, TE_QUANT_INT8, TE_QUANT_FP8})
    assert select_te_quant_scheme(_target(), "auto", family = "ltx-2") == TE_QUANT_FP8


def test_quantize_explicit_fp8_dynamic_refused_for_ltx2(monkeypatch):
    # The deny contract covers EXPLICIT requests too: black frames are a model-level
    # breakage, not a preference, so the encoder stays dense instead.
    _allow_te(monkeypatch, {TE_QUANT_FP8_DYNAMIC})
    calls: list = []
    monkeypatch.setattr(dp, "_cast_fp8_dynamic", lambda enc, tgt: calls.append(enc))
    pipe = types.SimpleNamespace(text_encoder = object())
    assert quantize_text_encoders(pipe, _target(), mode = "fp8_dynamic", family = "ltx-2") is None
    assert calls == []


# ── zero-output-row guard (per-row fp8 NaN protection) ───────────────────────────


class _FakeAmaxVec:
    def __init__(self, vals):
        self._vals = vals

    def __eq__(self, other):  # noqa: PLW0642 -- tensor-style elementwise compare
        return _FakeAmaxVec([v == other for v in self._vals])

    def any(self):
        return _FakeScalar(any(self._vals))


class _FakeScalar:
    def __init__(self, v):
        self._v = v

    def item(self):
        return self._v


class _FakeWeight:
    """Tensor-shaped stand-in supporting the exact chain the guard runs:
    ``weight.abs().amax(dim = -1) == 0 -> .any().item()``."""

    ndim = 2

    def __init__(self, rows):
        self._rows = rows

    def abs(self):
        return _FakeWeight([[abs(v) for v in r] for r in self._rows])

    def amax(self, dim = -1):
        return _FakeAmaxVec([max(r) for r in self._rows])


def test_weight_zero_output_row_detection():
    # A dead output row NaNs torchao's per-row fp8 (scale 0 -> 0/0); SDXL's
    # text_encoder_2 (OpenCLIP bigG) really ships one in layers.2.self_attn.out_proj --
    # measured: every fp8_dynamic SDXL render was black until the row is kept dense.
    zero_row = types.SimpleNamespace(weight = _FakeWeight([[0.1, 0.2], [0.0, 0.0]]))
    dense = types.SimpleNamespace(weight = _FakeWeight([[0.1, 0.2], [0.3, 0.0]]))
    assert dp._weight_has_zero_output_row(zero_row) is True
    assert dp._weight_has_zero_output_row(dense) is False
    # Non-2D / absent weights are not the per-row scheme's input: never flagged.
    w3 = _FakeWeight([[1.0]])
    w3.ndim = 3
    assert dp._weight_has_zero_output_row(types.SimpleNamespace(weight = w3)) is False
    assert dp._weight_has_zero_output_row(types.SimpleNamespace()) is False

    # An unreadable weight falls through to quantize_'s own handling.
    class _Boom:
        @property
        def weight(self):
            raise RuntimeError("meta tensor")

    assert dp._weight_has_zero_output_row(_Boom()) is False


def test_fp8_dynamic_filter_skips_zero_row_linear(monkeypatch):
    # The fp8_dynamic caster must leave a zero-output-row Linear dense while the rest
    # of the encoder still quantises (a family-wide deny would forfeit the whole win).
    _stub_torch(monkeypatch)
    captured: dict = {}
    _stub_transformer_quant(monkeypatch, captured)
    enc = types.SimpleNamespace(_keep_in_fp32_modules = [])

    dp._cast_fp8_dynamic(enc, _target())

    ff = captured["filter_fn"]
    dead = types.SimpleNamespace(weight = _FakeWeight([[0.5, 0.5], [0.0, 0.0]]))
    live = types.SimpleNamespace(weight = _FakeWeight([[0.5, 0.5], [0.5, 0.5]]))
    assert ff(dead, "text_model.encoder.layers.2.self_attn.out_proj") is False
    assert ff(live, "text_model.encoder.layers.2.mlp.fc1") is True


# ── partial in-place cast detection (fails the load, not a silent dense report) ────
class _TorchaoLikeTensor:
    """Detection keys on the tensor class's module path ("torchao" in __module__)."""


_TorchaoLikeTensor.__module__ = "torchao.quantization.linear_activation_quantized_tensor"


class _PartiallyCastEncoder:
    def __init__(self):
        self._swapped = False

    def named_parameters(self):
        if self._swapped:
            yield ("model.layers.0.mlp.up_proj.weight", _TorchaoLikeTensor())
        yield ("model.layers.1.mlp.up_proj.weight", types.SimpleNamespace())


def test_quantize_partial_cast_failure_fails_load(monkeypatch):
    # The caster mutates the encoder in place module-by-module: a mid-pass failure
    # that left torchao params behind must raise (the encoder cannot run as the dense
    # module a best-effort fallback would report), unlike the clean failure above.
    _stub_torch(monkeypatch)
    hooks = types.ModuleType("diffusers.hooks")
    casting = types.ModuleType("diffusers.hooks.layerwise_casting")
    casting.DEFAULT_SKIP_MODULES_PATTERN = ("norm",)

    def _swap_one_then_boom(module, **kwargs):
        module._swapped = True
        raise RuntimeError("encoder cast failed mid-pass")

    hooks.apply_layerwise_casting = _swap_one_then_boom
    monkeypatch.setitem(sys.modules, "diffusers.hooks", hooks)
    monkeypatch.setitem(sys.modules, "diffusers.hooks.layerwise_casting", casting)
    pipe = types.SimpleNamespace(text_encoder = _PartiallyCastEncoder())
    with pytest.raises(RuntimeError, match = "partially quantized"):
        quantize_text_encoders(pipe, _target(), mode = "fp8")
