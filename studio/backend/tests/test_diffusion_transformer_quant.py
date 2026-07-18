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
    # Even with every scheme available, auto picks fp8 on Blackwell: on a B200 fp8 is faster and
    # more accurate than nvfp4 for the DiT's shapes (nvfp4 only wins on very large GEMMs).
    _allow(monkeypatch, {TQ_NVFP4, TQ_MXFP8, TQ_FP8, TQ_INT8})
    assert select_transformer_quant_scheme(_target(), "auto") == TQ_FP8
    # fp8 unavailable: nvfp4 is DISABLED in the auto ladder (explicit opt-in only), so even
    # though the hardware supports nvfp4 here, auto skips it and picks mxfp8 next.
    _allow(monkeypatch, {TQ_NVFP4, TQ_MXFP8, TQ_INT8})
    assert select_transformer_quant_scheme(_target(), "auto") == TQ_MXFP8
    # Only mxfp8 + int8 left -> mxfp8 (still above int8).
    _allow(monkeypatch, {TQ_MXFP8, TQ_INT8})
    assert select_transformer_quant_scheme(_target(), "auto") == TQ_MXFP8
    # Only int8 usable -> int8.
    _allow(monkeypatch, {TQ_INT8})
    assert select_transformer_quant_scheme(_target(), "auto") == TQ_INT8


def test_auto_consumer_blackwell_prefers_int8(monkeypatch):
    # Consumer Blackwell (RTX 50xx): fp8 FP32-accumulate is throughput-halved while int8 is
    # full-rate, so auto prefers int8 even though fp8 is available (the data-center default).
    _stub_torch(monkeypatch, cc = (10, 0), device_name = "NVIDIA GeForce RTX 5090")
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
    _stub_torch(monkeypatch, cc = (8, 9), device_name = "NVIDIA RTX A5000")
    _allow(monkeypatch, {TQ_FP8, TQ_INT8})
    assert select_transformer_quant_scheme(_target(), "auto") == TQ_INT8


def test_auto_professional_rtx_prefers_fp8(monkeypatch):
    # Professional parts (RTX PRO 6000 Blackwell, RTX 6000 Ada) are classified datacenter
    # by the rest of the backend, so auto keeps fp8 first (not int8) -- matching llama_cpp.
    for device_name, cc in (
        ("NVIDIA RTX PRO 6000 Blackwell Server Edition", (10, 0)),
        ("NVIDIA RTX 6000 Ada Generation", (8, 9)),
    ):
        _stub_torch(monkeypatch, cc = cc, device_name = device_name)
        _allow(monkeypatch, {TQ_FP8, TQ_INT8})
        assert select_transformer_quant_scheme(_target(), "auto") == TQ_FP8


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
        "NVIDIA RTX A5000",  # workstation: A5000 token, not professional/datacenter
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
        "NVIDIA B300",  # Blackwell Ultra (matches llama_cpp datacenter regex)
        "NVIDIA GH200 480GB",  # Grace-Hopper superchip (was misread as consumer)
        "NVIDIA H100 80GB HBM3",
        "NVIDIA A100-SXM4-80GB",
        "NVIDIA A40",  # data-center Ampere (distinct token from RTX A4000)
        "NVIDIA L40S",
        "NVIDIA L4",
        "Tesla V100-SXM2-16GB",
        "NVIDIA RTX PRO 6000 Blackwell Server Edition",  # professional -> datacenter-class
        "NVIDIA RTX 6000 Ada Generation",  # professional -> datacenter-class
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


def test_require_bf16_schemes_excludes_nvfp4():
    # fp8 and mxfp8 assert a bf16 weight, so they gate on it; nvfp4 handles fp32 fine, so it is NOT
    # gated (leaving its large fp32 projections quantised, not dense).
    from core.inference.diffusion_transformer_quant import (
        _REQUIRE_BF16_SCHEMES,
        TQ_FP8,
        TQ_MXFP8,
        TQ_NVFP4,
        TQ_INT8,
    )

    assert TQ_FP8 in _REQUIRE_BF16_SCHEMES
    assert TQ_MXFP8 in _REQUIRE_BF16_SCHEMES
    assert TQ_NVFP4 not in _REQUIRE_BF16_SCHEMES
    assert TQ_INT8 not in _REQUIRE_BF16_SCHEMES


def test_make_filter_fn_require_bf16_skips_non_bf16(monkeypatch):
    # fp8 / mxfp8 assert a bf16 weight, so require_bf16 must skip a fp32 Linear (which Wan / Hunyuan
    # keep) while keeping the bf16 ones, else a single fp32 layer raises inside quantize_. int8 and
    # nvfp4 leave it off (they handle fp32).
    torch = types.ModuleType("torch")
    torch.bfloat16, torch.float32 = "bf16", "fp32"

    class _Lin:
        def __init__(self, i, o, dtype):
            self.in_features, self.out_features = i, o
            self.weight = types.SimpleNamespace(dtype = dtype)

    torch.nn = types.SimpleNamespace(Linear = _Lin)
    monkeypatch.setitem(sys.modules, "torch", torch)

    gated = make_filter_fn(512, require_bf16 = True)
    assert gated(_Lin(1024, 4096, torch.bfloat16), "blocks.0.attn.to_q") is True
    assert gated(_Lin(1024, 4096, torch.float32), "blocks.0.attn.to_q") is False  # fp32 -> skip
    assert gated(types.SimpleNamespace(in_features = 1024, out_features = 4096), "no_weight") is False
    # int8 (require_bf16 off, the default) still quantises the fp32 linear.
    assert make_filter_fn(512)(_Lin(1024, 4096, torch.float32), "blocks.0.attn.to_q") is True


def test_make_filter_fn_int8_excludes_modulation_and_embedders(monkeypatch):
    # int8 skips the M=1 AdaLN modulation / conditioning-embedder projections (they crash
    # torch._int_mm's M>16), keeping the attention / FFN layers. fp8 keeps everything.
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
    # Runtime and offline prequant must apply the SAME exclusion, or an artifact quantises a layer
    # runtime skips (int8's M=1 linears -> _int_mm crash; fp8's padded-conditioning embedder ->
    # black frames). int8 gets the family-independent exclusion; scaled_mm excludes nothing
    # WITHOUT a family.
    from core.inference.diffusion_transformer_quant import (
        _INT8_EXCLUDE_NAME_TOKENS,
        exclude_tokens_for_scheme,
    )
    assert exclude_tokens_for_scheme(TQ_INT8) == _INT8_EXCLUDE_NAME_TOKENS
    for scheme in (TQ_FP8, TQ_NVFP4, TQ_MXFP8):
        assert exclude_tokens_for_scheme(scheme) == ()


def test_exclude_tokens_for_scheme():
    # The shared scheme->exclusion decision used by BOTH the runtime path and the offline builder,
    # so an offline checkpoint skips exactly the runtime layers. int8 excludes the M=1 modulation /
    # embedder tokens on every family; scaled_mm excludes nothing by default.
    from core.inference.diffusion_transformer_quant import (
        _INT8_EXCLUDE_NAME_TOKENS,
        exclude_tokens_for_scheme,
    )

    assert exclude_tokens_for_scheme(TQ_INT8) == _INT8_EXCLUDE_NAME_TOKENS
    assert exclude_tokens_for_scheme(TQ_FP8) == ()
    assert exclude_tokens_for_scheme(TQ_NVFP4) == ()
    assert exclude_tokens_for_scheme(TQ_MXFP8) == ()


def test_exclude_tokens_for_scheme_family():
    # Per-family fp8 exclusion: on Wan the per-row fp8 scale divides by a zero padding row in
    # condition_embedder.text_embedder (-> inf -> black frames), so fp8 keeps condition_embedder
    # bf16 while the 30-block stack stays fp8. int8 tolerates zero rows; unknown/Hunyuan get no
    # fp8 exclusion.
    from core.inference.diffusion_transformer_quant import (
        _INT8_EXCLUDE_NAME_TOKENS,
        exclude_tokens_for_scheme,
    )

    assert exclude_tokens_for_scheme(TQ_FP8, "wan2.2-ti2v-5b") == ("condition_embedder",)
    assert exclude_tokens_for_scheme(TQ_FP8, "wan2.2-t2v-a14b") == ("condition_embedder",)
    # I2V-A14B shares the T2V DiT pair, so the fp8 recipe transfers unchanged.
    assert exclude_tokens_for_scheme(TQ_FP8, "wan2.2-i2v-a14b") == ("condition_embedder",)
    assert exclude_tokens_for_scheme(TQ_MXFP8, "wan2.2-ti2v-5b") == ("condition_embedder",)
    # Hunyuan is not localisable (fp8 stays denied), and an unknown family gets nothing.
    assert exclude_tokens_for_scheme(TQ_FP8, "hunyuanvideo-1.5") == ()
    assert exclude_tokens_for_scheme(TQ_FP8, "z-image") == ()
    # int8 tolerates zero ROWS (per-token; no divide), so most families need no extra skip...
    assert exclude_tokens_for_scheme(TQ_INT8, "wan2.2-ti2v-5b") == _INT8_EXCLUDE_NAME_TOKENS
    # ...but the trim shrinks HunyuanVideo-1.5's text streams to their VALID token counts, where
    # int8 fails two ways: M=0 passes through UNPROJECTED (byt5/image embedders -> cond-type add
    # crash), and torch._int_mm needs M > 16 (the ~6-token empty negative prompt). All text-stream
    # linears stay bf16; the M ~ 32k video-stream linears keep int8 coverage.
    from core.inference.diffusion_transformer_quant import _HUNYUAN15_INT8_EXCLUDES

    for fam in ("hunyuanvideo-1.5", "hunyuanvideo-1.5-720p"):
        assert (
            exclude_tokens_for_scheme(TQ_INT8, fam)
            == _INT8_EXCLUDE_NAME_TOKENS + _HUNYUAN15_INT8_EXCLUDES
        )
        assert "context_embedder" in _HUNYUAN15_INT8_EXCLUDES  # covers context_embedder_2 too
    # Qwen-Image never pads its text stream (unlike FLUX's 512-token T5), so a short prompt
    # runs the text-stream linears at M <= 16 and torch._int_mm raises; they stay bf16.
    from core.inference.diffusion_transformer_quant import _QWENIMAGE_INT8_EXCLUDES

    for fam in ("qwen-image", "qwen-image-edit"):
        assert (
            exclude_tokens_for_scheme(TQ_INT8, fam)
            == _INT8_EXCLUDE_NAME_TOKENS + _QWENIMAGE_INT8_EXCLUDES
        )
    for token in ("txt_in", "add_q_proj", "to_add_out", "txt_mlp"):
        assert token in _QWENIMAGE_INT8_EXCLUDES


# ── apply ───────────────────────────────────────────────────────────────────────


def test_resolve_fast_accum(monkeypatch):
    # None auto-detects by GPU class; an explicit bool forces it.
    monkeypatch.setattr(tq, "_is_consumer_gpu", lambda *a: True)
    assert tq._resolve_fast_accum(None) is True
    monkeypatch.setattr(tq, "_is_consumer_gpu", lambda *a: False)
    assert tq._resolve_fast_accum(None) is False
    assert tq._resolve_fast_accum(True) is True  # forced on (e.g. on a data-center card)
    assert tq._resolve_fast_accum(False) is False  # forced off (e.g. on a consumer card)


def test_fp8_config_uses_per_row_granularity():
    """FP8 must use PerRow (per-token activation + per-channel weight) scaling. torchao's
    default is per-TENSOR: on a DiT with extreme activation outliers (z-image's ~6.6e4) one
    outlier forces a tensor-wide scale that pushes normal values below fp8 resolution and the
    denoise collapses to noise. This is the regression guard for that fix (validated on B200:
    per-tensor fp8 = noise, per-row fp8 = matches bf16)."""
    torchao_quant = pytest.importorskip("torchao.quantization")
    per_row = getattr(torchao_quant, "PerRow", None)
    if per_row is None:
        pytest.skip("torchao build without PerRow granularity")
    cfg = tq._make_quant_config(TQ_FP8)
    gran = getattr(cfg, "granularity", None)
    assert gran is not None, "fp8 config must set an explicit granularity, not torchao's default"
    grans = gran if isinstance(gran, (list, tuple)) else [gran]
    assert grans and all(isinstance(g, per_row) for g in grans), f"expected all PerRow, got {gran}"


def test_fp8_config_pins_torch_kernel_preference():
    """FP8 must pin KernelPreference.TORCH. The AUTO default silently switches the weight
    quantize to the MSLK kernel whenever an mslk package is importable, which changes fp8
    scale rounding bitwise (measured 8/8 FLUX matrices differ) and would break the hosted
    prequant bit-identity invariant; the mslk path is also slower under torch.compile."""
    pytest.importorskip("torchao.quantization")
    try:
        from torchao.quantization.quantize_.common.kernel_preference import KernelPreference
    except Exception:
        pytest.skip("torchao build without KernelPreference")
    cfg = tq._make_quant_config(TQ_FP8)
    if not hasattr(cfg, "kernel_preference"):
        pytest.skip("torchao config without kernel_preference")
    assert cfg.kernel_preference == KernelPreference.TORCH


def test_quantize_transformer_applies_and_marks(monkeypatch):
    monkeypatch.setattr(
        tq, "select_transformer_quant_scheme", lambda target, mode, family = None: TQ_FP8
    )
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
    monkeypatch.setattr(
        tq, "select_transformer_quant_scheme", lambda target, mode, family = None: None
    )
    pipe = types.SimpleNamespace(transformer = types.SimpleNamespace())
    assert quantize_transformer(pipe, _target(), mode = "auto") is None


def test_quantize_transformer_tolerates_failure(monkeypatch):
    monkeypatch.setattr(
        tq, "select_transformer_quant_scheme", lambda target, mode, family = None: TQ_INT8
    )
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


# ── family scheme deny (measured model-level breakage) ────────────────────────


def test_family_deny_auto_skips_fp8_for_qwen(monkeypatch):
    # B200, all schemes: auto must NOT pick fp8 / nvfp4 / mxfp8 for the Qwen DiT (per-row fp8 is
    # black there) and falls through to int8, excellent on Qwen.
    _stub_torch(monkeypatch, cc = (10, 0))
    _allow(monkeypatch, {TQ_FP8, TQ_NVFP4, TQ_MXFP8, TQ_INT8})
    assert select_transformer_quant_scheme(_target(), "auto", family = "qwen-image") == TQ_INT8
    assert select_transformer_quant_scheme(_target(), "auto", family = "qwen-image-edit") == TQ_INT8


def test_family_deny_refuses_explicit_fp8_for_qwen(monkeypatch):
    # An explicit fp8 request on qwen-image returns None (same as an unsupported scheme: build
    # GGUF instead). int8 stays honored on qwen; fp8 stays honored outside the deny table.
    _stub_torch(monkeypatch, cc = (10, 0))
    _allow(monkeypatch, {TQ_FP8, TQ_INT8})
    assert select_transformer_quant_scheme(_target(), "fp8", family = "qwen-image") is None
    assert select_transformer_quant_scheme(_target(), "int8", family = "qwen-image") == TQ_INT8
    assert select_transformer_quant_scheme(_target(), "fp8", family = "z-image") == TQ_FP8


def test_family_allows_fp8_for_wan(monkeypatch):
    # The Wan fp8 black frame was a single input embedder dividing by a zero padding row (fixed by
    # the condition_embedder exclude, not a deny), so auto now picks fp8 for both 5B TI2V and A14B.
    # mxfp8 / nvfp4 stay denied, so with only those + int8 auto still lands on int8.
    _stub_torch(monkeypatch, cc = (10, 0))
    _allow(monkeypatch, {TQ_FP8, TQ_NVFP4, TQ_MXFP8, TQ_INT8})
    assert select_transformer_quant_scheme(_target(), "auto", family = "wan2.2-ti2v-5b") == TQ_FP8
    assert select_transformer_quant_scheme(_target(), "auto", family = "wan2.2-t2v-a14b") == TQ_FP8
    assert select_transformer_quant_scheme(_target(), "auto", family = "wan2.2-i2v-a14b") == TQ_FP8
    _allow(monkeypatch, {TQ_NVFP4, TQ_MXFP8, TQ_INT8})  # fp8 unavailable -> denied mx/nvfp4 skipped
    assert select_transformer_quant_scheme(_target(), "auto", family = "wan2.2-ti2v-5b") == TQ_INT8


def test_family_deny_wan_fp8_allowed_mxfp8_nvfp4_refused(monkeypatch):
    # An explicit fp8 request on a Wan family is now honored (safe via the condition_embedder
    # exclude); int8 stays honored; mxfp8 / nvfp4 stay refused (None -> GGUF) until validated.
    _stub_torch(monkeypatch, cc = (10, 0))
    _allow(monkeypatch, {TQ_FP8, TQ_MXFP8, TQ_NVFP4, TQ_INT8})
    assert select_transformer_quant_scheme(_target(), "fp8", family = "wan2.2-ti2v-5b") == TQ_FP8
    assert select_transformer_quant_scheme(_target(), "int8", family = "wan2.2-ti2v-5b") == TQ_INT8
    assert select_transformer_quant_scheme(_target(), "mxfp8", family = "wan2.2-ti2v-5b") is None
    assert select_transformer_quant_scheme(_target(), "nvfp4", family = "wan2.2-ti2v-5b") is None


def test_family_deny_auto_skips_fp8_for_hunyuan(monkeypatch):
    # HunyuanVideo-1.5 DiT renders black on fp8 (LPIPS 0.82); both repacks deny fp8/mxfp8/nvfp4
    # and fall to int8. ltx-2 is NOT denied (its fp8 is clean), so the deny is per family.
    _stub_torch(monkeypatch, cc = (10, 0))
    _allow(monkeypatch, {TQ_FP8, TQ_NVFP4, TQ_MXFP8, TQ_INT8})
    assert select_transformer_quant_scheme(_target(), "auto", family = "hunyuanvideo-1.5") == TQ_INT8
    assert (
        select_transformer_quant_scheme(_target(), "auto", family = "hunyuanvideo-1.5-720p")
        == TQ_INT8
    )
    # ltx-2 keeps the ladder head (fp8) -- it is not a black-frame family.
    assert select_transformer_quant_scheme(_target(), "auto", family = "ltx-2") == TQ_FP8


def test_family_deny_no_family_keeps_ladder(monkeypatch):
    # Without a family (or an unknown one) the ladder is unchanged: fp8 first on B200.
    _stub_torch(monkeypatch, cc = (10, 0))
    _allow(monkeypatch, {TQ_FP8, TQ_INT8})
    assert select_transformer_quant_scheme(_target(), "auto") == TQ_FP8
    assert select_transformer_quant_scheme(_target(), "auto", family = "sdxl") == TQ_FP8


def test_is_int8_memory_fallback(monkeypatch):
    # True only where AUTO lands on int8 as a denied FALLBACK on a data-center fp8-capable GPU
    # (Hunyuan: fp8 denied -> int8), where dense+compile beats int8, so the loader prefers dense.
    # False where int8 is a real accelerator: fp8 families, consumer GPUs, pre-Ada parts.
    from core.inference.diffusion_transformer_quant import is_int8_memory_fallback

    # data-center Blackwell, all schemes available: Hunyuan denies fp8/mx/nvfp4 -> auto int8 -> True.
    _stub_torch(monkeypatch, cc = (10, 0), device_name = "NVIDIA B200")
    _allow(monkeypatch, {TQ_FP8, TQ_NVFP4, TQ_MXFP8, TQ_INT8})
    assert is_int8_memory_fallback(_target(), "hunyuanvideo-1.5") is True
    assert is_int8_memory_fallback(_target(), "hunyuanvideo-1.5-720p") is True
    # Wan / LTX resolve to fp8 (a speed win), not int8 -> False (keep quantising).
    assert is_int8_memory_fallback(_target(), "wan2.2-ti2v-5b") is False
    assert is_int8_memory_fallback(_target(), "wan2.2-t2v-a14b") is False
    assert is_int8_memory_fallback(_target(), "ltx-2") is False
    assert is_int8_memory_fallback(_target(), None) is False

    # Consumer GPU (fp8 accumulate halved -> int8 can be a speed win): never prefer dense.
    _stub_torch(monkeypatch, cc = (10, 0), device_name = "NVIDIA GeForce RTX 5090")
    _allow(monkeypatch, {TQ_FP8, TQ_NVFP4, TQ_MXFP8, TQ_INT8})
    assert is_int8_memory_fallback(_target(), "hunyuanvideo-1.5") is False

    # Ampere data-center (sm_80, no fp8 tensor cores): int8 is the genuine accelerator -> False.
    _stub_torch(monkeypatch, cc = (8, 0), device_name = "NVIDIA A100")
    _allow(monkeypatch, {TQ_INT8})
    assert is_int8_memory_fallback(_target(), "hunyuanvideo-1.5") is False


def test_quantize_transformer_threads_family(monkeypatch):
    # quantize_transformer passes the family down to the selector, so a denied
    # (family, scheme) pair never reaches torchao.
    _stub_torch(monkeypatch, cc = (10, 0))
    _allow(monkeypatch, {TQ_FP8, TQ_INT8})
    pipe = types.SimpleNamespace(transformer = types.SimpleNamespace())
    called = {}
    tqz = types.ModuleType("torchao.quantization")

    def _quantize(
        module,
        config,
        filter_fn = None,
    ):
        called["scheme"] = True

    tqz.quantize_ = _quantize
    tqz.Int8DynamicActivationInt8WeightConfig = lambda: "int8-cfg"
    tqz.Float8DynamicActivationFloat8WeightConfig = lambda **kw: "fp8-cfg"
    tqz.PerRow = lambda: "per-row"
    monkeypatch.setitem(sys.modules, "torchao.quantization", tqz)
    assert quantize_transformer(pipe, _target(), mode = "fp8", family = "qwen-image") is None
    assert called == {}


def test_quantize_transformer_fp8_wan_excludes_condition_embedder(monkeypatch):
    # The Wan fp8 fix is a FILTER exclusion, not a deny: quantize_transformer threads the family
    # into the filter so condition_embedder stays bf16 while the 30-block stack is fp8. Capture the
    # filter that reaches torchao and check it on representative FQNs.
    monkeypatch.setattr(
        tq, "select_transformer_quant_scheme", lambda target, mode, family = None: TQ_FP8
    )
    monkeypatch.setattr(tq, "_make_quant_config", lambda scheme, fast_accum = None: "cfg")

    # torch stub with a real nn.Linear class so the captured filter's isinstance + bf16 gate runs.
    torch = types.ModuleType("torch")
    torch.bfloat16 = "bfloat16"

    class _Linear:
        def __init__(
            self,
            inf,
            outf,
            dtype = "bfloat16",
        ):
            self.in_features, self.out_features = inf, outf
            self.weight = types.SimpleNamespace(dtype = dtype)

    torch.nn = types.SimpleNamespace(Linear = _Linear)
    monkeypatch.setitem(sys.modules, "torch", torch)

    captured: dict = {}
    tqz = types.ModuleType("torchao.quantization")
    tqz.quantize_ = lambda module, config, filter_fn = None: captured.update(fn = filter_fn)
    monkeypatch.setitem(sys.modules, "torchao.quantization", tqz)

    pipe = types.SimpleNamespace(transformer = types.SimpleNamespace())
    assert quantize_transformer(pipe, _target(), mode = "fp8", family = "wan2.2-ti2v-5b") == TQ_FP8
    filt = captured["fn"]
    big = _Linear(4096, 5120)  # a FLOP-heavy bf16 linear (passes min_features + bf16 gate)
    # condition_embedder.* is kept bf16 (the divide-by-zero origin); the block stack is fp8.
    assert filt(big, "condition_embedder.text_embedder.linear_1") is False
    assert filt(big, "condition_embedder.time_embedder.linear_1") is False
    assert filt(big, "blocks.0.attn1.to_q") is True
    assert filt(big, "blocks.0.ffn.net.0.proj") is True
    assert (
        filt(big, "blocks.0.attn2.to_k") is True
    )  # cross-attn K/V stay fp8 (embedder bias rescues rows)


# ── partial in-place quant detection (torchao_quantized_param_fqns) ────────────────
class _TorchaoLikeTensor:
    """Stands in for a torchao tensor subclass: detection keys on the class's module
    path, so the fake just claims a torchao __module__."""


_TorchaoLikeTensor.__module__ = "torchao.dtypes.affine_quantized_tensor"


class _MutableDiT:
    """A fake transformer whose quantize_ pass 'swapped' one weight before failing."""

    def __init__(self):
        self._swapped = False

    def named_parameters(self):
        if self._swapped:
            yield ("blocks.0.attn.to_q.weight", _TorchaoLikeTensor())
        yield ("blocks.1.attn.to_q.weight", types.SimpleNamespace())


def test_torchao_param_scan_detects_swapped_weights():
    dit = _MutableDiT()
    assert tq.torchao_quantized_param_fqns(dit) == []
    dit._swapped = True
    assert tq.torchao_quantized_param_fqns(dit) == ["blocks.0.attn.to_q.weight"]
    # Unscannable object -> no leftovers reported (best-effort, the pre-check path).
    assert tq.torchao_quantized_param_fqns(object()) == []


def test_quantize_transformer_partial_failure_raises(monkeypatch):
    # A mid-pass quantize_ exception can leave earlier layers quantized. That module can't run as
    # dense, so the load must FAIL with a clear error instead of reporting a dense fallback.
    monkeypatch.setattr(
        tq, "select_transformer_quant_scheme", lambda target, mode, family = None: TQ_FP8
    )
    monkeypatch.setattr(tq, "_make_quant_config", lambda scheme, fast_accum = None: "cfg")
    tqz = types.ModuleType("torchao.quantization")

    def _convert_one_then_boom(
        module,
        config,
        filter_fn = None,
    ):
        module._swapped = True  # the in-place swap of the first submodule
        raise RuntimeError("OOM mid-conversion")

    tqz.quantize_ = _convert_one_then_boom
    monkeypatch.setitem(sys.modules, "torchao.quantization", tqz)
    pipe = types.SimpleNamespace(transformer = _MutableDiT())
    with pytest.raises(RuntimeError, match = "partially quantized"):
        quantize_transformer(pipe, _target(), mode = "fp8")


def test_quantize_transformer_clean_failure_still_falls_back_dense(monkeypatch):
    # A failure that swapped NOTHING keeps the best-effort contract: dense fallback.
    monkeypatch.setattr(
        tq, "select_transformer_quant_scheme", lambda target, mode, family = None: TQ_FP8
    )
    monkeypatch.setattr(tq, "_make_quant_config", lambda scheme, fast_accum = None: "cfg")
    tqz = types.ModuleType("torchao.quantization")

    def _boom(
        module,
        config,
        filter_fn = None,
    ):
        raise RuntimeError("failed before any swap")

    tqz.quantize_ = _boom
    monkeypatch.setitem(sys.modules, "torchao.quantization", tqz)
    pipe = types.SimpleNamespace(transformer = _MutableDiT())
    assert quantize_transformer(pipe, _target(), mode = "fp8") is None
