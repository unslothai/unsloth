# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit tests for VAE quantisation (``diffusion_vae_quant.py``).

Hermetic: torch + the diffusers / torchao casters are stubbed via ``sys.modules`` so
gating, the conv filter, and the apply path run without a GPU, real diffusers, or real
torchao. Mirrors tests/test_diffusion_precision.py's stubbing style.
"""

from __future__ import annotations

import sys
import types

import pytest

import core.inference.diffusion_vae_quant as vq
from core.inference.diffusion_vae_quant import (
    VAE_QUANT_AUTO,
    VAE_QUANT_FP8,
    VAE_QUANT_FP8_DYNAMIC,
    _cast_vae_fp8,
    _cast_vae_fp8_dynamic,
    normalize_vae_quant,
    quantize_vae,
    select_vae_quant_scheme,
    vae_quant_supported,
)


def _target(
    *,
    device = "cuda",
    dtype = "bfloat16",
    cc = (10, 0),
):
    return types.SimpleNamespace(device = device, dtype = dtype, _cc = cc)


class _Weight:
    """A stand-in for a conv / linear weight tensor: exposes ``.shape`` and ``.dim()``."""

    def __init__(self, shape):
        self.shape = shape

    def dim(self):
        return len(self.shape)


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
    # The conv filter isinstance-checks nn.Linear / nn.Conv2d / nn.Conv3d, and the layerwise
    # caster reads torch.float8_e4m3fn, so the stub torch must expose both.
    torch.nn = types.SimpleNamespace(
        Linear = type("Linear", (), {}),
        Conv2d = type("Conv2d", (), {}),
        Conv3d = type("Conv3d", (), {}),
    )
    torch.cuda = types.SimpleNamespace(
        get_device_capability = lambda *a: cc,
        synchronize = lambda *a, **k: None,
        is_available = lambda: True,
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    return torch


def _stub_torchao(monkeypatch, captured):
    # torchao's fp8_dynamic conv config + quantize_. Records the (config, filter_fn) so the
    # PerTensor granularity and the conv filter closure can be asserted.
    tq = types.ModuleType("torchao.quantization")
    tq.quantize_ = lambda module, config, filter_fn = None: captured.update(
        module = module, config = config, filter_fn = filter_fn
    )
    tq.Float8DynamicActivationFloat8WeightConfig = lambda granularity = None: ("fp8dyn", granularity)
    tq.PerTensor = lambda: "pertensor"
    monkeypatch.setitem(sys.modules, "torchao.quantization", tq)
    return tq


def _stub_diffusers(monkeypatch, recorder):
    hooks = types.ModuleType("diffusers.hooks")
    casting = types.ModuleType("diffusers.hooks.layerwise_casting")
    casting.DEFAULT_SKIP_MODULES_PATTERN = ("norm",)
    hooks.apply_layerwise_casting = lambda module, **kw: recorder.append(("fp8", module, kw))
    monkeypatch.setitem(sys.modules, "diffusers.hooks", hooks)
    monkeypatch.setitem(sys.modules, "diffusers.hooks.layerwise_casting", casting)


def _stub_capability(monkeypatch, cc):
    """Stub the transformer module's ``_capability`` that select_vae_quant_scheme imports."""
    dtq = types.ModuleType("core.inference.diffusion_transformer_quant")
    dtq._capability = lambda: cc
    monkeypatch.setitem(sys.modules, "core.inference.diffusion_transformer_quant", dtq)
    return dtq


def _allow_vae(monkeypatch, allowed):
    """Force vae_quant_supported to accept only ``allowed`` (simulates the hardware gate)."""
    monkeypatch.setattr(vq, "vae_quant_supported", lambda target, mode: mode in allowed)


# ── normalisation ─────────────────────────────────────────────────────────────


def test_normalize_vae_quant():
    assert normalize_vae_quant(None) is None
    assert normalize_vae_quant("") is None
    assert normalize_vae_quant("none") is None
    # "off" disables (like the TE / transformer normalisers) -> dense.
    assert normalize_vae_quant("off") is None
    # "auto" passes through for select_vae_quant_scheme to resolve.
    assert normalize_vae_quant("AUTO") == VAE_QUANT_AUTO
    assert normalize_vae_quant("FP8") == VAE_QUANT_FP8
    # Hyphens fold to underscores so "fp8-dynamic" is accepted.
    assert normalize_vae_quant("FP8-Dynamic") == VAE_QUANT_FP8_DYNAMIC
    # int8 / nvfp4 have no VAE scheme -> rejected.
    with pytest.raises(ValueError):
        normalize_vae_quant("int8")
    with pytest.raises(ValueError):
        normalize_vae_quant("nvfp4")


# ── gating ────────────────────────────────────────────────────────────────────


def test_vae_quant_supported_fp8_requires_cuda_bf16_and_fp8(monkeypatch):
    _stub_torch(monkeypatch, with_fp8 = True, cc = (8, 9))
    assert vae_quant_supported(_target(), VAE_QUANT_FP8) is True
    assert vae_quant_supported(_target(device = "cpu"), VAE_QUANT_FP8) is False
    assert vae_quant_supported(_target(dtype = "float16"), VAE_QUANT_FP8) is False
    # No fp8 dtype at all -> unsupported.
    _stub_torch(monkeypatch, with_fp8 = False, cc = (8, 9))
    assert vae_quant_supported(_target(), VAE_QUANT_FP8) is False


def test_vae_quant_supported_fp8_dynamic_requires_sm89(monkeypatch):
    # Compute fp8 conv (torch._scaled_mm) needs fp8-GEMM silicon: Ada sm_89+ / Hopper / Blackwell.
    _stub_torch(monkeypatch, cc = (8, 9))
    assert vae_quant_supported(_target(), VAE_QUANT_FP8_DYNAMIC) is True
    _stub_torch(monkeypatch, cc = (9, 0))
    assert vae_quant_supported(_target(), VAE_QUANT_FP8_DYNAMIC) is True
    # Ampere (8.0) has no fp8 GEMM.
    _stub_torch(monkeypatch, cc = (8, 0))
    assert vae_quant_supported(_target(), VAE_QUANT_FP8_DYNAMIC) is False


# ── auto ladder (select_vae_quant_scheme) ───────────────────────────────────────


def test_select_datacenter_uses_layerwise_fp8(monkeypatch):
    # ``auto`` engages layerwise fp8 ONLY -- the accuracy sweep keeps fp8_dynamic out of the
    # auto ladder (only in-bar on a couple of VAEs). Even on data-center fp8-GEMM silicon it
    # resolves to fp8, and the fp8_dynamic conv probe is never consulted for auto.
    _stub_capability(monkeypatch, (10, 0))
    _allow_vae(monkeypatch, {VAE_QUANT_FP8_DYNAMIC, VAE_QUANT_FP8})
    monkeypatch.setattr(
        vq, "_vae_fp8_dynamic_probe", lambda device: pytest.fail("auto must not probe fp8_dynamic")
    )
    assert select_vae_quant_scheme(_target(), "auto", family = "flux.1") == VAE_QUANT_FP8


def test_select_offload_uses_layerwise_fp8(monkeypatch):
    # Under offload ``auto`` still resolves to layerwise fp8 (the storage-only scheme that
    # survives Module.to()); the fp8_dynamic conv probe is never consulted.
    _stub_capability(monkeypatch, (10, 0))
    _allow_vae(monkeypatch, {VAE_QUANT_FP8_DYNAMIC, VAE_QUANT_FP8})
    monkeypatch.setattr(
        vq, "_vae_fp8_dynamic_probe", lambda device: pytest.fail("probe must not run under offload")
    )
    assert select_vae_quant_scheme(_target(), "auto", offload_active = True) == VAE_QUANT_FP8


def test_select_force_fp32_stays_dense(monkeypatch):
    # A force-fp32 (Wan) family never quantises, for auto or an explicit request.
    _stub_capability(monkeypatch, (10, 0))
    _allow_vae(monkeypatch, {VAE_QUANT_FP8_DYNAMIC, VAE_QUANT_FP8})
    assert select_vae_quant_scheme(_target(), "auto", force_fp32 = True) is None
    assert select_vae_quant_scheme(_target(), "fp8", force_fp32 = True) is None


def test_select_family_deny_skips_scheme(monkeypatch):
    # The only scheme ``auto`` walks is layerwise fp8, so denying fp8 for a family leaves it
    # dense (None). (This is the SDXL case in the real deny list: fp8 marginal -> stay dense.)
    _stub_capability(monkeypatch, (10, 0))
    _allow_vae(monkeypatch, {VAE_QUANT_FP8_DYNAMIC, VAE_QUANT_FP8})
    monkeypatch.setattr(vq, "_vae_fp8_dynamic_probe", lambda device: True)
    monkeypatch.setattr(vq, "_VAE_FAMILY_SCHEME_DENY", {"badfam": frozenset({VAE_QUANT_FP8})})
    assert select_vae_quant_scheme(_target(), "auto", family = "badfam") is None


def test_select_no_capability_is_none(monkeypatch):
    _stub_capability(monkeypatch, None)
    _allow_vae(monkeypatch, {VAE_QUANT_FP8_DYNAMIC, VAE_QUANT_FP8})
    assert select_vae_quant_scheme(_target(), "auto") is None


def test_select_support_gate_uses_fp8(monkeypatch):
    # fp8-capable silicon with fp8 supported -> ``auto`` resolves to layerwise fp8.
    _stub_capability(monkeypatch, (8, 9))
    _allow_vae(monkeypatch, {VAE_QUANT_FP8})
    assert select_vae_quant_scheme(_target(), "auto") == VAE_QUANT_FP8


def test_select_auto_never_uses_fp8_dynamic(monkeypatch):
    # Even with fp8_dynamic hardware-supported and its conv probe passing, ``auto`` stays on
    # layerwise fp8: fp8_dynamic is deliberately kept out of the auto ladder (explicit opt-in).
    _stub_capability(monkeypatch, (10, 0))
    _allow_vae(monkeypatch, {VAE_QUANT_FP8_DYNAMIC, VAE_QUANT_FP8})
    monkeypatch.setattr(vq, "_vae_fp8_dynamic_probe", lambda device: True)
    assert select_vae_quant_scheme(_target(), "auto") == VAE_QUANT_FP8


def test_select_explicit_passthrough(monkeypatch):
    # An explicit request is returned as-is (quantize_vae re-gates it); no ladder walk,
    # so no transformer-module stub is needed.
    assert select_vae_quant_scheme(_target(), "fp8") == VAE_QUANT_FP8
    assert select_vae_quant_scheme(_target(), "fp8_dynamic") == VAE_QUANT_FP8_DYNAMIC
    assert select_vae_quant_scheme(_target(), None) is None
    assert select_vae_quant_scheme(_target(), "none") is None


def test_select_explicit_family_deny_returns_none(monkeypatch):
    monkeypatch.setattr(vq, "_VAE_FAMILY_SCHEME_DENY", {"badfam": frozenset({VAE_QUANT_FP8})})
    assert select_vae_quant_scheme(_target(), "fp8", family = "badfam") is None


# ── conv-aware filter (_cast_vae_fp8_dynamic) ───────────────────────────────────


def test_fp8_dynamic_conv_filter(monkeypatch):
    # The real filter closure the PerTensor fp8 conv config receives: %16-channel Conv2d/Conv3d
    # and Linear quantise; off-%16 channels and the conv_out / norm_out head stay dense.
    torch = _stub_torch(monkeypatch)
    captured: dict = {}
    _stub_torchao(monkeypatch, captured)
    _cast_vae_fp8_dynamic(object(), _target())
    # PerTensor granularity (NOT the DiT's per-row) is what the config was built with.
    assert captured["config"] == ("fp8dyn", "pertensor")
    ff = captured["filter_fn"]
    nn = torch.nn

    def _mod(cls, shape, kernel_size = None):
        m = cls()
        m.weight = _Weight(shape)
        if kernel_size is not None:
            m.kernel_size = kernel_size
        return m

    # Conv2d (4D) / Conv3d (5D) with both channel dims a multiple of 16 quantise.
    assert ff(_mod(nn.Conv2d, (128, 128, 3, 3), (3, 3)), "decoder.up.0.resnets.0.conv1") is True
    assert ff(_mod(nn.Conv3d, (64, 64, 3, 3, 3), (3, 3, 3)), "decoder.mid_block.conv3d") is True
    # nn.Linear (mid-block attention projections) also quantise (no kernel_size attr).
    assert ff(_mod(nn.Linear, (512, 512)), "decoder.mid_block.attentions.0.to_q") is True
    # POINTWISE (1x1 / 1x1x1) convs excluded even at %16 channels: torchao 0.17's fp8 conv
    # kernel rejects them ("Activation and filter channels must match") -> crash at decode.
    assert ff(_mod(nn.Conv2d, (128, 128, 1, 1), (1, 1)), "decoder.mid_block.attentions.0.proj_conv") is False
    assert ff(_mod(nn.Conv3d, (64, 64, 1, 1, 1), (1, 1, 1)), "decoder.time_mix.conv") is False
    # Channels not a multiple of 16 excluded (torchao would skip them regardless): the RGB
    # in/out head (C=3) and any off-16 dim.
    assert ff(_mod(nn.Conv2d, (128, 3, 3, 3), (3, 3)), "encoder.conv_in") is False
    assert ff(_mod(nn.Conv2d, (24, 128, 3, 3), (3, 3)), "decoder.up.1.upsamplers.0.conv") is False
    # conv_out / proj_out / norm_out excluded by NAME even with %16 channels.
    assert ff(_mod(nn.Conv2d, (16, 128, 3, 3)), "decoder.conv_out") is False
    assert ff(_mod(nn.Conv2d, (128, 128, 3, 3)), "decoder.conv_norm_out") is False
    assert ff(_mod(nn.Conv2d, (128, 128, 3, 3)), "decoder.norm_out.conv") is False
    assert ff(_mod(nn.Linear, (512, 512)), "decoder.proj_out") is False

    # A non-conv/linear module (e.g. a GroupNorm) is excluded outright.
    class _GroupNorm:
        pass

    gn = _GroupNorm()
    gn.weight = _Weight((128,))
    assert ff(gn, "decoder.mid_block.resnets.0.norm1") is False
    # A weight with dim() < 2 (a 1D param) is excluded.
    assert ff(_mod(nn.Conv2d, (128,)), "decoder.some.bias_only") is False


def test_cast_vae_fp8_layerwise_skips_head_and_norms(monkeypatch):
    # The layerwise storage cast passes the decoder head + norm tokens through to diffusers'
    # skip_modules_pattern (on top of the diffusers default), so they stay dense.
    _stub_torch(monkeypatch)
    recorder: list = []
    _stub_diffusers(monkeypatch, recorder)
    vae = object()
    _cast_vae_fp8(vae, _target())
    assert len(recorder) == 1
    _, mod, kw = recorder[0]
    assert mod is vae
    assert kw["storage_dtype"] == "float8_e4m3fn"
    assert kw["compute_dtype"] == "bfloat16"
    skip = kw["skip_modules_pattern"]
    # The diffusers default is preserved and the keep-dense tokens are appended.
    assert "norm" in skip
    for tok in ("conv_out", "proj_out", "conv_norm_out", "norm_out"):
        assert tok in skip


# ── apply (quantize_vae) ────────────────────────────────────────────────────────


def test_quantize_vae_disabled_returns_none(monkeypatch):
    pipe = types.SimpleNamespace(vae = object())
    assert quantize_vae(pipe, _target(), mode = None) is None
    assert quantize_vae(pipe, _target(), mode = "none") is None


def test_quantize_vae_force_fp32_stays_dense(monkeypatch):
    # A force-fp32 (Wan) family never casts, for an explicit scheme or auto.
    _stub_torch(monkeypatch, cc = (10, 0))
    monkeypatch.setattr(vq, "_cast_vae_fp8_dynamic", lambda v, t: pytest.fail("must not cast"))
    monkeypatch.setattr(vq, "_cast_vae_fp8", lambda v, t: pytest.fail("must not cast"))
    pipe = types.SimpleNamespace(vae = object())
    assert quantize_vae(pipe, _target(), mode = "fp8", force_fp32 = True) is None
    assert quantize_vae(pipe, _target(), mode = "auto", force_fp32 = True) is None


def test_quantize_vae_offload_skips_fp8_dynamic(monkeypatch):
    # Explicit fp8_dynamic under offload is skipped (torchao tensors reject Module.to());
    # layerwise fp8 still engages.
    _stub_torch(monkeypatch, cc = (10, 0))
    _allow_vae(monkeypatch, {VAE_QUANT_FP8_DYNAMIC, VAE_QUANT_FP8})
    monkeypatch.setattr(
        vq, "_cast_vae_fp8_dynamic", lambda v, t: pytest.fail("torchao must not run under offload")
    )
    pipe = types.SimpleNamespace(vae = object())
    assert quantize_vae(pipe, _target(), mode = "fp8_dynamic", offload_active = True) is None
    fp8_calls: list = []
    monkeypatch.setattr(vq, "_cast_vae_fp8", lambda v, t: fp8_calls.append(v))
    assert quantize_vae(pipe, _target(), mode = "fp8", offload_active = True) == VAE_QUANT_FP8
    assert len(fp8_calls) == 1


def test_quantize_vae_unsupported_hw_is_noop(monkeypatch):
    # An explicit scheme on hardware that does not support it applies nothing.
    _stub_torch(monkeypatch, cc = (8, 0))
    _allow_vae(monkeypatch, set())
    monkeypatch.setattr(vq, "_cast_vae_fp8", lambda v, t: pytest.fail("must not cast"))
    pipe = types.SimpleNamespace(vae = object())
    assert quantize_vae(pipe, _target(), mode = "fp8") is None


def test_quantize_vae_none_vae_is_noop(monkeypatch):
    # A pipeline with no VAE attribute is a best-effort no-op even when the mode is supported.
    _stub_torch(monkeypatch, cc = (10, 0))
    _allow_vae(monkeypatch, {VAE_QUANT_FP8})
    pipe = types.SimpleNamespace()  # no .vae
    assert quantize_vae(pipe, _target(), mode = "fp8") is None


def test_quantize_vae_explicit_fp8_applies(monkeypatch):
    _stub_torch(monkeypatch, cc = (10, 0))
    _allow_vae(monkeypatch, {VAE_QUANT_FP8})
    calls: list = []
    monkeypatch.setattr(vq, "_cast_vae_fp8", lambda v, t: calls.append(v))
    vae = object()
    pipe = types.SimpleNamespace(vae = vae)
    assert quantize_vae(pipe, _target(), mode = "fp8") == VAE_QUANT_FP8
    assert calls == [vae]


def test_quantize_vae_tolerates_caster_failure(monkeypatch):
    # The caster raising leaves the VAE dense (best-effort) -> None.
    _stub_torch(monkeypatch, cc = (10, 0))
    _allow_vae(monkeypatch, {VAE_QUANT_FP8})

    def _boom(v, t):
        raise RuntimeError("fp8 unsupported for this layer")

    monkeypatch.setattr(vq, "_cast_vae_fp8", _boom)
    pipe = types.SimpleNamespace(vae = object())
    assert quantize_vae(pipe, _target(), mode = "fp8") is None


def test_quantize_vae_auto_resolves_and_applies(monkeypatch):
    # End-to-end: mode="auto" resolves via the ladder (layerwise fp8) then applies that caster.
    _stub_torch(monkeypatch, cc = (10, 0))
    _stub_capability(monkeypatch, (10, 0))
    _allow_vae(monkeypatch, {VAE_QUANT_FP8_DYNAMIC, VAE_QUANT_FP8})
    monkeypatch.setattr(vq, "_cast_vae_fp8_dynamic", lambda v, t: pytest.fail("auto must not use fp8_dynamic"))
    calls: list = []
    monkeypatch.setattr(vq, "_cast_vae_fp8", lambda v, t: calls.append(v))
    vae = object()
    pipe = types.SimpleNamespace(vae = vae)
    assert quantize_vae(pipe, _target(), mode = "auto", family = "flux.1") == VAE_QUANT_FP8
    assert calls == [vae]


def test_quantize_vae_explicit_fp8_dynamic_probe_gates(monkeypatch):
    # An explicit fp8_dynamic request runs the conv smoke probe: a build whose torchao lacks a
    # working fp8 conv path (probe False) stays dense; a passing probe applies the caster.
    _stub_torch(monkeypatch, cc = (10, 0))
    _allow_vae(monkeypatch, {VAE_QUANT_FP8_DYNAMIC, VAE_QUANT_FP8})
    monkeypatch.setattr(vq, "_vae_fp8_dynamic_probe", lambda device: False)
    monkeypatch.setattr(vq, "_cast_vae_fp8_dynamic", lambda v, t: pytest.fail("probe failed: must not cast"))
    pipe = types.SimpleNamespace(vae = object())
    assert quantize_vae(pipe, _target(), mode = "fp8_dynamic", family = "flux.2-klein") is None
    monkeypatch.setattr(vq, "_vae_fp8_dynamic_probe", lambda device: True)
    calls: list = []
    monkeypatch.setattr(vq, "_cast_vae_fp8_dynamic", lambda v, t: calls.append(v))
    assert quantize_vae(pipe, _target(), mode = "fp8_dynamic", family = "flux.2-klein") == VAE_QUANT_FP8_DYNAMIC
    assert len(calls) == 1


def test_real_family_deny_list_policy(monkeypatch):
    # The shipped _VAE_FAMILY_SCHEME_DENY (from the B200 sweep), exercised through the real
    # select path -- no deny-list monkeypatch. Confirms the per-family auto/explicit outcomes.
    _stub_capability(monkeypatch, (10, 0))
    _allow_vae(monkeypatch, {VAE_QUANT_FP8_DYNAMIC, VAE_QUANT_FP8})
    # SDXL denies BOTH schemes -> auto stays dense (its layerwise fp8 was marginal).
    assert select_vae_quant_scheme(_target(), "auto", family = "sdxl") is None
    assert select_vae_quant_scheme(_target(), "fp8", family = "sdxl") is None
    # Qwen-Image: auto -> layerwise fp8 (safe); explicit fp8_dynamic refused (catastrophic).
    assert select_vae_quant_scheme(_target(), "auto", family = "qwen-image") == VAE_QUANT_FP8
    assert select_vae_quant_scheme(_target(), "fp8_dynamic", family = "qwen-image") is None
    # FLUX.1 / LTX-2 keep layerwise fp8 on auto but deny explicit fp8_dynamic.
    assert select_vae_quant_scheme(_target(), "auto", family = "ltx-2") == VAE_QUANT_FP8
    assert select_vae_quant_scheme(_target(), "fp8_dynamic", family = "flux.1") is None
    # FLUX.2 / Hunyuan keep fp8_dynamic available as an explicit opt-in (measured in-bar).
    assert select_vae_quant_scheme(_target(), "fp8_dynamic", family = "flux.2-klein") == VAE_QUANT_FP8_DYNAMIC
    assert select_vae_quant_scheme(_target(), "fp8_dynamic", family = "hunyuanvideo-1.5") == VAE_QUANT_FP8_DYNAMIC
