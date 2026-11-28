# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Mergeable QAT-LoRA: fake quantize the merged weight Q(W + s * B @ A) instead of the frozen base alone.

Default QAT with LoRA trains Q(W) x + s * B A x: the adapter runs unquantized beside a fake quantized base,
and the export merges W + s * B A in 16 bits and then quantizes it, so the exported model is not the model
that was trained (a LoRA update smaller than a quantization step can vanish in the export entirely).

With qat_lora_merged = True (or UNSLOTH_QAT_LORA_MERGED=1) every LoRA Linear over a torchao QAT base computes
Q(W + s * B @ A) x, the weight the torchao export writes, so the exported checkpoint is the trained model.
The gradient uses the straight-through estimator through Q: dL/dA and dL/dB are exactly the ordinary LoRA
formulas (s * (G B)^T X and s * G^T (X A^T)), and dL/dX = G Q(W + s B A). So the backward costs what LoRA
costs; the forward adds one merge and one fake quantization of the weight per call.

How: the base layer's weight fake quantizer returns the residual R = Q(W + s B A) - s B A (no grad), so the
unchanged LoRA forward (PEFT's, or Unsloth's fused LoRA kernels, which read the base weight through that same
quantizer) computes R x + s B A x = Q(W + s B A) x up to 16-bit rounding of R. The LoRA branch of an
activation quantized base reads the same fake quantized input as the base. torchao's NVFP4 / MX QAT
linears quantize inside forward(), so their forward runs on W + s B A and subtracts s B A x with the adapter
weights detached. Live memory: one weight-sized residual per layer while autograd holds it (freed by
gradient checkpointing, exactly like base-only QAT without the in-place rounding). UNSLOTH_QAT_LORA_MERGED_CACHE=1
keeps each residual until the adapter changes (one per optimizer step under gradient accumulation), at the
cost of one 16-bit copy of every adapted weight.

The merge is the one save.py's _merge_lora writes (fp32 W^T + s A^T B^T, cast back), shared through
qat_lora_merged_weight, so the exported 16-bit weights are bitwise the weights fake quantized in training.
Disabled adapters, merged adapters and disabled quantizers (QAD teacher) fall back to the base-only behaviour.
Refused: lora_dropout > 0 (the dropped input cannot be merged), DoRA and other LoRA variants, lora_bias,
fake quantized adapters (UNSLOTH_QAT_FAKE_QUANTIZE_LORA=1) and more than one adapter on a layer."""

import os
import types

import torch
import torch.nn.functional as F

__all__ = [
    "qat_lora_merged_requested",
    "qat_lora_merged_weight",
    "install_qat_lora_merged",
    "is_qat_lora_merged",
    "qat_lora_merged_stats",
]

_ENV = "UNSLOTH_QAT_LORA_MERGED"
_CACHE_ENV = "UNSLOTH_QAT_LORA_MERGED_CACHE"
_FLAG = "_unsloth_qat_lora_merged"
_TEACHER = "_unsloth_qat_teacher"
_STATS = {"layers": 0, "microscaling": 0, "calls": 0, "cache_hits": 0}
_TRUE = ("1", "true", "yes", "on")


def qat_lora_merged_requested(value = None):
    """The explicit argument when given, else UNSLOTH_QAT_LORA_MERGED."""
    if value is not None:
        if not isinstance(value, bool):
            raise TypeError(f"Unsloth: qat_lora_merged must be True, False or None, got {value!r}.")
        return value
    return os.environ.get(_ENV, "0").strip().lower() in _TRUE


def qat_lora_merged_stats():
    return dict(_STATS)


def is_qat_lora_merged(module):
    return bool(module.__dict__.get(_FLAG, False))


def _merged_fp32(W, A, B, scale):
    """fp32 (W + s * B @ A) laid out as save.py's _merge_lora computes it: W^T.addmm_(A^T, B^T, alpha = s)."""
    Wt = W.to(torch.float32, copy = True).t()
    Wt.addmm_(A.t().to(torch.float32), B.t().to(torch.float32), alpha = scale)
    return Wt.t()


def qat_lora_merged_weight(
    W,
    A,
    B,
    scale,
    dtype = None,
):
    """The merged 16-bit weight that training fake quantizes and the export writes."""
    with torch.no_grad():
        return _merged_fp32(W, A, B, scale).to(W.dtype if dtype is None else dtype)


def _active_adapter(layer):
    """(A, B, scale) of the single active adapter, or None when the base alone runs."""
    if layer.disable_adapters or layer.merged:
        return None
    for adapter in layer.active_adapters:
        if adapter in layer.lora_A:
            return (
                layer.lora_A[adapter].weight,
                layer.lora_B[adapter].weight,
                layer.scaling[adapter],
            )
    return None


def _cache_enabled():
    return os.environ.get(_CACHE_ENV, "0").strip().lower() in _TRUE


def _cache_key(W, A, B, scale):
    autocast = torch.is_autocast_enabled(W.device.type)
    return (
        W.data_ptr(),
        W._version,
        A.data_ptr(),
        A._version,
        B.data_ptr(),
        B._version,
        float(scale),
        W.dtype,
        autocast,
        torch.get_autocast_dtype(W.device.type) if autocast else None,
    )


class MergedLoRAWeightFakeQuantizer(torch.nn.Module):
    """Stands in for a QAT base layer's weight_fake_quantizer: returns Q(W + s B A) - s B A while the adapter is
    active, so base + LoRA computes Q(W + s B A) x. The wrapped torchao quantizer stays a child (`inner`)."""

    def __init__(self, inner, layer):
        super().__init__()
        self.inner = inner
        # Not a child module: the LoRA layer owns this quantizer, a child link would make a module cycle.
        self.__dict__["_lora_layer"] = layer
        self.__dict__["_cache"] = None

    @property
    def enabled(self):
        return getattr(self.inner, "enabled", True)

    @enabled.setter
    def enabled(self, value):
        self.inner.enabled = value

    @property
    def config(self):
        return getattr(self.inner, "config", None)

    def extra_repr(self):
        return "merged LoRA: Q(W + s * B @ A) - s * B @ A"

    def forward(self, W):
        inner = self.inner
        if not getattr(inner, "enabled", True):
            return W
        adapter = _active_adapter(self.__dict__["_lora_layer"])
        if adapter is None:
            return inner(W)
        A, B, scale = adapter
        caching = (
            not torch.compiler.is_compiling()
            and _cache_enabled()
            and self.training
            and type(W) in (torch.Tensor, torch.nn.Parameter)
        )
        if caching:
            key = _cache_key(W, A, B, scale)
            cached = self.__dict__["_cache"]
            if cached is not None and cached[0] == key:
                _STATS["cache_hits"] += 1
                return cached[1]
        if not torch.compiler.is_compiling():
            _STATS["calls"] += 1
        with torch.no_grad():
            merged = _merged_fp32(W, A, B, scale)
            quantized = inner(merged.to(W.dtype))
            # R = Q(W + s B A) - s B A, in fp32 before the one rounding to the weight dtype.
            residual = (quantized.to(torch.float32) - (merged - W.to(torch.float32))).to(
                quantized.dtype
            )
        if caching:
            self.__dict__["_cache"] = (key, residual)
        elif self.__dict__["_cache"] is not None:
            self.__dict__["_cache"] = None
        return residual

    def clear_cache(self):
        self.__dict__["_cache"] = None


def _activation_quantize_lora_input(module, args):
    """forward_pre_hook on lora_A: the LoRA branch sees the base layer's fake quantized input."""
    quantizer = module.__dict__.get("_unsloth_qat_lora_input_quantizer")
    if quantizer is None or not getattr(quantizer, "enabled", True) or not args:
        return None
    return (quantizer(args[0]), *args[1:])


def _microscaling_weight_forward(base):
    """The NVFP4 / MX QAT forward of `base` on an explicit weight, or None for other classes."""
    from . import _utils

    try:
        from torchao.prototype import qat as prototype_qat
    except Exception:
        return None
    nvfp4 = getattr(prototype_qat, "NVFP4FakeQuantizedLinear", None)
    mx = getattr(prototype_qat, "MXFakeQuantizedLinear", None)
    if isinstance(nvfp4, type) and isinstance(base, nvfp4):
        return _utils._nvfp4_any_rank_forward
    if isinstance(mx, type) and isinstance(base, mx):
        return _utils._mx_autocast_forward
    return None


def _merged_microscaling_forward(self, x):
    """NVFP4 / MX QAT base inside a merged LoRA layer: Q(W + s B A) x - s B A x, the adapter detached here
    (the LoRA layer adds s B A x back with gradients), so the sum is Q(W + s B A) x and dX = G Q(W + s B A)."""
    weight_forward = self.__dict__["_unsloth_qat_lora_weight_forward"]
    layer = self.__dict__["_unsloth_qat_lora_layer"]
    adapter = None if self.__dict__.get(_TEACHER, False) else _active_adapter(layer)
    if adapter is None:
        return weight_forward(self, x)
    A, B, scale = adapter
    if not torch.compiler.is_compiling():
        _STATS["calls"] += 1
    with torch.no_grad():
        merged = qat_lora_merged_weight(self.weight, A, B, scale)
    out = weight_forward(self, x, weight = merged)
    lora_x = x.to(A.dtype)
    correction = F.linear(F.linear(lora_x, A.detach()), B.detach()) * scale
    return out - correction.to(out.dtype)


def _refusal(layer, name):
    if len(layer.lora_A) != 1:
        return f"{name} has {len(layer.lora_A)} adapters; merged QAT-LoRA trains one"
    adapter = next(iter(layer.lora_A))
    dropout = layer.lora_dropout[adapter] if adapter in layer.lora_dropout else None
    if dropout is not None and not isinstance(dropout, torch.nn.Identity):
        if float(getattr(dropout, "p", 0.0)) > 0.0:
            return f"{name} uses lora_dropout = {dropout.p}; merged QAT-LoRA needs lora_dropout = 0"
    if adapter in (getattr(layer, "lora_variant", None) or {}) or (
        getattr(layer, "use_dora", None) or {}
    ).get(adapter, False):
        return f"{name} uses DoRA or another LoRA variant; merged QAT-LoRA supports plain LoRA"
    for factor in (layer.lora_A[adapter], layer.lora_B[adapter]):
        if getattr(factor, "bias", None) is not None:
            return f"{name} uses lora_bias; merged QAT-LoRA supports adapters without a bias"
        if factor._modules.get("weight_fake_quantizer") is not None:
            return f"{name} fake quantizes its adapter weights (UNSLOTH_QAT_FAKE_QUANTIZE_LORA=1)"
    return None


def install_qat_lora_merged(model):
    """Switch every LoRA Linear over a torchao QAT base to Q(W + s B A). Idempotent; returns the layer count."""
    try:
        from peft.tuners.lora import Linear as LoraLinear
    except Exception:
        return 0
    layers = []
    for name, layer in model.named_modules():
        if not isinstance(layer, LoraLinear):
            continue
        base = layer.get_base_layer()
        quantizer = base._modules.get("weight_fake_quantizer")
        weight_forward = None if quantizer is not None else _microscaling_weight_forward(base)
        if quantizer is None and weight_forward is None:
            continue
        reason = _refusal(layer, name)
        if reason is not None:
            raise ValueError(
                "Unsloth: qat_lora_merged = True cannot train this model: " + reason + "."
            )
        layers.append((layer, base, quantizer, weight_forward))
    count = microscaling = 0
    for layer, base, quantizer, weight_forward in layers:
        if quantizer is not None:
            if not isinstance(quantizer, MergedLoRAWeightFakeQuantizer):
                base.weight_fake_quantizer = MergedLoRAWeightFakeQuantizer(quantizer, layer)
            activation = base._modules.get("activation_fake_quantizer")
            if activation is not None:
                for lora_A in layer.lora_A.values():
                    if "_unsloth_qat_lora_input_quantizer" not in lora_A.__dict__:
                        lora_A.__dict__["_unsloth_qat_lora_input_quantizer"] = activation
                        lora_A.register_forward_pre_hook(_activation_quantize_lora_input)
        else:
            if "_unsloth_qat_lora_weight_forward" not in base.__dict__:
                base.__dict__["_unsloth_qat_lora_weight_forward"] = weight_forward
                base.__dict__["_unsloth_qat_lora_layer"] = layer
                base.forward = types.MethodType(_merged_microscaling_forward, base)
            microscaling += 1
        layer.__dict__[_FLAG] = True
        count += 1
    _STATS["layers"] = count
    _STATS["microscaling"] = microscaling
    if count:
        try:
            from ..precision import record
            record(model, "qat_lora_merged", count)
        except Exception:
            pass
    return count
