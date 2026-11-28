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

How: for weight-only recipes the base layer's weight fake quantizer returns the residual R = Q(W + s B A) - s B A
(no grad), so the unchanged LoRA forward (PEFT's, or Unsloth's fused LoRA kernels, which read the base weight
through that same quantizer) computes R x + s B A x = Q(W + s B A) x up to 16-bit rounding of R. Activation
quantized bases (int8 / fp8 activations) never take the fused kernels, and torchao's NVFP4 / MX linears quantize
inside forward(): there the LoRA layer runs the base forward on Q(W + s B A), bit for bit the exported layer (the
activation quantizer would amplify any rounding difference), and adds its adapter branch with value exactly 0 on
a detached input to carry the gradients. Live memory: one weight-sized tensor per layer while autograd holds it
(freed by gradient checkpointing, exactly like base-only QAT without the in-place rounding). The QAT weight memo
(models/_qat_weight_memo.py) caches it, keyed on the adapter's storage and version, within its memory budget, so
gradient accumulation micro batches and checkpoint replays reuse it until the optimizer step.

The merge is the one save.py's _merge_lora writes (fp32 W^T + s A^T B^T, cast back), shared through
qat_lora_merged_weight, so the exported 16-bit weights are bitwise the weights fake quantized in training.
Disabled adapters, merged adapters and disabled quantizers (QAD teacher) fall back to the base-only behaviour.
Refused: lora_dropout > 0 (the dropped input cannot be merged), DoRA and other LoRA variants, lora_bias,
fake quantized adapters (UNSLOTH_QAT_FAKE_QUANTIZE_LORA=1) and more than one adapter on a layer."""

import contextlib
import copy
import functools
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
    "nearest_merge_rounding",
]

_ENV = "UNSLOTH_QAT_LORA_MERGED"
_FLAG = "_unsloth_qat_lora_merged"
_TEACHER = "_unsloth_qat_teacher"
_MERGE_ROUNDING_ENV = "UNSLOTH_MERGE_ROUNDING"
_COMPILE_ENV = "UNSLOTH_QAT_LORA_MERGED_COMPILE"
_STATS = {"layers": 0, "microscaling": 0, "calls": 0}
_TRUE = ("1", "true", "yes", "on")


def qat_lora_merged_requested(value = None):
    """The explicit argument when given, else UNSLOTH_QAT_LORA_MERGED."""
    if value is not None:
        if not isinstance(value, bool):
            raise TypeError(f"Unsloth: qat_lora_merged must be True, False or None, got {value!r}.")
        return value
    return os.environ.get(_ENV, "0").strip().lower() in _TRUE


@contextlib.contextmanager
def nearest_merge_rounding(model):
    """Round the 16-bit LoRA merge to nearest while exporting a merged QAT-LoRA model.

    unsloth_zoo's merge may round stochastically (UNSLOTH_MERGE_ROUNDING), which keeps sub-ulp LoRA updates in a
    16-bit export on average but moves about a quarter of the merged weights by one ulp, and so across
    quantization steps: the quantized export would no longer be the trained model. Other models: unchanged."""
    if not any(is_qat_lora_merged(m) for m in model.modules()):
        yield
        return
    previous = os.environ.get(_MERGE_ROUNDING_ENV)
    os.environ[_MERGE_ROUNDING_ENV] = "nearest"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(_MERGE_ROUNDING_ENV, None)
        else:
            os.environ[_MERGE_ROUNDING_ENV] = previous


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


# Opaque custom ops, so a torch.compile region runs these eagerly, bit for bit what the export computes:
# inductor otherwise fuses the 16-bit cast of the merged weight into the quantizer and skips its rounding
# (torch._inductor.config.emulate_precision_casts is off), which moves weights across quantization steps.
# Compile caches key on the op name: bump the suffix on schema changes.
_QUANTIZERS = {}
_QUANTIZER_KEYS = {}


def _quantizer_key(quantizer):
    """One registry entry per (class, config). The entry is a private copy, never a model's own module: a
    model's quantizer carries runtime state (`enabled`, a QAD teacher's pass-through) that must not reach another
    layer or model through the shared entry. Each wrapper checks its own quantizer's state before the op."""
    cls = type(quantizer)
    signature = (cls.__module__, cls.__qualname__, repr(getattr(quantizer, "config", None)))
    key = _QUANTIZER_KEYS.get(signature)
    if key is None:
        canonical = copy.deepcopy(quantizer)
        canonical.__dict__.pop("forward", None)
        if "enabled" in vars(canonical):
            canonical.enabled = True
        canonical.requires_grad_(False)
        _drop_qparams(canonical)
        key = _QUANTIZER_KEYS[signature] = len(_QUANTIZERS)
        _QUANTIZERS[key] = canonical
    return key


def _residual_tail(merged, W, quantizer):
    quantized = quantizer(merged.to(W.dtype))
    # R = Q(W + s B A) - s B A, in fp32 before the one rounding to the weight dtype.
    return (quantized.to(torch.float32) - (merged - W.to(torch.float32))).to(quantized.dtype)


_COMPILED_TAILS = {}


def _compiled_tail(key, quantizer):
    """UNSLOTH_QAT_LORA_MERGED_COMPILE=1: the quantize + residual step as one inductor kernel per recipe, about
    3x less memory traffic. Not bitwise: Triton's fp32 division is not correctly rounded, so about 1e-4 of the
    weights land one quantization step away from the export. Off by default."""
    if os.environ.get(_COMPILE_ENV, "0").strip().lower() not in _TRUE:
        return None
    fn = _COMPILED_TAILS.get(key)
    if fn is None:
        fn = _COMPILED_TAILS[key] = torch.compile(
            functools.partial(_residual_tail, quantizer = quantizer),
            dynamic = True,
            options = {"emulate_precision_casts": True},
        )
    return fn or None


def _drop_qparams(quantizer):
    """torchao's Intx quantizer keeps its last scale / zero point as attributes. These are dynamic recipes, so drop
    them: the shared registry entry must not hold one layer's tensors (nor a CUDA graph pool's) after the op."""
    for name in ("scale", "zero_point"):
        value = quantizer.__dict__.get(name)
        if isinstance(value, torch.Tensor) and not isinstance(value, torch.nn.Parameter):
            quantizer.__dict__[name] = None


def _residual_eager(
    W,
    A,
    B,
    scale,
    quantizer,
    key = None,
):
    merged = _merged_fp32(W, A, B, scale)
    fn = _compiled_tail(key, quantizer) if key is not None and W.is_cuda else None
    if fn is not None:
        try:
            return fn(merged, W)
        except Exception:
            _COMPILED_TAILS[key] = False  # this recipe does not compile here: eager from now on
    quantized = quantizer(merged.to(W.dtype))
    # In place, one fp32 weight-sized buffer; the same values as _residual_tail.
    return merged.sub_(W).neg_().add_(quantized).to(quantized.dtype)


@torch.library.custom_op("unsloth::qat_lora_merged_residual_v1", mutates_args = ())
def _residual_op(
    W: torch.Tensor, A: torch.Tensor, B: torch.Tensor, scale: float, key: int
) -> torch.Tensor:
    quantizer = _QUANTIZERS[key]
    try:
        with torch.no_grad():
            return _residual_eager(W, A, B, scale, quantizer, key)
    finally:
        _drop_qparams(quantizer)


@_residual_op.register_fake
def _(W, A, B, scale, key):
    return torch.empty_like(W)


@torch.library.custom_op("unsloth::qat_lora_merged_quantized_v1", mutates_args = ())
def _quantized_op(
    W: torch.Tensor, A: torch.Tensor, B: torch.Tensor, scale: float, key: int
) -> torch.Tensor:
    quantizer = _QUANTIZERS[key]
    try:
        with torch.no_grad():
            return quantizer(_merged_fp32(W, A, B, scale).to(W.dtype))
    finally:
        _drop_qparams(quantizer)


@_quantized_op.register_fake
def _(W, A, B, scale, key):
    return torch.empty_like(W)


@torch.library.custom_op("unsloth::qat_lora_merged_weight_v1", mutates_args = ())
def _merged_weight_op(
    W: torch.Tensor, A: torch.Tensor, B: torch.Tensor, scale: float
) -> torch.Tensor:
    return qat_lora_merged_weight(W, A, B, scale)


@_merged_weight_op.register_fake
def _(W, A, B, scale):
    return torch.empty_like(W)


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


class MergedLoRAWeightFakeQuantizer(torch.nn.Module):
    """Stands in for a QAT base layer's weight_fake_quantizer: returns Q(W + s B A) - s B A while the adapter is
    active, so base + LoRA computes Q(W + s B A) x. The wrapped torchao quantizer stays a child (`inner`)."""

    def __init__(self, inner, layer):
        super().__init__()
        self.inner = inner
        # Not a child module: the LoRA layer owns this quantizer, a child link would make a module cycle.
        self.__dict__["_lora_layer"] = layer
        self.__dict__["_key"] = _quantizer_key(inner)

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

    def _unsloth_memo_extra_key(self):
        """What the cached output (models/_qat_weight_memo.py) depends on besides W: the active adapter, by
        storage and version, so an optimizer step (an in-place update) recomputes it."""
        adapter = _active_adapter(self.__dict__["_lora_layer"])
        if adapter is None:
            return ("base",)
        A, B, scale = adapter
        if type(A.data) is not torch.Tensor or type(B.data) is not torch.Tensor:
            return False
        return (A.data_ptr(), A._version, B.data_ptr(), B._version, float(scale))

    def forward(self, W):
        inner = self.inner
        if not getattr(inner, "enabled", True):
            return W
        adapter = _active_adapter(self.__dict__["_lora_layer"])
        if adapter is None:
            return inner(W)
        A, B, scale = adapter
        if not torch.compiler.is_compiling():
            _STATS["calls"] += 1
        if self.__dict__.get(
            "_full", False
        ):  # the LoRA layer adds a zero-valued adapter branch itself
            return _quantized_op(
                W.detach(), A.detach(), B.detach(), float(scale), self.__dict__["_key"]
            )
        return _residual_op(W.detach(), A.detach(), B.detach(), float(scale), self.__dict__["_key"])


def _exact_lora_forward(self, x, *args, **kwargs):
    """LoRA Linear over an activation quantized base (fake quant int8 / fp8 activations): the base forward, whose
    weight quantizer returns Q(W + s B A) here, is bit for bit the exported layer; the adapter branch is added with
    value exactly 0 to carry the straight-through LoRA gradients, on a detached input so dX comes from the base."""
    base = self.get_base_layer()
    adapter = _active_adapter(self)
    quantizer = base._modules.get("weight_fake_quantizer")
    # A switched-off quantizer returns the plain base weight, and an export strips it: the adapter adds its value.
    if adapter is None or args or kwargs or quantizer is None or not quantizer.enabled:
        return type(self).forward(self, x, *args, **kwargs)
    A, _, scale = adapter
    out = base(x)
    name = next(a for a in self.active_adapters if a in self.lora_A)
    lora = self.lora_B[name](self.lora_A[name](x.detach().to(A.dtype))) * scale
    return out + (lora - lora.detach()).to(out.dtype)


def _refuse_mixed_batch(self, *args, **kwargs):
    base = self.get_base_layer()
    if base._modules.get("weight_fake_quantizer") is None and "weight_config" not in base.__dict__:
        return type(self)._mixed_batch_forward(
            self, *args, **kwargs
        )  # QAT stripped: plain LoRA again
    raise NotImplementedError(
        "Unsloth: merged QAT-LoRA layers compute the base and the adapter as one merged weight, so a batch that "
        "mixes adapters per row (adapter_names = ...) cannot run on them. Run one adapter per batch."
    )


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


def _merged_microscaling_forward(self, x, *args, **kwargs):
    """LoRA Linear over an NVFP4 / MX QAT base: the base's own low-precision forward on W + s B A, bit for bit
    the exported layer (W4A4 / W8A8 amplify any rounding difference through the activation quantizer), plus
    the adapter branch with value exactly 0 that carries the straight-through LoRA gradients. The adapter reads
    a detached input, so dX = G Q(W + s B A) comes from the base alone."""
    base = self.get_base_layer()
    adapter = None if base.__dict__.get(_TEACHER, False) else _active_adapter(self)
    # "weight_config": still a torchao NVFP4 / MX QAT linear (an export strips it to nn.Linear).
    if adapter is None or args or kwargs or "weight_config" not in base.__dict__:
        return type(self).forward(self, x, *args, **kwargs)
    A, B, scale = adapter
    if not torch.compiler.is_compiling():
        _STATS["calls"] += 1
    merged = _merged_weight_op(base.weight.detach(), A.detach(), B.detach(), float(scale))
    out = self.__dict__["_unsloth_qat_lora_weight_forward"](base, x, weight = merged)
    adapter_name = next(a for a in self.active_adapters if a in self.lora_A)
    lora = self.lora_B[adapter_name](self.lora_A[adapter_name](x.detach().to(A.dtype))) * scale
    return out + (lora - lora.detach()).to(out.dtype)


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
    base = layer.get_base_layer()
    weight = base._parameters.get("weight")
    if weight is None or type(weight.data) is not torch.Tensor or weight.device.type == "meta":
        return f"{name} has no plain base weight (sharded, offloaded or a tensor subclass)"
    config = getattr(base._modules.get("weight_fake_quantizer"), "config", None)
    if getattr(config, "range_learning", False) or not getattr(config, "is_dynamic", True):
        return f"{name} uses a static or learned weight scale; merged QAT-LoRA needs dynamic scales"
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
            if base._modules.get("activation_fake_quantizer") is not None:
                # Never on the fused LoRA kernels (they skip activation quantized layers): exact path.
                base.weight_fake_quantizer.__dict__["_full"] = True
                if "_unsloth_qat_lora_exact" not in layer.__dict__:
                    layer.__dict__["_unsloth_qat_lora_exact"] = True
                    layer.forward = types.MethodType(_exact_lora_forward, layer)
        else:
            if "_unsloth_qat_lora_weight_forward" not in layer.__dict__:
                layer.__dict__["_unsloth_qat_lora_weight_forward"] = weight_forward
                layer.forward = types.MethodType(_merged_microscaling_forward, layer)
            microscaling += 1
        # PEFT's per-row adapter selection would split what the merged weight fuses: refuse it.
        layer._mixed_batch_forward = types.MethodType(_refuse_mixed_batch, layer)
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
