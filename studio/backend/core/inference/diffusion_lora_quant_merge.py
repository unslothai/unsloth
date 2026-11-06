# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Merges the LoRA and LoKR adapters of a ComfyUI-quantized single file into its quantized weights.

A quantized single file loaded with adapters bakes them before the compile. Run as layers, a LoRA adds two small GEMMs
per adapted Linear and a full-rank LoKR about a quarter of the Linear's own GEMM in bf16 (measured: +0.55 s per step on
Krea 2 fp8, +0.66 s on MOSS-Image int8 ConvRot, both above the base model's own step). ComfyUI patches the adapters
into the weights instead: dequantize, add the delta, quantize back. This does the same for torchao ``Float8Tensor`` and
``Int8Tensor`` weights (ConvRot rotation included) and plain float weights, then removes the adapter layers, so the
compiled model runs the base layers alone.

The original quantized data of every touched layer is kept in host memory. A weight change copies it back into the same
tensors and merges again, so compiled code and captured graphs, which read those tensors, stay valid."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Optional

MERGE_ENV = "UNSLOTH_DIFFUSION_LORA_QUANT_MERGE"
_MERGED_ATTR = "_unsloth_lora_quant_merged"
# Host RAM left free after the backups.
_HOST_RESERVE = 8 << 30
_FP8_MAX = 448.0
_INT8_MAX = 127.0


def enabled() -> bool:
    return (os.environ.get(MERGE_ENV) or "").strip().lower() not in ("0", "off", "false", "no")


def merged(pipe: Any) -> bool:
    return getattr(pipe, _MERGED_ATTR, None) is not None


@dataclass
class _Term:
    """One adapter's contribution to one layer: ``scale * weight * delta``, delta = ``B @ A`` or ``kron(w1, w2)``."""

    adapter: str
    kind: str  # "lora" or "lokr"
    a: Any  # A (r, in) or w1
    b: Any  # B (out, r) or w2
    scale: float
    rows: Optional[tuple[int, int]] = None


@dataclass
class _Layer:
    module: Any
    group: int  # ConvRot group, 0 for none
    backup: dict = field(default_factory = dict)
    terms: list = field(default_factory = list)


def _weight_kind(weight: Any) -> Optional[str]:
    import torch

    name = type(weight).__name__
    if name in ("Float8Tensor", "Int8Tensor"):
        qdata, scale = getattr(weight, "qdata", None), getattr(weight, "scale", None)
        # Per-row symmetric, the layout a ComfyUI file's fp8 / int8 layers load into; anything else keeps its adapters.
        if (
            qdata is None
            or scale is None
            or qdata.dim() != 2
            or tuple(scale.shape) != (qdata.shape[0], 1)
        ):
            return None
        zero = getattr(weight, "zero_point", None)
        if zero is not None and bool(torch.count_nonzero(zero)):
            return None
        return "fp8" if name == "Float8Tensor" else "int8"
    if type(weight) is torch.Tensor or (
        isinstance(weight, torch.nn.Parameter) and type(weight.data) is torch.Tensor
    ):
        return "dense" if weight.is_floating_point() else None
    return None


def _tensors(weight: Any, kind: str) -> dict:
    if kind == "dense":
        return {"data": weight.data}
    return {"qdata": weight.qdata, "scale": weight.scale}


def _group(module: Any) -> int:
    from .diffusion_convrot import is_rotated_linear
    return int(module.convrot_groupsize) if is_rotated_linear(module) else 0


def _dequant(module: Any, kind: str, group: int, device: Any) -> Any:
    """The layer's float32 weight (ConvRot rotation undone)."""
    import torch

    weight = module.weight
    if kind == "dense":
        return weight.data.to(device = device, dtype = torch.float32)
    w = weight.qdata.to(device = device, dtype = torch.float32) * weight.scale.to(
        device = device, dtype = torch.float32
    )
    if group:
        from .diffusion_convrot import build_convrot_hadamard

        # Stored as W @ blockdiag(H).T; H is symmetric and orthogonal, so W = that @ H.
        h = build_convrot_hadamard(group, device = device, dtype = torch.float32)
        rows, cols = w.shape
        w = (w.reshape(rows, cols // group, group) @ h).reshape(rows, cols)
    return w


def _stochastic_fp8(x: Any, dtype: Any, generator: Any) -> Any:
    """Rounds ``x`` (float32, already within the fp8 range) to e4m3 at random, in proportion to its distance from the
    two neighbours, as ComfyUI rounds a patched fp8 weight: a delta below half a step survives on average instead of
    always rounding away."""
    import torch

    mag = x.abs()
    # e4m3: 3 mantissa bits, smallest normal exponent -6 (the subnormal step below it is 2^-9).
    exponent = torch.floor(torch.log2(mag.clamp(min = 2.0**-6))).clamp(max = 8)
    step = torch.exp2(exponent - 3)
    noise = torch.rand(x.shape, device = x.device, generator = generator)
    rounded = torch.floor(x / step + noise) * step
    return rounded.clamp(-_FP8_MAX, _FP8_MAX).to(dtype)


def _requant_(module: Any, kind: str, group: int, w: Any, seed: int) -> None:
    """Writes ``w`` (float32) into the layer's own tensors in the layout they already have, with a scale recomputed for
    the merged weight and seeded stochastic rounding (the same codes for the same weights every time)."""
    import torch

    weight = module.weight
    if kind == "dense":
        weight.data.copy_(w.to(weight.dtype))
        return
    if group:
        from .diffusion_convrot import build_convrot_hadamard

        h = build_convrot_hadamard(group, device = w.device, dtype = torch.float32)
        rows, cols = w.shape
        w = (w.reshape(rows, cols // group, group) @ h.T).reshape(rows, cols)
    amax = w.abs().amax(dim = -1, keepdim = True)
    top = _FP8_MAX if kind == "fp8" else _INT8_MAX
    scale = (amax / top).clamp(min = 1e-12)
    generator = torch.Generator(device = w.device)
    generator.manual_seed(seed)
    q = w / scale
    if kind == "fp8":
        codes = _stochastic_fp8(q, weight.qdata.dtype, generator)
    else:
        noise = torch.rand(q.shape, device = q.device, generator = generator)
        codes = torch.floor(q + noise).clamp(-128.0, 127.0).to(torch.int8)
    weight.qdata.copy_(codes.to(weight.qdata.device))
    weight.scale.copy_(scale.to(weight.scale))


def _delta(term: _Term, device: Any) -> Any:
    import torch

    a = term.a.to(device = device, dtype = torch.float32)
    b = term.b.to(device = device, dtype = torch.float32)
    if term.kind == "lora":
        out = b @ a
    else:
        out = torch.kron(a, b)
        if term.rows is not None:
            out = out[term.rows[0] : term.rows[1]]
    return out


def _peft_terms(layer: Any) -> Optional[list]:
    """The active adapters of a PEFT Linear LoRA layer at weight 1, or None when one cannot merge exactly."""
    import torch

    base = layer.get_base_layer()
    if not isinstance(base, torch.nn.Linear) or getattr(layer, "fan_in_fan_out", False):
        return None
    out = []
    for adapter in getattr(layer, "active_adapters", None) or ():
        if adapter not in getattr(layer, "lora_A", {}):
            continue
        if (getattr(layer, "use_dora", {}) or {}).get(adapter) or (
            getattr(layer, "lora_bias", {}) or {}
        ).get(adapter):
            return None
        a = layer.lora_A[adapter].weight.detach()
        b = layer.lora_B[adapter].weight.detach()
        # set_adapter_weights ran at weight 1 before this read: scaling is the adapter's own alpha / r.
        out.append(_Term(adapter, "lora", a.cpu(), b.cpu(), float(layer.scaling[adapter])))
    return out


def plan(pipe: Any, denoisers: list) -> Optional[list]:
    """Every adapted layer of ``denoisers`` with its adapter terms, or None when anything cannot merge: an adapter
    layer outside the denoisers, a non-Linear or DoRA layer, or a weight format with no requantizer here."""
    from . import diffusion_lora_files as lora_files

    ids = {id(m) for d in denoisers for m in d.modules()}
    layers: dict[int, _Layer] = {}
    for component in (getattr(pipe, "components", None) or {}).values():
        modules = getattr(component, "modules", None)
        if not callable(modules):
            continue
        for m in modules():
            if hasattr(m, "lora_A") and callable(getattr(m, "get_base_layer", None)):
                if id(m) not in ids:
                    return None  # a text encoder carries the adapter too
                terms = _peft_terms(m)
                if terms is None:
                    return None
                if not terms:
                    continue
                base = m.get_base_layer()
                entry = layers.setdefault(id(base), _Layer(base, _group(base)))
                entry.terms.extend(terms)
    for adapter_name, adapters in pipe.__dict__.get(lora_files.KRON_ATTR, {}).items():
        for adapter in adapters:
            base = adapter.wrapper.get_base_layer()
            if id(base) not in ids:
                return None
            w1 = getattr(adapter.wrapper, adapter.prefix + "_w1").detach()
            w2 = getattr(adapter.wrapper, adapter.prefix + "_w2").detach()
            rows = next((r for p, r in adapter.wrapper.slots if p == adapter.prefix), None)
            entry = layers.setdefault(id(base), _Layer(base, _group(base)))
            entry.terms.append(
                _Term(adapter_name, "lokr", w1.cpu(), w2.cpu(), float(adapter.alpha_scale), rows)
            )
    if not layers:
        return None
    for entry in layers.values():
        weight = getattr(entry.module, "weight", None)
        if weight is None or _weight_kind(weight) is None:
            return None
    return list(layers.values())


def _host_room(layers: list) -> bool:
    nbytes = sum(
        t.numel() * t.element_size()
        for e in layers
        for t in _tensors(e.module.weight, _weight_kind(e.module.weight)).values()
    )
    try:
        import psutil
        return int(psutil.virtual_memory().available) - nbytes >= _HOST_RESERVE
    except Exception:  # noqa: BLE001 - unreadable: assume it fits, as the load itself did
        return True


def _apply(layers: list, weights: dict, device: Any) -> None:
    import torch

    # Full float32 matmuls (the rotation, B @ A), whatever the speed profile set: the same weights merge to the
    # same codes before and after the profile engages.
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        with torch.no_grad():
            for index, entry in enumerate(layers):
                module = entry.module
                kind = _weight_kind(module.weight)
                live = [t for t in entry.terms if weights.get(t.adapter, 0.0)]
                if not live:
                    continue
                w = _dequant(module, kind, entry.group, device)
                for term in live:
                    w.add_(_delta(term, device), alpha = term.scale * float(weights[term.adapter]))
                _requant_(module, kind, entry.group, w, seed = index)
                del w
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32


def _restore(layers: list) -> None:
    import torch
    with torch.no_grad():
        for entry in layers:
            kind = _weight_kind(entry.module.weight)
            for key, live in _tensors(entry.module.weight, kind).items():
                live.copy_(entry.backup[key])


def merge(
    pipe: Any,
    denoisers: list,
    weights: dict,
    device: Any,
    logger: Any = None,
) -> bool:
    """Merges the attached adapters at ``weights`` (name -> weight) and removes the adapter layers. False, with the
    adapters left attached, when they cannot all be merged or the host has no room for the backups."""
    if not enabled() or merged(pipe):
        return False
    from . import diffusion_lora_files as lora_files

    names = list(weights)
    try:
        lora_files.set_adapter_weights(pipe, names, [1.0] * len(names))
        layers = plan(pipe, denoisers)
    finally:
        lora_files.set_adapter_weights(pipe, names, [weights[n] for n in names])
    if layers is None or not _host_room(layers):
        return False
    for entry in layers:
        kind = _weight_kind(entry.module.weight)
        entry.backup = {
            k: t.detach().to("cpu", copy = True)
            for k, t in _tensors(entry.module.weight, kind).items()
        }
    try:
        _apply(layers, weights, device)
    except Exception as exc:  # noqa: BLE001 - put the weights back, keep the adapters
        _restore(layers)
        if logger is not None:
            logger.warning(
                "diffusion.lora_merge: merge failed, keeping the adapter layers: %s", exc
            )
        return False
    lora_files.unload_adapters(pipe)
    setattr(pipe, _MERGED_ATTR, {"layers": layers, "weights": dict(weights), "device": device})
    if logger is not None:
        logger.info(
            "diffusion.lora_merge: %d adapter(s) merged into %d quantized layer(s); the adapter layers are removed",
            len(names),
            len(layers),
        )
    return True


def set_weights(pipe: Any, weights: dict) -> bool:
    """Re-merges at new weights (an omitted name is 0) in place; False when nothing was merged."""
    state = getattr(pipe, _MERGED_ATTR, None)
    if state is None:
        return False
    layers = state["layers"]
    device = state["device"]
    for entry in layers:
        live = next(iter(_tensors(entry.module.weight, _weight_kind(entry.module.weight)).values()))
        if live.device.type != "meta":
            device = live.device
            break
    _restore(layers)
    try:
        _apply(layers, weights, device)
    except Exception:
        # Never leave half the layers merged: the base weights, then the error.
        _restore(layers)
        state["weights"] = {}
        raise
    state["weights"] = dict(weights)
    return True
