# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Linear / Embedding weights stay at stored half dtype and upcast exactly per call. UNSLOTH_STUDIO_MANUAL_CAST=0
disables. LoRA adapters on them stay at compute dtype, unfused (UNSLOTH_STUDIO_MANUAL_CAST_LORA=0 disables)."""

from __future__ import annotations

import os
from typing import Any, Optional

MANUAL_CAST_ENV = "UNSLOTH_STUDIO_MANUAL_CAST"
MANUAL_CAST_ATTR = "_unsloth_manual_cast"
MANUAL_CAST_LORA_ENV = "UNSLOTH_STUDIO_MANUAL_CAST_LORA"
_PEFT_PATCHED = "_unsloth_manual_cast_lora"

_CLASSES: dict[str, Any] = {}


def manual_cast_disabled() -> bool:
    return (os.environ.get(MANUAL_CAST_ENV) or "").strip().lower() in ("0", "off", "false", "no")


def manual_cast_lora_disabled() -> bool:
    return (os.environ.get(MANUAL_CAST_LORA_ENV) or "").strip().lower() in (
        "0",
        "off",
        "false",
        "no",
    )


def lora_compute_dtype(base: Any) -> Any:
    if not _CLASSES or type(base) not in (_CLASSES["linear"], _CLASSES["embedding"]):
        return None
    dtype = getattr(base, "_unsloth_compute_dtype", None)
    weight = getattr(base, "weight", None)
    if dtype is None or weight is None or weight.dtype == dtype or manual_cast_lora_disabled():
        return None
    return dtype


def _patch_peft_adapter_dtype() -> None:
    """PEFT casts adapters to the base weight dtype (bf16 GEMMs, rounded fp32 files); use compute dtype instead."""
    try:
        from peft.tuners.tuners_utils import BaseTunerLayer
    except Exception:  # noqa: BLE001 -- no PEFT, no adapters to place
        return
    original = BaseTunerLayer.__dict__.get("_move_adapter_to_device_of_base_layer")
    if original is None or getattr(original, _PEFT_PATCHED, False):
        return
    import torch

    def _move_adapter_to_device_of_base_layer(
        self,
        adapter_name,
        device = None,
    ):
        base = self.get_base_layer()
        dtype = lora_compute_dtype(base)
        if dtype is None:
            return original(self, adapter_name, device)
        if device is None:
            device = base.weight.device
        meta = torch.device("meta")
        for name in (*self.adapter_layer_names, *self.other_param_names):
            layer = getattr(self, name, None)
            if not hasattr(layer, "keys") or adapter_name not in layer.keys():
                continue
            item = layer[adapter_name]
            if not isinstance(item, (torch.nn.Module, torch.Tensor)):
                continue
            tensors = item.parameters() if isinstance(item, torch.nn.Module) else (item,)
            if any(t.device == meta for t in tensors):
                continue
            if isinstance(item, torch.nn.Module) or item.is_floating_point():
                layer[adapter_name] = item.to(device, dtype = dtype)
            else:
                layer[adapter_name] = item.to(device)

    setattr(_move_adapter_to_device_of_base_layer, _PEFT_PATCHED, True)
    _move_adapter_to_device_of_base_layer.__wrapped__ = original
    BaseTunerLayer._move_adapter_to_device_of_base_layer = _move_adapter_to_device_of_base_layer


def _classes() -> tuple[Any, Any]:
    if not _CLASSES:
        import torch
        import torch.nn.functional as F

        class ManualCastLinear(torch.nn.Linear):
            def forward(self, x):
                dtype = self._unsloth_compute_dtype
                weight = self.weight if self.weight.dtype == dtype else self.weight.to(dtype)
                bias = self.bias
                if bias is not None and bias.dtype != dtype:
                    bias = bias.to(dtype)
                return F.linear(x if x.dtype == dtype else x.to(dtype), weight, bias)

        class ManualCastEmbedding(torch.nn.Embedding):
            def forward(self, ids):
                out = F.embedding(
                    ids,
                    self.weight,
                    self.padding_idx,
                    self.max_norm,
                    self.norm_type,
                    self.scale_grad_by_freq,
                    self.sparse,
                )
                dtype = self._unsloth_compute_dtype
                return out if out.dtype == dtype else out.to(dtype)

        _CLASSES["linear"] = ManualCastLinear
        _CLASSES["embedding"] = ManualCastEmbedding
    return _CLASSES["linear"], _CLASSES["embedding"]


def _keep_fp32_patterns(module: Any) -> tuple[str, ...]:
    pats = getattr(module, "_keep_in_fp32_modules", None) or ()
    strict = getattr(module, "_keep_in_fp32_modules_strict", None) or ()
    return tuple(str(p) for p in (*pats, *strict))


def _first_float_owner(module: Any, keep: tuple[str, ...]) -> Optional[str]:
    """Owner of the param ``module.dtype`` reads; converted whole so pipelines cast latents to the compute dtype."""
    for name, param in module.named_parameters():
        if keep and any(k in name for k in keep):
            continue
        if param.is_floating_point():
            return name.rpartition(".")[0]
    return None


def apply_manual_cast_(module: Any, compute_dtype: Any) -> int:
    """Swap plain half Linear / Embedding for manual-cast twins, convert other half tensors; returns bytes kept.
    Subclasses (quantised, GGUF, LoRA) are left alone: they do their own cast."""
    import torch

    linear_cls, embedding_cls = _classes()
    _patch_peft_adapter_dtype()
    half = (torch.bfloat16, torch.float16)
    keep = _keep_fp32_patterns(module)
    first_owner = _first_float_owner(module, keep)
    kept_bytes = 0
    with torch.no_grad():
        for mname, sub in module.named_modules():
            swap = None
            if type(sub) is torch.nn.Linear:
                swap = linear_cls
            elif type(sub) is torch.nn.Embedding:
                swap = embedding_cls
            weight = getattr(sub, "weight", None)
            if (
                swap is not None
                and mname != first_owner
                and isinstance(weight, torch.Tensor)
                and weight.dtype in half
                and weight.device.type != "meta"
            ):
                sub.__class__ = swap
                sub._unsloth_compute_dtype = compute_dtype
                kept_bytes += weight.numel() * weight.element_size()
                bias = getattr(sub, "bias", None)
                if isinstance(bias, torch.Tensor):
                    kept_bytes += bias.numel() * bias.element_size()
                continue
            for _pname, param in list(sub.named_parameters(recurse = False)):
                if param.dtype in half:
                    param.data = param.data.to(compute_dtype)
            for bname, buf in list(sub.named_buffers(recurse = False)):
                if buf is not None and buf.dtype in half:
                    sub._buffers[bname] = buf.to(compute_dtype)
    setattr(
        module, MANUAL_CAST_ATTR, {"compute_dtype": str(compute_dtype), "kept_bytes": kept_bytes}
    )
    return kept_bytes


def undo_same_width_manual_cast_(module: Any) -> int:
    """Revert same-width manual-cast Linear / Embedding (bf16 for fp16 compute) to plain compute-dtype modules,
    since that cast saves no memory; returns bytes converted. Call after a quant rebuilt the big Linears."""
    import torch

    if not _CLASSES:
        return 0
    linear_cls, embedding_cls = _CLASSES["linear"], _CLASSES["embedding"]
    plain = {linear_cls: torch.nn.Linear, embedding_cls: torch.nn.Embedding}
    converted = 0
    with torch.no_grad():
        for sub in module.modules():
            base = plain.get(type(sub))
            if base is None:
                # never read .weight of anything else: a quantised Linear dequantises it on access
                continue
            dtype = getattr(sub, "_unsloth_compute_dtype", None)
            weight = getattr(sub, "weight", None)
            if not isinstance(dtype, torch.dtype) or not isinstance(weight, torch.Tensor):
                continue
            if (
                weight.dtype != dtype
                and weight.element_size() != torch.empty(0, dtype = dtype).element_size()
            ):
                continue
            for name in ("weight", "bias"):
                t = getattr(sub, name, None)
                if isinstance(t, torch.Tensor) and t.is_floating_point() and t.dtype != dtype:
                    converted += t.numel() * t.element_size()
                    t.data = t.data.to(dtype)
            sub.__class__ = base
            del sub._unsloth_compute_dtype
    info = manual_cast_info(module)
    if info is not None:
        if any(type(sub) in plain for sub in module.modules()):
            info["kept_bytes"] = max(0, int(info.get("kept_bytes", 0)) - converted)
        else:
            delattr(module, MANUAL_CAST_ATTR)
    return converted


def manual_cast_info(module: Any) -> Optional[dict]:
    info = getattr(module, MANUAL_CAST_ATTR, None)
    return info if isinstance(info, dict) else None
