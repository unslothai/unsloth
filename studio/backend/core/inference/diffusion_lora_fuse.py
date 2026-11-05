# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Folds an unchanged LoRA set into a resident dense denoiser's weights, and restores them bit-exactly before a change.

PEFT runs every adapted layer as ``base(x) + B(A(x)) * scale``: two extra GEMMs per layer on every step. Once the
same set renders twice, the second render folds ``B @ A * scale`` into the base weights (fp32 add, one rounding to the
weight dtype) and marks each layer merged, so PEFT's forward runs the base layer alone. The original weights are kept
on the card when it has ample room (a restore is then a device copy), else in pinned host chunks, and copied back
before any change, so the weights-only fast path (``set_adapters``) runs on exactly the weights it would have seen
unfused. Opt out with UNSLOTH_DIFFUSION_LORA_FUSE=0."""

from __future__ import annotations

import os
from typing import Any, Optional

LORA_FUSE_ENV = "UNSLOTH_DIFFUSION_LORA_FUSE"
BACKUP_ENV = "UNSLOTH_DIFFUSION_LORA_FUSE_BACKUP"
_DEVICE_BACKUP_MIN_FREE = 8 << 30
_FUSED_ATTR = "_unsloth_lora_fused"
_ALIGN = 64
_CHUNK = 64 << 20  # a power of two: no rounding waste in the pinned host allocator


def enabled() -> bool:
    return (os.environ.get(LORA_FUSE_ENV) or "").strip().lower() not in ("0", "off", "false", "no")


def fused(pipe: Any) -> bool:
    return bool(getattr(pipe, _FUSED_ATTR, None))


def _lora_layers(module: Any) -> list:
    return [
        m
        for m in module.modules()
        if hasattr(m, "lora_A")
        and hasattr(m, "get_base_layer")
        and callable(getattr(m, "get_delta_weight", None))
    ]


def _plan(denoisers: list) -> Optional[list]:
    """(layer, active adapters, base weight) per adapted layer, or None when any layer cannot fold exactly."""
    import torch

    plan = []
    for denoiser in denoisers:
        for layer in _lora_layers(denoiser):
            if getattr(layer, "merged", False) or getattr(layer, "disable_adapters", False):
                return None
            active = [
                a for a in (getattr(layer, "active_adapters", None) or ()) if a in layer.lora_A
            ]
            if not active:
                continue
            if any(
                (getattr(layer, "use_dora", {}) or {}).get(a)
                or (getattr(layer, "lora_bias", {}) or {}).get(a)
                for a in active
            ):
                return None  # DoRA / LoRA bias change more than the weight
            weight = getattr(layer.get_base_layer(), "weight", None)
            if (
                not isinstance(weight, torch.Tensor)
                or type(weight.data) is not torch.Tensor  # quantised / GGUF / torchao subclasses
                or not weight.is_floating_point()
                or weight.device.type == "meta"
            ):
                return None
            plan.append((layer, active, weight))
    return plan or None


def _delta_fp32(layer: Any, adapter: str) -> Any:
    """``scale * (B @ A)`` computed in fp32. PEFT's ``get_delta_weight`` multiplies in the adapter dtype on the GPU
    (one bf16 rounding before the add); here its own math runs on fp32 copies of A and B, which are put back as they
    were, so the layout rules (fan_in_fan_out, conv kernels) stay PEFT's."""
    import torch

    mods = [
        m
        for m in (layer.lora_A[adapter], layer.lora_B[adapter])
        if getattr(m, "weight", None) is not None
    ]
    saved = [m.weight.data for m in mods]
    try:
        for m in mods:
            m.weight.data = m.weight.data.float()
        with torch.no_grad():
            return layer.get_delta_weight(adapter).float()
    finally:
        for m, w in zip(mods, saved):
            m.weight.data = w


def fuse(
    pipe: Any,
    denoisers: list,
    logger: Any = None,
) -> bool:
    """Fold the active adapters into ``denoisers``' weights; True when they are folded (now or already)."""
    if fused(pipe):
        return True
    plan = _plan(denoisers)
    if plan is None:
        return False
    import torch

    device = plan[0][2].device
    if any(w.device != device for _l, _a, w in plan):
        return False
    # Packed into power-of-two chunks: the pinned host allocator rounds every block up to a power of two, so one
    # slab of the whole backup would hold up to twice its size.
    places, sizes, used = [], [], _CHUNK
    for _layer, _active, weight in plan:
        nbytes = -(-weight.numel() * weight.element_size() // _ALIGN) * _ALIGN
        if nbytes > _CHUNK:
            places.append((len(sizes), 0))
            sizes.append(nbytes)
            used = _CHUNK
            continue
        if used + nbytes > _CHUNK:
            sizes.append(_CHUNK)
            used = 0
        places.append((len(sizes) - 1, used))
        used += nbytes
    total = sum(sizes)
    on_device = _backup_on_device(device, total)
    try:
        if on_device:
            chunks = [torch.empty(n, dtype = torch.uint8, device = device) for n in sizes]
        else:
            chunks = [
                torch.empty(n, dtype = torch.uint8, pin_memory = device.type == "cuda") for n in sizes
            ]
    except Exception:  # noqa: BLE001 - no room for the backup: stay unfused
        return False
    entries = []
    try:
        for (layer, active, weight), (ci, off) in zip(plan, places):
            nbytes = weight.numel() * weight.element_size()
            backup = chunks[ci][off : off + nbytes].view(weight.dtype).view(weight.shape)
            backup.copy_(weight.detach())
            delta = None
            for adapter in active:
                d = _delta_fp32(layer, adapter)
                delta = d if delta is None else delta + d
            with torch.no_grad():
                # W + scale * (B @ A) in fp32, rounded to the weight dtype once (as a weight-patching loader does)
                weight.copy_((weight.float() + delta).to(weight.dtype))
            layer.merged_adapters = list(active)
            entries.append((layer, weight, backup))
        _sync(device)
    except Exception as exc:  # noqa: BLE001 - put back what was folded, stay unfused
        _restore(entries)
        if logger is not None:
            logger.warning("diffusion.lora_fuse: fold failed, staying unfused: %s", exc)
        return False
    setattr(pipe, _FUSED_ATTR, {"entries": entries, "chunks": chunks})
    if logger is not None:
        logger.info(
            "diffusion.lora_fuse: folded the unchanged LoRA set into %d layers (%d MiB kept on the %s to restore)",
            len(entries),
            total >> 20,
            "card" if on_device else "host",
        )
    return True


def _restore(entries: list) -> None:
    import torch
    with torch.no_grad():
        for layer, weight, backup in entries:
            weight.copy_(backup)
            layer.merged_adapters = []
    if entries:
        _sync(entries[0][1].device)


def _backup_on_device(device: Any, nbytes: int) -> bool:
    """Keep the backup beside the weights (a restore is then a device copy, not a host upload) only when the card keeps
    max(8 GiB, a quarter of it) free after it; UNSLOTH_DIFFUSION_LORA_FUSE_BACKUP=host / device forces either."""
    mode = (os.environ.get(BACKUP_ENV) or "auto").strip().lower()
    if device.type != "cuda" or mode == "host":
        return False
    if mode == "device":
        return True
    try:
        import torch
        free, total = torch.cuda.mem_get_info(device)
    except Exception:  # noqa: BLE001 - unreadable: the host
        return False
    return int(free) - int(nbytes) >= max(_DEVICE_BACKUP_MIN_FREE, int(total) // 4)


def _sync(device: Any) -> None:
    if device.type == "cuda":
        import torch
        torch.cuda.synchronize(device)


def unfuse(pipe: Any) -> bool:
    """Copy the original weights back (bit-exact) and run the adapters unfused again; True when anything was folded."""
    state = getattr(pipe, _FUSED_ATTR, None)
    if not state:
        return False
    try:
        _restore(state["entries"])
    finally:
        setattr(pipe, _FUSED_ATTR, None)
    return True
