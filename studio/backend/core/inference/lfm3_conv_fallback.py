# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""LFM3's remote Triton conv dies with ``PassManager::run failed`` where Triton cannot build for the card (gfx101x on Triton 3.9); use its torch path there."""

from __future__ import annotations

import sys
from typing import Any, Callable, Optional

from core.inference.rocm_bf16 import _device_gfx_arch, is_rocm_torch

_GATE = "can_use_lfm3_conv"


def _core_probe() -> Callable[[int], bool]:
    try:
        from unsloth.models._utils import triton_runs_on_device
    except Exception:  # noqa: BLE001 -- no probe: keep the remote code's own choice
        return lambda index: True
    return triton_runs_on_device


def _model_cuda_indices(model: Any) -> set[int]:
    indices = set()
    try:
        for param in model.parameters():
            device = param.device
            if device.type == "cuda":
                indices.add(device.index or 0)
    except Exception:  # noqa: BLE001 -- no parameters to read: nothing on a GPU to gate
        pass
    return indices


def _guarded(original: Any, runs_on: Callable[[int], bool]):
    def gate(hidden_states, weight, *args, **kwargs):
        # Module-wide gate: a model on a card Triton does support keeps the kernel.
        device = getattr(hidden_states, "device", None)
        if getattr(device, "type", None) != "cuda" or not runs_on(device.index or 0):
            return False
        return original(hidden_states, weight, *args, **kwargs)

    gate._unsloth_triton_guard = True
    return gate


def disable_lfm3_triton_conv_where_unsupported(
    model: Any,
    torch: Any,
    log: Any = None,
    runs_on: Optional[Callable[[int], bool]] = None,
) -> list[str]:
    """Make every loaded remote-code module's ``can_use_lfm3_conv`` refuse tensors on a ROCm card Triton cannot run on."""
    if not is_rocm_torch(torch):
        return []
    runs_on = runs_on or _core_probe()
    failing = sorted(index for index in _model_cuda_indices(model) if not runs_on(index))
    if not failing:
        return []
    patched = []
    for name, module in list(sys.modules.items()):
        if not name.startswith("transformers_modules.") or module is None:
            continue
        original = getattr(module, _GATE, None)
        if callable(original) and not getattr(original, "_unsloth_triton_guard", False):
            setattr(module, _GATE, _guarded(original, runs_on))
            patched.append(name)
    if patched and log is not None:
        archs = ", ".join(_device_gfx_arch(torch, index) or f"cuda:{index}" for index in failing)
        log.info(
            f"LFM3 Triton conv disabled on {archs} (Triton cannot build for it), "
            f"using the torch conv: {', '.join(patched)}"
        )
    return patched
