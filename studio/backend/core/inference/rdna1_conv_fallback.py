# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""LFM3's remote Triton conv kernel fails to compile on RDNA1 (``PassManager::run failed``); use its torch path there."""

from __future__ import annotations

import re
import sys
from typing import Any

from core.inference.rocm_bf16 import _device_gfx_arch, is_rocm_torch

_RDNA1_ARCH = re.compile(r"^gfx101[0-9a-f]$")
_GATE = "can_use_lfm3_conv"


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


def _rdna1_aware(original: Any, torch: Any):
    archs: dict[int, str] = {}

    def gate(hidden_states, weight, *args, **kwargs):
        # The gate is module-wide, so a model on another card in this process keeps the Triton kernel.
        device = getattr(hidden_states, "device", None)
        if getattr(device, "type", None) != "cuda":
            return False
        index = device.index or 0
        if index not in archs:
            archs[index] = _device_gfx_arch(torch, index)
        if _RDNA1_ARCH.match(archs[index]):
            return False
        return original(hidden_states, weight, *args, **kwargs)

    gate._unsloth_rdna1 = True
    return gate


def disable_lfm3_triton_conv_on_rdna1(model: Any, torch: Any, log: Any = None) -> list[str]:
    """Make every loaded remote-code module's ``can_use_lfm3_conv`` refuse RDNA1 tensors when the model sits on an RDNA1 card."""
    if not is_rocm_torch(torch):
        return []
    archs = {_device_gfx_arch(torch, index) for index in _model_cuda_indices(model)}
    rdna1 = sorted(a for a in archs if _RDNA1_ARCH.match(a))
    if not rdna1:
        return []
    patched = []
    for name, module in list(sys.modules.items()):
        if not name.startswith("transformers_modules.") or module is None:
            continue
        original = getattr(module, _GATE, None)
        if callable(original) and not getattr(original, "_unsloth_rdna1", False):
            setattr(module, _GATE, _rdna1_aware(original, torch))
            patched.append(name)
    if patched and log is not None:
        log.info(
            f"LFM3 Triton conv disabled on {', '.join(rdna1)} (fails to compile on RDNA1), "
            f"using the torch conv: {', '.join(patched)}"
        )
    return patched
