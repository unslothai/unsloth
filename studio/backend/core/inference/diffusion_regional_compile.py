# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Names the repeated blocks for Diffusers transformers that declare none (``compile_repeated_blocks`` raises on an
empty list), on the instance only. Only classes verified to compile without graph breaks are listed."""

from __future__ import annotations

from typing import Any, Optional

_VERIFIED_BLOCKS: dict[str, tuple[str, ...]] = {
    "Lumina2Transformer2DModel": ("Lumina2TransformerBlock",),
    "HiDreamImageTransformer2DModel": (
        "HiDreamImageTransformerBlock",
        "HiDreamImageSingleTransformerBlock",
    ),
}


def discover_repeated_blocks(model: Any) -> tuple[str, ...]:
    """``_no_split_modules`` classes that occur at least twice inside an ``nn.ModuleList`` of ``model``."""
    names = tuple(getattr(model, "_no_split_modules", None) or ())
    if not names:
        return ()
    try:
        import torch.nn as nn
        counts: dict[str, int] = {}
        for sub in model.modules():
            if isinstance(sub, nn.ModuleList):
                for child in sub.modules():
                    name = type(child).__name__
                    if name in names:
                        counts[name] = counts.get(name, 0) + 1
    except Exception:  # noqa: BLE001 - a probe, never a failed load
        return ()
    return tuple(n for n in names if counts.get(n, 0) >= 2)


def verified_repeated_blocks(class_name: Optional[str]) -> tuple[str, ...]:
    return _VERIFIED_BLOCKS.get(class_name or "", ())


def ensure_repeated_blocks(model: Any) -> tuple[str, ...]:
    """The block names ``compile_repeated_blocks`` will use, supplying them for a verified class that declares none."""
    declared = tuple(getattr(model, "_repeated_blocks", None) or ())
    if declared:
        return declared
    verified = verified_repeated_blocks(type(model).__name__)
    if not verified:
        return ()
    found = discover_repeated_blocks(model)
    blocks = tuple(n for n in verified if n in found)
    if not blocks:
        return ()
    if type(model).__name__ == "HiDreamImageTransformer2DModel":
        install_traceable_moe(model)
    model._repeated_blocks = list(blocks)
    return blocks


def _moe_infer_dense(self: Any, x: Any, flat_expert_indices: Any, flat_expert_weights: Any) -> Any:
    # `where`, not a zero weight: an unpicked expert that overflows in fp16 must add 0, not 0 * inf = NaN.
    import torch

    k = self.num_activated_experts
    idx = flat_expert_indices.view(-1, k)
    weights = flat_expert_weights.view(-1, k).to(x.dtype)
    zero = x.new_zeros(())
    out = torch.zeros_like(x)
    for i, expert in enumerate(self.experts):
        hit = idx == i
        weight = (weights * hit).sum(-1, keepdim = True)
        out = out + torch.where(hit.any(-1, keepdim = True), expert(x) * weight, zero)
    return out


def install_traceable_moe(model: Any) -> int:
    """Replace HiDream's routed-expert loop, whose host read of per-expert counts (``bincount().cpu()``) breaks the
    graph in every block. Dense experts cost 2x the routed FLOPs (top-2 of 4) but need no host sync or gather."""
    patched = 0
    for sub in model.modules():
        if type(sub).__name__ != "MOEFeedForwardSwiGLU":
            continue
        if not callable(getattr(sub, "moe_infer", None)) or not hasattr(sub, "experts"):
            continue
        sub.moe_infer = _moe_infer_dense.__get__(sub)
        patched += 1
    return patched
