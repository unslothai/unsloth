# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Supplies ``_repeated_blocks`` (on the instance) for verified Diffusers transformers that declare none."""

from __future__ import annotations

from typing import Any, Optional

_VERIFIED_BLOCKS: dict[str, tuple[str, ...]] = {
    "Lumina2Transformer2DModel": ("Lumina2TransformerBlock",),
    "CosmosTransformer3DModel": ("CosmosTransformerBlock",),
    "HiDreamImageTransformer2DModel": (
        "HiDreamImageTransformerBlock",
        "HiDreamImageSingleTransformerBlock",
    ),
}


def discover_repeated_blocks(model: Any) -> tuple[str, ...]:
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
    blocks = tuple(getattr(model, "_repeated_blocks", None) or ())
    if not blocks:
        verified = verified_repeated_blocks(type(model).__name__)
        found = discover_repeated_blocks(model) if verified else ()
        blocks = tuple(n for n in verified if n in found)
        if not blocks:
            return ()
        model._repeated_blocks = list(blocks)
    # Also when Diffusers declares the blocks itself: its released moe_infer still host-syncs.
    if type(model).__name__ == "HiDreamImageTransformer2DModel":
        install_traceable_moe(model)
    return blocks


# HiDream routes each token to k of E experts (k = 2 of 4). The dense path below runs every expert on every token; on a
# bf16 CUDA denoiser the experts instead run as three grouped GEMMs over the tokens sorted by expert (torch._grouped_mm,
# device offsets: no host sync, static shapes). Same per-token math up to the GEMMs' reduction order (rel L2 2.5e-3 vs
# dense at HiDream-I1 dims; the released loop is 4.5e-3 from dense), the two picked contributions are summed in the dense
# path's order. One MoE layer, 4096 tokens, compiled: A100 8.25 -> 5.41 ms, L4 38.9 -> 21.0 ms, RTX PRO 6000 5.90 ->
# 3.57 ms. ``UNSLOTH_DIFFUSION_HIDREAM_GROUPED_MOE=0`` keeps the dense experts.
GROUPED_MOE_ENV = "UNSLOTH_DIFFUSION_HIDREAM_GROUPED_MOE"
_GROUPED_ATTR = "_unsloth_grouped_moe"
_GROUPED_CALLS = [0]


def grouped_moe_enabled() -> bool:
    import os
    return (os.environ.get(GROUPED_MOE_ENV) or "").strip().lower() not in (
        "0",
        "off",
        "false",
        "no",
    )


def _grouped_supported() -> bool:
    """torch._grouped_mm exists and every visible CUDA card is sm80+ (NVIDIA; ROCm keeps the dense experts)."""
    try:
        import torch

        if not hasattr(torch, "_grouped_mm") or getattr(torch.version, "hip", None):
            return False
        if not torch.cuda.is_available():
            return False
        return all(
            torch.cuda.get_device_capability(i) >= (8, 0) for i in range(torch.cuda.device_count())
        )
    except Exception:  # noqa: BLE001
        return False


def _plain_expert(e: Any) -> bool:
    """A stock HiDream SwiGLU expert: three bias-free plain Linears with plain (not quantized / subclass) weights and no
    hooks. Traced: attribute and type reads only."""
    import torch
    from torch import nn

    for lin in (e.w1, e.w2, e.w3):
        if (
            type(lin) is not nn.Linear
            or lin.bias is not None
            or type(lin.weight) is not nn.Parameter
        ):
            return False
        d = lin.__dict__
        if "forward" in d or d.get("_forward_hooks") or d.get("_forward_pre_hooks"):
            return False
    d = e.__dict__
    if "forward" in d or d.get("_forward_hooks") or d.get("_forward_pre_hooks"):
        return False
    return e.w1.weight.dtype is torch.bfloat16


def _grouped_ready(self: Any, x: Any) -> bool:
    import torch

    if (
        not self.__dict__.get(_GROUPED_ATTR, False)
        or not x.is_cuda
        or x.dtype is not torch.bfloat16
    ):
        return False
    nn_module = torch.nn.modules.module
    if nn_module._global_forward_hooks or nn_module._global_forward_pre_hooks:
        return False
    return all(_plain_expert(e) and e.w1.weight.device == x.device for e in self.experts)


def _moe_infer_grouped(
    self: Any, x: Any, flat_expert_indices: Any, flat_expert_weights: Any
) -> Any:
    """``_moe_infer_dense``'s result with each expert run only on its own tokens (three grouped GEMMs)."""
    import torch
    import torch.nn.functional as F

    experts = self.experts
    n_exp = len(experts)
    k = self.num_activated_experts
    n = x.shape[0]
    if not torch.compiler.is_compiling():
        _GROUPED_CALLS[0] += 1
    # [E, D, H] / [E, H, D] transposed views of stacked copies (transient: no second resident set of expert weights)
    w1 = torch.stack([e.w1.weight for e in experts]).transpose(1, 2)
    w3 = torch.stack([e.w3.weight for e in experts]).transpose(1, 2)
    w2 = torch.stack([e.w2.weight for e in experts]).transpose(1, 2)
    flat = flat_expert_indices.view(-1)
    order = flat.argsort(stable = True)
    # per-expert counts without bincount (data-dependent size: a graph break)
    counts = (flat.unsqueeze(1) == torch.arange(n_exp, device = flat.device)).sum(0)
    offs = counts.cumsum(0).to(torch.int32)
    xs = x.index_select(0, order // k)
    h = F.silu(torch._grouped_mm(xs, w1, offs = offs)) * torch._grouped_mm(xs, w3, offs = offs)
    y = torch._grouped_mm(h, w2, offs = offs)
    y = y * flat_expert_weights.view(-1, 1).to(x.dtype).index_select(0, order)
    ys = torch.empty_like(y).index_copy_(0, order, y).view(n, k, -1)
    if k == 1:
        return torch.zeros_like(x) + ys[:, 0]
    # the dense path adds the experts in index order, one rounding per add: reproduce that order for k = 2
    idx = flat.view(n, k)
    lo_first = (idx[:, :1] <= idx[:, 1:2]).unsqueeze(-1)
    first = torch.where(lo_first, ys[:, :1], ys[:, 1:2]).squeeze(1)
    second = torch.where(lo_first, ys[:, 1:2], ys[:, :1]).squeeze(1)
    out = torch.zeros_like(x) + first
    out = out + second
    for j in range(2, k):  # k > 2 (not HiDream): later slots in their own order
        out = out + ys[:, j]
    return out


def _moe_infer_dense(self: Any, x: Any, flat_expert_indices: Any, flat_expert_weights: Any) -> Any:
    # `where`, not a zero weight: an unpicked expert that overflows in fp16 must add 0, not 0 * inf = NaN.
    import torch

    if _grouped_ready(self, x):
        return _moe_infer_grouped(self, x, flat_expert_indices, flat_expert_weights)
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
    """Swap HiDream's ``moe_infer`` (host ``bincount().cpu()`` breaks every block's graph) for dense experts."""
    patched = 0
    grouped = grouped_moe_enabled() and _grouped_supported()
    for sub in model.modules():
        if type(sub).__name__ != "MOEFeedForwardSwiGLU":
            continue
        if not callable(getattr(sub, "moe_infer", None)) or not hasattr(sub, "experts"):
            continue
        sub.moe_infer = _moe_infer_dense.__get__(sub)
        sub.__dict__[_GROUPED_ATTR] = grouped
        patched += 1
    return patched
