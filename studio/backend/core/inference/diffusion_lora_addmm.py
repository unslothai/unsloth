# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unfolded LoRA delta accumulated in the GEMM epilogue: ``addmm(base(x), x @ A^T, B^T, alpha = scale)``.

PEFT's ``Linear.forward`` runs ``result + lora_B(lora_A(x)) * scaling``: the full-width B output, a scaled copy and
the sum are three [rows, out] passes per adapted layer. When the adapters stay unfolded (GGUF, a group-offloaded
pipe, the first render after a change) that is every step. Here the sum and the scale ride in the B GEMM's epilogue
(fp32, one rounding), out of place so a base output that something else holds is never written. L4, FLUX.2 klein 4B
at 1024 px, rank-32 adapter on 100 layers, s/step over no adapter: Q8_0 GGUF +20.9% -> +10.4%, bf16 resident
(group policy, unfolded) +23.6% -> +9.9%, bf16 streamed (low_vram) +70% -> +51%.

Only the plain case is taken: one or more vanilla adapters (no dropout, DoRA, LoRA bias, variant), float scales,
no mixed batch, nothing merged or disabled, adapters, input and base output in one dtype, no autograd. Anything
else runs PEFT's own forward. UNSLOTH_DIFFUSION_LORA_ADDMM=0 disables."""

from __future__ import annotations

import os
from typing import Any

LORA_ADDMM_ENV = "UNSLOTH_DIFFUSION_LORA_ADDMM"
_PATCHED = "_unsloth_lora_addmm"
CALLS = [0, 0]  # [epilogue path, PEFT path], eager calls only


def _count(path: int) -> None:
    import torch

    # a traced write to a global is guarded on its value: every compiled call would recompile
    if not torch.compiler.is_compiling():
        CALLS[path] += 1


def enabled() -> bool:
    return (os.environ.get(LORA_ADDMM_ENV) or "").strip().lower() not in ("0", "off", "false", "no")


def _hooked(module: Any) -> bool:
    """Whether a module's own forward does more than its matmul: offload / execution hooks (diffusers group or leaf
    offload, accelerate cpu / sequential offload) move its weights onto the device only inside that forward."""
    return bool(
        getattr(module, "_diffusers_hook", None) is not None
        or getattr(module, "_hf_hook", None) is not None
        or getattr(module, "_forward_pre_hooks", None)
        or getattr(module, "_forward_hooks", None)
    )


def _plain_adapters(layer: Any, x: Any) -> Any:
    """The active adapters when every one is a vanilla LoRA the epilogue path computes exactly, else None. An adapter
    whose modules carry offload hooks, or whose weights are not already on ``x``'s device, runs PEFT's forward, which
    goes through those hooks."""
    import torch

    dtype = x.dtype

    if layer.disable_adapters or layer.merged or getattr(layer, "lora_variant", None):
        return None
    active = [a for a in layer.active_adapters if a in layer.lora_A]
    for a in active:
        if not isinstance(layer.lora_dropout[a], torch.nn.Identity):
            return None
        if (getattr(layer, "use_dora", None) or {}).get(a) or (
            getattr(layer, "lora_bias", None) or {}
        ).get(a):
            return None
        if not isinstance(layer.scaling[a], (int, float)):
            return None
        if _hooked(layer.lora_A[a]) or _hooked(layer.lora_B[a]):
            return None
        wa, wb = layer.lora_A[a].weight, layer.lora_B[a].weight
        if wa.dtype != dtype or wb.dtype != dtype:
            return None
        if wa.device != x.device or wb.device != x.device:
            return None
        if (
            getattr(layer.lora_B[a], "bias", None) is not None
            or getattr(layer.lora_A[a], "bias", None) is not None
        ):
            return None
    return active


def _forward(original: Any) -> Any:
    def forward(self: Any, x: Any, *args: Any, **kwargs: Any) -> Any:
        import torch

        # autograd (training, or a grad-mode call) keeps PEFT's graph
        if args or kwargs or not enabled() or x.dim() < 2 or torch.is_grad_enabled():
            _count(1)
            return original(self, x, *args, **kwargs)
        active = _plain_adapters(self, x)
        if active is None:
            _count(1)
            return original(self, x, *args, **kwargs)
        result = self.base_layer(x)
        if not active:
            return result
        if (
            type(result) is not torch.Tensor
            or result.dtype != x.dtype
            or result.requires_grad
            or result.shape[:-1] != x.shape[:-1]
        ):
            # Rare base layers (a subclass output, another dtype): PEFT's own sum on the output already computed.
            _count(1)
            for a in active:
                result = result + self.lora_B[a](self.lora_A[a](x)) * self.scaling[a]
            return result
        _count(0)
        x2 = x.reshape(-1, x.shape[-1])
        out = result.reshape(-1, result.shape[-1])
        for a in active:
            h = x2 @ self.lora_A[a].weight.t()
            out = torch.addmm(out, h, self.lora_B[a].weight.t(), alpha = float(self.scaling[a]))
        return out.view(result.shape)

    setattr(forward, _PATCHED, True)
    forward._unsloth_original = original
    return forward


def install() -> bool:
    """Patch ``peft.tuners.lora.layer.Linear.forward`` once; False when PEFT is missing or its forward moved."""
    try:
        from peft.tuners.lora import layer as lora_layer
    except Exception:  # noqa: BLE001 -- no PEFT, no adapters
        return False
    linear = getattr(lora_layer, "Linear", None)
    original = getattr(linear, "forward", None) if linear is not None else None
    if original is None:
        return False
    if getattr(original, _PATCHED, False):
        return True
    linear.forward = _forward(original)
    return True
