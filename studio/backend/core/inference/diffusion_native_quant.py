# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Torchao-free weight-only int8 / fp8 for the diffusion transformer, for hosts torchao cannot serve.

ROCm reaches here (gfx1151 refuses fp8 ``_scaled_mm``, and torchao's int8 W8A8 measured 0.48x bf16 speed
there), and so does Windows ROCm, where torchao cannot even import (torch ships no
``_c10d_functional.all_gather_into_tensor``) and Studio installs a stub. Each large Linear keeps its weight
in int8 or fp8 e4m3fn with one scale per output row and dequantises it to the activation dtype per forward:
the transformer's weights halve, the arithmetic stays bf16.

Measured on gfx1151 (Qwen-Image-2.1, 1024px, 20 steps): int8 weight-only ran at 0.97x bf16 speed with the
transformer at 6.64 GiB instead of 13.25. Weights and scales are plain buffers, so unlike torchao tensors
they survive the ``Module.to()`` calls the offload hooks make.

Imports torch lazily, like the rest of the quant modules, so the module loads on a torch-free host.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Callable, Optional

NATIVE_INT8 = "int8"
NATIVE_FP8 = "fp8"
_QMAX = {NATIVE_INT8: 127.0, NATIVE_FP8: 448.0}
# int8 W8A8 switch: "1" opts AMD in, "0" opts the NVIDIA offload route out (``native_int8_act``).
NATIVE_INT8_ACT_ENV = "UNSLOTH_NATIVE_INT8_ACT"
# ConvRot Hadamard spreads the DiT outlier channels that pin the per-row int8 scale; "0" turns it off.
NATIVE_INT8_ROT_ENV = "UNSLOTH_NATIVE_INT8_ROT"
# torch._int_mm wants M > 16 and K, N multiples of 8; smaller or odd shapes keep the weight-only path.
_INT_MM_MIN_ROWS = 17


def int8_act_requested() -> bool:
    import os
    return os.environ.get(NATIVE_INT8_ACT_ENV, "").strip() == "1"


def int8_act_disabled() -> bool:
    import os
    return os.environ.get(NATIVE_INT8_ACT_ENV, "").strip() == "0"


def int8_rotation_group() -> int:
    import os

    if os.environ.get(NATIVE_INT8_ROT_ENV, "").strip() == "0":
        return 0
    from .diffusion_convrot import DEFAULT_CONVROT_GROUPSIZE

    return DEFAULT_CONVROT_GROUPSIZE


@lru_cache(maxsize = 1)
def native_linear_class():
    """``NativeWeightOnlyLinear``, defined on first use so this module imports torch-free."""
    import torch
    from torch import nn
    from torch.nn import functional as F

    class NativeWeightOnlyLinear(nn.Module):
        """A Linear whose weight is stored as int8 / fp8 with a per-output-row scale."""

        def __init__(
            self,
            linear: Any,
            scheme: str,
            act_int8: bool = False,
            rot_group: int = 0,
        ):
            super().__init__()
            if scheme not in _QMAX:
                raise ValueError(f"unsupported native scheme {scheme!r}")
            self.in_features = int(linear.in_features)
            self.out_features = int(linear.out_features)
            self.scheme = scheme
            self.compute_dtype = linear.weight.dtype
            self.act_int8 = bool(act_int8) and scheme == NATIVE_INT8
            self.rot_group = (
                int(rot_group)
                if self.act_int8 and rot_group and self.in_features % int(rot_group) == 0
                else 0
            )
            weight = linear.weight.detach()
            with torch.no_grad():
                w = weight.float()
                if self.rot_group:
                    from .diffusion_convrot import build_convrot_hadamard

                    h = build_convrot_hadamard(self.rot_group, device = w.device, dtype = torch.float32)
                    g = self.rot_group
                    w = (w.reshape(self.out_features, -1, g) @ h.T).reshape(self.out_features, -1)
                    # Non-persistent: rebuilt from the group size, exact in any float dtype (entries are +-2^-k).
                    self.register_buffer("rot_h", h.to(self.compute_dtype), persistent = False)
                scale = w.abs().amax(dim = 1, keepdim = True).clamp(min = 1e-12) / _QMAX[scheme]
                if scheme == NATIVE_INT8:
                    wq = (w / scale).round_().clamp_(-127, 127).to(torch.int8)
                else:
                    wq = (w / scale).to(torch.float8_e4m3fn)
                del w
            # Integer views, so a module-wide ``.to(dtype)`` (which casts every floating buffer) can neither widen
            # the fp8 payload back to bf16 nor round the fp32 scales.
            self.register_buffer("weight_q", wq.view(torch.uint8) if scheme == NATIVE_FP8 else wq)
            self.register_buffer(
                "weight_scale", scale.squeeze(1).to(torch.float32).contiguous().view(torch.int32)
            )
            self.bias = linear.bias

        def _stored_weight(self, dtype: Any) -> Any:
            wq = (
                self.weight_q.view(torch.float8_e4m3fn)
                if self.scheme == NATIVE_FP8
                else self.weight_q
            )
            scale = self.weight_scale.view(torch.float32)
            return (wq.to(torch.float32) * scale[:, None]).to(dtype)

        def _rotate(self, x: Any) -> Any:
            g = self.rot_group
            return (x.reshape(-1, self.in_features // g, g) @ self.rot_h.to(x.dtype)).reshape(
                x.shape
            )

        def dequantized_weight(self, dtype: Any) -> Any:
            """The weight in the model's own basis, rotation undone (``H`` is orthogonal and symmetric)."""
            w = self._stored_weight(torch.float32 if self.rot_group else dtype)
            if not self.rot_group:
                return w
            g = self.rot_group
            h = self.rot_h.to(torch.float32)
            return (
                (w.reshape(self.out_features, -1, g) @ h).reshape(self.out_features, -1).to(dtype)
            )

        @property
        def weight(self) -> Any:
            """The dense weight, rebuilt on each read. Code that reads a wrapped Linear's weight directly
            (PEFT's DoRA forward reads ``base_layer.weight``) keeps working; nothing holds it."""
            return self.dequantized_weight(self.compute_dtype)

        def _int_mm_ok(self, rows: int) -> bool:
            return (
                rows >= _INT_MM_MIN_ROWS
                and self.in_features % 8 == 0
                and self.out_features % 8 == 0
            )

        def _forward_int_mm(self, x: Any) -> Any:
            # Must stay fast in eager (Windows ROCm has no Triton): upcasts ride mixed-dtype ops, no .float() copies.
            x2 = x.reshape(-1, self.in_features)
            x_scale = torch.linalg.vector_norm(
                x2, ord = float("inf"), dim = 1, keepdim = True, dtype = torch.float32
            )
            x_scale = x_scale.clamp_(min = 1e-12).div_(127.0)
            xq = torch.mul(x2, x_scale.reciprocal()).round_().clamp_(-127, 127).to(torch.int8)
            acc = torch._int_mm(xq, self.weight_q.t())
            w_scale = self.weight_scale.view(torch.float32)
            out = torch.mul(acc, x_scale)
            if self.bias is not None:
                out = torch.addcmul(self.bias.float(), out, w_scale)
            else:
                out = out.mul_(w_scale)
            return out.to(x.dtype).reshape(*x.shape[:-1], self.out_features)

        def forward(self, x: Any) -> Any:
            if self.rot_group:
                x = self._rotate(x)
            if (
                self.act_int8
                and x.is_cuda
                and self._int_mm_ok(x.numel() // max(1, self.in_features))
            ):
                return self._forward_int_mm(x)
            # Rotated or not, x and the stored weight share a basis here, so no un-rotation is needed.
            return F.linear(x, self._stored_weight(x.dtype), self.bias)

        def extra_repr(self) -> str:
            return (
                f"in_features={self.in_features}, out_features={self.out_features}, "
                f"scheme={self.scheme}, act_int8={self.act_int8}, rot_group={self.rot_group}, bias={self.bias is not None}"
            )

    return NativeWeightOnlyLinear


def is_native_linear(module: Any) -> bool:
    return type(module).__name__ == "NativeWeightOnlyLinear" and hasattr(module, "weight_q")


def is_native_quantised(module: Any) -> bool:
    """Whether any layer of ``module`` has been swapped for a native weight-only Linear."""
    try:
        return any(is_native_linear(sub) for sub in module.modules())
    except Exception:  # noqa: BLE001 -- a probe must never replace the failure it describes
        return False


def apply_native_weight_quant(
    transformer: Any,
    scheme: str,
    *,
    filter_fn: Callable[[Any, str], bool],
    act_int8: bool,
    logger: Any = None,
) -> int:
    """Swap every Linear ``filter_fn`` keeps for a native weight-only twin; returns layers swapped.

    One layer at a time, like torchao's ``quantize_``, so the build peak is the dense model plus one
    layer rather than 1.5x it; on unified memory that peak is what the OS kills for. A failure part-way
    leaves a partial conversion, which ``is_native_quantised`` reports so the loader refuses it the
    same way it refuses a partial torchao pass."""
    cls = native_linear_class()
    act_int8 = scheme == NATIVE_INT8 and bool(act_int8)
    rot_group = int8_rotation_group() if act_int8 else 0
    # Names only: holding the old modules would keep every bf16 weight alive until the pass ends.
    names = [name for name, mod in transformer.named_modules() if name and filter_fn(mod, name)]
    for name in names:
        parent_name, _, leaf = name.rpartition(".")
        parent = transformer.get_submodule(parent_name) if parent_name else transformer
        setattr(
            parent, leaf, cls(getattr(parent, leaf), scheme, act_int8 = act_int8, rot_group = rot_group)
        )
    if logger is not None:
        logger.info(
            "diffusion.transformer_quant: %s %s (torchao-free) on %d linears",
            scheme,
            ("W8A8 torch._int_mm" + (f" ConvRot g{rot_group}" if rot_group else ""))
            if act_int8
            else "weight-only",
            len(names),
        )
    return len(names)


def _first_native_linear(module: Any) -> Any:
    try:
        return next((sub for sub in module.modules() if is_native_linear(sub)), None)
    except Exception:  # noqa: BLE001 -- a probe must never raise
        return None


def native_quant_signature(module: Any) -> Optional[str]:
    layer = _first_native_linear(module)
    if layer is None:
        return None
    if not getattr(layer, "act_int8", False):
        return f"{layer.scheme}-wo"
    rot = int(getattr(layer, "rot_group", 0) or 0)
    return f"{layer.scheme}-w8a8" + (f"-rot{rot}" if rot else "")


def native_quant_reason(module: Any, scheme: str) -> str:
    layer = _first_native_linear(module)
    if layer is not None and getattr(layer, "act_int8", False):
        rot = int(getattr(layer, "rot_group", 0) or 0)
        return (
            f"W8A8: {scheme} weights and activations through torch._int_mm"
            + (f" with a g{rot} ConvRot rotation" if rot else "")
            + " (torchao-free; plain buffers, so the offload hooks can move it)"
        )
    return (
        f"weight-only: {scheme} weights, bf16 compute "
        "(torchao-free, a memory saving rather than a speed-up)"
    )


def native_weight_error(linear: Any, scheme: str) -> Optional[float]:
    """Relative Frobenius error of one Linear's native reconstruction; for tests and probes."""
    try:
        import torch

        layer = native_linear_class()(linear, scheme)
        ref = linear.weight.detach().float()
        rec = layer.dequantized_weight(torch.float32)
        return float((rec - ref).norm() / ref.norm().clamp(min = 1e-12))
    except Exception:  # noqa: BLE001
        return None
