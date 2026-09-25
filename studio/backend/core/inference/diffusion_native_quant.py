# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Torchao-free weight-only int8 / fp8 for the diffusion transformer (ROCm, Windows torchao stub).

Plain buffers, unlike torchao tensors, survive the offload hooks' ``Module.to()``; torch imported lazily.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Callable, Optional

NATIVE_INT8 = "int8"
NATIVE_FP8 = "fp8"
_QMAX = {NATIVE_INT8: 127.0, NATIVE_FP8: 448.0}


@lru_cache(maxsize = 1)
def native_linear_class():
    """``NativeWeightOnlyLinear``, defined on first use so this module imports torch-free."""
    import torch
    from torch import nn
    from torch.nn import functional as F

    class NativeWeightOnlyLinear(nn.Module):
        """A Linear whose weight is stored as int8 / fp8 with a per-output-row scale."""

        def __init__(self, linear: Any, scheme: str):
            super().__init__()
            if scheme not in _QMAX:
                raise ValueError(f"unsupported native scheme {scheme!r}")
            self.in_features = int(linear.in_features)
            self.out_features = int(linear.out_features)
            self.scheme = scheme
            self.compute_dtype = linear.weight.dtype
            weight = linear.weight.detach()
            with torch.no_grad():
                w = weight.float()
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

        def dequantized_weight(self, dtype: Any) -> Any:
            wq = (
                self.weight_q.view(torch.float8_e4m3fn)
                if self.scheme == NATIVE_FP8
                else self.weight_q
            )
            scale = self.weight_scale.view(torch.float32)
            return (wq.to(torch.float32) * scale[:, None]).to(dtype)

        @property
        def weight(self) -> Any:
            """The dense weight, rebuilt on each read. Code that reads a wrapped Linear's weight directly
            (PEFT's DoRA forward reads ``base_layer.weight``) keeps working; nothing holds it."""
            return self.dequantized_weight(self.compute_dtype)

        def forward(self, x: Any) -> Any:
            return F.linear(x, self.dequantized_weight(x.dtype), self.bias)

        def extra_repr(self) -> str:
            return (
                f"in_features={self.in_features}, out_features={self.out_features}, "
                f"scheme={self.scheme}, bias={self.bias is not None}"
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
    logger: Any = None,
) -> int:
    """Swap every Linear ``filter_fn`` keeps for a native weight-only twin; returns layers swapped. One layer
    at a time; a failure part-way leaves a partial conversion the caller must refuse."""
    cls = native_linear_class()
    names = [name for name, mod in transformer.named_modules() if name and filter_fn(mod, name)]
    for name in names:
        parent_name, _, leaf = name.rpartition(".")
        parent = transformer.get_submodule(parent_name) if parent_name else transformer
        setattr(parent, leaf, cls(getattr(parent, leaf), scheme))
    if logger is not None:
        logger.info(
            "diffusion.transformer_quant: %s weight-only (torchao-free) on %d linears",
            scheme,
            len(names),
        )
    return len(names)


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
