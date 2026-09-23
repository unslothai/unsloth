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
# Opt-in W8A8 for int8: quantise each activation row to int8 and run torch._int_mm on the stored int8 weight
# instead of dequantising the weight. "1" enables it; anything else keeps weight-only.
NATIVE_INT8_ACT_ENV = "UNSLOTH_NATIVE_INT8_ACT"
# torch._int_mm wants M > 16 and K, N multiples of 8; smaller or odd shapes keep the weight-only path.
_INT_MM_MIN_ROWS = 17


def int8_act_requested() -> bool:
    import os

    return os.environ.get(NATIVE_INT8_ACT_ENV, "").strip() == "1"


@lru_cache(maxsize = 1)
def native_linear_class():
    """``NativeWeightOnlyLinear``, defined on first use so this module imports torch-free."""
    import torch
    from torch import nn
    from torch.nn import functional as F

    class NativeWeightOnlyLinear(nn.Module):
        """A Linear whose weight is stored as int8 / fp8 with a per-output-row scale."""

        def __init__(self, linear: Any, scheme: str, act_int8: bool = False):
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
            self.act_int8 = bool(act_int8) and scheme == NATIVE_INT8

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

        def _int_mm_ok(self, rows: int) -> bool:
            return (
                rows >= _INT_MM_MIN_ROWS
                and self.in_features % 8 == 0
                and self.out_features % 8 == 0
            )

        def _forward_int_mm(self, x: Any) -> Any:
            """W8A8: per-row symmetric int8 activations x the stored int8 weight, int32 accumulate, one rescale."""
            x2 = x.reshape(-1, self.in_features)
            xf = x2.float()
            x_scale = xf.abs().amax(dim = 1, keepdim = True).clamp(min = 1e-12) / 127.0
            xq = (xf / x_scale).round_().clamp_(-127, 127).to(torch.int8)
            acc = torch._int_mm(xq, self.weight_q.t())
            out = acc.float() * x_scale * self.weight_scale.view(torch.float32)[None, :]
            out = out.to(x.dtype)
            if self.bias is not None:
                out = out + self.bias.to(x.dtype)
            return out.reshape(*x.shape[:-1], self.out_features)

        def forward(self, x: Any) -> Any:
            if self.act_int8 and x.is_cuda and self._int_mm_ok(x.numel() // max(1, self.in_features)):
                return self._forward_int_mm(x)
            return F.linear(x, self.dequantized_weight(x.dtype), self.bias)

        def extra_repr(self) -> str:
            return (
                f"in_features={self.in_features}, out_features={self.out_features}, "
                f"scheme={self.scheme}, act_int8={self.act_int8}, bias={self.bias is not None}"
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
    """Swap every Linear ``filter_fn`` keeps for a native weight-only twin; returns layers swapped.

    One layer at a time, like torchao's ``quantize_``, so the build peak is the dense model plus one
    layer rather than 1.5x it; on unified memory that peak is what the OS kills for. A failure part-way
    leaves a partial conversion, which ``is_native_quantised`` reports so the loader refuses it the
    same way it refuses a partial torchao pass."""
    cls = native_linear_class()
    act_int8 = scheme == NATIVE_INT8 and int8_act_requested()
    # Names only: holding the old modules would keep every bf16 weight alive until the pass ends.
    names = [name for name, mod in transformer.named_modules() if name and filter_fn(mod, name)]
    for name in names:
        parent_name, _, leaf = name.rpartition(".")
        parent = transformer.get_submodule(parent_name) if parent_name else transformer
        setattr(parent, leaf, cls(getattr(parent, leaf), scheme, act_int8 = act_int8))
    if logger is not None:
        logger.info(
            "diffusion.transformer_quant: %s %s (torchao-free) on %d linears",
            scheme,
            "W8A8 torch._int_mm" if act_int8 else "weight-only",
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
