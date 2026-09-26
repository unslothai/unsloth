# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Keep compressed-tensors ``pack-quantized`` INT4/INT8 Linears packed for LoRA training.

The checkpoint's own ``weight_packed`` / ``weight_scale`` / ``weight_zero_point`` / ``weight_g_idx``
load straight into ``Int4PackedLinear`` (no decompress, no re-quantization). ``.weight`` is the packed
int32 tensor carrying an ``Int4QuantState``, so Unsloth's LoRA kernels treat it like a bitsandbytes
weight and decode it exactly per use (kernels/int4_packed.py). ``UNSLOTH_COMPRESSED_TENSORS_INT4=nf4``
selects the older NF4 re-quantization instead.
"""

import functools
import os
import types

import torch
from torch import nn

__all__ = [
    "Int4PackedLinear",
    "int4_packed_route_enabled",
    "int4_packed_supported_plan",
    "make_int4_packed_linear",
    "finalize_int4_packed_linears",
    "INT4_ROUTE_ENV",
]

INT4_ROUTE_ENV = "UNSLOTH_COMPRESSED_TENSORS_INT4"
_PACKED_BITS = (2, 4, 8)  # 32 // bits must be a power of two for the kernels.


def int4_packed_route_enabled() -> bool:
    route = os.environ.get(INT4_ROUTE_ENV, "packed").strip().lower()
    if route in ("nf4", "bnb", "bitsandbytes", "0", "off"):
        return False
    if not torch.cuda.is_available() or getattr(torch.version, "hip", None):
        return False
    try:
        import triton  # noqa: F401
    except Exception:
        return False
    return True


def int4_packed_supported_plan(plan) -> bool:
    """Every config group is a group / channel INT scheme the kernels decode exactly."""
    groups = (plan or {}).get("config_groups") or {}
    if not groups:
        return False
    for group in groups.values():
        weights = group.get("weights") or {}
        if int(weights.get("num_bits", 0)) not in _PACKED_BITS:
            return False
        strategy = str(weights.get("strategy", "group")).lower()
        if strategy not in ("group", "channel"):
            return False
        if strategy == "group" and not int(weights.get("group_size") or 0) > 0:
            return False
    return True


class _Int4LinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, packed, module, bias):
        from ..kernels.int4_packed import int4_matmul

        qs = module.quant_state
        out = int4_matmul(x.to(qs.dtype), packed, qs)
        if bias is not None:
            out = out + bias.to(out.dtype)
        ctx.module = module
        ctx.x_dtype = x.dtype
        ctx.save_for_backward(packed)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        from ..kernels.int4_packed import int4_matmul_t

        (packed,) = ctx.saved_tensors
        grad_x = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_x = int4_matmul_t(grad_output, packed, ctx.module.quant_state).to(ctx.x_dtype)
        if ctx.needs_input_grad[3]:
            grad_bias = grad_output.reshape(-1, grad_output.shape[-1]).sum(0)
        return grad_x, None, None, grad_bias


class Int4PackedLinear(nn.Linear):
    """``nn.Linear`` over a compressed-tensors packed integer weight (frozen)."""

    _unsloth_int4_packed_linear = True

    @property
    def quant_state(self):
        from ..kernels.int4_packed import Int4QuantState

        packed = self._parameters["weight_packed"]
        qs = self.__dict__.get("_int4_quant_state")
        scale = self._parameters["weight_scale"]
        zero = self._parameters.get("weight_zero_point")
        g_idx = self._parameters.get("weight_g_idx")
        if (
            qs is None
            or qs.scale is not scale
            or qs.zero_point is not zero
            or (g_idx is not None and qs.g_idx is not g_idx)
        ):
            qs = Int4QuantState(
                scale,
                zero,
                g_idx,
                (self.out_features, self.in_features),
                self._int4_bits,
                self._int4_group_size,
                self.__dict__.get("_int4_dtype") or torch.bfloat16,
            )
            self.__dict__["_int4_quant_state"] = qs
            packed.quant_state = qs
        elif getattr(packed, "quant_state", None) is not qs:
            packed.quant_state = qs
        return qs

    @property
    def weight(self):
        # The packed int32 words with `.quant_state`: the bitsandbytes contract Unsloth's kernels read.
        packed = self._parameters["weight_packed"]
        if packed.__dict__.get("quant_state") is None:
            self.quant_state
        return packed

    def forward(self, x):
        return _Int4LinearFunction.apply(x, self._parameters["weight_packed"], self, self.bias)

    def dequantize_weight(self, dtype = None):
        from ..kernels.int4_packed import int4_dequantize

        qs = self.quant_state
        return int4_dequantize(self._parameters["weight_packed"], qs, dtype or qs.dtype)

    def reset_parameters(self):
        return

    def _apply(self, fn, *args, **kwargs):
        # Casts never touch the packed words, scales or zero points (a fp32 scale must stay exact);
        # only moves apply to them, and the compute dtype follows a floating cast.
        try:
            probe = fn(torch.empty(0, dtype = torch.bfloat16))
            dtype, device = probe.dtype, probe.device
        except Exception:
            dtype = device = None
        frozen = {n: self._parameters.pop(n) for n in list(self._parameters) if n.startswith("weight_")}
        try:
            result = super()._apply(fn, *args, **kwargs)
        finally:
            for name, param in frozen.items():
                if param is not None and param.device.type != "meta":
                    target = fn(torch.empty(0, device = param.device)).device
                    if target != param.device:
                        param.data = param.data.to(target)
                self._parameters[name] = param
        if dtype is not None and dtype in (torch.float16, torch.bfloat16, torch.float32):
            self.__dict__["_int4_dtype"] = dtype
        self.__dict__.pop("_int4_quant_state", None)
        packed = self._parameters.get("weight_packed")
        if packed is not None:
            packed.__dict__.pop("quant_state", None)
        return result

    def extra_repr(self):
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, bits={self._int4_bits}, group_size={self._int4_group_size}"
        )


def make_int4_packed_linear(module, scheme, shapes, dtype, device = "meta"):
    """``Int4PackedLinear`` shell for ``module`` whose parameters take the checkpoint tensors as-is.

    ``shapes`` maps suffix (``weight_packed`` ...) to ``(shape, torch dtype)`` read from the header."""
    new = Int4PackedLinear.__new__(Int4PackedLinear)
    nn.Module.__init__(new)
    new.in_features = module.in_features
    new.out_features = module.out_features
    weights = scheme.weights
    new._int4_bits = int(weights.num_bits)
    group_size = weights.group_size
    new._int4_group_size = int(group_size) if group_size else module.in_features
    new.__dict__["_int4_dtype"] = dtype
    for suffix in ("weight_packed", "weight_scale", "weight_zero_point", "weight_g_idx"):
        if suffix not in shapes:
            continue
        shape, tdtype = shapes[suffix]
        new.register_parameter(
            suffix,
            nn.Parameter(torch.empty(shape, dtype = tdtype, device = device), requires_grad = False),
        )
    if "weight_shape" in shapes:
        shape, tdtype = shapes["weight_shape"]
        new.register_buffer("weight_shape", torch.empty(shape, dtype = tdtype, device = device))
    bias = getattr(module, "bias", None)
    if isinstance(bias, nn.Parameter):
        new.bias = bias
    else:
        new.register_parameter("bias", None)
    new.quantization_scheme = scheme
    return new


def finalize_int4_packed_linears(model, dtype = None) -> int:
    count = 0
    for module in model.modules():
        if isinstance(module, Int4PackedLinear):
            for name, param in module._parameters.items():
                if name.startswith("weight_") and param is not None:
                    param.requires_grad_(False)
            if isinstance(dtype, torch.dtype) and dtype.is_floating_point:
                module.__dict__["_int4_dtype"] = dtype
            module.__dict__.pop("_int4_quant_state", None)
            module._parameters["weight_packed"].__dict__.pop("quant_state", None)
            module.quant_state
            count += 1
    return count


# PEFT merge on a packed base would write into a throwaway view: densify first, restore after.
_PACKED_STATE = "_unsloth_int4_packed_state"


def _swap_class(module, cls):
    old = module.__dict__.get("_old_forward")
    hooked = old is not None and getattr(old, "__func__", None) is type(module).forward
    module.__class__ = cls
    if hooked:
        module._old_forward = types.MethodType(cls.forward, module)


def _densify(module):
    weight = module.dequantize_weight()
    saved = {
        name: module._parameters.pop(name)
        for name in list(module._parameters)
        if name.startswith("weight_")
    }
    module.__dict__[_PACKED_STATE] = saved
    _swap_class(module, nn.Linear)
    module.weight = nn.Parameter(weight, requires_grad = False)


def _restore_packed(module):
    saved = module.__dict__.pop(_PACKED_STATE)
    del module._parameters["weight"]
    _swap_class(module, Int4PackedLinear)
    for name, param in saved.items():
        module.register_parameter(name, param)
    module.__dict__.pop("_int4_quant_state", None)


def patch_peft_merge_for_int4_packed_linears() -> bool:
    try:
        from peft.tuners.lora.layer import Linear as LoraLinear
    except Exception:
        return False
    if getattr(LoraLinear.merge, "_unsloth_int4_patched", False):
        return True
    original_merge = LoraLinear.merge
    original_unmerge = LoraLinear.unmerge

    @functools.wraps(original_merge)
    def merge(self, *args, **kwargs):
        base = self.get_base_layer()
        if not isinstance(base, Int4PackedLinear):
            return original_merge(self, *args, **kwargs)
        _densify(base)
        try:
            return original_merge(self, *args, **kwargs)
        finally:
            if not self.merged_adapters:
                _restore_packed(base)

    @functools.wraps(original_unmerge)
    def unmerge(self, *args, **kwargs):
        result = original_unmerge(self, *args, **kwargs)
        base = self.get_base_layer()
        if _PACKED_STATE in base.__dict__ and not self.merged_adapters:
            # The merged delta was subtracted again; the exact packed weights come back.
            _restore_packed(base)
        return result

    merge._unsloth_int4_patched = True
    unmerge._unsloth_int4_patched = True
    LoraLinear.merge = merge
    LoraLinear.unmerge = unmerge
    return True


patch_peft_merge_for_int4_packed_linears()
