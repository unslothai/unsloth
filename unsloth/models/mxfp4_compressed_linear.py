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

"""Train through compressed-tensors ``mxfp4-pack-quantized`` Linears without unpacking them.

Forward and backward dequantize the frozen weight per use (exact in bf16/fp16), so no 16-bit copy is kept.
Remote-code MoE experts are stacked into unsloth_zoo ``Mxfp4StackedExperts`` instead (see ``compressed_tensors_bnb``).
"""

import functools
import os
import types
from typing import Optional

import torch
from torch import nn

__all__ = [
    "Mxfp4PackedLinear",
    "mxfp4_keep_packed_enabled",
    "dequantize_mxfp4_packed",
    "is_mxfp4_scheme",
    "make_mxfp4_packed_linear",
    "adopt_compressed_mxfp4_modules",
    "stack_packed_expert_linears",
    "install_compressed_tensors_keep_packed",
    "patch_peft_merge_for_mxfp4_packed_linears",
    "patch_peft_inits_for_mxfp4_packed_linears",
]

MXFP4_KEEP_PACKED_ENV = "UNSLOTH_MXFP4_KEEP_PACKED"
MXFP4_FORMAT = "mxfp4-pack-quantized"

try:
    from unsloth_zoo.mxfp4_dequant import mxfp4_dequantize as _zoo_mxfp4_dequantize
except Exception:
    _zoo_mxfp4_dequantize = None

_FP4_VALUES = (
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
)  # fmt: skip
_LUT_CACHE = {}


def mxfp4_keep_packed_enabled() -> bool:
    return os.environ.get(MXFP4_KEEP_PACKED_ENV, "1").strip() != "0"


def _dequantize_torch(blocks, scales, dtype):
    key = (blocks.device, dtype)
    lut = _LUT_CACHE.get(key)
    if lut is None:
        lut = _LUT_CACHE[key] = torch.tensor(_FP4_VALUES, dtype = dtype, device = blocks.device)
    *prefix, G, B = blocks.shape
    out = torch.empty(*prefix, G, B * 2, dtype = dtype, device = blocks.device)
    out[..., 0::2] = lut[(blocks & 0x0F).long()]
    out[..., 1::2] = lut[(blocks >> 4).long()]
    out = torch.ldexp(out, (scales.to(torch.int32) - 127).unsqueeze(-1)).to(dtype)
    return out.reshape(*prefix, G * B * 2)


def dequantize_mxfp4_packed(
    packed,
    scale,
    dtype = torch.bfloat16,
):
    """Packed ``[out, in / 2]`` + E8M0 scales ``[out, in / 32]`` -> ``[out, in]``; zoo kernel if self-verified, else torch."""
    out_features, half = packed.shape
    blocks = packed.view(out_features, half // 16, 16)
    if _zoo_mxfp4_dequantize is not None and dtype in (torch.bfloat16, torch.float16):
        return _zoo_mxfp4_dequantize(blocks, scale, dtype = dtype)
    return _dequantize_torch(blocks, scale, dtype)


def _compute_dtype(x):
    device_type = x.device.type
    try:
        if torch.is_autocast_enabled(device_type):
            return torch.get_autocast_dtype(device_type)
    except Exception:
        pass
    return x.dtype


class _Mxfp4PackedLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, packed, scale, bias):
        dtype = _compute_dtype(x)
        weight = dequantize_mxfp4_packed(packed, scale, dtype)
        out = torch.nn.functional.linear(
            x.to(dtype),
            weight,
            None if bias is None else bias.to(dtype),
        )
        del weight
        ctx.save_for_backward(packed, scale)
        ctx.x_dtype = x.dtype
        return out

    @staticmethod
    def backward(ctx, grad_output):
        packed, scale = ctx.saved_tensors
        grad_x = grad_bias = None
        if ctx.needs_input_grad[0]:
            weight = dequantize_mxfp4_packed(packed, scale, grad_output.dtype)
            grad_x = torch.matmul(grad_output, weight).to(ctx.x_dtype)
            del weight
        if ctx.needs_input_grad[3]:
            grad_bias = grad_output.reshape(-1, grad_output.shape[-1]).sum(0)
        return grad_x, None, None, grad_bias


class Mxfp4PackedLinear(nn.Linear):
    # Read by kernels.utils.has_mxfp4_base without importing this.
    _unsloth_mxfp4_packed_linear = True
    compute_dtype = torch.bfloat16

    def forward(self, x):
        return _Mxfp4PackedLinearFunction.apply(x, self.weight_packed, self.weight_scale, self.bias)

    def dequantize_weight(self, dtype = None):
        return dequantize_mxfp4_packed(
            self.weight_packed, self.weight_scale, dtype or self.compute_dtype
        )

    @property
    def weight(self):
        # Fresh exact decode per read, never kept: writes are lost (merge densifies first).
        return self.dequantize_weight()

    def reset_parameters(self):
        return

    def _apply(self, fn, *args, **kwargs):
        # Casts leave the uint8 bytes alone; the decode follows the dtype.
        try:
            dtype = fn(torch.empty(0, dtype = self.compute_dtype)).dtype
        except Exception:
            dtype = self.compute_dtype
        result = super()._apply(fn, *args, **kwargs)
        if dtype.is_floating_point:
            self.compute_dtype = dtype
        return result

    def extra_repr(self):
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, format={MXFP4_FORMAT}"
        )


def is_mxfp4_scheme(scheme, default_format = None) -> bool:
    # Weight-only: activation-quantized schemes cannot run here.
    if any(
        getattr(scheme, f, None) is not None for f in ("input_activations", "output_activations")
    ):
        return False
    fmt = getattr(scheme, "format", None) or default_format
    fmt = getattr(fmt, "value", fmt)
    if fmt != MXFP4_FORMAT:
        return False
    weights = getattr(scheme, "weights", None)
    if weights is None:
        return True
    kind = getattr(weights, "type", "float")
    kind = str(getattr(kind, "value", kind)).lower()
    return (
        kind == "float"
        and int(getattr(weights, "num_bits", 4) or 4) == 4
        and int(getattr(weights, "group_size", 32) or 32) == 32
    )


def _set_compressed_attributes(module, scheme):
    module.quantization_scheme = scheme
    try:
        from compressed_tensors.quantization import QuantizationStatus
        module.quantization_status = QuantizationStatus.COMPRESSED
    except Exception:
        module.quantization_status = "compressed"


def make_mxfp4_packed_linear(
    in_features: int,
    out_features: int,
    bias: Optional[torch.Tensor] = None,
    scheme = None,
    device = None,
    dtype = None,
) -> Mxfp4PackedLinear:
    if in_features % 32 != 0:
        raise ValueError(
            f"Unsloth: an MXFP4 Linear needs in_features divisible by 32, got {in_features}."
        )
    module = Mxfp4PackedLinear.__new__(Mxfp4PackedLinear)
    nn.Module.__init__(module)
    module.in_features = in_features
    module.out_features = out_features
    module.weight_packed = nn.Parameter(
        torch.empty(out_features, in_features // 2, dtype = torch.uint8, device = device),
        requires_grad = False,
    )
    module.weight_scale = nn.Parameter(
        torch.empty(out_features, in_features // 32, dtype = torch.uint8, device = device),
        requires_grad = False,
    )
    if isinstance(bias, nn.Parameter):
        module.bias = bias
    else:
        module.register_parameter("bias", None)
    if dtype is not None and dtype.is_floating_point:
        module.compute_dtype = dtype
    _set_compressed_attributes(module, scheme)
    return module


def _drop_compressed_tensors_forward(module):
    """Drop compressed-tensors' per-instance forward; other instance wrappers (offload) stay."""
    forward = module.__dict__.get("forward")
    code = getattr(getattr(forward, "__func__", forward), "__code__", None)
    if code is not None and code.co_name == "quantized_forward":
        del module.__dict__["forward"]


def _restore_dense_linear(module, dtype):
    device = module._parameters["weight_packed"].device
    for name in [n for n in module._parameters if n.startswith("weight_")]:
        del module._parameters[name]
    module.weight = nn.Parameter(
        torch.empty(module.out_features, module.in_features, dtype = dtype, device = device),
        requires_grad = False,
    )
    for name in ("quantization_scheme", "quantization_status"):
        module.__dict__.pop(name, None)
    _drop_compressed_tensors_forward(module)


def adopt_compressed_mxfp4_modules(
    model,
    default_format = None,
    packed_names = None,
    dtype = None,
) -> int:
    """Adopt every MXFP4 Linear as ``Mxfp4PackedLinear`` and drop the decompress hook; all or nothing (0).
    Linears not in ``packed_names`` go dense: Kimi-K3 ships its latent MoE projections in bf16."""
    if not mxfp4_keep_packed_enabled():
        return 0
    candidates, others, dense = [], 0, []
    for name, module in model.named_modules():
        if isinstance(module, Mxfp4PackedLinear):
            continue
        packed = getattr(module, "_parameters", {}).get("weight_packed")
        if packed is None:
            # e.g. FP8 still needs the model-wide decompress hook.
            if getattr(module, "quantization_scheme", None) is not None:
                others += 1
            continue
        scale = module._parameters.get("weight_scale")
        scheme = getattr(module, "quantization_scheme", None)
        if not (
            type(module) is nn.Linear
            and "weight" not in module._parameters
            and packed.dtype == torch.uint8
            and scale is not None
            and scale.dtype == torch.uint8
            and is_mxfp4_scheme(scheme, default_format)
        ):
            others += 1
        elif packed_names is not None and name not in packed_names:
            dense.append(module)
        else:
            candidates.append(module)
    # The hook would decompress adopted modules too, so adopt only when it can be dropped.
    if others or not candidates:
        return 0
    for module in dense:
        _restore_dense_linear(module, dtype or torch.get_default_dtype())
    for module in candidates:
        module.__class__ = Mxfp4PackedLinear
        if dtype is not None and dtype.is_floating_point:
            module.compute_dtype = dtype
        _drop_compressed_tensors_forward(module)
    hook = getattr(model, "ct_decompress_hook", None)
    if hook is not None:
        hook.remove()
        try:
            delattr(model, "ct_decompress_hook")
        except AttributeError:
            pass
    return len(candidates)


def stack_packed_expert_linears(
    model,
    block_names,
    dtype = None,
) -> list:
    """Stack adopted per-expert Linears; gate_up is released before down is allocated (one-layer transient)."""
    from .compressed_tensors_bnb import _new_stacked_experts, _plain_expert_shape

    stacked = []
    for name in block_names:
        block = model.get_submodule(name)
        experts = block.experts
        if not isinstance(experts, nn.ModuleList) or not all(
            isinstance(getattr(e, w, None), Mxfp4PackedLinear)
            for e in experts
            for w in ("w1", "w2", "w3")
        ):
            continue
        dims = _plain_expert_shape(experts)
        if dims is None:
            continue
        hidden, inter = dims
        E, device = len(experts), experts[0].w1.weight_packed.device
        new = _new_stacked_experts(experts, dims, dtype or experts[0].w1.compute_dtype, "meta")
        empty = lambda *shape: torch.empty(shape, dtype = torch.uint8, device = device)  # noqa: E731
        for target, out_rows, in_features, projections in (
            (
                "gate_up",
                2 * inter,
                hidden,
                (("w1", slice(0, inter)), ("w3", slice(inter, 2 * inter))),
            ),
            ("down", hidden, inter, (("w2", slice(None)),)),
        ):
            stacked_blocks = empty(E, out_rows, in_features // 32, 16)
            stacked_scales = empty(E, out_rows, in_features // 32)
            with torch.no_grad():
                for e, expert in enumerate(experts):
                    for proj, rows in projections:
                        linear = getattr(expert, proj)
                        blocks = stacked_blocks[e, rows]
                        blocks.copy_(linear.weight_packed.view(blocks.shape))
                        stacked_scales[e, rows].copy_(linear.weight_scale)
                        del linear._parameters["weight_packed"], linear._parameters["weight_scale"]
            setattr(new, f"{target}_blocks", nn.Parameter(stacked_blocks, requires_grad = False))
            setattr(new, f"{target}_scales", nn.Parameter(stacked_scales, requires_grad = False))
        block.experts = new.finalize()
        stacked.append(name)
    return stacked


def _full_finetuning() -> bool:
    return os.environ.get("UNSLOTH_ENABLE_FULL_FINETUNING", "0") == "1"


def install_compressed_tensors_keep_packed() -> bool:
    """Stock compressed-tensors loader keeps MXFP4 packed (adopt, then stack experts) under the same
    all-or-nothing plan; declined, full finetuning or explicit dequantize = stock. Idempotent."""
    try:
        from transformers.quantizers.quantizer_compressed_tensors import (
            CompressedTensorsHfQuantizer,
        )
    except Exception:
        return False
    if "_unsloth_keep_packed" in CompressedTensorsHfQuantizer.__dict__:
        return True
    original_before = CompressedTensorsHfQuantizer._process_model_before_weight_loading
    original_after = CompressedTensorsHfQuantizer._process_model_after_weight_loading

    def _process_model_before_weight_loading(self, model, **kwargs):
        result = original_before(self, model, **kwargs)
        self._unsloth_mxfp4_blocks = []
        config = getattr(self, "quantization_config", None)
        # `dequantize=True` (transformers 5.5+) or `run_compressed=False` (older).
        if (
            not mxfp4_keep_packed_enabled()
            or _full_finetuning()
            or getattr(config, "dequantize", False)
            or getattr(self, "run_compressed", True) is False
        ):
            return result
        fmt = getattr(config, "format", None) or getattr(
            getattr(config, "quantization_config", None), "format", None
        )
        try:
            from .compressed_tensors_bnb import (
                _checkpoint_keys,
                keep_mxfp4_experts_packed,
                plan_mxfp4_keep_packed,
            )

            # Skip planning non-MXFP4 checkpoints: key matching cost ~1 s on INT4 Kimi-K2.7.
            if not keep_mxfp4_experts_packed(config.to_dict()):
                return result
            keys = _checkpoint_keys(kwargs.get("checkpoint_files"))
            plan = plan_mxfp4_keep_packed(model, keys) if keys else None
        except Exception:
            plan = None
        if plan is None:
            return result
        packed_names = set(plan.linears) | {
            f"{name}.experts.{index}.{proj}"
            for name, block, _, _ in plan.blocks
            for index in range(len(block.experts))
            for proj in ("w1", "w2", "w3")
        }
        # transformers 5.x does not pass the load dtype here.
        dtype = kwargs.get("dtype")
        if not isinstance(dtype, torch.dtype):
            dtype = getattr(model, "dtype", None)
        adopted = adopt_compressed_mxfp4_modules(
            model,
            default_format = fmt,
            packed_names = packed_names,
            dtype = dtype if isinstance(dtype, torch.dtype) else None,
        )
        if adopted:
            self._unsloth_mxfp4_blocks = [name for name, _, _, _ in plan.blocks]
        return result

    def _process_model_after_weight_loading(self, model, **kwargs):
        result = original_after(self, model, **kwargs)
        blocks = getattr(self, "_unsloth_mxfp4_blocks", None)
        if blocks:
            self._unsloth_mxfp4_blocks = []
            stacked = stack_packed_expert_linears(model, blocks)
            print(
                f"Unsloth: Kept the MXFP4 weights packed ({len(stacked)} expert stacks); "
                "they are dequantized on the fly."
            )
        return result

    CompressedTensorsHfQuantizer._process_model_before_weight_loading = (
        _process_model_before_weight_loading
    )
    CompressedTensorsHfQuantizer._process_model_after_weight_loading = (
        _process_model_after_weight_loading
    )
    CompressedTensorsHfQuantizer._unsloth_keep_packed = True
    return True


# PEFT merge on a packed base: `weight.data += delta` would hit a throwaway decode, silently a no-op.
# Merge densifies to an exact-decode nn.Linear (bytes kept aside); unmerging the last adapter restores them.

_PACKED_STATE = "_unsloth_mxfp4_packed_state"


def _swap_class(module, cls):
    # accelerate keeps the wrapped forward as `_old_forward`; it must follow the class.
    old = module.__dict__.get("_old_forward")
    hooked = old is not None and getattr(old, "__func__", None) is type(module).forward
    module.__class__ = cls
    if hooked:
        module._old_forward = types.MethodType(cls.forward, module)


def _densify(module):
    weight = module.dequantize_weight()
    packed = module._parameters.pop("weight_packed")
    scale = module._parameters.pop("weight_scale")
    module.__dict__[_PACKED_STATE] = (packed, scale, module.compute_dtype)
    _swap_class(module, nn.Linear)
    module.weight = nn.Parameter(weight, requires_grad = False)


def _restore_packed(module):
    packed, scale, _ = module.__dict__.pop(_PACKED_STATE)
    device, dtype = module.weight.device, module.weight.dtype
    del module._parameters["weight"]
    _swap_class(module, Mxfp4PackedLinear)
    module.compute_dtype = dtype
    for name, param in (("weight_packed", packed), ("weight_scale", scale)):
        if param.device != device:
            param = nn.Parameter(param.data.to(device), requires_grad = False)
        module.register_parameter(name, param)


def patch_peft_merge_for_mxfp4_packed_linears() -> bool:
    try:
        from peft.tuners.lora.layer import Linear as LoraLinear
    except Exception:
        return False
    if getattr(LoraLinear.merge, "_unsloth_mxfp4_patched", False):
        return True
    original_merge = LoraLinear.merge
    original_unmerge = LoraLinear.unmerge

    def merge(self, *args, **kwargs):
        base = self.get_base_layer()
        if not isinstance(base, Mxfp4PackedLinear):
            return original_merge(self, *args, **kwargs)
        _densify(base)
        try:
            return original_merge(self, *args, **kwargs)
        finally:
            if not self.merged_adapters:
                _restore_packed(base)

    def unmerge(self, *args, **kwargs):
        result = original_unmerge(self, *args, **kwargs)
        base = self.get_base_layer()
        if _PACKED_STATE in base.__dict__ and not self.merged_adapters:
            _restore_packed(base)
        return result

    merge._unsloth_mxfp4_patched = True
    unmerge._unsloth_mxfp4_patched = True
    LoraLinear.merge = merge
    LoraLinear.unmerge = unmerge
    return True


# These inits subtract the adapter from `weight.data`, a throwaway decode on a packed base: silent model change.
_BASE_WEIGHT_INITS = ("pissa_init", "olora_init", "corda_init", "loftq_init", "lora_ga_init")


def _refuse_on_packed_base(original):
    @functools.wraps(original)
    def init(self, *args, **kwargs):
        if isinstance(self.get_base_layer(), Mxfp4PackedLinear):
            raise NotImplementedError(
                f"Unsloth: `{original.__name__}` rewrites the base weight, which stays packed in "
                "MXFP4 here and cannot be written. Use the default LoRA initialisation."
            )
        return original(self, *args, **kwargs)

    init._unsloth_mxfp4_patched = True
    return init


def patch_peft_inits_for_mxfp4_packed_linears() -> bool:
    try:
        from peft.tuners.lora.layer import LoraLayer
    except Exception:
        return False
    for name in _BASE_WEIGHT_INITS:
        original = getattr(LoraLayer, name, None)
        if original is not None and not getattr(original, "_unsloth_mxfp4_patched", False):
            setattr(LoraLayer, name, _refuse_on_packed_base(original))
    return True


patch_peft_merge_for_mxfp4_packed_linears()
patch_peft_inits_for_mxfp4_packed_linears()
