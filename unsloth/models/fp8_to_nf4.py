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
"""Load a block-FP8 checkpoint straight into bitsandbytes NF4 (``load_in_4bit = True``).

The fp8 block is parked off the config so transformers runs a bnb 4bit load; each FP8 tensor is
dequantized inside its weight converter right before bnb quantizes it, so only one converter's
tensors are ever 16-bit and NF4 is bit-identical to loading the dequantized checkpoint. The
checkpoint's ``modules_to_not_convert`` stay out of bitsandbytes.
"""

import contextvars
import copy
import functools
import inspect
import os
import re
from typing import Optional

import torch

__all__ = [
    "UNSLOTH_FP8_TO_NF4_ATTR",
    "track_explicit_4bit_request",
    "requests_bnb_4bit",
    "explicit_4bit_requested",
    "fp8_block_quantization_config",
    "fp8_to_nf4_unavailable_reason",
    "maybe_arm_fp8_to_nf4",
    "fp8_to_nf4_armed",
    "fp8_to_nf4_disabled",
    "disarm_fp8_to_nf4",
    "fp8_to_nf4_planner_quantization_config",
]

UNSLOTH_FP8_TO_NF4_ATTR = "_unsloth_fp8_to_nf4"
_ENV = "UNSLOTH_FP8_TO_NF4"
_ENV_QUANTIZE_ALL = "UNSLOTH_FP8_TO_NF4_QUANTIZE_16BIT"

# None outside a load; True / False once the outermost from_pretrained has looked at its call.
_EXPLICIT_4BIT = contextvars.ContextVar("unsloth_explicit_load_in_4bit", default = None)
# Configs armed inside the outermost from_pretrained; disarmed when it returns or raises, so an
# exception before the load's own restore (prefetch, device-map planning) cannot leak a stripped config.
_ARMED_CONFIGS = contextvars.ContextVar("unsloth_fp8_to_nf4_armed_configs", default = None)

_FP8_DTYPES = tuple(
    getattr(torch, name)
    for name in ("float8_e4m3fn", "float8_e5m2", "float8_e4m3fnuz", "float8_e5m2fnuz")
    if hasattr(torch, name)
)
_FP8_SAFETENSORS = ("F8_E4M3", "F8_E5M2", "F8_E4M3FNUZ", "F8_E5M2FNUZ")
_SCALE_SUFFIXES = (".weight_scale_inv", ".activation_scale")
# Checkpoint dtypes a scaled weight may be stored in; anything else (int8-packed FP4) is refused.
_FLOAT_SAFETENSORS = _FP8_SAFETENSORS + ("BF16", "F16", "F32")
_PACKED_EXPERT_DTYPES = ("fp4", "mxfp4", "nvfp4", "int4")


def requests_bnb_4bit(quantization_config) -> bool:
    if quantization_config is None:
        return False
    if isinstance(quantization_config, dict):
        get = quantization_config.get
    else:
        get = lambda key, default = None: getattr(quantization_config, key, default)
    method = get("quant_method", "") or ""
    method = str(getattr(method, "value", method)).lower()
    if method and "bitsandbytes" not in method:
        return False
    return bool(get("load_in_4bit", False))


def track_explicit_4bit_request(fn):
    """Record whether the caller itself asked for 4bit: ``load_in_4bit`` defaults to True, so only a
    passed argument is a request (a user ``quantization_config`` clears the loader's 4bit flag, so it
    never arms). The outermost from_pretrained decides; the FastModel delegations inside it pass the
    flag on explicitly."""
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        signature = None

    @functools.wraps(fn)
    def _wrapper(*args, **kwargs):
        if _EXPLICIT_4BIT.get() is not None:
            return fn(*args, **kwargs)
        explicit = False
        try:
            if signature is not None:
                arguments = signature.bind_partial(*args, **kwargs).arguments
                explicit = bool(arguments.get("load_in_4bit", False))
        except TypeError:
            explicit = bool(kwargs.get("load_in_4bit", False))
        explicit = explicit or requests_bnb_4bit(kwargs.get("quantization_config", None))
        token = _EXPLICIT_4BIT.set(explicit)
        armed_token = _ARMED_CONFIGS.set([])
        try:
            return fn(*args, **kwargs)
        finally:
            for armed in _ARMED_CONFIGS.get() or []:
                disarm_fp8_to_nf4(armed)
            _ARMED_CONFIGS.reset(armed_token)
            _EXPLICIT_4BIT.reset(token)

    return _wrapper


def explicit_4bit_requested() -> bool:
    return bool(_EXPLICIT_4BIT.get())


def _quant_dict(quant) -> Optional[dict]:
    if quant is None:
        return None
    if isinstance(quant, dict):
        return quant
    to_dict = getattr(quant, "to_dict", None)
    if callable(to_dict):
        try:
            return to_dict()
        except Exception:
            return None
    return None


def _configs_with_quantization(config) -> list:
    """The config objects holding a quantization_config: the root, else its text config
    (transformers' get_hf_quantizer reads both)."""
    found = []
    if config is None:
        return found
    if getattr(config, "quantization_config", None) is not None:
        found.append(config)
    get_text_config = getattr(config, "get_text_config", None)
    if callable(get_text_config):
        try:
            text_config = get_text_config(decoder = True)
        except TypeError:
            text_config = get_text_config()
        except Exception:
            text_config = None
        if (
            text_config is not None
            and text_config is not config
            and getattr(text_config, "quantization_config", None) is not None
        ):
            found.append(text_config)
    return found


def _block_size(quant: dict):
    block = quant.get("weight_block_size")
    if isinstance(block, (int, float)):
        block = [block, block]
    if not isinstance(block, (list, tuple)) or len(block) != 2:
        return None
    try:
        block = [int(block[0]), int(block[1])]
    except (TypeError, ValueError):
        return None
    if block[0] <= 0 or block[1] <= 0:
        return None
    return block


def fp8_block_quantization_config(config) -> Optional[dict]:
    """The checkpoint's fine-grained (block) fp8 quantization_config as a dict, else None."""
    for holder in _configs_with_quantization(config):
        quant = _quant_dict(getattr(holder, "quantization_config", None))
        if not quant or str(quant.get("quant_method", "")).lower() != "fp8":
            continue
        if _block_size(quant) is None:
            return None
        return quant
    return None


def _transformers_supports_fp8_to_nf4() -> Optional[str]:
    try:
        from transformers import core_model_loading as cml
        from transformers.conversion_mapping import get_model_conversion_mapping  # noqa: F401
        from transformers.quantizers import auto as quantizers_auto
        from transformers.quantizers.quantizer_bnb_4bit import Bnb4BitHfQuantizer
    except Exception:
        return "this transformers has no weight-conversion loader (transformers >= 5.16 is needed)"
    for name in (
        "WeightConverter",
        "WeightRenaming",
        "ConversionOps",
        "rename_source_key",
        "spawn_materialize",
    ):
        if not hasattr(cml, name):
            return f"transformers.core_model_loading has no `{name}`"
    if not hasattr(cml.WeightTransform, "add_tensor"):
        return "transformers' WeightTransform has no add_tensor"
    if not isinstance(getattr(quantizers_auto, "AUTO_QUANTIZER_MAPPING", None), dict):
        return "transformers has no AUTO_QUANTIZER_MAPPING"
    # Converters are rebuilt with these (5.5 to 5.16 added them one by one).
    if not hasattr(cml, "_IdentityOp"):
        return "transformers.core_model_loading has no `_IdentityOp`"
    if not callable(getattr(Bnb4BitHfQuantizer, "update_weight_conversions", None)):
        return "transformers' quantizers have no update_weight_conversions hook"
    # Slots, not the signature: Unsloth wraps WeightConverter.__init__ as (*args, **kwargs).
    converter_fields = set(getattr(cml.WeightConverter, "__slots__", ())) | set(
        getattr(cml.WeightConverter, "__dataclass_fields__", {})
    )
    if "force_cpu" not in converter_fields:
        return "transformers' WeightConverter predates force_cpu (transformers >= 5.16 is needed)"
    return None


def fp8_to_nf4_unavailable_reason() -> Optional[str]:
    reason = _transformers_supports_fp8_to_nf4()
    if reason is not None:
        return reason
    try:
        import bitsandbytes  # noqa: F401
    except Exception:
        return "bitsandbytes is not installed"
    return None


def fp8_to_nf4_disabled() -> bool:
    return os.environ.get(_ENV, "").strip().lower() in ("0", "false", "off", "no")


def fp8_to_nf4_armed(config) -> bool:
    return config is not None and getattr(config, UNSLOTH_FP8_TO_NF4_ATTR, None) is not None


def maybe_arm_fp8_to_nf4(
    config,
    load_in_4bit,
    load_in_8bit,
    verbose = True,
) -> bool:
    """Arm an fp8 -> NF4 load. True means: keep load_in_4bit, the fp8 block was parked on `config`
    and removed from it, and the load must pass this `config` to from_pretrained."""
    if not load_in_4bit or load_in_8bit or config is None:
        return False
    quant = fp8_block_quantization_config(config)
    if quant is None:
        return False
    if fp8_to_nf4_disabled():
        return False
    if str(quant.get("expert_dtype") or "").lower() in _PACKED_EXPERT_DTYPES:
        # DeepSeek-V4-Flash: fp8 attention, packed FP4 experts. Not handled here.
        if verbose and explicit_4bit_requested():
            print(
                "Unsloth: this fp8 checkpoint packs its experts in FP4, which the fp8 -> 4bit load does "
                "not handle. Loading it as it is."
            )
        return False
    mode = os.environ.get(_ENV, "").strip().lower()
    if not (explicit_4bit_requested() or mode in ("1", "true", "on", "yes", "force")):
        return False
    reason = fp8_to_nf4_unavailable_reason()
    if reason is not None:
        if verbose:
            print(
                f"Unsloth: `load_in_4bit = True` on an fp8 checkpoint cannot quantize to 4bit here: "
                f"{reason}. Loading the fp8 weights instead."
            )
        return False
    if quant.get("modules_to_convert"):
        try:
            from transformers.integrations.finegrained_fp8 import (  # noqa: F401
                FP8Embedding,
                replace_with_fp8_embedding,
            )
        except Exception:
            if verbose:
                print(
                    "Unsloth: this fp8 checkpoint keeps fp8 embedding tables (modules_to_convert), which "
                    "need transformers >= 5.17 to load in 4bit. Loading the fp8 weights instead."
                )
            return False
    if not install_fp8_to_nf4_quantizer():
        if verbose:
            print(
                "Unsloth: could not register the fp8 -> 4bit loader with transformers. Loading the "
                "fp8 weights instead."
            )
        return False
    plan = {
        "original_quantization_config": dict(quant),
        "weight_block_size": _block_size(quant),
        "fmt": quant.get("fmt", "e4m3"),
        "activation_scheme": quant.get("activation_scheme"),
        "modules_to_not_convert": list(quant.get("modules_to_not_convert") or []),
        "modules_to_convert": list(quant.get("modules_to_convert") or []),
    }
    holders = _configs_with_quantization(config)
    try:
        stripped = []
        for holder in holders:
            if (
                str((_quant_dict(holder.quantization_config) or {}).get("quant_method", "")).lower()
                == "fp8"
            ):
                stripped.append(holder is not config)
                try:
                    delattr(holder, "quantization_config")
                except AttributeError:
                    holder.__dict__.pop("quantization_config", None)
        plan["stripped_text_config"] = any(stripped)
        plan["stripped_root_config"] = not all(stripped) if stripped else False
        setattr(config, UNSLOTH_FP8_TO_NF4_ATTR, plan)
        armed_configs = _ARMED_CONFIGS.get()
        if armed_configs is not None:
            armed_configs.append(config)
    except Exception:
        for holder in holders:
            if getattr(holder, "quantization_config", None) is None:
                holder.quantization_config = quant
        return False
    if verbose:
        print(
            "Unsloth: fp8 checkpoint with `load_in_4bit = True`: dequantizing each fp8 tensor to 16bit "
            "while loading and quantizing it to 4bit (NF4) right away, so no 16bit copy of the model is "
            f"ever held. Set {_ENV}=0 to load the fp8 weights instead."
        )
    return True


def disarm_fp8_to_nf4(config) -> bool:
    """Undo maybe_arm_fp8_to_nf4 when the load ends up not quantizing (e.g. full finetuning)."""
    global _STATE
    plan = getattr(config, UNSLOTH_FP8_TO_NF4_ATTR, None) if config is not None else None
    if plan is None:
        return False
    # A load that raised never reached _process_model_after_weight_loading.
    _STATE = None
    original = plan["original_quantization_config"]
    if plan.get("stripped_root_config"):
        config.quantization_config = dict(original)
    if plan.get("stripped_text_config"):
        get_text_config = getattr(config, "get_text_config", None)
        text_config = None
        if callable(get_text_config):
            try:
                text_config = get_text_config(decoder = True)
            except TypeError:
                text_config = get_text_config()
        if text_config is not None:
            text_config.quantization_config = dict(original)
    try:
        delattr(config, UNSLOTH_FP8_TO_NF4_ATTR)
    except AttributeError:
        config.__dict__.pop(UNSLOTH_FP8_TO_NF4_ATTR, None)
    return True


def _quantize_16bit_requested() -> bool:
    return os.environ.get(_ENV_QUANTIZE_ALL, "0").strip().lower() in ("1", "true", "on", "yes")


def fp8_to_nf4_planner_quantization_config(config, llm_int8_skip_modules = None) -> Optional[dict]:
    """What the device-map planner must size an armed load as: a bitsandbytes 4bit load."""
    plan = getattr(config, UNSLOTH_FP8_TO_NF4_ATTR, None) if config is not None else None
    if plan is None:
        return None
    skip = list(llm_int8_skip_modules or [])
    # Quantize-all mode quantizes the checkpoint's 16-bit Linears too: size them as 4bit.
    for name in [] if _quantize_16bit_requested() else plan.get("modules_to_not_convert") or []:
        if name not in skip:
            skip.append(name)
    return {
        "quant_method": "bitsandbytes",
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
        "llm_int8_skip_modules": skip,
    }


class _LoadState:
    """Book-keeping for the one fp8 -> NF4 load in flight. Module level, never on an op or a
    converter: the loader deep-copies those once per target key."""

    def __init__(self, plan, dtype, headers):
        self.block = plan["weight_block_size"]
        self.dtype = dtype
        self.fp8_keys = {k for k, v in headers.items() if v in _FP8_SAFETENSORS}
        # `<module>.scale` is DeepSeek-V4's name for weight_scale_inv (renamed by the fp8 quantizer too).
        self.dot_scale_keys = {
            k
            for k in headers
            if k.endswith(".scale") and (k[: -len(".scale")] + ".weight") in self.fp8_keys
        }
        # Activation scales (static fp8) are not needed for a weight dequant, so they are not tracked.
        self.scale_keys = {
            k for k in headers if k.endswith(".weight_scale_inv")
        } | self.dot_scale_keys
        self.native_keys = self.fp8_keys | self.scale_keys
        # Refined once renaming is known: dropped MTP layers never reach a converter.
        self.expected_fp8 = set(self.fp8_keys)
        self.expected_scales = set(self.scale_keys)
        self.headers = headers
        self.routed_native = set()
        self.dequantized = 0
        self.kept_fp8 = 0


_STATE: Optional[_LoadState] = None


def _is_scale_pattern(pattern: str) -> bool:
    base = pattern[:-1] if pattern.endswith("$") else pattern
    return base.endswith(("weight_scale_inv", "activation_scale"))


def _scale_pattern_for(pattern: str) -> str:
    anchored = pattern.endswith("$")
    base = pattern[:-1] if anchored else pattern
    if base.endswith(".weight"):
        scale = base[: -len(".weight")] + ".weight_scale_inv"
    elif base == "weight":
        scale = "weight_scale_inv"
    else:
        scale = base + "_scale_inv"
    return scale + "$" if anchored else scale


def _scale_as_fp32(scale: torch.Tensor) -> torch.Tensor:
    if scale.dtype == torch.uint8:
        return (scale.to(torch.float32) - 127.0).exp2()
    return scale.to(torch.float32)


def _dequantize_block_fp8(
    weight,
    scale,
    block,
    dtype,
    out = None,
):
    """``(weight.float() * scale_per_block).to(dtype)``, element by element the same product as a
    full expand of the scale grid, so the result is bit-identical to any dequant using that formula.
    Ragged last blocks (dims not a multiple of the block) are supported. Works in row chunks to
    bound the fp32 transient."""
    if weight.dtype not in _FP8_DTYPES:
        raise TypeError(f"expected an fp8 weight, got {weight.dtype}")
    if weight.dim() > 2:
        weight2 = weight.reshape(-1, *weight.shape[-2:])
        if scale.numel() == 1:
            scale2 = scale.reshape(1, 1, 1).expand(weight2.shape[0], 1, 1)
        else:
            scale2 = scale.reshape(-1, *scale.shape[-2:])
        if out is None:
            out = torch.empty(weight.shape, dtype = dtype, device = weight.device)
        out2 = out.view(-1, *weight.shape[-2:])
        for i in range(weight2.shape[0]):
            _dequantize_block_fp8(weight2[i], scale2[i], block, dtype, out = out2[i])
        return out
    rows, cols = weight.shape
    if scale.dim() == 0 or scale.numel() == 1:
        scale = scale.reshape(1, 1)
    scale = scale.reshape(scale.shape[-2], scale.shape[-1]) if scale.dim() != 2 else scale
    scale_rows, scale_cols = scale.shape
    if scale_rows == 1 and scale_cols == 1:
        block_m, block_n = rows, cols
    elif -(-rows // block[0]) == scale_rows and -(-cols // block[1]) == scale_cols:
        block_m, block_n = block
    elif rows % scale_rows == 0 and cols % scale_cols == 0:
        block_m, block_n = rows // scale_rows, cols // scale_cols
    else:
        raise ValueError(
            f"fp8 weight {tuple(weight.shape)} does not fit its scale grid {tuple(scale.shape)} "
            f"with block {block}"
        )
    scale = _scale_as_fp32(scale).to(weight.device)
    if out is None:
        out = torch.empty((rows, cols), dtype = dtype, device = weight.device)
    # About 64M fp32 elements per chunk, a whole number of row blocks.
    chunk_blocks = max(1, (1 << 26) // max(1, block_m * cols))
    for block_row in range(0, scale_rows, chunk_blocks):
        row_start = block_row * block_m
        row_end = min(rows, (block_row + chunk_blocks) * block_m)
        piece = weight[row_start:row_end].to(torch.float32)
        piece_scale = scale[block_row : block_row + chunk_blocks]
        expanded = piece_scale.repeat_interleave(block_m, dim = 0)[: row_end - row_start]
        expanded = expanded.repeat_interleave(block_n, dim = 1)[:, :cols]
        piece.mul_(expanded)
        out[row_start:row_end].copy_(piece)
        del piece, expanded
    return out


def _output_dtype(model, full_layer_name, default):
    """The load dtype, except fp32 for a target the model keeps in fp32 (_keep_in_fp32_modules)."""
    if model is None or not full_layer_name:
        return default
    module_name, _, attr = full_layer_name.rpartition(".")
    try:
        module = model.get_submodule(module_name) if module_name else model
    except AttributeError:
        return default
    param = getattr(module, attr, None)
    if (
        isinstance(param, torch.Tensor)
        and param.dtype == torch.float32
        and type(param).__name__ != "Params4bit"
        and not type(module).__module__.startswith("bitsandbytes")
    ):
        return torch.float32
    return default


def _target_keeps_fp8(model, full_layer_name) -> bool:
    """An fp8 container module (FP8Embedding) that holds its own scale: load its bytes as stored."""
    if model is None or not full_layer_name:
        return False
    module_name, _, attr = full_layer_name.rpartition(".")
    try:
        module = model.get_submodule(module_name) if module_name else model
    except AttributeError:
        return False
    param = getattr(module, attr, None)
    if not isinstance(param, torch.Tensor) or param.dtype not in _FP8_DTYPES:
        return False
    return any(
        isinstance(getattr(module, name, None), torch.Tensor)
        for name in (attr + "_scale", attr + "_scale_inv", "weight_scale", "weight_scale_inv")
    )


def _merge_accepts_stacked_tensor() -> bool:
    try:
        from transformers.core_model_loading import MergeModulelist

        stacked = torch.zeros(2, 1, 1)
        merged = MergeModulelist(dim = 0).convert(
            {"a": stacked}, source_patterns = ["a"], target_patterns = ["b"]
        )
        return next(iter(merged.values())) is stacked
    except Exception:
        return False


def _build_classes():
    from transformers.core_model_loading import ConversionOps, WeightConverter, _IdentityOp

    class Fp8BlockDequantizeToHalf(ConversionOps):
        """Dequantize fp8 sources with their scales to the load dtype; drop scale entries. `stack`
        pre-allocates the stacked output a following MergeModulelist(dim=0) accepts as is."""

        def __init__(
            self,
            generic = False,
            stack = False,
            fuse = None,
        ):
            self.generic = generic
            self.stack = stack
            # (patterns in merge order, concat dim): replaces MergeModulelist + Concatenate by writing
            # each expert straight into the final stack, the only 16-bit tensor ever held.
            self.fuse = fuse

        def _fused(
            self,
            input_dict,
            target_patterns,
            state,
            model = None,
            full_layer_name = None,
        ):
            patterns, cat_dim = self.fuse
            groups = []
            for pattern in patterns:
                weights = input_dict.get(pattern)
                if weights is None:
                    continue
                weights = weights if isinstance(weights, list) else [weights]
                scales = input_dict.get(_scale_pattern_for(pattern))
                scales = (
                    (scales if isinstance(scales, list) else [scales])
                    if scales is not None
                    else None
                )
                if scales is not None and len(scales) != len(weights):
                    raise ValueError(
                        f"Unsloth: {len(weights)} fp8 weights but {len(scales)} scales"
                    )
                groups.append((weights, scales))
            if not groups:
                return {}
            shapes = [(len(w), *w[0].shape) for w, _ in groups]
            for weights, _ in groups:
                if any(tuple(w.shape) != tuple(weights[0].shape) for w in weights):
                    raise ValueError("Unsloth: experts of one projection have different shapes")
            for shape in shapes[1:]:
                if any(a != b for i, (a, b) in enumerate(zip(shape, shapes[0])) if i != cat_dim):
                    raise ValueError(
                        f"Unsloth: expert stacks {shapes} cannot be concatenated on dim {cat_dim}"
                    )
            final = list(shapes[0])
            final[cat_dim] = sum(shape[cat_dim] for shape in shapes)
            device = groups[0][0][0].device
            dtype = _output_dtype(model, full_layer_name, state.dtype)
            out = torch.empty(final, dtype = dtype, device = device)
            offset = 0
            for (weights, scales), shape in zip(groups, shapes):
                part = out.narrow(cat_dim, offset, shape[cat_dim])
                offset += shape[cat_dim]
                for i in range(len(weights)):
                    weight = weights[i]
                    weights[i] = None
                    if weight.dtype in _FP8_DTYPES:
                        if scales is None:
                            raise RuntimeError(
                                "Unsloth: an fp8 expert weight has no weight_scale_inv"
                            )
                        _dequantize_block_fp8(weight, scales[i], state.block, dtype, out = part[i])
                        state.dequantized += 1
                    elif weight.is_floating_point() and weight.element_size() >= 2:
                        part[i].copy_(weight)
                    else:
                        raise RuntimeError(
                            f"Unsloth: an expert weight is stored as {weight.dtype}, which the fp8 -> 4bit "
                            f"load cannot dequantize. Set {_ENV}=0 to load the checkpoint as it is."
                        )
                    del weight
            return {target_patterns[0]: out}

        @torch.no_grad()
        def convert(
            self,
            input_dict,
            source_patterns = None,
            target_patterns = None,
            full_layer_name = None,
            model = None,
            **kwargs,
        ):
            state = _STATE
            if state is None:
                raise RuntimeError(
                    "Unsloth: fp8 -> 4bit dequantize ran outside an fp8 -> 4bit load"
                )
            block, dtype = state.block, _output_dtype(model, full_layer_name, state.dtype)
            if _target_keeps_fp8(model, full_layer_name):
                # modules_to_convert (Qwen3.8's n-gram table): stays fp8 with its own per-tensor scale.
                state.kept_fp8 += sum(
                    len(v) if isinstance(v, list) else 1
                    for k, v in input_dict.items()
                    if not _is_scale_pattern(k)
                )
                return input_dict
            if self.fuse is not None:
                return self._fused(input_dict, target_patterns, state, model, full_layer_name)
            result = {}
            for key in list(input_dict.keys()):
                if _is_scale_pattern(key):
                    continue
                value = input_dict[key]
                scale_key = _scale_pattern_for(key)
                if scale_key not in input_dict:
                    alt = scale_key[:-1] if scale_key.endswith("$") else scale_key + "$"
                    scale_key = alt if alt in input_dict else None
                weights = value if isinstance(value, list) else [value]
                if scale_key is None:
                    if any(isinstance(w, torch.Tensor) and w.dtype in _FP8_DTYPES for w in weights):
                        raise RuntimeError(
                            f"Unsloth: fp8 tensor for `{full_layer_name}` has no weight_scale_inv in the checkpoint"
                        )
                    if self.generic:
                        target = full_layer_name if full_layer_name is not None else "weight"
                        result[target] = value
                    else:
                        result[key] = value
                    continue
                scales = input_dict[scale_key]
                scales = scales if isinstance(scales, list) else [scales]
                if len(scales) != len(weights):
                    raise ValueError(
                        f"Unsloth: {len(weights)} fp8 weights but {len(scales)} scales for `{full_layer_name}`"
                    )
                out = None
                if self.stack and isinstance(value, list) and len(weights) > 0:
                    first = weights[0]
                    out = torch.empty(
                        (len(weights), *first.shape), dtype = dtype, device = first.device
                    )
                    outputs = out
                else:
                    outputs = []
                for i in range(len(weights)):
                    weight, scale = weights[i], scales[i]
                    weights[i] = None
                    if weight.dtype in _FP8_DTYPES:
                        converted = _dequantize_block_fp8(
                            weight, scale, block, dtype, out = out[i] if out is not None else None
                        )
                        state.dequantized += 1
                    elif weight.is_floating_point() and weight.element_size() >= 2:
                        # A 16-bit tensor that shipped a scale anyway: take it as stored.
                        converted = weight.to(dtype)
                        if out is not None:
                            out[i].copy_(converted)
                    else:
                        raise RuntimeError(
                            f"Unsloth: `{full_layer_name}` is stored as {weight.dtype} with a scale, which the "
                            f"fp8 -> 4bit load cannot dequantize. Set {_ENV}=0 to load the checkpoint as it is."
                        )
                    if out is None:
                        outputs.append(converted)
                    del weight
                if self.generic:
                    target = full_layer_name if full_layer_name is not None else "weight"
                    result[target] = outputs if out is not None else outputs[0]
                else:
                    result[key] = (
                        outputs if (out is not None or isinstance(value, list)) else outputs[0]
                    )
            return result

        @property
        def reverse_op(self):
            return _IdentityOp()

    class Fp8SourceConverter(WeightConverter):
        """A WeightConverter that loads fp8 weights and their scales at their stored dtype. An
        on-the-fly bitsandbytes load casts every source to the model dtype, which would round the
        fp32 scales to bf16."""

        __slots__ = ()

        def add_tensor(self, target_key, source_key, source_pattern, future):
            state = _STATE
            if state is not None and source_key in state.native_keys:
                native = getattr(future, "_unsloth_native_materialize", None)
                if native is None:
                    # Cast to the model dtype, a fp32 scale would round to bf16: refuse, do not guess.
                    raise RuntimeError(
                        f"Unsloth: cannot read `{source_key}` at its stored dtype during an fp8 -> 4bit load. "
                        f"Set {_ENV}=0 to load the fp8 weights as they are."
                    )
                future = native
                state.routed_native.add(source_key)
            return super().add_tensor(target_key, source_key, source_pattern, future)

    return Fp8BlockDequantizeToHalf, Fp8SourceConverter


def _install_spawn_materialize_tag() -> bool:
    """Tag each lazily materialized source with a twin that keeps the stored dtype."""
    from transformers import core_model_loading as cml

    original = cml.spawn_materialize
    if getattr(original, "_unsloth_fp8_to_nf4", False):
        return True
    try:
        signature = inspect.signature(original)
    except (TypeError, ValueError):
        return False
    if not {"thread_pool", "tensor", "device", "dtype"} <= set(signature.parameters):
        return False
    materialize_copy = getattr(cml, "_materialize_copy", None)

    @functools.wraps(original)
    def spawn_materialize(*args, **kwargs):
        job = original(*args, **kwargs)
        if _STATE is None or not callable(job) or materialize_copy is None:
            return job
        try:
            bound = signature.bind(*args, **kwargs)
        except TypeError:
            return job
        arguments = bound.arguments
        if (
            arguments.get("sharding_op", None) is not None
            or arguments.get("thread_pool", None) is not None
        ):
            return job
        tensor = arguments.get("tensor")
        device = arguments.get("device", None)
        try:
            job._unsloth_native_materialize = functools.partial(
                materialize_copy, tensor, device, None
            )
        except Exception:
            pass
        return job

    spawn_materialize._unsloth_fp8_to_nf4 = True
    cml.spawn_materialize = spawn_materialize
    return True


def _refuse_packed_scaled_weights(headers):
    """A scaled weight stored as int8 / uint8 is packed FP4 (or another format): it would be cast
    to bf16 as integers. Fail before loading anything."""
    for key, dtype in headers.items():
        if not key.endswith((".weight_scale_inv", ".scale")):
            continue
        weight = headers.get(_weight_key_of_scale(key))
        if weight is not None and weight not in _FLOAT_SAFETENSORS:
            raise RuntimeError(
                f"Unsloth: `{_weight_key_of_scale(key)}` is stored as {weight} with a block scale, which the "
                f"fp8 -> 4bit load cannot dequantize. Set {_ENV}=0 to load the checkpoint as it is."
            )


def _keep_fp8_embeddings(model, patterns) -> int:
    """The checkpoint's modules_to_convert are fp8 embedding tables too large to expand (Qwen3.8's
    51B-weight n-gram table): keep them fp8 with their per-tensor scale, as an fp8 load does."""
    try:
        from transformers.integrations.finegrained_fp8 import (
            FP8Embedding,
            replace_with_fp8_embedding,
        )
    except Exception:
        return 0
    replace_with_fp8_embedding(model, list(patterns))
    return sum(1 for module in model.modules() if isinstance(module, FP8Embedding))


def _read_headers(checkpoint_files) -> dict:
    from safetensors import safe_open

    headers = {}
    for path in checkpoint_files or ():
        path = str(path)
        if not path.endswith(".safetensors"):
            raise RuntimeError(
                f"Unsloth: fp8 -> 4bit loading needs safetensors shards, got `{os.path.basename(path)}`"
            )
        with safe_open(path, framework = "pt") as handle:
            for key in handle.keys():
                headers[key] = handle.get_slice(key).get_dtype()
    return headers


def _generalize(name: str) -> str:
    escaped = re.escape(name)
    return re.sub(r"(?<=\\\.)\d+(?=\\\.|$)", r"\\d+", escaped)


def _compress_patterns(names, others) -> list:
    """Regexes matching every name in `names` and none in `others`: layer indices generalized
    where that stays exact, else the exact names. Anchored, since transformers re.match()es them."""
    families = {}
    for name in sorted(names):
        families.setdefault(_generalize(name), []).append(name)
    patterns = []
    for pattern, members in families.items():
        regex = re.compile(pattern + "$")
        if any(regex.match(other) for other in others):
            patterns.extend(re.escape(member) + "$" for member in members)
        else:
            patterns.append(pattern + "$")
    return patterns


def _high_precision_linears(model, headers, weight_conversions) -> tuple:
    """nn.Linear modules whose checkpoint weight is stored in 16/32 bits (the fp8 checkpoint's
    modules_to_not_convert), found by renaming every checkpoint key exactly as the loader will."""
    from copy import deepcopy
    from transformers import core_model_loading as cml

    # Copies: renaming marks a transform used, and the used ones are what saving reverses.
    conversions = deepcopy(list(weight_conversions))
    renamings = [c for c in conversions if isinstance(c, cml.WeightRenaming)]
    converters = [c for c in conversions if isinstance(c, cml.WeightConverter)]
    meta_state_dict = model.state_dict()
    prefix = getattr(model, "base_model_prefix", None)
    sources_fp8 = {}
    reached = set()
    for key, dtype in headers.items():
        if key.endswith(_SCALE_SUFFIXES) or key.endswith(".scale"):
            continue
        renamed, _ = cml.rename_source_key(key, renamings, converters, prefix, meta_state_dict)
        if renamed not in meta_state_dict and key in meta_state_dict:
            renamed = key
        if renamed in meta_state_dict:
            reached.add(key)
        sources_fp8[renamed] = sources_fp8.get(renamed, False) or dtype in _FP8_SAFETENSORS
    keep, quantize = [], []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        target = f"{name}.weight"
        if target not in sources_fp8:
            continue
        (quantize if sources_fp8[target] else keep).append(name)
    return keep, quantize, reached


def _weight_key_of_scale(key):
    for suffix in (".weight_scale_inv", ".scale"):
        if key.endswith(suffix):
            return key[: -len(suffix)] + ".weight"
    return key


def _restore_plain_linears(model, names, dtype) -> int:
    """Swap bitsandbytes Linear4bit back to nn.Linear (still on meta, before any weight loads)."""
    try:
        import bitsandbytes as bnb
    except Exception:
        return 0
    restored = 0
    for name in names:
        try:
            module = model.get_submodule(name)
        except AttributeError:
            continue
        if not isinstance(module, bnb.nn.Linear4bit):
            continue
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        device = module.weight.device
        plain = torch.nn.Linear(
            module.in_features,
            module.out_features,
            bias = module.bias is not None,
            device = device,
            dtype = dtype,
        )
        setattr(parent, child, plain)
        restored += 1
    return restored


_installed_class = None


def install_fp8_to_nf4_quantizer() -> bool:
    """Register a Bnb4BitHfQuantizer subclass for `bitsandbytes_4bit` that only differs from its
    parent on a load whose config carries the parked fp8 block. Composes with another subclass."""
    global _installed_class
    if _transformers_supports_fp8_to_nf4() is not None:
        return False
    from transformers.quantizers import auto as quantizers_auto
    from transformers.quantizers.quantizer_bnb_4bit import Bnb4BitHfQuantizer
    from transformers.core_model_loading import WeightConverter, WeightRenaming

    mapping = quantizers_auto.AUTO_QUANTIZER_MAPPING
    current = mapping.get("bitsandbytes_4bit")
    if current is None or not issubclass(current, Bnb4BitHfQuantizer):
        return False
    if getattr(current, "_unsloth_fp8_to_nf4", False):
        return _install_spawn_materialize_tag()
    if not _install_spawn_materialize_tag():
        return False
    DequantOp, SourceConverter = _build_classes()
    stack_ok = _merge_accepts_stacked_tensor()

    def _rebuild(conv):
        from transformers.core_model_loading import Concatenate, MergeModulelist

        weight_sources = [p for p in conv.source_patterns if p.endswith(".weight")]
        if not weight_sources:
            return None
        other = [p for p in conv.source_patterns if not p.endswith(".weight")]
        sources = (
            [p + "$" for p in weight_sources]
            + [p[: -len(".weight")] + ".weight_scale_inv$" for p in weight_sources]
            + other
        )
        ops = list(conv.operations)
        first = ops[0] if ops else None
        merges = isinstance(first, MergeModulelist) and getattr(first, "dim", None) == 0
        if (
            merges
            and len(ops) == 2
            and type(ops[1]) is Concatenate
            and not other
            and len(conv.target_patterns) == 1
        ):
            operations = [DequantOp(fuse = ([p + "$" for p in weight_sources], ops[1].dim % 3))]
        else:
            operations = [DequantOp(stack = bool(stack_ok and merges))] + ops
        rebuilt = SourceConverter(
            source_patterns = sources,
            target_patterns = list(conv._original_target_patterns),
            operations = operations,
            force_cpu = getattr(conv, "force_cpu", False),
        )
        rebuilt.scope_prefix = conv.scope_prefix
        rebuilt.base_model_prefix = conv.base_model_prefix
        return rebuilt

    class UnslothFp8ToNf4Bnb4BitHfQuantizer(current):
        _unsloth_fp8_to_nf4 = True
        _unsloth_fp8_plan = None
        _unsloth_fp8_rebuilt = None
        _unsloth_fp8_model = None
        _unsloth_fp8_kept = 0

        def _process_model_before_weight_loading(self, model, **kwargs):
            global _STATE
            config = getattr(model, "config", None)
            plan = getattr(config, UNSLOTH_FP8_TO_NF4_ATTR, None) if config is not None else None
            if plan is not None:
                self._unsloth_fp8_plan = plan
                # preprocess_model does not forward dtype; the meta model was built in the load dtype.
                dtype = kwargs.get("dtype", None)
                if not isinstance(dtype, torch.dtype):
                    dtype = getattr(model, "dtype", None)
                if dtype not in (torch.bfloat16, torch.float16, torch.float32):
                    dtype = torch.bfloat16
                self._unsloth_fp8_model = model
                headers = _read_headers(kwargs.get("checkpoint_files"))
                _refuse_packed_scaled_weights(headers)
                _STATE = _LoadState(plan, dtype, headers)
                if plan.get("modules_to_convert"):
                    _keep_fp8_embeddings(model, plan["modules_to_convert"])
            return super()._process_model_before_weight_loading(model, **kwargs)

        def update_weight_conversions(self, weight_conversions):
            if self._unsloth_fp8_plan is None or _STATE is None:
                return super().update_weight_conversions(weight_conversions)
            model = self._unsloth_fp8_model
            self._unsloth_fp8_model = None
            # Here, not before loading: only now are the caller's key_mapping renamings known.
            keep, quantize = [], []
            quantize_all = _quantize_16bit_requested()
            if model is not None:
                keep, quantize, reached = _high_precision_linears(
                    model, _STATE.headers, weight_conversions
                )
                _STATE.expected_fp8 = _STATE.fp8_keys & reached
                _STATE.expected_scales = {
                    key
                    for key in _STATE.scale_keys
                    if _weight_key_of_scale(key) in _STATE.expected_fp8
                }
            if quantize_all:
                # UNSLOTH_FP8_TO_NF4_QUANTIZE_16BIT=1 quantizes them too, like a load of the bf16 repo.
                keep = []
            self._unsloth_fp8_kept = _restore_plain_linears(model, keep, _STATE.dtype)
            if keep:
                patterns = _compress_patterns(keep, quantize)
                # A copy: the caller may reuse their BitsAndBytesConfig for another load.
                quantization_config = copy.copy(self.quantization_config)
                skip = list(quantization_config.llm_int8_skip_modules or [])
                quantization_config.llm_int8_skip_modules = skip + patterns
                self.quantization_config = quantization_config
                if isinstance(getattr(self, "modules_to_not_convert", None), list):
                    self.modules_to_not_convert.extend(patterns)
            updated = []
            rebuilt_from = {}
            if _STATE.dot_scale_keys:
                updated.append(
                    WeightRenaming(
                        source_patterns = r"^(.+)\.scale$", target_patterns = r"\1.weight_scale_inv"
                    )
                )
            for conv in weight_conversions:
                if isinstance(conv, WeightConverter):
                    rebuilt = _rebuild(conv)
                    if rebuilt is not None:
                        rebuilt_from[id(rebuilt)] = conv
                        updated.append(rebuilt)
                        continue
                updated.append(conv)
            updated.append(
                SourceConverter(
                    source_patterns = ["weight$", "weight_scale_inv", "activation_scale"],
                    target_patterns = "weight",
                    operations = [DequantOp(generic = True)],
                )
            )
            self._unsloth_fp8_rebuilt = rebuilt_from
            return super().update_weight_conversions(updated)

        def _process_model_after_weight_loading(self, model, **kwargs):
            global _STATE
            if self._unsloth_fp8_plan is not None:
                state = _STATE
                _STATE = None
                self._unsloth_fp8_plan = None
                self._unsloth_fp8_model = None
                rebuilt_from = self._unsloth_fp8_rebuilt or {}
                self._unsloth_fp8_rebuilt = None
                _finish_fp8_to_nf4_load(
                    model, state, rebuilt_from, getattr(self, "_unsloth_fp8_kept", 0)
                )
            return super()._process_model_after_weight_loading(model, **kwargs)

    mapping["bitsandbytes_4bit"] = UnslothFp8ToNf4Bnb4BitHfQuantizer
    _installed_class = UnslothFp8ToNf4Bnb4BitHfQuantizer
    return True


def _finish_fp8_to_nf4_load(model, state, rebuilt_from, kept):
    config = getattr(model, "config", None)
    if config is not None:
        try:
            delattr(config, UNSLOTH_FP8_TO_NF4_ATTR)
        except AttributeError:
            config.__dict__.pop(UNSLOTH_FP8_TO_NF4_ATTR, None)
    # Saving reverses model._weight_conversions: put the model's own converters back.
    conversions = getattr(model, "_weight_conversions", None)
    if isinstance(conversions, list):
        restored = []
        for conv in conversions:
            original = rebuilt_from.get(id(conv))
            if original is not None:
                try:
                    original._was_used = True
                except Exception:
                    pass
                restored.append(original)
            elif type(conv).__name__ == "Fp8SourceConverter":
                continue
            elif getattr(conv, "_original_source_patterns", None) == [r"^(.+)\.scale$"]:
                continue
            else:
                restored.append(conv)
        model._weight_conversions = restored
    if state is None:
        raise RuntimeError("Unsloth: the fp8 -> 4bit load lost its state")
    problems = []
    missed_scales = sorted(state.expected_scales - state.routed_native)
    if missed_scales:
        problems.append(
            f"{len(missed_scales)} fp8 scales were never applied (e.g. `{missed_scales[0]}`), so their "
            "weights would load unscaled"
        )
    leftover = [
        name
        for name, param in model.named_parameters()
        if param.dtype in _FP8_DTYPES and not _target_keeps_fp8(model, name)
    ]
    if leftover:
        problems.append(f"{len(leftover)} parameters are still fp8 (e.g. `{leftover[0]}`)")
    if state.dequantized + state.kept_fp8 < len(state.expected_fp8):
        problems.append(
            f"only {state.dequantized} of {len(state.expected_fp8) - state.kept_fp8} fp8 tensors were dequantized"
        )
    if problems:
        raise RuntimeError(
            "Unsloth: loading this fp8 checkpoint in 4bit failed: " + "; ".join(problems) + ". "
            f"Set {_ENV}=0 to load the fp8 weights as they are."
        )
    kept_fp8 = (
        f" {state.kept_fp8} fp8 embedding shards (modules_to_convert) stay fp8."
        if state.kept_fp8
        else ""
    )
    kept_note = (
        f" {kept} Linear layers the checkpoint stores in 16bit stay in 16bit "
        f"({_ENV_QUANTIZE_ALL}=1 quantizes them too)."
        if kept
        else ""
    )
    print(
        f"Unsloth: dequantized {state.dequantized} fp8 tensors to 4bit while loading.{kept_note}{kept_fp8}"
    )
