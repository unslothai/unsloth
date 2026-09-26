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
"""Re-quantize a compressed-tensors ``pack-quantized`` W4A16/W8A16 checkpoint into bitsandbytes
4-bit per tensor during ``from_pretrained`` (no 16-bit copy on disk). Native MoE stacks keep
transformers' ``DecompressExperts``; needs the transformers 5 weight-conversion loader."""

import inspect
import os
import re
from typing import Any, Optional

import torch

__all__ = [
    "compressed_tensors_bnb_plan",
    "arm_compressed_tensors_bnb_loading",
    "install_compressed_tensors_bnb_quantizer",
    "UNSLOTH_COMPRESSED_TENSORS_ATTR",
]

# Popped by the quantizer after loading so it never reaches a saved config.
UNSLOTH_COMPRESSED_TENSORS_ATTR = "_unsloth_compressed_tensors_bnb"

_SUPPORTED_FORMATS = ("pack-quantized", "mxfp4-pack-quantized")


def _weights_supported(fmt: str, weights: dict) -> bool:
    kind = str(weights.get("type", "int")).lower()
    bits = int(weights.get("num_bits", 0) or 0)
    if fmt == "mxfp4-pack-quantized":
        return kind == "float" and bits == 4 and int(weights.get("group_size", 0) or 0) == 32
    return kind == "int" and bits in (2, 3, 4, 5, 6, 7, 8)


def _describe(plan: dict) -> str:
    names = set()
    top = plan.get("format")
    for group in (plan.get("config_groups") or {}).values():
        fmt = group.get("format") or top
        weights = group.get("weights") or {}
        if fmt == "mxfp4-pack-quantized":
            names.add("MXFP4")
        else:
            names.add(f"INT{int(weights.get('num_bits', 0) or 0)}")
    return "/".join(sorted(names)) or "INT"


def _quant_dict(config) -> Optional[dict]:
    quant = getattr(config, "quantization_config", None)
    if quant is None:
        return None
    if hasattr(quant, "to_dict"):
        try:
            quant = quant.to_dict()
        except Exception:
            return None
    if not isinstance(quant, dict):
        return None
    # Legacy llm-compressor checkpoints nest the real config one level down.
    inner = quant.get("quantization_config")
    if isinstance(inner, dict) and "config_groups" in inner:
        merged = dict(inner)
        merged.setdefault("quant_method", quant.get("quant_method"))
        if "sparsity_config" in quant:
            merged["sparsity_config"] = quant["sparsity_config"]
        quant = merged
    return quant


def _config_and_subconfigs(config) -> list:
    seen, out, stack = set(), [], [config]
    while stack:
        cfg = stack.pop()
        if cfg is None or id(cfg) in seen:
            continue
        seen.add(id(cfg))
        out.append(cfg)
        for name, value in list(getattr(cfg, "__dict__", {}).items()):
            if name.startswith("_"):
                continue
            if (
                hasattr(value, "to_dict")
                and hasattr(value, "__dict__")
                and not isinstance(value, (str, bytes, dict, list, tuple))
            ):
                stack.append(value)
    return out


def _transformers_supports_weight_converters() -> bool:
    try:
        from transformers.core_model_loading import WeightConverter  # noqa: F401
        from transformers.quantizers.base import HfQuantizer
        return hasattr(HfQuantizer, "update_weight_conversions")
    except Exception:
        return False


def compressed_tensors_bnb_plan(config) -> Optional[dict]:
    """Quant dict for weight-only, non-sparse pack-quantized (int) or mxfp4-pack-quantized checkpoints; else ``None``."""
    quant = _quant_dict(config)
    if quant is None:
        return None
    method = str(quant.get("quant_method") or "").lower()
    if method not in ("compressed-tensors", "compressed_tensors", "sparseml"):
        return None
    fmt = quant.get("format")
    if fmt not in _SUPPORTED_FORMATS:
        return None
    groups = quant.get("config_groups") or {}
    if not isinstance(groups, dict) or len(groups) == 0:
        return None
    for group in groups.values():
        if not isinstance(group, dict):
            return None
        weights = group.get("weights")
        if not isinstance(weights, dict):
            return None
        group_format = group.get("format")
        if group_format is not None and group_format not in _SUPPORTED_FORMATS:
            return None
        if not _weights_supported(group_format or fmt, weights):
            return None
        if group.get("input_activations") is not None:
            return None
        if group.get("output_activations") is not None:
            return None
    sparsity = quant.get("sparsity_config")
    if isinstance(sparsity, dict):
        if str(sparsity.get("format", "dense")) != "dense":
            return None
    return quant


def arm_compressed_tensors_bnb_loading(config, verbose: bool = True) -> Optional[dict]:
    """Swap the checkpoint quant config for the plan + quantizer; ``None`` (config untouched) if unsupported."""
    plan = compressed_tensors_bnb_plan(config)
    if plan is None:
        return None
    if not _transformers_supports_weight_converters():
        if verbose:
            print(
                f"Unsloth: This checkpoint is compressed-tensors packed {_describe(plan)}. Re-quantizing it "
                "to bitsandbytes 4-bit on the fly needs transformers 5.8 or later; loading it as published instead."
            )
        return None
    try:
        import compressed_tensors  # noqa: F401
    except Exception:
        if verbose:
            print(
                f"Unsloth: This checkpoint is compressed-tensors packed {_describe(plan)} but `compressed_tensors` "
                "is not installed; loading it as published. `pip install compressed-tensors` to train it in 4-bit."
            )
        return None
    # Parse now: compressed-tensors retires fields (`actorder = "group"`, 0.19.0) and a late failure is costly.
    try:
        _build_quantization_config(plan)
    except Exception as error:
        if verbose:
            print(
                f"Unsloth: This checkpoint is compressed-tensors packed {_describe(plan)} but the installed "
                f"compressed-tensors cannot read its quantization config ({type(error).__name__}); "
                "loading it as published. Upgrading or downgrading `compressed-tensors` may help."
            )
        return None
    if not install_compressed_tensors_bnb_quantizer():
        # No converter hook (transformers < 5.8) or foreign bnb quantizer: keep the checkpoint config.
        if verbose:
            if not _transformers_supports_weight_converters():
                reason = "this transformers has no quantizer weight-conversion hook (5.8 or later has it)"
            else:
                reason = "another library owns the bitsandbytes quantizer"
            print(
                f"Unsloth: This checkpoint is compressed-tensors packed {_describe(plan)} but "
                f"{reason}; loading it as published."
            )
        return None
    # Composite configs (Kimi-K2.7) copy the quant config onto sub-configs transformers also reads.
    for sub in _config_and_subconfigs(config):
        try:
            delattr(sub, "quantization_config")
        except AttributeError:
            pass
        if "quantization_config" in getattr(sub, "__dict__", {}):
            sub.__dict__.pop("quantization_config", None)
    setattr(config, UNSLOTH_COMPRESSED_TENSORS_ATTR, plan)
    if verbose and keep_mxfp4_experts_packed(plan):
        print(
            "Unsloth: Checkpoint is compressed-tensors packed MXFP4. Keeping the MXFP4 weights packed "
            "and dequantizing them on the fly; 16-bit weights go to bitsandbytes 4-bit. "
            "Set UNSLOTH_MXFP4_KEEP_PACKED=0 to re-quantize everything to bitsandbytes 4-bit."
        )
    elif verbose:
        if _int4_packed_plan(plan):
            how = (
                "Keeping its Linear weights packed and decoding them exactly per use "
                f"(set {'UNSLOTH_COMPRESSED_TENSORS_INT4'}=nf4 to re-quantize to bitsandbytes 4-bit instead)."
            )
        else:
            how = (
                "Decompressing each weight on the fly and re-quantizing to bitsandbytes 4-bit "
                "(no 16-bit copy on disk)."
            )
        print(f"Unsloth: Checkpoint is compressed-tensors packed {_describe(plan)}. {how}")
    return plan


# dtype_plan None = keep storage dtype (else int32 packed words are cast to bf16); keyed by the
# renamed name (``<module>.weight`` or the merged stack), so both spellings are added.
# the *renamed* parameter name (``<module>.weight`` for a plain Linear, the merged stack name
# for native MoE experts), which is why both spellings are added.


def _checkpoint_keys(checkpoint_files) -> list:
    keys = []
    try:
        from safetensors import safe_open
    except Exception:
        return keys
    others = [str(p) for p in (checkpoint_files or []) if not str(p).endswith(".safetensors")]
    if others:
        # Pickled shards are cast to the model dtype before any converter sees them.
        raise RuntimeError(
            "Unsloth: re-quantizing a compressed-tensors packed checkpoint on the fly needs "
            f"safetensors shards; {os.path.basename(others[0])} is not one. Convert the "
            "checkpoint to safetensors, or load it as published without `load_in_4bit = True`."
        )
    for path in checkpoint_files or []:
        if not str(path).endswith(".safetensors"):
            continue
        try:
            with safe_open(str(path), framework = "pt") as f:
                keys.extend(f.keys())
        except Exception:
            continue
    return keys


_SAFETENSORS_DTYPES = {
    "I8": torch.int8,
    "I16": torch.int16,
    "I32": torch.int32,
    "I64": torch.int64,
    "U8": torch.uint8,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "F64": torch.float64,
}


def _packed_tensor_meta(checkpoint_files) -> dict:
    """``{module: {suffix: (shape, dtype)}}`` for every packed module, from the shard headers only."""
    from safetensors import safe_open

    meta: dict = {}
    for path in checkpoint_files or []:
        if not str(path).endswith(".safetensors"):
            continue
        with safe_open(str(path), framework = "pt") as f:
            for key in f.keys():
                module, _, suffix = key.rpartition(".")
                if suffix not in _PACKED_SUFFIXES:
                    continue
                piece = f.get_slice(key)
                dtype = _SAFETENSORS_DTYPES.get(piece.get_dtype())
                if dtype is None:
                    return {}
                meta.setdefault(module, {})[suffix] = (tuple(piece.get_shape()), dtype)
    return {m: v for m, v in meta.items() if "weight_packed" in v}


def adopt_int4_packed_linears(model, ct_config, checkpoint_files, dtype) -> tuple:
    """Swap every packed plain ``nn.Linear`` for an ``Int4PackedLinear`` shell.

    Returns ``(swapped, leftover)``: ``leftover`` packed modules (routers, renamed prefixes) keep the
    per-module decompress converter; stacked MoE experts keep their own converter."""
    from torch import nn
    from .compressed_tensors_int4 import Int4PackedLinear, make_int4_packed_linear

    meta = _packed_tensor_meta(checkpoint_files)
    swaps, leftover = [], []
    for name, shapes in meta.items():
        try:
            module = model.get_submodule(name)
        except AttributeError:
            module = None
        if isinstance(module, Int4PackedLinear):
            swaps.append((name, None, None, None))
            continue
        if module is None and ".experts." in name:
            continue
        if type(module) is not nn.Linear:
            leftover.append(name)
            continue
        scheme = _scheme_for_module(ct_config, name, module)
        if (
            int(scheme.weights.num_bits) not in (2, 4, 8)
            or shapes["weight_packed"][0][0] != module.out_features
            or "weight_scale" not in shapes
        ):
            leftover.append(name)
            continue
        swaps.append((name, module, scheme, shapes))
    for name, module, scheme, shapes in swaps:
        if module is None:
            continue
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        device = module.weight.device if module.weight is not None else "meta"
        setattr(parent, child, make_int4_packed_linear(module, scheme, shapes, dtype, device))
    return [name for name, *_ in swaps], leftover


def _generalize(name: str) -> str:
    escaped = re.escape(name)
    return re.sub(r"(?<=\\.)\d+(?=\\.|$)", r"\\d+", escaped)


def packed_weight_dtype_plan(keys) -> dict:
    """Regex -> ``None`` for ``dtype_plan``; exact names where widening would catch unpacked modules."""
    packed = [k[: -len(".weight_packed")] for k in keys if k.endswith(".weight_packed")]
    if not packed:
        return {}
    plain = {k[: -len(".weight")] for k in keys if k.endswith(".weight")}
    packed_set = set(packed)
    families: dict = {}
    for name in packed:
        families.setdefault(_generalize(name), []).append(name)
    plan: dict = {}
    for pattern, members in families.items():
        rx = re.compile(pattern + r"$")
        collides = any(rx.match(p) for p in plain if p not in packed_set)
        if collides:
            for name in members:
                plan[re.escape(name) + r"\.weight$"] = None
        else:
            plan[pattern + r"\.weight$"] = None
    return plan


_PACKED_SUFFIXES = (
    "weight_packed",
    "weight_scale",
    "weight_shape",
    "weight_zero_point",
    "weight_g_idx",
)


def _build_quantization_config(plan: dict):
    from compressed_tensors.quantization import QuantizationConfig

    clean = {
        k: v for k, v in plan.items() if k not in ("sparsity_config", "global_compression_ratio")
    }
    clean.setdefault("quant_method", "compressed-tensors")
    return QuantizationConfig.model_validate(clean)


def _experts_scheme(ct_config):
    try:
        from transformers.integrations.compressed_tensors import get_experts_scheme
        return get_experts_scheme(ct_config)
    except Exception:
        groups = list(ct_config.config_groups.values())
        for group in groups:
            if any(t.startswith("re:") and "experts" in t for t in group.targets):
                return group
        return groups[0]


def _scheme_for_sources(ct_config, weight_sources):
    groups = list(ct_config.config_groups.values())
    if len(groups) == 1:
        return groups[0]
    chosen = []
    for pattern in weight_sources:
        name = pattern[: -len(".weight")] if pattern.endswith(".weight") else pattern
        name = name.replace("*", "0")
        scheme = _scheme_for_module(ct_config, name, None)
        if all(scheme is not s for s in chosen):
            chosen.append(scheme)
    if len(chosen) > 1:
        raise RuntimeError(
            "Unsloth: the compressed-tensors checkpoint quantizes the projections of one expert "
            f"bucket ({', '.join(weight_sources)}) under different config groups; re-quantizing it "
            "on the fly is not supported. Load it as published without `load_in_4bit = True`."
        )
    return chosen[0] if chosen else _experts_scheme(ct_config)


def _layer_expert_scheme(ct_config, full_layer_name, packed_key, n_experts, default):
    if len(ct_config.config_groups) < 2:
        return default
    parent = full_layer_name.rsplit(".", 1)[0]
    pattern = packed_key.rstrip("$")
    for suffix in (".weight_packed", "_packed"):
        if pattern.endswith(suffix):
            pattern = pattern[: -len(suffix)]
            break
    if pattern.endswith(".weight"):
        pattern = pattern[: -len(".weight")]
    pattern = pattern.lstrip("^").lstrip(".")
    parts = parent.split(".")
    tail = None
    for start in range(len(parts)):
        head = ".".join(parts[start:])
        if pattern.startswith(head + "."):
            tail = pattern[len(head) :]
            break
    if tail is None or "*" not in tail:
        return default
    schemes = []
    for i in range(n_experts):
        scheme = _scheme_for_module(ct_config, parent + tail.replace("*", str(i), 1), None)
        if all(scheme is not s for s in schemes):
            schemes.append(scheme)
    if len(schemes) > 1:
        raise RuntimeError(
            f"Unsloth: the compressed-tensors checkpoint quantizes the experts of {parent} under "
            "different config groups; re-quantizing it on the fly is not supported. Load it as "
            "published without `load_in_4bit = True`."
        )
    return schemes[0] if schemes else default


def drop_load_only_conversions(model) -> int:
    conversions = getattr(model, "_weight_conversions", None)
    if not conversions:
        return 0
    kept, dropped = [], 0
    for conv in conversions:
        ops = getattr(conv, "operations", None) or []
        if any(
            isinstance(op, (_DecompressPackedWeights, _WithOriginalSources, _StackPackedExperts))
            for op in ops
        ):
            dropped += 1
            continue
        kept.append(conv)
    if dropped:
        try:
            model._weight_conversions = kept
        except Exception:
            return 0
    return dropped


def _scheme_for_module(ct_config, name: str, module: Optional[torch.nn.Module]):
    groups = list(ct_config.config_groups.values())
    if len(groups) == 1:
        return groups[0]
    # compressed-tensors precedence: an exact module path, then a regex, then a class name.
    for group in groups:
        if name in group.targets:
            return group
    for group in groups:
        for target in group.targets:
            if target.startswith("re:") and re.match(target[3:], name):
                return group
    try:
        from compressed_tensors.utils.match import is_match
        if module is not None:
            for group in groups:
                if is_match(name, module, group.targets):
                    return group
    except Exception:
        pass
    class_name = type(module).__name__ if module is not None else "Linear"
    for group in groups:
        for target in group.targets:
            if target == class_name or target == "Linear":
                return group
    return groups[0]


def _decompress_one_triton(scheme, packed, scale, shape, zero_point, g_idx, dtype):
    """Exact Triton decode (bit-identical to compressed-tensors) for CUDA tensors; ``None`` otherwise."""
    if packed.device.type != "cuda" or dtype not in (torch.bfloat16, torch.float16, torch.float32):
        return None
    weights = scheme.weights
    bits = int(weights.num_bits)
    strategy = str(getattr(weights.strategy, "value", weights.strategy)).lower()
    if bits not in (2, 4, 8) or strategy not in ("group", "channel") or scale.dim() != 2:
        return None
    if getattr(torch.version, "hip", None):
        return None
    try:
        from ..kernels.int4_packed import Int4QuantState, int4_dequantize
    except Exception:
        return None
    rows = packed.shape[0]
    cols = (
        int(shape[1])
        if shape is not None and shape.numel() == 2
        else packed.shape[1] * (32 // bits)
    )
    group = int(weights.group_size) if strategy == "group" and weights.group_size else cols
    if scale.shape[0] != rows or cols > packed.shape[1] * (32 // bits):
        return None
    qs = Int4QuantState(
        scale,
        None if weights.symmetric else zero_point,
        g_idx if g_idx is not None and g_idx.device.type == "cuda" and -1 not in g_idx else None,
        (rows, cols),
        bits,
        group,
        dtype,
    )
    if not weights.symmetric and zero_point is None:
        return None
    if g_idx is not None and qs.g_idx is None:
        return None
    return int4_dequantize(packed, qs, dtype)


def _decompress_one(compressor, scheme, packed, scale, shape, zero_point, g_idx, dtype):
    fast = _decompress_one_triton(scheme, packed, scale, shape, zero_point, g_idx, dtype)
    if fast is not None:
        return fast
    state = {"weight_packed": packed, "weight_scale": scale}
    if shape is not None and shape.numel():
        state["weight_shape"] = shape
    else:
        storage_bits = 8 if packed.dtype == torch.uint8 else 32
        pack_factor = storage_bits // int(scheme.weights.num_bits)
        state["weight_shape"] = torch.tensor([packed.shape[0], packed.shape[1] * pack_factor])
    if zero_point is not None:
        state["weight_zero_point"] = zero_point
    if g_idx is not None:
        state["weight_g_idx"] = g_idx
    out = compressor.decompress(state, scheme)
    weight = out["weight"]
    if dtype is not None and weight.dtype != dtype:
        weight = weight.to(dtype)
    return weight.contiguous()


class _DecompressPackedWeights:
    def __init__(
        self,
        ct_config,
        dtype,
        stacked: bool,
        scheme = None,
    ):
        self.ct_config = ct_config
        self.dtype = dtype
        self.stacked = stacked
        self.scheme = scheme

    def _compressor(self, scheme):
        from compressed_tensors.compressors import BaseCompressor
        fmt = scheme.format or getattr(self.ct_config, "format", None) or "pack-quantized"
        return BaseCompressor.get_value_from_registry(str(fmt))

    def convert(
        self,
        input_dict,
        source_patterns = None,
        target_patterns = None,
        full_layer_name = None,
        model = None,
        **kwargs,
    ):
        packed_keys = [k for k in input_dict if "weight_packed" in k]
        if not packed_keys:
            return input_dict
        out = {}
        as_list = lambda v: v if isinstance(v, list) else [v]  # noqa: E731
        for packed_key in packed_keys:
            packed = input_dict[packed_key]
            get = lambda suffix: input_dict.get(packed_key.replace("weight_packed", suffix))  # noqa: E731
            scale, shape, zero_point, g_idx = (
                get(s)
                for s in ("weight_scale", "weight_shape", "weight_zero_point", "weight_g_idx")
            )
            scheme = self.scheme
            if self.stacked and full_layer_name:
                scheme = _layer_expert_scheme(
                    self.ct_config, full_layer_name, packed_key, len(as_list(packed)), scheme
                )
            if scheme is None:
                module = None
                if model is not None and full_layer_name is not None and not self.stacked:
                    try:
                        from transformers.quantizers.quantizers_utils import get_module_from_name
                        module, _ = get_module_from_name(model, full_layer_name)
                    except Exception:
                        module = None
                scheme = _scheme_for_module(
                    self.ct_config, (full_layer_name or "").rsplit(".", 1)[0], module
                )
            compressor = self._compressor(scheme)
            packed_list = as_list(packed)
            n = len(packed_list)
            scales, shapes, zps, gidxs = (
                (as_list(v) if v is not None else [None] * n)
                for v in (scale, shape, zero_point, g_idx)
            )
            if self.stacked:
                stacked_out = None
                for i, p in enumerate(packed_list):
                    w = _decompress_one(
                        compressor, scheme, p, scales[i], shapes[i], zps[i], gidxs[i], self.dtype
                    )
                    if stacked_out is None:
                        stacked_out = torch.empty((n, *w.shape), dtype = w.dtype, device = w.device)
                    stacked_out[i].copy_(w)
                    del w
                value = stacked_out
            else:
                value = _decompress_one(
                    compressor,
                    scheme,
                    packed_list[0],
                    scales[0],
                    shapes[0],
                    zps[0],
                    gidxs[0],
                    self.dtype,
                )
            if self.stacked:
                out[packed_key] = value
            else:
                out[(target_patterns or ["weight"])[0]] = value
        for key, value in input_dict.items():
            if not any(s in key for s in _PACKED_SUFFIXES) and key not in out:
                out[key] = value
        return out

    @property
    def reverse_op(self):
        try:
            from transformers.core_model_loading import _IdentityOp
            return _IdentityOp()
        except Exception:
            return None


class _WithOriginalSources:
    """Run a converter's own op under the source contract it was declared with.

    The decompression step leaves each expert bucket under its `<weight>_packed$`
    key and the converter's `source_patterns` name the packed metadata. An op
    that iterates `source_patterns` and requires every one of them present
    (transformers' ErnieFuseAndSplitTextVisionExperts) then fails on the consumed
    scale and shape patterns. Here every decompressed bucket is renamed to the
    plain weight pattern and the op sees the original `source_patterns`, which
    is exactly what it sees on an unpacked checkpoint.
    """

    def __init__(
        self,
        op,
        original_sources,
        weight_sources,
        receives_buckets = True,
    ):
        self.op = op
        self.original_sources = list(original_sources)
        self.weight_sources = list(weight_sources)
        # Later ops in a chain (Concatenate after MergeModulelist) get the previous op's output.
        self.receives_buckets = receives_buckets

    def convert(
        self,
        input_dict,
        source_patterns = None,
        target_patterns = None,
        **kwargs,
    ):
        if not self.receives_buckets:
            return self.op.convert(
                input_dict,
                source_patterns = self.original_sources,
                target_patterns = target_patterns,
                **kwargs,
            )
        keeps_stack = _merge_takes_a_stacked_tensor(self.op)
        for pattern in self.weight_sources:
            if (pattern + "_packed$") in input_dict and (pattern + "$") in input_dict:
                # The loader drops indices of mixed subsets, so a merge would reorder experts.
                raise RuntimeError(
                    "Unsloth: the compressed-tensors checkpoint packs only some of the experts of "
                    f"`{pattern}`; a partially packed expert bucket cannot be re-quantized on the fly. "
                    "Load it as published without `load_in_4bit = True`."
                )
        renamed = {}
        for key, value in input_dict.items():
            new_key = key
            for pattern in self.weight_sources:
                if key in (pattern + "_packed$", pattern + "$"):
                    new_key = pattern
                    if isinstance(value, torch.Tensor) and not keeps_stack:
                        value = list(value.unbind(0))
                    break
            renamed[new_key] = value
        return self.op.convert(
            renamed,
            source_patterns = self.original_sources,
            target_patterns = target_patterns,
            **kwargs,
        )

    @property
    def reverse_op(self):
        return getattr(self.op, "reverse_op", None)

    def __getattr__(self, name):
        # copy.deepcopy (the loader copies each converter) probes attributes before `op` exists.
        try:
            op = self.__dict__["op"]
        except KeyError:
            raise AttributeError(name) from None
        return getattr(op, name)


# MXFP4 stays packed: MoE experts -> Mxfp4StackedExperts, other Linears -> Mxfp4PackedLinear (UNSLOTH_MXFP4_KEEP_PACKED=0 opts out).

_PACKED_EXPERT_KEY = re.compile(r"^(.*)\.experts\.(\d+)\.(w[123])\.weight_(packed|scale)$")
_STACKED_EXPERT_TARGETS = (
    "experts.gate_up_blocks",
    "experts.gate_up_scales",
    "experts.down_blocks",
    "experts.down_scales",
)


def _plan_is_mxfp4(plan) -> bool:
    if not isinstance(plan, dict):
        return False
    top = plan.get("format")
    groups = (plan.get("config_groups") or {}).values()
    return bool(groups) and all(
        (group.get("format") or top) == "mxfp4-pack-quantized" for group in groups
    )


def _int4_packed_plan(plan) -> bool:
    """Packed INT route: INT groups only (MXFP4 groups also have 4 bits but take the MXFP4 route)."""
    if not isinstance(plan, dict):
        return False
    top = plan.get("format")
    for group in (plan.get("config_groups") or {}).values():
        weights = group.get("weights") or {}
        if (group.get("format") or top) == "mxfp4-pack-quantized" or str(
            weights.get("type", "int")
        ).lower() != "int":
            return False
    from .compressed_tensors_int4 import int4_packed_route_enabled, int4_packed_supported_plan

    return int4_packed_route_enabled() and int4_packed_supported_plan(plan)


def _zoo_saves_packed_modules() -> bool:
    """Without unsloth_zoo's packed save patch a full save writes a checkpoint nothing reloads."""
    try:
        from transformers import PreTrainedModel
        from unsloth_zoo.temporary_patches import mxfp4 as zoo_mxfp4

        zoo_mxfp4._densified_module_names
        zoo_mxfp4.patch_save_pretrained_mxfp4()
    except Exception:
        return False
    return getattr(PreTrainedModel.save_pretrained, "_unsloth_mxfp4_patched", False)


def keep_mxfp4_experts_packed(plan) -> bool:
    from .mxfp4_compressed_linear import mxfp4_keep_packed_enabled
    return mxfp4_keep_packed_enabled() and _plan_is_mxfp4(plan) and _zoo_saves_packed_modules()


def packed_expert_prefixes(keys) -> dict:
    """{moe prefix: expert count}; empty if any layer is only partly packed (converters would claim its keys)."""
    seen: dict = {}
    for key in keys:
        match = _PACKED_EXPERT_KEY.match(key)
        if match is not None:
            prefix, index, proj, kind = match.groups()
            seen.setdefault(prefix, set()).add((int(index), proj, kind))
    extras = {
        k.rsplit(".experts.", 1)[0]
        for k in keys
        if ".experts." in k and k.endswith(("weight_shape", "weight_zero_point", "weight_g_idx"))
    }
    prefixes = {}
    for prefix, entries in seen.items():
        count = 1 + max(index for index, _, _ in entries)
        complete = len(entries) == count * 6
        if not complete or prefix in extras:
            return {}
        prefixes[prefix] = count
    return prefixes


def _prefix_for(name: str, prefixes: dict) -> Optional[str]:
    for prefix in prefixes:
        if prefix == name or prefix.endswith("." + name) or name.endswith("." + prefix):
            return prefix
    return None


def _plain_expert_shape(experts) -> Optional[tuple]:
    first = experts[0]
    hidden = intermediate = None
    for expert in experts:
        w1, w2, w3 = (getattr(expert, n, None) for n in ("w1", "w2", "w3"))
        if not all(isinstance(w, torch.nn.Linear) and w.bias is None for w in (w1, w2, w3)):
            return None
        if not hasattr(expert, "act_fn") or type(expert) is not type(first):
            return None
        shape = (w1.in_features, w1.out_features)
        if (w3.in_features, w3.out_features) != shape or (w2.in_features, w2.out_features) != shape[
            ::-1
        ]:
            return None
        if hidden is None:
            hidden, intermediate = shape
        elif (hidden, intermediate) != shape:
            return None
    if hidden % 32 or intermediate % 32:
        return None
    return hidden, intermediate


def _swappable_expert_blocks(model, prefixes) -> Optional[list]:
    """``None`` unless every packed MoE block is the plain remote-code layout: all or nothing."""
    try:
        from unsloth_zoo.mxfp4_stacked_experts import Mxfp4StackedExperts  # noqa: F401
    except Exception:
        return None
    from .remote_moe_shims import is_remote_deepseek_moe

    plan = []
    for name, module in model.named_modules():
        experts = getattr(module, "experts", None)
        if not isinstance(experts, torch.nn.ModuleList) or len(experts) == 0:
            continue
        prefix = _prefix_for(name, prefixes)
        if prefix is None:
            continue
        # A stack runs only through the remote MoE shim's dispatch.
        if (
            prefixes[prefix] != len(experts)
            or getattr(module, "ep_size", 1) > 1
            or not is_remote_deepseek_moe(module)
        ):
            return None
        dims = _plain_expert_shape(experts)
        if dims is None:
            return None
        plan.append((name, module, prefix, dims))
    if len({prefix for _, _, prefix, _ in plan}) != len(prefixes):
        return None
    return plan


def _new_stacked_experts(experts, dims, dtype, device):
    from unsloth_zoo.mxfp4_stacked_experts import Mxfp4StackedExperts

    first = experts[0]
    # Kimi's SiTU takes the concatenated [gate, up]; any other act_fn is act(gate) * up.
    fused = getattr(getattr(first, "config", None), "hidden_act", None) == "situ"
    return Mxfp4StackedExperts(
        len(experts),
        dims[0],
        dims[1],
        first.act_fn,
        fused,
        dtype = dtype,
        device = device,
    )


def _swap_in_stacks(blocks, dtype) -> list:
    swapped = []
    for name, module, _, dims in blocks:
        first = module.experts[0]
        module.experts = _new_stacked_experts(module.experts, dims, dtype, first.w1.weight.device)
        swapped.append(name)
    return swapped


def _match_packed_linears(model, keys) -> Optional[dict]:
    """Linear per packed checkpoint module (names may differ by leading prefixes); ``None`` if any has no home."""
    prefixes = [k[: -len(".weight_packed")] for k in keys if k.endswith(".weight_packed")]
    if not prefixes:
        return None
    linears = {
        name: module
        for name, module in model.named_modules()
        if name and isinstance(module, torch.nn.Linear)
    }
    by_suffix: dict = {}
    for name in linears:
        parts = name.split(".")
        for i in range(1, len(parts)):
            by_suffix.setdefault(".".join(parts[i:]), []).append(name)
    matched = {}
    for prefix in prefixes:
        found = prefix if prefix in linears else None
        if found is None:
            owners = by_suffix.get(prefix, ())
            found = owners[0] if len(owners) == 1 else None
        if found is None:
            parts = prefix.split(".")
            for i in range(1, len(parts)):
                if ".".join(parts[i:]) in linears:
                    found = ".".join(parts[i:])
                    break
        if found is None or found in matched:
            return None
        matched[found] = linears[found]
    return matched


class Mxfp4KeepPackedPlan:
    def __init__(self, blocks, linears):
        self.blocks = blocks
        self.linears = linears


def plan_mxfp4_keep_packed(model, keys) -> Optional[Mxfp4KeepPackedPlan]:
    """All-or-nothing keep-packed plan; ``None`` falls back to the existing decompress route."""
    packed = [k[: -len(".weight_packed")] for k in keys if k.endswith(".weight_packed")]
    if not packed:
        return None
    # Kept packed, a scale missing from the checkpoint would stay uninitialised bytes.
    present = set(keys)
    if any(prefix + ".weight_scale" not in present for prefix in packed):
        return None
    prefixes = packed_expert_prefixes(keys)
    if not prefixes and any(_PACKED_EXPERT_KEY.match(k) for k in keys):
        return None
    blocks = (_swappable_expert_blocks(model, prefixes) or []) if prefixes else []
    stacked = {prefix for _, _, prefix, _ in blocks}

    def _stacked(key):
        match = _PACKED_EXPERT_KEY.match(key)
        return match is not None and match.group(1) in stacked

    rest = [k for k in keys if not _stacked(k)]
    linears = {}
    if any(k.endswith(".weight_packed") for k in rest):
        linears = _match_packed_linears(model, rest)
        if linears is None or any(m.in_features % 32 for m in linears.values()):
            return None
    return Mxfp4KeepPackedPlan(blocks, sorted(linears))


def _keep_packed_linears(
    model,
    names,
    ct_config,
    dtype = None,
) -> int:
    from .mxfp4_compressed_linear import make_mxfp4_packed_linear
    for name in names:
        old = model.get_submodule(name)
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        device = next(iter(old.parameters()), torch.empty(0, device = "meta")).device
        new = make_mxfp4_packed_linear(
            old.in_features,
            old.out_features,
            bias = old.bias,
            scheme = _scheme_for_module(ct_config, name, old),
            device = device,
            dtype = dtype,
        )
        setattr(parent, child, new)
    return len(names)


class _StackPackedExperts:
    def __init__(self, scales: bool):
        self.scales = scales

    def convert(
        self,
        input_dict,
        source_patterns = None,
        target_patterns = None,
        **kwargs,
    ):
        from unsloth_zoo.mxfp4_stacked_experts import stack_packed_experts

        as_list = lambda v: v if isinstance(v, list) else [v]  # noqa: E731
        per_projection = [
            as_list(input_dict.pop(pattern)) for pattern in source_patterns if pattern in input_dict
        ]
        return {target_patterns[0]: stack_packed_experts(per_projection, blocks = not self.scales)}

    @property
    def reverse_op(self):
        try:
            from transformers.core_model_loading import _IdentityOp
            return _IdentityOp()
        except Exception:
            return None


def finalize_packed_mxfp4_experts(model) -> int:
    count = 0
    for module in model.modules():
        if getattr(type(module), "_unsloth_mxfp4_stacked_experts", False):
            module.finalize()
            count += 1
    return count


_with_sources_classes = {}


def _merge_takes_a_stacked_tensor(op):
    try:
        from transformers.core_model_loading import MergeModulelist
    except Exception:  # pragma: no cover
        return False
    if not isinstance(op, MergeModulelist):
        return False
    try:
        source = inspect.getsource(type(op).convert)
    except (OSError, TypeError):
        return False
    return "isinstance(tensors, torch.Tensor)" in source


def _with_original_sources(
    op,
    original_sources,
    weight_sources,
    receives_buckets = True,
):
    """Subclass of ``op``'s own class: WeightConverter only allows many-to-many for its Ernie ops."""
    cls = _with_sources_classes.get(type(op))
    if cls is None:
        cls = _with_sources_classes[type(op)] = type(
            "WithOriginalSources" + type(op).__name__, (_WithOriginalSources, type(op)), {}
        )
    return cls(op, original_sources, weight_sources, receives_buckets)


_installed = False


def install_compressed_tensors_bnb_quantizer() -> bool:
    """Idempotent; ``False`` on transformers without the weight-conversion loader."""
    global _installed
    if _installed:
        return True
    if not _transformers_supports_weight_converters():
        return False
    from transformers.quantizers import auto as quantizers_auto
    from transformers.quantizers.quantizer_bnb_4bit import Bnb4BitHfQuantizer
    from transformers.core_model_loading import ConversionOps, WeightConverter

    op_cls = type("DecompressPackedWeights", (_DecompressPackedWeights, ConversionOps), {})
    with_sources_cls = _with_original_sources
    stack_cls = type("StackPackedExperts", (_StackPackedExperts, ConversionOps), {})

    class UnslothBnb4BitHfQuantizer(Bnb4BitHfQuantizer):
        _unsloth_ct_config = None
        _unsloth_ct_dtype = None
        _unsloth_dtype_plan = None
        _unsloth_plan_dicts = ()
        _unsloth_packed_experts = ()
        _unsloth_packed_linears = ()
        _unsloth_keep_packed = False
        _unsloth_int4_packed = ()
        _unsloth_int4_leftover = ()

        def _process_model_before_weight_loading(self, model, **kwargs):
            config = getattr(model, "config", None)
            plan = (
                getattr(config, UNSLOTH_COMPRESSED_TENSORS_ATTR, None)
                if config is not None
                else None
            )
            if plan is not None:
                # Read, not popped: the device-map planner runs this on a meta model first, and
                # popping here left the real load without converters (random experts, Kimi on 4 GPUs).
                self._unsloth_ct_config = _build_quantization_config(plan)
                dtype = kwargs.get("dtype", None)
                if not isinstance(dtype, torch.dtype):
                    dtype = getattr(model, "dtype", None) or torch.bfloat16
                self._unsloth_ct_dtype = dtype
                keys = _checkpoint_keys(kwargs.get("checkpoint_files"))
                self._unsloth_dtype_plan = packed_weight_dtype_plan(keys)
                # Stacks swapped in now; packed Linears after bitsandbytes lays out the 16-bit ones.
                keep = (
                    plan_mxfp4_keep_packed(model, keys) if keep_mxfp4_experts_packed(plan) else None
                )
                self._unsloth_packed_experts = _swap_in_stacks(keep.blocks, dtype) if keep else []
                self._unsloth_packed_linears = keep.linears if keep else []
                self._unsloth_keep_packed = keep is not None
                self._unsloth_plan_dicts = []
                original_get_dtype_plan = model._get_dtype_plan
                quantizer = self

                def _get_dtype_plan(dtype, _original = original_get_dtype_plan):
                    merged = dict(_original(dtype))
                    merged.update(quantizer._unsloth_dtype_plan)
                    quantizer._unsloth_plan_dicts.append(merged)
                    return merged

                model._get_dtype_plan = _get_dtype_plan
                self._unsloth_int4_packed = []
                self._unsloth_int4_leftover = []
                if keep is None and _int4_packed_plan(plan):
                    swapped, leftover = adopt_int4_packed_linears(
                        model, self._unsloth_ct_config, kwargs.get("checkpoint_files"), dtype
                    )
                    self._unsloth_int4_packed = swapped
                    self._unsloth_int4_leftover = leftover if swapped else []
                    if self._unsloth_int4_packed:
                        # Load the checkpoint tensors as stored: a cast would change the int32 words and
                        # round a fp32 scale.
                        self._unsloth_dtype_plan[
                            r"\.weight_(packed|scale|zero_point|g_idx|shape)$"
                        ] = None
            else:
                self._unsloth_keep_packed = False
            result = super()._process_model_before_weight_loading(model, **kwargs)
            if plan is not None and self._unsloth_packed_linears:
                _keep_packed_linears(
                    model,
                    self._unsloth_packed_linears,
                    self._unsloth_ct_config,
                    self._unsloth_ct_dtype,
                )
            return result

        def _process_model_after_weight_loading(self, model, **kwargs):
            config = getattr(model, "config", None)
            if (
                config is not None
                and getattr(config, UNSLOTH_COMPRESSED_TENSORS_ATTR, None) is not None
            ):
                try:
                    delattr(config, UNSLOTH_COMPRESSED_TENSORS_ATTR)
                except AttributeError:
                    config.__dict__.pop(UNSLOTH_COMPRESSED_TENSORS_ATTR, None)
            # Load-only: save_pretrained would reverse them into packed names without metadata.
            drop_load_only_conversions(model)
            if getattr(self, "_unsloth_int4_packed", None):
                from .compressed_tensors_int4 import finalize_int4_packed_linears
                finalize_int4_packed_linears(model, self._unsloth_ct_dtype)
            stacks = finalize_packed_mxfp4_experts(model)
            if stacks or self._unsloth_packed_linears:
                print(
                    f"Unsloth: Kept the MXFP4 weights packed ({stacks} expert stacks, "
                    f"{len(self._unsloth_packed_linears)} Linears); they are dequantized on the fly "
                    "(UNSLOTH_MXFP4_KEEP_PACKED=0 re-quantizes them to bitsandbytes 4-bit instead)."
                )
            return super()._process_model_after_weight_loading(model, **kwargs)

        def _unsloth_stacking_converters(self, WeightConverter, stack_cls):
            if not self._unsloth_packed_experts:
                return []
            converters = []
            for suffix, is_scale in (("weight_packed", False), ("weight_scale", True)):
                kind = "scales" if is_scale else "blocks"
                for projections, target in (
                    (("w1", "w3"), f"experts.gate_up_{kind}"),
                    (("w2",), f"experts.down_{kind}"),
                ):
                    converters.append(
                        WeightConverter(
                            source_patterns = [f"experts.*.{w}.{suffix}$" for w in projections],
                            target_patterns = target,
                            operations = [stack_cls(is_scale)],
                        )
                    )
            self._unsloth_keep_storage_dtype(list(_STACKED_EXPERT_TARGETS))
            return converters

        def _unsloth_keep_storage_dtype(self, target_patterns):
            for target in target_patterns:
                key = re.escape(target) + r"$"
                self._unsloth_dtype_plan[key] = None
                for plan in self._unsloth_plan_dicts:
                    plan[key] = None

        def update_weight_conversions(self, weight_conversions):
            ct_config = self._unsloth_ct_config
            if ct_config is None:
                return super().update_weight_conversions(weight_conversions)
            if getattr(self, "_unsloth_keep_packed", False):
                return (
                    list(weight_conversions)
                    + self._unsloth_stacking_converters(WeightConverter, stack_cls)
                    + self.get_weight_conversions()
                )
            dtype = self._unsloth_ct_dtype
            updated = []
            for conv in weight_conversions:
                if isinstance(conv, WeightConverter) and any(
                    "experts" in p for p in conv.source_patterns
                ):
                    weight_sources = [p for p in conv.source_patterns if p.endswith(".weight")]
                    if weight_sources:
                        scheme = _scheme_for_sources(ct_config, weight_sources)
                        other = [p for p in conv.source_patterns if not p.endswith(".weight")]
                        # Plain `.weight` kept so `ignore`d unpacked experts still merge; anchored.
                        new_sources = (
                            [p + "_packed$" for p in weight_sources]
                            + [p + "_scale$" for p in weight_sources]
                            + [p + "_shape$" for p in weight_sources]
                            + [p + "_zero_point$" for p in weight_sources]
                            + [p + "_g_idx$" for p in weight_sources]
                            + [p + "$" for p in weight_sources]
                            + other
                        )
                        original_sources = list(
                            getattr(conv, "_original_source_patterns", conv.source_patterns)
                        )
                        conv = WeightConverter(
                            source_patterns = new_sources,
                            target_patterns = conv._original_target_patterns,
                            operations = [op_cls(ct_config, dtype, stacked = True, scheme = scheme)]
                            + [
                                with_sources_cls(
                                    op,
                                    original_sources,
                                    weight_sources,
                                    receives_buckets = index == 0,
                                )
                                for index, op in enumerate(conv.operations)
                            ],
                        )
                        self._unsloth_keep_storage_dtype(conv._original_target_patterns)
                updated.append(conv)
            if not getattr(self, "_unsloth_int4_packed", None):
                sources = [s + "$" for s in _PACKED_SUFFIXES]
            else:
                # Only modules left out of the packed route; the lookbehind keeps the rename to `weight`.
                sources = [
                    f"(?<={re.escape(name)}\\.){s}$"
                    for name in getattr(self, "_unsloth_int4_leftover", None) or ()
                    for s in _PACKED_SUFFIXES
                ]
            if sources:
                updated.append(
                    WeightConverter(
                        source_patterns = sources,
                        target_patterns = "weight",
                        operations = [op_cls(ct_config, dtype, stacked = False)],
                    )
                )
            # Call up the MRO so a composed subclass's hook still runs.
            parent = getattr(super(), "update_weight_conversions", None)
            if parent is None:
                return updated + list(self.get_weight_conversions())
            return parent(updated)

    mapping = getattr(quantizers_auto, "AUTO_QUANTIZER_MAPPING", None)
    if mapping is None:
        return False
    for key in ("bitsandbytes_4bit",):
        current = mapping.get(key)
        if current is Bnb4BitHfQuantizer:
            mapping[key] = UnslothBnb4BitHfQuantizer
        elif current is not None and not issubclass(current, Bnb4BitHfQuantizer):
            return False
        elif (
            current is not None
            and current is not UnslothBnb4BitHfQuantizer
            and issubclass(current, Bnb4BitHfQuantizer)
        ):
            mapping[key] = type(
                "UnslothBnb4BitHfQuantizer", (UnslothBnb4BitHfQuantizer, current), {}
            )
    _installed = True
    return True
