# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Load a compressed-tensors weight-only integer checkpoint (W4A16 / W8A16,
``pack-quantized``) straight into bitsandbytes 4-bit.

A checkpoint published in the compressed-tensors packed integer format cannot be
trained as published: PEFT has no ``weight`` to attach a LoRA to, only
``weight_packed``, and the decompression hook compressed-tensors registers
expands the whole model to 16-bit on the first forward. The choice so far was
to load it as published (and fail at ``get_peft_model``) or to decompress the
whole checkpoint to a 16-bit copy on disk first, which for a 600 GB INT4
checkpoint means 2 TB of scratch space.

This module does the decompression per tensor while the checkpoint streams
through ``from_pretrained``: the packed weight, its scale, shape, zero point and
activation-order index are collected by a transformers ``WeightConverter``,
unpacked to the model dtype with compressed-tensors' own decompressor, and
handed to the bitsandbytes quantize op in the same loading step. The 16-bit
tensor never exists for more than one layer at a time, and the result is the
ordinary Unsloth QLoRA model: ``Linear4bit`` everywhere the checkpoint was
packed, the LoRA-friendly ``weight`` name, and no extra disk.

Native MoE models whose per-expert projections transformers merges into 3-D
expert stacks keep transformers' own ``DecompressExperts`` op for the merge;
the per-Linear converter below only catches packed weights nothing else claims
(remote-code MoE models with an ``nn.ModuleList`` of expert ``Linear`` modules,
dense models, attention).

Needs the transformers 5 weight-conversion loader; on older transformers the
loader keeps its previous behaviour (load as published).
"""

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

# Private attribute the loader parks the plan on the model config under; the quantizer
# consumes and deletes it before the first weight loads so it never reaches a saved config.
UNSLOTH_COMPRESSED_TENSORS_ATTR = "_unsloth_compressed_tensors_bnb"

_SUPPORTED_FORMATS = ("pack-quantized",)


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
    """``config`` followed by every config-like attribute nested inside it (text, vision,
    audio sub-configs), depth first, without duplicates."""
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
    """Return the compressed-tensors quantization dict when ``config`` describes a checkpoint
    this module can re-quantize on the fly into bitsandbytes 4-bit, else ``None``.

    Accepted: ``quant_method`` compressed-tensors (or the legacy ``sparseml`` spelling),
    ``format`` pack-quantized, every config group an integer weight-only scheme (no
    activation quantization), and no sparsity compression. FP8, NVFP4, MXFP4, activation
    quantized (W8A8) and sparse checkpoints are left to their own loaders.
    """
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
        if str(weights.get("type", "int")).lower() != "int":
            return None
        if int(weights.get("num_bits", 0)) not in (2, 3, 4, 5, 6, 7, 8):
            return None
        if group.get("input_activations") is not None:
            return None
        if group.get("output_activations") is not None:
            return None
        group_format = group.get("format")
        if group_format is not None and group_format not in _SUPPORTED_FORMATS:
            return None
    sparsity = quant.get("sparsity_config")
    if isinstance(sparsity, dict):
        if str(sparsity.get("format", "dense")) != "dense":
            return None
    return quant


def arm_compressed_tensors_bnb_loading(config, verbose: bool = True) -> Optional[dict]:
    """Prepare ``config`` for a bitsandbytes 4-bit load of a packed compressed-tensors
    checkpoint: drop the checkpoint's own quantization config (so transformers accepts the
    bitsandbytes one), park the plan on the config for the quantizer, and register the
    quantizer subclass. Returns the plan, or ``None`` when the checkpoint does not qualify or
    transformers is too old, in which case ``config`` is untouched."""
    plan = compressed_tensors_bnb_plan(config)
    if plan is None:
        return None
    if not _transformers_supports_weight_converters():
        if verbose:
            print(
                "Unsloth: This checkpoint is compressed-tensors packed INT4/INT8. Re-quantizing it "
                "to bitsandbytes 4-bit on the fly needs transformers >= 5; loading it as published instead."
            )
        return None
    try:
        import compressed_tensors  # noqa: F401
    except Exception:
        if verbose:
            print(
                "Unsloth: This checkpoint is compressed-tensors packed INT4/INT8 but `compressed_tensors` "
                "is not installed; loading it as published. `pip install compressed-tensors` to train it in 4-bit."
            )
        return None
    # The plan is only useful if the installed compressed-tensors can still parse it. Fields do get
    # retired between releases (`actorder = "group"` was removed in 0.19.0 and real published
    # checkpoints carry it), and the parse happens deep inside `from_pretrained`, long after the
    # checkpoint's own quantization config has been dropped from `config` below. Parse it here,
    # while declining is still free.
    try:
        _build_quantization_config(plan)
    except Exception as error:
        if verbose:
            print(
                "Unsloth: This checkpoint is compressed-tensors packed INT4/INT8 but the installed "
                f"compressed-tensors cannot read its quantization config ({type(error).__name__}); "
                "loading it as published. Upgrading or downgrading `compressed-tensors` may help."
            )
        return None
    if not install_compressed_tensors_bnb_quantizer():
        # A foreign quantizer class owns the bitsandbytes slot: it would load the packed
        # tensors without the converters, so leave the checkpoint's own config in place.
        if verbose:
            print(
                "Unsloth: This checkpoint is compressed-tensors packed INT4/INT8 but another library owns "
                "the bitsandbytes quantizer; loading it as published."
            )
        return None
    # Composite configs (Kimi-K2.7: KimiK25Config over a DeepseekV3Config text config) copy the
    # checkpoint's quantization config onto their sub-configs, and transformers also looks at
    # the decoder text config when deciding whether the model is pre-quantized.
    for sub in _config_and_subconfigs(config):
        try:
            delattr(sub, "quantization_config")
        except AttributeError:
            pass
        if "quantization_config" in getattr(sub, "__dict__", {}):
            sub.__dict__.pop("quantization_config", None)
    setattr(config, UNSLOTH_COMPRESSED_TENSORS_ATTR, plan)
    if verbose:
        bits = sorted({int(g["weights"]["num_bits"]) for g in plan["config_groups"].values()})
        print(
            f"Unsloth: Checkpoint is compressed-tensors packed INT{'/'.join(map(str, bits))}. "
            f"Decompressing each weight on the fly and re-quantizing to bitsandbytes 4-bit (no 16-bit copy on disk)."
        )
    return plan


# ---------------------------------------------------------------------------------------------
# Keeping the packed integer tensors out of the loader's dtype cast
# ---------------------------------------------------------------------------------------------
#
# transformers materialises every checkpoint tensor in the model dtype before the conversion
# ops see it, which would turn an int32 ``weight_packed`` into bf16 garbage. The loader's
# ``dtype_plan`` (a name-pattern -> dtype map) is consulted first, and a ``None`` entry means
# "keep the storage dtype", so the packed modules are registered there. The plan is keyed by
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
        # A pickled shard is read whole in the model dtype before any converter sees it, which
        # turns the packed int32 words into bf16 garbage with nothing left to decompress.
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


def _generalize(name: str) -> str:
    """Regex for ``name`` with every numeric path component widened to a digit run."""
    escaped = re.escape(name)
    return re.sub(r"(?<=\\.)\d+(?=\\.|$)", r"\\d+", escaped)


def packed_weight_dtype_plan(keys) -> dict:
    """Map of regexes (one per family of packed modules) to ``None`` for the loader's
    ``dtype_plan``. Exact names are used for any family whose widened pattern would also
    catch a module the checkpoint stores unpacked."""
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


# ---------------------------------------------------------------------------------------------
# The quantizer subclass and its conversion ops
# ---------------------------------------------------------------------------------------------

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
    """The config group that quantized the expert projections a converter collects. With one
    group there is nothing to choose. With several, each source pattern (a glob such as
    ``mlp.experts.*.gate_proj.weight``) is matched as a module name against the groups; the
    projections of one converter must agree, since one op decompresses them all."""
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
    """The config group of one concrete expert bucket. The converter is shared by every
    layer, so with several groups the scheme is resolved from this layer's module names
    (``model.layers.5.mlp.experts.{i}.gate_proj``); experts of one bucket must agree."""
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
    pattern = pattern.lstrip("^")
    # The pattern starts somewhere inside the layer path (``mlp.experts.*.gate_proj``).
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
    """Remove this module's converters from ``model._weight_conversions`` so save_pretrained
    does not reverse them. Returns how many were dropped."""
    conversions = getattr(model, "_weight_conversions", None)
    if not conversions:
        return 0
    kept, dropped = [], 0
    for conv in conversions:
        ops = getattr(conv, "operations", None) or []
        if any(isinstance(op, (_DecompressPackedWeights, _WithOriginalSources)) for op in ops):
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
    """Config group that quantized ``name``: a single group applies to everything; with
    several, match the module name against each group's ``targets`` (regex ``re:`` or class
    name) the way compressed-tensors does."""
    groups = list(ct_config.config_groups.values())
    if len(groups) == 1:
        return groups[0]
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
            if target.startswith("re:"):
                if re.match(target[3:], name):
                    return group
            elif target == class_name or target == "Linear":
                return group
    return groups[0]


def _decompress_one(compressor, scheme, packed, scale, shape, zero_point, g_idx, dtype):
    state = {"weight_packed": packed, "weight_scale": scale}
    if shape is not None and shape.numel():
        state["weight_shape"] = shape
    else:
        pack_factor = 32 // int(scheme.weights.num_bits)
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
    """transformers ``ConversionOps`` that turns the (packed, scale, shape, zero point, g_idx)
    group of one Linear, or of a bucket of expert Linears, into the 16-bit weight."""

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
        fmt = scheme.format or "pack-quantized"
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
            # The loader hands every collected source over as a list of tensors: one entry for a
            # plain Linear, one per expert for a bucket collected by a `*` pattern (in bucket order).
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
                # Downstream merge ops look the bucket up under the packed source pattern.
                out[packed_key] = value
            else:
                out[(target_patterns or ["weight"])[0]] = value
        # Anything the converter collected that is not part of a packed group passes through.
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

    def __init__(self, op, original_sources, weight_sources):
        self.op = op
        self.original_sources = list(original_sources)
        self.weight_sources = list(weight_sources)

    def convert(
        self,
        input_dict,
        source_patterns = None,
        target_patterns = None,
        **kwargs,
    ):
        try:
            from transformers.core_model_loading import MergeModulelist
        except Exception:  # pragma: no cover
            MergeModulelist = ()
        keeps_stack = isinstance(self.op, MergeModulelist) if MergeModulelist else False
        for pattern in self.weight_sources:
            if (pattern + "_packed$") in input_dict and (pattern + "$") in input_dict:
                # Some experts of one bucket packed and others not: the loader hands both
                # subsets over without their indices, so a merge would silently reorder them.
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
                    # A bucket arrives as one pre-stacked tensor. MergeModulelist takes that as
                    # is; every other op expects the per-expert list the loader collected.
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
        return getattr(self.op, name)


_with_sources_classes = {}


def _with_original_sources(op, original_sources, weight_sources):
    """Wrap ``op`` in a subclass of its own class. transformers' ``WeightConverter`` allows a
    many-to-many mapping only when ``operations`` holds an instance of its internal Ernie ops,
    so the adapter must still be one. ``_WithOriginalSources`` comes first in the MRO, so
    ``convert`` is the adapter's; the op's own state stays on ``self.op``."""
    cls = _with_sources_classes.get(type(op))
    if cls is None:
        cls = _with_sources_classes[type(op)] = type(
            "WithOriginalSources" + type(op).__name__, (_WithOriginalSources, type(op)), {}
        )
    return cls(op, original_sources, weight_sources)


_installed = False


def install_compressed_tensors_bnb_quantizer() -> bool:
    """Register a ``Bnb4BitHfQuantizer`` subclass under the bitsandbytes 4-bit quant method that
    adds the packed-weight converters when the model config carries a plan. Idempotent; a
    no-op on transformers without the weight-conversion loader."""
    global _installed
    if _installed:
        return True
    if not _transformers_supports_weight_converters():
        return False
    from transformers.quantizers import auto as quantizers_auto
    from transformers.quantizers.quantizer_bnb_4bit import Bnb4BitHfQuantizer
    from transformers.core_model_loading import ConversionOps, WeightConverter

    # Make the ops real ConversionOps so transformers' isinstance checks are happy.
    op_cls = type("DecompressPackedWeights", (_DecompressPackedWeights, ConversionOps), {})
    with_sources_cls = _with_original_sources

    class UnslothBnb4BitHfQuantizer(Bnb4BitHfQuantizer):
        _unsloth_ct_config = None
        _unsloth_ct_dtype = None
        _unsloth_dtype_plan = None
        _unsloth_plan_dicts = ()

        def _process_model_before_weight_loading(self, model, **kwargs):
            config = getattr(model, "config", None)
            plan = (
                getattr(config, UNSLOTH_COMPRESSED_TENSORS_ATTR, None)
                if config is not None
                else None
            )
            if plan is not None:
                # Read, not consumed: the device-map planner runs this same hook on a meta model
                # built from the same config object before the real load, and a plan taken off
                # the config there left the real load with no converters, so every packed
                # expert was reported missing, re-initialised and bitsandbytes-quantized from
                # random values (Kimi-K2.7-Code on four GPUs). The plan comes off the config
                # once the weights are in, in `_process_model_after_weight_loading`.
                self._unsloth_ct_config = _build_quantization_config(plan)
                dtype = kwargs.get("dtype", None)
                if not isinstance(dtype, torch.dtype):
                    dtype = getattr(model, "dtype", None) or torch.bfloat16
                self._unsloth_ct_dtype = dtype
                # Keep every packed tensor in its storage dtype through materialisation.
                self._unsloth_dtype_plan = packed_weight_dtype_plan(
                    _checkpoint_keys(kwargs.get("checkpoint_files"))
                )
                self._unsloth_plan_dicts = []
                original_get_dtype_plan = model._get_dtype_plan
                quantizer = self

                def _get_dtype_plan(dtype, _original = original_get_dtype_plan):
                    merged = dict(_original(dtype))
                    merged.update(quantizer._unsloth_dtype_plan)
                    quantizer._unsloth_plan_dicts.append(merged)
                    return merged

                model._get_dtype_plan = _get_dtype_plan
            return super()._process_model_before_weight_loading(model, **kwargs)

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
            # The decompression converters are load-only. transformers keeps every converter
            # it used on `model._weight_conversions` and reverses them in save_pretrained, which
            # would file the bitsandbytes (or merged) tensors under the packed names without
            # recreating any of the packed metadata: an unloadable checkpoint. The model is a
            # bitsandbytes model now and saves under bitsandbytes names.
            drop_load_only_conversions(model)
            return super()._process_model_after_weight_loading(model, **kwargs)

        def _unsloth_keep_storage_dtype(self, target_patterns):
            """Merged expert stacks are named after the converter target, not the packed
            source; add them to every dtype plan handed out so far and to future ones."""
            for target in target_patterns:
                key = re.escape(target) + r"$"
                self._unsloth_dtype_plan[key] = None
                for plan in self._unsloth_plan_dicts:
                    plan[key] = None

        def update_weight_conversions(self, weight_conversions):
            ct_config = self._unsloth_ct_config
            if ct_config is None:
                return super().update_weight_conversions(weight_conversions)
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
                        # The plain `.weight` sources stay so that expert layers the checkpoint
                        # left unpacked (its `ignore` list) still merge; anchored, so they do
                        # not also match the `_scale` / `_packed` keys.
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
                            # Only the op that receives the decompressed buckets is adapted; a
                            # later op in the chain (Concatenate after MergeModulelist) gets the
                            # previous op's output under the previous op's own contract.
                            operations = [op_cls(ct_config, dtype, stacked = True, scheme = scheme)]
                            + [
                                with_sources_cls(
                                    conv.operations[0], original_sources, weight_sources
                                )
                            ]
                            + list(conv.operations[1:]),
                        )
                        self._unsloth_keep_storage_dtype(conv._original_target_patterns)
                updated.append(conv)
            # Packed Linears nothing above claimed: remote-code MoE experts, dense layers, attention.
            updated.append(
                WeightConverter(
                    source_patterns = [s + "$" for s in _PACKED_SUFFIXES],
                    target_patterns = "weight",
                    operations = [op_cls(ct_config, dtype, stacked = False)],
                )
            )
            updated.extend(self.get_weight_conversions())
            return updated

    mapping = getattr(quantizers_auto, "AUTO_QUANTIZER_MAPPING", None)
    if mapping is None:
        return False
    for key in ("bitsandbytes_4bit",):
        current = mapping.get(key)
        if current is Bnb4BitHfQuantizer:
            mapping[key] = UnslothBnb4BitHfQuantizer
        elif current is not None and not issubclass(current, Bnb4BitHfQuantizer):
            # Someone else replaced it with a foreign class; do not fight over it.
            return False
        elif (
            current is not None
            and current is not UnslothBnb4BitHfQuantizer
            and issubclass(current, Bnb4BitHfQuantizer)
        ):
            # Another subclass is installed; layer ours on top of it so both behaviours survive.
            mapping[key] = type(
                "UnslothBnb4BitHfQuantizer", (UnslothBnb4BitHfQuantizer, current), {}
            )
    _installed = True
    return True
