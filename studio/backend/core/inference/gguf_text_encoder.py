# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Lazy GGUF text-encoder utilities for diffusion pipelines.

The stock Transformers GGUF loader is a checkpoint converter: it
dequantizes every GGUF tensor into a normal PyTorch tensor before the
model is usable. That defeats the low-VRAM goal for FLUX.2, whose text
encoder is a 24B Mistral backbone. This module keeps the Mistral GGUF
language weights quantized as uint8 buffers and dequantizes only inside
the module forward pass.

The first target is deliberately narrow: FLUX.2 text-to-image prompt
encoding. It implements the language backbone used by
``Mistral3ForConditionalGeneration`` and returns hidden states compatible
with ``Flux2Pipeline._get_mistral_3_small_prompt_embeds``. It does not
implement Pixtral image inputs or prompt upsampling generation.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any
import re
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _reverse_repeat_tuple
from transformers import AutoConfig, PreTrainedModel, T5Config, T5EncoderModel
from transformers.models.mistral3.modeling_mistral3 import Mistral3ForConditionalGeneration
from transformers.models.mistral.modeling_mistral import MistralModel, MistralRotaryEmbedding
from transformers.models.gemma3.modeling_gemma3 import Gemma3ForCausalLM, Gemma3RotaryEmbedding
from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaRotaryEmbedding
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel, Qwen3VLTextRotaryEmbedding


@dataclass(frozen = True)
class _GGUFNameTarget:
    hf_name: str
    reverse_permute_heads: int | None = None
    value_offset: float = 0.0


@dataclass(frozen = True)
class Qwen2VLMmprojTarget:
    hf_name: str
    qkv_part: str | None = None
    stack_index: int | None = None


@dataclass(frozen = True)
class LazyGGUFReplacementStats:
    loaded: int
    lazy_linear: int
    lazy_embedding: int
    materialized: int


@dataclass(frozen = True)
class Qwen2VLMmprojReplacementStats:
    loaded: int
    materialized: int
    qkv_groups: int
    stacked_patch_embeddings: int


@dataclass(frozen = True)
class TextEncoderGGUFInfo:
    path: Path
    architecture: str | None
    model_type: str | None
    supported_by_lazy_loader: bool
    requires_mmproj: bool
    mmproj_path: Path | None


_SUPPORTED_LAZY_TEXT_ARCHITECTURES = frozenset(
    {
        "mistral3",
        "t5",
        "t5encoder",
        "llama",
        "qwen2vl",
        "qwen3",
        "qwen3vl",
        "gemma3",
    }
)
_TEXT_ARCHITECTURES_REQUIRING_MMPROJ = frozenset({"qwen2vl"})


def _normalize_resident_device(device: torch.device | str | None) -> torch.device | None:
    if device is None:
        return None
    return torch.device(device)


def _pin_cpu_resident_gguf_tensors_enabled(pin_memory: bool | None = None) -> bool:
    if pin_memory is not None:
        return bool(pin_memory)
    value = os.environ.get("UNSLOTH_STUDIO_GGUF_PIN_CPU_RESIDENT", "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _pin_cpu_tensor_for_transfer(
    tensor: torch.Tensor,
    *,
    pin_memory: bool | None = None,
) -> torch.Tensor:
    if (
        tensor.device.type != "cpu"
        or not _pin_cpu_resident_gguf_tensors_enabled(pin_memory)
        or not torch.cuda.is_available()
    ):
        return tensor
    try:
        if tensor.is_pinned():
            return tensor
    except Exception:
        pass
    try:
        return tensor.pin_memory()
    except Exception:
        return tensor


def _resident_tensor_needs_refresh(
    tensor: torch.Tensor,
    resident: torch.device,
    *,
    pin_memory: bool | None = None,
) -> bool:
    if tensor.device != resident:
        return True
    if resident.type != "cpu" or not _pin_cpu_resident_gguf_tensors_enabled(pin_memory):
        return False
    try:
        return not tensor.is_pinned()
    except Exception:
        return False


def _copy_gguf_parameter_to_device(
    param: Any,
    device: Any,
    *,
    pin_memory: bool | None = None,
) -> Any:
    """Copy an upstream GGUFParameter while preserving quant metadata."""

    quant_type = getattr(param, "quant_type", None)
    quant_shape = getattr(param, "quant_shape", None)
    moved = param.detach().to(device, non_blocking = True)
    moved = _pin_cpu_tensor_for_transfer(moved, pin_memory = pin_memory)
    param_cls = type(param)
    try:
        copied = param_cls(
            moved,
            requires_grad = getattr(param, "requires_grad", False),
            quant_type = quant_type,
        )
    except Exception:
        try:
            from diffusers.quantizers.gguf.utils import GGUFParameter

            copied = GGUFParameter(
                moved,
                requires_grad = getattr(param, "requires_grad", False),
                quant_type = quant_type,
            )
        except Exception:
            copied = nn.Parameter(
                moved,
                requires_grad = getattr(param, "requires_grad", False),
            )
    if quant_shape is not None:
        try:
            copied.quant_shape = tuple(int(dim) for dim in quant_shape)
        except Exception:
            copied.quant_shape = quant_shape
    return copied


def _first_tensor_device(value: Any) -> torch.device | None:
    if isinstance(value, torch.Tensor):
        return value.device
    if isinstance(value, dict):
        for item in value.values():
            device = _first_tensor_device(item)
            if device is not None:
                return device
    if isinstance(value, (list, tuple)):
        for item in value:
            device = _first_tensor_device(item)
            if device is not None:
                return device
    return None


def _module_and_leaf(root: nn.Module, name: str) -> tuple[nn.Module, str]:
    parts = name.split(".")
    module = root
    for part in parts[:-1]:
        module = getattr(module, part)
    return module, parts[-1]


def _open_gguf_reader(path: Path) -> Any:
    gguf = _require_gguf()
    return gguf.GGUFReader(str(path))


def _read_gguf_scalar_field(reader: Any, field_name: str, field_type: type) -> Any:
    field = reader.get_field(field_name)
    if field is None:
        return None
    if field_type is str:
        value = field.parts[field.data[-1]]
        if isinstance(value, str):
            return value
        if isinstance(value, bytes):
            return value.decode("utf-8")
        if isinstance(value, (bytearray, memoryview)):
            return bytes(value).decode("utf-8")
        tobytes = getattr(value, "tobytes", None)
        if callable(tobytes):
            try:
                return tobytes().decode("utf-8")
            except Exception:
                pass
        return str(value)
    if field_type in (int, float, bool):
        value = field.parts[field.data[-1]]
        if hasattr(value, "item"):
            value = value.item()
        return field_type(value)
    raise TypeError(f"Unsupported GGUF field type: {field_type!r}")


def strip_gguf_quant_suffix(name: str) -> str:
    """Remove common GGUF quant suffixes from a filename stem.

    This is used before looking for a sibling mmproj file, e.g.
    ``Qwen2.5-VL-7B-Instruct-Q4_K_M`` becomes
    ``Qwen2.5-VL-7B-Instruct``.
    """

    pattern = r"[-_]?(?:ud-)?i?q[0-9]_[a-z0-9_\-]{1,8}$"
    match = re.search(pattern, name, re.IGNORECASE)
    if match:
        return name[: match.start()]
    return name


def resolve_text_encoder_mmproj_gguf(path: str | Path) -> Path | None:
    """Resolve a sibling mmproj GGUF for text encoders that need one."""

    text_path = Path(path)
    root = text_path.parent
    if not root.is_dir():
        return None
    text_stem = strip_gguf_quant_suffix(text_path.stem).lower()
    matches: list[Path] = []
    generic_matches: list[Path] = []
    for candidate in sorted(root.iterdir(), key = lambda p: p.name.lower()):
        if not candidate.is_file() or candidate.suffix.lower() != ".gguf":
            continue
        candidate_stem = candidate.stem.lower()
        if "mmproj" not in candidate_stem:
            continue
        if text_stem and text_stem in candidate_stem:
            matches.append(candidate)
        elif candidate_stem.startswith("mmproj"):
            generic_matches.append(candidate)
    if matches:
        return matches[0]
    if len(generic_matches) == 1:
        return generic_matches[0]
    return None


def inspect_text_encoder_gguf(path: str | Path) -> TextEncoderGGUFInfo:
    gguf_path = Path(path)
    reader = _open_gguf_reader(gguf_path)
    architecture = _read_gguf_scalar_field(reader, "general.architecture", str)
    model_type = _read_gguf_scalar_field(reader, "general.type", str)
    if isinstance(architecture, str):
        architecture = architecture.lower()
    if isinstance(model_type, str):
        model_type = model_type.lower()
    requires_mmproj = architecture in _TEXT_ARCHITECTURES_REQUIRING_MMPROJ
    return TextEncoderGGUFInfo(
        path = gguf_path,
        architecture = architecture,
        model_type = model_type,
        supported_by_lazy_loader = architecture in _SUPPORTED_LAZY_TEXT_ARCHITECTURES,
        requires_mmproj = requires_mmproj,
        mmproj_path = resolve_text_encoder_mmproj_gguf(gguf_path) if requires_mmproj else None,
    )


def map_llama_style_text_gguf_name(
    name: str,
    *,
    root_prefix: str = "",
    include_lm_head: bool = True,
    reverse_permute_qk: bool = False,
    num_attention_heads: int | None = None,
    num_key_value_heads: int | None = None,
) -> _GGUFNameTarget | None:
    """Map llama.cpp text GGUF tensor names to Transformers module names.

    The llama.cpp GGUF names are shared across llama, qwen2vl, qwen3,
    and qwen3vl text backbones. This helper makes that mapping explicit
    so architecture-specific lazy loaders can target the right
    module/parameter without materializing a full state dict.
    """

    prefix = root_prefix
    if prefix and not prefix.endswith("."):
        prefix = f"{prefix}."

    if name == "token_embd.weight":
        return _GGUFNameTarget(f"{prefix}embed_tokens.weight")
    if name == "output_norm.weight":
        return _GGUFNameTarget(f"{prefix}norm.weight")
    if name == "output.weight":
        if include_lm_head:
            return _GGUFNameTarget("lm_head.weight")
        return None

    parts = name.split(".")
    if len(parts) != 4 or parts[0] != "blk" or parts[3] not in {"weight", "bias"}:
        return None

    layer = parts[1]
    leaf = parts[2]
    suffix = parts[3]
    layer_prefix = f"{prefix}layers.{layer}"
    mapping = {
        "attn_q": f"{layer_prefix}.self_attn.q_proj.{suffix}",
        "attn_k": f"{layer_prefix}.self_attn.k_proj.{suffix}",
        "attn_v": f"{layer_prefix}.self_attn.v_proj.{suffix}",
        "attn_output": f"{layer_prefix}.self_attn.o_proj.{suffix}",
        "attn_q_norm": f"{layer_prefix}.self_attn.q_norm.{suffix}",
        "attn_k_norm": f"{layer_prefix}.self_attn.k_norm.{suffix}",
        "attn_v_norm": f"{layer_prefix}.self_attn.v_norm.{suffix}",
        "ffn_gate": f"{layer_prefix}.mlp.gate_proj.{suffix}",
        "ffn_up": f"{layer_prefix}.mlp.up_proj.{suffix}",
        "ffn_down": f"{layer_prefix}.mlp.down_proj.{suffix}",
        "attn_norm": f"{layer_prefix}.input_layernorm.{suffix}",
        "ffn_norm": f"{layer_prefix}.post_attention_layernorm.{suffix}",
    }
    hf_name = mapping.get(leaf)
    if hf_name is None:
        return None

    reverse_heads = None
    if reverse_permute_qk and suffix == "weight":
        if leaf == "attn_q":
            reverse_heads = num_attention_heads
        elif leaf == "attn_k":
            reverse_heads = num_key_value_heads
    return _GGUFNameTarget(hf_name, reverse_heads)


def map_qwen2vl_text_gguf_name(name: str) -> _GGUFNameTarget | None:
    return map_llama_style_text_gguf_name(
        name,
        root_prefix = "model.language_model",
        include_lm_head = True,
        reverse_permute_qk = False,
    )


def map_qwen3_text_gguf_name(name: str) -> _GGUFNameTarget | None:
    return map_llama_style_text_gguf_name(
        name,
        root_prefix = "model",
        include_lm_head = False,
        reverse_permute_qk = False,
    )


def map_llama_text_gguf_name(
    name: str,
    *,
    num_attention_heads: int | None = None,
    num_key_value_heads: int | None = None,
) -> _GGUFNameTarget | None:
    return map_llama_style_text_gguf_name(
        name,
        root_prefix = "model",
        include_lm_head = False,
        reverse_permute_qk = True,
        num_attention_heads = num_attention_heads,
        num_key_value_heads = num_key_value_heads,
    )


def map_qwen3vl_text_gguf_name(name: str) -> _GGUFNameTarget | None:
    return map_llama_style_text_gguf_name(
        name,
        root_prefix = "",
        include_lm_head = False,
        reverse_permute_qk = False,
    )


def map_gemma3_text_gguf_name(name: str) -> _GGUFNameTarget | None:
    if name == "token_embd.weight":
        return _GGUFNameTarget("model.embed_tokens.weight")
    if name == "output_norm.weight":
        return _GGUFNameTarget("model.norm.weight", value_offset = -1.0)
    if name == "output.weight":
        return None

    parts = name.split(".")
    if len(parts) != 4 or parts[0] != "blk" or parts[3] not in {"weight", "bias"}:
        return None

    layer = parts[1]
    leaf = parts[2]
    suffix = parts[3]
    prefix = f"model.layers.{layer}"
    mapping = {
        "attn_q": f"{prefix}.self_attn.q_proj.{suffix}",
        "attn_k": f"{prefix}.self_attn.k_proj.{suffix}",
        "attn_v": f"{prefix}.self_attn.v_proj.{suffix}",
        "attn_output": f"{prefix}.self_attn.o_proj.{suffix}",
        "attn_q_norm": f"{prefix}.self_attn.q_norm.{suffix}",
        "attn_k_norm": f"{prefix}.self_attn.k_norm.{suffix}",
        "attn_v_norm": f"{prefix}.self_attn.v_norm.{suffix}",
        "ffn_gate": f"{prefix}.mlp.gate_proj.{suffix}",
        "ffn_up": f"{prefix}.mlp.up_proj.{suffix}",
        "ffn_down": f"{prefix}.mlp.down_proj.{suffix}",
        "attn_norm": f"{prefix}.input_layernorm.{suffix}",
        "post_attention_norm": f"{prefix}.post_attention_layernorm.{suffix}",
        "ffn_norm": f"{prefix}.pre_feedforward_layernorm.{suffix}",
        "post_ffw_norm": f"{prefix}.post_feedforward_layernorm.{suffix}",
    }
    hf_name = mapping.get(leaf)
    if hf_name is None:
        return None
    needs_norm_correction = (
        suffix == "weight"
        and leaf
        in {
            "attn_norm",
            "post_attention_norm",
            "ffn_norm",
            "post_ffw_norm",
            "attn_q_norm",
            "attn_k_norm",
        }
    )
    return _GGUFNameTarget(
        hf_name,
        value_offset = -1.0 if needs_norm_correction else 0.0,
    )


def map_t5_text_gguf_name(name: str) -> _GGUFNameTarget | None:
    """Map T5/T5Encoder GGUF tensor names to HF T5 names."""

    if name == "token_embd.weight":
        return _GGUFNameTarget("shared.weight")
    if name in {"output_norm.weight", "enc.output_norm.weight"}:
        return _GGUFNameTarget("encoder.final_layer_norm.weight")

    parts = name.split(".")
    if len(parts) != 5 or parts[0] != "enc" or parts[1] != "blk" or parts[4] not in {"weight", "bias"}:
        return None

    layer = parts[2]
    leaf = parts[3]
    suffix = parts[4]
    prefix = f"encoder.block.{layer}"
    mapping = {
        "attn_q": f"{prefix}.layer.0.SelfAttention.q.{suffix}",
        "attn_k": f"{prefix}.layer.0.SelfAttention.k.{suffix}",
        "attn_v": f"{prefix}.layer.0.SelfAttention.v.{suffix}",
        "attn_o": f"{prefix}.layer.0.SelfAttention.o.{suffix}",
        "attn_rel_b": f"{prefix}.layer.0.SelfAttention.relative_attention_bias.{suffix}",
        "attn_norm": f"{prefix}.layer.0.layer_norm.{suffix}",
        "ffn_gate": f"{prefix}.layer.1.DenseReluDense.wi_0.{suffix}",
        "ffn_up": f"{prefix}.layer.1.DenseReluDense.wi_1.{suffix}",
        "ffn_down": f"{prefix}.layer.1.DenseReluDense.wo.{suffix}",
        "ffn_norm": f"{prefix}.layer.1.layer_norm.{suffix}",
    }
    hf_name = mapping.get(leaf)
    if hf_name is None:
        return None
    return _GGUFNameTarget(hf_name)


def map_mistral3_causal_gguf_name(
    name: str,
    *,
    num_attention_heads: int | None = None,
    num_key_value_heads: int | None = None,
) -> _GGUFNameTarget | None:
    """Map Mistral3/Ministral3 causal GGUF names to HF module names."""

    return map_llama_style_text_gguf_name(
        name,
        root_prefix = "model",
        include_lm_head = True,
        reverse_permute_qk = True,
        num_attention_heads = num_attention_heads,
        num_key_value_heads = num_key_value_heads,
    )


def map_qwen2vl_mmproj_gguf_name(name: str) -> Qwen2VLMmprojTarget | None:
    """Map Qwen2VL mmproj GGUF tensor names to Transformers names.

    Qwen2VL mmproj files store the visual tower separately from the text
    GGUF. Most tensors are one-to-one after applying the GGUF-to-HF
    name mapping, while `attn_q/k/v` must be grouped into HF's fused
    `attn.qkv` tensor and dual `v.patch_embd.weight*` tensors must be
    stacked.
    """

    if name == "v.post_ln.weight":
        return Qwen2VLMmprojTarget("model.visual.merger.ln_q.weight")
    if name.startswith("mm."):
        return Qwen2VLMmprojTarget(f"model.visual.merger.mlp.{name.removeprefix('mm.')}")
    if name == "v.patch_embd.weight":
        return Qwen2VLMmprojTarget("model.visual.patch_embed.proj.weight", stack_index = 0)
    if name == "v.patch_embd.weight.1":
        return Qwen2VLMmprojTarget("model.visual.patch_embed.proj.weight", stack_index = 1)

    parts = name.split(".")
    if len(parts) != 5 or parts[0] != "v" or parts[1] != "blk" or parts[4] not in {"weight", "bias"}:
        return None

    layer = parts[2]
    leaf = parts[3]
    suffix = parts[4]
    prefix = f"model.visual.blocks.{layer}"
    if leaf in {"attn_q", "attn_k", "attn_v"}:
        return Qwen2VLMmprojTarget(
            f"{prefix}.attn.qkv.{suffix}",
            qkv_part = leaf.removeprefix("attn_"),
        )
    mapping = {
        "attn_out": f"{prefix}.attn.proj.{suffix}",
        "ffn_gate": f"{prefix}.mlp.gate_proj.{suffix}",
        "ffn_up": f"{prefix}.mlp.up_proj.{suffix}",
        "ffn_down": f"{prefix}.mlp.down_proj.{suffix}",
        "ln1": f"{prefix}.norm1.{suffix}",
        "ln2": f"{prefix}.norm2.{suffix}",
    }
    hf_name = mapping.get(leaf)
    if hf_name is None:
        return None
    return Qwen2VLMmprojTarget(hf_name)


def patch_gguf_text_encoder_for_resident_device(
    root: Any,
    resident_device: Any,
    *,
    pin_memory: bool | None = None,
) -> int:
    """Keep GGUF text-encoder weights resident on a chosen device.

    This covers two native Studio cases:

    * the hand-written lazy modules in this file, which store quantized
      GGUF bytes in ``qweight`` / ``qbias`` buffers and dequantize inside
      forward;
    * upstream Diffusers GGUF modules, which expose a ``weight`` parameter
      with ``quant_type`` metadata and dequantize / fused-matmul inside
      their own forward path.

    The second case follows the same native residency contract as
    diffusion transformers: module ``.to(...)`` calls should not
    permanently move quantized bytes away from the offload device, while
    forward briefly copies the active quantized weight to the activation
    device before upstream dequantization or fused execution.
    """

    import types

    if root is None or not hasattr(root, "modules"):
        return 0

    resident = torch.device(resident_device)
    pin_cpu_resident = _pin_cpu_resident_gguf_tensors_enabled(pin_memory)
    patched = 0

    for module in root.modules():
        buffers = getattr(module, "_buffers", {})
        if hasattr(module, "_resident_device") and any(name in buffers for name in ("qweight", "qbias")):
            module._resident_device = resident
            module._pin_cpu_resident = pin_cpu_resident
            if hasattr(module, "_place_resident_qweight"):
                module._place_resident_qweight()
            patched += 1
            continue

        weight = getattr(module, "weight", None)
        if weight is None or not hasattr(weight, "quant_type"):
            continue
        if not hasattr(module, "forward_native") and type(module).__name__ != "GGUFLinear":
            continue

        if getattr(module, "_unsloth_gguf_resident_patched", False):
            module._unsloth_gguf_resident_device = resident
            module._unsloth_gguf_pin_cpu_resident = pin_cpu_resident
            for param_name, current in list(module._parameters.items()):
                if current is None or not hasattr(current, "quant_type"):
                    continue
                if _resident_tensor_needs_refresh(
                    current,
                    resident,
                    pin_memory = pin_cpu_resident,
                ):
                    module._parameters[param_name] = _copy_gguf_parameter_to_device(
                        current,
                        resident,
                        pin_memory = pin_cpu_resident,
                    )
            patched += 1
            continue

        original_apply = module._apply
        original_forward = module.forward
        module._unsloth_gguf_resident_device = resident
        module._unsloth_gguf_pin_cpu_resident = pin_cpu_resident
        module._unsloth_gguf_original_apply = original_apply
        module._unsloth_gguf_original_forward = original_forward

        def _resident_apply(self, fn, _original_apply = original_apply):
            resident_params = {
                name: self._parameters.pop(name)
                for name, param in list(self._parameters.items())
                if param is not None and hasattr(param, "quant_type")
            }
            try:
                result = _original_apply(fn)
            finally:
                for name, resident_param in resident_params.items():
                    if _resident_tensor_needs_refresh(
                        resident_param,
                        self._unsloth_gguf_resident_device,
                        pin_memory = getattr(self, "_unsloth_gguf_pin_cpu_resident", False),
                    ):
                        resident_param = _copy_gguf_parameter_to_device(
                            resident_param,
                            self._unsloth_gguf_resident_device,
                            pin_memory = getattr(self, "_unsloth_gguf_pin_cpu_resident", False),
                        )
                    self._parameters[name] = resident_param
            return result

        def _resident_forward(self, *args, _original_forward = original_forward, **kwargs):
            target_device = _first_tensor_device(args)
            if target_device is None:
                target_device = _first_tensor_device(kwargs)
            original_params: dict[str, Any] = {}
            if target_device is not None and target_device.type != "meta":
                for name, param in list(self._parameters.items()):
                    if param is None or not hasattr(param, "quant_type"):
                        continue
                    if param.device == target_device:
                        continue
                    original_params[name] = param
                    self._parameters[name] = _copy_gguf_parameter_to_device(
                        param,
                        target_device,
                        pin_memory = False,
                    )
            if original_params:
                try:
                    return _original_forward(*args, **kwargs)
                finally:
                    self._parameters.update(original_params)
            return _original_forward(*args, **kwargs)

        module._apply = types.MethodType(_resident_apply, module)
        module.forward = types.MethodType(_resident_forward, module)
        module._unsloth_gguf_resident_patched = True
        for param_name, param in list(module._parameters.items()):
            if param is None or not hasattr(param, "quant_type"):
                continue
            if _resident_tensor_needs_refresh(
                param,
                resident,
                pin_memory = pin_cpu_resident,
            ):
                module._parameters[param_name] = _copy_gguf_parameter_to_device(
                    param,
                    resident,
                    pin_memory = pin_cpu_resident,
                )
        patched += 1
    return patched


class _LazyGGUFOffloadMixin:
    _resident_device: torch.device | None
    _pin_cpu_resident: bool

    def _place_resident_qweight(self) -> None:
        if self._resident_device is None:
            return
        for name in ("qweight", "qbias"):
            qbuffer = self._buffers.get(name)
            if qbuffer is None:
                continue
            if qbuffer.device != self._resident_device:
                qbuffer = qbuffer.to(self._resident_device)
            self._buffers[name] = _pin_cpu_tensor_for_transfer(
                qbuffer,
                pin_memory = getattr(self, "_pin_cpu_resident", None),
            )

    def _compute_quant_buffer(self, name: str, target_device: torch.device) -> torch.Tensor:
        qbuffer = self._buffers[name]
        if qbuffer.device == target_device:
            return qbuffer
        return qbuffer.to(target_device, non_blocking = True)

    def _compute_qweight(self, target_device: torch.device) -> torch.Tensor:
        return self._compute_quant_buffer("qweight", target_device)

    def _apply(self, fn):
        qbuffer_names = [name for name in ("qweight", "qbias") if name in self._buffers]
        if self._resident_device is None or not qbuffer_names:
            return super()._apply(fn)

        qbuffers = {name: self._buffers.pop(name) for name in qbuffer_names}
        try:
            result = super()._apply(fn)
        finally:
            self._buffers.update(qbuffers)
            self._place_resident_qweight()
        return result


def _require_gguf():
    try:
        import gguf  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "Lazy FLUX.2 text-encoder GGUF loading requires the gguf package."
        ) from exc
    return gguf


def _qwen2_5_vl_model_class():
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "Qwen2VL text-encoder GGUF loading requires a Transformers build "
            "with Qwen2_5_VLForConditionalGeneration."
        ) from exc
    return Qwen2_5_VLForConditionalGeneration


def _qwen2_5_vl_rotary_classes():
    try:
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
            Qwen2_5_VisionRotaryEmbedding,
            Qwen2_5_VLRotaryEmbedding,
        )
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "Qwen2VL text-encoder GGUF loading requires a Transformers build "
            "with Qwen2.5-VL rotary modules."
        ) from exc
    return Qwen2_5_VLRotaryEmbedding, Qwen2_5_VisionRotaryEmbedding


def _qwen3_rotary_class():
    try:
        from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "Qwen3 text-encoder GGUF loading requires a Transformers build "
            "with Qwen3RotaryEmbedding."
        ) from exc
    return Qwen3RotaryEmbedding


def _materialize_qwen2vl_rotary_buffers(text_encoder: nn.Module, config: Any) -> None:
    """Replace Qwen2.5-VL rotary modules created on ``meta`` with CPU buffers."""

    text_rotary_cls, vision_rotary_cls = _qwen2_5_vl_rotary_classes()
    text_encoder.model.language_model.rotary_emb = text_rotary_cls(config.text_config)
    for layer in getattr(text_encoder.model.language_model, "layers", []):
        self_attn = getattr(layer, "self_attn", None)
        if self_attn is not None and hasattr(self_attn, "rotary_emb"):
            self_attn.rotary_emb = text_rotary_cls(config.text_config)

    vision_config = config.vision_config
    head_dim = vision_config.hidden_size // vision_config.num_heads
    text_encoder.model.visual.rotary_pos_emb = vision_rotary_cls(head_dim // 2)


def _materialize_qwen3_rotary_buffers(text_encoder: nn.Module, config: Any) -> None:
    """Replace Qwen3 rotary modules created on ``meta`` with CPU buffers."""

    text_encoder.model.rotary_emb = _qwen3_rotary_class()(config)


def _materialize_llama_rotary_buffers(text_encoder: nn.Module, config: Any) -> None:
    text_encoder.model.rotary_emb = LlamaRotaryEmbedding(config)


def _materialize_qwen3vl_rotary_buffers(text_encoder: nn.Module, config: Any) -> None:
    text_encoder.rotary_emb = Qwen3VLTextRotaryEmbedding(config = config)


def _ensure_qwen3vl_rope_scaling(config: Any) -> None:
    if getattr(config, "rope_scaling", None) is None:
        config.rope_scaling = {"rope_type": "default", "mrope_section": [24, 20, 20]}
    elif "mrope_section" not in config.rope_scaling:
        config.rope_scaling = {**config.rope_scaling, "mrope_section": [24, 20, 20]}


def _materialize_gemma3_buffers(text_encoder: nn.Module, config: Any) -> None:
    if hasattr(text_encoder.model.embed_tokens, "embed_scale"):
        text_encoder.model.embed_tokens.embed_scale = torch.tensor(
            float(config.hidden_size) ** 0.5,
            dtype = torch.float32,
        )
    text_encoder.model.rotary_emb = Gemma3RotaryEmbedding(config = config)
    local_config = copy.deepcopy(config)
    local_config.rope_theta = local_config.rope_local_base_freq
    local_config.rope_scaling = {"rope_type": "default"}
    text_encoder.model.rotary_emb_local = Gemma3RotaryEmbedding(config = local_config)


def _materialize_ministral3_rotary_buffers(text_encoder: nn.Module, config: Any) -> None:
    try:
        from transformers.models.ministral3.modeling_ministral3 import (
            Ministral3RotaryEmbedding,
        )
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "Ministral3 GGUF loading requires a Transformers build with "
            "Ministral3RotaryEmbedding."
        ) from exc
    text_encoder.model.rotary_emb = Ministral3RotaryEmbedding(config)


def _gguf_field_scalar(reader: Any, name: str, default: Any = None) -> Any:
    field = reader.get_field(name)
    if field is None:
        return default
    value = field.parts[field.data[-1]]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "tolist"):
        value = value.tolist()
        while isinstance(value, list) and len(value) == 1:
            value = value[0]
    return value


def _infer_t5_config_from_gguf(reader: Any) -> T5Config:
    vocab_size = None
    d_model = int(_gguf_field_scalar(reader, "t5encoder.embedding_length", 0) or 0)
    d_ff = int(_gguf_field_scalar(reader, "t5encoder.feed_forward_length", 0) or 0)
    num_layers = int(_gguf_field_scalar(reader, "t5encoder.block_count", 0) or 0)
    num_heads = int(_gguf_field_scalar(reader, "t5encoder.attention.head_count", 0) or 0)
    d_kv = int(_gguf_field_scalar(reader, "t5encoder.attention.key_length", 0) or 0)
    relative_attention_num_buckets = int(
        _gguf_field_scalar(reader, "t5encoder.attention.relative_buckets_count", 32) or 32
    )
    layer_norm_epsilon = float(
        _gguf_field_scalar(
            reader,
            "t5encoder.attention.layer_norm_rms_epsilon",
            _gguf_field_scalar(reader, "t5encoder.attention.layer_norm_epsilon", 1e-6),
        )
        or 1e-6
    )

    has_gate = False
    max_layer = -1
    for tensor in reader.tensors:
        if tensor.name == "token_embd.weight":
            shape = _gguf_tensor_logical_shape(tensor, reader = reader)
            if len(shape) == 2:
                vocab_size, d_model_from_tensor = shape
                if not d_model:
                    d_model = int(d_model_from_tensor)
        elif tensor.name.endswith(".ffn_gate.weight"):
            has_gate = True
        if tensor.name.startswith("enc.blk."):
            parts = tensor.name.split(".")
            if len(parts) >= 3 and parts[2].isdigit():
                max_layer = max(max_layer, int(parts[2]))
        if tensor.name.endswith(".ffn_up.weight") and not d_ff:
            shape = _gguf_tensor_logical_shape(tensor, reader = reader)
            if len(shape) == 2:
                d_ff = int(shape[0])

    if not num_layers and max_layer >= 0:
        num_layers = max_layer + 1
    if not d_kv and d_model and num_heads:
        d_kv = d_model // num_heads
    missing = [
        name
        for name, value in {
            "vocab_size": vocab_size,
            "d_model": d_model,
            "d_ff": d_ff,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "d_kv": d_kv,
        }.items()
        if not value
    ]
    if missing:
        raise RuntimeError(
            "Could not infer a T5 encoder config from GGUF metadata; "
            f"missing {', '.join(missing)}."
        )

    return T5Config(
        vocab_size = int(vocab_size),
        d_model = int(d_model),
        d_ff = int(d_ff),
        num_layers = int(num_layers),
        num_heads = int(num_heads),
        d_kv = int(d_kv),
        feed_forward_proj = "gated-gelu" if has_gate else "relu",
        dense_act_fn = "gelu_new" if has_gate else "relu",
        is_encoder_decoder = False,
        is_decoder = False,
        use_cache = False,
        dropout_rate = 0.0,
        layer_norm_epsilon = layer_norm_epsilon,
        relative_attention_num_buckets = relative_attention_num_buckets,
    )


def _dequantize_gguf_bytes(
    qweight: torch.Tensor,
    quant_type: Any,
    *,
    dtype: torch.dtype | None = None,
    logical_shape: tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Dequantize a GGUF uint8 tensor without materializing at load time."""

    from diffusers.quantizers.gguf.utils import (
        GGML_QUANT_SIZES,
        dequantize_functions,
    )

    block_size, type_size = GGML_QUANT_SIZES[quant_type]
    shape = logical_shape or (*qweight.shape[:-1], qweight.shape[-1] // type_size * block_size)
    blocks = qweight.view(torch.uint8).reshape((-1, type_size))
    dequant = dequantize_functions[quant_type](blocks, block_size, type_size, dtype)
    return dequant.reshape(shape)


def _reverse_permute_qk(weight: torch.Tensor, num_heads: int) -> torch.Tensor:
    """Match Transformers' GGUF q/k tensor post-processing lazily.

    This mirrors ``LlamaTensorProcessor._reverse_permute_weights`` in
    Transformers' GGUF converter, but runs on the dequantized temporary
    weight inside the forward pass so the resident weight stays quantized.
    """

    dim = weight.shape[0] // num_heads // 2
    return weight.reshape(num_heads, dim, 2, *weight.shape[1:]).transpose(1, 2).reshape(weight.shape)


class LazyGGUFLinear(_LazyGGUFOffloadMixin, nn.Module):
    def __init__(
        self,
        qweight: torch.Tensor,
        quant_type: Any,
        *,
        in_features: int,
        out_features: int,
        compute_dtype: torch.dtype,
        reverse_permute_heads: int | None = None,
        resident_device: torch.device | str | None = None,
        bias: torch.Tensor | None = None,
        qbias: torch.Tensor | None = None,
        bias_quant_type: Any | None = None,
        bias_logical_shape: tuple[int, ...] | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.quant_type = quant_type
        self.compute_dtype = compute_dtype
        self.reverse_permute_heads = reverse_permute_heads
        self.bias_quant_type = bias_quant_type
        self.bias_logical_shape = (
            tuple(bias_logical_shape) if bias_logical_shape is not None else None
        )
        self._resident_device = _normalize_resident_device(resident_device)
        self._pin_cpu_resident = _pin_cpu_resident_gguf_tensors_enabled()
        self.register_buffer(
            "qweight",
            qweight if qweight.is_contiguous() else qweight.contiguous(),
            persistent = False,
        )
        if qbias is not None:
            self.register_buffer(
                "qbias",
                qbias if qbias.is_contiguous() else qbias.contiguous(),
                persistent = False,
            )
            self.register_parameter("bias", None)
        elif bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(bias.to(dtype = compute_dtype), requires_grad = False)
        self._place_resident_qweight()

    @property
    def weight(self) -> torch.Tensor:
        """Metadata-only compatibility shim for modules that inspect weight.dtype.

        Some Diffusers model code reads ``linear.weight.dtype`` to pick an
        activation dtype without using the weight tensor itself. Returning
        the packed ``qweight`` would report ``uint8`` and materializing the
        dense weight here would defeat lazy GGUF residency, so expose a
        zero-element tensor with the compute dtype instead.
        """

        device = self.qweight.device
        if device.type == "meta":
            device = torch.device("cpu")
        return torch.empty(0, device = device, dtype = self.compute_dtype)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        qweight = self._compute_qweight(inputs.device)
        # Match diffusers' GGUFLinear.forward_native: dequantize in the
        # dequantizer default precision, then cast to compute dtype.
        weight = _dequantize_gguf_bytes(qweight, self.quant_type)
        if self.reverse_permute_heads is not None:
            weight = _reverse_permute_qk(weight, self.reverse_permute_heads)
        weight = weight.to(dtype = self.compute_dtype)
        if "qbias" in self._buffers:
            qbias = self._compute_quant_buffer("qbias", inputs.device)
            bias = _dequantize_gguf_bytes(
                qbias,
                self.bias_quant_type,
                logical_shape = self.bias_logical_shape,
            )
        else:
            bias = self.bias
        if bias is not None and self.reverse_permute_heads is not None:
            bias = _reverse_permute_qk(bias, self.reverse_permute_heads)
        if bias is not None:
            bias = bias.to(device = inputs.device, dtype = self.compute_dtype)
        return F.linear(inputs, weight, bias)


class LazyGGUFEmbedding(_LazyGGUFOffloadMixin, nn.Module):
    def __init__(
        self,
        qweight: torch.Tensor,
        quant_type: Any,
        *,
        num_embeddings: int,
        embedding_dim: int,
        compute_dtype: torch.dtype,
        resident_device: torch.device | str | None = None,
        output_scale: torch.Tensor | float | None = None,
        padding_idx: int | None = None,
        max_norm: float | None = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.quant_type = quant_type
        self.compute_dtype = compute_dtype
        self._resident_device = _normalize_resident_device(resident_device)
        self._pin_cpu_resident = _pin_cpu_resident_gguf_tensors_enabled()
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse
        if output_scale is not None:
            if isinstance(output_scale, torch.Tensor):
                scale = output_scale.detach().to(device = "cpu", dtype = torch.float32)
            else:
                scale = torch.tensor(float(output_scale), dtype = torch.float32)
            self.register_buffer("output_scale", scale, persistent = False)
        else:
            self.output_scale = None
        self.register_buffer(
            "qweight",
            qweight if qweight.is_contiguous() else qweight.contiguous(),
            persistent = False,
        )
        self._place_resident_qweight()

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        flat = input_ids.reshape(-1)
        unique_ids, inverse = torch.unique(flat, sorted = True, return_inverse = True)
        rows = self.qweight.index_select(0, unique_ids.to(self.qweight.device))
        if rows.device != input_ids.device:
            rows = rows.to(input_ids.device, non_blocking = True)
        weight = _dequantize_gguf_bytes(rows, self.quant_type, dtype = self.compute_dtype)
        weight = weight.to(dtype = self.compute_dtype)
        if self.max_norm is not None:
            local_padding_idx = None
            if self.padding_idx is not None:
                matches = (unique_ids == int(self.padding_idx)).nonzero(as_tuple = False)
                if matches.numel():
                    local_padding_idx = int(matches[0].item())
            gathered = F.embedding(
                inverse.to(weight.device),
                weight,
                local_padding_idx,
                self.max_norm,
                self.norm_type,
                self.scale_grad_by_freq,
                self.sparse,
            )
        else:
            gathered = weight.index_select(0, inverse.to(weight.device))
        output = gathered.reshape(*input_ids.shape, self.embedding_dim)
        if self.output_scale is not None:
            output = output * self.output_scale.to(device = output.device, dtype = output.dtype)
        return output


class LazyGGUFConv2d(_LazyGGUFOffloadMixin, nn.Module):
    """GGUF-aware Conv2d that keeps packed weights and dequantizes in forward."""

    def __init__(
        self,
        qweight: torch.Tensor,
        quant_type: Any,
        *,
        in_channels: int,
        out_channels: int,
        kernel_size: Any,
        stride: Any,
        padding: Any,
        dilation: Any,
        groups: int,
        compute_dtype: torch.dtype,
        padding_mode: str = "zeros",
        resident_device: torch.device | str | None = None,
        bias: torch.Tensor | None = None,
        qbias: torch.Tensor | None = None,
        bias_quant_type: Any | None = None,
        bias_logical_shape: tuple[int, ...] | None = None,
        logical_shape: tuple[int, ...] | None = None,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = int(groups)
        self.padding_mode = padding_mode
        self._reversed_padding_repeated_twice = _reverse_repeat_tuple(self.padding, 2)
        self.quant_type = quant_type
        self.compute_dtype = compute_dtype
        self.logical_shape = tuple(logical_shape) if logical_shape is not None else None
        self.bias_quant_type = bias_quant_type
        self.bias_logical_shape = (
            tuple(bias_logical_shape) if bias_logical_shape is not None else None
        )
        self._resident_device = _normalize_resident_device(resident_device)
        self._pin_cpu_resident = _pin_cpu_resident_gguf_tensors_enabled()
        self.register_buffer(
            "qweight",
            qweight if qweight.is_contiguous() else qweight.contiguous(),
            persistent = False,
        )
        if qbias is not None:
            self.register_buffer(
                "qbias",
                qbias if qbias.is_contiguous() else qbias.contiguous(),
                persistent = False,
            )
            self.register_parameter("bias", None)
        elif bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(bias.to(dtype = compute_dtype), requires_grad = False)
        self._place_resident_qweight()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        qweight = self._compute_qweight(inputs.device)
        weight = _dequantize_gguf_bytes(
            qweight,
            self.quant_type,
            dtype = self.compute_dtype,
            logical_shape = self.logical_shape,
        ).to(device = inputs.device, dtype = inputs.dtype)
        if "qbias" in self._buffers:
            qbias = self._compute_quant_buffer("qbias", inputs.device)
            bias = _dequantize_gguf_bytes(
                qbias,
                self.bias_quant_type,
                dtype = self.compute_dtype,
                logical_shape = self.bias_logical_shape,
            ).to(device = inputs.device, dtype = inputs.dtype)
        else:
            bias = self.bias
        if bias is not None:
            bias = bias.to(device = inputs.device, dtype = inputs.dtype)
        if self.padding_mode != "zeros":
            inputs = F.pad(inputs, self._reversed_padding_repeated_twice, mode = self.padding_mode)
            padding = _pair(0)
        else:
            padding = self.padding
        return F.conv2d(
            inputs,
            weight,
            bias,
            self.stride,
            padding,
            self.dilation,
            self.groups,
        )


class _LazyGGUFNormMixin(_LazyGGUFOffloadMixin):
    def _init_lazy_norm_parameter(
        self,
        *,
        name: str,
        qvalue: torch.Tensor | None,
        quant_type: Any | None,
        dense_value: torch.Tensor | None,
        compute_dtype: torch.dtype,
        logical_shape: tuple[int, ...] | None,
    ) -> None:
        setattr(self, f"{name}_quant_type", quant_type)
        setattr(self, f"{name}_logical_shape", tuple(logical_shape) if logical_shape is not None else None)
        qbuffer_name = f"q{name}"
        if qvalue is not None:
            self.register_buffer(
                qbuffer_name,
                qvalue if qvalue.is_contiguous() else qvalue.contiguous(),
                persistent = False,
            )
            self.register_parameter(name, None)
        elif dense_value is not None:
            setattr(
                self,
                name,
                nn.Parameter(dense_value.to(dtype = compute_dtype), requires_grad = False),
            )
        else:
            self.register_parameter(name, None)

    def _norm_parameter(self, name: str, inputs: torch.Tensor) -> torch.Tensor | None:
        qbuffer_name = f"q{name}"
        if qbuffer_name in self._buffers:
            qbuffer = self._compute_quant_buffer(qbuffer_name, inputs.device)
            quant_type = getattr(self, f"{name}_quant_type")
            logical_shape = getattr(self, f"{name}_logical_shape")
            value = _dequantize_gguf_bytes(
                qbuffer,
                quant_type,
                dtype = self.compute_dtype,
                logical_shape = logical_shape,
            )
            return value.to(device = inputs.device, dtype = inputs.dtype)
        value = getattr(self, name, None)
        if value is None:
            return None
        return value.to(device = inputs.device, dtype = inputs.dtype)


class LazyGGUFLayerNorm(_LazyGGUFNormMixin, nn.Module):
    """GGUF-aware LayerNorm that dequantizes affine params in forward."""

    def __init__(
        self,
        *,
        normalized_shape: Any,
        eps: float,
        compute_dtype: torch.dtype,
        resident_device: torch.device | str | None = None,
        qweight: torch.Tensor | None = None,
        weight_quant_type: Any | None = None,
        weight: torch.Tensor | None = None,
        weight_logical_shape: tuple[int, ...] | None = None,
        qbias: torch.Tensor | None = None,
        bias_quant_type: Any | None = None,
        bias: torch.Tensor | None = None,
        bias_logical_shape: tuple[int, ...] | None = None,
    ) -> None:
        super().__init__()
        self.normalized_shape = (
            (int(normalized_shape),)
            if isinstance(normalized_shape, int)
            else tuple(normalized_shape)
        )
        self.eps = float(eps)
        self.compute_dtype = compute_dtype
        self._resident_device = _normalize_resident_device(resident_device)
        self._pin_cpu_resident = _pin_cpu_resident_gguf_tensors_enabled()
        self._init_lazy_norm_parameter(
            name = "weight",
            qvalue = qweight,
            quant_type = weight_quant_type,
            dense_value = weight,
            compute_dtype = compute_dtype,
            logical_shape = weight_logical_shape,
        )
        self._init_lazy_norm_parameter(
            name = "bias",
            qvalue = qbias,
            quant_type = bias_quant_type,
            dense_value = bias,
            compute_dtype = compute_dtype,
            logical_shape = bias_logical_shape,
        )
        self._place_resident_qweight()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        weight = self._norm_parameter("weight", inputs)
        bias = self._norm_parameter("bias", inputs)
        return F.layer_norm(inputs, self.normalized_shape, weight, bias, self.eps)


class LazyGGUFGroupNorm(_LazyGGUFNormMixin, nn.Module):
    """GGUF-aware GroupNorm that dequantizes affine params in forward."""

    def __init__(
        self,
        *,
        num_groups: int,
        num_channels: int,
        eps: float,
        compute_dtype: torch.dtype,
        resident_device: torch.device | str | None = None,
        qweight: torch.Tensor | None = None,
        weight_quant_type: Any | None = None,
        weight: torch.Tensor | None = None,
        weight_logical_shape: tuple[int, ...] | None = None,
        qbias: torch.Tensor | None = None,
        bias_quant_type: Any | None = None,
        bias: torch.Tensor | None = None,
        bias_logical_shape: tuple[int, ...] | None = None,
    ) -> None:
        super().__init__()
        self.num_groups = int(num_groups)
        self.num_channels = int(num_channels)
        self.eps = float(eps)
        self.compute_dtype = compute_dtype
        self._resident_device = _normalize_resident_device(resident_device)
        self._pin_cpu_resident = _pin_cpu_resident_gguf_tensors_enabled()
        self._init_lazy_norm_parameter(
            name = "weight",
            qvalue = qweight,
            quant_type = weight_quant_type,
            dense_value = weight,
            compute_dtype = compute_dtype,
            logical_shape = weight_logical_shape,
        )
        self._init_lazy_norm_parameter(
            name = "bias",
            qvalue = qbias,
            quant_type = bias_quant_type,
            dense_value = bias,
            compute_dtype = compute_dtype,
            logical_shape = bias_logical_shape,
        )
        self._place_resident_qweight()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        weight = self._norm_parameter("weight", inputs)
        bias = self._norm_parameter("bias", inputs)
        return F.group_norm(inputs, self.num_groups, weight, bias, self.eps)


class LazyFlux2MistralTextEncoder(Mistral3ForConditionalGeneration):
    """Text-only lazy GGUF replacement for FLUX.2's Mistral3 text encoder."""

    def __init__(self, config: Any, language_model: MistralModel, *, compute_dtype: torch.dtype) -> None:
        PreTrainedModel.__init__(self, config)
        self.language_model = language_model
        self.compute_dtype = compute_dtype

    @property
    def dtype(self) -> torch.dtype:
        return self.compute_dtype

    @property
    def device(self) -> torch.device:
        for param in self.parameters(recurse = True):
            return param.device
        for name, tensor in self.named_buffers(recurse = True):
            if name.rsplit(".", 1)[-1] in {"qweight", "qbias"}:
                continue
            return tensor.device
        for tensor in self.buffers(recurse = True):
            return tensor.device
        return torch.device("cpu")

    def forward(self, *args, **kwargs):
        return self.language_model(*args, **kwargs)

    @classmethod
    def from_gguf(
        cls,
        gguf_path: str | Path,
        *,
        base_repo_or_path: str | Path,
        compute_dtype: torch.dtype = torch.bfloat16,
        resident_device: torch.device | str | None = None,
        token: str | None = None,
    ) -> "LazyFlux2MistralTextEncoder":
        base_path = Path(base_repo_or_path).expanduser()
        if base_path.exists():
            config = AutoConfig.from_pretrained(base_path / "text_encoder", token = token)
        else:
            config = AutoConfig.from_pretrained(
                str(base_repo_or_path),
                subfolder = "text_encoder",
                token = token,
            )
        with torch.device("meta"):
            language_model = MistralModel(config.text_config)
        language_model.rotary_emb = MistralRotaryEmbedding(config.text_config)

        gguf_reader = _replace_mistral_modules_with_lazy_gguf(
            language_model,
            gguf_path = Path(gguf_path),
            compute_dtype = compute_dtype,
            resident_device = resident_device,
        )
        language_model.requires_grad_(False)
        text_encoder = cls(config, language_model, compute_dtype = compute_dtype)
        # Keep mmap-backed NumPy arrays from GGUFReader alive for any
        # quantized tensors that were wrapped without a CPU copy.
        text_encoder._gguf_reader = gguf_reader
        return text_encoder


class LazyMistral3TextEncoder(LazyFlux2MistralTextEncoder):
    """Generic hidden-state Mistral3 GGUF encoder for diffusion prompts."""


class LazyMinistral3PromptEnhancer:
    """Factory for ERNIE's Ministral3 prompt-enhancer GGUF component."""

    @classmethod
    def from_gguf(
        cls,
        gguf_path: str | Path,
        *,
        base_repo_or_path: str | Path,
        subfolder: str = "pe",
        compute_dtype: torch.dtype = torch.bfloat16,
        resident_device: torch.device | str | None = None,
        token: str | None = None,
    ) -> Any:
        try:
            from transformers.models.ministral3.modeling_ministral3 import (
                Ministral3ForCausalLM,
            )
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise RuntimeError(
                "ERNIE prompt-enhancer GGUF loading requires a Transformers "
                "build with Ministral3ForCausalLM."
            ) from exc

        base_path = Path(base_repo_or_path).expanduser()
        if base_path.exists():
            config = AutoConfig.from_pretrained(base_path / subfolder, token = token)
        else:
            config = AutoConfig.from_pretrained(
                str(base_repo_or_path),
                subfolder = subfolder,
                token = token,
            )

        with torch.device("meta"):
            prompt_enhancer = Ministral3ForCausalLM(config)
        _materialize_ministral3_rotary_buffers(prompt_enhancer, config)

        gguf_reader, stats = replace_mapped_text_modules_with_lazy_gguf(
            prompt_enhancer,
            Path(gguf_path),
            map_name = lambda name: map_mistral3_causal_gguf_name(
                name,
                num_attention_heads = getattr(config, "num_attention_heads", None),
                num_key_value_heads = getattr(config, "num_key_value_heads", None),
            ),
            compute_dtype = compute_dtype,
            resident_device = resident_device,
        )
        prompt_enhancer.requires_grad_(False)
        prompt_enhancer._gguf_reader = gguf_reader
        prompt_enhancer._gguf_text_replacement_stats = stats
        return prompt_enhancer


class LazyQwen2VLTextEncoder:
    """Factory for native Qwen2VL text GGUF + mmproj loading."""

    @classmethod
    def from_gguf(
        cls,
        gguf_path: str | Path,
        *,
        base_repo_or_path: str | Path,
        mmproj_gguf_path: str | Path | None = None,
        compute_dtype: torch.dtype = torch.bfloat16,
        resident_device: torch.device | str | None = None,
        token: str | None = None,
    ) -> Any:
        gguf_path = Path(gguf_path)
        if mmproj_gguf_path is None:
            mmproj_gguf_path = resolve_text_encoder_mmproj_gguf(gguf_path)
        if mmproj_gguf_path is None:
            raise RuntimeError(
                "Qwen2VL text GGUF loading requires a sibling mmproj GGUF file."
            )
        mmproj_gguf_path = Path(mmproj_gguf_path)

        base_path = Path(base_repo_or_path).expanduser()
        if base_path.exists():
            config = AutoConfig.from_pretrained(base_path / "text_encoder", token = token)
        else:
            config = AutoConfig.from_pretrained(
                str(base_repo_or_path),
                subfolder = "text_encoder",
                token = token,
            )

        model_cls = _qwen2_5_vl_model_class()
        with torch.device("meta"):
            text_encoder = model_cls(config)
        _materialize_qwen2vl_rotary_buffers(text_encoder, config)

        text_reader, text_stats = replace_mapped_text_modules_with_lazy_gguf(
            text_encoder,
            gguf_path,
            map_name = map_qwen2vl_text_gguf_name,
            compute_dtype = compute_dtype,
            resident_device = resident_device,
        )
        mmproj_reader, mmproj_stats = replace_qwen2vl_mmproj_modules_with_gguf(
            text_encoder,
            mmproj_gguf_path,
            compute_dtype = compute_dtype,
        )
        text_encoder.requires_grad_(False)
        # Keep mmap-backed GGUFReader instances alive for lazily wrapped
        # text tensors and for parity with the FLUX.2 lazy loader.
        text_encoder._gguf_reader = text_reader
        text_encoder._mmproj_gguf_reader = mmproj_reader
        text_encoder._gguf_text_replacement_stats = text_stats
        text_encoder._gguf_mmproj_replacement_stats = mmproj_stats
        text_encoder._gguf_mmproj_path = mmproj_gguf_path
        return text_encoder


class LazyQwen3TextEncoder(Qwen3ForCausalLM):
    """Lazy GGUF Qwen3 text encoder for Z-Image pipelines."""

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    @classmethod
    def from_gguf(
        cls,
        gguf_path: str | Path,
        *,
        base_repo_or_path: str | Path,
        compute_dtype: torch.dtype = torch.bfloat16,
        resident_device: torch.device | str | None = None,
        token: str | None = None,
    ) -> "LazyQwen3TextEncoder":
        base_path = Path(base_repo_or_path).expanduser()
        if base_path.exists():
            config = AutoConfig.from_pretrained(base_path / "text_encoder", token = token)
        else:
            config = AutoConfig.from_pretrained(
                str(base_repo_or_path),
                subfolder = "text_encoder",
                token = token,
            )

        with torch.device("meta"):
            text_encoder = cls(config)
        text_encoder.lm_head = nn.Identity()
        _materialize_qwen3_rotary_buffers(text_encoder, config)

        gguf_reader, stats = replace_mapped_text_modules_with_lazy_gguf(
            text_encoder,
            Path(gguf_path),
            map_name = map_qwen3_text_gguf_name,
            compute_dtype = compute_dtype,
            resident_device = resident_device,
        )
        text_encoder.requires_grad_(False)
        text_encoder._gguf_reader = gguf_reader
        text_encoder._gguf_text_replacement_stats = stats
        return text_encoder


class LazyLlamaTextEncoder(LlamaForCausalLM):
    """Lazy GGUF llama-family text encoder using hidden-state outputs."""

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    @classmethod
    def from_gguf(
        cls,
        gguf_path: str | Path,
        *,
        base_repo_or_path: str | Path,
        subfolder: str = "text_encoder",
        compute_dtype: torch.dtype = torch.bfloat16,
        resident_device: torch.device | str | None = None,
        token: str | None = None,
    ) -> "LazyLlamaTextEncoder":
        base_path = Path(base_repo_or_path).expanduser()
        if base_path.exists():
            config = AutoConfig.from_pretrained(base_path / subfolder, token = token)
        else:
            config = AutoConfig.from_pretrained(
                str(base_repo_or_path),
                subfolder = subfolder,
                token = token,
            )
        if hasattr(config, "text_config"):
            config = config.text_config

        with torch.device("meta"):
            text_encoder = cls(config)
        text_encoder.lm_head = nn.Identity()
        _materialize_llama_rotary_buffers(text_encoder, config)

        gguf_reader, stats = replace_mapped_text_modules_with_lazy_gguf(
            text_encoder,
            Path(gguf_path),
            map_name = lambda name: map_llama_text_gguf_name(
                name,
                num_attention_heads = getattr(config, "num_attention_heads", None),
                num_key_value_heads = getattr(config, "num_key_value_heads", None),
            ),
            compute_dtype = compute_dtype,
            resident_device = resident_device,
        )
        text_encoder.requires_grad_(False)
        text_encoder._gguf_reader = gguf_reader
        text_encoder._gguf_text_replacement_stats = stats
        return text_encoder


class LazyQwen3VLTextEncoder(Qwen3VLTextModel):
    """Lazy GGUF Qwen3VL text-only encoder."""

    @classmethod
    def from_gguf(
        cls,
        gguf_path: str | Path,
        *,
        base_repo_or_path: str | Path,
        subfolder: str = "text_encoder",
        compute_dtype: torch.dtype = torch.bfloat16,
        resident_device: torch.device | str | None = None,
        token: str | None = None,
    ) -> "LazyQwen3VLTextEncoder":
        base_path = Path(base_repo_or_path).expanduser()
        if base_path.exists():
            config = AutoConfig.from_pretrained(base_path / subfolder, token = token)
        else:
            config = AutoConfig.from_pretrained(
                str(base_repo_or_path),
                subfolder = subfolder,
                token = token,
            )
        if hasattr(config, "text_config"):
            config = config.text_config
        _ensure_qwen3vl_rope_scaling(config)

        with torch.device("meta"):
            text_encoder = cls(config)
        _materialize_qwen3vl_rotary_buffers(text_encoder, config)

        gguf_reader, stats = replace_mapped_text_modules_with_lazy_gguf(
            text_encoder,
            Path(gguf_path),
            map_name = map_qwen3vl_text_gguf_name,
            compute_dtype = compute_dtype,
            resident_device = resident_device,
        )
        text_encoder.requires_grad_(False)
        text_encoder._gguf_reader = gguf_reader
        text_encoder._gguf_text_replacement_stats = stats
        return text_encoder


class LazyGemma3TextEncoder(Gemma3ForCausalLM):
    """Lazy GGUF Gemma3 text encoder using hidden-state outputs."""

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    @classmethod
    def from_gguf(
        cls,
        gguf_path: str | Path,
        *,
        base_repo_or_path: str | Path,
        subfolder: str = "text_encoder",
        compute_dtype: torch.dtype = torch.bfloat16,
        resident_device: torch.device | str | None = None,
        token: str | None = None,
    ) -> "LazyGemma3TextEncoder":
        base_path = Path(base_repo_or_path).expanduser()
        if base_path.exists():
            config = AutoConfig.from_pretrained(base_path / subfolder, token = token)
        else:
            config = AutoConfig.from_pretrained(
                str(base_repo_or_path),
                subfolder = subfolder,
                token = token,
            )
        if hasattr(config, "text_config"):
            config = config.text_config

        with torch.device("meta"):
            text_encoder = cls(config)
        text_encoder.lm_head = nn.Identity()
        _materialize_gemma3_buffers(text_encoder, config)

        gguf_reader, stats = replace_mapped_text_modules_with_lazy_gguf(
            text_encoder,
            Path(gguf_path),
            map_name = map_gemma3_text_gguf_name,
            compute_dtype = compute_dtype,
            resident_device = resident_device,
        )
        text_encoder.requires_grad_(False)
        text_encoder._gguf_reader = gguf_reader
        text_encoder._gguf_text_replacement_stats = stats
        return text_encoder


class LazyT5TextEncoder(T5EncoderModel):
    """Lazy GGUF T5/T5Encoder text encoder for FLUX.1 and SD3-style pipelines."""

    @classmethod
    def from_gguf(
        cls,
        gguf_path: str | Path,
        *,
        base_repo_or_path: str | Path | None = None,
        subfolder: str = "text_encoder",
        compute_dtype: torch.dtype = torch.bfloat16,
        resident_device: torch.device | str | None = None,
        token: str | None = None,
    ) -> "LazyT5TextEncoder":
        config = None
        if base_repo_or_path is not None:
            base_path = Path(base_repo_or_path).expanduser()
            if base_path.exists():
                config = AutoConfig.from_pretrained(base_path / subfolder, token = token)
            else:
                config = AutoConfig.from_pretrained(
                    str(base_repo_or_path),
                    subfolder = subfolder,
                    token = token,
                )
        if config is None:
            config_reader = _open_gguf_reader(Path(gguf_path))
            config = _infer_t5_config_from_gguf(config_reader)

        with torch.device("meta"):
            text_encoder = cls(config)

        gguf_reader, stats = replace_mapped_text_modules_with_lazy_gguf(
            text_encoder,
            Path(gguf_path),
            map_name = map_t5_text_gguf_name,
            compute_dtype = compute_dtype,
            resident_device = resident_device,
        )
        if text_encoder.shared is not text_encoder.encoder.embed_tokens:
            text_encoder.encoder.embed_tokens = text_encoder.shared
        text_encoder.requires_grad_(False)
        text_encoder._gguf_reader = gguf_reader
        text_encoder._gguf_text_replacement_stats = stats
        return text_encoder


def _target_for_gguf_name(name: str, *, num_attention_heads: int, num_key_value_heads: int) -> _GGUFNameTarget | None:
    if name == "token_embd.weight":
        return _GGUFNameTarget("embed_tokens.weight")
    if name == "output_norm.weight":
        return _GGUFNameTarget("norm.weight")
    if name == "output.weight":
        # FLUX.2 prompt embeddings never use lm_head / generation.
        return None

    parts = name.split(".")
    if len(parts) != 4 or parts[0] != "blk" or parts[3] not in {"weight", "bias"}:
        return None

    layer = parts[1]
    leaf = parts[2]
    suffix = parts[3]
    prefix = f"layers.{layer}"
    mapping = {
        "attn_q": _GGUFNameTarget(f"{prefix}.self_attn.q_proj.{suffix}", num_attention_heads),
        "attn_k": _GGUFNameTarget(f"{prefix}.self_attn.k_proj.{suffix}", num_key_value_heads),
        "attn_v": _GGUFNameTarget(f"{prefix}.self_attn.v_proj.{suffix}"),
        "attn_output": _GGUFNameTarget(f"{prefix}.self_attn.o_proj.{suffix}"),
        "ffn_gate": _GGUFNameTarget(f"{prefix}.mlp.gate_proj.{suffix}"),
        "ffn_up": _GGUFNameTarget(f"{prefix}.mlp.up_proj.{suffix}"),
        "ffn_down": _GGUFNameTarget(f"{prefix}.mlp.down_proj.{suffix}"),
        "attn_norm": _GGUFNameTarget(f"{prefix}.input_layernorm.{suffix}"),
        "ffn_norm": _GGUFNameTarget(f"{prefix}.post_attention_layernorm.{suffix}"),
    }
    return mapping.get(leaf)


def _gguf_full_precision_types(gguf: Any) -> set[Any]:
    return {
        gguf.GGMLQuantizationType.F32,
        gguf.GGMLQuantizationType.F16,
    }


def _gguf_bfloat16_type(gguf: Any) -> Any:
    return getattr(gguf.GGMLQuantizationType, "BF16", None)


def _read_gguf_orig_shape_metadata(reader: Any, tensor_name: str) -> tuple[int, ...] | None:
    """Read optional original-shape metadata for a GGUF tensor."""

    if reader is None or not tensor_name:
        return None
    get_field = getattr(reader, "get_field", None)
    if not callable(get_field):
        return None
    field = get_field(f"comfy.gguf.orig_shape.{tensor_name}")
    if field is None:
        return None
    try:
        dims = []
        for part_idx in field.data:
            value = field.parts[part_idx]
            if hasattr(value, "item"):
                value = value.item()
            elif isinstance(value, (list, tuple)):
                value = value[0]
                if hasattr(value, "item"):
                    value = value.item()
            else:
                try:
                    value = value[0]
                    if hasattr(value, "item"):
                        value = value.item()
                except Exception:
                    pass
            dims.append(int(value))
        return tuple(dims)
    except Exception:
        return None


def _gguf_tensor_logical_shape(tensor: Any, *, reader: Any = None) -> tuple[int, ...]:
    orig_shape = _read_gguf_orig_shape_metadata(reader, getattr(tensor, "name", ""))
    if orig_shape is not None:
        return orig_shape
    return tuple(int(v) for v in reversed(tensor.shape))


def _torch_from_gguf_data(tensor: Any, *, copy: bool) -> torch.Tensor:
    data = tensor.data
    if isinstance(data, torch.Tensor):
        return data.clone() if copy else data
    if copy:
        data = data.copy()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message = "The given NumPy array is not writable.*")
        return torch.from_numpy(data)


def _materialize_gguf_tensor(
    tensor: Any,
    *,
    compute_dtype: torch.dtype,
    gguf: Any,
    reader: Any = None,
) -> torch.Tensor:
    raw = _torch_from_gguf_data(tensor, copy = True)
    qtype = tensor.tensor_type
    if qtype in _gguf_full_precision_types(gguf):
        shape = _gguf_tensor_logical_shape(tensor, reader = reader)
        numel = 1
        for dim in shape:
            numel *= dim
        if raw.numel() == numel:
            raw = raw.reshape(shape)
        return raw.to(dtype = compute_dtype)
    if qtype == _gguf_bfloat16_type(gguf):
        return _dequantize_gguf_bytes(
            raw,
            qtype,
            dtype = compute_dtype,
            logical_shape = _gguf_tensor_logical_shape(tensor, reader = reader),
        ).to(dtype = compute_dtype)
    return _dequantize_gguf_bytes(
        raw,
        qtype,
        dtype = compute_dtype,
        logical_shape = _gguf_tensor_logical_shape(tensor, reader = reader),
    ).to(dtype = compute_dtype)


def _apply_gguf_target_value_transform(
    value: torch.Tensor,
    target: _GGUFNameTarget,
) -> torch.Tensor:
    if target.value_offset:
        value = value + value.new_tensor(target.value_offset)
    return value


def replace_mapped_text_modules_with_lazy_gguf(
    root: nn.Module,
    gguf_path: str | Path,
    *,
    map_name: Any,
    compute_dtype: torch.dtype,
    resident_device: torch.device | str | None = None,
) -> tuple[Any, LazyGGUFReplacementStats]:
    """Replace mapped GGUF tensors in a Transformers model with lazy modules.

    The mapper returns HF state-dict names for a specific GGUF text
    architecture. Quantized Linear/Embedding weights are kept packed and
    replaced with lazy modules; small full-precision tensors and biases
    are materialized as frozen parameters. This is the architecture-neutral
    primitive needed by Qwen2VL/Qwen3/Gemma/T5 loaders.
    """

    gguf = _require_gguf()
    reader = _open_gguf_reader(Path(gguf_path))
    expected = set(root.state_dict().keys())
    mapped: dict[str, tuple[Any, _GGUFNameTarget]] = {}
    for tensor in reader.tensors:
        target = map_name(tensor.name)
        if target is None:
            continue
        if target.hf_name not in expected:
            if target.hf_name.endswith(".DenseReluDense.wi_1.weight"):
                relu_hf_name = target.hf_name.replace(
                    ".DenseReluDense.wi_1.weight",
                    ".DenseReluDense.wi.weight",
                )
                if relu_hf_name in expected:
                    target = _GGUFNameTarget(
                        relu_hf_name,
                        reverse_permute_heads = target.reverse_permute_heads,
                        value_offset = target.value_offset,
                    )
                else:
                    continue
            else:
                continue
        mapped[target.hf_name] = (tensor, target)

    loaded: set[str] = set()
    lazy_linear = 0
    lazy_embedding = 0
    materialized = 0

    for hf_name, (tensor, target) in mapped.items():
        if hf_name in loaded or hf_name.endswith(".bias"):
            continue

        parent, leaf = _module_and_leaf(root, hf_name)
        qtype = tensor.tensor_type
        if qtype in _gguf_full_precision_types(gguf) or qtype == _gguf_bfloat16_type(gguf):
            value = _materialize_gguf_tensor(
                tensor,
                compute_dtype = compute_dtype,
                gguf = gguf,
                reader = reader,
            )
            value = _apply_gguf_target_value_transform(value, target)
            setattr(parent, leaf, nn.Parameter(value, requires_grad = False))
            loaded.add(hf_name)
            materialized += 1
            continue

        if target.value_offset:
            value = _materialize_gguf_tensor(
                tensor,
                compute_dtype = compute_dtype,
                gguf = gguf,
                reader = reader,
            )
            value = _apply_gguf_target_value_transform(value, target)
            setattr(parent, leaf, nn.Parameter(value, requires_grad = False))
            loaded.add(hf_name)
            materialized += 1
            continue

        if leaf == "weight" and isinstance(parent, nn.Linear):
            bias_name = hf_name.removesuffix(".weight") + ".bias"
            bias = None
            qbias = None
            bias_quant_type = None
            bias_logical_shape = None
            if bias_name in mapped:
                bias_tensor, _bias_target = mapped[bias_name]
                bias_qtype = bias_tensor.tensor_type
                if (
                    _bias_target.value_offset
                    or bias_qtype in _gguf_full_precision_types(gguf)
                    or bias_qtype == _gguf_bfloat16_type(gguf)
                ):
                    bias = _materialize_gguf_tensor(
                        bias_tensor,
                        compute_dtype = compute_dtype,
                        gguf = gguf,
                        reader = reader,
                    )
                    bias = _apply_gguf_target_value_transform(bias, _bias_target)
                    materialized += 1
                else:
                    qbias = _torch_from_gguf_data(bias_tensor, copy = False)
                    bias_quant_type = bias_qtype
                    bias_logical_shape = _gguf_tensor_logical_shape(bias_tensor, reader = reader)
                loaded.add(bias_name)
            raw = _torch_from_gguf_data(tensor, copy = False)
            lazy = LazyGGUFLinear(
                raw,
                qtype,
                in_features = parent.in_features,
                out_features = parent.out_features,
                compute_dtype = compute_dtype,
                reverse_permute_heads = target.reverse_permute_heads,
                resident_device = resident_device,
                bias = bias,
                qbias = qbias,
                bias_quant_type = bias_quant_type,
                bias_logical_shape = bias_logical_shape,
            )
            grandparent_name = hf_name.removesuffix(".weight")
            grandparent, module_name = _module_and_leaf(root, grandparent_name)
            setattr(grandparent, module_name, lazy)
            loaded.add(hf_name)
            lazy_linear += 1
            continue

        if leaf == "weight" and isinstance(parent, nn.Embedding):
            raw = _torch_from_gguf_data(tensor, copy = False)
            output_scale = getattr(parent, "embed_scale", None)
            lazy = LazyGGUFEmbedding(
                raw,
                qtype,
                num_embeddings = parent.num_embeddings,
                embedding_dim = parent.embedding_dim,
                compute_dtype = compute_dtype,
                resident_device = resident_device,
                output_scale = output_scale,
            )
            grandparent_name = hf_name.removesuffix(".weight")
            grandparent, module_name = _module_and_leaf(root, grandparent_name)
            setattr(grandparent, module_name, lazy)
            loaded.add(hf_name)
            lazy_embedding += 1
            continue

        raise RuntimeError(f"Unsupported mapped GGUF target: {hf_name}")

    return reader, LazyGGUFReplacementStats(
        loaded = len(loaded),
        lazy_linear = lazy_linear,
        lazy_embedding = lazy_embedding,
        materialized = materialized,
    )


def replace_qwen2vl_mmproj_modules_with_gguf(
    root: nn.Module,
    mmproj_gguf_path: str | Path,
    *,
    compute_dtype: torch.dtype,
) -> tuple[Any, Qwen2VLMmprojReplacementStats]:
    """Load Qwen2VL mmproj GGUF tensors into a Transformers visual tower.

    This helper handles two Qwen2VL mmproj tensor layout differences
    before loading the visual module:

    * `v.patch_embd.weight` and `v.patch_embd.weight.1` are stacked into
      HF's 5D `visual.patch_embed.proj.weight`;
    * split `attn_q`, `attn_k`, and `attn_v` tensors are concatenated
      into HF's fused `attn.qkv` weight/bias.

    This helper performs those layout conversions while using the same
    native GGUF materialization path as the lazy text replacement
    primitive. It returns the live GGUF reader so mmap-backed tensors stay
    alive for callers that later add lazy mmproj modules.
    """

    gguf = _require_gguf()
    reader = _open_gguf_reader(Path(mmproj_gguf_path))
    expected = set(root.state_dict().keys())
    plain: dict[str, Any] = {}
    qkv: dict[str, dict[str, Any]] = {}
    stacked: dict[str, dict[int, Any]] = {}

    for tensor in reader.tensors:
        target = map_qwen2vl_mmproj_gguf_name(tensor.name)
        if target is None or target.hf_name not in expected:
            continue
        if target.qkv_part is not None:
            qkv.setdefault(target.hf_name, {})[target.qkv_part] = tensor
        elif target.stack_index is not None:
            stacked.setdefault(target.hf_name, {})[target.stack_index] = tensor
        else:
            plain[target.hf_name] = tensor

    loaded = 0
    materialized = 0
    qkv_groups = 0
    stacked_patch_embeddings = 0

    for hf_name, tensor in plain.items():
        parent, leaf = _module_and_leaf(root, hf_name)
        value = _materialize_gguf_tensor(
            tensor,
            compute_dtype = compute_dtype,
            gguf = gguf,
            reader = reader,
        )
        setattr(parent, leaf, nn.Parameter(value, requires_grad = False))
        loaded += 1
        materialized += 1

    for hf_name, parts in qkv.items():
        if not {"q", "k", "v"}.issubset(parts):
            missing = sorted({"q", "k", "v"} - set(parts))
            raise RuntimeError(f"Qwen2VL mmproj qkv target {hf_name} is missing {missing}")
        parent, leaf = _module_and_leaf(root, hf_name)
        value = torch.cat(
            [
                _materialize_gguf_tensor(
                    parts["q"],
                    compute_dtype = compute_dtype,
                    gguf = gguf,
                    reader = reader,
                ),
                _materialize_gguf_tensor(
                    parts["k"],
                    compute_dtype = compute_dtype,
                    gguf = gguf,
                    reader = reader,
                ),
                _materialize_gguf_tensor(
                    parts["v"],
                    compute_dtype = compute_dtype,
                    gguf = gguf,
                    reader = reader,
                ),
            ],
            dim = 0,
        )
        setattr(parent, leaf, nn.Parameter(value, requires_grad = False))
        loaded += 3
        materialized += 3
        qkv_groups += 1

    for hf_name, parts in stacked.items():
        if not {0, 1}.issubset(parts):
            missing = sorted({0, 1} - set(parts))
            raise RuntimeError(f"Qwen2VL mmproj stacked target {hf_name} is missing {missing}")
        parent, leaf = _module_and_leaf(root, hf_name)
        value = torch.stack(
            [
                _materialize_gguf_tensor(
                    parts[0],
                    compute_dtype = compute_dtype,
                    gguf = gguf,
                    reader = reader,
                ),
                _materialize_gguf_tensor(
                    parts[1],
                    compute_dtype = compute_dtype,
                    gguf = gguf,
                    reader = reader,
                ),
            ],
            dim = 2,
        )
        setattr(parent, leaf, nn.Parameter(value, requires_grad = False))
        loaded += 2
        materialized += 2
        stacked_patch_embeddings += 1

    return reader, Qwen2VLMmprojReplacementStats(
        loaded = loaded,
        materialized = materialized,
        qkv_groups = qkv_groups,
        stacked_patch_embeddings = stacked_patch_embeddings,
    )


def _replace_mistral_modules_with_lazy_gguf(
    language_model: MistralModel,
    *,
    gguf_path: Path,
    compute_dtype: torch.dtype,
    resident_device: torch.device | str | None = None,
) -> Any:
    gguf = _require_gguf()
    reader = gguf.GGUFReader(str(gguf_path))

    config = language_model.config
    expected = set(language_model.state_dict().keys())
    mapped: dict[str, tuple[Any, _GGUFNameTarget]] = {}
    loaded: set[str] = set()

    for tensor in reader.tensors:
        target = _target_for_gguf_name(
            tensor.name,
            num_attention_heads = config.num_attention_heads,
            num_key_value_heads = config.num_key_value_heads,
        )
        if target is None:
            continue

        if target.hf_name not in expected:
            continue

        mapped[target.hf_name] = (tensor, target)

    for hf_name, (tensor, target) in mapped.items():
        if hf_name in loaded:
            continue

        parent, leaf = _module_and_leaf(language_model, hf_name)
        quant_type = tensor.tensor_type

        if leaf == "weight" and isinstance(parent, nn.Linear) and quant_type not in _gguf_full_precision_types(gguf) and quant_type != _gguf_bfloat16_type(gguf):
            bias_name = hf_name.removesuffix(".weight") + ".bias"
            bias = None
            qbias = None
            bias_quant_type = None
            bias_logical_shape = None
            if bias_name in mapped:
                bias_tensor, _bias_target = mapped[bias_name]
                bias_qtype = bias_tensor.tensor_type
                if bias_qtype in _gguf_full_precision_types(gguf) or bias_qtype == _gguf_bfloat16_type(gguf):
                    bias = _materialize_gguf_tensor(
                        bias_tensor,
                        compute_dtype = compute_dtype,
                        gguf = gguf,
                        reader = reader,
                    )
                    if _bias_target.reverse_permute_heads is not None:
                        bias = _reverse_permute_qk(bias, _bias_target.reverse_permute_heads)
                else:
                    qbias = _torch_from_gguf_data(bias_tensor, copy = False)
                    bias_quant_type = bias_qtype
                    bias_logical_shape = _gguf_tensor_logical_shape(bias_tensor, reader = reader)
                loaded.add(bias_name)
            raw = _torch_from_gguf_data(tensor, copy = False)
            lazy = LazyGGUFLinear(
                raw,
                quant_type,
                in_features = parent.in_features,
                out_features = parent.out_features,
                compute_dtype = compute_dtype,
                reverse_permute_heads = target.reverse_permute_heads,
                resident_device = resident_device,
                bias = bias,
                qbias = qbias,
                bias_quant_type = bias_quant_type,
                bias_logical_shape = bias_logical_shape,
            )
            grandparent_name = hf_name.removesuffix(".weight")
            grandparent, module_name = _module_and_leaf(language_model, grandparent_name)
            setattr(grandparent, module_name, lazy)
        elif leaf == "weight" and isinstance(parent, nn.Embedding) and quant_type not in _gguf_full_precision_types(gguf) and quant_type != _gguf_bfloat16_type(gguf):
            raw = _torch_from_gguf_data(tensor, copy = False)
            lazy = LazyGGUFEmbedding(
                raw,
                quant_type,
                num_embeddings = parent.num_embeddings,
                embedding_dim = parent.embedding_dim,
                compute_dtype = compute_dtype,
                resident_device = resident_device,
            )
            grandparent_name = hf_name.removesuffix(".weight")
            grandparent, module_name = _module_and_leaf(language_model, grandparent_name)
            setattr(grandparent, module_name, lazy)
        elif quant_type in _gguf_full_precision_types(gguf) or quant_type == _gguf_bfloat16_type(gguf):
            value = _materialize_gguf_tensor(
                tensor,
                compute_dtype = compute_dtype,
                gguf = gguf,
                reader = reader,
            )
            if target.reverse_permute_heads is not None:
                value = _reverse_permute_qk(value, target.reverse_permute_heads)
            setattr(parent, leaf, nn.Parameter(value, requires_grad = False))
        else:
            raise RuntimeError(f"Unsupported lazy GGUF target: {hf_name}")

        loaded.add(hf_name)

    missing = sorted(k for k in expected if k not in loaded)
    if missing:
        raise RuntimeError(
            "Lazy Mistral GGUF loader did not initialize all text weights. "
            f"Missing examples: {missing[:8]} ({len(missing)} total)."
        )
    return reader
