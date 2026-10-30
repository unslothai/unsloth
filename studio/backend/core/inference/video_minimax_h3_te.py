# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Qwen3-VL conditioner (50 of 64 layers, no lm_head/norm, ConvRot INT8); lossless as H3 reads
hidden_states[50]; 51st pass-through slot avoids post-norm and keeps
len(layers)==config.num_hidden_layers (diffusers guard); inert unless configured; bad artifacts
fall back to released bf16 encoder."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Optional

# ConvRot primitives shared with the denoiser's hosted INT8 checkpoint; re-exported here.
from .diffusion_convrot import (  # noqa: F401  (re-export: callers and tests name them here)
    build_convrot_hadamard,
    rotate_convrot_activation,
)

# Not H3_COMPONENT_REPO: the VAEs moved to the GGUF mirror, the conditioner did not.
H3_TE_QUANT_REPO = "unsloth/MiniMax-H3-FP8"
# Hosted quantized conditioners, by ``text_encoder_quant`` scheme. nvfp4
# (``qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors``, 14.61 GiB) is deliberately absent: it is an AWQ NVFP4 layout with
# two levels of scale and a per-tensor global scale, a different loader and a different kernel, and shipping it needs
# its own numerical verification. The table is the one place to add it.
H3_TE_QUANT_FILES: dict[str, str] = {
    # Built from MiniMaxAI text_encoder/ with quantize_convrot_int8_weight; reads the official path
    # only.
    "int8": "text_encoders/official/qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
}

# Default int8: 27 GB over 50 layers vs 66.7 GB dense 64-layer; transformers has no early exit.
# Not an INT8 GEMM: ConvRot dequantizes then F.linear; the win is bytes moved, not math.
H3_TE_QUANT_DEFAULT = "int8"

# From Hub safetensors headers; the load is storage-faithful, so file size == resident size.
H3_TE_QUANT_RESIDENT_GB: dict[str, float] = {
    "int8": 27.2,
}

# Must equal MiniMaxH3ModularPipeline.text_encoder_layer.
H3_TE_READ_LAYER = 50

# Validated per tensor at load so a re-upload with another group is refused, not decoded to noise.
H3_TE_CONVROT_GROUP = 256

# Any other comfy_quant format is unverified and refused.
_H3_TE_EXPECTED_QUANT = {"format": "int8_tensorwise", "convrot": True}

# Tensor-name suffixes that make up one quantized linear in the hosted layout.
_SCALE_SUFFIX = ".weight_scale"
_QUANT_SUFFIX = ".comfy_quant"


def h3_te_quant_scheme(mode: Optional[str]) -> Optional[str]:
    """Pure and non-raising; the request was validated at the route, unknown schemes give None."""
    if mode is None:
        return None
    normalized = str(mode).strip().lower().replace("-", "_")
    return normalized if normalized in H3_TE_QUANT_FILES else None


def h3_te_quant_filename(scheme: Optional[str]) -> Optional[str]:
    """The hosted checkpoint's path inside ``H3_TE_QUANT_REPO`` for ``scheme``, or None."""
    return H3_TE_QUANT_FILES.get(scheme or "")


def h3_te_quant_source(scheme: Optional[str]) -> str:
    """The repo to fetch the hosted quantized conditioner for ``scheme`` from: always ``H3_TE_QUANT_REPO``. A video
    load never reads from the Comfy-Org repack, not even bytes an older install cached under it."""
    del scheme
    return H3_TE_QUANT_REPO


def h3_te_resident_gb(scheme: Optional[str], *, bf16_gb: float) -> float:
    """Resident decimal GB of the conditioner this pick loads: the hosted quantized size when one
    exists for ``scheme``, else the released bfloat16 ``bf16_gb``."""
    resolved = h3_te_quant_scheme(scheme)
    return H3_TE_QUANT_RESIDENT_GB[resolved] if resolved else bf16_gb


# H is a normalized regular Hadamard (H @ H == I); dequantizing without the rotation gives noise.
@lru_cache(maxsize = None)
def _int8_convrot_linear_class() -> Any:
    """Built lazily so importing the module never imports torch, and once: a fresh class on a second load would retrace every compiled block."""
    import torch
    from torch import nn

    class Int8ConvRotLinear(nn.Module):
        """A Linear whose weight stays INT8 in the rotated basis for its whole residency.

        The parameter names are the hosted checkpoint's own (``weight``, ``weight_scale``, ``bias``)
        so the remapped state dict loads straight in under ``strict=True``.

        The forward keeps the weight quantized and dequantizes a bfloat16 view per call (at most
        262 MB, for the 25600x5120 MLP projections) rather than holding one. That is deliberate:
        the whole point of this path is that the encoder's RESIDENT footprint is 27 GB, and a
        cached dense view would put the 51 GB back. The per-output-channel scale is applied to the
        OUTPUT, not the weight, because it factors straight out of the matmul --
        ``sum_i x_i q_oi s_o == s_o * sum_i x_i q_oi`` -- which keeps it exact in float32 over a
        tiny tensor instead of approximate in bfloat16 over a huge one. INT8 values are integers
        below 256, so the ``.to(compute dtype)`` is exact in bfloat16.

        Weight-only (not W8A8): less error, and the conditioner runs once per generation.
        """

        def __init__(
            self, in_features: int, out_features: int, bias: bool, group_size: int
        ) -> None:
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.group_size = group_size
            self.register_buffer(
                "weight",
                torch.empty(out_features, in_features, dtype = torch.int8, device = "meta"),
                persistent = True,
            )
            self.register_buffer(
                "weight_scale",
                torch.empty(out_features, 1, dtype = torch.float32, device = "meta"),
                persistent = True,
            )
            if bias:
                self.register_buffer(
                    "bias",
                    torch.empty(out_features, dtype = torch.bfloat16, device = "meta"),
                    persistent = True,
                )
            else:
                self.bias = None

        def forward(self, x: Any) -> Any:  # noqa: D102
            h = build_convrot_hadamard(self.group_size, device = x.device, dtype = x.dtype)
            rotated = rotate_convrot_activation(x, h, self.group_size)
            out = torch.nn.functional.linear(rotated, self.weight.to(x.dtype))
            out = out * self.weight_scale.reshape(1, -1).to(dtype = out.dtype, device = out.device)
            if self.bias is not None:
                out = out + self.bias.to(out.dtype)
            return out

        def extra_repr(self) -> str:  # noqa: D102
            return (
                f"in_features={self.in_features}, out_features={self.out_features}, "
                f"bias={self.bias is not None}, int8_convrot(group={self.group_size})"
            )

    return Int8ConvRotLinear


def _terminator_layer_class() -> Any:
    """The 51st decoder slot: passes its input through untouched. See the module docstring."""
    from torch import nn

    class H3TextEncoderTerminatorLayer(nn.Module):
        """Zero-parameter stand-in for decoder layer 50 (0-based).

        MiniMax-H3 reads ``hidden_states[50]``, the input to this slot, so nothing this returns is
        ever consumed -- it exists only so the final ``norm`` lands on an entry no one reads and
        ``hidden_states[50]`` stays the raw post-layer-50 state. Holding real weights here would
        cost ~0.5 GB and change nothing."""

        def forward(self, hidden_states: Any, *args: Any, **kwargs: Any) -> Any:  # noqa: D102
            return hidden_states

    return H3TextEncoderTerminatorLayer


# Hosted keys use ComfyUI's flattened Qwen3-VL names; transformers nests them under ``model.``.
def h3_te_remap_key(key: str) -> str:
    """A hosted checkpoint tensor name in transformers' ``Qwen3VLForConditionalGeneration`` naming."""
    if key.startswith("visual."):
        return "model." + key
    if key.startswith("model."):
        return "model.language_model." + key[len("model.") :]
    return key


def _validate_comfy_quant(blob: Any, name: str) -> None:
    """Raise unless a tensor's ``comfy_quant`` metadata is the format this loader implements."""
    import json

    meta = json.loads(blob.cpu().numpy().tobytes().decode("utf-8").rstrip("\x00"))
    for field, expected in _H3_TE_EXPECTED_QUANT.items():
        if meta.get(field) != expected:
            raise ValueError(
                f"{name}: unsupported quant metadata {field}={meta.get(field)!r} "
                f"(this loader implements {field}={expected!r})"
            )
    group = int(meta.get("convrot_groupsize", H3_TE_CONVROT_GROUP))
    if group != H3_TE_CONVROT_GROUP:
        raise ValueError(
            f"{name}: ConvRot group {group} != the {H3_TE_CONVROT_GROUP} this loader implements"
        )


def load_h3_quantized_text_encoder(
    base: str,
    scheme: str,
    *,
    dtype: Any,
    hf_token: Optional[str] = None,
    cache_dir: Optional[str] = None,
    local_base: Optional[str] = None,
    local_files_only: bool = False,
    logger: Any = None,
) -> Optional[Any]:
    """Loads with local_files_only so a reload after eviction never re-downloads; stays on CPU since enable_auto_cpu_offload owns placement; None on any problem."""
    try:
        filename = h3_te_quant_filename(scheme)
        if filename is None:
            return None

        import torch
        import transformers
        from accelerate import init_empty_weights
        from safetensors import safe_open

        from utils.hf_xet_fallback import hf_hub_download_with_xet_fallback

        # Resolve as the stager did, so a cached legacy copy is read, not 401ed or re-pulled (27 GB).
        source_repo = h3_te_quant_source(scheme)
        path = hf_hub_download_with_xet_fallback(
            source_repo,
            filename,
            hf_token,
            cache_dir = cache_dir,
            # Pinned to cache_dir alone, a moved cache folder would re-pull 27 GB.
            reuse_other_cache_root = True,
            local_files_only = local_files_only,
            gguf_header_delta = True,
        )

        from .diffusion_families import pinned_revision_kwargs

        config = transformers.AutoConfig.from_pretrained(
            local_base or base,
            **pinned_revision_kwargs(local_base or base),
            subfolder = "text_encoder",
            token = hf_token,
            cache_dir = cache_dir,
            # local_base is None offline, so the hub id is read: the flag must be forwarded.
            local_files_only = local_files_only,
        )
        text_config = getattr(config, "text_config", config)
        released_layers = int(getattr(text_config, "num_hidden_layers", 0))
        if released_layers <= H3_TE_READ_LAYER:
            # Refuse a conditioner not deeper than the read layer; nothing else enforces the pairing.
            raise ValueError(
                f"{base} text_encoder has {released_layers} layers; MiniMax-H3 reads "
                f"hidden_states[{H3_TE_READ_LAYER}] and needs more than that"
            )
        # One slot past the read layer, so the read lands on the raw state, not the post-norm one.
        text_config.num_hidden_layers = H3_TE_READ_LAYER + 1

        # Params to meta (replaced by assign=True); non-persistent rotary buffers must be real on CPU,
        # else a dense 51 GB CPU rebuild.
        with init_empty_weights(include_buffers = False):
            encoder = transformers.Qwen3VLForConditionalGeneration(config)

        language_model = encoder.model.language_model
        language_model.layers[H3_TE_READ_LAYER] = _terminator_layer_class()()
        # norm only follows the read layer and the head is never called; dropping it saves 1.56 GB.
        language_model.norm = torch.nn.Identity()
        encoder.lm_head = torch.nn.Identity()

        int8_linear_cls = _int8_convrot_linear_class()
        with safe_open(path, framework = "pt", device = "cpu") as handle:
            names = set(handle.keys())
            quantized = {
                name[: -len(_SCALE_SUFFIX)] for name in names if name.endswith(_SCALE_SUFFIX)
            }
            # Swap before loading so the state dict lands on int8 buffers, not bf16 Linears.
            for prefix in sorted(quantized):
                quant_key = prefix + _QUANT_SUFFIX
                if quant_key not in names:
                    raise ValueError(f"{prefix}: quantized weight with no {_QUANT_SUFFIX} metadata")
                _validate_comfy_quant(handle.get_tensor(quant_key), prefix)
                target = h3_te_remap_key(prefix)
                parent_path, _, leaf = target.rpartition(".")
                parent = encoder.get_submodule(parent_path)
                existing = getattr(parent, leaf)
                slice_ = handle.get_slice(prefix + ".weight")
                out_features, in_features = slice_.get_shape()
                if (existing.out_features, existing.in_features) != (out_features, in_features):
                    raise ValueError(
                        f"{target}: checkpoint shape {(out_features, in_features)} != model "
                        f"{(existing.out_features, existing.in_features)}"
                    )
                setattr(
                    parent,
                    leaf,
                    int8_linear_cls(
                        in_features,
                        out_features,
                        bias = (prefix + ".bias") in names,
                        group_size = H3_TE_CONVROT_GROUP,
                    ),
                )

            # INT8 payload and fp32 scales keep their dtypes; dense parts follow the compute dtype.
            state_dict = {}
            for name in names:
                if name.endswith(_QUANT_SUFFIX):
                    continue
                tensor = handle.get_tensor(name)
                if (
                    not name.endswith(_SCALE_SUFFIX)
                    and tensor.is_floating_point()
                    and dtype is not None
                ):
                    tensor = tensor.to(dtype)
                state_dict[h3_te_remap_key(name)] = tensor

        # strict=True does not prove quantization: a dense re-upload loads as bf16 Linear and breaks
        # the VRAM preflight, so no plain Linear may survive in the decoder stack.
        dense = [
            name
            for name, module in language_model.layers.named_modules()
            if isinstance(module, torch.nn.Linear)
        ]
        if dense:
            raise ValueError(
                f"{len(dense)} decoder projection(s) are not quantized in this artifact, "
                f"e.g. {', '.join(sorted(dense)[:3])}; it is not the {scheme} checkpoint this "
                f"path budgets for"
            )
        encoder.load_state_dict(state_dict, strict = True, assign = True)
        # A dense CPU rebuild is the 51 GB allocation this path avoids, so leftover meta is a refusal.
        stranded = _meta_tensor_names(encoder)
        if stranded:
            raise ValueError(
                f"{len(stranded)} tensor(s) still on the meta device after loading, "
                f"e.g. {', '.join(stranded[:3])}"
            )
        encoder.eval()
        if logger is not None:
            logger.info(
                "video.h3_te_quant: loaded the hosted %s conditioner (%s, %s), "
                "%d ConvRot INT8 projections over %d decoder layers",
                scheme,
                source_repo,
                filename,
                len(quantized),
                H3_TE_READ_LAYER,
            )
        return encoder
    except Exception as exc:  # noqa: BLE001 -- fall back to the released bfloat16 encoder
        if logger is not None:
            logger.warning(
                "video.h3_te_quant: hosted %s conditioner unusable (%s); "
                "loading the released bfloat16 encoder instead",
                scheme,
                exc,
            )
        return None


def _meta_tensor_names(module: Any) -> list[str]:
    """Names of every parameter and buffer still on the meta device."""
    from itertools import chain
    return [
        name
        for name, tensor in chain(module.named_parameters(), module.named_buffers())
        if getattr(tensor, "is_meta", False)
    ]


# Group offloading pages the int8 conditioner in leaf by leaf instead of moving 27 GB whole.
H3_TE_STREAM_ENV = "UNSLOTH_H3_TE_STREAM"
# Encode footprint: the 1.56 GB embedding table + the prefetched leaf + activations (B200, 12 / 16 / 24 GB caps).
H3_TE_STREAMED_GB = 3.0
# Power-of-two arenas: torch's pinned allocator rounds each allocation up (13% extra pinned one by one).
_PIN_ARENA_BYTES = 1 << 31
_PIN_ALIGN = 512


def h3_te_stream_enabled() -> bool:
    """``UNSLOTH_H3_TE_STREAM=0`` keeps the conditioner in the CPU-offload rotation."""
    import os
    return str(os.environ.get(H3_TE_STREAM_ENV, "")).strip().lower() not in (
        "0",
        "off",
        "false",
        "no",
    )


def h3_te_pin_allowed(payload_bytes: int = 0, device: Any = None) -> bool:
    """Whether a payload may be pinned in full (not one arena), under the group switch and the ~1
    GiB Windows/WSL cap; never raises"""
    try:
        import os

        from .diffusion_memory import GROUP_OFFLOAD_PIN_ENV, _pin_budget_mib, _pinned_memory_capped

        from .host_pinning import host_pinning_allowed

        forced = str(os.environ.get(GROUP_OFFLOAD_PIN_ENV, "")).strip().lower()
        if forced in ("0", "off", "false", "no") or not host_pinning_allowed(device):
            return False
        if forced in ("1", "on", "true", "yes"):
            return True
        if _pinned_memory_capped():
            return False
        budget = _pin_budget_mib()
        return budget is not None and budget >= max(_PIN_ARENA_BYTES, int(payload_bytes)) >> 20
    except Exception:  # noqa: BLE001 -- unanswerable: stream unpinned
        return False


H3_TE_PIN_YIELD_ENV = "UNSLOTH_H3_TE_PIN_YIELD"


def h3_te_pin_yields(
    text_encoder: Any,
    streamed_denoiser: Any,
    device: Any = None,
    logger: Any = None,
) -> Optional[bool]:
    """False when the conditioner alone fits the pin budget but not beside the denoiser that will stream: the
    denoiser's blocks are read every step, the conditioner once per prompt, so the denoiser keeps the budget. None
    leaves the decision to ``stream_h3_text_encoder``. ``UNSLOTH_H3_TE_PIN_YIELD=0`` turns it off. Never raises."""
    try:
        import os

        if str(os.environ.get(H3_TE_PIN_YIELD_ENV, "")).strip().lower() in (
            "0",
            "off",
            "false",
            "no",
        ):
            return None
        if text_encoder is None or streamed_denoiser is None:
            return None
        te_bytes = _module_payload_bytes(text_encoder)
        dit_bytes = _host_payload_bytes(streamed_denoiser)
        if dit_bytes <= 0 or not h3_te_pin_allowed(te_bytes, device):
            return None
        if h3_te_pin_allowed(te_bytes + dit_bytes, device):
            return None
        if logger is not None:
            from .diffusion_memory import _pin_budget_mib
            logger.info(
                "video.h3_te_stream: the conditioner (%.1f GB) streams from pageable memory so the streamed "
                "denoiser (%.1f GB) can pin first (%s MiB pinnable)",
                te_bytes / 1e9,
                dit_bytes / 1e9,
                _pin_budget_mib(),
            )
        return False
    except Exception:  # noqa: BLE001 -- keep the stock decision
        return None


def _host_payload_bytes(module: Any) -> int:
    """Host bytes of ``module``'s CPU tensors, torchao subclasses at their inner (int8) size, not their bf16 shape."""
    from .diffusion_prequant import tensor_payload_bytes

    seen: set = set()
    total = 0
    for tensor in list(module.parameters()) + list(module.buffers()):
        if id(tensor) not in seen and tensor.device.type == "cpu":
            seen.add(id(tensor))
            total += tensor_payload_bytes(tensor)
    return total


def _module_payload_bytes(module: Any) -> int:
    seen: set = set()
    total = 0
    for tensor in list(module.parameters()) + list(module.buffers()):
        if id(tensor) not in seen and tensor.device.type == "cpu":
            seen.add(id(tensor))
            total += tensor.numel() * tensor.element_size()
    return total


def _round_up(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


def _pinned_arena(size: int) -> Any:
    import torch

    from .diffusion_memory import registered_pinned_empty

    arena = registered_pinned_empty(size)
    if arena is not None:
        return arena
    return torch.empty(size, dtype = torch.uint8, pin_memory = True)


def pin_module_in_place(
    module: Any,
    *,
    arena_bytes: int = _PIN_ARENA_BYTES,
    _arena_factory: Any = None,
    repoint: bool = False,
) -> int:
    """Replaces tensors in _parameters/_buffers rather than .data, which would keep the mmap alive."""
    import torch

    slots: dict[int, list] = {}  # id(tensor) -> [tensor, [(owner dict, name, is_param)]]
    for sub in module.modules():
        for table, is_param in ((sub._parameters, True), (sub._buffers, False)):
            for name, tensor in list(table.items()):
                if tensor is None:
                    continue
                entry = slots.setdefault(id(tensor), [tensor, []])
                entry[1].append((table, name, is_param))
    todo = []
    for tensor, owners in slots.values():
        if tensor.device.type != "cpu" or type(tensor.data) is not torch.Tensor:
            continue
        if tensor.numel() == 0 or (_arena_factory is None and tensor.is_pinned()):
            continue
        todo.append((tensor, owners))
    # Largest first, first fit.
    todo.sort(key = lambda item: item[0].numel() * item[0].element_size(), reverse = True)
    arenas: list[list] = []  # [arena tensor, used bytes]
    pinned = 0
    for tensor, owners in todo:
        nbytes = tensor.numel() * tensor.element_size()
        need = _round_up(nbytes, _PIN_ALIGN)
        slot = next((a for a in arenas if a[0].numel() - a[1] >= need), None)
        if slot is None:
            size = max(int(arena_bytes), 1 << (need - 1).bit_length())
            arena = _arena_factory(size) if _arena_factory is not None else _pinned_arena(size)
            slot = [arena, 0]
            arenas.append(slot)
        offset = slot[1]
        view = slot[0][offset : offset + nbytes].view(tensor.dtype).view(tensor.shape)
        view.copy_(tensor.detach())
        if repoint:
            # Only when the old storage is not an mmap view (weights the loader already copied or cast).
            tensor.data = view
        else:
            replacement = (
                torch.nn.Parameter(view, requires_grad = tensor.requires_grad)
                if isinstance(tensor, torch.nn.Parameter)
                else view
            )
            for table, name, _is_param in owners:
                table[name] = replacement
        slot[1] = offset + need
        pinned += nbytes
    return pinned


def stream_h3_text_encoder(
    manager: Any,
    text_encoder: Any,
    device: Any,
    *,
    pin: Optional[bool] = None,
    logger: Any = None,
) -> Optional[str]:
    """Stream MiniMax-H3's conditioner leaf by leaf via group offloading, outside the ComponentsManager rotation.

    Hooks go on ``text_encoder.model``, which the H3 encode step calls directly. ``pin=None`` decides from
    ``h3_te_pin_allowed``. Returns ``"stream"`` (pinned) / ``"stream_lazy"`` / ``"sync"`` (integrated GPU: no stream,
    nothing pinned), or None with nothing changed."""
    import torch

    onload = torch.device(device)
    target = getattr(text_encoder, "model", None)
    if onload.type != "cuda" or target is None or not h3_te_stream_enabled():
        return None
    try:
        import inspect

        from diffusers.hooks import apply_group_offloading
    except Exception:  # noqa: BLE001 -- no group offloading in this diffusers
        return None
    from .diffusion_memory import (
        _pin_vision_embedding_device,
        _remove_group_offload_hooks,
        install_group_offload_buffer_restore,
        install_group_offload_hooks_eager,
    )
    from .diffusion_prequant import _evict_rotation_hook, _unhook_from_manager

    from .host_pinning import host_pinning_allowed

    params = inspect.signature(apply_group_offloading).parameters
    # A stream makes diffusers pin each leaf; an integrated GPU streams synchronously from pageable memory instead.
    stream = host_pinning_allowed(onload)
    if not stream:
        pin = False
    elif pin is None:
        pin = h3_te_pin_allowed(_module_payload_bytes(text_encoder), onload)
    if pin:
        try:
            pin_module_in_place(text_encoder)
        except Exception as exc:  # noqa: BLE001 -- pinned host RAM refused: stream unpinned
            if logger is not None:
                logger.warning("video.h3_te_stream: pinning failed, streaming unpinned: %s", exc)
            pin = False
    kwargs: dict[str, Any] = {
        "onload_device": onload,
        "offload_device": torch.device("cpu"),
        "offload_type": "leaf_level",
        "use_stream": stream,
    }
    if stream:
        if "non_blocking" in params:
            kwargs["non_blocking"] = True
        if "record_stream" in params:
            kwargs["record_stream"] = True
        if "low_cpu_mem_usage" in params:
            # Already pinned in place, so diffusers' pin_memory() returns the same tensor: no second host copy.
            kwargs["low_cpu_mem_usage"] = not pin
        elif not pin:
            return None
    try:
        text_encoder.requires_grad_(False)
        # The int8 weights are buffers, which stock diffusers leaves on the device after offload.
        install_group_offload_buffer_restore()
        install_group_offload_hooks_eager()
        apply_group_offloading(target, **kwargs)
        # Image prompts: Qwen3-VL's position interpolation reads the offloaded embedding's CPU device (transformers 5.5).
        _pin_vision_embedding_device(target)
    except Exception as exc:  # noqa: BLE001 -- keep the rotation
        _remove_group_offload_hooks(target)
        if logger is not None:
            logger.warning(
                "video.h3_te_stream: group offloading refused, keeping the rotation: %s", exc
            )
        return None
    if not _unhook_from_manager(manager, text_encoder, logger = logger, what = "te_stream:hook"):
        # Still in the rotation: its pre_forward would move the whole module under the group hooks.
        _remove_group_offload_hooks(target)
        return None
    # The manager only evicts inside a managed pre_forward; else the last decode's VAE sits beside the encode.
    target.register_forward_pre_hook(_evict_rotation_hook(manager, onload))
    mode = ("stream" if pin else "stream_lazy") if stream else "sync"
    if logger is not None:
        logger.info(
            "video.h3_te_stream: conditioner streamed leaf by leaf on %s (%s host copy)",
            device,
            "pinned" if pin else "pageable",
        )
    return mode
