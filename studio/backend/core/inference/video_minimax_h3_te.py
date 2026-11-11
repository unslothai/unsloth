# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Qwen3-VL conditioner (50 of 64 layers, no lm_head/norm, ConvRot INT8); lossless as H3 reads
hidden_states[50]; 51st pass-through slot avoids post-norm and keeps
len(layers)==config.num_hidden_layers (diffusers guard); inert unless configured; bad artifacts
fall back to released bf16 encoder."""

from __future__ import annotations

import functools
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
            # float16: scale the weight before the matmul; the unscaled int8 sum overflows 65504.
            self.scale_weight_first = False

        def forward(self, x: Any) -> Any:  # noqa: D102
            h = build_convrot_hadamard(self.group_size, device = x.device, dtype = x.dtype)
            rotated = rotate_convrot_activation(x, h, self.group_size)
            if getattr(self, "scale_weight_first", False):
                weight = (self.weight.to(torch.float32) * self.weight_scale).to(x.dtype)
                out = torch.nn.functional.linear(rotated, weight)
                if self.bias is not None:
                    out = out + self.bias.to(out.dtype)
                return out
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
# Arena fills (copies plus zeroed tails) run on a pool, so they are parallel from any calling thread.
_PIN_COPY_THREADS = 8
_PIN_ZERO_CHUNK = 64 << 20


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


def _pinned_arena(size: int, fill: Any) -> Any:
    """A pinned uint8 arena of ``size`` bytes, every byte written by ``fill`` (before registering, where registered)."""
    import torch

    from .diffusion_memory import registered_pinned_empty

    arena = registered_pinned_empty(size, fill = fill)
    if arena is not None:
        return arena
    arena = torch.empty(size, dtype = torch.uint8, pin_memory = True)
    fill(arena)
    return arena


# Arenas sized to their payload (0.61 GB audio VAE pinned 2.15 GB); the host-looked-up vocab table stays mapped. =0 reverts.
H3_HOST_PIN_TRIM_ENV = "UNSLOTH_H3_HOST_PIN_TRIM"
_PIN_PAGE = 4096


def h3_host_pin_trim_enabled() -> bool:
    import os
    return str(os.environ.get(H3_HOST_PIN_TRIM_ENV, "")).strip().lower() not in (
        "0",
        "off",
        "false",
        "no",
    )


def pin_module_in_place(
    module: Any,
    *,
    arena_bytes: int = _PIN_ARENA_BYTES,
    _arena_factory: Any = None,
    repoint: bool = False,
    skip: tuple = (),
) -> int:
    """Replaces tensors in _parameters/_buffers rather than .data, which would keep the mmap alive."""
    import torch

    skipped = {id(t) for t in skip if t is not None}
    slots: dict[int, list] = {}  # id(tensor) -> [tensor, [(owner dict, name, is_param)]]
    for sub in module.modules():
        for table, is_param in ((sub._parameters, True), (sub._buffers, False)):
            for name, tensor in list(table.items()):
                if tensor is None or id(tensor) in skipped:
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
    # Largest first, first fit. Placed up front, so each arena is filled before it is registered.
    todo.sort(key = lambda item: item[0].numel() * item[0].element_size(), reverse = True)
    arenas: list[list] = []  # [arena bytes, used bytes, [(tensor, owners, offset)]]
    pinned = 0
    for tensor, owners in todo:
        nbytes = tensor.numel() * tensor.element_size()
        need = _round_up(nbytes, _PIN_ALIGN)
        slot = next((a for a in arenas if a[0] - a[1] >= need), None)
        if slot is None:
            slot = [max(int(arena_bytes), 1 << (need - 1).bit_length()), 0, []]
            arenas.append(slot)
        slot[2].append((tensor, owners, slot[1]))
        slot[1] += need
        pinned += nbytes
    if h3_host_pin_trim_enabled():
        # placement is final: an arena's unused tail would only be zeroed and locked
        for arena in arenas:
            arena[0] = min(arena[0], _round_up(arena[1], _PIN_PAGE))
    placed = []  # (tensor, owners, pinned view)
    for size, _used, entries in arenas:
        fill = functools.partial(_fill_arena, entries = entries)
        if _arena_factory is not None:
            arena = _arena_factory(size)
            fill(arena)
        else:
            arena = _pinned_arena(size, fill)
        placed.extend(
            (tensor, owners, _arena_view(arena, tensor, offset))
            for tensor, owners, offset in entries
        )
    for tensor, owners, view in placed:
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
    return pinned


def _arena_view(arena: Any, tensor: Any, offset: int) -> Any:
    nbytes = tensor.numel() * tensor.element_size()
    return arena[offset : offset + nbytes].view(tensor.dtype).view(tensor.shape)


def _fill_arena(arena: Any, *, entries: list) -> None:
    """Copies each tensor to its offset and zeroes the rest, on a thread pool (copy_, zero_ and their page faults
    release the GIL). Writing every byte faults the arena's pages in the copy itself, with no separate zero pass."""
    jobs = []  # (destination, source tensor or None to zero)
    end = 0
    for tensor, _owners, offset in entries:
        if offset > end:
            jobs.append((arena[end:offset], None))
        jobs.append((_arena_view(arena, tensor, offset), tensor))
        end = offset + tensor.numel() * tensor.element_size()
    jobs.extend(
        (arena[start : start + _PIN_ZERO_CHUNK], None)
        for start in range(end, arena.numel(), _PIN_ZERO_CHUNK)
    )

    def run(job: tuple) -> None:
        dst, src = job
        if src is None:
            dst.zero_()
        else:
            dst.copy_(src.detach())

    workers = min(_PIN_COPY_THREADS, len(jobs))
    if workers <= 1:
        for job in jobs:
            run(job)
        return
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers = workers, thread_name_prefix = "unsloth-h3-te-pin") as pool:
        # list() re-raises the first failed copy, before any owner is replaced.
        list(pool.map(run, jobs))


# The streamed conditioner's vocabulary table (151936 x 5120 bf16, 1.56 GB) stays on the host: the prompt's rows are
# gathered there and only they are uploaded, not the whole table per encode. ``UNSLOTH_H3_HOST_EMBED_ROWS=0`` streams
# the table like every other leaf.
H3_HOST_EMBED_ROWS_ENV = "UNSLOTH_H3_HOST_EMBED_ROWS"
_HOST_ROWS_ATTR = "_unsloth_host_rows_device"


def h3_host_embed_rows_enabled() -> bool:
    import os
    return str(os.environ.get(H3_HOST_EMBED_ROWS_ENV, "")).strip().lower() not in (
        "0",
        "off",
        "false",
        "no",
    )


def _host_rows_forward(self: Any, input: Any) -> Any:
    """``nn.Embedding.forward`` on the host table (a gather: bit-identical), the rows sent to the onload device."""
    import torch.nn.functional as F

    weight = self.weight
    ids = input if input.device == weight.device else input.to(weight.device)
    rows = F.embedding(
        ids,
        weight,
        self.padding_idx,
        self.max_norm,
        self.norm_type,
        self.scale_grad_by_freq,
        self.sparse,
    )
    return rows.to(self.__dict__[_HOST_ROWS_ATTR])


def _host_rows_table(target: Any) -> Any:
    """The table ``keep_h3_embedding_on_host`` will take out of the stream (same static checks), or None."""
    try:
        import torch

        if target is None or not h3_host_embed_rows_enabled():
            return None
        getter = getattr(target, "get_input_embeddings", None)
        emb = getter() if callable(getter) else None
        if type(emb) is not torch.nn.Embedding or _HOST_ROWS_ATTR in emb.__dict__:
            return None
        weight = emb.weight
        if (
            type(weight.data) is not torch.Tensor
            or weight.device.type != "cpu"
            or emb.max_norm is not None
            or emb.sparse
            or weight.dtype not in (torch.bfloat16, torch.float16, torch.float32)
        ):
            return None
        return weight
    except Exception:  # noqa: BLE001 -- pin it with the rest
        return None


def _pin_streamed_table(target: Any, weight: Any) -> None:
    """The table stayed in the stream after all: pin it as the whole-module pin would have."""
    from .diffusion_memory import _offload_groups

    owner = next(m for m in target.modules() if m._parameters.get("weight") is weight)
    pin_module_in_place(owner, repoint = True)
    for group in _offload_groups(target):
        cpu = getattr(group, "cpu_param_dict", None)
        if isinstance(cpu, dict) and weight in cpu:
            cpu[weight] = weight.data


def keep_h3_embedding_on_host(target: Any, logger: Any = None) -> bool:
    """Take the group-offloaded language embedding out of the stream, its lookup done on the host. Run right after
    ``apply_group_offloading``; True when installed. Plain frozen bf16/fp16/fp32 tables only. Never raises."""
    try:
        if not h3_host_embed_rows_enabled():
            return False
        import types

        import torch

        getter = getattr(target, "get_input_embeddings", None)
        emb = getter() if callable(getter) else None
        if type(emb) is not torch.nn.Embedding or _HOST_ROWS_ATTR in emb.__dict__:
            return False
        weight = emb.weight
        if (
            type(weight.data) is not torch.Tensor
            or weight.device.type != "cpu"
            or weight.requires_grad
            or emb.max_norm is not None
            or emb.sparse
            or weight.dtype not in (torch.bfloat16, torch.float16, torch.float32)
        ):
            return False
        registry = getattr(emb, "_diffusers_hook", None)
        hooks = getattr(registry, "hooks", None) or {}
        # a streamed (lazy prefetch) setup also tracks the leaf's first execution; anything else keeps the stock leaf
        if "group_offloading" not in hooks or not set(hooks) <= {
            "group_offloading",
            "layer_execution_tracker",
        }:
            return False
        group = getattr(hooks["group_offloading"], "group", None)
        if group is None or list(getattr(group, "modules", None) or ()) != [emb]:
            return False
        onload = getattr(group, "onload_device", None)
        if onload is None:
            return False
        name = next((n for n, m in target.named_modules() if m is emb), None)
        lazy = (getattr(getattr(target, "_diffusers_hook", None), "hooks", None) or {}).get(
            "lazy_prefetch_group_offloading"
        )
        emb.__dict__[_HOST_ROWS_ATTR] = torch.device(onload)
        try:
            if "layer_execution_tracker" in hooks:
                registry.remove_hook("layer_execution_tracker", recurse = False)
            registry.remove_hook("group_offloading", recurse = False)
        finally:
            # set even if a removal failed half way: the table is on the host, so the lookup must be too
            emb.forward = types.MethodType(_host_rows_forward, emb)
        names = getattr(lazy, "_layer_execution_tracker_module_names", None)
        if isinstance(names, set) and name is not None:
            names.discard(name)  # no longer streamed: the first forward must not expect it
        # diffusers may hold the table's pinned twin; the lookup reads what the parameter points at now
        host = getattr(group, "cpu_param_dict", None) or {}
        pinned = host.get(weight)
        if (
            pinned is not None
            and pinned.device.type == "cpu"
            and weight.data_ptr() != pinned.data_ptr()
        ):
            weight.data = pinned
    except Exception as exc:  # noqa: BLE001 -- the streamed leaf still works
        if logger is not None:
            logger.warning("video.h3_te_stream: embedding stays streamed: %s", exc)
        return False
    if logger is not None:
        logger.info(
            "video.h3_te_stream: vocabulary table (%.2f GB) stays on the host, prompt rows uploaded",
            weight.numel() * weight.element_size() / 1e9,
        )
    return True


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
    # Looked up on the host, so a pinned copy buys nothing; needs diffusers to take the pinned rest as it is.
    table = (
        _host_rows_table(target)
        if pin and h3_host_pin_trim_enabled() and "low_cpu_mem_usage" in params
        else None
    )
    if pin:
        try:
            pinned_bytes = pin_module_in_place(text_encoder, skip = (table,))
        except Exception as exc:  # noqa: BLE001 -- pinned host RAM refused: stream unpinned
            if logger is not None:
                logger.warning("video.h3_te_stream: pinning failed, streaming unpinned: %s", exc)
            pin = False
            table = None
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
            # With the table left mapped, the onload pins only what is not pinned yet: nothing once the table is out.
            kwargs["low_cpu_mem_usage"] = not pin or table is not None
        elif not pin:
            return None
    try:
        text_encoder.requires_grad_(False)
        # The int8 weights are buffers, which stock diffusers leaves on the device after offload.
        install_group_offload_buffer_restore()
        install_group_offload_hooks_eager()
        apply_group_offloading(target, **kwargs)
        on_host = keep_h3_embedding_on_host(target, logger = logger)
        if table is not None:
            if on_host:
                if logger is not None:
                    logger.info(
                        "video.h3_host_pin_trim: conditioner pins %.2f GB in arenas sized to their weights, its "
                        "vocabulary table (%.2f GB) stays on its mapping",
                        pinned_bytes / 1e9,
                        table.numel() * table.element_size() / 1e9,
                    )
            else:
                try:
                    _pin_streamed_table(target, table)
                except Exception as exc:  # noqa: BLE001 -- streams pageable: correct, slower
                    if logger is not None:
                        logger.warning("video.h3_host_pin_trim: table not pinned: %s", exc)
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


# A mapped conditioner larger than the page cache is re-read from disk every prompt (L4: +78-84 s); anon pages are not.
H3_TE_RESIDENT_ENV = "UNSLOTH_H3_TE_RESIDENT"
_H3_TE_RESIDENT_MIN_MIB = 1024
_H3_TE_COPY_THREADS = 4


def h3_te_resident_enabled() -> bool:
    """``UNSLOTH_H3_TE_RESIDENT=0`` leaves the streamed conditioner on its memory mapping."""
    import os
    return str(os.environ.get(H3_TE_RESIDENT_ENV, "")).strip().lower() not in (
        "0",
        "off",
        "false",
        "no",
    )


def _streamed_host_tensors(target: Any, groups: Optional[list] = None) -> list:
    """(group, tensor, host) pageable CPU tensors, language model first: vision is only read for image prompts."""
    import torch

    if groups is None:
        from .diffusion_memory import _offload_groups
        groups = _offload_groups(target)
    names = {id(m): n for n, m in target.named_modules()}

    def _late(group: Any) -> bool:
        mods = list(getattr(group, "modules", None) or ())
        leader = getattr(group, "onload_leader", None)
        name = names.get(id(mods[0] if mods else leader), "")
        return "visual" in name.split(".")

    out: list = []
    seen: set = set()
    for group in sorted(groups, key = _late):
        cpu = getattr(group, "cpu_param_dict", None) or {}
        tensors = (
            [t for m in getattr(group, "modules", None) or () for t in m.parameters()]
            + [t for m in getattr(group, "modules", None) or () for t in m.buffers()]
            + list(getattr(group, "parameters", None) or ())
            + list(getattr(group, "buffers", None) or ())
        )
        for tensor in tensors:
            host = cpu.get(tensor, tensor)
            if id(tensor) in seen or host is None or host.device.type != "cpu":
                continue
            seen.add(id(tensor))
            if type(host) is not torch.Tensor and type(host) is not torch.nn.Parameter:
                continue  # tensor subclasses keep their own storage layout
            if host.numel() == 0 or host.is_pinned():
                continue
            out.append((group, tensor, host))
    return out


def keep_h3_te_resident(
    text_encoder: Any,
    *,
    logger: Any = None,
    groups: Optional[list] = None,
    host_ram: Any = None,
) -> int:
    """Copy the share of the streamed conditioner that fits beside the reserve into anonymous RAM; run after every
    load-time host allocation. ``host_ram``: ``(total, available, commit)`` MiB. Returns MiB copied. Never raises."""
    try:
        target = getattr(text_encoder, "model", None)
        if target is None or not h3_te_resident_enabled():
            return 0
        undo = _undo_list(target)  # marks the share as managed: later calls top it up
        copied_ids = {id(e[3]) for e in undo}
        entries = [e for e in _streamed_host_tensors(target, groups) if id(e[2]) not in copied_ids]
        need = -(-sum(h.numel() * h.element_size() for _g, _t, h in entries) // (1 << 20))
        if need <= 0:
            return 0
        from .diffusion_small_host import (
            encoder_resident_share_mib,
            host_commit_mib,
            host_ram_mib,
        )

        if host_ram is None:
            total, available = host_ram_mib()
            commit = host_commit_mib()
        else:
            total, available, commit = host_ram
        share, why = encoder_resident_share_mib(
            need,
            host_total_mib = total,
            host_available_mib = available,
            host_commit_mib = commit,
            min_share_mib = min(need, _H3_TE_RESIDENT_MIN_MIB),
        )
        if share <= 0:
            if logger is not None:
                logger.info("video.h3_te_resident: the conditioner stays memory-mapped (%s)", why)
            return 0
        budget = share << 20
        chosen = []
        for entry in entries:
            nbytes = entry[2].numel() * entry[2].element_size()
            if nbytes <= budget:
                chosen.append(entry)
                budget -= nbytes
        copied = _copy_host_tensors_(chosen, undo = undo)
        mib = copied >> 20
        if logger is not None:
            logger.info(
                "video.h3_te_resident: %d of %d MiB of the streamed conditioner copied into host RAM (%s)",
                mib,
                need,
                why,
            )
        return mib
    except Exception as exc:  # noqa: BLE001 -- the mapping still works
        if logger is not None:
            logger.warning("video.h3_te_resident: kept the mapping: %s", exc)
        return 0


_TE_RESIDENT_UNDO_ATTR = "_unsloth_te_resident_undo"


def _undo_list(target: Any) -> list:
    undo = getattr(target, _TE_RESIDENT_UNDO_ATTR, None)
    if undo is None:
        undo = []
        try:
            setattr(target, _TE_RESIDENT_UNDO_ATTR, undo)
        except Exception:  # noqa: BLE001 -- no undo: the share cannot yield
            return []
    return undo


def top_up_h3_te_resident(text_encoder: Any, *, logger: Any = None) -> int:
    """``keep_h3_te_resident`` again after the load (a re-fit released resident denoiser host copies): copies the
    next share of the still-mapped conditioner. Only for a share the load already manages. Returns MiB copied."""
    target = getattr(text_encoder, "model", None)
    if target is None or getattr(target, _TE_RESIDENT_UNDO_ATTR, None) is None:
        return 0
    return keep_h3_te_resident(text_encoder, logger = logger)


def yield_h3_te_resident(
    text_encoder: Any,
    nbytes: int,
    *,
    logger: Any = None,
) -> int:
    """Return up to ``nbytes`` of the conditioner's resident share to its memory mapping (last copied first: vision,
    then the deepest language layers), so the anonymous copies are freed. Returns bytes given back. Never raises."""
    try:
        target = getattr(text_encoder, "model", None)
        undo = getattr(target, _TE_RESIDENT_UNDO_ATTR, None) if target is not None else None
        if not undo or nbytes <= 0:
            return 0
        import torch

        freed = 0
        with torch.no_grad():
            while undo and freed < nbytes:
                group, tensor, mapped, copy = undo.pop()
                cpu = getattr(group, "cpu_param_dict", None)
                if cpu is not None and cpu.get(tensor) is copy:
                    cpu[tensor] = mapped
                if tensor.device.type == "cpu" and tensor.data_ptr() == copy.data_ptr():
                    tensor.data = mapped
                freed += copy.numel() * copy.element_size()
                del copy
        if logger is not None:
            logger.info(
                "video.h3_te_resident: %d MiB of the conditioner's resident share returned to its mapping "
                "(host RAM for a demoted denoiser block)",
                freed >> 20,
            )
        return freed
    except Exception as exc:  # noqa: BLE001
        if logger is not None:
            logger.warning("video.h3_te_resident: could not yield the resident share: %s", exc)
        return 0


def _copy_host_tensors_(entries: list, undo: Optional[list] = None) -> int:
    import torch
    from concurrent.futures import ThreadPoolExecutor

    def _fresh(host: Any) -> Any:
        out = torch.empty_like(host, memory_format = torch.contiguous_format)
        out.copy_(host)  # releases the GIL: parallel page faults keep the disk queue full
        return out

    with torch.no_grad(), ThreadPoolExecutor(_H3_TE_COPY_THREADS) as pool:
        fresh = list(pool.map(lambda e: _fresh(e[2]), entries))
    copied = 0
    with torch.no_grad():
        for (group, tensor, host), new in zip(entries, fresh):
            cpu = getattr(group, "cpu_param_dict", None)
            if cpu is not None and tensor in cpu:
                cpu[tensor] = new
            if tensor.device.type == "cpu" and tensor.data_ptr() == host.data_ptr():
                tensor.data = new
            if undo is not None:
                undo.append((group, tensor, host, new))
            copied += new.numel() * new.element_size()
    return copied


# Once every denoiser block is resident, leftover budget keeps conditioner leaf groups on device in read order (48 GB:
# ~0.8 s per render saved); requests needing more take it back first. Vision tower keeps streaming. =0 disables.
H3_TE_DEVICE_RESIDENT_ENV = "UNSLOTH_H3_TE_DEVICE_RESIDENT"
_TE_DEVICE_ATTR = "_unsloth_te_device"
_TE_SAVED_IO_ATTR = "_unsloth_te_saved_io"
# Planned device bytes per tensor: the allocator's 512-byte rounding plus an unsplit remainder (at most 1 MiB).
_ALLOC_ROUND = 512
_ALLOC_SPLIT_SLACK = 1 << 20


def h3_te_device_resident_enabled() -> bool:
    import os
    return str(os.environ.get(H3_TE_DEVICE_RESIDENT_ENV, "")).strip().lower() not in (
        "0",
        "off",
        "false",
        "no",
    )


def _planned_device_bytes(group: Any) -> int:
    from .video_minimax_h3_residency import _group_tensors, _tensor_bytes

    total = 0
    for t in _group_tensors(group):
        n = _tensor_bytes(t)
        total += _round_up(n, _ALLOC_ROUND) + (_ALLOC_SPLIT_SLACK if n > _ALLOC_SPLIT_SLACK else 0)
    return total


def _allocated_bytes(device: Any) -> Optional[int]:
    try:
        import torch

        dev = torch.device(device)
        if dev.type != "cuda" or not torch.cuda.is_available():
            return None
        return int(torch.cuda.memory_allocated(dev))
    except Exception:  # noqa: BLE001
        return None


def _te_resident_onload(group: Any) -> Any:
    """Keep the side-stream sync: the next group may still be arriving via the previous group's prefetch."""
    stream = getattr(group, "stream", None)

    def onload_(*_a: Any, **_k: Any) -> None:
        if stream is not None:
            stream.synchronize()

    try:
        import torch
        return torch.compiler.disable(onload_)
    except Exception:  # noqa: BLE001
        return onload_


def _te_noop(*_a: Any, **_k: Any) -> None:
    return None


class H3TeDeviceShare:
    """The streamed conditioner's leaf groups resident on the device: a prefix in read order (language layers first)."""

    def __init__(
        self,
        text_encoder: Any,
        device: Any,
        *,
        logger: Any = None,
    ) -> None:
        from .diffusion_memory import _offload_groups
        from .video_minimax_h3_residency import _group_tensors

        target = getattr(text_encoder, "model", None)
        self.device = device
        self.logger = logger
        self.groups: list = []
        self.layer_of: list = []
        names = {id(m): n for n, m in target.named_modules()} if target is not None else {}
        for group in _offload_groups(target) if target is not None else []:
            mods = list(getattr(group, "modules", None) or ())
            leader = mods[0] if mods else getattr(group, "onload_leader", None)
            parts = names.get(id(leader), "").split(".")
            if (
                "visual" in parts
                or getattr(group, "stream", None) is None
                or not isinstance(getattr(group, "cpu_param_dict", None), dict)
                or not _group_tensors(group)
            ):
                continue
            layer = None
            if "layers" in parts and parts.index("layers") + 1 < len(parts):
                index = parts[parts.index("layers") + 1]
                layer = int(index) if index.isdigit() else None
            self.groups.append(group)
            self.layer_of.append(layer)
        self.planned = [_planned_device_bytes(g) for g in self.groups]
        self.layers = len({x for x in self.layer_of if x is not None})
        self._logged: Optional[int] = None

    @property
    def usable(self) -> bool:
        return bool(self.groups)

    def resident_count(self) -> int:
        n = 0
        for group in self.groups:
            if not group.__dict__.get(_TE_DEVICE_ATTR):
                break
            n += 1
        return n

    def resident_bytes(self) -> int:
        """Measured at onload; planned bytes off CUDA."""
        return sum(
            int(group.__dict__.get(_TE_DEVICE_ATTR) or 0)
            for group in self.groups
            if group.__dict__.get(_TE_DEVICE_ATTR)
        )

    def plan(self, budget_bytes: int) -> int:
        n = 0
        left = int(budget_bytes)
        while n < len(self.groups) and self.planned[n] <= left:
            left -= self.planned[n]
            n += 1
        return n

    def shrink(self, n: int) -> None:
        import torch
        demoted = False
        for group in reversed(self.groups[n:]):
            if not group.__dict__.get(_TE_DEVICE_ATTR):
                continue
            if not demoted and torch.cuda.is_available():
                torch.cuda.current_stream().synchronize()
            saved_on, saved_off = group.__dict__.get(_TE_SAVED_IO_ATTR, (None, None))
            for name, saved in (("onload_", saved_on), ("offload_", saved_off)):
                if saved is not None:
                    setattr(group, name, saved)
                else:
                    group.__dict__.pop(name, None)
            group.__dict__.pop(_TE_DEVICE_ATTR, None)
            group.offload_()  # back to its host copy, which was never dropped
            demoted = True

    def grow(self, n: int) -> None:
        import torch
        for group in self.groups[:n]:
            if group.__dict__.get(_TE_DEVICE_ATTR):
                continue
            before = _allocated_bytes(self.device)
            stream = getattr(group, "stream", None)
            try:
                group.onload_()
                if stream is not None:
                    stream.synchronize()
                if torch.cuda.is_available():
                    torch.cuda.current_stream().synchronize()
            except Exception:
                group.offload_()  # a half-onloaded group back on its host copy
                raise
            after = _allocated_bytes(self.device)
            held = after - before if before is not None and after is not None else None
            if held is None or held <= 0:
                held = _planned_device_bytes(group)
            group.__dict__[_TE_SAVED_IO_ATTR] = (
                group.__dict__.get("onload_"),
                group.__dict__.get("offload_"),
            )
            group.onload_ = _te_resident_onload(group)
            group.offload_ = _te_noop
            group.__dict__[_TE_DEVICE_ATTR] = max(1, int(held))

    def report(
        self,
        budget_bytes: int,
        *,
        force: bool = False,
    ) -> None:
        n = self.resident_count()
        if self.logger is None or (n == self._logged and not force):
            return
        self._logged = n
        full = sum(
            1
            for layer in {x for x in self.layer_of if x is not None}
            if all(
                self.groups[i].__dict__.get(_TE_DEVICE_ATTR)
                for i, x in enumerate(self.layer_of)
                if x == layer
            )
        )
        self.logger.info(
            "video.h3_te_resident_device: %d of %d TE layers resident on %s (%.2f GB, %d of %d leaf groups, "
            "%.2f GB left beside the resident denoiser)",
            full,
            self.layers,
            self.device,
            self.resident_bytes() / 1e9,
            n,
            len(self.groups),
            max(0, budget_bytes) / 1e9,
        )


def h3_te_device_share(
    text_encoder: Any,
    device: Any,
    mode: Any,
    *,
    logger: Any = None,
) -> Optional[H3TeDeviceShare]:
    """None unless the conditioner streams with a side stream. Never raises."""
    try:
        if mode not in ("stream", "stream_lazy") or not h3_te_device_resident_enabled():
            return None
        share = H3TeDeviceShare(text_encoder, device, logger = logger)
        return share if share.usable else None
    except Exception as exc:  # noqa: BLE001 -- the conditioner streams as before
        if logger is not None:
            logger.warning("video.h3_te_resident_device: staying streamed: %s", exc)
        return None


# "0": the H3 conditioner records every hidden state again.
H3_TE_READ_CAPTURE_ENV = "UNSLOTH_H3_TE_READ_CAPTURE"


def h3_te_read_capture_enabled() -> bool:
    import os
    return str(os.environ.get(H3_TE_READ_CAPTURE_ENV, "")).strip().lower() not in (
        "0",
        "off",
        "false",
        "no",
    )


def _is_compiling() -> bool:
    try:
        import torch
        return bool(torch.compiler.is_compiling())
    except Exception:  # noqa: BLE001
        return False


def install_h3_te_read_capture(
    text_encoder: Any,
    *,
    layer: int = H3_TE_READ_LAYER,
    logger: Any = None,
) -> bool:
    """H3 reads only ``hidden_states[layer]``: run without recording and keep layer ``layer``'s input via a pre-hook.
    Past the DeepStack layers that input equals ``hidden_states[layer]`` in transformers 4.x and 5.x (5.x records
    those layers before the visual merge), so DeepStack layers are refused. Idempotent; never raises."""
    try:
        model = getattr(text_encoder, "model", None)
        layers = getattr(getattr(model, "language_model", None), "layers", None)
        if model is None or layers is None or not isinstance(layer, int):
            return False
        if getattr(model, "_unsloth_h3_read_capture", None) is not None:
            return True
        config = getattr(text_encoder, "config", None)
        deepstack = getattr(
            getattr(config, "vision_config", None), "deepstack_visual_indexes", None
        )
        deepstack_layers = len(deepstack) if deepstack is not None else None
        if deepstack_layers is None or not (deepstack_layers < layer < len(layers)):
            return False
        stock = model.forward
        target = layers[layer]
    except Exception:  # noqa: BLE001 - optimisation only
        return False

    @functools.wraps(stock)
    def forward(*args: Any, **kwargs: Any) -> Any:
        # Hook registration is untraceable: a compiled caller takes the stock forward (same states, no graph break).
        if (
            not kwargs.get("output_hidden_states")
            or _is_compiling()
            or not h3_te_read_capture_enabled()
        ):
            return stock(*args, **kwargs)
        captured: list = []

        def _keep(_module: Any, hook_args: tuple, hook_kwargs: dict) -> None:
            captured.append(hook_args[0] if hook_args else hook_kwargs.get("hidden_states"))

        handle = target.register_forward_pre_hook(_keep, with_kwargs = True)
        try:
            out = stock(*args, **{**kwargs, "output_hidden_states": False})
        finally:
            handle.remove()
        if len(captured) != 1 or captured[0] is None:
            del out, captured
            return stock(*args, **kwargs)
        states = [None] * (len(layers) + 1)
        states[layer] = captured[0]
        out.hidden_states = tuple(states)
        return out

    forward.__wrapped__ = stock
    try:
        model.forward = forward
        model._unsloth_h3_read_capture = layer
    except Exception:  # noqa: BLE001
        return False
    if logger is not None:
        logger.info("video.h3_te_read_capture: on (keeps hidden_states[%d] only)", layer)
    return True
