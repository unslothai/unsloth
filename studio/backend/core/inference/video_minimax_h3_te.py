# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Load MiniMax-H3's Qwen3-VL conditioner from a hosted QUANTIZED checkpoint.

The H3 Diffusers path runs every component under ``ComponentsManager.enable_auto_cpu_offload``,
so its VRAM floor is the LARGEST SINGLE RESIDENT COMPONENT, not their sum. The released bfloat16
conditioner is 66.7 GB and the released denoiser is 66.3 GB, which is why seeding a 20 GB
pre-quantized denoiser moved that floor by nothing at all: the encoder simply became the largest
component instead. The encoder is the only remaining lever.

``diffusion_te_prequant`` cannot serve this one. Its artifacts are layerwise-fp8 STORAGE casts of
a dense encoder saved as ``torch.load``-able state dicts, which halve the bytes but keep every
released tensor. The hosted H3 conditioner is a different artifact in two independent ways, and
both savings stack:

  1. It carries 50 of the released 64 decoder layers, and no ``lm_head`` and no final ``norm``
     (47.97 GiB against 62.14 GiB for ``MiniMaxAI/MiniMax-H3`` ``text_encoder/``).
  2. Those 50 layers' four attention and three MLP projections are ConvRot INT8
     (25.28 GiB, i.e. 27.14 GB resident).

Dropping the tail is LOSSLESS here, not an approximation. MiniMax-H3 conditions the transformer on
``hidden_states[50]`` of the conditioner and never touches the language-model head -- see
``diffusers/modular_pipelines/minimax_h3/modular_pipeline.py`` (``text_encoder_layer`` returns 50,
"MiniMax-H3 reads `hidden_states[50]`, not the final one") and ``.../encoders.py``
``get_qwen3vl_prompt_embeds``, which calls ``text_encoder.model(..., output_hidden_states=True)``
and returns ``outputs.hidden_states[text_encoder_layer]``. Decoder layers 51-64 and ``lm_head``
therefore cannot influence the conditioning by construction. Comparing the hosted file's tensor
names against ``MiniMaxAI/MiniMax-H3``'s own shard index confirms the artifact drops EXACTLY that
set and nothing else: 902 names that map 1:1 onto the released ones, the remainder being decoder
layers 50-63 (0-based), ``model.language_model.norm.weight`` and ``lm_head.weight``.

There is one mechanical catch, and it is the reason this module builds 51 layers rather than 50.
``output_hidden_states`` yields ``[embeddings, layer_0_out, ..., layer_{N-1}_out]`` where the LAST
entry is post-``norm``. A stack truncated to exactly 50 layers therefore returns a NORMALIZED
``hidden_states[50]``, which is not the conditioning the released weights were trained against;
diffusers refuses that case outright (``encoders.py``: "The last hidden state of a stack truncated
to exactly 50 layers is post-norm"). Building a 51st slot that passes its input through unchanged
puts ``hidden_states[50]`` back where it belongs -- the raw output after 50 real layers, bit-identical
to what the full 64-layer conditioner produces there -- for no weights and no compute, and keeps
``len(layers) == config.num_hidden_layers`` honest so the diffusers guard passes on a true statement.

Best-effort throughout, exactly like ``diffusion_prequant``: any missing / unreadable / mismatched
artifact returns None and the caller loads the released bfloat16 encoder instead. Inert with
nothing configured.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Optional

from .diffusion_convrot import (  # noqa: F401  (re-export: callers and tests name them here)
    build_convrot_hadamard,
    rotate_convrot_activation,
)

# Not H3_COMPONENT_REPO: the VAEs moved to the GGUF mirror, the conditioner did not.
H3_TE_QUANT_REPO = "unsloth/MiniMax-H3-FP8"
# Legacy repack for caches predating the move; pairing lives in _SD_CPP_LEGACY_SOURCES.
H3_LEGACY_TE_QUANT_REPO = "Comfy-Org/MiniMax-H3"

# nvfp4 (AWQ, two-level scales) deliberately absent: needs its own loader, kernel and verification.
H3_TE_QUANT_FILES: dict[str, str] = {
    "int8": "text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
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

_SCALE_SUFFIX = ".weight_scale"
_QUANT_SUFFIX = ".comfy_quant"


def h3_te_quant_scheme(mode: Optional[str]) -> Optional[str]:
    """The hosted-conditioner scheme for a requested ``text_encoder_quant``, or None.

    Pure and non-raising: the request has already been validated by ``normalize_te_quant`` at the
    route, and a scheme with no hosted artifact simply keeps the released bfloat16 encoder."""
    if mode is None:
        return None
    normalized = str(mode).strip().lower().replace("-", "_")
    return normalized if normalized in H3_TE_QUANT_FILES else None


def h3_te_quant_filename(scheme: Optional[str]) -> Optional[str]:
    """The hosted checkpoint's path inside ``H3_TE_QUANT_REPO`` for ``scheme``, or None."""
    return H3_TE_QUANT_FILES.get(scheme or "")


def h3_te_quant_source(scheme: Optional[str]) -> str:
    """The repo to fetch the hosted quantized conditioner for ``scheme`` from: our mirror, or the
    repack it was mirrored from when this install already holds that exact artifact under the old
    id.

    The conditioner's half of ``h3_component_source``, and it matters more than the VAEs do: the
    artifact is ~27 GB, and the load that wants it has already dropped the dense encoder shards
    from its pull. So on an upgraded install a live re-download is not merely slow -- offline it
    leaves the pipeline with no encoder at all, because the base snapshot it would fall back to
    was never staged either.

    PURE, and the same shared owner (``prefer_cached_legacy_source``, both cache roots) the VAEs
    use, so planning, prefetching and loading cannot name different repos.
    """
    filename = h3_te_quant_filename(scheme)
    if filename is None:
        return H3_TE_QUANT_REPO
    try:
        from .diffusion_families import prefer_cached_legacy_source
        return prefer_cached_legacy_source(H3_TE_QUANT_REPO, (filename,))
    except Exception:  # noqa: BLE001 -- an unreadable cache just means "not cached"
        return H3_TE_QUANT_REPO


def h3_te_resident_gb(scheme: Optional[str], *, bf16_gb: float) -> float:
    """Resident decimal GB of the conditioner this pick loads: the hosted quantized size when one
    exists for ``scheme``, else the released bfloat16 ``bf16_gb``."""
    resolved = h3_te_quant_scheme(scheme)
    return H3_TE_QUANT_RESIDENT_GB[resolved] if resolved else bf16_gb


# H is a normalized regular Hadamard (H @ H == I); dequantizing without the rotation gives noise.
@lru_cache(maxsize = None)
def _int8_convrot_linear_class() -> Any:
    """The ConvRot INT8 ``nn.Linear`` stand-in, built lazily so importing this module never imports
    torch, and built exactly ONCE.

    One load already shares a single class across every projection, so the cache is about the
    SECOND load in a process: a fresh class there is a fresh ``___check_type_id`` guard, which
    retraces every compiled block that survived the first one. Same reason the denoiser's
    ``convrot_linear_class`` is cached."""
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
    """The hosted quantized Qwen3-VL conditioner for ``scheme``, on CPU, ready to seed into the
    modular pipeline; None on any problem so the caller loads the released bfloat16 encoder.

    ``base`` supplies the component CONFIG only (``<base>/text_encoder/config.json``); every weight
    comes from the hosted artifact, so the 62 GB dense encoder is never fetched. ``local_base`` is
    the already-staged snapshot of ``base`` when there is one, and it is preferred: the scoped
    pre-download keeps every component config precisely so the meta-init loaders can read them
    locally, and reading the config back out of the snapshot cannot go to the network at all.
    ``cache_dir`` pins the config resolution to the live cache root for the hub-id case, exactly as
    the artifact download above and every other loader call in this backend do -- unset, it
    resolves through huggingface_hub's import-time constant instead and can re-download into a root
    Unsloth no longer reads (or fail outright on an offline host that has already staged it).

    ``local_files_only`` is a load nobody asked for, which may not fetch anything. The artifact is
    ~27 GB, and the caller's staging phase (``_fetch_h3_te_quant``) has already accepted it -- so
    without the flag this is where that promise is broken, after the resident pipeline was evicted.
    It rides with the same other-root reuse the stager uses: the stager accepts a copy living only
    under huggingface_hub's import-time root, so a lookup pinned to ``cache_dir`` alone would refuse
    an artifact the load was cleared on and drop to the dense encoder the base pull already left
    behind. A genuine miss still returns None through the handler below.

    CPU on purpose: ``enable_auto_cpu_offload`` owns placement for every component, and a
    pre-placed encoder would only be moved again."""
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

        config = transformers.AutoConfig.from_pretrained(
            local_base or base,
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


def h3_te_pin_allowed(payload_bytes: int = 0) -> bool:
    """Whether ``payload_bytes`` (at least one arena) may be pinned. The full payload, not one arena: the hosted
    weights are mmap views already counted as available, so pinning them takes that much. Honours the group offload
    pin switch and the ~1 GiB pinned cap of Windows / WSL. Never raises."""
    try:
        import os

        from .diffusion_memory import GROUP_OFFLOAD_PIN_ENV, _pin_budget_mib, _pinned_memory_capped

        forced = str(os.environ.get(GROUP_OFFLOAD_PIN_ENV, "")).strip().lower()
        if forced in ("0", "off", "false", "no"):
            return False
        if forced in ("1", "on", "true", "yes"):
            return True
        if _pinned_memory_capped():
            return False
        budget = _pin_budget_mib()
        return budget is not None and budget >= max(_PIN_ARENA_BYTES, int(payload_bytes)) >> 20
    except Exception:  # noqa: BLE001 -- unanswerable: stream unpinned
        return False


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


def pin_module_in_place(
    module: Any,
    *,
    arena_bytes: int = _PIN_ARENA_BYTES,
    _arena_factory: Any = None,
    repoint: bool = False,
) -> int:
    """Move every CPU parameter / buffer of ``module`` into pinned host arenas, in place; returns pinned bytes.

    Tensors are REPLACED in ``_parameters`` / ``_buffers``, not re-pointed via ``.data``: the hosted safetensors
    tensors are views of one mmap, and ``.data =`` keeps the view's ``_base`` alive (27 GB of the mapping stayed
    resident). Pinned, non-CPU and subclass tensors are left alone; tied tensors stay tied."""
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
    todo.sort(key = lambda item: item[0].numel() * item[0].element_size(), reverse = True)
    arenas: list[list] = []  # [arena tensor, used bytes]
    pinned = 0
    for tensor, owners in todo:
        nbytes = tensor.numel() * tensor.element_size()
        need = _round_up(nbytes, _PIN_ALIGN)
        slot = next((a for a in arenas if a[0].numel() - a[1] >= need), None)
        if slot is None:
            size = max(int(arena_bytes), 1 << (need - 1).bit_length())
            arena = (
                _arena_factory(size)
                if _arena_factory is not None
                else torch.empty(size, dtype = torch.uint8, pin_memory = True)
            )
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
    ``h3_te_pin_allowed``. Returns ``"stream"`` (pinned) / ``"stream_lazy"``, or None with nothing changed."""
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

    params = inspect.signature(apply_group_offloading).parameters
    if pin is None:
        pin = h3_te_pin_allowed(_module_payload_bytes(text_encoder))
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
        "use_stream": True,
    }
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
    mode = "stream" if pin else "stream_lazy"
    if logger is not None:
        logger.info(
            "video.h3_te_stream: conditioner streamed leaf by leaf on %s (%s host copy)",
            device,
            "pinned" if pin else "pageable",
        )
    return mode
