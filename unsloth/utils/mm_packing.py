# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Packed multimodal rows: docs collated alone, concatenated; position_ids confine images to own doc.
mRoPE families (Qwen2-VL / 2.5-VL / 3-VL / 3.5) get [4, 1, T] position_ids: a per-document text plane (the
packed mask and varlen boundaries) over the three mRoPE planes each document's get_rope_index gives alone."""

from __future__ import annotations

import inspect
import os
from typing import Any, Callable, Optional, Sequence

import torch

__all__ = [
    "MROPE_PACKED_MULTIMODAL_MODEL_TYPES",
    "pack_multimodal_documents",
    "prepare_packed_multimodal_model",
    "SUPPORTED_PACKED_MULTIMODAL_MODEL_TYPES",
]

# Vision towers attend per image (cu_seqlens from the grid), so only the text model needs document bounds.
MROPE_PACKED_MULTIMODAL_MODEL_TYPES = frozenset(
    {"qwen2_vl", "qwen2_5_vl", "qwen3_vl", "qwen3_vl_moe", "qwen3_5", "qwen3_5_moe"}
)
_MROPE_ENV = (
    "UNSLOTH_VLM_MROPE_PACKING"  # "0" turns the mRoPE families off again (padding_free refused)
)


def _mrope_packing_enabled() -> bool:
    return os.environ.get(_MROPE_ENV, "1").strip().lower() not in ("0", "false", "no", "off")


SUPPORTED_PACKED_MULTIMODAL_MODEL_TYPES = frozenset({"gemma4"}) | (
    MROPE_PACKED_MULTIMODAL_MODEL_TYPES if _mrope_packing_enabled() else frozenset()
)

_TOKEN_KEYS = ("input_ids", "labels", "mm_token_type_ids", "token_type_ids")
# Media key -> value padding a mismatched trailing axis (the towers mask these out).
_MEDIA_PAD = {
    "pixel_values": 0.0,
    "image_position_ids": -1,
    "pixel_position_ids": -1,
    "input_features": 0.0,
    "input_features_mask": False,
    # mRoPE families: patch rows / grids / per-video seconds, concatenated in document order.
    "pixel_values_videos": 0.0,
    "image_grid_thw": 0,
    "video_grid_thw": 0,
    "second_per_grid_ts": 0.0,
}
_MROPE_ONLY = ("pixel_values_videos", "image_grid_thw", "video_grid_thw", "second_per_grid_ts")
_DROPPED = ("attention_mask", "position_ids", "packed_seq_lengths")


def _pad_to(t: torch.Tensor, shape: Sequence[int], value) -> torch.Tensor:
    if tuple(t.shape[1:]) == tuple(shape):
        return t
    out = t.new_full((t.shape[0], *shape), value)
    out[(slice(None), *(slice(0, n) for n in t.shape[1:]))] = t
    return out


def _strip_padding(doc: dict) -> dict:
    # Batch-of-one can still be padded (pad_to_multiple_of).
    am = doc.get("attention_mask")
    if am is None or bool(am.all()):
        return doc
    keep = am[0].bool()
    doc = dict(doc)
    for key in _TOKEN_KEYS:
        if key in doc and torch.is_tensor(doc[key]) and doc[key].dim() == 2:
            doc[key] = doc[key][:, keep]
    return doc


def pack_multimodal_documents(
    docs: Sequence[dict],
    ignore_index: int = -100,
    rope_index: Optional[Callable] = None,
) -> dict:
    """rope_index (from prepare_packed_multimodal_model): per-document 3D positions for mRoPE models."""
    if not docs:
        raise ValueError("Unsloth: no documents to pack")
    docs = [_strip_padding(d) for d in docs]
    for d in docs:
        if d["input_ids"].shape[0] != 1:
            raise ValueError("Unsloth: pack_multimodal_documents expects batch-of-one documents")
    unknown = (
        set().union(*(d.keys() for d in docs)) - set(_TOKEN_KEYS) - set(_MEDIA_PAD) - set(_DROPPED)
    )
    if unknown:
        # Unproven keys: refuse rather than drop silently.
        raise ValueError(f"Unsloth: cannot pack multimodal key(s) {sorted(unknown)}")
    if rope_index is None:
        grids = sorted(k for k in _MROPE_ONLY if any(k in d for d in docs))
        if grids:
            # Without per-document mRoPE positions the images would sit at text positions.
            raise ValueError(
                f"Unsloth: cannot pack multimodal key(s) {grids} without mRoPE positions"
            )

    lengths = [int(d["input_ids"].shape[1]) for d in docs]
    out: dict[str, Any] = {}
    for key in _TOKEN_KEYS:
        present = [key in d for d in docs]
        if not any(present):
            continue
        if not all(present):
            if key == "labels":
                raise ValueError("Unsloth: some documents have labels and some do not")
            parts = [d[key] if key in d else torch.zeros_like(d["input_ids"]) for d in docs]
        else:
            parts = [d[key] for d in docs]
        out[key] = torch.cat(parts, dim = 1)

    if "labels" in out:
        labels = out["labels"].clone()
        start = 0
        for n in lengths[:-1]:
            start += n
            labels[0, start] = ignore_index
        out["labels"] = labels

    device = out["input_ids"].device
    text = torch.cat([torch.arange(n, device = device) for n in lengths])[None]
    if rope_index is None:
        out["position_ids"] = text
    else:
        planes = torch.cat([rope_index(d) for d in docs], dim = -1)
        out["position_ids"] = torch.cat([text[None].to(planes.device), planes], dim = 0)
    out["packed_seq_lengths"] = torch.tensor(lengths, dtype = torch.int32)

    for key, pad in _MEDIA_PAD.items():
        parts = [d[key] for d in docs if key in d and d[key] is not None]
        if not parts:
            continue
        trailing = [max(p.shape[i] for p in parts) for i in range(1, parts[0].dim())]
        if any(p.dim() != parts[0].dim() for p in parts):
            raise ValueError(f"Unsloth: {key} rank differs between documents")
        if rope_index is not None and any(list(p.shape[1:]) != trailing for p in parts):
            # mRoPE towers read flat patch rows by grid: padding a row would invent patches.
            raise ValueError(f"Unsloth: {key} rows differ in width between documents")
        out[key] = torch.cat([_pad_to(p, trailing, pad) for p in parts], dim = 0)
    if "input_features" in out and "input_features_mask" not in out:
        raise ValueError("Unsloth: input_features without input_features_mask cannot be packed")
    return out


class _DocumentRopeIndex:
    """[3, 1, n] mRoPE positions of one document alone, exactly as the model computes them unpacked."""

    def __init__(self, get_rope_index: Callable, config):
        self.get_rope_index = get_rope_index
        try:
            self.params = set(inspect.signature(get_rope_index).parameters)
        except (TypeError, ValueError):
            self.params = set()
        # Model-type ids (text 0, image 1, video 2) when the processor did not emit mm_token_type_ids.
        self.token_types = [
            (getattr(config, name, None), value)
            for name, value in (("image_token_id", 1), ("video_token_id", 2))
        ]

    def _token_types(self, input_ids):
        types = torch.zeros_like(input_ids, dtype = torch.int32)
        for token_id, value in self.token_types:
            if isinstance(token_id, int):
                types[input_ids == token_id] = value
        return types

    def __call__(self, doc: dict) -> torch.Tensor:
        input_ids = doc["input_ids"]
        kwargs = {}
        for key in ("image_grid_thw", "video_grid_thw", "second_per_grid_ts", "mm_token_type_ids"):
            if key in self.params and doc.get(key) is not None:
                kwargs[key] = doc[key]
        if "mm_token_type_ids" in self.params and "mm_token_type_ids" not in kwargs:
            kwargs["mm_token_type_ids"] = self._token_types(input_ids)
        if "attention_mask" in self.params:
            kwargs["attention_mask"] = torch.ones_like(input_ids)
        with torch.no_grad():
            positions = self.get_rope_index(input_ids, **kwargs)
        if isinstance(positions, (tuple, list)):
            positions = positions[0]
        n = int(input_ids.shape[1])
        if not torch.is_tensor(positions) or tuple(positions.shape) != (3, 1, n):
            shape = (
                tuple(positions.shape) if torch.is_tensor(positions) else type(positions).__name__
            )
            raise ValueError(f"Unsloth: get_rope_index returned {shape}, expected (3, 1, {n})")
        return positions.to(torch.long)


def _find_rope_index(model) -> Optional[_DocumentRopeIndex]:
    """get_rope_index of the innermost module defining it (PEFT / CausalLM wrappers delegate)."""
    found = None
    for module in getattr(model, "modules", lambda: [])():
        if callable(getattr(type(module), "get_rope_index", None)):
            found = module
    if found is None:
        return None
    return _DocumentRopeIndex(found.get_rope_index, getattr(found, "config", None))


def _model_types(config) -> set:
    types = set()
    stack = [config]
    while stack:
        c = stack.pop()
        if c is None:
            continue
        mt = getattr(c, "model_type", None)
        if mt:
            types.add(mt)
        for sub in ("text_config", "vision_config", "audio_config"):
            s = getattr(c, sub, None)
            if s is not None and s is not c:
                stack.append(s)
    return types


def _transformers_isolates_packed_documents() -> bool:
    """Packed rows carry no mask: transformers derives the per-document one from the reset position_ids, which
    masking_utils does only on torch >= 2.6 (and not at all before masking_utils). Without it documents mix."""
    try:
        from transformers import masking_utils
    except Exception:
        return False
    if not hasattr(masking_utils, "find_packed_sequence_indices"):
        return False
    # transformers 4.x gates the packed mask on this flag; 5.x requires torch >= 2.6 and has no flag.
    return bool(getattr(masking_utils, "_is_torch_greater_or_equal_than_2_6", True))


def prepare_packed_multimodal_model(model) -> Optional[_DocumentRopeIndex]:
    """Refuses unsupported models; returns the per-document rope_index for mRoPE ones, else None."""
    types = _model_types(getattr(model, "config", None))
    if not types & SUPPORTED_PACKED_MULTIMODAL_MODEL_TYPES:
        raise ValueError(
            "Unsloth: padding_free=True in UnslothVisionDataCollator is only supported for "
            f"{sorted(SUPPORTED_PACKED_MULTIMODAL_MODEL_TYPES)} for now, got "
            f"{sorted(types) or 'an unknown model'}. Set padding_free=False."
        )
    if not _transformers_isolates_packed_documents():
        raise ValueError(
            "Unsloth: padding_free=True in UnslothVisionDataCollator needs transformers to build a "
            "per-document attention mask from packed position_ids, which this install does not do "
            f"(torch {torch.__version__}; it needs torch >= 2.6). Packed documents would attend to "
            "each other. Upgrade torch or set padding_free=False."
        )
    rope_index = None
    if types & MROPE_PACKED_MULTIMODAL_MODEL_TYPES and not types & {"gemma4"}:
        rope_index = _find_rope_index(model)
        if rope_index is None:
            raise ValueError(
                "Unsloth: padding_free=True needs the model's get_rope_index for per-document mRoPE "
                f"positions, and {type(model).__name__} has none. Set padding_free=False."
            )
        from .packing import _iter_stateful_mixers, patch_hybrid_linear_attention_varlen

        # Qwen3.5 linear-attention layers carry state across the row unless they run varlen.
        if isinstance(model, torch.nn.Module) and _iter_stateful_mixers(model):
            if not patch_hybrid_linear_attention_varlen(model):
                raise ValueError(
                    "Unsloth: padding_free=True needs per-document varlen kernels for the "
                    "linear-attention layers, which are unavailable here (see the warning above). "
                    "Set padding_free=False."
                )
    from .packing import _enable_hf_packed_attention, mark_allow_overlength

    _enable_hf_packed_attention()
    mark_allow_overlength(model)
    return rope_index
