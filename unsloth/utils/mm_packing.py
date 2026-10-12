# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Padding-free (packed) rows for multimodal documents (images, audio).

Each document is collated alone, then concatenated; ``position_ids`` restart per document, from
which transformers builds a mask so image blocks never cross a document boundary. Qwen2-VL style
models (4-row multimodal RoPE position ids) are refused.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch

__all__ = [
    "pack_multimodal_documents",
    "prepare_packed_multimodal_model",
    "SUPPORTED_PACKED_MULTIMODAL_MODEL_TYPES",
]

SUPPORTED_PACKED_MULTIMODAL_MODEL_TYPES = frozenset({"gemma4"})

_TOKEN_KEYS = ("input_ids", "labels", "mm_token_type_ids", "token_type_ids")
# Media key -> value padding a mismatched trailing axis (the towers mask these out).
_MEDIA_PAD = {
    "pixel_values": 0.0,
    "image_position_ids": -1,
    "pixel_position_ids": -1,
    "input_features": 0.0,
    "input_features_mask": False,
}
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


def pack_multimodal_documents(docs: Sequence[dict], ignore_index: int = -100) -> dict:
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
    out["position_ids"] = torch.cat([torch.arange(n, device = device) for n in lengths])[None]
    out["packed_seq_lengths"] = torch.tensor(lengths, dtype = torch.int32)

    for key, pad in _MEDIA_PAD.items():
        parts = [d[key] for d in docs if key in d and d[key] is not None]
        if not parts:
            continue
        trailing = [max(p.shape[i] for p in parts) for i in range(1, parts[0].dim())]
        if any(p.dim() != parts[0].dim() for p in parts):
            raise ValueError(f"Unsloth: {key} rank differs between documents")
        out[key] = torch.cat([_pad_to(p, trailing, pad) for p in parts], dim = 0)
    if "input_features" in out and "input_features_mask" not in out:
        raise ValueError("Unsloth: input_features without input_features_mask cannot be packed")
    return out


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


def prepare_packed_multimodal_model(model) -> None:
    types = _model_types(getattr(model, "config", None))
    if not types & SUPPORTED_PACKED_MULTIMODAL_MODEL_TYPES:
        raise ValueError(
            "Unsloth: padding_free=True in UnslothVisionDataCollator is only supported for "
            f"{sorted(SUPPORTED_PACKED_MULTIMODAL_MODEL_TYPES)} for now, got "
            f"{sorted(types) or 'an unknown model'}. Set padding_free=False."
        )
    from .packing import _enable_hf_packed_attention, mark_allow_overlength

    _enable_hf_packed_attention()
    mark_allow_overlength(model)
