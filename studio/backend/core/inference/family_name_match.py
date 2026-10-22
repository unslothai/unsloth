# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Separator-insensitive family-name matching (``qwen_image_2.1`` == ``Qwen-Image-2.1``) shared by
the image and video family tables. Path separators stay boundaries."""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Optional

_SEPARATOR_RUN = re.compile(r"[-_.\s]+")


@lru_cache(maxsize = 4096)
def normalize_family_name(text: str) -> str:
    return _SEPARATOR_RUN.sub("-", (text or "").lower())


@lru_cache(maxsize = 1024)
def _token_pattern(token: str) -> Optional[re.Pattern]:
    tok = normalize_family_name(token).strip("-")
    if not tok:
        return None
    return re.compile(r"(?:^|[-/\\])" + re.escape(tok) + r"(?:$|[-/\\])")


def token_in_name(token: str, needle: str) -> bool:
    """Whole-segment match after normalising: ``kontext`` never matches ``kontextual``."""
    pattern = _token_pattern(token)
    return pattern is not None and pattern.search(normalize_family_name(needle)) is not None


def token_length(token: str) -> int:
    return len(normalize_family_name(token).strip("-"))


def name_key_in(key: str, needle: str) -> bool:
    """Substring test, raw then normalised (the defaults tables' contract)."""
    low = (needle or "").lower()
    k = (key or "").lower()
    if k and k in low:
        return True
    nk = normalize_family_name(k)
    return bool(nk) and nk in normalize_family_name(low)


_LOCAL_PATH = re.compile(r"^(?:[/\\~]|\.{1,2}[/\\]|[a-zA-Z]:[/\\])")
_PATH_SPLIT = re.compile(r"[/\\]+")
_WEIGHT_FILE_SUFFIXES = (".gguf", ".safetensors", ".sft", ".ckpt", ".pt", ".pth", ".bin")
# Folders holding a model's files without naming it (diffusers component, ComfyUI kind folder, quant tier, and an
# original layout's expert / variant folder: Wan2.2 A14B's high_noise_model / low_noise_model, HunyuanVideo-1.5's
# transformer/480p_t2v / 720p_i2v...). The variant folder stays in the identity, so its tier still routes and an i2v
# folder still meets the family's refusal. Not the *_sr_* folders: a super-resolution transformer (98 input channels,
# SR conditioning) is no T2V model, so it must stay unrecognized rather than borrow the T2V family from its parent.
_CONTAINER_SEGMENT = re.compile(
    r"^(?:transformer(?:_\d)?|unet|vae|text_encoder(?:_\d)?|tokenizer(?:_\d)?|scheduler|"
    r"high_noise_model|low_noise_model|\d{3,4}p_(?:t2v|i2v)(?:_[a-z0-9]+)*|"
    r"diffusion_models|split_files|gguf|weights?|checkpoints?|ckpts?|models?|files|safetensors|original|"
    r"(?:u?d-)?(?:i?q\d\w*|f16|bf16|f32|fp8\w*|fp16|nvfp4|mxfp4))$",
    re.IGNORECASE,
)


def model_identity(text: Optional[str]) -> str:
    """The part of ``text`` naming the model: repo ids unchanged; a local path drops parent folders (an arbitrary
    keyword there must not pick a family), and an HF cache path ``models--org--name/snapshots/<sha>/f`` -> ``org/name/f``."""
    text = text or ""
    if not _LOCAL_PATH.match(text):
        return text
    parts = [p for p in _PATH_SPLIT.split(text) if p]
    for i in range(len(parts) - 1, -1, -1):
        seg = parts[i]
        if seg.lower().startswith("models--") and len(seg) > len("models--"):
            rest = parts[i + 1 :]
            if len(rest) >= 2 and rest[0].lower() == "snapshots":
                rest = rest[2:]
            return "/".join([seg[len("models--") :].replace("--", "/"), *rest])
    if not parts:
        return ""
    start = len(parts) - 1
    if start > 0 and parts[-1].lower().endswith(_WEIGHT_FILE_SUFFIXES):
        start -= 1
    while 0 < start and _CONTAINER_SEGMENT.match(parts[start]):
        start -= 1
    return "/".join(parts[start:])
