# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Separator-insensitive family-name matching shared by the image and video family tables.

ComfyUI, HF and GGUF repackers spell one model many ways (``qwen_image_2.1_bf16``,
``Qwen-Image-2.1``, ``flux-2-klein-4b``, ``FLUX.2-klein``, ``wan2.2_ti2v_5B``,
``Wan2_2-TI2V-5B``), so both a family token and the name it is looked up in are lowered and every
run of ``_``, ``-``, ``.`` or whitespace is folded to one ``-`` before the whole-segment test.
Path separators stay boundaries, so a token never matches across them. Pure string code."""

from __future__ import annotations

import re

_SEPARATOR_RUN = re.compile(r"[-_.\s]+")


def normalize_family_name(text: str) -> str:
    """Lowercase ``text`` and fold every ``_ - .`` / whitespace run into a single ``-``."""
    return _SEPARATOR_RUN.sub("-", (text or "").lower())


def token_in_name(token: str, needle: str) -> bool:
    """True when ``token`` is a whole ``-`` / path-delimited segment run of ``needle`` once both are
    normalised, so ``qwen-image`` matches ``qwen_image_bf16`` but ``kontext`` not ``kontextual``."""
    tok = normalize_family_name(token).strip("-")
    if not tok:
        return False
    hay = normalize_family_name(needle)
    return re.search(r"(?:^|[-/\\])" + re.escape(tok) + r"(?:$|[-/\\])", hay) is not None


def token_length(token: str) -> int:
    """Specificity of a token for longest-match ranking, independent of how it is spelled."""
    return len(normalize_family_name(token).strip("-"))


def name_key_in(key: str, needle: str) -> bool:
    """Substring test (the defaults / variant tables' contract), tried raw first and then with both
    sides normalised, so ``z-image-turbo`` also keys ``z_image_turbo_bf16``."""
    low = (needle or "").lower()
    k = (key or "").lower()
    if k and k in low:
        return True
    nk = normalize_family_name(k)
    return bool(nk) and nk in normalize_family_name(low)
