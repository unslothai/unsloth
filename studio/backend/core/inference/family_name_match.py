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
