# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Shared Studio attention backend policy.

The Studio API exposes logical attention choices instead of framework-
specific implementation names. Backends map these logical names to the
best implementation they support and fall back without hard failing.
"""

from __future__ import annotations

from typing import Optional


STUDIO_ATTENTION_BACKEND_AUTO = "auto"
STUDIO_ATTENTION_BACKEND_FLASH = "flash"
STUDIO_ATTENTION_BACKEND_SDPA = "sdpa"
STUDIO_ATTENTION_BACKEND_FLEX = "flex"
STUDIO_ATTENTION_BACKEND_XFORMERS = "xformers"
STUDIO_ATTENTION_BACKEND_DEFAULT = "default"

STUDIO_ATTENTION_BACKENDS = (
    STUDIO_ATTENTION_BACKEND_AUTO,
    STUDIO_ATTENTION_BACKEND_FLASH,
    STUDIO_ATTENTION_BACKEND_SDPA,
    STUDIO_ATTENTION_BACKEND_FLEX,
    STUDIO_ATTENTION_BACKEND_XFORMERS,
)

STUDIO_ATTENTION_AUTO_FALLBACK_ORDER = (
    STUDIO_ATTENTION_BACKEND_FLASH,
    STUDIO_ATTENTION_BACKEND_SDPA,
    STUDIO_ATTENTION_BACKEND_FLEX,
    STUDIO_ATTENTION_BACKEND_XFORMERS,
    STUDIO_ATTENTION_BACKEND_DEFAULT,
)


def normalize_attention_backend(value: Optional[str]) -> str:
    """Normalize the public attention backend request."""

    if value is None:
        return STUDIO_ATTENTION_BACKEND_AUTO
    normalized = str(value).strip().lower()
    if not normalized:
        return STUDIO_ATTENTION_BACKEND_AUTO
    if normalized not in STUDIO_ATTENTION_BACKENDS:
        allowed = ", ".join(STUDIO_ATTENTION_BACKENDS)
        raise ValueError(f"attention_backend must be one of: {allowed}")
    return normalized


def attention_fallback_order(requested: Optional[str]) -> tuple[str, ...]:
    """Return the logical fallback order for a public attention request."""

    normalized = normalize_attention_backend(requested)
    if normalized in (
        STUDIO_ATTENTION_BACKEND_AUTO,
        STUDIO_ATTENTION_BACKEND_FLASH,
    ):
        return STUDIO_ATTENTION_AUTO_FALLBACK_ORDER
    if normalized == STUDIO_ATTENTION_BACKEND_SDPA:
        return (
            STUDIO_ATTENTION_BACKEND_SDPA,
            STUDIO_ATTENTION_BACKEND_FLEX,
            STUDIO_ATTENTION_BACKEND_XFORMERS,
            STUDIO_ATTENTION_BACKEND_DEFAULT,
        )
    if normalized == STUDIO_ATTENTION_BACKEND_FLEX:
        return (
            STUDIO_ATTENTION_BACKEND_FLEX,
            STUDIO_ATTENTION_BACKEND_SDPA,
            STUDIO_ATTENTION_BACKEND_XFORMERS,
            STUDIO_ATTENTION_BACKEND_DEFAULT,
        )
    if normalized == STUDIO_ATTENTION_BACKEND_XFORMERS:
        return (
            STUDIO_ATTENTION_BACKEND_XFORMERS,
            STUDIO_ATTENTION_BACKEND_SDPA,
            STUDIO_ATTENTION_BACKEND_DEFAULT,
        )
    return (STUDIO_ATTENTION_BACKEND_DEFAULT,)


def supported_attention_options() -> dict[str, object]:
    """Public capability payload for Studio attention options."""

    return {
        "available": list(STUDIO_ATTENTION_BACKENDS),
        "default": STUDIO_ATTENTION_BACKEND_AUTO,
        "fallback_order": list(STUDIO_ATTENTION_AUTO_FALLBACK_ORDER),
        "policy": "flash -> sdpa -> flex -> xformers -> default",
        "never_fail_on_unavailable_backend": True,
    }
