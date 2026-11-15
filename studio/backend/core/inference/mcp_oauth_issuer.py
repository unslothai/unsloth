# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

from urllib.parse import urlsplit


def same_issuer(a: str, b: str) -> bool:
    """Exact match (RFC 8414 3.3), except a root issuer's trailing slash, which discovery drops."""
    if a == b:
        return True
    shorter, longer = sorted((a, b), key = len)
    if longer != f"{shorter}/":
        return False
    try:
        parts = urlsplit(shorter)
    except ValueError:
        return False
    return bool(parts.scheme and parts.netloc) and shorter == f"{parts.scheme}://{parts.netloc}"
