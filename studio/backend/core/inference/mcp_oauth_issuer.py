# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Which OAuth issuer spellings name the same authorization server."""

from __future__ import annotations

from urllib.parse import urlsplit


def same_issuer(a: str, b: str) -> bool:
    """Issuers compare as plain strings (RFC 8414 3.3), except a root issuer with and without its
    trailing slash: discovery drops that slash before building the metadata URL (RFC 8414 3), so
    both spellings fetch the same document. An issuer with a path stays exact."""
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
