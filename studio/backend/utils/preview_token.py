# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""HMAC capability tokens for public ``/p`` preview share links.

The preview ref (``run`` or ``run/checkpoint``) is a deterministic, guessable
outputs-root path, so it can't gate access on its own. We sign the canonical ref
with a dedicated server-side secret and require the resulting token on every
public preview request: guessing a ref no longer grants access, and rotating the
secret (``auth.storage.rotate_preview_link_secret``) revokes every link at once.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
from typing import Optional

from auth.storage import get_or_create_preview_link_secret

# Versioned so the token format can evolve without silently honoring old shapes.
_PREVIEW_TOKEN_VERSION = "v1"


def _canonical_payload(ref: str) -> bytes:
    # Sign the canonical ref only (never host/path) so links stay portable across
    # localhost / LAN IP / tunnel host changes.
    return f"preview:{_PREVIEW_TOKEN_VERSION}:{ref}".encode("utf-8")


def sign_preview_ref(ref: str) -> str:
    """Return the URL-safe HMAC capability token for a canonical preview ref."""
    mac = hmac.new(
        get_or_create_preview_link_secret(),
        _canonical_payload(ref),
        hashlib.sha256,
    ).digest()
    return base64.urlsafe_b64encode(mac).rstrip(b"=").decode("ascii")


def verify_preview_ref(ref: str, token: Optional[str]) -> bool:
    """Constant-time check that ``token`` is a valid capability for ``ref``."""
    if not token:
        return False
    # Compare as bytes: a non-ASCII token (e.g. a %-encoded query value) would make
    # hmac.compare_digest on two str raise TypeError -> treat it as simply invalid.
    try:
        provided = token.encode("ascii")
    except UnicodeEncodeError:
        return False
    return hmac.compare_digest(sign_preview_ref(ref).encode("ascii"), provided)
