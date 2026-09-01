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
from utils.workspace_context import LEGACY_WORKSPACE_SUBJECT, current_workspace_subject

# Versioned so the token format can evolve without silently honoring old shapes.
_PREVIEW_TOKEN_VERSION = "v1"


def _encode_subject(subject: str) -> str:
    return base64.urlsafe_b64encode(subject.encode("utf-8")).decode("ascii").rstrip("=")


def _canonical_payload(ref: str, subject: str) -> bytes:
    # Sign the canonical ref and the workspace it belongs to (never host/path) so
    # links stay portable across localhost / LAN IP / tunnel host changes.
    return f"preview:{_PREVIEW_TOKEN_VERSION}:{subject}:{ref}".encode("utf-8")


def _mac(ref: str, subject: str) -> str:
    digest = hmac.new(
        get_or_create_preview_link_secret(),
        _canonical_payload(ref, subject),
        hashlib.sha256,
    ).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def sign_preview_ref(ref: str, subject: Optional[str] = None) -> str:
    """The URL-safe capability token for a canonical preview ref.

    Carries the workspace: the outputs root is per account now, so a token that
    named the ref alone resolved against the owner's tree, which 404s every
    managed link and would serve the owner's checkpoint on a ref collision.
    """
    who = subject or current_workspace_subject()
    return f"{_encode_subject(who)}.{_mac(ref, who)}"


def preview_token_subject(token: Optional[str]) -> str:
    """The workspace a token names. Tokens minted before this read as the owner."""
    if not token or "." not in token:
        return LEGACY_WORKSPACE_SUBJECT
    encoded = token.split(".", 1)[0]
    padded = encoded + "=" * (-len(encoded) % 4)
    try:
        return base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8")
    except Exception:
        return LEGACY_WORKSPACE_SUBJECT


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
    if "." in token:
        subject = preview_token_subject(token)
        expected = f"{_encode_subject(subject)}.{_mac(ref, subject)}"
    else:
        # A link minted before the workspace was signed in. Only the owner could
        # have made one, so it verifies against the old ref-only payload.
        expected = base64.urlsafe_b64encode(
            hmac.new(
                get_or_create_preview_link_secret(),
                f"preview:{_PREVIEW_TOKEN_VERSION}:{ref}".encode("utf-8"),
                hashlib.sha256,
            ).digest()
        ).rstrip(b"=").decode("ascii")
    return hmac.compare_digest(expected.encode("ascii"), provided)
