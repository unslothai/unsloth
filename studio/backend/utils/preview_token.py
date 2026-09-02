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


def _incarnation(subject: str, *, create: bool) -> str:
    """This account's preview identity, empty for the owner and for a deleted one.

    The owner is the installation and cannot be deleted and recreated, so its
    links keep the original payload and stay valid across this change.
    """
    if subject == LEGACY_WORKSPACE_SUBJECT:
        return ""
    from auth.storage import preview_link_incarnation

    return preview_link_incarnation(subject, create = create)


def _canonical_payload(ref: str, subject: str, incarnation: str) -> bytes:
    # Sign the canonical ref and the workspace it belongs to (never host/path) so
    # links stay portable across localhost / LAN IP / tunnel host changes. And the
    # account incarnation, because a username is reusable: without it every link a
    # deleted account shared kept verifying, and pointed at the replacement's
    # checkpoint as soon as one appeared at the same ref.
    if not incarnation:
        return f"preview:{_PREVIEW_TOKEN_VERSION}:{subject}:{ref}".encode("utf-8")
    return f"preview:{_PREVIEW_TOKEN_VERSION}:{subject}:{incarnation}:{ref}".encode("utf-8")


def _mac(ref: str, subject: str, incarnation: str) -> str:
    digest = hmac.new(
        get_or_create_preview_link_secret(),
        _canonical_payload(ref, subject, incarnation),
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
    return f"{_encode_subject(who)}.{_mac(ref, who, _incarnation(who, create = True))}"


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
        # Never minted here: an account that has been deleted has no incarnation,
        # and creating one would be inventing the identity the token is checked
        # against. Its links fail, which is the point.
        incarnation = _incarnation(subject, create = False)
        if subject != LEGACY_WORKSPACE_SUBJECT and not incarnation:
            return False
        expected = f"{_encode_subject(subject)}.{_mac(ref, subject, incarnation)}"
    else:
        # A link minted before the workspace was signed in. Only the owner could
        # have made one, so it verifies against the old ref-only payload.
        expected = (
            base64.urlsafe_b64encode(
                hmac.new(
                    get_or_create_preview_link_secret(),
                    f"preview:{_PREVIEW_TOKEN_VERSION}:{ref}".encode("utf-8"),
                    hashlib.sha256,
                ).digest()
            )
            .rstrip(b"=")
            .decode("ascii")
        )
    return hmac.compare_digest(expected.encode("ascii"), provided)
