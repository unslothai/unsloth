# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""HMAC capability tokens for public ``/p`` preview share links; owner tokens keep the original shape, and a managed token carries its account id ahead of a signature covering it, so identically named runs in two accounts never open each other's."""

from __future__ import annotations

import base64
import hashlib
import hmac
from typing import Optional

from auth.storage import get_or_create_preview_link_secret
from utils.account_context import OWNER, AccountContext, current_account

# Versioned so the token format can evolve without silently honoring old shapes.
_PREVIEW_TOKEN_VERSION = "v1"
_ACCOUNT_TOKEN_VERSION = "v2"
_ACCOUNT_SEPARATOR = "."


def _canonical_payload(ref: str, account_id: Optional[str] = None) -> bytes:
    # Sign the canonical ref only, never host/path, so links stay portable across localhost / LAN IP
    # / tunnel host changes.
    if account_id is None:
        return f"preview:{_PREVIEW_TOKEN_VERSION}:{ref}".encode("utf-8")
    return f"preview:{_ACCOUNT_TOKEN_VERSION}:{account_id}:{ref}".encode("utf-8")


def _mac(ref: str, account_id: Optional[str]) -> str:
    digest = hmac.new(
        get_or_create_preview_link_secret(),
        _canonical_payload(ref, account_id),
        hashlib.sha256,
    ).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def sign_preview_ref(ref: str, account: Optional[AccountContext] = None) -> str:
    account = account or current_account()
    if account.is_owner:
        return _mac(ref, None)
    return f"{account.account_id}{_ACCOUNT_SEPARATOR}{_mac(ref, account.account_id)}"


def preview_token_account(ref: str, token: Optional[str]) -> Optional[AccountContext]:
    """The account whose outputs ``token`` opens for ``ref``, or None; constant-time on the signature."""
    if not token:
        return None
    # Compare as bytes: a non-ASCII token (e.g. a %-encoded query value) would make
    # hmac.compare_digest on two str raise TypeError -> treat it as simply invalid.
    try:
        provided = token.encode("ascii")
    except UnicodeEncodeError:
        return None
    account_id, separator, signature = token.partition(_ACCOUNT_SEPARATOR)
    if not separator:
        if hmac.compare_digest(_mac(ref, None).encode("ascii"), provided):
            return OWNER
        return None
    if not account_id or not signature or _ACCOUNT_SEPARATOR in signature:
        return None
    if not hmac.compare_digest(_mac(ref, account_id).encode("ascii"), signature.encode("ascii")):
        return None
    from auth.storage import get_account_by_id

    account = get_account_by_id(account_id)
    if account is None or account.is_owner:
        return None
    return account


def verify_preview_ref(ref: str, token: Optional[str]) -> bool:
    return preview_token_account(ref, token) is not None
