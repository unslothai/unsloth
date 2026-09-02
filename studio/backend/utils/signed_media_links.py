# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Short-lived HMAC links for files a plain client fetches without a bearer.

The gallery and RAG document links are downloaded by an image tag or pdf.js with
no Authorization header, so the route has no auth dependency and runs on the
ContextVar default. Once the galleries became per account that resolved the
owner's directory, and every managed account got a 404 for its own file.

The signed payload therefore names the workspace as well as the resource, and
the caller rebinds it before resolving anything. The subject is base64url so it
cannot contain the dot the payload splits on; usernames legally may.

Tokens minted before this carried no subject. They are still accepted and read
as the owner, which is who minted them on a single-user install.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import time
from typing import Optional, Tuple

from utils.workspace_context import LEGACY_WORKSPACE_SUBJECT, current_workspace_subject


def _encode_subject(subject: str) -> str:
    return base64.urlsafe_b64encode(subject.encode("utf-8")).decode("ascii").rstrip("=")


def _decode_subject(encoded: str) -> Optional[str]:
    padded = encoded + "=" * (-len(encoded) % 4)
    try:
        return base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8")
    except Exception:
        return None


def sign(secret: bytes, resource_id: str, ttl_seconds: int) -> str:
    """A token naming ``resource_id`` and the calling workspace, valid for the ttl."""
    exp = int(time.time()) + ttl_seconds
    payload = f"{resource_id}.{exp}.{_encode_subject(current_workspace_subject())}"
    sig = hmac.new(secret, payload.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{payload}.{sig}"


def verify(secret: bytes, token: str) -> Tuple[Optional[str], str]:
    """``(resource id, workspace)`` for a valid unexpired token, else ``(None, owner)``."""
    parts = token.rsplit(".", 3)
    if len(parts) == 4:
        resource_id, exp_s, encoded, sig = parts
        subject = _decode_subject(encoded)
        if subject is None:
            return None, LEGACY_WORKSPACE_SUBJECT
        payload = f"{resource_id}.{exp_s}.{encoded}"
    else:
        parts = token.rsplit(".", 2)
        if len(parts) != 3:
            return None, LEGACY_WORKSPACE_SUBJECT
        resource_id, exp_s, sig = parts
        subject = LEGACY_WORKSPACE_SUBJECT
        payload = f"{resource_id}.{exp_s}"
    expected = hmac.new(secret, payload.encode("utf-8"), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(sig, expected):
        return None, LEGACY_WORKSPACE_SUBJECT
    try:
        if int(exp_s) < int(time.time()):
            return None, LEGACY_WORKSPACE_SUBJECT
    except ValueError:
        return None, LEGACY_WORKSPACE_SUBJECT
    return resource_id, subject
