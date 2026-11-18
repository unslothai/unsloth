# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Who may send a large request body. FastAPI reads a body before the route's auth dependency runs,
so without this an unauthenticated client could make the server hold a body up to the full upload
cap per connection. Only bodies over ``PREAUTH_BODY_MAX_BYTES`` ask: every unauthenticated route
(login, refresh, desktop shell) takes a few hundred bytes."""

from __future__ import annotations

import re
from typing import Any, Optional
from urllib.parse import parse_qs

PREAUTH_BODY_MAX_BYTES = 1024 * 1024

# Public preview links carry their own capability (`?k=` or a bearer) instead of a session.
_PREVIEW_CHAT_RE = re.compile(r"^/p/([^/]+)(?:/([^/]+))?/v1/chat/completions/?$")


def _bearer(scope: dict) -> Optional[str]:
    values = [
        bytes(value).decode("latin-1")
        for name, value in scope.get("headers") or ()
        if bytes(name).lower() == b"authorization"
    ]
    if len(values) != 1:
        return None
    scheme, _, token = values[0].strip().partition(" ")
    token = token.strip()
    return token if scheme.lower() == "bearer" and token else None


def _query(scope: dict, name: str) -> Optional[str]:
    try:
        raw = (scope.get("query_string") or b"").decode("latin-1")
        values = parse_qs(raw).get(name)
    except Exception:
        return None
    return values[0] if values else None


def _session_jwt_valid(token: str) -> bool:
    import jwt

    from auth.authentication import ALGORITHM, _decode_subject_without_verification
    from auth.storage import get_user_and_secret

    subject = _decode_subject_without_verification(token)
    if subject is None:
        return False
    record = get_user_and_secret(subject)
    if record is None:
        return False
    try:
        payload = jwt.decode(token, record[2], algorithms = [ALGORITHM])
    except jwt.InvalidTokenError:
        return False
    return payload.get("sub") == subject


def _credential_valid(token: str) -> bool:
    from auth.authentication import bearer_is_valid_api_key
    from auth.storage import API_KEY_PREFIX

    if token.startswith(API_KEY_PREFIX):
        return bearer_is_valid_api_key(token)
    return _session_jwt_valid(token)


def caller_may_send_large_body(scope: dict, path: str) -> bool:
    """True when the request carries a credential that would pass the route's auth: a valid session
    or API key (header or ``?token=``), a preview capability on its preview route, or a transport the
    keyless API access setting admits. Blocking (SQLite), so run it off the event loop."""
    bearer = _bearer(scope)
    for token in (bearer, _query(scope, "token")):
        if token and _credential_valid(token):
            return True
    preview = _PREVIEW_CHAT_RE.match(path)
    if preview is not None:
        from utils.preview_token import preview_token_account
        ref = preview.group(1) if not preview.group(2) else f"{preview.group(1)}/{preview.group(2)}"
        if preview_token_account(ref, _query(scope, "k") or bearer) is not None:
            return True
    from starlette.requests import Request

    from utils.keyless_api_access import keyless_request_allowed

    return bool(keyless_request_allowed(Request(scope)))


async def caller_may_send_large_body_async(scope: dict, path: str) -> bool:
    from starlette.concurrency import run_in_threadpool
    try:
        return await run_in_threadpool(caller_may_send_large_body, scope, path)
    except Exception:
        # An unexpected failure must not turn a signed-in user's upload away.
        return True
