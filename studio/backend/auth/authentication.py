# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
import jwt

from .storage import (
    get_jwt_secret,
    get_user_and_secret,
    load_jwt_secret,
    save_refresh_token,
    verify_refresh_token,
)

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60
REFRESH_TOKEN_EXPIRE_DAYS = 7

security = HTTPBearer()  # Reads Authorization: Bearer <token>


def _get_secret_for_subject(subject: str) -> str:
    secret = get_jwt_secret(subject)
    if secret is None:
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail = "Invalid or expired token",
        )
    return secret


def _decode_subject_without_verification(token: str) -> Optional[str]:
    try:
        payload = jwt.decode(
            token,
            options = {"verify_signature": False, "verify_exp": False},
        )
    except jwt.InvalidTokenError:
        return None

    subject = payload.get("sub")
    return subject if isinstance(subject, str) else None


def create_access_token(
    subject: str,
    expires_delta: Optional[timedelta] = None,
) -> str:
    """
    Create a signed JWT for the given subject (e.g. username).

    Tokens are valid across restarts because the signing secret is stored in SQLite.
    """
    to_encode = {"sub": subject}
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes = ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire})
    return jwt.encode(
        to_encode,
        _get_secret_for_subject(subject),
        algorithm = ALGORITHM,
    )


def create_refresh_token(subject: str) -> str:
    """
    Create a random refresh token, store its hash in SQLite, and return it.

    Refresh tokens are opaque (not JWTs) and expire after REFRESH_TOKEN_EXPIRE_DAYS.
    """
    token = secrets.token_urlsafe(48)
    expires_at = datetime.now(timezone.utc) + timedelta(days = REFRESH_TOKEN_EXPIRE_DAYS)
    save_refresh_token(token, subject, expires_at.isoformat())
    return token


def refresh_access_token(refresh_token: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Validate a refresh token and issue a new access token.

    The refresh token itself is NOT consumed — it stays valid until expiry.
    Returns a new access_token or None if the refresh token is invalid/expired.
    """
    username = verify_refresh_token(refresh_token)
    if username is None:
        return None, None
    return create_access_token(subject = username), username


def reload_secret() -> None:
    """
    Keep legacy API compatibility for callers expecting auth storage init.

    Auth now resolves the current signing secret directly from SQLite.
    """
    load_jwt_secret()


async def get_current_subject(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> str:
    """Validate JWT and require the password-change flow to be completed."""
    return await _get_current_subject(
        credentials,
        allow_password_change = False,
    )


async def get_current_subject_allow_password_change(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> str:
    """Validate JWT but allow access to the password-change endpoint."""
    return await _get_current_subject(
        credentials,
        allow_password_change = True,
    )


async def _get_current_subject(
    credentials: HTTPAuthorizationCredentials,
    *,
    allow_password_change: bool,
) -> str:
    """
    FastAPI dependency to validate the JWT and return the subject.

    Use this as a dependency on routes that should be protected, e.g.:

        @router.get("/secure")
        async def secure_endpoint(current_subject: str = Depends(get_current_subject)):
            ...
    """
    token = credentials.credentials
    subject = _decode_subject_without_verification(token)
    if subject is None:
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail = "Invalid token payload",
        )

    record = get_user_and_secret(subject)
    if record is None:
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail = "Invalid or expired token",
        )

    _salt, _pwd_hash, jwt_secret, must_change_password = record
    try:
        payload = jwt.decode(token, jwt_secret, algorithms = [ALGORITHM])
        if payload.get("sub") != subject:
            raise HTTPException(
                status_code = status.HTTP_401_UNAUTHORIZED,
                detail = "Invalid token payload",
            )
        if must_change_password and not allow_password_change:
            raise HTTPException(
                status_code = status.HTTP_403_FORBIDDEN,
                detail = "Password change required",
            )
        return subject
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail = "Invalid or expired token",
        )


# ── Dual-auth for OpenAI-compatible endpoints ───────────────────

_optional_security = HTTPBearer(auto_error = False)


async def get_current_subject_or_api_key(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_optional_security),
) -> str:
    """Accept a valid JWT or the auto-generated API key for OpenAI-compatible endpoints."""
    if credentials is None:
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail = "Missing Authorization header",
        )

    token = credentials.credentials

    # Check auto-generated API key (fast path for external consumers)
    external_key = getattr(request.app.state, "external_api_key", None)
    if external_key and token == external_key:
        return "__api_user__"

    # Fall back to JWT validation (Studio frontend sessions)
    return await _get_current_subject(credentials, allow_password_change = False)


async def get_current_subject_or_api_key_anthropic(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_optional_security),
) -> str:
    """Accept x-api-key header, Authorization: Bearer, or JWT.

    The Anthropic SDK sends ``x-api-key: <key>`` instead of
    ``Authorization: Bearer <key>``. This dependency checks both so that
    the ``/v1/messages`` endpoint works with both Anthropic and OpenAI SDKs.
    """
    external_key = getattr(request.app.state, "external_api_key", None)

    # Check x-api-key header first (Anthropic SDK default)
    x_api_key = request.headers.get("x-api-key")
    if x_api_key and external_key and x_api_key == external_key:
        return "__api_user__"

    # Fall through to standard Bearer / JWT check
    if credentials is None:
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail = "Missing Authorization or x-api-key header",
        )

    token = credentials.credentials
    if external_key and token == external_key:
        return "__api_user__"

    return await _get_current_subject(credentials, allow_password_change = False)
