# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

from fastapi import Depends, HTTPException, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
import jwt

from .storage import (
    API_KEY_PREFIX,
    credential_generation,
    get_jwt_secret,
    get_user_and_secret,
    load_jwt_secret,
    save_refresh_token,
    validate_api_key_with_credential,
    verify_refresh_token,
)

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60
REFRESH_TOKEN_EXPIRE_DAYS = 7
X_API_KEY_HEADER = "x-api-key"
NOT_AUTHENTICATED_DETAIL = "Not authenticated"

security = HTTPBearer(auto_error = False)  # Reads Authorization: Bearer <token>
x_api_key_security = APIKeyHeader(name = X_API_KEY_HEADER, auto_error = False)


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
    *,
    desktop: bool = False,
    secret: Optional[str] = None,
) -> str:
    """
    Create a signed JWT for the given subject (e.g. username).

    Valid across restarts: the signing secret is stored in SQLite. Callers that
    already verified a credential pass ``secret`` so a rotation landing mid-request
    cannot sign the token with the credential that just replaced it.
    """
    to_encode = {"sub": subject}
    if desktop:
        to_encode["desktop"] = True
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes = ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire})
    return jwt.encode(
        to_encode,
        secret if secret is not None else _get_secret_for_subject(subject),
        algorithm = ALGORITHM,
    )


def is_desktop_access_token(token: str) -> bool:
    """Return true only for a valid desktop-issued JWT access token."""
    if token.startswith(API_KEY_PREFIX):
        return False

    subject = _decode_subject_without_verification(token)
    if subject is None:
        return False

    record = get_user_and_secret(subject)
    if record is None:
        return False

    _salt, _pwd_hash, jwt_secret, _must_change_password = record
    try:
        payload = jwt.decode(token, jwt_secret, algorithms = [ALGORITHM])
    except jwt.InvalidTokenError:
        return False

    return payload.get("sub") == subject and payload.get("desktop") is True


def create_refresh_token(
    subject: str,
    *,
    desktop: bool = False,
    secret: Optional[str] = None,
) -> str:
    """
    Create a random refresh token, store its hash in SQLite, and return it.

    Refresh tokens are opaque (not JWTs); expire after REFRESH_TOKEN_EXPIRE_DAYS.
    ``secret`` stamps the token with the credential version the caller verified,
    so a rotation cannot leave a token minted from the replaced credential valid.
    """
    token = secrets.token_urlsafe(48)
    expires_at = datetime.now(timezone.utc) + timedelta(days = REFRESH_TOKEN_EXPIRE_DAYS)
    save_refresh_token(
        token,
        subject,
        expires_at.isoformat(),
        is_desktop = desktop,
        secret_gen = credential_generation(secret) if secret is not None else None,
    )
    return token


def refresh_access_token(refresh_token: str) -> Tuple[Optional[str], Optional[str], bool]:
    """
    Validate a refresh token and issue a new access token.

    The refresh token is NOT consumed; it stays valid until expiry.
    Returns a new access_token, or None if the refresh token is invalid/expired.
    """
    verified = verify_refresh_token(refresh_token)
    if verified is None:
        return None, None, False
    username, is_desktop = verified
    return (
        create_access_token(subject = username, desktop = is_desktop),
        username,
        is_desktop,
    )


def reload_secret() -> None:
    """
    Legacy API compat for callers expecting auth storage init.

    Auth now resolves the current signing secret directly from SQLite.
    """
    load_jwt_secret()


def _resolve_credentials(
    credentials: Optional[HTTPAuthorizationCredentials],
    x_api_key: Optional[str],
) -> HTTPAuthorizationCredentials:
    if credentials is not None:
        return credentials
    if x_api_key is not None:
        if not x_api_key.startswith(API_KEY_PREFIX):
            raise HTTPException(
                status_code = status.HTTP_401_UNAUTHORIZED,
                detail = _invalid_api_key_detail(x_api_key),
            )
        return HTTPAuthorizationCredentials(
            scheme = X_API_KEY_HEADER,
            credentials = x_api_key,
        )
    raise HTTPException(
        status_code = status.HTTP_401_UNAUTHORIZED,
        detail = NOT_AUTHENTICATED_DETAIL,
    )


async def get_current_subject(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    x_api_key: Optional[str] = Depends(x_api_key_security),
) -> str:
    """Validate JWT and require the password-change flow to be completed."""
    subject, _generation = await _get_current_credential(
        _resolve_credentials(credentials, x_api_key),
        allow_password_change = False,
    )
    return subject


async def get_current_credential(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    x_api_key: Optional[str] = Depends(x_api_key_security),
) -> Tuple[str, Optional[str]]:
    """As get_current_subject, but also returns the credential generation.

    For routes that persist a new credential and must not do so on behalf of one
    a concurrent reset has revoked.
    """
    return await _get_current_credential(
        _resolve_credentials(credentials, x_api_key),
        allow_password_change = False,
    )


async def authenticated_via_api_key(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    x_api_key: Optional[str] = Depends(x_api_key_security),
) -> bool:
    """True when the caller used an sk-unsloth API key, not a UI session JWT.

    Lets routes treat programmatic API callers differently from the Unsloth UI
    (e.g. refuse a teardown the UI would allow).
    """
    resolved = _resolve_credentials(credentials, x_api_key)
    return resolved.credentials.startswith(API_KEY_PREFIX)


async def get_current_subject_allow_password_change(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    x_api_key: Optional[str] = Depends(x_api_key_security),
) -> str:
    """Validate JWT but allow access to the password-change endpoint."""
    subject, _generation = await _get_current_credential(
        _resolve_credentials(credentials, x_api_key),
        allow_password_change = True,
    )
    return subject


# The literal the examples ship with; pasted unedited more often than a revoked key.
API_KEY_PLACEHOLDER = f"{API_KEY_PREFIX}YOUR_KEY"


def _invalid_api_key_detail(token: str) -> str:
    """Why the key failed. Only the example placeholder is called out; every real
    key gets one indistinguishable message, so this leaks no key existence."""
    if token == API_KEY_PLACEHOLDER:
        return (
            "This is the placeholder key from the example. Create an API key in "
            f"Unsloth Studio under Settings > API and use it in place of {API_KEY_PLACEHOLDER}."
        )
    return "Invalid or expired API key"


async def _get_current_credential(
    credentials: HTTPAuthorizationCredentials, *, allow_password_change: bool
) -> Tuple[str, Optional[str]]:
    """Validate the bearer and return ``(subject, credential generation)``.

    The generation is the credential version this request actually authenticated
    against. Routes that persist new credentials must bind their write to it, or
    a reset landing mid-request would bless what it just revoked.
    """
    token = credentials.credentials

    # --- API key path (sk-unsloth-...) ---
    if token.startswith(API_KEY_PREFIX):
        verified = validate_api_key_with_credential(token)
        if verified is None:
            raise HTTPException(
                status_code = status.HTTP_401_UNAUTHORIZED,
                detail = _invalid_api_key_detail(token),
            )
        username, secret = verified
        return username, credential_generation(secret)

    # --- JWT path ---
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
        is_desktop = payload.get("desktop") is True
        if must_change_password and not allow_password_change and not is_desktop:
            raise HTTPException(
                status_code = status.HTTP_403_FORBIDDEN,
                detail = "Password change required",
            )
        return subject, credential_generation(jwt_secret)
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail = "Invalid or expired token",
        )
