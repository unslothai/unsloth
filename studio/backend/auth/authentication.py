# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import base64
import hashlib
import hmac
import json
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
import jwt

from .storage import (
    API_KEY_PREFIX,
    LINK_TOKEN_EXPIRE_SECONDS,
    consume_link_token,
    get_jwt_secret,
    get_user_and_secret,
    load_jwt_secret,
    save_link_token,
    save_refresh_token,
    validate_api_key,
    verify_refresh_token,
)

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60
REFRESH_TOKEN_EXPIRE_DAYS = 7

# Domain-separation label for the link-token signing key. The key is derived from
# the user's JWT secret (so a password change, which rotates that secret,
# invalidates outstanding link tokens) but is NOT the JWT secret itself: a link
# token must never be accepted as a bearer access token, so it is signed with a
# different key and can't validate on the access-token path.
_LINK_TOKEN_KEY_LABEL = b"unsloth-studio-link-token-v1"

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
    *,
    desktop: bool = False,
) -> str:
    """
    Create a signed JWT for the given subject (e.g. username).

    Valid across restarts: the signing secret is stored in SQLite.
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
        _get_secret_for_subject(subject),
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


def create_refresh_token(subject: str, *, desktop: bool = False) -> str:
    """
    Create a random refresh token, store its hash in SQLite, and return it.

    Refresh tokens are opaque (not JWTs); expire after REFRESH_TOKEN_EXPIRE_DAYS.
    """
    token = secrets.token_urlsafe(48)
    expires_at = datetime.now(timezone.utc) + timedelta(days = REFRESH_TOKEN_EXPIRE_DAYS)
    save_refresh_token(token, subject, expires_at.isoformat(), is_desktop = desktop)
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


# ---------------------------------------------------------------------------
# One-time link tokens (opt-in Colab same-tab handoff)
# ---------------------------------------------------------------------------


def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64url_decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + padding)


def _link_token_key(subject: str) -> Optional[bytes]:
    """Derive the link-token signing key for *subject* from their JWT secret.

    Returns None when the subject has no secret (unknown user), so a forged token
    naming a non-existent user is rejected before any comparison.
    """
    secret = get_jwt_secret(subject)
    if secret is None:
        return None
    return hmac.new(secret.encode("utf-8"), _LINK_TOKEN_KEY_LABEL, hashlib.sha256).digest()


def _decode_link_payload(payload_b64: str) -> Optional[dict]:
    try:
        return json.loads(_b64url_decode(payload_b64).decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None


def create_link_token(subject: str) -> str:
    """Mint a one-time, short-TTL HMAC-signed link token bound to *subject*.

    The token is ``<payload_b64>.<sig_b64>`` where the payload carries the
    subject, a random single-use id (jti), and an expiry, and the signature is
    HMAC-SHA256 over the payload under a key derived from the subject's JWT
    secret. The jti is recorded so the token can be exchanged exactly once.

    SECURITY: the returned value is a bearer credential. NEVER log it, and only
    ever place it on the private same-tab URL, never on a shared/public link.
    """
    key = _link_token_key(subject)
    if key is None:
        raise RuntimeError(f"Cannot mint a link token for unknown subject {subject!r}")
    jti = secrets.token_urlsafe(24)
    expires_at = datetime.now(timezone.utc) + timedelta(seconds = LINK_TOKEN_EXPIRE_SECONDS)
    expires_iso = expires_at.isoformat()
    save_link_token(jti, subject, expires_iso)
    payload = {"sub": subject, "jti": jti, "exp": expires_iso}
    payload_b64 = _b64url_encode(json.dumps(payload, separators = (",", ":")).encode("utf-8"))
    sig = hmac.new(key, payload_b64.encode("ascii"), hashlib.sha256).digest()
    return f"{payload_b64}.{_b64url_encode(sig)}"


def exchange_link_token(token: str) -> Optional[str]:
    """Validate and consume a one-time link token, returning its subject or None.

    Enforced in order: well-formed structure, a valid constant-time signature
    (bound to the named subject's derived key), matching subject claim, unexpired,
    and single-use consumption of the jti. Any failure returns None without a hint
    about which check failed.
    """
    if not isinstance(token, str) or token.count(".") != 1:
        return None
    payload_b64, sig_b64 = token.split(".", 1)
    if not payload_b64 or not sig_b64:
        return None

    # Read the claimed subject from the (still-unverified) payload only to select
    # the signing key; the signature check below is what actually authenticates it.
    claims = _decode_link_payload(payload_b64)
    if not isinstance(claims, dict):
        return None
    subject = claims.get("sub")
    if not isinstance(subject, str) or not subject:
        return None

    key = _link_token_key(subject)
    if key is None:
        return None
    expected_sig = hmac.new(key, payload_b64.encode("ascii"), hashlib.sha256).digest()
    try:
        provided_sig = _b64url_decode(sig_b64)
    except (ValueError, TypeError):
        return None
    if not hmac.compare_digest(expected_sig, provided_sig):
        return None

    jti = claims.get("jti")
    expires_iso = claims.get("exp")
    if not isinstance(jti, str) or not isinstance(expires_iso, str):
        return None
    # Expiry is defense-in-depth; consume_link_token also drops expired rows.
    try:
        if datetime.now(timezone.utc) > datetime.fromisoformat(expires_iso):
            return None
    except ValueError:
        return None

    if not consume_link_token(jti, subject):
        return None
    return subject


async def get_current_subject(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Validate JWT and require the password-change flow to be completed."""
    return await _get_current_subject(
        credentials,
        allow_password_change = False,
    )


async def authenticated_via_api_key(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> bool:
    """True when the caller used an sk-unsloth API key, not a UI session JWT.

    Lets routes treat programmatic API callers differently from the Unsloth UI
    (e.g. refuse a teardown the UI would allow).
    """
    return bool(credentials and credentials.credentials.startswith(API_KEY_PREFIX))


async def get_current_subject_allow_password_change(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> str:
    """Validate JWT but allow access to the password-change endpoint."""
    return await _get_current_subject(
        credentials,
        allow_password_change = True,
    )


async def _get_current_subject(
    credentials: HTTPAuthorizationCredentials, *, allow_password_change: bool
) -> str:
    """FastAPI dependency: validate the JWT and return the subject. Use on protected routes."""
    token = credentials.credentials

    # --- API key path (sk-unsloth-...) ---
    if token.startswith(API_KEY_PREFIX):
        username = validate_api_key(token)
        if username is None:
            raise HTTPException(
                status_code = status.HTTP_401_UNAUTHORIZED,
                detail = "Invalid or expired API key",
            )
        return username

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
        return subject
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail = "Invalid or expired token",
        )
