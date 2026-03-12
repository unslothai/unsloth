# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
import jwt

from .storage import load_jwt_secret, save_refresh_token, verify_refresh_token

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60
REFRESH_TOKEN_EXPIRE_DAYS = 7

# Load stable secret from SQLite (set during first-time setup)
# This will raise RuntimeError if auth hasn't been initialized yet
try:
    SECRET_KEY = load_jwt_secret()
except RuntimeError:
    # Fallback: use a temporary secret until setup is complete
    # This allows the app to start, but protected routes will fail until setup
    SECRET_KEY = secrets.token_urlsafe(64)

security = HTTPBearer()  # Reads Authorization: Bearer <token>


def create_access_token(
    subject: str,
    expires_delta: Optional[timedelta] = None,
) -> str:
    """
    Create a signed JWT for the given subject (e.g. username).

    Tokens are valid across restarts because SECRET_KEY is stored in SQLite.
    """
    to_encode = {"sub": subject}
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes = ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm = ALGORITHM)


def create_refresh_token(subject: str) -> str:
    """
    Create a random refresh token, store its hash in SQLite, and return it.

    Refresh tokens are opaque (not JWTs) and expire after REFRESH_TOKEN_EXPIRE_DAYS.
    """
    token = secrets.token_urlsafe(48)
    expires_at = datetime.now(timezone.utc) + timedelta(days = REFRESH_TOKEN_EXPIRE_DAYS)
    save_refresh_token(token, subject, expires_at.isoformat())
    return token


def refresh_access_token(refresh_token: str) -> Optional[str]:
    """
    Validate a refresh token and issue a new access token.

    The refresh token itself is NOT consumed — it stays valid until expiry.
    Returns a new access_token or None if the refresh token is invalid/expired.
    """
    username = verify_refresh_token(refresh_token)
    if username is None:
        return None
    return create_access_token(subject = username)


def reload_secret() -> None:
    """
    Reload the JWT secret from SQLite.

    Call this after setup to ensure new tokens use the persistent secret.
    """
    global SECRET_KEY
    SECRET_KEY = load_jwt_secret()


async def get_current_subject(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> str:
    """
    FastAPI dependency to validate the JWT and return the subject.

    Use this as a dependency on routes that should be protected, e.g.:

        @router.get("/secure")
        async def secure_endpoint(current_subject: str = Depends(get_current_subject)):
            ...
    """
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms = [ALGORITHM])
        subject: Optional[str] = payload.get("sub")
        if subject is None:
            raise HTTPException(
                status_code = status.HTTP_401_UNAUTHORIZED,
                detail = "Invalid token payload",
            )
        return subject
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail = "Invalid or expired token",
        )
