# SPDX-License-Identifier: AGPL-3.0-only - See /studio/LICENSE.AGPL-3.0
# Copyright © 2025 Unsloth AI

"""
Authentication API routes
"""

from fastapi import APIRouter, HTTPException, status
import secrets

from models.auth import (
    AuthSetupRequest,
    AuthLoginRequest,
    RefreshTokenRequest,
    AuthStatusResponse,
)
from models.users import Token
from auth import storage, hashing
from auth.authentication import (
    create_access_token,
    create_refresh_token,
    refresh_access_token,
    reload_secret,
)

router = APIRouter()


@router.get("/status", response_model = AuthStatusResponse)
async def auth_status() -> AuthStatusResponse:
    """
    Check whether auth has already been initialized.

    - initialized = False -> frontend should show "Set admin password" screen.
    - initialized = True  -> frontend should show normal login.
    """
    return AuthStatusResponse(initialized = storage.is_initialized())


@router.post("/setup", response_model = Token, status_code = status.HTTP_201_CREATED)
async def setup_auth(payload: AuthSetupRequest) -> Token:
    """
    First-time setup: create the admin user and a JWT secret.

    Requires a valid setup token (printed to the server console on startup).
    Can only be called once. Subsequent calls will return 400.
    """
    if storage.is_initialized():
        raise HTTPException(
            status_code = status.HTTP_400_BAD_REQUEST,
            detail = "Auth is already initialized.",
        )

    # Validate the one-time setup token
    if not storage.consume_setup_token(payload.setup_token):
        raise HTTPException(
            status_code = status.HTTP_403_FORBIDDEN,
            detail = "Invalid or expired setup token.",
        )

    # Generate a strong random JWT secret for this installation
    jwt_secret = secrets.token_urlsafe(64)

    # Create user + generate tokens atomically — rollback if anything fails
    try:
        storage.create_initial_user(
            username = payload.username,
            password = payload.password,
            jwt_secret = jwt_secret,
        )

        # Reload JWT secret from DB (so authentication.py picks it up)
        reload_secret()

        # Issue access + refresh tokens for the new user
        access_token = create_access_token(subject = payload.username)
        refresh_token = create_refresh_token(subject = payload.username)

    except Exception as e:
        # Rollback: remove the user row so setup can be retried
        storage.delete_user(payload.username)
        raise HTTPException(
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail = f"Setup failed (rolled back): {str(e)}",
        )

    return Token(
        access_token = access_token,
        refresh_token = refresh_token,
        token_type = "bearer",
    )


@router.post("/login", response_model = Token)
async def login(payload: AuthLoginRequest) -> Token:
    """
    Login with username/password and receive access + refresh tokens.
    """
    record = storage.get_user_and_secret(payload.username)
    if record is None:
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail = "Incorrect username or password",
        )

    salt, pwd_hash, _jwt_secret = record
    if not hashing.verify_password(payload.password, salt, pwd_hash):
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail = "Incorrect username or password",
        )

    access_token = create_access_token(subject = payload.username)
    refresh_token = create_refresh_token(subject = payload.username)
    return Token(
        access_token = access_token,
        refresh_token = refresh_token,
        token_type = "bearer",
    )


@router.post("/refresh", response_model = Token)
async def refresh(payload: RefreshTokenRequest) -> Token:
    """
    Exchange a valid refresh token for a new access token.

    The refresh token itself is reusable until it expires (7 days).
    """
    new_access_token = refresh_access_token(payload.refresh_token)
    if new_access_token is None:
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail = "Invalid or expired refresh token",
        )

    return Token(
        access_token = new_access_token,
        refresh_token = payload.refresh_token,
        token_type = "bearer",
    )
