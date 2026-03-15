# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Authentication API routes
"""

from fastapi import APIRouter, Depends, HTTPException, status

from models.auth import (
    AuthLoginRequest,
    RefreshTokenRequest,
    AuthStatusResponse,
    ChangePasswordRequest,
)
from models.users import Token
from auth import storage, hashing
from auth.authentication import (
    create_access_token,
    create_refresh_token,
    get_current_subject,
    get_current_subject_allow_password_change,
    refresh_access_token,
)

router = APIRouter()


@router.get("/status", response_model = AuthStatusResponse)
async def auth_status() -> AuthStatusResponse:
    """
    Check whether auth has already been initialized.

    - initialized = False -> frontend should wait for the seeded admin bootstrap.
    - initialized = True  -> frontend should show login or force the first password change.
    """
    return AuthStatusResponse(
        initialized = storage.is_initialized(),
        default_username = storage.DEFAULT_ADMIN_USERNAME,
        requires_password_change = storage.requires_password_change(
            storage.DEFAULT_ADMIN_USERNAME
        )
        if storage.is_initialized()
        else True,
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

    salt, pwd_hash, _jwt_secret, must_change_password = record
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
        must_change_password = must_change_password,
    )


@router.post("/refresh", response_model = Token)
async def refresh(payload: RefreshTokenRequest) -> Token:
    """
    Exchange a valid refresh token for a new access token.

    The refresh token itself is reusable until it expires (7 days).
    """
    new_access_token, username = refresh_access_token(payload.refresh_token)
    if new_access_token is None or username is None:
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail = "Invalid or expired refresh token",
        )

    return Token(
        access_token = new_access_token,
        refresh_token = payload.refresh_token,
        token_type = "bearer",
        must_change_password = storage.requires_password_change(username),
    )


@router.post("/change-password", response_model = Token)
async def change_password(
    payload: ChangePasswordRequest,
    current_subject: str = Depends(get_current_subject_allow_password_change),
) -> Token:
    """Allow the authenticated user to replace the default password."""
    record = storage.get_user_and_secret(current_subject)
    if record is None:
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail = "User session is invalid",
        )

    salt, pwd_hash, _jwt_secret, _must_change_password = record
    if not hashing.verify_password(payload.current_password, salt, pwd_hash):
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail = "Current password is incorrect",
        )
    if payload.current_password == payload.new_password:
        raise HTTPException(
            status_code = status.HTTP_400_BAD_REQUEST,
            detail = "New password must be different from the current password",
        )

    storage.update_password(current_subject, payload.new_password)
    storage.revoke_user_refresh_tokens(current_subject)
    access_token = create_access_token(subject = current_subject)
    refresh_token = create_refresh_token(subject = current_subject)
    return Token(
        access_token = access_token,
        refresh_token = refresh_token,
        token_type = "bearer",
        must_change_password = False,
    )


