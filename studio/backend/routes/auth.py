# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Authentication API routes
"""

from fastapi import APIRouter, Depends, HTTPException, status

from datetime import datetime, timedelta, timezone

from models.auth import (
    ApiKeyListResponse,
    ApiKeyResponse,
    AuthLoginRequest,
    AuthStatusResponse,
    ChangePasswordRequest,
    CreateApiKeyRequest,
    CreateApiKeyResponse,
    DesktopLoginRequest,
    RefreshTokenRequest,
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
            detail = "Incorrect password. Run 'unsloth studio reset-password' in your terminal to reset it.",
        )

    salt, pwd_hash, _jwt_secret, must_change_password = record
    if not hashing.verify_password(payload.password, salt, pwd_hash):
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail = "Incorrect password. Run 'unsloth studio reset-password' in your terminal to reset it.",
        )

    access_token = create_access_token(subject = payload.username)
    refresh_token = create_refresh_token(subject = payload.username)
    return Token(
        access_token = access_token,
        refresh_token = refresh_token,
        token_type = "bearer",
        must_change_password = must_change_password,
    )


@router.post("/desktop-login", response_model = Token)
async def desktop_login(payload: DesktopLoginRequest) -> Token:
    """Exchange a local desktop secret for normal admin-subject tokens."""
    username = storage.validate_desktop_secret(payload.secret)
    if username is None:
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail = "Desktop authentication failed",
        )

    return Token(
        access_token = create_access_token(subject = username, desktop = True),
        refresh_token = create_refresh_token(subject = username, desktop = True),
        token_type = "bearer",
        must_change_password = False,
    )


@router.post("/refresh", response_model = Token)
async def refresh(payload: RefreshTokenRequest) -> Token:
    """
    Exchange a valid refresh token for a new access token.

    The refresh token itself is reusable until it expires (7 days).
    """
    new_access_token, username, is_desktop = refresh_access_token(payload.refresh_token)
    if new_access_token is None or username is None:
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail = "Invalid or expired refresh token",
        )

    return Token(
        access_token = new_access_token,
        refresh_token = payload.refresh_token,
        token_type = "bearer",
        must_change_password = False
        if is_desktop
        else storage.requires_password_change(username),
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


# ---------------------------------------------------------------------------
# API key management
# ---------------------------------------------------------------------------


def _row_to_api_key_response(row: dict) -> ApiKeyResponse:
    return ApiKeyResponse(
        id = row["id"],
        name = row["name"],
        key_prefix = row["key_prefix"],
        created_at = row["created_at"],
        last_used_at = row.get("last_used_at"),
        expires_at = row.get("expires_at"),
        is_active = bool(row["is_active"]),
    )


@router.post("/api-keys", response_model = CreateApiKeyResponse)
async def create_api_key(
    payload: CreateApiKeyRequest,
    current_subject: str = Depends(get_current_subject),
) -> CreateApiKeyResponse:
    """Create a new API key. The raw key is returned once and cannot be retrieved later."""
    expires_at = None
    if payload.expires_in_days is not None:
        expires_at = (
            datetime.now(timezone.utc) + timedelta(days = payload.expires_in_days)
        ).isoformat()

    raw_key, row = storage.create_api_key(
        username = current_subject,
        name = payload.name,
        expires_at = expires_at,
    )
    return CreateApiKeyResponse(
        key = raw_key,
        api_key = _row_to_api_key_response(row),
    )


@router.get("/api-keys", response_model = ApiKeyListResponse)
async def list_api_keys(
    current_subject: str = Depends(get_current_subject),
) -> ApiKeyListResponse:
    """List all API keys for the authenticated user (raw keys are never exposed)."""
    rows = storage.list_api_keys(current_subject)
    return ApiKeyListResponse(
        api_keys = [_row_to_api_key_response(r) for r in rows],
    )


@router.delete("/api-keys/{key_id}")
async def revoke_api_key(
    key_id: int,
    current_subject: str = Depends(get_current_subject),
) -> dict:
    """Revoke (soft-delete) an API key."""
    if not storage.revoke_api_key(current_subject, key_id):
        raise HTTPException(
            status_code = status.HTTP_404_NOT_FOUND,
            detail = "API key not found",
        )
    return {"detail": "API key revoked"}
