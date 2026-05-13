# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Authentication API routes
"""

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status

import threading
import time
from collections import deque
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


# In-memory per-IP login rate limiter; multi-process deployment needs a shared store.
_LOGIN_BUCKETS: dict[str, deque] = {}
_LOGIN_BUCKETS_LOCK = threading.Lock()
_LOGIN_WINDOW_SECONDS = 60.0
_LOGIN_MAX_FAILS = 5
_LOGIN_LOCKOUT_SECONDS = 60


def _client_key(request: Request | None) -> str:
    if request is None or request.client is None:
        return "_unknown"
    return request.client.host or "_unknown"


def _record_login_failure(ip: str) -> int:
    now = time.monotonic()
    with _LOGIN_BUCKETS_LOCK:
        bucket = _LOGIN_BUCKETS.setdefault(ip, deque())
        while bucket and now - bucket[0] > _LOGIN_WINDOW_SECONDS:
            bucket.popleft()
        bucket.append(now)
        return len(bucket)


def _login_blocked(ip: str) -> int:
    """Return seconds until the next attempt is allowed, or 0."""
    now = time.monotonic()
    with _LOGIN_BUCKETS_LOCK:
        bucket = _LOGIN_BUCKETS.get(ip)
        if not bucket:
            return 0
        while bucket and now - bucket[0] > _LOGIN_WINDOW_SECONDS:
            bucket.popleft()
        if len(bucket) >= _LOGIN_MAX_FAILS:
            return max(1, int(_LOGIN_WINDOW_SECONDS - (now - bucket[0])))
        return 0


def _clear_login_bucket(ip: str) -> None:
    with _LOGIN_BUCKETS_LOCK:
        _LOGIN_BUCKETS.pop(ip, None)


@router.get("/status", response_model = AuthStatusResponse)
async def auth_status() -> AuthStatusResponse:
    """Auth initialization state; ``default_username`` is exposed for first-boot UI prefill only."""
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
async def login(payload: AuthLoginRequest, request: Request) -> Token:
    """Login with username/password. Rate-limited per source IP."""
    ip = _client_key(request)
    blocked_for = _login_blocked(ip)
    if blocked_for > 0:
        raise HTTPException(
            status_code = status.HTTP_429_TOO_MANY_REQUESTS,
            detail = (
                f"Too many failed login attempts from {ip}. "
                f"Try again in {blocked_for} seconds."
            ),
            headers = {"Retry-After": str(blocked_for)},
        )

    record = storage.get_user_and_secret(payload.username)
    if record is None:
        _record_login_failure(ip)
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail = "Incorrect password. Run 'unsloth studio reset-password' in your terminal to reset it.",
        )

    salt, pwd_hash, _jwt_secret, must_change_password = record
    if not hashing.verify_password(payload.password, salt, pwd_hash):
        _record_login_failure(ip)
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail = "Incorrect password. Run 'unsloth studio reset-password' in your terminal to reset it.",
        )

    _clear_login_bucket(ip)
    access_token = create_access_token(subject = payload.username)
    refresh_token = create_refresh_token(subject = payload.username)
    return Token(
        access_token = access_token,
        refresh_token = refresh_token,
        token_type = "bearer",
        must_change_password = must_change_password,
    )


@router.post("/logout", status_code = status.HTTP_204_NO_CONTENT)
async def logout(
    request: Request,
    current_subject: str = Depends(get_current_subject_allow_password_change),
) -> Response:
    """Revoke refresh tokens for the subject; the access token is stateless and expires on its own."""
    try:
        storage.revoke_user_refresh_tokens(current_subject)
    except Exception:
        pass
    try:
        request.app.state.bootstrap_password = None
    except AttributeError:
        pass
    return Response(status_code = status.HTTP_204_NO_CONTENT)


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
    """Exchange a refresh token for a new access+refresh pair (single-use)."""
    consumed = storage.consume_refresh_token(payload.refresh_token)
    if consumed is None:
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail = "Invalid or expired refresh token",
        )
    username, is_desktop = consumed
    new_access_token = create_access_token(subject = username, desktop = is_desktop)
    new_refresh_token = create_refresh_token(subject = username, desktop = is_desktop)

    return Token(
        access_token = new_access_token,
        refresh_token = new_refresh_token,
        token_type = "bearer",
        must_change_password = False
        if is_desktop
        else storage.requires_password_change(username),
    )


@router.post("/change-password", response_model = Token)
async def change_password(
    payload: ChangePasswordRequest,
    request: Request,
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
    try:
        request.app.state.bootstrap_password = None
    except AttributeError:
        pass
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
