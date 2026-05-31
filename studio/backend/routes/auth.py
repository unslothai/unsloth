# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Authentication API routes
"""

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status

import ipaddress
import os
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


# Per-(ip, username) bucket + per-IP aggregate. Account bucket stops one user's
# typos from blocking others; the aggregate stops username-rotation spray.
# Single-process only -- multi-worker deployments need a shared store.
_LOGIN_BUCKETS: dict[tuple[str, str], deque] = {}
_LOGIN_IP_BUCKETS: dict[str, deque] = {}
_LOGIN_BUCKETS_LOCK = threading.Lock()
_LOGIN_WINDOW_SECONDS = 60.0
_LOGIN_MAX_FAILS = 5
_LOGIN_IP_MAX_FAILS = 30
_LOGIN_LOCKOUT_SECONDS = 60
# Bucket-dict cap. On overflow we prune stale entries; if still full the
# failure folds into the per-IP aggregate only.
_LOGIN_MAX_BUCKETS = 4096
# Unrepresentable as a real username (leading NUL); folds unknown-user attempts
# into one slot so attacker cardinality cannot blow the bucket dict.
_UNKNOWN_LOGIN_USER = "\x00unknown-user"


def _trust_forwarded_for() -> bool:
    """Honour X-Forwarded-For only when UNSLOTH_STUDIO_TRUST_FORWARDED is set.

    Off by default so a direct caller cannot spoof the header.
    """
    return os.environ.get("UNSLOTH_STUDIO_TRUST_FORWARDED", "").lower() in (
        "1",
        "true",
        "yes",
    )


def _normalize_forwarded_addr(value: str) -> str:
    """Parse an XFF / Forwarded `for=` value into a bare IP (port-stripped)."""
    value = (value or "").strip().strip('"')
    if not value or value.lower() == "unknown":
        return ""
    if value.startswith("["):
        # Bracketed IPv6, optionally with port.
        end = value.find("]")
        if end <= 0:
            return ""
        host = value[1:end]
    elif value.count(":") == 1:
        # IPv4:port. Bare IPv6 has multiple colons and takes the else branch.
        head, _, tail = value.rpartition(":")
        host = head if tail.isdigit() and head else value
    else:
        host = value
    try:
        return str(ipaddress.ip_address(host))
    except ValueError:
        return ""


def _forwarded_for_from_element(element: str) -> str:
    """Pick the `for=` token out of a single ``Forwarded`` element."""
    for tok in element.split(";"):
        key, sep, val = tok.strip().partition("=")
        if sep and key.lower() == "for":
            return _normalize_forwarded_addr(val)
    return ""


def _client_ip(request: Request | None) -> str:
    if request is None:
        return "_unknown"
    if _trust_forwarded_for():
        xff = request.headers.get("x-forwarded-for", "")
        if xff:
            # First entry is the originating client.
            normalized = _normalize_forwarded_addr(xff.split(",", 1)[0])
            if normalized:
                return normalized
        fwd = request.headers.get("forwarded", "")
        if fwd:
            # First element only -- multi-element headers cannot fork buckets.
            normalized = _forwarded_for_from_element(fwd.split(",", 1)[0])
            if normalized:
                return normalized
    return (request.client.host if request.client else None) or "_unknown"


def _bucket_key(request: Request | None, username: str) -> tuple[str, str]:
    return (_client_ip(request), (username or "").casefold())


def _unknown_user_key(request: Request | None) -> tuple[str, str]:
    return (_client_ip(request), _UNKNOWN_LOGIN_USER)


def _prune_bucket(bucket: deque, now: float) -> None:
    while bucket and now - bucket[0] > _LOGIN_WINDOW_SECONDS:
        bucket.popleft()


def _prune_stale_buckets(now: float) -> None:
    """Drop empty / expired account buckets to bound memory under spray."""
    stale: list[tuple[str, str]] = []
    for key, bucket in _LOGIN_BUCKETS.items():
        _prune_bucket(bucket, now)
        if not bucket:
            stale.append(key)
    for key in stale:
        _LOGIN_BUCKETS.pop(key, None)


def _record_login_failure(key: tuple[str, str]) -> int:
    now = time.monotonic()
    ip, _username = key
    with _LOGIN_BUCKETS_LOCK:
        ip_bucket = _LOGIN_IP_BUCKETS.setdefault(ip, deque())
        _prune_bucket(ip_bucket, now)
        ip_bucket.append(now)

        if key not in _LOGIN_BUCKETS and len(_LOGIN_BUCKETS) >= _LOGIN_MAX_BUCKETS:
            _prune_stale_buckets(now)
        if key in _LOGIN_BUCKETS or len(_LOGIN_BUCKETS) < _LOGIN_MAX_BUCKETS:
            account_bucket = _LOGIN_BUCKETS.setdefault(key, deque())
            _prune_bucket(account_bucket, now)
            account_bucket.append(now)
            return len(account_bucket)
        # Bucket dict is at its cap; per-IP cap still applies via ip_bucket.
        return len(ip_bucket)


def _blocked_for(bucket: deque | None, now: float, max_fails: int) -> int:
    if not bucket:
        return 0
    _prune_bucket(bucket, now)
    if len(bucket) >= max_fails:
        return max(1, int(_LOGIN_WINDOW_SECONDS - (now - bucket[0])))
    return 0


def _login_blocked(key: tuple[str, str]) -> int:
    """Return seconds until the next attempt is allowed, or 0."""
    now = time.monotonic()
    ip, _username = key
    with _LOGIN_BUCKETS_LOCK:
        return max(
            _blocked_for(_LOGIN_BUCKETS.get(key), now, _LOGIN_MAX_FAILS),
            _blocked_for(_LOGIN_IP_BUCKETS.get(ip), now, _LOGIN_IP_MAX_FAILS),
        )


def _clear_login_bucket(key: tuple[str, str]) -> None:
    ip, _username = key
    with _LOGIN_BUCKETS_LOCK:
        _LOGIN_BUCKETS.pop(key, None)
        _LOGIN_IP_BUCKETS.pop(ip, None)


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
    """Login with username/password. Per-account + per-IP rate-limited."""
    key = _bucket_key(request, payload.username)
    unknown_key = _unknown_user_key(request)
    blocked_for = max(_login_blocked(key), _login_blocked(unknown_key))
    if blocked_for > 0:
        raise HTTPException(
            status_code = status.HTTP_429_TOO_MANY_REQUESTS,
            # IP is intentionally not interpolated into the body; behind a
            # proxy or NAT it is either misleading or an info leak.
            detail = (
                f"Too many failed login attempts. "
                f"Try again in {blocked_for} seconds."
            ),
            headers = {"Retry-After": str(blocked_for)},
        )

    record = storage.get_user_and_secret(payload.username)
    if record is None:
        # Record under a single sentinel key per IP so attacker-controlled
        # username cardinality does not allocate buckets without bound.
        _record_login_failure(unknown_key)
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail = "Incorrect password. Run 'unsloth studio reset-password' in your terminal to reset it.",
        )

    salt, pwd_hash, _jwt_secret, must_change_password = record
    if not hashing.verify_password(payload.password, salt, pwd_hash):
        _record_login_failure(key)
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail = "Incorrect password. Run 'unsloth studio reset-password' in your terminal to reset it.",
        )

    _clear_login_bucket(key)
    _clear_login_bucket(unknown_key)
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
