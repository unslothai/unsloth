# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Authentication API routes."""

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status

import base64
import ipaddress
import os
import shlex
import sys
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


def _reset_password_command() -> str:
    """Shell command shown in the 'incorrect password' hint.

    Prefer the absolute path to this install's ``unsloth`` launcher (sibling of
    the running interpreter) so the hint works even when its dir isn't on PATH.

    POSIX paths are shell-quoted. On Windows we use the bare absolute path only
    when it has no spaces (a quoted path differs between cmd and PowerShell);
    otherwise, or if the launcher can't be located, fall back to the PATH form.
    """
    try:
        bin_dir = os.path.dirname(os.path.abspath(sys.executable))
        if os.name == "nt":
            exe = os.path.join(bin_dir, "unsloth.exe")
            if os.path.isfile(exe) and " " not in exe:
                return f"{exe} studio reset-password"
        else:
            exe = os.path.join(bin_dir, "unsloth")
            if os.path.isfile(exe):
                return f"{shlex.quote(exe)} studio reset-password"
    except Exception:
        pass
    return "unsloth studio reset-password"


# Per-(ip, username) bucket + per-IP aggregate. Account bucket stops one user's
# typos from blocking others; the aggregate stops username-rotation spray.
# Single-process only; multi-worker deployments need a shared store.
_LOGIN_BUCKETS: dict[tuple[str, str], deque] = {}
_LOGIN_IP_BUCKETS: dict[str, deque] = {}
_LOGIN_BUCKETS_LOCK = threading.Lock()
_LOGIN_WINDOW_SECONDS = 60.0
_LOGIN_MAX_FAILS = 5
_LOGIN_IP_MAX_FAILS = 30
_LOGIN_LOCKOUT_SECONDS = 60
# Bucket-dict cap. On overflow, reclaim expired buckets; a new IP that still can't
# fit falls back to a sharded overflow rather than evicting a hot bucket.
_LOGIN_MAX_BUCKETS = 4096
# Last full stale-sweep time; rate-limits the O(n) sweep under a burst of new IPs.
_LAST_IP_PRUNE = 0.0
# Sharded overflow for per-IP failures that can't get their own bucket while the
# dict is saturated. Each shard is a small fixed-capacity dict ``ip -> [count,
# window_start]``: a per-IP count (so a source is throttled, and cleared on
# success, by its own failures -- no cross-IP collateral) with hard-bounded
# memory and O(1) lookups. When a shard is full a new IP evicts the lowest-count
# entry (and starts clean, never inheriting its count) rather than growing without
# bound, so a high-cardinality spray can't blow memory/CPU the way a per-failure
# deque could; a persistent attacker keeps a high count and is never the one
# evicted.
_LOGIN_IP_OVERFLOW_SHARDS = 256
_LOGIN_IP_OVERFLOW_MAX = 64  # distinct IPs tracked per shard
_LOGIN_IP_OVERFLOW: list[dict] = [dict() for _ in range(_LOGIN_IP_OVERFLOW_SHARDS)]


def _overflow_shard(ip: str) -> dict:
    return _LOGIN_IP_OVERFLOW[hash(ip) % _LOGIN_IP_OVERFLOW_SHARDS]


def _overflow_record(ip: str, now: float) -> int:
    """Record an overflow failure for ``ip`` and return its windowed count."""
    shard = _overflow_shard(ip)
    entry = shard.get(ip)
    if entry is not None:
        if now - entry[1] > _LOGIN_WINDOW_SECONDS:
            entry[0], entry[1] = 1, now
        else:
            # Only "at or above the per-IP threshold" matters for blocking, so cap
            # the count there. This also keeps the migration into a per-IP bucket
            # bounded -- without the cap a saturated source could accrue an
            # unbounded count, then materialize one deque entry per failure
            # (``[start] * carried``) on the next attempt, allocating an arbitrarily
            # large deque while holding the login lock.
            entry[0] = min(entry[0] + 1, _LOGIN_IP_MAX_FAILS)
        return entry[0]
    if len(shard) >= _LOGIN_IP_OVERFLOW_MAX:
        # Make room by dropping the lowest-count entry, but the new source starts
        # clean -- never inherit the evicted IP's failures, or an unrelated source
        # could be 429'd after one attempt. Worst case under a saturated shard is
        # that a heavy hitter briefly resets, not that a bystander is blocked.
        del shard[min(shard, key = lambda k: shard[k][0])]
    shard[ip] = [1, now]
    return 1


def _overflow_blocked(ip: str, now: float) -> int:
    """Seconds this IP is throttled by its own overflow count, or 0."""
    shard = _overflow_shard(ip)
    entry = shard.get(ip)
    if entry is None:
        return 0
    if now - entry[1] > _LOGIN_WINDOW_SECONDS:
        del shard[ip]
        return 0
    if entry[0] >= _LOGIN_IP_MAX_FAILS:
        return max(1, int(_LOGIN_WINDOW_SECONDS - (now - entry[1])))
    return 0


def _overflow_take(ip: str, now: float) -> tuple[int, float]:
    """Pop ip's overflow entry, returning its ``(count, window_start)`` so the
    count can migrate into a fresh per-IP bucket. ``(0, now)`` if none/expired."""
    entry = _overflow_shard(ip).pop(ip, None)
    if entry is None or now - entry[1] > _LOGIN_WINDOW_SECONDS:
        return 0, now
    # Cap the carried count so the bucket migration never allocates more than the
    # per-IP threshold worth of deque entries (defensive; _overflow_record already
    # clamps, but keep the bound at the consumption site too).
    return min(entry[0], _LOGIN_IP_MAX_FAILS), entry[1]


# Unrepresentable as a real username (leading NUL); folds unknown-user attempts
# into one slot so attacker cardinality can't blow the bucket dict.
_UNKNOWN_LOGIN_USER = "\x00unknown-user"


def _trust_forwarded_for() -> bool:
    """Honour X-Forwarded-For only when UNSLOTH_STUDIO_TRUST_FORWARDED is set.

    Off by default so a direct caller can't spoof the header.
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
        # IPv4:port. Bare IPv6 has multiple colons → else branch.
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
            # First element only; multi-element headers can't fork buckets.
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


def _prune_stale_ip_buckets(now: float) -> None:
    """Drop empty / expired per-IP buckets to bound memory under spray.

    The dict is otherwise reclaimed only on a successful login, so a failure-only
    spray from many (or spoofed) IPs would grow it without bound.
    """
    stale: list[str] = []
    for bucket_ip, bucket in _LOGIN_IP_BUCKETS.items():
        _prune_bucket(bucket, now)
        if not bucket:
            stale.append(bucket_ip)
    for bucket_ip in stale:
        _LOGIN_IP_BUCKETS.pop(bucket_ip, None)


def _record_login_failure(key: tuple[str, str]) -> int:
    global _LAST_IP_PRUNE
    now = time.monotonic()
    ip, _username = key
    with _LOGIN_BUCKETS_LOCK:
        # Keep the dict bounded without disabling throttling and without letting a
        # spray reset a hot bucket: for a new IP at the cap, reclaim expired buckets
        # (rate-limited) to make room.
        ip_bucket = _LOGIN_IP_BUCKETS.get(ip)
        if ip_bucket is None and len(_LOGIN_IP_BUCKETS) >= _LOGIN_MAX_BUCKETS:
            if now - _LAST_IP_PRUNE >= 1.0:
                _prune_stale_ip_buckets(now)
                _LAST_IP_PRUNE = now
        if ip_bucket is None and len(_LOGIN_IP_BUCKETS) >= _LOGIN_MAX_BUCKETS:
            # Still full -- every bucket is hot. Count this failure in the IP's
            # bounded overflow shard instead of evicting a live one, so the spray
            # stays throttled but can't push out (and reset) any IP's own counter.
            ip_fails = _overflow_record(ip, now)
        else:
            if ip_bucket is None:
                ip_bucket = _LOGIN_IP_BUCKETS[ip] = deque()
                # Carry over any overflow failures this IP accrued while the dict
                # was saturated, so straddling the overflow -> bucket transition
                # can't double the effective per-IP limit.
                carried, start = _overflow_take(ip, now)
                ip_bucket.extend([start] * carried)
            _prune_bucket(ip_bucket, now)
            ip_bucket.append(now)
            ip_fails = len(ip_bucket)

        if key not in _LOGIN_BUCKETS and len(_LOGIN_BUCKETS) >= _LOGIN_MAX_BUCKETS:
            _prune_stale_buckets(now)
        if key in _LOGIN_BUCKETS or len(_LOGIN_BUCKETS) < _LOGIN_MAX_BUCKETS:
            account_bucket = _LOGIN_BUCKETS.setdefault(key, deque())
            _prune_bucket(account_bucket, now)
            account_bucket.append(now)
            return len(account_bucket)
        # Both dicts at cap (sustained spray): fall back to the per-IP count.
        return ip_fails


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
        # Honor the IP's overflow shard regardless of current dict capacity: a
        # source counted there during saturation must stay throttled until those
        # failures age out, even if a bucket later frees up -- otherwise a fresh
        # bucket would reset it. Shards are empty outside saturation, so this is a
        # no-op in the common case.
        ip_blocked = max(
            _blocked_for(_LOGIN_IP_BUCKETS.get(ip), now, _LOGIN_IP_MAX_FAILS),
            _overflow_blocked(ip, now),
        )
        return max(_blocked_for(_LOGIN_BUCKETS.get(key), now, _LOGIN_MAX_FAILS), ip_blocked)


def _clear_login_bucket(key: tuple[str, str]) -> None:
    ip, _username = key
    with _LOGIN_BUCKETS_LOCK:
        _LOGIN_BUCKETS.pop(key, None)
        _LOGIN_IP_BUCKETS.pop(ip, None)
        # A successful login resets the IP's throttle, including any overflow it
        # accumulated during saturation (drop only this IP's entry, so a
        # shard-mate's throttle is untouched).
        _overflow_shard(ip).pop(ip, None)


# Sync def (not async): compute_identity_proof touches SQLite on the first call,
# so FastAPI runs it in the threadpool rather than blocking the event loop.
@router.get("/identity")
def identity(nonce: str, request: Request) -> dict:
    """Challenge-response proof this is the real local Studio: caller sends a nonce,
    gets HMAC(install identity secret, nonce, connection address + port).
    Unauthenticated and side-effect free; a process that can't read the same-user
    secret can't forge a proof, and binding to the address/port the connection
    landed on stops a squatter relaying a proof from the real Studio elsewhere."""
    try:
        raw = base64.urlsafe_b64decode(nonce)
    except Exception:
        raise HTTPException(
            status_code = status.HTTP_400_BAD_REQUEST, detail = "nonce must be base64url"
        )
    if not 16 <= len(raw) <= 128:
        raise HTTPException(
            status_code = status.HTTP_400_BAD_REQUEST, detail = "nonce must decode to 16-128 bytes"
        )
    # The address + port the connection actually landed on, from the socket
    # (request.scope is getsockname, so it is the real local address even when
    # bound to 0.0.0.0), never the client-controlled Host header.
    server = request.scope.get("server") or ("", 0)
    host = server[0] or ""
    port = server[1] if server[1] is not None else 0
    return {"proof": storage.compute_identity_proof(raw, host, port)}


@router.get("/status", response_model = AuthStatusResponse)
async def auth_status() -> AuthStatusResponse:
    """Auth initialization state; ``default_username`` is exposed for first-boot UI prefill only."""
    return AuthStatusResponse(
        initialized = storage.is_initialized(),
        default_username = storage.DEFAULT_ADMIN_USERNAME,
        requires_password_change = storage.requires_password_change(storage.DEFAULT_ADMIN_USERNAME)
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
            # IP not interpolated into the body; behind a proxy/NAT it's
            # misleading or an info leak.
            detail = (f"Too many failed login attempts. " f"Try again in {blocked_for} seconds."),
            headers = {"Retry-After": str(blocked_for)},
        )

    record = storage.get_user_and_secret(payload.username)
    if record is None:
        # Record under one sentinel key per IP so attacker-controlled username
        # cardinality can't allocate unbounded buckets.
        _record_login_failure(unknown_key)
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail = f"Incorrect password. To reset it, run this in your terminal: {_reset_password_command()}",
        )

    salt, pwd_hash, _jwt_secret, must_change_password = record
    if not hashing.verify_password(payload.password, salt, pwd_hash):
        _record_login_failure(key)
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail = f"Incorrect password. To reset it, run this in your terminal: {_reset_password_command()}",
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
    request: Request, current_subject: str = Depends(get_current_subject_allow_password_change)
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
        must_change_password = False if is_desktop else storage.requires_password_change(username),
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

    # Single transaction: a separate refresh-token purge could fail after the
    # password commit, leaving pre-change tokens able to mint access tokens.
    storage.update_password(current_subject, payload.new_password, revoke_refresh_tokens = True)
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
    payload: CreateApiKeyRequest, current_subject: str = Depends(get_current_subject)
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
async def list_api_keys(current_subject: str = Depends(get_current_subject)) -> ApiKeyListResponse:
    """List all API keys for the authenticated user (raw keys are never exposed)."""
    rows = storage.list_api_keys(current_subject)
    return ApiKeyListResponse(
        api_keys = [_row_to_api_key_response(r) for r in rows],
    )


@router.delete("/api-keys/{key_id}")
async def revoke_api_key(key_id: int, current_subject: str = Depends(get_current_subject)) -> dict:
    """Revoke (soft-delete) an API key."""
    if not storage.revoke_api_key(current_subject, key_id):
        raise HTTPException(
            status_code = status.HTTP_404_NOT_FOUND,
            detail = "API key not found",
        )
    return {"detail": "API key revoked"}
