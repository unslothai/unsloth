# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Authentication API routes."""

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status

import base64
import importlib.util
import ipaddress
import os
import shlex
import sys
import threading
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path

from models.auth import (
    ApiKeyListResponse,
    ApiKeyResponse,
    AuthLoginRequest,
    AuthStatusResponse,
    ChangePasswordRequest,
    CreateApiKeyRequest,
    CreateApiKeyResponse,
    DesktopInitialPasswordRequest,
    DesktopLoginRequest,
    LinkTokenRequest,
    RefreshTokenRequest,
)
from models.users import Token
from auth import storage, hashing
from auth.authentication import (
    authenticated_via_desktop_jwt,
    authenticated_without_credential,
    create_access_token,
    create_refresh_token,
    exchange_link_token_with_secret,
    get_current_credential,
    get_current_subject,
    get_current_subject_allow_password_change,
    refresh_access_token,
)

router = APIRouter()


def _require_a_credential_of_its_own(what: str):
    """Refuse a caller that nothing but keyless API access let in.

    For effects that outlive the setting: turning keyless access back off does not
    withdraw a key it handed out, restore one it destroyed, or undo a sign-out it
    forced. Listing keys is refused with them because it names the key to revoke.
    """

    def dependency(no_credential: bool = Depends(authenticated_without_credential)) -> None:
        if no_credential:
            raise HTTPException(
                status_code = status.HTTP_403_FORBIDDEN,
                detail = f"{what} can only be done from the Unsloth UI or with an existing API key.",
            )

    return dependency


# Byte-identical to _WINDOWS_CLI_ENTRYPOINT in unsloth_cli/commands/studio.py and to
# the bootstrap unsloth_cli/__main__.py documents for user-site installs.
_CLI_BOOTSTRAP = (
    "import sys, os; sys.path[:1] = [x for x in sys.path[:1] if getattr(sys.flags, 'safe_path', False) or x not in ('', os.getcwd())]; "
    "sys.argv[0] = 'unsloth'; from unsloth_cli import app; sys.exit(app())"
)


def _cli_is_inside(prefix: str) -> bool:
    """Whether unsloth_cli lives under *prefix*, so -I would still find it.

    Located rather than imported: this runs in a request handler, and a spec
    lookup answers the only question asked here, which is where the package is
    on disk and not whether it starts.
    """
    try:
        spec = importlib.util.find_spec("unsloth_cli")
        origin = getattr(spec, "origin", None)
        if not origin:
            # A namespace package, or nothing found. Either way there is no
            # location to compare, so do not claim isolation would work.
            return False
        return Path(origin).resolve().is_relative_to(Path(prefix).resolve())
    except (ImportError, OSError, ValueError, AttributeError):
        return False


def _reset_password_command() -> str:
    """Shell command shown in the 'incorrect password' hint.

    Prefer the absolute path to this install's ``unsloth`` launcher (sibling of
    the running interpreter) so the hint works even when its dir isn't on PATH.

    POSIX paths are shell-quoted. On Windows we use the bare absolute path only
    when it has no spaces (a quoted path differs between cmd and PowerShell);
    otherwise, or if the launcher can't be located, fall back to the PATH form.

    Windows never names unsloth.exe here, present or not. Existing is not the
    same as runnable: an Application Control policy leaves the generated,
    unsigned unsloth.exe on disk and denies it at CreateProcess (issue #8490),
    and a bare `unsloth` resolves to that same file because PATHEXT puts .EXE
    ahead of the .cmd shim. Whoever is locked out of Unsloth is exactly who needs
    this command to work, so it must not be the one a policy refuses. Preference
    order is therefore the interpreter's module entry, which needs no quoting in
    cmd or PowerShell, then `unsloth.cmd` -- spelling the extension is what stops
    PATHEXT reaching for the executable.

    -I only when the package is inside this interpreter's own prefix. -I implies
    -s, so a ``pip install --user`` install would be told to run a command that
    cannot find itself; unsloth_cli/__main__.py documents that exception and the
    bootstrap to use instead, and this prints that bootstrap. It is safe to show
    to either shell: the trampoline contains single quotes only, so one pair of
    double quotes wraps it identically in cmd and in PowerShell.
    """
    try:
        bin_dir = os.path.dirname(os.path.abspath(sys.executable))
        if os.name == "nt":
            python = os.path.abspath(sys.executable)
            if " " not in python:
                if _cli_is_inside(sys.prefix):
                    return f"{python} -I -m unsloth_cli studio reset-password"
                return f'{python} -X utf8 -c "{_CLI_BOOTSTRAP}" studio reset-password'
            # A spaced interpreter path cannot be written unquoted, so fall
            # through to the PATH form below.
        else:
            exe = os.path.join(bin_dir, "unsloth")
            if os.path.isfile(exe):
                return f"{shlex.quote(exe)} studio reset-password"
    except Exception:
        pass
    if os.name == "nt":
        return "unsloth.cmd studio reset-password"
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
    """Challenge-response proof this is the real local Unsloth: caller sends a nonce,
    gets HMAC(install identity secret, nonce, connection address + port).
    Unauthenticated and side-effect free; a process that can't read the same-user
    secret can't forge a proof, and binding to the address/port the connection
    landed on stops a squatter relaying a proof from the real Unsloth elsewhere."""
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


# FastAPI offloads sync reads; mutations stay on-loop to preserve atomic sequences.
@router.get("/status", response_model = AuthStatusResponse)
def auth_status() -> AuthStatusResponse:
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

    salt, pwd_hash, jwt_secret, must_change_password = record
    if not hashing.verify_password(payload.password, salt, pwd_hash):
        _record_login_failure(key)
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail = f"Incorrect password. To reset it, run this in your terminal: {_reset_password_command()}",
        )

    _clear_login_bucket(key)
    _clear_login_bucket(unknown_key)
    # Issue against the credential version just verified, not whatever is in the DB
    # now: a concurrent reset-password must not hand this login a post-reset session.
    access_token = create_access_token(subject = payload.username, secret = jwt_secret)
    refresh_token = create_refresh_token(subject = payload.username, secret = jwt_secret)
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
    _own_credential: None = Depends(_require_a_credential_of_its_own("Signing out")),
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
    verified = storage.validate_desktop_secret_with_credential(payload.secret)
    if verified is None:
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail = "Desktop authentication failed",
        )
    username, jwt_secret = verified

    return Token(
        access_token = create_access_token(subject = username, desktop = True, secret = jwt_secret),
        refresh_token = create_refresh_token(subject = username, desktop = True, secret = jwt_secret),
        token_type = "bearer",
        must_change_password = False,
    )


# Sync def (not async), like /identity: every step here is blocking SQLite work
# (token lookup, single-use consume, refresh-token insert) that can wait out the
# connection busy timeout while another writer holds the auth DB. On an async
# handler that wait pins the event loop and stalls every other request; FastAPI
# runs a sync handler in its threadpool instead. The failure buckets this route
# shares with /login are guarded by _LOGIN_BUCKETS_LOCK, so threadpool execution
# is safe.
@router.post("/link-exchange", response_model = Token)
def link_exchange(payload: LinkTokenRequest, request: Request) -> Token:
    """Exchange a one-time, short-TTL link token for normal session tokens.

    Powers the opt-in Colab same-tab handoff: the same-tab URL carries a
    single-use ``?link_token=...`` the UI posts here to obtain the same JWT the
    login form issues. The token is consumed here (a replay is rejected) and is
    never logged. Unauthenticated by design -- the token itself is the credential.

    Per-IP failure rate-limited like /login. This endpoint is unauthenticated and
    each attempt performs a SQLite lookup plus HMAC/base64 processing, so without a
    limiter an attacker could spray invalid tokens and saturate the threadpool with
    that work. There is no username here (the token is the credential), so failures
    fold into the per-IP aggregate bucket; the bound is checked BEFORE any storage
    work, and a successful exchange clears the bucket (mirrors /login).
    """
    ip_key = _unknown_user_key(request)
    blocked_for = _login_blocked(ip_key)
    if blocked_for > 0:
        raise HTTPException(
            status_code = status.HTTP_429_TOO_MANY_REQUESTS,
            # IP not interpolated into the body; behind a proxy/NAT it's
            # misleading or an info leak.
            detail = (f"Too many failed link-token exchanges. Try again in {blocked_for} seconds."),
            headers = {"Retry-After": str(blocked_for)},
        )

    exchanged = exchange_link_token_with_secret(payload.link_token)
    if exchanged is None:
        _record_login_failure(ip_key)
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail = "Invalid, expired, or already-used link token",
        )
    username, secret_at_exchange = exchanged
    access_token = create_access_token(subject = username)
    refresh_token = create_refresh_token(subject = username)
    # Bind session issuance to the JWT secret the link token validated against. A
    # concurrent password change rotates that secret (and revokes refresh tokens)
    # to invalidate every outstanding session; if it rotated between the single-use
    # consumption above and this issuance, revoke the tokens we just minted and
    # reject, so a pre-change link token cannot mint a session that survives the
    # change (consume-before-rotation TOCTOU). A rotation that lands after this
    # recheck is caught by that same refresh-token revocation and the JWT signature
    # change, so no issued session outlives the password change.
    if storage.get_jwt_secret(username) != secret_at_exchange:
        try:
            storage.consume_refresh_token(refresh_token)
        except Exception:
            pass
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail = "Invalid, expired, or already-used link token",
        )
    # A valid single-use token proves legitimacy: reset this IP's failure throttle,
    # exactly as a successful /login does.
    _clear_login_bucket(ip_key)
    return Token(
        access_token = access_token,
        refresh_token = refresh_token,
        token_type = "bearer",
        must_change_password = storage.requires_password_change(username),
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
    username, is_desktop, jwt_secret = consumed
    new_access_token = create_access_token(subject = username, desktop = is_desktop, secret = jwt_secret)
    new_refresh_token = create_refresh_token(
        subject = username, desktop = is_desktop, secret = jwt_secret
    )

    return Token(
        access_token = new_access_token,
        refresh_token = new_refresh_token,
        token_type = "bearer",
        must_change_password = False if is_desktop else storage.requires_password_change(username),
    )


@router.post("/desktop-initial-password", response_model = Token)
async def set_desktop_initial_password(
    payload: DesktopInitialPasswordRequest,
    request: Request,
    current_subject: str = Depends(get_current_subject_allow_password_change),
    is_desktop: bool = Depends(authenticated_via_desktop_jwt),
) -> Token:
    """Set the first real password from the desktop app, which never sees the seeded one.

    Desktop auth is passwordless, so the desktop user cannot complete the normal
    flow: it needs the generated bootstrap password that only the terminal ever
    printed. Remote browser logins do need a real password, so an
    already-authenticated desktop session may set it while the seeded credential
    is still in place. Once set, change-password owns every later change.
    """
    if not is_desktop:
        raise HTTPException(
            status_code = status.HTTP_403_FORBIDDEN,
            detail = "This action requires the Unsloth desktop app.",
        )

    record = storage.get_user_and_secret(current_subject)
    if record is None:
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail = "User session is invalid",
        )

    _salt, pwd_hash, _jwt_secret, must_change_password = record
    if not must_change_password:
        raise HTTPException(
            status_code = status.HTTP_409_CONFLICT,
            detail = "A password is already set. Change it instead.",
        )
    if any(ch.isspace() for ch in payload.new_password):
        raise HTTPException(
            status_code = status.HTTP_400_BAD_REQUEST,
            detail = "New password cannot contain spaces",
        )

    # Conditional on the credential just read: a web password change or a
    # reset-password landing while this request is in flight must not be
    # overwritten by a caller that verified no password at all.
    new_secret = storage.update_password(
        current_subject,
        payload.new_password,
        revoke_refresh_tokens = True,
        expect_password_hash = pwd_hash,
        preserve_desktop_secret = True,
    )
    if new_secret is None:
        raise HTTPException(
            status_code = status.HTTP_409_CONFLICT,
            detail = "The password changed while this request was in flight. Try again.",
        )
    try:
        request.app.state.bootstrap_password = None
    except AttributeError:
        pass
    access_token = create_access_token(subject = current_subject, desktop = True, secret = new_secret)
    refresh_token = create_refresh_token(subject = current_subject, desktop = True, secret = new_secret)
    return Token(
        access_token = access_token,
        refresh_token = refresh_token,
        token_type = "bearer",
        must_change_password = False,
    )


@router.post("/change-password", response_model = Token)
async def change_password(
    payload: ChangePasswordRequest,
    request: Request,
    current_subject: str = Depends(get_current_subject_allow_password_change),
    is_desktop: bool = Depends(authenticated_via_desktop_jwt),
    _own_credential: None = Depends(_require_a_credential_of_its_own("Changing passwords")),
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
    if any(ch.isspace() for ch in payload.new_password):
        raise HTTPException(
            status_code = status.HTTP_400_BAD_REQUEST,
            detail = "New password cannot contain spaces",
        )
    if payload.current_password == payload.new_password:
        raise HTTPException(
            status_code = status.HTTP_400_BAD_REQUEST,
            detail = "New password must be different from the current password",
        )

    # Single transaction: a separate refresh-token purge could fail after the
    # password commit, leaving pre-change tokens able to mint access tokens.
    # Conditional on the hash just verified: a reset-password that landed while
    # this request was in flight must not be overwritten by it.
    # The desktop app authenticates with a local secret rather than this
    # password; revoking that secret would break its auto-auth over a change it
    # made itself. A browser session still revokes it.
    new_secret = storage.update_password(
        current_subject,
        payload.new_password,
        revoke_refresh_tokens = True,
        expect_password_hash = pwd_hash,
        preserve_desktop_secret = is_desktop,
    )
    if new_secret is None:
        raise HTTPException(
            status_code = status.HTTP_409_CONFLICT,
            detail = "The password changed while this request was in flight. Sign in again.",
        )
    try:
        request.app.state.bootstrap_password = None
    except AttributeError:
        pass
    access_token = create_access_token(
        subject = current_subject, desktop = is_desktop, secret = new_secret
    )
    refresh_token = create_refresh_token(
        subject = current_subject, desktop = is_desktop, secret = new_secret
    )
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
    credential: tuple = Depends(get_current_credential),
    _own_credential: None = Depends(_require_a_credential_of_its_own("Managing API keys")),
) -> CreateApiKeyResponse:
    """Create a new API key. The raw key is returned once and cannot be retrieved later."""
    current_subject, generation = credential
    expires_at = None
    if payload.expires_in_days is not None:
        expires_at = (
            datetime.now(timezone.utc) + timedelta(days = payload.expires_in_days)
        ).isoformat()

    try:
        raw_key, row = storage.create_api_key(
            username = current_subject,
            name = payload.name,
            expires_at = expires_at,
            expect_gen = generation,
        )
    except storage.CredentialRotated:
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail = "Invalid or expired token",
        )
    return CreateApiKeyResponse(
        key = raw_key,
        api_key = _row_to_api_key_response(row),
    )


@router.get("/api-keys", response_model = ApiKeyListResponse)
def list_api_keys(
    current_subject: str = Depends(get_current_subject),
    _own_credential: None = Depends(_require_a_credential_of_its_own("Managing API keys")),
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
    _own_credential: None = Depends(_require_a_credential_of_its_own("Managing API keys")),
) -> dict:
    """Revoke (soft-delete) an API key."""
    if not storage.revoke_api_key(current_subject, key_id):
        raise HTTPException(
            status_code = status.HTTP_404_NOT_FOUND,
            detail = "API key not found",
        )
    return {"detail": "API key revoked"}
