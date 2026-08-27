# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import base64
import hashlib
import hmac
import json
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, Optional, Tuple

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.security.utils import get_authorization_scheme_param
import jwt
from starlette.concurrency import run_in_threadpool

from .storage import (
    API_KEY_PREFIX,
    DEFAULT_ADMIN_USERNAME,
    LINK_TOKEN_EXPIRE_SECONDS,
    consume_link_token,
    credential_generation,
    get_jwt_secret,
    get_user_and_secret,
    load_jwt_secret,
    save_link_token,
    save_refresh_token,
    validate_api_key_with_credential,
    verify_refresh_token,
)

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60
REFRESH_TOKEN_EXPIRE_DAYS = 7

# internal schemes, never sent by a client: no token at all, and a token to ignore if unusable
KEYLESS_SCHEME = "Keyless"
KEYLESS_FALLBACK_SCHEME = "KeylessBearer"
_KEYLESS_CREDENTIALS = HTTPAuthorizationCredentials(
    scheme = KEYLESS_SCHEME,
    credentials = "",
)


def is_keyless(credentials: Optional[HTTPAuthorizationCredentials]) -> bool:
    """True when the keyless API access setting had a hand in admitting this caller."""
    return credentials is not None and credentials.scheme in (
        KEYLESS_SCHEME,
        KEYLESS_FALLBACK_SCHEME,
    )


def _names_a_session(token: str) -> bool:
    """Whether this bearer claims an Unsloth sign-in this install actually knows.

    A session token stays authoritative even under keyless API access: letting an
    expired one through would leave the app running as the admin instead of prompting
    for a sign-in. The subject is confirmed against storage because the claim itself is
    unverified here, so a token merely shaped like a JWT -- which is a legal value for
    the ``api_key`` the OpenAI SDKs always send -- is treated as the credential it is.
    """
    subject = _decode_subject_without_verification(token)
    return subject is not None and get_user_and_secret(subject) is not None


def bearer_names_a_session(token: str) -> bool:
    """Public form of the session check, for callers that only have the raw token."""
    return _names_a_session(token)


def bearer_is_valid_api_key(token: str) -> bool:
    """Whether this bearer is an sk-unsloth key this install still accepts.

    Such a key authenticates as itself even while keyless API access is on, so the
    callers below must not treat it as a credential the setting had to stand in for.
    Asked ahead of the real validation, so it leaves ``last_used_at`` to that one.
    """
    return (
        token.startswith(API_KEY_PREFIX)
        and validate_api_key_with_credential(token, touch = False) is not None
    )


def admitted_without_credential(credentials: Optional[HTTPAuthorizationCredentials]) -> bool:
    """True when the keyless setting alone let this caller in.

    Narrower than ``is_keyless``, which also covers a working API key that happened
    to arrive while the setting was on. Routes whose effect outlives the setting need
    this stricter form: turning keyless access back off has to undo what it allowed.
    """
    if credentials is None:
        return False
    if credentials.scheme == KEYLESS_SCHEME:
        return True
    return credentials.scheme == KEYLESS_FALLBACK_SCHEME


def _request_would_use_keyless(request: Any) -> bool:
    """Classify a request before the security dependency has recorded its result."""
    from utils.keyless_api_access import APPROVED_DUMMY_BEARERS, keyless_request_allowed

    if not keyless_request_allowed(request):
        return False
    try:
        raw_headers = getattr(request, "scope", {}).get("headers") or ()
        values = [
            bytes(value).decode("latin-1")
            for name, value in raw_headers
            if bytes(name).lower() == b"authorization"
        ]
    except Exception:
        return False
    if not values:
        return True
    if len(values) != 1:
        return False
    scheme, token = get_authorization_scheme_param(values[0])
    return scheme.lower() == "bearer" and token in APPROVED_DUMMY_BEARERS


def request_admitted_without_credential(request: Request) -> bool:
    """``admitted_without_credential`` for a caller that holds only the request.

    Costs a key validation, so ask it late: past the cheap disqualifiers, next to the
    effect being guarded.
    """
    from utils.keyless_api_access import request_was_admitted_keyless

    recorded = request_was_admitted_keyless(request)
    return _request_would_use_keyless(request) if recorded is None else recorded


def admitted_without_session(request: Any) -> bool:
    """True when keyless API access lets this request through with no Unsloth sign-in.

    The single predicate behind both the auth dependency below and the route-level
    checks that ask whether a caller is the Unsloth UI or a programmatic client.
    """
    from utils.keyless_api_access import request_was_admitted_keyless

    recorded = request_was_admitted_keyless(request)
    return _request_would_use_keyless(request) if recorded is None else recorded


class _BearerOrKeyless(HTTPBearer):
    """Read ``Authorization: Bearer <token>``, admitting a caller without one.

    When the setting is off this behaves exactly like ``HTTPBearer``, errors included.
    """

    async def __call__(self, request: Request) -> Optional[HTTPAuthorizationCredentials]:
        from utils.keyless_api_access import (
            APPROVED_DUMMY_BEARERS,
            keyless_request_allowed,
            mark_keyless_admission,
            request_was_admitted_keyless,
        )

        raw_headers = getattr(request, "scope", {}).get("headers") or ()
        authorization = [
            bytes(value).decode("latin-1")
            for name, value in raw_headers
            if bytes(name).lower() == b"authorization"
        ]
        if len(authorization) > 1:
            mark_keyless_admission(request, False)
            raise HTTPException(
                status_code = status.HTTP_403_FORBIDDEN,
                detail = "Invalid authentication credentials",
            )
        header = authorization[0] if authorization else ""
        scheme, token = get_authorization_scheme_param(header)
        usable_bearer = bool(scheme.lower() == "bearer" and token)
        recorded = request_was_admitted_keyless(request)
        eligible = (
            await run_in_threadpool(keyless_request_allowed, request)
            if recorded is None
            else recorded
        )
        if not authorization and eligible:
            mark_keyless_admission(request, True)
            return _KEYLESS_CREDENTIALS
        dummy = eligible and usable_bearer and token in APPROVED_DUMMY_BEARERS
        mark_keyless_admission(request, dummy)
        if dummy:
            return HTTPAuthorizationCredentials(
                scheme = KEYLESS_FALLBACK_SCHEME,
                credentials = token,
            )
        if usable_bearer:
            return HTTPAuthorizationCredentials(scheme = scheme, credentials = token)
        return await super().__call__(request)


# Domain-separation label for the link-token signing key. The key is derived from
# the user's JWT secret (so a password change, which rotates that secret,
# invalidates outstanding link tokens) but is NOT the JWT secret itself: a link
# token must never be accepted as a bearer access token, so it is signed with a
# different key and can't validate on the access-token path. That separation now
# also has to hold against the keyless schemes above: _BearerOrKeyless only ever
# yields KEYLESS_SCHEME or a real Bearer credential, and neither path consults
# this key, so a link token has no way onto the access-token path.
_LINK_TOKEN_KEY_LABEL = b"unsloth-studio-link-token-v1"


# scheme_name pinned so the OpenAPI securitySchemes entry keeps its published name
security = _BearerOrKeyless(scheme_name = "HTTPBearer")  # Reads Authorization: Bearer <token>


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


# ---------------------------------------------------------------------------
# One-time link tokens (opt-in Colab same-tab handoff)
# ---------------------------------------------------------------------------


def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64url_decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + padding)


def _b64url_decode_canonical(value: str) -> Optional[bytes]:
    """Decode unpadded base64url, rejecting every non-canonical spelling.

    urlsafe_b64decode is not injective: it silently discards non-alphabet
    characters, accepts already-present "=" padding, and ignores the unused pad
    bits of the final character (RFC 4648 s3.5 puts the zero-pad-bits MUST on
    encoders, and only lets decoders reject). A 32-byte HMAC therefore has four
    spellings that decode to the same bytes, so flipping the last character of a
    signature still passes compare_digest whenever it is canonically "A" -- about
    1 in 16 tokens. Re-encoding the decoded bytes and requiring an exact match
    makes the mapping one-to-one, so any altered signature text is rejected.
    Returns None instead of raising, so callers reject without leaking which
    check failed.
    """
    try:
        raw = _b64url_decode(value)
    except (ValueError, TypeError):
        return None
    if _b64url_encode(raw) != value:
        return None
    return raw


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
    """Parse the (still unverified) payload, or None for anything unparseable.

    RecursionError is caught alongside the parse errors: on Python 3.10/3.11 the
    json scanner raises it at ~1000 nesting levels, and 1000 nested arrays encode
    to well under LINK_TOKEN_MAX_LENGTH, so an unauthenticated caller could
    otherwise turn every /link-exchange POST into a 500 that never reaches the
    failure counter and so is never rate limited. Both are rejected here as a
    plain bad token instead. (3.12+ raises no error for that depth; the catch
    keeps every supported version on the same path.)
    """
    raw = _b64url_decode_canonical(payload_b64)
    if raw is None:
        return None
    try:
        return json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError, RecursionError):
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


def exchange_link_token_with_secret(token: str) -> Optional[Tuple[str, str]]:
    """Validate and consume a one-time link token, returning ``(subject, secret)``.

    ``secret`` is the JWT secret whose derived key validated the signature. The
    session-issuing route binds the tokens it mints to this exact secret: a
    concurrent password change rotates the secret, and if it rotated between this
    consumption and issuance the route revokes the just-minted session and rejects.
    Without that binding a pre-change link token consumed just before the rotation
    committed could mint a session under the NEW secret and survive the change --
    the consume-before-rotation TOCTOU (cf. Keycloak CVE-2026-1035, where
    non-atomic single-use enforcement undermined refresh-token rotation).

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

    # Capture the secret (not just the derived key) so the caller can detect a
    # rotation that races issuance; None means an unknown user -> reject.
    secret = get_jwt_secret(subject)
    if secret is None:
        return None
    key = hmac.new(secret.encode("utf-8"), _LINK_TOKEN_KEY_LABEL, hashlib.sha256).digest()
    expected_sig = hmac.new(key, payload_b64.encode("ascii"), hashlib.sha256).digest()
    # Canonical decode: a permissive base64url decode would accept three sibling
    # spellings of the same signature bytes, so a tampered trailing character
    # could still verify (see _b64url_decode_canonical).
    provided_sig = _b64url_decode_canonical(sig_b64)
    if provided_sig is None:
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
    return subject, secret


def exchange_link_token(token: str) -> Optional[str]:
    """Validate and consume a one-time link token, returning its subject or None.

    Thin wrapper over :func:`exchange_link_token_with_secret` for callers that do
    not issue a session and so need only the subject.
    """
    result = exchange_link_token_with_secret(token)
    return result[0] if result is not None else None


async def get_current_subject(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Validate JWT and require the password-change flow to be completed."""
    subject, _generation = await _get_current_credential(
        credentials,
        allow_password_change = False,
    )
    return subject


async def get_current_credential(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> Tuple[str, Optional[str]]:
    """As get_current_subject, but also returns the credential generation.

    For routes that persist a new credential and must not do so on behalf of one
    a concurrent reset has revoked.
    """
    return await _get_current_credential(
        credentials,
        allow_password_change = False,
    )


async def authenticated_via_api_key(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> bool:
    """True when the caller used an sk-unsloth API key, not a UI session JWT.

    Lets routes treat programmatic API callers differently from the Unsloth UI
    (e.g. refuse a teardown the UI would allow). A keyless caller counts as an API
    caller too: it is the same programmatic surface, only without the key, so every
    guard an API key faces still applies to it.
    """
    if is_keyless(credentials):
        return True
    return bool(credentials and credentials.credentials.startswith(API_KEY_PREFIX))


async def credentials_for_token(
    request: Any, token: Optional[str]
) -> Optional[HTTPAuthorizationCredentials]:
    """What ``security`` would resolve for a bearer the route read for itself.

    Routes that take the token from somewhere the dependency cannot see, such as the
    ``?token=`` query param an ``<img src>`` has to use, would otherwise miss keyless
    API access entirely and answer 401 on a scope that covers them. None means no
    usable credential and no setting to stand in for one.
    """
    from utils.keyless_api_access import APPROVED_DUMMY_BEARERS, keyless_request_allowed

    # A real token is authoritative and never needs keyless classification. The
    # remaining settings/listener reads use SQLite and DNS, so keep them off the
    # event loop just like the normal credential lookup path.
    if token and token not in APPROVED_DUMMY_BEARERS:
        return HTTPAuthorizationCredentials(scheme = "Bearer", credentials = token)
    eligible = await run_in_threadpool(keyless_request_allowed, request)
    keyless = eligible and (token is None or token in APPROVED_DUMMY_BEARERS)
    if token:
        return HTTPAuthorizationCredentials(
            scheme = KEYLESS_FALLBACK_SCHEME if keyless else "Bearer",
            credentials = token,
        )
    return _KEYLESS_CREDENTIALS if keyless else None


async def authenticated_without_credential(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> bool:
    """Dependency form of ``admitted_without_credential``."""
    return admitted_without_credential(credentials)


def require_ui_session_for_local_commands(via_api_key: bool) -> None:
    """Refuse an sk-unsloth API key that asks to define a local (stdio) MCP command.

    stdio MCP runs a command on this host as the backend user, outside the
    python/terminal sandbox, so only a UI session may choose what runs. API keys
    keep http(s) MCP, and stdio servers the owner already configured.
    """
    if via_api_key:
        raise HTTPException(
            status_code = status.HTTP_403_FORBIDDEN,
            detail = "Local (stdio) MCP servers can only be configured from the Unsloth UI, "
            "not with an API key. Use an http:// or https:// MCP server instead.",
        )


async def allow_ambient_hf_token(via_api_key: bool = Depends(authenticated_via_api_key)) -> bool:
    """Whether a download this caller starts may fall back to the backend's own HF_TOKEN.

    A UI session already gets the saved token from Settings, so the ambient one grants it
    nothing new. ``require_ui_session`` refuses an sk-unsloth API key that same token, so it
    must not reach private repos by naming one in a download instead; it sends its own token
    in ``X-Unsloth-HF-Token``.
    """
    return not via_api_key


async def authenticated_via_desktop_jwt(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> bool:
    """True when the caller is the local desktop app, not a browser session or API key.

    Lets routes treat the desktop as an authority of its own: it authenticates
    with a local secret rather than the account password.
    """
    return await run_in_threadpool(is_desktop_access_token, credentials.credentials)


async def get_current_subject_allow_password_change(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> str:
    """Validate JWT but allow access to the password-change endpoint."""
    subject, _generation = await _get_current_credential(
        credentials,
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


def _admin_credential(*, allow_password_change: bool) -> Tuple[str, Optional[str]]:
    """Resolve the local admin for a caller admitted by the keyless API access setting."""
    record = get_user_and_secret(DEFAULT_ADMIN_USERNAME)
    if record is None:
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail = "Invalid or expired token",
        )
    _salt, _pwd_hash, jwt_secret, must_change_password = record
    if must_change_password and not allow_password_change:
        raise HTTPException(
            status_code = status.HTTP_403_FORBIDDEN,
            detail = "Password change required",
        )
    return DEFAULT_ADMIN_USERNAME, credential_generation(jwt_secret)


async def _get_current_credential(
    credentials: HTTPAuthorizationCredentials, *, allow_password_change: bool
) -> Tuple[str, Optional[str]]:
    """Validate the bearer and return ``(subject, credential generation)``.

    The generation is the credential version this request actually authenticated
    against. Routes that persist new credentials must bind their write to it, or
    a reset landing mid-request would bless what it just revoked.

    Credential reads run in the threadpool so stalled SQLite cannot block the event loop.
    """
    if credentials.scheme == KEYLESS_SCHEME:
        return await run_in_threadpool(
            _admin_credential, allow_password_change = allow_password_change
        )

    if credentials.scheme == KEYLESS_FALLBACK_SCHEME:
        from utils.keyless_api_access import APPROVED_DUMMY_BEARERS
        if credentials.credentials not in APPROVED_DUMMY_BEARERS:
            raise HTTPException(
                status_code = status.HTTP_401_UNAUTHORIZED,
                detail = "Invalid authentication credentials",
            )
        return await run_in_threadpool(
            _admin_credential, allow_password_change = allow_password_change
        )

    token = credentials.credentials

    # --- API key path (sk-unsloth-...) ---
    if token.startswith(API_KEY_PREFIX):
        verified = await run_in_threadpool(validate_api_key_with_credential, token)
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

    record = await run_in_threadpool(get_user_and_secret, subject)
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
