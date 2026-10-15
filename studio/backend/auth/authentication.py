# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, Optional, Tuple

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.security.utils import get_authorization_scheme_param
import jwt
from starlette.concurrency import run_in_threadpool

from utils.account_context import OWNER, AccountContext, bind_account

from .storage import (
    get_account,
    get_user_record,
    API_KEY_PREFIX,
    DEFAULT_ADMIN_USERNAME,
    credential_generation,
    get_jwt_secret,
    get_user_and_secret,
    load_jwt_secret,
    save_refresh_token,
    validate_api_key_account,
    validate_api_key_with_credential,
    verify_refresh_token,
)

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60
REFRESH_TOKEN_EXPIRE_DAYS = 7

# Internal schemes, never sent by a client.
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
    """Subject is checked in storage: the claim is unverified, so a JWT-shaped api_key still counts."""
    subject = _decode_subject_without_verification(token)
    return subject is not None and get_user_and_secret(subject) is not None


def bearer_names_a_session(token: str) -> bool:
    """Public form of the session check, for callers that only have the raw token."""
    return _names_a_session(token)


def bearer_is_valid_api_key(token: str) -> bool:
    """Checks an sk-unsloth key with touch=False, so only the later real validation updates last_used_at."""
    return (
        token.startswith(API_KEY_PREFIX)
        and validate_api_key_with_credential(token, touch = False) is not None
    )


def admitted_without_credential(credentials: Optional[HTTPAuthorizationCredentials]) -> bool:
    """Keyless setting alone let this caller in; routes that outlive the setting need this stricter form."""
    if credentials is None:
        return False
    if credentials.scheme == KEYLESS_SCHEME:
        return True
    return credentials.scheme == KEYLESS_FALLBACK_SCHEME


def _request_would_use_keyless(request: Any) -> bool:
    """Classify a request before the security dependency has recorded its result."""
    from utils.keyless_api_access import (
        APPROVED_DUMMY_BEARERS,
        is_empty_bearer,
        keyless_request_allowed,
    )

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
    if is_empty_bearer(values[0]):
        return True
    scheme, token = get_authorization_scheme_param(values[0])
    return scheme.lower() == "bearer" and token in APPROVED_DUMMY_BEARERS


def request_admitted_without_credential(request: Request) -> bool:
    """``admitted_without_credential`` for a caller that holds only the request. Costs a key
    validation, so ask it late: past the cheap disqualifiers, next to the effect being guarded."""
    from utils.keyless_api_access import request_was_admitted_keyless

    recorded = request_was_admitted_keyless(request)
    return _request_would_use_keyless(request) if recorded is None else recorded


def admitted_without_session(request: Any) -> bool:
    """True when keyless API access lets this request through with no Unsloth sign-in. The single
    predicate behind both the auth dependency below and the route-level checks that ask whether a
    caller is the Unsloth UI or a programmatic client."""
    from utils.keyless_api_access import request_was_admitted_keyless

    recorded = request_was_admitted_keyless(request)
    return _request_would_use_keyless(request) if recorded is None else recorded


class _BearerOrKeyless(HTTPBearer):
    """Read ``Authorization: Bearer <token>``, admitting a caller without one. When the setting is off this
    behaves exactly like ``HTTPBearer``, errors included.
    """

    async def __call__(self, request: Request) -> Optional[HTTPAuthorizationCredentials]:
        from utils.keyless_api_access import (
            APPROVED_DUMMY_BEARERS,
            is_empty_bearer,
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
        if (not authorization or is_empty_bearer(header)) and eligible:
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


# scheme_name pinned so the OpenAPI securitySchemes entry keeps its published name
security = _BearerOrKeyless(scheme_name = "HTTPBearer")


def _bind_owner() -> None:
    bind_account(OWNER)


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
    """Create a signed JWT for the given subject (e.g. username). Valid across restarts: the signing
    secret is stored in SQLite. Callers that already verified a credential pass ``secret`` so a
    rotation landing mid-request cannot sign the token with the credential that just replaced it."""
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

    if payload.get("sub") != subject or payload.get("desktop") is not True:
        return False
    account = get_account(subject)
    return account is not None and account.is_owner


def create_refresh_token(
    subject: str,
    *,
    desktop: bool = False,
    secret: Optional[str] = None,
) -> str:
    """``secret`` binds the token to the verified credential version, so a rotation revokes it."""
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
    """Validate a refresh token and issue a new access token. The refresh token is NOT consumed; it stays valid
    until expiry. Returns a new access_token, or None if the refresh token is invalid/expired.
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
    """Legacy API compat for callers expecting auth storage init. Auth now resolves the current signing
    secret directly from SQLite."""
    load_jwt_secret()


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
    """As get_current_subject, but also returns the credential generation, for routes that persist a
    new credential and must not do so on behalf of one a concurrent reset has revoked."""
    return await _get_current_credential(
        credentials,
        allow_password_change = False,
    )


async def authenticated_via_api_key(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> bool:
    """True for sk-unsloth keys and keyless callers, so every API-key guard also covers keyless access."""
    if is_keyless(credentials):
        return True
    return bool(credentials and credentials.credentials.startswith(API_KEY_PREFIX))


async def credentials_for_token(
    request: Any, token: Optional[str]
) -> Optional[HTTPAuthorizationCredentials]:
    """For a bearer the route reads itself (``?token=`` for img src), so keyless access still applies."""
    from utils.keyless_api_access import APPROVED_DUMMY_BEARERS, keyless_request_allowed

    if token is not None and not token.strip():
        token = None
    # Settings/listener reads hit SQLite and DNS; keep them off the event loop.
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


async def subject_for_header_or_query_token(request: Any, token: Optional[str]) -> str:
    """The subject of the bearer in ``Authorization``, or failing that in a ``?token=`` the route
    read for itself. An ``<img src>`` and the native save command fetch without a header."""
    header = request.headers.get("authorization") or ""
    header_token = header[7:] if header.lower().startswith("bearer ") else ""
    # A blank header counts as absent, so a ?token= is still honored.
    credentials = await credentials_for_token(request, header_token.strip() or token or None)
    if credentials is None:
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail = "Missing authentication token",
        )
    return await get_current_subject(credentials)


async def authenticated_without_credential(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> bool:
    """Dependency form of ``admitted_without_credential``."""
    return admitted_without_credential(credentials)


def require_ui_session_for_local_commands(via_api_key: bool) -> None:
    """stdio MCP runs a command on the host outside the sandbox, so only a UI session may define one."""
    if via_api_key:
        raise HTTPException(
            status_code = status.HTTP_403_FORBIDDEN,
            detail = "Local (stdio) MCP servers can only be configured from the Unsloth UI, "
            "not with an API key. Use an http:// or https:// MCP server instead.",
        )


async def allow_ambient_hf_token(via_api_key: bool = Depends(authenticated_via_api_key)) -> bool:
    """API keys get no ambient HF_TOKEN fallback, so they must send their own in X-Unsloth-HF-Token."""
    return not via_api_key


async def authenticated_via_desktop_jwt(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> bool:
    """True when the caller is the local desktop app, not a browser session or API key. Lets routes treat the
    desktop as an authority of its own: it authenticates with a local secret rather than the account password.
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


def _admin_credential() -> Tuple[str, Optional[str]]:
    """Resolve the local admin for a keyless caller, without the UI password gate."""
    record = get_user_and_secret(DEFAULT_ADMIN_USERNAME)
    if record is None:
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail = "Invalid or expired token",
        )
    _salt, _pwd_hash, jwt_secret, _must_change_password = record
    return DEFAULT_ADMIN_USERNAME, credential_generation(jwt_secret)


async def _get_current_credential(
    credentials: HTTPAuthorizationCredentials, *, allow_password_change: bool
) -> Tuple[str, Optional[str]]:
    """Routes that persist credentials must bind to this generation, or a reset mid-request blesses them."""
    if credentials.scheme == KEYLESS_SCHEME:
        _bind_owner()
        return await run_in_threadpool(_admin_credential)

    if credentials.scheme == KEYLESS_FALLBACK_SCHEME:
        from utils.keyless_api_access import APPROVED_DUMMY_BEARERS

        if credentials.credentials not in APPROVED_DUMMY_BEARERS:
            raise HTTPException(
                status_code = status.HTTP_401_UNAUTHORIZED,
                detail = "Invalid authentication credentials",
            )
        _bind_owner()
        return await run_in_threadpool(_admin_credential)

    token = credentials.credentials

    if token.startswith(API_KEY_PREFIX):
        verified = await run_in_threadpool(validate_api_key_account, token)
        if verified is None:
            raise HTTPException(
                status_code = status.HTTP_401_UNAUTHORIZED,
                detail = _invalid_api_key_detail(token),
            )
        record, secret = verified
        # A second lookup by username would bind whoever owns that name after a delete and re-create.
        bind_account(AccountContext(record["account_id"], record["username"], record["role"]))
        return record["username"], credential_generation(secret)

    subject = _decode_subject_without_verification(token)
    if subject is None:
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail = "Invalid token payload",
        )

    record = await run_in_threadpool(get_user_record, subject)
    if record is None or not record.get("is_active", 1):
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail = "Invalid or expired token",
        )

    jwt_secret = record["jwt_secret"]
    must_change_password = bool(record["must_change_password"])
    try:
        payload = jwt.decode(token, jwt_secret, algorithms = [ALGORITHM])
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail = "Invalid or expired token",
        )
    if payload.get("sub") != subject:
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail = "Invalid token payload",
        )
    bind_account(AccountContext(record["account_id"], record["username"], record["role"]))
    is_desktop = payload.get("desktop") is True and record.get("role") == "owner"
    if must_change_password and not allow_password_change and not is_desktop:
        raise HTTPException(
            status_code = status.HTTP_403_FORBIDDEN,
            detail = "Password change required",
        )
    return subject, credential_generation(jwt_secret)
