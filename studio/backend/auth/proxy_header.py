# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Sign-in from a forward-auth reverse proxy (Authelia, Authentik, oauth2-proxy, ...).

Opt-in with ``UNSLOTH_STUDIO_TRUSTED_USER_HEADER`` (for example ``Remote-User``). The header
is read only on a request that ``request_via_trusted_proxy`` admits: it landed on a loopback
socket and names a host in ``UNSLOTH_STUDIO_PROXY_HOSTS``. A LAN or tunnel client talking to
Studio directly never reaches that check. ``UNSLOTH_STUDIO_TRUSTED_PROXY_SECRET`` additionally
requires the proxy to send ``X-Unsloth-Proxy-Secret``, so other local processes cannot sign in
by setting the header themselves. The header value must name an existing, active Studio account
(its username, or the owner's sign-in name); no account is ever created from it.
"""

from __future__ import annotations

import hmac
import os
import re
from typing import NamedTuple, Optional

from utils.client_ip import request_via_trusted_proxy

USER_HEADER_ENV = "UNSLOTH_STUDIO_TRUSTED_USER_HEADER"
PROXY_SECRET_ENV = "UNSLOTH_STUDIO_TRUSTED_PROXY_SECRET"
PROXY_SECRET_HEADER = "x-unsloth-proxy-secret"

_HEADER_NAME = re.compile(r"[a-z0-9][a-z0-9-]{0,63}")
# Headers Studio or the proxy chain give another meaning; never an identity.
_REFUSED_HEADERS = frozenset(
    {
        "authorization",
        "cookie",
        "host",
        "x-api-key",
        "forwarded",
        "x-forwarded-for",
        "x-forwarded-host",
        "x-forwarded-proto",
        "x-real-ip",
        "cf-connecting-ip",
        PROXY_SECRET_HEADER,
    }
)
_MAX_VALUE_LENGTH = 64


class ProxyAccount(NamedTuple):
    username: str
    jwt_secret: str


def trusted_user_header() -> Optional[str]:
    """The configured header name, lowercased, or None when the feature is off or misconfigured."""
    name = os.environ.get(USER_HEADER_ENV, "").strip().lower()
    if not name or not _HEADER_NAME.fullmatch(name) or name in _REFUSED_HEADERS:
        return None
    return name


def _single_header(scope, name: bytes) -> Optional[str]:
    values = [value for key, value in scope.get("headers") or () if bytes(key).lower() == name]
    if len(values) != 1:
        return None
    try:
        return bytes(values[0]).decode("latin-1").strip()
    except UnicodeDecodeError:
        return None


def _proxy_secret_matches(scope) -> bool:
    expected = os.environ.get(PROXY_SECRET_ENV, "")
    if not expected:
        return True
    provided = _single_header(scope, PROXY_SECRET_HEADER.encode("ascii"))
    if provided is None:
        return False
    return hmac.compare_digest(provided.encode("latin-1"), expected.strip().encode("utf-8"))


def proxy_header_value(scope) -> Optional[str]:
    """The proxy-asserted user name, or None unless every trust condition holds."""
    header = trusted_user_header()
    if header is None or not isinstance(scope, dict):
        return None
    if not request_via_trusted_proxy(scope) or not _proxy_secret_matches(scope):
        return None
    value = _single_header(scope, header.encode("ascii"))
    if not value or len(value) > _MAX_VALUE_LENGTH:
        return None
    return value


def proxy_account(scope) -> Optional[ProxyAccount]:
    """The active account the trusted proxy names, ready to sign in, or None."""
    value = proxy_header_value(scope)
    if value is None:
        return None
    from auth import storage

    username = storage.resolve_login_username(value.casefold())
    record = storage.get_user_record(username)
    if record is None or not record.get("is_active", True):
        return None
    # A pending setup code or default password is finished at the password form, not skipped.
    if record.get("must_change_password"):
        return None
    secret = record.get("jwt_secret")
    if not secret:
        return None
    return ProxyAccount(record["username"], secret)
