# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Opt-in frontend access through an operator-managed local HTTPS reverse proxy.

No identity is inferred from proxy headers. Authentication remains the API's job.
"""

import ipaddress
import os
import re
from urllib.parse import urlsplit


PROXY_ORIGIN_ENV = "UNSLOTH_STUDIO_PROXY_ORIGIN"


def local_proxy_configured() -> bool:
    # Treat even an invalid nonempty setting as remote exposure for trust policy.
    # A typo must not re-enable loopback-only authentication/tool shortcuts.
    return bool(os.environ.get(PROXY_ORIGIN_ENV, "").strip())


def _https_origin(value: str) -> tuple[str, int] | None:
    """Strict serialized HTTPS origin, without userinfo, paths or URL suffixes."""
    if not value or any(c.isspace() or ord(c) < 32 for c in value):
        return None
    if any(c in value for c in ("@", "\\", "?", "#", "%")):
        return None
    try:
        parsed = urlsplit(value)
        host, port = parsed.hostname, parsed.port
    except ValueError:
        return None
    if parsed.scheme != "https" or parsed.path or not host or parsed.netloc.endswith(":"):
        return None
    if ":" in host:
        try:
            host = str(ipaddress.IPv6Address(host))
        except ValueError:
            return None
    elif not re.fullmatch(r"[a-z0-9]+(?:[a-z0-9.-]*[a-z0-9])?", host):
        return None
    if port is not None and not 1 <= port <= 65535:
        return None
    return host, 443 if port is None else port


def validate_local_proxy_origin() -> tuple[str, int] | None:
    value = os.environ.get(PROXY_ORIGIN_ENV, "").strip()
    if not value:
        return None
    origin = _https_origin(value)
    if origin is None:
        raise ValueError(
            f"{PROXY_ORIGIN_ENV} must be one HTTPS origin, for example "
            "https://studio.example.ts.net (no path, query, fragment or credentials)"
        )
    return origin


def local_proxy_frontend_request(scope) -> bool:
    """Admit the configured authority only on an actual loopback listener.

    Uvicorn may have rewritten scope['client'] from X-Forwarded-For already;
    scope['server'] still describes the accepting socket. The proxy must preserve
    Host and overwrite X-Forwarded-Host/Proto. It is responsible for restricting
    access to its public side; these headers do not authenticate a caller.
    """
    try:
        expected = validate_local_proxy_origin()
    except ValueError:
        return False
    if expected is None or scope.get("type") != "http":
        return False
    server = scope.get("server")
    if not server or not isinstance(server[0], str) or "%" in server[0]:
        return False
    try:
        address = ipaddress.ip_address(server[0])
    except ValueError:
        return False
    address = getattr(address, "ipv4_mapped", None) or address
    if not address.is_loopback:
        return False
    headers = {}
    relevant = {b"host", b"x-forwarded-host", b"x-forwarded-proto", b"origin"}
    for name, value in scope.get("headers", ()):
        if name in relevant:
            if name in headers:
                return False
            headers[name] = value.decode("latin-1")
    for name in (b"host", b"x-forwarded-host"):
        if _https_origin("https://" + headers.get(name, "")) != expected:
            return False
    if headers.get(b"x-forwarded-proto") != "https":
        return False
    origin = headers.get(b"origin")
    return origin is None or _https_origin(origin) == expected
