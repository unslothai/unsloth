# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Centralised HuggingFace endpoint configuration.

Backend code that constructs HF URLs directly (i.e. *outside* of
``huggingface_hub`` calls) should use :func:`get_hf_endpoint` instead of
hard-coding ``https://huggingface.co``. The value is read through
:func:`utils.utils.hf_endpoint_url` — the single source of truth for
``HF_ENDPOINT`` parsing — so both entry points stay in lockstep.

The datasets-server base URL is independent from the Hub mirror: most Hub
mirrors do not proxy the datasets-server API, so a mirrored ``HF_ENDPOINT``
never implicitly redirects datasets-server traffic.  Operators who do run a
mirrored datasets-server must set ``HF_DATASETS_SERVER`` explicitly.
"""

from __future__ import annotations

import ipaddress
import logging
import os
from urllib.parse import urlsplit, urlunsplit

from utils.utils import hf_endpoint_url

logger = logging.getLogger(__name__)

_DEFAULT_HF_ENDPOINT = "https://huggingface.co"
_DEFAULT_DATASETS_SERVER = "https://datasets-server.huggingface.co"

DEFAULTS_BY_HEALTH_KEY = {
    "hf_endpoint": _DEFAULT_HF_ENDPOINT,
    "hf_datasets_server": _DEFAULT_DATASETS_SERVER,
}

_ds_mirror_warned = False
# The CSP builder runs on every response, so each bad value is logged once.
_rejected_warned: set[str] = set()
_unreachable_warned: set[str] = set()

# Any of these would add CSP sources or directives rather than one origin.
_FORBIDDEN_CHARS = frozenset(" \t\r\n\f\v;,'\"\\")


def _split(candidate: str):
    """``urlsplit`` that answers None instead of raising on a malformed host."""
    try:
        return urlsplit(candidate)
    except ValueError:
        return None


def is_loopback_host(hostname: str | None) -> bool:
    if not hostname:
        return False
    host = hostname.strip("[]").lower()
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _port_is_valid(parts) -> bool:
    """``SplitResult.port`` raises rather than returning None on a bad port."""
    try:
        parts.port
    except ValueError:
        return False
    return True


def _sanitize(candidate: str, default: str, var_name: str) -> str:
    """Plain http(s) origins only, since the value is copied into the CSP connect-src directive."""
    if not candidate:
        return default
    canonical, reason = _check(candidate)
    if canonical is not None:
        return canonical
    if candidate not in _rejected_warned:
        _rejected_warned.add(candidate)
        logger.warning(
            "%s=%r %s; ignoring it and using %s instead.",
            var_name,
            candidate,
            reason,
            default,
        )
    return default


def _check(candidate: str) -> tuple[str | None, str | None]:
    """``(canonical, None)`` for a usable endpoint, else ``(None, reason)``."""
    if any(ch in _FORBIDDEN_CHARS for ch in candidate) or any(
        ord(ch) < 0x20 or ord(ch) == 0x7F for ch in candidate
    ):
        reason = "contains whitespace, a separator or a control character"
    elif not candidate.isascii():
        # Starlette headers are latin-1, so an IDN host in the CSP would 500 every response.
        reason = "contains non-ASCII characters; use the punycode (xn--) form of the host"
    elif (parts := _split(candidate)) is None:
        # urlsplit raises on "https://[".
        reason = "is not a parseable URL"
    else:
        if parts.scheme not in ("http", "https"):
            reason = "is not an http(s) URL"
        elif not parts.hostname:
            reason = "has no host"
        elif parts.username or parts.password:
            reason = "carries credentials"
        elif parts.query or parts.fragment:
            reason = "carries a query string or fragment"
        elif not _port_is_valid(parts):
            reason = "has an invalid port"
        elif parts.netloc.endswith(":"):
            reason = "has an empty port"
        elif "*" in parts.netloc:
            reason = "contains a wildcard host"
        elif parts.scheme == "http" and not is_loopback_host(parts.hostname):
            # Hub calls carry the user's token, so http off-box exposes it.
            reason = (
                "is plain HTTP to a non-loopback host, which would put the Hub token on the wire"
            )
        else:
            # Scheme folded per RFC 3986 3.1; the frontend keys its cache on this string.
            return _canonical(parts, parts.scheme + candidate[len(parts.scheme) :]), None
    return None, reason


def validate_hub_endpoint(raw: str) -> str:
    """Same rules as the env values, but a rejected value raises ValueError rather than falling back."""
    value = raw.strip().rstrip("/")
    if not value:
        return ""
    if "://" not in value:
        value = "https://" + value
    canonical, reason = _check(value)
    if canonical is None:
        raise ValueError(f"The endpoint {reason}.")
    return canonical


def is_private_host(hostname: str | None) -> bool:
    """Local-network address literals only; a name is not private even if it resolves to one."""
    if not hostname:
        return False
    try:
        address = ipaddress.ip_address(hostname.strip("[]").lower())
    except ValueError:
        return False
    return address.is_private or address.is_link_local


def endpoint_is_reachable_by(endpoint: str, client_host: str | None) -> bool:
    """Loopback endpoints go only to loopback clients; private ones only to local clients, not remote."""
    parts = _split(endpoint)
    if parts is None:
        return True
    host = parts.hostname
    if is_loopback_host(host):
        return is_loopback_host(client_host)
    if is_private_host(host):
        return is_loopback_host(client_host) or is_private_host(client_host)
    return True


def client_reachable_endpoint(client_host: str | None) -> str:
    """Falls back to the official Hub, logged once, when the client cannot reach the configured endpoint."""
    endpoint = get_hf_endpoint()
    if endpoint_is_reachable_by(endpoint, client_host):
        return endpoint
    if endpoint not in _unreachable_warned:
        _unreachable_warned.add(endpoint)
        logger.warning(
            "HF_ENDPOINT %s is not reachable from this client (%s); serving %s to "
            "that browser instead. The backend itself keeps using the configured "
            "endpoint.",
            endpoint,
            client_host or "client address unknown",
            _DEFAULT_HF_ENDPOINT,
        )
    return _DEFAULT_HF_ENDPOINT


def _canonical(parts, folded: str) -> str:
    """Compresses IPv6 literals: CSP host-sources match as strings, and browsers send the compressed
    form."""
    host = parts.hostname
    if not host or ":" not in host:
        return folded
    try:
        compressed = ipaddress.ip_address(host).compressed
    except ValueError:
        return folded
    if compressed == host:
        return folded
    netloc = f"[{compressed}]"
    if parts.port is not None:
        netloc += f":{parts.port}"
    return urlunsplit((parts.scheme.lower(), netloc, parts.path, "", ""))


def normalize_hf_endpoint_env() -> None:
    """Normalizes HF_ENDPOINT before huggingface_hub imports, which reads it raw and unvalidated."""
    raw = os.environ.get("HF_ENDPOINT")
    if raw is None:
        return
    if not raw.strip():
        # Blank is not an endpoint but huggingface_hub would read it verbatim.
        os.environ.pop("HF_ENDPOINT", None)
        return
    endpoint = get_hf_endpoint()
    if endpoint == _DEFAULT_HF_ENDPOINT:
        os.environ.pop("HF_ENDPOINT", None)
    else:
        os.environ["HF_ENDPOINT"] = endpoint


def csp_connect_sources() -> tuple[str, ...]:
    """Origins only, since a CSP host-source with a path matches exactly; saved-only endpoints stay out."""
    from utils.hub_settings import saved_only_endpoints

    hidden = saved_only_endpoints()
    return tuple(
        _origin_of(endpoint)
        for endpoint in (browser_hf_endpoint(), get_hf_datasets_server())
        if endpoint not in hidden
    )


def csp_asset_sources() -> tuple[str, ...]:
    """Only http origins: img-src and media-src already allow https, so an https mirror needs nothing."""
    return tuple(
        dict.fromkeys(source for source in csp_connect_sources() if source.startswith("http://"))
    )


def _origin_of(endpoint: str) -> str:
    parts = urlsplit(endpoint)
    return f"{parts.scheme}://{parts.netloc}" if parts.netloc else endpoint


def get_hf_endpoint() -> str:
    """The configured hub endpoint, sanitized and without a trailing slash, for path concatenation."""
    return _sanitize(hf_endpoint_url().rstrip("/"), _DEFAULT_HF_ENDPOINT, "HF_ENDPOINT")


def browser_hf_endpoint() -> str:
    """The endpoint the browser uses. The ModelScope adapter's loopback listener is for
    this process only; the browser reaches ModelScope through its authenticated mount."""
    from utils.hub_settings import MODELSCOPE, active_source
    return _DEFAULT_HF_ENDPOINT if active_source() == MODELSCOPE else get_hf_endpoint()


def get_hf_datasets_server() -> str:
    """A mirrored HF_ENDPOINT does not apply here; the datasets server needs its own HF_DATASETS_SERVER."""
    raw = (os.environ.get("HF_DATASETS_SERVER") or "").strip()
    if raw:
        endpoint = raw if "://" in raw else "https://" + raw
        return _sanitize(endpoint.rstrip("/"), _DEFAULT_DATASETS_SERVER, "HF_DATASETS_SERVER")
    global _ds_mirror_warned
    if not _ds_mirror_warned and browser_hf_endpoint() != _DEFAULT_HF_ENDPOINT:
        _ds_mirror_warned = True
        logger.warning(
            "HF_ENDPOINT is set to %s but HF_DATASETS_SERVER is unset; "
            "datasets-server calls will still go to %s. "
            "Set HF_DATASETS_SERVER to override.",
            get_hf_endpoint(),
            _DEFAULT_DATASETS_SERVER,
        )
    return _DEFAULT_DATASETS_SERVER
