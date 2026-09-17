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

# Reported when the configured value is not reachable by this client.
DEFAULTS_BY_HEALTH_KEY = {
    "hf_endpoint": _DEFAULT_HF_ENDPOINT,
    "hf_datasets_server": _DEFAULT_DATASETS_SERVER,
}

_ds_mirror_warned = False
# The CSP builder runs on every response, so each bad value is logged once.
_rejected_warned: set[str] = set()
# client_reachable_endpoint() runs per request too, so each configured endpoint
# logs its per-client fallback once instead of on every /api/health call.
_unreachable_warned: set[str] = set()

# A source list is whitespace-separated and semicolon-delimited: any of these in an
# endpoint would add sources or whole directives rather than one origin.
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
    """Return ``candidate`` when it is a plain http(s) origin, else ``default``.

    These values are interpolated into request URLs *and* into the
    ``connect-src`` directive of the Content-Security-Policy header, so a value
    carrying whitespace, a semicolon or a control character would inject extra
    CSP sources or directives. Operators set these env vars themselves, so this
    is a configuration guard rather than a defence against hostile input, but a
    silently broken policy is the worst way to find that out.
    """
    if not candidate:
        return default
    if any(ch in _FORBIDDEN_CHARS for ch in candidate) or any(
        ord(ch) < 0x20 or ord(ch) == 0x7F for ch in candidate
    ):
        reason = "contains whitespace, a separator or a control character"
    elif not candidate.isascii():
        # Starlette headers are latin-1, so an IDN host in the CSP 500s every
        # response, and the Tauri builder has no IDNA encoder either.
        reason = "contains non-ASCII characters; use the punycode (xn--) form of the host"
    elif (parts := _split(candidate)) is None:
        # urlsplit RAISES on "https://[", and _build_csp runs on every response.
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
            # "javascript:alert(1)" becomes a fine URL with a nonsense port.
            reason = "has an invalid port"
        elif parts.netloc.endswith(":"):
            reason = "has an empty port"
        elif "*" in parts.netloc:
            reason = "contains a wildcard host"
        elif parts.scheme == "http" and not is_loopback_host(parts.hostname):
            # Hub calls carry the user's token, so http off-box puts it on the wire.
            reason = (
                "is plain HTTP to a non-loopback host, which would put the Hub token on the wire"
            )
        else:
            # Folded: RFC 3986 3.1, and the frontend keys its cache on this string.
            return _canonical(parts, parts.scheme + candidate[len(parts.scheme) :])
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


def is_private_host(hostname: str | None) -> bool:
    """Is this an address literal that is only meaningful on some local network?

    RFC 1918 and friends, plus link-local and unique-local IPv6. Reserved and
    documentation ranges count too, which is the conservative direction: an
    address that is not routable on the internet is one whose meaning depends on
    where you stand. A NAME is not private by this test even if it resolves to
    such an address: there is nothing here to resolve it with, and a name at
    least means the same thing to both ends when their DNS agrees.
    """
    if not hostname:
        return False
    try:
        address = ipaddress.ip_address(hostname.strip("[]").lower())
    except ValueError:
        return False
    return address.is_private or address.is_link_local


def endpoint_is_reachable_by(endpoint: str, client_host: str | None) -> bool:
    """Would a browser at ``client_host`` reach the SAME host this endpoint names?

    A loopback endpoint names a proxy on the machine the BACKEND runs on, so it
    means the browser's own localhost anywhere else. A private-network address
    has the same problem one step out: through the managed tunnel, or from the
    internet, ``https://10.0.0.5:8443`` is an address on the VISITOR's network,
    where it is either dead or some unrelated service that would be offered the
    user's Hub token. A private address is therefore reported only to a client
    that is itself local, which keeps the ordinary LAN deployment working, and a
    loopback one only to a loopback client.
    """
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
    """The hub endpoint to hand to a browser at ``client_host``.

    Everything the backend does itself keeps using ``get_hf_endpoint()``; this is
    only for values that leave for a browser (``/api/health``, the publish link).
    A remote client that cannot reach the configured endpoint gets the official
    one instead, and the fallback is logged once per configured endpoint so the
    operator can see why a tunnelled browser is not using the mirror.
    """
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
    """Compress an IPv6 literal host, leaving everything else untouched.

    ``http://[0:0:0:0:0:0:0:1]`` and ``http://[::1]`` are the same host, but a
    CSP host-source is matched as a string (CSP3 6.7.2.5), and the browser sends
    the compressed form: the uncompressed source would not match its own request
    and the policy would block the very mirror it names.
    """
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
    """Make HF_ENDPOINT mean the same thing to huggingface_hub as it does here.

    The library reads the variable itself, at import, with no normalisation and
    no validation. A scheme-less ``hf-mirror.com`` therefore reaches it verbatim
    and every ``HfApi`` / ``snapshot_download`` call fails on a missing scheme
    while Studio's own requests work, and a value this module REJECTS -- a
    plain-HTTP mirror off the machine, say -- would still be handed the user's
    Hub token by the library while Studio itself fell back to huggingface.co.
    Rewriting the variable before huggingface_hub is imported gives the whole
    process, and the subprocesses that inherit this environment, one endpoint.
    """
    raw = os.environ.get("HF_ENDPOINT")
    if raw is None:
        return
    if not raw.strip():
        # Blank is not an endpoint, but the library would read it verbatim;
        # clear it so Studio and huggingface_hub agree on the default.
        os.environ.pop("HF_ENDPOINT", None)
        return
    endpoint = get_hf_endpoint()
    if endpoint == _DEFAULT_HF_ENDPOINT:
        # Rejected, or the official host: either way, unset asks for the default.
        os.environ.pop("HF_ENDPOINT", None)
    else:
        os.environ["HF_ENDPOINT"] = endpoint


def csp_connect_sources() -> tuple[str, str]:
    """The two endpoints as CSP ``connect-src`` sources, i.e. origins only.

    A CSP host-source carrying a path is matched *exactly* unless the path ends
    in a solidus (CSP3 6.7.2.7), so listing a path-prefixed mirror verbatim --
    ``https://hub.internal/hf`` -- allows exactly that one URL and blocks every
    ``/hf/api/models`` request under it, in Chrome, Edge, Firefox and Safari
    alike. The path belongs in the request URL, not in the policy, so the source
    is reduced to scheme://host[:port].
    """
    return (_origin_of(get_hf_endpoint()), _origin_of(get_hf_datasets_server()))


def csp_asset_sources() -> tuple[str, ...]:
    """Configured origins that ``img-src``/``media-src`` do not already cover.

    Those directives carry a bare ``https:``, so an https mirror needs nothing
    added. A loopback HTTP mirror does: its avatars (hf-owner-avatar.ts) and
    README images (hf-readme.ts) are same-origin-relative to the endpoint, and
    without this they are blocked while the API calls beside them succeed.

    Returning only the http origins is what keeps an unconfigured deployment's
    policy byte-identical to the pre-PR one.
    """
    return tuple(
        dict.fromkeys(source for source in csp_connect_sources() if source.startswith("http://"))
    )


def _origin_of(endpoint: str) -> str:
    parts = urlsplit(endpoint)
    return f"{parts.scheme}://{parts.netloc}" if parts.netloc else endpoint


def get_hf_endpoint() -> str:
    """Return the configured HuggingFace hub endpoint (no trailing slash).

    Wraps :func:`utils.utils.hf_endpoint_url` so callers get a value that is
    safe for ``f"{endpoint}/path"`` concatenation.
    """
    return _sanitize(hf_endpoint_url().rstrip("/"), _DEFAULT_HF_ENDPOINT, "HF_ENDPOINT")


def get_hf_datasets_server() -> str:
    """Return the datasets-server base URL (no trailing slash).

    Returns ``HF_DATASETS_SERVER`` when set, otherwise the official
    ``datasets-server.huggingface.co``.  A mirrored ``HF_ENDPOINT`` does
    **not** implicitly apply here — Hub mirrors rarely proxy the
    datasets-server API, so operators must opt in explicitly.
    """
    raw = (os.environ.get("HF_DATASETS_SERVER") or "").strip()
    if raw:
        endpoint = raw if "://" in raw else "https://" + raw
        return _sanitize(endpoint.rstrip("/"), _DEFAULT_DATASETS_SERVER, "HF_DATASETS_SERVER")
    global _ds_mirror_warned
    if not _ds_mirror_warned and get_hf_endpoint() != _DEFAULT_HF_ENDPOINT:
        _ds_mirror_warned = True
        logger.warning(
            "HF_ENDPOINT is set to %s but HF_DATASETS_SERVER is unset; "
            "datasets-server calls will still go to %s. "
            "Set HF_DATASETS_SERVER to override.",
            get_hf_endpoint(),
            _DEFAULT_DATASETS_SERVER,
        )
    return _DEFAULT_DATASETS_SERVER
