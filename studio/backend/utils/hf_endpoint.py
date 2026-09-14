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

import logging
import os
from urllib.parse import urlsplit

from utils.utils import hf_endpoint_url

logger = logging.getLogger(__name__)

_DEFAULT_HF_ENDPOINT = "https://huggingface.co"
_DEFAULT_DATASETS_SERVER = "https://datasets-server.huggingface.co"

_ds_mirror_warned = False
# Values already reported as unusable, so a per-request caller (the CSP builder runs
# on every response) logs each bad configuration once rather than per request.
_rejected_warned: set[str] = set()

# A CSP source list is whitespace-separated and semicolon-delimited, so any of these
# inside an endpoint would add sources or whole directives rather than one origin.
_FORBIDDEN_CHARS = frozenset(" \t\r\n\f\v;,'\"\\")


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
    else:
        parts = urlsplit(candidate)
        if parts.scheme not in ("http", "https"):
            reason = "is not an http(s) URL"
        elif not parts.hostname:
            reason = "has no host"
        elif parts.username or parts.password:
            reason = "carries credentials"
        elif parts.query or parts.fragment:
            reason = "carries a query string or fragment"
        elif not _port_is_valid(parts):
            # A scheme-less value keeps everything before the first ':' as the host,
            # so "javascript:alert(1)" becomes "https://javascript:alert(1)" -- a
            # syntactically fine URL whose port is nonsense.
            reason = "has an invalid port"
        elif parts.netloc.endswith(":"):
            # "host:" parses with port None, but is not a valid CSP host-source.
            reason = "has an empty port"
        elif "*" in parts.netloc:
            # HF_ENDPOINT="*" becomes "https://*", which as a CSP source allows
            # EVERY https origin -- the opposite of what the policy is for. A
            # wildcard is never a usable endpoint to send requests to either.
            reason = "contains a wildcard host"
        else:
            return candidate
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
