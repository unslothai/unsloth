# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Verify TLS against the OS trust store (corporate TLS-inspection proxies).

Python's ``ssl`` trusts only certifi's roots, so behind a TLS-inspecting proxy
(Cisco Umbrella, Zscaler, Netskope) every huggingface.co request fails with
``CERTIFICATE_VERIFY_FAILED``: the proxy re-signs traffic with a corporate CA
that lives only in the OS store. A shell user can export ``SSL_CERT_FILE``, but
GUI launches (macOS ``.app``, desktop shortcuts) never read shell profiles.

``truststore.inject_into_ssl()`` makes ``ssl.SSLContext`` verify against the OS
store instead, the runtime counterpart of ``UV_NATIVE_TLS`` in install.sh.
Injection is process-wide but does not survive a spawn, so every
network-touching entry point calls :func:`activate_native_tls` before its first
TLS connection; the ``python -c`` probes and the standalone prebuilt installers
carry an inline copy of the gating because they cannot import backend modules.

Defaults mirror install.sh: on for macOS and Windows, opt-in on Linux via
``UNSLOTH_STUDIO_NATIVE_TLS=1`` (distro OpenSSL configurations vary), opt-out
anywhere with ``0``. Explicit ``SSL_CERT_FILE``/``REQUESTS_CA_BUNDLE`` keep
working, but become additive rather than exclusive, since truststore keeps the
OS anchors alongside them; ``0`` is the way back to a bundle being the only
trust root.

Client side only: the injected class verifies a peer chain on every handshake,
so an ``SSLContext`` built after activation cannot serve TLS. Studio serves
plain HTTP on loopback and ``test_native_tls_entrypoints.py`` keeps it that way;
a future in-process HTTPS listener needs ``truststore.SSLContext`` for outbound
connections instead of this process-wide injection.
"""

from __future__ import annotations

import logging
import os
import sys

_NATIVE_TLS_ENV = "UNSLOTH_STUDIO_NATIVE_TLS"
_DEFAULT_ON_PLATFORMS = ("darwin", "win32")

_logger = logging.getLogger(__name__)
_activated = False


def native_tls_enabled() -> bool:
    """Resolve ``UNSLOTH_STUDIO_NATIVE_TLS`` against the platform default."""
    flag = os.environ.get(_NATIVE_TLS_ENV, "").strip().lower()
    if flag in {"1", "true", "yes"}:
        return True
    if flag in {"0", "false", "no"}:
        return False
    return sys.platform in _DEFAULT_ON_PLATFORMS


def activate_native_tls() -> bool:
    """Idempotently patch ``ssl`` to verify against the OS trust store.

    Returns True when injection is active. Failure is non-fatal: falling back to
    certifi is the pre-existing, strictly less permissive behaviour.
    """
    global _activated
    if _activated:
        return True
    if not native_tls_enabled():
        return False
    # uv child installers do their own TLS: rustls ignores in-process injection,
    # so point them at the OS store too (uv >= 0.11 reads UV_SYSTEM_CERTS, older
    # reads UV_NATIVE_TLS). Mirror one resolved value across both rather than
    # defaulting each to "1": uv takes either var as an opt-in, so an opt-out in
    # one spelling has to carry to the other or the unset name re-enables it.
    os.environ.setdefault("UV_SYSTEM_CERTS", os.environ.get("UV_NATIVE_TLS", "1"))
    os.environ.setdefault("UV_NATIVE_TLS", os.environ["UV_SYSTEM_CERTS"])
    try:
        import truststore
        truststore.inject_into_ssl()
    except Exception as exc:  # noqa: BLE001
        # Warn loudly: a silent certifi fallback is what this exists to prevent.
        # One line, no traceback -- truststore is simply absent on Python 3.9.
        _logger.warning("native TLS unavailable (%s); TLS keeps certifi defaults", exc)
        return False
    _activated = True
    return True
