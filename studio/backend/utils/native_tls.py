# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Verify TLS against the OS trust store (corporate TLS-inspection proxies).

Python's ``ssl`` verifies against certifi's bundled Mozilla roots and ignores
the OS trust store. Behind a TLS-inspecting proxy (Cisco Umbrella, Zscaler,
Netskope, ...) every huggingface.co request then fails with
``CERTIFICATE_VERIFY_FAILED``, because the proxy re-signs traffic with a
corporate CA that lives only in the OS store. A shell user can export
``SSL_CERT_FILE``, but GUI launches (macOS ``.app``, desktop shortcuts) never
read shell profiles, so the backend they spawn has no way to pick that up.

``truststore.inject_into_ssl()`` makes ``ssl.SSLContext`` verify against the
OS store (macOS Security framework, Windows CertStore, OpenSSL dirs on Linux)
— the runtime counterpart of ``UV_NATIVE_TLS`` in install.sh, and the same
mechanism pip enables by default since 24.2. Injection is process-wide but
does not survive into spawned interpreters, so each network-touching entry
point (main.py, download and training workers) calls
:func:`activate_native_tls` before its first TLS connection.

Defaults mirror install.sh: on for macOS and Windows (well-defined OS stores,
and the fleets where TLS inspection is common), opt-in on Linux via
``UNSLOTH_STUDIO_NATIVE_TLS=1`` (distro OpenSSL configurations vary), opt-out
anywhere with ``0``. Explicit ``SSL_CERT_FILE``/``REQUESTS_CA_BUNDLE``
overrides keep working: clients pass them as ``verify=...``/``cafile``, which
``load_verify_locations`` honours on the injected context too.
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
    """Resolve the ``UNSLOTH_STUDIO_NATIVE_TLS`` tri-state against the platform default."""
    flag = os.environ.get(_NATIVE_TLS_ENV, "").strip().lower()
    if flag in {"1", "true", "yes"}:
        return True
    if flag in {"0", "false", "no"}:
        return False
    return sys.platform in _DEFAULT_ON_PLATFORMS


def activate_native_tls() -> bool:
    """Idempotently patch ``ssl`` to verify against the OS trust store.

    Returns True when injection is active in this process. Failure is
    non-fatal: falling back to certifi is the pre-existing behaviour and the
    strictly less permissive direction.
    """
    global _activated
    if _activated:
        return True
    if not native_tls_enabled():
        return False
    try:
        import truststore
        truststore.inject_into_ssl()
    except Exception:
        _logger.debug("truststore injection failed; keeping certifi defaults", exc_info = True)
        return False
    _activated = True
    return True
