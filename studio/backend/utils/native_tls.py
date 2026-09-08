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

truststore is vendored at ``backend/vendor/`` rather than depended on, so no
Unsloth user gains a package for a proxy they do not have; see the README there.
Every consumer appends that directory to ``sys.path`` and imports the top-level
name, which keeps a truststore the user installed themselves in front of ours.

On by default for macOS and Windows, and on Linux only for the desktop app's own
backend (#9218); a headless ``unsloth studio`` keeps the
``UNSLOTH_STUDIO_NATIVE_TLS=1`` opt-in, since distro OpenSSL configurations vary
and an operator in a shell can export it. ``0`` opts out anywhere.

``SSL_CERT_FILE``/``REQUESTS_CA_BUNDLE`` stay exclusive, not additive: httpx
builds its context from ``SSL_CERT_FILE`` alone, so pointing it at a private CA
still costs you the public roots and the Hub with them. Install the CA in the OS
store instead, which is what this module then reaches.

Client side only: the injected class verifies a peer chain on every handshake,
so an ``SSLContext`` built after activation cannot serve TLS. Unsloth serves
plain HTTP on loopback and ``test_native_tls_entrypoints.py`` keeps it that way;
a future in-process HTTPS listener needs ``truststore.SSLContext`` for outbound
connections instead of this process-wide injection.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

_NATIVE_TLS_ENV = "UNSLOTH_STUDIO_NATIVE_TLS"
_DESKTOP_OWNER_KIND_ENV = "UNSLOTH_STUDIO_DESKTOP_OWNER_KIND"
_DEFAULT_ON_PLATFORMS = ("darwin", "win32")
_TRUTHY = ("1", "true", "yes")
_FALSEY = ("0", "false", "no")

# Resolved from this file so it is right in a checkout and an installed wheel alike; never built
# from the cwd or a hardcoded "studio/backend".
_VENDOR_DIR = str(Path(__file__).resolve().parent.parent / "vendor")

_logger = logging.getLogger(__name__)
_activated = False


def native_tls_enabled() -> bool:
    """Resolve ``UNSLOTH_STUDIO_NATIVE_TLS`` against the platform default.

    Linux is on only for the desktop's own backend: a .deb/AppImage launched
    from an icon reads no shell profile, so the opt-in is unreachable there
    (#9218), while a headless server has an operator who can export it.
    """
    flag = os.environ.get(_NATIVE_TLS_ENV, "").strip().lower()
    if flag in _TRUTHY:
        return True
    if flag in _FALSEY:
        return False
    if sys.platform in _DEFAULT_ON_PLATFORMS:
        return True
    if sys.platform.startswith("linux"):
        return _desktop_owned_process()
    return False


def _desktop_owned_process() -> bool:
    """True when this backend belongs to the Tauri desktop app.

    Read, never popped: main._load_desktop_owner owns this marker and pops it,
    but activation runs first (main.py:185, ahead of the loader), so reading it
    here cannot steal it.
    """
    return os.environ.get(_DESKTOP_OWNER_KIND_ENV, "") == "tauri"


def _uv_system_certs_wanted() -> bool:
    """Whether uv should move onto the OS store along with this process.

    True for an explicit opt-in and for the platforms install.sh already covers,
    False when only the desktop-owner default turned native TLS on. truststore
    ADDS the OS anchors to the ones a caller loaded, but uv's system certs
    REPLACE its bundled webpki roots, so an unusable SSL_CERT_FILE that uv
    happily ignores today becomes "No CA certificates were loaded from the
    system" instead, and core/training/worker.py runs uv with no pip fallback.
    """
    if os.environ.get(_NATIVE_TLS_ENV, "").strip().lower() in _TRUTHY:
        return True
    return sys.platform in _DEFAULT_ON_PLATFORMS


# Children that cannot import this module carry the gate as source, generated from the same constants so it cannot drift
# from native_tls_enabled(). The Linux desktop-owner clause also applies to children launched directly by the desktop.
# The children that cannot import it are the `python -c` probes and prebuilt_core.py, and each supplies os, sys and
# _TRUSTSTORE_VENDOR itself.
_INLINE_GATE = """\
_flag = os.environ.get({env!r}, '').strip().lower()
_owned = os.environ.get({owner_env!r}, '') == 'tauri'
if _flag in {truthy!r} or (
    _flag not in {falsey!r}
    and (sys.platform in {platforms!r} or (sys.platform.startswith('linux') and _owned))
):
    try:
        if _TRUSTSTORE_VENDOR not in sys.path:
            sys.path.append(_TRUSTSTORE_VENDOR)
        import truststore
        truststore.inject_into_ssl()
    except Exception:
        pass
del _flag, _owned
"""


def vendor_dir() -> str:
    """Where the vendored truststore lives, for a child that must be told."""
    return _VENDOR_DIR


def inline_gate_source() -> str:
    """The gate as executable source, for a child that cannot import this module.

    The child must bind ``_TRUSTSTORE_VENDOR`` to the vendor directory first.
    """
    return _INLINE_GATE.format(
        env = _NATIVE_TLS_ENV,
        owner_env = _DESKTOP_OWNER_KIND_ENV,
        truthy = _TRUTHY,
        falsey = _FALSEY,
        platforms = _DEFAULT_ON_PLATFORMS,
    )


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
    # main.py pops the desktop-owner marker, so children re-resolve from the flag
    # alone: spell the decision back into the env, the way the UV_* pair below is.
    # Assign, not setdefault: an opt-out already returned above, so the only value
    # left to preserve would be an unrecognized one, which reads as off in a child.
    uv_default = "1" if _uv_system_certs_wanted() else "0"
    os.environ[_NATIVE_TLS_ENV] = "1"
    # uv's rustls ignores in-process injection (uv >= 0.11 reads UV_SYSTEM_CERTS, older reads UV_NATIVE_TLS). Mirror one
    # value across both: uv takes either as an opt-in, so an opt-out in one spelling must carry to the other.
    # Written even when the answer is "0", because a worker sees the normalized flag above and not the desktop marker,
    # so an absent value would have it derive an opt-in this process declined.
    os.environ.setdefault("UV_SYSTEM_CERTS", os.environ.get("UV_NATIVE_TLS", uv_default))
    os.environ.setdefault("UV_NATIVE_TLS", os.environ["UV_SYSTEM_CERTS"])
    # append, not insert(0): a user-installed truststore must win over the vendored copy.
    if _VENDOR_DIR not in sys.path:
        sys.path.append(_VENDOR_DIR)
    try:
        import truststore
        truststore.inject_into_ssl()
    except Exception as exc:  # noqa: BLE001
        # Warn, no traceback: a silent certifi fallback is what this exists to prevent.
        _logger.warning("native TLS unavailable (%s); TLS keeps certifi defaults", exc)
        return False
    _activated = True
    return True
