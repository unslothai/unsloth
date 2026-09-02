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

Defaults mirror install.sh: on for macOS and Windows, opt-in on Linux via
``UNSLOTH_STUDIO_NATIVE_TLS=1`` (distro OpenSSL configurations vary), opt-out
anywhere with ``0``. Explicit ``SSL_CERT_FILE``/``REQUESTS_CA_BUNDLE`` keep
working, but become additive rather than exclusive, since truststore keeps the
OS anchors alongside them; ``0`` is the way back to a bundle being the only
trust root.

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
# main.py's _load_desktop_owner handshake marker. Read here, never popped:
# activation happens before main's loader consumes it.
_DESKTOP_OWNER_KIND_ENV = "UNSLOTH_STUDIO_DESKTOP_OWNER_KIND"
_DEFAULT_ON_PLATFORMS = ("darwin", "win32")
_TRUTHY = ("1", "true", "yes")
_FALSEY = ("0", "false", "no")

# Resolved from this file so it is right in a checkout and an installed wheel
# alike. Never build it from the cwd or a hardcoded "studio/backend".
_VENDOR_DIR = str(Path(__file__).resolve().parent.parent / "vendor")

_logger = logging.getLogger(__name__)
_activated = False


def native_tls_enabled() -> bool:
    """Resolve ``UNSLOTH_STUDIO_NATIVE_TLS`` against the platform default.

    On Linux the default flips to on when the backend runs as the Tauri
    desktop's owned process (#9218): the .deb/AppImage desktop is launched
    from a desktop icon, not a shell, so a self-signed CA trusted only in
    the OS store is unreachable both ways — the provider connectivity
    check fails with CERTIFICATE_VERIFY_FAILED, and exporting
    SSL_CERT_FILE replaces the roots instead of adding to them, taking
    Hugging Face downloads down too. The headless ``unsloth studio``
    server keeps the opt-in default (distro OpenSSL configurations vary
    there, and an operator running from a shell can export the env).
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

    Mirrors main._load_desktop_owner's handshake (kind == "tauri") without
    consuming the env vars it pops: native_tls activates BEFORE main's
    loader pops either variable, so reading the kind marker here cannot
    race or steal it.
    """
    return os.environ.get(_DESKTOP_OWNER_KIND_ENV, "") == "tauri"


# Children that cannot import this module (the `python -c` probes,
# prebuilt_core.py) carry the gate as source; generating it from the same
# constants stops it drifting from native_tls_enabled(). The Linux
# desktop-owner clause mirrors _desktop_owned_process, so a child launched
# directly by the desktop app (not via the backend) resolves the same flip.
# The child supplies os, sys and _TRUSTSTORE_VENDOR itself, which is what
# keeps the gate identical everywhere despite each child locating the vendor
# directory differently.
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
    # main.py pops the desktop-owner vars before the `python -c` probes are
    # spawned, so those children re-resolve from the tri-state flag alone.
    # Spell the resolved decision back into the env (same mirroring as the
    # UV_* pair below) and the children agree with this process by default.
    os.environ.setdefault(_NATIVE_TLS_ENV, "1")
    # uv's rustls ignores in-process injection (uv >= 0.11 reads UV_SYSTEM_CERTS,
    # older reads UV_NATIVE_TLS). Mirror one value across both: uv takes either as
    # an opt-in, so an opt-out in one spelling must carry to the other.
    os.environ.setdefault("UV_SYSTEM_CERTS", os.environ.get("UV_NATIVE_TLS", "1"))
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
