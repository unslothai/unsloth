# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Uses OS trust store for TLS-inspecting proxies; headless Linux needs UNSLOTH_STUDIO_NATIVE_TLS=1."""

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

# Resolved from this file so it is right in a checkout and an installed wheel alike; never built from the cwd or a hardcoded "studio/backend".
_VENDOR_DIR = str(Path(__file__).resolve().parent.parent / "vendor")

_logger = logging.getLogger(__name__)
_activated = False


def native_tls_enabled() -> bool:
    """Linux defaults on only for the desktop backend: icon-launched apps never read the shell opt-in."""
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
    """Reads the Tauri owner marker without popping it, since main pops it later."""
    return os.environ.get(_DESKTOP_OWNER_KIND_ENV, "") == "tauri"


def _uv_system_certs_wanted() -> bool:
    """System certs REPLACE the bundled roots of uv, so only opt-in or covered platforms; a bad SSL_CERT_FILE then fails uv with no pip fallback."""
    if os.environ.get(_NATIVE_TLS_ENV, "").strip().lower() in _TRUTHY:
        return True
    return sys.platform in _DEFAULT_ON_PLATFORMS


# Inline gate for children that cannot import this module (python -c probes, prebuilt_core.py); each must supply os, sys and _TRUSTSTORE_VENDOR.
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
    """The gate as executable source, for a child that cannot import this module. The child must bind ``_TRUSTSTORE_VENDOR`` to the vendor directory first."""
    return _INLINE_GATE.format(
        env = _NATIVE_TLS_ENV,
        owner_env = _DESKTOP_OWNER_KIND_ENV,
        truthy = _TRUTHY,
        falsey = _FALSEY,
        platforms = _DEFAULT_ON_PLATFORMS,
    )


def activate_native_tls() -> bool:
    """Idempotently patch ``ssl`` to verify against the OS trust store. Returns True when injection is active; failure is non-fatal, since falling back to certifi is the pre-existing, strictly less permissive behaviour."""
    global _activated
    if _activated:
        return True
    if not native_tls_enabled():
        return False
    # main.py pops the desktop marker, so write the decision into env for children; assign, not setdefault, since an unrecognized value would read as off.
    uv_default = "1" if _uv_system_certs_wanted() else "0"
    os.environ[_NATIVE_TLS_ENV] = "1"
    # Mirror one value to UV_SYSTEM_CERTS and UV_NATIVE_TLS, even when 0: uv takes either as opt-in, so an absent value would turn it on in a worker.
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
