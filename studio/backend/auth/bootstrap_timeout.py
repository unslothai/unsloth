# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Auto-shutdown for an exposed first-run Unsloth whose admin password is unchanged.

On a fresh install the seeded bootstrap admin password stays a valid login
credential until first login changes it. When the web UI is put on the network
(``--secure`` / ``0.0.0.0``) and nobody completes that first-login change within
a deadline, tear Unsloth down so a fresh, unconfigured instance does not stay
publicly reachable indefinitely. If the password was changed, Unsloth keeps
running.

Scope: web UI launches only (never ``--api-only``, which authenticates by API
key rather than the admin password, and never Colab). Configurable via
``UNSLOTH_STUDIO_BOOTSTRAP_TIMEOUT`` (seconds; default 3600; ``0`` disables).
"""

import os
import sys
import threading

BOOTSTRAP_TIMEOUT_ENV_VAR = "UNSLOTH_STUDIO_BOOTSTRAP_TIMEOUT"
DEFAULT_BOOTSTRAP_TIMEOUT_SECONDS = 3600


def bootstrap_timeout_seconds(env = None) -> int:
    """Resolve the deadline in seconds. ``0`` (or invalid/negative) disables it.

    A malformed value falls back to the default rather than disabling, so a typo
    cannot silently remove the protection.
    """
    env = os.environ if env is None else env
    raw = env.get(BOOTSTRAP_TIMEOUT_ENV_VAR)
    if raw is None or raw.strip() == "":
        return DEFAULT_BOOTSTRAP_TIMEOUT_SECONDS
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_BOOTSTRAP_TIMEOUT_SECONDS
    return value if value > 0 else 0


def _is_exposed_bind(host: str, secure: bool) -> bool:
    """True when this launch puts the web UI on the network (tunnel or non-loopback)."""
    if secure:
        return True
    if host in ("0.0.0.0", "::"):
        return True
    try:
        from utils.host_policy import is_external_host
    except Exception:
        return False
    return bool(is_external_host(host))


def should_arm_bootstrap_timeout(
    *,
    host: str,
    secure: bool,
    api_only: bool,
    frontend_served: bool,
    is_colab: bool,
    requires_change: bool,
    timeout_seconds: int,
) -> bool:
    """Whether to arm the deadline: only for an exposed web UI whose seeded admin
    password is still unchanged. Pure decision (no I/O) for cheap unit testing."""
    if timeout_seconds <= 0:
        return False
    if api_only or not frontend_served or is_colab:
        return False
    if not requires_change:
        return False
    return _is_exposed_bind(host, secure)


def _format_duration(seconds: int) -> str:
    """Human-friendly duration for the shutdown message (seconds under a minute)."""

    def _plural(n: int, unit: str) -> str:
        return f"{n} {unit}{'' if n == 1 else 's'}"

    if seconds < 60:
        return _plural(seconds, "second")
    minutes, rem = divmod(seconds, 60)
    label = _plural(minutes, "minute")
    if rem:
        label += f" {_plural(rem, 'second')}"
    return label


def enforce_bootstrap_password_deadline(
    storage,
    trigger_shutdown,
    *,
    timeout_seconds: int,
    logger = None,
) -> bool:
    """Deadline handler: shut down iff the seeded admin password is still unchanged.

    Returns True if it shut Unsloth down, False if it left it running (the
    password was changed in time).
    """
    try:
        still_default = storage.requires_password_change(storage.DEFAULT_ADMIN_USERNAME)
    except Exception:
        return False
    if not still_default:
        return False  # password changed in time -> leave Unsloth running

    message = (
        "\nUnsloth Studio was exposed on the network but its default admin "
        f"password was not changed within {_format_duration(timeout_seconds)}. "
        "Shutting down to avoid leaving an unsecured public instance running.\n"
        "Next time, sign in and change the password on first login, or set "
        f"{BOOTSTRAP_TIMEOUT_ENV_VAR}=0 to disable this timeout."
    )
    if logger is not None:
        logger.warning(message)
    print(message, file = sys.stderr, flush = True)
    try:
        trigger_shutdown()
    except Exception as e:  # shutdown is best-effort; never raise from the timer
        if logger is not None:
            logger.warning("Bootstrap-timeout shutdown failed: %s", e)
    return True


def arm_bootstrap_timeout(
    storage,
    trigger_shutdown,
    *,
    timeout_seconds: int,
    logger = None,
) -> "threading.Timer":
    """Start a daemon timer that enforces the deadline. Returns the Timer."""
    timer = threading.Timer(
        timeout_seconds,
        enforce_bootstrap_password_deadline,
        args = (storage, trigger_shutdown),
        kwargs = {"timeout_seconds": timeout_seconds, "logger": logger},
    )
    timer.daemon = True
    timer.start()
    return timer
