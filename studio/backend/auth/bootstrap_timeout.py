# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Shuts down an exposed UI still on its bootstrap password after a deadline; web UI only, never
--api-only or Colab; UNSLOTH_STUDIO_BOOTSTRAP_TIMEOUT (default 3600, 0 disables)."""

import os
import sys
import threading
import time
from typing import Optional

BOOTSTRAP_TIMEOUT_ENV_VAR = "UNSLOTH_STUDIO_BOOTSTRAP_TIMEOUT"
DEFAULT_BOOTSTRAP_TIMEOUT_SECONDS = 3600
# Retry cadence and window for a failed auth storage read at the deadline.
STORAGE_RETRY_SECONDS = 5
STORAGE_RETRY_WINDOW_SECONDS = 600

# Read back over HTTP because the stderr warning is lost on --secure / external-bind launches.
_deadline_at: Optional[float] = None


def record_bootstrap_deadline(timeout_seconds: int) -> None:
    global _deadline_at
    _deadline_at = time.monotonic() + timeout_seconds if timeout_seconds > 0 else None


def clear_bootstrap_deadline() -> None:
    global _deadline_at
    _deadline_at = None


def bootstrap_deadline_remaining_seconds() -> Optional[int]:
    """Seconds until shutdown, or None when untimed; floored at 0 for calls after expiry."""
    if _deadline_at is None:
        return None
    return max(0, int(round(_deadline_at - time.monotonic())))


def bootstrap_timeout_seconds(env = None) -> int:
    """Deadline in seconds; 0 or negative disables it, but a malformed value uses the default."""
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
    """Arms the deadline only for an exposed web UI whose seeded admin password is unchanged."""
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
    """Deadline handler: shut down iff the seeded admin password is still unchanged. Returns True if it
    shut Unsloth down, False if it left it running (the password was changed in time)."""
    # One-shot timer: a failed read retries for a bounded window, then counts as unsecured.
    give_up_at = time.monotonic() + STORAGE_RETRY_WINDOW_SECONDS
    while True:
        try:
            still_default = storage.requires_password_change(storage.DEFAULT_ADMIN_USERNAME)
            break
        except Exception as e:
            if logger is not None:
                logger.warning("Bootstrap deadline could not read auth storage, retrying: %s", e)
            if time.monotonic() >= give_up_at:
                still_default = None
                break
            time.sleep(STORAGE_RETRY_SECONDS)
    if still_default is not None and not still_default:
        return False

    if still_default is None:
        message = (
            "\nUnsloth Studio was exposed on the network and auth storage stayed unreadable for "
            f"{_format_duration(STORAGE_RETRY_WINDOW_SECONDS)} after the first-login deadline, so "
            "whether the default admin password was changed is unknown. "
            "Shutting down to avoid leaving an unsecured public instance running."
        )
    else:
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
    record_bootstrap_deadline(timeout_seconds)
    timer = threading.Timer(
        timeout_seconds,
        enforce_bootstrap_password_deadline,
        args = (storage, trigger_shutdown),
        kwargs = {"timeout_seconds": timeout_seconds, "logger": logger},
    )
    timer.daemon = True
    timer.start()
    return timer
