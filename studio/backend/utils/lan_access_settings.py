# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Persisted policy and launch policy for Settings > LAN access.

The listener itself lives in ``lan_access``; this decides whether the current
launch may own one, whether the user is allowed to turn it on, and remembers the
answer across restarts.
"""

from __future__ import annotations

from typing import Any, Optional

from loggers import get_logger

logger = get_logger(__name__)

LAN_ACCESS_AUTO_START_KEY = "lan_access_auto_start"
DEFAULT_LAN_ACCESS_AUTO_START = False


def _coerce_bool(value: Any) -> Optional[bool]:
    return value if isinstance(value, bool) else None


def get_lan_access_auto_start() -> bool:
    """Read the preference, failing closed on missing, invalid, or unreadable data."""
    try:
        from storage.studio_db import get_app_setting
        stored = get_app_setting(LAN_ACCESS_AUTO_START_KEY, None)
    except Exception:
        return False
    parsed = _coerce_bool(stored)
    return parsed if parsed is not None else DEFAULT_LAN_ACCESS_AUTO_START


def set_lan_access_auto_start(enabled: bool) -> bool:
    if not isinstance(enabled, bool):
        raise ValueError("LAN access auto-start must be true or false.")
    from storage.studio_db import upsert_app_settings

    upsert_app_settings({LAN_ACCESS_AUTO_START_KEY: enabled})
    return enabled


def _admin_password_ready() -> bool:
    try:
        from auth.storage import DEFAULT_ADMIN_USERNAME, requires_password_change
        return not requires_password_change(DEFAULT_ADMIN_USERNAME)
    except Exception:
        return False


def configure_lan_access(
    app_state, *, port: int, bind_host: str, secure: bool, is_colab: bool, frontend_served: bool
) -> None:
    """Publish immutable launch policy used by every settings request."""
    from utils.host_policy import is_external_host

    app_state.lan_access_port = port
    app_state.lan_access_wildcard_bind = bind_host in ("0.0.0.0", "::")
    app_state.lan_access_launch_managed = app_state.lan_access_wildcard_bind or is_external_host(
        bind_host
    )
    # --secure forces the loopback bind precisely so the raw port is never exposed
    app_state.lan_access_secure_launch = bool(secure)
    app_state.lan_access_is_colab = bool(is_colab)
    app_state.lan_access_frontend_served = bool(frontend_served)
    app_state.lan_access_ready = False


def _launch_urls(app_state) -> list[str]:
    """Where a launch-managed bind answers on this network.

    A wildcard launch cannot use ``server_url``: run.py builds that from
    ``_display_host_for_bind``, which resolves the machine's public IP for
    sharing, and behind NAT that address reaches nothing on the LAN.
    """
    if getattr(app_state, "lan_access_wildcard_bind", False):
        cached = getattr(app_state, "lan_access_wildcard_urls", None)
        if cached is None:
            from lan_access import detect_lan_addresses
            cached = _listener_urls(
                detect_lan_addresses(), getattr(app_state, "lan_access_port", None)
            )
            app_state.lan_access_wildcard_urls = cached
        return list(cached)
    url = getattr(app_state, "server_url", None)
    return [url] if url else []


def _listener_urls(addresses, port: Optional[int]) -> list[str]:
    if not port:
        return []
    return [f"http://{address}:{port}" for address in addresses]


def _public_urls(urls: list[str]) -> list[str]:
    """The subset reachable from the internet rather than only this network."""
    from urllib.parse import urlparse

    from lan_access import is_public_address

    return [url for url in urls if is_public_address(urlparse(url).hostname or "")]


def lan_access_status(app) -> dict:
    """Everything Settings > LAN access renders, resolved for the current launch."""
    from lan_access import lan_listener_status

    app_state = app.state
    listener = lan_listener_status()
    ready = bool(getattr(app_state, "lan_access_ready", False))
    is_colab = bool(getattr(app_state, "lan_access_is_colab", False))
    launch_managed = bool(getattr(app_state, "lan_access_launch_managed", False))
    frontend_served = bool(getattr(app_state, "lan_access_frontend_served", False))

    block_reason = None
    if not ready:
        block_reason = "server_starting"
    elif is_colab:
        block_reason = "colab"
    elif launch_managed:
        block_reason = "launch_managed"
    elif bool(getattr(app_state, "lan_access_secure_launch", False)):
        block_reason = "secure_launch"
    elif not _admin_password_ready():
        block_reason = "admin_password_change_required"

    running = bool(listener["running"])
    if launch_managed:
        state, urls, managed_by = "online", _launch_urls(app_state), "launch"
    elif running:
        state, urls, managed_by = (
            "online",
            _listener_urls(listener["addresses"], listener["port"]),
            "settings",
        )
    else:
        state = "error" if listener["error"] else "off"
        urls, managed_by = [], None

    controllable = block_reason is None
    return {
        "state": state,
        "urls": urls,
        "public_urls": _public_urls(urls),
        "error": listener["error"],
        "auto_start": get_lan_access_auto_start(),
        "managed_by": managed_by,
        "can_start": controllable and not running,
        "can_stop": controllable and running,
        "block_reason": block_reason,
        "serves_web_ui": frontend_served,
    }


def _server_loop(app_state):
    """The loop the primary server serves on, or a refusal if it is not serving."""
    loop = getattr(app_state, "lan_access_loop", None)
    if loop is None or loop.is_closed() or not loop.is_running():
        raise RuntimeError("server_not_running")
    return loop


def start_lan_access(app) -> dict:
    """Bring the LAN listener up for this launch. Repeated requests are idempotent."""
    from lan_access import start_lan_listener

    status = lan_access_status(app)
    if status["state"] == "online":
        return status
    if not status["can_start"]:
        raise RuntimeError(status["block_reason"] or "operation_in_progress")

    port = getattr(app.state, "lan_access_port", None)
    if not isinstance(port, int) or port <= 0:
        raise RuntimeError("server_port_unavailable")

    addresses = start_lan_listener(app, _server_loop(app.state), port)
    logger.info("LAN access started on %s", ", ".join(addresses))
    return lan_access_status(app)


def stop_lan_access(app) -> dict:
    """Take the LAN listener down without changing the auto-start preference."""
    from lan_access import clear_lan_listener_error, stop_lan_listener

    status = lan_access_status(app)
    if status["managed_by"] == "launch":
        raise RuntimeError("launch_managed")
    # a stop that could not confirm the port is closed leaves the host reachable
    # a stop that could not confirm the port is closed leaves the host reachable,
    # and lan_access keeps the trust flag with the listener state it describes
    if stop_lan_listener():
        clear_lan_listener_error()
    return lan_access_status(app)


def maybe_auto_start_lan_access(app) -> bool:
    """Start the listener at boot when the persisted preference allows it."""
    if not get_lan_access_auto_start():
        return False
    try:
        start_lan_access(app)
    except Exception as exc:
        # an optional preference must never take the whole server down with it
        logger.info("LAN access auto-start skipped: %s", exc)
        return False
    return True
