# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Persisted policy and launch policy for Settings > LAN access.

The listener itself lives in ``lan_access``; this decides whether the current
launch may own one, whether the user is allowed to turn it on, and remembers the
answer across restarts.
"""

from __future__ import annotations

import ipaddress
import socket
from typing import Any, Optional

from loggers import get_logger

logger = get_logger(__name__)

LAN_ACCESS_AUTO_START_KEY = "lan_access_auto_start"
DEFAULT_LAN_ACCESS_AUTO_START = False

_PRIVATE_LAN_NETWORKS = (
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
)


def _normalized_ip(address: str):
    """Parse one literal address, normalizing IPv4-mapped IPv6."""
    if not isinstance(address, str):
        return None
    value = address.strip().strip("[]")
    if "%" in value:
        value = value.split("%", 1)[0]
    try:
        parsed = ipaddress.ip_address(value)
    except ValueError:
        return None
    if isinstance(parsed, ipaddress.IPv6Address) and parsed.ipv4_mapped is not None:
        return parsed.ipv4_mapped
    return parsed


def _resolve_host_addresses(host: str, port: int) -> tuple:
    """Resolve a literal or hostname to normalized transport addresses."""
    literal = _normalized_ip(host)
    if literal is not None:
        return (literal,)
    if not isinstance(host, str) or not isinstance(port, int) or port < 0:
        return ()
    try:
        infos = socket.getaddrinfo(host, port, type = socket.SOCK_STREAM)
    except OSError:
        return ()
    addresses = []
    for _family, _kind, _protocol, _name, sockaddr in infos:
        if not sockaddr:
            continue
        parsed = _normalized_ip(sockaddr[0])
        if parsed is not None and parsed not in addresses:
            addresses.append(parsed)
    return tuple(addresses)


def _private_non_loopback(address) -> bool:
    return not address.is_loopback and any(address in network for network in _PRIVATE_LAN_NETWORKS)


def _all_addresses_are(host: str, port: int, predicate) -> bool:
    addresses = _resolve_host_addresses(host, port)
    return bool(addresses) and all(predicate(address) for address in addresses)


def request_is_loopback(request) -> bool:
    """Whether both authoritative transport endpoints are loopback."""
    scope = getattr(request, "scope", {})
    server = scope.get("server")
    client = scope.get("client")
    if not isinstance(server, (tuple, list)) or len(server) < 2:
        return False
    if not isinstance(client, (tuple, list)) or len(client) < 1:
        return False
    server_host, server_port = server[0], server[1]
    client_host = client[0]
    if not isinstance(server_host, str) or not isinstance(server_port, int):
        return False
    if not isinstance(client_host, str):
        return False
    return _all_addresses_are(server_host, server_port, lambda address: address.is_loopback) and (
        _all_addresses_are(client_host, server_port, lambda address: address.is_loopback)
    )


def _addresses_match(host: str, port: int, candidates) -> bool:
    request_addresses = set(_resolve_host_addresses(host, port))
    if not request_addresses:
        return False
    configured = set()
    for candidate in candidates or ():
        if isinstance(candidate, str):
            configured.update(_resolve_host_addresses(candidate, port))
    return bool(request_addresses & configured)


def request_on_lan_access(request) -> bool:
    """Classify a request from ASGI socket state, never client-controlled headers.

    Both the accepting endpoint and peer must be private and non-loopback. The
    accepting endpoint must also match either the exact live settings listener or
    the launch-managed bind and port published at startup.
    """
    from lan_access import lan_listener_status

    scope = getattr(request, "scope", {})
    server = scope.get("server")
    client = scope.get("client")
    if not isinstance(server, (tuple, list)) or len(server) < 2:
        return False
    if not isinstance(client, (tuple, list)) or len(client) < 1:
        return False
    server_host, server_port = server[0], server[1]
    client_host = client[0]
    if not isinstance(server_host, str) or not isinstance(server_port, int):
        return False
    if not isinstance(client_host, str):
        return False
    if not _all_addresses_are(server_host, server_port, _private_non_loopback):
        return False
    if not _all_addresses_are(client_host, server_port, _private_non_loopback):
        return False

    try:
        listener = lan_listener_status()
    except Exception:
        return False
    if not isinstance(listener, dict):
        return False
    if (
        listener.get("running") is True
        and listener.get("port") == server_port
        and _addresses_match(server_host, server_port, listener.get("addresses"))
    ):
        return True

    try:
        app_state = request.app.state
    except Exception:
        return False
    if bool(getattr(app_state, "lan_access_is_colab", False)):
        return False
    if bool(getattr(app_state, "lan_access_secure_launch", False)):
        return False
    if not bool(getattr(app_state, "lan_access_launch_managed", False)):
        return False
    if getattr(app_state, "lan_access_port", None) != server_port:
        return False
    if bool(getattr(app_state, "lan_access_wildcard_bind", False)):
        return True
    return _addresses_match(
        server_host,
        server_port,
        getattr(app_state, "lan_access_launch_addresses", ()),
    )


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
    from utils.host_policy import wildcard_ip_versions

    app_state.lan_access_port = port
    app_state.lan_access_wildcard_ip_versions = wildcard_ip_versions(bind_host)
    app_state.lan_access_wildcard_bind = bool(app_state.lan_access_wildcard_ip_versions)
    app_state.lan_access_bind_host = bind_host
    app_state.lan_access_launch_addresses = tuple(
        str(address) for address in _resolve_host_addresses(bind_host, port)
    )
    resolved_loopback = bool(app_state.lan_access_launch_addresses) and all(
        _normalized_ip(address).is_loopback for address in app_state.lan_access_launch_addresses
    )
    # An unresolved hostname is launch-managed but never trusted for keyless LAN
    # admission: request_on_lan_access requires its resolved address set.
    app_state.lan_access_launch_managed = (
        app_state.lan_access_wildcard_bind or not resolved_loopback
    )
    # --secure forces the loopback bind precisely so the raw port is never exposed
    app_state.lan_access_secure_launch = bool(secure)
    app_state.lan_access_is_colab = bool(is_colab)
    app_state.lan_access_frontend_served = bool(frontend_served)
    app_state.lan_access_ready = False


def _launch_urls(app_state) -> list[str]:
    """Where a launch-managed bind answers on this network.

    A wildcard launch cannot rely only on ``server_url``: run.py gives that
    direct base one LAN-reachable address, while Settings must show every
    currently reachable address in each family the launch serves.
    """
    if getattr(app_state, "lan_access_wildcard_bind", False):
        from lan_access import detect_lan_addresses

        addresses = []
        for ip_version in getattr(app_state, "lan_access_wildcard_ip_versions", ()) or (4,):
            for address in detect_lan_addresses(ip_version):
                if address not in addresses:
                    addresses.append(address)
        return _listener_urls(
            addresses,
            getattr(app_state, "lan_access_port", None),
        )
    url = getattr(app_state, "server_url", None)
    return [url] if url else []


def _listener_urls(addresses, port: Optional[int]) -> list[str]:
    if not port:
        return []
    urls = []
    for address in addresses:
        url_host = f"[{address}]" if ":" in address else address
        urls.append(f"http://{url_host}:{port}")
    return urls


def _public_urls(urls: list[str], resolved_addresses: tuple[str, ...] = ()) -> list[str]:
    """The subset reachable from the internet rather than only this network."""
    from urllib.parse import urlparse

    resolved = []
    for address in resolved_addresses:
        parsed = _normalized_ip(address)
        if parsed is not None:
            resolved.append(parsed)
    if any(address.is_global for address in resolved):
        return list(urls)
    public = []
    for url in urls:
        parsed = urlparse(url)
        port = parsed.port or 80
        addresses = _resolve_host_addresses(parsed.hostname or "", port)
        if addresses and any(address.is_global for address in addresses):
            public.append(url)
    return public


def _has_keyless_lan_url(urls: list[str]) -> bool:
    """Whether any of these URLs is one a keyless caller can actually reach.

    Resolution alone is not enough: `keyless_api_access._host_authority_is_direct` refuses a
    `Host` that names anything, so a hostname bind yields a URL that resolves to a private
    address and is still refused. Reporting it eligible is what made the LAN panel advertise
    `Bearer not-needed` against a URL that answers 401, so the literal is required here too.

    Admission decides, through the shared
    `keyless_api_access.keyless_authority_address_allowed`. A second copy of the test is what
    let an IPv4-mapped literal like `::ffff:192.168.1.24` be advertised while admission
    refused it: `_normalized_ip` un-maps, which is exactly what that form is refused for.
    """
    import ipaddress
    from urllib.parse import urlparse

    from utils.keyless_api_access import (
        KEYLESS_SCOPE_INFERENCE,
        keyless_authority_address_allowed,
    )

    for url in urls:
        parsed = urlparse(url)
        if not parsed.hostname:
            continue
        try:
            # urlparse already strips the IPv6 brackets and lowercases; parse the
            # remaining literal without normalising it
            address = ipaddress.ip_address(parsed.hostname)
        except ValueError:
            continue
        if not keyless_authority_address_allowed(address, KEYLESS_SCOPE_INFERENCE):
            continue
        if _all_addresses_are(parsed.hostname, parsed.port or 80, _private_non_loopback):
            return True
    return False


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
    try:
        from utils.keyless_api_access import get_keyless_api_access_settings
        keyless_scope, keyless_tools = get_keyless_api_access_settings()
    except Exception:
        keyless_scope, keyless_tools = "off", False
    return {
        "state": state,
        "urls": urls,
        "public_urls": _public_urls(
            urls,
            getattr(app_state, "lan_access_launch_addresses", ()) if launch_managed else (),
        ),
        "error": listener["error"],
        "auto_start": get_lan_access_auto_start(),
        "managed_by": managed_by,
        "can_start": controllable and not running,
        "can_stop": controllable and running,
        "block_reason": block_reason,
        "bind_host": getattr(app_state, "lan_access_bind_host", None),
        "wildcard_bind": bool(getattr(app_state, "lan_access_wildcard_bind", False)),
        "serves_web_ui": frontend_served,
        "keyless_lan_eligible": _has_keyless_lan_url(urls),
        "keyless_scope": keyless_scope,
        "keyless_tools": keyless_tools,
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
