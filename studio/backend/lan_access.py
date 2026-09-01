# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Runtime LAN listener for Unsloth Studio.

Unsloth binds 127.0.0.1 by default, so a phone or laptop on the same network
cannot reach it without relaunching with ``-H 0.0.0.0``. This module adds a
second uvicorn listener over the already-running app, on the machine's own
network addresses and the same port, and takes it away again -- no restart, and
the loopback socket keeps serving the desktop app throughout.

The listener binds each detected address explicitly rather than the wildcard:
``0.0.0.0`` collides with the loopback socket that already holds the port. It
runs on the primary server's event loop with ``lifespan="off"``, so the app's
startup and shutdown handlers stay owned by the primary server and never fire
twice.

The settings-managed listener remains IPv4 only. Address discovery also serves
an existing IPv6 wildcard launch, while link-local IPv6 addresses are omitted
because another device cannot use them without the listener's interface scope.
"""

from __future__ import annotations

import asyncio
import ipaddress
import platform
import socket
import subprocess
import sys
import threading
import time
from typing import Any, Optional

import uvicorn

from loggers import get_logger
from utils.host_policy import set_lan_connector_active

logger = get_logger(__name__)

# local socket work either way, so exceeding these means the event loop is wedged
_START_TIMEOUT = 10.0
# kept under the ~5s Windows console-close budget run.py's shutdown path works to
_STOP_TIMEOUT = 3.0

# a LAN request already accepted can run for minutes and stays a remote caller throughout; on expiry
# the trust flag is left active rather than downgraded
_DRAIN_TIMEOUT = 300.0

# networking mode rarely changes, but a failed wslinfo probe must eventually recover
_WSL_MODE_CACHE_TTL = 60.0

# uvicorn's own default, so a burst queues on the LAN socket as it does on loopback
_LISTEN_BACKLOG = 2048

_lock = threading.RLock()
_wsl_mode_lock = threading.Lock()
_wsl_mode_cache: Optional[tuple[float, str]] = None
_server: Any = None
_serve_loop: Any = None
_sockets: tuple[socket.socket, ...] = ()
_port: Optional[int] = None
_error: Optional[str] = None
# stopped listeners whose accepted requests are still running remain remote callers, so the trust
# flag stays up until every one has drained
_pending_drains = 0
# rebound whole, never mutated: request_on_lan_listener reads it without the lock
_bound_addresses: tuple[str, ...] = ()


def detect_lan_addresses(ip_version: int = 4) -> list[str]:
    """The machine's own reachable addresses for one IP version, default route first.

    Loopback, link-local (169.254/16) and multicast are dropped: none of them is
    an address another device on the network can open. A public address is kept
    -- a cloud VM binding its own public IP is the same operation as a laptop
    binding its Wi-Fi address, and the caller decides whether that is wanted.
    """
    # WSL's NAT-side address belongs to a private Hyper-V network a second device cannot open; mirrored
    # mode is different, since WSL joins the host's network
    if _wsl_networking_mode() not in (None, "mirrored"):
        return []

    if ip_version == 4:
        socket_family = socket.AF_INET
        route_probe = ("8.8.8.8", 80)
    elif ip_version == 6:
        socket_family = socket.AF_INET6
        route_probe = ("2001:4860:4860::8888", 80, 0, 0)
    else:
        raise ValueError("ip_version must be 4 or 6")

    addresses: list[str] = []

    def _add(candidate: str) -> None:
        candidate = candidate.split("%", 1)[0]
        try:
            parsed = ipaddress.ip_address(candidate)
        except ValueError:
            return
        if parsed.version != ip_version:
            return
        if parsed.is_loopback or parsed.is_link_local or parsed.is_multicast:
            return
        if parsed.is_unspecified or parsed.is_reserved:
            return
        normalized = str(parsed)
        if normalized not in addresses:
            addresses.append(normalized)

    # a UDP connect only fixes the local end of the socket; nothing is sent to the target
    probe = None
    try:
        probe = socket.socket(socket_family, socket.SOCK_DGRAM)
        probe.connect(route_probe)
        _add(probe.getsockname()[0])
    except OSError:
        pass
    finally:
        if probe is not None:
            probe.close()

    # a route probe picks one source address and an isolated LAN has no route at all, so neither it nor
    # the hostname enumerates the other adapters
    for address in _interface_addresses(ip_version):
        _add(address)
    return addresses


def _wsl_networking_mode() -> Optional[str]:
    """The active WSL networking mode, or ``None`` outside WSL.

    An older WSL without ``wslinfo`` is treated as unknown and therefore not
    advertised. Older releases use NAT, so failing closed avoids handing a phone
    an address that only the Windows host can route to.
    """
    global _wsl_mode_cache

    if sys.platform != "linux" or "microsoft" not in platform.release().casefold():
        return None
    with _wsl_mode_lock:
        now = time.monotonic()
        cached = _wsl_mode_cache
        if cached is not None and now - cached[0] < _WSL_MODE_CACHE_TTL:
            return cached[1]
        try:
            result = subprocess.run(
                ["wslinfo", "--networking-mode"],
                capture_output = True,
                check = False,
                text = True,
                encoding = "utf-8",
                timeout = 1,
            )
        except (OSError, subprocess.SubprocessError):
            mode = "unknown"
        else:
            mode = result.stdout.strip().casefold() or "unknown"
        _wsl_mode_cache = (time.monotonic(), mode)
        return mode


def _is_host_only_interface(name: str) -> bool:
    """True for Windows Hyper-V switches that do not face the physical LAN."""
    normalized = name.strip().casefold()
    if not normalized.startswith("vethernet ("):
        return False
    return any(
        marker in normalized
        for marker in ("default switch", "wsl", "hyper-v firewall", "host-only")
    )


def _interface_addresses(ip_version: int = 4) -> list[str]:
    """Addresses for one IP version on every interface that is up.

    Falls back to resolving the hostname where psutil is unavailable. That
    fallback is not an enumeration: a Linux host mapping its name to 127.0.1.1
    reports nothing, which is why it is the last resort rather than the source.
    """
    try:
        import psutil
    except ImportError:
        try:
            return [
                info[4][0]
                for info in socket.getaddrinfo(
                    socket.gethostname(),
                    None,
                    socket.AF_INET if ip_version == 4 else socket.AF_INET6,
                )
            ]
        except OSError:
            return []
    try:
        stats = psutil.net_if_stats()
        family = socket.AF_INET if ip_version == 4 else socket.AF_INET6
        addresses = []
        for name, entries in psutil.net_if_addrs().items():
            if _is_host_only_interface(name):
                continue
            interface = stats.get(name)
            if interface is not None and not interface.isup:
                continue
            addresses.extend(e.address for e in entries if e.family == family)
        return addresses
    except Exception:
        return []


def is_public_address(address: str) -> bool:
    """True when ``address`` is routable from the internet, not just this network.

    A VPS or dedicated box usually carries its public IPv4 straight on the NIC, so
    the addresses this module binds are not always the LAN addresses the name
    implies. Callers surface that rather than refusing it: a public-IP campus or
    office network is a legitimate place to serve, and only the operator knows
    which one they are on.
    """
    try:
        return ipaddress.ip_address(address).is_global
    except ValueError:
        return False


def _bind_listener(address: str, port: int) -> socket.socket:
    """A listening socket on exactly ``address:port``."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        # skipped on Windows, where SO_REUSEADDR lets a socket take over a live listener
        if sys.platform != "win32":
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((address, port))
        sock.listen(_LISTEN_BACKLOG)
        sock.set_inheritable(False)
    except BaseException:
        sock.close()
        raise
    return sock


def _listener_config(app, host: str, port: int):
    from utils.uvicorn_h11_shutdown import uvicorn_http_protocol
    return uvicorn.Config(
        app,
        host = host,
        port = port,
        # a second lifespan would re-fire the app's startup handlers
        lifespan = "off",
        # uvicorn.Config applies log_config eagerly, resetting run.py's startup log rewrite
        log_config = None,
        access_log = False,
        server_header = False,
        http = uvicorn_http_protocol(),
    )


def _running_on_event_loop() -> bool:
    """True when the caller is already inside a running event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


def _wait_until(predicate, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def start_lan_listener(
    app,
    loop,
    port: int,
    fallback_ports: tuple[int, ...] = (),
) -> tuple[str, ...]:
    """Serve ``app`` on LAN addresses at the first bindable candidate port."""
    global _server, _serve_loop, _sockets, _bound_addresses, _port, _error

    with _lock:
        if _server is not None:
            return _bound_addresses

        candidates = detect_lan_addresses()
        if not candidates:
            _error = "no_lan_address"
            raise RuntimeError(_error)

        sockets: list[socket.socket] = []
        bound: list[str] = []
        failures: list[str] = []
        attempted: list[str] = []
        for candidate_port in (port, *fallback_ports):
            sockets = []
            bound = []
            failures = []
            for address in candidates:
                try:
                    sockets.append(_bind_listener(address, candidate_port))
                except OSError as exc:
                    failures.append(f"{address}:{candidate_port} ({exc})")
                    continue
                bound.append(address)
            if sockets:
                port = candidate_port
                break
            attempted.extend(failures)

        if not sockets:
            _error = "bind_failed"
            logger.warning("LAN access could not bind: %s", "; ".join(attempted))
            raise RuntimeError(_error)
        if failures:
            logger.info("LAN access skipped unbindable addresses: %s", "; ".join(failures))

        server = uvicorn.Server(_listener_config(app, bound[0], port))
        # published before the socket can accept: a request served in between would still read the loopback-
        # only trust defaults
        set_lan_connector_active(True)
        serving = server.serve(sockets = sockets)
        try:
            future = asyncio.run_coroutine_threadsafe(serving, loop)
        except RuntimeError as exc:
            # the loop can close between _server_loop validating it and this call
            serving.close()
            _fail_start(sockets, port, exc)
            raise RuntimeError(_error) from exc
        started = _wait_until(lambda: server.started or future.done(), _START_TIMEOUT)
        if not started or not server.started:
            server.should_exit = True
            cause = future.exception(timeout = 0) if future.done() else None
            future.cancel()
            _fail_start(sockets, port, cause if cause is not None else "timed out")
            raise RuntimeError(_error)

        _server, _serve_loop, _sockets = server, loop, tuple(sockets)
        _bound_addresses, _port, _error = tuple(bound), port, None

    logger.info("LAN access listening on %s", ", ".join(f"{a}:{port}" for a in bound))
    return _bound_addresses


def _sync_lan_trust() -> None:
    """Publish the beyond-loopback flag from the authoritative state.

    Derived rather than assigned by callers: a repeated stop, or a start racing a
    stop in another worker thread, otherwise cleared a flag that a live listener
    or a still-draining one owned. The caller holds ``_lock``.
    """
    set_lan_connector_active(_server is not None or _pending_drains > 0)


def _release_listener_state() -> None:
    """Drop the listener references. The caller holds ``_lock``."""
    global _server, _serve_loop, _sockets, _port

    _server = _serve_loop = None
    _sockets = ()
    _port = None
    _sync_lan_trust()


def _fail_start(sockets, port: int, cause) -> None:
    """Undo a start that never came up. The caller holds ``_lock``."""
    global _error

    _close_sockets(sockets)
    _sync_lan_trust()
    _error = "listener_start_failed"
    logger.warning("LAN access listener did not start on port %s: %s", port, cause)


def _arm_drain_watcher(server) -> None:
    """Own the trust flag until ``server``'s accepted requests end. Caller holds ``_lock``."""
    global _pending_drains

    _pending_drains += 1
    threading.Thread(
        target = _clear_trust_after_drain,
        args = (server,),
        name = "lan-access-drain",
        daemon = True,
    ).start()


def _clear_trust_after_drain(server) -> None:
    """Hold the beyond-loopback flag until the stopped listener's requests finish.

    Closing the listening sockets stops new connections, but uvicorn then drains
    the accepted ones, and a request that started on the LAN is still a remote
    caller for its whole life.
    """
    global _pending_drains

    state = getattr(server, "server_state", None)
    deadline = time.monotonic() + _DRAIN_TIMEOUT
    while state is not None and state.connections and time.monotonic() < deadline:
        time.sleep(0.05)
    with _lock:
        if state is not None and state.connections:
            # ownership is never given up: a request that never ended is still remote
            logger.warning("LAN access kept the trust flag on: connections did not drain")
            return
        _pending_drains -= 1
        _sync_lan_trust()


def _close_sockets(sockets) -> None:
    for sock in sockets:
        try:
            sock.close()
        except OSError:
            pass


def stop_lan_listener() -> bool:
    """Release the LAN sockets and take the listener down. Idempotent.

    Returns whether the port is confirmed released. A False means the sockets may
    still be accepting, so the caller must keep treating the host as reachable.

    Waits for the sockets, not for ``serve()`` to return: uvicorn closes the
    sockets passed to it at the top of its shutdown and only then drains
    in-flight responses, so waiting on the serve task would make a Stop pressed
    from a LAN device wait out its own response.
    """
    global _server, _serve_loop, _sockets, _bound_addresses, _port, _error

    # a start holds _lock while waiting for this loop to run serve(), so a stop arriving on the loop
    # itself must not block on it
    if not _lock.acquire(blocking = not _running_on_event_loop()):
        logger.info("LAN access stop deferred: a listener change is in flight")
        return False
    # held across the wait so a start cannot begin rebinding sockets still closing
    try:
        server, loop, sockets = _server, _serve_loop, _sockets
        port = _port
        # closed before the wait so a request landing mid-teardown already reads as off
        _bound_addresses = ()

        if server is None:
            _release_listener_state()
            return True
        server.should_exit = True
        if _running_on_event_loop():
            # /api/shutdown tears down from a task on this very loop, so waiting would deadlock; ownership is
            # kept because uvicorn cannot close the sockets until the loop is free again
            logger.info("LAN access stopping")
            return True
        if loop is None or loop.is_closed() or not loop.is_running():
            # nothing is left to run uvicorn's shutdown, so release the sockets here
            _close_sockets(sockets)
            _release_listener_state()
            logger.info("LAN access stopped with its server loop")
            return True
        if _wait_until(lambda: all(sock.fileno() == -1 for sock in sockets), _STOP_TIMEOUT):
            # armed before the release so the flag is never briefly unowned
            _arm_drain_watcher(server)
            _release_listener_state()
            logger.info("LAN access stopped")
            return True
        # ownership is kept so a retry waits on these same sockets, and so a second stop cannot report
        # success while the port may still be accepting
        _error = "stop_timed_out"
        logger.warning("LAN access did not release port %s within %ss", port, _STOP_TIMEOUT)
        return False
    finally:
        _lock.release()


def lan_listener_status() -> dict:
    """Runtime view of the listener: whether it serves, where, and why not."""
    with _lock:
        return {
            "running": _server is not None,
            "addresses": list(_bound_addresses),
            "port": _port,
            "error": _error,
        }


def clear_lan_listener_error() -> None:
    """Drop a recorded failure so a retry starts from a clean status."""
    global _error
    with _lock:
        _error = None


def request_on_lan_listener(scope) -> bool:
    """True when this request arrived on a LAN listener socket, not on loopback.

    ``scope["server"]`` is the accepting socket's own address, so it identifies
    the listener a connection came in on without trusting any client header.
    """
    addresses = _bound_addresses
    if not addresses:
        return False
    server = scope.get("server")
    return bool(server) and server[0] in addresses


def close_lan_listener_lifecycle() -> None:
    """Shutdown hook: never raise, whatever state the listener is in."""
    try:
        stop_lan_listener()
    except Exception as exc:
        logger.warning("Error stopping the LAN listener: %s", exc)
