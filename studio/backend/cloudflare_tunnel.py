# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Free Cloudflare quick tunnel for Unsloth's 0.0.0.0 launches.

The raw http://<ip>:<port> is often unreachable (https-vs-http, blocked ports,
closed security groups); a cloudflared quick tunnel gives a free
https://*.trycloudflare.com URL that works anywhere, with no account or domain.

Best-effort throughout: any failure collapses to "no URL" and Unsloth keeps
running. Stdlib only (back-end imports are lazy) so it is safe to import early.
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Optional, Tuple

# Match only the URL; the negative lookahead drops cloudflared's own api.trycloudflare.com host from failure lines.
_URL_RE = re.compile(r"https://(?!api\.)[A-Za-z0-9-]+\.trycloudflare\.com")

# Until an edge connection registers, the quick-tunnel URL returns Cloudflare error 1033 (HTTP 530).
_REGISTERED_MARKER = "Registered tunnel connection"

_RELEASE_BASE = "https://github.com/cloudflare/cloudflared/releases/latest/download"

_READY_TIMEOUT = 15.0
_DOWNLOAD_TIMEOUT = 60

# A registered edge connection does not mean the hostname resolves yet, so the URL is fetched once before it is
# advertised.
_PUBLIC_PROBE_PATH = "/api/health"
_PUBLIC_PROBE_MARKER = "Unsloth UI Backend"
# One deadline for DNS propagation + the health probe, bounding the startup stall.
_PUBLIC_PROBE_TIMEOUT = 45.0
_PUBLIC_PROBE_ATTEMPT_TIMEOUT = 5.0
_PUBLIC_PROBE_RETRY_DELAY = 1.0

# Resolve via DoH first: an early OS lookup negative-caches the NXDOMAIN for up to 30 min.
_DNS_POLL_DELAY = 2.0
# Retry transient DoH failures, but give up fast when DoH is blocked outright.
_DNS_MAX_DOH_ERRORS = 3
_DOH_URL = "https://cloudflare-dns.com/dns-query?name={host}&type=A"
# The resolver negative-caches a miss of its own, so a query sent before the record can exist blinds the poll
# for that cache's lifetime. Hold off first.
_DNS_INITIAL_GRACE = 3.0
# A blinded poll cannot recover, so bound its share of the shared deadline.
_DNS_WAIT_MAX = 20.0

# Cloudflare's edge routes by TLS SNI, so it serves the tunnel before the hostname resolves anywhere.
_EDGE_HOST = "trycloudflare.com"
_EDGE_PROBE_RETRY_DELAY = 0.5
# Bound the wait so the hostname fallback keeps most of the shared deadline.
_EDGE_WAIT_MAX = 15.0
# A network that blocks the edge blocks every attempt, so stop spending the wait.
_EDGE_MAX_UNREACHABLE = 2


def _windows_hidden_kwargs() -> dict:
    """Suppress a child console window on Windows; no-op elsewhere."""
    if sys.platform != "win32":
        return {}
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    return {"creationflags": flags} if flags else {}


def _lifetime_kwargs() -> dict:
    """Bind cloudflared to the parent's lifetime (Linux PDEATHSIG). Lazy +
    best-effort so this module still loads standalone (storage_roots-style)."""
    try:
        from utils.process_lifetime import child_popen_kwargs
        return child_popen_kwargs()
    except Exception:
        return {}


def _adopt_pid(pid: int) -> None:
    """Record cloudflared so a force quit does not strand it (macOS has no
    PDEATHSIG). Best-effort, like _lifetime_kwargs above."""
    try:
        from utils.process_lifetime import adopt_pid
        adopt_pid(pid)
    except Exception:
        pass


def _forget_pid(pid: int) -> None:
    try:
        from utils.process_lifetime import forget_pid
        forget_pid(pid)
    except Exception:
        pass


def _spawn_child(spawn):
    """Fork on a process-lifetime thread so the PDEATHSIG above means "die with
    the parent process", not "die when the worker thread that forked me returns"."""
    try:
        from utils.process_lifetime import spawn_on_lifetime_thread
    except Exception:
        return spawn()
    return spawn_on_lifetime_thread(spawn)


def _asset_name() -> Optional[Tuple[str, bool]]:
    """(release asset filename, is_tgz) for this OS/arch, or None if unsupported."""
    system = platform.system().lower()
    machine = platform.machine().lower()
    is_x64 = machine in ("x86_64", "amd64", "x64")
    is_arm64 = machine in ("aarch64", "arm64")
    is_x86 = machine in ("i386", "i686", "x86")
    if system == "linux":
        if is_x64:
            return ("cloudflared-linux-amd64", False)
        if is_arm64:
            return ("cloudflared-linux-arm64", False)
    elif system == "darwin":
        if is_arm64:
            return ("cloudflared-darwin-arm64.tgz", True)
        if is_x64:
            return ("cloudflared-darwin-amd64.tgz", True)
    elif system == "windows":
        if is_x64:
            return ("cloudflared-windows-amd64.exe", False)
        if is_x86:
            return ("cloudflared-windows-386.exe", False)
    return None


def _cache_path() -> Optional[Path]:
    """studio_bin_root()/cloudflared(.exe), or None if the studio home is unresolvable."""
    try:
        from utils.paths.storage_roots import studio_bin_root
    except Exception:
        return None
    name = "cloudflared.exe" if sys.platform == "win32" else "cloudflared"
    return studio_bin_root() / name


def find_cloudflared() -> Optional[str]:
    """Locate an existing cloudflared: PATH first, then the Unsloth bin cache."""
    on_path = shutil.which("cloudflared")
    if on_path:
        return on_path
    cached = _cache_path()
    if cached is not None and cached.is_file() and os.access(cached, os.X_OK):
        return str(cached)
    return None


def _download(url: str, dest: Path) -> bool:
    """Download url to dest via urllib (temp file + atomic rename). Best-effort -> bool."""
    import tempfile
    import urllib.request

    tmp_path: Optional[Path] = None
    try:
        dest.parent.mkdir(parents = True, exist_ok = True)
        with tempfile.NamedTemporaryFile(
            prefix = dest.name + ".tmp-", dir = dest.parent, delete = False
        ) as handle:
            tmp_path = Path(handle.name)
            # GitHub's CDN 403s the default Python-urllib User-Agent.
            req = urllib.request.Request(url, headers = {"User-Agent": "unsloth-studio"})
            with urllib.request.urlopen(req, timeout = _DOWNLOAD_TIMEOUT) as response:
                shutil.copyfileobj(response, handle)
        if tmp_path.stat().st_size == 0:
            raise RuntimeError("empty download")
        os.replace(tmp_path, dest)
        return True
    except Exception:
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok = True)
            except Exception:
                pass
        return False


def _extract_tgz_member(tgz_path: Path, dest: Path) -> bool:
    """Extract just the `cloudflared` member from a darwin .tgz to dest.

    Rejects absolute paths and `..` traversal so a hostile archive cannot write
    outside dest. Best-effort -> bool.
    """
    import tarfile
    try:
        with tarfile.open(tgz_path, "r:gz") as tar:
            member = None
            for m in tar.getmembers():
                if not m.isfile() or os.path.basename(m.name) != "cloudflared":
                    continue
                if m.name.startswith("/") or ".." in Path(m.name).parts:
                    continue
                member = m
                break
            if member is None:
                return False
            src = tar.extractfile(member)
            if src is None:
                return False
            with src, open(dest, "wb") as out:
                shutil.copyfileobj(src, out)
        return True
    except Exception:
        return False


def ensure_cloudflared() -> Optional[str]:
    """Return a cloudflared path, downloading + caching the binary once if missing."""
    existing = find_cloudflared()
    if existing:
        return existing
    asset = _asset_name()
    cached = _cache_path()
    if asset is None or cached is None:
        return None
    name, is_tgz = asset
    url = f"{_RELEASE_BASE}/{name}"
    try:
        cached.parent.mkdir(parents = True, exist_ok = True)
        if is_tgz:
            tgz = cached.with_suffix(".tgz")
            if not _download(url, tgz) or not _extract_tgz_member(tgz, cached):
                tgz.unlink(missing_ok = True)
                return None
            tgz.unlink(missing_ok = True)
        elif not _download(url, cached):
            return None
        if sys.platform != "win32":
            os.chmod(cached, 0o755)
        return str(cached)
    except Exception:
        return None


def _wait_for_dns(host: str, deadline: float) -> None:
    import json
    import urllib.request

    now = time.monotonic()
    deadline = min(deadline, now + _DNS_WAIT_MAX)
    if deadline > now:
        time.sleep(min(_DNS_INITIAL_GRACE, deadline - now))
    errors = 0
    while True:
        answered = False
        try:
            req = urllib.request.Request(
                _DOH_URL.format(host = host),
                headers = {"Accept": "application/dns-json", "User-Agent": "unsloth-studio"},
            )
            with urllib.request.urlopen(req, timeout = 5) as response:
                answered = bool(json.loads(response.read(65536)).get("Answer"))
            errors = 0
        except Exception:
            errors += 1
            if errors >= _DNS_MAX_DOH_ERRORS:
                return
        if answered:
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(_DNS_POLL_DELAY, remaining))


def _edge_addresses() -> list:
    """Distinct Cloudflare frontends, from a name that resolves before any tunnel exists."""
    import socket

    addresses = []
    try:
        resolved = socket.getaddrinfo(_EDGE_HOST, 443, type = socket.SOCK_STREAM)
    except Exception:
        return addresses
    for info in resolved:
        address = info[4][0]
        # macOS reports the A records as IPv4-mapped under AF_INET6; the mapped and bare forms are one frontend.
        if address.startswith("::ffff:"):
            address = address[len("::ffff:") :]
        if address not in addresses:
            addresses.append(address)
    return addresses[:2]


def _probe_edge(
    address: str,
    host: str,
    timeout: float = _PUBLIC_PROBE_ATTEMPT_TIMEOUT,
) -> Optional[bool]:
    """Ask the edge for the marker as ``host``. None when the edge is unreachable."""
    import http.client
    import json
    import socket
    import ssl

    request = (
        f"GET {_PUBLIC_PROBE_PATH} HTTP/1.1\r\nHost: {host}\r\n"
        "User-Agent: unsloth-studio\r\nConnection: close\r\n\r\n"
    ).encode()
    try:
        with socket.create_connection((address, 443), timeout = timeout) as raw:
            with ssl.create_default_context().wrap_socket(raw, server_hostname = host) as tls:
                tls.sendall(request)
                response = http.client.HTTPResponse(tls, method = "GET")
                response.begin()
                body = response.read(4096)
    except Exception:
        return None
    try:
        return json.loads(body).get("service") == _PUBLIC_PROBE_MARKER
    except Exception:
        return False


def _verify_through_edge(host: str, deadline: float) -> bool:
    """Verify at the edge, which selects the tunnel by SNI rather than by address.

    Error 1033 and an intercepting proxy's own page are both answers and are not
    told apart here, so only the marker ends the wait. Nothing answering at all
    is this path being blocked, which the hostname may still get through.
    """
    addresses = _edge_addresses()
    if not addresses:
        return False
    deadline = min(deadline, time.monotonic() + _EDGE_WAIT_MAX)
    unreachable = 0
    while True:
        for address in addresses:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            answer = _probe_edge(address, host, min(_PUBLIC_PROBE_ATTEMPT_TIMEOUT, remaining))
            if answer:
                return True
            unreachable = unreachable + 1 if answer is None else 0
            if unreachable >= _EDGE_MAX_UNREACHABLE:
                return False
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(_EDGE_PROBE_RETRY_DELAY, remaining))


def verify_public_url(url: str, timeout: float = _PUBLIC_PROBE_TIMEOUT) -> bool:
    import json
    import urllib.request
    from urllib.parse import urlsplit

    deadline = time.monotonic() + timeout
    host = urlsplit(url).hostname
    if host:
        if _verify_through_edge(host, deadline):
            return True
        # The edge never served the tunnel, so fall back to the hostname and pay the DoH wait that keeps an
        # early OS lookup from caching the miss.
        _wait_for_dns(host, deadline)

    probe_url = f"{url.rstrip('/')}{_PUBLIC_PROBE_PATH}"
    while True:
        # Drain cloudflared's output: capture the first trycloudflare URL and the first edge-connection
        # registration, and keep draining so it never blocks on a full pipe.
        try:
            req = urllib.request.Request(probe_url, headers = {"User-Agent": "unsloth-studio"})
            with urllib.request.urlopen(req, timeout = _PUBLIC_PROBE_ATTEMPT_TIMEOUT) as response:
                body = response.read(4096)
            if json.loads(body).get("service") == _PUBLIC_PROBE_MARKER:
                return True
        except Exception:
            pass
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(_PUBLIC_PROBE_RETRY_DELAY, remaining))


def _process_exited(proc: subprocess.Popen) -> bool:
    try:
        return proc.poll() is not None
    except Exception:
        return False


def _origin_url(host: str, port: int) -> str:
    url_host = host.replace("%", "%25")
    if ":" in url_host and not url_host.startswith("["):
        url_host = f"[{url_host}]"
    return f"http://{url_host}:{port}"


class CloudflareTunnel:
    """A cloudflared quick tunnel to a local Studio endpoint. Best-effort throughout.

    Use a loopback address for wildcard binds so cloudflared's upstream stays
    local-only while matching Studio's active address family.
    """

    def __init__(
        self,
        port: int,
        binary: str,
        protocol: Optional[str] = None,
        origin_host: str = "localhost",
    ):
        self.port = port
        self.binary = binary
        self.origin_host = origin_host
        # None lets cloudflared pick quic; set "http2" to force it when quic is blocked.
        self.protocol = protocol
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._stopped = False
        self._url_event = threading.Event()
        self._ready_event = threading.Event()
        self.url: Optional[str] = None
        self.ready = False
        self.error: Optional[str] = None
        self.on_exit: Optional[Callable[["CloudflareTunnel"], None]] = None
        self._reader_exited = False
        self._runtime_active = False

    def start(self) -> None:
        cmd = [
            self.binary,
            "tunnel",
            "--url",
            _origin_url(self.origin_host, self.port),
            "--no-autoupdate",
        ]
        if self.protocol:
            cmd += ["--protocol", self.protocol]
        with self._lock:
            # Refuse to spawn once a stop() has marked the tunnel stopped: it would orphan a process nobody owns.
            if self._stopped:
                return
            _set_studio_tunnel_runtime_active(self, True)
            try:
                # PDEATHSIG binds to the forking thread, so spawning from the settings worker would kill cloudflared
                # when it returns.
                proc = _spawn_child(
                    lambda: subprocess.Popen(
                        cmd,
                        stdout = subprocess.PIPE,
                        stderr = subprocess.STDOUT,
                        stdin = subprocess.DEVNULL,
                        text = True,
                        encoding = "utf-8",
                        errors = "replace",
                        bufsize = 1,
                        **_windows_hidden_kwargs(),
                        **_lifetime_kwargs(),
                    )
                )
            except Exception:
                _set_studio_tunnel_runtime_active(self, False)
                raise
            # Adopt before dropping the lock: a racing stop() would otherwise forget it and this would record
            # whatever inherited the pid.
            _adopt_pid(proc.pid)
            self._proc = proc
        threading.Thread(
            target = self._reader, args = (proc,), name = "cloudflared-reader", daemon = True
        ).start()

    def _reader(self, proc: subprocess.Popen) -> None:
        try:
            if proc.stdout is not None:
                for line in proc.stdout:
                    # stdout closed -> cloudflared has exited. Record why, and unblock any waiters at once
                    # instead of letting them wait out the full timeout.
                    if self.url is None:
                        match = _URL_RE.search(line)
                        if match:
                            self.url = match.group(0)
                            self._url_event.set()
                    if not self.ready and _REGISTERED_MARKER in line:
                        self.ready = True
                        self._ready_event.set()
        except Exception:
            pass
        finally:
            if self.url is None:
                self.error = "cloudflared exited before emitting a tunnel URL"
            elif not self.ready:
                self.error = "cloudflared exited before the tunnel connection registered"
            else:
                self.error = "cloudflared exited"
            self._url_event.set()
            self._ready_event.set()
            with self._lock:
                self._reader_exited = True
                callback = self.on_exit
            if _process_exited(proc):
                _set_studio_tunnel_runtime_active(self, False)
            if callback is not None:
                callback(self)

    def wait_for_ready(self, timeout: float = _READY_TIMEOUT) -> Optional[str]:
        """Block until the tunnel is actually serving -- the URL has been minted
        *and* at least one edge connection has registered -- or until timeout.

        Returns the URL only when ready, so callers never advertise a URL that
        would return Cloudflare error 1033 (HTTP 530)."""
        self._ready_event.wait(timeout)
        return self.url if self.ready else None

    def stop(self) -> bool:
        """Terminate the tunnel and report whether process exit was confirmed."""
        with self._lock:
            # Mark stopped so a start() racing behind us refuses to spawn.
            self._stopped = True
            proc, self._proc = self._proc, None
        if proc is None:
            active = _studio_tunnel_runtime_active(self)
            if active:
                _retain_studio_tunnel_for_stop(self)
            return not active
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout = 5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    try:
                        proc.wait(timeout = 5)
                    except Exception:
                        pass
        except Exception:
            pass
        if _process_exited(proc):
            _forget_pid(proc.pid)
            _set_studio_tunnel_runtime_active(self, False)
            return True
        else:
            # Preserve both the stop handle and the fail-closed trust state when termination could not be
            # confirmed. A later stop can retry.
            with self._lock:
                if self._proc is None:
                    self._proc = proc
            _retain_studio_tunnel_for_stop(self)
            return False

    def is_running(self) -> bool:
        with self._lock:
            proc = self._proc
        try:
            return proc is not None and proc.poll() is None
        except Exception:
            return False

    def set_on_exit(self, callback: Callable[["CloudflareTunnel"], None]) -> None:
        with self._lock:
            self.on_exit = callback
            reader_exited = self._reader_exited
        if reader_exited:
            callback(self)

    def _publish_if_running(self, callback: Callable[[], None]) -> bool:
        with self._lock:
            try:
                running = (
                    not self._reader_exited and self._proc is not None and self._proc.poll() is None
                )
            except Exception:
                running = False
            if running:
                callback()
            return running


# Single serving process per Unsloth launch, so one module-level tunnel handle is enough; the lock guards the
# start/stop/shutdown races.
_active_tunnel: Optional[CloudflareTunnel] = None
_active_lock = threading.Lock()
_start_lock = threading.Lock()
# Latched by stop_studio_tunnel so a shutdown landing between retries cannot start a tunnel nobody will stop.
_shutdown_requested = False
_tunnel_generation = 0
_tunnel_lifecycle = 0
_accepting_starts = True
_tunnel_state = "off"
_tunnel_owner: Optional[str] = None
_tunnel_url: Optional[str] = None
_tunnel_error: Optional[str] = None
_tunnel_port: Optional[int] = None
_tunnel_url_callback: Optional[Callable[[Optional[str]], None]] = None
_tunnel_runtime_callback: Optional[Callable[[bool], None]] = None
_tunnel_runtime_lock = threading.Lock()
_tunnel_runtime_count = 0
_tunnels_pending_stop = set()
_TUNNEL_OWNERS = frozenset({"launch", "settings", "colab"})


def _set_studio_tunnel_runtime_active(tunnel: CloudflareTunnel, active: bool) -> None:
    global _tunnel_runtime_count
    with _tunnel_runtime_lock:
        if not active:
            _tunnels_pending_stop.discard(tunnel)
        if tunnel._runtime_active == active:
            return
        tunnel._runtime_active = active
        _tunnel_runtime_count += 1 if active else -1
        if _tunnel_runtime_callback is not None:
            try:
                _tunnel_runtime_callback(_tunnel_runtime_count > 0)
            except Exception:
                pass


def _studio_tunnel_runtime_active(tunnel: CloudflareTunnel) -> bool:
    with _tunnel_runtime_lock:
        return tunnel._runtime_active


def _retain_studio_tunnel_for_stop(tunnel: CloudflareTunnel) -> None:
    with _tunnel_runtime_lock:
        if tunnel._runtime_active:
            _tunnels_pending_stop.add(tunnel)


def _tunnels_pending_stop_snapshot() -> tuple:
    with _tunnel_runtime_lock:
        return tuple(_tunnels_pending_stop)


def set_studio_tunnel_runtime_callback(callback: Optional[Callable[[bool], None]]) -> None:
    global _tunnel_runtime_callback
    with _tunnel_runtime_lock:
        _tunnel_runtime_callback = callback
        if callback is not None:
            try:
                callback(_tunnel_runtime_count > 0)
            except Exception:
                pass


def open_studio_tunnel_lifecycle() -> None:
    """Open a new backend lifecycle and invalidate workers from any prior one."""
    global _tunnel_lifecycle, _accepting_starts
    with _active_lock:
        _tunnel_lifecycle += 1
        _accepting_starts = True


def capture_studio_tunnel_start_admission() -> Optional[Tuple[int, int]]:
    """Capture the lifecycle/generation that admitted an asynchronous start."""
    with _active_lock:
        if not _accepting_starts:
            return None
        return (_tunnel_lifecycle, _tunnel_generation)


def get_studio_tunnel_control_token() -> Tuple[int, int]:
    """Return the current lifecycle/generation for worker bookkeeping."""
    with _active_lock:
        return (_tunnel_lifecycle, _tunnel_generation)


def _set_tunnel_url_locked(url: Optional[str]) -> None:
    global _tunnel_url
    _tunnel_url = url
    if _tunnel_url_callback is not None:
        try:
            _tunnel_url_callback(url)
        except Exception:
            pass


def set_studio_tunnel_url_callback(callback: Optional[Callable[[Optional[str]], None]]) -> None:
    global _tunnel_url_callback
    with _active_lock:
        _tunnel_url_callback = callback
        _set_tunnel_url_locked(_tunnel_url)


def get_studio_tunnel_status() -> dict:
    with _active_lock:
        return {
            "state": _tunnel_state,
            "managed_by": _tunnel_owner,
            "url": _tunnel_url,
            "error": _tunnel_error,
            "port": _tunnel_port,
            "stop_pending": bool(_tunnels_pending_stop_snapshot()),
        }


def _set_failed(generation: int, owner: str, port: int, error: str) -> None:
    global _tunnel_state, _tunnel_owner, _tunnel_url, _tunnel_error, _tunnel_port
    with _active_lock:
        if generation != _tunnel_generation or _shutdown_requested:
            return
        _tunnel_state = "error"
        _tunnel_owner = owner
        _set_tunnel_url_locked(None)
        _tunnel_error = error
        _tunnel_port = port


def _active_tunnel_exited(tunnel: CloudflareTunnel) -> None:
    global _active_tunnel, _tunnel_state, _tunnel_owner
    global _tunnel_url, _tunnel_error, _tunnel_port
    with _active_lock:
        if _active_tunnel is not tunnel:
            return
        if _tunnel_state == "stopping":
            return
        generation = _tunnel_generation
        exit_owner, exit_port = _tunnel_owner, _tunnel_port
        exit_error = tunnel.error or "cloudflared exited"
        _tunnel_state = "stopping"
        _set_tunnel_url_locked(None)
        _tunnel_error = None
    stopped = tunnel.stop() is not False
    with _active_lock:
        stopped = stopped or not _studio_tunnel_runtime_active(tunnel)
        if not stopped and (_active_tunnel is None or _active_tunnel is tunnel):
            _active_tunnel = tunnel
            _tunnel_state = "error"
            _tunnel_owner = exit_owner
            _tunnel_error = "cloudflared could not be stopped"
            _tunnel_port = exit_port
        elif generation == _tunnel_generation:
            _active_tunnel = None
            _tunnel_state = "error"
            _tunnel_error = exit_error
        elif stopped and _active_tunnel is tunnel:
            _active_tunnel = None
            _tunnel_state = "off"
            _tunnel_owner = None
            _tunnel_error = None
            _tunnel_port = None


def _set_online_locked(url: str) -> None:
    global _tunnel_state, _tunnel_url, _tunnel_error
    _tunnel_state = "online"
    _set_tunnel_url_locked(url)
    _tunnel_error = None


def start_studio_tunnel(
    port: int,
    timeout: float = _READY_TIMEOUT,
    *,
    managed_by: str = "launch",
    admission: Optional[Tuple[int, int]] = None,
    origin_host: str = "localhost",
) -> Optional[str]:
    """Start a quick tunnel and return its public URL once it is actually
    serving, or None (best-effort).

    Waits for cloudflared to both mint the URL and register an edge connection,
    then fetches /api/health over the public URL, so the caller never advertises
    a link that yields Cloudflare error 1033 (HTTP 530) or an unresolvable host.
    If a URL is minted but no connection registers within the window (e.g. quic
    is blocked on this network), retries once forcing the http2 protocol. On any
    failure the tunnel is stopped and None is returned.
    """
    global _active_tunnel, _shutdown_requested, _tunnel_generation
    global _tunnel_state, _tunnel_owner, _tunnel_url, _tunnel_error, _tunnel_port
    if managed_by not in _TUNNEL_OWNERS:
        raise ValueError(f"Unknown Cloudflare tunnel owner: {managed_by}")
    with _active_lock:
        if not _accepting_starts or _tunnel_state == "stopping" or _tunnels_pending_stop_snapshot():
            return None
        if admission is not None and admission != (_tunnel_lifecycle, _tunnel_generation):
            return None
        requested_generation = _tunnel_generation
    with _start_lock:
        with _active_lock:
            if (
                _tunnel_state == "online"
                and _tunnel_owner == managed_by
                and _tunnel_port == port
                and _active_tunnel is not None
                and getattr(_active_tunnel, "origin_host", "localhost") == origin_host
            ):
                return _tunnel_url
            if (
                not _accepting_starts
                or requested_generation != _tunnel_generation
                or _tunnel_state == "stopping"
                or _tunnels_pending_stop_snapshot()
                or (admission is not None and admission != (_tunnel_lifecycle, _tunnel_generation))
            ):
                return None
            _shutdown_requested = False
            _tunnel_generation += 1
            generation = _tunnel_generation
            prior_at_start, _active_tunnel = _active_tunnel, None
            _tunnel_state = "starting"
            _tunnel_owner = managed_by
            _set_tunnel_url_locked(None)
            _tunnel_error = None
            _tunnel_port = port
        if prior_at_start is not None and prior_at_start.stop() is False:
            with _active_lock:
                if generation == _tunnel_generation:
                    _active_tunnel = prior_at_start
                    _tunnel_state = "error"
                    _tunnel_error = "cloudflared could not be stopped"
            return None

        binary = ensure_cloudflared()
        if not binary:
            _set_failed(generation, managed_by, port, "cloudflared is unavailable")
            return None

        for protocol in (None, "http2"):
            with _active_lock:
                if _shutdown_requested or generation != _tunnel_generation:
                    _active_tunnel = None
                    return None
                tunnel = CloudflareTunnel(
                    port,
                    binary,
                    protocol = protocol,
                    origin_host = origin_host,
                )
                prior, _active_tunnel = _active_tunnel, tunnel
            if prior is not None and prior.stop() is False:
                with _active_lock:
                    if generation == _tunnel_generation and _active_tunnel is tunnel:
                        _active_tunnel = prior
                        _tunnel_state = "error"
                        _tunnel_error = "cloudflared could not be stopped"
                return None
            registered = False
            try:
                tunnel.start()
                url = tunnel.wait_for_ready(timeout)
                registered = url is not None
                if url and not verify_public_url(url):
                    url = None
            except Exception:
                url = None
            if url:
                if hasattr(tunnel, "set_on_exit"):
                    tunnel.set_on_exit(_active_tunnel_exited)
                else:
                    tunnel.on_exit = _active_tunnel_exited
                aborted = False
                with _active_lock:
                    if (
                        generation != _tunnel_generation
                        or _shutdown_requested
                        or _active_tunnel is not tunnel
                    ):
                        # Detach and tear down: returning this URL would leave a live public tunnel no later
                        # stop_studio_tunnel() can reach.
                        aborted = True
                        was_active = False
                        if _active_tunnel is tunnel:
                            _active_tunnel = None
                    else:
                        if hasattr(tunnel, "_publish_if_running"):
                            running = tunnel._publish_if_running(lambda: _set_online_locked(url))
                        else:
                            _set_online_locked(url)
                            running = True
                        if running:
                            was_active = True
                        else:
                            _active_tunnel = None
                            _tunnel_state = "error"
                            _set_tunnel_url_locked(None)
                            _tunnel_error = tunnel.error or "cloudflared exited"
                            was_active = False
                if aborted or not was_active:
                    tunnel.stop()
                    return None
                if hasattr(tunnel, "is_running") and not tunnel.is_running():
                    _active_tunnel_exited(tunnel)
                    return None
                return url
            saw_url = tunnel.url is not None
            with _active_lock:
                was_active = _active_tunnel is tunnel
                if was_active:
                    _tunnel_state = "stopping"
            stopped = tunnel.stop() is not False
            with _active_lock:
                stopped = stopped or not _studio_tunnel_runtime_active(tunnel)
                if _shutdown_requested and _active_tunnel is tunnel:
                    if stopped:
                        _active_tunnel = None
                        _tunnel_state = "off"
                        _tunnel_owner = None
                        _tunnel_error = None
                        _tunnel_port = None
                    else:
                        _tunnel_state = "error"
                        _tunnel_error = "cloudflared could not be stopped"
                elif generation == _tunnel_generation and _active_tunnel is tunnel:
                    if stopped:
                        _active_tunnel = None
                        # Reset from "stopping" before the http2 retry, or stop_studio_tunnel() early-returns and stops
                        # nothing.
                        if _tunnel_state == "stopping" and protocol is None:
                            _tunnel_state = "starting"
                    else:
                        _active_tunnel = tunnel
                        _tunnel_state = "error"
                        _tunnel_error = "cloudflared could not be stopped"
            if not was_active:
                return None
            if not stopped:
                return None
            if not saw_url:
                _set_failed(generation, managed_by, port, "cloudflared did not produce a URL")
                return None
            if registered:
                _set_failed(generation, managed_by, port, "Cloudflare URL was not reachable")
                return None
        _set_failed(generation, managed_by, port, "cloudflared did not register a connection")
        return None


def stop_studio_tunnel(*, admission: Optional[Tuple[int, int]] = None) -> None:
    """Terminate the active tunnel, if any. Idempotent."""
    global _active_tunnel, _shutdown_requested, _tunnel_generation
    global _tunnel_state, _tunnel_owner, _tunnel_url, _tunnel_error, _tunnel_port
    with _active_lock:
        if admission is not None and admission != (_tunnel_lifecycle, _tunnel_generation):
            return
        if _tunnel_state == "stopping":
            # Latch so an in-flight start_studio_tunnel won't start a fresh tunnel (e.g. its http2 retry) after
            # we have already torn down.
            _shutdown_requested = True
            _tunnel_generation += 1
            return
        _shutdown_requested = True
        _tunnel_generation += 1
        stop_generation = _tunnel_generation
        tunnel = _active_tunnel
        pending = _tunnels_pending_stop_snapshot()
        tunnels = list(dict.fromkeys(((tunnel,) if tunnel is not None else ()) + pending))
        _tunnel_state = "stopping" if tunnels else "off"
        _set_tunnel_url_locked(None)
        _tunnel_error = None
    for candidate in tunnels:
        candidate.stop()
    with _active_lock:
        if stop_generation == _tunnel_generation or _tunnel_state == "stopping":
            pending = _tunnels_pending_stop_snapshot()
            if pending:
                _active_tunnel = pending[0]
                _tunnel_state = "error"
                _tunnel_error = "cloudflared could not be stopped"
            else:
                _active_tunnel = None
                _tunnel_state = "off"
                _tunnel_owner = None
                _tunnel_port = None


def close_studio_tunnel_lifecycle() -> None:
    """Permanently reject queued starts for this backend lifecycle, then stop."""
    global _tunnel_lifecycle, _accepting_starts
    with _active_lock:
        _accepting_starts = False
        _tunnel_lifecycle += 1
    stop_studio_tunnel()
