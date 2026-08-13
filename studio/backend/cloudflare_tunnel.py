# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Cloudflare tunnel lifecycle for Unsloth's remote Studio access.

The raw http://<ip>:<port> is often unreachable (https-vs-http, blocked ports,
closed security groups). Launches default to a free Temporary tunnel; settings can
instead run a previously provisioned custom tunnel at the user's custom hostname.

Best-effort throughout: any failure collapses to "no URL" and Unsloth keeps
running. Stdlib only (back-end imports are lazy) so it is safe to import early.
"""

from __future__ import annotations

import errno
import hashlib
import json
import logging
import os
import platform
import re
import secrets
import shutil
import signal
import subprocess
import sys
import threading
import time
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Callable, Optional, Tuple
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)

# cloudflared logs the temporary-tunnel URL; match only the URL so we do not depend
# on the surrounding wording, which Cloudflare may change. The negative lookahead
# drops cloudflared's own API host, which appears in failure lines such as
#   failed to request quick Tunnel: Post "https://api.trycloudflare.com/tunnel"
# and must never be mistaken for a usable tunnel URL.
_URL_RE = re.compile(r"https://(?!api\.)[A-Za-z0-9-]+\.trycloudflare\.com")

# cloudflared logs this once per edge connection it establishes. Until at least
# one appears the temporary-tunnel URL returns Cloudflare error 1033 (HTTP 530), so
# we wait for it before advertising the URL.
_REGISTERED_MARKER = "Registered tunnel connection"

_METRICS_RE = re.compile(r"metrics server on (\[[0-9a-fA-F:]+\]:[0-9]+|[0-9.]+:[0-9]+)")

_RELEASE_BASE = "https://github.com/cloudflare/cloudflared/releases/latest/download"

_READY_TIMEOUT = 15.0  # seconds to wait for the URL + a registered edge connection
# cloudflared drains in-flight requests for thirty seconds before it goes.
_CONNECTOR_DRAIN_WAIT = 35.0
_DOWNLOAD_TIMEOUT = 60  # urlopen timeout for the one-time binary download

# A registered edge connection does not mean the hostname resolves yet, so the
# URL is fetched once before it is advertised.
_PUBLIC_PROBE_PATH = "/api/health"
_PUBLIC_PROBE_MARKER = "Unsloth UI Backend"
# One deadline for DNS propagation + the health probe, bounding the startup stall.
_PUBLIC_PROBE_TIMEOUT = 45.0
_PUBLIC_PROBE_ATTEMPT_TIMEOUT = 5.0
_PUBLIC_PROBE_RETRY_DELAY = 1.0

# Wait for the hostname via DoH first: an early OS lookup negative-caches the
# NXDOMAIN for up to 30 min.
_DNS_POLL_DELAY = 2.0
# Retry transient DoH failures, but give up fast when DoH is blocked outright.
_DNS_MAX_DOH_ERRORS = 3
_DNS_ATTEMPT_TIMEOUT = 5.0
_DOH_URL = "https://cloudflare-dns.com/dns-query?name={host}&type=A"
# The resolver negative-caches a miss of its own, so a query sent before the
# record can exist blinds the poll for that cache's lifetime. Hold off first.
_DNS_INITIAL_GRACE = 3.0
# A blinded poll cannot recover, so bound its share of the shared deadline.
_DNS_WAIT_MAX = 20.0
# Outlive typical negative DNS caches before giving up on propagation.
_DNS_WATCH_TOTAL = 2100.0

# Cloudflare's edge routes by TLS SNI, so it serves the tunnel as soon as the
# connection registers -- before the hostname resolves anywhere. Probing there
# keeps DNS off the startup path entirely.
_EDGE_HOST = "trycloudflare.com"
_EDGE_PROBE_RETRY_DELAY = 0.5
# Bound the wait so the hostname fallback keeps most of the shared deadline.
_EDGE_WAIT_MAX = 15.0
# A network that blocks the edge blocks every attempt, so stop spending the wait.
_EDGE_MAX_UNREACHABLE = 2

# Tunnel kind is independent of its launch/settings owner.
TUNNEL_KINDS = frozenset({"temporary", "custom"})


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
        from utils.paths.storage_roots import studio_bin_root  # lazy: backend-only import
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


def _wait_for_dns(host: str, deadline: float) -> Optional[bool]:
    import urllib.request

    now = time.monotonic()
    deadline = min(deadline, now + _DNS_WAIT_MAX)
    if deadline > now:
        time.sleep(min(_DNS_INITIAL_GRACE, deadline - now))
    errors = 0
    heard_negative = False
    while True:
        resolved = False
        if time.monotonic() >= deadline:
            return False if heard_negative else None
        try:
            req = urllib.request.Request(
                _DOH_URL.format(host = host),
                headers = {"Accept": "application/dns-json", "User-Agent": "unsloth-studio"},
            )
            attempt = max(0.0, min(_DNS_ATTEMPT_TIMEOUT, deadline - time.monotonic()))
            with urllib.request.urlopen(req, timeout = attempt) as response:
                payload = json.loads(response.read(65536))
            status = payload.get("Status")
            # Only NOERROR and NXDOMAIN are authoritative DNS answers.
            if status not in (0, 3):
                raise RuntimeError("resolver could not answer")
            resolved = status == 0 and bool(payload.get("Answer"))
            heard_negative = heard_negative or not resolved
            errors = 0
        except Exception:
            errors += 1
            if errors >= _DNS_MAX_DOH_ERRORS:
                return None
        if resolved:
            return True
        time.sleep(max(0.0, min(_DNS_POLL_DELAY, deadline - time.monotonic())))


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
        # macOS reports the A records as IPv4-mapped under AF_INET6. The mapped
        # and bare forms are one frontend, and probing it twice buys nothing.
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
                status = response.status
                body = response.read(4096)
    except Exception:
        return None
    if status != 200:
        return False
    try:
        return json.loads(body).get("service") == _PUBLIC_PROBE_MARKER
    except Exception:
        return False


def _hostname_of(url: str) -> Optional[str]:
    return urlsplit(url).hostname


def _connector_reports_ready(address: str, timeout: float) -> bool:
    import urllib.request

    # Never proxy cloudflared's loopback readiness endpoint.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(f"http://{address}/ready", timeout = timeout) as response:
            body = json.loads(response.read(4096))
    except Exception:
        return False
    return int(body.get("readyConnections") or 0) > 0


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
    import urllib.request

    deadline = time.monotonic() + timeout
    host = urlsplit(url).hostname
    if host:
        if _verify_through_edge(host, deadline):
            return True
        # The edge never served the tunnel, so fall back to the hostname and pay
        # the DoH wait that keeps an early OS lookup from caching the miss.
        _wait_for_dns(host, deadline)

    probe_url = f"{url.rstrip('/')}{_PUBLIC_PROBE_PATH}"
    while True:
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


def temporary_tunnel_args(port: int, protocol: Optional[str] = None) -> list:
    args = ["tunnel", "--url", f"http://localhost:{port}", "--no-autoupdate"]
    if protocol:
        args += ["--protocol", protocol]
    return args


def _end_unrecorded_connector(proc: subprocess.Popen) -> bool:
    with suppress(Exception):
        proc.terminate()
        try:
            proc.wait(timeout = _CONNECTOR_DRAIN_WAIT)
        except subprocess.TimeoutExpired:
            proc.kill()
            with suppress(Exception):
                proc.wait(timeout = 5)
    return _process_exited(proc)


def write_custom_ingress(
    port: int,
    identity: dict,
    protocol: Optional[str] = None,
) -> Path:
    path = state_root() / "ingress.yml"
    # Run by UUID because provisioning deletes the account certificate.
    lines = [
        f"tunnel: {identity['tunnel_id']}",
        f"credentials-file: {json.dumps(str(identity['credentials']))}",
    ]
    if protocol:
        lines.append(f"protocol: {protocol}")
    lines += [
        "ingress:",
        f"  - hostname: {identity['hostname']}",
        f"    service: http://localhost:{port}",
        "  - service: http_status:404",
    ]
    path.write_text("\n".join(lines) + "\n", encoding = "utf-8")
    with suppress(OSError):
        os.chmod(path, 0o600)
    return path


def custom_tunnel_args(config_path: Path, tunnel_id: str) -> list:
    return [
        "tunnel",
        "--config",
        str(config_path),
        "--no-autoupdate",
        "--metrics",
        "127.0.0.1:0",
        "run",
        tunnel_id,
    ]


class CloudflareTunnel:
    """A running cloudflared connector. Best-effort throughout.

    Kinds differ only in `args` and in whether they know `url` up front; a temporary
    tunnel does not, because cloudflared mints the hostname.
    """

    def __init__(
        self,
        port: int,
        binary: str,
        protocol: Optional[str] = None,
        *,
        kind: str = "temporary",
        args: Optional[list] = None,
        url: Optional[str] = None,
        reservation = None,
    ):
        if kind not in TUNNEL_KINDS:
            raise ValueError(f"Unknown Cloudflare tunnel kind: {kind}")
        if kind == "custom":
            if args is None or url is None:
                raise ValueError(f"Cloudflare tunnel kind {kind} does not match its arguments")
        elif args is not None:
            raise ValueError(f"Cloudflare tunnel kind {kind} does not match its arguments")
        self.port = port
        self.binary = binary
        self.kind = kind
        self._args = list(args) if args is not None else temporary_tunnel_args(port, protocol)
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._stopped = False
        self._url_event = threading.Event()
        self._ready_event = threading.Event()
        self._exit_event = threading.Event()
        self.url: Optional[str] = url
        self.ready = False
        self._metrics_address: Optional[str] = None
        self.error: Optional[str] = None
        self.on_exit: Optional[Callable[["CloudflareTunnel"], None]] = None
        self._reader_exited = False
        self._runtime_active = False
        self._reservation = reservation

    def _env(self) -> Optional[dict]:
        if self.kind != "custom":
            return None
        if self._reservation is None:
            raise RuntimeError("a custom Cloudflare tunnel needs a connector reservation")

        return _child_env(self._reservation.token)

    def _release_reservation(self) -> None:
        with self._lock:
            reservation, self._reservation = self._reservation, None
        if reservation is not None:
            with suppress(Exception):
                reservation.release()

    def start(self) -> None:
        cmd = [self.binary, *self._args]
        unrecorded = failure = None
        with self._lock:
            # A stop() that landed before us (e.g. a shutdown in the caller's
            # register->start window) marks the tunnel stopped; spawning now would
            if self._stopped:
                return
            env = self._env()
            _set_studio_tunnel_runtime_active(self, True)
            try:
                # PDEATHSIG binds to the forking thread, so spawning from the
                # settings start worker would kill cloudflared when it returned.
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
                        env = env,
                        **_windows_hidden_kwargs(),
                        **_lifetime_kwargs(),
                    )
                )
            except Exception:
                _set_studio_tunnel_runtime_active(self, False)
                raise
            if self.kind == "custom":
                try:
                    self._reservation.record_connector(proc.pid)
                except Exception as error:
                    unrecorded, failure = proc, error
            if unrecorded is None:
                # Adopted before the lock drops: a stop() that got in first would
                # otherwise reap and forget it while nothing was tracked, and this
                # would then record whatever inherited the pid.
                _adopt_pid(proc.pid)
                self._proc = proc
        if unrecorded is not None:
            self._settle_connector(unrecorded, _end_unrecorded_connector(unrecorded))
            raise failure
        threading.Thread(
            target = self._reader, args = (proc,), name = "cloudflared-reader", daemon = True
        ).start()

    def _reader(self, proc: subprocess.Popen) -> None:
        # Drain cloudflared's output: capture the first trycloudflare URL and the
        # first edge-connection registration, and keep draining so it never
        # blocks on a full pipe.
        try:
            if proc.stdout is not None:
                for line in proc.stdout:
                    if self.url is None:
                        match = _URL_RE.search(line)
                        if match:
                            self.url = match.group(0)
                            self._url_event.set()
                    if self._metrics_address is None:
                        found = _METRICS_RE.search(line)
                        if found:
                            self._metrics_address = found.group(1)
                    if not self.ready and _REGISTERED_MARKER in line:
                        self.ready = True
                        self._ready_event.set()
        except Exception:
            pass
        finally:
            # stdout closed -> cloudflared has exited. Record why, and unblock any
            # waiters at once instead of letting them wait out the full timeout.
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
            self._exit_event.set()
            if _process_exited(proc):
                _set_studio_tunnel_runtime_active(self, False)
            if callback is not None:
                callback(self)

    def wait_for_ready(self, timeout: float = _READY_TIMEOUT) -> Optional[str]:
        """Block until the tunnel is actually serving -- the URL has been minted
        *and* at least one edge connection has registered -- or until timeout.

        Returns the URL only when ready, so callers never advertise a URL that
        would return Cloudflare error 1033 (HTTP 530)."""
        deadline = time.monotonic() + timeout
        self._ready_event.wait(timeout)
        if not self.ready or self.url is None:
            return None
        while True:
            with self._lock:
                done = self._reader_exited or self._stopped
            remaining = deadline - time.monotonic()
            if done or remaining <= 0:
                return None
            address = self._metrics_address
            serving = address is not None and _connector_reports_ready(
                address,
                min(remaining, _PUBLIC_PROBE_ATTEMPT_TIMEOUT),
            )
            remaining = deadline - time.monotonic()
            with self._lock:
                if self._reader_exited or self._stopped or remaining <= 0:
                    return None
                if serving:
                    return self.url
            if self._exit_event.wait(min(_EDGE_PROBE_RETRY_DELAY, remaining)):
                return None

    def stop(self) -> bool:
        """Terminate the tunnel and report whether process exit was confirmed."""
        with self._lock:
            # Mark stopped so a start() racing behind us refuses to spawn.
            self._stopped = True
            proc, self._proc = self._proc, None
        self._exit_event.set()
        self._url_event.set()
        self._ready_event.set()
        if proc is None:
            active = _studio_tunnel_runtime_active(self)
            if active:
                _retain_studio_tunnel_for_stop(self)
            else:
                self._release_reservation()
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
        return self._settle_connector(proc, _process_exited(proc))

    def _settle_connector(self, proc: subprocess.Popen, confirmed: bool) -> bool:
        if confirmed:
            _set_studio_tunnel_runtime_active(self, False)
            _forget_pid(proc.pid)
            self._release_reservation()
            return True
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


# Single serving process per Unsloth launch, so one module-level tunnel handle is
# enough; the lock guards the start/stop/shutdown races.
_active_tunnel: Optional[CloudflareTunnel] = None
_active_lock = threading.Lock()
_start_lock = threading.Lock()
# Latched by stop_studio_tunnel so a shutdown landing *between* a start's retry
# attempts aborts the loop instead of starting a tunnel nobody will ever stop.
_shutdown_requested = False
_tunnel_generation = 0
# DNS observations belong only to the tunnel generation that requested them.
_tunnel_dns: Tuple[int, str] = (0, "unknown")
_tunnel_lifecycle = 0
_accepting_starts = True
_tunnel_state = "off"
_tunnel_owner: Optional[str] = None
_tunnel_kind: Optional[str] = None
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
        dns = _tunnel_dns[1] if _tunnel_dns[0] == _tunnel_generation else "unknown"
        return {
            "state": _tunnel_state,
            "managed_by": _tunnel_owner,
            "kind": _tunnel_kind,
            "url": _tunnel_url,
            "error": _tunnel_error,
            "port": _tunnel_port,
            "stop_pending": bool(_tunnels_pending_stop_snapshot()),
            "connector_registered": bool(
                _active_tunnel and getattr(_active_tunnel, "ready", False)
            ),
            "tunnel_serving": _tunnel_state == "online",
            "dns": dns,
            # Whether the URL can be handed to anyone yet. A custom hostname
            # serves only once its record resolves, and every caller that
            # offers the link was deciding that for itself.
            "url_usable": _tunnel_kind != "custom" or dns == "resolved",
        }


def _watch_wanted(generation: int, give_up: float, url: str) -> bool:
    recorded_generation, recorded_state = _tunnel_dns
    return (
        time.monotonic() < give_up
        and _tunnel_url == url
        and generation == _tunnel_generation
        and not (recorded_generation == generation and recorded_state == "resolved")
    )


def _watch_hostname_resolution(url: str, generation: int) -> None:
    global _tunnel_dns
    host = _hostname_of(url)
    give_up = time.monotonic() + _DNS_WATCH_TOTAL
    while True:
        with _active_lock:
            if not _watch_wanted(generation, give_up, url):
                return
        resolved = (
            _wait_for_dns(host, min(give_up, time.monotonic() + _DNS_WAIT_MAX)) if host else None
        )
        state = {True: "resolved", False: "pending"}.get(resolved, "unknown")
        with _active_lock:
            if not _watch_wanted(generation, give_up, url):
                return
            _tunnel_dns = (generation, state)
        if resolved is True or not host:
            return
        time.sleep(max(0.0, min(_DNS_POLL_DELAY, give_up - time.monotonic())))


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
    global _active_tunnel, _tunnel_state, _tunnel_owner, _tunnel_kind
    global _tunnel_url, _tunnel_error, _tunnel_port, _tunnel_generation
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
        was_current = generation == _tunnel_generation
        ended = False
        if not stopped and (_active_tunnel is None or _active_tunnel is tunnel):
            _active_tunnel = tunnel
            _tunnel_state = "error"
            _tunnel_owner = exit_owner
            _tunnel_error = "cloudflared could not be stopped"
            _tunnel_port = exit_port
        elif was_current:
            ended = True
            _active_tunnel = None
            _tunnel_state = "error"
            _tunnel_error = exit_error
        elif stopped and _active_tunnel is tunnel:
            ended = True
            _active_tunnel = None
            _tunnel_state = "off"
            _tunnel_owner = None
            _tunnel_kind = None
            _tunnel_error = None
            _tunnel_port = None
        if was_current and ended:
            _tunnel_generation += 1


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
    kind: str = "temporary",
) -> Optional[str]:
    """Start the requested tunnel kind and return its URL once it is serving.

    Waits for an edge connection, then fetches /api/health over the public URL,
    so the caller never advertises a link that yields Cloudflare error 1033. A
    Temporary tunnel must first mint its URL and may retry with HTTP/2; a custom tunnel
    has a configured URL and makes one connector attempt. Any failure returns None.
    """
    global _active_tunnel, _shutdown_requested, _tunnel_generation
    global _tunnel_state, _tunnel_owner, _tunnel_url, _tunnel_error, _tunnel_port
    global _tunnel_kind
    if managed_by not in _TUNNEL_OWNERS:
        raise ValueError(f"Unknown Cloudflare tunnel owner: {managed_by}")
    if kind not in TUNNEL_KINDS:
        raise ValueError(f"Unknown Cloudflare tunnel kind: {kind}")
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
                and _tunnel_kind == kind
                and _tunnel_port == port
                and _active_tunnel is not None
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
            prior_owner, prior_kind, prior_port = _tunnel_owner, _tunnel_kind, _tunnel_port
            _tunnel_state = "starting"
            _tunnel_owner = managed_by
            _tunnel_kind = kind
            _set_tunnel_url_locked(None)
            _tunnel_error = None
            _tunnel_port = port
        if prior_at_start is not None and prior_at_start.stop() is False:
            with _active_lock:
                if generation == _tunnel_generation:
                    _active_tunnel = prior_at_start
                    _tunnel_state = "error"
                    _tunnel_owner, _tunnel_kind, _tunnel_port = prior_owner, prior_kind, prior_port
                    _tunnel_error = "cloudflared could not be stopped"
            return None

        binary = ensure_cloudflared()
        if not binary:
            _set_failed(generation, managed_by, port, "cloudflared is unavailable")
            return None

        identity = None
        if kind == "custom":
            identity = read_identity()
            if not identity_is_runnable(identity):
                _set_failed(
                    generation, managed_by, port, "Custom Cloudflare tunnel is not configured"
                )
                return None

        for protocol in (None,) if kind == "custom" else (None, "http2"):
            prepared = None
            if kind == "custom":
                reservation = reserve_connector()
                if reservation is None:
                    _set_failed(
                        generation,
                        managed_by,
                        port,
                        "Custom Cloudflare connector is already running",
                    )
                    return None
                try:
                    config = write_custom_ingress(port, identity, protocol)
                    prepared = CloudflareTunnel(
                        port,
                        binary,
                        kind = kind,
                        args = custom_tunnel_args(config, identity["tunnel_id"]),
                        url = f"https://{identity['hostname']}",
                        reservation = reservation,
                    )
                except Exception:
                    reservation.release()
                    _set_failed(
                        generation,
                        managed_by,
                        port,
                        "Custom Cloudflare tunnel could not be prepared",
                    )
                    return None
            cancelled = False
            with _active_lock:
                if _shutdown_requested or generation != _tunnel_generation:
                    _active_tunnel = None
                    cancelled = True
                else:
                    tunnel = prepared or CloudflareTunnel(port, binary, protocol = protocol)
                    prior, _active_tunnel = _active_tunnel, tunnel
            if cancelled:
                if prepared is not None:
                    prepared.stop()
                return None
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
                        # Stop/shutdown landed while coming up. Detach AND tear
                        # down: returning this URL would leave a live public
                        # tunnel no later stop_studio_tunnel() can reach.
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
                threading.Thread(
                    target = _watch_hostname_resolution,
                    args = (url, generation),
                    name = "tunnel-dns-watch",
                    daemon = True,
                ).start()
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
                        _tunnel_kind = None
                        _tunnel_error = None
                        _tunnel_port = None
                    else:
                        _tunnel_state = "error"
                        _tunnel_error = "cloudflared could not be stopped"
                elif generation == _tunnel_generation and _active_tunnel is tunnel:
                    if stopped:
                        _active_tunnel = None
                        # Attempt 1's teardown parked the state at "stopping".
                        # Leaving it there for the http2 retry makes
                        # stop_studio_tunnel() early-return and stop nothing.
                        if _tunnel_state == "stopping" and kind == "temporary" and protocol is None:
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
                error = (
                    "custom_hostname_unreachable"
                    if kind == "custom"
                    else "Cloudflare URL was not reachable"
                )
                _set_failed(generation, managed_by, port, error)
                return None
        _set_failed(generation, managed_by, port, "cloudflared did not register a connection")
        return None


def stop_studio_tunnel(
    *, admission: Optional[Tuple[int, int]] = None, kind: Optional[str] = None
) -> None:
    """Terminate the active tunnel, if any. Idempotent."""
    global _active_tunnel, _shutdown_requested, _tunnel_generation
    global _tunnel_state, _tunnel_owner, _tunnel_url, _tunnel_error, _tunnel_port
    global _tunnel_kind
    with _active_lock:
        if admission is not None and admission != (_tunnel_lifecycle, _tunnel_generation):
            return
        if kind is not None and _tunnel_kind != kind:
            return
        if _tunnel_state == "stopping":
            _shutdown_requested = True
            _tunnel_generation += 1
            return
        # Latch so an in-flight start_studio_tunnel won't start a fresh tunnel
        # (e.g. its http2 retry) after we have already torn down.
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
                _tunnel_kind = None
                _tunnel_port = None


def close_studio_tunnel_lifecycle() -> None:
    """Permanently reject queued starts for this backend lifecycle, then stop."""
    global _tunnel_lifecycle, _accepting_starts
    with _active_lock:
        _accepting_starts = False
        _tunnel_lifecycle += 1
    stop_studio_tunnel()


_RECORD = "provisioning.json"
_IDENTITY = "identity.json"
_ORPHANS = "orphans.json"
_ABANDONED = "abandoned-tunnels.json"
_CONNECTOR = "connector.json"

_HOSTNAME_LABEL_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")

_NAME_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
_NAME_RE = re.compile(r"^unsloth-[A-Z0-9]{6}$")

# cloudflared exposes these failure classes only in command output.
_CREATE_COLLISION = "tunnel with name already exists"
_ROUTE_CONFLICT = "already exists"
_ROUTE_AUTHORIZATION_FAILURES = (
    "failed to find zone",
    "zone could not be found",
)
_LOGIN_REFUSED = "which login would overwrite"
_LOGIN_REFUSED_RE = re.compile(r"existing certificate at (.+?) which login would overwrite")
_TUNNEL_ABSENT = "non-deleted tunnel named"
_LOGIN_URL_RE = re.compile(r"https://\S*cloudflare\.com/argotunnel\S*")
_TUNNEL_ID_RE = re.compile(r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}")
_CREDENTIALS_RE = re.compile(r"credentials written to (.+?\.json)", re.IGNORECASE)
_ROUTE_ADVICE = "overwrite any existing DNS records for this hostname."
_ROUTE_ADDED_RE = re.compile(r"added cname (\S+) which will route", re.IGNORECASE)

_CLI_TIMEOUT = 60.0
# cloudflared drains in-flight requests for thirty seconds before it goes.
_CONNECTOR_EXIT_WAIT = 35.0
_LOGIN_DEADLINE = 900.0

# Fall back only when the platform explicitly lacks file locking.
_LOCK_UNSUPPORTED = frozenset({errno.ENOTSUP, errno.EOPNOTSUPP})

# Keep the Windows lock past the readable holder message.
_LOCK_OFFSET = 1 << 30

_TOKEN_VAR = "UNSLOTH_CHILD_TOKEN"

_claim_lock = threading.Lock()
_connector_lock = threading.Lock()


class ProvisioningError(Exception):
    """Report a provisioning step failure with a stable error code."""

    def __init__(
        self,
        code,
        detail = "",
    ):
        super().__init__(detail or code)
        self.code, self.detail = code, detail


def origin_cert_path() -> Path:
    return Path.home() / ".cloudflared" / "cert.pem"


def _claim_path() -> Path:
    return Path.home() / ".unsloth" / "cloudflare" / "claim.lock"


def _state_dir() -> Path:
    from utils.paths.storage_roots import studio_root
    return studio_root() / "cloudflare"


def _writable_dir(path: Path) -> Path:
    path.mkdir(parents = True, exist_ok = True)
    _chmod(path, 0o700)
    return path


def state_root() -> Path:
    return _writable_dir(_state_dir())


def _chmod(path: Path, mode: int) -> None:
    with suppress(OSError):
        os.chmod(path, mode)


def _unlink(path: Path, *, required: bool = False) -> None:
    if required:
        path.unlink(missing_ok = True)
    else:
        with suppress(OSError):
            path.unlink(missing_ok = True)


def _read(name: str) -> Optional[object]:
    try:
        return json.loads((_state_dir() / name).read_text(encoding = "utf-8"))
    except (OSError, ValueError):
        return None


def _write(name: str, payload: object) -> None:
    path = _writable_dir(_state_dir()) / name
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(json.dumps(payload), encoding = "utf-8")
        _chmod(tmp, 0o600)
        os.replace(tmp, path)
    finally:
        _unlink(tmp)


def _discard(name: str, *, required: bool = False) -> None:
    _unlink(_state_dir() / name, required = required)


def _string_list(name: str) -> list:
    record = _read(name)
    return [item for item in record if isinstance(item, str)] if isinstance(record, list) else []


def read_identity() -> Optional[dict]:
    record = _read(_IDENTITY)
    return record if isinstance(record, dict) and record.get("hostname") else None


def identity_is_runnable(identity: object = None) -> bool:
    record = read_identity() if identity is None else identity
    if not isinstance(record, dict):
        return False
    required = (record.get("hostname"), record.get("tunnel_name"), record.get("tunnel_id"))
    credentials = record.get("credentials")
    return all(isinstance(value, str) and value for value in required) and (
        isinstance(credentials, str) and Path(credentials).is_file()
    )


def orphaned_hostnames() -> list:
    return _string_list(_ORPHANS)


def _set_orphan(hostname: str, *, stranded: bool) -> None:
    orphans = _string_list(_ORPHANS)
    if stranded == (hostname in orphans):
        return
    _write(_ORPHANS, orphans + [hostname] if stranded else [h for h in orphans if h != hostname])


def canonical_hostname(raw: str) -> str:
    """Normalize a hostname to lowercase A-label form without a trailing dot."""
    text = (raw or "").strip()
    invalid = ProvisioningError("invalid_hostname", "Enter a hostname like unsloth.example.com.")
    try:
        # Only a pasted URL is parsed as one. Splitting a bare string treats
        # userinfo and ports as delimiters and returns the shorter name that
        # survives them, so a mistyped "studio@example.com" would provision the
        # apex domain instead of being refused.
        host = (urlsplit(text).hostname or "") if "://" in text else text
    except ValueError:
        raise invalid from None
    labels = []
    for label in host.strip(".").split("."):
        if not label:
            raise invalid
        try:
            ascii_label = label if label.isascii() else label.encode("idna").decode("ascii")
        except UnicodeError:
            raise invalid from None
        if not label.isascii():
            # The stdlib codec is IDNA 2003, which maps some characters away
            # instead of encoding them: "faß" becomes "fass", a separate
            # registration. Refusing beats provisioning a domain nobody entered,
            # and the A-label can be pasted directly.
            try:
                round_trip = ascii_label.encode("ascii").decode("idna")
            except UnicodeError:
                raise invalid from None
            if round_trip != label.lower():
                raise invalid
        ascii_label = ascii_label.lower()
        if not _HOSTNAME_LABEL_RE.match(ascii_label):
            raise invalid
        labels.append(ascii_label)
    name = ".".join(labels)
    if len(labels) < 2 or len(name) > 253:
        raise invalid
    return name


def _new_token() -> str:
    return secrets.token_hex(16)


def _process_token(pid: Optional[int]) -> Optional[str]:
    if not _addressable(pid):
        return None
    try:
        import psutil
        return psutil.Process(pid).environ().get(_TOKEN_VAR) or None
    except Exception:
        return None


# Refuse to signal a PID unless its recorded identity still matches.
def _same_process(recorded: object, pid: Optional[int]) -> bool:
    actual = _process_token(pid)
    return actual is not None and actual == recorded


def _addressable(pid: object) -> bool:
    return isinstance(pid, int) and pid >= 2


def _pid_alive(pid: object) -> bool:
    if not _addressable(pid):
        return False
    try:
        import psutil
        return psutil.pid_exists(pid)
    except Exception:
        # Not os.kill(pid, 0): on Windows signal 0 is CTRL_C_EVENT, not a probe.
        return True


def _lock_exclusively(fd: int) -> bool:
    try:
        if sys.platform == "win32":
            import msvcrt
            os.lseek(fd, _LOCK_OFFSET, os.SEEK_SET)
            try:
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            finally:
                os.lseek(fd, 0, os.SEEK_SET)
        else:
            import fcntl
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except OSError as exc:
        if exc.errno in _LOCK_UNSUPPORTED:
            logger.warning("This filesystem cannot lock; other Studios are unguarded.")
            return True
        return False


def _release_lock(fd: int) -> None:
    if sys.platform == "win32":
        with suppress(OSError):
            import msvcrt
            os.lseek(fd, _LOCK_OFFSET, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)


def _busy_detail(path: Path, requested: str) -> str:
    holder = ""
    with suppress(OSError):
        with open(path, "rb") as stream:
            holder = stream.read(64).decode("utf-8", "replace").strip()
    return f"Another Cloudflare {holder or requested} step is already running."


@contextmanager
def certificate_state_claim(operation: str):
    path = _claim_path()
    _writable_dir(path.parent)
    fd, locked, held = None, False, False
    try:
        fd = os.open(str(path), os.O_CREAT | os.O_RDWR, 0o600)
        locked = _lock_exclusively(fd)
        if not locked:
            raise ProvisioningError("certificate_state_busy", _busy_detail(path, operation))
        held = _claim_lock.acquire(blocking = False)
        if not held:
            raise ProvisioningError("certificate_state_busy", _busy_detail(path, operation))
        with suppress(OSError):
            os.truncate(fd, 0)
            os.write(fd, operation.encode("utf-8"))
        yield
    finally:
        if fd is not None:
            if locked:
                _release_lock(fd)
            try:
                os.close(fd)
            except OSError:
                pass
        if held:
            _claim_lock.release()


def _connector_lock_path() -> Path:
    return _writable_dir(_state_dir()) / "connector.lock"


class ConnectorReservation:
    def __init__(self, fd: int):
        self._fd = fd
        self._guard = threading.Lock()
        self.token = _new_token()

    def record_connector(self, connector_pid: int) -> None:
        with self._guard:
            if self._fd is None:
                raise RuntimeError("the connector reservation has already been released")
            _write(
                _CONNECTOR,
                {
                    "connector_pid": connector_pid,
                    # A pid alone is a pid a later process may be reusing.
                    "connector_token": self.token,
                },
            )

    def release(self) -> None:
        with self._guard:
            fd, self._fd = self._fd, None
        if fd is None:
            return
        try:
            _discard(_CONNECTOR)
        finally:
            _release_lock(fd)
            with suppress(OSError):
                os.close(fd)
            _connector_lock.release()


def reserve_connector() -> Optional[ConnectorReservation]:
    fd, locked, held = None, False, False
    try:
        fd = os.open(str(_connector_lock_path()), os.O_CREAT | os.O_RDWR, 0o600)
        locked = _lock_exclusively(fd)
        if not locked:
            return None
        held = _connector_lock.acquire(blocking = False)
        if not held:
            return None
        if not _end_previous_connector(_read(_CONNECTOR)):
            return None
        _discard(_CONNECTOR)
        reservation = ConnectorReservation(fd)
        fd, held = None, False
        return reservation
    finally:
        if fd is not None:
            if locked:
                _release_lock(fd)
            with suppress(OSError):
                os.close(fd)
        if held:
            _connector_lock.release()


def _terminate_connector(pid: int) -> bool:
    try:
        import psutil

        process = psutil.Process(pid)
        os.kill(pid, signal.SIGTERM)
        process.wait(timeout = _CONNECTOR_EXIT_WAIT)
        return True
    except Exception:
        return not _pid_alive(pid)


def _end_previous_connector(record: object) -> bool:
    if not isinstance(record, dict):
        return True
    pid = record.get("connector_pid")
    if not _pid_alive(pid):
        return True
    if not _same_process(record.get("connector_token"), pid):
        return True
    return _terminate_connector(pid)


def reclaim_at_launch() -> None:
    try:
        with certificate_state_claim("cleanup"):
            from utils.remote_access_settings import clear_custom_remote_access_auto_start
            _settle(None, clear_auto_start = clear_custom_remote_access_auto_start)
    except ProvisioningError as exc:
        if exc.code != "certificate_state_busy":
            logger.warning("Cloudflare setup reclaim stopped: %s", exc.detail)
    except Exception:
        logger.warning("Cloudflare setup reclaim stopped.", exc_info = True)

    try:
        reservation = reserve_connector()
        if reservation is not None:
            reservation.release()
    except Exception:
        logger.warning("Cloudflare connector reclaim stopped.", exc_info = True)


# Strip ambient TUNNEL_* flag bindings before invoking cloudflared.
def _child_env(token: Optional[str] = None) -> dict:
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("TUNNEL_") and key != _TOKEN_VAR
    }
    if token is not None:
        env[_TOKEN_VAR] = token
    return env


# cloudflared opens login itself unless its platform launcher lookup fails.
def _login_env(token: str) -> dict:
    env = _child_env(token)
    env["PATH"] = ""
    env["NoDefaultCurrentDirectoryInExePath"] = "1"
    return env


def _cli_argv(binary: str, *args: str) -> list:
    return [binary, "tunnel", "--no-autoupdate", "--origincert", str(origin_cert_path()), *args]


def _cli(binary: str, *args: str):
    try:
        return subprocess.run(
            _cli_argv(binary, *args),
            capture_output = True,
            encoding = "utf-8",
            errors = "replace",
            timeout = _CLI_TIMEOUT,
            env = _child_env(),
            **_windows_hidden_kwargs(),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ProvisioningError("cloudflared_unreachable", str(exc)) from exc


def _output(result) -> str:
    return "\n".join(part for part in (result.stdout, result.stderr) if part)


def _error_summary(text: str) -> str:
    text = _LOGIN_URL_RE.sub("", text or "")
    lines = [line.strip() for line in text.splitlines() if line.strip()] or [""]
    _, advised, cause = lines[-1].partition(_ROUTE_ADVICE)
    return (cause.lstrip(": ").strip() if advised else lines[-1])[:300]


def _terminate(proc) -> None:
    try:
        proc.terminate()
        proc.wait(timeout = 5.0)
    except Exception:
        try:
            proc.kill()
            proc.wait(timeout = 5.0)
        except Exception:
            pass


def _spawn_login(binary: str, token: str):
    def _spawn():
        return subprocess.Popen(
            [binary, "tunnel", "--no-autoupdate", "login"],
            stdout = subprocess.PIPE,
            stderr = subprocess.STDOUT,
            encoding = "utf-8",
            errors = "replace",
            env = _login_env(token),
            **_lifetime_kwargs(),
            **_windows_hidden_kwargs(),
        )

    return _spawn_child(_spawn)


def _capture_certificate(record: dict) -> None:
    try:
        record["cert_digest"] = hashlib.sha256(origin_cert_path().read_bytes()).hexdigest()
    except OSError:
        return
    _write(_RECORD, record)


def _run_login(
    record: dict,
    binary: str,
    on_login_url: Optional[Callable[[str], None]],
    cancelled: Optional[Callable[[], bool]],
) -> None:
    seen = []
    proc = reader = None

    def _drain():
        for line in proc.stdout or ():
            seen.append(line)
            match = _LOGIN_URL_RE.search(line)
            if match is not None and on_login_url is not None:
                try:
                    # Never logged: this URL authorizes the user's Cloudflare account.
                    on_login_url(match.group(0))
                except Exception:
                    pass

    try:
        token = _new_token()
        proc = _spawn_login(binary, token)
        _adopt_pid(proc.pid)
        record["login_pid"] = proc.pid
        record["login_token"] = token
        _write(_RECORD, record)
        thread = threading.Thread(target = _drain, daemon = True, name = "cloudflared-login")
        thread.start()
        reader = thread
        deadline = time.monotonic() + _LOGIN_DEADLINE
        while proc.poll() is None:
            if cancelled is not None and cancelled():
                raise ProvisioningError("cancelled", "Cloudflare setup was cancelled.")
            if time.monotonic() >= deadline:
                raise ProvisioningError("login_timed_out", "The Cloudflare login did not finish.")
            time.sleep(0.2)
    finally:
        if proc is not None:
            if proc.poll() is None:
                _terminate(proc)
            _forget_pid(proc.pid)
            # Delete cert.pem only when its final digest proves this run created it.
            _capture_certificate(record)
            if reader is not None:
                reader.join(timeout = 2.0)
            with suppress(Exception):
                proc.stdout.close()
            record.pop("login_pid", None)
            record.pop("login_token", None)
            _write(_RECORD, record)

    text = "".join(seen)
    if _LOGIN_REFUSED in text:
        found = _LOGIN_REFUSED_RE.search(text)
        record.pop("cert_digest", None)
        _write(_RECORD, record)
        path = found.group(1) if found else str(origin_cert_path())
        raise ProvisioningError("certificate_exists", path)
    if proc.returncode != 0:
        raise ProvisioningError("login_failed", _error_summary(text))
    if not record.get("cert_digest"):
        raise ProvisioningError("login_failed", "The Cloudflare login wrote no certificate.")


def _generate_tunnel_name() -> str:
    return "unsloth-" + "".join(secrets.choice(_NAME_ALPHABET) for _ in range(6))


def _credentials_path(text: str, tunnel_id: str) -> Path:
    match = _CREDENTIALS_RE.search(text)
    beside_the_certificate = origin_cert_path().with_name(f"{tunnel_id}.json")
    return Path(match.group(1).strip()) if match else beside_the_certificate


def _create_tunnel(record: dict, binary: str) -> Tuple[str, str, Path]:
    for attempt in (0, 1):
        name = _generate_tunnel_name()
        record.setdefault("tunnel_names", []).append(name)
        _write(_RECORD, record)
        result = _cli(binary, "create", name)
        text = _output(result)
        if result.returncode == 0:
            match = _TUNNEL_ID_RE.search(text)
            if match is None:
                raise ProvisioningError("tunnel_create_failed", "cloudflared reported no id.")
            tunnel_id = match.group(0).lower()
            return name, tunnel_id, _credentials_path(text, tunnel_id)
        collided = _CREATE_COLLISION in text
        if collided:
            record["tunnel_names"].remove(name)
            _write(_RECORD, record)
        if not collided or attempt:
            raise ProvisioningError("tunnel_create_failed", _error_summary(text))


def _secure_credentials(source: Path, tunnel_id: str) -> Path:
    target = state_root() / f"{tunnel_id}.json"
    try:
        shutil.move(str(source), str(target))
    except OSError:
        target = source
    _chmod(target, 0o600)
    return target


def _route_hostname(record: dict, binary: str, name: str, hostname: str) -> None:
    record["route_attempted"] = True
    _write(_RECORD, record)
    result = _cli(binary, "route", "dns", name, hostname)
    text = _output(result)
    if result.returncode == 0:
        # A hostname outside the authorized zone is not refused: cloudflared takes
        # it as a label inside that zone and reports success having created a
        # record that can never serve what was asked for. The route is kept on the
        # record so the name it did create is still accounted for. Only a reported
        # name that disagrees is evidence; reporting none is not.
        created = _ROUTE_ADDED_RE.search(text)
        if created and created.group(1).rstrip(".").lower() != hostname.lower():
            # The record that exists is the one cleanup has to name, and the
            # hostname asked for was never created.
            record["hostname"] = created.group(1).rstrip(".").lower()
            _write(_RECORD, record)
            raise ProvisioningError("hostname_not_authorized", hostname)
        return
    lowered = text.lower()
    if any(marker in lowered for marker in _ROUTE_AUTHORIZATION_FAILURES):
        record.pop("route_attempted", None)
        _write(_RECORD, record)
        raise ProvisioningError("hostname_not_authorized", hostname)
    # An unclassified failure may have created the route before returning.
    if _ROUTE_CONFLICT not in lowered:
        raise ProvisioningError("route_failed", _error_summary(text))
    record.pop("route_attempted", None)
    _write(_RECORD, record)
    detail = f"A DNS record for {hostname} already exists and Studio will not replace it."
    raise ProvisioningError("dns_conflict", detail)


def _delete_tunnel(binary: Optional[str], name: object) -> bool:
    if not isinstance(name, str) or not _NAME_RE.match(name):
        return True
    if not binary:
        return False
    try:
        result = _cli(binary, "delete", name)
    except ProvisioningError as exc:
        logger.warning("Could not delete Cloudflare tunnel %s: %s", name, exc.detail)
        return False
    text = _output(result)
    if result.returncode == 0 or _TUNNEL_ABSENT in text.lower():
        return True
    logger.warning("Could not delete Cloudflare tunnel %s: %s", name, _error_summary(text))
    return False


def _abandon(binary: Optional[str], names) -> None:
    stranded = [
        name for name in names if isinstance(name, str) and not _delete_tunnel(binary, name)
    ]
    if stranded:
        _write(_ABANDONED, list(dict.fromkeys(_string_list(_ABANDONED) + stranded)))


def _reap_login_child(record: dict) -> bool:
    """End a recorded login child. False when one is still running."""
    pid = record.get("login_pid")
    if not _same_process(record.get("login_token"), pid):
        return True
    try:
        import psutil

        os.kill(pid, signal.SIGTERM)
        psutil.Process(pid).wait(timeout = 5.0)
        return True
    except Exception:
        return not _pid_alive(pid)


def _delete_our_certificate(record: dict, *, required: bool = False) -> None:
    path = origin_cert_path()
    digest = record.get("cert_digest")
    if not digest:
        return
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(fd, "rb") as certificate:
            actual = hashlib.sha256(certificate.read()).hexdigest()
            opened = os.fstat(certificate.fileno())
            current = path.stat(follow_symlinks = False)
            if actual == digest and (opened.st_dev, opened.st_ino) == (
                current.st_dev,
                current.st_ino,
            ):
                _unlink(path, required = required)
    except FileNotFoundError:
        return
    except OSError:
        if required:
            raise


def _settle(
    binary: Optional[str], *, clear_auto_start: Optional[Callable[[], object]] = None
) -> bool:
    record = _read(_RECORD)
    if not isinstance(record, dict):
        if (_state_dir() / _RECORD).exists():
            logger.warning("Cloudflare setup record is unreadable; leaving it in place.")
        return False
    if not _reap_login_child(record):
        # This record is the only thing naming that child. Clearing it now would
        # leave a login able to write the certificate with nothing able to
        # identify what wrote it, which is the state setup cannot recover from.
        logger.warning("Cloudflare login is still running; leaving its setup record in place.")
        return False
    identity = read_identity()
    teardown = record.get("operation") == "teardown"
    if teardown:
        if record.get("hostname"):
            _set_orphan(str(record["hostname"]), stranded = True)
        if (
            record.get("cloudflare_cleanup") != "manual"
            and record.get("tunnel_deleted") is not True
        ):
            _abandon(binary, record.get("tunnel_names") or ())
        if clear_auto_start is not None:
            clear_auto_start()
        credentials = record.get("credentials") or (
            identity.get("credentials") if identity else None
        )
        if credentials:
            _unlink(Path(str(credentials)), required = True)
        _discard(_IDENTITY, required = True)
    elif identity is None:
        if record.get("route_attempted") and record.get("hostname"):
            _set_orphan(str(record["hostname"]), stranded = True)
        _abandon(binary, record.get("tunnel_names") or ())
        if record.get("credentials"):
            _unlink(Path(str(record["credentials"])))
    _delete_our_certificate(record, required = teardown)
    _discard(_RECORD, required = teardown)
    return teardown


def _provision(
    record: dict,
    hostname: str,
    binary: str,
    on_login_url: Optional[Callable[[str], None]],
    cancelled: Optional[Callable[[], bool]],
) -> None:
    def _stop_if_cancelled() -> None:
        if cancelled is not None and cancelled():
            raise ProvisioningError("cancelled", "Cloudflare setup was cancelled.")

    if origin_cert_path().exists():
        raise ProvisioningError("certificate_exists", str(origin_cert_path()))
    _run_login(record, binary, on_login_url, cancelled)
    # Each step below creates something in the user's Cloudflare account, and the
    # CLI calls between them run long enough for a cancel to arrive mid-flight.
    # The caller's settle removes whatever was created before the raise.
    _stop_if_cancelled()
    name, tunnel_id, credentials = _create_tunnel(record, binary)
    record["credentials"] = str(_secure_credentials(credentials, tunnel_id))
    _write(_RECORD, record)
    _stop_if_cancelled()
    _route_hostname(record, binary, name, hostname)
    _stop_if_cancelled()
    _write(
        _IDENTITY,
        {
            "hostname": hostname,
            "tunnel_name": name,
            "tunnel_id": tunnel_id,
            "credentials": record["credentials"],
        },
    )
    _set_orphan(hostname, stranded = False)


def provision_custom_tunnel(
    hostname: str,
    *,
    binary: str,
    on_login_url: Optional[Callable[[str], None]] = None,
    cancelled: Optional[Callable[[], bool]] = None,
) -> Optional[dict]:
    """Create this install's tunnel and DNS record."""
    host = canonical_hostname(hostname)
    with certificate_state_claim("setup"):
        if read_identity() is not None:
            detail = "This machine already has a Cloudflare tunnel."
            raise ProvisioningError("identity_exists", detail)
        _settle(binary)
        if (_state_dir() / _RECORD).exists():
            raise ProvisioningError("setup_record_unreadable", str(_state_dir() / _RECORD))
        record = {"hostname": host}
        _write(_RECORD, record)
        try:
            _provision(record, host, binary, on_login_url, cancelled)
        finally:
            _settle(binary)
    return read_identity()


def teardown_custom_tunnel(*, clear_auto_start: Optional[Callable[[], object]] = None) -> bool:
    """Clear this install's custom-tunnel credentials for manual Cloudflare removal."""
    reservation = reserve_connector()
    if reservation is None:
        raise ProvisioningError(
            "connector_in_use", "Another backend is running this Cloudflare connector."
        )
    try:
        with certificate_state_claim("teardown"):
            identity = read_identity()
            if identity is None:
                return False
            hostname = str(identity.get("hostname") or "")
            name = identity.get("tunnel_name")
            credentials = identity.get("credentials")
            record = {
                "operation": "teardown",
                "cloudflare_cleanup": "manual",
                "hostname": hostname,
                "tunnel_names": [name],
                "credentials": credentials,
            }
            _write(_RECORD, record)
            _settle(None, clear_auto_start = clear_auto_start)
            return True
    finally:
        reservation.release()
