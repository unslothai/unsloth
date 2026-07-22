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
from typing import Optional, Tuple

# cloudflared logs the quick-tunnel URL; match only the URL so we do not depend
# on the surrounding wording, which Cloudflare may change. The negative lookahead
# drops cloudflared's own API host, which appears in failure lines such as
#   failed to request quick Tunnel: Post "https://api.trycloudflare.com/tunnel"
# and must never be mistaken for a usable tunnel URL.
_URL_RE = re.compile(r"https://(?!api\.)[A-Za-z0-9-]+\.trycloudflare\.com")

# cloudflared logs this once per edge connection it establishes. Until at least
# one appears the quick-tunnel URL returns Cloudflare error 1033 (HTTP 530), so
# we wait for it before advertising the URL.
_REGISTERED_MARKER = "Registered tunnel connection"

_RELEASE_BASE = "https://github.com/cloudflare/cloudflared/releases/latest/download"

_READY_TIMEOUT = 15.0  # seconds to wait for the URL + a registered edge connection
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
_DOH_URL = "https://cloudflare-dns.com/dns-query?name={host}&type=A"


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


def _wait_for_dns(host: str, deadline: float) -> None:
    import json
    import urllib.request
    while True:
        try:
            req = urllib.request.Request(
                _DOH_URL.format(host = host),
                headers = {"Accept": "application/dns-json", "User-Agent": "unsloth-studio"},
            )
            with urllib.request.urlopen(req, timeout = 5) as response:
                answered = bool(json.loads(response.read(65536)).get("Answer"))
        except Exception:
            return
        if answered:
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(_DNS_POLL_DELAY, remaining))


def verify_public_url(url: str, timeout: float = _PUBLIC_PROBE_TIMEOUT) -> bool:
    import json
    import urllib.request
    from urllib.parse import urlsplit

    deadline = time.monotonic() + timeout
    host = urlsplit(url).hostname
    if host:
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


class CloudflareTunnel:
    """A cloudflared quick tunnel to http://localhost:<port>. Best-effort throughout.

    Use localhost (not the wildcard bind) as the tunnel origin so cloudflared's
    upstream stays local-only.
    """

    def __init__(
        self,
        port: int,
        binary: str,
        protocol: Optional[str] = None,
    ):
        self.port = port
        self.binary = binary
        # None lets cloudflared pick its default (quic, with its own http2
        # fallback); set to "http2" to force it when quic is blocked.
        self.protocol = protocol
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._stopped = False
        self._url_event = threading.Event()
        self._ready_event = threading.Event()
        self.url: Optional[str] = None
        self.ready = False
        self.error: Optional[str] = None

    def start(self) -> None:
        cmd = [
            self.binary,
            "tunnel",
            "--url",
            f"http://localhost:{self.port}",
            "--no-autoupdate",
        ]
        if self.protocol:
            cmd += ["--protocol", self.protocol]
        with self._lock:
            # A stop() that landed before us (e.g. a shutdown in the caller's
            # register->start window) marks the tunnel stopped; spawning now would
            # orphan a process nobody owns, so refuse.
            if self._stopped:
                return
            proc = subprocess.Popen(
                cmd,
                stdout = subprocess.PIPE,
                stderr = subprocess.STDOUT,
                stdin = subprocess.DEVNULL,
                text = True,
                errors = "replace",
                bufsize = 1,
                **_windows_hidden_kwargs(),
                **_lifetime_kwargs(),
            )
            self._proc = proc
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
            self._url_event.set()
            self._ready_event.set()

    def wait_for_ready(self, timeout: float = _READY_TIMEOUT) -> Optional[str]:
        """Block until the tunnel is actually serving -- the URL has been minted
        *and* at least one edge connection has registered -- or until timeout.

        Returns the URL only when ready, so callers never advertise a URL that
        would return Cloudflare error 1033 (HTTP 530)."""
        self._ready_event.wait(timeout)
        return self.url if self.ready else None

    def stop(self) -> None:
        """Terminate the tunnel. Idempotent and safe to call from a signal handler."""
        with self._lock:
            # Mark stopped so a start() racing behind us refuses to spawn.
            self._stopped = True
            proc, self._proc = self._proc, None
        if proc is None:
            return
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


# Single serving process per Unsloth launch, so one module-level tunnel handle is
# enough; the lock guards the start/stop/shutdown races.
_active_tunnel: Optional[CloudflareTunnel] = None
_active_lock = threading.Lock()
# Latched by stop_studio_tunnel so a shutdown landing *between* a start's retry
# attempts aborts the loop instead of starting a tunnel nobody will ever stop.
_shutdown_requested = False


def start_studio_tunnel(port: int, timeout: float = _READY_TIMEOUT) -> Optional[str]:
    """Start a quick tunnel and return its public URL once it is actually
    serving, or None (best-effort).

    Waits for cloudflared to both mint the URL and register an edge connection,
    then fetches /api/health over the public URL, so the caller never advertises
    a link that yields Cloudflare error 1033 (HTTP 530) or an unresolvable host.
    If a URL is minted but no connection registers within the window (e.g. quic
    is blocked on this network), retries once forcing the http2 protocol. On any
    failure the tunnel is stopped and None is returned.
    """
    global _active_tunnel, _shutdown_requested
    binary = ensure_cloudflared()
    if not binary:
        return None
    with _active_lock:
        _shutdown_requested = False  # fresh session
    # Default protocol first (quic, with cloudflared's own http2 fallback); if a
    # URL appears but no connection registers, quic is likely blocked -> retry
    # once forcing http2.
    for protocol in (None, "http2"):
        # Create + register under the lock, and bail if a stop already landed
        # (e.g. between this and the previous attempt) so we never start a tunnel
        # after shutdown has run.
        with _active_lock:
            if _shutdown_requested:
                _active_tunnel = None
                return None
            tunnel = CloudflareTunnel(port, binary, protocol = protocol)
            prior, _active_tunnel = _active_tunnel, tunnel
        if prior is not None:
            prior.stop()
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
            return url
        saw_url = tunnel.url is not None
        # Not ready: drop it, but only if we are still the active tunnel.
        with _active_lock:
            was_active = _active_tunnel is tunnel
            if was_active:
                _active_tunnel = None
        tunnel.stop()
        # A concurrent shutdown or start took over while we waited; retrying would
        # spawn a tunnel nobody owns (orphaned after shutdown), so bail instead.
        if not was_active:
            return None
        # No URL at all is an API/network failure, not a protocol one; forcing
        # http2 will not help, so do not burn another window on it.
        if not saw_url:
            return None
        # probe failure after registering is DNS propagation; http2 would not help
        if registered:
            return None
    return None


def stop_studio_tunnel() -> None:
    """Terminate the active tunnel, if any. Idempotent."""
    global _active_tunnel, _shutdown_requested
    with _active_lock:
        # Latch so an in-flight start_studio_tunnel won't start a fresh tunnel
        # (e.g. its http2 retry) after we have already torn down.
        _shutdown_requested = True
        tunnel, _active_tunnel = _active_tunnel, None
    if tunnel is not None:
        tunnel.stop()
