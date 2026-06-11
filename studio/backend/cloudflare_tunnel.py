# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Free Cloudflare quick tunnel for Studio's 0.0.0.0 launches.

The raw http://<ip>:<port> is often unreachable (https-vs-http, blocked ports,
closed security groups); a cloudflared quick tunnel gives a free
https://*.trycloudflare.com URL that works anywhere, with no account or domain.

Best-effort throughout: any failure collapses to "no URL" and Studio keeps
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
from pathlib import Path
from typing import Optional, Tuple

# cloudflared logs the quick-tunnel URL; match only the URL so we do not depend
# on the surrounding wording, which Cloudflare may change.
_URL_RE = re.compile(r"https://[A-Za-z0-9-]+\.trycloudflare\.com")

_RELEASE_BASE = "https://github.com/cloudflare/cloudflared/releases/latest/download"

_URL_TIMEOUT = 15.0  # seconds to wait for the public URL before giving up
_DOWNLOAD_TIMEOUT = 60  # urlopen timeout for the one-time binary download


def _windows_hidden_kwargs() -> dict:
    """Suppress a child console window on Windows; no-op elsewhere."""
    if sys.platform != "win32":
        return {}
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    return {"creationflags": flags} if flags else {}


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
    """Locate an existing cloudflared: PATH first, then the Studio bin cache."""
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
            with urllib.request.urlopen(url, timeout = _DOWNLOAD_TIMEOUT) as response:
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


class CloudflareTunnel:
    """A cloudflared quick tunnel to http://localhost:<port>. Best-effort throughout.

    Use localhost (not the wildcard bind) as the tunnel origin so cloudflared's
    upstream stays local-only.
    """

    def __init__(self, port: int, binary: str):
        self.port = port
        self.binary = binary
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._url_event = threading.Event()
        self.url: Optional[str] = None
        self.error: Optional[str] = None

    def start(self) -> None:
        cmd = [
            self.binary,
            "tunnel",
            "--url",
            f"http://localhost:{self.port}",
            "--no-autoupdate",
        ]
        proc = subprocess.Popen(
            cmd,
            stdout = subprocess.PIPE,
            stderr = subprocess.STDOUT,
            stdin = subprocess.DEVNULL,
            text = True,
            errors = "replace",
            bufsize = 1,
            **_windows_hidden_kwargs(),
        )
        with self._lock:
            self._proc = proc
        threading.Thread(
            target = self._reader, args = (proc,), name = "cloudflared-reader", daemon = True
        ).start()

    def _reader(self, proc: subprocess.Popen) -> None:
        # Drain cloudflared's output, capture the first trycloudflare URL, and
        # keep draining so it never blocks on a full pipe.
        try:
            if proc.stdout is not None:
                for line in proc.stdout:
                    if self.url is None:
                        match = _URL_RE.search(line)
                        if match:
                            self.url = match.group(0)
                            self._url_event.set()
        except Exception:
            pass
        finally:
            if self.url is None:
                self.error = "cloudflared exited before emitting a tunnel URL"
                self._url_event.set()

    def wait_for_url(self, timeout: float = _URL_TIMEOUT) -> Optional[str]:
        self._url_event.wait(timeout)
        return self.url

    def stop(self) -> None:
        """Terminate the tunnel. Idempotent and safe to call from a signal handler."""
        with self._lock:
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


# Single serving process per Studio launch, so one module-level tunnel handle is
# enough; the lock guards the start/stop/shutdown races.
_active_tunnel: Optional[CloudflareTunnel] = None
_active_lock = threading.Lock()


def start_studio_tunnel(port: int, timeout: float = _URL_TIMEOUT) -> Optional[str]:
    """Start a quick tunnel and return its public URL, or None (best-effort).

    On any failure (no binary, no URL within timeout, early crash) the tunnel is
    stopped and None is returned, so the caller prints a hint and continues.
    """
    global _active_tunnel
    binary = ensure_cloudflared()
    if not binary:
        return None
    tunnel = CloudflareTunnel(port, binary)
    try:
        tunnel.start()
    except Exception:
        return None
    url = tunnel.wait_for_url(timeout)
    if not url:
        tunnel.stop()
        return None
    with _active_lock:
        prior, _active_tunnel = _active_tunnel, tunnel
    if prior is not None:
        prior.stop()
    return url


def stop_studio_tunnel() -> None:
    """Terminate the active tunnel, if any. Idempotent."""
    global _active_tunnel
    with _active_lock:
        tunnel, _active_tunnel = _active_tunnel, None
    if tunnel is not None:
        tunnel.stop()
