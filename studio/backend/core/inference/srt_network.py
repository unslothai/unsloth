# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Per-launch Unix sockets for SRT's private HTTP and SOCKS forwarders."""

from __future__ import annotations

import math
import logging
import os
from pathlib import Path
import socket
import tempfile
import threading
import shutil
import stat

from .network_proxy import AllowlistProxy, PROXY_ENV_KEYS, NO_PROXY_VALUE
from .network_proxy import _openssl_default_paths, _HASHED_CERT_RE, MAX_CAPATH_ENTRIES

logger = logging.getLogger(__name__)

MAX_TRUST_SNAPSHOT_BYTES = 16 * 1024 * 1024


class TlsTrustSnapshot:
    """Expose selected OpenSSL trust without exposing symlink target directories."""

    def __init__(self, base_environment=None, *, parent_dir=None):
        self._base = dict(os.environ if base_environment is None else base_environment)
        self._parent_dir = parent_dir
        self._directory = None
        self.environment = {}
        self.read_roots = ()
        self._started = False

    def start(self):
        if self._started:
            raise RuntimeError("TLS trust snapshot cannot be reused")
        self._started = True
        try:
            cafile, capath = _openssl_default_paths()
            cafile = self._base.get("SSL_CERT_FILE", cafile)
            capath = self._base.get("SSL_CERT_DIR", capath)
            roots = []
            # Preserve explicit absent/empty stores instead of installing a fallback.
            for key in ("SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE"):
                if key in self._base:
                    self.environment[key] = self._base[key]
            if cafile and os.path.isfile(cafile):
                cafile = os.path.realpath(cafile)
                roots.append(cafile)
                self.environment["SSL_CERT_FILE"] = cafile
                if "REQUESTS_CA_BUNDLE" not in self._base:
                    self.environment["REQUESTS_CA_BUNDLE"] = cafile
            requests_bundle = self._base.get("REQUESTS_CA_BUNDLE")
            if requests_bundle and os.path.isfile(requests_bundle):
                requests_bundle = os.path.realpath(requests_bundle)
                self.environment["REQUESTS_CA_BUNDLE"] = requests_bundle
                if requests_bundle not in roots:
                    roots.append(requests_bundle)
            if capath and os.path.isdir(capath):
                self._directory = Path(tempfile.mkdtemp(prefix="srt-trust-", dir=self._parent_dir))
                os.chmod(self._directory, 0o700)
                count = total = 0
                with os.scandir(capath) as entries:
                    for entry in entries:
                        if not _HASHED_CERT_RE.fullmatch(entry.name):
                            continue
                        # Follow only leaf trust entries; never copy a directory tree.
                        if not entry.is_file(follow_symlinks=True):
                            continue
                        count += 1
                        if count > MAX_CAPATH_ENTRIES:
                            raise ValueError("TLS trust snapshot exceeds certificate entry limit")
                        fd = os.open(entry.path, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0))
                        with os.fdopen(fd, "rb") as source:
                            if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
                                raise ValueError("TLS trust entry is not a regular file")
                            data = source.read(MAX_TRUST_SNAPSHOT_BYTES - total + 1)
                        total += len(data)
                        if total > MAX_TRUST_SNAPSHOT_BYTES:
                            raise ValueError("TLS trust snapshot exceeds byte limit")
                        target = self._directory / entry.name
                        with target.open("xb") as output:
                            output.write(data)
                        os.chmod(target, 0o600)
                self.environment["SSL_CERT_DIR"] = str(self._directory)
                roots.append(str(self._directory))
            self.read_roots = tuple(roots)
            return self
        except BaseException:
            self.close()
            raise

    def close(self):
        if self._directory is not None:
            shutil.rmtree(self._directory)
            self._directory = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *_exc):
        self.close()


class SrtNetworkTransport:
    """Own a proxy on HTTP UDS and refuse every SOCKS connection.

    SRT exposes these sockets as ports 3128 and 1080 inside its network namespace.
    HTTP is served directly by the existing HTTPS CONNECT policy;
    this adapter adds no forwarding policy, DNS resolution, or TLS interception.
    Its proxy owns bounded request buffers, tunnel concurrency and idle timeouts.
    """

    def __init__(
        self,
        proxy: AllowlistProxy,
        *,
        parent_dir: str | None = None,
        lifetime_seconds: float | None = 300,
    ) -> None:
        if lifetime_seconds is not None and (
            not math.isfinite(lifetime_seconds) or lifetime_seconds <= 0
        ):
            raise ValueError("network transport lifetime must be finite and positive")
        self.proxy = proxy
        self._parent_dir = parent_dir
        self._lifetime = lifetime_seconds
        self._directory: Path | None = None
        self._socks: socket.socket | None = None
        self._refuser: threading.Thread | None = None
        self._watchdog: threading.Thread | None = None
        self._closed = threading.Event()
        self._cleanup_done = threading.Event()
        self._cleanup_error: BaseException | None = None
        self._lock = threading.RLock()
        self._started = False

    @property
    def http_socket_path(self) -> str:
        if self._directory is None:
            raise RuntimeError("network transport has not started")
        return str(self._directory / "http.sock")

    @property
    def socks_socket_path(self) -> str:
        if self._directory is None:
            raise RuntimeError("network transport has not started")
        return str(self._directory / "socks.sock")

    @property
    def environment(self) -> dict[str, str]:
        """The private socket is the authority; no secret enters SRT's arguments."""
        return {
            **{key: "http://127.0.0.1:3128" for key in PROXY_ENV_KEYS},
            "NO_PROXY": NO_PROXY_VALUE,
            "no_proxy": NO_PROXY_VALUE,
        }

    def _listener(self, path: str) -> socket.socket:
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            listener.bind(path)
            os.chmod(path, 0o600)
            listener.listen(16)
            return listener
        except BaseException:
            listener.close()
            raise

    def start(self) -> "SrtNetworkTransport":
        with self._lock:
            if self._started or self._closed.is_set():
                raise RuntimeError("network transport cannot be reused")
            self._started = True
            try:
                self._directory = Path(tempfile.mkdtemp(prefix="srt-net-", dir=self._parent_dir))
                os.chmod(self._directory, 0o700)
                # The private listener API validates ownership before enabling
                # socket authority and takes ownership even when it raises.
                self.proxy.serve_private_unix_listener(self._listener(self.http_socket_path))
                self._socks = self._listener(self.socks_socket_path)
                self._socks.settimeout(0.1)
                self._refuser = threading.Thread(
                    target=self._refuse_socks,
                    name="srt-socks-refusal",
                    daemon=True,
                )
                self._refuser.start()
                if self._lifetime is not None:
                    self._watchdog = threading.Thread(
                        target=self._expire,
                        name="srt-network-lifetime",
                        daemon=True,
                    )
                    self._watchdog.start()
                return self
            except BaseException:
                self.close()
                raise

    def _refuse_socks(self) -> None:
        listener = self._socks
        assert listener is not None
        while not self._closed.is_set():
            try:
                client, _ = listener.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            with client:
                # No input or worker allocation: even idle and malformed clients
                # receive only a fixed SOCKS5 no-acceptable-method response.
                client.settimeout(0.1)
                try:
                    client.sendall(b"\x05\xff")
                except OSError:
                    pass

    def _expire(self) -> None:
        if not self._closed.wait(self._lifetime):
            try:
                self.close()
            except Exception:
                logger.exception("SRT network transport lifetime cleanup failed")

    def close(self) -> None:
        with self._lock:
            if self._closed.is_set():
                # The owner may be joining this watchdog. Other callers must
                # wait until sockets and paths really have been cleaned up.
                if threading.current_thread() is self._watchdog:
                    return
                owns_cleanup = False
            else:
                self._closed.set()
                owns_cleanup = True
        if not owns_cleanup:
            if not self._cleanup_done.wait(7):
                raise RuntimeError("SRT network transport cleanup did not finish")
            if self._cleanup_error is not None:
                raise RuntimeError("SRT network transport cleanup failed") from self._cleanup_error
            return
        try:
            self._close_owned()
        except BaseException as exc:
            self._cleanup_error = exc
            raise
        finally:
            self._cleanup_done.set()

    def _close_owned(self) -> None:
        if self._socks is not None:
            try:
                self._socks.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self._socks.close()
        proxy_error = None
        try:
            self.proxy.close()
        except Exception as exc:
            proxy_error = exc
        current = threading.current_thread()
        for worker in (self._refuser, self._watchdog):
            if worker is not None and worker is not current and worker.ident is not None:
                worker.join(timeout=1)
                if worker.is_alive():
                    raise RuntimeError("SRT network transport worker did not stop")
        if self._directory is not None:
            for name in ("http.sock", "socks.sock"):
                (self._directory / name).unlink(missing_ok=True)
            self._directory.rmdir()
        if proxy_error is not None:
            raise RuntimeError("SRT proxy cleanup failed") from proxy_error

    def __enter__(self) -> "SrtNetworkTransport":
        return self.start()

    def __exit__(self, *exc_info: object) -> None:
        self.close()
