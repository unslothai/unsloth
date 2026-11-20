# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Per-call HTTP proxy for sandboxed curl/wget: the sandbox's only way out, refusing metadata and local addresses."""

from __future__ import annotations

import ipaddress
import os
import selectors
import socket
import threading
from urllib.parse import urlsplit

from loggers import get_logger

logger = get_logger(__name__)

RELAY_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sandbox_egress_relay.py")
# Loopback stays direct: on Linux it is the call's own network namespace, on macOS Seatbelt refuses it.
NO_PROXY = "localhost,127.0.0.1,::1"

_HEAD_LIMIT = 64 * 1024
_HEAD_TIMEOUT_SECONDS = 30.0
_CONNECT_TIMEOUT_SECONDS = 15.0
_IDLE_TIMEOUT_SECONDS = 600.0
_MAX_CONNECTIONS = 64
_NAT64 = ipaddress.ip_network("64:ff9b::/96")
# RFC 8215 local-use translation: where the IPv4 sits depends on the site's prefix length, so refuse it all.
_LOCAL_NAT64 = ipaddress.ip_network("64:ff9b:1::/48")
_THIS_NETWORK = ipaddress.ip_network("0.0.0.0/8")
_HOP_HEADERS = frozenset({"connection", "keep-alive", "proxy-connection", "proxy-authorization"})


class EgressRefused(Exception):
    pass


def write_empty_rc_files(directory: str) -> None:
    # curl takes the first .curlrc it finds from CURL_HOME on, so an empty one here hides the workdir's.
    # wget reads .netrc credentials unless told not to; curl only with -n, which is not allowlisted.
    for name, text in ((".curlrc", ""), ("wgetrc", "netrc = off\n")):
        with open(os.path.join(directory, name), "w", encoding = "utf-8") as f:
            f.write(text)


def proxy_environment(url: str, rc_dir: str) -> dict[str, str]:
    env = {
        name: url
        for name in (
            "http_proxy",
            "https_proxy",
            "all_proxy",
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "ALL_PROXY",
        )
    }
    env.update(no_proxy = NO_PROXY, NO_PROXY = NO_PROXY)
    # wget reads its global rc before WGETRC and keeps its settings (post_file), so replace both.
    wgetrc = os.path.join(rc_dir, "wgetrc")
    env.update(CURL_HOME = rc_dir, WGETRC = wgetrc, SYSTEM_WGETRC = wgetrc)
    return env


def refused_address(address: str) -> bool:
    from .providers import _metadata_host

    try:
        ip = ipaddress.ip_address(address.split("%", 1)[0])
    except ValueError:
        return True
    if ip.version == 6 and ip in _LOCAL_NAT64:
        return True
    candidates = [ip]
    if ip.version == 6:
        candidates += [ip.ipv4_mapped, ip.sixtofour]
        if ip in _NAT64:
            candidates.append(ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF))
    for candidate in candidates:
        if candidate is None:
            continue
        if (
            candidate.is_loopback
            or candidate.is_link_local
            or candidate.is_unspecified
            or candidate.is_multicast
            or (candidate.version == 4 and (candidate.is_reserved or candidate in _THIS_NETWORK))
            or _metadata_host(candidate.compressed)
        ):
            return True
    return False


def resolve_allowed(host: str, port: int) -> list[tuple[int, tuple]]:
    """Every address the name resolves to is checked; the caller dials exactly these, so DNS cannot change it."""
    from .providers import _metadata_host

    if not host or _metadata_host(host.lower()):
        raise EgressRefused(f"{host or 'an empty host'} is a cloud-metadata host")
    try:
        infos = socket.getaddrinfo(host, port, type = socket.SOCK_STREAM)
    except (OSError, UnicodeError) as exc:
        raise OSError(f"could not resolve {host}: {exc}") from exc
    for info in infos:
        if refused_address(info[4][0]):
            raise EgressRefused(
                f"{host} resolves to {info[4][0]}, a cloud-metadata, link-local or loopback address"
            )
    return [(info[0], info[4]) for info in infos]


def _dial(host: str, port: int) -> socket.socket:
    error: OSError | None = None
    for family, sockaddr in resolve_allowed(host, port):
        sock = socket.socket(family, socket.SOCK_STREAM)
        sock.settimeout(_CONNECT_TIMEOUT_SECONDS)
        try:
            sock.connect(sockaddr)
            return sock
        except OSError as exc:
            sock.close()
            error = exc
    raise error or OSError(f"no address for {host}")


def _read_head(sock: socket.socket) -> tuple[bytes, bytes]:
    data = b""
    while b"\r\n\r\n" not in data:
        if len(data) > _HEAD_LIMIT:
            raise ValueError("request head too large")
        chunk = sock.recv(65536)
        if not chunk:
            raise ValueError("connection closed before the request head")
        data += chunk
    head, _, rest = data.partition(b"\r\n\r\n")
    return head, rest


def _split_authority(authority: str) -> tuple[str, int]:
    parsed = urlsplit(f"//{authority}")
    if not parsed.hostname or parsed.port is None:
        raise ValueError(f"bad CONNECT target: {authority}")
    return parsed.hostname, parsed.port


def _reply(sock: socket.socket, status: str, message: str) -> None:
    body = f"Unsloth sandbox: {message}\n".encode("utf-8", "replace")
    try:
        sock.sendall(
            f"HTTP/1.1 {status}\r\nContent-Type: text/plain\r\nConnection: close\r\n"
            f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
            + body
        )
    except OSError:
        pass


def _splice(
    first: socket.socket,
    second: socket.socket,
    may_send = None,
) -> dict[socket.socket, int]:
    """may_send(n) is asked before each n bytes read from `first` go on; False ends the splice."""
    peers = {first: second, second: first}
    read = {first: 0, second: 0}
    selector = selectors.DefaultSelector()
    try:
        for sock in peers:
            sock.setblocking(True)
            selector.register(sock, selectors.EVENT_READ)
        while selector.get_map():
            events = selector.select(_IDLE_TIMEOUT_SECONDS)
            if not events:
                return read
            for key, _ in events:
                sock = key.fileobj
                data = sock.recv(65536)
                if data:
                    if sock is first and may_send is not None and not may_send(len(data)):
                        return read
                    read[sock] += len(data)
                    peers[sock].sendall(data)
                    continue
                selector.unregister(sock)
                try:
                    peers[sock].shutdown(socket.SHUT_WR)
                except OSError:
                    pass
    except (OSError, ValueError):
        pass
    finally:
        selector.close()
    return read


class EgressProxy:
    """Listens on a unix socket (bubblewrap) or an ephemeral 127.0.0.1 port (Seatbelt) until close()."""

    def __init__(
        self,
        unix_path: str | None = None,
        gate = None,
    ) -> None:
        self._gate = gate
        family = socket.AF_UNIX if unix_path else socket.AF_INET
        listener = socket.socket(family, socket.SOCK_STREAM)
        try:
            listener.bind(unix_path or ("127.0.0.1", 0))
            listener.listen(_MAX_CONNECTIONS)
            listener.settimeout(0.25)
        except Exception:
            listener.close()
            raise
        self.port = None if unix_path else listener.getsockname()[1]
        self._listener = listener
        self._sockets: set[socket.socket] = set()
        self._lock = threading.Lock()
        self._closed = False
        self._thread = threading.Thread(target = self._serve, name = "unsloth-egress", daemon = True)
        self._thread.start()

    def _track(self, sock: socket.socket) -> bool:
        with self._lock:
            if self._closed:
                return False
            self._sockets.add(sock)
            return True

    def _release(self, sock: socket.socket | None) -> None:
        if sock is None:
            return
        with self._lock:
            self._sockets.discard(sock)
        try:
            sock.close()
        except OSError:
            pass

    def _serve(self) -> None:
        while not self._closed:
            try:
                client, _ = self._listener.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            if len(self._sockets) >= 2 * _MAX_CONNECTIONS or not self._track(client):
                client.close()
                continue
            threading.Thread(target = self._handle, args = (client,), daemon = True).start()

    def _handle(self, client: socket.socket) -> None:
        upstream = None
        try:
            client.settimeout(_HEAD_TIMEOUT_SECONDS)
            try:
                head, rest = _read_head(client)
                request_line, *headers = head.decode("latin-1").split("\r\n")
                method, target, version = request_line.split(" ")
                if method.upper() == "CONNECT":
                    host, port = _split_authority(target)
                    forward = None
                else:
                    url = urlsplit(target)
                    if url.scheme.lower() != "http" or not url.hostname:
                        raise ValueError(f"only http:// URLs and CONNECT are proxied: {target}")
                    host, port = url.hostname, url.port or 80
                    path = (url.path or "/") + (f"?{url.query}" if url.query else "")
                    kept = [
                        line
                        for line in headers
                        if line.split(":", 1)[0].strip().lower() not in _HOP_HEADERS
                    ]
                    # One request per upstream connection: a reused one would carry the next URL to this host.
                    forward = "\r\n".join(
                        [f"{method} {path} {version}", *kept, "Connection: close", "", ""]
                    ).encode("latin-1")
                if any(ord(c) <= 0x20 or ord(c) == 0x7F for c in host):
                    raise ValueError(f"bad host: {host!r}")
                refusal = self._gate.check(host, port) if self._gate is not None else None
                if refusal:
                    raise EgressRefused(refusal)
                upstream = _dial(host, port)
            except EgressRefused as exc:
                logger.info("Sandboxed download refused: %s", exc)
                _reply(client, "403 Forbidden", f"refused: {exc}")
                return
            except ValueError as exc:
                _reply(client, "400 Bad Request", str(exc))
                return
            except OSError as exc:
                _reply(client, "502 Bad Gateway", str(exc))
                return
            if not self._track(upstream):
                return
            client.settimeout(None)
            upstream.settimeout(None)
            gate = self._gate
            may_send = None if gate is None else (lambda size: gate.sending(host, port, size))
            first = (forward or b"") + rest
            if forward is None:
                client.sendall(b"HTTP/1.1 200 Connection established\r\n\r\n")
            if first:
                if may_send is not None and not may_send(len(first)):
                    return
                upstream.sendall(first)
            read = _splice(client, upstream, may_send)
            if gate is not None:
                gate.record(host, port, len(first) + read[client], read[upstream])
        except OSError:
            pass
        finally:
            self._release(upstream)
            self._release(client)

    def close(self) -> None:
        if self._gate is not None:
            self._gate.close()
        with self._lock:
            self._closed = True
            sockets = list(self._sockets)
            self._sockets.clear()
        # Shutdown only: closing here races a handler blocked in select(), which then sleeps out the idle timeout.
        for sock in sockets:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        self._listener.close()
        self._thread.join(timeout = 2)
