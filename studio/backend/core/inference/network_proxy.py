"""Loopback HTTP CONNECT proxy with a host allowlist for OS-isolated tool launches.

The sandboxes built by ``os_sandbox`` deny every network path by default. This
module is the one opening a user can enable per session: a proxy that runs in
the Studio backend process (outside the sandbox), accepts ``CONNECT`` tunnels
only, and lets them through only to a fixed set of hostnames on port 443.

Design points, each of which a test pins down:

* Tunnels only. Cleartext ``GET http://...`` is refused, so the proxy never has
  to parse, rewrite or forward request bodies; the sandboxed client speaks TLS
  end to end with the allowlisted host and the proxy cannot see the traffic.
* Hostnames only. IP literals in every spelling ``inet_aton`` accepts (dotted,
  decimal, octal, hex, ``127.1``) and bracketed IPv6 are refused, and after the
  allowlisted name resolves, every answer must be a public unicast address.
  A DNS answer that points at loopback, RFC 1918, link-local, CGNAT, multicast
  or an IPv4-mapped copy of those refuses the tunnel, so an allowlisted name
  cannot become a bridge back to the host or the LAN.
* Authenticated. Each launch mints a random credential that only travels to
  the sandboxed process through its environment; a request without it gets 407,
  so another local user on the same machine cannot use the tunnel.
* Bounded. Header size, concurrent tunnels, connect time and idle time all have
  caps; a client that misbehaves loses its connection, never the proxy.
* Audited. Allowed and denied hosts are counted per launch so the tool result
  can name the host a script tried and was refused, instead of leaving the
  model to guess why ``pip`` failed.
"""

from __future__ import annotations

import base64
import binascii
import hmac
import ipaddress
import os
import re
import secrets
import select
import socket
import threading
import time
from dataclasses import dataclass
from typing import Callable, Iterable

from loggers import get_logger

logger = get_logger(__name__)

# Hosts a model-driven Python or Terminal tool most often needs during an ML
# task. Everything is a hostname; the proxy refuses IP literals outright.
DEFAULT_ALLOWLIST: tuple[str, ...] = (
    "pypi.org",
    "files.pythonhosted.org",
    "huggingface.co",
    "*.huggingface.co",
    "cdn-lfs.huggingface.co",
    "*.hf.co",
    "github.com",
    "*.github.com",
    "*.githubusercontent.com",
    "download.pytorch.org",
    "*.kaggle.com",
    "*.tensorflow.org",
    "storage.googleapis.com",
)

ALLOWLIST_ENV = "UNSLOTH_STUDIO_TOOL_NETWORK_ALLOWLIST"

# Only TLS. A tunnel to 80 would let cleartext traffic leave the sandbox, and a
# tunnel to an arbitrary port would turn the proxy into a generic TCP relay.
ALLOWED_PORTS: frozenset[int] = frozenset({443})
# Tests that stand up a loopback upstream set this to False; production keeps
# the public-address rule so an allowlisted name resolving to 127.0.0.1 or
# 10.0.0.5 is refused.
REQUIRE_PUBLIC_ADDRESSES = True

MAX_HEADER_BYTES = 16 * 1024
MAX_TUNNELS = 64
CONNECT_TIMEOUT_SECONDS = 20.0
IDLE_TIMEOUT_SECONDS = 120.0
HEADER_TIMEOUT_SECONDS = 15.0
MAX_AUDITED_HOSTS = 128

_LABEL_RE = re.compile(r"^[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?$")
_PROXY_USER = "sandbox"
PROXY_ENV_KEYS = (
    "HTTPS_PROXY",
    "https_proxy",
    "HTTP_PROXY",
    "http_proxy",
    "ALL_PROXY",
    "all_proxy",
)
NO_PROXY_VALUE = "localhost,127.0.0.1,::1"


class AllowlistError(ValueError):
    """An allowlist entry that the proxy could never enforce safely."""


def _is_ip_literal(host: str) -> bool:
    candidate = host.strip("[]")
    try:
        ipaddress.ip_address(candidate)
        return True
    except ValueError:
        pass
    # inet_aton accepts every legacy IPv4 spelling (127.1, 0x7f000001,
    # 2130706433, 0177.0.0.1); ipaddress accepts none of them.
    try:
        socket.inet_aton(candidate)
        return True
    except OSError:
        return False


def normalize_host(host: str) -> str:
    """Return a lowercase ASCII hostname or raise ``AllowlistError``.

    IDNA-encodes Unicode names, strips one trailing dot, refuses IP literals
    (in any spelling), ``localhost`` and ``.local`` names, and every label that
    is not a valid DNS label.
    """
    if not isinstance(host, str):
        raise AllowlistError("host must be a string")
    stripped = host.strip().rstrip(".")
    if not stripped:
        raise AllowlistError("empty host")
    if len(stripped) > 253:
        raise AllowlistError("host is too long")
    if _is_ip_literal(stripped):
        raise AllowlistError(f"IP literals are not allowed: {host!r}")
    try:
        ascii_host = stripped.encode("idna").decode("ascii").lower()
    except UnicodeError as exc:
        raise AllowlistError(f"host is not a valid IDNA name: {host!r}") from exc
    if _is_ip_literal(ascii_host):
        raise AllowlistError(f"IP literals are not allowed: {host!r}")
    labels = ascii_host.split(".")
    for label in labels:
        if not _LABEL_RE.match(label):
            raise AllowlistError(f"invalid hostname label {label!r} in {host!r}")
    if labels[-1].isdigit():
        raise AllowlistError(f"numeric top-level label in {host!r}")
    if ascii_host == "localhost" or ascii_host.endswith(".localhost"):
        raise AllowlistError("localhost is never proxied")
    if ascii_host.endswith(".local"):
        raise AllowlistError("mDNS .local names are never proxied")
    return ascii_host


@dataclass(frozen = True)
class NetworkAllowlist:
    """Exact hosts plus ``*.suffix`` wildcards, normalized once at construction."""

    exact: frozenset[str]
    suffixes: frozenset[str]
    entries: tuple[str, ...]

    @classmethod
    def from_entries(cls, entries: Iterable[str]) -> "NetworkAllowlist":
        exact: set[str] = set()
        suffixes: set[str] = set()
        display: list[str] = []
        for raw in entries:
            entry = raw.strip()
            if not entry:
                continue
            if entry.startswith("*."):
                suffix = normalize_host(entry[2:])
                suffixes.add(suffix)
                shown = f"*.{suffix}"
            elif "*" in entry:
                raise AllowlistError(
                    f"only a leading '*.' wildcard is supported: {entry!r}"
                )
            else:
                shown = normalize_host(entry)
                exact.add(shown)
            if shown not in display:
                display.append(shown)
        if not exact and not suffixes:
            raise AllowlistError("the network allowlist is empty")
        return cls(frozenset(exact), frozenset(suffixes), tuple(display))

    @classmethod
    def from_env(cls, environ: dict[str, str] | None = None) -> "NetworkAllowlist":
        """The default list, replaced or extended by ``UNSLOTH_STUDIO_TOOL_NETWORK_ALLOWLIST``.

        A value starting with ``+`` extends the defaults; anything else replaces
        them. Entries are comma or whitespace separated.
        """
        source = os.environ if environ is None else environ
        raw = source.get(ALLOWLIST_ENV, "")
        if not raw.strip():
            return cls.from_entries(DEFAULT_ALLOWLIST)
        extend = raw.lstrip().startswith("+")
        body = raw.lstrip()[1:] if extend else raw
        entries = [item for item in re.split(r"[,\s]+", body) if item]
        if extend:
            entries = [*DEFAULT_ALLOWLIST, *entries]
        return cls.from_entries(entries)

    @property
    def hosts(self) -> tuple[str, ...]:
        return self.entries

    def allows(self, host: str) -> bool:
        try:
            normalized = normalize_host(host)
        except AllowlistError:
            return False
        if normalized in self.exact:
            return True
        parts = normalized.split(".")
        for index in range(1, len(parts)):
            if ".".join(parts[index:]) in self.suffixes:
                return True
        return False


@dataclass(frozen = True)
class ProxyCredential:
    token: str

    @classmethod
    def mint(cls) -> "ProxyCredential":
        return cls(secrets.token_urlsafe(32))

    def matches(self, header_value: str | None) -> bool:
        if not header_value:
            return False
        scheme, _, payload = header_value.strip().partition(" ")
        if scheme.lower() != "basic" or not payload:
            return False
        try:
            decoded = base64.b64decode(payload.strip(), validate = True).decode("utf-8")
        except (binascii.Error, UnicodeDecodeError, ValueError):
            return False
        user, _, presented = decoded.partition(":")
        return hmac.compare_digest(user, _PROXY_USER) and hmac.compare_digest(
            presented, self.token
        )

    def proxy_url(self, port: int) -> str:
        return f"http://{_PROXY_USER}:{self.token}@127.0.0.1:{port}"


def proxy_environment(port: int, credential: ProxyCredential) -> dict[str, str]:
    """Environment variables that point pip, requests, curl and git at the proxy."""
    url = credential.proxy_url(port)
    env = {key: url for key in PROXY_ENV_KEYS}
    env["NO_PROXY"] = NO_PROXY_VALUE
    env["no_proxy"] = NO_PROXY_VALUE
    return env


def public_address(address: str) -> bool:
    """True for a global unicast address; IPv4-mapped IPv6 is judged as IPv4."""
    try:
        ip = ipaddress.ip_address(address)
    except ValueError:
        return False
    mapped = getattr(ip, "ipv4_mapped", None)
    if mapped is not None:
        ip = mapped
    return bool(ip.is_global) and not ip.is_multicast


def _default_resolver(host: str, port: int) -> list[str]:
    infos = socket.getaddrinfo(host, port, socket.AF_UNSPEC, socket.SOCK_STREAM)
    addresses: list[str] = []
    for family, _, _, _, sockaddr in infos:
        if family not in (socket.AF_INET, getattr(socket, "AF_INET6", None)):
            continue
        address = str(sockaddr[0])
        if address not in addresses:
            addresses.append(address)
    return addresses


DEFAULT_RESOLVER: Callable[[str, int], list[str]] = _default_resolver


class NetworkAudit:
    """Per-launch allowed and denied counters, bounded and thread-safe."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._allowed: dict[str, int] = {}
        self._denied: dict[str, tuple[int, str]] = {}
        self._overflow = 0

    def record_allowed(self, host: str) -> None:
        with self._lock:
            if host not in self._allowed and len(self._allowed) >= MAX_AUDITED_HOSTS:
                self._overflow += 1
                return
            self._allowed[host] = self._allowed.get(host, 0) + 1

    def record_denied(self, host: str, reason: str) -> None:
        with self._lock:
            if host not in self._denied and len(self._denied) >= MAX_AUDITED_HOSTS:
                self._overflow += 1
                return
            count, _ = self._denied.get(host, (0, reason))
            self._denied[host] = (count + 1, reason)

    def summary(self) -> dict[str, object]:
        with self._lock:
            return {
                "allowed": dict(self._allowed),
                "denied": {
                    host: {"count": count, "reason": reason}
                    for host, (count, reason) in self._denied.items()
                },
                "unrecorded": self._overflow,
            }

    def denied_hosts(self) -> list[tuple[str, int, str]]:
        with self._lock:
            return [(host, count, reason) for host, (count, reason) in self._denied.items()]


class _Denied(Exception):
    def __init__(self, status: int, reason: str, host: str = "") -> None:
        super().__init__(reason)
        self.status = status
        self.reason = reason
        self.host = host


class AllowlistProxy:
    """One proxy per launch; ``close()`` stops the listener and every tunnel."""

    def __init__(
        self,
        allowlist: NetworkAllowlist,
        credential: ProxyCredential | None = None,
        *,
        resolver: Callable[[str, int], list[str]] | None = None,
        allowed_ports: Iterable[int] | None = None,
        require_public: bool | None = None,
        connect_timeout: float = CONNECT_TIMEOUT_SECONDS,
        idle_timeout: float = IDLE_TIMEOUT_SECONDS,
        max_tunnels: int = MAX_TUNNELS,
    ) -> None:
        self.allowlist = allowlist
        self.credential = credential or ProxyCredential.mint()
        self.audit = NetworkAudit()
        # Module-level defaults are read at construction so a test can point
        # the proxy at a loopback upstream without reaching into the sandbox
        # backend that builds it.
        self._resolver = resolver or DEFAULT_RESOLVER
        self._allowed_ports = frozenset(allowed_ports) if allowed_ports is not None else ALLOWED_PORTS
        self._require_public = REQUIRE_PUBLIC_ADDRESSES if require_public is None else require_public
        self._connect_timeout = connect_timeout
        self._idle_timeout = idle_timeout
        self._slots = threading.BoundedSemaphore(max_tunnels)
        self._listener: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._closed = threading.Event()
        self._tunnels: set[socket.socket] = set()
        self._tunnel_lock = threading.Lock()

    # -- lifecycle -----------------------------------------------------------

    @property
    def port(self) -> int:
        if self._listener is None:
            raise RuntimeError("the proxy is not listening")
        return int(self._listener.getsockname()[1])

    def listen_loopback(self) -> int:
        """Bind an ephemeral port on the host's loopback (macOS path)."""
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            listener.bind(("127.0.0.1", 0))
            listener.listen(64)
        except OSError:
            listener.close()
            raise
        self.serve_listener(listener)
        return self.port

    def serve_listener(self, listener: socket.socket) -> None:
        """Accept on a listener the caller already bound (Linux passes one from the sandbox)."""
        if self._listener is not None:
            raise RuntimeError("the proxy already serves a listener")
        listener.setblocking(True)
        self._listener = listener
        self._thread = threading.Thread(
            target = self._accept_loop, name = "studio-tool-network-proxy", daemon = True
        )
        self._thread.start()

    def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        listener = self._listener
        if listener is not None:
            try:
                listener.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            listener.close()
        with self._tunnel_lock:
            pending = list(self._tunnels)
        for sock in pending:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                sock.close()
            except OSError:
                pass
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout = 2.0)

    def __enter__(self) -> "AllowlistProxy":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- accept and tunnel ---------------------------------------------------

    def _accept_loop(self) -> None:
        listener = self._listener
        assert listener is not None
        while not self._closed.is_set():
            try:
                client, _ = listener.accept()
            except OSError:
                if self._closed.is_set():
                    return
                time.sleep(0.05)
                continue
            if not self._slots.acquire(blocking = False):
                self._refuse(client, 503, "too many concurrent tunnels")
                continue
            threading.Thread(
                target = self._serve_client, args = (client,), daemon = True
            ).start()

    def _track(self, sock: socket.socket) -> None:
        with self._tunnel_lock:
            self._tunnels.add(sock)

    def _untrack(self, sock: socket.socket) -> None:
        with self._tunnel_lock:
            self._tunnels.discard(sock)

    def _serve_client(self, client: socket.socket) -> None:
        upstream: socket.socket | None = None
        self._track(client)
        try:
            host, port = self._authorize(client)
            upstream = self._connect_upstream(host, port)
            self.audit.record_allowed(host)
            self._track(upstream)
            client.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
            self._splice(client, upstream)
        except _Denied as denied:
            if denied.host:
                self.audit.record_denied(denied.host, denied.reason)
            self._refuse(client, denied.status, denied.reason)
        except OSError as exc:
            logger.debug("Tool network tunnel ended: %s", exc)
        finally:
            for sock in (client, upstream):
                if sock is None:
                    continue
                self._untrack(sock)
                try:
                    sock.close()
                except OSError:
                    pass
            self._slots.release()

    def _refuse(self, client: socket.socket, status: int, reason: str) -> None:
        reasons = {
            400: "Bad Request",
            403: "Forbidden",
            405: "Method Not Allowed",
            407: "Proxy Authentication Required",
            502: "Bad Gateway",
            503: "Service Unavailable",
        }
        body = reason.encode("utf-8", "replace")
        head = (
            f"HTTP/1.1 {status} {reasons.get(status, 'Error')}\r\n"
            "Content-Type: text/plain; charset=utf-8\r\n"
            f"Content-Length: {len(body)}\r\n"
            "Connection: close\r\n"
        )
        if status == 407:
            head += 'Proxy-Authenticate: Basic realm="unsloth-studio-tool-sandbox"\r\n'
        head += "X-Unsloth-Sandbox-Network: refused\r\n\r\n"
        try:
            client.sendall(head.encode("ascii") + body)
        except OSError:
            pass
        finally:
            try:
                client.close()
            except OSError:
                pass

    def _read_head(self, client: socket.socket) -> bytes:
        client.settimeout(HEADER_TIMEOUT_SECONDS)
        buffer = b""
        while b"\r\n\r\n" not in buffer:
            try:
                chunk = client.recv(4096)
            except socket.timeout as exc:
                raise _Denied(400, "request header timed out") from exc
            if not chunk:
                raise _Denied(400, "connection closed before the request head")
            buffer += chunk
            if len(buffer) > MAX_HEADER_BYTES:
                raise _Denied(400, "request head too large")
        return buffer

    def _authorize(self, client: socket.socket) -> tuple[str, int]:
        head = self._read_head(client)
        try:
            text = head.split(b"\r\n\r\n", 1)[0].decode("latin-1")
        except UnicodeDecodeError as exc:  # pragma: no cover - latin-1 cannot fail
            raise _Denied(400, "undecodable request head") from exc
        lines = text.split("\r\n")
        parts = lines[0].split(" ")
        if len(parts) != 3 or not parts[2].startswith("HTTP/1."):
            raise _Denied(400, "malformed request line")
        method, target, _ = parts
        headers: dict[str, str] = {}
        for line in lines[1:]:
            name, sep, value = line.partition(":")
            if sep:
                headers[name.strip().lower()] = value.strip()
        if not self.credential.matches(headers.get("proxy-authorization")):
            raise _Denied(407, "proxy credential missing or wrong")
        if method.upper() != "CONNECT":
            shown = self._host_from_absolute_target(target)
            raise _Denied(
                405,
                "only CONNECT tunnels are proxied; cleartext http:// requests are refused",
                shown,
            )
        return self._parse_authority(target)

    @staticmethod
    def _host_from_absolute_target(target: str) -> str:
        if "://" not in target:
            return ""
        rest = target.split("://", 1)[1]
        authority = rest.split("/", 1)[0]
        authority = authority.rsplit("@", 1)[-1]
        host = authority.rsplit(":", 1)[0] if authority.count(":") == 1 else authority
        try:
            return normalize_host(host)
        except AllowlistError:
            return host[:253]

    def _parse_authority(self, target: str) -> tuple[str, int]:
        if target.startswith("["):
            raise _Denied(403, "IPv6 literals are not allowed", target[:64])
        host, sep, port_text = target.rpartition(":")
        if not sep or not host:
            raise _Denied(400, "CONNECT target must be host:port")
        if not port_text.isdigit():
            raise _Denied(400, "CONNECT port must be numeric")
        port = int(port_text)
        try:
            normalized = normalize_host(host)
        except AllowlistError as exc:
            raise _Denied(403, f"host refused: {exc}", host[:253]) from exc
        if port not in self._allowed_ports:
            raise _Denied(
                403,
                f"port {port} is not allowed; only "
                + ", ".join(str(item) for item in sorted(self._allowed_ports))
                + " may be tunneled",
                normalized,
            )
        if not self.allowlist.allows(normalized):
            raise _Denied(403, "host is not on the network allowlist", normalized)
        return normalized, port

    def _connect_upstream(self, host: str, port: int) -> socket.socket:
        try:
            addresses = self._resolver(host, port)
        except OSError as exc:
            raise _Denied(502, f"could not resolve {host}: {exc}", host) from exc
        if not addresses:
            raise _Denied(502, f"{host} did not resolve", host)
        if self._require_public:
            for address in addresses:
                if not public_address(address):
                    raise _Denied(
                        403,
                        f"{host} resolved to a non-public address and was refused",
                        host,
                    )
        last_error: OSError | None = None
        for address in addresses:
            try:
                upstream = socket.create_connection(
                    (address, port), timeout = self._connect_timeout
                )
            except OSError as exc:
                last_error = exc
                continue
            upstream.settimeout(None)
            return upstream
        raise _Denied(502, f"could not connect to {host}: {last_error}", host)

    def _splice(self, client: socket.socket, upstream: socket.socket) -> None:
        client.setblocking(False)
        upstream.setblocking(False)
        pairs = {client: upstream, upstream: client}
        open_sides = {client, upstream}
        while open_sides and not self._closed.is_set():
            readable, _, errored = select.select(
                list(open_sides), [], list(open_sides), self._idle_timeout
            )
            if errored or not readable:
                # Idle past the cap, or a socket error: end the tunnel.
                return
            for sock in readable:
                try:
                    data = sock.recv(65536)
                except (BlockingIOError, InterruptedError):
                    continue
                except OSError:
                    return
                peer = pairs[sock]
                if not data:
                    open_sides.discard(sock)
                    try:
                        peer.shutdown(socket.SHUT_WR)
                    except OSError:
                        return
                    continue
                try:
                    _send_all_blocking(peer, data, self._idle_timeout)
                except OSError:
                    return


def _send_all_blocking(sock: socket.socket, data: bytes, timeout: float) -> None:
    view = memoryview(data)
    deadline = time.monotonic() + timeout
    while view:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise socket.timeout("tunnel write timed out")
        _, writable, _ = select.select([], [sock], [], remaining)
        if not writable:
            continue
        try:
            sent = sock.send(view)
        except (BlockingIOError, InterruptedError):
            continue
        view = view[sent:]


def format_denied_trailer(audit: NetworkAudit) -> str:
    """The tool-result trailer naming refused hosts, or an empty string."""
    denied = audit.denied_hosts()
    if not denied:
        return ""
    lines = ["", "[network] Connections refused by the sandbox network allowlist:"]
    for host, count, reason in sorted(denied)[:20]:
        shown = host or "(no host)"
        suffix = f" ({count} attempts)" if count > 1 else ""
        lines.append(f"  - {shown}{suffix}: {reason}")
    if len(denied) > 20:
        lines.append(f"  - and {len(denied) - 20} more")
    return "\n".join(lines)
