# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
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
# Deliberately narrow: package indexes, model hubs and source hosting only. No
# general cloud storage or CDN wildcards (any bucket there is a ready-made drop
# box for exfiltration); operators extend the list through ALLOWLIST_ENV.
DEFAULT_ALLOWLIST: tuple[str, ...] = (
    "pypi.org",
    "files.pythonhosted.org",
    "huggingface.co",
    "*.huggingface.co",
    "hf.co",
    "*.hf.co",
    "github.com",
    "api.github.com",
    "codeload.github.com",
    "raw.githubusercontent.com",
    "objects.githubusercontent.com",
    "release-assets.githubusercontent.com",
    "download.pytorch.org",
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
# Every proxy in this process shares this cap as well, so a session that opens
# many launches at once cannot turn the backend into a few hundred threads.
MAX_TOTAL_TUNNELS = 256
# A ClientHello that does not fit in this many bytes is not inspected; the
# tunnel proceeds and the audit counts it as one without a checked name.
MAX_CLIENT_HELLO_BYTES = 8 * 1024
CONNECT_TIMEOUT_SECONDS = 20.0
IDLE_TIMEOUT_SECONDS = 120.0
HEADER_TIMEOUT_SECONDS = 15.0
MAX_AUDITED_HOSTS = 128

# ``fullmatch`` and not ``match``: with ``$`` a trailing newline still matched,
# so a label ending in a control character passed validation.
_LABEL_RE = re.compile(r"[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?")
# Every character a hostname may hold once IDNA encoding has run. Anything else
# (control characters, spaces, quotes) is refused before the label checks, so it
# can never reach the audit or the model-facing trailer.
_HOST_CHARS_RE = re.compile(r"[A-Za-z0-9.-]+")
# The audit boundary. A denied host comes straight off an untrusted CONNECT
# line, so it is coerced to the characters a hostname may hold plus the "*" a
# wildcard entry shows; a reason is coerced to printable ASCII. Neither can then
# carry a newline into the tool result the model reads.
_AUDIT_HOST_RE = re.compile(r"[^A-Za-z0-9.\-*]")
_AUDIT_REASON_RE = re.compile(r"[^\x20-\x7e]")
MAX_AUDIT_HOST_CHARS = 253
MAX_AUDIT_REASON_CHARS = 200
MAX_TRAILER_ENTRIES = 20

# How a denial is worded to the model. An upstream failure is not the allowlist
# refusing a host, and saying so sends the model chasing a policy that allowed
# the host all along.
POLICY_REFUSAL = "policy"
UPSTREAM_FAILURE = "upstream"
PROXY_REFUSAL = "proxy"
_DENIAL_HEADINGS = {
    POLICY_REFUSAL: "[network] Connections refused by the sandbox network allowlist:",
    UPSTREAM_FAILURE: "[network] Connections that could not be reached:",
    PROXY_REFUSAL: "[network] Connections the sandbox network proxy could not serve:",
}
_DENIAL_ORDER = (POLICY_REFUSAL, UPSTREAM_FAILURE, PROXY_REFUSAL)

_PROXY_USER = "sandbox"
_PROXY_USER_BYTES = _PROXY_USER.encode("ascii")
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


def sanitize_audit_host(host: object) -> str:
    """Coerce a host to something safe to show on one line of a tool result."""
    text = host if isinstance(host, str) else str(host)
    return _AUDIT_HOST_RE.sub("?", text)[:MAX_AUDIT_HOST_CHARS]


def sanitize_audit_reason(reason: object) -> str:
    """Coerce a denial reason to printable ASCII on a single line."""
    text = reason if isinstance(reason, str) else str(reason)
    return _AUDIT_REASON_RE.sub("?", text)[:MAX_AUDIT_REASON_CHARS]


def _denial_kind(status: int) -> str:
    if status == 502:
        return UPSTREAM_FAILURE
    if status == 503:
        return PROXY_REFUSAL
    return POLICY_REFUSAL


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
    except (OSError, ValueError, UnicodeEncodeError):
        # ValueError is what an embedded NUL raises, not OSError.
        return False


def normalize_host(host: str) -> str:
    """Return a lowercase ASCII hostname or raise ``AllowlistError``.

    IDNA-encodes Unicode names, strips one trailing dot, refuses IP literals
    (in any spelling), ``localhost`` and ``.local`` names, and every label that
    is not a valid DNS label.
    """
    if not isinstance(host, str):
        raise AllowlistError("host must be a string")
    # Every message below echoes the sanitized spelling: an AllowlistError text
    # can end up in the audit, and from there in front of the model.
    shown = sanitize_audit_host(host)
    stripped = host.strip().rstrip(".")
    if not stripped:
        raise AllowlistError("empty host")
    if len(stripped) > 253:
        raise AllowlistError("host is too long")
    if _is_ip_literal(stripped):
        raise AllowlistError(f"IP literals are not allowed: {shown!r}")
    try:
        ascii_host = stripped.encode("idna").decode("ascii").lower()
    except UnicodeError as exc:
        raise AllowlistError(f"host is not a valid IDNA name: {shown!r}") from exc
    if _is_ip_literal(ascii_host):
        raise AllowlistError(f"IP literals are not allowed: {shown!r}")
    if not _HOST_CHARS_RE.fullmatch(ascii_host):
        raise AllowlistError(f"host holds a character a hostname may not: {shown!r}")
    labels = ascii_host.split(".")
    for label in labels:
        if not _LABEL_RE.fullmatch(label):
            raise AllowlistError(
                f"invalid hostname label {sanitize_audit_host(label)!r} in {shown!r}"
            )
    if labels[-1].isdigit():
        raise AllowlistError(f"numeric top-level label in {shown!r}")
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
            decoded = base64.b64decode(payload.strip(), validate = True)
        except (binascii.Error, ValueError):
            return False
        # Bytes, not str: ``compare_digest`` raises TypeError on a str holding
        # non-ASCII, and an unauthenticated request must never raise.
        user, sep, presented = decoded.partition(b":")
        if not sep:
            return False
        user_ok = hmac.compare_digest(user, _PROXY_USER_BYTES)
        token_ok = hmac.compare_digest(presented, self.token.encode("utf-8"))
        return user_ok and token_ok

    def proxy_url(self, port: int) -> str:
        return f"http://{_PROXY_USER}:{self.token}@127.0.0.1:{port}"


def proxy_environment(port: int, credential: ProxyCredential) -> dict[str, str]:
    """Environment variables that point pip, requests, curl and git at the proxy."""
    url = credential.proxy_url(port)
    env = {key: url for key in PROXY_ENV_KEYS}
    env["NO_PROXY"] = NO_PROXY_VALUE
    env["no_proxy"] = NO_PROXY_VALUE
    return env


def _openssl_default_paths() -> tuple[str | None, str | None]:
    """OpenSSL's compiled-in cafile and capath as the host sees them (None when absent)."""
    try:
        import ssl

        paths = ssl.get_default_verify_paths()
    except Exception:  # noqa: BLE001 - a broken ssl build is treated as no defaults
        return None, None
    cafile = paths.cafile if paths.cafile and os.path.isfile(paths.cafile) else None
    capath = paths.capath if paths.capath and os.path.isdir(paths.capath) else None
    return cafile, capath


def _certifi_bundle() -> str | None:
    try:
        import certifi  # type: ignore[import-not-found]

        bundle = certifi.where()
    except Exception:  # noqa: BLE001 - optional dependency
        return None
    return bundle if isinstance(bundle, str) and os.path.isfile(bundle) else None


def tls_trust_paths() -> tuple[str, ...]:
    """Paths the sandbox must be able to read for TLS verification to work.

    OpenSSL's default store often lives outside the system roots the sandbox
    exposes: python.org and hosted-toolcache builds keep it under the
    interpreter's own ``etc/openssl``, which is not part of the runtime tree the
    backends bind. certifi's bundle is returned too so the environment fallback
    below has something readable to point at.
    """
    cafile, capath = _openssl_default_paths()
    candidates = [
        os.path.dirname(cafile) if cafile else None,
        capath,
        _certifi_bundle(),
    ]
    seen: list[str] = []
    for path in candidates:
        if path and os.path.exists(path) and path not in seen:
            seen.append(path)
    return tuple(seen)


def tls_trust_environment(base: dict[str, str] | None = None) -> dict[str, str]:
    """``SSL_CERT_FILE`` naming a bundle the sandboxed interpreter can verify against.

    The host's OpenSSL default cafile when it exists (it is exposed read-only by
    ``tls_trust_paths``), else certifi's bundle: python.org and hosted-toolcache
    builds on macOS report default cert paths that do not exist, so ``urllib``
    fails every HTTPS request with CERTIFICATE_VERIFY_FAILED while pip (which
    vendors certifi) works. A caller that already set the variable keeps its
    value; pip and requests read the same variables.
    """
    if base and any(key in base for key in ("SSL_CERT_FILE", "SSL_CERT_DIR")):
        return {}
    cafile, _capath = _openssl_default_paths()
    bundle = cafile or _certifi_bundle()
    if not bundle:
        return {}
    return {"SSL_CERT_FILE": bundle, "REQUESTS_CA_BUNDLE": bundle}


_NAT64_WELL_KNOWN = ipaddress.IPv6Network("64:ff9b::/96")
_NAT64_LOCAL_USE = ipaddress.IPv6Network("64:ff9b:1::/48")
_SIX_TO_FOUR = ipaddress.IPv6Network("2002::/16")


def _embedded_ipv4(ip: ipaddress.IPv6Address) -> ipaddress.IPv4Address | None:
    """The IPv4 address an IPv6 form carries, for the forms ``ipv4_mapped`` misses.

    ``ipv4_mapped`` only covers ``::ffff:0:0/96``. ``::7f00:1`` (deprecated
    IPv4-compatible) and ``64:ff9b::7f00:1`` (the RFC 6052 NAT64 form of
    127.0.0.1) are both reported global by ``ipaddress``, so on a host with a
    NAT64 gateway an allowlisted name could resolve back to loopback.
    """
    packed = ip.packed
    if packed[:12] == b"\x00" * 12:
        return ipaddress.IPv4Address(packed[12:16])
    if ip in _NAT64_WELL_KNOWN:
        return ipaddress.IPv4Address(packed[12:16])
    if ip in _NAT64_LOCAL_USE:
        # RFC 6052 skips the byte at offset 8 for prefixes shorter than /96.
        return ipaddress.IPv4Address(bytes((packed[6], packed[7], packed[9], packed[10])))
    if ip in _SIX_TO_FOUR:
        return ipaddress.IPv4Address(packed[2:6])
    return None


def public_address(address: str) -> bool:
    """True for a global unicast address; every IPv4-in-IPv6 form is judged as IPv4."""
    try:
        ip = ipaddress.ip_address(address)
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv6Address):
        mapped = ip.ipv4_mapped
        if mapped is None:
            mapped = _embedded_ipv4(ip)
        if mapped is not None:
            ip = mapped
    return bool(ip.is_global) and not ip.is_multicast


def _client_hello_sni(data: bytes) -> tuple[str, str]:
    """Read the SNI out of a TLS ClientHello.

    Returns ``(status, name)`` where status is one of ``incomplete`` (more bytes
    may decide it), ``not-tls`` (never will), ``absent`` (a ClientHello with no
    server name) or ``found``. The parse is deliberately total: any surprise is
    ``not-tls``, which lets the tunnel through rather than breaking a client
    speaking something the sandbox is allowed to speak.
    """
    if len(data) < 5:
        return ("incomplete", "")
    if data[0] != 0x16:  # not a TLS handshake record
        return ("not-tls", "")
    record_length = int.from_bytes(data[3:5], "big")
    if record_length == 0 or record_length > MAX_CLIENT_HELLO_BYTES:
        return ("not-tls", "")
    if len(data) < 5 + record_length:
        return ("incomplete", "")
    body = data[5 : 5 + record_length]
    try:
        if len(body) < 4 or body[0] != 0x01:  # not a ClientHello
            return ("not-tls", "")
        handshake_length = int.from_bytes(body[1:4], "big")
        hello = body[4 : 4 + handshake_length]
        if len(hello) < handshake_length:
            # Fragmented across records; do not guess at a name.
            return ("not-tls", "")
        pos = 2 + 32  # legacy_version and random
        if len(hello) < pos + 1:
            return ("not-tls", "")
        pos += 1 + hello[pos]  # legacy_session_id
        if len(hello) < pos + 2:
            return ("not-tls", "")
        pos += 2 + int.from_bytes(hello[pos : pos + 2], "big")  # cipher_suites
        if len(hello) < pos + 1:
            return ("not-tls", "")
        pos += 1 + hello[pos]  # legacy_compression_methods
        if len(hello) < pos + 2:
            return ("absent", "")
        extensions_end = min(len(hello), pos + 2 + int.from_bytes(hello[pos : pos + 2], "big"))
        pos += 2
        while pos + 4 <= extensions_end:
            kind = int.from_bytes(hello[pos : pos + 2], "big")
            length = int.from_bytes(hello[pos + 2 : pos + 4], "big")
            pos += 4
            chunk = hello[pos : pos + length]
            pos += length
            if kind != 0 or len(chunk) < 2:
                continue
            entries = chunk[2 : 2 + int.from_bytes(chunk[0:2], "big")]
            cursor = 0
            while cursor + 3 <= len(entries):
                name_kind = entries[cursor]
                name_length = int.from_bytes(entries[cursor + 1 : cursor + 3], "big")
                cursor += 3
                name = entries[cursor : cursor + name_length]
                cursor += name_length
                if name_kind == 0 and len(name) == name_length and name:
                    return ("found", name.decode("latin-1"))
            return ("absent", "")
        return ("absent", "")
    except (IndexError, ValueError):  # pragma: no cover - the slicing above is total
        return ("not-tls", "")


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
        # Keyed by (host, reason, kind). Keying by host alone kept only the most
        # recent reason, so a host blocked fifty times by the allowlist and once
        # by a DNS hiccup was reported to the model as a DNS problem.
        self._denied: dict[tuple[str, str, str], int] = {}
        self._overflow = 0
        self._sni_absent = 0

    def record_allowed(self, host: str) -> None:
        with self._lock:
            if host not in self._allowed and len(self._allowed) >= MAX_AUDITED_HOSTS:
                self._overflow += 1
                return
            self._allowed[host] = self._allowed.get(host, 0) + 1

    def record_sni_absent(self) -> None:
        """A tunnel whose first bytes carried no server name to check."""
        with self._lock:
            self._sni_absent += 1

    def record_denied(self, host: str, reason: str, kind: str = POLICY_REFUSAL) -> None:
        # Sanitize here, not at the point of display: this is the boundary where
        # an untrusted CONNECT authority enters data the model is shown.
        key = (
            sanitize_audit_host(host),
            sanitize_audit_reason(reason),
            kind if kind in _DENIAL_HEADINGS else POLICY_REFUSAL,
        )
        with self._lock:
            if key not in self._denied and len(self._denied) >= MAX_AUDITED_HOSTS:
                self._overflow += 1
                return
            self._denied[key] = self._denied.get(key, 0) + 1

    def summary(self) -> dict[str, object]:
        with self._lock:
            allowed = dict(self._allowed)
            entries = list(self._denied.items())
            overflow = self._overflow
            sni_absent = self._sni_absent
        denied: dict[str, dict[str, object]] = {}
        for (host, reason, kind), count in entries:
            record = denied.get(host)
            if record is None:
                denied[host] = {
                    "count": count,
                    "reason": reason,
                    "kind": kind,
                    "reasons": [{"reason": reason, "kind": kind, "count": count}],
                }
                continue
            record["count"] = int(record["count"]) + count
            reasons = record["reasons"]
            assert isinstance(reasons, list)
            reasons.append({"reason": reason, "kind": kind, "count": count})
        return {
            "allowed": allowed,
            "denied": denied,
            "unrecorded": overflow,
            "sni_absent": sni_absent,
        }

    def denied_entries(self) -> list[tuple[str, int, str, str]]:
        """One entry per (host, reason, kind), so no reason is overwritten."""
        with self._lock:
            return [
                (host, count, reason, kind)
                for (host, reason, kind), count in self._denied.items()
            ]

    def denied_hosts(self) -> list[tuple[str, int, str]]:
        return [(host, count, reason) for host, count, reason, _ in self.denied_entries()]


class _Denied(Exception):
    def __init__(self, status: int, reason: str, host: str = "", kind: str | None = None) -> None:
        super().__init__(reason)
        self.status = status
        self.reason = reason
        self.host = host
        self.kind = kind or _denial_kind(status)


class _TunnelAborted(Exception):
    """A tunnel refused after the 200, where no HTTP status can be sent."""

    def __init__(self, host: str, reason: str) -> None:
        super().__init__(reason)
        self.host = host
        self.reason = reason


class AllowlistProxy:
    """One proxy per launch; ``close()`` stops the listener and every tunnel."""

    # Shared by every instance, so the per-launch cap is not the only bound on
    # how many tunnels this process can hold. A test lowers it on the class.
    global_slots = threading.BoundedSemaphore(MAX_TOTAL_TUNNELS)

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
        header_timeout: float = HEADER_TIMEOUT_SECONDS,
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
        # A wall-clock budget for the whole request head, not per recv: a client
        # dribbling one byte every few seconds otherwise held a tunnel slot for
        # hours and could starve the launch of every slot.
        self._header_timeout = header_timeout
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
                # Recorded, so the model is told why pip failed instead of
                # seeing a bare proxy error with no explanation.
                self.audit.record_denied(
                    "",
                    "the sandbox network proxy was at its concurrent tunnel cap",
                    PROXY_REFUSAL,
                )
                self._dispatch_refusal(client, 503, "too many concurrent tunnels")
                continue
            if not self.global_slots.acquire(blocking = False):
                self._slots.release()
                self.audit.record_denied(
                    "",
                    "the backend was at its process-wide tunnel cap",
                    PROXY_REFUSAL,
                )
                self._dispatch_refusal(
                    client, 503, "too many concurrent tunnels in this backend"
                )
                continue
            try:
                threading.Thread(
                    target = self._serve_client, args = (client,), daemon = True
                ).start()
            except BaseException as exc:
                # RuntimeError("can't start new thread") would otherwise burn the
                # slot for the life of the launch and leak the accepted socket.
                logger.warning("Tool network proxy could not start a worker: %s", exc)
                self._release_slots()
                try:
                    client.close()
                except OSError:
                    pass

    def _dispatch_refusal(self, client: socket.socket, status: int, reason: str) -> None:
        """Answer a refusal off the accept thread, with a timeout on the write.

        ``accept`` hands back a blocking socket with no timeout, so a client
        advertising a zero receive window would otherwise wedge the single
        accept thread inside ``sendall`` and stop the proxy serving anyone.
        """
        try:
            client.settimeout(self._header_timeout)
        except OSError:
            pass
        try:
            threading.Thread(
                target = self._refuse,
                args = (client, status, reason),
                name = "studio-tool-network-refuse",
                daemon = True,
            ).start()
        except BaseException as exc:
            logger.warning("Tool network proxy could not start a refusal worker: %s", exc)
            self._refuse(client, status, reason)

    def _release_slots(self) -> None:
        for semaphore in (self._slots, self.global_slots):
            try:
                semaphore.release()
            except ValueError:  # pragma: no cover - a release without an acquire
                logger.warning("Tool network proxy released a tunnel slot twice")

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
            host, port, pipelined = self._authorize(client)
            upstream = self._connect_upstream(host, port)
            self.audit.record_allowed(host)
            self._track(upstream)
            client.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
            pipelined = self._check_server_name(client, host, pipelined)
            if pipelined:
                # A client that wrote its ClientHello in the same segment as the
                # CONNECT head would otherwise hang until the idle timeout.
                upstream.sendall(pipelined)
            self._splice(client, upstream)
        except _TunnelAborted as aborted:
            self.audit.record_denied(aborted.host, aborted.reason, POLICY_REFUSAL)
            logger.debug("Tool network tunnel aborted: %s", aborted.reason)
        except _Denied as denied:
            if denied.host or denied.kind != POLICY_REFUSAL:
                self.audit.record_denied(denied.host, denied.reason, denied.kind)
            self._refuse(client, denied.status, denied.reason)
        except OSError as exc:
            logger.debug("Tool network tunnel ended: %s", exc)
        except Exception as exc:  # noqa: BLE001 - one request must not kill the worker
            logger.warning(
                "Tool network proxy failed to serve a request: %s", exc, exc_info = True
            )
            self._refuse(client, 400, "the proxy could not process this request")
        finally:
            for sock in (client, upstream):
                if sock is None:
                    continue
                self._untrack(sock)
                try:
                    sock.close()
                except OSError:
                    pass
            self._release_slots()

    def _check_server_name(self, client: socket.socket, host: str, pipelined: bytes) -> bytes:
        """Refuse a tunnel whose ClientHello names a host other than the CONNECT one.

        Without this the allowlist checks only the name in the CONNECT line: a
        client can name an allowlisted host, then ask the shared front end it
        lands on for a different site entirely (domain fronting). A client that
        sends no server name is let through, since that is legal, but the audit
        counts it so the gap is visible.
        """
        buffer = pipelined
        deadline = time.monotonic() + self._header_timeout
        status, name = _client_hello_sni(buffer)
        while status == "incomplete" and len(buffer) < MAX_CLIENT_HELLO_BYTES:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                client.settimeout(max(0.1, remaining))
                chunk = client.recv(4096)
            except OSError:
                break
            if not chunk:
                break
            buffer += chunk
            status, name = _client_hello_sni(buffer)
        if status == "found":
            if name.strip().rstrip(".").lower() != host:
                raise _TunnelAborted(host, "SNI does not match the CONNECT host")
            return buffer
        self.audit.record_sni_absent()
        return buffer

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

    def _read_head(self, client: socket.socket) -> tuple[bytes, bytes]:
        """The request head and whatever the client pipelined behind it."""
        deadline = time.monotonic() + self._header_timeout
        buffer = b""
        while b"\r\n\r\n" not in buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise _Denied(400, "request header timed out")
            client.settimeout(max(0.1, remaining))
            try:
                chunk = client.recv(4096)
            except socket.timeout as exc:
                raise _Denied(400, "request header timed out") from exc
            if not chunk:
                raise _Denied(400, "connection closed before the request head")
            buffer += chunk
            if len(buffer) > MAX_HEADER_BYTES:
                raise _Denied(400, "request head too large")
        client.settimeout(self._header_timeout)
        head, _, pipelined = buffer.partition(b"\r\n\r\n")
        return head, pipelined

    def _authorize(self, client: socket.socket) -> tuple[str, int, bytes]:
        head, pipelined = self._read_head(client)
        try:
            text = head.decode("latin-1")
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
        host, port = self._parse_authority(target)
        return host, port, pipelined

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
            return sanitize_audit_host(host)

    def _parse_authority(self, target: str) -> tuple[str, int]:
        if target.startswith("["):
            raise _Denied(403, "IPv6 literals are not allowed", sanitize_audit_host(target[:64]))
        host, sep, port_text = target.rpartition(":")
        if not sep or not host:
            raise _Denied(400, "CONNECT target must be host:port")
        if not port_text.isdigit():
            raise _Denied(400, "CONNECT port must be numeric")
        port = int(port_text)
        try:
            normalized = normalize_host(host)
        except AllowlistError as exc:
            raise _Denied(403, f"host refused: {exc}", sanitize_audit_host(host)) from exc
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
        # One budget for the whole loop, not per answer: a name with a dozen
        # black-holed answers would otherwise hold a worker for a dozen timeouts.
        deadline = time.monotonic() + self._connect_timeout
        for address in addresses:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                if last_error is None:
                    last_error = socket.timeout("the connect budget ran out")
                break
            try:
                upstream = socket.create_connection(
                    (address, port), timeout = min(self._connect_timeout, remaining)
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
    """The tool-result trailer naming refused hosts, or an empty string.

    Every host and reason is sanitized again here, so the trailer holds one line
    per entry and no control characters even if the audit was filled by hand.
    """
    entries = audit.denied_entries()
    if not entries:
        return ""
    grouped: dict[str, list[tuple[str, int, str]]] = {}
    for host, count, reason, kind in entries:
        bucket = kind if kind in _DENIAL_HEADINGS else POLICY_REFUSAL
        grouped.setdefault(bucket, []).append(
            (sanitize_audit_host(host), count, sanitize_audit_reason(reason))
        )
    lines = [""]
    for kind in _DENIAL_ORDER:
        group = grouped.get(kind)
        if not group:
            continue
        lines.append(_DENIAL_HEADINGS[kind])
        for host, count, reason in sorted(group)[:MAX_TRAILER_ENTRIES]:
            shown = host or "(no host)"
            suffix = f" ({count} attempts)" if count > 1 else ""
            lines.append(f"  - {shown}{suffix}: {reason}")
        if len(group) > MAX_TRAILER_ENTRIES:
            lines.append(f"  - and {len(group) - MAX_TRAILER_ENTRIES} more")
    return "\n".join(lines)
