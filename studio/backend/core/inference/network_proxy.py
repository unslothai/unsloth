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
  cannot become a bridge back to the host or the LAN. Which IPv4-in-IPv6 forms
  are decoded, which are refused outright and which are knowingly let through is
  argued out in the block comment above ``_NAT64_WELL_KNOWN``.
* Authenticated. Each launch mints a random credential that only travels to
  the sandboxed process through its environment; a request without it gets 407,
  so another local user on the same machine cannot use the tunnel.
* Bounded. Header size, concurrent tunnels, connect time and idle time all have
  caps; a client that misbehaves loses its connection, never the proxy.
* Audited. Allowed and denied hosts are counted per launch so the tool result
  can name the host a script tried and was refused, instead of leaving the
  model to guess why ``pip`` failed.
* TLS, and to the host that was asked for. A tunnel must open with a TLS
  ClientHello naming the CONNECT host, or it is refused before a byte reaches
  the upstream. The block comment above ``_check_server_name`` argues out why
  a missing name and a stream that is not TLS at all are refusals and not
  allowances.

Two holes this design cannot close are named rather than implied, because a
sandbox whose UI says "only these hosts, over HTTPS" should be read with them
in mind:

* Domain fronting through the inner HTTP ``Host`` header. The proxy checks the
  CONNECT authority and the ClientHello SNI, both of which are outside the
  encryption; the request that then travels inside the TLS session can ask an
  allowlisted front end for any site it also serves. Seeing that header means
  terminating TLS, which would put the sandbox's traffic in cleartext in this
  process and hand it a certificate authority the sandbox trusts. That trade is
  refused, so this stays open and is stated here instead.
* An IPv6 answer under a NAT64 prefix that no operator named. Argued out in the
  block comment above ``_NAT64_WELL_KNOWN``.
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
from typing import Callable, Iterable, NoReturn

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
# A ClientHello whose handshake BODY does not fit in this many bytes is refused:
# a tunnel whose first message is TLS has to name the CONNECT host before it
# carries traffic, and a handshake this large is not one a client legitimately
# sends (even with post-quantum key shares a ClientHello stays a few kilobytes).
MAX_CLIENT_HELLO_BYTES = 8 * 1024
_TLS_RECORD_HEADER_BYTES = 5
_TLS_HANDSHAKE_HEADER_BYTES = 4
# The most records one ClientHello may be cut into. Real stacks send one, and the
# split hello a test pins down uses two; the bound is here so that a client
# dribbling a byte per record cannot grow the wire allowance below without end.
MAX_CLIENT_HELLO_RECORDS = 64
# What the receive loop may buffer off the wire. The cap above bounds the
# handshake body, which is what the reassembler measures, but the wire also
# carries the four byte handshake header and five more bytes for every record the
# body is cut into. Comparing wire bytes against the body cap refused a permitted
# hello purely because it was framed, so the framing is budgeted separately.
MAX_CLIENT_HELLO_WIRE_BYTES = (
    MAX_CLIENT_HELLO_BYTES
    + _TLS_HANDSHAKE_HEADER_BYTES
    + MAX_CLIENT_HELLO_RECORDS * _TLS_RECORD_HEADER_BYTES
)
# The largest a single TLS record may be: 2^14 of plaintext plus the expansion
# TLS 1.2 allows. Records are reassembled, so a hello spread over several of
# them is still read; the caps above bound the total either way.
_MAX_TLS_RECORD_BYTES = 16 * 1024 + 2048
# How many sockets may be waiting to be told "no" at once. A refusal writes one
# short response, but a client that never reads holds its worker until the
# header timeout, so without a bound every socket accepted past the tunnel cap
# would get a thread of its own.
MAX_REFUSAL_WORKERS = 8
CONNECT_TIMEOUT_SECONDS = 20.0
IDLE_TIMEOUT_SECONDS = 120.0
HEADER_TIMEOUT_SECONDS = 15.0
MAX_AUDITED_HOSTS = 128
# How long ``close`` waits for the accept thread, and then for every worker
# still holding a socket. Two bounded waits rather than one unbounded one: the
# cleanup path runs at the end of every tool call, so it may not hang, and it
# may not return while a thread it started is still splicing bytes.
CLOSE_JOIN_SECONDS = 2.0

# The whole record for a fatal ``unrecognized_name(112)`` alert: content type
# 21, legacy_record_version 0x0303 (RFC 8446 section 5.1 asks every record but
# an initial ClientHello for that value), a two byte body, level fatal(2),
# description 112. RFC 6066 section 3 assigns this alert to a server that will
# not serve the name the ClientHello asked for, which is this decision exactly,
# so a client reports "tlsv1 alert unrecognized name" and a reader can check the
# code against the RFC that defines the extension being enforced.
TLS_UNRECOGNIZED_NAME_ALERT = b"\x15\x03\x03\x00\x02\x02\x70"

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
                raise AllowlistError(f"only a leading '*.' wildcard is supported: {entry!r}")
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


def _openssl_verify_paths() -> object | None:
    """``ssl.get_default_verify_paths()``, or None on a build that cannot report it."""
    try:
        import ssl
        return ssl.get_default_verify_paths()
    except Exception:  # noqa: BLE001 - a broken ssl build is treated as no defaults
        return None


def _openssl_default_paths() -> tuple[str | None, str | None]:
    """The cafile and capath OpenSSL will actually use (None when absent).

    Not purely compiled-in: ``ssl.get_default_verify_paths`` lets ``SSL_CERT_FILE``
    and ``SSL_CERT_DIR`` shadow the build paths, exactly as OpenSSL does for the
    interpreter itself, so either of these may be a path an operator chose.
    """
    paths = _openssl_verify_paths()
    if paths is None:
        return None, None
    cafile = getattr(paths, "cafile", None)
    capath = getattr(paths, "capath", None)
    cafile = cafile if cafile and os.path.isfile(cafile) else None
    capath = capath if capath and os.path.isdir(capath) else None
    return cafile, capath


# The only names OpenSSL ever opens inside a hashed certificate directory:
# ``<8 hex digits>.<n>`` for a certificate and ``<8 hex digits>.r<n>`` for a CRL.
# A capath is looked up by name, never listed, so nothing else in one is needed.
_HASHED_CERT_RE = re.compile(r"[0-9a-f]{8}\.r?[0-9]+")
# How many of those one capath may contribute. A system store holds a few
# hundred; past this the directory is dropped whole rather than turned into a
# thousand bind arguments.
MAX_CAPATH_ENTRIES = 256


def _same_path(first: str, second: str) -> bool:
    try:
        return os.path.realpath(first) == os.path.realpath(second)
    except OSError:  # pragma: no cover - realpath does not raise on Linux or macOS
        return False


def _capath_trust_paths(capath: str | None) -> tuple[str, ...]:
    """What a capath may contribute to the set of paths the sandbox can read.

    A capath is a directory, so handing it over hands over everything that sits
    beside the certificates in it. Whose directory it is decides the treatment:

    * the directory OpenSSL was built with is passed through as a directory. It
      is a build constant rather than input, it is normally a system store the
      backends already bind, and enumerating it would add one bind per
      certificate to every launch for nothing.
    * a directory ``SSL_CERT_DIR`` names is never passed through as a directory.
      Only the entries OpenSSL can ever open in a hashed store are returned, so
      an operator who points the variable at a directory that also holds a key
      exposes the certificates and not the key. This is the sibling of the
      cafile rule above: an environment-controlled path is not ours to widen.

    Enumeration is a snapshot, so a certificate added after the launch is not
    visible to it; a tool call is short and the alternative is exposing the whole
    directory for the life of the sandbox.
    """
    if not capath:
        return ()
    paths = _openssl_verify_paths()
    compiled = getattr(paths, "openssl_capath", "") if paths is not None else ""
    if compiled and _same_path(capath, compiled):
        return (capath,)
    try:
        names = sorted(os.listdir(capath))
    except OSError:
        return ()
    entries: list[str] = []
    for name in names:
        if not _HASHED_CERT_RE.fullmatch(name):
            continue
        candidate = os.path.join(capath, name)
        if not os.path.isfile(candidate):
            # A directory named like a hashed certificate would be exposed with
            # everything under it, which is the exposure this function exists to
            # avoid.
            continue
        entries.append(candidate)
        if len(entries) > MAX_CAPATH_ENTRIES:
            logger.warning(
                "Tool sandbox: SSL_CERT_DIR holds more than %d hashed certificates, "
                "so none of them are exposed; set SSL_CERT_FILE to a bundle instead",
                MAX_CAPATH_ENTRIES,
            )
            return ()
    return tuple(entries)


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

    Nothing an environment variable chose is returned as a directory. The cafile
    is returned as the file itself, and a capath that ``SSL_CERT_DIR`` named is
    returned as its hashed certificate entries rather than as the directory
    holding them, so an operator pointing either variable at a path that happens
    to sit beside unrelated secrets exposes the trust material and not the
    secrets. Both backends take either shape: the Linux one binds each path with
    ``--ro-bind-try`` and the macOS profile emits a ``literal`` filter, plus a
    ``subpath`` filter for the directories, which is why which of the two a path
    is matters.
    """
    cafile, capath = _openssl_default_paths()
    candidates = [
        cafile,
        *_capath_trust_paths(capath),
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


# ---------------------------------------------------------------------------
# IPv4 addresses hidden inside IPv6 answers, and where this proxy draws the line.
#
# The danger is an allowlisted name resolving to an IPv6 address that some
# translator on the way out turns back into 127.0.0.1, 10.0.0.5 or
# 169.254.169.254. RFC 6052 section 5.3 states the goal exactly: packets sent to
# an IPv4-embedded IPv6 address "should ... be subject to the same filtering as
# those directly sent to ... the embedded IPv4 addresses". The whole difficulty
# is that doing so needs the translation prefix, and an answer's bytes do not
# carry it.
#
# Two readings of that were argued over this code, and neither can win outright:
#
# * Too narrow. RFC 6052 section 1.3 lets an operator pick any Network-Specific
#   Prefix, so a DNS64 under some prefix other than 64:ff9b can hand back
#   Pref64::10.0.0.5 and it reads as an ordinary global address.
# * Too broad. Decoding every address at every RFC 6052 offset and demanding all
#   six decodes be public refuses most of the real IPv6 internet. The /32 offset
#   is bytes 4 to 7, which are zero in nearly every address written with a "::"
#   (0.0.0.0 is not public), and the /96 offset is bytes 12 to 15, so
#   2001:4860:4860::8888 decodes to 0.0.136.136 and is refused.
#
# It is tempting to split the difference on structure: RFC 6052 section 2.2 does
# require the reserved octet at byte 8 to be zero, and the suffix bits after the
# embedded address to be zero, so an address failing that shape cannot be an
# encoding at that length. As a security gate it does not hold. The same section
# makes the suffix a SHOULD and then says translators receiving non-zero suffix
# bits "SHOULD ignore the bits' value and proceed as if the bits' value were
# zero", so an attacker sets one suffix bit, the shape test skips the offset, and
# the translator still delivers to 10.0.0.5. A gate an attacker opens by
# flipping a bit that changes nothing downstream is not a gate.
#
# RFC 7050 prefix discovery (query ipv4only.arpa, learn the real Pref64::/n) was
# proposed as the way out. It is rejected here, on three grounds:
#
# * It is unauthenticatable by design. RFC 8880 section 5 requires ipv4only.arpa
#   to be an insecure delegation and says validating resolvers "MUST NOT attempt
#   to validate answers received in response to queries for the IPv6 AAAA address
#   records", explicitly correcting RFC 7050's claim that DNSSEC could protect
#   them. RFC 7050 section 7 spells out the consequence: "fake positive AAAA
#   responses could cause hosts to erroneously detect Pref64::/n, thus allowing
#   an attacker to inject malicious Pref64::/n".
# * It fails open, and the same adversary decides when. RFC 7050 section 3 ends
#   the heuristic with "the Pref64::/n cannot be determined and the heuristic
#   procedure has failed" when the well-known addresses are not found, so anyone
#   who can shape the DNS answer this check is meant to catch can withhold one
#   more answer and switch the check off.
# * It costs a DNS round trip on a path that has none today. The proxy is minted
#   per launch, so that is per tool call, and RFC 8880 section 7.1 wants the
#   query pinned to the interface's own resolver, which is not what the resolver
#   this proxy is handed does.
#
# So the resolution is to stop guessing, and to say plainly what is refused and
# what is admitted.
#
# REFUSED, on purpose:
#
# * The Well-Known Prefix 64:ff9b::/96 carrying a non-public IPv4. RFC 6052
#   section 3.1: it "MUST NOT be used to represent non-global IPv4 addresses"
#   and translators "MUST drop these packets". A public IPv4 behind it still
#   tunnels, which is what an IPv6-only host with a NAT64 needs.
# * Every other address spelled 64:ff9b, which is all of RFC 8215's local-use
#   64:ff9b:1::/48 plus the unallocated remainder of 64:ff9b::/32. This is the
#   class knowingly given up: an operator carving a /96 out of 64:ff9b:1::/48
#   (RFC 8215 section 6 suggests exactly that) cannot tunnel through here. It is
#   refused because RFC 8215 section 5 removes the section 3.1 protection there
#   ("the restrictions on the use of the WKP ... do not apply"), so such an
#   address may legally encode 169.254.169.254; because IANA's IPv6
#   Special-Purpose registry marks the block Globally Reachable: False and
#   CPython 3.13 already classifies it private; and because an allowlisted public
#   host resolving into a local-use translation range is the bridge back to the
#   LAN this proxy exists to refuse. Such an operator sets NAT64_PREFIX_ENV.
# * Any address under a prefix the operator named in NAT64_PREFIX_ENV whose
#   embedded IPv4, read at that prefix's own offset, is not public. This is the
#   sound answer to the "too narrow" reading: the prefix comes from the operator
#   rather than from the network, so it cannot be spoofed or withheld, it is
#   decoded at one offset rather than six, and it costs no round trip.
#
# ADMITTED, on purpose, and this is the hole:
#
# * A Network-Specific Prefix that the operator did not name. With no
#   configuration, an address under some other operator's /96 NAT64 prefix is
#   judged as the ordinary global unicast address it is bit-for-bit
#   indistinguishable from, and if that network's translator maps it to
#   10.0.0.5 the tunnel is allowed. Closing this by inspection means refusing
#   ordinary IPv6, so it is left open and named here instead of hidden.
# ---------------------------------------------------------------------------

# RFC 6052 section 2.1. The Well-Known Prefix is a /96 and section 2.2 says it
# "can only be used in the last form of the table", so there is exactly one place
# its IPv4 address can sit.
_NAT64_WELL_KNOWN = ipaddress.IPv6Network("64:ff9b::/96")
# Every address spelled 64:ff9b: the Well-Known Prefix, RFC 8215's local-use
# 64:ff9b:1::/48, and the unallocated space between them.
_NAT64_RESERVED = ipaddress.IPv6Network("64:ff9b::/32")
_SIX_TO_FOUR = ipaddress.IPv6Network("2002::/16")
# Where RFC 6052 section 2.2 puts the four IPv4 bytes for each prefix length it
# allows. Byte 8 is the reserved octet and is skipped, so a /40 takes bytes 5, 6,
# 7 and 9 while a /64 takes bytes 9 to 12.
_RFC6052_OFFSETS: dict[int, tuple[int, int, int, int]] = {
    32: (4, 5, 6, 7),
    40: (5, 6, 7, 9),
    48: (6, 7, 9, 10),
    56: (7, 9, 10, 11),
    64: (9, 10, 11, 12),
    96: (12, 13, 14, 15),
}

# Pref64::/n values the operator states their own network translates, comma or
# whitespace separated. Only these six lengths are RFC 6052 encodings, so only
# these can be decoded; anything else is dropped with a warning rather than
# silently ignored.
NAT64_PREFIX_ENV = "UNSLOTH_STUDIO_TOOL_NAT64_PREFIXES"
_nat64_prefix_cache: tuple[str, tuple[ipaddress.IPv6Network, ...]] | None = None


def _parse_nat64_prefixes(raw: str) -> tuple[ipaddress.IPv6Network, ...]:
    prefixes: list[ipaddress.IPv6Network] = []
    for item in re.split(r"[,\s]+", raw.strip()):
        if not item:
            continue
        try:
            network = ipaddress.IPv6Network(item, strict = False)
        except ValueError:
            logger.warning(
                "%s holds an entry that is not an IPv6 prefix: %r", NAT64_PREFIX_ENV, item
            )
            continue
        if network.prefixlen not in _RFC6052_OFFSETS:
            logger.warning(
                "%s entry %r is not an RFC 6052 prefix length (32, 40, 48, 56, 64 or 96)",
                NAT64_PREFIX_ENV,
                item,
            )
            continue
        prefixes.append(network)
    return tuple(prefixes)


def _configured_nat64_prefixes() -> tuple[ipaddress.IPv6Network, ...]:
    """The operator's Pref64::/n list, parsed once per distinct value of the variable."""
    global _nat64_prefix_cache
    raw = os.environ.get(NAT64_PREFIX_ENV, "")
    cached = _nat64_prefix_cache
    if cached is not None and cached[0] == raw:
        return cached[1]
    parsed = _parse_nat64_prefixes(raw)
    # A racing thread recomputes the same tuple, so no lock is needed.
    _nat64_prefix_cache = (raw, parsed)
    return parsed


def _rfc6052_embedded(packed: bytes, prefix_length: int) -> ipaddress.IPv4Address:
    """The IPv4 address an RFC 6052 prefix of this length embeds."""
    offsets = _RFC6052_OFFSETS[prefix_length]
    return ipaddress.IPv4Address(bytes(packed[index] for index in offsets))


def _embedded_ipv4_candidates(ip: ipaddress.IPv6Address) -> tuple[ipaddress.IPv4Address, ...]:
    """Every IPv4 address this IPv6 address is known to carry, for the forms ``ipv4_mapped`` misses.

    ``ipv4_mapped`` only covers ``::ffff:0:0/96``. ``::7f00:1`` (the deprecated
    IPv4-compatible form), ``2002:7f00:1::`` (6to4 on a Python whose
    ``ipaddress`` does not yet list 2002::/16) and ``64:ff9b::7f00:1`` (the
    RFC 6052 NAT64 form of 127.0.0.1) are all reported global by older
    ``ipaddress`` modules, so on a host with a translator an allowlisted name
    could resolve back to loopback.

    "Known to carry" is the whole point: an offset is decoded only where the
    prefix is known, never guessed. See the block comment above for what that
    admits and what it refuses.
    """
    configured = tuple(
        _rfc6052_embedded(ip.packed, prefix.prefixlen)
        for prefix in _configured_nat64_prefixes()
        if ip in prefix
    )
    if configured:
        # The operator named this prefix, so it outranks every guess below.
        return configured
    mapped = ip.ipv4_mapped
    if mapped is not None:
        return (mapped,)
    packed = ip.packed
    if packed[:12] == b"\x00" * 12:
        return (ipaddress.IPv4Address(packed[12:16]),)
    if ip in _NAT64_WELL_KNOWN:
        return (ipaddress.IPv4Address(packed[12:16]),)
    if ip in _SIX_TO_FOUR:
        return (ipaddress.IPv4Address(packed[2:6]),)
    return ()


def _public_unicast(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return bool(ip.is_global) and not ip.is_multicast


# RFC 4291 section 2.4 gives 2000::/3 to global unicast, and it is still the
# only block IANA allocates from, so an IPv6 answer outside it is special
# purpose or unallocated rather than a host on the internet.
#
# Membership is tested by block because ``ipaddress`` cannot be relied on to
# know which blocks are reachable. On CPython 3.13, ``fec0::1`` reports
# ``is_private`` False and ``is_global`` True: RFC 3879 deprecated site-local
# addressing in 2004, so the module never listed fec0::/10 beside fe80::/10 and
# fc00::/7, yet gear that predates the deprecation still routes it and IANA
# still marks the block Globally Reachable: False. ``5f00::1`` (RFC 9602 SRv6
# SIDs, also Globally Reachable: False) reports the same way. Either would let
# an allowlisted name resolve onto a network the sandbox must not reach, which
# is the whole class this check exists to refuse.
#
# The one block outside 2000::/3 still admitted is the NAT64 Well-Known Prefix,
# and only because it is decoded to the IPv4 address it carries and judged as
# that address before this test is reached.
_GLOBAL_UNICAST = ipaddress.IPv6Network("2000::/3")


def public_address(address: str) -> bool:
    """True for a global unicast address; every IPv4-in-IPv6 form is judged as IPv4."""
    try:
        ip = ipaddress.ip_address(address)
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv6Address):
        candidates = _embedded_ipv4_candidates(ip)
        if candidates:
            return all(_public_unicast(candidate) for candidate in candidates)
        if ip in _NAT64_RESERVED:
            # Translation space that is not the Well-Known Prefix and that no
            # operator claimed. Which /96 inside it carries the IPv4 address is
            # unknowable from the bytes, and RFC 8215 permits a private one.
            return False
        if ip not in _GLOBAL_UNICAST:
            return False
    return _public_unicast(ip)


def _reassemble_handshake(data: bytes) -> tuple[str, bytes]:
    """Join the handshake bytes of consecutive TLS records into one message.

    A ClientHello may be split over several records, and each record carries its
    own five byte header, so the handshake message has to be reassembled before
    it can be parsed. Returns ``(status, message)`` with status ``complete`` (a
    whole handshake message, trimmed to its declared length), ``incomplete``
    (the records so far do not finish it) or ``malformed`` (these bytes are TLS
    records but they will never yield a ClientHello).
    """
    handshake = bytearray()
    position = 0
    while True:
        if len(data) - position < 5:
            return ("incomplete", bytes(handshake))
        if data[position] != 0x16:
            # Another content type before the ClientHello finished: this is a
            # TLS stream, but not one whose first message names a host.
            return ("malformed", bytes(handshake))
        record_length = int.from_bytes(data[position + 3 : position + 5], "big")
        if record_length == 0 or record_length > _MAX_TLS_RECORD_BYTES:
            return ("malformed", bytes(handshake))
        end = position + 5 + record_length
        if end > len(data):
            return ("incomplete", bytes(handshake))
        handshake += data[position + 5 : end]
        position = end
        if len(handshake) < 4:
            continue
        if handshake[0] != 0x01:  # not a ClientHello
            return ("malformed", b"")
        handshake_length = int.from_bytes(handshake[1:4], "big")
        if handshake_length > MAX_CLIENT_HELLO_BYTES:
            return ("malformed", b"")
        if len(handshake) >= 4 + handshake_length:
            return ("complete", bytes(handshake[: 4 + handshake_length]))


def _client_hello_sni(data: bytes) -> tuple[str, str]:
    """Read the SNI out of a TLS ClientHello, however its records are cut up.

    Returns ``(status, name)`` where status is one of ``incomplete`` (more bytes
    may decide it), ``not-tls`` (these bytes are not a TLS handshake at all),
    ``malformed`` (TLS records that will never yield a readable ClientHello),
    ``absent`` (a ClientHello with no server name) or ``found``.

    The parse is deliberately total, but a surprise inside a TLS stream is
    ``malformed`` and not ``not-tls``: reporting a split or corrupt handshake as
    "not TLS" let a client reach an allowlisted address with any server name at
    all simply by writing its ClientHello in two records.
    """
    if not data:
        return ("incomplete", "")
    if data[0] != 0x16:  # not a TLS handshake record
        return ("not-tls", "")
    status, message = _reassemble_handshake(data)
    if status != "complete":
        return (status, "")
    hello = message[4:]
    try:
        pos = 2 + 32  # legacy_version and random
        if len(hello) < pos + 1:
            return ("malformed", "")
        pos += 1 + hello[pos]  # legacy_session_id
        if len(hello) < pos + 2:
            return ("malformed", "")
        pos += 2 + int.from_bytes(hello[pos : pos + 2], "big")  # cipher_suites
        if len(hello) < pos + 1:
            return ("malformed", "")
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
        return ("malformed", "")


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
        self._non_tls = 0
        self._silent = 0

    def record_allowed(self, host: str) -> None:
        with self._lock:
            if host not in self._allowed and len(self._allowed) >= MAX_AUDITED_HOSTS:
                self._overflow += 1
                return
            self._allowed[host] = self._allowed.get(host, 0) + 1

    def record_sni_absent(self) -> None:
        """A tunnel refused because its ClientHello carried no server name.

        The counter is separate from the denial the caller also records, because
        an allowlisted host answering with no SNI is worth telling apart from
        one answering with the wrong SNI when a client turns out to need it.
        """
        with self._lock:
            self._sni_absent += 1

    def record_non_tls(self) -> None:
        """A tunnel refused because its first bytes were not a TLS handshake."""
        with self._lock:
            self._non_tls += 1

    def record_silent_close(self) -> None:
        """A tunnel the client closed before sending anything.

        Not a refusal: no byte was forwarded and none was refused, so this never
        reaches the trailer. It is counted because a launch full of these is a
        client failing before it speaks, which is otherwise invisible.
        """
        with self._lock:
            self._silent += 1

    def record_denied(
        self,
        host: str,
        reason: str,
        kind: str = POLICY_REFUSAL,
    ) -> None:
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
            non_tls = self._non_tls
            silent = self._silent
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
            "non_tls": non_tls,
            "silent_close": silent,
        }

    def denied_entries(self) -> list[tuple[str, int, str, str]]:
        """One entry per (host, reason, kind), so no reason is overwritten."""
        with self._lock:
            return [
                (host, count, reason, kind) for (host, reason, kind), count in self._denied.items()
            ]

    def denied_hosts(self) -> list[tuple[str, int, str]]:
        return [(host, count, reason) for host, count, reason, _ in self.denied_entries()]


class _Denied(Exception):
    def __init__(
        self,
        status: int,
        reason: str,
        host: str = "",
        kind: str | None = None,
    ) -> None:
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


def _close_quietly(sock: socket.socket) -> None:
    """Close a socket whose failure to close is not something a caller can act on."""
    try:
        sock.close()
    except OSError:
        pass


# ---------------------------------------------------------------------------
# What the server name check decides, and what it cannot.
#
# The CONNECT line names a host and the allowlist judges that name, but the
# name is the client's claim about where it is going. Everything after the 200
# is opaque, so the first bytes of the tunnel are the last chance to check that
# claim against something the far end will also act on.
#
# REFUSED, on purpose, and each of these used to be forwarded:
#
# * A ClientHello whose SNI is some other host. The tunnel lands on whatever
#   front end the allowlisted name resolves to, and asks it for a different
#   site.
# * A ClientHello with no SNI at all. Refusing this costs nothing a real client
#   needs: RFC 6066 section 3 says "Literal IPv4 and IPv6 addresses are not
#   permitted in HostName" and RECOMMENDS the extension whenever a client
#   locates a server by name, RFC 8446 section 9.2 makes it mandatory to
#   implement, and RFC 9113 section 9.2 turns that into "clients MUST send the
#   server_name TLS extension" when the server is identified by a domain name,
#   which is every HTTP/2 request an ML tool makes to an allowlisted host. This
#   proxy already refuses IP literals in the CONNECT authority, so the one case
#   where a client legitimately has no name to send cannot arrive here at all.
# * A stream whose first byte is not a TLS handshake record. This was the
#   concrete hole: an allowlisted host that also answers SSH on 443 turned the
#   proxy into an arbitrary TCP tunnel out of a sandbox whose UI promises HTTPS
#   to a fixed list of hosts.
# * A TLS stream that never yields a readable ClientHello inside the byte cap
#   and the header deadline, including one that sends nothing and waits the
#   deadline out. Failing open at the deadline would have made the refusals
#   above a formality: wait, then speak anything.
#
# ADMITTED, on purpose, and this is the hole:
#
# * Domain fronting through the inner HTTP ``Host`` header. The outer name is
#   allowlisted, the SNI agrees with it, and the request inside the TLS session
#   asks that front end for a host that is not on the list. Nothing outside the
#   encryption distinguishes the two, so catching it means terminating TLS in
#   this process, which is a larger loss than the one it prevents. Named here
#   rather than implied, and named again in the module docstring.
# * A client that encrypts its ClientHello (RFC 9849 ECH) is not admitted but
#   is worth naming beside it: the outer SNI is the ECH provider's public name,
#   which will not equal the CONNECT host, so such a tunnel is refused. That is
#   the safe direction, and nothing the sandbox runs does it today (OpenSSL 4.0
#   carries the code, curl needs an explicit --ech, CPython's ssl exposes no
#   API for it), but a client that turns it on will be refused rather than
#   silently downgraded.
# ---------------------------------------------------------------------------


class AllowlistProxy:
    """One proxy per launch.

    ``close()`` stops the listener, ends every tunnel and waits under a bound for
    the workers holding them, so the launch's cleanup does not return while a
    thread it started is still moving bytes.
    """

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
        max_refusal_workers: int = MAX_REFUSAL_WORKERS,
    ) -> None:
        self.allowlist = allowlist
        self.credential = credential or ProxyCredential.mint()
        self.audit = NetworkAudit()
        # Module-level defaults are read at construction so a test can point
        # the proxy at a loopback upstream without reaching into the sandbox
        # backend that builds it.
        self._resolver = resolver or DEFAULT_RESOLVER
        self._allowed_ports = (
            frozenset(allowed_ports) if allowed_ports is not None else ALLOWED_PORTS
        )
        self._require_public = (
            REQUIRE_PUBLIC_ADDRESSES if require_public is None else require_public
        )
        self._connect_timeout = connect_timeout
        self._idle_timeout = idle_timeout
        # A wall-clock budget for the whole request head, not per recv: a client
        # dribbling one byte every few seconds otherwise held a tunnel slot for
        # hours and could starve the launch of every slot.
        self._header_timeout = header_timeout
        self._slots = threading.BoundedSemaphore(max_tunnels)
        # Refusals get their own small pool. They are not tunnels, so they must
        # not take a tunnel slot, and they are not free either: each one can sit
        # in ``sendall`` until the header timeout.
        self._refusal_slots = threading.BoundedSemaphore(max(1, max_refusal_workers))
        self._listener: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._closed = threading.Event()
        # Guards both sets. Every socket a worker holds and every worker that
        # holds one has to be reachable from ``close``, and the two are added
        # and dropped together, so one lock covers them.
        self._tunnels: set[socket.socket] = set()
        self._workers: set[threading.Thread] = set()
        self._tunnel_lock = threading.Lock()

    # -- lifecycle -----------------------------------------------------------

    @property
    def port(self) -> int:
        if self._listener is None:
            raise RuntimeError("the proxy is not listening")
        return int(self._listener.getsockname()[1])

    def listen_loopback(self) -> int:
        """Bind an ephemeral port on the host's loopback (macOS path).

        The listener is made here and handed to ``serve_listener``, so from the
        caller's side this is the same contract as the Linux path: the proxy
        owns the socket and ``close`` is the only thing that closes it.
        """
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
        """Take ownership of a bound listener and accept on it.

        The ownership contract, which is the same for both backends and holds
        whether this returns or raises: from the moment of the call the listener
        belongs to the proxy, and ``close`` is the only thing that closes it. A
        hand-over this refuses (a second listener, or a proxy already closed) or
        cannot complete closes the listener here, so no caller is ever left
        holding a socket it does not know whether to close.

        The two callers differ only in where the socket comes from. The Linux
        backend has no route out of the sandbox's network namespace, so the
        sandboxed wrapper binds the listener inside it and passes the descriptor
        back through SCM_RIGHTS; that descriptor is the only copy the host has,
        and this method is handed it directly. The macOS backend has no network
        namespace, so ``listen_loopback`` binds the host's own loopback and
        hands that socket to this same method. Neither caller closes what it
        passed, and neither has to unwind differently on failure.
        """
        refusal: str | None = None
        if self._listener is not None:
            refusal = "the proxy already serves a listener"
        elif self._closed.is_set():
            refusal = "the proxy is closed"
        if refusal is not None:
            _close_quietly(listener)
            raise RuntimeError(refusal)
        try:
            listener.setblocking(True)
            self._listener = listener
            thread = threading.Thread(
                target = self._accept_loop, name = "studio-tool-network-proxy", daemon = True
            )
            thread.start()
        except BaseException:
            # A listener nobody accepts on is a descriptor nobody will ever
            # close, and on Linux it is the sandbox's only way out.
            self._listener = None
            _close_quietly(listener)
            raise
        self._thread = thread

    def close(self) -> None:
        """Stop the listener, end every tunnel, and wait for the workers.

        Called from the cleanup path of every tool call, so it waits under a
        bound rather than forever: sockets are shut down first, which wakes any
        worker blocked in ``select`` or ``recv``, then the workers are joined,
        and only then are the sockets closed. Joining before the close is what
        makes it deterministic; closing first would free descriptors a worker
        still holds and could hand it a number that has been reused.
        """
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
            workers = list(self._workers)
        for sock in pending:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        current = threading.current_thread()
        thread = self._thread
        if thread is not None and thread is not current:
            thread.join(timeout = CLOSE_JOIN_SECONDS)
        # One budget for every worker, not one each: a launch at the tunnel cap
        # must not turn a two second cleanup into a minute of them.
        deadline = time.monotonic() + CLOSE_JOIN_SECONDS
        alive = 0
        for worker in workers:
            if worker is current:
                continue
            worker.join(timeout = max(0.0, deadline - time.monotonic()))
            if worker.is_alive():
                alive += 1
        if alive:
            logger.warning(
                "Tool network proxy closed with %d worker thread(s) still running; "
                "their sockets are closed here so they cannot forward anything further",
                alive,
            )
        for sock in pending:
            _close_quietly(sock)

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
                self._dispatch_refusal(client, 503, "too many concurrent tunnels in this backend")
                continue
            try:
                self._spawn_worker(self._serve_client, (client,), "studio-tool-network-tunnel")
            except BaseException as exc:
                # RuntimeError("can't start new thread") would otherwise burn the
                # slot for the life of the launch and leak the accepted socket.
                logger.warning("Tool network proxy could not start a worker: %s", exc)
                self._release_slots()
                _close_quietly(client)

    def _dispatch_refusal(self, client: socket.socket, status: int, reason: str) -> None:
        """Answer a refusal off the accept thread, with a timeout on the write.

        ``accept`` hands back a blocking socket with no timeout, so a client
        advertising a zero receive window would otherwise wedge the single
        accept thread inside ``sendall`` and stop the proxy serving anyone.

        The workers are capped. Past the cap the socket is closed with no reply,
        which is the only answer left that neither blocks the accept loop nor
        adds a thread: a flood arriving after the tunnel cap must cost the
        backend a bounded number of threads, not one per connection.
        """
        try:
            client.settimeout(self._header_timeout)
        except OSError:
            pass
        if not self._refusal_slots.acquire(blocking = False):
            logger.debug("Tool network proxy closed a connection without a refusal reply")
            _close_quietly(client)
            return
        try:
            self._spawn_worker(
                self._refusal_worker,
                (client, status, reason),
                "studio-tool-network-refuse",
            )
        except BaseException as exc:
            # Never answer inline: that is the accept thread, and this client may
            # never read.
            logger.warning("Tool network proxy could not start a refusal worker: %s", exc)
            self._release_refusal_slot()
            _close_quietly(client)

    def _refusal_worker(self, client: socket.socket, status: int, reason: str) -> None:
        try:
            if self._track(client):
                # Tracked so ``close`` can shut this socket down: a client that
                # never reads its refusal would otherwise sit in ``sendall``
                # until the header timeout, well past the end of the tool call.
                self._refuse(client, status, reason)
            else:
                _close_quietly(client)
        finally:
            self._untrack(client)
            self._release_refusal_slot()
            self._retire_worker()

    def _release_refusal_slot(self) -> None:
        try:
            self._refusal_slots.release()
        except ValueError:  # pragma: no cover - a release without an acquire
            logger.warning("Tool network proxy released a refusal slot twice")

    def _release_slots(self) -> None:
        for semaphore in (self._slots, self.global_slots):
            try:
                semaphore.release()
            except ValueError:  # pragma: no cover - a release without an acquire
                logger.warning("Tool network proxy released a tunnel slot twice")

    def _track(self, sock: socket.socket) -> bool:
        """Put a socket under ``close``, or answer False when ``close`` already ran.

        ``close`` sets the flag before it takes this lock, so a socket is either
        in the set it snapshots or gets False here and is closed by its caller.
        Neither order leaves a socket that nothing will close.
        """
        with self._tunnel_lock:
            if self._closed.is_set():
                return False
            self._tunnels.add(sock)
            return True

    def _untrack(self, sock: socket.socket) -> None:
        with self._tunnel_lock:
            self._tunnels.discard(sock)

    def _spawn_worker(self, target: Callable[..., None], args: tuple, name: str) -> None:
        """Start a worker ``close`` can wait for.

        Registered before ``start`` and dropped by the worker itself, so there
        is no window where a thread is running and ``close`` cannot see it. A
        thread that fails to start is dropped here instead.
        """
        thread = threading.Thread(target = target, args = args, name = name, daemon = True)
        with self._tunnel_lock:
            self._workers.add(thread)
        try:
            thread.start()
        except BaseException:
            with self._tunnel_lock:
                self._workers.discard(thread)
            raise

    def _retire_worker(self) -> None:
        with self._tunnel_lock:
            self._workers.discard(threading.current_thread())

    def _serve_client(self, client: socket.socket) -> None:
        upstream: socket.socket | None = None
        try:
            if not self._track(client):
                # ``close`` already ran, so this connection is nobody's to
                # serve; the finally below closes it.
                return
            host, port, pipelined = self._authorize(client)
            upstream = self._connect_upstream(host, port)
            if not self._track(upstream):
                return
            client.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
            opening = self._check_server_name(client, host, pipelined)
            if opening is None:
                # The client hung up without sending anything, so no byte was
                # forwarded and nothing was refused. Nothing to report either.
                return
            # Recorded only once the tunnel is about to carry traffic: a tunnel
            # the server name check refused is a denial, and counting it as an
            # allowed host too would tell the model both at once.
            self.audit.record_allowed(host)
            if opening:
                # A client that wrote its ClientHello in the same segment as the
                # CONNECT head would otherwise hang until the idle timeout.
                upstream.sendall(opening)
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
            logger.warning("Tool network proxy failed to serve a request: %s", exc, exc_info = True)
            self._refuse(client, 400, "the proxy could not process this request")
        finally:
            for sock in (client, upstream):
                if sock is None:
                    continue
                self._untrack(sock)
                _close_quietly(sock)
            self._release_slots()
            self._retire_worker()

    def _check_server_name(
        self, client: socket.socket, host: str, pipelined: bytes
    ) -> bytes | None:
        """The bytes to forward, or None when the client left without sending any.

        Every other outcome raises ``_TunnelAborted`` before a byte reaches the
        upstream. What is refused and what is knowingly admitted is argued out
        in the block comment above this class; the short version is that a
        tunnel must open with a TLS ClientHello naming the CONNECT host, and a
        missing name, a stream that is not TLS, and a stream that never
        completes a hello are all refusals rather than allowances.

        A stream that begins with a TLS handshake record must produce a complete
        ClientHello within the byte cap and the header deadline, whatever record
        boundaries it uses. Treating a handshake split across records as "not
        TLS" was an earlier bypass of the name check itself.
        """
        buffer = pipelined
        deadline = time.monotonic() + self._header_timeout
        # Told apart because the two mean different things once the buffer is
        # empty: a peer that hung up refused nothing, while a deadline that ran
        # out is a client holding an unchecked tunnel open.
        hung_up = False
        status, name = _client_hello_sni(buffer)
        # The wire allowance, not the body cap: a hello inside the body cap still
        # spends five bytes on every record header it arrives in, and measuring
        # those against the body cap refused it before it could be parsed.
        while status == "incomplete" and len(buffer) < MAX_CLIENT_HELLO_WIRE_BYTES:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                client.settimeout(max(0.1, remaining))
                chunk = client.recv(4096)
            except socket.timeout:
                break
            except OSError:
                # The socket is gone, which is also how ``close`` ends a tunnel.
                hung_up = True
                break
            if not chunk:
                hung_up = True
                break
            buffer += chunk
            status, name = _client_hello_sni(buffer)
        if status == "found":
            if name.strip().rstrip(".").lower() == host:
                return buffer
            self._abort_tls_tunnel(client, host, "SNI does not match the CONNECT host")
        if status == "absent":
            self.audit.record_sni_absent()
            self._abort_tls_tunnel(
                client,
                host,
                "the TLS ClientHello named no server to check against the CONNECT host",
            )
        if status == "not-tls":
            self.audit.record_non_tls()
            # No alert here. These bytes are not TLS, so a TLS record would be
            # noise in whatever protocol this is, and the reason reaches the
            # model through the trailer either way.
            raise _TunnelAborted(
                host, "the tunnel did not open with TLS, and only HTTPS is proxied"
            )
        if not buffer:
            if hung_up:
                # Opened a tunnel, said nothing, closed it. Nothing was checked
                # because nothing was sent, and nothing was forwarded, so there
                # is no refusal to report; returning None ends the tunnel here
                # rather than splicing a stream that was never checked.
                self.audit.record_silent_close()
                return None
            # Silent until the deadline. Forwarding from here is the bypass that
            # would make every refusal above optional: wait, then speak anything.
            raise _TunnelAborted(host, "no TLS ClientHello arrived before the deadline")
        # "incomplete" here means the cap, the deadline or an EOF ended the wait,
        # and "malformed" that the records will never yield a hello. Either way a
        # TLS stream did not name its host, which is a refusal and not an allow.
        self._abort_tls_tunnel(client, host, "the TLS ClientHello did not name the CONNECT host")

    def _abort_tls_tunnel(self, client: socket.socket, host: str, reason: str) -> NoReturn:
        """End a TLS client's tunnel with an alert it can report, then abort.

        The 200 is already on the wire, so no HTTP status can carry this. A bare
        close reaches a TLS client as "EOF occurred in violation of protocol" or
        a reset, which is the mystery a user is then left to guess at; the alert
        reaches it as "tlsv1 alert unrecognized name", which names the thing
        this proxy actually enforces. Sent only where the stream already looked
        like TLS, so nothing else is ever handed a TLS record it did not ask
        for, and only towards the sandbox, never towards the upstream.
        """
        try:
            client.sendall(TLS_UNRECOGNIZED_NAME_ALERT)
        except OSError:
            pass
        raise _TunnelAborted(host, reason)

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
            _close_quietly(client)

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
            try:
                readable, _, errored = select.select(
                    list(open_sides), [], list(open_sides), self._idle_timeout
                )
            except (OSError, ValueError):
                # ``close`` shut these sockets down underneath the splice, which
                # ``select`` reports as EBADF or, once Python has blanked the
                # descriptor, as a negative file descriptor.
                return
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
                except (OSError, ValueError):
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
