# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""LAN announcements of worker agents; mutual TLS, not the announcement, decides trust."""

from __future__ import annotations

import ipaddress
import json
import re
import logging
import random
import socket
import struct
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from . import DISCOVERY_GROUP_V4, DISCOVERY_GROUP_V6, DISCOVERY_PORT, PROTOCOL_VERSION

logger = logging.getLogger(__name__)

MAX_DATAGRAM = 512
MAX_SIGHTINGS = 256
MAX_HOSTS_PER_NODE = 4


def lan_ipv4_addresses() -> List[str]:
    """Every up, non-loopback, private IPv4 address of this host, or [] when they cannot be listed."""
    try:
        import psutil
        addrs = psutil.net_if_addrs()
        stats = psutil.net_if_stats()
    except Exception:
        return []
    found: List[str] = []
    for name, entries in addrs.items():
        st = stats.get(name)
        if st is not None and not st.isup:
            continue
        for entry in entries:
            if entry.family != socket.AF_INET:
                continue
            try:
                addr = ipaddress.IPv4Address(entry.address)
            except ValueError:
                continue
            if addr.is_loopback or not (addr.is_private or addr.is_link_local):
                continue
            if entry.address not in found:
                found.append(entry.address)
    return found


@dataclass(frozen = True)
class Sighting:
    host: str
    agent_port: int
    fp: str
    seen_at: float


def encode_announcement(node_id: str, agent_port: int, fingerprint: str) -> bytes:
    body = json.dumps(
        {"v": PROTOCOL_VERSION, "node_id": node_id, "agent_port": agent_port, "fp": fingerprint},
        separators = (",", ":"),
    ).encode("utf-8")
    if len(body) > MAX_DATAGRAM:
        raise ValueError("announcement too large")
    return body


_NODE_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")


def decode_announcement(data: bytes) -> Optional[Dict[str, object]]:
    """The announcement's fields, or None for anything malformed."""
    if len(data) > MAX_DATAGRAM:
        return None
    try:
        obj = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None
    if not isinstance(obj, dict) or obj.get("v") != PROTOCOL_VERSION:
        return None
    node_id, port, fp = obj.get("node_id"), obj.get("agent_port"), obj.get("fp")
    # Anyone on the LAN can send this: a node id reaches logs and terminals, so no control bytes.
    if not isinstance(node_id, str) or not _NODE_ID.fullmatch(node_id):
        return None
    if isinstance(port, bool) or not isinstance(port, int) or not 0 < port < 65536:
        return None
    if not isinstance(fp, str) or len(fp) != 64:
        return None
    try:
        int(fp, 16)
    except ValueError:
        return None
    return {"node_id": node_id, "agent_port": port, "fp": fp.lower()}


class Announcer:
    """Multicasts this worker's agent address every ``interval`` seconds from a daemon thread."""

    def __init__(
        self,
        node_id: str,
        agent_port: int,
        fingerprint: str,
        *,
        interval: float = 5.0,
    ) -> None:
        self.payload = encode_announcement(node_id, agent_port, fingerprint)
        self.interval = interval
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._warned = False

    def _sockets(self):
        socks = []
        for iface in lan_ipv4_addresses() or [None]:
            try:
                s4 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
            except OSError as exc:
                self._warn(exc)
                continue
            try:
                s4.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 1)
                if iface is not None:
                    s4.setsockopt(
                        socket.IPPROTO_IP, socket.IP_MULTICAST_IF, socket.inet_aton(iface)
                    )
            except OSError as exc:
                s4.close()
                self._warn(exc)
                continue
            socks.append((s4, (DISCOVERY_GROUP_V4, DISCOVERY_PORT)))
        if socket.has_ipv6:
            try:
                s6 = socket.socket(socket.AF_INET6, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
                s6.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_MULTICAST_HOPS, 1)
                socks.append((s6, (DISCOVERY_GROUP_V6, DISCOVERY_PORT)))
            except OSError:
                pass
        return socks

    def _warn(self, exc: BaseException) -> None:
        if not self._warned:
            self._warned = True
            logger.warning("cluster discovery: cannot announce (%s); pair with --addr instead", exc)

    def send_once(self) -> bool:
        sent = sent_v4 = False
        failure: Optional[OSError] = None
        for sock, dest in self._sockets():
            try:
                sock.sendto(self.payload, dest)
                sent = True
                sent_v4 = sent_v4 or dest[0] == DISCOVERY_GROUP_V4
            except OSError as exc:
                if dest[0] == DISCOVERY_GROUP_V4:
                    failure = exc
            finally:
                sock.close()
        if failure is not None and not sent_v4:
            self._warn(failure)
        return sent

    def _run(self) -> None:
        while not self._stop.is_set():
            self.send_once()
            self._stop.wait(self.interval * random.uniform(0.8, 1.2))

    def start(self) -> "Announcer":
        if self._thread is None:
            self._thread = threading.Thread(target = self._run, name = "cluster-announcer", daemon = True)
            self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout = 2.0)
            self._thread = None


REOPEN_S = 2.0
REJOIN_S = 30.0


class Listener:
    """Collects announcements on the IPv4 group from a daemon thread; ``latest()`` is thread-safe."""

    def __init__(
        self,
        *,
        interval: float = 5.0,
        port: int = DISCOVERY_PORT,
        accept: Optional[Callable[[str, str], bool]] = None,
    ) -> None:
        self.expire_after = 3 * interval
        self.port = port
        # (node_id, fp) -> keep? Limited to paired workers, so made-up node ids cannot flood the table.
        self._accept = accept
        self._lock = threading.Lock()
        # node_id -> host -> Sighting: a spoofed announcement from another host must not hide the real one.
        self._seen: Dict[str, Dict[str, Sighting]] = {}
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._sock: Optional[socket.socket] = None
        self._ifaces: List[str] = []
        self._joined: List[str] = []
        self._joined_at = 0.0
        self.error: str = ""

    def _open(self) -> Optional[socket.socket]:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if hasattr(socket, "SO_REUSEPORT"):
                try:
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
                except OSError:
                    pass
            sock.bind(("", self.port))
            ifaces = lan_ipv4_addresses()
            joined = []
            for iface in ifaces:
                try:
                    sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, self._mreq(iface))
                    joined.append(iface)
                except OSError:
                    pass
            if not joined:
                sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, self._mreq("0.0.0.0"))
            self._ifaces, self._joined, self._joined_at = ifaces, joined, time.monotonic()
            sock.settimeout(0.5)
            return sock
        except OSError as exc:
            self.error = str(exc)
            logger.warning(
                "cluster discovery: cannot listen (%s); using the last known addresses", exc
            )
            return None

    @staticmethod
    def _mreq(iface: str) -> bytes:
        return struct.pack("4s4s", socket.inet_aton(DISCOVERY_GROUP_V4), socket.inet_aton(iface))

    def _interfaces_changed(self, sock: Optional[socket.socket] = None) -> bool:
        if time.monotonic() - self._joined_at < REJOIN_S:
            return False
        self._joined_at = time.monotonic()
        if lan_ipv4_addresses() != self._ifaces:
            return True
        for iface in self._ifaces:
            if iface in self._joined or sock is None:
                continue
            try:
                sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, self._mreq(iface))
                self._joined.append(iface)
            except OSError:
                pass
        return False

    def observe(
        self,
        data: bytes,
        host: str,
        now: Optional[float] = None,
    ) -> bool:
        ann = decode_announcement(data)
        if ann is None:
            return False
        if self._accept is not None:
            try:
                if not self._accept(ann["node_id"], ann["fp"]):
                    return False
            except Exception:
                return False
        now = time.time() if now is None else now
        sighting = Sighting(host = host, agent_port = ann["agent_port"], fp = ann["fp"], seen_at = now)
        with self._lock:
            if ann["node_id"] not in self._seen and len(self._seen) >= MAX_SIGHTINGS:
                self._prune(now)
                if len(self._seen) >= MAX_SIGHTINGS:
                    return False
            hosts = self._seen.setdefault(ann["node_id"], {})
            hosts[host] = sighting
            if len(hosts) > MAX_HOSTS_PER_NODE:
                del hosts[min(hosts, key = lambda h: hosts[h].seen_at)]
        return True

    def _prune(self, now: float) -> None:
        for node in list(self._seen):
            hosts = self._seen[node]
            for host in [h for h, s in hosts.items() if now - s.seen_at > self.expire_after]:
                del hosts[host]
            if not hosts:
                del self._seen[node]

    def latest(self, now: Optional[float] = None) -> Dict[str, Sighting]:
        """The most recent sighting per node."""
        now = time.time() if now is None else now
        with self._lock:
            self._prune(now)
            return {n: max(h.values(), key = lambda s: s.seen_at) for n, h in self._seen.items()}

    def candidates(
        self,
        node_id: str,
        now: Optional[float] = None,
    ) -> List[Sighting]:
        """Every unexpired sighting of ``node_id``, newest first: the caller tries each over mutual TLS."""
        now = time.time() if now is None else now
        with self._lock:
            self._prune(now)
            return sorted(self._seen.get(node_id, {}).values(), key = lambda s: -s.seen_at)

    def _run(self) -> None:
        while not self._stop.is_set():
            sock = self._sock
            if sock is None:
                self._sock = sock = self._open()
                if sock is None:
                    self._stop.wait(REOPEN_S)
                    continue
                self.error = ""
            if self._interfaces_changed(sock):
                self._sock = None
                sock.close()
                continue
            try:
                data, addr = sock.recvfrom(MAX_DATAGRAM + 1)
            except socket.timeout:
                continue
            except OSError:
                self._sock = None
                try:
                    sock.close()
                except OSError:
                    pass
                self._stop.wait(REOPEN_S)
                continue
            self.observe(data, addr[0])

    def start(self) -> "Listener":
        if self._thread is None:
            self._sock = self._open()  # here, so a caller sees `error` at once
            self._thread = threading.Thread(target = self._run, name = "cluster-discovery", daemon = True)
            self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout = 2.0)
            self._thread = None
        if self._sock is not None:
            self._sock.close()
            self._sock = None
