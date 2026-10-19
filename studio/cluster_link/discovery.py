# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""LAN announcements: a worker says where its agent listens, the master learns a paired worker's
current address. An announcement carries no secret and grants nothing: the master only uses it to
dial, and the mutual TLS handshake decides whether the far end is the paired worker."""

from __future__ import annotations

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
        try:
            s4 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
            s4.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 1)
            socks.append((s4, (DISCOVERY_GROUP_V4, DISCOVERY_PORT)))
        except OSError as exc:
            self._warn(exc)
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
        sent = False
        for sock, dest in self._sockets():
            try:
                sock.sendto(self.payload, dest)
                sent = True
            except OSError as exc:
                if dest[0] == DISCOVERY_GROUP_V4:
                    self._warn(exc)
            finally:
                sock.close()
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
        # (node_id, fp) -> keep? Serving passes its paired workers, so a flood of made-up node ids
        # cannot fill the table and push a real worker out.
        self._accept = accept
        self._lock = threading.Lock()
        # node_id -> host -> Sighting. Several hosts per node: anyone on the LAN can announce a paired
        # node's public fingerprint from another address, and last-writer-wins let that hide the real one.
        self._seen: Dict[str, Dict[str, Sighting]] = {}
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._sock: Optional[socket.socket] = None
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
            mreq = struct.pack(
                "4s4s", socket.inet_aton(DISCOVERY_GROUP_V4), socket.inet_aton("0.0.0.0")
            )
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
            sock.settimeout(0.5)
            return sock
        except OSError as exc:
            self.error = str(exc)
            logger.warning(
                "cluster discovery: cannot listen (%s); using the last known addresses", exc
            )
            return None

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
                # Not listening yet (no interface at start) or after an error: keep trying, or the
                # process stays deaf to workers that move until it restarts.
                self._sock = sock = self._open()
                if sock is None:
                    self._stop.wait(REOPEN_S)
                    continue
                self.error = ""
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
