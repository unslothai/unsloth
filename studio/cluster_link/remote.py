# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Reaching paired machines on other networks through tailcat. Every call is a no-op without its helper.

Mutual TLS still runs inside the tunnel: tailcat only provides reachability, never trust."""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from .paths import private_dir, state_dir, write_private

logger = logging.getLogger(__name__)
_LOCK = threading.Lock()

RELAYED = "reachable only through a relay, too slow to serve models"
LOCAL = "local network"
DIRECT = "direct internet connection"


def _tailnet() -> Any:
    if (os.environ.get("UNSLOTH_CLUSTER_TAILNET") or "").strip() == "0":
        return None
    try:
        from . import tailnet
    except ImportError:
        return None
    try:
        return tailnet if tailnet.available() else None
    except Exception:
        return None


def enabled() -> bool:
    return _tailnet() is not None


_HELPER: Dict[str, Any] = {}
_HELPER_LOCK = threading.Lock()


def ensure_helper(note: Optional[Callable[[str], None]] = None) -> bool:
    """Install the tailcat helper once per process if it is missing; False (LAN only) on any failure.

    The only place that may download it: ``tailnet.available()`` and ``helper_path()`` never do."""
    if (os.environ.get("UNSLOTH_CLUSTER_TAILNET") or "").strip() == "0":
        return False
    with _HELPER_LOCK:
        if "ok" not in _HELPER:
            ok, why = False, ""
            try:
                from . import tailnet

                install = getattr(tailnet, "install", None)
                if not tailnet.available() and install is not None:
                    install()
                ok = tailnet.available()
                why = "" if ok else "it is not installed"
            except Exception as exc:
                why = str(exc).strip() or type(exc).__name__
            _HELPER.update(ok = ok, why = why)
        ok, why = _HELPER["ok"], _HELPER["why"]
    if not ok and note is not None:
        note(
            f"Reaching machines on other networks is off here ({why}); using the local network only."
        )
    return ok


def relay_allowed() -> bool:
    return (os.environ.get("UNSLOTH_CLUSTER_ALLOW_RELAY") or "").strip() == "1"


def key_path(kind: str) -> Path:
    """``server`` (a worker's persistent address) or ``client`` (a master's identity)."""
    return state_dir() / "tailnet" / f"{kind}.key"


def own_pub(kind: str) -> Optional[str]:
    """This machine's public tailcat key of ``kind``, made on first use."""
    tn = _tailnet()
    if tn is None:
        return None
    key = key_path(kind)
    pub = key.with_suffix(".pub")
    if key.is_file() and pub.is_file():
        return pub.read_text(encoding = "utf-8").strip() or None
    private_dir(key.parent)
    value = (
        tn.new_server_key(key, fixed_region = True) if kind == "server" else tn.new_client_key(key)
    )
    write_private(pub, value.encode("utf-8"))
    return value


def forget_keys() -> None:
    for kind in ("server", "client"):
        for path in (key_path(kind), key_path(kind).with_suffix(".pub")):
            try:
                path.unlink()
            except OSError:
                pass


def serve(
    ports: Sequence[int], key: Optional[str], allow: Optional[List[str]]
) -> Optional[Tuple[Any, str]]:
    """A started helper serving ``ports`` and its secret address; ``key`` None is an ephemeral one."""
    tn = _tailnet()
    if tn is None:
        return None
    net = tn.Tailnet()
    try:
        net.start()
        address = net.serve(
            [int(p) for p in ports], key_path = key_path(key) if key else None, allow = allow
        )
    except BaseException:
        net.close()
        raise
    return net, address


def dial(address: str, port: int) -> Tuple[Any, int]:
    """An ephemeral client forwarding a local port to ``port`` at ``address`` (the pairing window)."""
    tn = _tailnet()
    if tn is None:
        raise RuntimeError("tailcat is not available")
    net = tn.Tailnet()
    try:
        net.start()
        return net, net.forward(address, int(port))
    except BaseException:
        net.close()
        raise


def worker_address(agent_port: int) -> Optional[Tuple[str, str]]:
    """(address, public key) of this worker's persistent tailcat server, made on first use."""
    pub = own_pub("server")
    if pub is None:
        return None
    served = serve([agent_port], "server", [])
    if served is None:
        return None
    net, address = served
    net.close()
    return address, pub


_CLIENT: Any = None
_FORWARDS: Dict[str, Tuple[str, Set[int]]] = {}  # node_id -> (address, remote ports)


def forward(node_id: str, address: str, port: int) -> Optional[int]:
    """Master side: a local port reaching ``port`` on the paired worker, or None."""
    global _CLIENT
    tn = _tailnet()
    if tn is None or not address:
        return None
    try:
        with _LOCK:
            if _CLIENT is None:
                own_pub("client")
                client = tn.Tailnet()
                client.start()
                _CLIENT = client
            client = _CLIENT
        local = client.forward(address, int(port), key_path = key_path("client"))
    except Exception as exc:
        logger.info("cluster link: no tailcat forward to %s (%s)", node_id, type(exc).__name__)
        return None
    with _LOCK:
        old = _FORWARDS.get(node_id)
        ports = old[1] if old and old[0] == address else set()
        ports.add(int(port))
        _FORWARDS[node_id] = (address, ports)
    return local


def path(node_id: str) -> Any:
    """The tunnel's current path to a forwarded worker (``direct``, ``rtt_ms``, ``via``), or None."""
    with _LOCK:
        client, entry = _CLIENT, _FORWARDS.get(node_id)
    if client is None or entry is None:
        return None
    try:
        return client.probe(entry[0])
    except Exception as exc:
        logger.info("cluster link: tailcat probe of %s failed (%s)", node_id, type(exc).__name__)
        return None


def usable(info: Any) -> bool:
    return info is not None and (bool(getattr(info, "direct", False)) or relay_allowed())


def drop(node_id: str) -> None:
    with _LOCK:
        client, entry = _CLIENT, _FORWARDS.pop(node_id, None)
    if client is not None and entry is not None:
        try:
            client.unforward(entry[0])
        except Exception:
            pass


def forwarded() -> List[str]:
    with _LOCK:
        return list(_FORWARDS)


def close() -> None:
    global _CLIENT
    with _LOCK:
        client, _CLIENT = _CLIENT, None
        _FORWARDS.clear()
    if client is not None:
        try:
            client.close()
        except Exception:
            pass


class WorkerTunnel:
    """A worker's persistent tailcat server for its agent (and native) ports, open to its masters' keys."""

    def __init__(self, ports: Sequence[int], allowed: Callable[[], List[str]]):
        self.ports = [int(p) for p in ports]
        self._allowed = allowed
        self._net: Any = None
        self._allow: Set[str] = set()

    def start(self) -> bool:
        if not enabled() or not key_path("server").is_file():
            return False
        allow = sorted(set(self._allowed()))
        served = serve(self.ports, "server", allow)
        if served is None:
            return False
        self._net, _address = served
        self._allow = set(allow)
        return True

    def sync(self) -> None:
        """Admit newly paired masters and revoke (closing the sessions of) removed ones."""
        if self._net is None:
            return
        want = set(self._allowed())
        for pub in sorted(want - self._allow):
            self._net.allow(pub)
        for pub in sorted(self._allow - want):
            self._net.revoke(pub)
        self._allow = want

    def close(self) -> None:
        net, self._net = self._net, None
        if net is not None:
            try:
                net.close()
            except Exception:
                pass
