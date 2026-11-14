# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""What each paired worker is doing right now, shared by `unsloth cluster list` and Studio."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

DISCOVERY_SETTLE_S = 6.5

# macOS answers "No route to host" for every outgoing LAN connection until the app has Local Network access.
LOCAL_NETWORK_HINT = (
    "on macOS, allow Local Network access for the app running Unsloth in "
    "System Settings > Privacy & Security > Local Network"
)


@dataclass
class WorkerStatus:
    name: str
    node_id: str
    where: str
    state: str  # ready | busy | rpc down | unreachable | relayed
    via: Optional[str] = None  # local network | direct internet connection
    reason: Optional[str] = None
    devices: List[str] = field(default_factory = list)
    discovered: bool = False


def addresses(peer: Any) -> List[Tuple[str, int]]:
    default = int(getattr(peer, "agent_port", 0) or 0)
    out: List[Tuple[str, int]] = []
    for addr in getattr(peer, "addrs", None) or []:
        host, sep, port = str(addr).rpartition(":")
        if sep and host and port.isdigit() and ":" not in host:
            out.append((host, int(port)))
        elif addr and default:
            out.append((str(addr), default))
    return out


def unreachable_reason(st: Any) -> str:
    if not isinstance(st, dict):
        return "not found on the LAN"
    raw = str(st.get("error") or "")
    low = raw.lower()
    if "errno 111" in low or "connection refused" in low or "connect call failed" in low:
        return "not running: start `unsloth worker` there"
    if "timeout" in low or "timed out" in low:
        return "timed out"
    if "certificate" in low or "ssl" in low or "tls" in low:
        return f"TLS refused: {raw}; re-pair it if it was reset"
    if "no route" in low or "unreachable" in low or "errno 113" in low:
        if sys.platform == "darwin":
            return f"no route to it on the network; {LOCAL_NETWORK_HINT}"
        return "no route to it on the network"
    return raw or "no answer"


def answered(st: Any) -> bool:
    return (
        isinstance(st, dict)
        and st.get("reachable", True) is not False
        and st.get("type") != "error"
    )


def _through_tailnet(peer: Any, ask: Callable) -> Optional[Tuple[str, Any]]:
    """("direct" | "relayed", status) for a worker answering through its tailcat address, else None."""
    from studio.cluster_link import AGENT_PORT_DEFAULT, remote

    if not getattr(peer, "tc_addr", "") or not remote.ensure_helper() or not remote.enabled():
        return None
    local = remote.forward(peer.node_id, peer.tc_addr, int(peer.agent_port or AGENT_PORT_DEFAULT))
    if local is None:
        return None
    st = ask(peer, "127.0.0.1", local)
    if not answered(st):
        return None
    info = remote.path(peer.node_id)
    return ("direct" if getattr(info, "direct", False) else "relayed"), st


def _drop_removed(live: set) -> None:
    """``workers`` is every live worker: a tunnel forward to anyone else belongs to a removed one."""
    from studio.cluster_link import remote
    for node_id in remote.forwarded():
        if node_id not in live:
            remote.drop(node_id)


def remote_relay_allowed() -> bool:
    from studio.cluster_link import remote
    return remote.relay_allowed()


def check_workers(
    master: Any,
    registry: Any,
    workers: List[Any],
    *,
    authorize: Callable[[Any], bool],
    settle_s: float = DISCOVERY_SETTLE_S,
    on_search: Optional[Callable[[int], None]] = None,
) -> List[WorkerStatus]:
    """Ask every worker (recorded address, then LAN discovery, then tailcat) and report each."""
    from studio.cluster_link.relay import RelayManager

    relays = RelayManager(master, authorize)

    def ask(peer: Any, host: str, port: int) -> Any:
        try:
            relays.ensure(peer.node_id, host, port)
            return relays.status(peer.node_id, timeout = 3.0)
        except Exception as exc:
            return {"type": "error", "error": type(exc).__name__, "reachable": False}

    found: Dict[str, Tuple[str, Any, bool]] = {}
    via: Dict[str, str] = {}
    try:
        for peer in workers:
            for i, (addr, aport) in enumerate(addresses(peer)):
                st = ask(peer, addr, aport)
                if i == 0 or answered(st):
                    found[peer.node_id] = (f"{addr}:{aport}", st, False)
                if answered(st):
                    if i:
                        with registry.transaction():
                            registry.record_address(peer.node_id, addr, aport)
                    break
        missing = [p for p in workers if not answered(found.get(p.node_id, (None, None))[1])]
        if missing:
            from studio.cluster_link.discovery import Listener

            fps = {p.node_id: p.fingerprint for p in workers}
            if on_search is not None:
                on_search(len(missing))
            listener = Listener(accept = lambda node_id, fp: fps.get(node_id) == fp).start()
            try:
                time.sleep(settle_s)
                for peer in missing:
                    for seen in listener.candidates(peer.node_id):
                        if seen.fp != peer.fingerprint:
                            continue
                        st = ask(peer, seen.host, seen.agent_port)
                        if answered(st):
                            found[peer.node_id] = (f"{seen.host}:{seen.agent_port}", st, True)
                            with registry.transaction():
                                registry.record_address(peer.node_id, seen.host, seen.agent_port)
                            break
            finally:
                listener.stop()
        for peer in [p for p in workers if not answered(found.get(p.node_id, (None, None))[1])]:
            path = _through_tailnet(peer, ask)
            if path is not None:
                found[peer.node_id] = ("through tailcat", path[1], False)
                via[peer.node_id] = path[0]
    finally:
        relays.stop()
        _drop_removed({p.node_id for p in workers})
    out: List[WorkerStatus] = []
    for peer in workers:
        where, st, discovered = found.get(peer.node_id, ("address unknown", None, False))
        row = WorkerStatus(name = peer.name, node_id = peer.node_id, where = where, state = "")
        if not answered(st):
            row.state, row.reason = "unreachable", unreachable_reason(st)
        elif via.get(peer.node_id) == "relayed" and not remote_relay_allowed():
            row.state = "relayed"
        else:
            row.state = "busy" if st.get("busy") else ("ready" if st.get("rpc_up") else "rpc down")
            if row.state == "ready":
                row.via = {
                    "direct": "direct internet connection",
                    "relayed": "through a relay: slow",
                }.get(via.get(peer.node_id), "local network")
            row.devices = [str(d) for d in st.get("devices") or []]
            row.discovered = discovered
        out.append(row)
    return out
