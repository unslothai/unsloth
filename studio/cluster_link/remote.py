# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Reaching paired machines on other networks through tailcat. Every call is a no-op without its helper.

Mutual TLS still runs inside the tunnel: tailcat only provides reachability, never trust."""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
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
INSTALL_RETRY_S = 600.0


def ensure_helper(note: Optional[Callable[[str], None]] = None) -> bool:
    """Install the tailcat helper if it is missing; False (LAN only) on any failure.

    Success is kept for the process, a failed install is retried after INSTALL_RETRY_S.
    The only place that may download it: ``tailnet.available()`` and ``helper_path()`` never do."""
    if (os.environ.get("UNSLOTH_CLUSTER_TAILNET") or "").strip() == "0":
        return False
    with _HELPER_LOCK:
        if not _HELPER.get("ok"):
            ok, why = False, ""
            try:
                from . import tailnet

                ok = tailnet.available()
                stale = time.monotonic() - _HELPER.get("at", -INSTALL_RETRY_S) >= INSTALL_RETRY_S
                if not ok and ("ok" not in _HELPER or stale):
                    install = getattr(tailnet, "install", None)
                    if install is not None:
                        _HELPER["at"] = time.monotonic()
                        install()
                    ok = tailnet.available()
                why = "" if ok else _HELPER.get("why") or "it is not installed"
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


def _identity_lock(held: bool) -> Any:
    from contextlib import nullcontext

    from .identity import identity_lock
    return nullcontext() if held else identity_lock()


def own_pub(kind: str, *, held: bool = False) -> Optional[str]:
    """This machine's public tailcat key of ``kind``, made on first use under the identity lock
    (``held``: the caller holds it). A key whose sidecar is missing is kept and its sidecar rewritten."""
    tn = _tailnet()
    if tn is None:
        return None
    key = key_path(kind)
    pub = key.with_suffix(".pub")
    with _identity_lock(held):
        if key.is_file() and pub.is_file():
            return pub.read_text(encoding = "utf-8").strip() or None
        private_dir(key.parent)
        value = (
            tn.new_server_key(key, fixed_region = True)
            if kind == "server"
            else tn.new_client_key(key)
        )
        write_private(pub, value.encode("utf-8"))
    return value


def forget_keys(*, held: bool = False) -> None:
    with _identity_lock(held):
        for kind in ("server", "client"):
            for path in (key_path(kind), key_path(kind).with_suffix(".pub")):
                try:
                    path.unlink()
                except OSError:
                    pass


async def owned(call: Callable[..., Any], *args: Any, close: Callable[[Any], None]) -> Any:
    """``call(*args)`` in a thread; a result that arrives after the caller was cancelled is closed."""
    fut = asyncio.ensure_future(asyncio.to_thread(call, *args))
    try:
        return await asyncio.shield(fut)
    except asyncio.CancelledError:
        fut.add_done_callback(
            lambda f: f.cancelled()
            or f.exception() is not None
            or f.result() is None
            or _quietly(close, f.result())
        )
        raise


def _quietly(fn: Callable[[Any], None], arg: Any) -> None:
    try:
        fn(arg)
    except Exception:
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


def worker_address(agent_port: int, *, held: bool = False) -> Optional[Tuple[str, str]]:
    """(address, public key) of this worker's persistent tailcat server, made on first use."""
    pub = own_pub("server", held = held)
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
    # Held across the helper call, so a concurrent drop() cannot be undone by this forward.
    with _LOCK:
        try:
            if _CLIENT is not None and getattr(_CLIENT, "dead", False):
                dead, _CLIENT = _CLIENT, None
                _FORWARDS.clear()
                _quietly(lambda c: c.close(), dead)
            if _CLIENT is None:
                own_pub("client")
                client = tn.Tailnet()
                client.start()
                _CLIENT = client
            client = _CLIENT
            old = _FORWARDS.get(node_id)
            if old and old[0] != address:
                del _FORWARDS[node_id]
                _quietly(lambda a: client.unforward(a), old[0])
            local = client.forward(address, int(port), key_path = key_path("client"))
        except Exception as exc:
            logger.info("cluster link: no tailcat forward to %s (%s)", node_id, type(exc).__name__)
            return None
        old = _FORWARDS.get(node_id)
        ports = old[1] if old else set()
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
            _quietly(lambda a: client.unforward(a), entry[0])


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
        net = self._net
        if net is None:
            return
        want = set(self._allowed())
        for pub in sorted(want - self._allow):
            net.allow(pub)
        for pub in sorted(self._allow - want):
            net.revoke(pub)
        self._allow = want

    def close(self) -> None:
        net, self._net = self._net, None
        if net is not None:
            try:
                net.close()
            except Exception:
                pass
