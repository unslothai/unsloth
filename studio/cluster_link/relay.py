# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The master side: one loopback listener per paired worker that ``llama-server --rpc`` connects to,
carried to that worker's agent over mutually authenticated TLS. It runs its own event loop in a daemon
thread, so RPC traffic never competes with the caller's loop (Studio's API)."""

from __future__ import annotations

import asyncio
import logging
import ssl
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Set, Tuple

from . import framing, tls
from .model import LocalIdentity
from .splice import STREAM_LIMIT, close_writer, tune_socket, splice

logger = logging.getLogger(__name__)

CONNECT_TIMEOUT_S = 10.0
FIRST_BYTE_TIMEOUT_S = 30.0
REPLY_TIMEOUT_S = 10.0
# A client that reconnects right after its last session ends must not be refused as busy while that
# session is still closing; a genuinely busy worker is still refused this fast, not after 8 s.
BUSY_GRACE_S = 0.25
# A live session re-checks its worker's authorization this often: `unsloth cluster remove` in another
# process only edits the registry, and authorization is otherwise checked at connect time only.
REVALIDATE_S = 2.0


class RelayError(RuntimeError):
    """The worker could not be reached, failed authentication, or refused the request."""


@dataclass
class _Relay:
    node_id: str
    host: str
    agent_port: int
    server: Optional[asyncio.AbstractServer] = None
    local_port: int = 0
    active: bool = False
    streams: int = 0
    last_error: str = ""
    counters: Dict[str, int] = field(default_factory = dict)
    conns: Set[asyncio.StreamWriter] = field(default_factory = set)
    idle: Optional[asyncio.Event] = None

    @property
    def endpoint(self) -> str:
        return f"127.0.0.1:{self.local_port}"


class RelayManager:
    def __init__(
        self, identity: LocalIdentity, authorize: Callable[[tls.PeerIdentity], bool]
    ) -> None:
        self.identity = identity
        self._authorize = authorize
        self._ctx = tls.mutual_context(
            server = False,
            certfile = identity.certfile,
            keyfile = identity.keyfile,
            rootfile = identity.rootfile,
        )
        self._relays: Dict[str, _Relay] = {}
        self._ensure_lock: Optional[asyncio.Lock] = None
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target = self._run, name = "unsloth-cluster-relay", daemon = True)
        self._thread.start()

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _call(
        self,
        coro: Any,
        timeout: Optional[float] = None,
    ) -> Any:
        if not self._loop.is_running() and self._loop.is_closed():
            raise RelayError("the relay manager is stopped")
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout)

    def ensure(self, node_id: str, host: str, agent_port: int) -> str:
        return self._call(self._ensure(node_id, host, int(agent_port)))

    def remove(self, node_id: str) -> None:
        self._call(self._remove(node_id))

    def endpoints(self) -> Dict[str, str]:
        return {n: r.endpoint for n, r in self._relays.items()}

    def counters(self, node_id: str) -> Dict[str, int]:
        relay = self._relays.get(node_id)
        return dict(relay.counters) if relay else {}

    def status(
        self,
        node_id: str,
        timeout: float = 3.0,
    ) -> Dict[str, Any]:
        try:
            reply = self._call(self._control(node_id, {"type": "status"}, timeout), timeout + 1.0)
        except Exception as exc:
            reply = {"type": "error", "error": str(exc) or type(exc).__name__, "reachable": False}
        else:
            reply["reachable"] = True
        relay = self._relays.get(node_id)
        if relay is not None:
            reply["relay"] = {
                "endpoint": relay.endpoint,
                "active": relay.active,
                "streams": relay.streams,
                "last_error": relay.last_error,
                "counters": dict(relay.counters),
            }
        return reply

    def busy(self, node_id: str) -> bool:
        relay = self._relays.get(node_id)
        if relay is not None and relay.active:
            return True
        return bool(self.status(node_id).get("busy"))

    def stop(self) -> None:
        if self._loop.is_closed():
            return
        try:
            self._call(self._stop_all(), 10.0)
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(5.0)
            if not self._loop.is_running():
                self._loop.close()

    async def _ensure(self, node_id: str, host: str, agent_port: int) -> str:
        # Serialized: start_server yields, and two callers creating the same relay at once each got a
        # listener, one of them never registered and so never closed by remove() or stop().
        if self._ensure_lock is None:
            self._ensure_lock = asyncio.Lock()
        async with self._ensure_lock:
            return await self._ensure_locked(node_id, host, agent_port)

    async def _ensure_locked(self, node_id: str, host: str, agent_port: int) -> str:
        relay = self._relays.get(node_id)
        if relay is not None:
            relay.host, relay.agent_port = host, agent_port
            return relay.endpoint
        relay = _Relay(node_id = node_id, host = host, agent_port = agent_port, idle = asyncio.Event())
        relay.idle.set()
        relay.server = await asyncio.start_server(
            lambda r, w: self._on_local(relay, r, w),
            "127.0.0.1",
            0,
            limit = STREAM_LIMIT,
        )
        relay.local_port = relay.server.sockets[0].getsockname()[1]
        self._relays[node_id] = relay
        return relay.endpoint

    async def _remove(self, node_id: str) -> None:
        relay = self._relays.pop(node_id, None)
        if relay is None:
            return
        if relay.server is not None:
            relay.server.close()
        for writer in list(relay.conns):
            await close_writer(writer, abort = True)

    async def _stop_all(self) -> None:
        servers = [r.server for r in self._relays.values() if r.server is not None]
        for node_id in list(self._relays):
            await self._remove(node_id)
        for server in servers:
            try:
                await asyncio.wait_for(server.wait_closed(), 2.0)
            except (asyncio.TimeoutError, Exception):
                pass
        current = asyncio.current_task()
        pending = [t for t in asyncio.all_tasks() if t is not current]
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions = True)

    async def _connect(
        self,
        relay_or_id: Any,
        timeout: float = CONNECT_TIMEOUT_S,
    ) -> Tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        relay = relay_or_id if isinstance(relay_or_id, _Relay) else self._relays.get(relay_or_id)
        if relay is None:
            raise RelayError(f"no relay for {relay_or_id}")
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(
                    relay.host,
                    relay.agent_port,
                    ssl = self._ctx,
                    server_hostname = "",
                    ssl_handshake_timeout = timeout,
                    limit = STREAM_LIMIT,
                ),
                timeout,
            )
        except (OSError, ssl.SSLError, asyncio.TimeoutError) as exc:
            raise RelayError(
                f"cannot reach {relay.node_id} at {relay.host}:{relay.agent_port}: {exc}"
            ) from None
        pid = tls.peer_identity(writer.get_extra_info("ssl_object"))
        ok = pid is not None and pid.role == "worker" and pid.node_id == relay.node_id
        if ok:
            try:
                ok = bool(self._authorize(pid))
            except Exception:
                logger.exception("cluster relay: authorization check failed")
                ok = False
        if not ok:
            await close_writer(writer, abort = True)
            raise RelayError(
                f"{relay.host}:{relay.agent_port} is not the paired worker {relay.node_id}"
            )
        tune_socket(writer)
        return reader, writer

    async def _revoke_when_unauthorized(
        self,
        relay: _Relay,
        pid: Optional[tls.PeerIdentity],
        upstream: asyncio.StreamWriter,
        local: asyncio.StreamWriter,
    ) -> None:
        while True:
            await asyncio.sleep(REVALIDATE_S)
            try:
                # The check reads the registry from disk: off the loop that carries the RPC bytes.
                ok = pid is not None and bool(await asyncio.to_thread(self._authorize, pid))
            except Exception:
                ok = False
            if not ok:
                relay.last_error = f"{relay.node_id} is no longer paired; session closed"
                logger.warning("cluster relay: %s", relay.last_error)
                upstream.transport.abort()
                local.transport.abort()
                return

    async def _wait_idle(self, relay: _Relay) -> bool:
        if not relay.active:
            return True
        try:
            await asyncio.wait_for(relay.idle.wait(), BUSY_GRACE_S)
        except asyncio.TimeoutError:
            return False
        return not relay.active

    async def _control(
        self, node_id: str, request: Dict[str, Any], timeout: float
    ) -> Dict[str, Any]:
        reader, writer = await self._connect(node_id, timeout)
        try:
            await framing.write_frame(writer, request)
            return await framing.read_frame(reader, timeout = timeout)
        finally:
            await close_writer(writer)

    async def _on_local(
        self, relay: _Relay, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        tune_socket(writer)
        if not await self._wait_idle(relay):
            # A probe of a held worker: fail it now rather than make it wait out a timeout.
            await close_writer(writer, abort = True)
            return
        relay.conns.add(writer)
        claimed = False
        upstream: Optional[asyncio.StreamWriter] = None
        try:
            try:
                first = await asyncio.wait_for(reader.read(STREAM_LIMIT), FIRST_BYTE_TIMEOUT_S)
            except (OSError, asyncio.TimeoutError):
                first = b""
            if not first:
                # A port check, or a client that left: the worker never hears of it.
                return
            if not await self._wait_idle(relay):
                return
            relay.active = claimed = True
            relay.idle.clear()
            try:
                up_reader, upstream = await self._connect(relay)
                await framing.write_frame(upstream, {"type": "open_rpc"})
                reply = await framing.read_frame(up_reader, timeout = REPLY_TIMEOUT_S)
            except (RelayError, framing.FrameError, OSError, ssl.SSLError) as exc:
                relay.last_error = str(exc)
                logger.warning("cluster relay %s: %s", relay.node_id, exc)
                return
            if reply.get("type") != "rpc_ok":
                relay.last_error = str(reply.get("error") or reply.get("type"))
                logger.warning("cluster relay %s: worker said %s", relay.node_id, relay.last_error)
                return
            relay.streams += 1
            relay.last_error = ""
            relay.counters["to_worker"] = relay.counters.get("to_worker", 0) + len(first)
            upstream.write(first)
            await upstream.drain()
            stream, upstream = upstream, None

            def release() -> None:
                nonlocal claimed
                if claimed:
                    claimed = False
                    relay.active = False
                    relay.idle.set()

            pid = tls.peer_identity(stream.get_extra_info("ssl_object"))
            watch = asyncio.ensure_future(
                self._revoke_when_unauthorized(relay, pid, stream, writer)
            )
            try:
                await splice(
                    reader,
                    writer,
                    up_reader,
                    stream,
                    relay.counters,
                    "to_worker",
                    "from_worker",
                    on_done = release,
                )
            finally:
                watch.cancel()
        except (OSError, ssl.SSLError) as exc:
            relay.last_error = str(exc)
        finally:
            relay.conns.discard(writer)
            if claimed:
                relay.active = False
                relay.idle.set()
            await close_writer(upstream, abort = True)
            await close_writer(writer, abort = True)
