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
import socket
import time

from .splice import (
    STREAM_LIMIT,
    SpliceSession,
    close_writer,
    local_client_is_ours,
    socket_splice,
    splice,
    threaded_sessions,
    tune_raw_socket,
    tune_socket,
)

logger = logging.getLogger(__name__)

CONNECT_TIMEOUT_S = 10.0
FIRST_BYTE_TIMEOUT_S = 30.0
REPLY_TIMEOUT_S = 10.0
# open_rpc may first move a worker out of native mode, which restarts its rpc-server and waits up to
# the agent's RPC_READY_TIMEOUT_S (60 s) for it: the reply wait covers that window.
OPEN_RPC_REPLY_TIMEOUT_S = 75.0
# A client that reconnects right after its last session ends must not be refused as busy while that
# session is still closing; a genuinely busy worker is still refused this fast, not after 8 s.
BUSY_GRACE_S = 0.25
# Re-check worker authorization this often: `cluster remove` elsewhere only edits the registry.
REVALIDATE_S = 2.0
# Backoff cap for re-sending native_close to a removed worker that did not answer.
NATIVE_CLOSE_RETRY_MAX_S = 30.0


def _drop(sock: Any) -> None:
    for step in (lambda: sock.shutdown(socket.SHUT_RDWR), sock.close):
        try:
            step()
        except (OSError, ValueError):
            pass


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
    listener: Optional[socket.socket] = None
    raw: Set[socket.socket] = field(default_factory = set)
    sessions: Set[SpliceSession] = field(default_factory = set)

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
        # Native grants carry no relay session to revalidate: one watch per granted worker does it.
        self._native_watches: Dict[str, asyncio.Future] = {}
        self._ensure_lock: Optional[asyncio.Lock] = None
        self.threaded = threaded_sessions()
        self._slot_lock = threading.Lock()
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

    def destination(self, node_id: str) -> Optional[Tuple[str, int]]:
        """Where an existing relay sends its sessions (host, agent port), or None."""
        relay = self._relays.get(node_id)
        return (relay.host, relay.agent_port) if relay is not None else None

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

    def native(
        self,
        node_id: str,
        timeout: float = 70.0,
    ) -> Dict[str, Any]:
        """Ask the worker to serve llama.cpp's own TLS to this master; {"type": "native_ok", "port",
        "fingerprint"} when it does. The worker restarts its rpc-server, so this waits for it."""
        try:
            return self._call(self._native(node_id, timeout), timeout + 1.0)
        except Exception as exc:
            return {"type": "error", "error": str(exc) or type(exc).__name__}

    def native_close(
        self,
        node_id: str,
        timeout: float = 15.0,
    ) -> bool:
        """Tell a worker to stop serving this master natively, which cuts a live session. Also for a
        worker removed a moment ago: it is no longer authorized here, but it is still the same node."""
        try:
            reply = self._call(self._native_close(node_id, timeout), timeout + 1.0)
        except Exception:
            return False
        return isinstance(reply, dict) and reply.get("type") == "ok"

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
        # Serialized: concurrent creators each got a listener, one never registered and never closed.
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
        if self.threaded:
            listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                listener.bind(("127.0.0.1", 0))
                listener.listen(8)
                listener.settimeout(0.5)
            except OSError:
                listener.close()
                raise
            relay.listener = listener
            relay.local_port = listener.getsockname()[1]
            threading.Thread(
                target = self._accept_loop,
                args = (relay, listener),
                name = "unsloth-cluster-relay-accept",
                daemon = True,
            ).start()
        else:
            relay.server = await asyncio.start_server(
                lambda r, w: self._on_local(relay, r, w),
                "127.0.0.1",
                0,
                limit = STREAM_LIMIT,
            )
            relay.local_port = relay.server.sockets[0].getsockname()[1]
        self._relays[node_id] = relay
        return relay.endpoint

    async def _native(self, node_id: str, timeout: float) -> Dict[str, Any]:
        relay = self._relays.get(node_id)
        reader, writer = await self._connect(node_id, timeout)
        try:
            pid = tls.peer_identity(writer.get_extra_info("ssl_object"))
            await framing.write_frame(writer, {"type": "native_rpc"})
            reply = await framing.read_frame(reader, timeout = timeout)
        finally:
            await close_writer(writer)
        if relay is not None and isinstance(reply, dict) and reply.get("type") == "native_ok":
            self._stop_native_watch(node_id)
            self._native_watches[node_id] = asyncio.ensure_future(
                self._revoke_native_when_unauthorized(relay, pid)
            )
        return reply

    async def _native_close(self, node_id: str, timeout: float) -> Dict[str, Any]:
        reply = await self._control(node_id, {"type": "native_close"}, timeout, paired = False)
        # Only an answer ends the watch: after a failed close it keeps revalidating and retrying.
        self._stop_native_watch(node_id)
        return reply

    def _stop_native_watch(self, node_id: str) -> None:
        watch = self._native_watches.pop(node_id, None)
        if watch is not None and watch is not asyncio.current_task():
            watch.cancel()

    async def _revoke_native_when_unauthorized(
        self, relay: _Relay, pid: Optional[tls.PeerIdentity]
    ) -> None:
        """The relay's 2 s revalidation for a native session, which does not pass through this
        process: once the worker is removed here, tell it to close, which cuts the session."""
        while True:
            await asyncio.sleep(REVALIDATE_S)
            try:
                ok = pid is not None and bool(await asyncio.to_thread(self._authorize, pid))
            except Exception:
                ok = False
            if ok:
                continue
            relay.last_error = f"{relay.node_id} is no longer paired; native session closed"
            logger.warning("cluster relay: %s", relay.last_error)
            break
        # The worker still trusts this master: keep asking until it acknowledges.
        delay = REVALIDATE_S
        while True:
            try:
                # The relay object, not its id: Studio may already have dropped the relay.
                reply = await self._control(relay, {"type": "native_close"}, 15.0, paired = False)
                if isinstance(reply, dict) and reply.get("type") == "ok":
                    break
                # A non-"ok" answer means the worker already left native mode for this master: done.
                logger.warning(
                    "cluster relay: %s answered native_close with %s", relay.node_id, reply
                )
                break
            except Exception as exc:
                logger.warning("cluster relay: could not close %s natively: %s", relay.node_id, exc)
            await asyncio.sleep(delay)
            delay = min(delay * 2, NATIVE_CLOSE_RETRY_MAX_S)
        if self._native_watches.get(relay.node_id) is asyncio.current_task():
            del self._native_watches[relay.node_id]

    async def _remove(self, node_id: str) -> None:
        relay = self._relays.pop(node_id, None)
        if relay is None:
            return
        if relay.server is not None:
            relay.server.close()
        for writer in list(relay.conns):
            await close_writer(writer, abort = True)
        if relay.listener is not None:
            listener, relay.listener = relay.listener, None
            try:
                listener.close()
            except OSError:
                pass
        for session in list(relay.sessions):
            session.abort()
        for sock in list(relay.raw):
            _drop(sock)

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
        paired: bool = True,
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
        if ok and paired:
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
        self,
        node_id: Any,  # a node id, or the _Relay itself once Studio may have dropped it
        request: Dict[str, Any],
        timeout: float,
        paired: bool = True,
    ) -> Dict[str, Any]:
        reader, writer = await self._connect(node_id, timeout, paired = paired)
        try:
            await framing.write_frame(writer, request)
            return await framing.read_frame(reader, timeout = timeout)
        finally:
            await close_writer(writer)

    def _accept_loop(self, relay: _Relay, listener: socket.socket) -> None:
        while relay.listener is listener:
            try:
                conn, _ = listener.accept()
            except socket.timeout:
                continue
            except OSError:
                if relay.listener is not listener:
                    return
                time.sleep(0.05)
                continue
            threading.Thread(
                target = self._on_local_threaded,
                args = (relay, conn),
                name = "unsloth-cluster-relay-conn",
                daemon = True,
            ).start()

    def _wait_idle_threaded(self, relay: _Relay) -> bool:
        deadline = time.monotonic() + BUSY_GRACE_S
        while relay.active:
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.005)
        return True

    def _claim(self, relay: _Relay) -> bool:
        with self._slot_lock:
            if relay.active:
                return False
            relay.active = True
        try:
            self._loop.call_soon_threadsafe(relay.idle.clear)
        except RuntimeError:
            pass  # the manager is stopping
        return True

    def _unclaim(self, relay: _Relay) -> None:
        with self._slot_lock:
            relay.active = False
        try:
            self._loop.call_soon_threadsafe(relay.idle.set)
        except RuntimeError:
            pass  # the manager is stopping

    def _connect_blocking(self, relay: _Relay) -> Any:
        try:
            raw = socket.create_connection(
                (relay.host, relay.agent_port), timeout = CONNECT_TIMEOUT_S
            )
        except OSError as exc:
            raise RelayError(
                f"cannot reach {relay.node_id} at {relay.host}:{relay.agent_port}: {exc}"
            ) from None
        try:
            sock = self._ctx.wrap_socket(raw)
        except (OSError, ssl.SSLError, ValueError) as exc:
            raw.close()
            raise RelayError(
                f"cannot reach {relay.node_id} at {relay.host}:{relay.agent_port}: {exc}"
            ) from None
        pid = tls.peer_identity(sock)
        ok = pid is not None and pid.role == "worker" and pid.node_id == relay.node_id
        if ok:
            try:
                ok = bool(self._authorize(pid))
            except Exception:
                logger.exception("cluster relay: authorization check failed")
                ok = False
        if not ok:
            _drop(sock)
            raise RelayError(
                f"{relay.host}:{relay.agent_port} is not the paired worker {relay.node_id}"
            )
        tune_raw_socket(sock)
        return sock

    def _on_local_threaded(self, relay: _Relay, conn: socket.socket) -> None:
        tune_raw_socket(conn)
        if not local_client_is_ours(conn):
            logger.warning("cluster relay: refused a local client owned by another user")
            _drop(conn)
            return
        if not self._wait_idle_threaded(relay):
            # A probe of a held worker: fail it now rather than make it wait out a timeout.
            _drop(conn)
            return
        relay.raw.add(conn)
        claimed = False
        upstream: Any = None
        try:
            conn.settimeout(FIRST_BYTE_TIMEOUT_S)
            try:
                first = conn.recv(STREAM_LIMIT)
            except OSError:
                first = b""
            if not first:
                return  # a port check, or a client that left: the worker never hears of it
            if not self._wait_idle_threaded(relay) or not self._claim(relay):
                return
            claimed = True
            try:
                upstream = self._connect_blocking(relay)
                relay.raw.add(upstream)
                framing.send_frame(upstream, {"type": "open_rpc"})
                reply = framing.recv_frame(upstream, timeout = OPEN_RPC_REPLY_TIMEOUT_S)
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
            relay.counters["threaded_streams"] = relay.counters.get("threaded_streams", 0) + 1
            relay.counters["to_worker"] = relay.counters.get("to_worker", 0) + len(first)

            def release() -> None:
                nonlocal claimed
                if claimed:
                    claimed = False
                    self._unclaim(relay)

            pid = tls.peer_identity(upstream)
            session = SpliceSession()
            relay.sessions.add(session)
            ended = threading.Event()

            def watch() -> None:
                while not ended.wait(REVALIDATE_S):
                    try:
                        ok = pid is not None and bool(self._authorize(pid))
                    except Exception:
                        ok = False
                    if not ok:
                        relay.last_error = f"{relay.node_id} is no longer paired; session closed"
                        logger.warning("cluster relay: %s", relay.last_error)
                        session.abort()
                        return

            threading.Thread(
                target = watch, name = "unsloth-cluster-relay-revalidate", daemon = True
            ).start()
            stream, upstream = upstream, None
            try:
                socket_splice(
                    conn,
                    stream,
                    relay.counters,
                    "to_worker",
                    "from_worker",
                    on_done = release,
                    session = session,
                    pending_a = first,
                )
            finally:
                ended.set()
                relay.sessions.discard(session)
                relay.raw.discard(stream)
        except (OSError, ssl.SSLError) as exc:
            relay.last_error = str(exc)
        finally:
            relay.raw.discard(conn)
            if claimed:
                self._unclaim(relay)
            if upstream is not None:
                relay.raw.discard(upstream)
                _drop(upstream)
            _drop(conn)

    async def _on_local(
        self, relay: _Relay, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        tune_socket(writer)
        sock = writer.get_extra_info("socket")
        if sock is None or not await asyncio.to_thread(local_client_is_ours, sock):
            logger.warning("cluster relay: refused a local client owned by another user")
            await close_writer(writer, abort = True)
            return
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
                reply = await framing.read_frame(up_reader, timeout = OPEN_RPC_REPLY_TIMEOUT_S)
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
