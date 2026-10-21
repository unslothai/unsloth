# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The worker agent: a worker's control plane. It supervises one ``ggml-rpc-server``: bound to loopback,
with one authenticated master's TLS connection spliced to it at a time, or (native mode, on that master's
request) serving llama.cpp's own mutual TLS on the LAN to that master alone."""

from __future__ import annotations

import asyncio
import collections
import json
import logging
import os
import shutil
import signal
import ssl
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Mapping, Optional, Sequence, Tuple

from . import (
    AGENT_PORT_DEFAULT,
    NATIVE_RPC_PORT_DEFAULT,
    PROTOCOL_VERSION,
    RPC_PORT_DEFAULT,
    SECURE_RPC_ENV,
    framing,
    native,
    tls,
)
from .model import LocalIdentity
from .paths import run_dir, state_dir
import socket

from .splice import (
    STREAM_LIMIT,
    SpliceSession,
    close_writer,
    is_private_address,
    socket_splice,
    splice,
    threaded_sessions,
)

logger = logging.getLogger(__name__)

HANDSHAKE_TIMEOUT_S = 10.0
RPC_CONNECT_TIMEOUT_S = 5.0
# A model load can arrive right after `unsloth worker` starts: give rpc-server this long to bind.
RPC_READY_TIMEOUT_S = 60.0
RESTART_BACKOFF_MAX_S = 30.0
STABLE_RUN_S = 60.0
BUSY_GRACE_S = 0.25
# A live session re-checks its master's authorization this often: `unsloth cluster leave` (or a
# re-pair) in another process only edits the registry, which is otherwise read at connect time.
REVALIDATE_S = 2.0
_TAIL = 50


def _die_with_agent_hook() -> Optional[Callable[[], None]]:
    """A preexec_fn arming PR_SET_PDEATHSIG: the kernel stops rpc-server when the agent dies, even by
    SIGKILL, so a killed worker never leaves an rpc-server holding its port. It fires when the forking
    THREAD exits; that is the agent's event loop thread, which lives as long as the agent."""
    if not sys.platform.startswith("linux"):
        return None
    try:
        import ctypes

        # Resolved here, not in the child: a forked child must not import or dlopen before exec.
        prctl = ctypes.CDLL(None, use_errno = True).prctl
    except Exception:
        return None
    # Compared against the agent's own pid, not 1: an agent that IS pid 1 (a container entrypoint)
    # is every child's parent, and a "parent is 1" test killed each rpc-server it started.
    agent_pid = os.getpid()

    def hook() -> None:
        prctl(1, signal.SIGTERM)
        if os.getppid() != agent_pid:
            os._exit(0)  # the agent died before the signal was armed

    return hook


_SETPRIV: List[Optional[str]] = []


def _setpriv() -> Optional[str]:
    """util-linux setpriv with --pdeathsig (2.33+), which arms the signal in the child itself."""
    if not _SETPRIV:
        found = shutil.which("setpriv")
        try:
            ok = (
                bool(found)
                and "--pdeathsig"
                in subprocess.run(
                    [found, "--help"], capture_output = True, text = True, timeout = 5
                ).stdout
            )
        except (OSError, subprocess.SubprocessError):
            ok = False
        _SETPRIV.append(found if ok else None)
    return _SETPRIV[0]


def _die_with_agent_launch() -> Tuple[List[str], Optional[Callable[[], None]]]:
    """An argv prefix and preexec_fn that make rpc-server die with the agent. preexec_fn is unsafe
    once the process has other threads (the LAN announcer, executor threads): use setpriv then, and
    the hook only while the agent is single-threaded. Neither: the next agent reaps a leftover."""
    hook = _die_with_agent_hook()
    if hook is None:
        return [], None
    setpriv = _setpriv()
    if setpriv:
        return [setpriv, "--pdeathsig", "TERM", "--"], None
    if threading.active_count() == 1:
        return [], hook
    return [], None


def _lock_rpc_port(port: int) -> int:
    """An exclusive, non-blocking lock on this rpc port, machine-wide for this user: a held lock
    means another live worker serves on the port."""
    fd = os.open(str(run_dir() / f"rpc-server-{port}.lock"), os.O_CREAT | os.O_RDWR, 0o600)
    try:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        raise RuntimeError(
            f"another `unsloth worker` already runs ggml-rpc-server on port {port}; stop it or "
            "start this one with a different --rpc-port"
        ) from None
    return fd


def _unlock(fd: Optional[int]) -> None:
    if fd is None:
        return
    try:
        if os.name == "nt":
            import msvcrt
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    except OSError:
        pass
    os.close(fd)


def _listen(host: str, port: int) -> socket.socket:
    """The agent's listening socket for threaded sessions (asyncio.start_server's bind, minus asyncio)."""
    infos = socket.getaddrinfo(host or None, port, type = socket.SOCK_STREAM, flags = socket.AI_PASSIVE)
    family, socktype, proto, _, addr = infos[0]
    sock = socket.socket(family, socktype, proto)
    try:
        if os.name != "nt":  # on Windows SO_REUSEADDR lets another process take the port
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(addr)
        sock.listen(16)
        # A short timeout lets the accept thread notice a stop on every platform.
        sock.settimeout(0.5)
    except OSError:
        sock.close()
        raise
    return sock


def _drop(sock: socket.socket) -> None:
    for step in (lambda: sock.shutdown(socket.SHUT_RDWR), sock.close):
        try:
            step()
        except (OSError, ValueError):
            pass


class Agent:
    def __init__(
        self,
        identity: LocalIdentity,
        authorize: Callable[[tls.PeerIdentity], bool],
        *,
        host: str = "0.0.0.0",
        port: int = AGENT_PORT_DEFAULT,
        rpc_binary: Optional[str] = None,
        rpc_port: int = RPC_PORT_DEFAULT,
        rpc_args: Sequence[str] = (),
        cache_dir: Optional[str] = None,
        allow_any_network: bool = False,
        rpc_target: Optional[Tuple[str, int]] = None,
        source_filter: Optional[Callable[[str], bool]] = None,
        rpc_env: Optional[Mapping[str, str]] = None,
        native_port: int = NATIVE_RPC_PORT_DEFAULT,
        native_tls: Optional[bool] = None,
    ) -> None:
        self.identity = identity
        self._authorize = authorize
        self.host = host
        self.port = port
        self.rpc_binary = rpc_binary
        self.rpc_port = rpc_port
        # Native mode: rpc-server serves llama.cpp's own mutual TLS on the LAN for the one master that
        # asked, instead of plain TCP on loopback behind this agent. One rpc-server either way: two
        # would each hold an accelerator context. None = use it when the bundle has it.
        self.native_port = native_port
        self._native_tls = native_tls
        self.native_available = False
        self.mode = "loopback"
        self._native_master: Optional[tls.PeerIdentity] = None
        self._native_watch: Optional[asyncio.Task] = None
        self._native_lock: Optional[asyncio.Lock] = None
        self._native_port_lock: Optional[int] = None
        self._planned_restart = False
        self.rpc_args = [str(a) for a in rpc_args]
        self.cache_dir = cache_dir
        self._base_env = dict(rpc_env) if rpc_env is not None else None
        self._rpc_target = rpc_target or ("127.0.0.1", rpc_port)
        if source_filter is not None:
            self._source_allowed = source_filter
        elif allow_any_network:
            self._source_allowed = lambda _host: True
        else:
            self._source_allowed = is_private_address
        self._server: Optional[asyncio.AbstractServer] = None
        self._proc: Optional[asyncio.subprocess.Process] = None
        self._supervisor: Optional[asyncio.Task] = None
        self._stopping = False
        self._active = 0
        self._idle: Optional[asyncio.Event] = None
        self._conns: set = set()
        self.tail: Deque[str] = collections.deque(maxlen = _TAIL)
        self.stats: Dict[str, int] = {
            "connections": 0,
            "rejected": 0,
            "rpc_streams": 0,
            "rpc_restarts": 0,
        }
        self.counters: Dict[str, int] = {}
        self._rpc_ready: Optional[asyncio.Event] = None
        self._port_lock: Optional[int] = None
        # Threaded sessions (default): an accept thread and one thread per connection, see splice.py.
        self.threaded = False
        self._listener: Optional[socket.socket] = None
        self._accept_thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._closed: Optional[asyncio.Event] = None
        self._slot_lock = threading.Lock()
        self._raw: set = set()
        self._sessions: set = set()

    @property
    def rpc_endpoint(self) -> str:
        return f"{self._rpc_target[0]}:{self._rpc_target[1]}"

    @property
    def rpc_up(self) -> bool:
        """rpc-server is running AND accepts on its loopback port: a live process that has not bound
        yet would fail the master's load."""
        if self.rpc_binary is None:
            return True
        alive = self._proc is not None and self._proc.returncode is None
        return alive and self._rpc_ready is not None and self._rpc_ready.is_set()

    @property
    def busy(self) -> bool:
        return self._active > 0

    @property
    def rpc_pid(self) -> Optional[int]:
        alive = self._proc is not None and self._proc.returncode is None
        return self._proc.pid if alive else None

    async def wait_rpc_ready(self, timeout: float) -> bool:
        if self.rpc_binary is None:
            return True
        if self._rpc_ready is None:
            return False
        try:
            await asyncio.wait_for(self._rpc_ready.wait(), timeout)
        except asyncio.TimeoutError:
            return False
        return self.rpc_up

    async def start(self) -> None:
        self._stopping = False
        self._rpc_ready = asyncio.Event()
        self._idle = asyncio.Event()
        self._idle.set()
        ctx = tls.mutual_context(
            server = True,
            certfile = self.identity.certfile,
            keyfile = self.identity.keyfile,
            rootfile = self.identity.rootfile,
        )
        self._loop = asyncio.get_running_loop()
        self._closed = asyncio.Event()
        self.threaded = threaded_sessions()
        if self.threaded:
            self._ctx = ctx
            self._listener = _listen(self.host, self.port)
            self.port = self._listener.getsockname()[1]
            self._accept_thread = threading.Thread(
                target = self._accept_loop, name = "unsloth-cluster-agent-accept", daemon = True
            )
            self._accept_thread.start()
        else:
            self._server = await asyncio.start_server(
                self._handle,
                self.host,
                self.port,
                ssl = ctx,
                ssl_handshake_timeout = HANDSHAKE_TIMEOUT_S,
                limit = STREAM_LIMIT,
                reuse_address = True,
            )
            sockets = self._server.sockets or []
            if sockets:
                self.port = sockets[0].getsockname()[1]
        self._native_lock = asyncio.Lock()
        self.mode = "loopback"
        self.native_available = False
        if self._native_tls is None:
            wanted = not native.disabled()
        else:
            wanted = bool(self._native_tls)
        # Only once the port is ours: a failed bind must not leave an rpc-server running.
        if self.rpc_binary is not None:
            try:
                # Held for the agent's whole life, released by the kernel however it ends: two
                # workers starting at once on one rpc port cannot both pass the check below.
                self._port_lock = _lock_rpc_port(self.rpc_port)
                # After the lock: a slow `--help` must not let a second worker pass it meanwhile.
                self.native_available = bool(
                    wanted and await asyncio.to_thread(native.binary_supports_tls, self.rpc_binary)
                )
                if self.native_available:
                    self._native_port_lock = _lock_rpc_port(self.native_port)
                    native.write_allowlist(native.allowlist_path(self.native_port), [])
                self._reap_stale_rpc()
            except (RuntimeError, OSError):
                if self._server is not None:
                    self._server.close()
                    self._server = None
                self._close_listener()
                _unlock(self._port_lock)
                _unlock(self._native_port_lock)
                self._port_lock = self._native_port_lock = None
                raise
            self._supervisor = asyncio.ensure_future(self._supervise())

    async def serve_forever(self) -> None:
        if self._server is None and self._listener is None:
            await self.start()
        try:
            if self._server is not None:
                await self._server.serve_forever()
            else:
                assert self._closed is not None
                await self._closed.wait()
        except asyncio.CancelledError:
            pass

    async def stop(self) -> None:
        try:
            await self._stop()
        finally:
            _unlock(self._port_lock)
            _unlock(self._native_port_lock)
            self._port_lock = self._native_port_lock = None

    async def _stop(self) -> None:
        self._stopping = True
        if self._server is not None:
            self._server.close()
            try:
                await asyncio.wait_for(self._server.wait_closed(), 2.0)
            except (asyncio.TimeoutError, Exception):
                pass
            self._server = None
        for writer in list(self._conns):
            await close_writer(writer, abort = True)
        if self._listener is not None:
            self._close_listener()
            for session in list(self._sessions):
                session.abort()
            for sock in list(self._raw):
                _drop(sock)
            if self._accept_thread is not None:
                await asyncio.to_thread(self._accept_thread.join, 2.0)
                self._accept_thread = None
        if self._closed is not None:
            self._closed.set()
        if self._native_watch is not None:
            self._native_watch.cancel()
            self._native_watch = None
        if self.native_available:
            # Nobody may connect to a later rpc-server on this port with a stale grant.
            try:
                native.write_allowlist(native.allowlist_path(self.native_port), [])
            except OSError:
                pass
        if self._supervisor is not None:
            self._supervisor.cancel()
            await asyncio.gather(self._supervisor, return_exceptions = True)
            self._supervisor = None
        await self._kill_rpc()

    def _cache(self) -> Path:
        cache = Path(self.cache_dir) if self.cache_dir else state_dir() / "llama_cache"
        cache.mkdir(parents = True, exist_ok = True)
        return cache

    def _rpc_env(self) -> Dict[str, str]:
        env = dict(self._base_env if self._base_env is not None else os.environ)
        env.update(SECURE_RPC_ENV)
        env["LLAMA_CACHE"] = str(self._cache())
        return env

    # PR_SET_PDEATHSIG is Linux only: on macOS and Windows a killed agent leaves its rpc-server
    # holding the port, so the next agent finds it through this record and stops it first.
    def _record_path(self) -> Path:
        return run_dir() / f"rpc-server-{self.rpc_port}.json"

    def _record_rpc(self, pid: int) -> None:
        try:
            import psutil
            created = psutil.Process(pid).create_time()
        except Exception:
            return
        try:
            self._record_path().write_text(
                json.dumps({"pid": pid, "created": created}), encoding = "utf-8"
            )
        except OSError:
            pass

    def _forget_rpc(self) -> None:
        try:
            self._record_path().unlink()
        except OSError:
            pass

    def _reap_stale_rpc(self) -> None:
        try:
            record = json.loads(self._record_path().read_text(encoding = "utf-8"))
        except Exception:
            record = None
        # This agent holds the port lock, so the agent that wrote the record is gone.
        try:
            import psutil

            proc = psutil.Process(int(record["pid"]))
            # The creation time pins the exact process: a reused pid is someone else's.
            if abs(proc.create_time() - float(record["created"])) > 0.01:
                return
            if not any("rpc-server" in str(part) for part in [proc.name(), *proc.cmdline()[:2]]):
                return
            proc.terminate()
            try:
                proc.wait(3)
            except psutil.TimeoutExpired:
                proc.kill()
            logger.warning(
                "stopped ggml-rpc-server %s left by an agent that did not exit cleanly", proc.pid
            )
        except Exception:
            pass
        finally:
            self._forget_rpc()

    def _rpc_argv(self) -> list:
        if self.mode == "native":
            # TLS 1.3 on the LAN: only a client whose certificate chains to the cluster root AND whose
            # fingerprint is in the allowlist (the one master that asked) gets a byte through.
            return [
                str(self.rpc_binary),
                "-H",
                self.host or "0.0.0.0",
                "-p",
                str(self.native_port),
                "-c",
                "--tls-cert",
                self.identity.certfile,
                "--tls-key",
                self.identity.keyfile,
                "--tls-ca",
                self.identity.rootfile,
                "--tls-allow",
                str(native.allowlist_path(self.native_port)),
                *self.rpc_args,
            ]
        return [
            str(self.rpc_binary),
            "-H",
            "127.0.0.1",
            "-p",
            str(self.rpc_port),
            "-c",
            *self.rpc_args,
        ]

    @property
    def _probe_target(self) -> Tuple[str, int]:
        if self.mode == "native":
            host = self.host if self.host not in ("", "0.0.0.0", "::") else "127.0.0.1"
            return host, self.native_port
        return self._rpc_target

    async def _drain_output(self, proc: asyncio.subprocess.Process) -> None:
        assert proc.stdout is not None
        while True:
            line = await proc.stdout.readline()
            if not line:
                return
            self.tail.append(line.decode("utf-8", "replace").rstrip())

    async def _supervise(self) -> None:
        backoff = 0.5
        while not self._stopping:
            started = time.monotonic()
            try:
                prefix, preexec = _die_with_agent_launch()
                self._proc = await asyncio.create_subprocess_exec(
                    *prefix,
                    *self._rpc_argv(),
                    stdin = asyncio.subprocess.DEVNULL,
                    stdout = asyncio.subprocess.PIPE,
                    stderr = asyncio.subprocess.STDOUT,
                    env = self._rpc_env(),
                    preexec_fn = preexec,
                )
            except OSError as exc:
                self.tail.append(f"could not start {self.rpc_binary}: {exc}")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, RESTART_BACKOFF_MAX_S)
                continue
            self._record_rpc(self._proc.pid)
            drain = asyncio.ensure_future(self._drain_output(self._proc))
            probe = asyncio.ensure_future(self._probe_rpc(self._proc))
            rc = await self._proc.wait()
            probe.cancel()
            if self._rpc_ready is not None:
                self._rpc_ready.clear()
            await asyncio.gather(drain, probe, return_exceptions = True)
            if self._stopping:
                return
            if self._planned_restart:
                # A mode switch or a revocation stopped it on purpose: back at once, not counted.
                self._planned_restart = False
                backoff = 0.5
                continue
            self.stats["rpc_restarts"] += 1
            logger.warning("ggml-rpc-server exited with %s; restarting", rc)
            if time.monotonic() - started > STABLE_RUN_S:
                backoff = 0.5
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, RESTART_BACKOFF_MAX_S)

    async def _probe_rpc(self, proc: asyncio.subprocess.Process) -> None:
        deadline = time.monotonic() + RPC_READY_TIMEOUT_S
        while proc.returncode is None and time.monotonic() < deadline:
            try:
                # A bare connect and close: a TLS rpc-server refuses the missing handshake and goes
                # back to accept, so this only proves the port is bound.
                _, writer = await asyncio.wait_for(
                    asyncio.open_connection(*self._probe_target), 1.0
                )
            except (OSError, asyncio.TimeoutError):
                await asyncio.sleep(0.1)
                continue
            # Connect and close without a command: rpc-server reads EOF and goes back to accept.
            await close_writer(writer, abort = True)
            if self._rpc_ready is not None:
                self._rpc_ready.set()
            return

    async def _kill_rpc(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None or proc.returncode is not None:
            return
        proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), 5.0)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
        self._forget_rpc()

    def _native_client(self) -> bool:
        """In native mode the session bypasses this agent: whether its rpc-server holds a connection on
        the TLS port is the only sign of one (Studio's busy check reads it)."""
        pid = self.rpc_pid
        if self.mode != "native" or pid is None:
            return False
        try:
            import psutil

            proc = psutil.Process(pid)
            lister = getattr(proc, "net_connections", None) or proc.connections  # psutil < 6
            conns = lister(kind = "tcp")
        except Exception:
            return False
        return any(
            c.laddr and c.laddr.port == self.native_port and c.status == "ESTABLISHED"
            for c in conns
        )

    def status(self) -> Dict[str, Any]:
        return {
            "type": "status",
            "version": PROTOCOL_VERSION,
            "node_id": self.identity.node_id,
            "busy": self.busy or self._native_client(),
            "rpc_up": self.rpc_up,
            "devices": [],
            "counters": dict(self.counters),
            "native": {
                "available": self.native_available,
                "active": self.mode == "native",
                "port": self.native_port if self.native_available else 0,
            },
        }

    # Native mode -------------------------------------------------------------------------------

    async def _restart_rpc(self) -> bool:
        """Stop rpc-server so the supervisor starts it again with the current mode's argv; a live
        session through it ends here. True once the new one accepts."""
        if self.rpc_binary is None:
            return True
        if self._rpc_ready is not None:
            self._rpc_ready.clear()
        proc = self._proc
        if proc is not None and proc.returncode is None:
            self._planned_restart = True
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), 5.0)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
        return await self.wait_rpc_ready(RPC_READY_TIMEOUT_S)

    async def _enter_native(self, pid: tls.PeerIdentity) -> Dict[str, Any]:
        if not self.native_available:
            return {"type": "error", "error": "this worker's llama.cpp has no TLS transport"}
        assert self._native_lock is not None
        async with self._native_lock:
            if self.busy:
                return {"type": "busy"}
            path = native.allowlist_path(self.native_port)
            switch = self.mode != "native"
            same_master = (
                self._native_master is not None
                and self._native_master.fingerprint == pid.fingerprint
            )
            await asyncio.to_thread(native.write_allowlist, path, [pid.fingerprint])
            self._native_master = pid
            self.mode = "native"
            if switch or not same_master:
                # A different master's live session must not survive the hand-over.
                ok = await self._restart_rpc()
            else:
                ok = await self.wait_rpc_ready(RPC_READY_TIMEOUT_S)
            if not ok:
                return {"type": "error", "error": "ggml-rpc-server did not start with TLS"}
            if self._native_watch is None or self._native_watch.done():
                self._native_watch = asyncio.ensure_future(self._watch_native())
            return {
                "type": "native_ok",
                "port": self.native_port,
                "fingerprint": self.identity.fingerprint,
            }

    async def _leave_native(self) -> bool:
        """Back to loopback rpc-server behind this agent: empties the allowlist and restarts, which
        cuts a live native session."""
        assert self._native_lock is not None
        async with self._native_lock:
            if self.mode != "native":
                return True
            await asyncio.to_thread(
                native.write_allowlist, native.allowlist_path(self.native_port), []
            )
            self.mode = "loopback"
            self._native_master = None
            watch, self._native_watch = self._native_watch, None
            if watch is not None and watch is not asyncio.current_task():
                watch.cancel()
            return await self._restart_rpc()

    async def _watch_native(self) -> None:
        """The relay's 2 s revalidation, for a session the agent does not carry: `unsloth cluster leave`
        or a re-pair edits the registry, and the next check revokes the master by restarting rpc-server."""
        while self.mode == "native" and not self._stopping:
            await asyncio.sleep(REVALIDATE_S)
            pid = self._native_master
            if pid is None or self.mode != "native":
                return
            try:
                ok = bool(await asyncio.to_thread(self._authorize, pid))
            except Exception:
                ok = False
            if not ok:
                logger.warning(
                    "cluster agent: the master is no longer paired; closing its TLS session"
                )
                await self._leave_native()
                return

    async def _control_native(self, kind: str, pid: Optional[tls.PeerIdentity]) -> Dict[str, Any]:
        if kind == "native_rpc":
            if pid is None:
                return {"type": "error", "error": "no master identity"}
            return await self._enter_native(pid)
        await self._leave_native()
        return {"type": "ok"}

    def _control_native_threaded(
        self, kind: str, pid: Optional[tls.PeerIdentity]
    ) -> Dict[str, Any]:
        loop = self._loop
        if loop is None or loop.is_closed():
            return {"type": "error", "error": "agent is stopping"}
        return asyncio.run_coroutine_threadsafe(self._control_native(kind, pid), loop).result(
            RPC_READY_TIMEOUT_S + 15
        )

    def _leave_native_threaded(self) -> None:
        if self.mode != "native":
            return
        loop = self._loop
        if loop is not None and not loop.is_closed():
            asyncio.run_coroutine_threadsafe(self._leave_native(), loop).result(
                RPC_READY_TIMEOUT_S + 15
            )

    async def _revoke_when_unauthorized(
        self, pid: Any, master: asyncio.StreamWriter, rpc: asyncio.StreamWriter
    ) -> None:
        while True:
            await asyncio.sleep(REVALIDATE_S)
            try:
                # Reads the registry from disk: off the loop that carries the RPC bytes.
                ok = pid is not None and bool(await asyncio.to_thread(self._authorize, pid))
            except Exception:
                ok = False
            if not ok:
                logger.warning("cluster agent: the master is no longer paired; closing its session")
                master.transport.abort()
                rpc.transport.abort()
                return

    def _admit(self, writer: asyncio.StreamWriter) -> bool:
        peer = writer.get_extra_info("peername")
        return self._admit_peer(str(peer[0]) if peer else "", writer.get_extra_info("ssl_object"))

    def _admit_peer(self, source: str, ssl_object: Any) -> bool:
        if not self._source_allowed(source):
            logger.warning("cluster agent: refused %s (not a private address)", source)
            return False
        pid = tls.peer_identity(ssl_object)
        if pid is None or pid.role != "master":
            return False
        try:
            return bool(self._authorize(pid))
        except Exception:
            logger.exception("cluster agent: authorization check failed")
            return False

    # Threaded sessions -------------------------------------------------------------------------

    def _close_listener(self) -> None:
        listener, self._listener = self._listener, None
        if listener is not None:
            try:
                listener.close()
            except OSError:
                pass

    def _accept_loop(self) -> None:
        listener = self._listener
        while listener is not None and not self._stopping:
            try:
                conn, addr = listener.accept()
            except socket.timeout:
                listener = self._listener
                continue
            except OSError:
                if self._stopping or self._listener is None:
                    return
                time.sleep(0.05)  # EMFILE and the like: do not spin
                continue
            threading.Thread(
                target = self._serve_conn,
                args = (conn, addr),
                name = "unsloth-cluster-agent-conn",
                daemon = True,
            ).start()

    def _serve_conn(self, conn: socket.socket, addr: Any) -> None:
        self._raw.add(conn)
        sock: Any = conn
        try:
            conn.settimeout(HANDSHAKE_TIMEOUT_S)
            try:
                sock = self._ctx.wrap_socket(conn, server_side = True)
            except (OSError, ssl.SSLError, ValueError):
                return  # like asyncio: a failed handshake never reaches the handler
            self._raw.add(sock)
            self.stats["connections"] += 1
            if not self._admit_peer(str(addr[0]) if addr else "", sock):
                self.stats["rejected"] += 1
                return
            msg = framing.recv_frame(sock, timeout = HANDSHAKE_TIMEOUT_S)
            kind = msg.get("type")
            if kind == "ping":
                framing.send_frame(sock, {"type": "pong", "version": PROTOCOL_VERSION})
            elif kind == "status":
                framing.send_frame(sock, self.status())
            elif kind in ("native_rpc", "native_close"):
                framing.send_frame(
                    sock, self._control_native_threaded(kind, tls.peer_identity(sock))
                )
            elif kind == "open_rpc":
                # The relay needs the loopback rpc-server back (an older master, or native turned off).
                self._leave_native_threaded()
                self._open_rpc_threaded(sock)
            else:
                framing.send_frame(sock, {"type": "error", "error": "unknown request"})
        except (framing.FrameError, OSError, ssl.SSLError, ValueError):
            pass
        finally:
            self._raw.discard(conn)
            self._raw.discard(sock)
            for s in {sock, conn}:
                try:
                    s.close()
                except OSError:
                    pass

    def _claim_slot(self) -> bool:
        deadline = time.monotonic() + BUSY_GRACE_S
        while True:
            with self._slot_lock:
                if not self._active:
                    self._active += 1
                    return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.005)

    def _open_rpc_threaded(self, sock: Any) -> None:
        if not self._claim_slot():
            framing.send_frame(sock, {"type": "busy"})
            return
        released = False

        def release() -> None:
            nonlocal released
            with self._slot_lock:
                if not released:
                    released = True
                    self._active -= 1

        try:
            # Right after start (or a restart) rpc-server may not have bound yet: wait, do not fail the load.
            loop = self._loop
            if self.rpc_binary is not None and loop is not None and not loop.is_closed():
                try:
                    asyncio.run_coroutine_threadsafe(
                        self.wait_rpc_ready(RPC_CONNECT_TIMEOUT_S * 3), loop
                    ).result(RPC_CONNECT_TIMEOUT_S * 3 + 5)
                except Exception:
                    pass
            try:
                up = socket.create_connection(self._rpc_target, timeout = RPC_CONNECT_TIMEOUT_S)
            except OSError as exc:
                framing.send_frame(
                    sock, {"type": "error", "error": f"rpc-server unreachable: {exc}"}
                )
                return
            self.stats["rpc_streams"] += 1
            self.stats["threaded_streams"] = self.stats.get("threaded_streams", 0) + 1
            framing.send_frame(sock, {"type": "rpc_ok"})
            pid = tls.peer_identity(sock)
            session = SpliceSession()
            self._sessions.add(session)
            ended = threading.Event()

            def watch() -> None:
                while not ended.wait(REVALIDATE_S):
                    try:
                        ok = pid is not None and bool(self._authorize(pid))
                    except Exception:
                        ok = False
                    if not ok:
                        logger.warning(
                            "cluster agent: the master is no longer paired; closing its session"
                        )
                        session.abort()
                        return

            watcher = threading.Thread(
                target = watch, name = "unsloth-cluster-agent-revalidate", daemon = True
            )
            watcher.start()
            try:
                socket_splice(
                    sock,
                    up,
                    self.counters,
                    "from_master",
                    "to_master",
                    on_done = release,
                    session = session,
                )
            finally:
                ended.set()
                self._sessions.discard(session)
        finally:
            release()

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self._conns.add(writer)
        self.stats["connections"] += 1
        try:
            if not self._admit(writer):
                self.stats["rejected"] += 1
                return
            msg = await framing.read_frame(reader, timeout = HANDSHAKE_TIMEOUT_S)
            kind = msg.get("type")
            if kind == "ping":
                await framing.write_frame(writer, {"type": "pong", "version": PROTOCOL_VERSION})
            elif kind == "status":
                await framing.write_frame(writer, self.status())
            elif kind in ("native_rpc", "native_close"):
                pid = tls.peer_identity(writer.get_extra_info("ssl_object"))
                await framing.write_frame(writer, await self._control_native(kind, pid))
            elif kind == "open_rpc":
                await self._leave_native()
                await self._open_rpc(reader, writer)
            else:
                await framing.write_frame(writer, {"type": "error", "error": "unknown request"})
        except (framing.FrameError, OSError, ssl.SSLError, asyncio.TimeoutError):
            pass
        finally:
            self._conns.discard(writer)
            await close_writer(writer)

    async def _open_rpc(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        assert self._idle is not None
        if self._active:
            try:
                await asyncio.wait_for(self._idle.wait(), BUSY_GRACE_S)
            except asyncio.TimeoutError:
                pass
        if self._active:
            await framing.write_frame(writer, {"type": "busy"})
            await close_writer(writer)
            return
        self._active += 1
        self._idle.clear()
        released = False

        def release() -> None:
            nonlocal released
            if not released:
                released = True
                self._active -= 1
                if not self._active:
                    self._idle.set()

        try:
            # Right after start (or a restart) rpc-server may not have bound yet: wait, do not fail the load.
            await self.wait_rpc_ready(RPC_CONNECT_TIMEOUT_S * 3)
            try:
                up_reader, up_writer = await asyncio.wait_for(
                    asyncio.open_connection(*self._rpc_target, limit = STREAM_LIMIT),
                    RPC_CONNECT_TIMEOUT_S,
                )
            except (OSError, asyncio.TimeoutError) as exc:
                await framing.write_frame(
                    writer, {"type": "error", "error": f"rpc-server unreachable: {exc}"}
                )
                await close_writer(writer)
                return
            self.stats["rpc_streams"] += 1
            await framing.write_frame(writer, {"type": "rpc_ok"})
            pid = tls.peer_identity(writer.get_extra_info("ssl_object"))
            watch = asyncio.ensure_future(self._revoke_when_unauthorized(pid, writer, up_writer))
            try:
                await splice(
                    reader,
                    writer,
                    up_reader,
                    up_writer,
                    self.counters,
                    "from_master",
                    "to_master",
                    on_done = release,
                )
            finally:
                watch.cancel()
        finally:
            release()
