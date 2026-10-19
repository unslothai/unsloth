# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The worker agent: the only network-facing piece of a worker. It supervises a ``ggml-rpc-server``
bound to loopback and, for an authenticated master, splices one TLS connection to it at a time."""

from __future__ import annotations

import asyncio
import collections
import json
import logging
import os
import signal
import ssl
import sys
import time
from pathlib import Path
from typing import Any, Callable, Deque, Dict, Mapping, Optional, Sequence, Tuple

from . import AGENT_PORT_DEFAULT, PROTOCOL_VERSION, RPC_PORT_DEFAULT, SECURE_RPC_ENV, framing, tls
from .model import LocalIdentity
from .paths import state_dir
from .splice import STREAM_LIMIT, close_writer, is_private_address, splice

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
    ) -> None:
        self.identity = identity
        self._authorize = authorize
        self.host = host
        self.port = port
        self.rpc_binary = rpc_binary
        self.rpc_port = rpc_port
        self.rpc_args = [str(a) for a in rpc_args]
        self.cache_dir = cache_dir
        # The base env for rpc-server, e.g. Studio's library paths for its llama.cpp build.
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
        # Only once the port is ours: a failed bind must not leave an rpc-server running.
        if self.rpc_binary is not None:
            try:
                self._reap_stale_rpc()
            except RuntimeError:
                self._server.close()
                self._server = None
                raise
            self._supervisor = asyncio.ensure_future(self._supervise())

    async def serve_forever(self) -> None:
        if self._server is None:
            await self.start()
        assert self._server is not None
        try:
            await self._server.serve_forever()
        except asyncio.CancelledError:
            pass

    async def stop(self) -> None:
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
        if self._supervisor is not None:
            self._supervisor.cancel()
            await asyncio.gather(self._supervisor, return_exceptions = True)
            self._supervisor = None
        await self._kill_rpc()

    # rpc-server supervision

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
        return self._cache() / f"rpc-server-{self.rpc_port}.json"

    def _record_rpc(self, pid: int) -> None:
        try:
            import psutil

            created = psutil.Process(pid).create_time()
            me = psutil.Process(os.getpid())
            owner = {"agent_pid": me.pid, "agent_created": me.create_time()}
        except Exception:
            return
        try:
            self._record_path().write_text(json.dumps({"pid": pid, "created": created, **owner}))
        except OSError:
            pass

    def _live_owner(self, record: dict) -> Optional[int]:
        """The pid of another agent that is still running and owns this record's rpc-server."""
        try:
            import psutil

            agent = psutil.Process(int(record["agent_pid"]))
            if agent.pid == os.getpid():
                return None
            if abs(agent.create_time() - float(record["agent_created"])) > 0.01:
                return None
            return agent.pid
        except Exception:
            return None

    def _forget_rpc(self) -> None:
        try:
            self._record_path().unlink()
        except OSError:
            pass

    def _reap_stale_rpc(self) -> None:
        try:
            record = json.loads(self._record_path().read_text())
        except Exception:
            record = None
        owner = self._live_owner(record) if isinstance(record, dict) else None
        if owner is not None:
            # Not left behind: another running worker serves on this rpc port.
            raise RuntimeError(
                f"another `unsloth worker` (pid {owner}) already runs ggml-rpc-server on port "
                f"{self.rpc_port}; stop it or start this one with a different --rpc-port"
            )
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
        return [
            str(self.rpc_binary),
            "-H",
            "127.0.0.1",
            "-p",
            str(self.rpc_port),
            "-c",
            *self.rpc_args,
        ]

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
                self._proc = await asyncio.create_subprocess_exec(
                    *self._rpc_argv(),
                    stdin = asyncio.subprocess.DEVNULL,
                    stdout = asyncio.subprocess.PIPE,
                    stderr = asyncio.subprocess.STDOUT,
                    env = self._rpc_env(),
                    preexec_fn = _die_with_agent_hook(),
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
                _, writer = await asyncio.wait_for(asyncio.open_connection(*self._rpc_target), 1.0)
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

    # connections

    def status(self) -> Dict[str, Any]:
        return {
            "type": "status",
            "version": PROTOCOL_VERSION,
            "node_id": self.identity.node_id,
            "busy": self.busy,
            "rpc_up": self.rpc_up,
            "devices": [],
            "counters": dict(self.counters),
        }

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
        source = str(peer[0]) if peer else ""
        if not self._source_allowed(source):
            logger.warning("cluster agent: refused %s (not a private address)", source)
            return False
        pid = tls.peer_identity(writer.get_extra_info("ssl_object"))
        if pid is None or pid.role != "master":
            return False
        try:
            return bool(self._authorize(pid))
        except Exception:
            logger.exception("cluster agent: authorization check failed")
            return False

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
            elif kind == "open_rpc":
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
