# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tailcat tunnels through the ``unsloth-link`` helper. Addresses and keys are secrets: never log them."""

from __future__ import annotations

import itertools
import json
import os
import socket
import subprocess
import sys
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import paths

ENV_BIN = "UNSLOTH_LINK_BIN"
ENV_DERP_MAP = "UNSLOTH_CLUSTER_DERP_MAP"
ENV_ALLOW_RELAY = "UNSLOTH_CLUSTER_ALLOW_RELAY"
ENV_DISABLE = "UNSLOTH_CLUSTER_TAILNET"
BINARY = "unsloth-link.exe" if os.name == "nt" else "unsloth-link"
START_TIMEOUT_S = 15.0
CALL_TIMEOUT_S = 60.0


class TailnetError(RuntimeError):
    """The helper refused or failed a request."""


class TailnetUnavailable(TailnetError):
    """No usable helper binary on this machine."""


@dataclass
class PathInfo:
    direct: bool
    rtt_ms: Optional[float]
    via: str


def enabled() -> bool:
    return (os.environ.get(ENV_DISABLE) or "").strip() != "0"


def relay_allowed() -> bool:
    return (os.environ.get(ENV_ALLOW_RELAY) or "").strip() == "1"


def _installer() -> Any:
    try:
        from studio import install_unsloth_link
    except ImportError:
        import install_unsloth_link
    return install_unsloth_link


def helper_path() -> Optional[Path]:
    """``UNSLOTH_LINK_BIN``, else the verified install for the current pins. Never downloads."""
    override = (os.environ.get(ENV_BIN) or "").strip()
    if override:
        path = Path(override).expanduser().absolute()
        return path if path.is_file() else None
    try:
        return _installer().installed_binary()
    except Exception:
        return None


def install(log: Any = None) -> Path:
    """Download and verify the pinned helper if it is missing; raises TailnetUnavailable when it cannot."""
    found = helper_path()
    if found is not None:
        return found
    if (os.environ.get(ENV_BIN) or "").strip():
        raise TailnetUnavailable(f"{ENV_BIN} does not name a file")
    try:
        installer = _installer()
        return installer.ensure(log = log or (lambda _msg: None))
    except Exception as exc:
        raise TailnetUnavailable(str(exc)) from exc


def available() -> bool:
    return enabled() and helper_path() is not None


def _command() -> List[str]:
    if not enabled():
        raise TailnetUnavailable(f"tailcat tunnels are disabled ({ENV_DISABLE}=0)")
    path = helper_path()
    if path is None:
        raise TailnetUnavailable(f"{BINARY} is not installed")
    return [str(path)]


def _derp_map_url(explicit: Optional[str] = None) -> Optional[str]:
    return explicit or (os.environ.get(ENV_DERP_MAP) or "").strip() or None


def _genkey(kind: str, path: Path, extra: List[str]) -> str:
    paths.private_dir(path.parent)
    argv = [*_command(), "genkey", "--kind", kind, "--path", str(path), *extra]
    url = _derp_map_url()
    if url:
        argv += ["--derpmap-url", url]
    try:
        out = subprocess.run(argv, capture_output = True, text = True, timeout = 60)
    except (OSError, subprocess.SubprocessError) as exc:
        raise TailnetError(f"{BINARY} genkey failed: {exc}") from exc
    if out.returncode != 0:
        raise TailnetError((out.stderr or "").strip() or f"{BINARY} genkey exited {out.returncode}")
    return out.stdout.strip()


def new_server_key(path: Path, *, fixed_region: bool = True) -> str:
    """Write a 0600 server key; returns its public ``nodekey:...``."""
    return _genkey("server", Path(path), ["--fixed-region"] if fixed_region else [])


def new_client_key(path: Path) -> str:
    """Write a 0600 client key; returns the public key servers allowlist."""
    return _genkey("client", Path(path), [])


def _lifetime() -> Tuple[Dict[str, Any], Any]:
    """(Popen kwargs, spawn) tying the helper to this process: PDEATHSIG armed from a lifetime
    thread on Linux, the Job Object Studio installs on Windows; the helper itself also exits on
    stdin EOF and once reparented (``--parent-pid``)."""
    try:
        lifetime = sys.modules.get("utils.process_lifetime")  # Studio's own instance
        if lifetime is None:
            from studio.backend.utils import process_lifetime as lifetime
        return lifetime.child_popen_kwargs(), lifetime.spawn_on_lifetime_thread
    except Exception:
        return {}, lambda spawn: spawn()


class Tailnet:
    """One helper process; it dies with this process (see _lifetime) and exits when stdin closes."""

    def __init__(self, *, derp_map_url: Optional[str] = None):
        self._derp_map_url = _derp_map_url(derp_map_url)
        self._proc: Optional[subprocess.Popen] = None
        self._ids = itertools.count(1)
        self._lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._waiting: Dict[int, Tuple[threading.Event, List[Dict[str, Any]]]] = {}
        self._ready = threading.Event()
        self._dead: Optional[str] = None
        self._closed = False
        self._stderr: deque = deque(maxlen = 20)

    @property
    def dead(self) -> bool:
        return self._dead is not None

    def start(self) -> None:
        with self._lock:
            if self._closed:
                raise TailnetError(self._dead or f"{BINARY} was closed")
            if self._proc is not None:
                return
            argv = [*_command(), "stdio"]
            if self._derp_map_url:
                argv += ["--derpmap-url", self._derp_map_url]
            kwargs, spawn = _lifetime()
            if os.name == "nt":
                kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            else:
                argv += ["--parent-pid", str(os.getpid())]
            try:
                proc = spawn(
                    lambda: subprocess.Popen(
                        argv,
                        stdin = subprocess.PIPE,
                        stdout = subprocess.PIPE,
                        stderr = subprocess.PIPE,
                        **kwargs,
                    )
                )
            except OSError as exc:
                raise TailnetUnavailable(f"cannot start {BINARY}: {exc}") from exc
            self._proc = proc
        threading.Thread(target = self._read, args = (proc,), daemon = True, name = "tailnet-out").start()
        threading.Thread(
            target = self._drain_stderr, args = (proc,), daemon = True, name = "tailnet-err"
        ).start()
        if not self._ready.wait(START_TIMEOUT_S) or self._dead:
            self.close()
            raise TailnetError(self._dead or f"{BINARY} did not start")

    def _read(self, proc: subprocess.Popen) -> None:
        assert proc.stdout is not None
        for raw in proc.stdout:
            try:
                msg = json.loads(raw)
            except ValueError:
                continue
            if msg.get("event") == "ready":
                self._ready.set()
                continue
            slot = self._waiting.get(msg.get("id"))
            if slot is not None:
                slot[1].append(msg)
                slot[0].set()
        tail = "; ".join(self._stderr)
        self._dead = self._dead or f"{BINARY} exited" + (f": {tail}" if tail else "")
        self._ready.set()
        for event, _ in list(self._waiting.values()):
            event.set()
        _reap(proc)

    def _drain_stderr(self, proc: subprocess.Popen) -> None:
        assert proc.stderr is not None
        for raw in proc.stderr:
            self._stderr.append(raw.decode("utf-8", "replace").rstrip())

    def _call(
        self,
        op: str,
        timeout: float = CALL_TIMEOUT_S,
        **fields: Any,
    ) -> Dict[str, Any]:
        proc = self._proc
        if proc is None:
            self.start()
            proc = self._proc
        if self._dead or proc is None:
            raise TailnetError(self._dead or f"{BINARY} was closed")
        rid = next(self._ids)
        event, box = threading.Event(), []
        self._waiting[rid] = (event, box)
        line = json.dumps({"id": rid, "op": op, **fields}).encode("utf-8") + b"\n"
        try:
            with self._write_lock:
                assert proc.stdin is not None
                proc.stdin.write(line)
                proc.stdin.flush()
            if not event.wait(timeout):
                raise TailnetError(f"{BINARY} did not answer {op} in {timeout:.0f} s")
        except (OSError, ValueError) as exc:  # ValueError: stdin closed under us
            raise TailnetError(self._dead or f"{BINARY} is gone: {exc}") from exc
        finally:
            self._waiting.pop(rid, None)
        if not box:
            raise TailnetError(self._dead or f"{BINARY} exited")
        reply = box[0]
        if not reply.get("ok"):
            raise TailnetError(str(reply.get("error") or f"{op} failed"))
        return reply

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._dead = self._dead or f"{BINARY} was closed"
            proc, self._proc = self._proc, None
        if proc is None:
            return
        # A writer blocked on a full pipe holds stdin: close it on the side so the kill stays reachable.
        threading.Thread(target = _close_quietly, args = (proc.stdin,), daemon = True).start()
        try:
            proc.wait(timeout = 5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout = 5)

    def serve(
        self, ports: List[int], *, key_path: Optional[Path], allow: Optional[List[str]]
    ) -> str:
        """Serve local 127.0.0.1 ports through the tunnel; returns the secret address.

        ``key_path`` None is an ephemeral key; ``allow`` None admits anyone holding the address.
        A second call adds ports (and allowed keys) to the running server.
        """
        reply = self._call(
            "serve",
            ports = [int(p) for p in ports],
            key_path = str(key_path) if key_path else "",
            allow = list(allow or []),
            allow_all = allow is None,
        )
        return str(reply["address"])

    def allow(self, pubkey: str) -> None:
        self._call("allow", pubkey = pubkey)

    def revoke(self, pubkey: str) -> None:
        self._call("revoke", pubkey = pubkey)

    def forward(
        self,
        address: str,
        remote_port: int,
        *,
        key_path: Optional[Path] = None,
    ) -> int:
        reply = self._call(
            "forward",
            address = address,
            remote_port = int(remote_port),
            key_path = str(key_path) if key_path else "",
        )
        return int(reply["local_port"])

    def unforward(
        self,
        address: str,
        remote_port: Optional[int] = None,
    ) -> None:
        self._call("unforward", address = address, remote_port = int(remote_port or 0))

    def probe(
        self,
        address: str,
        *,
        timeout_s: float = 10.0,
    ) -> PathInfo:
        reply = self._call(
            "probe", timeout = timeout_s + 15, address = address, timeout_s = float(timeout_s)
        )
        rtt = reply.get("rtt_ms")
        return PathInfo(
            direct = bool(reply.get("direct")),
            rtt_ms = float(rtt) if rtt is not None else None,
            via = str(reply.get("via") or ""),
        )

    def __enter__(self) -> "Tailnet":
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def _close_quietly(stream: Any) -> None:
    try:
        if stream is not None:
            stream.close()
    except (OSError, ValueError):
        pass


def _reap(proc: subprocess.Popen) -> None:
    """The helper hung up: collect it and drop its pipes, so no zombie or descriptor is left."""
    try:
        proc.wait(timeout = 5)
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            proc.wait(timeout = 5)
        except subprocess.TimeoutExpired:
            pass
    for stream in (proc.stdout, proc.stdin):
        _close_quietly(stream)


class FakeTailnet:
    """In-process stand-in for tests: "addresses" name local TCP ports, forwards are plain local proxies."""

    _servers: Dict[str, "FakeTailnet"] = {}
    _counter = itertools.count(1)

    def __init__(
        self,
        *,
        derp_map_url: Optional[str] = None,
        direct: bool = True,
        rtt_ms: float = 1.0,
    ):
        self.direct = direct
        self.rtt_ms = rtt_ms
        self.address: Optional[str] = None
        self.ports: List[int] = []
        self.allowed: Optional[set] = None
        self.client_pub = f"nodekey:fake{next(self._counter):060x}"
        self._forwards: Dict[Tuple[str, int], socket.socket] = {}
        self._live: Dict[str, List[socket.socket]] = {}
        self._closed = False

    def start(self) -> None:
        pass

    def close(self) -> None:
        self._closed = True
        for sock in self._forwards.values():
            sock.close()
        self._forwards.clear()
        if self.address:
            FakeTailnet._servers.pop(self.address, None)

    def serve(
        self, ports: List[int], *, key_path: Optional[Path], allow: Optional[List[str]]
    ) -> str:
        if self.address is None:
            self.address = f"tcfake{next(self._counter):08d}"
            self.allowed = None if allow is None else set(allow)
            FakeTailnet._servers[self.address] = self
        elif allow and self.allowed is not None:
            self.allowed.update(allow)
        self.ports = sorted(set(self.ports) | {int(p) for p in ports})
        return self.address

    def allow(self, pubkey: str) -> None:
        if self.allowed is not None:
            self.allowed.add(pubkey)

    def revoke(self, pubkey: str) -> None:
        if self.allowed is None:
            self.allowed = set()
        self.allowed.discard(pubkey)
        for sock in self._live.pop(pubkey, []):
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            sock.close()

    def _server(self, address: str) -> "FakeTailnet":
        server = FakeTailnet._servers.get(address)
        if server is None or server._closed:
            raise TailnetError("no answer: unknown address")
        return server

    def forward(
        self,
        address: str,
        remote_port: int,
        *,
        key_path: Optional[Path] = None,
    ) -> int:
        server = self._server(address)
        if int(remote_port) not in server.ports:
            raise TailnetError("port not served")
        listener = self._forwards.get((address, int(remote_port)))
        if listener is None:
            listener = socket.create_server(("127.0.0.1", 0))
            self._forwards[(address, int(remote_port))] = listener
            threading.Thread(
                target = self._accept, args = (listener, server, int(remote_port)), daemon = True
            ).start()
        return listener.getsockname()[1]

    def _accept(self, listener: socket.socket, server: "FakeTailnet", port: int) -> None:
        while True:
            try:
                local, _ = listener.accept()
            except OSError:
                return
            if server.allowed is not None and self.client_pub not in server.allowed:
                local.close()
                continue
            try:
                remote = socket.create_connection(("127.0.0.1", port), timeout = 5)
            except OSError:
                local.close()
                continue
            server._live.setdefault(self.client_pub, []).extend([local, remote])
            for a, b in ((local, remote), (remote, local)):
                threading.Thread(target = _pipe, args = (a, b), daemon = True).start()

    def unforward(
        self,
        address: str,
        remote_port: Optional[int] = None,
    ) -> None:
        for key in [
            k
            for k in self._forwards
            if k[0] == address and (remote_port is None or k[1] == remote_port)
        ]:
            self._forwards.pop(key).close()

    def probe(
        self,
        address: str,
        *,
        timeout_s: float = 10.0,
    ) -> PathInfo:
        self._server(address)
        via = "direct 127.0.0.1:41641" if self.direct else "relay fake"
        return PathInfo(direct = self.direct, rtt_ms = self.rtt_ms, via = via)

    def __enter__(self) -> "FakeTailnet":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def _pipe(src: socket.socket, dst: socket.socket) -> None:
    try:
        while True:
            data = src.recv(65536)
            if not data:
                break
            dst.sendall(data)
    except OSError:
        pass
    finally:
        try:
            dst.shutdown(socket.SHUT_WR)
        except OSError:
            pass
