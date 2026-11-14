# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A stand-in for studio.cluster_link.tailnet with the same API: an address is a file under
$FAKE_TAILNET_DIR and a forward is a real loopback TCP proxy, so it works across processes."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import socket
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


class TailnetError(RuntimeError):
    pass


class TailnetUnavailable(TailnetError):
    pass


@dataclass
class PathInfo:
    direct: bool
    rtt_ms: Optional[float]
    via: str


def _dir() -> Path:
    d = Path(os.environ["FAKE_TAILNET_DIR"])
    d.mkdir(parents = True, exist_ok = True)
    return d


def available() -> bool:
    return bool(os.environ.get("FAKE_TAILNET_DIR"))


def helper_path():
    return None


def _pub(path) -> str:
    return "nodekey:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _new_key(path) -> str:
    path = Path(path)
    path.write_text(secrets.token_hex(16))
    os.chmod(path, 0o600)
    return _pub(path)


def new_server_key(path, *, fixed_region: bool = True) -> str:
    return _new_key(path)


def new_client_key(path) -> str:
    return _new_key(path)


def _log(kind: str, **fields) -> None:
    with open(_dir() / "events.jsonl", "a") as f:
        f.write(json.dumps({"kind": kind, **fields}) + "\n")


def events() -> List[dict]:
    try:
        lines = (_dir() / "events.jsonl").read_text().splitlines()
    except FileNotFoundError:
        return []
    return [json.loads(line) for line in lines]


def _state(address: str) -> Optional[dict]:
    try:
        return json.loads((_dir() / address).read_text())
    except (FileNotFoundError, ValueError):
        return None


class _Proxy:
    def __init__(self, address: str, remote_port: int, client_pub: str):
        self.address, self.remote_port, self.client_pub = address, remote_port, client_pub
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(16)
        self.port = self.sock.getsockname()[1]
        self.conns: List[socket.socket] = []
        threading.Thread(target = self._accept, daemon = True).start()

    def _accept(self) -> None:
        while True:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return
            state = _state(self.address)
            allow = state.get("allow") if state else []
            if state is None or (allow is not None and self.client_pub not in allow):
                conn.close()
                continue
            try:
                up = socket.create_connection(("127.0.0.1", self.remote_port), timeout = 5)
                up.settimeout(None)
            except OSError:
                conn.close()
                continue
            self.conns += [conn, up]
            for a, b in ((conn, up), (up, conn)):
                threading.Thread(target = self._pipe, args = (a, b), daemon = True).start()

    @staticmethod
    def _pipe(a: socket.socket, b: socket.socket) -> None:
        try:
            while True:
                data = a.recv(65536)
                if not data:
                    break
                b.sendall(data)
        except OSError:
            pass
        for s in (a, b):
            try:
                s.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    def close(self) -> None:
        for s in [self.sock] + self.conns:
            try:
                s.close()
            except OSError:
                pass


class Tailnet:
    def __init__(self, *, derp_map_url: Optional[str] = None):
        self._served: List[str] = []
        self._proxies: Dict[tuple, _Proxy] = {}
        self._anon = "nodekey:anon-" + secrets.token_hex(8)

    def start(self) -> None:
        if not available():
            raise TailnetUnavailable("no fake tailnet dir")

    def close(self) -> None:
        for address in self._served:
            try:
                (_dir() / address).unlink()
            except FileNotFoundError:
                pass
        for proxy in self._proxies.values():
            proxy.close()
        self._served, self._proxies = [], {}

    def serve(
        self,
        ports,
        *,
        key_path = None,
        allow = None,
    ) -> str:
        seed = _pub(key_path) if key_path else "nodekey:" + secrets.token_hex(32)
        address = "tc" + seed.split(":", 1)[1][:40]
        relayed = os.environ.get("FAKE_TAILNET_RELAYED") == "1"
        state = {"ports": [int(p) for p in ports], "allow": allow, "relayed": relayed}
        (_dir() / address).write_text(json.dumps(state))
        self._served.append(address)
        _log("serve", address = address, ports = state["ports"], allow = allow)
        return address

    def _update(self, change) -> None:
        for address in self._served:
            state = _state(address)
            if state is not None:
                state["allow"] = change(list(state["allow"] or []))
                (_dir() / address).write_text(json.dumps(state))

    def allow(self, pubkey: str) -> None:
        self._update(lambda a: a + [pubkey])
        _log("allow", pub = pubkey)

    def revoke(self, pubkey: str) -> None:
        self._update(lambda a: [k for k in a if k != pubkey])
        _log("revoke", pub = pubkey)

    def forward(
        self,
        address: str,
        remote_port: int,
        *,
        key_path = None,
    ) -> int:
        state = _state(address)
        if state is None or int(remote_port) not in state["ports"]:
            raise TailnetError("no such tailcat server")
        key = (address, int(remote_port))
        if key not in self._proxies:
            pub = _pub(key_path) if key_path else self._anon
            self._proxies[key] = _Proxy(address, int(remote_port), pub)
        _log("forward", address = address, port = int(remote_port))
        return self._proxies[key].port

    def unforward(
        self,
        address: str,
        remote_port: Optional[int] = None,
    ) -> None:
        for key in [k for k in self._proxies if k[0] == address and remote_port in (None, k[1])]:
            self._proxies.pop(key).close()
        _log("unforward", address = address)

    def probe(
        self,
        address: str,
        *,
        timeout_s: float = 10.0,
    ) -> PathInfo:
        state = _state(address) or {}
        if state.get("relayed"):
            return PathInfo(False, 40.0, "relay fake")
        return PathInfo(True, 0.2, "direct 127.0.0.1")
