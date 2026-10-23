# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The cluster link data path: worker agent + master relay over mutual TLS, against a fake rpc-server,
plus the real CPU ggml-rpc-server / llama-server when a build is available."""

from __future__ import annotations

import asyncio
import contextlib
import builtins
import dataclasses
import io
import json
import os
import sys
import random
import signal
import socket
import struct
import subprocess
import textwrap
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
from typing import Any, Callable

import pytest

pytest.importorskip("cryptography")

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec

from studio.cluster_link import agent as agent_mod
from studio.cluster_link import identity, native, tls
from studio.cluster_link import relay as relay_mod
from studio.cluster_link import splice as splice_mod
from studio.cluster_link.agent import Agent
from studio.cluster_link.model import LocalIdentity
from studio.cluster_link.paths import run_dir
from studio.cluster_link.relay import RelayManager

MIB = 1024 * 1024
_REPO = str(Path(__file__).resolve().parents[2])
_LINUX = sys.platform.startswith("linux")
_SKIP_NT = pytest.mark.skipif(os.name == "nt", reason = "the fake rpc-server is a shebang script")
_THREADED_ENV = "UNSLOTH_CLUSTER_LINK_THREADED_SPLICE"


class Pki:
    """A throwaway cluster root and leaves, made by the real CA code under `directory`."""

    def __init__(self, directory: Path):
        self.dir = directory
        with mock.patch.object(identity, "_ca_dir", lambda: directory):
            self.key, self.cert = identity._create_ca(directory.name)
        self.rootfile = str(directory / "root.pem")

    def leaf(self, node_id: str, role: str) -> LocalIdentity:
        key = ec.generate_private_key(ec.SECP256R1())
        cert = identity._leaf(key.public_key(), self.key, self.cert, node_id, role, 30)
        certfile, keyfile = self.dir / f"{node_id}-{role}.pem", self.dir / f"{node_id}-{role}.key"
        certfile.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
        keyfile.write_bytes(identity._key_pem(key))
        return LocalIdentity(
            node_id = node_id,
            role = role,
            cluster_id = "test-cluster",
            certfile = str(certfile),
            keyfile = str(keyfile),
            rootfile = self.rootfile,
            fingerprint = tls.fingerprint(cert.public_bytes(serialization.Encoding.DER)),
        )


class LoopThread:
    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target = self.loop.run_forever, daemon = True)
        self.thread.start()

    def run(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(30)

    def stop(self):
        async def cancel_rest():
            rest = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
            for task in rest:
                task.cancel()
            await asyncio.gather(*rest, return_exceptions = True)

        self.run(cancel_rest())
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(5)
        self.loop.close()


class FakeRpc:
    """Stands in for ggml-rpc-server: echoes, floods, or drops the connection after N bytes."""

    def __init__(self, mode: str, drop_after: int):
        self.mode, self.drop_after = mode, drop_after
        self.connections = self.open = self.received = self.sent = self.port = 0
        self.seen = bytearray()  # first bytes of every connection, to spot a forwarded payload
        self.server = None

    async def start(self):
        self.server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        self.port = self.server.sockets[0].getsockname()[1]

    async def _handle(self, reader, writer):
        self.connections += 1
        self.open += 1
        try:
            if self.mode == "flood":
                await reader.read(1)
                chunk = b"\x5a" * (256 * 1024)
                while True:
                    writer.write(chunk)
                    await writer.drain()
                    self.sent += len(chunk)
            while data := await reader.read(256 * 1024):
                self.received += len(data)
                if len(self.seen) < 1 << 20:
                    self.seen += data[:4096]
                if self.mode == "drop" and self.received >= self.drop_after:
                    writer.transport.abort()
                    return
                writer.write(data)
                await writer.drain()
                self.sent += len(data)
        except (ConnectionError, OSError):
            pass
        finally:
            self.open -= 1
            writer.close()


def _wait(predicate: Callable[[], bool], timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


class Link:
    """A worker agent and master relay on loopback. Link options: target_mode, drop_after,
    agent_authorize, relay_authorize, worker_role, master_role, master_other_root, relay_node_id;
    the rest go to Agent."""

    def __init__(self, tmp: Path, **kw: Any):
        pop = kw.pop
        self.pki = Pki(tmp / "pki")
        self.worker = self.pki.leaf("w1", pop("worker_role", "worker"))
        issuer = Pki(tmp / "other") if pop("master_other_root", False) else self.pki
        master = issuer.leaf("m1", pop("master_role", "master"))
        self.master = dataclasses.replace(master, rootfile = self.pki.rootfile)
        mfp, wfp = self.master.fingerprint, self.worker.fingerprint
        agent_auth = pop(
            "agent_authorize", lambda pid: pid.node_id == "m1" and pid.fingerprint == mfp
        )
        relay_auth = pop(
            "relay_authorize", lambda pid: pid.node_id == "w1" and pid.fingerprint == wfp
        )
        relay_node_id = pop("relay_node_id", "w1")
        self.loop = LoopThread()
        self.target = FakeRpc(pop("target_mode", "echo"), pop("drop_after", 0))
        if kw.setdefault("rpc_binary", None) is None:
            self.loop.run(self.target.start())
            kw["rpc_target"] = ("127.0.0.1", self.target.port)
        kw = {"rpc_port": 0, "source_filter": None, "cache_dir": str(tmp / "cache"), **kw}
        self.agent = Agent(self.worker, agent_auth, host = "127.0.0.1", port = 0, **kw)
        self.loop.run(self.agent.start())
        self.relay = RelayManager(self.master, relay_auth)
        self.endpoint = self.relay.ensure(relay_node_id, "127.0.0.1", self.agent.port)

    def connect(self, data: bytes = b"") -> socket.socket:
        host, port = self.endpoint.rsplit(":", 1)
        sock = socket.create_connection((host, int(port)), timeout = 10)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.sendall(data)
        return sock

    def ping(self, data: bytes = b"ping") -> socket.socket:
        sock = self.connect(data)
        assert _recv_exact(sock, len(data)) == data
        return sock

    def close(self):
        self.relay.stop()
        self.loop.run(self.agent.stop())
        if self.target.server is not None:
            self.loop.loop.call_soon_threadsafe(self.target.server.close)
        self.loop.stop()


@pytest.fixture(autouse = True)
def _own_run_dir(tmp_path, monkeypatch):
    # rpc port ownership records are machine-wide: keep each test (and its subprocesses) to its own.
    monkeypatch.setenv("UNSLOTH_CLUSTER_RUN_DIR", str(tmp_path / "run"))


@pytest.fixture
def make_link(tmp_path):
    links = []

    def factory(**kwargs) -> Link:
        links.append(Link(tmp_path / f"l{len(links)}", **kwargs))
        return links[-1]

    yield factory
    for link in links:
        link.close()


@pytest.fixture
def procs():
    """Popen objects and bare pids, killed at teardown."""
    started = []
    yield started
    for p in started:
        with contextlib.suppress(ProcessLookupError):
            os.kill(p, signal.SIGKILL) if isinstance(p, int) else (p.kill(), p.wait(10))


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n and (chunk := sock.recv(min(1 << 20, n - len(buf)))):
        buf += chunk
    return bytes(buf)


def _closed(sock: socket.socket, timeout: float = 5.0) -> bool:
    sock.settimeout(timeout)
    try:
        return sock.recv(1) == b""
    except (ConnectionResetError, BrokenPipeError):
        return True
    except socket.timeout:
        return False


def _rst_close(sock: socket.socket) -> None:
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
    sock.close()


def _echo(link: Link, payload: bytes, fragment: bool) -> bytes:
    sock = link.connect()
    rng = random.Random(len(payload))

    def send():
        i = 0
        while i < len(payload):
            n = rng.randint(1, 9000) if fragment else len(payload)
            sock.sendall(payload[i : i + n])
            i += n
            if fragment and rng.random() < 0.01:
                time.sleep(0.001)

    sender = threading.Thread(target = send)
    sender.start()
    got = _recv_exact(sock, len(payload))
    sender.join()
    sock.close()
    return got


def _idle(link: Link) -> bool:
    return not link.agent.busy and not link.relay._relays["w1"].active


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _gone(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


@pytest.mark.parametrize(
    "mode, size, fragment",
    [("async", n, False) for n in (1, 65537, MIB + 3, 64 * MIB)]
    + [("async", 3 * MIB + 17, True), ("threaded", MIB + 5, True), ("threaded", 8 * MIB + 3, True)],
)
def test_round_trip_is_byte_exact_on_the_path_the_kill_switch_selects(
    make_link, monkeypatch, mode, size, fragment
):
    # Fragmented: concurrent upload + download in random pieces, never stalls a direction.
    monkeypatch.setenv(_THREADED_ENV, str(int(mode == "threaded")))
    link = make_link()
    payload = os.urandom(size)
    assert _echo(link, payload, fragment) == payload
    assert link.agent.stats["rpc_streams"] == 1
    want = 1 if mode == "threaded" else 0
    assert link.agent.threaded is (want == 1) and link.relay.threaded is (want == 1)
    assert link.agent.stats.get("threaded_streams", 0) == want
    counters = link.relay.counters("w1")
    assert counters.get("threaded_streams", 0) == want
    assert counters["to_worker"] == size and counters["from_worker"] == size


def test_a_slow_reader_backpressures_the_rpc_server(make_link):
    link = make_link(target_mode = "flood")
    sock = link.connect(b"g")
    time.sleep(1.5)
    sent = link.target.sent
    assert 0 < sent < 64 * MIB, f"{sent / MIB:.1f} MiB queued for a reader that read nothing"
    sock.close()


def test_a_port_check_never_reaches_the_worker(make_link):
    link = make_link()
    before = link.agent.stats["connections"]
    for _ in range(3):
        link.connect().close()
    time.sleep(0.3)
    assert link.agent.stats["connections"] == before
    assert link.target.connections == 0


def test_status_reports_busy_and_a_second_client_is_refused_at_once(make_link):
    link = make_link()
    status = link.relay.status("w1")
    assert status["reachable"] and status["type"] == "status"
    assert status["rpc_up"] is True and status["busy"] is False
    assert status["node_id"] == "w1"
    first = link.ping(b"x")
    assert link.relay.status("w1")["busy"] is True
    assert link.relay.busy("w1") is True
    second = link.connect()
    started = time.monotonic()
    assert _closed(second, timeout = 2.0)
    assert time.monotonic() - started < 1.0
    first.sendall(b"y")
    assert _recv_exact(first, 1) == b"y"
    first.close()
    assert _wait(lambda: not link.agent.busy)
    link.relay.ensure("w1", "127.0.0.1", 1)
    status = link.relay.status("w1", timeout = 1.0)
    assert status["reachable"] is False and status["type"] == "error"


@pytest.mark.parametrize("mode", ["async", "threaded"])
def test_the_relay_refuses_a_second_client_without_dialing_the_worker(make_link, monkeypatch, mode):
    monkeypatch.setenv(_THREADED_ENV, str(int(mode == "threaded")))
    link = make_link()
    relay = link.relay._relays["w1"]
    early = link.connect()  # admitted while idle, sends its first byte only once a session is live
    assert _wait(lambda: len(relay.raw if mode == "threaded" else relay.conns) == 1)
    first = link.ping(b"live")
    dialed = link.agent.stats["connections"]
    silent = link.connect()
    assert _closed(silent, 3.0)  # refused at the door, not after the first-byte timeout
    early.sendall(b"x")
    assert _closed(early)  # refused at the claim
    assert not link.relay._claim(relay)
    assert link.agent.stats["connections"] == dialed and link.target.connections == 1
    for sock in (first, silent, early):
        sock.close()


@pytest.mark.parametrize("mode", ["async", "threaded"])
def test_the_agent_refuses_a_second_relay_session_while_one_is_live(make_link, monkeypatch, mode):
    monkeypatch.setenv(_THREADED_ENV, str(int(mode == "threaded")))
    link = make_link()
    first = link.ping(b"live")
    other = RelayManager(link.master, lambda pid: True)  # a second relay to the same worker
    try:
        host, port = other.ensure("w1", "127.0.0.1", link.agent.port).rsplit(":", 1)
        second = socket.create_connection((host, int(port)), timeout = 10)
        second.sendall(b"x")
        assert _closed(second)
        assert link.target.connections == 1 and link.target.open == 1
        second.close()
    finally:
        other.stop()
    first.close()


def test_back_to_back_sessions_are_not_refused_as_busy(make_link):
    # The busy slot frees when the splice ends, so an immediate reconnect is served, not refused.
    link = make_link()
    for i in range(30):
        link.ping(b"s%d" % i).close()
    assert link.agent.stats["rpc_streams"] == 30
    assert link.target.connections == 30


def test_a_client_reconnecting_while_its_last_session_drains_is_served(make_link):
    # A teardown with MBs in flight must not hold the slot past BUSY_GRACE_S or leak bytes onward.
    link = make_link()
    _rst_close(link.connect(os.urandom(8 * MIB)))
    payload = os.urandom(4096)
    second = link.connect(payload)
    second.settimeout(10)
    assert _recv_exact(second, len(payload)) == payload
    second.close()


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param(dict(agent_authorize = lambda pid: False), id = "master_not_paired_by_worker"),
        pytest.param(dict(master_role = "worker"), id = "worker_cert_cannot_act_as_master"),
        pytest.param(dict(worker_role = "master"), id = "master_cert_cannot_act_as_worker"),
        pytest.param(dict(relay_node_id = "w2"), id = "relay_refuses_another_node_id"),
        pytest.param(dict(relay_authorize = lambda pid: False), id = "relay_unauthorized_worker"),
        pytest.param(dict(master_other_root = True), id = "cert_from_another_cluster_root"),
        pytest.param(dict(source_filter = lambda host: False), id = "non_private_source"),
    ],
)
def test_an_unauthorized_peer_is_refused(make_link, kwargs):
    link = make_link(**kwargs)
    # A random marker, not a connection count: a stray client of an earlier test can land on the reused port.
    marker = b"\x0e" + os.urandom(15)
    assert _closed(link.connect(marker))
    time.sleep(0.2)
    assert marker not in link.target.seen
    if "source_filter" in kwargs:
        assert link.agent.stats["rejected"] >= 1


@pytest.mark.parametrize("mode", ["async", "threaded"])
@pytest.mark.parametrize("side", ["relay", "agent"])
def test_revocation_cuts_a_live_session(make_link, monkeypatch, side, mode):
    # `cluster remove` (master) / `cluster leave` (worker) only edit the registry: the open session
    # must still end promptly.
    monkeypatch.setattr(relay_mod, "REVALIDATE_S", 0.2)
    monkeypatch.setattr(agent_mod, "REVALIDATE_S", 0.2)
    monkeypatch.setenv(_THREADED_ENV, str(int(mode == "threaded")))
    trusted = {"ok": True}
    if side == "relay":
        link = make_link(relay_authorize = lambda pid: trusted["ok"] and pid.node_id == "w1")
    else:
        link = make_link(agent_authorize = lambda pid: trusted["ok"] and pid.node_id == "m1")
    sock = link.ping()
    assert link.agent.stats.get("threaded_streams", 0) == (1 if mode == "threaded" else 0)
    trusted["ok"] = False
    assert _closed(sock, 5.0), f"the {side} kept a revoked session open"
    if side == "relay":
        last_error = link.relay.status("w1").get("relay", {}).get("last_error", "")
        assert "no longer paired" in last_error
    assert _wait(lambda: link.target.open == 0)
    assert _wait(lambda: _idle(link))
    assert _wait(lambda: not _session_threads())


def test_private_address_filter():
    ok = ["127.0.0.1", "10.1.2.3", "172.16.0.9", "192.168.1.5", "169.254.0.1", "::1"]
    for host in ok + ["fe80::1%eth0", "fd00::5", "::ffff:192.168.1.1"]:
        assert splice_mod.is_private_address(host), host
    for bad in ("8.8.8.8", "1.1.1.1", "2606:4700::1111", "::ffff:8.8.8.8", "not-an-ip", ""):
        assert not splice_mod.is_private_address(bad), bad


_BIN_DIR = Path(os.environ.get("UNSLOTH_TEST_LLAMA_BIN_DIR") or "/nonexistent")


def _listening(port: int) -> bool:
    try:
        socket.create_connection(("127.0.0.1", port), timeout = 1).close()
        return True
    except OSError:
        return False


@pytest.mark.skipif(
    not ((_BIN_DIR / "ggml-rpc-server").is_file() and (_BIN_DIR / "llama-server").is_file()),
    reason = "no CPU llama.cpp build (set UNSLOTH_TEST_LLAMA_BIN_DIR)",
)
def test_real_rpc_server_lists_its_device_through_the_relay(make_link):
    rpc_port = _free_port()
    link = make_link(rpc_binary = str(_BIN_DIR / "ggml-rpc-server"), rpc_port = rpc_port)
    assert _wait(lambda: link.agent.rpc_pid is not None, 15)
    assert _wait(lambda: _listening(rpc_port), 15), list(link.agent.tail)
    out = subprocess.run(
        [str(_BIN_DIR / "llama-server"), "--rpc", link.endpoint, "--list-devices"],
        capture_output = True,
        text = True,
        encoding = "utf-8",
        errors = "replace",
        timeout = 120,
        env = dict(os.environ, LD_LIBRARY_PATH = str(_BIN_DIR), GGML_RPC_NO_RDMA = "1"),
    )
    text = out.stdout + out.stderr
    assert "RPC" in text, text[-2000:]
    assert link.agent.stats["rpc_streams"] >= 1
    assert link.relay.counters("w1").get("from_worker", 0) > 0
    pid = link.agent.rpc_pid
    os.kill(pid, 9)
    assert _wait(lambda: link.agent.rpc_pid not in (None, pid), 20)
    assert link.agent.stats["rpc_restarts"] >= 1


AGENT_SCRIPT = """
import asyncio, json, os, sys, threading, time
sys.path.insert(0, {repo!r})
import studio.cluster_link.agent as A
{prelude}
me = A.LocalIdentity(**json.loads(sys.argv[1]))
agent = A.Agent(me, lambda pid: True, host = "127.0.0.1", port = 0, rpc_binary = sys.argv[2],
                rpc_port = 1, cache_dir = (sys.argv[3:] or [None])[0])
async def main():
    await agent.start()
{body}
asyncio.run(main())
"""
_PRINT_PID = """
    while agent.rpc_pid is None:
        await asyncio.sleep(0.05)
    print(agent.rpc_pid, flush = True)
    await asyncio.sleep(600)
"""
_NO_PDEATHSIG = "A._die_with_agent_hook = lambda: None  # as on macOS / Windows"


def _sleeper(tmp: Path, secs: int = 600) -> Path:
    fake = tmp / "fake-rpc-server"
    fake.write_text(f"#!{sys.executable}\nimport time\ntime.sleep({secs})\n", encoding = "utf-8")
    fake.chmod(0o755)
    return fake


def _agent_argv(tmp: Path, prelude: str) -> list:
    """`python -c <agent> <identity json> <fake rpc-server>`; append a cache dir as argv[3]."""
    me = Pki(tmp / "pki").leaf("w1", "worker")
    script = AGENT_SCRIPT.format(repo = _REPO, prelude = prelude, body = _PRINT_PID)
    return [sys.executable, "-c", script, json.dumps(dataclasses.asdict(me)), str(_sleeper(tmp))]


def _start(procs: list, argv: list, **kw) -> subprocess.Popen:
    procs.append(subprocess.Popen(argv, stdout = subprocess.PIPE, text = True, **kw))
    return procs[-1]


def _agent(me: LocalIdentity, **kw) -> Agent:
    return Agent(me, lambda pid: True, host = "127.0.0.1", **{"port": 0, "rpc_port": 1, **kw})


@pytest.mark.skipif(not _LINUX, reason = "PR_SET_PDEATHSIG is Linux only")
@pytest.mark.parametrize("threads", [False, True], ids = ["pdeathsig", "threaded_agent_setpriv"])
def test_a_killed_agent_takes_its_rpc_server_with_it(tmp_path, procs, threads):
    # preexec_fn is unsafe once threads exist: rpc-server must still be tied to the agent (setpriv).
    if threads and agent_mod._setpriv() is None:
        pytest.skip("needs Linux with util-linux setpriv --pdeathsig")
    prelude = "threading.Thread(target = time.sleep, args = (600,), daemon = True).start()"
    agent = _start(procs, _agent_argv(tmp_path, prelude if threads else ""))
    child = int(agent.stdout.readline())
    procs.append(child)
    os.kill(child, 0)
    agent.send_signal(signal.SIGKILL)
    agent.wait(10)
    assert _wait(lambda: _gone(child), 10.0), f"rpc-server {child} outlived its agent"


@pytest.mark.skipif(not _LINUX, reason = "PR_SET_PDEATHSIG is Linux only")
def test_an_agent_running_as_pid_1_keeps_its_rpc_server(tmp_path):
    # A container with `unsloth worker` as entrypoint: the agent is pid 1, and its rpc-server's parent.
    body = """
    await asyncio.sleep(2.0)
    # The fake never binds a port, so "running" is the live pid, not rpc_up (which needs the port).
    print(json.dumps({"alive": agent.rpc_pid is not None, "restarts": agent.stats["rpc_restarts"]}))
    await agent.stop()
"""
    argv = _agent_argv(tmp_path, "os.getpid = lambda: 1\nos.getppid = lambda: 1")
    argv[2] = argv[2].replace(_PRINT_PID, body)
    out = subprocess.run(argv, capture_output = True, text = True, timeout = 60)
    lines = out.stdout.strip().splitlines()
    assert lines, out.stderr[-2000:]
    assert json.loads(lines[-1]) == {"alive": True, "restarts": 0}


@_SKIP_NT
def test_the_next_agent_stops_a_left_behind_rpc_server_but_never_a_live_ones(tmp_path, procs):
    psutil = pytest.importorskip("psutil")
    argv = _agent_argv(tmp_path, _NO_PDEATHSIG) + [str(tmp_path / "cache")]
    first = _start(procs, argv)
    orphan = int(first.stdout.readline())
    procs.append(orphan)
    first.send_signal(signal.SIGKILL)
    first.wait(10)
    assert psutil.pid_exists(orphan)  # nothing took it down: the case this guards
    fresh = int(_start(procs, argv).stdout.readline())
    procs.append(fresh)
    assert fresh != orphan
    assert _wait(
        lambda: not psutil.pid_exists(orphan) or psutil.Process(orphan).status() == "zombie", 10.0
    )
    # A second agent on the port of a live one (its own --cache-dir: the ownership record must not
    # live in either cache) is refused and leaves that server running.
    second = subprocess.run(
        argv[:-1] + [str(tmp_path / "cache-2")], capture_output = True, text = True, timeout = 30
    )
    assert second.returncode != 0 and "already runs ggml-rpc-server" in second.stderr
    assert psutil.Process(fresh).is_running() and psutil.Process(fresh).status() != "zombie"


@_SKIP_NT
def test_two_workers_starting_at_once_on_one_rpc_port_one_wins(tmp_path, procs):
    # Both could pass the "is the port owned?" check: the per-port lock lets exactly one start.
    argv = _agent_argv(tmp_path, _NO_PDEATHSIG)
    both = [_start(procs, argv + [str(tmp_path / f"c{i}")], stderr = subprocess.PIPE) for i in "01"]
    time.sleep(3)
    alive = [p for p in both if p.poll() is None]
    dead = [p for p in both if p.poll() is not None]
    assert len(alive) == 1 and len(dead) == 1, [p.poll() for p in both]
    assert "already runs ggml-rpc-server" in dead[0].stderr.read()
    procs.append(int(alive[0].stdout.readline()))


def test_the_rpc_port_lock_admits_one_holder_at_a_time():
    # The lock itself, no child process: runs natively on every OS (flock / msvcrt).
    first = agent_mod._lock_rpc_port(50999)
    try:
        with pytest.raises(RuntimeError, match = "already runs ggml-rpc-server"):
            agent_mod._lock_rpc_port(50999)
        agent_mod._unlock(agent_mod._lock_rpc_port(50998))  # another port is independent
    finally:
        agent_mod._unlock(first)
    agent_mod._unlock(agent_mod._lock_rpc_port(50999))  # free again once released


@_SKIP_NT
def test_an_rpc_server_left_on_another_port_is_reaped_but_never_a_foreign_process(tmp_path, procs):
    # A killed agent (no PDEATHSIG) left its rpc-server; the next agent starts on another --rpc-port.
    psutil = pytest.importorskip("psutil")
    fake = str(_sleeper(tmp_path, 60))

    def spawn(port, *argv):
        proc = subprocess.Popen(list(argv))
        procs.append(proc)

        def started():
            try:
                return len(psutil.Process(proc.pid).cmdline()) >= 2
            except psutil.Error:
                return False

        _wait(started)
        record = {"pid": proc.pid, "created": psutil.Process(proc.pid).create_time()}
        (run_dir() / f"rpc-server-{port}.json").write_text(json.dumps(record), encoding = "utf-8")
        return proc

    orphan = spawn(50101, fake)  # its agent is gone: the port lock is free
    held = spawn(50102, fake)  # a live agent still holds this port
    foreign = spawn(50103, sys.executable, "-c", "import time; time.sleep(60)")  # not rpc-server
    live = agent_mod._lock_rpc_port(50102)
    try:
        me = SimpleNamespace(rpc_port = 50200)
        me._record_path = lambda: run_dir() / "rpc-server-50200.json"
        Agent._reap_stale_rpc(me)
    finally:
        agent_mod._unlock(live)
    orphan.wait(5)
    assert orphan.returncode is not None
    assert held.poll() is None and foreign.poll() is None
    assert not (run_dir() / "rpc-server-50101.json").exists()
    assert (run_dir() / "rpc-server-50102.json").exists()


@_SKIP_NT
def test_a_failed_bind_starts_no_rpc_server(tmp_path):
    taken = socket.socket()
    taken.bind(("127.0.0.1", 0))
    taken.listen(1)
    marker = tmp_path / "started"
    fake = tmp_path / "fake-rpc-server"
    fake.write_text(f"#!/bin/sh\ntouch {marker}\nexec sleep 600\n")
    fake.chmod(0o755)
    me = Pki(tmp_path / "pki").leaf("w1", "worker")
    agent = _agent(me, port = taken.getsockname()[1], rpc_binary = str(fake))

    async def main():
        with pytest.raises(OSError):
            await agent.start()
        await asyncio.sleep(0.5)
        assert agent._supervisor is None and agent.rpc_pid is None

    try:
        asyncio.run(main())
    finally:
        taken.close()
    assert not marker.exists()


@_SKIP_NT
def test_rpc_up_waits_for_the_port_not_just_the_process(tmp_path):
    # Reporting a process that has not bound yet as up made a load right after `unsloth worker` start
    # fail on that worker.
    port = _free_port()
    slow = tmp_path / "slow-rpc-server"
    slow.write_text(
        textwrap.dedent(f"""\
        #!{sys.executable}
        import socket, time
        time.sleep(1.5)
        s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", {port})); s.listen(4)
        while True:
            c, _ = s.accept(); c.close()
        """)
    )
    slow.chmod(0o755)
    me = Pki(tmp_path / "pki").leaf("w1", "worker")
    agent = _agent(me, rpc_binary = str(slow), rpc_port = port, cache_dir = str(tmp_path / "c"))

    async def main():
        await agent.start()
        try:
            await asyncio.sleep(0.5)
            early = (agent.rpc_pid is not None, agent.rpc_up, agent.status()["rpc_up"])
            ready = await agent.wait_rpc_ready(10.0)
            return early, ready, agent.status()["rpc_up"]
        finally:
            await agent.stop()

    early, ready, later = asyncio.run(main())
    assert early == (True, False, False)
    assert ready and later


def test_stream_sockets_get_nodelay_and_keepalive():
    async def main():
        server = await asyncio.start_server(lambda r, w: w.close(), "127.0.0.1", 0)
        _, writer = await asyncio.open_connection(*server.sockets[0].getsockname()[:2])
        splice_mod.tune_socket(writer)
        sock = writer.get_extra_info("socket")
        seen = {
            "keepalive": sock.getsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE),
            "nodelay": sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY),
        }
        if hasattr(socket, "TCP_KEEPIDLE"):
            seen["idle"] = sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE)
        writer.close()
        server.close()
        return seen

    seen = asyncio.run(main())
    assert seen["keepalive"] and seen["nodelay"]
    if "idle" in seen:
        assert seen["idle"] == splice_mod.KEEPALIVE_IDLE_S


def test_a_tls_record_drain_stops_at_a_would_block_or_eof_and_keeps_the_bytes_in_hand():
    import ssl

    script = [b"x" * 100, ssl.SSLWantReadError(), b"y" * 50, b"", b""]

    class Src:  # a TLS socket with buffered records
        pending = staticmethod(lambda: 1)

        def recv_into(self, view):
            step = script.pop(0)
            if isinstance(step, Exception):
                raise step
            view[: len(step)] = step
            return len(step)

    src, out, counters = Src(), bytearray(), {}
    dst = SimpleNamespace(sendall = out.extend)
    end = splice_mod._End()
    locks = {id(src): threading.Lock()}
    splice_mod._pump(src, dst, locks, counters, "k", None, splice_mod.SpliceSession(), end)
    assert bytes(out) == b"x" * 100 + b"y" * 50 and counters["k"] == 150
    assert end.first is True and not script


def test_splice_forwards_bytes_buffered_before_it_started_and_closes_promptly():
    async def main():
        got = bytearray()
        upstream_done, spliced = asyncio.Event(), asyncio.Event()

        async def upstream(reader, writer):
            got.extend(await reader.read())  # until the splice closes us
            upstream_done.set()
            writer.close()

        up = await asyncio.start_server(upstream, "127.0.0.1", 0)

        async def front(reader, writer):
            await reader.readexactly(5)  # the "frame"; the rest stays in the StreamReader buffer
            await asyncio.sleep(0.2)
            u_reader, u_writer = await asyncio.open_connection(*up.sockets[0].getsockname()[:2])
            t0 = time.monotonic()
            await splice_mod.splice(reader, writer, u_reader, u_writer)
            assert time.monotonic() - t0 < 1.5, "close waited on the old protocol"
            # The agent's handler awaits this afterwards: it must not sit out the close timeout.
            await asyncio.wait_for(writer.wait_closed(), 0.5)
            await asyncio.wait_for(u_writer.wait_closed(), 0.5)
            spliced.set()

        srv = await asyncio.start_server(front, "127.0.0.1", 0)
        _, writer = await asyncio.open_connection(*srv.sockets[0].getsockname()[:2])
        writer.write(b"HELLO" + b"early-bytes|")
        await writer.drain()
        await asyncio.sleep(0.4)
        writer.write(b"late-bytes")
        await writer.drain()
        writer.write_eof()
        await asyncio.wait_for(spliced.wait(), 5)
        await asyncio.wait_for(upstream_done.wait(), 5)
        writer.close()
        srv.close()
        up.close()
        return bytes(got)

    assert asyncio.run(main()) == b"early-bytes|late-bytes"


def test_links_send_no_session_tickets(tmp_path):
    # Links never resume, so the two TLS 1.3 tickets a server sends by default are wasted records.
    pki = Pki(tmp_path / "pki")

    def ctx(role, server):
        me = pki.leaf(role[0] + "1", role)
        return tls.mutual_context(
            server = server, certfile = me.certfile, keyfile = me.keyfile, rootfile = me.rootfile
        )

    server_ctx, client_ctx = ctx("worker", True), ctx("master", False)
    a, b = socket.socketpair()
    result = {}

    def serve():
        with server_ctx.wrap_socket(a, server_side = True) as s:
            s.sendall(b"hello")
            result["echo"] = s.recv(5)

    thread = threading.Thread(target = serve)
    thread.start()
    with client_ctx.wrap_socket(b, server_hostname = None) as c:
        assert c.recv(5) == b"hello"
        c.sendall(b"world")
        thread.join(5)
        assert not c.session.has_ticket
    assert result["echo"] == b"world"


def test_the_link_trace_records_both_hops_only_when_asked(make_link, tmp_path, monkeypatch):
    link = make_link()
    link.ping(b"untraced").close()
    trace = tmp_path / "trace.jsonl"
    monkeypatch.setenv("UNSLOTH_CLUSTER_LINK_TRACE", str(trace))
    sock = link.connect()
    for i in range(1, 21):
        sock.sendall(b"q" * i)
        assert _recv_exact(sock, i) == b"q" * i
    sock.close()

    def lines():
        return trace.read_text(encoding = "utf-8").splitlines() if trace.exists() else []

    assert _wait(lambda: len(lines()) == 2)
    rows = [json.loads(line) for line in lines()]
    keys = sorted(k for row in rows for k in row["dirs"])
    assert keys == ["from_master", "from_worker", "to_master", "to_worker"]
    total = 2 * sum(range(1, 21))
    for row in rows:
        # The relay reads the client's first byte itself (the port-check guard) before the splice.
        assert total - 1 <= sum(d["bytes"] for d in row["dirs"].values()) <= total
        assert all(v["n"] >= 1 for v in row["turnaround_us"].values())


_SESSION_THREADS = {
    "unsloth-cluster-agent-conn",
    "unsloth-cluster-relay-conn",
    "unsloth-cluster-agent-revalidate",
    "unsloth-cluster-relay-revalidate",
    "unsloth-cluster-splice",
}


def _session_threads():
    return [t for t in threading.enumerate() if t.name in _SESSION_THREADS and t.is_alive()]


def _accept_threads():
    return [t for t in threading.enumerate() if t.name == "unsloth-cluster-relay-accept"]


def _fds() -> int:
    return len(os.listdir("/proc/self/fd"))


@pytest.mark.parametrize("mode", ["async", "threaded"])
def test_abrupt_disconnects_leave_no_session_threads_or_sockets(make_link, monkeypatch, mode):
    # A client RST or an rpc-server drop mid-session tears down both sides and frees the slot.
    monkeypatch.setenv(_THREADED_ENV, str(int(mode == "threaded")))
    link = make_link()
    if mode == "threaded":
        time.sleep(0.6)  # both threaded listeners idle past their 0.5 s accept timeout
    before = _fds() if _LINUX else 0
    for _ in range(5):
        sock = link.ping(b"abc")
        assert link.target.open == 1
        _rst_close(sock)  # RST from the client mid-session
        assert _wait(lambda: link.target.open == 0)
        assert _wait(lambda: _idle(link))
    link.ping(b"again").close()
    dropper = make_link(target_mode = "drop", drop_after = 4)
    sock = dropper.connect(b"abcd")
    assert _closed(sock)  # the rpc-server end vanishes mid-session
    sock.close()
    assert _wait(lambda: _idle(dropper))
    assert _wait(lambda: not _session_threads(), 10.0), [t.name for t in _session_threads()]
    for lk in (link, dropper):
        assert not lk.agent._sessions and not lk.agent._raw
        assert not lk.relay._relays["w1"].sessions and not lk.relay._relays["w1"].raw
    if _LINUX:
        # The second link's own listeners and loop sockets are the only growth allowed.
        assert _wait(lambda: _fds() <= before + 12), (before, _fds())


@pytest.mark.parametrize("side", ["agent", "relay"])
def test_stopping_either_side_cuts_a_live_threaded_session(make_link, monkeypatch, side):
    monkeypatch.setenv(_THREADED_ENV, "1")
    link = make_link()
    sock = link.ping(b"live")
    assert _wait(lambda: link.agent._sessions and link.agent._raw)
    if side == "agent":
        link.loop.run(link.agent.stop())
    else:
        link.relay.stop()
    assert _closed(sock)
    sock.close()
    assert _wait(lambda: not link.agent._sessions and not link.agent._raw)
    assert _wait(lambda: link.target.open == 0)
    assert _wait(lambda: not _session_threads(), 10.0), [t.name for t in _session_threads()]


@pytest.mark.parametrize("mode", ["async", "threaded"])
def test_concurrent_ensure_for_one_worker_makes_one_listener(tmp_path, monkeypatch, mode):
    started, real_start = [], relay_mod.asyncio.start_server

    async def counting_start(*args, **kwargs):
        started.append(await real_start(*args, **kwargs))
        await asyncio.sleep(0.05)  # widen the window two creators used to race through
        return started[-1]

    # The asyncio listener is what two creators raced on; the threaded one binds synchronously.
    monkeypatch.setattr(relay_mod.asyncio, "start_server", counting_start)
    monkeypatch.setenv(_THREADED_ENV, str(int(mode == "threaded")))
    relays = RelayManager(Pki(tmp_path / "pki").leaf("m1", "master"), lambda pid: True)
    endpoints, gate = [], threading.Barrier(8)

    def ensure():
        gate.wait()
        endpoints.append(relays.ensure("w1", "127.0.0.1", 50070))

    threads = [threading.Thread(target = ensure) for _ in range(8)]
    try:
        for t in threads:
            t.start()
        for t in threads:
            t.join(30)
        assert len(endpoints) == 8 and set(endpoints) == {relays.endpoints()["w1"]}
        if mode == "async":
            assert len(started) == 1, f"{len(started)} listeners for one worker"
        else:
            assert len([t for t in _accept_threads() if t.is_alive()]) == 1
    finally:
        relays.stop()
    assert _wait(lambda: not _accept_threads())


def _fake_rpc_binary(tmp: Path, tls_build: bool, tls_fails: bool) -> Path:
    """`--help` lists the TLS flags only for a TLS build; otherwise it logs its argv (one JSON line
    per start), binds -H/-p and accepts forever."""
    tmp.mkdir(parents = True, exist_ok = True)
    fake = tmp / "fake-rpc-server"
    usage = "  --tls-cert F  --tls-key F  --tls-ca F  --tls-allow F" if tls_build else ""
    fake.write_text(
        textwrap.dedent(f"""\
        #!{sys.executable}
        import json, socket, sys
        if "--help" in sys.argv:
            print("usage: ggml-rpc-server -H host -p port{usage}")
            sys.exit(0)
        with open({str(tmp / "argv.jsonl")!r}, "a") as f:
            f.write(json.dumps(sys.argv[1:]) + "\\n")
        if {tls_fails!r} and "--tls-cert" in sys.argv:
            sys.exit("rejected the TLS arguments")
        a = sys.argv
        host, port = a[a.index("-H") + 1], int(a[a.index("-p") + 1])
        s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((host, port)); s.listen(8)
        while True:
            c, _ = s.accept(); c.close()
        """)
    )
    fake.chmod(0o755)
    return fake


def _last_argv(link: Link) -> list:
    lines = (link.fake.parent / "argv.jsonl").read_text().splitlines()
    return json.loads([line for line in lines if line.strip()][-1])


def _arg(argv: list, flag: str) -> str:
    return argv[argv.index(flag) + 1]


@pytest.fixture
def native_link(make_link, tmp_path):
    def factory(**kw) -> Link:
        tls_build, tls_fails = kw.pop("tls_build", True), kw.pop("tls_fails", False)
        fake = _fake_rpc_binary(
            tmp_path / f"bin{len(list(tmp_path.glob('bin*')))}", tls_build, tls_fails
        )
        link = make_link(
            rpc_binary = str(fake), rpc_port = _free_port(), native_port = _free_port(), **kw
        )
        assert link.loop.run(link.agent.wait_rpc_ready(10.0))
        link.fake = fake
        return link

    return factory


def _allowlist(link: Link) -> list:
    return native.allowlist_path(link.agent.native_port).read_text().split()


def _assert_back_on_loopback(link: Link, pid_native: int) -> None:
    assert _wait(lambda: link.agent.mode == "loopback", 15.0)
    assert _allowlist(link) == []
    assert _wait(lambda: link.agent.rpc_pid not in (None, pid_native), 10.0)
    assert _wait(lambda: "--tls-cert" not in _last_argv(link), 10.0)
    argv = _last_argv(link)
    assert _arg(argv, "-H") == "127.0.0.1" and _arg(argv, "-p") == str(link.agent.rpc_port)
    assert link.loop.run(link.agent.wait_rpc_ready(10.0))


def _flaky_native_close(link: Link, fails: int) -> list:
    plain, failures = link.relay._control, []

    async def flaky(
        node,
        request,
        timeout,
        paired = True,
    ):
        if request.get("type") == "native_close" and len(failures) < fails:
            failures.append(request)
            raise OSError("network unreachable")
        return await plain(node, request, timeout, paired = paired)

    link.relay._control = flaky
    return failures


def _native_ok(link: Link) -> int:
    assert link.relay.native("w1")["type"] == "native_ok"
    return link.agent.rpc_pid


@_SKIP_NT
@pytest.mark.parametrize("why", ["plain_build", "kill_switch"])
def test_native_tls_is_off_without_a_tls_build_or_with_the_kill_switch(
    native_link, monkeypatch, why
):
    if why == "kill_switch":
        monkeypatch.setenv(native.DISABLE_ENV, "0")
    link = native_link(tls_build = why == "kill_switch")
    assert link.agent.native_available is False
    assert link.relay.status("w1")["native"]["available"] is False
    assert link.relay.native("w1")["type"] == "error"


@_SKIP_NT
def test_a_native_request_restarts_rpc_server_with_tls_for_that_master_only(native_link):
    link = native_link()
    port = link.agent.native_port
    assert link.relay.status("w1")["native"] == {"available": True, "active": False, "port": port}
    reply = link.relay.native("w1")
    assert reply == {"type": "native_ok", "port": port, "fingerprint": link.worker.fingerprint}
    argv = _last_argv(link)
    assert _arg(argv, "-p") == str(port)
    assert _arg(argv, "--tls-cert") == link.worker.certfile
    assert _arg(argv, "--tls-key") == link.worker.keyfile
    assert _arg(argv, "--tls-ca") == link.worker.rootfile
    assert Path(_arg(argv, "--tls-allow")).read_text().split() == [link.master.fingerprint]
    assert link.relay.status("w1")["native"]["active"] is True
    assert link.agent.stats["rpc_restarts"] == 0  # a planned switch, not a crash


@_SKIP_NT
@pytest.mark.parametrize(
    "side, close_failures",
    [
        pytest.param("agent", 0, id = "master_revoked_on_worker"),
        pytest.param("relay", 0, id = "worker_removed_on_master"),
        pytest.param("relay", 2, id = "failed_native_close_is_retried"),
    ],
)
def test_revocation_cuts_native_mode(native_link, monkeypatch, side, close_failures):
    # Native bypasses the relay and `cluster remove`/`leave` only edit the registry: the revalidation
    # watch must close it, retrying a failed native_close until the worker answers.
    monkeypatch.setattr(agent_mod if side == "agent" else relay_mod, "REVALIDATE_S", 0.2)
    monkeypatch.setattr(relay_mod, "NATIVE_CLOSE_RETRY_MAX_S", 0.4, raising = False)
    trusted = {"ok": True}
    if side == "relay":
        link = native_link(relay_authorize = lambda pid: trusted["ok"] and pid.node_id == "w1")
    else:
        link = native_link(agent_authorize = lambda pid: trusted["ok"] and pid.node_id == "m1")
    pid_native = _native_ok(link)
    failures = _flaky_native_close(link, close_failures)
    trusted["ok"] = False
    _assert_back_on_loopback(link, pid_native)
    assert len(failures) == close_failures
    if side == "relay":
        assert _wait(lambda: "w1" not in link.relay._native_watches, 5.0)


@_SKIP_NT
def test_a_failed_explicit_native_close_keeps_the_revocation_watch(native_link, monkeypatch):
    # Studio's one native_close attempt fails: the watch must survive and retry.
    monkeypatch.setattr(relay_mod, "REVALIDATE_S", 0.2)
    monkeypatch.setattr(relay_mod, "NATIVE_CLOSE_RETRY_MAX_S", 0.4, raising = False)
    trusted = {"ok": True}
    link = native_link(relay_authorize = lambda pid: trusted["ok"] and pid.node_id == "w1")
    pid_native = _native_ok(link)
    _flaky_native_close(link, 1)
    assert link.relay.native_close("w1") is False  # the explicit close fails once
    trusted["ok"] = False  # the worker is removed on the master
    link.relay.remove("w1")  # as _drop_stale_relays does next
    _assert_back_on_loopback(link, pid_native)


@_SKIP_NT
def test_native_close_and_a_relay_session_both_return_to_loopback(native_link):
    link = native_link()
    _native_ok(link)
    assert link.relay.native_close("w1") is True
    assert link.agent.mode == "loopback"
    assert _allowlist(link) == []
    # Back to native, then a relay session (an older master, or native turned off) takes it back.
    _native_ok(link)
    sock = link.connect(b"x")
    assert _wait(lambda: link.agent.mode == "loopback", 10.0)
    sock.close()
    assert _wait(lambda: "--tls-cert" not in _last_argv(link), 10.0)


@_SKIP_NT
@pytest.mark.parametrize("mode", ["threaded", "async"])
def test_a_relay_open_is_busy_while_a_native_session_is_live(native_link, monkeypatch, mode):
    # Native sessions bypass _active: an open_rpc must not restart rpc-server under one.
    monkeypatch.setenv(_THREADED_ENV, str(int(mode == "threaded")))
    link = native_link()
    pid_native = _native_ok(link)
    monkeypatch.setattr(link.agent, "_native_client", lambda: True)
    sock = link.connect(b"x")
    assert _closed(sock, 10.0)
    sock.close()
    assert link.agent.mode == "native"
    assert link.agent.rpc_pid == pid_native
    assert "--tls-cert" in _last_argv(link)


@_SKIP_NT
def test_a_removed_worker_still_takes_a_native_close(native_link):
    # Closing a just-removed worker's native session: no longer authorized, but closing is allowed.
    link = native_link(relay_authorize = lambda pid: False)
    assert link.relay.status("w1")["reachable"] is False  # status needs the worker still paired
    peer = tls.PeerIdentity(link.master.fingerprint, "m1", "master")
    link.loop.run(link.agent._enter_native(peer))
    assert link.agent.mode == "native"
    assert link.relay.native_close("w1") is True
    assert link.agent.mode == "loopback"


@_SKIP_NT
def test_a_failed_native_start_rolls_back_to_loopback(native_link, monkeypatch):
    # TLS rpc-server dies at start: the worker must return to loopback for the relay fallback.
    monkeypatch.setattr(agent_mod, "RPC_READY_TIMEOUT_S", 3.0)
    link = native_link(tls_fails = True)
    assert link.relay.native("w1")["type"] == "error"
    assert link.agent.mode == "loopback"
    assert _allowlist(link) == []
    argv = _last_argv(link)
    assert "--tls-cert" not in argv and _arg(argv, "-H") == "127.0.0.1"
    assert link.loop.run(link.agent.wait_rpc_ready(10.0))
    assert link.relay.status("w1")["rpc_up"] is True


def test_the_relay_waits_out_a_native_to_loopback_restart():
    # open_rpc on a worker still in native mode restarts its rpc-server before answering.
    assert relay_mod.OPEN_RPC_REPLY_TIMEOUT_S > agent_mod.RPC_READY_TIMEOUT_S


@pytest.mark.skipif(not _LINUX, reason = "the /proc owner lookup is Linux-only")
def test_the_relay_serves_only_local_clients_of_this_user(make_link, monkeypatch):
    # Every local account can reach a 127.0.0.1 port, and the relay proxies with this user's cluster
    # credentials: a client owned by another uid gets nothing, and the worker never hears of it.
    link = make_link()
    link.ping(b"mine").close()  # this user's own client
    before = link.agent.stats["connections"]
    monkeypatch.setattr(splice_mod, "_loopback_owner_uid", lambda c, s: os.getuid() + 1)
    assert _closed(link.connect(b"steal"))
    time.sleep(0.3)
    assert link.agent.stats["connections"] == before
    assert link.target.connections == 1


@pytest.mark.skipif(not _LINUX, reason = "the /proc owner lookup is Linux-only")
def test_the_owner_lookup_matches_the_exact_tuple_of_real_connections():
    # Two live connections share the client port but not the client address: each is attributed to
    # its own row, and the 127.0.0.2 one is still found (its exact tuple, not the 127.0.0.1 one).
    owner, ours = splice_mod._loopback_owner_uid, splice_mod.local_client_is_ours
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(2)
    first = socket.create_connection(server.getsockname())
    conn1, _ = server.accept()
    second = socket.socket()
    socks = [first, second, conn1, server]
    try:
        assert owner(conn1.getpeername(), conn1.getsockname()) == os.getuid() and ours(conn1)
        try:
            second.bind(("127.0.0.2", first.getsockname()[1]))
        except OSError:
            pytest.skip("127.0.0.2 is not bindable here")
        second.connect(server.getsockname())
        conn2, _ = server.accept()
        socks.append(conn2)
        assert conn2.getpeername() == ("127.0.0.2", first.getsockname()[1])
        assert owner(conn2.getpeername(), conn2.getsockname()) == os.getuid()
        assert ours(conn1) and ours(conn2)
        # A tuple no live connection has: refused.
        assert owner(("127.0.0.3", first.getsockname()[1]), conn2.getsockname()) is None
    finally:
        for s in socks:
            s.close()


_TCP_HEADER = "  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode"
_OUR_UID, _OTHER_UID = 4242, 31337
# Client port 0xA1B2 (41394), listener port 0x1F90 (8080). Addresses are host-order 32-bit words
# (little-endian here): 0100007F = 127.0.0.1, 0200007F = 127.0.0.2.
_V4_LOCAL, _V4_CLIENT1, _V4_CLIENT2 = "0100007F:1F90", "0100007F:A1B2", "0200007F:A1B2"
_V6_LOOP = "00000000000000000000000001000000"  # ::1
_V6_MAPPED_LOOP = "0000000000000000FFFF00000100007F"  # ::ffff:127.0.0.1
_V4 = (("127.0.0.1", 0xA1B2), ("127.0.0.1", 0x1F90))
_V4_FROM_2 = (("127.0.0.2", 0xA1B2), ("127.0.0.1", 0x1F90))
_V6 = (("::1", 0xA1B2, 0, 0), ("::1", 0x1F90, 0, 0))
_MAPPED = (("::ffff:127.0.0.1", 0xA1B2, 0, 0), ("::ffff:127.0.0.1", 0x1F90, 0, 0))
_MAPPED_FROM_2 = (("::ffff:127.0.0.2", 0xA1B2, 0, 0), ("::ffff:127.0.0.1", 0x1F90, 0, 0))


def _row(i, local, remote, state, uid):
    return f"  {i:>2}: {local} {remote} {state} 00000000:00000000 00:00000000 00000000 {uid:>5}        0 {1000 + i} 1 0000000000000000 20 4 30 10 -1"


def _v4(state, uid = _OUR_UID):
    return _row(0, _V4_CLIENT1, _V4_LOCAL, state, uid)


def _v6(state, loop = _V6_LOOP):
    return _row(0, f"{loop}:A1B2", f"{loop}:1F90", state, _OUR_UID)


# Another account reuses this user's (TIME_WAIT) source port on 127.0.0.2: a port-only lookup
# returned this user's uid and admitted it.
_ATTACK = [
    _v4("06"),
    _row(1, _V4_LOCAL, _V4_CLIENT2, "01", _OUR_UID),  # the listener's accepted end
    _row(2, _V4_CLIENT2, _V4_LOCAL, "01", _OTHER_UID),  # the attacker's live client end
]
_OURS_BESIDE_STALE = [
    _row(0, _V4_CLIENT2, _V4_LOCAL, "06", _OTHER_UID),  # stale, someone else, other address
    _row(1, _V4_CLIENT1, _V4_LOCAL, "01", _OUR_UID),
]


@pytest.mark.skipif(
    not _LINUX or sys.byteorder != "little",
    reason = "Linux owner lookup; fixture addresses are little-endian",
)
@pytest.mark.parametrize(
    "tcp, tcp6, names, admitted",
    [
        pytest.param(_ATTACK, [], _V4_FROM_2, False, id = "other_address_same_ports_refused"),
        pytest.param([_v4("01")], [], _V4_FROM_2, False, id = "never_borrow_a_near_match"),
        pytest.param([_v4("06")], [], _V4, False, id = "exact_tuple_time_wait_refused"),
        pytest.param([_v4("05")], [], _V4, False, id = "exact_tuple_fin_wait2_refused"),
        pytest.param([_v4("07")], [], _V4, False, id = "exact_tuple_close_refused"),
        pytest.param(_OURS_BESIDE_STALE, [], _V4, True, id = "exact_live_tuple_ours_admitted"),
        pytest.param([_v4("01", _OTHER_UID)], [], _V4, False, id = "other_uid_refused"),
        pytest.param([], [_v6("01", _V6_MAPPED_LOOP)], _V4, True, id = "v6_client_v4_listener"),
        pytest.param([_v4("01")], [], _MAPPED, True, id = "dual_stack_listener_v4_client"),
        pytest.param([], [_v6("01")], _V6, True, id = "ipv6_loopback_both_ends"),
        pytest.param([_v4("06")], [_v6("06")], _MAPPED_FROM_2, False, id = "mapped_127_0_0_2"),
        pytest.param([], [_v6("06")], _V6, False, id = "ipv6_time_wait_not_live"),
    ],
)
def test_the_owner_lookup_admits_only_the_exact_live_tuple_of_this_user(
    monkeypatch, tcp, tcp6, names, admitted
):
    files = {
        "/proc/net/tcp": "\n".join([_TCP_HEADER, *tcp]) + "\n",
        "/proc/net/tcp6": "\n".join([_TCP_HEADER, *tcp6]) + "\n",
    }
    real_open = builtins.open

    def fake_open(path, *args, **kwargs):
        return io.StringIO(files[path]) if path in files else real_open(path, *args, **kwargs)

    monkeypatch.setattr(splice_mod, "open", fake_open, raising = False)
    monkeypatch.setattr(splice_mod.os, "getuid", lambda: _OUR_UID)
    sock = SimpleNamespace(getpeername = lambda: names[0], getsockname = lambda: names[1])
    assert bool(splice_mod.local_client_is_ours(sock)) is admitted
