# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The cluster link data path: worker agent + master relay over mutual TLS, against a fake rpc-server,
plus the real CPU ggml-rpc-server / llama-server when a build is available."""

from __future__ import annotations

import asyncio
import datetime
import os
import sys
import random
import socket
import struct
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

import pytest

pytest.importorskip("cryptography")

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

from studio.cluster_link import tls
from studio.cluster_link.agent import Agent
from studio.cluster_link.model import LocalIdentity
from studio.cluster_link.relay import RelayManager

MIB = 1024 * 1024
_NOW = datetime.datetime.now(datetime.timezone.utc)


class Pki:
    """A throwaway cluster root and leaf certificates shaped like the real ones."""

    def __init__(
        self,
        directory: Path,
        name: str = "root",
    ):
        self.dir = directory
        self.dir.mkdir(parents = True, exist_ok = True)
        self.key = ec.generate_private_key(ec.SECP256R1())
        self.name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, f"unsloth cluster {name}")])
        cert = (
            x509.CertificateBuilder()
            .subject_name(self.name)
            .issuer_name(self.name)
            .public_key(self.key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(_NOW - datetime.timedelta(minutes = 5))
            .not_valid_after(_NOW + datetime.timedelta(days = 30))
            .add_extension(x509.BasicConstraints(ca = True, path_length = 0), critical = True)
            .add_extension(
                x509.KeyUsage(
                    digital_signature = False,
                    content_commitment = False,
                    key_encipherment = False,
                    data_encipherment = False,
                    key_agreement = False,
                    key_cert_sign = True,
                    crl_sign = True,
                    encipher_only = False,
                    decipher_only = False,
                ),
                critical = True,
            )
            .add_extension(
                x509.SubjectKeyIdentifier.from_public_key(self.key.public_key()), critical = False
            )
            .sign(self.key, hashes.SHA256())
        )
        self.rootfile = str(self.dir / f"{name}.pem")
        Path(self.rootfile).write_bytes(cert.public_bytes(serialization.Encoding.PEM))

    def leaf(
        self,
        node_id: str,
        role: str,
        rootfile: Optional[str] = None,
    ) -> LocalIdentity:
        key = ec.generate_private_key(ec.SECP256R1())
        cert = (
            x509.CertificateBuilder()
            .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, node_id)]))
            .issuer_name(self.name)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(_NOW - datetime.timedelta(minutes = 5))
            .not_valid_after(_NOW + datetime.timedelta(days = 30))
            .add_extension(x509.BasicConstraints(ca = False, path_length = None), critical = True)
            .add_extension(
                x509.ExtendedKeyUsage(
                    [ExtendedKeyUsageOID.SERVER_AUTH, ExtendedKeyUsageOID.CLIENT_AUTH]
                ),
                critical = False,
            )
            .add_extension(
                x509.SubjectAlternativeName(
                    [
                        x509.UniformResourceIdentifier(tls.NODE_URI_PREFIX + node_id),
                        x509.UniformResourceIdentifier(tls.ROLE_URI_PREFIX + role),
                    ]
                ),
                critical = False,
            )
            .add_extension(
                x509.AuthorityKeyIdentifier.from_issuer_public_key(self.key.public_key()),
                critical = False,
            )
            .sign(self.key, hashes.SHA256())
        )
        certfile = self.dir / f"{node_id}-{role}.pem"
        keyfile = self.dir / f"{node_id}-{role}.key"
        certfile.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
        keyfile.write_bytes(
            key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            )
        )
        return LocalIdentity(
            node_id = node_id,
            role = role,
            cluster_id = "test-cluster",
            certfile = str(certfile),
            keyfile = str(keyfile),
            rootfile = rootfile or self.rootfile,
            fingerprint = tls.fingerprint(cert.public_bytes(serialization.Encoding.DER)),
        )


class LoopThread:
    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target = self.loop.run_forever, daemon = True)
        self.thread.start()

    def run(
        self,
        coro,
        timeout: float = 30.0,
    ):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout)

    def stop(self):
        async def cancel_rest():
            current = asyncio.current_task()
            rest = [t for t in asyncio.all_tasks() if t is not current]
            for task in rest:
                task.cancel()
            await asyncio.gather(*rest, return_exceptions = True)

        self.run(cancel_rest())
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(5)
        self.loop.close()


class FakeRpc:
    """Stands in for ggml-rpc-server: echoes, floods, or drops the connection after N bytes."""

    def __init__(
        self,
        mode: str = "echo",
        drop_after: int = 0,
    ):
        self.mode = mode
        self.drop_after = drop_after
        self.connections = 0
        self.open = 0
        self.received = 0
        self.sent = 0
        self.server = None
        self.port = 0

    async def start(self):
        self.server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        self.port = self.server.sockets[0].getsockname()[1]

    async def close(self):
        self.server.close()

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
            while True:
                data = await reader.read(256 * 1024)
                if not data:
                    break
                self.received += len(data)
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
    def __init__(
        self,
        tmp: Path,
        *,
        target_mode: str = "echo",
        drop_after: int = 0,
        agent_authorize: Optional[Callable] = None,
        relay_authorize: Optional[Callable] = None,
        worker_role: str = "worker",
        master_role: str = "master",
        master_other_root: bool = False,
        source_filter: Optional[Callable[[str], bool]] = None,
        relay_node_id: str = "w1",
        rpc_binary: Optional[str] = None,
        rpc_port: int = 0,
        **agent_kwargs: Any,
    ):
        self.pki = Pki(tmp / "pki")
        self.worker = self.pki.leaf("w1", worker_role)
        if master_other_root:
            self.master = Pki(tmp / "other", "other").leaf(
                "m1", master_role, rootfile = self.pki.rootfile
            )
        else:
            self.master = self.pki.leaf("m1", master_role)
        self.loop = LoopThread()
        self.target = FakeRpc(target_mode, drop_after)
        if rpc_binary is None:
            self.loop.run(self.target.start())
        master_fp = self.master.fingerprint
        worker_fp = self.worker.fingerprint
        self.agent = Agent(
            self.worker,
            agent_authorize or (lambda pid: pid.node_id == "m1" and pid.fingerprint == master_fp),
            host = "127.0.0.1",
            port = 0,
            rpc_binary = rpc_binary,
            rpc_port = rpc_port,
            cache_dir = str(tmp / "cache"),
            rpc_target = None if rpc_binary else ("127.0.0.1", self.target.port),
            source_filter = source_filter,
            **agent_kwargs,
        )
        self.loop.run(self.agent.start())
        self.relay = RelayManager(
            self.master,
            relay_authorize or (lambda pid: pid.node_id == "w1" and pid.fingerprint == worker_fp),
        )
        self.endpoint = self.relay.ensure(relay_node_id, "127.0.0.1", self.agent.port)
        self.relay_node_id = relay_node_id

    def connect(self) -> socket.socket:
        host, port = self.endpoint.rsplit(":", 1)
        sock = socket.create_connection((host, int(port)), timeout = 10)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        return sock

    def close(self):
        self.relay.stop()
        self.loop.run(self.agent.stop())
        if self.target.server is not None:
            self.loop.run(self.target.close())
        self.loop.stop()


@pytest.fixture(autouse = True)
def _own_run_dir(tmp_path, monkeypatch):
    # rpc port ownership records are machine-wide: keep each test (and its subprocesses) to its own.
    monkeypatch.setenv("UNSLOTH_CLUSTER_RUN_DIR", str(tmp_path / "run"))


@pytest.fixture
def make_link(tmp_path):
    links = []

    def factory(**kwargs) -> Link:
        link = Link(tmp_path / f"l{len(links)}", **kwargs)
        links.append(link)
        return link

    yield factory
    for link in links:
        link.close()


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(1 << 20, n - len(buf)))
        if not chunk:
            break
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


def _echo(
    link: Link,
    payload: bytes,
    fragment: bool = False,
) -> bytes:
    sock = link.connect()
    rng = random.Random(len(payload))

    def send():
        if not fragment:
            sock.sendall(payload)
            return
        i = 0
        while i < len(payload):
            n = rng.randint(1, 9000)
            sock.sendall(payload[i : i + n])
            i += n
            if rng.random() < 0.01:
                time.sleep(0.001)

    sender = threading.Thread(target = send)
    sender.start()
    got = _recv_exact(sock, len(payload))
    sender.join()
    sock.close()
    return got


@pytest.mark.parametrize("size", [1, 65537, MIB + 3, 64 * MIB])
def test_round_trip_is_byte_exact(make_link, size):
    link = make_link()
    payload = os.urandom(size)
    assert _echo(link, payload) == payload
    assert link.agent.stats["rpc_streams"] == 1
    counters = link.relay.counters("w1")
    assert counters["to_worker"] == size and counters["from_worker"] == size


def test_fragmented_writes_arrive_intact(make_link):
    link = make_link()
    payload = os.urandom(3 * MIB + 17)
    assert _echo(link, payload, fragment = True) == payload


def test_a_slow_reader_backpressures_the_rpc_server(make_link):
    link = make_link(target_mode = "flood")
    sock = link.connect()
    sock.sendall(b"g")
    time.sleep(1.5)
    sent = link.target.sent
    assert 0 < sent < 64 * MIB, f"{sent / MIB:.1f} MiB queued for a reader that read nothing"
    sock.close()


def test_a_client_reset_tears_down_the_worker_side(make_link):
    link = make_link()
    sock = link.connect()
    sock.sendall(b"abc")
    assert _recv_exact(sock, 3) == b"abc"
    assert link.target.open == 1
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
    sock.close()
    assert _wait(lambda: link.target.open == 0)
    assert _wait(lambda: not link.agent.busy)
    assert _echo(link, b"again") == b"again"


def test_an_rpc_server_drop_closes_the_client(make_link):
    link = make_link(target_mode = "drop", drop_after = 4)
    sock = link.connect()
    sock.sendall(b"abcd")
    assert _closed(sock)
    assert _wait(lambda: not link.agent.busy)


def test_a_port_check_never_reaches_the_worker(make_link):
    link = make_link()
    before = link.agent.stats["connections"]
    for _ in range(3):
        link.connect().close()
    time.sleep(0.3)
    assert link.agent.stats["connections"] == before
    assert link.target.connections == 0


def test_a_second_client_is_refused_at_once_while_a_stream_is_active(make_link):
    link = make_link()
    first = link.connect()
    first.sendall(b"x")
    assert _recv_exact(first, 1) == b"x"
    second = link.connect()
    started = time.monotonic()
    assert _closed(second, timeout = 2.0)
    assert time.monotonic() - started < 1.0
    first.sendall(b"y")
    assert _recv_exact(first, 1) == b"y"
    first.close()


def _refused(link: Link) -> None:
    sock = link.connect()
    sock.sendall(b"\x0e" + b"\0" * 15)
    assert _closed(sock)
    assert link.target.connections == 0


def test_a_master_the_worker_has_not_paired_is_refused(make_link):
    _refused(make_link(agent_authorize = lambda pid: False))


def test_a_worker_certificate_cannot_act_as_master(make_link):
    _refused(make_link(master_role = "worker"))


def test_a_master_certificate_cannot_act_as_worker(make_link):
    _refused(make_link(worker_role = "master"))


def test_the_relay_refuses_a_worker_with_another_node_id(make_link):
    _refused(make_link(relay_node_id = "w2"))


def test_the_relay_refuses_an_unauthorized_worker(make_link):
    _refused(make_link(relay_authorize = lambda pid: False))


def test_a_certificate_from_another_cluster_root_is_refused(make_link):
    _refused(make_link(master_other_root = True))


def test_a_non_private_source_is_refused(make_link):
    link = make_link(source_filter = lambda host: False)
    _refused(link)
    assert link.agent.stats["rejected"] >= 1


def test_status_reports_rpc_and_busy(make_link):
    link = make_link()
    status = link.relay.status("w1")
    assert status["reachable"] and status["type"] == "status"
    assert status["rpc_up"] is True and status["busy"] is False
    assert status["node_id"] == "w1"
    sock = link.connect()
    sock.sendall(b"x")
    assert _recv_exact(sock, 1) == b"x"
    assert link.relay.status("w1")["busy"] is True
    assert link.relay.busy("w1") is True
    sock.close()
    assert _wait(lambda: not link.agent.busy)


def test_status_of_an_unreachable_worker(make_link):
    link = make_link()
    link.relay.ensure("w1", "127.0.0.1", 1)
    status = link.relay.status("w1", timeout = 1.0)
    assert status["reachable"] is False and status["type"] == "error"


def test_private_address_filter():
    from studio.cluster_link.splice import is_private_address
    for ok in (
        "127.0.0.1",
        "10.1.2.3",
        "172.16.0.9",
        "192.168.1.5",
        "169.254.0.1",
        "::1",
        "fe80::1%eth0",
        "fd00::5",
        "::ffff:192.168.1.1",
    ):
        assert is_private_address(ok), ok
    for bad in ("8.8.8.8", "1.1.1.1", "2606:4700::1111", "::ffff:8.8.8.8", "not-an-ip", ""):
        assert not is_private_address(bad), bad


_BIN_DIR = Path(os.environ.get("UNSLOTH_TEST_LLAMA_BIN_DIR") or "/nonexistent")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.skipif(
    not ((_BIN_DIR / "ggml-rpc-server").is_file() and (_BIN_DIR / "llama-server").is_file()),
    reason = "no CPU llama.cpp build (set UNSLOTH_TEST_LLAMA_BIN_DIR)",
)
def test_real_rpc_server_lists_its_device_through_the_relay(make_link, tmp_path):
    rpc_port = _free_port()
    link = make_link(rpc_binary = str(_BIN_DIR / "ggml-rpc-server"), rpc_port = rpc_port)
    assert _wait(lambda: link.agent.rpc_pid is not None, 15)

    def listening() -> bool:
        try:
            socket.create_connection(("127.0.0.1", rpc_port), timeout = 1).close()
            return True
        except OSError:
            return False

    assert _wait(listening, 15), list(link.agent.tail)
    env = dict(os.environ, LD_LIBRARY_PATH = str(_BIN_DIR), GGML_RPC_NO_RDMA = "1")
    out = subprocess.run(
        [str(_BIN_DIR / "llama-server"), "--rpc", link.endpoint, "--list-devices"],
        capture_output = True,
        text = True,
        encoding = "utf-8",
        errors = "replace",
        timeout = 120,
        env = env,
    )
    text = out.stdout + out.stderr
    assert "RPC" in text, text[-2000:]
    assert link.agent.stats["rpc_streams"] >= 1
    assert link.relay.counters("w1").get("from_worker", 0) > 0
    pid = link.agent.rpc_pid
    os.kill(pid, 9)
    assert _wait(lambda: link.agent.rpc_pid not in (None, pid), 20)
    assert link.agent.stats["rpc_restarts"] >= 1


def test_back_to_back_sessions_are_not_refused_as_busy(make_link):
    link = make_link()
    for i in range(30):
        assert _echo(link, b"s%d" % i) == b"s%d" % i
    assert link.agent.stats["rpc_streams"] == 30


AGENT_SCRIPT = """
import asyncio, json, sys
sys.path.insert(0, {repo!r})
from studio.cluster_link.agent import Agent
from studio.cluster_link.model import LocalIdentity
me = LocalIdentity(**json.loads(sys.argv[1]))
agent = Agent(me, lambda pid: True, host = "127.0.0.1", port = 0, rpc_binary = sys.argv[2], rpc_port = 1)
async def main():
    await agent.start()
    while agent.rpc_pid is None:
        await asyncio.sleep(0.05)
    print(agent.rpc_pid, flush = True)
    await asyncio.sleep(600)
asyncio.run(main())
"""


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason = "PR_SET_PDEATHSIG is Linux only")
def test_a_killed_agent_takes_its_rpc_server_with_it(tmp_path):
    import dataclasses
    import json
    import signal
    import subprocess

    me = Pki(tmp_path / "pki").leaf("w1", "worker")
    fake = tmp_path / "fake-rpc-server"
    fake.write_text("#!/bin/sh\nexec sleep 600\n")
    fake.chmod(0o755)
    agent = subprocess.Popen(
        [
            sys.executable,
            "-c",
            AGENT_SCRIPT.format(repo = str(Path(__file__).resolve().parents[2])),
            json.dumps(dataclasses.asdict(me)),
            str(fake),
        ],
        stdout = subprocess.PIPE,
        text = True,
    )
    try:
        child = int(agent.stdout.readline())
        os.kill(child, 0)
        agent.send_signal(signal.SIGKILL)
        agent.wait(10)

        def gone() -> bool:
            try:
                os.kill(child, 0)
            except ProcessLookupError:
                return True
            return False

        assert _wait(gone, 10.0), f"rpc-server {child} outlived its agent"
    finally:
        if agent.poll() is None:
            agent.kill()


def test_concurrent_ensure_for_one_worker_makes_one_relay(tmp_path, monkeypatch):
    import threading

    from studio.cluster_link import relay as relay_mod

    started = []
    real_start = relay_mod.asyncio.start_server

    async def counting_start(*args, **kwargs):
        server = await real_start(*args, **kwargs)
        started.append(server)
        await asyncio.sleep(0.05)  # widen the window two creators used to race through
        return server

    monkeypatch.setattr(relay_mod.asyncio, "start_server", counting_start)
    # The asyncio listener is what two creators raced on; the threaded one binds synchronously.
    monkeypatch.delenv("UNSLOTH_CLUSTER_LINK_THREADED_SPLICE", raising = False)
    pki = Pki(tmp_path / "pki")
    relays = RelayManager(pki.leaf("m1", "master"), lambda pid: True)
    try:
        endpoints, gate = [], threading.Barrier(8)

        def ensure():
            gate.wait()
            endpoints.append(relays.ensure("w1", "127.0.0.1", 50070))

        threads = [threading.Thread(target = ensure) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(30)
        assert set(endpoints) == {relays.endpoints()["w1"]}
        assert len(started) == 1, f"{len(started)} listeners for one worker"
    finally:
        relays.stop()


PID1_SCRIPT = """
import asyncio, json, os, sys
sys.path.insert(0, {repo!r})
from studio.cluster_link.agent import Agent
from studio.cluster_link.model import LocalIdentity
me = LocalIdentity(**json.loads(sys.argv[1]))
# What a container sees when `unsloth worker` is its entrypoint: the agent is pid 1, so it is
# also the parent its rpc-server reports.
os.getpid = lambda: 1
os.getppid = lambda: 1
async def main():
    agent = Agent(me, lambda pid: True, host = "127.0.0.1", port = 0, rpc_binary = sys.argv[2], rpc_port = 1)
    await agent.start()
    await asyncio.sleep(2.0)
    # The fake never binds a port, so "running" is the live pid, not rpc_up (which needs the port).
    print(json.dumps({{"alive": agent.rpc_pid is not None, "restarts": agent.stats["rpc_restarts"]}}), flush = True)
    await agent.stop()
asyncio.run(main())
"""


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason = "PR_SET_PDEATHSIG is Linux only")
def test_an_agent_running_as_pid_1_keeps_its_rpc_server(tmp_path):
    import dataclasses
    import json
    import subprocess

    me = Pki(tmp_path / "pki").leaf("w1", "worker")
    fake = tmp_path / "fake-rpc-server"
    fake.write_text("#!/bin/sh\nexec sleep 600\n")
    fake.chmod(0o755)
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            PID1_SCRIPT.format(repo = str(Path(__file__).resolve().parents[2])),
            json.dumps(dataclasses.asdict(me)),
            str(fake),
        ],
        capture_output = True,
        text = True,
        timeout = 60,
    )
    lines = out.stdout.strip().splitlines()
    assert lines, out.stderr[-2000:]
    assert json.loads(lines[-1]) == {"alive": True, "restarts": 0}


def test_stream_sockets_get_nodelay_and_keepalive():
    import asyncio as aio
    import socket as so

    from studio.cluster_link import splice as sp

    async def main():
        seen = {}

        async def handle(reader, writer):
            sp.tune_socket(writer)
            sock = writer.get_extra_info("socket")
            seen["keepalive"] = sock.getsockopt(so.SOL_SOCKET, so.SO_KEEPALIVE)
            seen["nodelay"] = sock.getsockopt(so.IPPROTO_TCP, so.TCP_NODELAY)
            if hasattr(so, "TCP_KEEPIDLE"):
                seen["idle"] = sock.getsockopt(so.IPPROTO_TCP, so.TCP_KEEPIDLE)
            writer.close()

        server = await aio.start_server(handle, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        _, w = await aio.open_connection("127.0.0.1", port)
        for _ in range(100):
            if seen:
                break
            await aio.sleep(0.02)
        w.close()
        server.close()
        return seen

    seen = asyncio.run(main())
    assert seen["keepalive"] and seen["nodelay"]
    if "idle" in seen:
        assert seen["idle"] == sp.KEEPALIVE_IDLE_S


@pytest.mark.skipif(os.name == "nt", reason = "the fake rpc-server is a shebang script")
def test_a_failed_bind_starts_no_rpc_server(tmp_path):
    me = Pki(tmp_path / "pki").leaf("w1", "worker")
    taken = socket.socket()
    taken.bind(("127.0.0.1", 0))
    taken.listen(1)
    marker = tmp_path / "started"
    fake = tmp_path / "fake-rpc-server"
    fake.write_text(f"#!/bin/sh\ntouch {marker}\nexec sleep 600\n")
    fake.chmod(0o755)
    agent = Agent(
        me,
        lambda pid: True,
        host = "127.0.0.1",
        port = taken.getsockname()[1],
        rpc_binary = str(fake),
        rpc_port = 1,
    )

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


def test_splice_forwards_bytes_buffered_before_it_started_and_closes_promptly():
    from studio.cluster_link.splice import splice
    async def main():
        got = bytearray()
        upstream_done = asyncio.Event()

        async def upstream(reader, writer):
            got.extend(await reader.read())  # until the splice closes us
            upstream_done.set()
            writer.close()

        up = await asyncio.start_server(upstream, "127.0.0.1", 0)
        up_port = up.sockets[0].getsockname()[1]
        spliced = asyncio.Event()

        async def front(reader, writer):
            await reader.readexactly(5)  # the "frame"; the rest stays in the StreamReader buffer
            await asyncio.sleep(0.2)
            u_reader, u_writer = await asyncio.open_connection("127.0.0.1", up_port)
            t0 = time.monotonic()
            await splice(reader, writer, u_reader, u_writer)
            assert time.monotonic() - t0 < 1.5, "close waited on the old protocol"
            # The agent's handler awaits this afterwards: it must not sit out the close timeout.
            await asyncio.wait_for(writer.wait_closed(), 0.5)
            await asyncio.wait_for(u_writer.wait_closed(), 0.5)
            spliced.set()

        srv = await asyncio.start_server(front, "127.0.0.1", 0)
        reader, writer = await asyncio.open_connection("127.0.0.1", srv.sockets[0].getsockname()[1])
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


NO_PDEATHSIG_AGENT = """
import asyncio, json, sys
sys.path.insert(0, {repo!r})
import studio.cluster_link.agent as A
A._die_with_agent_hook = lambda: None  # as on macOS / Windows
me = A.LocalIdentity(**json.loads(sys.argv[1]))
agent = A.Agent(me, lambda pid: True, host = "127.0.0.1", port = 0, rpc_binary = sys.argv[2],
                rpc_port = 1, cache_dir = sys.argv[3])
async def main():
    await agent.start()
    while agent.rpc_pid is None:
        await asyncio.sleep(0.05)
    print(agent.rpc_pid, flush = True)
    await asyncio.sleep(600)
asyncio.run(main())
"""


@pytest.mark.skipif(os.name == "nt", reason = "the fake rpc-server is a shebang script")
def test_the_next_agent_stops_an_rpc_server_a_killed_agent_left_behind(tmp_path):
    import dataclasses
    import json
    import signal
    import subprocess

    psutil = pytest.importorskip("psutil")
    me = Pki(tmp_path / "pki").leaf("w1", "worker")
    fake = tmp_path / "fake-rpc-server"
    fake.write_text(f"#!{sys.executable}\nimport time\ntime.sleep(600)\n")
    fake.chmod(0o755)
    argv = [
        sys.executable,
        "-c",
        NO_PDEATHSIG_AGENT.format(repo = str(Path(__file__).resolve().parents[2])),
        json.dumps(dataclasses.asdict(me)),
        str(fake),
        str(tmp_path / "cache"),
    ]
    children = []
    try:
        first = subprocess.Popen(argv, stdout = subprocess.PIPE, text = True)
        children.append(first)
        orphan = int(first.stdout.readline())
        first.send_signal(signal.SIGKILL)
        first.wait(10)
        assert psutil.pid_exists(orphan)  # nothing took it down: the case this guards
        children.append(psutil.Process(orphan))
        second = subprocess.Popen(argv, stdout = subprocess.PIPE, text = True)
        children.append(second)
        fresh = int(second.stdout.readline())
        children.append(psutil.Process(fresh))
        assert fresh != orphan
        assert _wait(
            lambda: not psutil.pid_exists(orphan) or psutil.Process(orphan).status() == "zombie",
            10.0,
        )
    finally:
        for c in children:
            try:
                c.kill()
            except Exception:
                pass
        for c in children[2:]:
            if isinstance(c, subprocess.Popen):
                c.wait(10)


@pytest.mark.skipif(os.name == "nt", reason = "the fake rpc-server is a shebang script")
def test_rpc_up_waits_for_the_port_not_just_the_process(tmp_path):
    # Reporting a process that has not bound yet as up made a load right after `unsloth worker` start
    # fail on that worker.
    import textwrap

    me = Pki(tmp_path / "pki").leaf("w1", "worker")
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
    agent = Agent(
        me,
        lambda pid: True,
        host = "127.0.0.1",
        port = 0,
        rpc_binary = str(slow),
        rpc_port = port,
        cache_dir = str(tmp_path / "cache"),
    )

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


def test_revoking_a_worker_cuts_its_live_session(make_link, monkeypatch):
    # `unsloth cluster remove` runs in another process and only edits the registry: a session that is
    # already open must still end, not keep carrying tensors until Studio next refreshes its nodes.
    import studio.cluster_link.relay as relay_mod

    monkeypatch.setattr(relay_mod, "REVALIDATE_S", 0.2)
    paired = {"w1": True}
    link = make_link(relay_authorize = lambda pid: paired.get(pid.node_id, False))
    sock = link.connect()
    sock.sendall(b"ping")
    assert _recv_exact(sock, 4) == b"ping"  # the session is live
    paired["w1"] = False
    assert _closed(sock, 5.0), "the relay kept a revoked worker's session open"
    assert "no longer paired" in link.relay.status("w1").get("relay", {}).get("last_error", "")


def test_a_worker_leaving_cuts_the_masters_live_session(make_link, monkeypatch):
    # `unsloth cluster leave` on the worker only edits its registry; the master side still trusts the
    # worker, so the agent itself must end the session it is serving.
    import studio.cluster_link.agent as agent_mod

    monkeypatch.setattr(agent_mod, "REVALIDATE_S", 0.2)
    trusted = {"m1": True}
    link = make_link(agent_authorize = lambda pid: trusted.get(pid.node_id, False))
    sock = link.connect()
    sock.sendall(b"ping")
    assert _recv_exact(sock, 4) == b"ping"
    trusted["m1"] = False
    assert _closed(sock, 5.0), "the agent kept serving a master it no longer trusts"


@pytest.mark.skipif(os.name == "nt", reason = "the fake rpc-server is a shebang script")
def test_a_second_agent_on_the_same_rpc_port_does_not_kill_the_first_ones_server(tmp_path):
    import dataclasses
    import json
    import subprocess

    psutil = pytest.importorskip("psutil")
    me = Pki(tmp_path / "pki").leaf("w1", "worker")
    fake = tmp_path / "fake-rpc-server"
    fake.write_text(f"#!{sys.executable}\nimport time\ntime.sleep(600)\n")
    fake.chmod(0o755)
    argv = [
        sys.executable,
        "-c",
        NO_PDEATHSIG_AGENT.format(repo = str(Path(__file__).resolve().parents[2])),
        json.dumps(dataclasses.asdict(me)),
        str(fake),
        str(tmp_path / "cache"),
    ]
    first = subprocess.Popen(argv, stdout = subprocess.PIPE, text = True)
    server = None
    try:
        server = psutil.Process(int(first.stdout.readline()))
        # Its own --cache-dir: the ownership record must not live in either cache.
        second = subprocess.run(
            argv[:-1] + [str(tmp_path / "cache-2")], capture_output = True, text = True, timeout = 30
        )
        assert second.returncode != 0 and "already runs ggml-rpc-server" in second.stderr
        assert server.is_running() and server.status() != "zombie"
    finally:
        first.kill()
        first.wait(10)
        if server is not None:
            try:
                server.kill()
            except psutil.NoSuchProcess:
                pass


def test_rpc_server_dies_with_the_agent_even_when_the_agent_has_threads(tmp_path):
    # preexec_fn is unsafe once the process has threads (the LAN announcer starts first): the launch
    # must still tie rpc-server to the agent, through setpriv when it is there.
    import dataclasses
    import json
    import signal
    import subprocess

    from studio.cluster_link import agent as agent_mod

    if not sys.platform.startswith("linux") or agent_mod._setpriv() is None:
        pytest.skip("needs Linux with util-linux setpriv --pdeathsig")
    me = Pki(tmp_path / "pki").leaf("w1", "worker")
    fake = tmp_path / "fake-rpc-server"
    fake.write_text(f"#!{sys.executable}\nimport time\ntime.sleep(600)\n")
    fake.chmod(0o755)
    script = THREADED_AGENT.format(repo = str(Path(__file__).resolve().parents[2]))
    proc = subprocess.Popen(
        [sys.executable, "-c", script, json.dumps(dataclasses.asdict(me)), str(fake)],
        stdout = subprocess.PIPE,
        text = True,
    )
    try:
        child = int(proc.stdout.readline())
        proc.send_signal(signal.SIGKILL)
        proc.wait(10)

        def gone() -> bool:
            try:
                os.kill(child, 0)
            except ProcessLookupError:
                return True
            return False

        assert _wait(gone, 10.0), f"rpc-server {child} outlived its threaded agent"
    finally:
        if proc.poll() is None:
            proc.kill()


THREADED_AGENT = """
import asyncio, json, sys, threading, time
sys.path.insert(0, {repo!r})
from studio.cluster_link.agent import Agent
from studio.cluster_link.model import LocalIdentity
threading.Thread(target = time.sleep, args = (600,), daemon = True).start()  # like the announcer
me = LocalIdentity(**json.loads(sys.argv[1]))
agent = Agent(me, lambda pid: True, host = "127.0.0.1", port = 0, rpc_binary = sys.argv[2], rpc_port = 1)
async def main():
    await agent.start()
    while agent.rpc_pid is None:
        await asyncio.sleep(0.05)
    print(agent.rpc_pid, flush = True)
    await asyncio.sleep(600)
asyncio.run(main())
"""


@pytest.mark.skipif(os.name == "nt", reason = "the fake rpc-server is a shebang script")
def test_two_workers_starting_at_once_on_one_rpc_port_one_wins(tmp_path):
    # Both used to pass the "is the port owned?" check before either recorded itself, then fought
    # over the port forever. The per-port lock lets exactly one start.
    import dataclasses
    import json
    import subprocess

    me = Pki(tmp_path / "pki").leaf("w1", "worker")
    fake = tmp_path / "fake-rpc-server"
    fake.write_text(f"#!{sys.executable}\nimport time\ntime.sleep(600)\n")
    fake.chmod(0o755)
    script = NO_PDEATHSIG_AGENT.format(repo = str(Path(__file__).resolve().parents[2]))
    procs = [
        subprocess.Popen(
            [
                sys.executable,
                "-c",
                script,
                json.dumps(dataclasses.asdict(me)),
                str(fake),
                str(tmp_path / f"c{i}"),
            ],
            stdout = subprocess.PIPE,
            stderr = subprocess.PIPE,
            text = True,
        )
        for i in range(2)
    ]
    children = []
    try:
        time.sleep(3)
        alive = [p for p in procs if p.poll() is None]
        dead = [p for p in procs if p.poll() is not None]
        assert len(alive) == 1 and len(dead) == 1, [p.poll() for p in procs]
        assert "already runs ggml-rpc-server" in dead[0].stderr.read()
        children.append(int(alive[0].stdout.readline()))
    finally:
        for p in procs:
            if p.poll() is None:
                p.kill()
                p.wait(10)
        for pid in children:
            try:
                os.kill(pid, 9)
            except ProcessLookupError:
                pass


def test_the_rpc_port_lock_admits_one_holder_at_a_time(tmp_path):
    # The lock itself, no child process: runs natively on every OS (flock / msvcrt).
    from studio.cluster_link import agent as agent_mod

    first = agent_mod._lock_rpc_port(50999)
    try:
        with pytest.raises(RuntimeError, match = "already runs ggml-rpc-server"):
            agent_mod._lock_rpc_port(50999)
        other = agent_mod._lock_rpc_port(50998)  # another port is independent
        agent_mod._unlock(other)
    finally:
        agent_mod._unlock(first)
    agent_mod._unlock(agent_mod._lock_rpc_port(50999))  # free again once released


def test_links_send_no_session_tickets(tmp_path):
    # Links never resume, so the two TLS 1.3 tickets a server sends by default are wasted records.
    pki = Pki(tmp_path / "pki")
    worker, master = pki.leaf("w1", "worker"), pki.leaf("m1", "master")
    server_ctx = tls.mutual_context(
        server = True, certfile = worker.certfile, keyfile = worker.keyfile, rootfile = worker.rootfile
    )
    client_ctx = tls.mutual_context(
        server = False, certfile = master.certfile, keyfile = master.keyfile, rootfile = master.rootfile
    )
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


def test_a_client_reconnecting_at_once_is_served(make_link):
    # The busy slot is released when the splice ends, before the upstream socket finishes closing,
    # so a llama-server that reconnects straight away must be served, not refused as busy.
    link = make_link()
    for i in range(5):
        payload = os.urandom(1000 + i)
        sock = link.connect()
        sock.sendall(payload)
        assert _recv_exact(sock, len(payload)) == payload, f"session {i}"
        sock.close()
    assert link.target.connections == 5


def test_a_client_reconnecting_while_its_last_session_drains_is_served(make_link):
    # The previous session ends with megabytes still in flight: its teardown must not keep the slot
    # past BUSY_GRACE_S and refuse the reconnect, and none of its bytes may reach the new session.
    link = make_link()
    first = link.connect()
    first.sendall(os.urandom(8 * MIB))
    first.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
    first.close()
    payload = os.urandom(4096)
    second = link.connect()
    second.sendall(payload)
    second.settimeout(10)
    assert _recv_exact(second, len(payload)) == payload
    second.close()


def test_the_link_trace_records_both_hops_only_when_asked(make_link, tmp_path, monkeypatch):
    link = make_link()
    assert _echo(link, b"untraced") == b"untraced"
    trace = tmp_path / "trace.jsonl"
    monkeypatch.setenv("UNSLOTH_CLUSTER_LINK_TRACE", str(trace))
    sock = link.connect()
    for i in range(20):
        sock.sendall(b"q" * (i + 1))
        assert _recv_exact(sock, i + 1) == b"q" * (i + 1)
    sock.close()
    assert _wait(
        lambda: trace.exists() and len(trace.read_text(encoding = "utf-8").splitlines()) == 2
    )
    import json

    rows = [json.loads(line) for line in trace.read_text(encoding = "utf-8").splitlines()]
    keys = sorted(k for row in rows for k in row["dirs"])
    assert keys == ["from_master", "from_worker", "to_master", "to_worker"]
    for row in rows:
        # The relay reads the client's first byte itself (the port-check guard) before the splice.
        assert (
            2 * sum(range(1, 21)) - 1
            <= sum(d["bytes"] for d in row["dirs"].values())
            <= 2 * sum(range(1, 21))
        )
        assert all(v["n"] >= 1 for v in row["turnaround_us"].values())


# Threaded sessions: the default byte path once a session is admitted (splice.socket_splice).

_SESSION_THREADS = (
    "unsloth-cluster-agent-conn",
    "unsloth-cluster-relay-conn",
    "unsloth-cluster-agent-revalidate",
    "unsloth-cluster-relay-revalidate",
    "unsloth-cluster-splice",
)


def _session_threads():
    return [t for t in threading.enumerate() if t.name in _SESSION_THREADS and t.is_alive()]


@pytest.mark.parametrize("mode", ["threaded", "async"])
def test_sessions_take_the_path_the_kill_switch_selects(make_link, monkeypatch, mode):
    if mode == "async":
        monkeypatch.delenv("UNSLOTH_CLUSTER_LINK_THREADED_SPLICE", raising = False)
    else:
        monkeypatch.setenv("UNSLOTH_CLUSTER_LINK_THREADED_SPLICE", "1")
    link = make_link()
    payload = os.urandom(MIB + 5)
    assert _echo(link, payload, fragment = True) == payload
    want = 1 if mode == "threaded" else 0
    assert link.agent.threaded is (mode == "threaded") and link.relay.threaded is (
        mode == "threaded"
    )
    assert link.agent.stats.get("threaded_streams", 0) == want
    assert link.relay.counters("w1").get("threaded_streams", 0) == want
    counters = link.relay.counters("w1")
    assert counters["to_worker"] == len(payload) and counters["from_worker"] == len(payload)


def test_a_threaded_session_carries_a_bulk_reply_byte_exact_both_ways(make_link, monkeypatch):
    # Upload and download at once, each fragmented: the one thread per session must never stall a
    # direction behind the other or reorder bytes.
    monkeypatch.setenv("UNSLOTH_CLUSTER_LINK_THREADED_SPLICE", "1")
    link = make_link()
    payload = os.urandom(8 * MIB + 3)
    assert _echo(link, payload, fragment = True) == payload
    assert link.agent.stats["threaded_streams"] == 1


@pytest.mark.parametrize("side", ["relay", "agent"])
def test_revocation_cuts_a_live_threaded_session(make_link, monkeypatch, side):
    import studio.cluster_link.agent as agent_mod
    import studio.cluster_link.relay as relay_mod

    monkeypatch.setattr(relay_mod, "REVALIDATE_S", 0.2)
    monkeypatch.setattr(agent_mod, "REVALIDATE_S", 0.2)
    monkeypatch.setenv("UNSLOTH_CLUSTER_LINK_THREADED_SPLICE", "1")
    trusted = {"ok": True}
    if side == "relay":
        link = make_link(relay_authorize = lambda pid: trusted["ok"] and pid.node_id == "w1")
    else:
        link = make_link(agent_authorize = lambda pid: trusted["ok"] and pid.node_id == "m1")
    sock = link.connect()
    sock.sendall(b"ping")
    assert _recv_exact(sock, 4) == b"ping"
    assert link.agent.stats["threaded_streams"] == 1
    trusted["ok"] = False
    assert _closed(sock, 5.0), f"the {side} kept a revoked session open"
    assert _wait(lambda: link.target.open == 0)
    assert _wait(lambda: not link.agent.busy and not link.relay._relays["w1"].active)
    assert _wait(lambda: not _session_threads())


def _fds() -> int:
    return len(os.listdir("/proc/self/fd"))


def test_abrupt_disconnects_leave_no_session_threads_or_sockets(make_link, monkeypatch):
    monkeypatch.setenv("UNSLOTH_CLUSTER_LINK_THREADED_SPLICE", "1")
    link = make_link()
    linux = sys.platform.startswith("linux")
    before = _fds() if linux else 0
    for _ in range(5):
        sock = link.connect()
        sock.sendall(b"abc")
        assert _recv_exact(sock, 3) == b"abc"
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
        sock.close()  # RST from the client mid-session
        assert _wait(lambda: link.target.open == 0)
        assert _wait(lambda: not link.agent.busy and not link.relay._relays["w1"].active)
    dropper = make_link(target_mode = "drop", drop_after = 4)
    sock = dropper.connect()
    sock.sendall(b"abcd")
    assert _closed(sock)  # the rpc-server end vanishes mid-session
    sock.close()
    assert _wait(lambda: not _session_threads(), 10.0), [t.name for t in _session_threads()]
    for lk in (link, dropper):
        assert not lk.agent._sessions and not lk.agent._raw
        relay = lk.relay._relays["w1"]
        assert not relay.sessions and not relay.raw
    if linux:
        # The second link's own listeners and loop sockets are the only growth allowed.
        assert _wait(lambda: _fds() <= before + 12), (before, _fds())


def test_concurrent_ensure_makes_one_threaded_listener(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_CLUSTER_LINK_THREADED_SPLICE", "1")
    pki = Pki(tmp_path / "pki")
    relays = RelayManager(pki.leaf("m1", "master"), lambda pid: True)
    try:
        endpoints, gate = [], threading.Barrier(8)

        def ensure():
            gate.wait()
            endpoints.append(relays.ensure("w1", "127.0.0.1", 50070))

        threads = [threading.Thread(target = ensure) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(30)
        assert len(endpoints) == 8 and set(endpoints) == {relays.endpoints()["w1"]}
        accept = [
            t
            for t in threading.enumerate()
            if t.name == "unsloth-cluster-relay-accept" and t.is_alive()
        ]
        assert len(accept) == 1
    finally:
        relays.stop()
    assert _wait(
        lambda: not [t for t in threading.enumerate() if t.name == "unsloth-cluster-relay-accept"]
    )


# Native mode: llama.cpp's own TLS rpc-server on request ---------------------------------------


def _fake_rpc_binary(
    tmp: Path,
    tls_build: bool,
    tls_fails: bool = False,
) -> Path:
    """A ggml-rpc-server stand-in: `--help` lists the TLS flags only for a TLS build; otherwise it logs
    its argv (one JSON line per start), binds -H/-p and accepts forever."""
    import textwrap

    tmp.mkdir(parents = True, exist_ok = True)
    log = tmp / "argv.jsonl"
    fake = tmp / "fake-rpc-server"
    usage = "  --tls-cert F  --tls-key F  --tls-ca F  --tls-allow F" if tls_build else ""
    fake.write_text(
        textwrap.dedent(f"""\
        #!{sys.executable}
        import json, socket, sys
        if "--help" in sys.argv:
            print("usage: ggml-rpc-server -H host -p port{usage}")
            sys.exit(0)
        with open({str(log)!r}, "a") as f:
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


def _argv_log(fake: Path) -> list:
    import json

    path = fake.parent / "argv.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _native_link(
    make_link,
    tmp_path,
    tls_build = True,
    tls_fails = False,
    **kw,
) -> Any:
    fake = _fake_rpc_binary(
        tmp_path / f"bin{len(list(tmp_path.glob('bin*')))}", tls_build, tls_fails
    )
    link = make_link(rpc_binary = str(fake), rpc_port = _free_port(), native_port = _free_port(), **kw)
    assert link.loop.run(link.agent.wait_rpc_ready(10.0))
    link.fake = fake
    return link


@pytest.mark.skipif(os.name == "nt", reason = "the fake rpc-server is a shebang script")
def test_native_tls_is_offered_only_by_a_tls_build(make_link, tmp_path):
    plain = _native_link(make_link, tmp_path, tls_build = False)
    assert plain.relay.status("w1")["native"]["available"] is False
    assert plain.relay.native("w1")["type"] == "error"
    tls_link = _native_link(make_link, tmp_path)
    assert tls_link.relay.status("w1")["native"] == {
        "available": True,
        "active": False,
        "port": tls_link.agent.native_port,
    }


@pytest.mark.skipif(os.name == "nt", reason = "the fake rpc-server is a shebang script")
def test_the_kill_switch_keeps_a_tls_build_on_the_relay(make_link, tmp_path, monkeypatch):
    from studio.cluster_link import native

    monkeypatch.setenv(native.DISABLE_ENV, "0")
    link = _native_link(make_link, tmp_path)
    assert link.agent.native_available is False
    assert link.relay.native("w1")["type"] == "error"


@pytest.mark.skipif(os.name == "nt", reason = "the fake rpc-server is a shebang script")
def test_a_native_request_restarts_rpc_server_with_tls_for_that_master_only(make_link, tmp_path):
    link = _native_link(make_link, tmp_path)
    reply = link.relay.native("w1")
    assert reply == {
        "type": "native_ok",
        "port": link.agent.native_port,
        "fingerprint": link.worker.fingerprint,
    }
    argv = _argv_log(link.fake)[-1]
    assert argv[argv.index("-p") + 1] == str(link.agent.native_port)
    assert argv[argv.index("--tls-cert") + 1] == link.worker.certfile
    assert argv[argv.index("--tls-key") + 1] == link.worker.keyfile
    assert argv[argv.index("--tls-ca") + 1] == link.worker.rootfile
    allow = Path(argv[argv.index("--tls-allow") + 1])
    assert allow.read_text().split() == [link.master.fingerprint]
    assert link.relay.status("w1")["native"]["active"] is True
    assert link.agent.stats["rpc_restarts"] == 0  # a planned switch, not a crash


@pytest.mark.skipif(os.name == "nt", reason = "the fake rpc-server is a shebang script")
def test_revoking_the_master_cuts_native_mode(make_link, tmp_path, monkeypatch):
    from studio.cluster_link import native

    import studio.cluster_link.agent as agent_mod

    monkeypatch.setattr(agent_mod, "REVALIDATE_S", 0.2)
    trusted = {"ok": True}
    link = _native_link(
        make_link, tmp_path, agent_authorize = lambda pid: trusted["ok"] and pid.node_id == "m1"
    )
    assert link.relay.native("w1")["type"] == "native_ok"
    pid_native = link.agent.rpc_pid
    allow = native.allowlist_path(link.agent.native_port)
    trusted["ok"] = False
    assert _wait(lambda: link.agent.mode == "loopback", 10.0)
    assert allow.read_text().split() == []
    assert _wait(lambda: link.agent.rpc_pid not in (None, pid_native), 10.0)
    assert _wait(lambda: "--tls-cert" not in _argv_log(link.fake)[-1], 10.0)
    argv = _argv_log(link.fake)[-1]
    assert argv[argv.index("-H") + 1] == "127.0.0.1" and argv[argv.index("-p") + 1] == str(
        link.agent.rpc_port
    )


@pytest.mark.skipif(os.name == "nt", reason = "the fake rpc-server is a shebang script")
def test_native_close_and_a_relay_session_both_return_to_loopback(make_link, tmp_path):
    from studio.cluster_link import native

    link = _native_link(make_link, tmp_path)
    assert link.relay.native("w1")["type"] == "native_ok"
    assert link.relay.native_close("w1") is True
    assert link.agent.mode == "loopback"
    assert native.allowlist_path(link.agent.native_port).read_text().split() == []
    # Back to native, then a relay session (an older master, or native turned off) takes it back.
    assert link.relay.native("w1")["type"] == "native_ok"
    sock = link.connect()
    sock.sendall(b"x")
    assert _wait(lambda: link.agent.mode == "loopback", 10.0)
    sock.close()
    assert _wait(lambda: "--tls-cert" not in _argv_log(link.fake)[-1], 10.0)


@pytest.mark.skipif(os.name == "nt", reason = "the fake rpc-server is a shebang script")
def test_a_removed_worker_still_takes_a_native_close(make_link, tmp_path):
    # The master cuts a native session of a worker it just removed: no longer authorized there, but
    # the certificate still names that node, and closing is all it is asked.
    link = _native_link(make_link, tmp_path, relay_authorize = lambda pid: False)
    assert link.relay.status("w1")["reachable"] is False  # status needs the worker still paired
    link.loop.run(
        link.agent._enter_native(tls.PeerIdentity(link.master.fingerprint, "m1", "master"))
    )
    assert link.agent.mode == "native"
    assert link.relay.native_close("w1") is True
    assert link.agent.mode == "loopback"


@pytest.mark.skipif(os.name == "nt", reason = "the fake rpc-server is a shebang script")
def test_a_failed_native_start_rolls_back_to_loopback(make_link, tmp_path, monkeypatch):
    # The TLS rpc-server dies at start: the master falls back to the relay, which needs the worker back
    # on its loopback rpc-server, not stuck in native mode reporting rpc down.
    from studio.cluster_link import native

    import studio.cluster_link.agent as agent_mod

    monkeypatch.setattr(agent_mod, "RPC_READY_TIMEOUT_S", 3.0)
    link = _native_link(make_link, tmp_path, tls_fails = True)
    assert link.relay.native("w1")["type"] == "error"
    assert link.agent.mode == "loopback"
    assert native.allowlist_path(link.agent.native_port).read_text().split() == []
    argv = _argv_log(link.fake)[-1]
    assert "--tls-cert" not in argv
    assert argv[argv.index("-H") + 1] == "127.0.0.1"
    assert link.loop.run(link.agent.wait_rpc_ready(10.0))
    assert link.relay.status("w1")["rpc_up"] is True


def test_the_relay_waits_out_a_native_to_loopback_restart():
    # open_rpc on a worker still in native mode restarts its rpc-server before answering.
    import studio.cluster_link.agent as agent_mod
    import studio.cluster_link.relay as relay_mod
    assert relay_mod.OPEN_RPC_REPLY_TIMEOUT_S > agent_mod.RPC_READY_TIMEOUT_S


@pytest.mark.skipif(os.name == "nt", reason = "the fake rpc-server is a shebang script")
def test_removing_a_native_worker_on_the_master_alone_cuts_its_session(
    make_link, tmp_path, monkeypatch
):
    # `unsloth cluster remove` only edits the master's registry; the native session does not pass
    # through the relay, so the relay manager's own watch must notice and close it within seconds.
    from studio.cluster_link import native

    import studio.cluster_link.relay as relay_mod

    monkeypatch.setattr(relay_mod, "REVALIDATE_S", 0.2)
    trusted = {"ok": True}
    link = _native_link(
        make_link, tmp_path, relay_authorize = lambda pid: trusted["ok"] and pid.node_id == "w1"
    )
    assert link.relay.native("w1")["type"] == "native_ok"
    pid_native = link.agent.rpc_pid
    trusted["ok"] = False
    assert _wait(lambda: link.agent.mode == "loopback", 10.0)
    assert native.allowlist_path(link.agent.native_port).read_text().split() == []
    assert _wait(lambda: link.agent.rpc_pid not in (None, pid_native), 10.0)


@pytest.mark.skipif(os.name == "nt", reason = "the fake rpc-server is a shebang script")
def test_a_failed_native_revocation_is_retried_until_the_worker_answers(
    make_link, tmp_path, monkeypatch
):
    # The first native_close after a remove cannot reach the agent (a blip on the LAN): the watch must
    # keep asking, or the worker, which still trusts this master, keeps its native session forever.
    from studio.cluster_link import native

    import studio.cluster_link.relay as relay_mod

    monkeypatch.setattr(relay_mod, "REVALIDATE_S", 0.2)
    monkeypatch.setattr(relay_mod, "NATIVE_CLOSE_RETRY_MAX_S", 0.4, raising = False)
    trusted = {"ok": True}
    link = _native_link(
        make_link, tmp_path, relay_authorize = lambda pid: trusted["ok"] and pid.node_id == "w1"
    )
    assert link.relay.native("w1")["type"] == "native_ok"
    pid_native = link.agent.rpc_pid
    plain, failures = link.relay._control, []

    async def flaky(
        node,
        request,
        timeout,
        paired = True,
    ):
        if request.get("type") == "native_close" and len(failures) < 2:
            failures.append(request)
            raise OSError("network unreachable")
        return await plain(node, request, timeout, paired = paired)

    link.relay._control = flaky
    trusted["ok"] = False
    assert _wait(lambda: link.agent.mode == "loopback", 15.0)
    assert len(failures) == 2
    assert native.allowlist_path(link.agent.native_port).read_text().split() == []
    assert _wait(lambda: link.agent.rpc_pid not in (None, pid_native), 10.0)
    assert _wait(lambda: "w1" not in link.relay._native_watches, 5.0)


@pytest.mark.skipif(os.name == "nt", reason = "the fake rpc-server is a shebang script")
def test_a_failed_explicit_native_close_keeps_the_revocation_watch(
    make_link, tmp_path, monkeypatch
):
    # Studio's removal path calls native_close() itself; if that one attempt cannot reach the agent, the
    # watch must survive and retry, or the removed worker keeps its native session for good.
    from studio.cluster_link import native

    import studio.cluster_link.relay as relay_mod

    monkeypatch.setattr(relay_mod, "REVALIDATE_S", 0.2)
    monkeypatch.setattr(relay_mod, "NATIVE_CLOSE_RETRY_MAX_S", 0.4, raising = False)
    trusted = {"ok": True}
    link = _native_link(
        make_link, tmp_path, relay_authorize = lambda pid: trusted["ok"] and pid.node_id == "w1"
    )
    assert link.relay.native("w1")["type"] == "native_ok"
    pid_native = link.agent.rpc_pid
    plain, failures = link.relay._control, []

    async def flaky(
        node,
        request,
        timeout,
        paired = True,
    ):
        if request.get("type") == "native_close" and not failures:
            failures.append(request)
            raise OSError("network unreachable")
        return await plain(node, request, timeout, paired = paired)

    link.relay._control = flaky
    assert link.relay.native_close("w1") is False  # the explicit close fails once
    trusted["ok"] = False  # the worker is removed on the master
    link.relay.remove("w1")  # as _drop_stale_relays does next
    assert _wait(lambda: link.agent.mode == "loopback", 15.0)
    assert native.allowlist_path(link.agent.native_port).read_text().split() == []
    assert _wait(lambda: link.agent.rpc_pid not in (None, pid_native), 10.0)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason = "owner lookup is Linux-only")
def test_the_relay_serves_only_local_clients_of_this_user(make_link, monkeypatch):
    # Every local account can reach a 127.0.0.1 port, and the relay proxies with this user's cluster
    # credentials: a client owned by another uid gets nothing, and the worker never hears of it.
    import studio.cluster_link.splice as splice_mod

    link = make_link()
    assert _echo(link, b"mine") == b"mine"  # this user's own client
    before = link.agent.stats["connections"]
    monkeypatch.setattr(splice_mod, "_loopback_owner_uid", lambda c, s: os.getuid() + 1)
    sock = link.connect()
    sock.sendall(b"steal")
    assert _closed(sock)
    time.sleep(0.3)
    assert link.agent.stats["connections"] == before
    assert link.target.connections == 1


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason = "owner lookup is Linux-only")
def test_the_loopback_owner_of_a_real_connection_is_found():
    from studio.cluster_link.splice import _loopback_owner_uid, local_client_is_ours

    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    client = socket.create_connection(server.getsockname())
    conn, _ = server.accept()
    try:
        assert _loopback_owner_uid(client.getsockname()[1], server.getsockname()[1]) == os.getuid()
        assert local_client_is_ours(conn)
    finally:
        for s in (client, conn, server):
            s.close()
