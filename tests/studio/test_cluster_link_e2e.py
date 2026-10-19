# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""End to end on one host, CPU only: a master and two paired workers (separate state dirs, agents on
127.0.0.2 / 127.0.0.3 with real ggml-rpc-servers on loopback), a forced two-worker layer split through
the relays, and the same split over direct RPC: greedy output must be identical. An unpaired client
gets no RPC, and a worker's rpc-server is not reachable beyond the address it is bound to."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
LINK = REPO / "studio" / "cluster_link"
BIN_DIR = Path(
    os.environ.get("UNSLOTH_TEST_LLAMA_BIN_DIR")
    or "/mnt/disks/unslothai/daniel2/workspace_34/temp/llama_cpu_builds/pr187"
)
MODEL = Path(
    os.environ.get("UNSLOTH_TEST_TINY_GGUF")
    or "/mnt/disks/unslothai/daniel2/workspace_34/temp/fake_gguf/qwen3-0.6b.gguf"
)
LLAMA_SERVER = BIN_DIR / "llama-server"
RPC_SERVER = BIN_DIR / "ggml-rpc-server"

pytestmark = pytest.mark.skipif(
    not all((LINK / f"{m}.py").is_file() for m in ("identity", "pairing", "agent", "relay"))
    or not LLAMA_SERVER.is_file()
    or not RPC_SERVER.is_file()
    or not MODEL.is_file(),
    reason = "needs studio/cluster_link identity/pairing/agent/relay and CPU llama.cpp binaries",
)

# Run inside each node's own process: its state dir is its identity.
PRELUDE = f"""
import asyncio, sys
sys.path.insert(0, {str(REPO)!r})
import studio
if {str(REPO / "studio")!r} not in list(studio.__path__):
    studio.__path__.append({str(REPO / "studio")!r})
from studio.cluster_link import identity

def registry():
    try:
        return identity.Registry.load()
    except TypeError:
        r = identity.Registry(); r.load(); return r
"""

MASTER_PAIR = (
    PRELUDE
    + """
from studio.cluster_link.pairing import PairingWindow
async def main():
    w = PairingWindow(identity.master_identity(create = True), host = "127.0.0.1",
                      port = int(sys.argv[1]), ttl = 60, addrs = ["127.0.0.1"])
    print("TOKEN " + await w.start(), flush = True)
    peer = await w.wait(60)
    w.close()
    print("PAIRED " + peer.node_id, flush = True)
asyncio.run(main())
"""
)

WORKER_JOIN = (
    PRELUDE
    + """
from studio.cluster_link.pairing import join
me = asyncio.run(join(sys.stdin.readline().strip(), name = sys.argv[1]))
print("JOINED " + me.node_id, flush = True)
"""
)

WORKER_AGENT = (
    PRELUDE
    + """
from studio.cluster_link.agent import Agent
me = identity.worker_identity()
def authorize(peer):
    return registry().authorize(peer, "master") is not None
agent = Agent(me, authorize, host = sys.argv[1], port = int(sys.argv[2]), rpc_binary = sys.argv[3],
              rpc_port = int(sys.argv[4]), cache_dir = sys.argv[5])
async def main():
    await agent.start()
    print("READY", flush = True)
    await agent.serve_forever()
asyncio.run(main())
"""
)


def free_port(host: str = "127.0.0.1") -> int:
    with socket.socket() as s:
        s.bind((host, 0))
        return s.getsockname()[1]


def node_env(state: Path) -> dict:
    env = dict(os.environ)
    env["UNSLOTH_CLUSTER_STATE_DIR"] = str(state)
    env["GGML_RPC_NO_RDMA"] = "1"
    return env


def wait_line(
    proc: subprocess.Popen,
    prefix: str,
    timeout: float = 60.0,
) -> str:
    deadline = time.time() + timeout
    while time.time() < deadline:
        line = proc.stdout.readline()
        if not line:
            if proc.poll() is not None:
                raise AssertionError(
                    f"exited {proc.returncode} before {prefix!r}: {proc.stderr.read()[-2000:]}"
                )
            continue
        if line.startswith(prefix):
            return line[len(prefix) :].strip()
    raise AssertionError(f"timed out waiting for {prefix!r}")


def wait_port(
    host: str,
    port: int,
    timeout: float = 30.0,
) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout = 1.0):
                return
        except OSError:
            time.sleep(0.2)
    raise AssertionError(f"{host}:{port} never opened")


@pytest.fixture
def procs():
    started = []
    yield started
    for p in started:
        if p.poll() is None:
            p.send_signal(signal.SIGTERM)
    for p in started:
        try:
            p.wait(timeout = 10)
        except subprocess.TimeoutExpired:
            p.kill()


def spawn(procs, argv, env, **kw):
    p = subprocess.Popen(
        argv, env = env, stdout = subprocess.PIPE, stderr = subprocess.PIPE, text = True, **kw
    )
    procs.append(p)
    return p


def pair(procs, master_state: Path, worker_state: Path, name: str) -> str:
    port = free_port()
    master = spawn(procs, [sys.executable, "-c", MASTER_PAIR, str(port)], node_env(master_state))
    token = wait_line(master, "TOKEN ")
    joiner = subprocess.run(
        [sys.executable, "-c", WORKER_JOIN, name],
        input = token + "\n",
        env = node_env(worker_state),
        capture_output = True,
        text = True,
        timeout = 60,
    )
    assert joiner.returncode == 0, joiner.stderr[-2000:]
    assert token not in joiner.stdout + joiner.stderr
    return wait_line(master, "PAIRED ")


def complete(procs, rpc: str, env: dict) -> str:
    port = free_port()
    server = spawn(
        procs,
        [
            str(LLAMA_SERVER),
            "-m",
            str(MODEL),
            "--rpc",
            rpc,
            "--device",
            "RPC0,RPC1",
            "-sm",
            "layer",
            "-ngl",
            "99",
            "-c",
            "512",
            "-np",
            "1",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        env,
    )
    deadline = time.time() + 180
    while time.time() < deadline:
        assert server.poll() is None, server.stderr.read()[-3000:]
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout = 2) as r:
                if r.status == 200:
                    break
        except Exception:
            time.sleep(0.5)
    body = json.dumps(
        {
            "prompt": "The capital of France is",
            "n_predict": 24,
            "temperature": 0,
            "cache_prompt": False,
        }
    ).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/completion",
        data = body,
        headers = {"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout = 120) as r:
        text = json.loads(r.read())["content"]
    server.send_signal(signal.SIGTERM)
    server.wait(timeout = 30)
    return text


def test_a_split_through_paired_relays_matches_direct_rpc(tmp_path, procs, monkeypatch):
    master_state, states = tmp_path / "master", [tmp_path / "w1", tmp_path / "w2"]
    ids = [pair(procs, master_state, s, f"w{i + 1}") for i, s in enumerate(states)]
    assert len(set(ids)) == 2

    hosts = ["127.0.0.2", "127.0.0.3"]
    agent_ports, rpc_ports = [], []
    for state, host in zip(states, hosts):
        ap, rp = free_port(host), free_port()
        agent = spawn(
            procs,
            [
                sys.executable,
                "-c",
                WORKER_AGENT,
                host,
                str(ap),
                str(RPC_SERVER),
                str(rp),
                str(state / "cache"),
            ],
            node_env(state),
        )
        wait_line(agent, "READY")
        agent_ports.append(ap)
        rpc_ports.append(rp)
    # rpc-server is bound to 127.0.0.1 only: the agent's address does not reach it.
    for rp in rpc_ports:
        wait_port("127.0.0.1", rp)
        with pytest.raises(OSError):
            socket.create_connection(("127.0.0.2", rp), timeout = 2).close()

    monkeypatch.setenv("UNSLOTH_CLUSTER_STATE_DIR", str(master_state))
    sys.path.insert(0, str(REPO))
    import studio

    if str(REPO / "studio") not in list(studio.__path__):
        studio.__path__.append(str(REPO / "studio"))
    from studio.cluster_link import identity, tls
    from studio.cluster_link.relay import RelayManager

    def registry():
        try:
            return identity.Registry.load()
        except TypeError:
            r = identity.Registry()
            r.load()
            return r

    relays = RelayManager(
        identity.master_identity(create = False),
        lambda peer: registry().authorize(peer, "worker") is not None,
    )
    try:
        endpoints = [relays.ensure(i, h, p) for i, h, p in zip(ids, hosts, agent_ports)]
        for node_id in ids:
            st = relays.status(node_id, timeout = 10.0)
            assert st.get("rpc_up") and not st.get("busy"), st
        linked = complete(procs, ",".join(endpoints), node_env(master_state))
    finally:
        relays.stop()

    # Same split over direct RPC on fresh loopback rpc-servers.
    direct_ports = [free_port(), free_port()]
    for dp in direct_ports:
        spawn(
            procs,
            [str(RPC_SERVER), "-H", "127.0.0.1", "-p", str(dp)],
            node_env(tmp_path / "direct"),
        )
        wait_port("127.0.0.1", dp)
    direct = complete(
        procs, ",".join(f"127.0.0.1:{p}" for p in direct_ports), node_env(tmp_path / "direct")
    )
    assert linked and linked == direct

    # An unpaired client (no certificate) gets nothing from an agent.
    async def unpaired() -> bytes:
        reader, writer = await asyncio.open_connection(
            hosts[0], agent_ports[0], ssl = tls.bootstrap_client_context()
        )
        writer.write(b"\x00\x00\x00\x13" + b'{"type":"open_rpc"}')
        await writer.drain()
        return await asyncio.wait_for(reader.read(64), 5)

    try:
        answer = asyncio.run(unpaired())
    except Exception:
        answer = b""
    assert answer == b""
