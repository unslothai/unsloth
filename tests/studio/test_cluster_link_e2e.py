# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""CPU-only, one host: a master and two paired workers; a layer split through the relays (and over
native TLS) must match direct RPC, and an unpaired client or a non-loopback dial gets no RPC."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import socket
import ssl
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[2]
LINK = REPO / "studio" / "cluster_link"
# A CPU llama.cpp build with GGML_RPC=ON and a small GGUF; the test skips without them.
BIN_DIR = Path(os.environ.get("UNSLOTH_TEST_LLAMA_BIN_DIR") or "/nonexistent")
MODEL = Path(os.environ.get("UNSLOTH_TEST_TINY_GGUF") or "/nonexistent")
LLAMA_SERVER = BIN_DIR / "llama-server"
RPC_SERVER = BIN_DIR / "ggml-rpc-server"
HOSTS = ["127.0.0.2", "127.0.0.3"]

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

MASTER_PAIR = PRELUDE + """
from studio.cluster_link.pairing import PairingWindow
async def main():
    w = PairingWindow(identity.master_identity(create = True), host = "127.0.0.1",
                      port = int(sys.argv[1]), ttl = 60, addrs = ["127.0.0.1"])
    print("TOKEN " + await w.start(), flush = True)
    peer = await w.wait(60)
    w.close()
    print("PAIRED " + peer.node_id, flush = True)
asyncio.run(main())
"""  # fmt: skip

WORKER_JOIN = PRELUDE + """
from studio.cluster_link.pairing import join
me = asyncio.run(join(sys.stdin.readline().strip(), name = sys.argv[1]))
print("JOINED " + me.node_id, flush = True)
"""  # fmt: skip

WORKER_AGENT = PRELUDE + """
from studio.cluster_link.agent import Agent
me = identity.worker_identity()
def authorize(peer):
    return registry().authorize(peer, "master") is not None
agent = Agent(me, authorize, host = sys.argv[1], port = int(sys.argv[2]), rpc_binary = sys.argv[3],
              rpc_port = int(sys.argv[4]), cache_dir = sys.argv[5], rpc_args = ("-t", "4"),
              native_port = int(sys.argv[6]) if len(sys.argv) > 6 else 50053)
async def main():
    await agent.start()
    print("READY", flush = True)
    await agent.serve_forever()
asyncio.run(main())
"""  # fmt: skip


def free_port(host = "127.0.0.1"):
    with socket.socket() as s:
        s.bind((host, 0))
        return s.getsockname()[1]


def node_env(state: Path, **extra) -> dict:
    env = dict(os.environ, GGML_RPC_NO_RDMA = "1", **extra)
    env.update(UNSLOTH_CLUSTER_STATE_DIR = str(state), UNSLOTH_CLUSTER_RUN_DIR = str(state / "run"))
    return env


def wait_line(proc, prefix):
    deadline = time.time() + 60.0
    while time.time() < deadline:
        line = proc.stdout.readline()
        if not line:
            if proc.poll() is not None:
                raise AssertionError(
                    f"exited {proc.returncode} before {prefix!r}: {log_tail(proc)}"
                )
            continue
        if line.startswith(prefix):
            return line[len(prefix) :].strip()
    raise AssertionError(f"timed out waiting for {prefix!r}")


def wait_port(port, host = "127.0.0.1"):
    deadline = time.time() + 30.0
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


def spawn(
    procs,
    argv,
    env,
    *,
    quiet = False,
):
    # Undrained pipes block the child (llama-server stops mid-load): output goes to a file.
    log = tempfile.NamedTemporaryFile("w+", prefix = "e2e_", suffix = ".log", delete = False)
    stdout = log if quiet else subprocess.PIPE
    p = subprocess.Popen(argv, env = env, stdout = stdout, stderr = log, text = True)
    p.log_path = log.name
    procs.append(p)
    return p


def log_tail(p, n = 3000) -> str:
    try:
        return Path(p.log_path).read_text(errors = "replace")[-n:]
    except OSError:
        return ""


def pair(procs, master_state: Path, worker_state: Path, name: str) -> str:
    master = spawn(
        procs, [sys.executable, "-c", MASTER_PAIR, str(free_port())], node_env(master_state)
    )
    token = wait_line(master, "TOKEN ")
    joiner = subprocess.run(
        [sys.executable, "-c", WORKER_JOIN, name], input = token + "\n", env = node_env(worker_state),
        capture_output = True, text = True, timeout = 60,
    )  # fmt: skip
    assert joiner.returncode == 0, joiner.stderr[-2000:]
    assert token not in joiner.stdout + joiner.stderr
    return wait_line(master, "PAIRED ")


def start_workers(procs, tmp_path, native_ports):
    """Pair two workers with tmp_path/master and start their agents on HOSTS."""
    states = [tmp_path / "w1", tmp_path / "w2"]
    ids = [pair(procs, tmp_path / "master", s, f"w{i + 1}") for i, s in enumerate(states)]
    assert len(set(ids)) == 2
    w = SimpleNamespace(ids = ids, agent_ports = [], rpc_ports = [], agents = [])
    for i, (state, host) in enumerate(zip(states, HOSTS)):
        ap, rp = free_port(host), free_port()
        argv = [sys.executable, "-c", WORKER_AGENT, host, str(ap), str(RPC_SERVER), str(rp)]
        argv += [str(state / "cache")] + ([str(native_ports[i])] if native_ports else [])
        agent = spawn(procs, argv, node_env(state))
        wait_line(agent, "READY")
        w.agents.append(agent)
        w.agent_ports.append(ap)
        w.rpc_ports.append(rp)
    return w


def master_relays(monkeypatch, master_state):
    """The master's identity, registry and RelayManager, read from master_state in this process."""
    monkeypatch.setenv("UNSLOTH_CLUSTER_STATE_DIR", str(master_state))
    sys.path.insert(0, str(REPO))
    import studio

    if str(REPO / "studio") not in list(studio.__path__):
        studio.__path__.append(str(REPO / "studio"))
    from studio.cluster_link import identity
    from studio.cluster_link.relay import RelayManager

    def registry():
        try:
            return identity.Registry.load()
        except TypeError:
            r = identity.Registry()
            r.load()
            return r

    master = identity.master_identity(create = False)
    relays = RelayManager(master, lambda peer: registry().authorize(peer, "worker") is not None)
    return master, registry, relays


def complete(procs, rpc: str, env: dict) -> str:
    port = free_port()
    # Few threads on both sides: a busy shared host makes 96 spinning threads crawl.
    flags = "--device RPC0,RPC1 -sm layer -ngl 99 -c 512 -np 1 -t 4 --host 127.0.0.1 --port"
    argv = [str(LLAMA_SERVER), "-m", str(MODEL), "--rpc", rpc, *flags.split(), str(port)]
    server = spawn(procs, argv, env, quiet = True)
    deadline, healthy = time.time() + 180, False
    while time.time() < deadline and not healthy:
        assert server.poll() is None, log_tail(server)
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout = 2) as r:
                healthy = r.status == 200
        except Exception:
            time.sleep(0.5)
    assert healthy, "llama-server never became healthy:\n" + log_tail(server)
    body = {
        "prompt": "The capital of France is",
        "n_predict": 24,
        "temperature": 0,
        "cache_prompt": False,
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/completion",
        data = json.dumps(body).encode(),
        headers = {"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout = 300) as r:
        text = json.loads(r.read())["content"]
    server.send_signal(signal.SIGTERM)
    server.wait(timeout = 30)
    return text


def test_a_split_through_paired_relays_matches_direct_rpc(tmp_path, procs, monkeypatch):
    master_state = tmp_path / "master"
    w = start_workers(procs, tmp_path, None)
    # rpc-server is bound to 127.0.0.1 only: the agent's address does not reach it.
    for rp in w.rpc_ports:
        wait_port(rp)
        with pytest.raises(OSError):
            socket.create_connection(("127.0.0.2", rp), timeout = 2).close()

    _, _, relays = master_relays(monkeypatch, master_state)
    from studio.cluster_link import tls

    try:
        endpoints = [relays.ensure(i, h, p) for i, h, p in zip(w.ids, HOSTS, w.agent_ports)]
        for node_id in w.ids:
            st = relays.status(node_id, timeout = 10.0)
            assert st.get("rpc_up") and not st.get("busy"), st
        linked = complete(procs, ",".join(endpoints), node_env(master_state))
    finally:
        relays.stop()

    direct_ports = [free_port(), free_port()]
    for dp in direct_ports:
        # The same thread count as the agents' rpc-servers: the two arms must differ only in transport.
        argv = [str(RPC_SERVER), "-H", "127.0.0.1", "-p", str(dp), "-t", "4"]
        spawn(procs, argv, node_env(tmp_path / "direct"))
        wait_port(dp)
    rpc = ",".join(f"127.0.0.1:{p}" for p in direct_ports)
    direct = complete(procs, rpc, node_env(tmp_path / "direct"))
    assert linked and linked == direct

    async def unpaired() -> bytes:
        ctx = tls.bootstrap_client_context()
        reader, writer = await asyncio.open_connection(HOSTS[0], w.agent_ports[0], ssl = ctx)
        writer.write(b"\x00\x00\x00\x13" + b'{"type":"open_rpc"}')
        await writer.drain()
        return await asyncio.wait_for(reader.read(64), 5)

    # The agent must be listening, so a refusal below is the TLS layer, not a dead agent or closed port.
    socket.create_connection((HOSTS[0], w.agent_ports[0]), timeout = 5).close()
    try:
        answer = asyncio.run(unpaired())
    except (ssl.SSLError, ConnectionResetError, asyncio.IncompleteReadError, BrokenPipeError):
        answer = b""
    assert answer == b""
    assert w.agents[0].poll() is None, "the agent died instead of refusing the unpaired client"


def _tls_build() -> bool:
    try:
        out = subprocess.run(
            [str(RPC_SERVER), "--help"], capture_output = True, text = True, timeout = 15
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return "--tls-cert" in out.stdout + out.stderr


def test_a_split_over_native_tls_matches_the_relays(tmp_path, procs, monkeypatch):
    # llama-server dials tls://<fp>@host:port natively; greedy output must equal the relayed split.
    if not _tls_build():
        pytest.skip("this llama.cpp build has no TLS transport (GGML_RPC_TLS)")
    master_state = tmp_path / "master"
    native_ports = [free_port(h) for h in HOSTS]
    w = start_workers(procs, tmp_path, native_ports)

    master, registry, relays = master_relays(monkeypatch, master_state)
    peers = {p.node_id: p for p in registry().workers()}
    try:
        relay_endpoints = [relays.ensure(i, h, p) for i, h, p in zip(w.ids, HOSTS, w.agent_ports)]
        tls_endpoints = []
        for node_id, host, port in zip(w.ids, HOSTS, native_ports):
            assert relays.status(node_id, timeout = 10.0)["native"]["available"] is True
            reply = relays.native(node_id)
            assert reply["type"] == "native_ok", reply
            assert reply["fingerprint"] == peers[node_id].fingerprint and reply["port"] == port
            tls_endpoints.append(f"tls://{reply['fingerprint']}@{host}:{port}")
        env = node_env(
            master_state, GGML_RPC_TLS_CERT = master.certfile, GGML_RPC_TLS_KEY = master.keyfile,
            GGML_RPC_TLS_CA = master.rootfile,
        )  # fmt: skip
        native_text = complete(procs, ",".join(tls_endpoints), env)
        for node_id in w.ids:
            assert relays.status(node_id, timeout = 10.0)["native"]["active"] is True
        # The relay path takes each worker back to its loopback rpc-server.
        relayed_text = complete(procs, ",".join(relay_endpoints), node_env(master_state))
        for node_id in w.ids:
            assert relays.status(node_id, timeout = 10.0)["native"]["active"] is False
    finally:
        relays.stop()
    assert native_text and native_text == relayed_text
    # Without the master certificate llama.cpp cannot reach a native rpc-server at all.
    stranger = subprocess.run(
        [str(LLAMA_SERVER), "--rpc", tls_endpoints[0], "--list-devices"],
        env = node_env(tmp_path / "stranger"), capture_output = True, text = True, timeout = 60,
    )  # fmt: skip
    assert tls_endpoints[0] not in stranger.stdout
