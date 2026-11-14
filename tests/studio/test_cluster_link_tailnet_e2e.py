# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Real unsloth-link helpers on one host with a shared loopback DERP: pairing whose LAN address cannot
be reached goes through tailcat, workers then serve through their own tunnels, and a layer split over
them matches direct RPC. Skipped unless UNSLOTH_LINK_BIN names a built helper."""

from __future__ import annotations

import os
import re
import socket
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_cluster_link_e2e as e2e  # noqa: E402
from test_cluster_link_e2e import procs  # noqa: E402,F401

REPO = e2e.REPO
HELPER = Path(os.environ.get("UNSLOTH_LINK_BIN") or "/nonexistent")
UNREACHABLE = "192.0.2.10"  # TEST-NET-1: the LAN path must fail, so pairing has to use tailcat

pytestmark = pytest.mark.skipif(
    not HELPER.is_file(), reason = "needs a built unsloth-link (UNSLOTH_LINK_BIN)"
)

CLI = "import sys; sys.argv[0] = 'unsloth'; from unsloth_cli import app; app()"

# Bound to 127.0.0.1 only, so the LAN addresses the worker reported never reach it: only its tunnel does.
WORKER_AGENT = e2e.PRELUDE + """
from studio.cluster_link.agent import Agent
me = identity.worker_identity()
def authorize(peer):
    return registry().authorize(peer, "master") is not None
def tunnel_allow():
    m = registry().master()
    return [m.tc_pub] if m is not None and m.tc_pub else []
agent = Agent(me, authorize, host = "127.0.0.1", port = int(sys.argv[1]), rpc_binary = sys.argv[2],
              rpc_port = int(sys.argv[3]), cache_dir = sys.argv[4], rpc_args = ("-t", "4"),
              tunnel_allow = tunnel_allow)
async def main():
    await agent.start()
    print("READY " + ("tunnel" if agent.tunnel is not None else "lan-only"), flush = True)
    await agent.serve_forever()
asyncio.run(main())
"""  # fmt: skip


@pytest.fixture
def derp_map(procs):
    """A loopback DERP every helper in the test shares, through a DERP map URL like the public one."""
    p = subprocess.Popen(
        [str(HELPER), "devderp"], stdin = subprocess.PIPE, stdout = subprocess.PIPE, text = True
    )
    procs.append(p)
    url = p.stdout.readline().strip()
    assert url.startswith("http://127.0.0.1:"), url
    return url


def env(state: Path, derp: str, **extra) -> dict:
    out = e2e.node_env(state, UNSLOTH_LINK_BIN = str(HELPER), UNSLOTH_CLUSTER_DERP_MAP = derp, **extra)
    out.update(CUDA_VISIBLE_DEVICES = "", PYTHONPATH = str(REPO))
    out.pop("TS_DEBUG_TAILCAT_LOCAL_DERP", None)
    return out


def cli(
    state: Path,
    derp: str,
    *args: str,
    stdin: str = "",
    timeout: float = 120,
    **extra: str,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", CLI, *args], input = stdin, env = env(state, derp, **extra),
        capture_output = True, text = True, timeout = timeout, cwd = str(REPO),
    )  # fmt: skip


def pair_through_tailcat(
    procs, master: Path, worker: Path, name: str, derp: str, agent_port: int
) -> dict:
    """`cluster pair` with an unreachable --addr and `cluster join`: returns both sides' words and output."""
    port = e2e.free_port()
    argv = [sys.executable, "-c", CLI, "cluster", "pair", "--ttl", "120s", "--addr", UNREACHABLE]
    p = subprocess.Popen(
        argv + ["--port", str(port)], env = env(master, derp), stdin = subprocess.PIPE,
        stdout = subprocess.PIPE, stderr = subprocess.STDOUT, text = True, cwd = str(REPO),
    )  # fmt: skip
    procs.append(p)
    code = None
    lines = []
    while code is None:
        line = p.stdout.readline()
        assert line or p.poll() is None, "".join(lines)
        lines.append(line)
        if line.strip().startswith("unsloth-pair-"):
            code = line.strip()
    assert code.startswith("unsloth-pair-2."), "no tailcat address in the code:\n" + "".join(lines)
    p.stdin.write("y\n")
    p.stdin.flush()
    joined = cli(
        worker, derp, "cluster", "join", "--token-stdin", "--name", name, "--port", str(agent_port),
        stdin = code + "\n",
    )  # fmt: skip
    rest, _ = p.communicate(timeout = 180)
    master_out = "".join(lines) + rest
    assert p.returncode == 0, master_out
    assert joined.returncode == 0, joined.stdout + joined.stderr
    assert code not in joined.stdout + joined.stderr
    worker_words = re.search(r"allow it there: (\w+ \w+ \w+)", joined.stdout)
    master_words = re.search(rf"{name} wants to join \((\w+ \w+ \w+)\)", master_out)
    assert worker_words and master_words, (joined.stdout, master_out)
    return {"worker": worker_words.group(1), "master": master_words.group(1), "out": master_out}


def master_registry(monkeypatch, master: Path, derp: str):
    monkeypatch.setenv("UNSLOTH_LINK_BIN", str(HELPER))
    monkeypatch.setenv("UNSLOTH_CLUSTER_DERP_MAP", derp)
    monkeypatch.delenv("TS_DEBUG_TAILCAT_LOCAL_DERP", raising = False)
    return e2e.master_relays(monkeypatch, master)


def test_pairing_and_a_split_go_through_tailcat_when_the_lan_cannot(
    tmp_path, procs, monkeypatch, derp_map
):
    master = tmp_path / "master"
    states = [tmp_path / "w1", tmp_path / "w2"]
    agent_ports = [e2e.free_port(), e2e.free_port()]
    for i, state in enumerate(states):
        seen = pair_through_tailcat(procs, master, state, f"w{i + 1}", derp_map, agent_ports[i])
        assert seen["worker"] == seen["master"], seen

    _, registry, relays = master_registry(monkeypatch, master, derp_map)
    peers = {p.name: p for p in registry().workers()}
    assert sorted(peers) == ["w1", "w2"]
    for peer in peers.values():
        assert peer.tc_addr.startswith("tc") and peer.tc_pub.startswith("nodekey:")
        assert not any(a.startswith(("127.", UNREACHABLE)) for a in peer.addrs), peer.addrs
        assert peer.tc_addr not in repr(peer)

    if not (e2e.LLAMA_SERVER.is_file() and e2e.RPC_SERVER.is_file() and e2e.MODEL.is_file()):
        relays.stop()
        pytest.skip(
            reason = "pairing checked; the split needs UNSLOTH_TEST_LLAMA_BIN_DIR and UNSLOTH_TEST_TINY_GGUF"
        )

    for state, ap in zip(states, agent_ports):
        argv = [
            sys.executable,
            "-c",
            WORKER_AGENT,
            str(ap),
            str(e2e.RPC_SERVER),
            str(e2e.free_port()),
        ]
        agent = e2e.spawn(procs, argv + [str(state / "cache")], env(state, derp_map))
        assert e2e.wait_line(agent, "READY ") == "tunnel", e2e.log_tail(agent)

    listed = cli(master, derp_map, "cluster", "list", timeout = 180)
    assert listed.returncode == 0, listed.stdout + listed.stderr
    for name in ("w1", "w2"):
        line = next(l for l in listed.stdout.splitlines() if l.startswith(name))
        assert "through tailcat" in line and "ready (direct internet connection)" in line, (
            listed.stdout + listed.stderr
        )

    from studio.cluster_link import remote

    try:
        endpoints = []
        for name in ("w1", "w2"):
            peer = peers[name]
            local = remote.forward(peer.node_id, peer.tc_addr, peer.agent_port)
            assert local, f"no tailcat forward to {name}"
            assert remote.path(peer.node_id).direct
            endpoints.append(relays.ensure(peer.node_id, "127.0.0.1", local))
            st = relays.status(peer.node_id, timeout = 15.0)
            assert st.get("rpc_up") and not st.get("busy"), st
        tunnelled = e2e.complete(procs, ",".join(endpoints), e2e.node_env(master))
    finally:
        relays.stop()

    direct_ports = [e2e.free_port(), e2e.free_port()]
    for dp in direct_ports:
        argv = [str(e2e.RPC_SERVER), "-H", "127.0.0.1", "-p", str(dp), "-t", "4"]
        e2e.spawn(procs, argv, e2e.node_env(tmp_path / "direct"))
        e2e.wait_port(dp)
    direct = e2e.complete(
        procs, ",".join(f"127.0.0.1:{p}" for p in direct_ports), e2e.node_env(tmp_path / "direct")
    )
    assert tunnelled and tunnelled == direct

    removed = cli(master, derp_map, "cluster", "remove", "w1")
    assert removed.returncode == 0, removed.stdout + removed.stderr
    assert [p.name for p in registry().workers()] == ["w2"]
    listed = cli(master, derp_map, "cluster", "list", timeout = 180)
    assert "w1" not in listed.stdout and "w2" in listed.stdout, listed.stdout
    remote.close()


def test_a_worker_reachable_only_through_a_relay_is_not_served_unless_allowed(
    tmp_path, procs, monkeypatch, derp_map
):
    if not e2e.RPC_SERVER.is_file():
        pytest.skip(reason = "needs ggml-rpc-server (UNSLOTH_TEST_LLAMA_BIN_DIR)")
    master, state, ap = tmp_path / "master", tmp_path / "w1", e2e.free_port()
    pair_through_tailcat(procs, master, state, "w1", derp_map, ap)
    argv = [sys.executable, "-c", WORKER_AGENT, str(ap), str(e2e.RPC_SERVER), str(e2e.free_port())]
    agent = e2e.spawn(procs, argv + [str(state / "cache")], env(state, derp_map))
    assert e2e.wait_line(agent, "READY ") == "tunnel", e2e.log_tail(agent)
    # magicsock's own knob: every packet goes through DERP, as behind a NAT that defeats traversal.
    relayed = {"TS_DEBUG_ALWAYS_USE_DERP": "1"}
    listed = cli(master, derp_map, "cluster", "list", timeout = 180, **relayed)
    assert "w1" in listed.stdout and "relayed only: too slow to serve models" in listed.stdout, (
        listed.stdout + listed.stderr
    )
    allowed = cli(
        master, derp_map, "cluster", "list", timeout = 180, UNSLOTH_CLUSTER_ALLOW_RELAY = "1", **relayed
    )
    assert "ready (through a relay: slow)" in allowed.stdout, allowed.stdout + allowed.stderr

    _, registry, relays = master_registry(monkeypatch, master, derp_map)
    monkeypatch.setenv("TS_DEBUG_ALWAYS_USE_DERP", "1")
    from studio.cluster_link import remote

    try:
        peer = registry().workers()[0]
        assert remote.forward(peer.node_id, peer.tc_addr, peer.agent_port)
        info = remote.path(peer.node_id)
        assert info is not None and not info.direct and not remote.usable(info)
        monkeypatch.setenv("UNSLOTH_CLUSTER_ALLOW_RELAY", "1")
        assert remote.usable(info)
    finally:
        relays.stop()
        remote.close()
