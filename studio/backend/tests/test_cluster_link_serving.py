# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Paired workers (studio/cluster_link) as cluster nodes: precedence after the env and cluster.json,
relay endpoints, dropped workers with a reason, busy detection, and the launch env."""

from __future__ import annotations

import contextlib
import asyncio
import json
import subprocess
from types import SimpleNamespace

import pytest

from core.inference import cluster_serving as cs
from core.inference import spark_serving as ss

FP_A = "a" * 64
FP_B = "b" * 64


class FakeRegistry:
    peers: list = []
    recorded: list = []

    @classmethod
    def open(cls):
        return cls()

    @contextlib.contextmanager
    def transaction(self):
        yield self

    def workers(self):
        return list(type(self).peers)

    def authorize(self, peer, role):
        for p in type(self).peers:
            if p.fingerprint == peer.fingerprint and role == "worker" and not p.removed:
                return p
        return None

    def record_address(self, node_id, host, port):
        type(self).recorded.append((node_id, host, port))


class FakeRelays:
    def __init__(self):
        self.ports = {}
        self.statuses = {}
        self.removed = []
        self.stopped = False

    def ensure(self, node_id, host, agent_port):
        port = self.ports.setdefault(node_id, 41000 + len(self.ports))
        return f"127.0.0.1:{port}"

    def status(
        self,
        node_id,
        timeout = 3.0,
    ):
        answer = self.statuses.get(node_id)
        if isinstance(answer, Exception):
            raise answer
        return answer if answer is not None else {"busy": False, "rpc_up": True, "devices": []}

    def busy(self, node_id):
        return bool(self.status(node_id).get("busy"))

    def remove(self, node_id):
        self.removed.append(node_id)

    def stop(self):
        self.stopped = True


def peer(
    node_id,
    name,
    fp,
    addrs = ("192.168.1.20",),
    port = 50070,
    removed = False,
):
    return SimpleNamespace(
        node_id = node_id,
        name = name,
        role = "worker",
        fingerprint = fp,
        cert_pem = "",
        addrs = list(addrs),
        agent_port = port,
        removed = removed,
    )


@pytest.fixture
def relays():
    return FakeRelays()


@pytest.fixture(autouse = True)
def clean(monkeypatch, tmp_path, relays):
    for name in (cs.ENV_NODES, cs.ENV_MANAGED, cs.ENV_TOPOLOGY, ss.ENV_TOGGLE):
        monkeypatch.delenv(name, raising = False)
    monkeypatch.setenv("UNSLOTH_CLUSTER_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setattr(cs, "_config_path", lambda: tmp_path / "cluster.json")
    cs.reset_for_tests()
    FakeRegistry.peers = [
        peer("node-a", "desk", FP_A),
        peer("node-b", "laptop", FP_B, ("192.168.1.21",)),
    ]
    FakeRegistry.recorded = []
    fake_identity = SimpleNamespace(
        master_identity = lambda create = False: object(), Registry = FakeRegistry
    )
    monkeypatch.setattr(cs, "_link_identity", lambda: fake_identity)
    monkeypatch.setattr(cs, "_relays", lambda: relays)
    monkeypatch.setattr(cs, "_sightings", lambda: {})
    yield
    cs.reset_for_tests()


def test_paired_workers_make_the_cluster_configured():
    assert cs.configured()
    FakeRegistry.peers = []
    assert not cs.configured()


def test_not_a_master_means_no_paired_cluster(monkeypatch):
    monkeypatch.setattr(
        cs,
        "_link_identity",
        lambda: SimpleNamespace(master_identity = lambda create = False: None, Registry = FakeRegistry),
    )
    assert not cs.configured()
    with pytest.raises(cs.ClusterConfigError):
        cs.load_nodes()


def test_paired_workers_become_link_nodes_on_relays():
    nodes = cs.load_nodes()
    assert [n.node_id for n in nodes] == ["node-a", "node-b"]
    assert all(n.link and not n.managed for n in nodes)
    assert [n.endpoint for n in nodes] == ["127.0.0.1:41000", "127.0.0.1:41001"]
    # host stays the worker's own address: host-keyed checks never see two "127.0.0.1" nodes.
    assert [n.host for n in nodes] == ["192.168.1.20", "192.168.1.21"]
    assert cs._one_managed_worker_per_host(nodes) == nodes


def test_env_and_cluster_json_still_win(monkeypatch, tmp_path):
    (tmp_path / "cluster.json").write_text(json.dumps({"nodes": ["10.0.0.9:50052"]}))
    assert [n.endpoint for n in cs.load_nodes()] == ["10.0.0.9:50052"]
    monkeypatch.setenv(cs.ENV_NODES, "10.0.0.2:50052")
    assert [n.endpoint for n in cs.load_nodes()] == ["10.0.0.2:50052"]
    assert not any(n.link for n in cs.load_nodes())


def test_removed_workers_are_skipped_and_their_relay_dropped(relays):
    cs.load_nodes()
    FakeRegistry.peers = [FakeRegistry.peers[0], peer("node-b", "laptop", FP_B, removed = True)]
    nodes = cs.load_nodes()
    assert [n.node_id for n in nodes] == ["node-a"]
    assert relays.removed == ["node-b"]


def test_a_worker_without_an_address_is_dropped_with_a_reason():
    FakeRegistry.peers = [peer("node-a", "desk", FP_A, addrs = ())]
    with pytest.raises(cs.ClusterConfigError, match = "desk: no known address"):
        cs.load_nodes()


def test_a_sighting_overrides_the_recorded_address_only_with_the_paired_fingerprint(monkeypatch):
    sighting = SimpleNamespace(host = "192.168.1.77", agent_port = 50099, fp = FP_A)
    spoof = SimpleNamespace(host = "192.168.1.66", agent_port = 50070, fp = "c" * 64)
    monkeypatch.setattr(cs, "_sightings", lambda: {"node-a": sighting, "node-b": spoof})
    nodes = cs.load_nodes()
    assert (nodes[0].host, nodes[0].rpc_port) == ("192.168.1.77", 50099)
    assert nodes[1].host == "192.168.1.21"


def test_an_unreachable_worker_is_dropped_never_downgraded(relays):
    st = cs.state()
    st.refresh_nodes()
    relays.statuses["node-b"] = ConnectionRefusedError("refused")

    async def check():
        return await asyncio.gather(*(st._reachable(n) for n in st.nodes))

    answers = asyncio.run(check())
    assert answers == [True, False]
    assert "ConnectionRefusedError" in cs.LINK_DROPPED["laptop"]
    assert st.status()["link_dropped"] == {"laptop": "ConnectionRefusedError: refused"}
    # the reachable one recorded its verified address
    assert FakeRegistry.recorded == [("node-a", "192.168.1.20", 50070)]


def test_a_worker_whose_rpc_server_is_down_is_dropped(relays):
    st = cs.state()
    st.refresh_nodes()
    relays.statuses["node-a"] = {"busy": False, "rpc_up": False}
    assert asyncio.run(st._reachable(st.nodes[0])) is False
    assert "not running" in cs.LINK_DROPPED["desk"]


def test_a_held_link_worker_is_read_from_the_agent_not_probed(monkeypatch, relays):
    st = cs.state()
    st.refresh_nodes()
    a, b = st.nodes
    device = cs.Device(
        name = "RPC0", description = a.endpoint, total_bytes = 1, free_bytes = 1, endpoint = a.endpoint
    )
    st._seen = {a.endpoint: [device], b.endpoint: []}
    st._pre_load = (
        "layer_split",
        "",
        SimpleNamespace(topology = "layer_split", devices = [device]),
        [],
    )
    st._backend = SimpleNamespace(is_loaded = True)
    relays.statuses["node-a"] = {"busy": True, "rpc_up": True}
    monkeypatch.setattr(
        cs, "list_devices", lambda *a, **k: pytest.fail("a held link worker was probed")
    )
    held = asyncio.run(st._held_workers("llama-server", [a.endpoint, b.endpoint]))
    assert list(held) == [a.endpoint]


def test_secure_rpc_env_only_for_link_relays():
    nodes = cs.load_nodes()
    assert cs.secure_rpc_env(["llama-server", "--rpc", nodes[0].endpoint]) == {
        "GGML_RPC_NO_RDMA": "1"
    }
    assert cs.secure_rpc_env(["llama-server", f"--rpc={nodes[1].endpoint}"]) == {
        "GGML_RPC_NO_RDMA": "1"
    }
    assert cs.secure_rpc_env(["llama-server", "--rpc", "10.0.0.2:50052"]) == {}
    cs.reset_for_tests()
    assert cs.secure_rpc_env(["llama-server", "--rpc", nodes[0].endpoint]) == {}


def test_the_device_probe_through_a_relay_carries_the_secure_env(monkeypatch):
    nodes = cs.load_nodes()
    seen = {}

    def fake_run(argv, **kw):
        seen["env"] = kw.get("env")
        return subprocess.CompletedProcess(argv, 0, stdout = "", stderr = "")

    monkeypatch.setattr(cs.subprocess, "run", fake_run)
    cs.list_devices("/bin/true", [nodes[0].endpoint])
    assert seen["env"]["GGML_RPC_NO_RDMA"] == "1"
    monkeypatch.delenv("GGML_RPC_NO_RDMA", raising = False)
    cs.list_devices("/bin/true", ["10.0.0.2:50052"])
    assert "GGML_RPC_NO_RDMA" not in (seen["env"] or {})


def test_reset_stops_the_relays(monkeypatch):
    stopped = []
    monkeypatch.setattr(cs, "_RELAYS", SimpleNamespace(stop = lambda: stopped.append(1)))
    cs._LINK_ENDPOINTS["127.0.0.1:1"] = "x"
    cs.reset_for_tests()
    assert stopped == [1] and cs._RELAYS is None and not cs._LINK_ENDPOINTS
