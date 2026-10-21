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

    def load(self):
        return self

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
        self.dest = {}
        self.statuses = {}
        self.removed = []
        self.stopped = False

    def ensure(self, node_id, host, agent_port):
        port = self.ports.setdefault(node_id, 41000 + len(self.ports))
        self.dest[node_id] = (host, agent_port)
        return f"127.0.0.1:{port}"

    def destination(self, node_id):
        return self.dest.get(node_id)

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


# The real one, before the autouse fixture swaps in the fake.
REAL_RELAYS = cs._relays


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


def test_a_spoofed_newer_sighting_does_not_take_the_worker_away(monkeypatch, relays):
    # Fingerprints are public: a LAN host announces node-a's fingerprint from its own address. Only the
    # real worker answers over mutual TLS, so serving keeps the address where it does.
    real = ("192.168.1.20", 50070)
    spoof = SimpleNamespace(host = "10.6.6.6", agent_port = 50070, fp = FP_A, seen_at = 2.0)
    honest = SimpleNamespace(host = real[0], agent_port = real[1], fp = FP_A, seen_at = 1.0)
    monkeypatch.setattr(cs, "_sightings", lambda: {"node-a": spoof})
    monkeypatch.setattr(
        cs, "_sighting_candidates", lambda node_id: [spoof, honest] if node_id == "node-a" else []
    )
    pointed = {}
    plain_ensure = relays.ensure

    def ensure(node_id, host, agent_port):
        pointed[node_id] = (host, agent_port)
        return plain_ensure(node_id, host, agent_port)

    def status(node_id, timeout = 3.0):
        if node_id == "node-a" and pointed.get(node_id) != real:
            raise ConnectionRefusedError("not the paired worker")
        return {"busy": False, "rpc_up": True, "devices": []}

    monkeypatch.setattr(relays, "ensure", ensure)
    monkeypatch.setattr(relays, "status", status)
    nodes = cs.load_nodes()
    assert (nodes[0].host, nodes[0].rpc_port) == real
    assert pointed["node-a"] == real


def test_discovery_only_keeps_announcements_of_paired_workers():
    cs._PAIRED_FPS.clear()
    cs._PAIRED_FPS["node-a"] = FP_A
    assert cs._paired_announcement("node-a", FP_A)
    assert not cs._paired_announcement("node-a", FP_B)
    assert not cs._paired_announcement("stranger", FP_A)


def test_removing_the_last_worker_still_closes_its_relay(relays):
    cs.load_nodes()
    FakeRegistry.peers = [
        peer(p.node_id, p.name, p.fingerprint, removed = True) for p in FakeRegistry.peers
    ]
    with pytest.raises(cs.ClusterConfigError):
        cs.load_nodes()
    assert sorted(relays.removed) == ["node-a", "node-b"]


def test_a_failed_status_is_never_treated_as_reachable(relays):
    # The real RelayManager.status() reports a refused or failed connection in the dict, not by raising.
    relays.statuses["node-a"] = {"type": "error", "error": "TLS refused", "reachable": False}
    answer, why = cs.link_status("node-a", 1.0)
    assert answer is None and "TLS refused" in why
    relays.statuses["node-a"] = {"type": "status", "busy": False, "rpc_up": True, "reachable": True}
    answer, why = cs.link_status("node-a", 1.0)
    assert answer is not None and why == ""


def test_a_worker_not_at_its_recorded_address_waits_for_discovery_before_a_load(
    monkeypatch, relays
):
    # node-b serves on another port than it paired with: only its announcement says where. A listener
    # started moments ago has heard none, so the load waits one announcement round, then plans with it.
    import asyncio
    import time as _time

    moved = ("192.168.1.21", 50123)
    pointed, sightings = {}, {}
    plain_ensure = relays.ensure

    def ensure(node_id, host, agent_port):
        pointed[node_id] = (host, agent_port)
        return plain_ensure(node_id, host, agent_port)

    def status(node_id, timeout = 3.0):
        if node_id == "node-b" and pointed.get(node_id) != moved:
            return {"type": "error", "error": "refused", "reachable": False}
        return {"busy": False, "rpc_up": True, "devices": []}

    monkeypatch.setattr(relays, "ensure", ensure)
    monkeypatch.setattr(relays, "status", status)
    monkeypatch.setattr(cs, "_sighting_candidates", lambda node_id: sightings.get(node_id, []))
    monkeypatch.setattr(cs, "_LISTENER", object())
    monkeypatch.setattr(cs, "_LISTENER_STARTED", _time.monotonic())
    monkeypatch.setattr(cs, "DISCOVERY_SETTLE_S", 0.3)
    cs.load_nodes()
    assert "node-b" in cs._UNVERIFIED  # tried only the recorded address, which is node-a's old port
    assert 0 < cs.discovery_settle_s() <= 0.3

    slept = []

    async def fake_sleep(seconds):
        slept.append(seconds)
        sightings["node-b"] = [
            SimpleNamespace(host = moved[0], agent_port = moved[1], fp = FP_B, seen_at = 1.0)
        ]

    monkeypatch.setattr(cs.asyncio, "sleep", fake_sleep)

    async def plan_stub(*args, **kwargs):
        return SimpleNamespace(llama_extra_args = [])

    state = cs.state()
    monkeypatch.setattr(state, "_plan_load", plan_stub)
    monkeypatch.setattr(state, "_serving_split", lambda: False)
    asyncio.run(state.before_load(SimpleNamespace(model_path = "m.gguf", gguf_variant = None), 1))
    assert slept and slept[0] > 0
    nodes = cs.load_nodes()
    assert ("node-b", moved) in [(n.node_id, (n.host, n.rpc_port)) for n in nodes]
    assert not cs._UNVERIFIED and cs.discovery_settle_s() == 0.0


def test_a_reset_master_gets_a_new_relay_manager(monkeypatch):
    # `cluster leave` on a master with no workers, then a new `cluster pair`: a new root. A running
    # Studio must stop presenting the old certificate, which no new worker trusts.
    import studio.cluster_link.relay as relay_mod

    made = []

    class Manager:
        def __init__(self, identity, authorize):
            self.identity, self.stopped = identity, False
            made.append(self)

        def stop(self):
            self.stopped = True

    current = {"master": SimpleNamespace(fingerprint = "1" * 64)}
    monkeypatch.setattr(relay_mod, "RelayManager", Manager)
    monkeypatch.setattr(
        cs,
        "_link_identity",
        lambda: SimpleNamespace(
            master_identity = lambda create = False: current["master"], Registry = FakeRegistry
        ),
    )
    first = REAL_RELAYS()
    assert REAL_RELAYS() is first and len(made) == 1
    current["master"] = SimpleNamespace(fingerprint = "2" * 64)
    second = REAL_RELAYS()
    assert second is not first and first.stopped
    assert second.identity.fingerprint == "2" * 64


def test_a_status_read_does_not_check_workers_over_the_network(monkeypatch, relays):
    # status() runs on Studio's event loop: listing paired workers must not wait on any of them.
    asked = []
    monkeypatch.setattr(relays, "status", lambda node_id, timeout = 3.0: asked.append(node_id))
    out = cs.state().status()
    assert asked == [] and out["enabled"]
    assert {n["name"] for n in out["nodes"]} == {"desk", "laptop"}


def test_an_older_recorded_address_is_tried_when_the_newest_is_stale(monkeypatch, relays):
    # Discovery blocked (no sightings): the registry's address history is all there is.
    FakeRegistry.peers = [peer("node-a", "desk", FP_A, ("192.168.1.99", "192.168.1.20"))]
    pointed = {}
    plain_ensure = relays.ensure

    def ensure(node_id, host, agent_port):
        pointed[node_id] = host
        return plain_ensure(node_id, host, agent_port)

    def status(node_id, timeout = 3.0):
        if pointed.get(node_id) != "192.168.1.20":
            return {"type": "error", "error": "unreachable", "reachable": False}
        return {"busy": False, "rpc_up": True, "devices": []}

    monkeypatch.setattr(relays, "ensure", ensure)
    monkeypatch.setattr(relays, "status", status)
    nodes = cs.load_nodes()
    assert [(n.node_id, n.host) for n in nodes] == [("node-a", "192.168.1.20")]


def test_after_the_master_is_reset_a_load_falls_back_instead_of_crashing(monkeypatch):
    # The last worker removed and `cluster leave` run while Studio still holds a relay: the next
    # refresh must close it and report "no cluster", not build a relay manager for no identity.
    stopped = []
    monkeypatch.setattr(
        cs,
        "_RELAYS",
        SimpleNamespace(
            identity = SimpleNamespace(fingerprint = "1" * 64), stop = lambda: stopped.append(True)
        ),
    )
    monkeypatch.setitem(cs._LINK_ENDPOINTS, "127.0.0.1:41000", "node-a")
    FakeRegistry.peers = []
    monkeypatch.setattr(cs, "_relays", REAL_RELAYS)
    monkeypatch.setattr(
        cs,
        "_link_identity",
        lambda: SimpleNamespace(master_identity = lambda create = False: None, Registry = FakeRegistry),
    )
    with pytest.raises(cs.ClusterConfigError):
        cs.load_nodes()
    assert stopped and cs._RELAYS is None and not cs._LINK_ENDPOINTS


# Native TLS: llama.cpp's own transport when both ends have it --------------------------------

NATIVE_STATUS = {
    "busy": False,
    "rpc_up": True,
    "devices": [],
    "native": {"available": True, "active": False, "port": 50053},
}


def _offer_native(
    monkeypatch,
    relays,
    reply = None,
    local = True,
):
    calls = {"native": [], "close": []}
    relays.statuses["node-a"] = dict(NATIVE_STATUS)

    def native(node_id, timeout = 70.0):
        calls["native"].append(node_id)
        return reply or {"type": "native_ok", "port": 50053, "fingerprint": FP_A}

    def native_close(node_id, timeout = 15.0):
        calls["close"].append(node_id)
        return True

    relays.native = native
    relays.native_close = native_close
    monkeypatch.setattr(cs, "_local_tls_client", lambda: local)
    return calls


def test_a_worker_offering_native_tls_is_planned_on_a_pinned_tls_endpoint(monkeypatch, relays):
    calls = _offer_native(monkeypatch, relays)
    nodes = {n.node_id: n for n in cs.load_nodes()}
    a, b = nodes["node-a"], nodes["node-b"]
    assert a.transport == "native_tls" and a.link
    assert a.endpoint == f"tls://{FP_A}@192.168.1.20:50053"
    # host / rpc_port stay the agent's address: they are what gets recorded and re-verified.
    assert (a.host, a.rpc_port) == ("192.168.1.20", 50070)
    assert b.transport == "link" and b.endpoint.startswith("127.0.0.1:")
    assert calls["native"] == ["node-a"]


def test_native_tls_needs_the_local_bundle_too(monkeypatch, relays):
    calls = _offer_native(monkeypatch, relays, local = False)
    nodes = {n.node_id: n for n in cs.load_nodes()}
    assert nodes["node-a"].transport == "link"
    assert calls["native"] == []


def test_a_native_offer_under_another_certificate_stays_on_the_relay(monkeypatch, relays):
    calls = _offer_native(
        monkeypatch, relays, reply = {"type": "native_ok", "port": 50053, "fingerprint": FP_B}
    )
    nodes = {n.node_id: n for n in cs.load_nodes()}
    assert nodes["node-a"].transport == "link"
    assert calls["close"] == ["node-a"]


def test_a_refused_native_request_falls_back_to_the_relay_never_plaintext(monkeypatch, relays):
    _offer_native(monkeypatch, relays, reply = {"type": "busy"})
    node = {n.node_id: n for n in cs.load_nodes()}["node-a"]
    assert node.transport == "link" and node.endpoint.startswith("127.0.0.1:")


def test_the_tls_launch_env_carries_the_master_certificate_not_argv(monkeypatch, relays):
    _offer_native(monkeypatch, relays)
    master = SimpleNamespace(certfile = "/m/cert.pem", keyfile = "/m/key.pem", rootfile = "/m/root.pem")
    fake_identity = SimpleNamespace(
        master_identity = lambda create = False: master, Registry = FakeRegistry
    )
    monkeypatch.setattr(cs, "_link_identity", lambda: fake_identity)
    nodes = {n.node_id: n for n in cs.load_nodes()}
    argv = ["llama-server", "--rpc", f"{nodes['node-a'].endpoint},{nodes['node-b'].endpoint}"]
    assert cs.secure_rpc_env(argv) == {
        "GGML_RPC_NO_RDMA": "1",
        "GGML_RPC_TLS_CERT": "/m/cert.pem",
        "GGML_RPC_TLS_KEY": "/m/key.pem",
        "GGML_RPC_TLS_CA": "/m/root.pem",
    }
    assert not any("/m/" in a for a in argv)
    # A relay-only launch gets no certificate at all.
    assert cs.secure_rpc_env(["llama-server", "--rpc", nodes["node-b"].endpoint]) == {
        "GGML_RPC_NO_RDMA": "1"
    }


def test_removing_a_native_worker_asks_it_to_cut_the_session(monkeypatch, relays):
    calls = _offer_native(monkeypatch, relays)
    assert {n.node_id: n for n in cs.load_nodes()}["node-a"].transport == "native_tls"
    FakeRegistry.peers[0].removed = True
    cs.load_nodes()
    assert calls["close"] == ["node-a"] and "node-a" in relays.removed


def test_a_status_read_reuses_the_native_grant_without_asking_again(monkeypatch, relays):
    calls = _offer_native(monkeypatch, relays)
    cs.load_nodes()
    nodes = {n.node_id: n for n in cs._link_nodes(verify = False)}
    assert nodes["node-a"].transport == "native_tls" and calls["native"] == ["node-a"]


@pytest.mark.skipif(__import__("os").name == "nt", reason = "the fake rpc-server is a shell script")
def test_the_local_tls_probe_reads_the_bundled_rpc_server(monkeypatch, tmp_path):
    from studio.cluster_link import native
    monkeypatch.undo()  # the real _local_tls_client, not a fixture stand-in
    for tls_build in (True, False):
        fake = tmp_path / f"rpc-{tls_build}"
        fake.write_text(
            "#!/bin/sh\necho 'usage: ggml-rpc-server"
            + (" --tls-cert F" if tls_build else "")
            + "'\n"
        )
        fake.chmod(0o755)
        monkeypatch.setattr(ss, "rpc_server_binary", lambda f = fake: str(f))
        monkeypatch.delenv(native.DISABLE_ENV, raising = False)
        assert cs._local_tls_client() is tls_build
        monkeypatch.setenv(native.DISABLE_ENV, "0")
        assert cs._local_tls_client() is False


def test_every_status_read_sees_workers_paired_or_removed_since(monkeypatch, relays):
    # `cluster pair` / `cluster remove` run in another process while Studio stays up: the status API
    # must follow the registry without waiting for a model load to refresh the nodes.
    st = cs.state()
    assert {n["name"] for n in st.status()["nodes"]} == {"desk", "laptop"}
    planned = list(st.nodes)
    FakeRegistry.peers = [p for p in FakeRegistry.peers if p.name == "desk"]
    assert {n["name"] for n in st.status()["nodes"]} == {"desk"}
    assert st.nodes == planned  # a load in flight keeps the nodes it planned against
    FakeRegistry.peers = FakeRegistry.peers + [peer("node-c", "attic", "c" * 64, ("192.168.1.30",))]
    assert {n["name"] for n in st.status()["nodes"]} == {"desk", "attic"}


def test_a_status_read_never_repoints_a_verified_relay_at_a_spoofed_sighting(monkeypatch, relays):
    # Planning verified node-a at its real address; a LAN host then announces node-a's public id and
    # fingerprint from elsewhere. A status read checks nothing over the network, so it must leave the
    # relay where verification put it, or the next RPC connection goes to the spoofer.
    real = ("192.168.1.20", 50070)
    honest = SimpleNamespace(host = real[0], agent_port = real[1], fp = FP_A, seen_at = 1.0)
    sightings = {"node-a": [honest]}
    monkeypatch.setattr(cs, "_sightings", lambda: {n: s[0] for n, s in sightings.items()})
    monkeypatch.setattr(cs, "_sighting_candidates", lambda node_id: sightings.get(node_id, []))
    st = cs.state()
    st.refresh_nodes()  # verified planning pass
    assert relays.destination("node-a") == real
    spoof = SimpleNamespace(host = "10.6.6.6", agent_port = 50070, fp = FP_A, seen_at = 2.0)
    sightings["node-a"] = [spoof, honest]
    st.status()
    assert relays.destination("node-a") == real
