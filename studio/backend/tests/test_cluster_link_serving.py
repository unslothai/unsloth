# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Paired workers (studio/cluster_link) as cluster nodes: precedence, relays, drops, launch env."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import subprocess
import threading
import time
from types import SimpleNamespace

import pytest

from core.inference import cluster_serving as cs
from core.inference import spark_serving as ss

FP_A = "a" * 64
FP_B = "b" * 64
REAL = ("192.168.1.20", 50070)  # node-a's recorded agent address
REFUSED = {"type": "error", "error": "refused", "reachable": False}
SPOOF = SimpleNamespace(host = "10.6.6.6", agent_port = 50070, fp = FP_A, seen_at = 2.0)
HONEST = SimpleNamespace(host = REAL[0], agent_port = REAL[1], fp = FP_A, seen_at = 1.0)


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

    def record_address(self, node_id, host, port):
        type(self).recorded.append((node_id, host, port))


class FakeRelays:
    def __init__(self):
        self.ports, self.dest, self.statuses, self.removed = {}, {}, {}, []
        self.only = {}  # node_id -> (the one (host, port) that answers, answer anywhere else)
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
        if node_id in self.only and self.dest.get(node_id) != self.only[node_id][0]:
            answer = self.only[node_id][1]
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


def _identity(master):
    return SimpleNamespace(master_identity = lambda create = False: master, Registry = FakeRegistry)


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
    fake_identity = _identity(object())
    monkeypatch.setattr(cs, "_link_identity", lambda: fake_identity)
    monkeypatch.setattr(cs, "_relays", lambda: relays)
    monkeypatch.setattr(cs, "_sightings", lambda: {})
    yield
    cs.reset_for_tests()


def _fake_manager(
    monkeypatch,
    master,
    delay = 0.0,
):
    import studio.cluster_link.relay as relay_mod

    made = []

    class Manager:
        def __init__(self, identity, authorize):
            self.identity, self.stopped = identity, False
            time.sleep(delay)  # widens the check-then-create window
            made.append(self)

        def stop(self):
            self.stopped = True

    monkeypatch.setattr(relay_mod, "RelayManager", Manager)
    monkeypatch.setattr(cs, "_link_identity", lambda: _identity(master()))
    return made


def test_paired_workers_on_a_master_make_the_cluster_configured(monkeypatch):
    assert cs.configured()
    FakeRegistry.peers = []
    assert not cs.configured()
    FakeRegistry.peers = [peer("node-a", "desk", FP_A)]
    monkeypatch.setattr(cs, "_link_identity", lambda: _identity(None))  # not a master
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


@pytest.mark.parametrize(
    "gone", [pytest.param({"node-b"}, id = "one"), pytest.param({"node-a", "node-b"}, id = "last")]
)
def test_removed_workers_are_skipped_and_their_relay_closed(relays, gone):
    cs.load_nodes()
    for p in FakeRegistry.peers:
        p.removed = p.node_id in gone
    if len(gone) == 2:
        with pytest.raises(cs.ClusterConfigError):
            cs.load_nodes()
    else:
        assert [n.node_id for n in cs.load_nodes()] == ["node-a"]
    assert sorted(relays.removed) == sorted(gone)


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


def test_an_unreachable_or_rpc_down_worker_is_dropped_never_downgraded(relays):
    st = cs.state()
    st.refresh_nodes()
    relays.statuses["node-b"] = ConnectionRefusedError("refused")

    async def check():
        return await asyncio.gather(*(st._reachable(n) for n in st.nodes))

    assert asyncio.run(check()) == [True, False]
    assert "ConnectionRefusedError" in cs.LINK_DROPPED["laptop"]
    assert st.status()["link_dropped"] == {"laptop": "ConnectionRefusedError: refused"}
    assert FakeRegistry.recorded == [("node-a", "192.168.1.20", 50070)]
    relays.statuses["node-a"] = {"busy": False, "rpc_up": False}  # its rpc-server is down
    assert asyncio.run(st._reachable(st.nodes[0])) is False
    assert "not running" in cs.LINK_DROPPED["desk"]


@pytest.mark.parametrize("transport_changed", [False, True], ids = ["same", "native_to_relay"])
def test_a_held_link_worker_is_read_from_the_agent_not_probed(
    monkeypatch, relays, transport_changed
):
    # A live split on native TLS whose refresh fell back to the relay is still that rpc-server.
    st = cs.state()
    st.refresh_nodes()
    a, b = st.nodes
    held_at = f"tls://{FP_A}@192.168.1.20:50053" if transport_changed else a.endpoint
    assert (a.endpoint != held_at) is transport_changed
    device = cs.Device(
        name = "RPC0", description = held_at, total_bytes = 1, free_bytes = 1, endpoint = held_at
    )
    st._seen = {held_at: [device], b.endpoint: []}
    plan = SimpleNamespace(topology = "layer_split", devices = [device])
    st._pre_load = ("layer_split", "", plan, [])
    st._backend = SimpleNamespace(is_loaded = True)
    relays.statuses["node-a"] = {"busy": True, "rpc_up": True}
    monkeypatch.setattr(
        cs, "list_devices", lambda *a, **k: pytest.fail("a held link worker was probed")
    )
    held = asyncio.run(st._held_workers("llama-server", [a.endpoint, b.endpoint]))
    assert list(held) == [a.endpoint]
    assert [d.endpoint for d in held[a.endpoint]] == [a.endpoint]


def test_secure_rpc_env_only_for_link_relays_and_their_device_probe(monkeypatch):
    nodes = cs.load_nodes()
    relay_env = {"GGML_RPC_NO_RDMA": "1"}
    assert cs.secure_rpc_env(["llama-server", "--rpc", nodes[0].endpoint]) == relay_env
    assert cs.secure_rpc_env(["llama-server", f"--rpc={nodes[1].endpoint}"]) == relay_env
    assert cs.secure_rpc_env(["llama-server", "--rpc", "10.0.0.2:50052"]) == {}
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
    cs.reset_for_tests()
    assert cs.secure_rpc_env(["llama-server", "--rpc", nodes[0].endpoint]) == {}


def test_a_status_refresh_never_hides_a_live_endpoint_from_a_concurrent_launch(monkeypatch):
    # Endpoint registry changes must look atomic, or the launch drops GGML_RPC_NO_RDMA / TLS vars.
    nodes = cs._link_nodes(verify = False)
    seen, probes = [], []

    def launch_env():
        seen.extend(cs.secure_rpc_env(["llama-server", "--rpc", n.endpoint]) for n in nodes)

    class Probing(dict):
        def _probe(self):
            launch = threading.Thread(target = launch_env)
            launch.start()
            launch.join(0.2)  # runs now unless the refresh holds the registry lock
            probes.append(launch)

        def __setitem__(self, key, value):
            super().__setitem__(key, value)
            self._probe()

        def pop(self, *args):
            out = super().pop(*args)
            self._probe()
            return out

    monkeypatch.setattr(cs, "_LINK_ENDPOINTS", Probing(cs._LINK_ENDPOINTS))
    cs._link_nodes(verify = False)
    for launch in probes:
        launch.join(10.0)
        assert not launch.is_alive()
    assert seen and all(env == {"GGML_RPC_NO_RDMA": "1"} for env in seen)


def test_reset_stops_the_relays(monkeypatch):
    stopped = []
    monkeypatch.setattr(cs, "_RELAYS", SimpleNamespace(stop = lambda: stopped.append(1)))
    cs._LINK_ENDPOINTS["127.0.0.1:1"] = "x"
    cs.reset_for_tests()
    assert stopped == [1] and cs._RELAYS is None and not cs._LINK_ENDPOINTS


def test_a_spoofed_newer_sighting_does_not_take_the_worker_away(monkeypatch, relays):
    # Fingerprints are public: serving keeps the address where the real worker answers mutual TLS.
    monkeypatch.setattr(cs, "_sightings", lambda: {"node-a": SPOOF})
    candidates = lambda n: [SPOOF, HONEST] if n == "node-a" else []  # noqa: E731
    monkeypatch.setattr(cs, "_sighting_candidates", candidates)
    relays.only["node-a"] = (REAL, ConnectionRefusedError("not the paired worker"))
    nodes = cs.load_nodes()
    assert (nodes[0].host, nodes[0].rpc_port) == REAL and relays.dest["node-a"] == REAL


def test_discovery_only_keeps_announcements_of_paired_workers():
    cs._PAIRED_FPS.clear()
    cs._PAIRED_FPS["node-a"] = FP_A
    assert cs._paired_announcement("node-a", FP_A)
    assert not cs._paired_announcement("node-a", FP_B)
    assert not cs._paired_announcement("stranger", FP_A)


def test_a_failed_status_is_never_treated_as_reachable(relays):
    # The real RelayManager.status() reports a refused connection in the dict, not by raising.
    relays.statuses["node-a"] = {"type": "error", "error": "TLS refused", "reachable": False}
    answer, why = cs.link_status("node-a", 1.0)
    assert answer is None and "TLS refused" in why
    relays.statuses["node-a"] = {"type": "status", "busy": False, "rpc_up": True, "reachable": True}
    answer, why = cs.link_status("node-a", 1.0)
    assert answer is not None and why == ""


def test_a_worker_not_at_its_recorded_address_waits_for_discovery_before_a_load(
    monkeypatch, relays
):
    # node-b serves on an unpaired port: the load waits one announcement round, then plans with it.
    moved, sightings, slept = ("192.168.1.21", 50123), {}, []
    relays.only["node-b"] = (moved, REFUSED)
    monkeypatch.setattr(cs, "_sighting_candidates", lambda node_id: sightings.get(node_id, []))
    monkeypatch.setattr(cs, "_LISTENER", object())
    monkeypatch.setattr(cs, "_LISTENER_STARTED", time.monotonic())
    monkeypatch.setattr(cs, "DISCOVERY_SETTLE_S", 0.3)
    cs.load_nodes()
    assert "node-b" in cs._UNVERIFIED  # tried only the recorded address
    assert 0 < cs.discovery_settle_s() <= 0.3

    async def fake_sleep(seconds):
        slept.append(seconds)
        found = SimpleNamespace(host = moved[0], agent_port = moved[1], fp = FP_B, seen_at = 1.0)
        sightings["node-b"] = [found]

    async def plan_stub(*args, **kwargs):
        return SimpleNamespace(llama_extra_args = [])

    monkeypatch.setattr(cs.asyncio, "sleep", fake_sleep)
    state = cs.state()
    monkeypatch.setattr(state, "_plan_load", plan_stub)
    monkeypatch.setattr(state, "_serving_split", lambda: False)
    asyncio.run(state.before_load(SimpleNamespace(model_path = "m.gguf", gguf_variant = None), 1))
    assert slept and slept[0] > 0
    nodes = cs.load_nodes()
    assert ("node-b", moved) in [(n.node_id, (n.host, n.rpc_port)) for n in nodes]
    assert not cs._UNVERIFIED and cs.discovery_settle_s() == 0.0


def test_the_discovery_wait_runs_from_the_failed_verification_not_the_listener_start(
    monkeypatch, relays
):
    # A long-running listener must still wait one announcement round for a worker that just moved.
    relays.only["node-b"] = (("192.168.1.21", 50123), REFUSED)
    monkeypatch.setattr(cs, "_sighting_candidates", lambda node_id: [])
    monkeypatch.setattr(cs, "_LISTENER", object())
    monkeypatch.setattr(cs, "_LISTENER_STARTED", time.monotonic() - 3600)
    monkeypatch.setattr(cs, "DISCOVERY_SETTLE_S", 0.3)
    cs.load_nodes()
    assert "node-b" in cs._UNVERIFIED
    assert 0 < cs.discovery_settle_s() <= 0.3
    # Still unreachable on the next load: the wait counts from the first failure, so it is not repeated.
    cs._UNVERIFIED["node-b"] = time.monotonic() - 10
    cs.load_nodes()
    assert "node-b" in cs._UNVERIFIED and cs.discovery_settle_s() == 0.0


def test_a_reset_master_gets_a_new_relay_manager(monkeypatch):
    # Leave + re-pair makes a new root: a running Studio must drop the old certificate.
    current = {"master": SimpleNamespace(fingerprint = "1" * 64)}
    made = _fake_manager(monkeypatch, lambda: current["master"])
    first = REAL_RELAYS()
    assert REAL_RELAYS() is first and len(made) == 1
    current["master"] = SimpleNamespace(fingerprint = "2" * 64)
    second = REAL_RELAYS()
    assert second is not first and first.stopped
    assert second.identity.fingerprint == "2" * 64


def test_concurrent_first_callers_share_one_relay_manager(monkeypatch):
    # A status poll (event loop) and a load (worker thread) can both be first: one manager, ever.
    master = SimpleNamespace(fingerprint = "1" * 64)
    made, got = _fake_manager(monkeypatch, lambda: master, delay = 0.05), []
    threads = [threading.Thread(target = lambda: got.append(REAL_RELAYS())) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(made) == 1 and all(g is made[0] for g in got)


def test_status_reads_follow_the_registry_without_asking_any_worker(monkeypatch, relays):
    # status() runs on Studio's event loop; pair/remove in another process shows without a load.
    asked = []
    monkeypatch.setattr(relays, "status", lambda node_id, timeout = 3.0: asked.append(node_id))
    st = cs.state()
    out = st.status()
    assert out["enabled"] and {n["name"] for n in out["nodes"]} == {"desk", "laptop"}
    planned = list(st.nodes)
    FakeRegistry.peers = [p for p in FakeRegistry.peers if p.name == "desk"]
    assert {n["name"] for n in st.status()["nodes"]} == {"desk"}
    assert st.nodes == planned  # a load in flight keeps the nodes it planned against
    FakeRegistry.peers = FakeRegistry.peers + [peer("node-c", "attic", "c" * 64, ("192.168.1.30",))]
    assert {n["name"] for n in st.status()["nodes"]} == {"desk", "attic"}
    assert asked == []


def test_an_older_recorded_address_is_tried_when_the_newest_is_stale(relays):
    # Discovery blocked (no sightings): the registry's address history is all there is.
    FakeRegistry.peers = [peer("node-a", "desk", FP_A, ("192.168.1.99", "192.168.1.20"))]
    relays.only["node-a"] = (REAL, {"type": "error", "error": "unreachable", "reachable": False})
    assert [(n.node_id, n.host) for n in cs.load_nodes()] == [("node-a", "192.168.1.20")]
    # Recorded host:port pairs are each tried too.
    p = SimpleNamespace(addrs = ["10.0.0.5:50099", "10.0.0.5:50070"], agent_port = 50099)
    assert cs._peer_addresses(p, 50070) == [("10.0.0.5", 50099), ("10.0.0.5", 50070)]


def test_after_the_master_is_reset_a_load_falls_back_instead_of_crashing(monkeypatch):
    # Last worker removed and `cluster leave` run: the refresh closes the relay, reports no cluster.
    stopped = []
    old_root = SimpleNamespace(fingerprint = "1" * 64)
    old = SimpleNamespace(identity = old_root, stop = lambda: stopped.append(1))
    monkeypatch.setattr(cs, "_RELAYS", old)
    monkeypatch.setitem(cs._LINK_ENDPOINTS, "127.0.0.1:41000", "node-a")
    FakeRegistry.peers = []
    monkeypatch.setattr(cs, "_relays", REAL_RELAYS)
    monkeypatch.setattr(cs, "_link_identity", lambda: _identity(None))
    with pytest.raises(cs.ClusterConfigError):
        cs.load_nodes()
    assert stopped and cs._RELAYS is None and not cs._LINK_ENDPOINTS


def _offer_native(
    monkeypatch,
    relays,
    reply = None,
    local = True,
):
    calls = {"native": [], "close": []}
    relays.statuses["node-a"] = {
        "busy": False,
        "rpc_up": True,
        "devices": [],
        "native": {"available": True, "active": False, "port": 50053},
    }

    def native(node_id, timeout = 70.0):
        calls["native"].append(node_id)
        return reply or {"type": "native_ok", "port": 50053, "fingerprint": FP_A}

    def native_close(node_id, timeout = 15.0):
        calls["close"].append(node_id)
        return True

    relays.native, relays.native_close = native, native_close
    monkeypatch.setattr(cs, "_local_tls_client", lambda: local)
    return calls


def test_a_worker_offering_native_tls_is_planned_on_a_pinned_tls_endpoint(monkeypatch, relays):
    calls = _offer_native(monkeypatch, relays)
    nodes = {n.node_id: n for n in cs.load_nodes()}
    a, b = nodes["node-a"], nodes["node-b"]
    assert a.transport == "native_tls" and a.link
    assert a.endpoint == f"tls://{FP_A}@192.168.1.20:50053"
    # host / rpc_port stay the agent's address: they are what gets recorded and re-verified.
    assert (a.host, a.rpc_port) == REAL
    assert b.transport == "link" and b.endpoint.startswith("127.0.0.1:")
    assert calls["native"] == ["node-a"]
    # A status read reuses the grant without asking again.
    again = {n.node_id: n for n in cs._link_nodes(verify = False)}
    assert again["node-a"].transport == "native_tls"
    assert calls["native"] == ["node-a"]


@pytest.mark.parametrize(
    "offer, asked, closed",
    [
        pytest.param({"local": False}, [], [], id = "no_local_tls_bundle"),
        pytest.param(
            {"reply": {"type": "native_ok", "port": 50053, "fingerprint": FP_B}},
            ["node-a"],
            ["node-a"],
            id = "another_certificate",
        ),
        pytest.param({"reply": {"type": "busy"}}, ["node-a"], [], id = "refused_request"),
    ],
)
def test_native_tls_falls_back_to_the_relay_never_plaintext(
    monkeypatch, relays, offer, asked, closed
):
    calls = _offer_native(monkeypatch, relays, **offer)
    node = {n.node_id: n for n in cs.load_nodes()}["node-a"]
    assert node.transport == "link" and node.endpoint.startswith("127.0.0.1:")
    assert calls == {"native": asked, "close": closed}


def test_the_tls_launch_env_carries_the_master_certificate_not_argv(monkeypatch, relays):
    _offer_native(monkeypatch, relays)
    master = SimpleNamespace(certfile = "/m/cert.pem", keyfile = "/m/key.pem", rootfile = "/m/root.pem")
    monkeypatch.setattr(cs, "_link_identity", lambda: _identity(master))
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


def test_a_status_read_cuts_a_removed_native_worker_off_the_event_loop(monkeypatch, relays):
    calls = _offer_native(monkeypatch, relays)
    assert {n.node_id: n for n in cs.load_nodes()}["node-a"].transport == "native_tls"
    release, closed = threading.Event(), threading.Event()

    def slow_close(node_id, timeout = 15.0):
        release.wait(10)  # the worker takes its time to answer
        calls["close"].append(node_id)
        closed.set()
        return True

    relays.native_close = slow_close
    FakeRegistry.peers[0].removed = True
    started = time.monotonic()
    nodes = cs._link_nodes(verify = False)  # what /status runs, on Studio's event loop
    assert time.monotonic() - started < 2.0
    assert "node-a" not in {n.node_id for n in nodes} and "node-a" not in cs._NATIVE
    release.set()
    # The revocation still happens, just not on the caller's thread.
    assert closed.wait(10)
    for _ in range(100):
        if "node-a" in relays.removed:
            break
        time.sleep(0.05)
    assert calls["close"] == ["node-a"] and "node-a" in relays.removed


@pytest.mark.parametrize(
    "fields",
    [{"engine": "vllm"}, {"alongside": True}, {"force_alongside": True}],
)
def test_a_passthrough_load_skips_paired_worker_discovery(monkeypatch, relays, fields):
    state = cs.state()
    refreshed, slept = [], []

    async def fake_sleep(seconds):
        slept.append(seconds)

    monkeypatch.setattr(state, "refresh_nodes", lambda *a, **k: refreshed.append(1) or [])
    monkeypatch.setattr(cs.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(cs, "discovery_settle_s", lambda: 6.5)
    request = SimpleNamespace(model_path = "m.gguf", gguf_variant = None, **fields)
    assert asyncio.run(state.before_load(request, 1)) is request
    assert refreshed == [] and slept == [] and state._passthrough


def _fake_rpc_server(path, tls):
    flag = " --tls-cert F" if tls else ""
    path.write_text(f"#!/bin/sh\necho 'usage: ggml-rpc-server{flag}'\n")
    path.chmod(0o755)
    return str(path)


@pytest.mark.skipif(os.name == "nt", reason = "the fake rpc-server is a shell script")
def test_the_local_tls_probe_reads_only_the_rpc_server_of_the_dialing_build(monkeypatch, tmp_path):
    from studio.cluster_link import native

    monkeypatch.undo()  # the real _local_tls_client, not a fixture stand-in
    monkeypatch.setattr(ss, "llama_server_binary", lambda: str(tmp_path / "llama-server"))
    for tls_build in (True, False):
        fake = _fake_rpc_server(tmp_path / f"rpc-{tls_build}", tls_build)
        monkeypatch.setattr(ss, "rpc_server_binary", lambda f = fake: f)
        monkeypatch.delenv(native.DISABLE_ENV, raising = False)
        assert cs._local_tls_client() is tls_build
        monkeypatch.setenv(native.DISABLE_ENV, "0")
        assert cs._local_tls_client() is False

    # A custom llama-server with no rpc-server beside it: the fallback's TLS says nothing about it.
    (tmp_path / "bundle").mkdir()
    (tmp_path / "custom").mkdir()
    fake = _fake_rpc_server(tmp_path / "bundle" / "ggml-rpc-server", True)
    monkeypatch.setattr(ss, "rpc_server_binary", lambda: fake)
    custom = str(tmp_path / "custom" / "llama-server")
    monkeypatch.setattr(ss, "llama_server_binary", lambda: custom)
    monkeypatch.delenv(native.DISABLE_ENV, raising = False)
    assert native.binary_supports_tls(fake) is True
    assert cs._local_tls_client() is False
    monkeypatch.setattr(ss, "llama_server_binary", lambda: None)
    assert cs._local_tls_client() is False


def test_a_status_read_never_repoints_a_verified_relay_at_a_spoofed_sighting(monkeypatch, relays):
    # A LAN host announces node-a's id and fingerprint after planning verified it.
    sightings = {"node-a": [HONEST]}
    monkeypatch.setattr(cs, "_sightings", lambda: {n: s[0] for n, s in sightings.items()})
    monkeypatch.setattr(cs, "_sighting_candidates", lambda node_id: sightings.get(node_id, []))
    st = cs.state()
    st.refresh_nodes()  # verified planning pass
    assert relays.destination("node-a") == REAL
    sightings["node-a"] = [SPOOF, HONEST]
    st.status()
    assert relays.destination("node-a") == REAL


def test_a_status_read_racing_a_load_never_repoints_the_relay_it_verified(monkeypatch, relays):
    # The load verifies the real address mid status read: the read's ensure must not undo it.
    monkeypatch.setattr(cs, "_sighting_candidates", lambda n: [SPOOF] if n == "node-a" else [])
    relays.ensure("node-a", SPOOF.host, SPOOF.agent_port)  # where the load's first try pointed it
    relays.only["node-a"] = (REAL, {"reachable": False, "error": "refused"})
    destination, load = relays.destination, []

    def racing_destination(node_id):
        seen = destination(node_id)
        if node_id == "node-a" and not load:
            load.append(threading.Thread(target = cs._link_nodes, kwargs = {"verify": True}))
            load[0].start()
            load[0].join(0.5)  # finishes in this window unless the status read holds it off
        return seen

    relays.destination = racing_destination
    cs._link_nodes(verify = False)
    load[0].join(10.0)
    assert not load[0].is_alive()
    assert relays.destination("node-a") == REAL


def test_a_paired_worker_whose_probe_finds_no_device_is_reported_dropped(monkeypatch, relays):
    # A live relay to a stopped worker passes the reachability check; only the probe sees it gone.
    st = cs.state()
    st.refresh_nodes()
    a, b = st.nodes

    async def reachable(node):
        return True

    async def nothing(*args, **kwargs):
        return {}

    monkeypatch.setattr(st, "_reachable", reachable)
    monkeypatch.setattr(st, "_ensure_workers", nothing)
    monkeypatch.setattr(st, "_held_workers", nothing)
    monkeypatch.setattr(ss, "llama_server_binary", lambda: "llama-server")
    only_b = cs.Device(
        name = "RPC0",
        description = b.endpoint,
        total_bytes = 8 << 30,
        free_bytes = 8 << 30,
        endpoint = b.endpoint,
    )
    monkeypatch.setattr(cs, "list_devices", lambda *args, **kwargs: ([only_b], ""))
    st._host_projector = 0
    request = SimpleNamespace()
    asyncio.run(st._place(request, 1, ss, [], [], 1 << 30, 0.0, 0.0, None))
    assert set(cs.LINK_DROPPED) == {a.name}
    assert "did not answer" in cs.LINK_DROPPED[a.name]
