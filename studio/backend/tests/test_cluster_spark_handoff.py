# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Handing a model between the two-Spark path and an N-node cluster config, in both directions."""

from __future__ import annotations

import asyncio
import socket
from types import SimpleNamespace

import pytest

from core.inference import cluster_serving as cs
from core.inference import rpc_serving as rs
from core.inference import spark_serving as ss

GIB = 2**30

DOWN: set = set()

LISTING = """Available devices:
  RPC0: 10.0.0.2:50052 (122502 MiB, 118000 MiB free)
  RPC1: 10.0.0.3:50052 (65536 MiB, 60000 MiB free)
  RPC2: 10.0.0.4:50052 (24576 MiB, 23000 MiB free)
  CUDA0: NVIDIA GB10 (122502 MiB, 110000 MiB free)
"""


def run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse = True)
def clean(monkeypatch, tmp_path):
    for name in (cs.ENV_NODES, cs.ENV_MANAGED, cs.ENV_TOPOLOGY, ss.ENV_TOGGLE):
        monkeypatch.delenv(name, raising = False)
    monkeypatch.setattr(cs, "_config_path", lambda: tmp_path / "cluster.json")
    cs.reset_for_tests()
    rs.reset_for_tests()
    yield
    cs.reset_for_tests()
    rs.reset_for_tests()


def _cluster_env(
    monkeypatch,
    listing = LISTING,
    model_bytes = 250 * GIB,
):
    monkeypatch.setenv(cs.ENV_NODES, "10.0.0.2,10.0.0.3,10.0.0.4")
    monkeypatch.setattr(rs, "llama_server_binary", lambda: "/b/llama-server")
    monkeypatch.setattr(rs, "cached_repo_file", lambda path, variant: path)
    monkeypatch.setattr(rs, "gguf_size_bytes", lambda path: model_bytes if path else None)
    monkeypatch.setattr(rs, "estimate_kv_bytes", lambda *a, **k: 0)
    monkeypatch.setattr(rs, "auto_projector_file", lambda *a, **k: None)
    monkeypatch.setattr(cs, "_remote_gguf_header", lambda *a, **k: ({}, {}))
    monkeypatch.setattr(cs, "_local_gguf_header", lambda *a, **k: ({}, {}))
    monkeypatch.setattr(cs, "_ctx_compute_bytes", lambda *a, **k: 0)
    monkeypatch.setattr(cs, "_remote_projector_bytes", lambda path, token: 0)
    monkeypatch.setattr(cs, "_shares_host_memory", lambda device: True)
    monkeypatch.setattr(cs, "_kv_unified_supported", lambda binary: True)
    monkeypatch.setattr(cs, "_keeps_resident", lambda: False)
    monkeypatch.setattr(
        cs, "_host_ram", lambda held = 0.0: cs.Device("CPU", "host RAM", 16 * GIB, 12 * GIB)
    )
    seen = []

    def fake_list(binary, endpoints, *a, **kw):
        seen.append((binary, list(endpoints)))
        devices = cs.parse_device_list(listing, endpoints)
        st = cs._STATE
        if listing is LISTING and st is not None and getattr(st._backend, "is_loaded", False):
            for d in devices:
                if d.remote and d.endpoint not in st._started:
                    d.free_bytes = max(0, d.free_bytes - int(st._resident.get(d.key, 0)))
        return [d for d in devices if not d.remote or d.endpoint in endpoints], ""

    monkeypatch.setattr(cs, "list_devices", fake_list)

    async def reachable(host, port, timeout, **kw):
        return host not in DOWN

    DOWN.clear()
    monkeypatch.setattr(rs, "wait_for_port", reachable)
    return seen


def _request(**kw):
    base = dict(
        model_path = "/models/big.gguf",
        gguf_variant = None,
        llama_extra_args = None,
        max_seq_length = 0,
        cache_type_kv = None,
        disable_vision = False,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def test_a_cluster_load_gets_the_split_and_status_describes_it(monkeypatch):
    seen = _cluster_env(monkeypatch)
    out = run(ss.before_load(_request(llama_extra_args = ["--seed", "1"]), 4))
    assert seen == [("/b/llama-server", ["10.0.0.2:50052", "10.0.0.3:50052", "10.0.0.4:50052"])]
    assert out.llama_extra_args[:2] == ["--seed", "1"]
    assert out.llama_extra_args[2:4] == ["--rpc", "10.0.0.2:50052,10.0.0.3:50052,10.0.0.4:50052"]
    assert ss.enabled() is True and ss.current_topology() == "layer_split"
    status = ss.status()
    assert status["mode"] == "cluster" and status["topology"] == "layer_split"
    assert [d["name"] for d in status["plan"]["devices"]] == ["RPC0", "RPC1", "RPC2", "CUDA0"]
    backend = SimpleNamespace(
        _process = SimpleNamespace(args = ["/b/llama-server"] + out.llama_extra_args)
    )
    assert run(ss.after_load(backend, 4)) is False
    assert ss.current_topology() == "layer_split"
    assert ss._STATE is None or ss._STATE.peer_process is None


def test_configuring_a_cluster_detaches_a_two_spark_topology_first(monkeypatch):
    _cluster_env(monkeypatch)
    detached = []
    legacy = ss.state()
    legacy.peer_process = object()

    async def fake_detach():
        detached.append(1)
        legacy.peer_process = None

    monkeypatch.setattr(legacy, "detach", fake_detach)
    run(ss.before_load(_request(alongside = True), 4))
    assert detached == [], "an alongside load leaves the primary's topology alone"
    run(ss.before_load(_request(), 4))
    assert detached == [1]


def test_dropping_the_cluster_on_a_spark_frees_its_workers_before_the_spark_split(monkeypatch):
    stopped = []

    class Worker:
        alive = True

        async def stop(self, timeout = 10.0):
            stopped.append(1)

    cs.state().workers["10.0.0.2:50052"] = Worker()
    monkeypatch.setattr(ss, "enabled", lambda: True)
    seen = []

    async def legacy_before_load(self, request, n_parallel, **kw):
        seen.append(list(stopped))
        await self._on_plan_accepted()
        seen.append(list(stopped))
        return request

    monkeypatch.setattr(ss.SparkServing, "before_load", legacy_before_load)
    run(ss.before_load(_request(), 4))
    assert seen == [[], [1]], "the cluster's workers stop once the two-Spark plan is accepted"


def test_a_two_spark_refusal_keeps_the_cluster_serving(monkeypatch):
    stopped = []

    class Worker:
        alive = True

        async def stop(self, timeout = 10.0):
            stopped.append(1)

    cs.state().workers["10.0.0.2:50052"] = Worker()
    monkeypatch.setattr(ss, "enabled", lambda: True)

    async def legacy_before_load(self, request, n_parallel, **kw):
        raise ss.SparkLoadDoesNotFit("no topology holds this load")

    monkeypatch.setattr(ss.SparkServing, "before_load", legacy_before_load)
    with pytest.raises(ss.SparkLoadDoesNotFit):
        run(ss.before_load(_request(), 4))
    assert stopped == [] and "10.0.0.2:50052" in cs.state().workers


def test_a_refused_cluster_plan_leaves_the_two_spark_split_serving(monkeypatch):
    _cluster_env(monkeypatch, model_bytes = 900 * GIB)
    detached = []
    legacy = ss.state()
    legacy.peer_process = object()

    async def fake_detach():
        detached.append(1)

    monkeypatch.setattr(legacy, "detach", fake_detach)
    with pytest.raises(ss.SparkLoadDoesNotFit):
        run(ss.before_load(_request(), 4))
    assert detached == []


def test_a_refused_teardown_rolls_the_cluster_plan_back(monkeypatch):
    _cluster_env(monkeypatch)
    legacy = ss.state()
    legacy.peer_process = object()
    detached = []

    async def fake_detach():
        detached.append(1)

    def refuse():
        raise RuntimeError("409 active generations")

    monkeypatch.setattr(legacy, "detach", fake_detach)
    with pytest.raises(RuntimeError):
        run(ss.before_load(_request(), 4, confirm_teardown = refuse))
    state = cs.state()
    assert detached == [] and state.in_flight is False and state.block == []


def test_completion_goes_to_the_planner_that_took_the_load(monkeypatch):
    _cluster_env(monkeypatch)
    run(ss.before_load(_request(), 4))
    run(ss.after_load(SimpleNamespace(_process = SimpleNamespace(args = [])), 4))
    # A load planned while the cluster is off is the two-Spark path's, even if the config comes back.
    assert cs.state().in_flight is False
    legacy_after = []

    async def fake_after(self, backend, n):
        legacy_after.append(1)

    monkeypatch.setattr(ss.SparkServing, "after_load", fake_after)
    monkeypatch.setattr(ss, "enabled", lambda: True)
    run(ss.after_load(SimpleNamespace(), 4))
    assert legacy_after == [1]


def test_an_engine_load_detaches_a_two_spark_topology_once_it_commits(monkeypatch):
    _cluster_env(monkeypatch)
    detached = []
    legacy = ss.state()
    legacy.peer_process = object()

    async def fake_detach():
        detached.append(1)
        legacy.peer_process = None

    monkeypatch.setattr(legacy, "detach", fake_detach)
    run(ss.before_load(_request(engine = "vllm"), 4))
    run(ss.load_failed())
    assert detached == [], "a failed engine load leaves the split serving"
    run(ss.before_load(_request(engine = "vllm"), 4))
    assert detached == []
    run(ss.after_load(SimpleNamespace(is_loaded = False), 4))
    assert detached == [1]
    legacy.peer_process = object()
    run(ss.before_load(_request(engine = "vllm", alongside = True), 4))
    run(ss.after_load(SimpleNamespace(is_loaded = False), 4))
    assert detached == [1], "an alongside engine load keeps the primary's split"


def test_a_rejected_replacement_unloads_a_model_whose_cluster_workers_went(monkeypatch):
    _cluster_env(monkeypatch)
    out = run(ss.before_load(_request(), 4))
    unloaded = []
    backend = SimpleNamespace(
        _process = SimpleNamespace(args = ["/b/llama-server"] + out.llama_extra_args),
        is_loaded = True,
        unload_model = lambda: unloaded.append(1),
    )
    run(ss.after_load(backend, 4))

    class Worker:
        alive = True

        async def stop(self, timeout = 10.0):
            pass

    cs.state().workers["10.0.0.2:50052"] = Worker()
    monkeypatch.delenv(cs.ENV_NODES)
    monkeypatch.setattr(ss, "enabled", lambda: True)

    async def legacy_before_load(self, request, n_parallel, **kw):
        await self._on_plan_accepted()
        return request

    monkeypatch.setattr(ss.SparkServing, "before_load", legacy_before_load)
    run(ss.before_load(_request(), 4))
    assert not cs.state().workers
    run(ss.load_failed())
    assert unloaded == [1], "its remote layers are gone, so it cannot answer"


def test_a_failed_engine_load_detaches_the_two_spark_primary_it_unloaded(monkeypatch):
    _cluster_env(monkeypatch)
    detached = []
    legacy = ss.state()
    legacy.peer_process = object()
    legacy.attached_backend = SimpleNamespace(is_loaded = True)

    async def fake_detach():
        detached.append(1)
        legacy.peer_process = legacy.attached_backend = None

    monkeypatch.setattr(legacy, "detach", fake_detach)
    run(ss.before_load(_request(engine = "vllm"), 4))
    run(ss.load_failed())
    assert detached == [], "the primary still serves"
    run(ss.before_load(_request(engine = "vllm"), 4))
    legacy.attached_backend.is_loaded = False
    run(ss.load_failed())
    assert detached == [1]


def test_a_failed_cluster_replacement_unloads_the_two_spark_primary_it_orphaned(monkeypatch):
    _cluster_env(monkeypatch)
    unloaded = []
    legacy = ss.state()
    primary = SimpleNamespace(is_loaded = True, unload_model = lambda: unloaded.append(1))
    primary._process = object()
    legacy.attached_backend = primary
    legacy.peer_process = SimpleNamespace(name = "ggml-rpc-server", alive = True)

    async def fake_detach():
        legacy.peer_process = legacy.attached_backend = None

    monkeypatch.setattr(legacy, "detach", fake_detach)
    run(ss.before_load(_request(), 4))
    run(ss.load_failed())
    assert unloaded == [1], "its remote layers are gone, so it cannot answer"


def _llama_header(n_kv):
    return {
        "general.architecture": "llama",
        "llama.block_count": 80,
        "llama.attention.head_count": 64,
        "llama.attention.head_count_kv": n_kv,
        "llama.embedding_length": 8192,
    }


def test_detaching_a_two_spark_split_forces_a_relaunch(monkeypatch):
    _cluster_env(monkeypatch)
    legacy = ss.state()
    legacy.peer_process = SimpleNamespace(name = "ggml-rpc-server", alive = True)

    async def fake_detach():
        legacy.peer_process = legacy.attached_backend = None

    monkeypatch.setattr(legacy, "detach", fake_detach)
    monkeypatch.setattr(legacy, "_note_retired_split", lambda process: None)
    out = run(ss.before_load(_request(), 4))
    assert out.force_reload is True
    out = run(ss.before_load(_request(), 4))
    assert not getattr(out, "force_reload", False)


def test_a_malformed_cluster_config_keeps_a_serving_two_spark_split(monkeypatch):
    _cluster_env(monkeypatch)
    monkeypatch.setenv(cs.ENV_NODES, "bad host!")
    legacy = ss.state()
    legacy.peer_process = SimpleNamespace(name = "ggml-rpc-server", alive = True)
    detached = []

    async def fake_detach():
        detached.append(1)

    async def legacy_before_load(self, request, n_parallel, **kw):
        return "legacy"

    monkeypatch.setattr(legacy, "detach", fake_detach)
    monkeypatch.setattr(ss.SparkServing, "before_load", legacy_before_load)
    assert run(ss.before_load(_request(), 4)) == "legacy" and detached == []


def test_a_cancelled_cluster_plan_keeps_the_two_spark_split(monkeypatch):
    import asyncio as _asyncio

    _cluster_env(monkeypatch)
    peer = object()
    monkeypatch.setattr(ss, "_STATE", SimpleNamespace(peer_process = peer, router = None))
    detached = []
    ss._STATE.detach = lambda: detached.append(1)
    cancel = _asyncio.Event()
    cancel.set()
    out = run(ss.before_load(_request(), 4, cancel_event = cancel))
    assert detached == [] and not getattr(out, "force_reload", False)
    assert ss._STATE.peer_process is peer


def test_status_names_the_two_spark_split_a_malformed_config_keeps(monkeypatch):
    _cluster_env(monkeypatch)
    legacy = ss.state()
    legacy.peer_process = SimpleNamespace(name = "ggml-rpc-server", alive = True)
    legacy.topology = "layer_split"
    monkeypatch.setattr(ss.SparkServing, "status", lambda self: {"topology": self.topology})
    monkeypatch.setenv(cs.ENV_NODES, "bad host!")
    assert ss.current_topology() == "layer_split"
    assert ss.status()["topology"] == "layer_split" and ss.status()["cluster_config_error"]


# --pipeline-groups (unslothai/llama.cpp PR #187) on an N-node layer split.
LISTING_2 = """Available devices:
  RPC0: 10.0.0.2:50052 (24576 MiB, 23000 MiB free)
  CUDA0: NVIDIA L4 (24576 MiB, 23000 MiB free)
"""
KV_PER_CELL = 64 * 1024  # 4096 cells = 256 MiB of KV for one context


def _legacy_worker_on(
    monkeypatch,
    host = "10.0.0.2",
    port = 50052,
):
    legacy = ss.state()
    unloaded = []
    primary = SimpleNamespace(is_loaded = True, unload_model = lambda: unloaded.append(1))
    primary._process = object()
    legacy.attached_backend = primary
    legacy.peer_process = SimpleNamespace(
        name = "ggml-rpc-server",
        alive = True,
        peer = host,
        argv = ["/b/ggml-rpc-server", "-H", host, "-p", str(port)],
    )

    async def fake_detach():
        legacy.peer_process = legacy.attached_backend = None

    monkeypatch.setattr(legacy, "detach", fake_detach)

    async def ensure(self):
        # The two-Spark worker still holds the port: the managed one cannot bind it.
        held = legacy.peer_process is not None
        return {"10.0.0.2:50052": "ggml-rpc-server did not take 10.0.0.2:50052"} if held else {}

    monkeypatch.setattr(cs.ClusterServing, "_ensure_workers", ensure)
    return legacy, unloaded


def test_a_two_spark_worker_on_a_managed_nodes_port_is_retired_before_planning(monkeypatch):
    _cluster_env(monkeypatch)
    monkeypatch.setenv(cs.ENV_MANAGED, "1")
    legacy, _unloaded = _legacy_worker_on(monkeypatch)
    out = run(ss.before_load(_request(), 4))
    rpc = out.llama_extra_args[out.llama_extra_args.index("--rpc") + 1]
    assert "10.0.0.2:50052" in rpc.split(","), "the peer is planned on the first load"
    assert legacy.peer_process is None and out.force_reload is True


def test_a_refused_load_after_retiring_the_two_spark_worker_unloads_its_primary(monkeypatch):
    _cluster_env(monkeypatch, model_bytes = 900 * GIB)
    monkeypatch.setenv(cs.ENV_MANAGED, "1")
    _legacy, unloaded = _legacy_worker_on(monkeypatch)
    with pytest.raises(ss.SparkLoadDoesNotFit):
        run(ss.before_load(_request(), 4))
    assert unloaded == [1], "its remote layers went with the worker"


def test_a_two_spark_worker_on_another_port_is_left_to_the_plan(monkeypatch):
    _cluster_env(monkeypatch, model_bytes = 900 * GIB)
    monkeypatch.setenv(cs.ENV_MANAGED, "1")
    legacy, unloaded = _legacy_worker_on(monkeypatch, port = 50053)
    with pytest.raises(ss.SparkLoadDoesNotFit):
        run(ss.before_load(_request(), 4))
    assert legacy.peer_process is not None and unloaded == []


def test_a_two_spark_worker_recorded_by_address_matches_a_node_named_by_host_name(monkeypatch):
    _cluster_env(monkeypatch)
    monkeypatch.setenv(cs.ENV_NODES, "spark-peer.local,10.0.0.3,10.0.0.4")
    monkeypatch.setenv(cs.ENV_MANAGED, "1")
    _legacy_worker_on(monkeypatch)
    real = socket.getaddrinfo

    def resolve(host, *a, **k):
        if host == "spark-peer.local":
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.2", 0))]
        return real(host, *a, **k)

    monkeypatch.setattr(socket, "getaddrinfo", resolve)
    cs.state().refresh_nodes()
    assert ss._legacy_worker_holds_a_cluster_port()
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda host, *a, **k: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.9", 0))],
    )
    assert not ss._legacy_worker_holds_a_cluster_port(), "another machine on that port"


def test_two_spark_serving_answers_the_layer_split_hooks():
    """With two-Spark serving installed, every hook the routes call goes through it."""
    assert rs._provider() is ss
