# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""N-node serving over llama.cpp RPC: config, device listing, placement, launch args, reload
stripping, and that nothing of it runs (or changes the two-Spark path) without a cluster config."""

from __future__ import annotations

import asyncio
import json
import os
import re
import shlex
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.inference import cluster_serving as cs
from core.inference import spark_serving as ss

GIB = 2**30
MIB = 2**20

# Hosts the fake reachability probe reports down.
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
    ss.reset_for_tests()
    yield
    cs.reset_for_tests()
    ss.reset_for_tests()


def test_nodes_from_the_env_and_from_json(tmp_path):
    nodes = cs.parse_nodes_env("10.0.0.2, me@mac-studio.local:50060", managed = True)
    assert [(n.host, n.rpc_port, n.ssh_user, n.managed) for n in nodes] == [
        ("10.0.0.2", 50052, None, True),
        ("mac-studio.local", 50060, "me", True),
    ]
    assert (
        nodes[1].endpoint == "mac-studio.local:50060"
        and nodes[1].ssh_target == "me@mac-studio.local"
    )
    data = {"nodes": [{"host": "10.0.0.3", "rpc_port": 50053, "managed": True}, "10.0.0.4"]}
    parsed = cs.parse_nodes_json(data)
    assert [(n.endpoint, n.managed) for n in parsed] == [
        ("10.0.0.3:50053", True),
        ("10.0.0.4:50052", False),
    ]


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "10.0.0.2,10.0.0.2",
        "host:0",
        "host:99999",
        "[::1]:50052",
        "@host",
        "bad host",
        "h;rm -rf",
    ],
)
def test_bad_node_lists_are_refused(bad):
    with pytest.raises(cs.ClusterConfigError):
        cs.parse_nodes_env(bad)


def test_device_listing_maps_every_rpc_device_to_its_endpoint():
    endpoints = ["10.0.0.2:50052", "10.0.0.3:50052", "10.0.0.4:50052"]
    devices = cs.parse_device_list(LISTING, endpoints)
    assert [(d.name, d.endpoint) for d in devices] == [
        ("RPC0", "10.0.0.2:50052"),
        ("RPC1", "10.0.0.3:50052"),
        ("RPC2", "10.0.0.4:50052"),
        ("CUDA0", None),
    ]
    assert devices[1].free_bytes == 60000 * MIB
    # Two devices behind one endpoint (a multi-GPU worker) both belong to it.
    two = cs.parse_device_list(
        "  RPC0: 10.0.0.2:50052 (10 MiB, 9 MiB free)\n  RPC1: 10.0.0.2:50052 (10 MiB, 9 MiB free)\n",
        ["10.0.0.2:50052"],
    )
    assert [d.endpoint for d in two] == ["10.0.0.2:50052"] * 2


def _devices():
    return cs.parse_device_list(LISTING, ["10.0.0.2:50052", "10.0.0.3:50052", "10.0.0.4:50052"])


def test_a_model_that_fits_here_is_not_split():
    plan = cs.plan_cluster(40 * GIB, 4 * GIB, 0, _devices())
    assert plan.topology == "single" and [d.name for d in plan.devices] == ["CUDA0"]


TWO_LOCAL = """Available devices:
  RPC0: 10.0.0.2:50052 (122502 MiB, 118000 MiB free)
  CUDA0: NVIDIA H100 (81559 MiB, 80000 MiB free)
  CUDA1: NVIDIA H100 (81559 MiB, 80000 MiB free)
  BLAS: Accelerate (0 MiB, 0 MiB free)
"""


def test_every_local_gpu_counts_and_sidecars_go_on_the_first():
    devices = cs.parse_device_list(TWO_LOCAL, ["10.0.0.2:50052"])
    assert [d.name for d in devices] == ["RPC0", "CUDA0", "CUDA1"]
    # 120 GiB fits the two local cards together, so it is not sent over the wire.
    plan = cs.plan_cluster(120 * GIB, 0, 0, devices)
    assert plan.topology == "single" and [d.name for d in plan.devices] == ["CUDA0", "CUDA1"]
    plan = cs.plan_cluster(220 * GIB, 0, 4 * GIB, devices)
    assert plan.topology == "layer_split"
    assert [d.name for d in plan.devices] == ["RPC0", "CUDA0", "CUDA1"]
    usable = [cs.usable_bytes(d) for d in devices]
    usable[1] -= 4 * GIB
    for w, u in zip(plan.weights, usable):
        assert w == pytest.approx(u / sum(usable))


def test_an_endpoint_is_matched_whole_not_as_a_prefix_of_another():
    endpoints = ["10.0.0.2:5005", "10.0.0.2:50052"]
    listing = (
        "  RPC0: 10.0.0.2:5005 (10 MiB, 9 MiB free)\n  RPC1: 10.0.0.2:50052 (10 MiB, 9 MiB free)\n"
    )
    assert [d.endpoint for d in cs.parse_device_list(listing, endpoints)] == endpoints
    # A build whose description says more still maps by the whole endpoint.
    listing = "  RPC0: RPC backend at 10.0.0.2:50052 (10 MiB, 9 MiB free)\n"
    assert cs.parse_device_list(listing, endpoints)[0].endpoint == "10.0.0.2:50052"


def test_a_large_model_is_split_by_usable_memory_with_this_machine_last():
    plan = cs.plan_cluster(230 * GIB, 20 * GIB, 2 * GIB, _devices())
    assert plan.topology == "layer_split"
    assert [d.name for d in plan.devices] == ["RPC0", "RPC1", "RPC2", "CUDA0"]
    assert abs(sum(plan.weights) - 1.0) < 1e-9
    usable = [cs.usable_bytes(d) for d in _devices()]
    usable[-1] -= 2 * GIB  # the coordinator also holds the sidecars
    for w, u in zip(plan.weights, usable):
        assert w == pytest.approx(u / sum(usable))
    # Every device's share of weights + KV fits its own usable memory, the sidecars included.
    for d, w, u in zip(plan.devices, plan.weights, usable):
        assert w * 250 * GIB <= u


def test_a_load_no_set_of_nodes_holds_is_refused_with_the_numbers():
    plan = cs.plan_cluster(400 * GIB, 0, 0, _devices())
    assert plan.topology == "refuse" and "add a node" in plan.reason
    assert cs.plan_cluster(None, 0, 0, _devices()).topology == "single"


def test_forced_topologies():
    assert (
        cs.plan_cluster(10 * GIB, 0, 0, _devices(), forced = "layer_split").topology == "layer_split"
    )
    assert cs.plan_cluster(250 * GIB, 0, 0, _devices(), forced = "single").topology == "single"


def test_a_coordinator_without_an_accelerator_splits_over_the_workers_only():
    remote_only = [d for d in _devices() if d.remote]
    plan = cs.plan_cluster(150 * GIB, 0, 0, remote_only)
    assert plan.topology == "layer_split" and [d.name for d in plan.devices] == [
        "RPC0",
        "RPC1",
        "RPC2",
    ]
    assert cs.plan_cluster(150 * GIB, 0, 1 * GIB, remote_only).topology == "refuse"


def test_split_args_are_one_rpc_list_with_matching_devices_and_weights():
    plan = cs.plan_cluster(250 * GIB, 0, 0, _devices())
    args = cs.split_extra_args(["10.0.0.2:50052", "10.0.0.3:50052", "10.0.0.4:50052"], plan)
    assert args[:2] == ["--rpc", "10.0.0.2:50052,10.0.0.3:50052,10.0.0.4:50052"]
    assert args[2:4] == ["--device", "RPC0,RPC1,RPC2,CUDA0"]
    assert args[4:6] == ["-sm", "layer"] and args[-2:] == ["--cache-ram", "0"]
    weights = [float(w) for w in args[7].split(",")]
    assert len(weights) == 4 and abs(sum(weights) - 1.0) < 1e-3


def test_only_the_recorded_block_is_stripped_on_inheritance():
    plan = cs.plan_cluster(250 * GIB, 0, 0, _devices())
    block = cs.split_extra_args(["10.0.0.2:50052", "10.0.0.3:50052", "10.0.0.4:50052"], plan)
    cs._remember_block(block)
    assert cs.strip_managed_block(["--seed", "1"] + block) == ["--seed", "1"]
    # Through the helper the reload path calls, next to the two-Spark block.
    assert ss.strip_layer_split_extra_args(["--seed", "1"] + block) == ["--seed", "1"]
    users = ["--rpc", "10.9.9.9:1", "--device", "RPC0,CUDA0"]
    assert ss.strip_layer_split_extra_args(users) == users
    cs.reset_for_tests()
    assert cs.strip_managed_block(["--seed", "1"] + block) == ["--seed", "1"] + block


def test_nothing_changes_without_a_cluster_config(monkeypatch):
    assert not cs.configured()
    calls = []
    monkeypatch.setattr(cs.ClusterServing, "before_load", lambda *a, **k: calls.append(1))
    request = SimpleNamespace(model_path = "m.gguf", llama_extra_args = None)
    monkeypatch.setattr(ss, "is_spark", lambda: False)
    assert run(ss.before_load(request, 4)) is request
    assert ss.status() == {"enabled": False, "topology": None, "reason": "not a paired DGX Spark"}
    assert calls == [] and cs._STATE is None


def test_the_spark_toggle_also_turns_the_cluster_off(monkeypatch):
    monkeypatch.setenv(cs.ENV_NODES, "10.0.0.2")
    monkeypatch.setenv(ss.ENV_TOGGLE, "0")
    monkeypatch.setattr(ss, "is_spark", lambda: False)
    request = SimpleNamespace(model_path = "m.gguf", llama_extra_args = None)
    assert run(ss.before_load(request, 4)) is request and cs._STATE is None
    assert ss.status()["enabled"] is False


def _cluster_env(
    monkeypatch,
    listing = LISTING,
    model_bytes = 250 * GIB,
):
    monkeypatch.setenv(cs.ENV_NODES, "10.0.0.2,10.0.0.3,10.0.0.4")
    monkeypatch.setattr(ss, "llama_server_binary", lambda: "/b/llama-server")
    monkeypatch.setattr(ss, "cached_repo_file", lambda path, variant: path)
    monkeypatch.setattr(ss, "gguf_size_bytes", lambda path: model_bytes if path else None)
    monkeypatch.setattr(ss, "estimate_kv_bytes", lambda *a, **k: 0)
    monkeypatch.setattr(ss, "auto_projector_file", lambda *a, **k: None)
    monkeypatch.setattr(cs, "_remote_gguf_header", lambda *a, **k: ({}, {}))
    monkeypatch.setattr(cs, "_local_gguf_header", lambda *a, **k: ({}, {}))
    monkeypatch.setattr(cs, "_ctx_compute_bytes", lambda *a, **k: 0)
    seen = []

    def fake_list(binary, endpoints):
        # llama.cpp lists only the endpoints it was given.
        seen.append((binary, list(endpoints)))
        devices = cs.parse_device_list(listing, endpoints)
        return [d for d in devices if not d.remote or d.endpoint in endpoints], ""

    monkeypatch.setattr(cs, "list_devices", fake_list)

    async def reachable(host, port, timeout, **kw):
        return host not in DOWN

    DOWN.clear()
    monkeypatch.setattr(ss, "wait_for_port", reachable)
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
    # The two-Spark state was never created.
    assert ss._STATE is None or ss._STATE.peer_process is None


def test_a_cluster_load_that_fits_here_is_left_alone(monkeypatch):
    _cluster_env(monkeypatch, model_bytes = 10 * GIB)
    request = _request()
    assert run(ss.before_load(request, 4)) is request
    assert ss.status()["topology"] == "single"


def test_a_cluster_load_nothing_holds_is_refused_and_the_old_state_kept(monkeypatch):
    _cluster_env(monkeypatch)
    run(ss.before_load(_request(), 4))
    _cluster_env(monkeypatch, model_bytes = 900 * GIB)
    with pytest.raises(ss.SparkLoadDoesNotFit):
        run(ss.before_load(_request(), 4))
    assert ss.status()["topology"] == "layer_split"


@pytest.mark.parametrize(
    "kw, reason",
    [
        ({"llama_extra_args": ["--rpc", "10.9.9.9:1"]}, "caller supplied --rpc"),
        ({"gpu_ids": [1]}, "explicit GPU selection"),
    ],
)
def test_caller_placement_is_left_to_the_caller(monkeypatch, kw, reason):
    _cluster_env(monkeypatch)
    request = _request(**kw)
    assert run(ss.before_load(request, 4)) is request
    assert reason in ss.status()["reason"]


@pytest.mark.parametrize("kw", [{"alongside": True}, {"force_alongside": True}, {"engine": "vllm"}])
def test_alongside_and_other_engines_pass_through(monkeypatch, kw):
    seen = _cluster_env(monkeypatch)
    request = _request(**kw)
    assert run(ss.before_load(request, 4)) is request and seen == []


def test_a_bad_cluster_config_serves_here_and_says_why(monkeypatch, tmp_path):
    (tmp_path / "cluster.json").write_text(json.dumps({"nodes": "nope"}), encoding = "utf-8")
    monkeypatch.setattr(ss, "is_spark", lambda: False)
    request = _request()
    assert run(ss.before_load(request, 4)) is request
    status = ss.status()
    assert status["topology"] == "single" and "nodes" in status["config_error"]


def test_a_failed_load_puts_the_previous_topology_back(monkeypatch):
    _cluster_env(monkeypatch)
    out = run(ss.before_load(_request(), 4))
    backend = SimpleNamespace(
        _process = SimpleNamespace(args = ["/b/llama-server"] + out.llama_extra_args),
        is_loaded = True,
    )
    run(ss.after_load(backend, 4))
    _cluster_env(monkeypatch, model_bytes = 10 * GIB)
    run(ss.before_load(_request(), 4))
    assert ss.status()["topology"] == "single"
    run(ss.load_failed())  # the split it would have replaced is still loaded
    assert ss.status()["topology"] == "layer_split"


def test_an_unreachable_worker_is_left_out_and_named(monkeypatch):
    seen = _cluster_env(monkeypatch, model_bytes = 150 * GIB)
    DOWN.add("10.0.0.3")
    out = run(ss.before_load(_request(), 4))
    assert seen[-1][1] == ["10.0.0.2:50052", "10.0.0.4:50052"]
    assert "10.0.0.3:50052" in ss.status()["last_error"]
    assert out.llama_extra_args[1] == "10.0.0.2:50052,10.0.0.4:50052"


def test_an_unreachable_worker_names_the_connect_failure():
    stderr = (
        "/src/ggml-rpc.cpp:743: Failed to connect to 10.0.0.3:50052\n\n"
        "/b/libllama-common.so.0(+0x1873)[0x72f9483c79e3]\n"
    )
    proc = SimpleNamespace(stdout = "", stderr = stderr)
    import subprocess as sp

    orig = sp.run
    try:
        cs.subprocess.run = lambda *a, **k: proc
        devices, error = cs.list_devices("/b/llama-server", ["10.0.0.3:50052"])
    finally:
        cs.subprocess.run = orig
    assert devices == [] and error == "Failed to connect to 10.0.0.3:50052"


def test_an_endpoint_with_no_device_is_reported(monkeypatch):
    listing = "  RPC0: 10.0.0.2:50052 (122502 MiB, 118000 MiB free)\n  CUDA0: GB10 (122502 MiB, 110000 MiB free)\n"
    _cluster_env(monkeypatch, listing = listing, model_bytes = 150 * GIB)
    out = run(ss.before_load(_request(), 4))
    assert "10.0.0.3:50052" in ss.status()["last_error"]
    assert out.llama_extra_args[1] == "10.0.0.2:50052"


def test_an_environment_projector_is_charged_to_this_machine(monkeypatch, tmp_path):
    _cluster_env(monkeypatch, model_bytes = 90 * GIB)
    monkeypatch.setattr(ss, "llama_server_supports", lambda flag, binary = None: True)
    mmproj = tmp_path / "mmproj.gguf"
    mmproj.write_bytes(b"x")
    request = _request()
    assert run(ss.before_load(request, 4)) is request  # 90 GiB fits here alone
    sizes = {str(mmproj): 10 * GIB}
    monkeypatch.setattr(ss, "gguf_size_bytes", lambda path: sizes.get(str(path), 90 * GIB))
    monkeypatch.setenv("LLAMA_ARG_MMPROJ", str(mmproj))
    run(ss.before_load(_request(), 4))
    assert ss.status()["topology"] == "layer_split"


def test_a_cancelled_load_stops_preparing(monkeypatch):
    _cluster_env(monkeypatch)
    event = asyncio.Event()
    event.set()
    request = _request()
    assert run(ss.before_load(request, 4, cancel_event = event)) is request
    assert "cancelled" in ss.status()["reason"]


def test_the_model_being_replaced_counts_as_freed(monkeypatch):
    _cluster_env(monkeypatch)
    out = run(ss.before_load(_request(), 4))
    plan = cs.state().plan
    backend = SimpleNamespace(
        _process = SimpleNamespace(args = ["/b/llama-server"] + out.llama_extra_args),
        is_loaded = True,
    )
    run(ss.after_load(backend, 4))
    # The probe now sees the served model's buffers as used.
    held = plan.resident()
    lines = []
    for d in _devices():
        free = d.free_bytes - held.get(d.key, 0)
        lines.append(
            f"  {d.name}: {d.description} ({d.total_bytes // MIB} MiB, {int(free) // MIB} MiB free)"
        )
    _cluster_env(monkeypatch, listing = "\n".join(lines) + "\n")
    run(ss.before_load(_request(model_path = "/models/other.gguf"), 4))
    assert ss.status()["topology"] == "layer_split"
    # Once it is unloaded, nothing is added back.
    backend.is_loaded = False
    with pytest.raises(ss.SparkLoadDoesNotFit):
        run(ss.before_load(_request(model_path = "/models/other.gguf"), 4))


def test_managed_workers_are_started_over_ssh_and_kept_across_reloads(monkeypatch):
    _cluster_env(monkeypatch)
    monkeypatch.setenv(cs.ENV_NODES, "me@10.0.0.2,10.0.0.3,10.0.0.4")
    monkeypatch.setenv(cs.ENV_MANAGED, "1")
    monkeypatch.setattr(
        ss, "rpc_server_binary", lambda: "/home/me/.unsloth/llama.cpp/build/bin/ggml-rpc-server"
    )
    ssh = []

    async def fake_ssh(target, remote, **kw):
        ssh.append(target)
        return 0, "$HOME/.unsloth/llama.cpp/build/bin/ggml-rpc-server\n", ""

    started = []

    async def fake_start(self):
        started.append((self.peer, list(self.argv)))
        self.proc = SimpleNamespace(returncode = None)

    async def fake_stop(self, timeout = 10.0):
        self.proc = SimpleNamespace(returncode = 0)

    async def owns(process, host, port, timeout, **kw):
        return True

    monkeypatch.setattr(ss, "ssh_run", fake_ssh)
    monkeypatch.setattr(ss.PeerProcess, "start", fake_start)
    monkeypatch.setattr(ss.PeerProcess, "stop", fake_stop)
    monkeypatch.setattr(ss, "wait_for_own_port", owns)
    run(ss.before_load(_request(), 4))
    assert [p for p, _ in started] == ["me@10.0.0.2", "10.0.0.3", "10.0.0.4"]
    assert started[0][1] == [
        "sh",
        "-c",
        cs._CUDA_RUNTIME,
        "$HOME/.unsloth/llama.cpp/build/bin/ggml-rpc-server",
        "-H",
        "10.0.0.2",
        "-p",
        "50052",
    ]
    run(ss.before_load(_request(), 4))
    assert len(started) == 3, "running workers are reused"

    # A forced reload keeps them too: restarting one would drop the served model's buffers.
    run(ss.before_load(_request(force_reload = True), 4))
    assert len(started) == 3
    # A node dropped from the config is retired once a load without it commits.
    workers = dict(cs.state().workers)
    monkeypatch.setenv(cs.ENV_NODES, "me@10.0.0.2")
    _cluster_env(monkeypatch, model_bytes = 150 * GIB)
    monkeypatch.setenv(cs.ENV_NODES, "me@10.0.0.2")
    out = run(ss.before_load(_request(), 4))
    assert set(cs.state().workers) == set(workers), "nothing is stopped before the load commits"
    backend = SimpleNamespace(
        _process = SimpleNamespace(args = ["/b/llama-server"] + out.llama_extra_args)
    )
    run(ss.after_load(backend, 4))
    assert list(cs.state().workers) == ["10.0.0.2:50052"]
    assert not workers["10.0.0.3:50052"].alive and not workers["10.0.0.4:50052"].alive
    run(ss.shutdown())
    assert cs.state().workers == {}


def test_ssh_takes_a_nodes_own_login_and_keeps_the_spark_one(monkeypatch):
    monkeypatch.setattr(ss, "_ssh_user", lambda: "studio")
    assert ss.ssh_argv("10.0.0.2", "true")[-2] == "studio@10.0.0.2"
    assert ss.ssh_argv("me@10.0.0.2", "true")[-2] == "me@10.0.0.2"


def test_caller_placement_after_a_split_leaves_no_plan_to_count_as_resident(monkeypatch):
    _cluster_env(monkeypatch)
    run(ss.before_load(_request(), 4))
    assert cs.state().plan.topology == "layer_split"
    run(ss.before_load(_request(gpu_ids = [0]), 4))
    assert cs.state().plan is None and cs.state().block == []
    run(ss.after_load(SimpleNamespace(_process = SimpleNamespace(args = []), is_loaded = True), 4))
    assert cs.state()._resident == {}


def test_an_alongside_load_keeps_the_primarys_residency(monkeypatch):
    _cluster_env(monkeypatch)
    out = run(ss.before_load(_request(), 4))
    backend = SimpleNamespace(
        _process = SimpleNamespace(args = ["/b/llama-server"] + out.llama_extra_args),
        is_loaded = True,
    )
    run(ss.after_load(backend, 4))
    held = dict(cs.state()._resident)
    assert held
    run(ss.before_load(_request(alongside = True), 4))
    run(ss.after_load(SimpleNamespace(), 4))
    assert cs.state()._resident == held
    run(ss.before_load(_request(engine = "llama.cpp"), 4))
    run(ss.after_load(SimpleNamespace(), 4))
    assert cs.state()._resident == {}


def test_an_environment_projector_replaces_the_automatic_one(monkeypatch, tmp_path):
    _cluster_env(monkeypatch, model_bytes = 80 * GIB)
    mmproj = tmp_path / "mmproj.gguf"
    mmproj.write_bytes(b"x")
    sizes = {str(mmproj): 5 * GIB, "/models/auto-mmproj.gguf": 15 * GIB}
    monkeypatch.setattr(ss, "gguf_size_bytes", lambda path: sizes.get(str(path), 80 * GIB))
    monkeypatch.setattr(ss, "auto_projector_file", lambda *a, **k: "/models/auto-mmproj.gguf")
    monkeypatch.setenv("LLAMA_ARG_MMPROJ", str(mmproj))
    request = _request()
    # 80 + 5 fits here; charging the automatic 15 as well would not.
    assert run(ss.before_load(request, 4)) is request
    assert ss.status()["topology"] == "single"


def test_an_rpc_device_tied_to_no_worker_is_not_planned_with(monkeypatch):
    listing = LISTING.replace("RPC1: 10.0.0.3:50052", "RPC1: some other build")
    _cluster_env(monkeypatch, listing = listing)
    monkeypatch.setattr(
        cs,
        "list_devices",
        lambda binary, endpoints: (cs.parse_device_list(listing, endpoints), ""),
    )
    request = _request()
    assert run(ss.before_load(request, 4)) is request
    assert "could not tell which worker" in ss.status()["reason"]


def test_an_uncached_hub_model_reserves_room_for_its_kv(monkeypatch):
    _cluster_env(monkeypatch, model_bytes = 85 * GIB)
    monkeypatch.setattr(ss, "cached_repo_file", lambda path, variant: None)
    monkeypatch.setattr(ss, "remote_gguf_size_bytes", lambda *a, **k: 85 * GIB)
    monkeypatch.setattr(cs, "_remote_gguf_header", lambda *a, **k: (_llama_header(8), {}))
    run(ss.before_load(_request(max_seq_length = 131072), 4))
    # 85 GiB of weights alone fits here; with its 40 GiB of KV it does not, so it is split.
    assert ss.status()["topology"] == "layer_split"
    run(ss.before_load(_request(max_seq_length = 0), 4))
    assert ss.status()["topology"] == "single"


def test_removing_the_config_retires_the_workers_once_a_load_commits(monkeypatch):
    stopped = []

    class Worker:
        alive = True

        async def stop(self, timeout = 10.0):
            stopped.append(1)

    cs.state().workers["10.0.0.2:50052"] = Worker()
    monkeypatch.setattr(ss, "is_spark", lambda: False)
    run(ss.before_load(_request(alongside = True), 4))
    run(ss.after_load(SimpleNamespace(), 4))
    assert stopped == [], "an alongside load does not replace the model the workers serve"
    run(ss.before_load(_request(), 4))
    assert stopped == [], "nothing stops before the load commits"
    run(ss.after_load(SimpleNamespace(), 4))
    assert stopped == [1] and cs.state().workers == {}


def test_the_projector_is_pinned_to_the_local_device_it_was_charged_to(monkeypatch):
    _cluster_env(monkeypatch)
    monkeypatch.setattr(ss, "llama_server_supports", lambda flag, binary = None: True)
    out = run(ss.before_load(_request(), 4))
    args = out.llama_extra_args
    assert args[args.index("--mmproj-device") + 1] == "CUDA0"
    assert "--device-draft" not in args
    # The caller's own choice stands, and a build without the flag gets none.
    out = run(ss.before_load(_request(llama_extra_args = ["-mmdev", "none"]), 4))
    assert "--mmproj-device" not in out.llama_extra_args
    monkeypatch.setattr(ss, "llama_server_supports", lambda flag, binary = None: False)
    out = run(ss.before_load(_request(), 4))
    assert "--mmproj-device" not in out.llama_extra_args


def test_a_disabled_projector_is_not_charged(monkeypatch):
    _cluster_env(monkeypatch, model_bytes = 85 * GIB)
    monkeypatch.setattr(ss, "llama_server_supports", lambda flag, binary = None: True)
    sizes = {"/models/auto-mmproj.gguf": 15 * GIB}
    monkeypatch.setattr(ss, "gguf_size_bytes", lambda path: sizes.get(str(path), 85 * GIB))
    monkeypatch.setattr(ss, "auto_projector_file", lambda *a, **k: "/models/auto-mmproj.gguf")
    run(ss.before_load(_request(), 4))
    assert ss.status()["topology"] == "layer_split"
    request = _request(llama_extra_args = ["--no-mmproj"])
    assert run(ss.before_load(request, 4)) is request
    assert ss.status()["topology"] == "single"


def test_turning_serving_off_retires_the_cluster_at_commit(monkeypatch):
    _cluster_env(monkeypatch)
    out = run(ss.before_load(_request(), 4))
    backend = SimpleNamespace(
        _process = SimpleNamespace(args = ["/b/llama-server"] + out.llama_extra_args),
        is_loaded = True,
    )
    run(ss.after_load(backend, 4))
    assert cs.state()._backend is backend and not cs.state().workers  # unmanaged nodes
    monkeypatch.setenv(ss.ENV_TOGGLE, "0")
    run(ss.before_load(_request(), 4))
    run(ss.after_load(SimpleNamespace(is_loaded = True), 4))
    assert cs.state()._backend is None and cs.state()._resident == {}


def test_a_worker_the_committed_split_uses_is_kept_when_it_turns_unmanaged(monkeypatch):
    _cluster_env(monkeypatch)
    stopped = []

    class Worker:
        alive = True

        async def stop(self, timeout = 10.0):
            stopped.append(1)

    cs.state().workers["10.0.0.3:50052"] = Worker()
    out = run(ss.before_load(_request(), 4))
    assert "10.0.0.3:50052" in out.llama_extra_args[1]
    run(ss.after_load(SimpleNamespace(_process = SimpleNamespace(args = out.llama_extra_args)), 4))
    assert stopped == [] and "10.0.0.3:50052" in cs.state().workers


def test_one_managed_worker_failing_does_not_stop_the_split(monkeypatch):
    _cluster_env(monkeypatch, model_bytes = 150 * GIB)
    monkeypatch.setenv(cs.ENV_MANAGED, "1")
    monkeypatch.setattr(ss, "rpc_server_binary", lambda: "/b/ggml-rpc-server")

    async def fake_ssh(target, remote, **kw):
        if target == "10.0.0.3":
            return 1, "MISSING\n", ""
        return 0, "/b/ggml-rpc-server\n", ""

    async def fake_start(self):
        self.proc = SimpleNamespace(returncode = None)

    async def owns(process, host, port, timeout, **kw):
        return True

    monkeypatch.setattr(ss, "ssh_run", fake_ssh)
    monkeypatch.setattr(ss.PeerProcess, "start", fake_start)
    monkeypatch.setattr(ss, "wait_for_own_port", owns)
    out = run(ss.before_load(_request(), 4))
    assert ss.status()["topology"] == "layer_split"
    assert (
        out.llama_extra_args[out.llama_extra_args.index("--rpc") + 1]
        == "10.0.0.2:50052,10.0.0.4:50052"
    )
    assert "10.0.0.3 has no ggml-rpc-server" in ss.status()["last_error"]


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
        await self._on_plan_accepted()  # the two-Spark planner accepted the load
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


def test_a_sidecar_larger_than_the_device_it_loads_on_is_refused():
    devices = cs.parse_device_list(TWO_LOCAL, ["10.0.0.2:50052"])
    plan = cs.plan_cluster(220 * GIB, 0, 80 * GIB, devices)
    assert plan.topology == "refuse" and "CUDA0" in plan.reason


def test_the_managed_override_applies_to_json_nodes(monkeypatch, tmp_path):
    (tmp_path / "cluster.json").write_text(
        json.dumps({"nodes": [{"host": "10.0.0.2"}, {"host": "10.0.0.3"}]}), encoding = "utf-8"
    )
    assert [n.managed for n in cs.load_nodes()] == [False, False]
    monkeypatch.setenv(cs.ENV_MANAGED, "1")
    assert [n.managed for n in cs.load_nodes()] == [True, True]


def test_manual_gpu_memory_mode_is_left_to_the_caller(monkeypatch):
    seen = _cluster_env(monkeypatch)
    request = _request(gpu_memory_mode = "manual")
    assert run(ss.before_load(request, 4)) is request
    assert "manual" in ss.status()["reason"] and seen == []


def test_a_load_that_bypassed_the_planner_clears_the_served_split(monkeypatch):
    _cluster_env(monkeypatch)
    out = run(ss.before_load(_request(), 4))
    backend = SimpleNamespace(
        _process = SimpleNamespace(args = ["/b/llama-server"] + out.llama_extra_args),
        is_loaded = True,
    )
    run(ss.after_load(backend, 4))
    assert cs.state()._resident
    run(ss.reconcile_internal_load())
    state = cs.state()
    assert state.topology == "single" and state.block == [] and state._resident == {}


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


def test_a_failed_replacement_forgets_a_split_it_already_tore_down(monkeypatch):
    _cluster_env(monkeypatch)
    out = run(ss.before_load(_request(), 4))
    backend = SimpleNamespace(
        _process = SimpleNamespace(args = ["/b/llama-server"] + out.llama_extra_args),
        is_loaded = True,
    )
    run(ss.after_load(backend, 4))
    run(ss.before_load(_request(model_path = "/models/other.gguf"), 4))
    backend.is_loaded = False  # the replacement unloaded it, then failed to start
    run(ss.load_failed())
    assert ss.status()["topology"] == "single" and ss.status()["split_args"] is None


def test_a_worker_that_breaks_the_combined_listing_is_dropped_and_listed_again(monkeypatch):
    _cluster_env(monkeypatch, model_bytes = 150 * GIB)
    calls = []

    def fake_list(binary, endpoints):
        calls.append(list(endpoints))
        if "10.0.0.3:50052" in endpoints:
            return [], "Failed to connect to 10.0.0.3:50052"
        return [
            d
            for d in cs.parse_device_list(LISTING, endpoints)
            if not d.remote or d.endpoint in endpoints
        ], ""

    monkeypatch.setattr(cs, "list_devices", fake_list)
    out = run(ss.before_load(_request(), 4))
    assert len(calls) == 2 and "10.0.0.3:50052" not in calls[1]
    assert (
        out.llama_extra_args[out.llama_extra_args.index("--rpc") + 1]
        == "10.0.0.2:50052,10.0.0.4:50052"
    )


@pytest.mark.parametrize(
    "extra",
    [["--device", "CUDA1"], ["-sm", "none"], ["--tensor-split=1,0"], ["--main-gpu", "1"]],
)
def test_caller_device_or_split_settings_are_left_alone(monkeypatch, extra):
    seen = _cluster_env(monkeypatch)
    request = _request(llama_extra_args = extra)
    assert run(ss.before_load(request, 4)) is request
    assert "placement is theirs" in ss.status()["reason"] and seen == []


def test_an_engine_load_clears_the_served_split(monkeypatch):
    _cluster_env(monkeypatch)
    out = run(ss.before_load(_request(), 4))
    backend = SimpleNamespace(
        _process = SimpleNamespace(args = ["/b/llama-server"] + out.llama_extra_args),
        is_loaded = True,
    )
    run(ss.after_load(backend, 4))
    run(ss.before_load(_request(engine = "vllm"), 4))
    run(ss.after_load(SimpleNamespace(), 4))
    status = ss.status()
    assert status["topology"] == "single" and status["split_args"] is None


def test_adapters_are_spread_with_the_layers_not_charged_to_this_machine(monkeypatch, tmp_path):
    lora = tmp_path / "adapter.gguf"
    lora.write_bytes(b"x")
    _cluster_env(monkeypatch, model_bytes = 150 * GIB)
    sizes = {str(lora): 20 * GIB}
    monkeypatch.setattr(ss, "gguf_size_bytes", lambda path: sizes.get(str(path), 150 * GIB))
    run(ss.before_load(_request(llama_extra_args = ["--lora", str(lora)]), 4))
    plan = ss.status()["plan"]
    assert plan["need_gib"] == pytest.approx(170.0, abs = 0.01)
    weights = [d["weight"] for d in plan["devices"]]
    usable = [cs.usable_bytes(d) for d in _devices()]
    # No coordinator deduction: the weights follow raw usable memory.
    assert weights == pytest.approx([u / sum(usable) for u in usable], abs = 1e-3)


def test_the_main_gpu_environment_pin_is_caller_placement(monkeypatch):
    seen = _cluster_env(monkeypatch)
    monkeypatch.setenv("LLAMA_ARG_MAIN_GPU", "1")
    request = _request()
    assert run(ss.before_load(request, 4)) is request and seen == []


def test_a_load_completes_through_the_cluster_even_if_the_config_went_away(monkeypatch):
    _cluster_env(monkeypatch)
    out = run(ss.before_load(_request(), 4))
    monkeypatch.delenv(cs.ENV_NODES)  # removed while the load was running
    backend = SimpleNamespace(
        _process = SimpleNamespace(args = ["/b/llama-server"] + out.llama_extra_args),
        is_loaded = True,
    )
    run(ss.after_load(backend, 4))
    assert cs.state()._backend is backend and not cs.state().in_flight


def test_an_inherited_context_size_is_budgeted(monkeypatch):
    _cluster_env(monkeypatch, model_bytes = 85 * GIB)
    monkeypatch.setattr(ss, "estimate_kv_bytes", lambda path, ctx, *a, **k: ctx * 2 * 2**20)
    run(ss.before_load(_request(), 4))
    assert ss.status()["topology"] == "single"
    monkeypatch.setenv("LLAMA_ARG_CTX_SIZE", "16384")  # 16384 * 2 MiB = 32 GiB of KV
    run(ss.before_load(_request(), 4))
    assert ss.status()["topology"] == "layer_split"


def test_no_worker_is_started_for_a_model_that_is_not_a_sizable_gguf(monkeypatch):
    _cluster_env(monkeypatch)
    monkeypatch.setenv(cs.ENV_MANAGED, "1")
    monkeypatch.setattr(ss, "gguf_size_bytes", lambda path: None)
    monkeypatch.setattr(ss, "remote_gguf_size_bytes", lambda *a, **k: None)
    started = []

    async def fake_ssh(target, remote, **kw):
        started.append(target)
        return 0, "/b/ggml-rpc-server\n", ""

    monkeypatch.setattr(ss, "ssh_run", fake_ssh)
    request = _request(model_path = "org/some-safetensors-model")
    assert run(ss.before_load(request, 4)) is request
    assert started == []


def test_a_refused_plan_leaves_nothing_in_flight(monkeypatch):
    _cluster_env(monkeypatch, model_bytes = 900 * GIB)
    with pytest.raises(ss.SparkLoadDoesNotFit):
        run(ss.before_load(_request(), 4))
    assert cs.state().in_flight is False and cs.state()._pre_load is None


def test_forced_single_starts_and_probes_nothing(monkeypatch):
    seen = _cluster_env(monkeypatch)
    monkeypatch.setenv(cs.ENV_TOPOLOGY, "single")
    monkeypatch.setenv(cs.ENV_MANAGED, "1")
    started = []

    async def fake_ssh(target, remote, **kw):
        started.append(target)
        return 0, "/b/ggml-rpc-server\n", ""

    monkeypatch.setattr(ss, "ssh_run", fake_ssh)
    request = _request()
    assert run(ss.before_load(request, 4)) is request
    assert started == [] and seen == []


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


def test_a_ssh_that_cannot_spawn_fails_only_that_node(monkeypatch):
    _cluster_env(monkeypatch, model_bytes = 150 * GIB)
    monkeypatch.setenv(cs.ENV_NODES, "10.0.0.2,me@10.0.0.3,10.0.0.4")
    monkeypatch.setenv(cs.ENV_MANAGED, "1")
    monkeypatch.setattr(ss, "rpc_server_binary", lambda: "/b/ggml-rpc-server")

    async def fake_ssh(target, remote, **kw):
        return 0, "/b/ggml-rpc-server\n", ""

    async def fake_start(self):
        if self.peer == "me@10.0.0.3":
            raise OSError("Resource temporarily unavailable")
        self.proc = SimpleNamespace(returncode = None)

    async def owns(process, host, port, timeout, **kw):
        return True

    monkeypatch.setattr(ss, "ssh_run", fake_ssh)
    monkeypatch.setattr(ss.PeerProcess, "start", fake_start)
    monkeypatch.setattr(ss, "wait_for_own_port", owns)
    out = run(ss.before_load(_request(), 4))
    assert (
        out.llama_extra_args[out.llama_extra_args.index("--rpc") + 1]
        == "10.0.0.2:50052,10.0.0.4:50052"
    )
    assert "could not start ssh to 10.0.0.3" in ss.status()["last_error"]


def test_a_loaded_split_is_still_reported_after_the_config_goes(monkeypatch):
    _cluster_env(monkeypatch)
    out = run(ss.before_load(_request(), 4))
    backend = SimpleNamespace(
        _process = SimpleNamespace(args = ["/b/llama-server"] + out.llama_extra_args),
        is_loaded = True,
    )
    run(ss.after_load(backend, 4))
    monkeypatch.delenv(cs.ENV_NODES)
    monkeypatch.setattr(ss, "is_spark", lambda: False)
    status = ss.status()
    assert status["topology"] == "layer_split" and status["retiring"] is True
    assert ss.current_topology() == "layer_split"


def test_completion_goes_to_the_planner_that_took_the_load(monkeypatch):
    _cluster_env(monkeypatch)
    run(ss.before_load(_request(), 4))
    run(ss.after_load(SimpleNamespace(_process = SimpleNamespace(args = [])), 4))
    # A later load planned while the cluster is off is the two-Spark path's, even if the cluster
    # config comes back before it completes.
    assert cs.state().in_flight is False
    legacy_after = []

    async def fake_after(self, backend, n):
        legacy_after.append(1)

    monkeypatch.setattr(ss.SparkServing, "after_load", fake_after)
    monkeypatch.setattr(ss, "enabled", lambda: True)
    run(ss.after_load(SimpleNamespace(), 4))
    assert legacy_after == [1]


def test_a_projector_that_cannot_be_pinned_keeps_the_load_local(monkeypatch):
    _cluster_env(monkeypatch)
    monkeypatch.setattr(ss, "auto_projector_file", lambda *a, **k: "/models/mmproj.gguf")
    sizes = {"/models/mmproj.gguf": 1 * GIB}
    monkeypatch.setattr(ss, "gguf_size_bytes", lambda path: sizes.get(str(path), 250 * GIB))
    monkeypatch.setattr(ss, "llama_server_supports", lambda flag, binary = None: False)
    request = _request()
    assert run(ss.before_load(request, 4)) is request
    assert "cannot pin the projector" in ss.status()["reason"]


def test_a_failed_engine_load_that_unloaded_the_split_forgets_it(monkeypatch):
    _cluster_env(monkeypatch)
    out = run(ss.before_load(_request(), 4))
    backend = SimpleNamespace(
        _process = SimpleNamespace(args = ["/b/llama-server"] + out.llama_extra_args),
        is_loaded = True,
    )
    run(ss.after_load(backend, 4))
    run(ss.before_load(_request(engine = "vllm"), 4))
    backend.is_loaded = False
    run(ss.load_failed())
    assert ss.status()["topology"] == "single" and ss.status()["split_args"] is None


@pytest.mark.parametrize("extra", [["-ngl", "0"], ["--n-gpu-layers=20"], ["--gpu-layers", "99"]])
def test_a_caller_gpu_layer_count_is_left_alone(monkeypatch, extra):
    seen = _cluster_env(monkeypatch)
    request = _request(llama_extra_args = extra)
    assert run(ss.before_load(request, 4)) is request and seen == []


def test_a_spawn_the_platform_refuses_fails_only_that_node(monkeypatch):
    _cluster_env(monkeypatch, model_bytes = 150 * GIB)
    monkeypatch.setenv(cs.ENV_NODES, "10.0.0.2,me@10.0.0.3,10.0.0.4")
    monkeypatch.setenv(cs.ENV_MANAGED, "1")
    monkeypatch.setattr(ss, "rpc_server_binary", lambda: "/b/ggml-rpc-server")

    async def fake_ssh(target, remote, **kw):
        return 0, "/b/ggml-rpc-server\n", ""

    async def fake_start(self):
        if self.peer == "me@10.0.0.3":
            raise ValueError("preexec_fn is not supported on Windows platforms")
        self.proc = SimpleNamespace(returncode = None)

    async def owns(process, host, port, timeout, **kw):
        return True

    monkeypatch.setattr(ss, "ssh_run", fake_ssh)
    monkeypatch.setattr(ss.PeerProcess, "start", fake_start)
    monkeypatch.setattr(ss, "wait_for_own_port", owns)
    out = run(ss.before_load(_request(), 4))
    assert ss.status()["topology"] == "layer_split"
    assert "10.0.0.3" in ss.status()["last_error"]


def test_every_split_device_keeps_a_whole_layer_free():
    # A small worker beside two large ones: 5% of the total would not cover one layer there.
    listing = (
        "  RPC0: 10.0.0.2:50052 (122502 MiB, 118000 MiB free)\n"
        "  RPC1: 10.0.0.3:50052 (12288 MiB, 12000 MiB free)\n"
        "  CUDA0: NVIDIA GB10 (122502 MiB, 110000 MiB free)\n"
    )
    devices = cs.parse_device_list(listing, ["10.0.0.2:50052", "10.0.0.3:50052"])
    layer = 4 * GIB
    plan = cs.plan_cluster(200 * GIB, 0, 0, devices, layer_bytes = layer)
    assert plan.topology == "layer_split"
    for d, w in zip(plan.devices, plan.weights):
        assert w * 200 * GIB + layer <= cs.usable_bytes(d) + 1


def test_a_split_reads_its_layer_count_from_the_gguf(monkeypatch):
    _cluster_env(monkeypatch)
    seen = {}
    real = cs.plan_cluster

    def spy(*a, **k):
        seen["layer_bytes"] = k.get("layer_bytes")
        return real(*a, **k)

    monkeypatch.setattr(cs, "plan_cluster", spy)
    monkeypatch.setattr(cs, "_gguf_block_count", lambda path, ss: 50)
    run(ss.before_load(_request(), 4))
    assert seen["layer_bytes"] == pytest.approx(250 * GIB / 50)


def test_a_first_hub_load_reserves_its_projector(monkeypatch):
    _cluster_env(monkeypatch, model_bytes = 90 * GIB)
    monkeypatch.setattr(ss, "llama_server_supports", lambda flag, binary = None: True)
    monkeypatch.setattr(ss, "cached_repo_file", lambda path, variant: None)
    monkeypatch.setattr(ss, "remote_gguf_size_bytes", lambda *a, **k: 90 * GIB)
    # An Auto context is priced at the fitter's floor, from the Hub header.
    monkeypatch.setattr(cs, "_remote_gguf_header", lambda *a, **k: (_hub_header, {}))
    monkeypatch.setattr(cs, "_remote_projector_bytes", lambda path, token: 10 * GIB)
    run(ss.before_load(_request(model_path = "org/vision-GGUF", max_seq_length = 0), 4))
    assert ss.status()["topology"] == "layer_split"
    monkeypatch.setattr(cs, "_remote_projector_bytes", lambda path, token: 0)
    run(ss.before_load(_request(model_path = "org/vision-GGUF", max_seq_length = 0), 4))
    assert ss.status()["topology"] == "single"


def test_managed_must_be_a_json_boolean():
    with pytest.raises(cs.ClusterConfigError, match = "managed"):
        cs.parse_nodes_json({"nodes": [{"host": "10.0.0.2", "managed": "false"}]})
    assert (
        cs.parse_nodes_json({"nodes": [{"host": "10.0.0.2", "managed": False}]})[0].managed is False
    )


def test_a_sidecar_must_fit_the_first_local_gpu_even_when_the_model_fits_locally():
    devices = cs.parse_device_list(TWO_LOCAL, ["10.0.0.2:50052"])
    # 100 GiB fits CUDA0 + CUDA1 together, but a 75 GiB projector does not fit CUDA0 alone.
    plan = cs.plan_cluster(30 * GIB, 0, 75 * GIB, devices)
    assert plan.topology == "refuse" and "CUDA0" in plan.reason


def test_the_sidecar_stays_pinned_to_the_gpu_it_was_charged_to(monkeypatch):
    listing = (
        "  RPC0: 10.0.0.2:50052 (122502 MiB, 118000 MiB free)\n"
        "  CUDA0: NVIDIA L4 (23034 MiB, 22000 MiB free)\n"
        "  CUDA1: NVIDIA H100 (81559 MiB, 80000 MiB free)\n"
    )
    devices = cs.parse_device_list(listing, ["10.0.0.2:50052"])
    plan = cs.plan_cluster(150 * GIB, 0, 19 * GIB, devices, layer_bytes = 2 * GIB)
    assert plan.topology == "layer_split"
    assert "CUDA0" not in [d.name for d in plan.devices]  # no room for layers beside the sidecar
    assert plan.sidecar_device.name == "CUDA0"


@pytest.mark.parametrize("extra", [["-ot", "blk.*=CUDA0"], ["--n-cpu-moe", "10"], ["--cpu-moe"]])
def test_tensor_overrides_are_caller_placement(monkeypatch, extra):
    seen = _cluster_env(monkeypatch)
    request = _request(llama_extra_args = extra)
    assert run(ss.before_load(request, 4)) is request and seen == []


def test_a_failed_engine_load_forgets_the_split_even_with_managed_workers(monkeypatch):
    _cluster_env(monkeypatch)
    out = run(ss.before_load(_request(), 4))
    backend = SimpleNamespace(
        _process = SimpleNamespace(args = ["/b/llama-server"] + out.llama_extra_args),
        is_loaded = True,
    )
    run(ss.after_load(backend, 4))
    cs.state().workers["10.0.0.2:50052"] = SimpleNamespace(alive = True, snapshot = lambda: {})
    run(ss.before_load(_request(engine = "vllm"), 4))
    backend.is_loaded = False
    run(ss.load_failed())
    assert ss.status()["topology"] == "single" and ss.status()["split_args"] is None


def test_the_spec_draft_model_alias_is_sized(tmp_path):
    drafter = tmp_path / "draft.gguf"
    drafter.write_bytes(b"x" * 1000)
    sized, unknown = ss.sidecar_bytes(["--spec-draft-model", str(drafter)])
    assert sized == 1000 and not unknown


def test_underscore_spellings_are_read_like_llama_cpp_reads_them(monkeypatch):
    seen = _cluster_env(monkeypatch)
    request = _request(llama_extra_args = ["--tensor_split", "1,0"])
    assert run(ss.before_load(request, 4)) is request and seen == []
    _cluster_env(monkeypatch, model_bytes = 85 * GIB)
    monkeypatch.setattr(ss, "estimate_kv_bytes", lambda path, ctx, *a, **k: ctx * 2 * 2**20)
    run(ss.before_load(_request(llama_extra_args = ["--ctx_size", "16384"]), 4))
    assert ss.status()["topology"] == "layer_split"


def test_a_failed_load_while_retiring_forgets_an_unloaded_split(monkeypatch):
    _cluster_env(monkeypatch)
    out = run(ss.before_load(_request(), 4))
    backend = SimpleNamespace(
        _process = SimpleNamespace(args = ["/b/llama-server"] + out.llama_extra_args),
        is_loaded = True,
    )
    run(ss.after_load(backend, 4))
    cs.state().workers["10.0.0.2:50052"] = SimpleNamespace(alive = True, snapshot = lambda: {})
    monkeypatch.delenv(cs.ENV_NODES)
    monkeypatch.setattr(ss, "is_spark", lambda: False)
    run(ss.before_load(_request(), 4))  # through the non-cluster path, retirement pending
    backend.is_loaded = False
    run(ss.load_failed())
    assert ss.current_topology() != "layer_split"


def test_host_names_are_deduplicated_case_insensitively():
    with pytest.raises(cs.ClusterConfigError, match = "twice"):
        cs.parse_nodes_env("gpu.local,GPU.local.")


def test_a_local_fit_records_the_sidecar_on_the_first_gpu_within_its_budget():
    devices = cs.parse_device_list(TWO_LOCAL, ["10.0.0.2:50052"])
    plan = cs.plan_cluster(120 * GIB, 0, 20 * GIB, devices)
    assert plan.topology == "single"
    held = plan.resident()
    for d in plan.devices:
        assert held[d.key] <= cs.usable_bytes(d) + 1


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


@pytest.mark.parametrize("env", [False, True])
def test_an_image_projector_the_vision_switch_drops_is_not_charged(monkeypatch, tmp_path, env):
    _cluster_env(monkeypatch, model_bytes = 85 * GIB)
    monkeypatch.setattr(ss, "llama_server_supports", lambda flag, binary = None: True)
    mmproj = tmp_path / "mmproj.gguf"
    mmproj.write_bytes(b"x")
    sizes = {str(mmproj): 15 * GIB}
    monkeypatch.setattr(ss, "gguf_size_bytes", lambda path: sizes.get(str(path), 85 * GIB))
    image = {str(mmproj): True}
    monkeypatch.setattr(ss, "_projector_accepts_image", lambda path: image[str(path)])
    if env:
        monkeypatch.setenv("LLAMA_ARG_MMPROJ", str(mmproj))
    extra = [] if env else ["--mmproj", str(mmproj)]
    request = _request(llama_extra_args = extra, disable_vision = True)
    out = run(ss.before_load(request, 4))
    assert out is request and ss.status()["topology"] == "single"  # 85 GiB fits here
    image[str(mmproj)] = False  # an audio-only projector stays loaded, so it counts
    out = run(ss.before_load(_request(llama_extra_args = extra, disable_vision = True), 4))
    assert ss.status()["topology"] == "layer_split" and "--mmproj-device" in out.llama_extra_args


def test_the_tensor_parallel_toggle_is_caller_placement(monkeypatch):
    seen = _cluster_env(monkeypatch)
    request = _request(tensor_parallel = True)
    assert run(ss.before_load(request, 4)) is request and seen == []
    assert "tensor_parallel" in ss.status()["reason"]


def test_a_failed_replacement_keeps_the_served_block_recognised(monkeypatch):
    _cluster_env(monkeypatch)
    out = run(ss.before_load(_request(), 4))
    block = list(cs.state().block)
    backend = SimpleNamespace(
        _process = SimpleNamespace(args = ["/b/llama-server"] + out.llama_extra_args),
        is_loaded = True,
    )
    run(ss.after_load(backend, 4))
    for i in range(8):
        cs._remember_block(["--rpc", f"10.9.9.{i}:1"])
    _cluster_env(monkeypatch, model_bytes = 280 * GIB)
    run(ss.before_load(_request(), 4))
    assert cs.state().block != block
    run(ss.load_failed())
    assert cs.state().block == block
    assert cs.strip_managed_block(["--seed", "1"] + block) == ["--seed", "1"]


def test_a_failing_worker_is_matched_by_its_whole_endpoint(monkeypatch):
    _cluster_env(monkeypatch, model_bytes = 150 * GIB)
    monkeypatch.setenv(cs.ENV_NODES, "10.0.0.2:5005,10.0.0.2:50052")
    listing = """Available devices:
  RPC0: 10.0.0.2:5005 (122502 MiB, 118000 MiB free)
  CUDA0: NVIDIA GB10 (122502 MiB, 110000 MiB free)
"""
    calls = []

    def fake_list(binary, endpoints):
        calls.append(list(endpoints))
        if "10.0.0.2:50052" in endpoints:
            return [], "Failed to connect to 10.0.0.2:50052"
        return cs.parse_device_list(listing, endpoints), ""

    monkeypatch.setattr(cs, "list_devices", fake_list)
    out = run(ss.before_load(_request(), 4))
    assert calls[1] == ["10.0.0.2:5005"]
    assert out.llama_extra_args[out.llama_extra_args.index("--rpc") + 1] == "10.0.0.2:5005"


@pytest.mark.parametrize("env", [False, True])
def test_an_explicit_fit_target_is_kept_free_on_every_device(monkeypatch, env):
    _cluster_env(monkeypatch, model_bytes = 85 * GIB)
    request = _request()
    assert run(ss.before_load(request, 4)) is request  # fits here with the default reserve
    if env:
        monkeypatch.setenv("LLAMA_ARG_FIT_TARGET", "30000")
    extra = [] if env else ["--fit-target", "30000"]
    run(ss.before_load(_request(llama_extra_args = extra), 4))
    assert ss.status()["topology"] == "layer_split"


def test_a_host_managed_twice_is_a_config_error(monkeypatch):
    monkeypatch.setenv(cs.ENV_NODES, "10.0.0.2:50052,10.0.0.2:50053")
    assert [n.endpoint for n in cs.load_nodes()] == ["10.0.0.2:50052", "10.0.0.2:50053"]
    monkeypatch.setenv(cs.ENV_MANAGED, "1")
    with pytest.raises(cs.ClusterConfigError, match = "managed"):
        cs.load_nodes()


def test_a_managed_host_takes_no_second_endpoint(monkeypatch, tmp_path):
    nodes = [
        {"host": "10.0.0.2", "managed": True},
        {"host": "10.0.0.2", "rpc_port": 50053},
    ]
    (tmp_path / "cluster.json").write_text(json.dumps({"nodes": nodes}), encoding = "utf-8")
    with pytest.raises(cs.ClusterConfigError, match = "managed"):
        cs.load_nodes()


def test_a_hung_managed_worker_is_stopped_once_a_load_without_it_commits(monkeypatch):
    _cluster_env(monkeypatch, model_bytes = 150 * GIB)
    monkeypatch.setenv(cs.ENV_MANAGED, "1")
    stopped = []

    class Worker:
        alive = True

        def __init__(self, endpoint):
            self.endpoint = endpoint

        async def stop(self, timeout = 10.0):
            stopped.append(self.endpoint)

    for e in ("10.0.0.2:50052", "10.0.0.3:50052", "10.0.0.4:50052"):
        cs.state().workers[e] = Worker(e)
    DOWN.add("10.0.0.3")  # its ssh lives on, its rpc-server no longer answers
    out = run(ss.before_load(_request(), 4))
    rpc = out.llama_extra_args[out.llama_extra_args.index("--rpc") + 1]
    assert rpc == "10.0.0.2:50052,10.0.0.4:50052"
    backend = SimpleNamespace(
        _process = SimpleNamespace(args = ["/b/llama-server"] + out.llama_extra_args),
        is_loaded = True,
    )
    run(ss.after_load(backend, 4))
    assert stopped == ["10.0.0.3:50052"]
    assert set(cs.state().workers) == {"10.0.0.2:50052", "10.0.0.4:50052"}


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
    monkeypatch.delenv(cs.ENV_NODES)  # the cluster config goes; this is a paired Spark
    monkeypatch.setattr(ss, "enabled", lambda: True)

    async def legacy_before_load(self, request, n_parallel, **kw):
        await self._on_plan_accepted()
        return request

    monkeypatch.setattr(ss.SparkServing, "before_load", legacy_before_load)
    run(ss.before_load(_request(), 4))
    assert not cs.state().workers
    run(ss.load_failed())  # the route rejected the replacement after the plan
    assert unloaded == [1], "its remote layers are gone, so it cannot answer"


def test_a_first_hub_load_with_vision_off_still_reserves_its_projector(monkeypatch):
    _cluster_env(monkeypatch, model_bytes = 90 * GIB)
    monkeypatch.setattr(ss, "llama_server_supports", lambda flag, binary = None: True)
    monkeypatch.setattr(ss, "cached_repo_file", lambda path, variant: None)
    monkeypatch.setattr(ss, "remote_gguf_size_bytes", lambda *a, **k: 90 * GIB)
    # An Auto context is priced at the fitter's floor, from the Hub header.
    monkeypatch.setattr(cs, "_remote_gguf_header", lambda *a, **k: (_hub_header, {}))
    monkeypatch.setattr(cs, "_remote_projector_bytes", lambda path, token: 10 * GIB)
    request = _request(model_path = "org/audio-GGUF", max_seq_length = 0, disable_vision = True)
    out = run(ss.before_load(request, 4))
    assert ss.status()["topology"] == "layer_split"  # an audio-only projector survives the switch
    assert "--mmproj-device" in out.llama_extra_args


def test_a_guessed_kv_reserve_is_not_counted_as_freed(monkeypatch):
    _cluster_env(monkeypatch, model_bytes = 90 * GIB)
    monkeypatch.setattr(ss, "cached_repo_file", lambda path, variant: None)
    monkeypatch.setattr(ss, "remote_gguf_size_bytes", lambda *a, **k: 90 * GIB)
    # An Auto context is priced at the fitter's floor, from the Hub header.
    monkeypatch.setattr(cs, "_remote_gguf_header", lambda *a, **k: (_hub_header, {}))
    monkeypatch.setattr(cs, "_remote_gguf_header", lambda *a, **k: (_llama_header(64), {}))
    out = run(ss.before_load(_request(model_path = "org/big-GGUF", max_seq_length = 8192), 4))
    assert ss.status()["topology"] == "layer_split"  # 90 GiB + an estimated 20 GiB of KV
    backend = SimpleNamespace(
        _process = SimpleNamespace(args = ["/b/llama-server"] + out.llama_extra_args),
        is_loaded = True,
    )
    run(ss.after_load(backend, 4))
    plan = cs.state().plan
    # Weights only, a layer short per device for whole-layer rounding.
    expected = sum(max(0.0, w * 90 * GIB - plan.layer_bytes) for w in plan.weights)
    assert plan.layer_bytes > 0
    assert sum(cs.state()._resident.values()) == pytest.approx(expected)


@pytest.mark.parametrize("env", [False, True])
def test_host_resident_kv_is_caller_placement(monkeypatch, env):
    # On unified memory host KV shares the pool with the weights, so it is not planned around.
    seen = _cluster_env(monkeypatch)
    if env:
        monkeypatch.setenv("LLAMA_ARG_NO_KV_OFFLOAD", "1")
    extra = [] if env else ["--no-kv-offload"]
    request = _request(max_seq_length = 8192, llama_extra_args = extra)
    assert run(ss.before_load(request, 4)) is request and seen == []
    assert "host memory" in ss.status()["reason"]


@pytest.mark.parametrize("value", ["auto", "-1", "AUTO"])
def test_automatic_gpu_layers_are_not_caller_placement(monkeypatch, value):
    seen = _cluster_env(monkeypatch)
    monkeypatch.setenv("LLAMA_ARG_N_GPU_LAYERS", value)
    run(ss.before_load(_request(llama_extra_args = ["-ngl", value]), 4))
    assert ss.status()["topology"] == "layer_split" and seen
    monkeypatch.setenv("LLAMA_ARG_N_GPU_LAYERS", "40")
    request = _request()
    assert run(ss.before_load(request, 4)) is request
    assert "LLAMA_ARG_N_GPU_LAYERS" in ss.status()["reason"]


def test_a_worker_restarted_by_this_load_gets_no_residency_credit(monkeypatch):
    _cluster_env(monkeypatch)
    out = run(ss.before_load(_request(), 4))
    backend = SimpleNamespace(
        _process = SimpleNamespace(args = ["/b/llama-server"] + out.llama_extra_args),
        is_loaded = True,
    )
    run(ss.after_load(backend, 4))
    credited = []
    real_plan = cs.plan_cluster

    def spy(model_bytes, kv_bytes, extra, devices, **kw):
        credited.extend((d.endpoint, d.free_bytes) for d in devices)
        return real_plan(model_bytes, kv_bytes, extra, devices, **kw)

    monkeypatch.setattr(cs, "plan_cluster", spy)

    async def restart_one(self):
        self._started.add("10.0.0.3:50052")
        return {}

    monkeypatch.setattr(cs.ClusterServing, "_ensure_workers", restart_one)
    run(ss.before_load(_request(), 4))
    free = dict(credited)
    listed = {d.endpoint: d.free_bytes for d in cs.parse_device_list(LISTING, list(free))}
    assert free["10.0.0.3:50052"] == listed["10.0.0.3:50052"]  # its buffers died with it
    assert free["10.0.0.2:50052"] > listed["10.0.0.2:50052"]


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


@pytest.mark.parametrize("extra", [["-ncffn", "4"], ["--n-cpu-ffn=4"], []])
def test_cpu_ffn_offload_is_caller_placement(monkeypatch, extra):
    seen = _cluster_env(monkeypatch)
    if not extra:
        monkeypatch.setenv("LLAMA_ARG_N_CPU_FFN", "4")
    request = _request(llama_extra_args = extra)
    assert run(ss.before_load(request, 4)) is request and seen == []
    assert "placement is theirs" in ss.status()["reason"]


def test_a_vocoder_keeps_the_load_local(monkeypatch):
    seen = _cluster_env(monkeypatch)
    request = _request(llama_extra_args = ["-mv", "/models/vocoder.gguf"])
    assert run(ss.before_load(request, 4)) is request and seen == []
    assert "vocoder" in ss.status()["reason"]


@pytest.mark.parametrize("env", [False, True])
def test_a_projector_kept_in_host_ram_is_not_pinned_to_a_gpu(monkeypatch, env):
    _cluster_env(monkeypatch)
    monkeypatch.setattr(ss, "llama_server_supports", lambda flag, binary = None: True)
    if env:
        monkeypatch.setenv("LLAMA_ARG_MMPROJ_OFFLOAD", "0")
    extra = [] if env else ["--no-mmproj-offload"]
    out = run(ss.before_load(_request(llama_extra_args = extra), 4))
    assert ss.status()["topology"] == "layer_split"
    assert "--mmproj-device" not in out.llama_extra_args


@pytest.mark.parametrize("env", [False, True])
def test_the_fitter_off_keeps_the_load_local(monkeypatch, env):
    seen = _cluster_env(monkeypatch)
    if env:
        monkeypatch.setenv("LLAMA_ARG_FIT", "0")
    extra = [] if env else ["--fit", "off"]
    request = _request(llama_extra_args = extra, max_seq_length = 0)
    assert run(ss.before_load(request, 4)) is request and seen == []
    assert "fitter is off" in ss.status()["reason"]
    request = _request(llama_extra_args = extra, max_seq_length = 8192)
    assert run(ss.before_load(request, 4)) is request and seen == []


def test_a_local_fit_over_several_gpus_claims_no_residency(monkeypatch):
    listing = """Available devices:
  RPC0: 10.0.0.2:50052 (122502 MiB, 118000 MiB free)
  CUDA0: NVIDIA H100 (81559 MiB, 80000 MiB free)
  CUDA1: NVIDIA H100 (81559 MiB, 80000 MiB free)
"""
    _cluster_env(monkeypatch, listing = listing, model_bytes = 60 * GIB)
    monkeypatch.setenv(cs.ENV_NODES, "10.0.0.2")
    request = _request()
    assert run(ss.before_load(request, 4)) is request
    assert ss.status()["topology"] == "single"
    run(ss.after_load(SimpleNamespace(_process = None, is_loaded = True), 4))
    assert cs.state()._resident == {}


def test_a_probe_that_times_out_without_a_name_asks_each_worker_alone(monkeypatch):
    _cluster_env(monkeypatch, model_bytes = 150 * GIB)
    calls = []

    def fake_list(binary, endpoints):
        calls.append(list(endpoints))
        if "10.0.0.3:50052" in endpoints:
            return [], "timed out after 30 seconds"
        return [
            d
            for d in cs.parse_device_list(LISTING, endpoints)
            if not d.remote or d.endpoint in endpoints
        ], ""

    monkeypatch.setattr(cs, "list_devices", fake_list)
    out = run(ss.before_load(_request(), 4))
    assert ss.status()["topology"] == "layer_split"
    rpc = out.llama_extra_args[out.llama_extra_args.index("--rpc") + 1]
    assert rpc == "10.0.0.2:50052,10.0.0.4:50052"


def test_the_requested_cache_type_beats_an_inherited_one(monkeypatch):
    _cluster_env(monkeypatch)
    seen = []
    monkeypatch.setattr(ss, "estimate_kv_bytes", lambda path, ctx, k, v: seen.append((k, v)) or 0)
    monkeypatch.setenv("LLAMA_ARG_CACHE_TYPE_K", "q4_0")
    monkeypatch.setenv("LLAMA_ARG_CACHE_TYPE_V", "q4_0")
    run(ss.before_load(_request(max_seq_length = 8192, cache_type_kv = "f32"), 4))
    run(
        ss.before_load(
            _request(max_seq_length = 8192, cache_type_kv = "f32", llama_extra_args = ["-ctk", "q8_0"]), 4
        )
    )
    assert seen == [("f32", "f32"), ("q8_0", "f32")]


@pytest.mark.parametrize(
    "key, value",
    [("LLAMA_ARG_CPU_MOE", "0"), ("LLAMA_ARG_N_CPU_MOE", "0"), ("LLAMA_ARG_N_CPU_FFN", "0")],
)
def test_cpu_placement_env_that_places_nothing_is_not_placement(monkeypatch, key, value):
    seen = _cluster_env(monkeypatch)
    monkeypatch.setenv(key, value)
    run(ss.before_load(_request(), 4))
    assert ss.status()["topology"] == "layer_split" and seen


def test_a_host_ram_projector_is_still_charged(monkeypatch, tmp_path):
    # On unified memory the host RAM it moves to is the same pool the layers use.
    _cluster_env(monkeypatch, model_bytes = 85 * GIB)
    monkeypatch.setattr(ss, "llama_server_supports", lambda flag, binary = None: True)
    mmproj = tmp_path / "mmproj.gguf"
    mmproj.write_bytes(b"x")
    sizes = {str(mmproj): 15 * GIB}
    monkeypatch.setattr(ss, "gguf_size_bytes", lambda path: sizes.get(str(path), 85 * GIB))
    out = run(
        ss.before_load(
            _request(llama_extra_args = ["--mmproj", str(mmproj), "--no-mmproj-offload"]), 4
        )
    )
    assert ss.status()["topology"] == "layer_split"  # 85 + 15 GiB do not fit here
    assert "--mmproj-device" not in out.llama_extra_args


@pytest.mark.skipif(sys.platform == "win32", reason = "the worker command runs under sh on the peer")
@pytest.mark.parametrize("venv", ["studio/unsloth_studio", "unsloth_studio"])
def test_a_managed_worker_finds_the_studio_venvs_cuda_runtime(tmp_path, venv):
    root = tmp_path / "home" / ".unsloth"
    bindir = root / "llama.cpp" / "build" / "bin"
    bindir.mkdir(parents = True)
    nv = root / venv / "lib" / "python3.12" / "site-packages"
    for sub in ("nvidia/cuda_runtime/lib", "nvidia/cublas/lib", "torch/lib"):
        (nv / sub).mkdir(parents = True)
    fake = bindir / "ggml-rpc-server"
    fake.write_text(
        '#!/bin/sh\necho "pid=$$ args=$*"\necho "LD=$LD_LIBRARY_PATH"\n', encoding = "utf-8"
    )
    fake.chmod(0o755)
    argv = cs._with_cuda_runtime([str(fake), "-H", "10.0.0.2", "-p", "50052"])
    out = subprocess.run(
        ["sh", "-c", " ".join(shlex.quote(a) for a in argv) + " & echo started=$!; wait"],
        capture_output = True,
        text = True,
        env = {"PATH": os.environ["PATH"], "LD_LIBRARY_PATH": "/keep"},
    ).stdout
    started = re.search(r"started=(\d+)", out).group(1)
    assert (
        f"pid={started} args=-H 10.0.0.2 -p 50052" in out
    )  # exec keeps the pid the reaper watches
    ld = re.search(r"LD=(.*)", out).group(1).split(":")
    assert ld[0] == str(bindir) and ld[-1] == "/keep"
    for sub in ("nvidia/cuda_runtime/lib", "nvidia/cublas/lib", "torch/lib"):
        assert any(os.path.realpath(d) == str(nv / sub) for d in ld), sub


def test_a_sized_kv_cache_is_not_counted_as_freed_either(monkeypatch):
    _cluster_env(monkeypatch, model_bytes = 90 * GIB)
    monkeypatch.setattr(ss, "estimate_kv_bytes", lambda *a, **k: 20 * GIB)
    out = run(ss.before_load(_request(max_seq_length = 8192), 4))
    assert ss.status()["topology"] == "layer_split"
    backend = SimpleNamespace(
        _process = SimpleNamespace(args = ["/b/llama-server"] + out.llama_extra_args),
        is_loaded = True,
    )
    run(ss.after_load(backend, 4))
    assert sum(cs.state()._resident.values()) == pytest.approx(90 * GIB)


def test_a_projector_the_args_disable_is_not_charged(monkeypatch, tmp_path):
    _cluster_env(monkeypatch, model_bytes = 85 * GIB)
    monkeypatch.setattr(ss, "llama_server_supports", lambda flag, binary = None: True)
    mmproj = tmp_path / "mmproj.gguf"
    mmproj.write_bytes(b"x")
    sizes = {str(mmproj): 15 * GIB}
    monkeypatch.setattr(ss, "gguf_size_bytes", lambda path: sizes.get(str(path), 85 * GIB))
    request = _request(llama_extra_args = ["--mmproj", str(mmproj), "--no-mmproj"])
    assert run(ss.before_load(request, 4)) is request
    assert ss.status()["topology"] == "single"


@pytest.mark.parametrize("env", [False, True])
def test_a_zero_context_keeps_the_load_local(monkeypatch, env):
    seen = _cluster_env(monkeypatch)
    if env:
        monkeypatch.setenv("LLAMA_ARG_CTX_SIZE", "0")
    extra = [] if env else ["-c", "0"]
    request = _request(llama_extra_args = extra, max_seq_length = 0)
    assert run(ss.before_load(request, 4)) is request and seen == []
    assert "context size of 0" in ss.status()["reason"]


@pytest.mark.parametrize("env", [False, True])
def test_a_negative_fit_ctx_keeps_the_load_local(monkeypatch, env):
    seen = _cluster_env(monkeypatch)
    if env:
        monkeypatch.setenv("LLAMA_ARG_FIT_CTX", "-1")
    extra = [] if env else ["--fit-ctx", "-1"]
    request = _request(llama_extra_args = extra, max_seq_length = 0)
    assert run(ss.before_load(request, 4)) is request and seen == []


def test_a_fit_ctx_floor_is_the_context_priced(monkeypatch):
    _cluster_env(monkeypatch)
    seen = []
    monkeypatch.setattr(ss, "estimate_kv_bytes", lambda path, ctx, k, v: seen.append(ctx) or 0)
    run(ss.before_load(_request(max_seq_length = 0, llama_extra_args = ["--fit-ctx", "16384"]), 4))
    assert seen == [16384]


@pytest.mark.parametrize("env", [False, True])
def test_a_projector_url_keeps_the_load_local(monkeypatch, env):
    seen = _cluster_env(monkeypatch)
    if env:
        monkeypatch.setenv("LLAMA_ARG_MMPROJ_URL", "https://example.com/mmproj.gguf")
    extra = [] if env else ["--mmproj-url", "https://example.com/mmproj.gguf"]
    request = _request(llama_extra_args = extra)
    assert run(ss.before_load(request, 4)) is request and seen == []
    assert "URL" in ss.status()["reason"]
    # With the vision switch off the URL projector is dropped, so the split goes ahead.
    run(ss.before_load(_request(llama_extra_args = extra, disable_vision = True), 4))
    assert ss.status()["topology"] == "layer_split"


def test_an_argv_projector_replaces_the_environment_one(monkeypatch, tmp_path):
    _cluster_env(monkeypatch, model_bytes = 85 * GIB)
    monkeypatch.setattr(ss, "llama_server_supports", lambda flag, binary = None: True)
    env_mm, argv_mm = tmp_path / "env-mmproj.gguf", tmp_path / "argv-mmproj.gguf"
    env_mm.write_bytes(b"x")
    argv_mm.write_bytes(b"x")
    sizes = {str(env_mm): 15 * GIB, str(argv_mm): 1 * GIB}
    monkeypatch.setattr(ss, "gguf_size_bytes", lambda path: sizes.get(str(path), 85 * GIB))
    monkeypatch.setenv("LLAMA_ARG_MMPROJ", str(env_mm))
    run(ss.before_load(_request(), 4))
    assert ss.status()["topology"] == "layer_split"  # the 15 GiB environment projector loads
    request = _request(llama_extra_args = ["--mmproj", str(argv_mm)])
    assert run(ss.before_load(request, 4)) is request
    assert ss.status()["topology"] == "single"  # only the 1 GiB argv one does


def test_the_device_probe_runs_with_the_llama_server_environment(monkeypatch):
    from core.inference.llama_cpp import LlamaCppBackend

    monkeypatch.setattr(
        LlamaCppBackend,
        "_llama_server_env_for_binary",
        staticmethod(lambda binary, **kw: {"MARK": binary}),
    )
    seen = {}

    def fake_run(argv, **kw):
        seen.update(kw)
        return SimpleNamespace(stdout = LISTING, stderr = "", returncode = 0)

    monkeypatch.setattr(cs.subprocess, "run", fake_run)
    devices, error = cs.list_devices("/b/llama-server", ["10.0.0.2:50052"])
    assert seen["env"] == {"MARK": "/b/llama-server"} and devices and not error


@pytest.mark.parametrize(
    "extra, env",
    [(["--moe-cache-mib", "4096"], None), ([], "4096"), (["--moe-cache-mib", "auto"], None)],
)
def test_a_moe_gpu_cache_is_caller_placement(monkeypatch, extra, env):
    seen = _cluster_env(monkeypatch)
    if env:
        monkeypatch.setenv("LLAMA_ARG_MOE_CACHE_MIB", env)
    request = _request(llama_extra_args = extra)
    assert run(ss.before_load(request, 4)) is request and seen == []
    monkeypatch.setenv("LLAMA_ARG_MOE_CACHE_MIB", "0")
    run(ss.before_load(_request(llama_extra_args = ["--moe-cache-mib", "0"]), 4))
    assert ss.status()["topology"] == "layer_split"


def test_a_caller_pinned_projector_is_caller_placement(monkeypatch):
    seen = _cluster_env(monkeypatch)
    monkeypatch.setattr(ss, "llama_server_supports", lambda flag, binary = None: True)
    monkeypatch.setattr(ss, "auto_projector_file", lambda *a, **k: "/models/auto-mmproj.gguf")
    request = _request(llama_extra_args = ["--mmproj-device", "CUDA1"])
    assert run(ss.before_load(request, 4)) is request and seen == []
    assert "pins the projector" in ss.status()["reason"]


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
    run(ss.load_failed())  # the route rejected the replacement after the split was torn down
    assert unloaded == [1], "its remote layers are gone, so it cannot answer"


def test_a_restart_mark_survives_a_refused_load(monkeypatch):
    _cluster_env(monkeypatch)
    out = run(ss.before_load(_request(), 4))
    backend = SimpleNamespace(
        _process = SimpleNamespace(args = ["/b/llama-server"] + out.llama_extra_args),
        is_loaded = True,
    )
    run(ss.after_load(backend, 4))
    restarted = [True]

    async def ensure(self):
        if restarted.pop() if restarted else False:
            self._started.add("10.0.0.3:50052")
        return {}

    monkeypatch.setattr(cs.ClusterServing, "_ensure_workers", ensure)
    _cluster_env(monkeypatch, model_bytes = 2000 * GIB)
    with pytest.raises(ss.SparkLoadDoesNotFit):
        run(ss.before_load(_request(), 4))  # restarts 10.0.0.3, then refuses
    credited = []
    real_plan = cs.plan_cluster

    def spy(model_bytes, kv_bytes, extra, devices, **kw):
        credited.extend((d.endpoint, d.free_bytes) for d in devices)
        return real_plan(model_bytes, kv_bytes, extra, devices, **kw)

    monkeypatch.setattr(cs, "plan_cluster", spy)
    _cluster_env(monkeypatch)
    run(ss.before_load(_request(), 4))
    free = dict(credited)
    listed = {d.endpoint: d.free_bytes for d in cs.parse_device_list(LISTING, list(free))}
    assert free["10.0.0.3:50052"] == listed["10.0.0.3:50052"]


def test_an_unverified_hub_projector_is_not_counted_as_freed(monkeypatch):
    _cluster_env(monkeypatch, model_bytes = 90 * GIB)
    monkeypatch.setattr(ss, "llama_server_supports", lambda flag, binary = None: True)
    monkeypatch.setattr(ss, "cached_repo_file", lambda path, variant: None)
    monkeypatch.setattr(ss, "remote_gguf_size_bytes", lambda *a, **k: 90 * GIB)
    # An Auto context is priced at the fitter's floor, from the Hub header.
    monkeypatch.setattr(cs, "_remote_gguf_header", lambda *a, **k: (_hub_header, {}))
    monkeypatch.setattr(cs, "_remote_projector_bytes", lambda path, token: 10 * GIB)
    request = _request(model_path = "org/vision-GGUF", max_seq_length = 0, disable_vision = True)
    out = run(ss.before_load(request, 4))
    backend = SimpleNamespace(
        _process = SimpleNamespace(args = ["/b/llama-server"] + out.llama_extra_args),
        is_loaded = True,
    )
    run(ss.after_load(backend, 4))
    plan = cs.state().plan
    # The weights only (a layer short per device): the uncertain projector is not freed.
    expected = sum(max(0.0, w * 90 * GIB - plan.layer_bytes) for w in plan.weights)
    assert sum(cs.state()._resident.values()) == pytest.approx(expected)


@pytest.mark.parametrize(
    "flag", ["-otd", "--override-tensor-draft", "--spec-draft-override-tensor"]
)
def test_draft_tensor_overrides_are_caller_placement(monkeypatch, flag):
    seen = _cluster_env(monkeypatch)
    request = _request(llama_extra_args = [flag, "blk.*=CUDA1"])
    assert run(ss.before_load(request, 4)) is request and seen == []


@pytest.mark.parametrize(
    "kw, extra",
    [({"spec_draft_n_max": 8}, []), ({}, ["--spec-draft-n-max", "8"]), ({}, ["--draft-max", "8"])],
)
def test_draft_tuning_alone_does_not_keep_auto_speculation(monkeypatch, kw, extra):
    _cluster_env(monkeypatch)
    out = run(ss.before_load(_request(speculative_type = None, llama_extra_args = extra, **kw), 4))
    assert out.speculative_type == "off"


@pytest.mark.parametrize(
    "extra, env",
    [
        (["-md", "/models/d.gguf"], {}),
        (["--spec-draft-model=/models/d.gguf"], {}),
        (["-hfd", "org/draft-GGUF"], {}),
        (["--spec-type", "draft-simple"], {}),
        (["--spec-type", "ngram-mod,draft-mtp"], {}),
        (["--spec-type", "mtp"], {}),
        (["--spec-type", "ngram-mod"], {"LLAMA_ARG_SPEC_DRAFT_HF_REPO": "org/draft-GGUF"}),
        (["--spec-type", "ngram-mod"], {"LLAMA_ARG_MODEL_DRAFT": "/models/d.gguf"}),
        (["--spec-default"], {"LLAMA_ARG_SPEC_TYPE": "draft-mtp"}),
    ],
)
def test_the_callers_drafter_or_mtp_keeps_the_load_local(monkeypatch, extra, env):
    seen = _cluster_env(monkeypatch)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    request = _request(llama_extra_args = extra)
    assert run(ss.before_load(request, 4)) is request and seen == []
    assert "speculative decoding" in ss.status()["reason"]


@pytest.mark.parametrize(
    "env",
    [
        {"LLAMA_ARG_SPEC_DRAFT_MODEL": "/models/d.gguf"},
        {"LLAMA_ARG_MODEL_DRAFT": "/models/d.gguf"},
        {"LLAMA_ARG_SPEC_TYPE": "draft-mtp"},
    ],
)
def test_speculation_the_launch_scrubs_from_the_environment_does_not_count(monkeypatch, env):
    _cluster_env(monkeypatch)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    out = run(ss.before_load(_request(speculative_type = None), 4))
    assert ss.status()["topology"] == "layer_split" and out.speculative_type == "off"


@pytest.mark.parametrize(
    "mode, after",
    [
        (None, "off"),
        ("auto", "off"),
        ("default", "off"),
        ("mtp", "off"),
        ("draft-mtp", "off"),
        ("mtp+ngram", "off"),
        ("dspark", "off"),
        ("draft-dspark", "off"),
        ("draft-dflash", "off"),
        ("ngram", "ngram"),
        ("ngram-mod", "ngram-mod"),
        ("off", "off"),
    ],
)
def test_a_split_runs_studio_speculation_off_except_ngram(monkeypatch, mode, after):
    _cluster_env(monkeypatch)
    out = run(ss.before_load(_request(speculative_type = mode), 4))
    assert ss.status()["topology"] == "layer_split"
    assert out.speculative_type == after
    assert ("speculative decoding off" in ss.status()["reason"]) is (after != (mode or "auto"))


@pytest.mark.parametrize("extra", [["--spec-type", "ngram-mod"], ["--spec-default"]])
def test_a_callers_ngram_speculation_splits_and_is_left_alone(monkeypatch, extra):
    _cluster_env(monkeypatch)
    out = run(ss.before_load(_request(speculative_type = None, llama_extra_args = extra), 4))
    assert ss.status()["topology"] == "layer_split" and out.speculative_type is None


def test_a_model_that_fits_keeps_its_speculation(monkeypatch):
    _cluster_env(monkeypatch, model_bytes = 10 * GIB)
    request = _request(speculative_type = "mtp")
    assert run(ss.before_load(request, 4)) is request
    assert ss.status()["topology"] == "single"


@pytest.mark.parametrize("flag", ["--no-mmproj", "--no-mmproj-auto"])
def test_an_inherited_projector_url_survives_no_mmproj(monkeypatch, flag):
    seen = _cluster_env(monkeypatch)
    monkeypatch.setenv("LLAMA_ARG_MMPROJ_URL", "https://example.com/mmproj.gguf")
    request = _request(llama_extra_args = [flag])
    assert run(ss.before_load(request, 4)) is request and seen == []
    assert "URL" in ss.status()["reason"]
    monkeypatch.delenv("LLAMA_ARG_MMPROJ_URL")
    request = _request(llama_extra_args = [flag, "--mmproj-url", "https://example.com/m.gguf"])
    run(ss.before_load(request, 4))
    assert ss.status()["topology"] == "layer_split"  # an argv URL is suppressed with it


def test_an_uncached_model_whose_kv_cannot_be_read_stays_local(monkeypatch):
    seen = _cluster_env(monkeypatch)
    monkeypatch.setattr(ss, "cached_repo_file", lambda path, variant: None)
    monkeypatch.setattr(ss, "remote_gguf_size_bytes", lambda *a, **k: 250 * GIB)
    request = _request(model_path = "org/big-GGUF", max_seq_length = 8192)
    assert run(ss.before_load(request, 4)) is request and seen == []
    assert "could not be sized" in ss.status()["reason"]


_hub_header = {
    "general.architecture": "tiny",
    "tiny.block_count": 90,
    "tiny.attention.head_count": 1,
    "tiny.embedding_length": 64,
}


def _llama_header(n_kv):
    return {
        "general.architecture": "llama",
        "llama.block_count": 80,
        "llama.attention.head_count": 64,
        "llama.attention.head_count_kv": n_kv,
        "llama.embedding_length": 8192,
    }


def _gguf_header(kvs, tensors = ()):
    import struct

    body = b""
    for key, (vtype, value) in kvs.items():
        body += struct.pack("<Q", len(key)) + key.encode() + struct.pack("<I", vtype)
        if vtype == 8:
            body += struct.pack("<Q", len(value)) + value.encode()
        elif vtype == 9:
            atype, items = value
            body += struct.pack("<IQ", atype, len(items))
            for item in items:
                body += (
                    struct.pack("<Q", len(item)) + item.encode()
                    if atype == 8
                    else struct.pack("<I", item)
                )
        elif vtype == 6:
            body += struct.pack("<f", value)
        else:
            body += struct.pack("<I", value)
    for name, dims, ttype in tensors:
        body += struct.pack("<Q", len(name)) + name.encode() + struct.pack("<I", len(dims))
        body += b"".join(struct.pack("<Q", d) for d in dims) + struct.pack("<IQ", ttype, 0)
    return struct.pack("<IIQQ", 0x46554747, 3, len(tensors), len(kvs)) + body


def test_the_header_reader_matches_the_local_kv_estimate(monkeypatch, tmp_path):
    import io

    kvs = {
        "general.architecture": (8, "llama"),
        "general.tags": (9, (8, ["a", "bb"])),
        "llama.block_count": (4, 80),
        "llama.rope.freq_base": (6, 500000.0),
        "llama.attention.head_count": (4, 64),
        "llama.attention.head_count_kv": (9, (4, [8, 2, 8])),
        "llama.embedding_length": (4, 8192),
        "tokenizer.ggml.tokens": (9, (8, ["x"] * 3)),
    }
    tensors = [
        ("token_embd.weight", [8192, 1000], 1),  # F16
        ("blk.0.attn_q.weight", [8192, 8192], 0),  # F32
        ("blk.1.ffn_up.weight", [256, 64], 12),  # Q4_K: 256 elements in 144 bytes
        ("blk.1.attn_q.weight", [8192, 8192], 1),
    ]
    vals, blocks = cs._read_gguf_header(io.BytesIO(_gguf_header(kvs, tensors)), tensors = True)
    assert vals == {
        "general.architecture": "llama",
        "llama.block_count": 80,
        "llama.attention.head_count": 64,
        "llama.attention.head_count_kv": 8,
        "llama.embedding_length": 8192,
    }
    # Tied embeddings: llama.cpp copies the embedding to the output device, one more lump.
    assert blocks == {0: 8192 * 8192 * 4, 1: 64 * 144 + 8192 * 8192 * 2, -2: 8192 * 1000 * 2}
    untied = tensors + [("output.weight", [8192, 1000], 1)]
    _vals, untied_blocks = cs._read_gguf_header(io.BytesIO(_gguf_header(kvs, untied)), tensors = True)
    # Untied: the output matrix is the lump, the embedding (-2) stays in host RAM.
    assert untied_blocks[-1] == untied_blocks[-2] == 8192 * 1000 * 2
    kv = cs._kv_from_scalars(vals, 8192, None, None, ss)
    assert kv == 80 * 8192 * 8 * (8192 // 64) * 4  # f16 K + V, as estimate_kv_bytes computes

    class FakeFs:
        def __init__(self, token = None):
            pass

        def open(
            self,
            path,
            mode,
            block_size = None,
        ):
            opened.append(path)
            return io.BytesIO(_gguf_header(kvs, tensors))

    import huggingface_hub
    from utils.models import model_config

    monkeypatch.setattr(huggingface_hub, "HfFileSystem", FakeFs)
    variant = SimpleNamespace(filename = "Q4/big-00001-of-00002.gguf", size_bytes = 1)
    monkeypatch.setattr(model_config, "list_gguf_variants", lambda repo, token: ([variant], False))
    opened = []
    doubled = {b: 2 * n for b, n in blocks.items()}
    assert cs._remote_gguf_header("org/big-GGUF", None, None, ss) == (vals, doubled)
    assert opened == [
        "org/big-GGUF/Q4/big-00001-of-00002.gguf",
        "org/big-GGUF/Q4/big-00002-of-00002.gguf",
    ]
    shard = tmp_path / "big.gguf"
    shard.write_bytes(_gguf_header(kvs, tensors))
    monkeypatch.setattr(ss, "gguf_shard_paths", lambda path: [str(shard), str(shard)])
    assert cs._local_gguf_header(str(shard), ss) == (vals, doubled)


def test_the_rounding_reserve_is_the_largest_block(monkeypatch):
    _cluster_env(monkeypatch)
    monkeypatch.setattr(cs, "_gguf_block_count", lambda path, ss: 100)
    run(ss.before_load(_request(), 4))
    average = cs.state().plan.reason
    monkeypatch.setattr(cs, "_local_gguf_header", lambda *a, **k: ({}, {0: 1 * GIB, 1: 5 * GIB}))
    run(ss.before_load(_request(), 4))
    # One layer is kept free per device: 4 x 2.5 GiB on average, 4 x 5 GiB for the largest.
    assert "267.5 GiB usable" in average and "257.5 GiB usable" in cs.state().plan.reason


def test_a_cached_model_whose_kv_cannot_be_read_stays_local(monkeypatch):
    seen = _cluster_env(monkeypatch)
    monkeypatch.setattr(ss, "estimate_kv_bytes", lambda *a, **k: None)
    request = _request(max_seq_length = 8192)
    assert run(ss.before_load(request, 4)) is request and seen == []
    assert "could not be sized" in ss.status()["reason"]
    monkeypatch.setattr(cs, "_local_gguf_header", lambda *a, **k: (_llama_header(8), {}))
    run(ss.before_load(request, 4))
    assert ss.status()["topology"] == "layer_split"  # the header reader sized it


def test_a_projector_beside_a_quant_subdirectory_is_charged(monkeypatch, tmp_path):
    quant = tmp_path / "snapshot" / "UD-Q4_K_XL"
    quant.mkdir(parents = True)
    weight = quant / "big-00001-of-00002.gguf"
    weight.write_bytes(b"x")
    mmproj = tmp_path / "snapshot" / "mmproj-F16.gguf"
    mmproj.write_bytes(b"x")
    _cluster_env(monkeypatch, model_bytes = 85 * GIB)
    monkeypatch.setattr(ss, "llama_server_supports", lambda flag, binary = None: True)
    sizes = {str(mmproj.resolve()): 10 * GIB}
    monkeypatch.setattr(
        ss, "gguf_size_bytes", lambda path: sizes.get(str(Path(path).resolve()), 85 * GIB)
    )
    run(ss.before_load(_request(model_path = str(weight)), 4))
    # 85 GiB of weights fits here (97.9 GiB usable); with the projector at 1.4x it does not.
    assert ss.status()["topology"] == "layer_split"
    monkeypatch.setattr(cs, "_quant_dir_projector", lambda *a, **k: None)
    run(ss.before_load(_request(model_path = str(weight)), 4))
    assert ss.status()["topology"] == "single"


def test_a_gpu_projector_is_charged_its_runtime_size(monkeypatch, tmp_path):
    mmproj = tmp_path / "mmproj.gguf"
    mmproj.write_bytes(b"x")
    _cluster_env(monkeypatch, model_bytes = 85 * GIB)
    monkeypatch.setattr(ss, "llama_server_supports", lambda flag, binary = None: True)
    sizes = {str(mmproj): 10 * GIB}
    monkeypatch.setattr(ss, "gguf_size_bytes", lambda path: sizes.get(str(path), 85 * GIB))
    request = _request(llama_extra_args = ["--mmproj", str(mmproj)])
    out = run(ss.before_load(request, 4))
    assert ss.status()["topology"] == "layer_split"  # 85 + 14 GiB
    assert out.llama_extra_args[out.llama_extra_args.index("--mmproj-device") + 1] == "CUDA0"
    monkeypatch.setattr(cs, "_MMPROJ_VRAM_SAFETY", 1.0)
    run(ss.before_load(request, 4))
    assert ss.status()["topology"] == "single"  # 85 + 10 GiB


@pytest.mark.parametrize(
    "kw, extra, env",
    [
        ({"n_ubatch": 2048}, [], {}),
        ({}, ["-ub", "1024"], {}),
        ({}, [], {"LLAMA_ARG_UBATCH": "4096"}),
    ],
)
def test_a_larger_micro_batch_keeps_the_load_local(monkeypatch, kw, extra, env):
    seen = _cluster_env(monkeypatch)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    request = _request(llama_extra_args = extra, **kw)
    assert run(ss.before_load(request, 4)) is request and seen == []
    assert "micro-batch" in ss.status()["reason"]


@pytest.mark.parametrize("kw", [{"n_ubatch": 256}, {"n_batch": 512, "n_ubatch": 4096}])
def test_a_default_or_smaller_micro_batch_still_splits(monkeypatch, kw):
    _cluster_env(monkeypatch)
    run(ss.before_load(_request(**kw), 4))
    assert ss.status()["topology"] == "layer_split"


def test_a_failed_load_after_retiring_stops_the_workers(monkeypatch):
    _cluster_env(monkeypatch)
    out = run(ss.before_load(_request(), 4))
    backend = SimpleNamespace(
        _process = SimpleNamespace(args = ["/b/llama-server"] + out.llama_extra_args),
        is_loaded = True,
    )
    run(ss.after_load(backend, 4))
    stopped = []

    async def stop():
        stopped.append(1)

    cs.state().workers["10.0.0.2:50052"] = SimpleNamespace(
        alive = True, snapshot = lambda: {}, stop = stop
    )
    monkeypatch.delenv(cs.ENV_NODES)
    monkeypatch.setattr(ss, "is_spark", lambda: False)
    run(ss.before_load(_request(), 4))
    backend.is_loaded = False
    run(ss.load_failed())
    assert stopped == [1] and not cs.state().workers
    assert not cs.state().retire_on_commit


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
    assert not getattr(out, "force_reload", False)  # nothing was detached this time


def test_a_restarted_served_worker_forces_a_relaunch(monkeypatch):
    _cluster_env(monkeypatch)
    out = run(ss.before_load(_request(), 4))
    backend = SimpleNamespace(
        _process = SimpleNamespace(args = ["/b/llama-server"] + out.llama_extra_args),
        is_loaded = True,
    )
    run(ss.after_load(backend, 4))
    out = run(ss.before_load(_request(), 4))
    assert not getattr(out, "force_reload", False)

    async def restart(self):
        self._started.add("10.0.0.3:50052")
        return {}

    monkeypatch.setattr(cs.ClusterServing, "_ensure_workers", restart)
    out = run(ss.before_load(_request(), 4))
    assert out.force_reload is True


def test_retiring_a_served_split_forces_a_relaunch(monkeypatch):
    _cluster_env(monkeypatch)
    out = run(ss.before_load(_request(), 4))
    backend = SimpleNamespace(
        _process = SimpleNamespace(args = ["/b/llama-server"] + out.llama_extra_args),
        is_loaded = True,
    )
    run(ss.after_load(backend, 4))
    monkeypatch.delenv(cs.ENV_NODES)
    monkeypatch.setattr(ss, "is_spark", lambda: False)
    out = run(ss.before_load(_request(), 4))
    assert out.force_reload is True and cs.state().retire_on_commit


def test_recurrent_state_is_charged_per_slot():
    vals = {
        "general.architecture": "qwen3next",
        "qwen3next.block_count": 48,
        "qwen3next.ssm.conv_kernel": 4,
        "qwen3next.ssm.inner_size": 4096,
        "qwen3next.ssm.state_size": 128,
        "qwen3next.ssm.group_count": 16,
    }
    per_layer = 3 * (4096 + 2 * 16 * 128) + 128 * 4096
    assert cs._ssm_state_bytes(vals, 8) == per_layer * 4 * 48 * 8
    assert cs._ssm_state_bytes({"general.architecture": "llama"}, 8) == 0


def test_a_state_space_model_without_attention_kv_is_sized(monkeypatch):
    _cluster_env(monkeypatch)
    monkeypatch.setattr(ss, "estimate_kv_bytes", lambda *a, **k: None)
    header = {
        "general.architecture": "mamba2",
        "mamba2.block_count": 64,
        "mamba2.ssm.conv_kernel": 4,
        "mamba2.ssm.inner_size": 8192,
        "mamba2.ssm.state_size": 128,
        "mamba2.ssm.group_count": 8,
    }
    monkeypatch.setattr(cs, "_local_gguf_header", lambda *a, **k: (header, {}))
    run(ss.before_load(_request(max_seq_length = 8192, ctx_checkpoints = 0), 4))
    assert ss.status()["topology"] == "layer_split"
    assert cs.state().plan.kv_bytes == cs._ssm_state_bytes(header, 4)


@pytest.mark.parametrize("env", [False, True])
def test_a_host_projector_url_keeps_the_load_local(monkeypatch, env):
    seen = _cluster_env(monkeypatch)
    if env:
        monkeypatch.setenv("LLAMA_ARG_MMPROJ_URL", "https://example.com/mmproj.gguf")
    extra = ["--no-mmproj-offload"] + ([] if env else ["-mmu", "https://example.com/m.gguf"])
    request = _request(llama_extra_args = extra)
    assert run(ss.before_load(request, 4)) is request and seen == []
    assert "URL" in ss.status()["reason"]


@pytest.mark.parametrize("extra, freed", [(["--no-mmproj-offload"], 0), ([], 21 * GIB)])
def test_a_host_ram_projector_is_not_counted_as_freed(monkeypatch, tmp_path, extra, freed):
    _cluster_env(monkeypatch, model_bytes = 85 * GIB)
    monkeypatch.setattr(ss, "llama_server_supports", lambda flag, binary = None: True)
    mmproj = tmp_path / "mmproj.gguf"
    mmproj.write_bytes(b"x")
    sizes = {str(mmproj): 15 * GIB}
    monkeypatch.setattr(ss, "gguf_size_bytes", lambda path: sizes.get(str(path), 85 * GIB))
    out = run(ss.before_load(_request(llama_extra_args = ["--mmproj", str(mmproj)] + extra), 4))
    assert ss.status()["topology"] == "layer_split"
    backend = SimpleNamespace(
        _process = SimpleNamespace(args = ["/b/llama-server"] + out.llama_extra_args),
        is_loaded = True,
    )
    run(ss.after_load(backend, 4))
    assert sum(cs.state()._resident.values()) == pytest.approx(85 * GIB + freed)


@pytest.mark.parametrize(
    "kw, extra, env",
    [
        ({"load_mode": "mlock"}, [], {}),
        ({"load_mode": "mmap+mlock"}, [], {}),
        ({}, ["--mlock"], {}),
        ({}, ["--load-mode", "mmap+mlock"], {}),
        ({}, [], {"LLAMA_ARG_MLOCK": "1"}),
        ({}, [], {"LLAMA_ARG_LOAD_MODE": "mlock"}),
    ],
)
def test_a_locked_host_copy_keeps_the_load_local(monkeypatch, kw, extra, env):
    seen = _cluster_env(monkeypatch)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    request = _request(llama_extra_args = extra, **kw)
    assert run(ss.before_load(request, 4)) is request and seen == []
    assert "mlock" in ss.status()["reason"]


@pytest.mark.parametrize("kw, env", [({"load_mode": "none"}, {}), ({}, {"LLAMA_ARG_MLOCK": "0"})])
def test_load_modes_without_a_locked_copy_still_split(monkeypatch, kw, env):
    _cluster_env(monkeypatch)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    run(ss.before_load(_request(**kw), 4))
    assert ss.status()["topology"] == "layer_split"


def test_the_loader_estimates_context_compute_from_a_file_or_a_hub_header(monkeypatch):
    import io

    path = os.environ.get("UNSLOTH_TEST_GGUF")
    if not path or not os.path.isfile(path):
        pytest.skip("needs UNSLOTH_TEST_GGUF")
    from core.inference.llama_cpp import LlamaCppBackend

    est = LlamaCppBackend.__new__(LlamaCppBackend)
    est._model_identifier = None
    est._read_gguf_metadata(path)
    want = est._compute_buffer_ctx_bytes(
        131072, 512, None, layer_split = True, flash_attn = True, n_parallel = 4
    )
    assert want > 0
    assert cs._ctx_compute_bytes(path, None, 131072, 512, None, True, 4) == want
    end = []
    with open(path, "rb") as f:
        cs._read_gguf_header(f, tensors = True, kv_end = end)
        f.seek(0)
        header_kv = f.read(end[0])
    assert cs._ctx_compute_bytes(None, header_kv, 131072, 512, None, True, 4) == want
    assert cs._ctx_compute_bytes(None, None, 131072, 512, None, True, 4) is None


def test_context_compute_is_kept_free_on_every_device(monkeypatch):
    _cluster_env(monkeypatch)
    run(ss.before_load(_request(max_seq_length = 8192), 4))
    plain = cs.state().plan.reason
    monkeypatch.setattr(cs, "_ctx_compute_bytes", lambda *a, **k: 2 * GIB if not a[7:] else 0)
    run(ss.before_load(_request(max_seq_length = 8192), 4))
    # 4 devices x 2 GiB less usable.
    assert "277.5 GiB usable" in plain and "269.5 GiB usable" in cs.state().plan.reason
    monkeypatch.setattr(cs, "_ctx_compute_bytes", lambda *a, **k: None)
    request = _request(max_seq_length = 8192)
    assert run(ss.before_load(request, 4)) is request
    assert "could not be sized" in ss.status()["reason"]


@pytest.mark.parametrize("extra, cells", [([], 8192), (["--no-kv-unified"], 64 * 256)])
def test_private_kv_streams_are_sized_per_slot(monkeypatch, extra, cells):
    _cluster_env(monkeypatch)
    seen = []
    monkeypatch.setattr(ss, "estimate_kv_bytes", lambda path, n, *a: seen.append(n) or 0)
    run(ss.before_load(_request(max_seq_length = 8192, llama_extra_args = extra), 64))
    assert seen == [cells]


def test_the_host_embedding_is_not_counted_as_freed(monkeypatch):
    _cluster_env(monkeypatch)
    monkeypatch.setattr(cs, "_local_gguf_header", lambda *a, **k: ({}, {0: GIB, -2: 3 * GIB}))
    out = run(ss.before_load(_request(), 4))
    backend = SimpleNamespace(
        _process = SimpleNamespace(args = ["/b/llama-server"] + out.llama_extra_args),
        is_loaded = True,
    )
    run(ss.after_load(backend, 4))
    plan = cs.state().plan
    assert plan.host_bytes == 3 * GIB
    held = 250 * GIB - 3 * GIB
    expected = sum(max(0.0, w * held - plan.layer_bytes) for w in plan.weights)
    assert sum(cs.state()._resident.values()) == pytest.approx(expected)


def test_adapter_tensors_join_the_layer_they_modify(monkeypatch, tmp_path):
    lora = tmp_path / "lora.gguf"
    lora.write_bytes(
        _gguf_header(
            {"general.architecture": (8, "llama")}, [("blk.3.attn_q.weight.lora_a", [64, 8192], 0)]
        )
    )
    _cluster_env(monkeypatch)
    monkeypatch.setattr(
        ss, "gguf_size_bytes", lambda path: 200 * GIB if path == "/models/big.gguf" else GIB
    )
    monkeypatch.setattr(cs, "_gguf_block_count", lambda path, ss: 100)
    monkeypatch.setattr(cs, "_local_gguf_header", lambda *a, **k: ({}, {3: GIB, 7: 2 * GIB}))
    run(ss.before_load(_request(), 4))
    assert cs.state()._max_block == 2 * GIB
    run(ss.before_load(_request(llama_extra_args = ["--lora", str(lora)]), 4))
    assert cs.state()._max_block == 2 * GIB  # block 3 grows to 1 GiB + 2 MiB, still not the max
    big = tmp_path / "big.gguf"
    big.write_bytes(
        _gguf_header(
            {"general.architecture": (8, "llama")},
            [("blk.3.ffn_up.weight.lora_b", [GIB // 4, 2], 0)],
        )
    )
    run(ss.before_load(_request(llama_extra_args = ["--lora", str(big)]), 4))
    assert cs.state()._max_block == 3 * GIB


def test_an_invalid_config_keeps_a_served_splits_workers(monkeypatch, tmp_path):
    _cluster_env(monkeypatch)
    out = run(ss.before_load(_request(), 4))
    backend = SimpleNamespace(
        _process = SimpleNamespace(args = ["/b/llama-server"] + out.llama_extra_args),
        is_loaded = True,
    )
    run(ss.after_load(backend, 4))
    stopped = []

    async def stop():
        stopped.append(1)

    cs.state().nodes = [replace(n, managed = True) for n in cs.state().nodes]
    cs.state().workers["10.0.0.2:50052"] = SimpleNamespace(
        alive = True, snapshot = lambda: {}, stop = stop
    )
    monkeypatch.setenv(cs.ENV_NODES, "bad host!")
    request = _request()
    assert run(ss.before_load(request, 4)) is request
    assert "cluster config" in ss.status()["reason"]
    run(ss.after_load(backend, 4))  # the route reused the loaded primary
    assert stopped == [] and "10.0.0.2:50052" in cs.state().workers


@pytest.mark.parametrize(
    "kw, extra, count",
    [({}, [], 32), ({"ctx_checkpoints": 4}, [], 4), ({}, ["--ctx-checkpoints", "0"], 0)],
)
def test_context_checkpoints_are_charged_to_this_machine(monkeypatch, kw, extra, count):
    _cluster_env(monkeypatch)
    monkeypatch.setattr(
        cs, "_ctx_compute_bytes", lambda *a, **k: 10 * 2**20 if a[7:] == ("checkpoint",) else 0
    )
    out = run(ss.before_load(_request(max_seq_length = 8192, llama_extra_args = extra, **kw), 4))
    plan = cs.state().plan
    charged = count * 4 * 10 * 2**20
    assert plan.extra_bytes == pytest.approx(charged)
    assert plan.uncertain_extra_bytes == pytest.approx(charged)  # never counted as freed
    assert out is not None


@pytest.mark.parametrize("flag", ["-mlock", "-lm"])
def test_short_mlock_spellings_keep_the_load_local(monkeypatch, flag):
    seen = _cluster_env(monkeypatch)
    extra = [flag] if flag == "-mlock" else [flag, "mlock"]
    request = _request(llama_extra_args = extra)
    assert run(ss.before_load(request, 4)) is request and seen == []


def test_asymmetric_key_and_value_widths_are_sized_apart():
    vals = dict(_llama_header(8))
    vals["llama.attention.key_length"] = 128
    vals["llama.attention.value_length"] = 256
    kv = cs._kv_from_scalars(vals, 1024, "q8_0", "f16", ss)
    per_cell = 128 * ss.kv_bytes_per_elem("q8_0") + 256 * ss.kv_bytes_per_elem("f16")
    assert kv == int(80 * 1024 * 8 * per_cell)


def test_lora_output_tensors_join_the_output_lump(monkeypatch, tmp_path):
    lora = tmp_path / "lora.gguf"
    lora.write_bytes(
        _gguf_header(
            {"general.architecture": (8, "llama")},
            [
                ("blk.0.attn_q.weight.lora_a", [64, 64], 0),
                ("output.weight.lora_b", [GIB // 4, 2], 0),
            ],
        )
    )
    _cluster_env(monkeypatch)
    monkeypatch.setattr(
        ss, "gguf_size_bytes", lambda path: 200 * GIB if path == "/models/big.gguf" else GIB
    )
    monkeypatch.setattr(cs, "_gguf_block_count", lambda path, ss: 100)
    monkeypatch.setattr(cs, "_local_gguf_header", lambda *a, **k: ({}, {0: GIB, -1: GIB}))
    run(ss.before_load(_request(llama_extra_args = ["--lora", str(lora)]), 4))
    assert cs.state()._max_block == 3 * GIB  # the 1 GiB output plus its 2 GiB adapter


def test_the_compute_reserve_prices_the_lighter_cache_axis_and_planned_flash_attention(monkeypatch):
    _cluster_env(monkeypatch)
    seen = []
    monkeypatch.setattr(cs, "_ctx_compute_bytes", lambda *a, **k: seen.append(a) or 0)
    monkeypatch.setattr(ss, "llama_server_supports", lambda flag, binary = None: True)
    extra = ["-ctk", "f16", "-ctv", "q4_0"]
    run(ss.before_load(_request(max_seq_length = 8192, llama_extra_args = extra), 4))
    assert [a for a in seen if not a[7:]][0][4] == "q4_0" and [a for a in seen if not a[7:]][0][
        5
    ] is True  # a quantized V forces flash attention on
    seen.clear()
    run(ss.before_load(_request(max_seq_length = 8192, llama_extra_args = ["-fa", "auto"]), 4))
    assert [a for a in seen if not a[7:]][0][
        5
    ] is False  # auto is priced as the costlier answer, as the loader does


def test_an_invalid_config_keeps_the_served_split_record(monkeypatch):
    _cluster_env(monkeypatch)
    out = run(ss.before_load(_request(), 4))
    backend = SimpleNamespace(
        _process = SimpleNamespace(args = ["/b/llama-server"] + out.llama_extra_args),
        is_loaded = True,
    )
    run(ss.after_load(backend, 4))
    block = list(cs.state().block)
    monkeypatch.setenv(cs.ENV_NODES, "bad host!")
    run(ss.before_load(_request(), 4))
    run(ss.after_load(backend, 4))  # the route reused the loaded primary
    assert cs.state().block == block and cs.state().topology == "layer_split"
    assert "still serving" in ss.status()["reason"]


def test_the_loaders_recurrent_estimate_is_charged(monkeypatch):
    _cluster_env(monkeypatch)
    monkeypatch.setattr(
        cs, "_ctx_compute_bytes", lambda *a, **k: 3 * GIB if a[7:] == ("recurrent",) else 0
    )
    run(ss.before_load(_request(max_seq_length = 8192, ctx_checkpoints = 0), 4))
    assert cs.state().plan.kv_bytes == 3 * GIB  # a KDA layout the header formula does not read


def test_a_quantized_v_cache_without_flash_attention_is_sized_as_f16(monkeypatch):
    _cluster_env(monkeypatch)
    monkeypatch.setattr(cs, "_local_gguf_header", lambda *a, **k: (_llama_header(8), {}))
    extra = ["-ctk", "q8_0", "-ctv", "q4_0"]
    for supports, v_type in ((True, "q4_0"), (False, "f16")):
        monkeypatch.setattr(ss, "llama_server_supports", lambda flag, binary = None: supports)
        run(ss.before_load(_request(max_seq_length = 8192, llama_extra_args = extra), 4))
        want = cs._kv_from_scalars(_llama_header(8), 8192, "q8_0", v_type, ss)
        assert cs.state().plan.kv_bytes == want


def test_a_served_split_planned_single_forces_a_relaunch(monkeypatch):
    _cluster_env(monkeypatch)
    out = run(ss.before_load(_request(), 4))
    backend = SimpleNamespace(
        _process = SimpleNamespace(args = ["/b/llama-server"] + out.llama_extra_args),
        is_loaded = True,
    )
    run(ss.after_load(backend, 4))
    monkeypatch.setattr(ss, "gguf_size_bytes", lambda path: 10 * GIB if path else None)
    out = run(ss.before_load(_request(), 4))
    assert ss.status()["topology"] == "single" and out.force_reload is True


def test_a_different_model_after_a_config_error_clears_the_split_record(monkeypatch):
    _cluster_env(monkeypatch)
    out = run(ss.before_load(_request(), 4))
    backend = SimpleNamespace(
        _process = SimpleNamespace(args = ["/b/llama-server"] + out.llama_extra_args),
        is_loaded = True,
    )
    run(ss.after_load(backend, 4))
    monkeypatch.setenv(cs.ENV_NODES, "bad host!")
    run(ss.before_load(_request(model_path = "/models/other.gguf"), 4))
    other = SimpleNamespace(
        _process = SimpleNamespace(args = ["/b/llama-server", "-m", "/models/other.gguf"]),
        is_loaded = True,
    )
    run(ss.after_load(other, 4))
    assert cs.state().block == [] and cs.state().plan is None
    assert cs.state().topology == "single"


def test_an_auto_context_is_priced_at_the_fitters_floor(monkeypatch):
    _cluster_env(monkeypatch)
    seen = []
    monkeypatch.setattr(ss, "estimate_kv_bytes", lambda path, n, *a: seen.append(n) or 0)
    run(ss.before_load(_request(max_seq_length = 0), 4))
    assert seen == [4096]
    header = dict(_llama_header(8), **{"llama.context_length": 2048})
    monkeypatch.setattr(cs, "_local_gguf_header", lambda *a, **k: (header, {}))
    run(ss.before_load(_request(max_seq_length = 0), 4))
    assert cs.state().plan.kv_bytes == cs._kv_from_scalars(header, 2048, None, None, ss)


def test_a_projector_raises_the_micro_batch_compute_is_priced_at(monkeypatch, tmp_path):
    _cluster_env(monkeypatch)
    mmproj = tmp_path / "mmproj.gguf"
    mmproj.write_bytes(b"x")
    from core.inference import llama_cpp

    monkeypatch.setattr(llama_cpp, "_mmproj_required_ubatch", lambda *a, **k: 2048)
    monkeypatch.setattr(
        ss, "gguf_size_bytes", lambda path: GIB if str(path) == str(mmproj) else 250 * GIB
    )
    seen = []
    monkeypatch.setattr(cs, "_ctx_compute_bytes", lambda *a, **k: seen.append(a[3]) or 0)
    run(
        ss.before_load(_request(max_seq_length = 8192, llama_extra_args = ["--mmproj", str(mmproj)]), 4)
    )
    assert seen[0] == 2048
    seen.clear()
    run(ss.before_load(_request(max_seq_length = 8192, disable_vision = True), 4))
    assert seen[0] == 512


def test_recurrent_layers_reserve_their_own_share(monkeypatch):
    _cluster_env(monkeypatch)
    header = dict(_llama_header(8), **{"llama.attention.head_count_kv.zeros": 10})
    monkeypatch.setattr(cs, "_local_gguf_header", lambda *a, **k: (header, {0: GIB}))
    monkeypatch.setattr(
        cs, "_ctx_compute_bytes", lambda *a, **k: 10 * GIB if a[7:] == ("recurrent",) else 0
    )
    run(ss.before_load(_request(max_seq_length = 8192, ctx_checkpoints = 0), 4))
    plan = cs.state().plan
    attention = plan.kv_bytes - 10 * GIB
    assert plan.layer_bytes == pytest.approx(GIB + attention / 80 + GIB)  # 10 GiB over 10 layers


def test_skipped_nextn_blocks_are_not_counted_as_freed(monkeypatch):
    _cluster_env(monkeypatch)
    header = dict(_llama_header(8), **{"llama.nextn_predict_layers": 1})
    monkeypatch.setattr(cs, "_local_gguf_header", lambda *a, **k: (header, {78: GIB, 79: 4 * GIB}))
    run(ss.before_load(_request(), 4))
    assert cs.state().plan.host_bytes == 4 * GIB  # block 79, the MTP head, is never loaded


def test_checkpoints_are_priced_with_the_heavier_cache_axis(monkeypatch):
    _cluster_env(monkeypatch)
    seen = []
    monkeypatch.setattr(cs, "_ctx_compute_bytes", lambda *a, **k: seen.append(a) or 0)
    extra = ["-ctk", "q4_0", "-ctv", "f16"]
    run(ss.before_load(_request(max_seq_length = 8192, llama_extra_args = extra), 4))
    checkpoint = [a for a in seen if a[7:] == ("checkpoint",)]
    assert checkpoint and checkpoint[0][4] == "f16"


@pytest.mark.parametrize(
    "env, extra",
    [
        ({cs.ENV_TOPOLOGY: "single"}, []),
        ({}, ["--device", "CUDA0"]),
        ({}, ["--no-kv-offload"]),
    ],
)
def test_every_path_that_drops_a_served_split_forces_a_relaunch(monkeypatch, env, extra):
    _cluster_env(monkeypatch)
    out = run(ss.before_load(_request(), 4))
    backend = SimpleNamespace(
        _process = SimpleNamespace(args = ["/b/llama-server"] + out.llama_extra_args),
        is_loaded = True,
    )
    run(ss.after_load(backend, 4))
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    out = run(ss.before_load(_request(llama_extra_args = extra), 4))
    assert ss.status()["topology"] == "single" and out.force_reload is True


def test_the_loaders_architecture_aware_kv_estimate_wins(monkeypatch):
    _cluster_env(monkeypatch)
    monkeypatch.setattr(cs, "_local_gguf_header", lambda *a, **k: (_llama_header(8), {}))
    monkeypatch.setattr(
        cs, "_ctx_compute_bytes", lambda *a, **k: 5 * GIB if a[7:] == ("kv",) else 0
    )
    run(ss.before_load(_request(max_seq_length = 8192, ctx_checkpoints = 0), 4))
    assert cs.state().plan.kv_bytes == 5 * GIB  # e.g. compact sliding-window layers
    monkeypatch.setattr(cs, "_ctx_compute_bytes", lambda *a, **k: 0)
    run(ss.before_load(_request(max_seq_length = 8192, ctx_checkpoints = 0), 4))
    assert cs.state().plan.kv_bytes == cs._kv_from_scalars(_llama_header(8), 8192, None, None, ss)
