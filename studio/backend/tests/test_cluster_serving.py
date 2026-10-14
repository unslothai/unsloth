# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""N-node serving over llama.cpp RPC: config, device listing, placement, launch args, reload
stripping, and that nothing of it runs (or changes the two-Spark path) without a cluster config."""

from __future__ import annotations

import asyncio
import json
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
    run(ss.before_load(_request(), 4))
    _cluster_env(monkeypatch, model_bytes = 10 * GIB)
    run(ss.before_load(_request(), 4))
    assert ss.status()["topology"] == "single"
    run(ss.load_failed())
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
    run(ss.before_load(_request(max_seq_length = 131072), 4))
    # 85 GiB of weights alone fits here; with the KV it might not, so it is split.
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
    assert args[args.index("--device-draft") + 1] == "CUDA0"
    # The caller's own choice stands, and a build without the flag gets none.
    out = run(ss.before_load(_request(llama_extra_args = ["-mmdev", "none"]), 4))
    assert "--mmproj-device" not in out.llama_extra_args
    monkeypatch.setattr(ss, "llama_server_supports", lambda flag, binary = None: False)
    out = run(ss.before_load(_request(), 4))
    assert "--mmproj-device" not in out.llama_extra_args


def test_a_disabled_projector_is_not_charged(monkeypatch):
    _cluster_env(monkeypatch, model_bytes = 85 * GIB)
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


def test_a_split_turns_automatic_speculation_off_but_keeps_a_chosen_one(monkeypatch):
    _cluster_env(monkeypatch)
    out = run(ss.before_load(_request(speculative_type = None), 4))
    assert out.speculative_type == "off"
    assert "speculative decoding off" in ss.status()["reason"]
    out = run(ss.before_load(_request(speculative_type = "mtp"), 4))
    assert out.speculative_type == "mtp"
    out = run(
        ss.before_load(
            _request(speculative_type = None, llama_extra_args = ["--spec-type", "ngram-mod"]), 4
        )
    )
    assert out.speculative_type is None


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
        return request

    monkeypatch.setattr(ss.SparkServing, "before_load", legacy_before_load)
    run(ss.before_load(_request(), 4))
    assert seen == [[1]], "the cluster's workers stop before the two-Spark path plans"


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
