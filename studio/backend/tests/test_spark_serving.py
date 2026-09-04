# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The two-Spark serving orchestrator, off any real hardware.

``spark_cluster`` is replaced by a stub so the tests pin the contract this module has
with it (what it asks the planner, which paths it launches from) and the gate that
keeps all of it off a normal machine. ssh never runs: the remote calls are patched.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

from core.inference import spark_serving as ss
from core.inference.spark_router import CONVERSATION_FIELD
from .spark_fake_llama import FakeLlama

GIB = 2**30


class StubCluster:
    """The slice of studio.spark_cluster the orchestrator uses, recording every call."""

    RPC_DEFAULT_PORT = 50052
    SPARK_USABLE_GIB = 121.69
    SERVE_OVERHEAD_GIB = 8.0

    def __init__(
        self,
        spark: bool = True,
        peer: Optional[str] = "192.168.200.13",
        topology: str = "single",
    ):
        self.spark = spark
        self.peer = peer
        self.topology = topology
        self.planner_calls: List[Dict[str, Any]] = []
        self.preflight_result: Dict[str, Any] = {
            "ok": True,
            "problems": [],
            "notes": ["bundles match"],
        }

    def is_dgx_spark(self) -> bool:
        return self.spark

    def peer_ip_for(self) -> Optional[str]:
        return self.peer

    def load_config(self) -> Dict[str, Any]:
        return {}

    def cabled_rails(self) -> List[Dict[str, Any]]:
        return [{"ipv4": ["192.168.200.12"]}] if self.spark else []

    def llama_bundle_dir(self) -> Path:
        return Path.home() / ".unsloth" / "llama.cpp"

    def rpc_server_binary(self) -> Optional[str]:
        return str(Path.home() / ".unsloth" / "llama.cpp" / "build" / "bin" / "ggml-rpc-server")

    def rpc_protocol_preflight(
        self,
        peer_ip: str,
        port: int = 50052,
    ) -> Dict[str, Any]:
        return dict(self.preflight_result, peer = peer_ip, port = port)

    def recommend_topology(
        self,
        model_bytes,
        kv_bytes_per_user,
        users,
        prompt_tokens,
        per_node_free_bytes,
        prefill_heavy = False,
    ):
        self.planner_calls.append(
            dict(
                model_bytes = model_bytes,
                kv_bytes_per_user = kv_bytes_per_user,
                users = users,
                prompt_tokens = prompt_tokens,
                per_node_free_bytes = per_node_free_bytes,
                prefill_heavy = prefill_heavy,
            )
        )
        return {"topology": self.topology, "reason": f"stub says {self.topology}", "speedup": 1.3}


@pytest.fixture
def cluster(monkeypatch):
    stub = StubCluster()
    ss.reset_for_tests()
    monkeypatch.setattr(ss, "_CLUSTER", stub)
    monkeypatch.setattr(ss, "_CLUSTER_LOOKED_UP", True)
    monkeypatch.delenv(ss.ENV_TOGGLE, raising = False)
    monkeypatch.delenv(ss.ENV_TOPOLOGY, raising = False)
    monkeypatch.delenv(ss.ENV_PEER, raising = False)
    yield stub
    ss.reset_for_tests()


def run(coro):
    return asyncio.run(coro)


# ── The gate ─────────────────────────────────────────────────────────────


def test_off_by_default_everywhere_but_a_paired_spark(cluster, monkeypatch):
    assert ss.enabled()
    cluster.peer = None
    assert not ss.enabled(), "a Spark with no configured peer stays single"
    cluster.peer = "192.168.200.13"
    cluster.spark = False
    assert not ss.enabled(), "not a Spark: nothing runs"
    cluster.spark = True
    monkeypatch.setenv(ss.ENV_TOGGLE, "0")
    assert not ss.enabled(), "the kill switch wins"
    monkeypatch.delenv(ss.ENV_TOGGLE)
    monkeypatch.setenv(ss.ENV_PEER, "10.0.0.2")
    assert ss.peer_address() == "10.0.0.2"


def test_module_entry_points_are_no_ops_off_a_spark(cluster):
    cluster.spark = False
    assert ss.status() == {"enabled": False, "topology": None, "reason": "not a paired DGX Spark"}
    assert ss.current_topology() is None
    assert ss.route_base_url(object()) is None
    payload: Dict[str, Any] = {}
    ss.tag_conversation(payload, "thread")
    assert payload == {}
    request = SimpleNamespace(model_path = "x")
    assert run(ss.before_load(request, 16)) is request
    run(ss.after_load(SimpleNamespace(is_loaded = True), 16))
    run(ss.shutdown())


def test_import_path_has_no_posix_only_calls():
    source = Path(ss.__file__).read_text(encoding = "utf-8")
    for name in ("os.fork", "os.setsid", "os.killpg", "signal.SIGHUP", "fcntl", "resource."):
        assert name not in source, name
    assert "asyncio.create_subprocess_exec" in source


# ── Planner contract ─────────────────────────────────────────────────────


def test_plan_topology_hands_the_shared_planner_bytes_users_and_budget(cluster):
    cluster.topology = "replicas"
    out = ss.plan_topology(16 * GIB, users = 16, kv_bytes_per_user = 0.25 * GIB)
    assert out["topology"] == "replicas" and out["reason"] == "stub says replicas"
    call = cluster.planner_calls[-1]
    assert call["model_bytes"] == 16 * GIB and call["kv_bytes_per_user"] == 0.25 * GIB
    assert call["users"] == 16 and call["prompt_tokens"] == ss.PROMPT_TOKENS_DEFAULT
    assert call["per_node_free_bytes"] == pytest.approx((121.69 - 8.0) * GIB)
    assert call["prefill_heavy"] is False
    # An unknown size is never guessed.
    assert ss.plan_topology(None, users = 32)["topology"] == "single"
    assert len(cluster.planner_calls) == 1
    # A planner answer outside the three names degrades to single.
    cluster.topology = "tensor-parallel"
    assert ss.plan_topology(1, users = 1)["topology"] == "single"


def test_forced_topology_overrides_and_says_so(cluster, monkeypatch):
    cluster.topology = "single"
    monkeypatch.setenv(ss.ENV_TOPOLOGY, "replicas")
    plan = ss.state().decide(model_bytes = GIB, users = 1, kv_bytes_per_user = 0)
    assert plan["topology"] == "replicas" and plan["recommended"] == "single"
    assert ss.ENV_TOPOLOGY in plan["reason"]


def test_gguf_size_counts_every_shard(tmp_path):
    for i in (1, 2, 3):
        (tmp_path / f"m-0000{i}-of-00003.gguf").write_bytes(b"x" * (10 * i))
    assert ss.gguf_size_bytes(str(tmp_path / "m-00001-of-00003.gguf")) == 60
    (tmp_path / "single.gguf").write_bytes(b"y" * 7)
    assert ss.gguf_size_bytes(str(tmp_path / "single.gguf")) == 7
    assert ss.gguf_size_bytes(str(tmp_path / "missing.gguf")) is None
    assert ss.gguf_size_bytes(None) is None


def test_cached_repo_file_finds_the_variant_in_the_hub_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    snap = tmp_path / "models--unsloth--Demo-GGUF" / "snapshots" / "abc"
    snap.mkdir(parents = True)
    (snap / "Demo-UD-Q4_K_XL.gguf").write_bytes(b"a" * 5)
    (snap / "mmproj-F16.gguf").write_bytes(b"b" * 5)
    (snap / "Demo-Q8_0.gguf").write_bytes(b"c" * 9)
    assert ss.cached_repo_file("unsloth/Demo-GGUF", "UD-Q4_K_XL") == str(
        snap / "Demo-UD-Q4_K_XL.gguf"
    )
    assert ss.cached_repo_file("unsloth/Nope-GGUF", "Q8_0") is None
    local = tmp_path / "local.gguf"
    local.write_bytes(b"z")
    assert ss.cached_repo_file(str(local), None) == str(local)


# ── Command construction ─────────────────────────────────────────────────


def test_replica_argv_repoints_only_host_and_port():
    local = [
        "/home/u/.unsloth/llama.cpp/build/bin/llama-server",
        "-m",
        "/m.gguf",
        "--port",
        "41234",
        "--parallel",
        "16",
        "-c",
        "8192",
        "--flash-attn",
        "on",
    ]
    out = ss.replica_argv(
        local,
        binary = "$HOME/.unsloth/llama.cpp/build/bin/llama-server",
        host = "192.168.200.13",
        port = 41234,
    )
    assert out == [
        "$HOME/.unsloth/llama.cpp/build/bin/llama-server",
        "-m",
        "/m.gguf",
        "--parallel",
        "16",
        "-c",
        "8192",
        "--flash-attn",
        "on",
        "--host",
        "192.168.200.13",
        "--port",
        "41234",
    ]


def test_rpc_server_and_layer_split_arguments():
    assert ss.rpc_server_argv(
        "/b/ggml-rpc-server", bind = "192.168.200.13", port = 50052, cache = True
    ) == ["/b/ggml-rpc-server", "-H", "192.168.200.13", "-p", "50052", "-c"]
    assert ss.rpc_server_argv("/b/ggml-rpc-server", bind = "0.0.0.0", port = 50053, cache = False) == [
        "/b/ggml-rpc-server",
        "-H",
        "0.0.0.0",
        "-p",
        "50053",
    ]
    extra = ss.layer_split_extra_args("192.168.200.13", 50052)
    assert extra == ["--rpc", "192.168.200.13:50052", "--device", "CUDA0,RPC0", "-sm", "layer"]
    assert not any("pipeline" in a for a in extra), "llama.cpp enables pipelining itself"


def test_peer_binary_candidates_prefer_the_local_bundle_path(cluster):
    local = str(Path.home() / ".unsloth" / "llama.cpp" / "build" / "bin" / "llama-server")
    candidates = ss.peer_binary_candidates(local, "llama-server")
    assert candidates[0] == "$HOME/.unsloth/llama.cpp/build/bin/llama-server"
    assert candidates[-1] == "llama-server"
    assert len(candidates) == len(set(candidates))
    script = ss.find_binary_script(candidates)
    assert script.endswith("echo MISSING; exit 1") and "command -v llama-server" in script


def test_peer_process_remote_command_prints_pid_then_execs():
    process = ss.PeerProcess(
        "llama-server", "192.168.200.13", ["/b/llama-server", "-m", "/p/a b.gguf"]
    )
    assert (
        process.remote_command == "echo UNSLOTH_SPARK_PID=$$; exec /b/llama-server -m '/p/a b.gguf'"
    )
    assert not process.alive
    snap = process.snapshot()
    assert snap["remote_pid"] is None and snap["alive"] is False


# ── Load hooks ───────────────────────────────────────────────────────────


class _FakeRequest:
    def __init__(self, model_path: str, **kw):
        self.model_path = model_path
        self.gguf_variant = kw.get("gguf_variant")
        self.context_length = kw.get("context_length", 0)
        self.llama_extra_args = kw.get("llama_extra_args")

    def model_copy(self, update):
        clone = _FakeRequest(
            self.model_path,
            gguf_variant = self.gguf_variant,
            context_length = self.context_length,
            llama_extra_args = self.llama_extra_args,
        )
        for k, v in update.items():
            setattr(clone, k, v)
        return clone


def _patch_remote(
    monkeypatch,
    *,
    binary = "$HOME/.unsloth/llama.cpp/build/bin/ggml-rpc-server",
    model_present = True,
    port_opens = True,
):
    calls: List[str] = []

    async def fake_ssh_run(
        peer,
        remote,
        timeout = 20.0,
    ):
        # Same contract as the real remote shell: the lookup script prints the first
        # executable and exits 0, or MISSING and exits 1, and nothing after it runs;
        # the file check prints YES or NO.
        calls.append(remote)
        if remote.startswith("test -f"):
            return 0, "YES\n" if model_present else "NO\n", ""
        if "echo MISSING" in remote:
            if binary == "MISSING":
                return 1, "MISSING\n", ""
            return 0, f"{binary}\n", ""
        return 0, "", ""

    started: List[ss.PeerProcess] = []

    async def fake_start(self):
        started.append(self)
        self.started_at = None  # never "started then died", so no relaunch in tests

    async def fake_stop(self, timeout = 10.0):
        return None

    async def fake_wait(host, port, timeout):
        return port_opens

    monkeypatch.setattr(ss, "ssh_run", fake_ssh_run)
    monkeypatch.setattr(ss.PeerProcess, "start", fake_start)
    monkeypatch.setattr(ss.PeerProcess, "stop", fake_stop)
    monkeypatch.setattr(ss, "wait_for_port", fake_wait)
    return calls, started


def test_before_load_turns_a_too_large_model_into_a_layer_split(cluster, monkeypatch, tmp_path):
    cluster.topology = "layer_split"
    model = tmp_path / "big.gguf"
    model.write_bytes(b"x" * 1024)
    calls, started = _patch_remote(monkeypatch)
    request = _FakeRequest(str(model), llama_extra_args = ["--seed", "1"])
    out = run(ss.before_load(request, 4))
    assert out is not request
    assert out.llama_extra_args == [
        "--seed",
        "1",
        "--rpc",
        "192.168.200.13:50052",
        "--device",
        "CUDA0,RPC0",
        "-sm",
        "layer",
    ]
    state = ss.state()
    assert state.topology == "layer_split" and state.peer == "192.168.200.13"
    assert state.preflight["ok"] is True
    assert started and started[0].name == "ggml-rpc-server"
    assert started[0].argv == [
        "$HOME/.unsloth/llama.cpp/build/bin/ggml-rpc-server",
        "-H",
        "192.168.200.13",
        "-p",
        "50052",
        "-c",
    ]
    assert (
        cluster.planner_calls[-1]["model_bytes"] == 1024 and cluster.planner_calls[-1]["users"] == 4
    )
    run(ss.shutdown())
    assert ss.state().topology == "single"


def test_before_load_refuses_a_layer_split_the_preflight_rejects(cluster, monkeypatch, tmp_path):
    cluster.topology = "layer_split"
    cluster.preflight_result = {
        "ok": False,
        "problems": ["llama.cpp bundle mismatch: b10715 here, b10796 there"],
        "notes": [],
    }
    model = tmp_path / "big.gguf"
    model.write_bytes(b"x")
    _calls, started = _patch_remote(monkeypatch)
    request = _FakeRequest(str(model))
    out = run(ss.before_load(request, 4))
    assert out is request and out.llama_extra_args is None
    assert not started
    assert ss.state().topology == "single"
    assert "bundle mismatch" in ss.state().reason


def test_before_load_without_rpc_cache_when_the_peer_lacks_the_model(
    cluster, monkeypatch, tmp_path
):
    cluster.topology = "layer_split"
    model = tmp_path / "big.gguf"
    model.write_bytes(b"x")
    _calls, started = _patch_remote(monkeypatch, model_present = False)
    run(ss.before_load(_FakeRequest(str(model)), 4))
    assert started[0].argv[-1] != "-c"
    assert ss.state().peer_model_present is False
    run(ss.shutdown())


def test_before_load_leaves_single_and_replicas_alone(cluster, monkeypatch, tmp_path):
    cluster.topology = "replicas"
    model = tmp_path / "m.gguf"
    model.write_bytes(b"x")
    _calls, started = _patch_remote(monkeypatch)
    request = _FakeRequest(str(model))
    assert run(ss.before_load(request, 16)) is request
    assert not started and ss.state().topology == "single"


class _FakeBackend:
    def __init__(
        self,
        port: int,
        gguf_path: str,
        slots: int = 16,
    ):
        self._port = port
        self._process = SimpleNamespace(
            args = [
                "/home/x/.unsloth/llama.cpp/build/bin/llama-server",
                "-m",
                gguf_path,
                "--port",
                str(port),
                "--parallel",
                str(slots),
            ]
        )
        self._gguf_path = gguf_path
        self._effective_context_length = 4096
        self.effective_parallel_slots = slots
        self._healthy = True

    @property
    def is_loaded(self):
        return self._process is not None and self._healthy

    @property
    def gguf_path(self):
        return self._gguf_path


def test_after_load_attaches_replicas_behind_the_router(cluster, monkeypatch, tmp_path):
    cluster.topology = "replicas"
    monkeypatch.setenv(ss.ENV_PEER, "127.0.0.1")
    model = tmp_path / "m.gguf"
    model.write_bytes(b"x" * 100)
    _calls, started = _patch_remote(
        monkeypatch, binary = "$HOME/.unsloth/llama.cpp/build/bin/llama-server"
    )

    async def scenario():
        fake = await FakeLlama("main").start()
        backend = _FakeBackend(fake.port, str(model))
        try:
            await ss.after_load(backend, 16)
            state = ss.state()
            assert state.topology == "replicas", state.reason
            assert state.router is not None and state.router.running
            assert started and started[0].name == "llama-server"
            assert started[0].argv[-4:] == ["--host", "127.0.0.1", "--port", str(fake.port)]
            assert "--port" not in started[0].argv[:-4]
            # Every existing code path now goes through the router.
            assert ss.route_base_url(backend) == state.router.base_url
            payload: Dict[str, Any] = {}
            ss.tag_conversation(payload, "thread-7")
            assert payload == {CONVERSATION_FIELD: "thread-7"}
            # A respawn on a new port goes direct until the supervisor re-points.
            backend._port = fake.port + 1
            assert ss.route_base_url(backend) is None
            backend._port = fake.port
            status = ss.status()
            assert status["topology"] == "replicas" and status["router"]["healthy_backends"] >= 1
            assert [b["name"] for b in status["router"]["backends"]] == ["main", "peer"]
            assert ss.current_topology() == "replicas"
            # Unloading this node's server tears the peer down.
            backend._process = None
            await asyncio.sleep(ss.SUPERVISOR_INTERVAL_S * 2.5)
            assert ss.state().topology == "single"
            assert ss.state().router is None and ss.state().peer_process is None
        finally:
            await ss.shutdown()
            await fake.stop()

    run(scenario())


def test_after_load_stays_single_when_the_peer_lacks_the_model_or_binary(
    cluster, monkeypatch, tmp_path
):
    cluster.topology = "replicas"
    monkeypatch.setenv(ss.ENV_PEER, "127.0.0.1")
    model = tmp_path / "m.gguf"
    model.write_bytes(b"x")
    _calls, started = _patch_remote(
        monkeypatch, binary = "$HOME/.unsloth/llama.cpp/build/bin/llama-server", model_present = False
    )
    backend = _FakeBackend(12345, str(model))
    run(ss.after_load(backend, 16))
    assert not started and ss.state().topology == "single"
    assert "does not have" in ss.state().reason and "rsync" in ss.state().reason
    assert ss.route_base_url(backend) is None
    _calls, started = _patch_remote(monkeypatch, binary = "MISSING")
    run(ss.after_load(backend, 16))
    assert not started and "provision" in ss.state().reason


def test_after_load_records_single_below_the_replica_threshold(cluster, monkeypatch, tmp_path):
    cluster.topology = "single"
    model = tmp_path / "m.gguf"
    model.write_bytes(b"x")
    _calls, started = _patch_remote(monkeypatch)
    run(ss.after_load(_FakeBackend(12345, str(model), slots = 2), 2))
    assert not started and ss.state().topology == "single"
    assert ss.state().reason == "stub says single"
    assert cluster.planner_calls[-1]["users"] == 2
