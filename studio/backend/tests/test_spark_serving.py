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
import time
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
        self.bundle: Optional[Path] = None  # the fixture points this at an empty tmp dir
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
        return self.bundle or Path.home() / ".unsloth" / "llama.cpp"

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
def cluster(monkeypatch, tmp_path):
    stub = StubCluster()
    # An empty bundle: no llama-server to probe, so the machine running the tests
    # never decides whether a layer split gets pipeline groups.
    stub.bundle = tmp_path / "bundle"
    stub.bundle.mkdir()
    ss.reset_for_tests()
    monkeypatch.setattr(ss, "_CLUSTER", stub)
    monkeypatch.setattr(ss, "_CLUSTER_LOOKED_UP", True)
    monkeypatch.delenv(ss.ENV_TOGGLE, raising = False)
    monkeypatch.delenv(ss.ENV_TOPOLOGY, raising = False)
    monkeypatch.delenv(ss.ENV_PEER, raising = False)
    monkeypatch.delenv(ss.ENV_PIPELINE_GROUPS, raising = False)
    yield stub
    ss.reset_for_tests()


# A stand-in llama-server: prints a --help that may or may not name the flag, and
# records every run beside itself so a test can see whether the probe ran at all.
_FAKE_HELP_WITH_FLAG = """usage: llama-server [options]
  -np, --parallel N            number of server slots (default: 4)
  --pipeline-groups N          number of pipeline groups the slots are split over
"""
_FAKE_HELP_WITHOUT_FLAG = """usage: llama-server [options]
  -np, --parallel N            number of server slots (default: 4)
  --kv-unified                 one KV buffer shared by all slots
"""


def write_fake_llama_server(
    directory: Path,
    help_text: str,
    *,
    body: str = "",
) -> Path:
    directory.mkdir(parents = True, exist_ok = True)
    script = directory / "llama-server"
    script.write_text(
        "#!/bin/sh\n" 'echo run >> "$0.calls"\n' + body + "cat <<'EOF'\n" + help_text + "EOF\n",
        encoding = "utf-8",
    )
    script.chmod(0o755)
    return script


def probe_runs(script: Path) -> int:
    calls = Path(str(script) + ".calls")
    return len(calls.read_text().splitlines()) if calls.exists() else 0


def run(coro):
    return asyncio.run(coro)


def test_ssh_user_is_this_login_and_never_a_fixed_one(monkeypatch):
    """The peer is reached as the account that ran `provision`: the environment's login,
    else the login database, never a hardcoded account."""
    import getpass

    monkeypatch.setenv("USER", "alice")
    assert ss._ssh_user() == "alice"
    for var in ("USER", "USERNAME", "LOGNAME"):
        monkeypatch.delenv(var, raising = False)
    assert ss._ssh_user() == getpass.getuser()
    assert ss.ssh_argv("192.168.200.13", "true")[-2] == f"{getpass.getuser()}@192.168.200.13"


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
        "--slot-save-path",
        "/home/u/.unsloth/studio/cache/llama-slots",
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
    assert not any("pipeline" in a for a in extra), "no groups asked for: today's launch"
    # Pipeline groups ride on the same launch, with a slot count that gives every
    # group a slot; the --parallel here overrides the emitted one (last wins).
    grouped = ss.layer_split_extra_args("192.168.200.13", 50052, pipeline_groups = 2, slots = 4)
    assert grouped == [
        "--rpc",
        "192.168.200.13:50052",
        "--device",
        "CUDA0,RPC0",
        "-sm",
        "layer",
        "--pipeline-groups",
        "2",
        "--parallel",
        "4",
    ]
    assert ss.layer_split_extra_args("p", 1, pipeline_groups = 1, slots = 4) == extra[:0] + [
        "--rpc",
        "p:1",
        "--device",
        "CUDA0,RPC0",
        "-sm",
        "layer",
    ]


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
        self.max_seq_length = kw.get("max_seq_length", 0)
        self.cache_type_kv = kw.get("cache_type_kv")
        self.llama_extra_args = kw.get("llama_extra_args")

    def model_copy(self, update):
        clone = _FakeRequest(
            self.model_path,
            gguf_variant = self.gguf_variant,
            max_seq_length = self.max_seq_length,
            cache_type_kv = self.cache_type_kv,
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
        # Looks alive for as long as the test runs, so nothing relaunches it.
        self.proc = SimpleNamespace(returncode = None)
        self.started_at = None

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


def test_launch_files_names_every_sidecar_the_launch_uses(tmp_path):
    weights = tmp_path / "m.gguf"
    mmproj = tmp_path / "mmproj.gguf"
    weights.write_bytes(b"w")
    mmproj.write_bytes(b"p")
    argv = [
        "/b/llama-server",
        "-m",
        str(weights),
        "--mmproj",
        str(mmproj),
        "--slot-save-path",
        str(tmp_path),
        "--alias",
        "unsloth/demo",
        "-c",
        "4096",
    ]
    assert ss.launch_files(argv, str(weights)) == [str(weights), str(mmproj)]


def test_before_load_reuses_a_live_rpc_server_and_after_load_reconciles(
    cluster, monkeypatch, tmp_path
):
    cluster.topology = "layer_split"
    model = tmp_path / "big.gguf"
    model.write_bytes(b"x")
    _calls, started = _patch_remote(monkeypatch)
    first = run(ss.before_load(_FakeRequest(str(model)), 4))
    assert len(started) == 1 and "--rpc" in first.llama_extra_args
    # A reload (no-op or not) keeps the running rpc-server: it is model-agnostic.
    again = run(ss.before_load(_FakeRequest(str(model)), 4))
    assert len(started) == 1 and "--rpc" in again.llama_extra_args
    assert ss.state().peer_process is started[0]
    # The launch that carries --rpc is the split; it is attached, not torn down.
    split_backend = _FakeBackend(4242, str(model))
    split_backend._process.args += ["--rpc", "192.168.200.13:50052"]
    run(ss.after_load(split_backend, 4))
    assert ss.state().topology == "layer_split" and ss.state().peer_process is started[0]
    assert ss.state().attached_backend is split_backend
    # A launch without --rpc (the split was dropped, or another model loaded) frees it.
    run(ss.after_load(_FakeBackend(4243, str(model)), 4))
    assert ss.state().peer_process is None and ss.state().topology == "single"
    run(ss.shutdown())


def test_a_failed_load_stops_what_the_pre_load_step_started(cluster, monkeypatch, tmp_path):
    cluster.topology = "layer_split"
    model = tmp_path / "big.gguf"
    model.write_bytes(b"x")
    _calls, started = _patch_remote(monkeypatch)
    run(ss.before_load(_FakeRequest(str(model)), 4))
    assert ss.state().topology == "layer_split" and started
    run(ss.load_failed())
    assert ss.state().topology == "single" and ss.state().peer_process is None
    # Nothing resident after the load reads the same way.
    run(ss.before_load(_FakeRequest(str(model)), 4))
    assert len(started) == 2
    run(ss.after_load(SimpleNamespace(is_loaded = False), 4))
    assert ss.state().peer_process is None


def test_after_load_keeps_replicas_across_a_no_op_reload(cluster, monkeypatch, tmp_path):
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
            router = ss.state().router
            assert router is not None and len(started) == 1
            # The same server again (the load was skipped): nothing restarts.
            await ss.after_load(backend, 16)
            assert ss.state().router is router and len(started) == 1
            # A real reload lands on a new port: the replica follows it.
            backend._port = fake.port  # same fake, but a new process identity
            backend._process = SimpleNamespace(args = list(backend._process.args))
            ss.state().attached_port = fake.port + 1
            await ss.after_load(backend, 16)
            assert ss.state().router is not router and len(started) == 2
        finally:
            await ss.shutdown()
            await fake.stop()

    run(scenario())


def test_peer_process_never_shows_the_api_key():
    argv = ["/b/llama-server", "-m", "/m.gguf", "--api-key", "sk-secret", "--port", "1"]
    process = ss.PeerProcess("llama-server", "192.168.200.13", argv)
    assert "sk-secret" in process.remote_command
    assert "sk-secret" not in process.redacted_command
    assert "sk-secret" not in process.snapshot()["command"]
    assert ss.redacted_argv(["--api-key=sk-secret"]) == ["--api-key=<redacted>"]


def test_kv_estimate_reads_the_gguf_header_and_scales_with_cache_type(tmp_path):
    gguf = pytest.importorskip("gguf")
    path = tmp_path / "tiny.gguf"
    writer = gguf.GGUFWriter(str(path), "llama")
    writer.add_block_count(2)
    writer.add_head_count(4)
    writer.add_head_count_kv(2)
    writer.add_embedding_length(64)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    # 2 (K and V) x 2 layers x 1024 tokens x 2 kv heads x 16 head dim x bytes per element.
    assert ss.estimate_kv_bytes(str(path), 1024) == 2 * 2 * 1024 * 2 * 16 * 2
    assert ss.estimate_kv_bytes(str(path), 1024, "q8_0") == int(2 * 2 * 1024 * 2 * 16 * (34 / 32))
    assert ss.estimate_kv_bytes(str(path), 0) is None
    assert ss.estimate_kv_bytes(str(tmp_path / "missing.gguf"), 1024) is None
    assert ss.kv_bytes_per_elem("q4_0") == 18 / 32 and ss.kv_bytes_per_elem(None) == 2.0


def test_ssh_user_defers_to_the_cluster_module(cluster, monkeypatch):
    """One rule for the login on both sides of the ssh: spark_cluster's."""
    monkeypatch.setenv("USER", "alice")
    cluster._ssh_user = lambda: "bob"
    assert ss._ssh_user() == "bob"
    assert ss.ssh_argv("192.168.200.13", "true")[-2] == "bob@192.168.200.13"
    del cluster._ssh_user
    assert ss._ssh_user() == "alice"


# ── Pipeline groups: the --help probe and the layer-split launch ─────────────


def test_llama_server_supports_probes_help_once_per_binary_and_mtime(cluster, tmp_path):
    script = write_fake_llama_server(cluster.bundle / "build" / "bin", _FAKE_HELP_WITH_FLAG)
    assert ss.llama_server_binary() == str(script)
    assert ss.llama_server_supports("--pipeline-groups") is True
    assert ss.llama_server_supports("--parallel") is True
    assert ss.llama_server_supports("--pipeline") is False, "a prefix is not the flag"
    assert ss.llama_server_supports("--no-such-flag") is False
    assert probe_runs(script) == 1, "one --help run answers every flag"
    # A reinstall at the same path is a new mtime and is probed again.
    write_fake_llama_server(cluster.bundle / "build" / "bin", _FAKE_HELP_WITHOUT_FLAG)
    os.utime(script, (time.time() + 5, time.time() + 5))
    assert ss.llama_server_supports("--pipeline-groups") is False
    assert probe_runs(script) == 2


def test_llama_server_supports_is_false_without_the_flag_or_binary(cluster, tmp_path):
    assert ss.llama_server_binary() is None
    assert ss.llama_server_supports("--pipeline-groups") is False, "no binary in the bundle"
    assert ss.llama_server_supports("--pipeline-groups", str(tmp_path / "missing")) is False
    script = write_fake_llama_server(cluster.bundle / "bin", _FAKE_HELP_WITHOUT_FLAG)
    assert ss.llama_server_binary() == str(script)
    assert ss.llama_server_supports("--pipeline-groups") is False
    assert ss.llama_server_supports("", str(script)) is False
    # A binary that dies before printing anything is a binary without the flag.
    crashing = write_fake_llama_server(tmp_path / "crash", _FAKE_HELP_WITH_FLAG, body = "exit 3\n")
    assert ss.llama_server_supports("--pipeline-groups", str(crashing)) is False
    # So is one that is not executable at all; nothing raises.
    dud = tmp_path / "dud" / "llama-server"
    dud.parent.mkdir()
    dud.write_text("not a program")
    assert ss.llama_server_supports("--pipeline-groups", str(dud)) is False


def test_llama_server_supports_treats_a_hang_as_no_flag(cluster, monkeypatch, tmp_path):
    script = write_fake_llama_server(tmp_path / "slow", _FAKE_HELP_WITH_FLAG, body = "sleep 5\n")
    monkeypatch.setattr(ss, "HELP_PROBE_TIMEOUT_S", 0.3)
    started = time.monotonic()
    assert ss.llama_server_supports("--pipeline-groups", str(script)) is False
    assert time.monotonic() - started < 3.0
    assert ss.llama_server_supports("--pipeline-groups", str(script)) is False
    assert probe_runs(script) == 1, "the timeout is cached; one stall per build"


def test_pipeline_groups_plan_gives_every_group_a_slot(cluster, monkeypatch):
    write_fake_llama_server(cluster.bundle / "build" / "bin", _FAKE_HELP_WITH_FLAG)
    for asked, slots in ((1, 2), (2, 2), (3, 4), (4, 4), (5, 6), (16, 16)):
        plan = ss.pipeline_groups_plan(asked)
        assert plan["pipeline_groups"] == 2 and plan["slots"] == slots, (asked, plan)
        assert plan["slots"] % 2 == 0 and plan["slots"] >= 2
        assert plan["requested_slots"] == asked and plan["reason"] is None
    # A pass-through slot count wins over the request's, as it does in the launch.
    plan = ss.pipeline_groups_plan(8, ["--seed", "1", "-np", "3"])
    assert plan["requested_slots"] == 3 and plan["slots"] == 4
    assert ss.pipeline_groups_plan(8, ["--parallel=5"])["slots"] == 6
    # N groups need a multiple of N.
    monkeypatch.setenv(ss.ENV_PIPELINE_GROUPS, "3")
    plan = ss.pipeline_groups_plan(4)
    assert plan["pipeline_groups"] == 3 and plan["slots"] == 6
    assert ss.pipeline_groups_plan(1)["slots"] == 3
    # 0 and 1 disable; garbage disables and says so; none of these run the probe.
    for value in ("0", "1", "two"):
        monkeypatch.setenv(ss.ENV_PIPELINE_GROUPS, value)
        plan = ss.pipeline_groups_plan(4)
        assert plan["pipeline_groups"] == 0 and plan["slots"] == 4, (value, plan)
        assert ss.ENV_PIPELINE_GROUPS in plan["reason"], (value, plan)


def test_pipeline_groups_plan_is_off_when_the_bundle_lacks_the_flag(cluster):
    write_fake_llama_server(cluster.bundle / "build" / "bin", _FAKE_HELP_WITHOUT_FLAG)
    plan = ss.pipeline_groups_plan(3)
    assert plan == {
        "pipeline_groups": 0,
        "reason": "bundle llama-server lacks --pipeline-groups",
        "slots": 3,
        "requested_slots": 3,
    }


def test_before_load_adds_pipeline_groups_when_the_bundle_has_the_flag(
    cluster, monkeypatch, tmp_path
):
    cluster.topology = "layer_split"
    script = write_fake_llama_server(cluster.bundle / "build" / "bin", _FAKE_HELP_WITH_FLAG)
    model = tmp_path / "big.gguf"
    model.write_bytes(b"x")
    _calls, started = _patch_remote(monkeypatch)
    out = run(ss.before_load(_FakeRequest(str(model), llama_extra_args = ["--seed", "1"]), 3))
    assert out.llama_extra_args == [
        "--seed",
        "1",
        "--rpc",
        "192.168.200.13:50052",
        "--device",
        "CUDA0,RPC0",
        "-sm",
        "layer",
        "--pipeline-groups",
        "2",
        "--parallel",
        "4",
    ], "three slots asked for; two groups need an even count"
    assert started and started[0].name == "ggml-rpc-server"
    status = ss.status()
    assert status["topology"] == "layer_split"
    assert status["pipeline_groups"] == 2 and status["pipeline_groups_reason"] is None
    # A second load reuses the rpc-server and the cached probe: still two groups.
    out = run(ss.before_load(_FakeRequest(str(model)), 4))
    assert out.llama_extra_args[-4:] == ["--pipeline-groups", "2", "--parallel", "4"]
    assert probe_runs(script) == 1
    run(ss.shutdown())
    status = ss.status()
    assert status["pipeline_groups"] == 0
    assert "not a layer split" in status["pipeline_groups_reason"]


def test_before_load_launches_as_before_without_the_flag(cluster, monkeypatch, tmp_path):
    cluster.topology = "layer_split"
    write_fake_llama_server(cluster.bundle / "build" / "bin", _FAKE_HELP_WITHOUT_FLAG)
    model = tmp_path / "big.gguf"
    model.write_bytes(b"x")
    _patch_remote(monkeypatch)
    out = run(ss.before_load(_FakeRequest(str(model)), 3))
    assert out.llama_extra_args == [
        "--rpc",
        "192.168.200.13:50052",
        "--device",
        "CUDA0,RPC0",
        "-sm",
        "layer",
    ], "no --pipeline-groups and no --parallel override on a bundle without the flag"
    status = ss.status()
    assert status["topology"] == "layer_split" and status["pipeline_groups"] == 0
    assert status["pipeline_groups_reason"] == "bundle llama-server lacks --pipeline-groups"
    run(ss.shutdown())


def test_pipeline_groups_env_override_disables_or_sets_n(cluster, monkeypatch, tmp_path):
    cluster.topology = "layer_split"
    script = write_fake_llama_server(cluster.bundle / "build" / "bin", _FAKE_HELP_WITH_FLAG)
    model = tmp_path / "big.gguf"
    model.write_bytes(b"x")
    _patch_remote(monkeypatch)
    monkeypatch.setenv(ss.ENV_PIPELINE_GROUPS, "0")
    out = run(ss.before_load(_FakeRequest(str(model)), 3))
    assert "--pipeline-groups" not in out.llama_extra_args
    assert "--parallel" not in out.llama_extra_args
    assert probe_runs(script) == 0, "disabled by env: the binary is never run"
    status = ss.status()
    assert status["pipeline_groups"] == 0
    assert status["pipeline_groups_reason"] == f"disabled by {ss.ENV_PIPELINE_GROUPS}=0"
    run(ss.shutdown())
    monkeypatch.setenv(ss.ENV_PIPELINE_GROUPS, "3")
    out = run(ss.before_load(_FakeRequest(str(model)), 4))
    assert out.llama_extra_args[-4:] == ["--pipeline-groups", "3", "--parallel", "6"]
    assert ss.status()["pipeline_groups"] == 3
    run(ss.shutdown())


def test_pipeline_groups_never_run_for_single_or_replicas(cluster, monkeypatch, tmp_path):
    script = write_fake_llama_server(cluster.bundle / "build" / "bin", _FAKE_HELP_WITH_FLAG)
    model = tmp_path / "m.gguf"
    model.write_bytes(b"x")
    _patch_remote(monkeypatch)
    for topology in ("single", "replicas"):
        cluster.topology = topology
        request = _FakeRequest(str(model))
        out = run(ss.before_load(request, 16))
        assert out is request and out.llama_extra_args is None
        status = ss.status()
        assert status["pipeline_groups"] == 0
        assert status["pipeline_groups_reason"] == "not a layer split (topology single)"
    assert probe_runs(script) == 0, "the probe belongs to the layer split alone"
    # Off a Spark the payload is the fixed refusal, with no probe and no new field.
    cluster.spark = False
    assert ss.status() == {
        "enabled": False,
        "topology": None,
        "reason": "not a paired DGX Spark",
    }
    assert probe_runs(script) == 0
