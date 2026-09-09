# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The two-Spark serving orchestrator, off any real hardware. ``spark_cluster`` is a stub, so
these pin the contract the module has with it and the gate that keeps all of it off a normal
machine; ssh never runs."""

from __future__ import annotations

import asyncio
import os
import shutil
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

    rpc_binary: Optional[str] = None

    def rpc_server_binary(self) -> Optional[str]:
        return self.rpc_binary or str(
            Path.home() / ".unsloth" / "llama.cpp" / "build" / "bin" / "ggml-rpc-server"
        )

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
    # An empty bundle, so the test machine never decides whether a split gets groups.
    stub.bundle = tmp_path / "bundle"
    stub.bundle.mkdir()
    (tmp_path / "home").mkdir(exist_ok = True)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    _which = shutil.which
    monkeypatch.setattr(
        shutil,
        "which",
        lambda cmd, *a, **k: None if "llama-server" in str(cmd) else _which(cmd, *a, **k),
    )
    monkeypatch.setenv("UNSLOTH_LLAMA_CPP_PATH", str(stub.bundle))
    monkeypatch.delenv("UNSLOTH_STUDIO_MANAGED_LLAMA_CPP_PATH", raising = False)
    monkeypatch.delenv("LLAMA_SERVER_PATH", raising = False)
    ss.reset_for_tests()
    monkeypatch.setattr(ss, "_CLUSTER", stub)
    monkeypatch.setattr(ss, "_CLUSTER_LOOKED_UP", True)
    monkeypatch.delenv(ss.ENV_TOGGLE, raising = False)
    monkeypatch.delenv(ss.ENV_TOPOLOGY, raising = False)
    monkeypatch.delenv(ss.ENV_PEER, raising = False)
    monkeypatch.delenv(ss.ENV_PIPELINE_GROUPS, raising = False)
    monkeypatch.delenv(ss.ENV_MTP, raising = False)
    yield stub
    ss.reset_for_tests()


_FAKE_HELP_WITH_FLAG = """usage: llama-server [options]
  -np, --parallel N            number of server slots (default: 4)
  --pipeline-groups N          number of pipeline groups the slots are split over
  --spec-type none,draft-simple,draft-mtp,ngram-simple
  --spec-draft-n-max N         number of tokens to draft for speculative decoding (default: 3)
"""
_FAKE_HELP_WITHOUT_FLAG = """usage: llama-server [options]
  -np, --parallel N            number of server slots (default: 4)
  --kv-unified                 one KV buffer shared by all slots
"""
_FAKE_HELP_SPEC_ONLY = """usage: llama-server [options]
  -np, --parallel N            number of server slots (default: 4)
  --spec-type none,draft-simple,draft-mtp,ngram-simple
  --spec-draft-n-max N         number of tokens to draft for speculative decoding (default: 3)
"""


def write_fake_llama_server(
    directory: Path,
    help_text: str,
    *,
    body: str = "",
    hidden_flags: tuple = (),
    refuses_groups_with_drafter: bool = False,
) -> Path:
    """A stand-in for the real parser and the real load path. ``hidden_flags`` is a flag the
    fork strips from argv and never prints; ``refuses_groups_with_drafter`` is the fork before
    PR #187, which -- like the real one -- refuses inside ``load_model`` and NOT at ``--help``,
    which is why the usage text cannot probe it."""
    directory.mkdir(parents = True, exist_ok = True)
    script = directory / "llama-server"
    reject = "--pipeline-groups" not in help_text and "--pipeline-groups" not in hidden_flags
    check = (
        'for a in "$@"; do case "$a" in --pipeline-groups|--pipeline-groups=*) '
        'echo "error: invalid argument: $a" >&2; exit 1;; esac; done\n'
        if reject
        else ""
    )
    spec_known = "--spec-type" in help_text or "--spec-type" in hidden_flags
    # The load path: only reached when a model is named, i.e. never by a --help probe.
    load = (
        'g=; s=; m=; n=; for a in "$@"; do\n'
        '  if [ -n "$n" ]; then m="$a"; n=; continue; fi\n'
        '  case "$a" in --pipeline-groups*) g=1;; --spec-type*|--model-draft*|-md) s=1;; '
        "-m|--model) n=1;; esac\n"
        "done\n"
        'if [ -n "$m" ]; then\n'
    )
    if refuses_groups_with_drafter:
        load += (
            '  if [ -n "$g" ] && [ -n "$s" ]; then echo "error: --pipeline-groups > 1 is not '
            'supported together with speculative decoding (--model-draft / MTP)" >&2; exit 1; fi\n'
        )
    load += (
        '  echo "error: llama_model_loader: failed to load model from $m" >&2\n' "  exit 1\n" "fi\n"
    )
    if not spec_known:
        # A build with no --spec-type at all stops at the common parser, as llama.cpp does.
        load = (
            'for a in "$@"; do case "$a" in --spec-type|--spec-type=*) '
            'echo "error: invalid argument: $a" >&2; exit 1;; esac; done\n'
        ) + load
    script.write_text(
        "#!/bin/sh\n"
        'echo run >> "$0.calls"\n' + check + load + body + "cat <<'EOF'\n" + help_text + "EOF\n",
        encoding = "utf-8",
    )
    script.chmod(0o755)
    return script


def write_gguf(path: Path, arch: str, **uint32_keys: int) -> Path:
    """A minimal GGUF: the architecture plus the given ``<key>: value`` uint32 fields."""
    gguf = pytest.importorskip("gguf")
    writer = gguf.GGUFWriter(str(path), arch)
    for key, value in uint32_keys.items():
        writer.add_uint32(key, int(value))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return path


def probe_runs(script: Path) -> int:
    calls = Path(str(script) + ".calls")
    return len(calls.read_text().splitlines()) if calls.exists() else 0


def run(coro):
    return asyncio.run(coro)


def test_ssh_user_is_this_login_and_never_a_fixed_one(monkeypatch):
    import getpass

    monkeypatch.setenv("USER", "alice")
    assert ss._ssh_user() == "alice"
    for var in ("USER", "USERNAME", "LOGNAME"):
        monkeypatch.delenv(var, raising = False)
    assert ss._ssh_user() == getpass.getuser()
    assert ss.ssh_argv("192.168.200.13", "true")[-2] == f"{getpass.getuser()}@192.168.200.13"


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


def test_plan_topology_hands_the_shared_planner_bytes_users_and_budget(cluster):
    cluster.topology = "replicas"
    out = ss.plan_topology(16 * GIB, users = 16, kv_bytes_per_user = 0.25 * GIB)
    assert out["topology"] == "replicas" and out["reason"] == "stub says replicas"
    call = cluster.planner_calls[-1]
    assert call["model_bytes"] == 16 * GIB and call["kv_bytes_per_user"] == 0.25 * GIB
    assert call["users"] == 16 and call["prompt_tokens"] == ss.PROMPT_TOKENS_DEFAULT
    assert call["per_node_free_bytes"] == pytest.approx((121.69 - 8.0) * GIB)
    assert call["prefill_heavy"] is False
    assert ss.plan_topology(None, users = 32)["topology"] == "single"
    assert len(cluster.planner_calls) == 1
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
        "--spec-type",
        "draft-mtp",
        "--spec-draft-n-max",
        "3",
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
        "--spec-type",
        "draft-mtp",
        "--spec-draft-n-max",
        "3",
        "--host",
        "192.168.200.13",
        "--port",
        "41234",
    ], "the peer runs the same speculation as this node"


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
    assert extra == [
        "--rpc",
        "192.168.200.13:50052",
        "--device",
        "RPC0,CUDA0",
        "-sm",
        "layer",
        "--tensor-split",
        "0.5,0.5",
        "--cache-ram",
        "0",
    ]
    assert not any("pipeline" in a for a in extra), "no groups asked for: today's launch"
    grouped = ss.layer_split_extra_args("192.168.200.13", 50052, pipeline_groups = 2)
    assert grouped == [
        "--rpc",
        "192.168.200.13:50052",
        "--device",
        "RPC0,CUDA0",
        "-sm",
        "layer",
        "--tensor-split",
        "0.5,0.5",
        "--cache-ram",
        "0",
        "--pipeline-groups",
        "2",
    ]
    assert not any(
        a in ("-np", "--parallel") for a in grouped
    ), "--parallel in the pass-through is refused by llama_server_args and fails the load"
    assert ss.layer_split_extra_args("p", 1, pipeline_groups = 1) == extra[:0] + [
        "--rpc",
        "p:1",
        "--device",
        "RPC0,CUDA0",
        "-sm",
        "layer",
        "--tensor-split",
        "0.5,0.5",
        "--cache-ram",
        "0",
    ]


def test_peer_binary_candidates_prefer_the_local_bundle_path(cluster):
    local = str(Path.home() / ".unsloth" / "llama.cpp" / "build" / "bin" / "llama-server")
    candidates = ss.peer_binary_candidates(local, "llama-server")
    assert candidates[0] == "$HOME/.unsloth/llama.cpp/build/bin/llama-server"
    assert candidates[-1] == "llama-server"
    assert len(candidates) == len(set(candidates))
    script = ss.find_binary_script(candidates)
    assert script.endswith("echo MISSING; exit 1") and "command -v llama-server" in script


def test_peer_process_remote_command_prints_the_servers_pid_and_reaps_it():
    """A Studio killed without its shutdown path used to leave the peer's rpc-server
    holding the peer's GPU."""
    process = ss.PeerProcess(
        "llama-server", "192.168.200.13", ["/b/llama-server", "-m", "/p/a b.gguf"]
    )
    command = process.remote_command
    assert command.startswith("/b/llama-server -m '/p/a b.gguf' & srv=$!")
    assert "echo UNSLOTH_SPARK_PID=$srv" in command, "the reported pid is the server's own"
    assert "watch=$PPID" in command and 'kill -0 "$watch"' in command
    assert 'kill "$srv"' in command and 'kill -9 "$srv"' in command
    assert 'wait "$srv"' in command, "a server that exits on its own still reports its status"
    for statement in command.split(";"):
        if "kill " in statement and "kill -0" not in statement:
            assert (
                '"$srv"' in statement
            ), f"the reaper kills something it did not start: {statement}"
    assert "pkill" not in command and "pgrep" not in command and "killall" not in command
    # A read on the ssh channel never returns on a half-open socket, so it polls.
    assert "read" not in command
    assert f"sleep {ss.PEER_REAP_POLL_S}" in command
    assert not process.alive
    snap = process.snapshot()
    assert snap["remote_pid"] is None and snap["alive"] is False


def test_the_ssh_client_carrying_the_peer_cannot_outlive_this_process():
    """Without PR_SET_PDEATHSIG the ssh client is reparented to init when Studio is killed,
    keeps the remote session open, and the peer-side watch never fires."""
    import inspect

    source = inspect.getsource(ss.PeerProcess.start)
    assert "preexec_fn = _die_with_parent" in source
    assert "PR_SET_PDEATHSIG" in inspect.getsource(ss._die_with_parent)
    ss._die_with_parent()


class _FakeRequest:
    def __init__(self, model_path: str, **kw):
        self.model_path = model_path
        self.gguf_variant = kw.get("gguf_variant")
        self.max_seq_length = kw.get("max_seq_length", 0)
        self.cache_type_kv = kw.get("cache_type_kv")
        self.llama_extra_args = kw.get("llama_extra_args")
        self.speculative_type = kw.get("speculative_type")
        self.spec_draft_n_max = kw.get("spec_draft_n_max")
        self.n_parallel = kw.get("n_parallel")
        self.disable_vision = bool(kw.get("disable_vision", False))
        self.force_reload = bool(kw.get("force_reload", False))

    def model_copy(self, update):
        clone = _FakeRequest(
            self.model_path,
            gguf_variant = self.gguf_variant,
            max_seq_length = self.max_seq_length,
            cache_type_kv = self.cache_type_kv,
            llama_extra_args = self.llama_extra_args,
            speculative_type = self.speculative_type,
            spec_draft_n_max = self.spec_draft_n_max,
            n_parallel = self.n_parallel,
            disable_vision = self.disable_vision,
            force_reload = self.force_reload,
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
        # As the real remote shell: the lookup exits at the first hit, so nothing after it runs.
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
        "RPC0,CUDA0",
        "-sm",
        "layer",
        "--tensor-split",
        "0.5,0.5",
        "--cache-ram",
        "0",
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
        argv_extra: Optional[List[str]] = None,
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
            + list(argv_extra or [])
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
            assert ss.route_base_url(backend) == state.router.base_url
            payload: Dict[str, Any] = {}
            ss.tag_conversation(payload, "thread-7")
            assert payload == {CONVERSATION_FIELD: "thread-7"}
            backend._port = fake.port + 1
            assert ss.route_base_url(backend) is None
            backend._port = fake.port
            status = ss.status()
            assert status["topology"] == "replicas" and status["router"]["healthy_backends"] >= 1
            assert [b["name"] for b in status["router"]["backends"]] == ["main", "peer"]
            assert ss.current_topology() == "replicas"
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


def test_launch_files_names_every_shard_not_just_the_one_in_argv(tmp_path):
    # llama-server takes the first shard and opens the rest itself, so a peer holding only
    # that one passes preflight and then fails the load.
    shards = [tmp_path / f"m-{i:05d}-of-00003.gguf" for i in (1, 2, 3)]
    for shard in shards:
        shard.write_bytes(b"w")
    argv = ["/b/llama-server", "-m", str(shards[0]), "-c", "4096"]
    assert ss.launch_files(argv, str(shards[0])) == [str(s) for s in shards]


def test_before_load_reuses_a_live_rpc_server_and_after_load_reconciles(
    cluster, monkeypatch, tmp_path
):
    cluster.topology = "layer_split"
    model = tmp_path / "big.gguf"
    model.write_bytes(b"x")
    _calls, started = _patch_remote(monkeypatch)
    first = run(ss.before_load(_FakeRequest(str(model)), 4))
    assert len(started) == 1 and "--rpc" in first.llama_extra_args
    again = run(ss.before_load(_FakeRequest(str(model)), 4))
    assert len(started) == 1 and "--rpc" in again.llama_extra_args
    assert ss.state().peer_process is started[0]
    split_backend = _FakeBackend(4242, str(model))
    split_backend._process.args += ["--rpc", "192.168.200.13:50052"]
    run(ss.after_load(split_backend, 4))
    assert ss.state().topology == "layer_split" and ss.state().peer_process is started[0]
    assert ss.state().attached_backend is split_backend
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
            await ss.after_load(backend, 16)
            assert ss.state().router is router and len(started) == 1
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
    monkeypatch.setenv("USER", "alice")
    cluster._ssh_user = lambda: "bob"
    assert ss._ssh_user() == "bob"
    assert ss.ssh_argv("192.168.200.13", "true")[-2] == "bob@192.168.200.13"
    del cluster._ssh_user
    assert ss._ssh_user() == "alice"


def test_llama_server_supports_probes_help_once_per_binary_and_mtime(cluster, tmp_path):
    script = write_fake_llama_server(cluster.bundle / "build" / "bin", _FAKE_HELP_WITH_FLAG)
    assert ss.llama_server_binary() == str(script)
    assert ss.llama_server_supports("--pipeline-groups") is True
    assert ss.llama_server_supports("--parallel") is True
    assert ss.llama_server_supports("--pipeline") is False, "a prefix is not the flag"
    assert ss.llama_server_supports("--no-such-flag") is False
    assert probe_runs(script) == 1, "one --help run answers every flag"
    write_fake_llama_server(cluster.bundle / "build" / "bin", _FAKE_HELP_WITHOUT_FLAG)
    os.utime(script, (time.time() + 5, time.time() + 5))
    assert ss.llama_server_supports("--pipeline-groups") is False
    assert probe_runs(script) == 2


def test_llama_server_supports_is_false_without_the_flag_or_binary(cluster, tmp_path):
    assert ss.llama_server_binary() is None
    assert ss.llama_server_supports("--pipeline-groups") is False, "no binary in the bundle"
    assert ss.llama_server_supports("--pipeline-groups", str(tmp_path / "missing")) is False
    write_fake_llama_server(cluster.bundle / "bin", _FAKE_HELP_WITH_FLAG)
    assert ss.llama_server_binary() is None
    assert ss.llama_server_supports("--pipeline-groups") is False
    script = write_fake_llama_server(cluster.bundle / "build" / "bin", _FAKE_HELP_WITHOUT_FLAG)
    assert ss.llama_server_binary() == str(script)
    assert ss.llama_server_supports("--pipeline-groups") is False
    assert ss.llama_server_supports("", str(script)) is False
    crashing = write_fake_llama_server(tmp_path / "crash", _FAKE_HELP_WITH_FLAG, body = "exit 3\n")
    assert ss.llama_server_supports("--pipeline-groups", str(crashing)) is False
    dud = tmp_path / "dud" / "llama-server"
    dud.parent.mkdir()
    dud.write_text("not a program")
    assert ss.llama_server_supports("--pipeline-groups", str(dud)) is False


def test_the_orchestrator_probes_the_binary_the_backend_launches(cluster, monkeypatch, tmp_path):
    """One resolver, or the orchestrator decides for a binary that never runs: it used to walk
    ``spark_cluster``'s layouts, which include ``<root>/bin`` where the backend's do not, so the
    two ends of the link ran different builds. This fails the moment they disagree again."""
    from core.inference.llama_cpp import LlamaCppBackend

    for parts in (("build", "bin"), (), ("bin",), ("build", "bin", "Release"), ("nowhere",)):
        root = tmp_path / ("layout_" + ("_".join(parts) or "flat"))
        write_fake_llama_server(root.joinpath(*parts), _FAKE_HELP_WITH_FLAG)
        monkeypatch.setenv("UNSLOTH_LLAMA_CPP_PATH", str(root))
        cluster.bundle = root
        ss.reset_for_tests()
        backend_choice = LlamaCppBackend._find_llama_server_binary()
        assert ss.llama_server_binary() == backend_choice, (
            f"layout {parts or ('flat',)}: the orchestrator would probe "
            f"{ss.llama_server_binary()} while the backend launches {backend_choice}"
        )
    # And not vacuously equal: without this the loop passes with both resolvers finding nothing.
    monkeypatch.setenv("UNSLOTH_LLAMA_CPP_PATH", str(tmp_path / "layout_build_bin"))
    ss.reset_for_tests()
    assert ss.llama_server_binary() == str(tmp_path / "layout_build_bin/build/bin/llama-server")
    monkeypatch.setenv("UNSLOTH_LLAMA_CPP_PATH", str(tmp_path / "layout_bin"))
    ss.reset_for_tests()
    assert ss.llama_server_binary() is None


def test_the_rpc_server_is_taken_from_beside_the_launched_llama_server(cluster, tmp_path):
    """The peer's copy comes from beside the llama-server this node launches."""
    launched = write_fake_llama_server(cluster.bundle / "build" / "bin", _FAKE_HELP_WITH_FLAG)
    beside = launched.parent / "ggml-rpc-server"
    beside.write_text("#!/bin/sh\nexit 0\n")
    beside.chmod(0o755)
    other = tmp_path / "other" / "ggml-rpc-server"
    other.parent.mkdir(parents = True, exist_ok = True)
    other.write_text("#!/bin/sh\nexit 0\n")
    other.chmod(0o755)
    cluster.rpc_binary = str(other)
    assert ss.rpc_server_binary() == str(beside)
    assert ss.peer_binary_candidates(ss.rpc_server_binary(), "ggml-rpc-server")[0].endswith(
        "/build/bin/ggml-rpc-server"
    )
    beside.unlink()
    assert ss.rpc_server_binary() == str(other)


def test_llama_server_supports_treats_a_hang_as_no_flag(cluster, monkeypatch, tmp_path):
    script = write_fake_llama_server(tmp_path / "slow", _FAKE_HELP_WITH_FLAG, body = "sleep 5\n")
    monkeypatch.setattr(ss, "HELP_PROBE_TIMEOUT_S", 0.3)
    started = time.monotonic()
    assert ss.llama_server_supports("--pipeline-groups", str(script)) is False
    assert time.monotonic() - started < 3.0
    assert ss.llama_server_supports("--pipeline-groups", str(script)) is False
    assert probe_runs(script) == 1, "the timeout is cached; one stall per build"


def test_pipeline_groups_default_is_two_when_the_bundle_has_the_flag(cluster, monkeypatch):
    script = write_fake_llama_server(cluster.bundle / "build" / "bin", _FAKE_HELP_WITH_FLAG)
    monkeypatch.delenv(ss.ENV_PIPELINE_GROUPS, raising = False)
    assert ss.PIPELINE_GROUPS_DEFAULT == 2
    plan = ss.pipeline_groups_plan(3)
    assert plan["pipeline_groups"] == 2 and plan["slots"] == 4 and plan["reason"] is None
    assert probe_runs(script) == 1


def test_pipeline_groups_plan_gives_every_group_a_slot(cluster, monkeypatch):
    write_fake_llama_server(cluster.bundle / "build" / "bin", _FAKE_HELP_WITH_FLAG)
    monkeypatch.setenv(ss.ENV_PIPELINE_GROUPS, "2")
    for asked, slots in ((1, 2), (2, 2), (3, 4), (4, 4), (5, 6), (16, 16)):
        plan = ss.pipeline_groups_plan(asked)
        assert plan["pipeline_groups"] == 2 and plan["slots"] == slots, (asked, plan)
        assert plan["slots"] % 2 == 0 and plan["slots"] >= 2
        assert plan["requested_slots"] == asked and plan["reason"] is None
    plan = ss.pipeline_groups_plan(8, ["--seed", "1", "-np", "3"])
    assert plan["requested_slots"] == 3 and plan["slots"] == 4
    assert ss.pipeline_groups_plan(8, ["--parallel=5"])["slots"] == 6
    monkeypatch.setenv(ss.ENV_PIPELINE_GROUPS, "3")
    plan = ss.pipeline_groups_plan(4)
    assert plan["pipeline_groups"] == 3 and plan["slots"] == 6
    assert ss.pipeline_groups_plan(1)["slots"] == 3
    for value in ("0", "1", "two"):
        monkeypatch.setenv(ss.ENV_PIPELINE_GROUPS, value)
        plan = ss.pipeline_groups_plan(4)
        assert plan["pipeline_groups"] == 0 and plan["slots"] == 4, (value, plan)
        assert ss.ENV_PIPELINE_GROUPS in plan["reason"], (value, plan)


def test_pipeline_groups_plan_is_off_when_the_bundle_lacks_the_flag(cluster, monkeypatch):
    write_fake_llama_server(cluster.bundle / "build" / "bin", _FAKE_HELP_WITHOUT_FLAG)
    monkeypatch.setenv(ss.ENV_PIPELINE_GROUPS, "2")
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
    monkeypatch.setenv(ss.ENV_PIPELINE_GROUPS, "2")
    out = run(ss.before_load(_FakeRequest(str(model), llama_extra_args = ["--seed", "1"]), 3))
    assert out.llama_extra_args == [
        "--seed",
        "1",
        "--rpc",
        "192.168.200.13:50052",
        "--device",
        "RPC0,CUDA0",
        "-sm",
        "layer",
        "--tensor-split",
        "0.5,0.5",
        "--cache-ram",
        "0",
        "--pipeline-groups",
        "2",
    ], "three slots asked for; two groups need an even count"
    assert out.n_parallel == 4, "two groups need an even slot count, on the request field"
    assert started and started[0].name == "ggml-rpc-server"
    status = ss.status()
    assert status["topology"] == "layer_split"
    assert status["pipeline_groups"] == 2 and status["pipeline_groups_reason"] is None
    out = run(ss.before_load(_FakeRequest(str(model)), 4))
    assert out.llama_extra_args[-2:] == ["--pipeline-groups", "2"]
    assert out.n_parallel == 4, "the slot count travels as the request field, not as argv"
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
    monkeypatch.setenv(ss.ENV_PIPELINE_GROUPS, "2")
    out = run(ss.before_load(_FakeRequest(str(model)), 3))
    assert out.llama_extra_args == [
        "--rpc",
        "192.168.200.13:50052",
        "--device",
        "RPC0,CUDA0",
        "-sm",
        "layer",
        "--tensor-split",
        "0.5,0.5",
        "--cache-ram",
        "0",
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
    assert out.llama_extra_args[-2:] == ["--pipeline-groups", "3"]
    assert out.n_parallel == 6, "the slot count travels as the request field, not as argv"
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
    cluster.spark = False
    assert ss.status() == {
        "enabled": False,
        "topology": None,
        "reason": "not a paired DGX Spark",
    }
    assert probe_runs(script) == 0


def test_llama_server_accepts_finds_a_flag_the_usage_hides(cluster, tmp_path):
    """The --help text alone says every fork build lacks a flag the fork hides."""
    hidden = write_fake_llama_server(
        cluster.bundle / "build" / "bin",
        _FAKE_HELP_SPEC_ONLY,
        hidden_flags = ("--pipeline-groups",),
    )
    assert ss.llama_server_supports("--pipeline-groups") is False
    assert ss.llama_server_accepts("--pipeline-groups") is True
    assert ss.llama_server_accepts("--pipeline-groups") is True
    assert probe_runs(hidden) == 2, "one --help run plus one acceptance run, both cached"
    plan = ss.pipeline_groups_plan(3)
    assert plan["pipeline_groups"] == 2 and plan["slots"] == 4 and plan["reason"] is None
    assert probe_runs(hidden) == 2
    plain = write_fake_llama_server(tmp_path / "plain", _FAKE_HELP_WITHOUT_FLAG)
    assert ss.llama_server_accepts("--pipeline-groups", binary = str(plain)) is False
    assert ss.llama_server_accepts("--pipeline-groups", binary = str(plain)) is False
    assert probe_runs(plain) == 1
    assert ss.llama_server_accepts("--pipeline-groups", binary = str(tmp_path / "none")) is False
    assert ss.llama_server_accepts("", binary = str(plain)) is False


def test_gguf_nextn_predict_layers_reads_the_header_and_never_raises(tmp_path):
    head = write_gguf(tmp_path / "mtp.gguf", "qwen35", **{"qwen35.nextn_predict_layers": 1})
    assert ss.gguf_nextn_predict_layers(str(head)) == 1
    assert ss.gguf_has_mtp_head(str(head)) is True
    plain = write_gguf(tmp_path / "plain.gguf", "qwen35moe", **{"qwen35moe.block_count": 4})
    assert ss.gguf_nextn_predict_layers(str(plain)) is None
    assert ss.gguf_has_mtp_head(str(plain)) is False
    zero = write_gguf(tmp_path / "zero.gguf", "qwen35", **{"qwen35.nextn_predict_layers": 0})
    assert ss.gguf_nextn_predict_layers(str(zero)) == 0
    assert ss.gguf_has_mtp_head(str(zero)) is False
    other = write_gguf(tmp_path / "other.gguf", "llama", **{"qwen35.nextn_predict_layers": 1})
    assert ss.gguf_has_mtp_head(str(other)) is False
    # A split file keeps its header in the first shard; any shard of it answers.
    write_gguf(tmp_path / "big-00001-of-00003.gguf", "qwen35", **{"qwen35.nextn_predict_layers": 1})
    (tmp_path / "big-00003-of-00003.gguf").write_bytes(b"not a header")
    assert ss.gguf_has_mtp_head(str(tmp_path / "big-00003-of-00003.gguf")) is True
    assert ss.gguf_has_mtp_head(str(tmp_path / "big-00002-of-00003.gguf")) is True
    junk = tmp_path / "junk.gguf"
    junk.write_bytes(b"x" * 64)
    assert ss.gguf_has_mtp_head(str(junk)) is False
    assert ss.gguf_has_mtp_head(str(tmp_path / "missing.gguf")) is False
    assert ss.gguf_has_mtp_head(None) is False and ss.gguf_has_mtp_head("") is False
    assert ss.gguf_has_mtp_head(str(tmp_path)) is False


def test_mtp_plan_verdicts(cluster, monkeypatch, tmp_path):
    head = str(write_gguf(tmp_path / "mtp.gguf", "qwen35", **{"qwen35.nextn_predict_layers": 1}))
    plain = str(write_gguf(tmp_path / "plain.gguf", "qwen35moe"))
    for path in (None, "", str(tmp_path / "missing.gguf")):
        plan = ss.mtp_plan(path)
        assert plan["mtp"] == "unknown" and plan["request"] == {}, plan
    # No head: settled by the header, the binary is never run.
    plan = ss.mtp_plan(plain)
    assert plan["mtp"] == "no head" and plan["request"] == {}
    assert ss.llama_server_binary() is None
    assert ss.mtp_plan(head)["mtp"] == "server too old"
    script = write_fake_llama_server(cluster.bundle / "build" / "bin", _FAKE_HELP_WITHOUT_FLAG)
    plan = ss.mtp_plan(head)
    assert plan["mtp"] == "server too old" and plan["request"] == {}
    assert plan["reason"] == "bundle llama-server lacks --spec-type"
    assert probe_runs(script) == 1
    write_fake_llama_server(cluster.bundle / "build" / "bin", _FAKE_HELP_SPEC_ONLY)
    os.utime(script, (time.time() + 5, time.time() + 5))
    plan = ss.mtp_plan(head)
    assert plan["mtp"] == "enabled"
    assert plan["request"] == {"spec_draft_n_max": 3}
    assert "1 MTP layer(s)" in plan["reason"] and "--spec-draft-n-max 3" in plan["reason"]
    for rows, depth in ((None, 3), (1, 3), (8, 3), (31, 3), (32, 2), (64, 1), (128, 1)):
        plan = ss.mtp_plan(head, users = rows)
        assert plan["request"] == {"spec_draft_n_max": depth}, rows
        assert f"--spec-draft-n-max {depth}" in plan["reason"], rows
        if depth != 3:
            assert f"measured best at {rows} rows" in plan["reason"], rows
    plan = ss.mtp_plan(head, spec_draft_n_max = 3, users = 64)
    assert plan["mtp"] == "user override" and plan["request"] == {}
    assert ss.mtp_plan(head, ["--seed", "1", "-np", "4"])["mtp"] == "enabled"
    for mode in (None, "auto", "default", "AUTO"):
        assert ss.mtp_plan(head, speculative_type = mode)["mtp"] == "enabled", mode
    for extras in (
        ["--spec-type", "ngram-simple"],
        ["--spec-type=draft-mtp"],
        ["--spec-default"],
        ["--model-draft", "/d.gguf"],
        ["-md", "/d.gguf"],
        ["--spec-draft-n-max=8"],
        ["--spec-draft-model", "/d.gguf"],
        ["--draft-max", "8"],
        ["--draft", "4"],
        ["-hfd", "unsloth/x"],
    ):
        plan = ss.mtp_plan(head, extras)
        assert plan["mtp"] == "user override" and plan["request"] == {}, extras
        assert extras[0].partition("=")[0] in plan["reason"], (extras, plan)
    for mode in ("off", "mtp", "ngram", "dflash", "draft-mtp", "none"):
        plan = ss.mtp_plan(head, speculative_type = mode)
        assert plan["mtp"] == "user override" and plan["request"] == {}, mode
    plan = ss.mtp_plan(head, spec_draft_n_max = 2)
    assert plan["mtp"] == "user override" and plan["reason"] == "spec_draft_n_max=2"
    monkeypatch.setenv(ss.ENV_MTP, "0")
    plan = ss.mtp_plan(head)
    assert plan["mtp"] == "disabled by env" and plan["request"] == {"speculative_type": "off"}
    assert plan["reason"] == f"{ss.ENV_MTP}=0"
    assert ss.mtp_plan(plain)["mtp"] == "disabled by env"
    assert ss.mtp_plan(None)["mtp"] == "disabled by env"
    assert ss.mtp_plan(head, ["--spec-type", "ngram-simple"])["mtp"] == "user override"
    assert ss.mtp_plan(head, speculative_type = "mtp")["mtp"] == "user override"
    monkeypatch.setenv(ss.ENV_MTP, "1")
    assert ss.mtp_plan(head)["mtp"] == "enabled"
    assert probe_runs(script) == 2, "one run per build; the verdicts above reused it"


def test_before_load_asks_for_the_spark_draft_depth_in_every_topology(
    cluster, monkeypatch, tmp_path
):
    """The backend's auto mode emits the --spec-type, so this sets only the depth."""
    write_fake_llama_server(cluster.bundle / "build" / "bin", _FAKE_HELP_WITH_FLAG)
    model = write_gguf(tmp_path / "m.gguf", "qwen35", **{"qwen35.nextn_predict_layers": 1})
    _calls, started = _patch_remote(monkeypatch)
    for topology in ("single", "replicas"):
        cluster.topology = topology
        request = _FakeRequest(str(model), llama_extra_args = ["--seed", "1"])
        out = run(ss.before_load(request, 16))
        assert out is not request
        assert out.spec_draft_n_max == 3 and out.speculative_type is None
        assert out.llama_extra_args == ["--seed", "1"], "no flag of its own in the extras"
        assert not started
        status = ss.status()
        assert status["mtp"] == "enabled", status
        assert "--spec-draft-n-max 3" in status["mtp_reason"]
    # Below the crossover the head wins over the groups, which halve the rows per group.
    cluster.topology = "layer_split"
    out = run(ss.before_load(_FakeRequest(str(model)), 4))
    assert out.spec_draft_n_max == 3 and out.speculative_type is None
    assert out.llama_extra_args == [
        "--rpc",
        "192.168.200.13:50052",
        "--device",
        "RPC0,CUDA0",
        "-sm",
        "layer",
        "--tensor-split",
        "0.5,0.5",
        "--cache-ram",
        "0",
    ]
    status = ss.status()
    assert status["topology"] == "layer_split" and status["mtp"] == "enabled"
    assert status["pipeline_groups"] == 0
    assert status["split_config"] == ss.SPLIT_CONFIG_SPEC
    assert status["pipeline_groups_reason"] == status["split_config_reason"]
    assert status["pipeline_groups_reason"] == (
        "--pipeline-groups not added: 4 rows is below the measured crossover of 16, where 2 "
        "groups halve the rows per group and measured 0.97x of one context with speculation "
        "at 8 rows"
    )
    run(ss.shutdown())
    assert ss.status()["mtp"] == "enabled", "the peer going away does not touch this node's launch"


def test_layer_split_with_no_head_keeps_its_groups_and_launches_without_speculation(
    cluster, monkeypatch, tmp_path
):
    write_fake_llama_server(cluster.bundle / "build" / "bin", _FAKE_HELP_WITH_FLAG)
    head = write_gguf(tmp_path / "m.gguf", "qwen35", **{"qwen35.nextn_predict_layers": 1})
    plain = write_gguf(tmp_path / "plain.gguf", "qwen35moe")
    _patch_remote(monkeypatch)
    cluster.topology = "layer_split"
    grouped = ["--pipeline-groups", "2"]
    out = run(ss.before_load(_FakeRequest(str(plain)), 3))
    assert out.llama_extra_args[-2:] == grouped and out.speculative_type == "off"
    assert out.n_parallel == 4
    assert out.spec_draft_n_max is None
    status = ss.status()
    assert status["pipeline_groups"] == 2 and status["mtp"] == "no head"
    assert status["split_config"] == ss.SPLIT_CONFIG_GROUPS
    assert "no speculation to keep" in status["split_config_reason"]
    assert "speculation off for --pipeline-groups 2" in status["mtp_reason"]
    run(ss.shutdown())
    monkeypatch.setenv(ss.ENV_MTP, "0")
    out = run(ss.before_load(_FakeRequest(str(head)), 3))
    assert out.llama_extra_args[-2:] == grouped and out.speculative_type == "off"
    assert ss.status()["pipeline_groups"] == 2 and ss.status()["mtp"] == "disabled by env"
    run(ss.shutdown())
    monkeypatch.delenv(ss.ENV_MTP)
    # A GGUF not on disk yet has no size, so it is never a split before the load.
    request = _FakeRequest(str(tmp_path / "later.gguf"))
    assert run(ss.before_load(request, 3)) is request
    assert ss.status()["mtp"] == "unknown" and ss.status()["topology"] == "single"
    request = _FakeRequest(str(head), llama_extra_args = ["--spec-type", "ngram-simple"])
    out = run(ss.before_load(request, 3))
    assert "--pipeline-groups" not in out.llama_extra_args
    assert out.llama_extra_args[:2] == ["--spec-type", "ngram-simple"]
    status = ss.status()
    assert status["pipeline_groups"] == 0 and status["mtp"] == "user override"
    assert status["split_config"] == ss.SPLIT_CONFIG_SPEC
    assert status["pipeline_groups_reason"] == (
        "--pipeline-groups not added: 3 rows is below the measured crossover of 16, where 2 "
        "groups halve the rows per group and measured 0.97x of one context with speculation "
        "at 8 rows, and the speculation is the caller's"
    )
    run(ss.shutdown())
    for request in (
        _FakeRequest(str(head), speculative_type = "off"),
        _FakeRequest(str(head), llama_extra_args = ["--spec-type", "none"]),
    ):
        out = run(ss.before_load(request, 3))
        assert out.llama_extra_args[-2:] == grouped, out.llama_extra_args
        assert out.spec_draft_n_max is None
        assert ss.status()["pipeline_groups"] == 2 and ss.status()["mtp"] == "user override"
        run(ss.shutdown())
    assert ss.caller_speculation_off("off") and ss.caller_speculation_off("NONE")
    assert ss.caller_speculation_off(None, ["--spec-type=none"])
    assert not ss.caller_speculation_off("mtp")
    assert not ss.caller_speculation_off(None, ["--spec-type", "none", "--model-draft", "/d"])
    assert not ss.caller_speculation_off(None, ["--seed", "1"])
    groups = {"pipeline_groups": 0, "slots": 3, "requested_slots": 3, "reason": "x"}
    mtp = {"mtp": "enabled", "reason": "r", "request": {"spec_draft_n_max": 3}}
    ss.reconcile_split_speculation(groups, mtp)
    assert groups["pipeline_groups"] == 0 and groups["reason"] == "x", "one context: no conflict"
    assert mtp["request"] == {"spec_draft_n_max": 3}


def test_split_takes_groups_and_speculation_above_the_crossover(cluster, monkeypatch, tmp_path):
    """Per-group speculative state is unslothai/llama.cpp PR #187."""
    script = write_fake_llama_server(cluster.bundle / "build" / "bin", _FAKE_HELP_WITH_FLAG)
    head = write_gguf(tmp_path / "m.gguf", "qwen35", **{"qwen35.nextn_predict_layers": 1})
    _patch_remote(monkeypatch)
    cluster.topology = "layer_split"
    out = run(ss.before_load(_FakeRequest(str(head)), 32))
    assert out.llama_extra_args[-2:] == ["--pipeline-groups", "2"]
    assert out.n_parallel == 32, "the slot count travels as the request field, not as argv"
    assert out.spec_draft_n_max == ss.mtp_draft_n_max(32) == 2 and out.speculative_type is None
    status = ss.status()
    assert status["pipeline_groups"] == 2 and status["mtp"] == "enabled"
    assert status["split_config"] == ss.SPLIT_CONFIG_BOTH == "groups + speculation"
    assert "at or above the measured crossover of 16" in status["split_config_reason"]
    assert "1.36x of one context with speculation" in status["split_config_reason"]
    assert "1.09x of 2 groups alone" in status["split_config_reason"]
    assert "kept together with --pipeline-groups 2" in status["mtp_reason"]
    assert probe_runs(script) == 2, "one --help run plus one combined probe, both cached"


def test_split_crossover_sits_at_the_measured_row_count(cluster, monkeypatch, tmp_path):
    """8 and 32 rows are measured; 16 is the interpolated midpoint where the rule turns."""
    write_fake_llama_server(cluster.bundle / "build" / "bin", _FAKE_HELP_WITH_FLAG)
    head = write_gguf(tmp_path / "m.gguf", "qwen35", **{"qwen35.nextn_predict_layers": 1})
    _patch_remote(monkeypatch)
    cluster.topology = "layer_split"
    for rows, config in (
        (1, ss.SPLIT_CONFIG_SPEC),
        (8, ss.SPLIT_CONFIG_SPEC),
        (15, ss.SPLIT_CONFIG_SPEC),
        (16, ss.SPLIT_CONFIG_BOTH),
        (32, ss.SPLIT_CONFIG_BOTH),
    ):
        out = run(ss.before_load(_FakeRequest(str(head)), rows))
        status = ss.status()
        assert status["split_config"] == config, (rows, status)
        assert status["mtp"] == "enabled", rows
        assert out.spec_draft_n_max == ss.mtp_draft_n_max(rows), rows
        if config == ss.SPLIT_CONFIG_BOTH:
            assert status["pipeline_groups"] == 2, rows
            assert out.llama_extra_args[-2:] == ["--pipeline-groups", "2"], rows
            assert out.n_parallel == rows, rows
        else:
            assert status["pipeline_groups"] == 0, rows
            assert "--pipeline-groups" not in out.llama_extra_args, rows
            assert "below the measured crossover of 16" in status["split_config_reason"]
        run(ss.shutdown())


def test_a_split_stops_speculating_at_the_row_count_where_the_drafter_starts_losing(
    cluster, monkeypatch, tmp_path
):
    """Above the boundary a split keeps its groups and emits no draft flag of any kind."""
    write_fake_llama_server(cluster.bundle / "build" / "bin", _FAKE_HELP_WITH_FLAG)
    head = write_gguf(tmp_path / "m.gguf", "qwen35", **{"qwen35.nextn_predict_layers": 1})
    _patch_remote(monkeypatch)
    cluster.topology = "layer_split"
    for rows in (8, 32):
        out = run(ss.before_load(_FakeRequest(str(head)), rows))
        status = ss.status()
        assert status["mtp"] == "enabled", rows
        assert out.spec_draft_n_max == ss.mtp_draft_n_max(rows), rows
        assert out.speculative_type is None, rows
        assert ss.launched_spec_flags(out.llama_extra_args or []) == (None, None), rows
        run(ss.shutdown())
    for rows in (64, 128):
        out = run(ss.before_load(_FakeRequest(str(head)), rows))
        assert out.speculative_type == "off", rows
        assert out.spec_draft_n_max is None, rows
        extras = out.llama_extra_args or []
        assert ss.launched_spec_flags(extras) == (None, None), rows
        assert not any(
            str(a).partition("=")[0].startswith(("--spec", "--draft", "--model-draft"))
            for a in extras
        ), (rows, extras)
        assert extras[-2:] == ["--pipeline-groups", "2"], rows
        status = ss.status()
        assert status["mtp"] == ss.MTP_OFF_FOR_SPLIT_ROWS == "off for the split rows", rows
        assert f"{rows} rows is at or above 64" in status["mtp_reason"], rows
        assert "-7.9 percent" in status["mtp_reason"] and "-22.9 percent" in status["mtp_reason"]
        assert status["split_config"] == ss.SPLIT_CONFIG_GROUPS, rows
        assert status["split_config_reason"] == status["mtp_reason"], rows
        run(ss.shutdown())
    # A drafter the CALLER asked for is never taken away by this boundary.
    request = _FakeRequest(str(head), llama_extra_args = ["--model-draft", "/models/draft.gguf"])
    out = run(ss.before_load(request, 64))
    assert out.llama_extra_args[:2] == ["--model-draft", "/models/draft.gguf"]
    assert out.speculative_type is None and out.spec_draft_n_max is None
    assert ss.status()["mtp"] == "user override"
    run(ss.shutdown())
    # The matrix was measured on the split alone, so single and replicas keep the drafter.
    for topology in ("single", "replicas"):
        cluster.topology = topology
        for rows in (8, 32, 64, 128):
            out = run(ss.before_load(_FakeRequest(str(head)), rows))
            assert out.speculative_type is None, (topology, rows)
            assert out.spec_draft_n_max == ss.mtp_draft_n_max(rows), (topology, rows)
            assert ss.status()["mtp"] == "enabled", (topology, rows)
            run(ss.shutdown())
    sc = _load_spark_cluster()
    assert ss.SPLIT_MTP_OFF_ROWS == sc.SPLIT_MTP_OFF_ROWS == 64
    for rows in (None, 1, 8, 32, 63):
        assert ss.split_mtp_wins(rows) and sc.split_mtp_wins(rows), rows
    for rows in (64, 65, 128, 4096):
        assert not ss.split_mtp_wins(rows) and not sc.split_mtp_wins(rows), rows
    # 64 is a measured point where the best depth LOSES and 32 one where it WINS.
    cells = sc.MTP_DRAFT_N_MAX_X_ROWS_TOKS
    assert sorted(cells) == [32, 64, 128]
    assert ss.SPLIT_MTP_OFF_ROWS in cells and 32 in cells

    def _best_over_off(rows):
        drafting = {depth: toks for depth, toks in cells[rows].items() if depth}
        return max(drafting.values()) / cells[rows][0]

    assert _best_over_off(32) > 1.0, "speculation is a win at the point below the boundary"
    for rows in (64, 128):
        assert _best_over_off(rows) < 1.0, rows
    # The quoted percentages come from the unrounded leg means, not from the rounded table.
    for rows, pct in sc.SPLIT_MTP_BEST_DEPTH_VS_OFF_PCT.items():
        assert abs((_best_over_off(rows) - 1) * 100 - pct) < 0.2, rows
    assert sc.SPLIT_MTP_BEST_DEPTH_VS_OFF_PCT == {32: 11.0, 64: -7.9, 128: -22.9}
    assert sc.GROUPS_X_MTP_CROSSOVER_ROWS == ss.GROUPS_X_MTP_MIN_ROWS == 16
    assert sc.SPLIT_MTP_OFF_ROWS != sc.GROUPS_X_MTP_CROSSOVER_ROWS
    assert sc.MTP_DRAFT_N_MAX_BY_ROWS == ss.MTP_DRAFT_N_MAX_BY_ROWS == {32: 2, 64: 1}
    for rows, want in ((None, 3), (1, 3), (8, 3), (31, 3), (32, 2), (63, 2)):
        assert ss.mtp_draft_n_max(rows) == sc.mtp_draft_n_max(rows) == want, rows
    note = sc.split_mtp_note()
    for text in ("64 concurrent rows", "+11.0 percent", "-7.9 percent", "-22.9 percent", "n-max 2"):
        assert text in note, (text, note)
    assert "one Spark and two replicas keep MTP" in note
    groups_note = sc.groups_x_mtp_note()
    assert "16 rows up" in groups_note and "drops the drafter entirely" in groups_note


def test_split_keeps_todays_behaviour_when_the_server_refuses_the_pair(
    cluster, monkeypatch, tmp_path
):
    script = write_fake_llama_server(
        cluster.bundle / "build" / "bin",
        _FAKE_HELP_WITH_FLAG,
        refuses_groups_with_drafter = True,
    )
    head = write_gguf(tmp_path / "m.gguf", "qwen35", **{"qwen35.nextn_predict_layers": 1})
    plain = write_gguf(tmp_path / "plain.gguf", "qwen35moe")
    _patch_remote(monkeypatch)
    cluster.topology = "layer_split"
    assert ss.llama_server_accepts(ss.PIPELINE_GROUPS_FLAG) is True
    assert ss.llama_server_accepts_groups_with_drafter(2) is False
    assert ss.llama_server_accepts_groups_with_drafter(1) is False, "one group is not the pair"
    out = run(ss.before_load(_FakeRequest(str(head)), 32))
    assert "--pipeline-groups" not in out.llama_extra_args
    assert out.spec_draft_n_max == ss.mtp_draft_n_max(32) == 2
    status = ss.status()
    assert status["pipeline_groups"] == 0 and status["mtp"] == "enabled"
    assert status["split_config"] == ss.SPLIT_CONFIG_SPEC
    assert "refuses it together with a drafter" in status["split_config_reason"]
    run(ss.shutdown())
    out = run(ss.before_load(_FakeRequest(str(plain)), 32))
    assert out.llama_extra_args[-2:] == ["--pipeline-groups", "2"]
    assert out.n_parallel == 32, "the slot count travels as the request field, not as argv"
    assert ss.status()["split_config"] == ss.SPLIT_CONFIG_GROUPS
    assert probe_runs(script) >= 2


def test_groups_keep_parallel_a_multiple_of_the_group_count(cluster, monkeypatch, tmp_path):
    """The crossover is judged on what the load asked for, not on the rounded count."""
    write_fake_llama_server(cluster.bundle / "build" / "bin", _FAKE_HELP_WITH_FLAG)
    head = write_gguf(tmp_path / "m.gguf", "qwen35", **{"qwen35.nextn_predict_layers": 1})
    _patch_remote(monkeypatch)
    cluster.topology = "layer_split"
    monkeypatch.setenv(ss.ENV_PIPELINE_GROUPS, "3")
    request = _FakeRequest(str(head), llama_extra_args = ["--parallel", "32"])
    out = run(ss.before_load(request, 4))
    assert out.llama_extra_args[-2:] == ["--pipeline-groups", "3"]
    assert out.n_parallel == 33, "the slot count travels as the request field, not as argv"
    assert out.n_parallel % 3 == 0
    assert out.spec_draft_n_max == 3
    assert ss.status()["split_config"] == ss.SPLIT_CONFIG_BOTH
    run(ss.shutdown())
    out = run(ss.before_load(_FakeRequest(str(head)), 8))
    assert "--parallel" not in out.llama_extra_args
    assert ss.status()["split_config"] == ss.SPLIT_CONFIG_SPEC


def test_mmproj_control_vectors_and_idle_sleep_cost_the_groups_not_the_speculation(
    cluster, monkeypatch, tmp_path
):
    """These are one per server even after PR #187, so the groups go, not the speculation."""
    write_fake_llama_server(cluster.bundle / "build" / "bin", _FAKE_HELP_WITH_FLAG)
    head = write_gguf(tmp_path / "m.gguf", "qwen35", **{"qwen35.nextn_predict_layers": 1})
    _patch_remote(monkeypatch)
    cluster.topology = "layer_split"
    for extras in (
        ["--mmproj", "/models/mmproj-F16.gguf"],
        ["-mm", "/models/mmproj-F16.gguf"],
        ["--control-vector", "/models/happy.gguf"],
        ["--sleep-idle-seconds", "30"],
    ):
        out = run(ss.before_load(_FakeRequest(str(head), llama_extra_args = list(extras)), 32))
        assert "--pipeline-groups" not in out.llama_extra_args, extras
        assert out.llama_extra_args[:2] == extras[:2], "the caller's flags are untouched"
        assert out.spec_draft_n_max == ss.mtp_draft_n_max(32) == 2, extras
        status = ss.status()
        assert status["pipeline_groups"] == 0 and status["mtp"] == "enabled", extras
        assert status["split_config"] == ss.SPLIT_CONFIG_SPEC, extras
        assert status["pipeline_groups_reason"] == (
            f"--pipeline-groups not added: the server refuses it together with {extras[0]}"
        ), extras
        run(ss.shutdown())
    assert ss.extra_args_refuse_pipeline_groups(["--seed", "1"]) is None
    assert ss.extra_args_refuse_pipeline_groups(None) is None
    assert ss.extra_args_refuse_pipeline_groups(["--mmproj=/p.gguf"]) == "--mmproj"
    assert ss.extra_args_refuse_pipeline_groups(["--control-vector-scaled", "/v", "0.5"]) == (
        "--control-vector-scaled"
    )
    plan = ss.pipeline_groups_plan(32, ["--sleep-idle-seconds", "30"])
    assert plan["pipeline_groups"] == 0 and plan["slots"] == 32


def test_a_users_override_of_either_flag_wins_over_the_crossover(cluster, monkeypatch, tmp_path):
    """A caller's drafter is carried by the groups: PR #187 takes --model-draft per group."""
    write_fake_llama_server(cluster.bundle / "build" / "bin", _FAKE_HELP_WITH_FLAG)
    head = write_gguf(tmp_path / "m.gguf", "qwen35", **{"qwen35.nextn_predict_layers": 1})
    _patch_remote(monkeypatch)
    cluster.topology = "layer_split"
    monkeypatch.setenv(ss.ENV_PIPELINE_GROUPS, "0")
    out = run(ss.before_load(_FakeRequest(str(head)), 32))
    assert "--pipeline-groups" not in out.llama_extra_args
    assert out.spec_draft_n_max == ss.mtp_draft_n_max(32) == 2
    status = ss.status()
    assert status["split_config"] == ss.SPLIT_CONFIG_SPEC and status["mtp"] == "enabled"
    assert ss.ENV_PIPELINE_GROUPS in status["split_config_reason"]
    run(ss.shutdown())
    monkeypatch.delenv(ss.ENV_PIPELINE_GROUPS)
    monkeypatch.setenv(ss.ENV_MTP, "0")
    out = run(ss.before_load(_FakeRequest(str(head)), 32))
    assert out.llama_extra_args[-2:] == ["--pipeline-groups", "2"]
    assert out.n_parallel == 32, "the slot count travels as the request field, not as argv"
    assert out.speculative_type == "off" and out.spec_draft_n_max is None
    status = ss.status()
    assert status["split_config"] == ss.SPLIT_CONFIG_GROUPS and status["mtp"] == "disabled by env"
    run(ss.shutdown())
    monkeypatch.delenv(ss.ENV_MTP)
    request = _FakeRequest(str(head), llama_extra_args = ["--model-draft", "/models/draft.gguf"])
    out = run(ss.before_load(request, 32))
    assert out.llama_extra_args[:2] == ["--model-draft", "/models/draft.gguf"]
    assert out.llama_extra_args[-2:] == ["--pipeline-groups", "2"]
    assert out.n_parallel == 32, "the slot count travels as the request field, not as argv"
    assert out.speculative_type is None and out.spec_draft_n_max is None
    status = ss.status()
    assert status["split_config"] == ss.SPLIT_CONFIG_BOTH and status["mtp"] == "user override"
    run(ss.shutdown())
    request = _FakeRequest(str(head), speculative_type = "off")
    out = run(ss.before_load(request, 32))
    assert out.llama_extra_args[-2:] == ["--pipeline-groups", "2"]
    assert out.n_parallel == 32, "the slot count travels as the request field, not as argv"
    assert out.speculative_type == "off"
    assert ss.status()["split_config"] == ss.SPLIT_CONFIG_GROUPS


def test_the_crossover_constants_are_the_planners_measured_numbers(cluster):
    import importlib.util

    path = Path(ss.__file__).resolve().parents[3] / "spark_cluster.py"
    spec = importlib.util.spec_from_file_location("spark_cluster_for_crossover_test", path)
    sc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sc)
    assert ss.GROUPS_X_MTP_MIN_ROWS == sc.GROUPS_X_MTP_CROSSOVER_ROWS == 16
    assert ss.GROUPS_X_MTP_OVER_MTP_ONLY == sc.GROUPS_X_MTP_OVER_MTP_ONLY
    assert ss.GROUPS_X_MTP_OVER_GROUPS_ONLY == sc.GROUPS_X_MTP_OVER_GROUPS_ONLY
    for rows, cells in sc.GROUPS_X_MTP_DECODE_TOKS.items():
        _one_context, mtp_only, groups_only, both = cells
        assert round(both / mtp_only, 2) == sc.GROUPS_X_MTP_OVER_MTP_ONLY[rows], rows
        assert round(both / groups_only, 2) == sc.GROUPS_X_MTP_OVER_GROUPS_ONLY[rows], rows
    # 16 is the geometric midpoint of the two measured points, and both sides are measured.
    assert min(sc.GROUPS_X_MTP_DECODE_TOKS) == 8 and max(sc.GROUPS_X_MTP_DECODE_TOKS) == 32
    assert sc.GROUPS_X_MTP_CROSSOVER_ROWS**2 == 8 * 32
    assert sc.groups_x_mtp_wins(16) and sc.groups_x_mtp_wins(32)
    assert not sc.groups_x_mtp_wins(8) and not sc.groups_x_mtp_wins(0)
    note = sc.groups_x_mtp_note()
    assert "PR #187" in note and "152.5" in note and "16 rows up" in note
    # The cells have to describe the launch the PRODUCT makes: the first version left out
    # --kv-unified, worth 1.27x on its own, so the constants described a launch never made.
    assert (
        "--kv-unified" in sc.GROUPS_X_MTP_MEASUREMENT
    ), "the cells must be measured with the flags Studio actually launches"
    import inspect

    source = inspect.getsource(ss)
    mirror = source[
        source.index("# Mirrors of spark_cluster.GROUPS_X_MTP_") : source.index(
            "GROUPS_X_MTP_MIN_ROWS = 16"
        )
    ]
    assert "--kv-unified" in mirror, "the mirrored cells must name the flag they were measured with"


def _load_spark_cluster():
    """spark_cluster is not importable as a package from here; load it by path."""
    import importlib.util

    path = Path(ss.__file__).resolve().parents[3] / "spark_cluster.py"
    spec = importlib.util.spec_from_file_location("spark_cluster_for_rows_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_draft_depth_follows_the_rows_and_the_crossover_does_not_move():
    sc = _load_spark_cluster()
    assert ss.MTP_DRAFT_N_MAX == sc.MTP_DRAFT_N_MAX == 3
    assert ss.MTP_DRAFT_N_MAX_BY_ROWS == sc.MTP_DRAFT_N_MAX_BY_ROWS == {32: 2, 64: 1}
    for rows in (None, 1, 4, 8, 16, 31):
        assert ss.mtp_draft_n_max(rows) == sc.mtp_draft_n_max(rows) == 3, rows
    for rows, want in ((32, 2), (48, 2), (63, 2), (64, 1), (128, 1), (512, 1)):
        assert ss.mtp_draft_n_max(rows) == sc.mtp_draft_n_max(rows) == want, rows
    # The rule is the measurement, so the table and the rule cannot drift apart.
    for rows, cells in sc.MTP_DRAFT_N_MAX_X_ROWS_TOKS.items():
        drafting = {depth: toks for depth, toks in cells.items() if depth}
        assert sc.mtp_draft_n_max(rows) == max(drafting, key = drafting.get), rows
        # every swept row count has the drafter-off arm in it, so "best vs off" is sayable
        assert 0 in cells, rows
    # Acceptance is a property of the depth, not of the rows: it is keyed by depth alone.
    assert sorted(sc.MTP_DRAFT_N_MAX_ACCEPTANCE) == [1, 2, 3]
    for text in (sc.MTP_DRAFT_N_MAX_X_ROWS_MEASUREMENT,):
        assert "--kv-unified" in text and "--pipeline-groups 2" in text
    # A DEPTH change only: whether a split runs groups AND speculation is unchanged.
    assert sc.GROUPS_X_MTP_CROSSOVER_ROWS == ss.GROUPS_X_MTP_MIN_ROWS == 16
    assert sc.groups_x_mtp_wins(16) and not sc.groups_x_mtp_wins(8)
    assert sc.SPLIT_MTP_OFF_ROWS == ss.SPLIT_MTP_OFF_ROWS == 64
    assert sc.SPLIT_MTP_OFF_ROWS not in (sc.GROUPS_X_MTP_CROSSOVER_ROWS,)
    for rows in (1, 8, 16, 31, 32, 63):
        assert sc.split_mtp_wins(rows), rows
        assert sc.mtp_draft_n_max(rows) == ss.mtp_draft_n_max(rows), rows


def test_the_layer_boundary_is_explicit_and_the_rows_table_is_consistent():
    """llama.cpp's DEFAULT split divides by each device's free memory at load time, so the
    boundary is not the same twice: two arms of the same model landed on different boundaries
    and were quietly incomparable."""
    sc = _load_spark_cluster()
    extra = ss.layer_split_extra_args("192.168.200.13", 50052)
    assert "--tensor-split" in extra, "the split must not inherit llama.cpp's free-memory boundary"
    assert extra[extra.index("--tensor-split") + 1] == ss.SPLIT_TENSOR_SPLIT_EVEN
    assert ss.SPLIT_TENSOR_SPLIT_EVEN == sc.SPLIT_TENSOR_SPLIT_EVEN, "mirror drifted"
    # The device order is what keeps the output block, and therefore the F32 logits, local.
    assert extra.index("--device") < extra.index("--tensor-split")
    assert extra[extra.index("--device") + 1] == "RPC0,CUDA0"

    by_blocks = sc.SPLIT_TENSOR_SPLIT_MEASURED_PEER_BLOCKS
    assert (
        max(by_blocks, key = by_blocks.get) == 33
    ), "the even split of 66 assignment slots puts 33 blocks on the first device"
    for blocks in (30, 34, 36, 27):
        assert by_blocks[blocks] < by_blocks[33], blocks

    rows = sorted(sc.SPLIT_GROUPS_ROWS_TOKS)
    assert rows == [8, 16, 32, 64, 128], "five points; the whole value of the block was the fifth"
    gains = [sc.SPLIT_GROUPS_ROWS_TOKS[r][1] / sc.SPLIT_GROUPS_ROWS_TOKS[r][0] for r in rows]
    assert all(g > 1.0 for g in gains), "two groups won at every measured concurrency"
    assert abs(gains[0] - gains[1]) < 0.02, "8 and 16 rows are one point, not two"
    assert gains[1:] == sorted(gains[1:]), "from 16 rows up the gain grows with rows"
    assert gains[-1] / gains[1] > 1.25, "and it grows a lot: 1.33x at 16 rows, 1.74x at 128"
    assert sc.SPLIT_GROUPS_MIN_ROWS == min(rows), "not a crossing: the lowest point measured"
    assert "--kv-unified" in sc.SPLIT_GROUPS_ROWS_MEASUREMENT
    assert "512*R" in sc.SPLIT_GROUPS_ROWS_MEASUREMENT, (
        "the rows point is only meaningful if -c scales with --parallel; an earlier table held "
        "--parallel at 32 and varied only the client count, and reached the opposite verdict "
        "at 8 rows"
    )
    # This block carried no drafter, so it must not move the speculation crossover.
    assert sc.GROUPS_X_MTP_CROSSOVER_ROWS == 16
    assert "no speculation" in sc.SPLIT_GROUPS_ROWS_MEASUREMENT


def test_rows_sizing_is_capped_by_ttft_and_never_rounds_the_slot_count_up():
    sc = _load_spark_cluster()
    # The throughput table alone says "use 128 rows"; the cap is the last point whose p90
    # TTFT is inside 14 s.
    rows = sorted(sc.SPLIT_ROWS_TTFT)
    assert rows == [8, 16, 32, 64, 128], "five points, the same five the throughput table has"
    toks = [sc.SPLIT_ROWS_TTFT[r][0] for r in rows]
    med = [sc.SPLIT_ROWS_TTFT[r][1] for r in rows]
    p90 = [sc.SPLIT_ROWS_TTFT[r][2] for r in rows]
    assert toks == sorted(toks), "throughput rises with rows"
    assert med == sorted(med) and p90 == sorted(p90), "and so does TTFT"
    assert (toks[2] / toks[0]) / (med[2] / med[0]) > 0.7
    assert (toks[4] / toks[2]) / (med[4] / med[2]) < 0.45, "above 32 rows TTFT wins the race"
    assert sc.SPLIT_ROWS_INTERACTIVE_MAX == 64
    assert sc.SPLIT_ROWS_TTFT[sc.SPLIT_ROWS_INTERACTIVE_MAX][2] < 14.0, "p90 inside 14 s"
    assert sc.SPLIT_ROWS_TTFT[128][2] > 20.0, "and 128 is not, which is why it is not the cap"
    assert sc.SPLIT_ROWS_THROUGHPUT_MAX == max(rows)
    assert sc.SPLIT_ROWS_MIN == min(rows), "nothing below 8 rows has ever been run"

    # Sizing must NOT round up: an oversized server loses on throughput and TTFT at once.
    assert sc.split_rows_for_users(1) == sc.SPLIT_ROWS_MIN
    assert sc.split_rows_for_users(32) == 32
    assert sc.split_rows_for_users(64) == 64
    assert sc.split_rows_for_users(128) == sc.SPLIT_ROWS_INTERACTIVE_MAX
    assert sc.split_rows_for_users(128, interactive = False) == sc.SPLIT_ROWS_THROUGHPUT_MAX
    assert sc.split_rows_for_users(4096, interactive = False) == sc.SPLIT_ROWS_THROUGHPUT_MAX
    for offered, (tok_ratio, ttft_ratio) in sc.SPLIT_ROWS_OVERSIZED_SLOTS.items():
        if offered < sc.SPLIT_ROWS_THROUGHPUT_MAX:
            assert tok_ratio < 0.85 and ttft_ratio > 1.0, (
                f"oversizing the slots at {offered} offered has to be recorded as a LOSS on "
                "both axes, or the sizing rule reads as arbitrary"
            )

    # No fitting: a fit through two of these points was wrong by 13 percent at the far end.
    assert sc.split_rows_ttft_s(64) == sc.SPLIT_ROWS_TTFT[64][2]
    assert sc.split_rows_ttft_s(100) == sc.SPLIT_ROWS_TTFT[64][2]
    assert sc.split_rows_ttft_s(2) == sc.SPLIT_ROWS_TTFT[8][2]

    assert ss.SPLIT_ROWS_INTERACTIVE_MAX == sc.SPLIT_ROWS_INTERACTIVE_MAX, "mirror drifted"
    assert ss.SPLIT_ROWS_THROUGHPUT_MAX == sc.SPLIT_ROWS_THROUGHPUT_MAX, "mirror drifted"

    # A saturating burst: read as steady-state latency these overstate every rows point.
    assert "arriving at once" in sc.SPLIT_ROWS_TTFT_MEASUREMENT
    assert "saturating burst" in sc.SPLIT_ROWS_TTFT_MEASUREMENT
    assert "--kv-unified" in sc.SPLIT_ROWS_TTFT_MEASUREMENT
    assert "512*R" in sc.SPLIT_ROWS_TTFT_MEASUREMENT
    assert "no speculation" in sc.SPLIT_ROWS_TTFT_MEASUREMENT
    assert sc.GROUPS_X_MTP_CROSSOVER_ROWS == 16
    assert ss.GROUPS_X_MTP_MIN_ROWS == sc.GROUPS_X_MTP_CROSSOVER_ROWS


def test_before_load_leaves_the_callers_speculation_alone(cluster, monkeypatch, tmp_path):
    write_fake_llama_server(cluster.bundle / "build" / "bin", _FAKE_HELP_WITH_FLAG)
    model = write_gguf(tmp_path / "m.gguf", "qwen35", **{"qwen35.nextn_predict_layers": 1})
    cluster.topology = "single"
    request = _FakeRequest(str(model), llama_extra_args = ["--spec-type", "ngram-simple"])
    assert run(ss.before_load(request, 4)) is request
    status = ss.status()
    assert status["mtp"] == "user override"
    assert status["mtp_reason"] == "--spec-type in the pass-through arguments"
    request = _FakeRequest(str(model))
    out = run(ss.before_load(request, 4, inherited_extra_args = ["--seed", "1", "--spec-default"]))
    assert out is request and out.spec_draft_n_max is None
    assert ss.status()["mtp"] == "user override"
    request = _FakeRequest(str(model), speculative_type = "off")
    assert run(ss.before_load(request, 4)) is request
    assert ss.status()["mtp_reason"] == "speculative_type=off"
    request = _FakeRequest(str(model), spec_draft_n_max = 8)
    assert run(ss.before_load(request, 4)) is request and request.spec_draft_n_max == 8
    monkeypatch.setenv(ss.ENV_MTP, "0")
    out = run(ss.before_load(_FakeRequest(str(model)), 4))
    assert out.speculative_type == "off" and out.spec_draft_n_max is None
    assert ss.status()["mtp"] == "disabled by env"
    assert ss.status()["mtp_reason"] == f"{ss.ENV_MTP}=0"
    monkeypatch.delenv(ss.ENV_MTP)
    plain = write_gguf(tmp_path / "plain.gguf", "qwen35moe")
    request = _FakeRequest(str(plain))
    assert run(ss.before_load(request, 4)) is request
    assert ss.status()["mtp"] == "no head"
    old = write_fake_llama_server(tmp_path / "old", _FAKE_HELP_WITHOUT_FLAG)
    monkeypatch.setattr(ss, "llama_server_binary", lambda: str(old))
    request = _FakeRequest(str(model))
    assert run(ss.before_load(request, 4)) is request
    assert ss.status()["mtp"] == "server too old"
    assert ss.status()["mtp_reason"] == "bundle llama-server lacks --spec-type"
    cluster.spark = False
    request = _FakeRequest(str(model))
    assert run(ss.before_load(request, 4)) is request
    assert "mtp" not in ss.status()


def test_after_load_reports_the_spec_type_that_launched(cluster, monkeypatch, tmp_path):
    write_fake_llama_server(cluster.bundle / "build" / "bin", _FAKE_HELP_WITH_FLAG)
    model = write_gguf(tmp_path / "m.gguf", "qwen35", **{"qwen35.nextn_predict_layers": 1})
    cluster.topology = "single"
    run(ss.before_load(_FakeRequest(str(model)), 4))
    assert ss.status()["mtp"] == "enabled"
    backend = _FakeBackend(
        41000, str(model), 4, argv_extra = ["--spec-type", "draft-mtp", "--spec-draft-n-max", "3"]
    )
    run(ss.after_load(backend, 4))
    status = ss.status()
    assert status["topology"] == "single" and status["mtp"] == "enabled"
    assert status["mtp_reason"] == "launched with --spec-type draft-mtp --spec-draft-n-max 3"
    run(ss.before_load(_FakeRequest(str(model)), 4))
    run(ss.after_load(_FakeBackend(41000, str(model), 4), 4))
    status = ss.status()
    assert status["mtp"] == "not launched" and "without --spec-type" in status["mtp_reason"]
    run(ss.before_load(_FakeRequest(str(model)), 4))
    run(
        ss.after_load(
            _FakeBackend(41000, str(model), 4, argv_extra = ["--spec-type=ngram-mod,draft-mtp"]), 4
        )
    )
    assert ss.status()["mtp"] == "enabled"
    run(ss.before_load(_FakeRequest(str(tmp_path / "not-cached.gguf")), 4))
    assert ss.status()["mtp"] == "unknown"
    run(
        ss.after_load(
            _FakeBackend(41000, str(model), 4, argv_extra = ["--spec-type", "ngram-simple"]), 4
        )
    )
    status = ss.status()
    assert status["mtp"] == "other speculation"
    assert status["mtp_reason"] == "launched with --spec-type ngram-simple"
    run(ss.before_load(_FakeRequest(str(model), speculative_type = "mtp"), 4))
    run(
        ss.after_load(
            _FakeBackend(41000, str(model), 4, argv_extra = ["--spec-type", "draft-mtp"]), 4
        )
    )
    status = ss.status()
    assert status["mtp"] == "user override"
    assert status["mtp_reason"] == "launched with --spec-type draft-mtp"
    assert ss.launched_spec_flags(["x", "--spec-draft-n-max", "abc"]) == (None, None)
    assert ss.launched_spec_flags(["x", "--spec-type", "none", "--spec-type", "draft-mtp"]) == (
        "draft-mtp",
        None,
    )


def test_the_split_never_puts_parallel_in_the_pass_through(cluster, monkeypatch, tmp_path):
    """Regression: the groups' slot count used to be appended to ``llama_extra_args`` as
    ``--parallel N``, which is on Studio's pass-through denylist, so the load answered 400 and
    every layer split with pipeline groups failed before llama-server started. Nothing this
    module emits may be on that denylist."""
    from core.inference import llama_server_args

    write_fake_llama_server(cluster.bundle / "build" / "bin", _FAKE_HELP_WITH_FLAG)
    head = write_gguf(tmp_path / "m.gguf", "qwen35", **{"qwen35.nextn_predict_layers": 1})
    _patch_remote(monkeypatch)
    cluster.topology = "layer_split"
    out = run(ss.before_load(_FakeRequest(str(head)), 32))
    assert ss.status()["split_config"] == ss.SPLIT_CONFIG_BOTH
    assert out.n_parallel == 32
    denied = set()
    for group in llama_server_args._DENYLIST_GROUPS:
        denied |= set(group)
    emitted = {a for a in out.llama_extra_args if str(a).startswith("-")}
    assert not (emitted & denied), (emitted & denied, "the load route refuses these with 400")
    for groups in (2, 3, 4):
        extras = ss.layer_split_extra_args("192.168.200.13", 50052, pipeline_groups = groups)
        assert not ({a for a in extras if a.startswith("-")} & denied)


def test_the_slot_count_stays_inside_the_request_field_range(cluster, monkeypatch):
    """n_parallel has a range, not free argv, so rounding a full server UP is refused."""
    write_fake_llama_server(cluster.bundle / "build" / "bin", _FAKE_HELP_WITH_FLAG)
    monkeypatch.setenv(ss.ENV_PIPELINE_GROUPS, "3")
    plan = ss.pipeline_groups_plan(ss.PARALLEL_MAX)
    assert plan["pipeline_groups"] == 3 and plan["slots"] == 63 <= ss.PARALLEL_MAX
    assert plan["requested_slots"] == ss.PARALLEL_MAX
    monkeypatch.setenv(ss.ENV_PIPELINE_GROUPS, "2")
    assert ss.pipeline_groups_plan(ss.PARALLEL_MAX)["slots"] == ss.PARALLEL_MAX
    monkeypatch.setenv(ss.ENV_PIPELINE_GROUPS, str(ss.PARALLEL_MAX + 1))
    plan = ss.pipeline_groups_plan(ss.PARALLEL_MAX)
    assert plan["pipeline_groups"] == 0
    assert "do not fit in the 64-slot maximum" in plan["reason"]


def test_the_drafter_probe_is_a_load_not_a_help(cluster, tmp_path):
    """Regression: probing the pair with ``--help`` answered yes on the fork that refuses it
    as well as on the fork that runs it, because the validation is in ``load_model``, which
    ``--help`` exits long before."""
    refusing = write_fake_llama_server(
        tmp_path / "old" / "build" / "bin",
        _FAKE_HELP_WITH_FLAG,
        refuses_groups_with_drafter = True,
    )
    taking = write_fake_llama_server(tmp_path / "new" / "build" / "bin", _FAKE_HELP_WITH_FLAG)
    assert (
        ss.llama_server_accepts(
            "--pipeline-groups", "2", str(refusing), extra = ("--spec-type", "draft-mtp")
        )
        is True
    )
    assert (
        ss.llama_server_accepts(
            "--pipeline-groups", "2", str(taking), extra = ("--spec-type", "draft-mtp")
        )
        is True
    )
    assert ss.llama_server_accepts_groups_with_drafter(2, str(refusing)) is False
    assert ss.llama_server_accepts_groups_with_drafter(2, str(taking)) is True
    calls = probe_runs(taking)
    assert ss.llama_server_accepts_groups_with_drafter(2, str(taking)) is True
    assert probe_runs(taking) == calls
    assert ss.llama_server_accepts_groups_with_drafter(1, str(taking)) is False
    no_spec = write_fake_llama_server(
        tmp_path / "nospec" / "build" / "bin",
        _FAKE_HELP_WITHOUT_FLAG,
        hidden_flags = ("--pipeline-groups",),
    )
    assert ss.llama_server_accepts_groups_with_drafter(2, str(no_spec)) is False


def test_studios_own_projector_costs_the_groups_not_the_load(cluster, monkeypatch, tmp_path):
    """Regression: the backend adds ``--mmproj`` AFTER before_load and the server refuses it
    together with the groups inside load_model, so the default 27B split died. A repo with no
    projector on disk failed the same way, because the backend downloaded one during the load,
    so a directory scan cannot clear a repo and only the Vision switch can."""
    write_fake_llama_server(cluster.bundle / "build" / "bin", _FAKE_HELP_WITH_FLAG)
    repo = tmp_path / "hub" / "models--unsloth--Qwen3.8-27B-GGUF"
    blobs = repo / "blobs"
    blobs.mkdir(parents = True)
    snapshot = repo / "snapshots" / "abc"
    snapshot.mkdir(parents = True)
    real = write_gguf(blobs / "0123456789", "qwen35", **{"qwen35.nextn_predict_layers": 1})
    model = snapshot / "Qwen3.8-27B-UD-Q4_K_XL.gguf"
    model.symlink_to(real)
    why = ss.projector_blocks_pipeline_groups(str(model))
    assert why and "vision off" in why and "fetches this repo" in why
    assert ss.pipeline_groups_plan(32, projector = why)["pipeline_groups"] == 0
    # Vision off: llama-server is launched with --no-mmproj-auto, so the groups may come.
    assert ss.projector_blocks_pipeline_groups(str(model), disable_vision = True) is None
    # A plain local GGUF outside a hub cache has nothing to fetch: it keeps its groups.
    loose = tmp_path / "loose.gguf"
    loose.write_bytes(b"x")
    assert ss.projector_blocks_pipeline_groups(str(loose)) is None
    (snapshot / "mmproj-F16.gguf").write_bytes(b"x")
    assert "mmproj-F16.gguf" in ss.projector_blocks_pipeline_groups(str(model))

    _patch_remote(monkeypatch)
    cluster.topology = "layer_split"
    out = run(ss.before_load(_FakeRequest(str(model)), 32))
    status = ss.status()
    assert "--pipeline-groups" not in out.llama_extra_args
    assert status["pipeline_groups"] == 0 and status["mtp"] == "enabled"
    assert status["split_config"] == ss.SPLIT_CONFIG_SPEC
    assert "vision off" in status["split_config_reason"]
    run(ss.shutdown())
    (snapshot / "mmproj-F16.gguf").unlink()
    out = run(ss.before_load(_FakeRequest(str(model), disable_vision = True), 32))
    status = ss.status()
    assert out.llama_extra_args[-2:] == ["--pipeline-groups", "2"]
    assert status["split_config"] == ss.SPLIT_CONFIG_BOTH and status["pipeline_groups"] == 2
    run(ss.shutdown())


def test_a_peer_that_stopped_answering_is_restarted_not_reused(cluster, monkeypatch, tmp_path):
    """Regression: a reload on a layer split failed to connect, because the peer process was
    reused on the strength of its ssh session alone and a session can outlive the server it
    carries."""
    write_fake_llama_server(cluster.bundle / "build" / "bin", _FAKE_HELP_WITH_FLAG)
    model = tmp_path / "big.gguf"
    model.write_bytes(b"x")
    _calls, started = _patch_remote(monkeypatch)
    cluster.topology = "layer_split"
    run(ss.before_load(_FakeRequest(str(model)), 4))
    assert len(started) == 1 and ss.status()["topology"] == "layer_split"
    first = ss.state().peer_process
    run(ss.before_load(_FakeRequest(str(model)), 4))
    assert len(started) == 1 and ss.state().peer_process is first
    run(ss.before_load(_FakeRequest(str(model), force_reload = True), 4))
    assert len(started) == 2, "a forced reload reused the peer the old server still holds"
    monkeypatch.setattr(ss, "wait_for_port", _port_answers(False, then = True))
    run(ss.before_load(_FakeRequest(str(model)), 4))
    assert len(started) == 3, "the dead peer was reused instead of restarted"
    assert ss.status()["topology"] == "layer_split"
    run(ss.shutdown())


def _port_answers(first: bool, *, then: bool):
    state = {"n": 0}

    async def _answer(host, port, timeout):
        state["n"] += 1
        return first if state["n"] == 1 else then

    return _answer


def test_relaunch_budget_resets_after_a_peer_recovers(monkeypatch):
    # The budget bounds one crash loop. Three clean restarts spread over a long run must not
    # leave the peer permanently unrecoverable.
    import asyncio

    state = ss.SparkServing() if hasattr(ss, "SparkServing") else None
    if state is None:
        import pytest
        pytest.skip("SparkServing type not exposed")

    class _Proc:
        name, peer, returncode, remote_pid = "llama-server", "peer", 1, 7
        tail: list = []
        started_at = 1.0
        alive = True
        async def stop(self, timeout = None): pass
        async def start(self): pass

    real_sleep = asyncio.sleep
    monkeypatch.setattr(ss.asyncio, "sleep", lambda *_a, **_k: real_sleep(0))
    state.peer_process = _Proc()
    state.attached_backend = object()
    state.relaunch_attempts = len(ss.RELAUNCH_BACKOFF_S) - 1

    asyncio.run(state._relaunch_peer())

    assert state.relaunch_attempts == 0
    assert not state.relaunch_gave_up


def test_load_failed_keeps_a_topology_whose_model_is_still_loaded():
    # The route validates before it unloads, so a rejected replacement load leaves the previous
    # model serving. Its split or router must survive that.
    import asyncio

    state = ss.SparkServing()
    detached = []

    async def _detach():
        detached.append(True)

    state.detach = _detach
    state.peer_process = object()
    state.attached_backend = type("B", (), {"is_loaded": True})()
    asyncio.run(state.load_failed())
    assert not detached, "tore down a topology whose model is still loaded"

    state.attached_backend = type("B", (), {"is_loaded": False})()
    asyncio.run(state.load_failed())
    assert detached, "kept a topology after the model really went away"


def test_ssh_run_kills_and_reaps_a_timed_out_child(monkeypatch):
    # An unreachable peer times these out on a supervisor loop, so abandoning the child leaks
    # one live ssh per probe and then one zombie per probe.
    import asyncio

    killed, waited = [], []

    class _Proc:
        returncode = None
        async def communicate(self):
            await asyncio.sleep(10)
        def kill(self):
            killed.append(True)
        async def wait(self):
            waited.append(True)

    async def _spawn(*a, **k):
        return _Proc()

    monkeypatch.setattr(ss.asyncio, "create_subprocess_exec", _spawn)
    rc, out, err = asyncio.run(ss.ssh_run("peer", "true", timeout = 0.01))

    assert rc == 255
    assert killed, "timed-out ssh child was never killed"
    assert waited, "timed-out ssh child was never reaped"


def test_estimate_kv_bytes_prices_k_and_v_apart(tmp_path):
    # K and V are configurable independently. Pricing V as K understates an asymmetric cache,
    # and understating is the direction that puts a model on a node that cannot hold it.
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

    cells = 2 * 1024 * 2 * 16  # layers x tokens x kv heads x head dim
    q4, f32 = 18 / 32, 4.0

    # V defaults to K, so the symmetric answer is unchanged.
    assert ss.estimate_kv_bytes(str(path), 1024, "q4_0") == int(cells * (q4 + q4))
    # K=q4_0 with V=f32 must be priced on both axes, not as two q4_0.
    asymmetric = ss.estimate_kv_bytes(str(path), 1024, "q4_0", "f32")
    assert asymmetric == int(cells * (q4 + f32))
    assert asymmetric > 4 * ss.estimate_kv_bytes(str(path), 1024, "q4_0")


def test_after_load_prices_the_aggregate_context_not_one_slot():
    # Once --parallel splits the cache, _effective_context_length is the per-slot window while
    # the aggregate is published separately. Pricing the per-slot value and then dividing by
    # slots reconstructs one slot's cache, short by up to the slot count, which is the direction
    # that puts a model on a node that cannot hold it.
    class _Backend:
        _kv_cache_context_total = 32768
        _effective_context_length = 4096   # 8 slots x 4096
        requested_n_ctx = 4096

    b = _Backend()
    picked = int(
        getattr(b, "_kv_cache_context_total", None)
        or getattr(b, "_effective_context_length", None)
        or getattr(b, "requested_n_ctx", 0)
        or 0
    )
    assert picked == 32768

    # An older backend that does not publish the aggregate must still work.
    class _Old:
        _effective_context_length = 4096
        requested_n_ctx = 4096

    o = _Old()
    picked_old = int(
        getattr(o, "_kv_cache_context_total", None)
        or getattr(o, "_effective_context_length", None)
        or getattr(o, "requested_n_ctx", 0)
        or 0
    )
    assert picked_old == 4096
