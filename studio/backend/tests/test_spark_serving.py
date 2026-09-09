# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The two-Spark serving orchestrator, off any real hardware. ``spark_cluster`` is a stub, so
these pin the contract the module has with it and the gate that keeps all of it off a normal
machine; ssh never runs."""

from __future__ import annotations

import asyncio
import os
import tempfile
import threading
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
        self.fits_any_topology = True
        self.reason = ""

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
        out = {
            "topology": self.topology,
            "reason": self.reason or f"stub says {self.topology}",
            "speedup": 1.3,
        }
        # The real planner reports this on every answer, and says `single` even when NOTHING
        # holds the load, so a stub that omitted it could not exercise the refusal.
        out["fits_any_topology"] = self.fits_any_topology
        return out


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
    # Rail discovery is cached with a TTL so the polled status endpoint does not walk sysfs on
    # the event loop every time. That cache is module state, and these tests reconfigure the
    # cluster between cases, so it is dropped here rather than left to leak across them.
    ss.reset_peer_discovery_cache()
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
    which is why the usage text cannot probe it.

    POSIX only, and deliberately so rather than by oversight. The stand-in has to be a file the
    resolver will name AND the OS will run, and on Windows nothing satisfies both: the backend
    looks for ``llama-server.exe`` (llama_cpp.py ``_find_llama_server_binary``), which is the
    right name for a real distribution, and Windows runs a ``.exe`` only if it is a PE image.
    Renaming this ``/bin/sh`` script to ``.exe`` would get it found and then fail at spawn, so
    the skip is at the fixture instead: one honest reason on every test that needs to execute a
    fake server. What those tests cover -- argv assembly, the capability probe, topology -- has
    no platform branch, so POSIX coverage is the real coverage. The platform-specific code that
    DOES branch is tested directly, without a spawn, in the ``is_executable_file`` tests above.
    """
    if os.name == "nt":
        pytest.skip("the fake llama-server is a POSIX shell script; Windows cannot exec it")
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
    # Rail discovery is cached for _PEER_DISCOVERY_TTL_S, so a peer that appears or disappears
    # mid-test has to say so. Real cabling does not change between two statements; this does.
    assert ss.enabled()
    cluster.peer = None
    ss.reset_peer_discovery_cache()
    assert not ss.enabled(), "a Spark with no configured peer stays single"
    cluster.peer = "192.168.200.13"
    ss.reset_peer_discovery_cache()
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
    model_stale = False,
    port_opens = True,
    pid_alive = True,
    absent_paths = (),
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
        if remote.startswith("stat -c"):
            # The peer answers with the SAME size and mtime as the local file, which is what a
            # correctly replicated node looks like. `model_present = False` makes it answer
            # NOSTAT for every path, i.e. the file is not there at all.
            import re as _re

            paths = _re.findall(r"stat -c '%s %Y' (\S+)", remote)
            lines = []
            for raw in paths:
                path = raw.strip("'\"")
                if path in set(absent_paths):
                    # This ONE file is missing on the peer while everything else matches, which
                    # is what isolates a single launch input rather than a whole absent node.
                    lines.append("NOSTAT")
                    continue
                if not model_present:
                    lines.append("NOSTAT")
                    continue
                try:
                    st = os.stat(path)
                except OSError:
                    lines.append("NOSTAT")
                    continue
                if model_stale:
                    # present at the same path, and a different file: the case an existence
                    # test cannot tell from a correctly replicated node.
                    lines.append(f"{st.st_size + 1} {int(st.st_mtime) + 60}")
                else:
                    lines.append(f"{st.st_size} {int(st.st_mtime)}")
            return 0, "\n".join(lines) + "\n", ""
        if remote.startswith("kill -0"):
            # Ownership: is the child this run started still there? A real start records the
            # pid the remote wrapper printed, so the double records one too.
            return (0, "PIDLIVE\n", "") if pid_alive else (0, "PIDGONE\n", "")
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
        # The real start reads UNSLOTH_SPARK_PID off the remote wrapper. Ownership is decided
        # against that pid, so a double without one answers "not ours" to everything.
        self.remote_pid = 4242

    async def fake_stop(self, timeout = 10.0):
        return None

    async def fake_wait(
        host,
        port,
        timeout,
        *,
        cancelled = None,
    ):
        # Honours the cancel check the same way the real one does, so a test can drive a
        # cancelled attach through this double.
        if cancelled is not None and cancelled():
            return False
        return port_opens

    monkeypatch.setattr(ss, "ssh_run", fake_ssh_run)
    monkeypatch.setattr(ss.PeerProcess, "start", fake_start)
    monkeypatch.setattr(ss.PeerProcess, "stop", fake_stop)
    monkeypatch.setattr(ss, "wait_for_port", fake_wait)
    # The settle exists so a real remote bind error has time to become an exit. The double
    # execs nothing, so there is no race to wait out; test_a_stranger_on_the_port covers it.
    monkeypatch.setattr(ss, "PEER_OWNERSHIP_SETTLE_S", 0.0)
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

    async def _answer(
        host,
        port,
        timeout,
        *,
        cancelled = None,
    ):
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

        async def stop(self, timeout = None):
            pass

        async def start(self):
            pass

    class _Backend:
        def __init__(self, healthy, primary):
            self.healthy, self.primary = healthy, primary

    class _Router:
        def __init__(self, healthy):
            self.backends = [_Backend(True, True), _Backend(healthy, False)]

    real_sleep = asyncio.sleep
    monkeypatch.setattr(ss.asyncio, "sleep", lambda *_a, **_k: real_sleep(0))
    proc = _Proc()
    state.peer_process = proc
    state.attached_backend = object()
    # Recovery is the ROUTER seeing the peer answer, not the peer merely being alive three
    # seconds later, so the budget only resets when there is a healthy non-primary backend.
    state.router = _Router(True)
    state.relaunch_attempts = len(ss.RELAUNCH_BACKOFF_S) - 1

    asyncio.run(state._relaunch_peer())

    assert state.relaunch_attempts == 0
    assert not state.relaunch_gave_up


def test_a_peer_that_never_becomes_healthy_spends_the_relaunch_budget(monkeypatch):
    # A large model spends minutes inside load_model before it can OOM, so a peer that dies
    # there is alive at the three second settle EVERY time. Resetting the budget on that alone
    # meant the three attempt bound was never reached: the peer reloaded and failed forever,
    # paying for the whole load each cycle. Recovery has to be the router seeing it answer.
    import asyncio

    state = ss.SparkServing()

    class _Proc:
        name, peer, returncode, remote_pid = "llama-server", "peer", 1, 7
        tail: list = []
        started_at = 1.0
        alive = True

        async def stop(self, timeout = None):
            pass

        async def start(self):
            pass

    class _Backend:
        def __init__(self, healthy, primary):
            self.healthy, self.primary = healthy, primary

    class _Router:
        # The primary is healthy throughout; only the peer never comes up. A check that looked
        # at "any healthy backend" rather than a non-primary one would pass on this forever.
        backends = [_Backend(True, True), _Backend(False, False)]

    real_sleep = asyncio.sleep
    monkeypatch.setattr(ss.asyncio, "sleep", lambda *_a, **_k: real_sleep(0))
    monkeypatch.setattr(ss, "RELAUNCH_HEALTHY_TIMEOUT_S", 0.05)
    monkeypatch.setattr(ss, "RELAUNCH_HEALTH_POLL_S", 0.01)
    state.peer_process = _Proc()
    state.attached_backend = object()
    state.router = _Router()

    asyncio.run(state._relaunch_peer())

    assert state.relaunch_gave_up, "the budget is spent, not reset, by a peer that never answers"
    assert state.relaunch_attempts == len(ss.RELAUNCH_BACKOFF_S)
    events = [entry.get("event") for entry in state.relaunch_log]
    assert "recovered" not in events
    assert events.count("relaunched but never healthy") == len(ss.RELAUNCH_BACKOFF_S)


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
        _effective_context_length = 4096  # 8 slots x 4096
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


def test_current_topology_does_not_pay_for_rail_discovery(cluster, monkeypatch):
    # Status polls call this constantly. Going through enabled() cost a sysfs walk and an `ip`
    # fork, measured at 16.2 ms per call on a paired Spark, on the event loop.
    probed = []
    monkeypatch.setattr(ss, "peer_address", lambda: probed.append(1) or "192.168.200.13")

    st = ss.state()
    st.attached_backend = None
    st.peer_process = None
    st.router = None
    assert ss.current_topology() is None
    assert not probed, "nothing is attached, so discovery must not run"

    # With a topology attached the answer is the real one, still without discovery.
    st.peer_process = object()
    st.topology = "layer_split"
    assert ss.current_topology() == "layer_split"
    assert not probed
    st.peer_process = None
    st.topology = "single"


def test_underscore_spellings_are_read_the_way_llama_server_reads_them():
    # common/arg.cpp folds `_` to `-` in every `--` token before the lookup, so these are
    # spellings the server accepts and a raw comparison used to miss all four.
    assert ss.extra_args_own_speculation(["--spec_type", "none"]) == "--spec-type"
    assert ss.extra_args_own_speculation(["--model_draft", "/d.gguf"]) == "--model-draft"
    assert ss.extra_args_refuse_pipeline_groups(["--control_vector", "/v.gguf"]) == (
        "--control-vector"
    )
    assert ss.extra_args_refuse_pipeline_groups(["--sleep_idle_seconds=30"]) == (
        "--sleep-idle-seconds"
    )
    assert ss.launched_spec_flags(["--spec_type", "draft-mtp", "--spec_draft_n_max=3"]) == (
        "draft-mtp",
        3,
    )
    assert ss._extra_args_slots(["--parallel_", "4"]) is None  # not an option at all
    assert ss._extra_args_slots(["--parallel", "4"]) == 4

    # Shorts keep their exact spelling, as they do in arg.cpp: the fold is guarded on "--".
    assert ss.extra_args_own_speculation(["-md", "/d.gguf"]) == "-md"
    assert ss.extra_args_refuse_pipeline_groups(["-mm", "/p.gguf"]) == "-mm"


def test_split_reload_keeps_the_extras_the_apply_path_does_not_round_trip(
    cluster, monkeypatch, tmp_path
):
    # A settings Apply reload omits llama_extra_args, and _resolve_inherited_extra_args puts
    # the previous same-model load's extras back afterwards, but only while the field is still
    # None. Writing the split flags into a bare list used to drop them for real.
    cluster.topology = "layer_split"
    write_fake_llama_server(cluster.bundle / "build" / "bin", _FAKE_HELP_WITH_FLAG)
    model = tmp_path / "big.gguf"
    model.write_bytes(b"x")
    _patch_remote(monkeypatch)
    monkeypatch.setenv(ss.ENV_PIPELINE_GROUPS, "2")

    request = _FakeRequest(str(model))
    assert getattr(request, "llama_extra_args", None) is None
    out = run(ss.before_load(request, 3, inherited_extra_args = ["--seed", "1"]))
    assert out.llama_extra_args[:2] == ["--seed", "1"]
    assert "--rpc" in out.llama_extra_args


def test_split_reload_reads_an_inherited_flag_the_groups_are_refused_with(
    cluster, monkeypatch, tmp_path
):
    # Same reload, but the inherited extras carry a flag llama-server refuses together with
    # --pipeline-groups. Planning against the bare field appended the groups anyway and the
    # server then failed the whole load.
    cluster.topology = "layer_split"
    write_fake_llama_server(cluster.bundle / "build" / "bin", _FAKE_HELP_WITH_FLAG)
    model = tmp_path / "big.gguf"
    model.write_bytes(b"x")
    _patch_remote(monkeypatch)
    monkeypatch.setenv(ss.ENV_PIPELINE_GROUPS, "2")

    out = run(
        ss.before_load(
            _FakeRequest(str(model)), 3, inherited_extra_args = ["--control-vector", "/v.gguf"]
        )
    )
    assert ss.PIPELINE_GROUPS_FLAG not in out.llama_extra_args
    assert ss.state().pipeline_groups in (0, 1)


def test_group_rounding_never_costs_the_caller_slots(cluster, monkeypatch):
    # _start_layer_split writes this slot count straight into request.n_parallel, so rounding
    # down to fit the cap used to silently serve fewer concurrent users than were asked for.
    monkeypatch.setattr(ss, "llama_server_supports", lambda flag: True)
    monkeypatch.setenv(ss.ENV_PIPELINE_GROUPS, "40")
    plan = ss.pipeline_groups_plan(ss.PARALLEL_MAX, [])
    assert plan["pipeline_groups"] == 0
    assert plan["slots"] == ss.PARALLEL_MAX
    assert "rounding down" in plan["reason"]

    # A group count that divides is untouched, and so is one that rounds up within the cap.
    monkeypatch.setenv(ss.ENV_PIPELINE_GROUPS, "2")
    plan = ss.pipeline_groups_plan(ss.PARALLEL_MAX, [])
    assert plan["pipeline_groups"] == 2 and plan["slots"] == ss.PARALLEL_MAX
    plan = ss.pipeline_groups_plan(3, [])
    assert plan["pipeline_groups"] == 2 and plan["slots"] == 4

    # A one-slot rounding loss is worth the groups, which measured about 1.4x.
    monkeypatch.setenv(ss.ENV_PIPELINE_GROUPS, "3")
    plan = ss.pipeline_groups_plan(ss.PARALLEL_MAX, [])
    assert plan["pipeline_groups"] == 3 and plan["slots"] == 63

    # A request already over the maximum was going to be clamped either way, so the clamp
    # there is not the groups' doing and does not count against them.
    monkeypatch.setenv(ss.ENV_PIPELINE_GROUPS, "2")
    plan = ss.pipeline_groups_plan(ss.PARALLEL_MAX * 2, [])
    assert plan["pipeline_groups"] == 2 and plan["slots"] == ss.PARALLEL_MAX


def test_an_undecided_mtp_verdict_is_left_to_the_backend(cluster, monkeypatch):
    # mtp_plan returns "unknown" when the GGUF is not on disk yet, which is what an uncached
    # repo forced to a layer split always is. Writing "off" took automatic MTP away from a
    # model that does have a head.
    monkeypatch.setattr(ss, "llama_server_accepts_groups_with_drafter", lambda groups: True)
    groups = {"pipeline_groups": 2, "slots": 4, "requested_slots": 4, "reason": None}
    mtp = {"mtp": "unknown", "reason": "GGUF not on disk before the load", "request": {}}
    ss.reconcile_split_speculation(groups, mtp)
    assert mtp["request"].get("speculative_type") is None

    # A GGUF that is on disk and has no head is still turned off: that verdict is decided.
    groups = {"pipeline_groups": 2, "slots": 4, "requested_slots": 4, "reason": None}
    mtp = {"mtp": "no head", "reason": "no nextn_predict_layers", "request": {}}
    ss.reconcile_split_speculation(groups, mtp)
    assert mtp["request"]["speculative_type"] == "off"

    # And so is "unknown" on a server that refuses the groups together with a drafter, where
    # leaving it undecided would fail the load rather than lose a speedup.
    monkeypatch.setattr(ss, "llama_server_accepts_groups_with_drafter", lambda groups: False)
    groups = {"pipeline_groups": 2, "slots": 4, "requested_slots": 4, "reason": None}
    mtp = {"mtp": "unknown", "reason": "GGUF not on disk before the load", "request": {}}
    ss.reconcile_split_speculation(groups, mtp)
    assert mtp["request"]["speculative_type"] == "off"


def test_a_load_that_bypassed_the_planner_stops_using_the_peer(cluster):
    # The auto-switch and preview paths call the loader directly, so nothing plans or
    # re-attaches for them. Leaving the peer up means the router alternates requests between
    # the model this load replaces and the new one.
    st = ss.state()
    stopped = []

    class _Peer:
        async def stop(self):
            stopped.append("peer")

    class _Router:
        async def stop(self):
            stopped.append("router")

    st.topology = "replicas"
    st.attached_backend = object()
    st.peer_process = _Peer()
    st.router = _Router()

    run(ss.reconcile_internal_load())
    assert stopped == ["router", "peer"]
    assert st.topology == "single"
    assert st.attached_backend is None

    # Nothing attached: no work, on a Spark or anywhere else.
    stopped.clear()
    run(ss.reconcile_internal_load())
    assert stopped == []
    assert st.topology == "single"


def _snapshot(root, repo, digest, files, mtime):
    snap = root / ("models--" + repo.replace("/", "--")) / "snapshots" / digest
    snap.mkdir(parents = True)
    for name, size in files.items():
        target = snap / name
        target.parent.mkdir(parents = True, exist_ok = True)
        target.write_bytes(b"x" * size)
    os.utime(snap, (mtime, mtime))
    return snap


def test_the_sized_gguf_is_the_newest_snapshot_not_the_first_hash(monkeypatch, tmp_path):
    # The loader takes snapshots newest first. Lexicographic order is the hash, which says
    # nothing about age, so a stale copy could be sized and planned against while a different
    # one loaded: a smaller stale copy plans single and then the real load does not fit.
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    repo = "unsloth/Big-GGUF"
    _snapshot(tmp_path, repo, "aaa0", {"big-Q4_K_M.gguf": 4096}, mtime = 1_000)
    _snapshot(tmp_path, repo, "zzz9", {"big-Q4_K_M.gguf": 16384}, mtime = 2_000)

    chosen = ss.cached_repo_file(repo, None)
    assert chosen is not None and "zzz9" in chosen
    assert ss.gguf_size_bytes(chosen) == 16384


def test_a_companion_gguf_is_never_sized_in_place_of_the_weights(monkeypatch, tmp_path):
    # An MTP or draft companion beside the weights is a few hundred megabytes. Sizing it
    # plans single for a model that needs both Sparks.
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    repo = "unsloth/With-Companions"
    _snapshot(
        tmp_path,
        repo,
        "abc1",
        {
            "MTP/model-Q4_K_M.gguf": 512,
            "model-Q4_K_M.gguf": 32768,
            "mmproj-model-f16.gguf": 256,
        },
        mtime = 1_000,
    )

    chosen = ss.cached_repo_file(repo, None)
    assert chosen is not None, "the weights are cached"
    assert "MTP" not in chosen and "mmproj" not in os.path.basename(chosen)
    assert ss.gguf_size_bytes(chosen) == 32768


def test_an_uncached_oversized_repo_splits_on_its_first_load(cluster, monkeypatch, tmp_path):
    # Nothing re-plans after the download, and the single-node launch the model would have to
    # survive first is exactly what does not fit, so a first load had to know the size before
    # the download started or the split could never happen at all.
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    write_fake_llama_server(cluster.bundle / "build" / "bin", _FAKE_HELP_WITH_FLAG)
    _patch_remote(monkeypatch)

    asked = []
    monkeypatch.setattr(
        ss,
        "remote_gguf_size_bytes",
        lambda path, variant, token = None: asked.append(path) or 200 * 1024**3,
    )

    cluster.topology = "layer_split"
    out = run(ss.before_load(_FakeRequest("unsloth/Huge-GGUF"), 4))
    assert asked == ["unsloth/Huge-GGUF"], "the hub is asked only when nothing is cached"
    # The planner is given the real size instead of nothing, which is what decides this.
    assert cluster.planner_calls[-1]["model_bytes"] == 200 * 1024**3
    assert ss.state().topology == "layer_split"
    assert "--rpc" in (out.llama_extra_args or [])


def test_a_hub_that_cannot_answer_leaves_the_old_sizeless_behaviour(cluster, monkeypatch, tmp_path):
    # Offline, gated, rate-limited: all of them fall back to planning single, which is what
    # happened before and is the right answer to an unknown size.
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    write_fake_llama_server(cluster.bundle / "build" / "bin", _FAKE_HELP_WITH_FLAG)
    _patch_remote(monkeypatch)
    monkeypatch.setattr(ss, "remote_gguf_size_bytes", lambda path, variant, token = None: None)

    cluster.topology = "layer_split"  # the planner would split, if it were ever asked
    request = _FakeRequest("unsloth/Huge-GGUF")
    assert run(ss.before_load(request, 4)) is request
    assert cluster.planner_calls == [], "an unknown size does not reach the planner at all"
    assert ss.state().topology == "single"
    assert "unknown" in (ss.state().plan or {}).get("reason", "")


def test_remote_sizing_never_raises_out_of_the_load(monkeypatch):
    # It is a network call in a load path, so every failure has to stay inside it.
    def _boom(*_a, **_k):
        raise RuntimeError("hub is down")

    monkeypatch.setattr(ss, "_pick_variant", _boom)
    assert ss.remote_gguf_size_bytes("unsloth/Huge-GGUF", None) is None
    # A local path or a bare name is not a repo and is not asked about at all.
    assert ss.remote_gguf_size_bytes("/models/x.gguf", None) is None
    assert ss.remote_gguf_size_bytes("mymodel", None) is None


def test_a_stranger_on_the_port_is_not_adopted_as_ours(monkeypatch):
    # A port that is already occupied answers on the first probe while the child that could
    # not bind it exits. Adopting it leaves the split attached to a server nothing here
    # manages: the supervisor relaunches a dead child, and if the stranger leaves the split
    # cannot be recovered.
    monkeypatch.setattr(ss, "PEER_OWNERSHIP_SETTLE_S", 0.01)

    async def always_open(
        host,
        port,
        timeout,
        *,
        cancelled = None,
    ):
        return True

    monkeypatch.setattr(ss, "wait_for_port", always_open)

    class _Child:
        # remote_pid and peer because ownership is settled against the pid the remote wrapper
        # printed, not against the local ssh session's state: the wrapper only re-checks its
        # child once per PEER_REAP_POLL_S, which outlasts the settle.
        peer = "1.2.3.4"
        remote_pid = 4242

        def __init__(self, dies):
            self._dies = dies
            self.alive = True

        async def die_after_the_probe(self):
            self.alive = not self._dies

    async def pid_is_live(peer, remote, timeout = 20.0):
        return 0, "PIDLIVE\n", ""

    monkeypatch.setattr(ss, "ssh_run", pid_is_live)

    ours = _Child(dies = False)
    assert run(ss.wait_for_own_port(ours, "1.2.3.4", 50052, 5.0)) is True

    # The same successful probe, but our child exited on its bind error.
    class _Doomed:
        alive = True
        peer = "1.2.3.4"
        remote_pid = 4242

    doomed = _Doomed()

    async def open_then_kill(
        host,
        port,
        timeout,
        *,
        cancelled = None,
    ):
        doomed.alive = False
        return True

    monkeypatch.setattr(ss, "wait_for_port", open_then_kill)
    assert run(ss.wait_for_own_port(doomed, "1.2.3.4", 50052, 5.0)) is False

    # Already gone before the first probe: not asked about at all.
    probed = []
    monkeypatch.setattr(
        ss,
        "wait_for_port",
        lambda h, p, t, **kw: probed.append(1) or always_open(h, p, t, **kw),
    )
    dead = _Doomed()
    dead.alive = False
    assert run(ss.wait_for_own_port(dead, "1.2.3.4", 50052, 5.0)) is False
    assert probed == []


def test_a_cancelled_replica_attach_does_not_leave_the_peer_running(cluster, monkeypatch, tmp_path):
    # Cancellation does not reach after_load's except Exception, so without cleanup here the
    # peer llama-server keeps its memory and its port with nothing left that knows about it.
    cluster.topology = "replicas"
    monkeypatch.setenv(ss.ENV_PEER, "127.0.0.1")
    model = tmp_path / "m.gguf"
    model.write_bytes(b"x")
    _calls, started = _patch_remote(
        monkeypatch, binary = "$HOME/.unsloth/llama.cpp/build/bin/llama-server"
    )

    stopped = []
    real_stop = ss.PeerProcess.stop

    async def counting_stop(self, timeout = 10.0):
        stopped.append(self.name)
        return await real_stop(self, timeout = timeout)

    async def cancel_while_waiting(
        process,
        host,
        port,
        timeout,
        *,
        cancelled = None,
    ):
        raise asyncio.CancelledError()

    monkeypatch.setattr(ss.PeerProcess, "stop", counting_stop)
    monkeypatch.setattr(ss, "wait_for_own_port", cancel_while_waiting)

    with pytest.raises(asyncio.CancelledError):
        run(ss.after_load(_FakeBackend(12345, str(model)), 16))
    assert started, "the peer llama-server was launched"
    assert stopped == ["llama-server"]
    assert ss.state().peer_process is None
    assert ss.state().router is None


def test_a_replica_port_taken_by_a_stranger_is_not_routed_to(cluster, monkeypatch, tmp_path):
    # A healthy llama-server already on that port would be admitted as the peer backend and
    # served generation traffic for whatever model it is holding.
    cluster.topology = "replicas"
    monkeypatch.setenv(ss.ENV_PEER, "127.0.0.1")
    model = tmp_path / "m.gguf"
    model.write_bytes(b"x")
    _calls, started = _patch_remote(
        monkeypatch, binary = "$HOME/.unsloth/llama.cpp/build/bin/llama-server"
    )

    async def not_ours(
        process,
        host,
        port,
        timeout,
        *,
        cancelled = None,
    ):
        return False

    monkeypatch.setattr(ss, "wait_for_own_port", not_ours)

    backend = _FakeBackend(12345, str(model))
    run(ss.after_load(backend, 16))
    assert started, "the peer llama-server was launched"
    assert ss.state().topology == "single"
    assert "did not take" in ss.state().reason
    assert ss.state().peer_process is None
    assert ss.route_base_url(backend) is None


def test_a_replica_inherits_the_settings_that_live_only_in_the_environment(
    cluster, monkeypatch, tmp_path
):
    # llama.cpp's common_arg reads LLAMA_ARG_* itself, and Studio leaves the KV cache types
    # there rather than in argv. ssh carries no environment, so a replica built from the argv
    # alone ran on f16: a different cache from the primary, more memory than the topology was
    # priced against, and a different answer depending on which replica served the request.
    cluster.topology = "replicas"
    monkeypatch.setenv(ss.ENV_PEER, "127.0.0.1")
    monkeypatch.setenv("LLAMA_ARG_CACHE_TYPE_K", " Q8_0 ")
    monkeypatch.setenv("LLAMA_ARG_CACHE_TYPE_V", "q8_0")
    monkeypatch.setenv("LLAMA_ARG_HOST", "0.0.0.0")
    monkeypatch.setenv("UNSLOTH_SECRET_NOT_YOURS", "x")
    model = tmp_path / "m.gguf"
    model.write_bytes(b"x")
    _calls, started = _patch_remote(
        monkeypatch, binary = "$HOME/.unsloth/llama.cpp/build/bin/llama-server"
    )

    run(ss.after_load(_FakeBackend(12345, str(model)), 16))
    assert started, "the peer llama-server was launched"
    peer_argv = started[0].argv
    assert peer_argv[0] == "env"
    # Normalised the way llama.cpp needs: it compares the raw string to ggml_type_name.
    assert "LLAMA_ARG_CACHE_TYPE_K=q8_0" in peer_argv
    assert "LLAMA_ARG_CACHE_TYPE_V=q8_0" in peer_argv
    # The endpoint is the replica's own, and nothing outside the namespace crosses over.
    assert not any(a.startswith("LLAMA_ARG_HOST=") for a in peer_argv)
    assert not any("UNSLOTH_SECRET_NOT_YOURS" in a for a in peer_argv)
    assert peer_argv[peer_argv.index("--host") + 1] == "127.0.0.1"


def test_a_replica_with_nothing_in_the_environment_is_launched_unchanged(
    cluster, monkeypatch, tmp_path
):
    cluster.topology = "replicas"
    monkeypatch.setenv(ss.ENV_PEER, "127.0.0.1")
    for name in list(os.environ):
        if name.startswith("LLAMA_ARG_"):
            monkeypatch.delenv(name, raising = False)
    model = tmp_path / "m.gguf"
    model.write_bytes(b"x")
    _calls, started = _patch_remote(
        monkeypatch, binary = "$HOME/.unsloth/llama.cpp/build/bin/llama-server"
    )

    run(ss.after_load(_FakeBackend(12345, str(model)), 16))
    assert started and started[0].argv[0] != "env"


def test_a_peer_gpu_holding_someone_elses_work_is_left_alone(cluster, monkeypatch, tmp_path):
    # A remote topology is priced against the whole node budget, so a resident training run
    # means one of the two ends up out of memory.
    cluster.peer_gpu_busy = lambda peer, timeout = 25: {
        "busy": True,
        "known": True,
        "processes": [{"pid": 4242, "used_mib": 90000}],
        "reason": "1 compute process(es) resident",
    }
    monkeypatch.setenv(ss.ENV_PEER, "127.0.0.1")
    model = tmp_path / "m.gguf"
    model.write_bytes(b"x")

    cluster.topology = "replicas"
    _calls, started = _patch_remote(
        monkeypatch, binary = "$HOME/.unsloth/llama.cpp/build/bin/llama-server"
    )
    run(ss.after_load(_FakeBackend(12345, str(model)), 16))
    assert not started, "nothing was launched on the busy peer"
    assert ss.state().topology == "single"
    assert "already in use" in ss.state().reason and "4242" in ss.state().reason


def test_a_split_does_not_start_an_rpc_server_on_a_busy_peer_gpu(cluster, monkeypatch, tmp_path):
    cluster.peer_gpu_busy = lambda peer, timeout = 25: {
        "busy": True,
        "known": True,
        "processes": [{"pid": 4242, "used_mib": 90000}],
        "reason": "1 compute process(es) resident",
    }
    cluster.topology = "layer_split"
    monkeypatch.setenv(ss.ENV_PEER, "127.0.0.1")
    model = tmp_path / "m.gguf"
    model.write_bytes(b"x")
    _calls, started = _patch_remote(monkeypatch)

    request = _FakeRequest(str(model))
    assert run(ss.before_load(request, 4)) is request
    assert not started and ss.state().topology == "single"
    assert "already in use" in ss.state().reason


def test_a_peer_gpu_probe_that_cannot_answer_changes_nothing(cluster, monkeypatch, tmp_path):
    # peer_gpu_busy fails CLOSED, which is right for the rsync it was written for and wrong
    # here: refusing on an unanswered probe turns a slow ssh into serving on one node.
    cluster.peer_gpu_busy = lambda peer, timeout = 25: {
        "busy": True,
        "known": False,
        "processes": [],
        "reason": "could not reach the peer to check its GPU",
    }
    cluster.topology = "layer_split"
    monkeypatch.setenv(ss.ENV_PEER, "127.0.0.1")
    model = tmp_path / "m.gguf"
    model.write_bytes(b"x")
    _calls, started = _patch_remote(monkeypatch)

    out = run(ss.before_load(_FakeRequest(str(model)), 4))
    assert started and ss.state().topology == "layer_split"
    assert "--rpc" in (out.llama_extra_args or [])


def test_our_own_peer_process_does_not_count_as_the_gpu_being_busy(cluster, monkeypatch):
    cluster.peer_gpu_busy = lambda peer, timeout = 25: {
        "busy": True,
        "known": True,
        "processes": [{"pid": 777, "used_mib": 50000}],
        "reason": "1 compute process(es) resident",
    }
    assert run(ss.peer_gpu_conflict("127.0.0.1")) is not None
    assert run(ss.peer_gpu_conflict("127.0.0.1", own_pids = [777])) is None


def test_an_rpc_split_configured_by_environment_is_not_treated_as_unsplit(
    cluster, monkeypatch, tmp_path
):
    # LLAMA_ARG_RPC is a supported way to ask for a split and common_arg reads it directly, so
    # it never reaches argv. Calling that server unsplit launches a full peer replica onto a
    # GPU already serving the split.
    cluster.topology = "replicas"
    monkeypatch.setenv(ss.ENV_PEER, "127.0.0.1")
    monkeypatch.setenv("LLAMA_ARG_RPC", "192.168.200.13:50052")
    model = tmp_path / "m.gguf"
    model.write_bytes(b"x")
    _calls, started = _patch_remote(
        monkeypatch, binary = "$HOME/.unsloth/llama.cpp/build/bin/llama-server"
    )

    run(ss.after_load(_FakeBackend(12345, str(model)), 16))
    assert not started, "no replica on a GPU already serving a split"
    assert ss.state().topology == "layer_split"
    assert "user-supplied --rpc" in ss.state().reason

    assert ss.argv_or_env_rpc(["llama-server", "--rpc", "h:1"], env = {}) is True
    assert ss.argv_or_env_rpc(["llama-server", "--rpc=h:1"], env = {}) is True
    assert ss.argv_or_env_rpc(["llama-server"], env = {"LLAMA_ARG_RPC": "h:1"}) is True
    assert ss.argv_or_env_rpc(["llama-server"], env = {"LLAMA_ARG_RPC": "  "}) is False
    assert ss.argv_or_env_rpc(["llama-server"], env = {}) is False


def test_a_caller_who_placed_their_own_rpc_keeps_it(cluster, monkeypatch, tmp_path):
    # Appending a second --rpc plus a managed device order either overrides a working manual
    # split or reaches llama-server as two conflicting placements.
    cluster.topology = "layer_split"
    monkeypatch.setenv(ss.ENV_PEER, "127.0.0.1")
    model = tmp_path / "m.gguf"
    model.write_bytes(b"x")
    _calls, started = _patch_remote(monkeypatch)

    request = _FakeRequest(str(model), llama_extra_args = ["--rpc", "10.0.0.5:50052"])
    out = run(ss.before_load(request, 4))
    assert out is request
    assert not started, "no peer rpc-server started for a placement we do not own"
    assert ss.state().topology == "single"
    assert "caller-supplied --rpc" in ss.state().reason
    assert out.llama_extra_args == ["--rpc", "10.0.0.5:50052"]


def test_the_combined_capability_probe_does_not_run_on_the_event_loop(
    cluster, monkeypatch, tmp_path
):
    # Its first call for a binary is a subprocess.run with a 30 s timeout, and the loop holds
    # every active stream.
    cluster.topology = "layer_split"
    write_fake_llama_server(cluster.bundle / "build" / "bin", _FAKE_HELP_WITH_FLAG)
    monkeypatch.setenv(ss.ENV_PIPELINE_GROUPS, "2")
    model = tmp_path / "m.gguf"
    model.write_bytes(b"x")
    _patch_remote(monkeypatch)

    loop_threads = []
    real = ss.reconcile_split_speculation

    def recording(*args, **kwargs):
        loop_threads.append(threading.current_thread().name)
        return real(*args, **kwargs)

    monkeypatch.setattr(ss, "reconcile_split_speculation", recording)
    main = threading.current_thread().name
    run(ss.before_load(_FakeRequest(str(model)), 4))
    assert loop_threads and all(name != main for name in loop_threads), loop_threads


def test_resident_sidecars_are_charged_to_the_node(cluster, monkeypatch, tmp_path):
    # A projector or a drafter is resident for the whole load. Pricing the base GGUF alone
    # understates the node, and understating is the direction that plans single for something
    # which then does not fit.
    cluster.topology = "single"
    model = tmp_path / "m.gguf"
    model.write_bytes(b"x" * 4096)
    projector = tmp_path / "mmproj.gguf"
    projector.write_bytes(b"x" * 1024)
    drafter = tmp_path / "draft.gguf"
    drafter.write_bytes(b"x" * 2048)
    _patch_remote(monkeypatch)

    run(
        ss.before_load(
            _FakeRequest(
                str(model),
                llama_extra_args = ["--mmproj", str(projector), "-md", str(drafter)],
            ),
            4,
        )
    )
    assert cluster.planner_calls[-1]["model_bytes"] == 4096 + 1024 + 2048


def test_the_adapter_operand_forms_are_taken_apart_not_ignored(tmp_path):
    # llama.cpp takes a comma-separated list with an optional :SCALE per entry. A raw token
    # test sees one string that is neither a path nor a file, so preflight reported success
    # and the peer then failed to launch for a sidecar it did not have.
    a = tmp_path / "a.gguf"
    a.write_bytes(b"x" * 10)
    b = tmp_path / "b.gguf"
    b.write_bytes(b"x" * 20)

    assert ss.sidecar_operand_paths("/a.gguf,/b.gguf") == ["/a.gguf", "/b.gguf"]
    assert ss.sidecar_operand_paths("/a.gguf:0.5") == ["/a.gguf"]
    assert ss.sidecar_operand_paths("/a.gguf:0.5,/b.gguf:1") == ["/a.gguf", "/b.gguf"]
    # Not every colon is a scale.
    assert ss.sidecar_operand_paths("/models/a:b.gguf") == ["/models/a:b.gguf"]

    args = ["--lora", f"{a},{b}", "--lora-scaled", f"{a}:0.5"]
    assert ss.sidecar_files(args) == [str(a), str(b), str(a)]
    assert ss.sidecar_bytes(args) == (10 + 20 + 10, False)

    # Relative operands resolve against the primary's working directory, as llama-server does.
    monkey_cwd = str(tmp_path)
    assert ss.sidecar_files(["--lora", "a.gguf"], cwd = monkey_cwd) == [str(a)]

    # And a launch that names them is preflighted for them.
    model = tmp_path / "m.gguf"
    model.write_bytes(b"x")
    files = ss.launch_files(["llama-server", "--lora", f"{a},{b}"], str(model))
    assert str(a) in files and str(b) in files


def test_a_sidecar_that_cannot_be_sized_does_not_cost_the_topology(cluster, monkeypatch, tmp_path):
    # A drafter the backend has yet to download is ordinary. Answering "size unknown" to it
    # would mean single, the one topology that cannot hold a large model.
    cluster.topology = "layer_split"
    monkeypatch.setenv(ss.ENV_PEER, "127.0.0.1")
    model = tmp_path / "m.gguf"
    model.write_bytes(b"x" * 4096)
    _calls, started = _patch_remote(monkeypatch)

    out = run(
        ss.before_load(
            _FakeRequest(str(model), llama_extra_args = ["-md", "/not/downloaded/yet.gguf"]), 4
        )
    )
    assert cluster.planner_calls[-1]["model_bytes"] == 4096
    assert started and ss.state().topology == "layer_split"
    assert "--rpc" in (out.llama_extra_args or [])


def test_a_forced_split_does_not_apply_to_a_load_that_is_not_a_gguf(cluster, monkeypatch, tmp_path):
    # before_load runs ahead of model classification. Forcing the topology anyway started an
    # rpc-server for a Transformers load, which then succeeded, after_load returned early
    # because no llama backend was loaded, and nothing ever detached the peer: the RPC port
    # stayed occupied and the status stayed on layer_split.
    monkeypatch.setenv(ss.ENV_TOPOLOGY, "layer_split")
    monkeypatch.setenv(ss.ENV_PEER, "127.0.0.1")
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "empty-cache"))
    cluster.topology = "single"
    _calls, started = _patch_remote(monkeypatch)
    monkeypatch.setattr(ss, "remote_gguf_size_bytes", lambda path, variant, token = None: None)

    request = _FakeRequest("meta-llama/Llama-3.1-8B-Instruct")
    assert run(ss.before_load(request, 4)) is request
    assert not started, "no rpc-server for weights that will never touch it"
    assert ss.state().topology == "single"

    # A GGUF load with the same environment is still forced.
    model = tmp_path / "m.gguf"
    model.write_bytes(b"x")
    _calls, started = _patch_remote(monkeypatch)
    out = run(ss.before_load(_FakeRequest(str(model)), 4))
    assert started and ss.state().topology == "layer_split"
    assert "--rpc" in (out.llama_extra_args or [])


def test_what_counts_as_a_gguf_load_before_the_loader_says(tmp_path):
    model = tmp_path / "m.gguf"
    model.write_bytes(b"x")
    assert ss.looks_like_a_gguf_load("repo/x", None, str(model)) is True
    assert ss.looks_like_a_gguf_load("repo/x", "Q4_K_M", None) is True
    assert ss.looks_like_a_gguf_load("/models/m.GGUF", None, None) is True
    assert ss.looks_like_a_gguf_load("meta-llama/Llama-3.1-8B-Instruct", None, None) is False


def test_the_capability_probe_follows_the_binary_the_loader_will_launch(monkeypatch, tmp_path):
    # A flag decided against one build and passed to another fails every affected load.
    chosen = tmp_path / "custom" / "llama-server"
    chosen.parent.mkdir(parents = True)
    write_fake_llama_server(chosen.parent, _FAKE_HELP_WITH_FLAG)
    other = tmp_path / "bundle" / "llama-server"
    other.parent.mkdir(parents = True)
    write_fake_llama_server(other.parent, _FAKE_HELP_WITHOUT_FLAG)

    monkeypatch.setattr(ss, "llama_server_binary", lambda: str(chosen))
    assert ss.llama_server_supports(ss.PIPELINE_GROUPS_FLAG) is True
    monkeypatch.setattr(ss, "llama_server_binary", lambda: str(other))
    assert ss.llama_server_supports(ss.PIPELINE_GROUPS_FLAG) is False


def test_every_tool_loop_round_names_the_same_conversation():
    # Once rolling compaction rewrites the first user turn, successive rounds of one agent run
    # derive different fallback hashes and jump between replicas, discarding the prefix KV
    # that sticky routing exists to preserve.
    source = (Path(ss.__file__).resolve().parent / "llama_cpp.py").read_text()
    start = source.index("def generate_chat_completion_with_tools(")
    body = source[start:]
    for marker in ("payload = {", "stream_payload = {"):
        assert marker in body, marker
    # Each payload the tool loop sends is tagged.
    assert body.count("tag_conversation(payload, thread_id)") >= 1
    assert body.count("tag_conversation(stream_payload, thread_id)") >= 1


def test_a_peer_running_a_different_llama_server_build_is_not_made_a_replica(
    cluster, monkeypatch, tmp_path
):
    # The replica is launched from the primary's complete argv, so an older peer build rejects
    # a flag the primary was given and the advertised replica never comes up, and two builds
    # that both start can answer the same request differently.
    cluster.topology = "replicas"
    monkeypatch.setenv(ss.ENV_PEER, "127.0.0.1")
    model = tmp_path / "m.gguf"
    model.write_bytes(b"x")

    monkeypatch.setattr(ss, "local_llama_server_version", lambda binary = None: "6109 (aaaaaaa)")
    real_ssh_calls: list = []

    def _remote(
        peer,
        remote,
        timeout = 20.0,
    ):
        real_ssh_calls.append(remote)

    _calls, started = _patch_remote(
        monkeypatch, binary = "$HOME/.unsloth/llama.cpp/build/bin/llama-server"
    )
    outer = ss.ssh_run

    async def versioned_ssh(
        peer,
        remote,
        timeout = 20.0,
    ):
        if "--version" in remote:
            return 0, "version: 5000 (bbbbbbb)\nbuilt with gcc\n", ""
        return await outer(peer, remote, timeout = timeout)

    monkeypatch.setattr(ss, "ssh_run", versioned_ssh)
    run(ss.after_load(_FakeBackend(12345, str(model)), 16))
    assert not started, "no replica from a different build"
    assert ss.state().topology == "single"
    assert "different llama-server build" in ss.state().reason
    assert "6109 (aaaaaaa)" in ss.state().reason and "5000 (bbbbbbb)" in ss.state().reason

    # The same build is admitted.
    async def matching_ssh(
        peer,
        remote,
        timeout = 20.0,
    ):
        if "--version" in remote:
            return 0, "version: 6109 (aaaaaaa)\n", ""
        return await outer(peer, remote, timeout = timeout)

    monkeypatch.setattr(ss, "ssh_run", matching_ssh)
    run(ss.after_load(_FakeBackend(12345, str(model)), 16))
    assert started and ss.state().topology == "replicas"


def test_an_unreadable_build_is_not_evidence_of_a_mismatch(cluster, monkeypatch, tmp_path):
    # A version that cannot be read on either end is not evidence, and refusing on it would
    # drop the pair to one node whenever a probe is slow.
    cluster.topology = "replicas"
    monkeypatch.setenv(ss.ENV_PEER, "127.0.0.1")
    model = tmp_path / "m.gguf"
    model.write_bytes(b"x")
    _calls, started = _patch_remote(
        monkeypatch, binary = "$HOME/.unsloth/llama.cpp/build/bin/llama-server"
    )
    monkeypatch.setattr(ss, "local_llama_server_version", lambda binary = None: None)

    run(ss.after_load(_FakeBackend(12345, str(model)), 16))
    assert started and ss.state().topology == "replicas"

    assert ss.parse_llama_server_version("version: 6109 (a1b2c3d)") == "6109 (a1b2c3d)"
    assert ss.parse_llama_server_version("no version here") is None


def test_a_replacement_load_does_not_look_like_an_unload_to_the_supervisor(cluster, monkeypatch):
    # The loader clears _process before its download and preparation phase, so a replacement
    # for an already-split model shows that transient state for the whole of it. Tearing down
    # there kills the rpc-server the replacement has already been configured to use.
    st = ss.state()
    detached = []

    async def fake_detach():
        detached.append(1)

    monkeypatch.setattr(st, "detach", fake_detach)
    monkeypatch.setattr(ss, "SUPERVISOR_INTERVAL_S", 0.01)
    st.attached_backend = SimpleNamespace(_process = None, _port = 8080, _healthy = True)
    st.attached_port = 8080

    async def scenario():
        st.load_in_progress = True
        supervisor = asyncio.ensure_future(st._supervise())
        await asyncio.sleep(0.1)
        assert not detached, "a load in progress is not an unload"
        # The load ends without a server: now it really is gone.
        st.load_in_progress = False
        await asyncio.sleep(0.1)
        supervisor.cancel()
        try:
            await supervisor
        except (asyncio.CancelledError, Exception):
            pass
        assert detached == [1]

    run(scenario())


def test_an_explicit_gpu_pin_does_not_get_a_split_with_the_wrong_device_order(
    cluster, monkeypatch, tmp_path
):
    # The backend strips every --device pass-through when gpu_ids is set, so a split would
    # launch without RPC0,CUDA0 and llama.cpp's CUDA-first default would put the output layer
    # and the logits on the peer: the measured slow path, silently.
    cluster.topology = "layer_split"
    monkeypatch.setenv(ss.ENV_PEER, "127.0.0.1")
    model = tmp_path / "m.gguf"
    model.write_bytes(b"x")
    _calls, started = _patch_remote(monkeypatch)

    request = _FakeRequest(str(model))
    request.gpu_ids = [0]
    out = run(ss.before_load(request, 4))
    assert out is request
    assert not started and ss.state().topology == "single"
    assert "GPU pin" in ss.state().reason

    # Without the pin the split is planned as before.
    _calls, started = _patch_remote(monkeypatch)
    out = run(ss.before_load(_FakeRequest(str(model)), 4))
    assert started and "--rpc" in (out.llama_extra_args or [])


def test_a_generated_chat_template_is_copied_rather_than_demanded(cluster, monkeypatch, tmp_path):
    # A chat-template override is written to a uniquely named file per load and named in argv.
    # The peer cannot already have a file this process just created, so demanding it meant
    # those loads never got replicas at all.
    cluster.topology = "replicas"
    monkeypatch.setenv(ss.ENV_PEER, "127.0.0.1")
    model = tmp_path / "m.gguf"
    model.write_bytes(b"x")
    template = Path(tempfile.gettempdir()) / f"unsloth_chat_template_{os.getpid()}.jinja"
    template.write_text("{% if x %}'quotes' and {braces}{% endif %}", encoding = "utf-8")

    try:
        _calls, started = _patch_remote(
            monkeypatch,
            binary = "$HOME/.unsloth/llama.cpp/build/bin/llama-server",
            model_present = True,
        )
        backend = _FakeBackend(12345, str(model))
        backend._process.args = list(backend._process.args) + [
            "--chat-template-file",
            str(template),
        ]
        run(ss.after_load(backend, 16))
        assert started and ss.state().topology == "replicas"
        copied = [c for c in _calls if "base64 -d" in c]
        assert len(copied) == 1 and str(template) in copied[0]
        # And it is NOT in the file-presence check, which it could never satisfy.
        checks = [c for c in _calls if c.startswith("stat -c")]
        assert checks and str(template) not in checks[0]
    finally:
        template.unlink(missing_ok = True)


def test_only_files_this_process_wrote_are_copied(tmp_path):
    outside = tmp_path / "weights.gguf"
    outside.write_bytes(b"x")
    inside = Path(tempfile.gettempdir()) / f"unsloth_chat_template_probe_{os.getpid()}.jinja"
    inside.write_text("x", encoding = "utf-8")
    try:
        found = ss.generated_launch_files([str(outside), str(inside), "/nope/missing.jinja"])
        assert found == [str(inside)]
    finally:
        inside.unlink(missing_ok = True)


def test_windows_picks_the_exe_and_refuses_a_file_it_cannot_run(monkeypatch, tmp_path):
    # os.access(path, os.X_OK) is true for ANY existing file on Windows, so it guarded nothing,
    # and the candidate list put the extensionless names first for every platform, so a stray
    # extensionless file beat the real .exe and was returned as the binary.
    stray = tmp_path / "ggml-rpc-server"
    stray.write_text("not a program", encoding = "utf-8")
    real = tmp_path / "ggml-rpc-server.exe"
    real.write_bytes(b"MZ")
    server = tmp_path / "llama-server.exe"
    server.write_bytes(b"MZ")
    # Held as strings: once os.name says nt, Path() builds a WindowsPath this system refuses.
    stray, real, server = str(stray), str(real), str(server)
    missing = os.path.join(str(tmp_path), "missing.exe")
    directory = str(tmp_path)

    monkeypatch.setattr(ss.os, "name", "nt")
    monkeypatch.setenv("PATHEXT", ".COM;.EXE;.BAT;.CMD")

    names = ss.rpc_server_names()
    assert names[0] == "ggml-rpc-server.com" and "ggml-rpc-server.exe" in names
    assert "ggml-rpc-server" not in names, "an extensionless file is not runnable on Windows"

    assert ss.is_executable_file(stray) is False
    assert ss.is_executable_file(real) is True
    assert ss.is_executable_file(missing) is False
    assert ss.is_executable_file(directory) is False

    # And the resolver picks the .exe even though the stray file shares the stem.
    monkeypatch.setattr(ss, "llama_server_binary", lambda: server)
    assert ss.rpc_server_binary() == real


def test_posix_still_answers_on_the_bit_and_the_bare_name(monkeypatch, tmp_path):
    assert ss.os.name != "nt", "this test describes the POSIX side"
    assert ss.executable_suffixes() == ("",)
    assert ss.rpc_server_names() == ("ggml-rpc-server", "rpc-server")

    plain = tmp_path / "ggml-rpc-server"
    plain.write_text("#!/bin/sh\n", encoding = "utf-8")
    assert ss.is_executable_file(plain) is False, "no execute bit"
    plain.chmod(0o755)
    assert ss.is_executable_file(plain) is True

    server = tmp_path / "llama-server"
    server.write_text("#!/bin/sh\n", encoding = "utf-8")
    server.chmod(0o755)
    monkeypatch.setattr(ss, "llama_server_binary", lambda: str(server))
    assert ss.rpc_server_binary() == str(plain)


def test_the_replica_never_inherits_an_env_var_the_primary_refuses():
    """The primary scrubs DENIED_ENV_VARS from a COPY of the environment, so they are still in
    os.environ when replica_env reads it. Inheriting one puts the peer in a configuration the
    primary would not run: an api key or TLS makes the router's plain-HTTP health probe fail
    forever, and LLAMA_ARG_MODEL points the replica at a different model that answers fine."""
    from core.inference.llama_server_args import DENIED_ENV_VARS

    source = {
        "LLAMA_ARG_CACHE_TYPE_K": "Q8_0",  # the whole point of replica_env: must cross
        "LLAMA_ARG_HOST": "10.0.0.1",  # endpoint, deliberately not inherited
        "LLAMA_ARG_PORT": "9999",
        "PATH": "/usr/bin",  # outside the namespace
    }
    for name in DENIED_ENV_VARS:
        if name.startswith("LLAMA_ARG_"):
            source[name] = "x"

    out = ss.replica_env(source)
    assert out == {"LLAMA_ARG_CACHE_TYPE_K": "q8_0"}, out
    leaked = sorted(n for n in out if n in set(DENIED_ENV_VARS))
    assert not leaked, f"the replica would launch with denied settings: {leaked}"


def test_the_replica_gets_sidecar_paths_the_peer_can_actually_open(tmp_path):
    """The preflight resolves sidecars against this process's cwd; the peer resolves a bare
    name against its own login directory. They have to agree, or preflight passes and the
    launch then dies on a file it just confirmed."""
    argv = [
        "/bundle/llama-server",
        "-m",
        "/models/m.gguf",
        "--lora",
        "adapter.gguf",
        "--control-vector-scaled=cv.gguf:0.5,/abs/other.gguf:2",
        "--mmproj",
        "/already/abs.gguf",
        "--host",
        "127.0.0.1",
        "--port",
        "1",
    ]
    out = ss.replica_argv(
        argv,
        binary = "/bundle/llama-server",
        host = "10.0.0.2",
        port = 9,
        cwd = "/work",
    )
    assert "/work/adapter.gguf" in out, out
    assert "--control-vector-scaled=/work/cv.gguf:0.5,/abs/other.gguf:2" in out, out
    assert "/already/abs.gguf" in out, "an absolute path must be left alone"
    assert "adapter.gguf" not in out, "the bare relative name must not reach the peer"
    # the scale and list forms survive intact, and the endpoint is still repointed
    assert out[-4:] == ["--host", "10.0.0.2", "--port", "9"], out[-4:]


def test_a_missing_shard_reports_the_size_unknown_rather_than_short(tmp_path):
    """Undercounting is the one error this must not make: a short total prices a 120 GiB model
    as 30, plans `single`, and is found out as an OOM after the rest of the shards arrive.
    None is handled -- before_load asks the hub instead."""
    first = tmp_path / "m-00001-of-00003.gguf"
    first.write_bytes(b"a" * 100)
    (tmp_path / "m-00002-of-00003.gguf").write_bytes(b"b" * 100)
    # third shard deliberately absent
    assert ss.gguf_size_bytes(str(first)) is None

    (tmp_path / "m-00003-of-00003.gguf").write_bytes(b"c" * 100)
    assert ss.gguf_size_bytes(str(first)) == 300


def test_a_projector_that_arrives_only_through_the_environment_still_costs_the_groups(monkeypatch):
    """llama.cpp's common_arg reads LLAMA_ARG_MMPROJ itself, so a projector can reach the server
    without ever appearing in argv. A refusal that only read argv saw a clean launch and enabled
    the groups anyway -- the same one-setting-two-routes shape as the sidecar paths."""
    monkeypatch.delenv("LLAMA_ARG_MMPROJ", raising = False)
    monkeypatch.delenv("LLAMA_ARG_MMPROJ_URL", raising = False)
    assert ss.extra_args_refuse_pipeline_groups([]) is None
    assert ss.extra_args_refuse_pipeline_groups(["--mmproj", "p.gguf"]) == "--mmproj"

    monkeypatch.setenv("LLAMA_ARG_MMPROJ", "/models/p.gguf")
    assert ss.extra_args_refuse_pipeline_groups([]) == "LLAMA_ARG_MMPROJ"
    monkeypatch.delenv("LLAMA_ARG_MMPROJ")
    monkeypatch.setenv("LLAMA_ARG_MMPROJ_URL", "https://example/p.gguf")
    assert ss.extra_args_refuse_pipeline_groups([]) == "LLAMA_ARG_MMPROJ_URL"
    # an empty value is not a projector
    monkeypatch.setenv("LLAMA_ARG_MMPROJ_URL", "  ")
    assert ss.extra_args_refuse_pipeline_groups([]) is None


def test_the_env_route_absolutises_the_same_sidecar_paths_the_argv_route_does():
    """The peer resolves a bare name against its own login directory whichever route carried it,
    so fixing only replica_argv fixed only the callers that happened to use argv."""
    source = {
        "LLAMA_ARG_MMPROJ": "proj.gguf",
        "LLAMA_ARG_SPEC_DRAFT_MODEL": "draft.gguf",
        "LLAMA_ARG_CHAT_TEMPLATE_FILE": "tmpl.jinja",
        "LLAMA_ARG_MMPROJ_URL": "https://example/p.gguf",  # a URL, must be left alone
        "LLAMA_ARG_SPEC_DRAFT_HF_REPO": "org/repo",  # a repo id, must be left alone
        "LLAMA_ARG_CACHE_TYPE_K": "Q8_0",
    }
    out = ss.replica_env(source, cwd = "/work")
    assert out["LLAMA_ARG_MMPROJ"] == "/work/proj.gguf"
    assert out["LLAMA_ARG_SPEC_DRAFT_MODEL"] == "/work/draft.gguf"
    assert out["LLAMA_ARG_CHAT_TEMPLATE_FILE"] == "/work/tmpl.jinja"
    assert out["LLAMA_ARG_MMPROJ_URL"] == "https://example/p.gguf", "a URL is not a path"
    assert out["LLAMA_ARG_SPEC_DRAFT_HF_REPO"] == "org/repo", "a repo id is not a path"
    assert out["LLAMA_ARG_CACHE_TYPE_K"] == "q8_0", "the lowering still applies"
    # already absolute stays put
    assert ss.replica_env({"LLAMA_ARG_MMPROJ": "/abs/p.gguf"}, cwd = "/work") == {
        "LLAMA_ARG_MMPROJ": "/abs/p.gguf"
    }


def test_extras_are_inherited_on_exact_requested_identity_and_never_on_a_shared_stem(monkeypatch):
    """The Spark helper runs before anything is resolved, so it compares what the CALLER asked
    for against what the previous caller asked for. Requested against requested needs no
    heuristic; the substring fallback it replaces is what let `org/qwen-7b` inherit
    `org/qwen-7b-instruct`'s `--lora`, `--mmproj` and `--model-draft`.

    On a layer split this is the ONLY path by which extras reach the launch, because
    `_with_rpc_args` materialises `llama_extra_args` and a non-None field makes the strict
    resolver skip inheritance -- so both directions matter: inheriting wrongly launches another
    model's adapter, and failing to inherit silently drops the user's own."""
    import routes.inference as ri

    class _Backend:
        def __init__(self, extras, rid, rvar):
            self.extra_args = extras
            self.extra_args_requested_source = (rid, rvar)
            self.extra_args_source = ("some/resolved-path", rvar)

    class _Req:
        def __init__(
            self,
            mp,
            gv = None,
            lea = None,
        ):
            self.model_path, self.gguf_variant, self.llama_extra_args = mp, gv, lea

    def run(
        stored_id,
        stored_var,
        req_path,
        req_var,
        extras = ("--lora", "a.gguf"),
    ):
        monkeypatch.setattr(
            ri, "get_llama_cpp_backend", lambda: _Backend(list(extras), stored_id, stored_var)
        )
        return ri._spark_inherited_extra_args(_Req(req_path, req_var))

    # the case the deleted substring fallback existed to serve: a hub id whose resolved stem
    # differs. Comparing requested against requested makes the resolved stem irrelevant.
    assert run("unsloth/Qwen3-8B-GGUF", "UD-Q4_K_XL", "unsloth/Qwen3-8B-GGUF", "UD-Q4_K_XL") == [
        "--lora",
        "a.gguf",
    ]
    # the bug, both directions
    assert run("org/qwen-7b-instruct", None, "org/qwen-7b", None) is None
    assert run("org/qwen-7b", None, "org/qwen-7b-instruct", None) is None
    # a user's own sidecar survives a warm reload of a genuinely identical model
    assert run("org/m", "Q4", "org/m", "Q4", extras = ("--lora", "/abs/adapter.gguf")) == [
        "--lora",
        "/abs/adapter.gguf",
    ]
    # the variant is part of the identity: two quants are different files with different sidecars
    assert run("org/m", "Q4", "org/m", "Q8") is None
    assert run("Org/M", "Q4", "  org/m  ", "q4") == ["--lora", "a.gguf"]
    # nothing recorded -> no guessing, which is what this replaced
    assert run(None, None, "org/m", None) is None
    # an explicit field always wins; inheritance is only for a request that omits it
    monkeypatch.setattr(
        ri, "get_llama_cpp_backend", lambda: _Backend(["--lora", "a"], "org/m", None)
    )
    assert ri._spark_inherited_extra_args(_Req("org/m", None, lea = [])) is None


def test_a_peer_holding_a_different_file_at_the_same_path_does_not_become_a_replica(
    cluster, monkeypatch, tmp_path
):
    """Existence is not identity. A replica is interchangeable with the primary only if it is
    serving the SAME weights. A peer with a stale quant at the same path passes `test -f`, and
    the router then alternates requests between two different models while the binary, argv and
    environment parity checks all report a matched pair -- so one conversation gets
    model-dependent answers with nothing anywhere reporting a problem."""
    cluster.topology = "replicas"
    monkeypatch.setenv(ss.ENV_PEER, "127.0.0.1")
    model = tmp_path / "m.gguf"
    model.write_bytes(b"x" * 4096)

    _calls, started = _patch_remote(
        monkeypatch,
        binary = "$HOME/.unsloth/llama.cpp/build/bin/llama-server",
        model_present = True,
        model_stale = True,
    )
    run(ss.after_load(_FakeBackend(12345, str(model)), 16))

    assert ss.state().topology == "single", "a mismatched peer must not be put in rotation"
    assert not started, "and no replica should have been launched"
    assert "does not have the same" in ss.state().reason
    assert str(model) in ss.state().reason, "the reason has to name the file that disagrees"
    # the probe really did ask for identity, not just presence
    assert any(c.startswith("stat -c") for c in _calls)


def test_an_unknown_mtp_verdict_is_still_turned_off_on_a_wide_split():
    """On the FIRST load of an uncached MTP-capable GGUF the header is not on disk, so mtp_plan
    says `unknown`. Leaving that undecided let the backend switch its own MTP on after the
    download, at a width where the measured rule says every split is faster with no drafter."""
    for verdict in ("enabled", "unknown"):
        mtp = {"mtp": verdict, "reason": "before", "request": {"spec_draft_n_max": 3}}
        groups = {"pipeline_groups": 0, "requested_slots": ss.SPLIT_MTP_OFF_ROWS}
        ss.reconcile_split_speculation(groups, mtp)
        assert mtp["mtp"] == ss.MTP_OFF_FOR_SPLIT_ROWS, verdict
        assert mtp["request"]["speculative_type"] == "off", verdict
        assert "spec_draft_n_max" not in mtp["request"], verdict

    # a drafter the CALLER asked for is never taken away
    mtp = {"mtp": "user override", "reason": "caller", "request": {}}
    groups = {"pipeline_groups": 0, "requested_slots": ss.SPLIT_MTP_OFF_ROWS}
    ss.reconcile_split_speculation(groups, mtp)
    assert mtp["mtp"] == "user override"

    # and below the threshold the unknown case is left alone to be decided on the real header
    mtp = {"mtp": "unknown", "reason": "before", "request": {}}
    groups = {"pipeline_groups": 0, "requested_slots": 1}
    ss.reconcile_split_speculation(groups, mtp)
    assert mtp["mtp"] == "unknown"


def test_a_replica_is_given_a_model_load_deadline_not_the_rpc_servers():
    """A replica is a llama-server: it reads the whole model before it binds. Reusing the
    rpc-server's 20 s -- the case that constant's own comment excludes -- killed a peer that was
    loading a near-node-capacity GGUF and fell back to one node while it was perfectly healthy."""
    assert ss.PEER_REPLICA_START_TIMEOUT_S > ss.PEER_START_TIMEOUT_S
    # matched to the primary's own readiness budget, so both ends get the same time
    assert ss.PEER_REPLICA_START_TIMEOUT_S == 600.0
    import inspect

    src = inspect.getsource(ss.SparkServing._start_replicas)
    assert "PEER_REPLICA_START_TIMEOUT_S" in src
    assert "PEER_START_TIMEOUT_S)" not in src, "the replica must not use the rpc-server deadline"


def test_a_replica_does_not_get_back_a_setting_the_primary_scrubbed(cluster, monkeypatch, tmp_path):
    # The primary is spawned with a CONDITIONALLY sanitized copy of the environment, and the
    # scrubs go well past DENIED_ENV_VARS: disable_vision drops LLAMA_ARG_MMPROJ and
    # _MMPROJ_URL, a CPU-forced replay drops LLAMA_ARG_OVERRIDE_TENSOR and the tensor split,
    # the memory fit drops _CTX_SIZE, _THREADS, _KV_UNIFIED, _N_PARALLEL and _FLASH_ATTN.
    # Rebuilding the replica's environment from os.environ handed every one of them back to
    # the peer, so the peer loaded a projector the primary had refused and the router then
    # alternated between two servers that were not the same server.
    cluster.topology = "replicas"
    monkeypatch.setenv(ss.ENV_PEER, "127.0.0.1")
    monkeypatch.setenv("LLAMA_ARG_MMPROJ", str(tmp_path / "mmproj.gguf"))
    monkeypatch.setenv("LLAMA_ARG_CTX_SIZE", "131072")
    monkeypatch.setenv("LLAMA_ARG_CACHE_TYPE_K", "q8_0")
    model = tmp_path / "m.gguf"
    model.write_bytes(b"x")
    _calls, started = _patch_remote(
        monkeypatch, binary = "$HOME/.unsloth/llama.cpp/build/bin/llama-server"
    )

    backend = _FakeBackend(12345, str(model))
    # What the primary was ACTUALLY spawned with: vision off, and a context the fit shrank.
    backend._launched_env = {
        "PATH": os.environ.get("PATH", ""),
        "LLAMA_ARG_CACHE_TYPE_K": "q8_0",
        "LLAMA_ARG_CTX_SIZE": "8192",
    }
    backend.launched_env = dict(backend._launched_env)

    run(ss.after_load(backend, 16))
    assert started, "the peer llama-server was launched"
    peer_argv = started[0].argv
    assert not any(
        "LLAMA_ARG_MMPROJ" in a for a in peer_argv
    ), "the peer got back a projector the primary was launched without"
    assert "LLAMA_ARG_CTX_SIZE=8192" in peer_argv, "the peer must run the primary's context"
    assert "LLAMA_ARG_CTX_SIZE=131072" not in peer_argv
    # Still carried across, so this is not just dropping everything.
    assert "LLAMA_ARG_CACHE_TYPE_K=q8_0" in peer_argv


def test_a_backend_that_cannot_say_still_gets_the_old_environment_behaviour(
    cluster, monkeypatch, tmp_path
):
    # Not every caller has a recorded launch environment (an older backend, a partial double).
    # Falling back to os.environ is what this did before and is strictly better than nothing.
    cluster.topology = "replicas"
    monkeypatch.setenv(ss.ENV_PEER, "127.0.0.1")
    monkeypatch.setenv("LLAMA_ARG_CACHE_TYPE_V", "q8_0")
    model = tmp_path / "m.gguf"
    model.write_bytes(b"x")
    _calls, started = _patch_remote(
        monkeypatch, binary = "$HOME/.unsloth/llama.cpp/build/bin/llama-server"
    )

    backend = _FakeBackend(12345, str(model))
    assert getattr(backend, "launched_env", None) is None
    run(ss.after_load(backend, 16))
    assert "LLAMA_ARG_CACHE_TYPE_V=q8_0" in started[0].argv


def test_the_hub_is_asked_with_the_requests_own_token(monkeypatch):
    # A private or gated repo answers an anonymous query with an authorization failure, which
    # is swallowed as "size unknown" and plans single -- and then the authenticated loader
    # downloads the model anyway, so a repo larger than one Spark reaches the one topology
    # that cannot hold it. The failure looks exactly like being offline.
    import sys
    import types

    seen = []

    class _Variant:
        filename, size_bytes = "model-Q4_K_M.gguf", 200 * 1024**3

    module = types.ModuleType("utils.models.model_config")

    def _list(repo_id, hf_token = None):
        seen.append((repo_id, hf_token))
        if not hf_token:
            raise PermissionError("gated repo")
        return ([_Variant()], False)

    module.list_gguf_variants = _list
    monkeypatch.setitem(sys.modules, "utils.models.model_config", module)

    assert ss.remote_gguf_size_bytes("private/Huge-GGUF", None) is None
    assert ss.remote_gguf_size_bytes("private/Huge-GGUF", None, "hf_tok") == 200 * 1024**3
    assert seen == [("private/Huge-GGUF", None), ("private/Huge-GGUF", "hf_tok")]


def test_the_planner_forwards_the_requests_token_to_the_hub(cluster, monkeypatch, tmp_path):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    write_fake_llama_server(cluster.bundle / "build" / "bin", _FAKE_HELP_WITH_FLAG)
    _patch_remote(monkeypatch)

    seen = []
    monkeypatch.setattr(
        ss,
        "remote_gguf_size_bytes",
        lambda path, variant, token = None: seen.append(token) or 200 * 1024**3,
    )
    cluster.topology = "layer_split"
    request = _FakeRequest("private/Huge-GGUF")
    request.hf_token = "hf_tok"
    run(ss.before_load(request, 4))
    assert seen == ["hf_tok"], "the request's token has to reach the sizing query"


def test_a_cancel_during_replica_startup_stops_waiting_and_leaves_nothing_behind(
    cluster, monkeypatch, tmp_path
):
    # A scoped unload arriving mid-load only SETS the event. Nothing here read it, so a cancel
    # during replica startup was ignored for up to PEER_REPLICA_START_TIMEOUT_S -- 600 seconds
    # -- while the unload it triggered had already returned "unloaded". The request then went
    # on to attach a peer to a backend the caller had been told was gone.
    import threading

    cluster.topology = "replicas"
    monkeypatch.setenv(ss.ENV_PEER, "127.0.0.1")
    model = tmp_path / "m.gguf"
    model.write_bytes(b"x")
    # port_opens is True on purpose: the wait would SUCCEED if the cancel were not read, so
    # this fails against the old code by attaching rather than by timing out.
    _calls, started = _patch_remote(monkeypatch)

    cancel = threading.Event()
    cancel.set()

    backend = _FakeBackend(12345, str(model))
    run(ss.after_load(backend, 16, cancel_event = cancel))

    assert started, "the peer llama-server was launched before the cancel was noticed"
    assert ss.state().topology == "single"
    assert ss.state().peer_process is None, "a cancelled attach must not leave the peer running"
    assert ss.state().router is None
    assert ss.route_base_url(backend) is None
    assert "cancel" in ss.state().reason.lower()
    # The event does not outlive the load it belongs to.
    assert ss.state()._cancel_event is None


def test_rail_discovery_is_not_walked_again_on_every_status_poll(cluster, monkeypatch):
    # enabled() reaches peer_ip_for(), which walks sysfs and forks `ip`: 16.2 ms, measured, and
    # the status endpoint is declared async, so every poll spent it on the event loop and
    # stalled whatever was streaming beside it. current_topology already sidesteps this; the
    # dedicated status endpoint cannot, because answering "not enabled" is the point of it.
    calls = []
    real = cluster.peer_ip_for

    def counting(*a, **k):
        calls.append(1)
        return real(*a, **k)

    cluster.peer_ip_for = counting
    ss.reset_peer_discovery_cache()

    assert ss.enabled()
    assert len(calls) == 1
    for _ in range(20):
        assert ss.enabled()
    assert len(calls) == 1, "the rails were re-walked on a repeat poll"

    # Not cached for the process lifetime: a cable plugged in later is still picked up.
    ss.reset_peer_discovery_cache()
    assert ss.enabled()
    assert len(calls) == 2


def test_the_size_is_taken_from_the_variant_the_loader_would_open(monkeypatch):
    # Both listers sort largest first, so variants[0] is a repo's BF16 rather than its default.
    # The load calls _pick_best_gguf, which prefers UD-Q4. Sizing one file and opening another
    # is not a rounding error: a repo whose BF16 exceeds the pair while its Q4 needs two nodes
    # and fits reads as "no topology fits", and the Q4 is left to a single node that OOMs.
    import sys
    import types

    class _Variant:
        def __init__(self, filename, quant, size_bytes):
            self.filename, self.quant, self.size_bytes = filename, quant, size_bytes

    bf16 = _Variant("Qwen3-8B-BF16.gguf", "BF16", 300 * 1024**3)
    q4 = _Variant("Qwen3-8B-UD-Q4_K_XL.gguf", "UD-Q4_K_XL", 120 * 1024**3)

    module = types.ModuleType("utils.models.model_config")
    module.list_gguf_variants = lambda repo_id, hf_token = None: ([bf16, q4], False)
    module._pick_best_gguf = lambda names: next(
        (n for n in names if "UD-Q4_K_XL" in n), names[0] if names else None
    )
    monkeypatch.setitem(sys.modules, "utils.models.model_config", module)

    assert ss.remote_gguf_size_bytes("unsloth/Qwen3-8B-GGUF", None) == 120 * 1024**3
    # A caller who names a variant still gets exactly that one, largest or not.
    assert ss.remote_gguf_size_bytes("unsloth/Qwen3-8B-GGUF", "BF16") == 300 * 1024**3


def test_a_repo_whose_picker_cannot_answer_still_gets_a_size(monkeypatch):
    # The picker is asked, not depended on: an import failure or an unrecognised naming scheme
    # falls back to the old answer rather than to no answer, which would plan `single`.
    import sys
    import types

    class _Variant:
        filename, quant, size_bytes = "weights.gguf", "F16", 42

    module = types.ModuleType("utils.models.model_config")
    module.list_gguf_variants = lambda repo_id, hf_token = None: ([_Variant()], False)

    def _boom(names):
        raise RuntimeError("no picker here")

    module._pick_best_gguf = _boom
    monkeypatch.setitem(sys.modules, "utils.models.model_config", module)
    assert ss.remote_gguf_size_bytes("org/repo", None) == 42


def test_a_relative_chat_template_is_absolutised_for_the_replica(cluster, monkeypatch, tmp_path):
    # llama.cpp resolves --chat-template-file against the server's working directory, and ssh
    # starts the replica in the peer's LOGIN directory. A relative path either is not there --
    # and the peer burns the whole startup window before falling back to one node -- or a file
    # of the same name IS there, and the replica comes up healthy formatting prompts
    # differently from the primary, which every parity check still calls a matched pair.
    # The env twin LLAMA_ARG_CHAT_TEMPLATE_FILE was already absolutised; the argv route was not.
    cluster.topology = "replicas"
    monkeypatch.setenv(ss.ENV_PEER, "127.0.0.1")
    monkeypatch.chdir(tmp_path)
    model = tmp_path / "m.gguf"
    model.write_bytes(b"x")
    template = tmp_path / "mine.jinja"
    template.write_text("{{ x }}", encoding = "utf-8")

    _calls, started = _patch_remote(
        monkeypatch, binary = "$HOME/.unsloth/llama.cpp/build/bin/llama-server"
    )
    backend = _FakeBackend(12345, str(model))
    backend._process.args = list(backend._process.args) + ["--chat-template-file", "mine.jinja"]
    run(ss.after_load(backend, 16))

    assert started and ss.state().topology == "replicas"
    peer_argv = started[0].argv
    assert str(template) in peer_argv, "the peer was handed a path relative to its own login dir"
    assert "mine.jinja" not in peer_argv
    # And it is preflighted, so a peer that does not have it is found out before the launch
    # rather than after the startup window.
    checks = [c for c in _calls if c.startswith("stat -c")]
    assert checks and str(template) in checks[0]


def test_the_inline_form_of_the_template_flag_is_absolutised_too(tmp_path):
    # `--flag=value` is the form the env-vs-argv sweep keeps finding on the wrong side of a
    # guard, so it is pinned rather than assumed.
    argv = ["/bin/llama-server", "-m", "/m.gguf", "--chat-template-file=mine.jinja"]
    out = ss.replica_argv(argv, binary = "/bin/llama-server", host = "h", port = 1, cwd = "/base")
    assert "--chat-template-file=/base/mine.jinja" in out
    # An absolute path is left exactly as it is, and a colon in it is not read as a :SCALE.
    argv = ["/bin/llama-server", "--chat-template-file", "/tmp/t:1.5.jinja"]
    out = ss.replica_argv(argv, binary = "/bin/llama-server", host = "h", port = 1, cwd = "/base")
    assert "/tmp/t:1.5.jinja" in out


def test_a_managed_split_is_retired_before_a_callers_own_rpc_is_adopted(
    cluster, monkeypatch, tmp_path
):
    # A managed layer split has a peer ggml-rpc-server and NO router, so testing the router
    # alone left the old rpc-server running and still tracked while the new llama-server used
    # the caller's own placement: it holds its share of the peer GPU, so the manual topology
    # contends with it or does not fit, and status goes on reporting the managed split.
    cluster.topology = "layer_split"
    monkeypatch.setenv(ss.ENV_PEER, "127.0.0.1")
    model = tmp_path / "m.gguf"
    model.write_bytes(b"x")
    write_fake_llama_server(cluster.bundle / "build" / "bin", _FAKE_HELP_WITH_FLAG)
    _calls, started = _patch_remote(monkeypatch)

    run(ss.before_load(_FakeRequest(str(model)), 4))
    managed = ss.state().peer_process
    assert managed is not None and ss.state().router is None, "a managed split, no router"

    stopped = []
    real_stop = ss.PeerProcess.stop

    async def counting_stop(self, timeout = 10.0):
        stopped.append(self.name)
        return await real_stop(self, timeout = timeout)

    monkeypatch.setattr(ss.PeerProcess, "stop", counting_stop)

    # The next load brings its own --rpc at a DIFFERENT endpoint, so nothing here manages the
    # placement any more. A different endpoint, not merely a different flag: our own managed
    # split reaches this same branch with an --rpc this module put there.
    backend = _FakeBackend(12345, str(model), argv_extra = ["--rpc", "10.0.0.9:50052"])
    run(ss.after_load(backend, 16))

    assert stopped, "the managed rpc-server was left running beside the caller's own split"
    assert ss.state().peer_process is None
    assert ss.state().topology == "layer_split"
    assert "user-supplied --rpc" in ss.state().reason


def test_our_own_managed_split_is_not_torn_down_by_its_own_rpc_flag(cluster, monkeypatch, tmp_path):
    # The reconcile branch is reached by the managed split too, whose --rpc this module wrote.
    # Detaching on the presence of the flag rather than on the endpoint kills the live split
    # on every reconcile, which is a worse failure than the one being fixed.
    cluster.topology = "layer_split"
    monkeypatch.setenv(ss.ENV_PEER, "127.0.0.1")
    model = tmp_path / "m.gguf"
    model.write_bytes(b"x")
    write_fake_llama_server(cluster.bundle / "build" / "bin", _FAKE_HELP_WITH_FLAG)
    _calls, started = _patch_remote(monkeypatch)

    request = run(ss.before_load(_FakeRequest(str(model)), 4))
    managed = ss.state().peer_process
    assert managed is not None
    ours = [a for a in request.llama_extra_args if a.count(":")] or []
    endpoint = next(a for a in ours if a.startswith("127.0.0.1:"))

    backend = _FakeBackend(12345, str(model), argv_extra = ["--rpc", endpoint])
    run(ss.after_load(backend, 16))

    assert ss.state().peer_process is managed, "the managed split was torn down by its own flag"
    assert ss.state().topology == "layer_split"


def test_a_reload_the_route_will_refuse_does_not_stop_the_running_peer(
    cluster, monkeypatch, tmp_path
):
    # The route validates the pass-through args AFTER before_load, "up front so a managed-flag
    # collision returns 400 before any model work" -- but starting or restarting a peer IS model
    # work and happens here first. Restoring the topology metadata is not enough for it: a
    # forced reload stops the peer's rpc-server on the way through, the outgoing llama-server's
    # RPC allocations die with it, and nothing can put that back, so a request refused with a
    # 400 took a working split offline.
    cluster.topology = "layer_split"
    monkeypatch.setenv(ss.ENV_PEER, "127.0.0.1")
    model = tmp_path / "m.gguf"
    model.write_bytes(b"x")
    write_fake_llama_server(cluster.bundle / "build" / "bin", _FAKE_HELP_WITH_FLAG)
    _calls, started = _patch_remote(monkeypatch)

    run(ss.before_load(_FakeRequest(str(model)), 4))
    managed = ss.state().peer_process
    assert managed is not None and managed.alive, "a live managed split to protect"
    topology, reason = ss.state().topology, ss.state().reason

    stopped = []
    real_stop = ss.PeerProcess.stop

    async def counting_stop(self, timeout = 10.0):
        stopped.append(self.name)
        return await real_stop(self, timeout = timeout)

    monkeypatch.setattr(ss.PeerProcess, "stop", counting_stop)

    # A forced reload whose extras the route will refuse with a 400: --parallel is managed.
    doomed = _FakeRequest(str(model))
    doomed.force_reload = True
    doomed.llama_extra_args = ["--parallel", "32"]
    out = run(ss.before_load(doomed, 4))

    assert out is doomed, "the request comes back untouched for the route to reject"
    assert stopped == [], "a request that will be refused must not stop the running peer"
    assert ss.state().peer_process is managed
    assert (ss.state().topology, ss.state().reason) == (topology, reason)


def test_a_valid_reload_still_restarts_the_peer_for_its_new_client(cluster, monkeypatch, tmp_path):
    # The counterpart: the guard must not become a way for a forced reload to skip the restart
    # the rpc-server's one-client-at-a-time contract needs.
    cluster.topology = "layer_split"
    monkeypatch.setenv(ss.ENV_PEER, "127.0.0.1")
    model = tmp_path / "m.gguf"
    model.write_bytes(b"x")
    write_fake_llama_server(cluster.bundle / "build" / "bin", _FAKE_HELP_WITH_FLAG)
    _calls, started = _patch_remote(monkeypatch)

    run(ss.before_load(_FakeRequest(str(model)), 4))
    assert ss.state().peer_process is not None
    good = _FakeRequest(str(model))
    good.force_reload = True
    good.llama_extra_args = ["--lora", str(tmp_path / "a.gguf")]
    (tmp_path / "a.gguf").write_bytes(b"x")
    run(ss.before_load(good, 4))
    assert len(started) == 2, "a valid forced reload still restarts the peer rpc-server"


def test_the_supervisor_does_not_repoint_the_router_mid_load(cluster, monkeypatch, tmp_path):
    # A replacement GGUF gets its new port before the load finishes. Repointing there moved
    # attached_port, and after_load then read its own matching backend and port as a no-op
    # reload and returned WITHOUT replacing the peer: the router alternates between the new
    # primary model and the stale peer model, which is what the replica path exists to prevent.
    cluster.topology = "replicas"
    monkeypatch.setenv(ss.ENV_PEER, "127.0.0.1")
    model = tmp_path / "m.gguf"
    model.write_bytes(b"x")
    _calls, started = _patch_remote(
        monkeypatch, binary = "$HOME/.unsloth/llama.cpp/build/bin/llama-server"
    )

    backend = _FakeBackend(12345, str(model))
    run(ss.after_load(backend, 16))
    state = ss.state()
    assert state.router is not None and state.attached_port == 12345

    moved = []

    async def record(name, host, port):
        moved.append((name, host, port))

    state.router.set_backend_address = record
    backend._port = 23456  # the replacement's port, before its load has finished

    state.load_in_progress = True
    run(state._repoint_primary(backend))
    assert moved == [], "the router was repointed while a load was still in progress"
    assert state.attached_port == 12345

    state.load_in_progress = False
    run(state._repoint_primary(backend))
    assert moved == [("main", "127.0.0.1", 23456)], "and it still repoints once the load is done"
    assert state.attached_port == 23456


def test_the_peer_is_searched_for_the_rpc_name_the_local_bundle_resolved(
    cluster, monkeypatch, tmp_path
):
    # rpc_server_binary() resolves the supported legacy name `rpc-server` locally, but the peer
    # lookup discarded that basename and asked only for `ggml-rpc-server`, so a symmetrically
    # provisioned peer holding only `rpc-server` was reported missing and every layer split
    # that needed it fell back to one node.
    cluster.topology = "layer_split"
    monkeypatch.setenv(ss.ENV_PEER, "127.0.0.1")
    model = tmp_path / "m.gguf"
    model.write_bytes(b"x")
    write_fake_llama_server(cluster.bundle / "build" / "bin", _FAKE_HELP_WITH_FLAG)
    legacy = tmp_path / "bin" / "rpc-server"
    legacy.parent.mkdir(parents = True, exist_ok = True)
    legacy.write_text("#!/bin/sh\n", encoding = "utf-8")
    legacy.chmod(0o755)
    monkeypatch.setattr(ss, "rpc_server_binary", lambda: str(legacy))

    _calls, started = _patch_remote(monkeypatch)
    run(ss.before_load(_FakeRequest(str(model)), 4))

    lookup = next(c for c in _calls if "MISSING" in c)
    assert "rpc-server" in lookup
    assert "/rpc-server" in lookup, "the name the local bundle resolved is not being asked for"


def test_the_plan_prices_the_context_the_extras_actually_ask_for(cluster, monkeypatch, tmp_path):
    # The pass-through block is appended after the managed flags and llama.cpp is last-wins, so
    # a --ctx-size in the extras is what the server allocates against. Pricing max_seq_length
    # instead planned a model whose WEIGHTS fit one Spark but whose real cache does not as
    # `single`, and it then spilled or shrank instead of getting the split it needed.
    cluster.topology = "replicas"
    monkeypatch.setenv(ss.ENV_PEER, "127.0.0.1")
    model = tmp_path / "m.gguf"
    model.write_bytes(b"x")
    _patch_remote(monkeypatch)

    priced = []
    monkeypatch.setattr(
        ss,
        "estimate_kv_bytes",
        lambda path, ctx, k = None, v = None: priced.append((ctx, k, v)) or 0,
    )
    request = _FakeRequest(str(model))
    request.max_seq_length = 4096
    request.cache_type_kv = "f16"
    request.llama_extra_args = ["--ctx-size", "131072", "-ctk", "q8_0", "--cache-type-v", "q4_0"]
    run(ss.before_load(request, 4))

    assert priced == [
        (131072, "q8_0", "q4_0")
    ], "the plan priced the request's fields, not what the load will run with"


def test_the_cache_types_are_read_from_the_environment_and_apart(cluster, monkeypatch, tmp_path):
    # Studio leaves the cache types in the ENVIRONMENT rather than materialising them, which is
    # the whole reason replica_env exists. And K and V are configurable apart: pricing V as K
    # understates an asymmetric cache by up to 4x, which is the direction that OOMs a node.
    cluster.topology = "replicas"
    monkeypatch.setenv(ss.ENV_PEER, "127.0.0.1")
    monkeypatch.setenv("LLAMA_ARG_CACHE_TYPE_K", "q4_0")
    monkeypatch.setenv("LLAMA_ARG_CACHE_TYPE_V", "f32")
    model = tmp_path / "m.gguf"
    model.write_bytes(b"x")
    _patch_remote(monkeypatch)

    priced = []
    monkeypatch.setattr(
        ss,
        "estimate_kv_bytes",
        lambda path, ctx, k = None, v = None: priced.append((ctx, k, v)) or 0,
    )
    request = _FakeRequest(str(model))
    request.max_seq_length = 8192
    request.cache_type_kv = "f16"
    run(ss.before_load(request, 4))
    assert priced == [(8192, "q4_0", "f32")]

    # An extras value overrides the environment, since it reaches the server last.
    priced.clear()
    request.llama_extra_args = ["--cache-type-k", "q8_0"]
    run(ss.before_load(request, 4))
    assert priced == [(8192, "q8_0", "f32")]


def test_an_empty_gpu_list_is_automatic_placement_not_a_pin(cluster, monkeypatch, tmp_path):
    # `gpu_ids: []` is the documented automatic form and every other placement site in the
    # backend reads it by truthiness. Reading it as an explicit pin refused the split for a
    # GGUF that needs one and sent the request to the single node that cannot hold it.
    cluster.topology = "layer_split"
    monkeypatch.setenv(ss.ENV_PEER, "127.0.0.1")
    model = tmp_path / "m.gguf"
    model.write_bytes(b"x")
    write_fake_llama_server(cluster.bundle / "build" / "bin", _FAKE_HELP_WITH_FLAG)
    _patch_remote(monkeypatch)

    request = _FakeRequest(str(model))
    request.gpu_ids = []
    out = run(ss.before_load(request, 4))
    assert ss.state().topology == "layer_split", "an empty list was read as an explicit pin"
    assert "--rpc" in out.llama_extra_args

    # A real pin still owns placement, which is the behaviour being preserved.
    request2 = _FakeRequest(str(model))
    request2.gpu_ids = [0]
    run(ss.before_load(request2, 4))
    assert ss.state().topology == "single"
    assert "GPU pin" in ss.state().reason or "GPU selection" in ss.state().reason


def test_a_local_directory_of_gguf_variants_is_sized_not_skipped(tmp_path):
    # A local export is a DIRECTORY of quants, and the loader opens one of them. Answering
    # "size unknown" for it planned `single`, so a local export larger than one Spark could
    # never reach the split it needs.
    directory = tmp_path / "export"
    directory.mkdir()
    (directory / "model-Q4_K_M.gguf").write_bytes(b"x" * 16)
    (directory / "model-BF16.gguf").write_bytes(b"x" * 64)
    (directory / "mmproj-model.gguf").write_bytes(b"x" * 4)

    picked = ss.cached_repo_file(str(directory), None)
    assert picked is not None, "a local directory of GGUFs must resolve to a file"
    assert "mmproj" not in osp_basename(picked), "a companion is not the weights"

    # And a named variant still selects that one.
    assert osp_basename(ss.cached_repo_file(str(directory), "BF16") or "") == "model-BF16.gguf"


def osp_basename(path: str) -> str:
    import os.path
    return os.path.basename(str(path))


def test_only_generated_templates_are_shipped_to_the_peer_by_value(tmp_path, monkeypatch):
    # The directory alone used to be the whole test, so a model living directly under /tmp --
    # an ordinary place to put one -- was classified as generated and handed to
    # replicate_generated_files, which reads the file whole, base64-encodes it and puts it on
    # an ssh command line. For a multi-gigabyte GGUF that is Studio's memory and the OS
    # argument limit, instead of the one-sentence "the peer does not have it" preflight.
    monkeypatch.setattr(ss.tempfile, "gettempdir", lambda: str(tmp_path))
    model = tmp_path / "model.gguf"
    model.write_bytes(b"x" * 4096)
    lora = tmp_path / "adapter.gguf"
    lora.write_bytes(b"x" * 16)
    template = tmp_path / "unsloth_chat_template_1234.jinja"
    template.write_text("{{ x }}", encoding = "utf-8")

    out = ss.generated_launch_files([str(model), str(lora), str(template)])
    assert out == [str(template)]

    # A file that matches the name but is far too big to be a template is not shipped either.
    huge = tmp_path / "unsloth_chat_template_9999.jinja"
    huge.write_bytes(b"x" * (ss._GENERATED_MAX_BYTES + 1))
    assert ss.generated_launch_files([str(huge)]) == []


def test_a_dead_child_behind_a_slow_reaper_does_not_claim_the_port(monkeypatch):
    # process.alive is the LOCAL ssh session. The remote wrapper re-checks its child only once
    # per PEER_REAP_POLL_S (5s), which is longer than PEER_OWNERSHIP_SETTLE_S (1.5s), so after a
    # failed bind the ssh process can still look alive here while the child is already gone and
    # a stranger owns the port. Adopting that listener attaches a split to a foreign rpc-server,
    # or routes replica traffic to whatever model it holds.
    assert ss.PEER_OWNERSHIP_SETTLE_S < ss.PEER_REAP_POLL_S, (
        "the settle is shorter than the reap interval, which is why the local view is not enough"
    )

    async def always_open(host, port, timeout, *, cancelled = None):
        return True

    monkeypatch.setattr(ss, "wait_for_port", always_open)
    monkeypatch.setattr(ss, "PEER_OWNERSHIP_SETTLE_S", 0)

    class _SlowReaped:
        # The ssh session has NOT noticed yet: alive is true throughout.
        alive = True
        peer = "1.2.3.4"
        remote_pid = 4242

    asked = []

    async def pid_is_gone(peer, remote, timeout = 20.0):
        asked.append(remote)
        return 0, "PIDGONE\n", ""

    monkeypatch.setattr(ss, "ssh_run", pid_is_gone)
    assert run(ss.wait_for_own_port(_SlowReaped(), "1.2.3.4", 50052, 5.0)) is False
    assert asked and asked[0].startswith("kill -0 4242")

    # An unanswerable probe reads as not ours, which is the safe direction.
    async def ssh_broken(peer, remote, timeout = 20.0):
        raise OSError("no route to host")

    monkeypatch.setattr(ss, "ssh_run", ssh_broken)
    assert run(ss.wait_for_own_port(_SlowReaped(), "1.2.3.4", 50052, 5.0)) is False


def test_a_peer_that_already_exited_is_not_signalled_by_pid(monkeypatch):
    # Once the ssh session is gone the pid is no longer ours: the remote wrapper watches $PPID
    # and kills the child itself, and if the peer reused the number the kill lands on somebody
    # else's process under the same account. start() clears the field for this reason, but the
    # relaunch path calls stop() FIRST, after a backoff of up to 45 seconds.
    sent = []

    async def record(peer, remote, timeout = 20.0):
        sent.append(remote)
        return 0, "", ""

    monkeypatch.setattr(ss, "ssh_run", record)

    process = ss.PeerProcess("llama-server", "1.2.3.4", ["/bin/llama-server"])
    process.remote_pid = 4242
    process.proc = SimpleNamespace(returncode = 1)  # the ssh session has exited
    run(process.stop())
    assert sent == [], "a pid we no longer own must not be signalled"
    assert process.remote_pid is None, "and it must not be carried into the next attempt"

    # While the session IS up, the kill still happens: this must not become a leak.
    sent.clear()
    live = ss.PeerProcess("llama-server", "1.2.3.4", ["/bin/llama-server"])
    live.remote_pid = 4343
    live.proc = SimpleNamespace(returncode = None, terminate = lambda: None)

    async def _wait():
        return 0

    live.proc.wait = _wait
    run(live.stop())
    assert sent and "kill 4343" in sent[0]


def test_an_env_configured_drafter_is_the_callers_and_is_not_turned_off(monkeypatch):
    # LLAMA_ARG_SPEC_TYPE is a first-class backend route -- llama.cpp's common_arg reads it
    # directly -- so a caller can configure a drafter with nothing in argv. Reading argv alone
    # called that setting ours, and on a split reconcile_split_speculation then wrote
    # speculative_type="off" over it: an explicit choice silently disabled.
    for name in list(os.environ):
        if name.startswith("LLAMA_ARG_SPEC") or name.endswith("_DRAFT"):
            monkeypatch.delenv(name, raising = False)
    assert ss.extra_args_own_speculation([]) is None

    monkeypatch.setenv("LLAMA_ARG_SPEC_TYPE", "draft-mtp")
    assert ss.extra_args_own_speculation([]) == "LLAMA_ARG_SPEC_TYPE"
    # Blank is not a setting, the same rule the rest of the module applies to LLAMA_ARG_*.
    monkeypatch.setenv("LLAMA_ARG_SPEC_TYPE", "   ")
    assert ss.extra_args_own_speculation([]) is None
    monkeypatch.setenv("LLAMA_ARG_SPEC_DRAFT_MODEL", "/models/draft.gguf")
    assert ss.extra_args_own_speculation([]) == "LLAMA_ARG_SPEC_DRAFT_MODEL"

    # argv still wins the naming, so an operator reading mtp_reason is told which route to change.
    assert ss.extra_args_own_speculation(["--spec-type", "draft-mtp"]) == "--spec-type"
    assert (
        ss._speculation_owner_reason("--spec-type") == "--spec-type in the pass-through arguments"
    )
    assert (
        ss._speculation_owner_reason("LLAMA_ARG_SPEC_TYPE")
        == "LLAMA_ARG_SPEC_TYPE in the environment"
    )

    # And the plan defers to it rather than planning MTP over the top.
    plan = ss.mtp_plan(None, [], users = 64)
    assert plan["mtp"] == "user override"
    assert "LLAMA_ARG_SPEC_DRAFT_MODEL" in plan["reason"]


def test_a_load_no_topology_can_hold_is_refused_before_the_model_work(
    cluster, monkeypatch, tmp_path
):
    # recommend_topology labels an over-budget load `single` because it has to answer with SOME
    # topology, and its own reason says saying so up front is the only useful answer. Nothing
    # read the flag, so the request went on to attempt on ONE node what the planner had just
    # established does not fit across TWO: a late OOM instead of the diagnosis already written.
    model = tmp_path / "m.gguf"
    model.write_bytes(b"x")
    cluster.topology = "single"
    cluster.fits_any_topology = False
    cluster.reason = "needs 400.0 GiB, against 227.4 GiB across both Sparks"

    with pytest.raises(ss.SparkLoadDoesNotFit) as caught:
        run(ss.before_load(_FakeRequest(str(model)), 4))
    assert "both Sparks" in str(caught.value), "the planner's own sentence is what is reported"
    assert ss.state().topology == "single"

    # A load that DOES fit is untouched, so this is a refusal and not a new gate.
    cluster.fits_any_topology = True
    assert run(ss.before_load(_FakeRequest(str(model)), 4)) is not None


def test_environment_only_launch_files_are_preflighted_on_the_peer(
    cluster, monkeypatch, tmp_path
):
    # replica_env forwards LLAMA_ARG_MMPROJ, LLAMA_ARG_SPEC_DRAFT_MODEL and
    # LLAMA_ARG_CHAT_TEMPLATE_FILE to the peer, so a projector, drafter or template can reach the
    # replica without appearing in argv. Preflighting argv alone meant a missing one cost the
    # whole replica startup window, and a DIFFERENT file at the same path let the peer come up
    # healthy with another projector while every parity check still said matched pair.
    cluster.topology = "replicas"
    monkeypatch.setenv(ss.ENV_PEER, "127.0.0.1")
    model = tmp_path / "m.gguf"
    model.write_bytes(b"x")
    projector = tmp_path / "mmproj.gguf"
    projector.write_bytes(b"x" * 8)
    template = tmp_path / "t.jinja"
    template.write_text("{{ x }}", encoding = "utf-8")

    _calls, started = _patch_remote(
        monkeypatch, binary = "$HOME/.unsloth/llama.cpp/build/bin/llama-server"
    )
    backend = _FakeBackend(12345, str(model))
    backend.launched_env = {
        "LLAMA_ARG_MMPROJ": str(projector),
        "LLAMA_ARG_CHAT_TEMPLATE_FILE": str(template),
    }
    run(ss.after_load(backend, 16))

    checks = [c for c in _calls if c.startswith("stat -c")]
    assert checks, "the peer was asked about the launch files"
    assert str(projector) in checks[0], "an env-only projector was never checked on the peer"
    assert str(template) in checks[0], "an env-only chat template was never checked on the peer"


def test_an_env_only_file_the_peer_does_not_have_costs_the_replicas_at_once(
    cluster, monkeypatch, tmp_path
):
    # The point of preflighting it: falling back here takes a round trip, not the full replica
    # startup window.
    cluster.topology = "replicas"
    monkeypatch.setenv(ss.ENV_PEER, "127.0.0.1")
    model = tmp_path / "m.gguf"
    model.write_bytes(b"x")
    projector = tmp_path / "mmproj.gguf"
    projector.write_bytes(b"x" * 8)

    # The MODEL is present on the peer and matches; only the env-only projector is missing.
    # Anything less specific would pass before the fix, because an absent model already
    # refuses on the argv path.
    _calls, started = _patch_remote(
        monkeypatch,
        binary = "$HOME/.unsloth/llama.cpp/build/bin/llama-server",
        absent_paths = (str(projector),),
    )
    backend = _FakeBackend(12345, str(model))
    backend.launched_env = {"LLAMA_ARG_MMPROJ": str(projector)}
    run(ss.after_load(backend, 16))

    assert ss.state().topology == "single"
    assert str(projector) in ss.state().reason, "the reason must name the file that is missing"
    assert not started, "the peer llama-server must not be launched at all"


def test_a_value_that_names_no_local_file_is_not_demanded_of_the_peer(tmp_path):
    # A stale or URL-shaped setting names nothing here, so it is no evidence about the peer, and
    # demanding it would refuse replicas for something llama-server itself would ignore.
    real = tmp_path / "mmproj.gguf"
    real.write_bytes(b"x")
    assert ss.env_launch_files({"LLAMA_ARG_MMPROJ": str(real)}) == [str(real)]
    assert ss.env_launch_files({"LLAMA_ARG_MMPROJ": str(tmp_path / "gone.gguf")}) == []
    assert ss.env_launch_files({"LLAMA_ARG_MMPROJ": ""}) == []
    assert ss.env_launch_files(None) == []
    # Not in _REPLICA_ENV_PATHS: a URL and a repo id are not local files.
    assert ss.env_launch_files({"LLAMA_ARG_MMPROJ_URL": "https://example/x.gguf"}) == []


def test_every_branch_that_commits_extra_args_also_records_the_requested_identity():
    """The two are a pair and must never disagree.

    ``_spark_inherited_extra_args`` compares the REQUESTED identity -- what the caller typed --
    against the incoming request, because that is the only same-namespace comparison available
    before the load has resolved anything. If a branch commits ``_extra_args`` and updates
    ``_extra_args_source`` but leaves ``_extra_args_requested_source`` holding a previous load's
    identity, those extras are read as belonging to that earlier model: load GGUF A, then a
    DiffusionGemma model with its own extras, then A again with the field omitted, and A is
    launched with the diffusion arguments.

    Written as an invariant over every branch rather than as that one scenario, so the next
    commit point someone adds is covered by construction. That is the whole reason the pair
    exists; a scenario test would pass again the moment a third branch appeared.
    """
    import ast
    from pathlib import Path

    src = Path(ss.__file__).resolve().parent / "llama_cpp.py"
    tree = ast.parse(src.read_text(encoding = "utf-8"))

    def _assigns(statements, attribute):
        for node in statements:
            for target in getattr(node, "targets", []):
                if (
                    isinstance(target, ast.Attribute)
                    and target.attr == attribute
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                ):
                    return True
        return False

    resolved = []
    for node in ast.walk(tree):
        for field in ("body", "orelse", "finalbody"):
            block = getattr(node, field, None)
            if not isinstance(block, list):
                continue
            if not _assigns(block, "_extra_args_source"):
                continue
            resolved.append(getattr(block[0], "lineno", "?"))
            assert _assigns(block, "_extra_args_requested_source"), (
                f"llama_cpp.py near line {getattr(block[0], 'lineno', '?')} commits "
                "_extra_args_source without recording or clearing "
                "_extra_args_requested_source; the pair must be written together"
            )

    assert len(resolved) >= 2, "the commit points moved; this invariant is no longer being checked"


def test_a_refused_oversized_load_leaves_the_running_topology_exactly_as_it_was(
    cluster, monkeypatch, tmp_path
):
    # The route catches SparkLoadDoesNotFit OUTSIDE the block that calls load_failed, so nothing
    # else rolls this back. Two things had to be undone: the reported topology, which would
    # otherwise describe a resident two-node model as `single` for the rest of its life, and
    # load_in_progress -- the flag the supervisor uses to decide that a cleared _process is a
    # load rather than an unload. Left true it would ignore a real unload forever, leaving the
    # peer and the router serving a model nothing thinks is loaded.
    model = tmp_path / "m.gguf"
    model.write_bytes(b"x")
    cluster.topology = "replicas"
    monkeypatch.setenv(ss.ENV_PEER, "127.0.0.1")
    _patch_remote(monkeypatch, binary = "$HOME/.unsloth/llama.cpp/build/bin/llama-server")
    run(ss.after_load(_FakeBackend(12345, str(model)), 16))
    state = ss.state()
    assert state.topology == "replicas" and state.router is not None

    before = (state.topology, state.reason, state.pipeline_groups, state.split_config)
    cluster.fits_any_topology = False
    cluster.reason = "needs 400.0 GiB, against 227.4 GiB across both Sparks"

    with pytest.raises(ss.SparkLoadDoesNotFit):
        run(ss.before_load(_FakeRequest(str(model)), 4))

    assert (
        state.topology,
        state.reason,
        state.pipeline_groups,
        state.split_config,
    ) == before, "the refusal overwrote the topology of a model that is still resident"
    assert state.load_in_progress is False, "the supervisor would ignore every later unload"
    assert state.router is not None, "and the live router must be untouched"


def test_a_completed_load_does_not_leave_a_snapshot_to_be_restored(
    cluster, monkeypatch, tmp_path
):
    # _pre_load_state described the topology BEFORE the load that has now finished. Any later
    # path that returns before taking a fresh one -- the pass-through refusal, for instance --
    # would have load_failed restore that stale description over a live topology: a split or
    # replica set created from `single` reported as `single` after a harmless 400, and the next
    # split tearing down a peer it should have reused.
    model = tmp_path / "m.gguf"
    model.write_bytes(b"x")
    cluster.topology = "replicas"
    monkeypatch.setenv(ss.ENV_PEER, "127.0.0.1")
    _patch_remote(monkeypatch, binary = "$HOME/.unsloth/llama.cpp/build/bin/llama-server")

    state = ss.state()
    # A snapshot from when nothing was attached, i.e. what the first load would have taken.
    run(ss.before_load(_FakeRequest(str(model)), 4))
    backend = _FakeBackend(12345, str(model))
    run(ss.after_load(backend, 16))
    assert state.topology == "replicas"
    assert state._pre_load_state is None, "a finished attempt must not leave a snapshot behind"

    # The route rejects the next request with a 400 and its wrapper calls load_failed. With a
    # stale snapshot this restored `single` over the live replica set.
    # is_loaded is a property on the double, and it already reads true here: the previous
    # model is still resident, which is the whole case load_failed's restore exists for.
    assert backend.is_loaded is True
    state.attached_backend = backend
    run(ss.load_failed())
    assert state.topology == "replicas", "load_failed restored a snapshot from a finished load"


def test_a_projector_set_only_in_the_environment_is_charged_to_the_node(
    cluster, monkeypatch, tmp_path
):
    # LLAMA_ARG_MMPROJ is resident for the whole load and is forwarded to the replica, so
    # charging zero for it let a plan pick `single` or `replicas` whose processes each exceed
    # the node budget. Understating is the direction that OOMs.
    model = tmp_path / "m.gguf"
    model.write_bytes(b"x" * 4096)
    projector = tmp_path / "mmproj.gguf"
    projector.write_bytes(b"y" * 8192)
    monkeypatch.setenv("LLAMA_ARG_MMPROJ", str(projector))
    _patch_remote(monkeypatch)

    run(ss.before_load(_FakeRequest(str(model)), 4))
    charged = cluster.planner_calls[-1]["model_bytes"]
    assert charged >= 4096 + 8192, f"the env projector was not charged: {charged}"

    # And it is charged once, not twice, when argv names the same file.
    cluster.planner_calls.clear()
    request = _FakeRequest(str(model))
    request.llama_extra_args = ["--mmproj", str(projector)]
    run(ss.before_load(request, 4))
    assert cluster.planner_calls[-1]["model_bytes"] == 4096 + 8192
