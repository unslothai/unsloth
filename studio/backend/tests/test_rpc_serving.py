# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The helpers layer-split serving shares: binaries, GGUF sizing, KV estimates, ssh and peer processes."""

from __future__ import annotations

import asyncio
import os
import shutil
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pytest

from core.inference import rpc_serving as ss

# These assert POSIX paths, accounts and process groups, which a Windows runner does not have.
_POSIX_ONLY = pytest.mark.skipif(os.name == "nt", reason = "POSIX paths and processes")

GIB = 2**30


class StubBundle:
    """Stands in for ``studio.llama_rpc``: where the llama.cpp bundle and its rpc-server are."""

    def __init__(self) -> None:
        self.bundle: Optional[Path] = None
        self.rpc_binary: Optional[str] = None

    def llama_bundle_dir(self) -> Path:
        return self.bundle or Path.home() / ".unsloth" / "llama.cpp"

    def rpc_server_binary(self) -> Optional[str]:
        return self.rpc_binary or str(
            Path.home() / ".unsloth" / "llama.cpp" / "build" / "bin" / "ggml-rpc-server"
        )


@pytest.fixture
def cluster(monkeypatch, tmp_path):
    stub = StubBundle()
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
    monkeypatch.setattr(ss, "_LLAMA_RPC", stub)
    monkeypatch.setattr(ss, "_LLAMA_RPC_LOOKED_UP", True)
    monkeypatch.delenv(ss.ENV_TOGGLE, raising = False)
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


def write_fake_llama_server(
    directory: Path,
    help_text: str,
    *,
    body: str = "",
    hidden_flags: tuple = (),
    refuses_groups_with_drafter: bool = False,
) -> Path:
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


def probe_runs(script: Path) -> int:
    calls = Path(str(script) + ".calls")
    return len(calls.read_text(encoding = "utf-8").splitlines()) if calls.exists() else 0


def run(coro):
    return asyncio.run(coro)


@_POSIX_ONLY
def test_ssh_user_is_this_login_and_never_a_fixed_one(monkeypatch):
    import getpass

    monkeypatch.setenv("USER", "alice")
    assert ss._ssh_user() == "alice"
    for var in ("USER", "USERNAME", "LOGNAME"):
        monkeypatch.delenv(var, raising = False)
    assert ss._ssh_user() == getpass.getuser()
    assert ss.ssh_argv("192.168.200.13", "true")[-2] == f"{getpass.getuser()}@192.168.200.13"


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


def test_peer_binary_candidates_prefer_the_local_bundle_path(cluster):
    local = str(Path.home() / ".unsloth" / "llama.cpp" / "build" / "bin" / "llama-server")
    candidates = ss.peer_binary_candidates(local, "llama-server")
    assert candidates[0] == "$HOME/.unsloth/llama.cpp/build/bin/llama-server"
    assert candidates[-1] == "llama-server"
    assert len(candidates) == len(set(candidates))
    script = ss.find_binary_script(candidates)
    assert script.endswith("echo MISSING; exit 1") and "command -v llama-server" in script


def test_peer_process_remote_command_prints_the_servers_pid_and_reaps_it():
    """A Studio killed without its shutdown path must not leave the peer rpc-server holding the GPU."""
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
    assert "read" not in command
    assert f"sleep {ss.PEER_REAP_POLL_S}" in command
    assert not process.alive
    snap = process.snapshot()
    assert snap["remote_pid"] is None and snap["alive"] is False


def test_the_ssh_client_carrying_the_peer_cannot_outlive_this_process():
    """Without PR_SET_PDEATHSIG the ssh client outlives a killed Studio."""
    import inspect

    source = inspect.getsource(ss.PeerProcess.start)
    assert "preexec_fn = _die_with_parent" in source
    assert "PR_SET_PDEATHSIG" in inspect.getsource(ss._die_with_parent)
    ss._die_with_parent()


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
    assert ss.estimate_kv_bytes(str(path), 1024) == 2 * 2 * 1024 * 2 * 16 * 2
    assert ss.estimate_kv_bytes(str(path), 1024, "q8_0") == int(2 * 2 * 1024 * 2 * 16 * (34 / 32))
    assert ss.estimate_kv_bytes(str(path), 0) is None
    assert ss.estimate_kv_bytes(str(tmp_path / "missing.gguf"), 1024) is None
    assert ss.kv_bytes_per_elem("q4_0") == 18 / 32 and ss.kv_bytes_per_elem(None) == 2.0


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


def test_kv_sizing_reads_an_inherited_llama_arg_ctx_size(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setenv("LLAMA_ARG_CTX_SIZE", "8192")
    request = SimpleNamespace(max_seq_length = 0, cache_type_kv = None)
    # no panel value and no -c: auto-layers omits -c and the child runs at the env length
    assert ss.effective_kv_settings(request, [])[0] == 8192
    # a panel value is passed as -c and wins over the environment
    assert ss.effective_kv_settings(SimpleNamespace(max_seq_length = 4096), [])[0] == 4096
    # a pass-through -c wins too, and -c 0 asks for the native length, not the env's
    assert ss.effective_kv_settings(request, ["-c", "2048"])[0] == 2048
    assert ss.effective_kv_settings(request, ["--ctx-size", "0"])[0] == 0
    monkeypatch.setenv("LLAMA_ARG_CTX_SIZE", "nope")
    assert ss.effective_kv_settings(request, [])[0] == 0


def test_kv_sizing_folds_long_option_underscores(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.delenv("LLAMA_ARG_CTX_SIZE", raising = False)
    monkeypatch.delenv("LLAMA_ARG_CACHE_TYPE_K", raising = False)
    monkeypatch.delenv("LLAMA_ARG_CACHE_TYPE_V", raising = False)
    request = SimpleNamespace(max_seq_length = 4096, cache_type_kv = "f16")
    extras = ["--ctx_size", "65536", "--cache_type_k", "q8_0", "--cache_type_v=q4_0"]
    assert ss.effective_kv_settings(request, extras) == (65536, "q8_0", "q4_0")


def test_estimate_kv_bytes_prices_k_and_v_apart(tmp_path):
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

    assert ss.estimate_kv_bytes(str(path), 1024, "q4_0") == int(cells * (q4 + q4))
    asymmetric = ss.estimate_kv_bytes(str(path), 1024, "q4_0", "f32")
    assert asymmetric == int(cells * (q4 + f32))
    assert asymmetric > 4 * ss.estimate_kv_bytes(str(path), 1024, "q4_0")


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
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    repo = "unsloth/Big-GGUF"
    _snapshot(tmp_path, repo, "aaa0", {"big-Q4_K_M.gguf": 4096}, mtime = 1_000)
    _snapshot(tmp_path, repo, "zzz9", {"big-Q4_K_M.gguf": 16384}, mtime = 2_000)

    chosen = ss.cached_repo_file(repo, None)
    assert chosen is not None and "zzz9" in chosen
    assert ss.gguf_size_bytes(chosen) == 16384


def test_a_companion_gguf_is_never_sized_in_place_of_the_weights(monkeypatch, tmp_path):
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


def test_remote_sizing_never_raises_out_of_the_load(monkeypatch):
    def _boom(*_a, **_k):
        raise RuntimeError("hub is down")

    monkeypatch.setattr(ss, "_pick_variant", _boom)
    assert ss.remote_gguf_size_bytes("unsloth/Huge-GGUF", None) is None
    assert ss.remote_gguf_size_bytes("/models/x.gguf", None) is None
    assert ss.remote_gguf_size_bytes("mymodel", None) is None


def test_a_stranger_on_the_port_is_not_adopted_as_ours(monkeypatch):
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
        peer = "1.2.3.4"
        remote_pid = 4242

        def __init__(self, dies):
            self._dies = dies
            self.alive = True

        async def die_after_the_probe(self):
            self.alive = not self._dies

    listener = {"pid": 4242}

    async def pid_is_live(
        peer,
        remote,
        timeout = 20.0,
    ):
        if remote.startswith("ss "):
            line = (
                f'LISTEN 0 512 1.2.3.4:50052 0.0.0.0:* users:(("rpc",pid={listener["pid"]},fd=3))'
            )
            return 0, line + "\n", ""
        return 0, "PIDLIVE\n", ""

    monkeypatch.setattr(ss, "ssh_run", pid_is_live)

    ours = _Child(dies = False)
    assert run(ss.wait_for_own_port(ours, "1.2.3.4", 50052, 5.0)) is True
    listener["pid"] = 999
    assert run(ss.wait_for_own_port(_Child(dies = False), "1.2.3.4", 50052, 1.5)) is False
    listener["pid"] = 4242

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


def test_the_capability_probe_follows_the_binary_the_loader_will_launch(monkeypatch, tmp_path):
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


def test_a_missing_shard_reports_the_size_unknown_rather_than_short(tmp_path):
    """Undercounting shards would plan `single` and OOM after the rest arrive."""
    first = tmp_path / "m-00001-of-00003.gguf"
    first.write_bytes(b"a" * 100)
    (tmp_path / "m-00002-of-00003.gguf").write_bytes(b"b" * 100)
    assert ss.gguf_size_bytes(str(first)) is None

    (tmp_path / "m-00003-of-00003.gguf").write_bytes(b"c" * 100)
    assert ss.gguf_size_bytes(str(first)) == 300


def test_a_projector_that_arrives_only_through_the_environment_still_costs_the_groups(monkeypatch):
    """LLAMA_ARG_MMPROJ reaches the server without appearing in argv."""
    monkeypatch.delenv("LLAMA_ARG_MMPROJ", raising = False)
    monkeypatch.delenv("LLAMA_ARG_MMPROJ_URL", raising = False)
    assert ss.extra_args_refuse_pipeline_groups([]) is None
    assert ss.extra_args_refuse_pipeline_groups(["--mmproj", "p.gguf"]) == "--mmproj"

    monkeypatch.setenv("LLAMA_ARG_MMPROJ", "/models/p.gguf")
    assert ss.extra_args_refuse_pipeline_groups([]) == "LLAMA_ARG_MMPROJ"
    monkeypatch.delenv("LLAMA_ARG_MMPROJ")
    monkeypatch.setenv("LLAMA_ARG_MMPROJ_URL", "https://example/p.gguf")
    assert ss.extra_args_refuse_pipeline_groups([]) == "LLAMA_ARG_MMPROJ_URL"
    monkeypatch.setenv("LLAMA_ARG_MMPROJ_URL", "  ")
    assert ss.extra_args_refuse_pipeline_groups([]) is None


def test_the_hub_is_asked_with_the_requests_own_token(monkeypatch):
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


def test_the_size_is_taken_from_the_variant_the_loader_would_open(monkeypatch):
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
    assert ss.remote_gguf_size_bytes("unsloth/Qwen3-8B-GGUF", "BF16") == 300 * 1024**3


def test_a_repo_whose_picker_cannot_answer_still_gets_a_size(monkeypatch):
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


def test_a_local_directory_of_gguf_variants_is_sized_not_skipped(tmp_path):
    directory = tmp_path / "export"
    directory.mkdir()
    (directory / "model-Q4_K_M.gguf").write_bytes(b"x" * 16)
    (directory / "model-BF16.gguf").write_bytes(b"x" * 64)
    (directory / "mmproj-model.gguf").write_bytes(b"x" * 4)

    picked = ss.cached_repo_file(str(directory), None)
    assert picked is not None, "a local directory of GGUFs must resolve to a file"
    assert "mmproj" not in osp_basename(picked), "a companion is not the weights"

    assert osp_basename(ss.cached_repo_file(str(directory), "BF16") or "") == "model-BF16.gguf"


def osp_basename(path: str) -> str:
    import os.path
    return os.path.basename(str(path))


def test_a_dead_child_behind_a_slow_reaper_does_not_claim_the_port(monkeypatch):
    assert (
        ss.PEER_OWNERSHIP_SETTLE_S < ss.PEER_REAP_POLL_S
    ), "the settle is shorter than the reap interval, which is why the local view is not enough"

    async def always_open(
        host,
        port,
        timeout,
        *,
        cancelled = None,
    ):
        return True

    monkeypatch.setattr(ss, "wait_for_port", always_open)
    monkeypatch.setattr(ss, "PEER_OWNERSHIP_SETTLE_S", 0)

    class _SlowReaped:
        alive = True
        peer = "1.2.3.4"
        remote_pid = 4242

    asked = []

    async def pid_is_gone(
        peer,
        remote,
        timeout = 20.0,
    ):
        asked.append(remote)
        return 0, "PIDGONE\n", ""

    monkeypatch.setattr(ss, "ssh_run", pid_is_gone)
    assert run(ss.wait_for_own_port(_SlowReaped(), "1.2.3.4", 50052, 5.0)) is False
    assert asked and asked[0].startswith("kill -0 4242")

    async def ssh_broken(
        peer,
        remote,
        timeout = 20.0,
    ):
        raise OSError("no route to host")

    monkeypatch.setattr(ss, "ssh_run", ssh_broken)
    assert run(ss.wait_for_own_port(_SlowReaped(), "1.2.3.4", 50052, 5.0)) is False


def test_a_peer_that_already_exited_is_not_signalled_by_pid(monkeypatch):
    sent = []

    async def record(
        peer,
        remote,
        timeout = 20.0,
    ):
        sent.append(remote)
        return 0, "", ""

    monkeypatch.setattr(ss, "ssh_run", record)

    process = ss.PeerProcess("llama-server", "1.2.3.4", ["/bin/llama-server"])
    process.remote_pid = 4242
    process.proc = SimpleNamespace(returncode = 1)  # the ssh session has exited
    run(process.stop())
    assert sent == [], "a pid we no longer own must not be signalled"
    assert process.remote_pid is None, "and it must not be carried into the next attempt"

    sent.clear()
    live = ss.PeerProcess("llama-server", "1.2.3.4", ["/bin/llama-server"])
    live.remote_pid = 4343
    live.proc = SimpleNamespace(returncode = None, terminate = lambda: None)

    async def _wait():
        return 0

    live.proc.wait = _wait
    run(live.stop())
    assert sent and "kill 4343" in sent[0]


def test_a_value_that_names_no_local_file_is_not_demanded_of_the_peer(tmp_path):
    real = tmp_path / "mmproj.gguf"
    real.write_bytes(b"x")
    assert ss.env_launch_files({"LLAMA_ARG_MMPROJ": str(real)}) == [str(real)]
    assert ss.env_launch_files({"LLAMA_ARG_MMPROJ": str(tmp_path / "gone.gguf")}) == []
    assert ss.env_launch_files({"LLAMA_ARG_MMPROJ": ""}) == []
    assert ss.env_launch_files(None) == []
    assert ss.env_launch_files({"LLAMA_ARG_MMPROJ_URL": "https://example/x.gguf"}) == []


def test_the_status_route_is_the_owners_only(monkeypatch):
    """Regression: only the owner may read worker command lines, log tails and host paths."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    import routes.rpc_serving as route
    import utils.account_context as account_context
    from auth.authentication import get_current_subject

    app = FastAPI()
    app.include_router(route.router, prefix = "/api/inference/spark")
    app.dependency_overrides[get_current_subject] = lambda: "someone"
    monkeypatch.setattr(ss, "status", lambda: {"enabled": True, "peer": {"command": "/home/o/x"}})
    client = TestClient(app)
    monkeypatch.setattr(account_context, "current_account", lambda: SimpleNamespace(is_owner = False))
    assert client.get("/api/inference/spark/status").status_code == 403
    monkeypatch.setattr(account_context, "current_account", lambda: SimpleNamespace(is_owner = True))
    response = client.get("/api/inference/spark/status")
    assert response.status_code == 200 and response.json()["enabled"] is True


def test_without_a_provider_the_hooks_route_nothing_and_count_one_replica():
    assert ss.route_base_url(object()) is None
    assert ss.replica_count(object()) == 1
    payload = {"messages": []}
    ss.tag_conversation(payload, "thread")
    assert payload == {"messages": []}


def test_a_registered_provider_answers_every_hook(monkeypatch):
    calls = []

    class Provider:
        def enabled(self):
            return True

        def route_base_url(self, backend):
            return "http://127.0.0.1:9"

        def replica_count(self, backend):
            return 2

        def current_topology(self):
            return "replicas"

        def status(self):
            return {"enabled": True, "topology": "replicas"}

        def tag_conversation(self, payload, thread_id):
            payload["tagged"] = thread_id

        async def before_load(self, request, n_parallel, **kw):
            calls.append(("before", sorted(kw)))
            return "planned"

        async def after_load(self, backend, n_parallel, **kw):
            calls.append("after")
            return True

        async def load_failed(self):
            calls.append("failed")

        async def reconcile_internal_load(self):
            calls.append("internal")

        async def shutdown(self):
            calls.append("shutdown")

    monkeypatch.setattr(ss, "_PROVIDER", Provider())
    payload = {}
    ss.tag_conversation(payload, "t")
    assert (ss.enabled(), ss.route_base_url(None), ss.replica_count(None)) == (
        True,
        "http://127.0.0.1:9",
        2,
    )
    assert ss.current_topology() == "replicas" and ss.status()["topology"] == "replicas"
    assert payload == {"tagged": "t"}
    assert run(ss.before_load(object(), 4)) == "planned"
    assert run(ss.after_load(object(), 4)) is True
    run(ss.load_failed())
    run(ss.reconcile_internal_load())
    run(ss.shutdown())
    assert calls == [
        ("before", ["cancel_event", "confirm_teardown", "inherited_extra_args"]),
        "after",
        "failed",
        "internal",
        "shutdown",
    ]


def test_every_tool_loop_round_names_the_same_conversation():
    source = (Path(ss.__file__).resolve().parent / "llama_cpp.py").read_text(encoding = "utf-8")
    start = source.index("def generate_chat_completion_with_tools(")
    body = source[start:]
    for marker in ("payload = {", "stream_payload = {"):
        assert marker in body, marker
    assert body.count("tag_conversation(payload, thread_id)") >= 1
    assert body.count("tag_conversation(stream_payload, thread_id)") >= 1


def test_every_branch_that_commits_extra_args_also_records_the_requested_identity():
    """The two are a pair and must never disagree."""
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


def test_a_load_the_account_may_not_make_is_refused_before_the_split_is_touched(monkeypatch):
    """Access is refused before before_load can tear down the served split."""
    import routes.inference as ri
    from fastapi import HTTPException

    touched = []

    async def _before(*a, **k):
        touched.append("before_load")
        return a[0]

    def _deny(path):
        raise HTTPException(status_code = 403, detail = f"no access to {path}")

    monkeypatch.setattr(ss, "enabled", lambda: True)
    monkeypatch.setattr(ss, "before_load", _before)
    monkeypatch.setattr(ri.account_access, "managed_account", lambda: True)
    monkeypatch.setattr(ri.account_access, "require_model_access", _deny)
    monkeypatch.setattr(ri, "_defers_access_to_native_grant", lambda request: False)
    request = SimpleNamespace(
        model_path = "org/forbidden-GGUF",
        engine = "auto",
        n_parallel = 4,
        load_request_id = None,
    )
    with pytest.raises(HTTPException) as raised:
        run(ri._run_tracked_load_model_impl(request, None, "someone"))
    assert raised.value.status_code == 403
    assert touched == []


def test_the_route_hands_its_reload_gate_to_the_split_teardown(monkeypatch):
    """The active-chat 409 must run before the split teardown."""
    import routes.inference as ri
    from fastapi import HTTPException

    gate_calls = []

    def _gate(*, cancel):
        gate_calls.append(cancel)
        raise HTTPException(status_code = 409, detail = "active generations")

    async def _before(request, slots, **k):
        confirmed = k["confirm_teardown"]()
        if hasattr(confirmed, "__await__"):
            await confirmed
        return request

    monkeypatch.setattr(ss, "enabled", lambda: True)
    monkeypatch.setattr(ss, "before_load", _before)
    monkeypatch.setattr(ri.account_access, "managed_account", lambda: False)
    monkeypatch.setattr(ri.account_access, "require_idle_other_accounts", lambda: None)
    request = SimpleNamespace(
        model_path = "org/model-GGUF", engine = "auto", n_parallel = 4, load_request_id = None
    )
    with pytest.raises(HTTPException) as raised:
        run(ri._run_tracked_load_model_impl(request, None, "someone", on_reload_confirmed = _gate))
    assert raised.value.status_code == 409 and gate_calls == [False]


def test_a_load_cancelled_during_the_replica_startup_is_undone_and_refused(monkeypatch):
    """Regression: a load cancelled during replica start must not report success."""
    import routes.inference as ri
    from fastapi import HTTPException

    unloaded = []

    async def _before(request, *a, **k):
        return request

    async def _impl(*a, **k):
        return {"status": "loaded"}

    async def _after(*a, **k):
        return True

    monkeypatch.setattr(ss, "enabled", lambda: True)
    monkeypatch.setattr(ss, "before_load", _before)
    monkeypatch.setattr(ss, "after_load", _after)
    monkeypatch.setattr(ri, "_load_model_impl", _impl)
    monkeypatch.setattr(ri.account_access, "managed_account", lambda: False)
    monkeypatch.setattr(
        ri,
        "get_llama_cpp_backend",
        lambda: SimpleNamespace(unload_model = lambda: unloaded.append(1)),
    )
    request = SimpleNamespace(
        model_path = "org/model-GGUF", engine = "auto", n_parallel = 4, load_request_id = None
    )
    with pytest.raises(HTTPException) as raised:
        run(ri._run_tracked_load_model_impl(request, None, "someone"))
    assert raised.value.status_code == 409 and unloaded == [1]


def test_the_load_summary_is_read_after_after_load_drops_a_missing_split(monkeypatch):
    """Regression: a server launched without the planned split must not be reported as split."""
    import routes.inference as ri
    from models.inference import LoadResponse

    split = {"on": True}

    async def _before(request, *a, **k):
        return request

    async def _impl(*a, **k):
        return LoadResponse(status = "loaded", model = "m", display_name = "m", inference = {})

    async def _after(*a, **k):
        split["on"] = False  # the launched argv lacked the planned block: state is now single
        return False

    monkeypatch.setattr(ss, "enabled", lambda: True)
    monkeypatch.setattr(ss, "before_load", _before)
    monkeypatch.setattr(ss, "after_load", _after)
    monkeypatch.setattr(ri, "_load_model_impl", _impl)
    monkeypatch.setattr(ri.account_access, "managed_account", lambda: False)
    monkeypatch.setattr(ri, "get_llama_cpp_backend", lambda: SimpleNamespace())
    monkeypatch.setattr(
        ri,
        "_cluster_ux",
        lambda name, *a, **k: "Split across this machine and studio-2."
        if name == "split_summary" and split["on"]
        else None,
    )
    request = SimpleNamespace(
        model_path = "org/model-GGUF", engine = "auto", n_parallel = 4, load_request_id = None
    )
    response = run(ri._run_tracked_load_model_impl(request, None, "someone"))
    assert isinstance(response, LoadResponse) and response.cluster_summary is None


def test_a_callers_own_rpc_placement_is_never_read_as_ours(monkeypatch):
    """With nothing recorded, an argument list shaped like a split block is still the caller's."""
    monkeypatch.setattr(ss, "_PROVIDER", None)
    monkeypatch.setattr(ss, "_PROVIDER_LOOKED_UP", True, raising = False)  # no provider installed
    users = [
        "--rpc", "10.0.0.2:50052", "--device", "RPC0,CUDA0", "-sm", "layer",
        "--tensor-split", "0.5,0.5", "--cache-ram", "0",
    ]  # fmt: skip
    assert ss.strip_layer_split_extra_args(users) == users


def test_split_extras_follow_the_same_gguf_whatever_the_reload_names_it(monkeypatch, tmp_path):
    """A reload that names the repo instead of its snapshot, or drops a variant it sent before, loads
    the same file; the planner must see the extras the route will inherit, not plan without them."""
    import routes.inference as ri

    cache = tmp_path / "hub"
    snapshot = cache / "models--org--m" / "snapshots" / "abc"
    snapshot.mkdir(parents = True)
    (snapshot / "m-Q4_K_M.gguf").write_bytes(b"GGUF")
    other = tmp_path / "elsewhere"
    other.mkdir()
    (other / "m-Q4_K_M.gguf").write_bytes(b"GGUF")
    monkeypatch.setenv("HF_HUB_CACHE", str(cache))

    class _Backend:
        def __init__(self, typed, variant):
            self.extra_args = ["-c", "8192", "--rope-scaling", "yarn"]
            self.extra_args_requested_source = (typed, variant)
            self.extra_args_source = (typed, variant)

    def inherits(
        stored,
        request_path,
        request_variant = None,
    ):
        monkeypatch.setattr(ri, "get_llama_cpp_backend", lambda: _Backend(*stored))
        request = SimpleNamespace(
            model_path = request_path, gguf_variant = request_variant, llama_extra_args = None
        )
        return ri._split_inherited_extra_args(request) is not None

    assert inherits((str(snapshot), "Q4_K_M"), "org/m", "Q4_K_M")
    assert inherits(("org/m", "Q4_K_M"), str(snapshot), "Q4_K_M")
    assert inherits(("org/m", "Q4_K_M"), "org/m")  # the variant it resolves to anyway
    assert not inherits(("org/m", "Q4_K_M"), "org/m", "Q8_0")
    assert not inherits((str(other), "Q4_K_M"), str(snapshot), "Q4_K_M")  # a shared stem


def test_split_extras_drop_a_template_file_the_bundled_gemma4_template_replaces(monkeypatch):
    """The planner's extras stand in for the route's inheritance, so they get the same stripping."""
    import routes.inference as ri

    stored = ["--chat-template-file", "/tmp/old.jinja", "-c", "8192"]
    repo = "unsloth/gemma-4-31B-it-GGUF"

    class _Backend:
        extra_args = list(stored)
        extra_args_requested_source = (repo, "Q4_K_M")
        extra_args_source = (repo, "Q4_K_M")

    monkeypatch.setattr(ri, "get_llama_cpp_backend", lambda: _Backend())
    request = SimpleNamespace(
        model_path = repo, gguf_variant = "Q4_K_M", llama_extra_args = None,
        chat_template_override = None, model_fields_set = set(),
    )  # fmt: skip
    inherited = ri._split_inherited_extra_args(request)
    assert "--chat-template-file" not in inherited and inherited[-2:] == ["-c", "8192"]
    other = SimpleNamespace(**{**vars(request), "model_path": "org/plain-GGUF"})

    class _Plain(_Backend):
        extra_args_requested_source = ("org/plain-GGUF", "Q4_K_M")
        extra_args_source = ("org/plain-GGUF", "Q4_K_M")

    monkeypatch.setattr(ri, "get_llama_cpp_backend", lambda: _Plain())
    assert ri._split_inherited_extra_args(other)[:2] == ["--chat-template-file", "/tmp/old.jinja"]
