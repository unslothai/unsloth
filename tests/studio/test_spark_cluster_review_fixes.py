# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Two-Spark provisioning and the `spark train --run` launch lifecycle."""

from __future__ import annotations

import asyncio
import importlib.util
import itertools
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
_POSIX_ONLY = pytest.mark.skipif(sys.platform == "win32", reason = "runs the peer's bash script")


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, REPO / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def pipeline():
    return _load("spark_pipeline_for_review_fixes", "studio/spark_pipeline.py")


@pytest.fixture
def cluster():
    return _load("spark_cluster_for_review_fixes", "studio/spark_cluster.py")


@pytest.fixture
def lb():
    return _load("spark_lb_for_review_fixes", "studio/spark_lb.py")


def _uv_venv(tmp_path: Path):
    """A venv whose interpreter is a symlink into a uv-managed Python under the user's home."""
    home = tmp_path / "home"
    runtime = home / ".local" / "share" / "uv" / "python" / "cpython-3.13.5-linux-aarch64-gnu"
    (runtime / "bin").mkdir(parents = True)
    (runtime / "lib").mkdir()
    (runtime / "bin" / "python3.13").write_bytes(b"\x7fELF")
    venv = home / ".unsloth" / "studio" / "unsloth_studio"
    (venv / "bin").mkdir(parents = True)
    (venv / "pyvenv.cfg").write_text(f"home = {runtime}/bin\n", encoding = "utf-8")
    os.symlink(str(runtime / "bin" / "python3.13"), str(venv / "bin" / "python"))
    os.symlink("python", str(venv / "bin" / "python3"))
    return home, runtime, venv


@_POSIX_ONLY
def test_provisioning_copies_the_venvs_python_runtime(cluster, monkeypatch, tmp_path) -> None:
    home, runtime, _ = _uv_venv(tmp_path)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("UNSLOTH_STUDIO_HOME", raising = False)
    monkeypatch.delenv("STUDIO_HOME", raising = False)
    monkeypatch.setattr(cluster.Path, "home", classmethod(lambda cls: home))
    paths = dict((label, path) for path, label in cluster.provision_paths())
    assert paths.get("Python runtime") == os.path.realpath(str(runtime))
    assert cluster.peer_destination(paths["Python runtime"]).startswith("~/.local/share/uv/")


@_POSIX_ONLY
def test_relocation_fails_when_the_peer_lacks_the_interpreter(cluster, tmp_path) -> None:
    """rsync -a kept bin/python pointing at a uv Python the peer never had, relocation still exited
    0, and provisioning reported success for a venv whose torchrun cannot start."""
    _, runtime, venv = _uv_venv(tmp_path)
    for f in (runtime / "bin").iterdir():
        f.unlink()
    script = cluster.venv_relocate_script(str(venv), str(venv))
    res = subprocess.run(["bash", "-s"], input = script.encode(), capture_output = True)
    assert res.returncode != 0
    assert b"does not" in res.stderr


@_POSIX_ONLY
def test_relocation_repoints_the_interpreter_under_the_peers_home(cluster, tmp_path) -> None:
    ours, runtime, venv = _uv_venv(tmp_path / "ours")
    peer = tmp_path / "peer"
    subprocess.run(["cp", "-a", str(ours), str(peer)], check = True)
    peer_venv = peer / venv.relative_to(ours)
    peer_runtime = "$HOME/" + str(runtime.relative_to(ours))
    script = cluster.venv_relocate_script(
        str(venv), "$HOME/.unsloth/studio/unsloth_studio", str(runtime), peer_runtime
    )
    env = dict(os.environ, HOME = str(peer))
    subprocess.run(["bash", "-s"], input = script.encode(), env = env, check = True)
    target = os.readlink(str(peer_venv / "bin" / "python"))
    assert target == str(peer / runtime.relative_to(ours) / "bin" / "python3.13")
    assert str(peer) in (peer_venv / "pyvenv.cfg").read_text(encoding = "utf-8")


def _fake_pipeline_run(
    cluster,
    monkeypatch,
    *,
    local_rc = 1,
    sleep = None,
):
    monkeypatch.setattr(cluster, "_ssh_user", lambda: "someuser")
    monkeypatch.setattr(cluster, "venv_activate_sh", lambda: '"$HOME/a"')
    monkeypatch.setattr(cluster, "peer_home", lambda ip, user: "/home/peer")
    monkeypatch.setattr(
        cluster, "stage_run_inputs", lambda command, ip, user, home: (command, [], [])
    )
    monkeypatch.setattr(cluster.time, "sleep", sleep or (lambda s: None))
    stopped, sent = [], []
    monkeypatch.setattr(
        cluster,
        "stop_peer_by_pidfile",
        lambda ip, user, opts, pid_file: stopped.append(pid_file) or True,
    )

    class _Done:
        def __init__(self, rc):
            self.returncode = rc

    def run(*a, **k):
        sent.append(a[0])
        return _Done(0 if "ssh" in a[0] else local_rc)

    monkeypatch.setattr(cluster.subprocess, "run", run)
    return stopped, sent


def test_runs_on_different_ports_keep_their_own_peer_pid(cluster, monkeypatch) -> None:
    """Both runs wrote /tmp/unsloth_pp_stage1.pid, so the first one's failure cleanup killed the
    second one's peer rank."""
    stopped, sent = _fake_pipeline_run(cluster, monkeypatch)
    base = "torchrun --nnodes=2 --nproc_per_node=1 --master_addr=10.0.0.1 --master_port"
    plan = {
        "env": {},
        "node0": f"{base}=29501 --node_rank=0 true",
        "node1": f"{base}=29501 --node_rank=1 true",
        "peer_ip": "192.0.2.7",
    }
    assert cluster.run_pipeline(plan) == 1
    assert stopped and stopped[0] != cluster._PEER_STAGE_PID and "29501" in stopped[0]
    launch = next(c for c in sent if c[0] == "ssh" and "-f" in c)[-1]
    assert cluster._PEER_STAGE_PID not in launch and stopped[0] in launch
    stopped.clear()
    plan = {**plan, "node0": plan["node0"].replace("29501", "29500")}
    cluster.run_pipeline(plan)
    assert stopped == [cluster._PEER_STAGE_PID]


def test_ctrl_c_during_the_startup_wait_stops_the_peer(cluster, monkeypatch) -> None:
    def interrupt(seconds):
        raise KeyboardInterrupt

    stopped, _ = _fake_pipeline_run(cluster, monkeypatch, sleep = interrupt)
    plan = {"env": {}, "node0": "true", "node1": "true", "peer_ip": "192.0.2.7"}
    with pytest.raises(KeyboardInterrupt):
        cluster.run_pipeline(plan)
    assert stopped == [cluster._PEER_STAGE_PID]


def test_an_unexpanded_tilde_save_is_collected_from_the_peers_home(
    cluster, monkeypatch, tmp_path
) -> None:
    """`--save=~/out` reached the launcher literally: the peer stage was looked for under
    `$HOME/~/out`, and the stale-stage clear missed it the same way."""
    sent = []

    class _Done:
        returncode = 0
        stderr = ""

    monkeypatch.setattr(cluster.subprocess, "run", lambda cmd, **k: sent.append(cmd) or _Done())
    monkeypatch.setenv("HOME", str(tmp_path))
    assert cluster.collect_stage_outputs("~/out", "192.0.2.7", "u", "/home/peer") is None
    assert "u@192.0.2.7:/home/peer/out/stage1" in sent[0]
    assert cluster._peer_save_path("out", "/home/peer") == "/home/peer/out"
    assert cluster._peer_save_path("/data/out", "/home/peer") == "/data/out"


def test_the_pipeline_expands_a_tilde_save(pipeline) -> None:
    source = (REPO / "studio" / "spark_pipeline.py").read_text(encoding = "utf-8")
    main = source.split("\ndef main(")[1].split("\ndef ")[0]
    assert "args.save = osp.expanduser(args.save)" in main


def test_a_nonpositive_step_count_is_refused_before_any_setup(pipeline) -> None:
    parser = pipeline.build_parser()
    for bad in ("0", "-3"):
        with pytest.raises(SystemExit):
            parser.parse_args(["--model", "m", "--steps", bad])
    assert parser.parse_args(["--model", "m", "--steps", "1"]).steps == 1


def test_the_venv_and_its_runtime_are_mirrored_and_caches_stay_additive(
    cluster, monkeypatch, tmp_path
) -> None:
    dirs = {}
    for label in (cluster._VENV_LABEL, cluster._PYTHON_RUNTIME_LABEL, "HF cache"):
        d = tmp_path / label.replace(" ", "_")
        d.mkdir()
        dirs[label] = str(d)
    calls = []

    def fake_run(cmd, *a, **k):
        calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, stdout = "", stderr = "")

    monkeypatch.setattr(cluster.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(cluster, "provision_paths", lambda: [(p, l) for l, p in dirs.items()])
    monkeypatch.setattr(cluster, "_ssh_user", lambda: "me")
    monkeypatch.setattr(
        cluster, "peer_gpu_busy", lambda ip: {"busy": False, "reason": "", "processes": []}
    )
    monkeypatch.setattr(cluster, "_relocate_peer_venv", lambda *a: "")
    monkeypatch.setattr(cluster.subprocess, "run", fake_run)

    def deleting(force):
        calls.clear()
        cluster.provision_peer("192.0.2.7", no_fast = True, force = force)
        return {
            label: "--delete" in next(c for c in calls if c[0] == "rsync" and c[-2] == p + "/")
            for label, p in dirs.items()
        }

    assert deleting(force = False) == {
        cluster._VENV_LABEL: True,
        cluster._PYTHON_RUNTIME_LABEL: True,
        "HF cache": False,
    }
    # --force only bypasses the busy-peer refusal; the venv mirror stays exact.
    assert deleting(force = True) == deleting(force = False)


def test_switched_setup_labels_each_block_with_its_own_node(cluster, capsys) -> None:
    def plan(octet):
        return [
            {"netdev": "enp1", "address": f"192.168.100.{octet}", "prefix": "24", "mtu": "9000"}
        ]

    cluster._print_manual_steps(plan(13), plan(12), extra_plans = [plan(14)], node_indices = [0, 2])
    out = capsys.readouterr().out
    assert out.index("Spark node 0") < out.index("192.168.100.12") < out.index("Spark node 2")
    assert out.index("Spark node 2") < out.index("192.168.100.14")


def test_hyperclovax_multiplies_by_logits_scaling_and_granite_divides(pipeline) -> None:
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace

    logits = torch.full((1, 2, 3), 4.0)
    granite = SimpleNamespace(model_type = "granite", logits_scaling = 8.0)
    clova = SimpleNamespace(model_type = "hyperclovax", logits_scaling = 0.5)
    assert torch.allclose(pipeline._softcap_logits(granite, logits), logits / 8.0)
    assert torch.allclose(pipeline._softcap_logits(clova, logits), logits * 0.5)


def test_a_nonpositive_batch_is_refused_before_any_setup(pipeline) -> None:
    parser = pipeline.build_parser()
    for bad in ("0", "-4"):
        with pytest.raises(SystemExit):
            parser.parse_args(["--model", "m", "--batch", bad])


def test_a_gguf_without_head_count_kv_prices_kv_with_the_attention_heads(cluster) -> None:
    meta = {
        "general.architecture": "llama",
        "llama.block_count": 2,
        "llama.attention.head_count": 4,
        "llama.embedding_length": 64,
    }
    assert cluster.gguf_kv_bytes_per_token(meta) == 2 * 4 * (16 + 16) * 2.0


def test_an_incomplete_split_gguf_has_no_size(cluster, tmp_path) -> None:
    files = []
    for i in (1, 2):
        f = tmp_path / f"m-0000{i}-of-00003.gguf"
        f.write_bytes(b"x" * 10)
        files.append((f.name, str(f)))
    report = cluster._size_one_group(files)
    assert report["bytes"] is None and "only 2" in report["why"]
    f = tmp_path / "m-00003-of-00003.gguf"
    f.write_bytes(b"x" * 10)
    assert cluster._size_one_group(files + [(f.name, str(f))])["bytes"] == 30


def test_a_sequence_needs_a_next_token_to_predict(pipeline) -> None:
    parser = pipeline.build_parser()
    for bad in ("1", "0", "-2"):
        with pytest.raises(SystemExit):
            parser.parse_args(["--model", "m", "--seq", bad])
    assert parser.parse_args(["--model", "m", "--seq", "2"]).seq == 2


def test_serve_refuses_a_nonpositive_context_or_slot_count(cluster, monkeypatch, capsys) -> None:
    monkeypatch.setattr(cluster, "is_dgx_spark", lambda: True)
    for ctx, slots in ((0, 16), (8192, 0), (-1, 4)):
        assert cluster._cmd_serve("m.gguf", ctx = ctx, slots = slots) == 2
    assert "must both be at least 1" in capsys.readouterr().out


def test_an_update_of_a_paired_install_says_the_peer_needs_provisioning() -> None:
    text = (REPO / "install.sh").read_text(encoding = "utf-8")
    start = text.index('if [ "$_sp_state" = "configured" ]; then')
    block = text[start : text.index("return 0", start)]
    assert "studio.spark_cluster provision" in block and "unsloth spark provision" in block


@pytest.mark.parametrize(
    "batch, microbatches, seq, needle",
    [
        (0, 1, 512, "--batch (0) must be at least 1."),
        (-8, 4, 512, "--batch (-8) must be at least 1."),
        (8, 0, 512, "--microbatches (0) must be at least 1."),
        (8, 4, 1, "--seq (1) must be at least 2"),
        (8, 4, 0, "--seq (0) must be at least 2"),
        (10, 4, 512, "--batch (10) must be a multiple of --microbatches (4)"),
    ],
)
def test_estimate_refuses_the_shapes_train_refuses(
    cluster, monkeypatch, capsys, batch, microbatches, seq, needle
) -> None:
    monkeypatch.setattr(cluster, "is_dgx_spark", lambda: True)
    monkeypatch.setattr(cluster, "model_size_report", lambda m: {"gib": 1.0, "why": ""})
    monkeypatch.setattr(cluster, "bf16_resident_gib", lambda m: None)
    monkeypatch.setattr(
        cluster, "model_dimensions", lambda m: {"hidden": 64, "layers": 2, "vocab": 128}
    )
    monkeypatch.setattr(cluster, "peer_ip_for", lambda: None)
    assert cluster.training_shape_error(batch, microbatches, seq) is not None
    assert cluster._cmd_estimate("m", batch, microbatches, seq, False, False) == 2
    out = capsys.readouterr().out
    assert needle in out and "OK:" not in out
    # A valid shape still estimates.
    assert cluster.training_shape_error(8, 4, 512) is None
    assert cluster._cmd_estimate("m", 8, 4, 512, False, False) == 0


def test_a_directory_holding_one_gguf_is_served_by_llama_cpp(cluster, tmp_path) -> None:
    model_dir = tmp_path / "my model"
    model_dir.mkdir()
    (model_dir / "m-Q4_K_M.gguf").write_bytes(b"x" * 16)
    (model_dir / "README.md").write_text("card")
    gguf = str(model_dir / "m-Q4_K_M.gguf")
    for axis in ("single", "replicas", "layer-split"):
        lines = cluster._serve_commands(axis, 2, str(model_dir))
        text = "\n".join(lines)
        assert not any(line.startswith("vllm serve") for line in lines), (axis, text)
        assert "ray start" not in text, (axis, text)
        assert shlex.quote(gguf) in text, (axis, text)
    # A sharded GGUF series is pointed at shard 1.
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    for i in (2, 1):
        (shard_dir / f"m-0000{i}-of-00002.gguf").write_bytes(b"x" * 8)
    single = cluster._serve_commands("single", 1, str(shard_dir))[0]
    assert str(shard_dir / "m-00001-of-00002.gguf") in single and "vllm" not in single
    # A safetensors directory still goes to vLLM.
    hf_dir = tmp_path / "hf"
    hf_dir.mkdir()
    (hf_dir / "model.safetensors").write_bytes(b"x" * 8)
    assert cluster._serve_commands("single", 1, str(hf_dir))[0].startswith("vllm serve")


def _kv_gguf(path: Path) -> Path:
    """A GGUF whose header declares 4 layers x 2 KV heads x (64 + 64) dims."""
    import struct

    def kv(key: str, value_type: int, payload: bytes) -> bytes:
        name = key.encode()
        return struct.pack("<Q", len(name)) + name + struct.pack("<I", value_type) + payload

    entries = [
        kv("general.architecture", 8, struct.pack("<Q", 5) + b"llama"),
        kv("llama.block_count", 4, struct.pack("<I", 4)),
        kv("llama.attention.head_count_kv", 4, struct.pack("<I", 2)),
        kv("llama.attention.key_length", 4, struct.pack("<I", 64)),
        kv("llama.attention.value_length", 4, struct.pack("<I", 64)),
    ]
    header = b"GGUF" + struct.pack("<I", 3) + struct.pack("<QQ", 0, len(entries))
    path.write_bytes(header + b"".join(entries))
    return path


def test_kv_per_user_is_read_through_a_gguf_directory_or_cached_repo(
    cluster, tmp_path, monkeypatch
) -> None:
    expected = 4 * 2 * (64 + 64) * 2.0 * 1024 / 2**30
    model_dir = tmp_path / "gguf dir"
    model_dir.mkdir()
    gguf = _kv_gguf(model_dir / "m-Q4_K_M.gguf")
    (model_dir / "README.md").write_text("card")
    assert cluster.serving_kv_gib_per_user(str(gguf), 1024)["gib"] == pytest.approx(expected)
    assert cluster.serving_kv_gib_per_user(str(model_dir), 1024)["gib"] == pytest.approx(expected)
    # The same GGUF reached as a cached HF repo id.
    cache = tmp_path / "hub"
    snapshot = cache / "models--org--m-GGUF" / "snapshots" / "abc"
    snapshot.mkdir(parents = True)
    _kv_gguf(snapshot / "m-Q4_K_M.gguf")
    (cache / "models--org--m-GGUF" / "refs").mkdir()
    (cache / "models--org--m-GGUF" / "refs" / "main").write_text("abc")
    monkeypatch.setattr(cluster, "hf_hub_cache", lambda: str(cache))
    assert cluster.model_size_report("org/m-GGUF").get("group", "").startswith("gguf:")
    assert cluster.serving_kv_gib_per_user("org/m-GGUF", 1024)["gib"] == pytest.approx(expected)
    # A safetensors directory still has no KV estimate.
    hf_dir = tmp_path / "hf"
    hf_dir.mkdir()
    (hf_dir / "model.safetensors").write_bytes(b"x" * 8)
    assert cluster.serving_kv_gib_per_user(str(hf_dir), 1024)["gib"] is None


def _gguf_commands(plan) -> list:
    return [c for c in plan["commands"] if "llama-server" in c or "spark serve" in c]


@pytest.mark.parametrize(
    "intent, concurrency, n_nodes, size, kv",
    [
        ("throughput", 16, 2, 10.0, 0.5),  # replicas
        ("throughput", 2, 2, 10.0, 0.5),  # single
        ("capacity", 4, 1, 10.0, 0.5),  # one node
        ("throughput", 2, 2, 20.0, 0.5),  # prefill-heavy below: layer split
    ],
)
def test_every_planned_gguf_command_carries_the_planned_context_and_users(
    cluster, intent, concurrency, n_nodes, size, kv
) -> None:
    plan = cluster.plan_deployment(
        size,
        n_nodes = n_nodes,
        intent = intent,
        concurrency = concurrency,
        model = "/m/model.gguf",
        kv_gib_per_user = kv,
        ctx = 4096,
        prefill_heavy = size == 20.0,
    )
    commands = _gguf_commands(plan)
    assert commands, plan["commands"]
    per_server = -(-concurrency // n_nodes) if plan["axis"] == "replicas" else concurrency
    for line in commands:
        if "spark serve" in line:
            assert f"--ctx 4096 --slots {concurrency}" in line, line
        else:
            assert f"--ctx-size {4096 * per_server} --parallel {per_server}" in line, line


def test_replica_fit_prices_only_each_replicas_share_of_the_users(cluster) -> None:
    """100 GiB model, 1 GiB KV per user, 16 users, two 113.69 GiB nodes: 108 GiB per replica."""
    gib = 2**30
    serving = cluster.recommend_topology(100 * gib, 1 * gib, 16, 512, 113.69 * gib, replicas = 2)
    assert serving["topology"] == "replicas", serving["reason"]
    assert serving["replica_node_bytes"] == 108 * gib
    # One copy holding every user's KV still does not fit, so the default stays a layer split.
    assert (
        cluster.recommend_topology(100 * gib, 1 * gib, 16, 512, 113.69 * gib)["topology"]
        == "layer_split"
    )

    plan = cluster.plan_deployment(
        100.0,
        n_nodes = 2,
        intent = "throughput",
        concurrency = 16,
        model = "/m/model.gguf",
        kv_gib_per_user = 1.0,
        ctx = 8192,
    )
    assert plan["serving"]["topology"] == "replicas"
    assert plan["axis"] == "replicas", plan["recommendation"]
    assert any("--ctx-size 65536 --parallel 8" in c for c in plan["commands"])
    # `spark serve --slots` is per engine, so every replica keeps all of them.
    plan = cluster.plan_deployment(
        100.0,
        two_sparks = True,
        concurrency = 16,
        kv_gib_per_user = 1.0,
        split_kv_across_replicas = False,
    )
    assert plan["serving"]["topology"] == "layer_split"


def test_training_plan_refuses_a_model_beyond_the_layer_split_capacity(cluster) -> None:
    budget = cluster.SPARK_USABLE_GIB - cluster.SERVE_OVERHEAD_GIB
    # Inside the two-node aggregate the layer split is still the plan.
    assert cluster.plan_training(budget * 1.5, n_nodes = 2, model = "m")["axis"] == (
        "pipeline-parallel"
    )
    # Beyond it no node count helps: the launcher is torchrun --nnodes=2.
    for nodes in (2, 4):
        plan = cluster.plan_training(300.0, n_nodes = nodes, model = "m")
        assert plan["axis"] == "none", plan["recommendation"]
        assert plan["commands"] == []
        assert plan["fits_one_node"] is False
        assert "--layer-split" not in " ".join(plan["commands"])
        assert "aggregate" in plan["recommendation"]
    one = cluster.plan_training(300.0, n_nodes = 1, model = "m")
    assert one["axis"] == "none"
    assert "Pair a second Spark" not in one["recommendation"]
