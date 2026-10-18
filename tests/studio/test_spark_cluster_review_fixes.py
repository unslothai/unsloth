# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Two-Spark provisioning and the `spark train --run` launch lifecycle."""

from __future__ import annotations

import asyncio
import importlib.util
import itertools
import json
import os
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
    # The default port keeps the established paths.
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
