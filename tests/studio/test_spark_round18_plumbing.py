# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Commands that are printed to be pasted, and paths that are interpolated into shells.

Four of these are the same defect in four places: a path is put into a command without being
quoted, so a Studio home or a checkpoint with a space in it silently runs something else. Two
are the reverse, a bare name assumed to be on PATH when the installer never puts it there. The
rest are states a probe collapsed together, and an architecture's own module names.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, REPO / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def cluster():
    return _load("spark_cluster_for_r18", "studio/spark_cluster.py")


@pytest.fixture
def pipeline():
    return _load("spark_pipeline_for_r18", "studio/spark_pipeline.py")


def test_generated_serve_commands_quote_a_model_path_with_a_space(cluster) -> None:
    model = "/models/large model/q4.gguf"
    for axis in ("tensor-parallel", "replicas", "layer-split", "single"):
        for line in cluster._serve_commands(axis, 2, model):
            assert "large model" not in line or "'/models/large model/q4.gguf'" in line, (
                axis,
                line,
            )
    # The placeholder still reads as a placeholder rather than a quoted string.
    assert any("<model>" in line for line in cluster._serve_commands("single", 1))


def test_the_peer_launch_survives_a_staged_path_with_a_space(cluster, monkeypatch) -> None:
    """`stage_run_inputs` returns a shlex.join-ed command, and it used to be pasted inside a
    hand-written `bash -c '...'`, where its quotes closed the outer string early."""
    import shlex

    seen = {}

    def _run(argv, **kw):
        if argv[0] == "ssh":
            seen.setdefault("remote", argv[-1])

        class R:
            returncode = 0
            stdout = ""
            stderr = ""

        return R()

    monkeypatch.setattr(cluster.subprocess, "run", _run)
    monkeypatch.setattr(cluster, "peer_home", lambda ip, user: "/home/peer")
    monkeypatch.setattr(cluster, "venv_activate_sh", lambda: '"$HOME/a/bin/activate"')
    node1 = shlex.join(
        ["torchrun", "--node_rank=1", "-m", "studio.spark_pipeline", "--model", "/m/large model"]
    )
    monkeypatch.setattr(cluster, "stage_run_inputs", lambda c, ip, u, h: (node1, [], []))
    monkeypatch.setattr(cluster, "wait_for_peer_stage", lambda *a, **k: True)
    monkeypatch.setattr(cluster, "collect_stage_outputs", lambda *a, **k: None)
    monkeypatch.setattr(cluster.time, "sleep", lambda *a: None)
    cluster.run_pipeline(
        {
            "env": {"NCCL_DEBUG": "WARN"},
            "node0": "torchrun --node_rank=0 -m studio.spark_pipeline",
            "node1": node1,
            "peer_ip": "192.168.200.13",
            "local_ip": "192.168.200.12",
        }
    )
    remote = seen["remote"]
    # One `bash -c` argument, and the staged path is still one word inside it.
    body = remote.split("bash -c ", 1)[1]
    assert body.startswith("'")
    inner = shlex.split(body)[0]
    assert "--model /m/large model" in inner or "'/m/large model'" in inner
    assert shlex.split(inner.split("exec ", 1)[1])[-1] == "/m/large model"


def test_rank_zero_of_the_nccl_probe_resolves_torchrun(cluster) -> None:
    src = (REPO / "studio" / "spark_cluster.py").read_text()
    probe = src.split("def nccl_bandwidth", 1)[1].split("\ndef ", 1)[0]
    assert "local_common = _common(managed_torchrun())" in probe
    assert "--node_rank=0" in probe and "{local_common}" in probe


def test_the_printed_training_commands_include_activation(cluster, monkeypatch, capsys) -> None:
    monkeypatch.setattr(cluster, "venv_activate_sh", lambda: '"$HOME/a/bin/activate"')
    cluster._print_launch(
        {
            "env": {"NCCL_NET_GDR_LEVEL": "0"},
            "node0": "torchrun --node_rank=0 x",
            "node1": "torchrun --node_rank=1 x",
            "local_ip": "192.168.200.12",
            "peer_ip": "192.168.200.13",
        }
    )
    out = capsys.readouterr().out
    assert '. "$HOME/a/bin/activate"' in out
    assert "export NCCL_NET_GDR_LEVEL=0" in out


def test_a_silent_listener_is_not_the_same_as_an_absent_one(cluster, monkeypatch) -> None:
    """A peer that accepts and never answers HELLO holds the port, so a new server cannot
    bind it. It used to be classed with `refused`, the normal pre-launch state."""

    def _probe(
        host,
        port = 0,
        **kw,
    ):
        return {"host": host, "port": port, "state": "silent", "version": None}

    monkeypatch.setattr(cluster, "rpc_hello_probe_detail", _probe)
    monkeypatch.setattr(cluster, "llama_bundle_identity", lambda: {})
    monkeypatch.setattr(cluster, "peer_llama_bundle_identity", lambda ip: {})
    monkeypatch.setattr(
        cluster,
        "compare_llama_bundles",
        lambda a, b: {"ok": True, "problems": [], "notes": []},
    )
    res = cluster.rpc_protocol_preflight("192.168.200.13", 50052)
    assert res["ok"] is False
    assert any("never answered the HELLO" in p for p in res["problems"]), res


def test_one_cabled_rail_is_planned_but_reported(cluster) -> None:
    rails = [{"ib_device": "rocep1s0", "netdev": "enp1s0f0np0", "ipv4": ["192.168.200.12"]}]
    report = cluster.rail_plan_report(rails, node_index = 0, n_nodes = 2)
    assert report["ok"] is True and report["degraded"] is True
    assert any("DEGRADED" in n for n in report["notes"])

    both = rails + [{"ib_device": "rocep1s0f1", "netdev": "enp1s0f1np1", "ipv4": []}]
    full = cluster.rail_plan_report(both, node_index = 0, n_nodes = 2)
    assert full["degraded"] is False and not any("DEGRADED" in n for n in full["notes"])


def test_the_embedding_is_dropped_by_the_architectures_own_name(pipeline) -> None:
    pytest.importorskip("torch")
    import torch.nn as nn

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.embed_in = nn.Embedding(8, 4)
            self.model.layers = nn.ModuleList([nn.Identity() for _ in range(4)])
            self.model.norm = nn.Identity()
            self.lm_head = nn.Linear(4, 8)

    m = Model()
    owner, _ = pipeline.find_layers(m)
    name, _ = pipeline._first_named(owner, pipeline._EMBED_NAMES)
    assert name == "embed_in"

    # And `build_stage_model` drops it by that name rather than by `embed_tokens`, which is
    # what left the whole table resident on every rank.
    src = (REPO / "studio" / "spark_pipeline.py").read_text()
    body = src.split("def build_stage_model", 1)[1].split("\ndef ", 1)[0]
    assert "_first_named(owner, _EMBED_NAMES)" in body
    assert 'hasattr(owner, "embed_tokens")' not in body
