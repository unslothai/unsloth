# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Gaps that only open once the surrounding code is right.

Three are the second half of an earlier fix: the planner grew a KV input that the public
`plan` path still passed as zero, the drift check compares rails that are present and so
cannot see one that vanished, and waiting for the peer rank to disappear is not the same as
waiting for it to succeed. The rest are a failure mode that has no exception to raise --
a blackholed backend, an ssh that returns 255, a parameter divided twice.
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
    return _load("spark_cluster_for_r19", "studio/spark_cluster.py")


@pytest.fixture
def lb():
    return _load("spark_lb_for_r19", "studio/spark_lb.py")


def test_the_public_plan_passes_the_gguf_kv_through(monkeypatch) -> None:
    """`plan` and `serve` are supposed to give the same answer for the same model. `serve`
    computed the KV and `plan` handed the planner a zero, so `plan` recommended replicas for a
    model whose weights fit and whose KV does not."""
    from unsloth_cli.commands import spark as cmd

    seen = {}

    class _SC:
        # The real signature, because the shim feature-detects with `inspect.signature`.
        @staticmethod
        def plan_deployment(
            size_gib,
            two_sparks = None,
            *,
            n_nodes = None,
            intent = "throughput",
            concurrency = 1,
            model = "<model>",
            prompt_tokens = None,
            prefill_heavy = False,
            kv_gib_per_user = 0.0,
        ):
            seen.update(
                n_nodes = n_nodes, concurrency = concurrency, kv_gib_per_user = kv_gib_per_user
            )
            return {"topology": "replicas"}

    cmd._plan_deployment(_SC, 105.0, 2, concurrency = 16, kv_gib_per_user = 1.0)
    assert seen["kv_gib_per_user"] == 1.0

    source = (REPO / "unsloth_cli" / "commands" / "spark.py").read_text()
    body = source.split('@spark_app.command("plan")')[1].split("@spark_app.command")[0]
    assert "serving_kv_gib_per_user(model, ctx)" in body
    assert "kv_gib_per_user = kv.get(\"gib\") or 0.0" in body


def test_a_planned_rail_that_disappears_is_drift(cluster, monkeypatch) -> None:
    saved = {
        "enabled": True,
        "rails": [
            {"netdev": "enp1s0f0np0", "address": "192.168.200.12", "mtu": "9000"},
            {"netdev": "enp1s0f1np1", "address": "192.168.201.12", "mtu": "9000"},
        ],
    }
    monkeypatch.setattr(cluster, "load_config", lambda: saved)
    present = [
        {
            "netdev": "enp1s0f0np0",
            "ib_device": "rocep1s0",
            "ipv4": ["192.168.200.12"],
            "mtu": 9000,
        }
    ]
    monkeypatch.setattr(cluster, "cabled_rails", lambda: present)
    problems = cluster.cluster_config_problems()
    assert any("enp1s0f1np1" in p and "not cabled" in p for p in problems), problems

    monkeypatch.setattr(
        cluster,
        "cabled_rails",
        lambda: present
        + [
            {
                "netdev": "enp1s0f1np1",
                "ib_device": "rocep1s0f1",
                "ipv4": ["192.168.201.12"],
                "mtu": 9000,
            }
        ],
    )
    assert cluster.cluster_config_problems() == []


def test_the_ddp_script_path_is_quoted(cluster, monkeypatch) -> None:
    monkeypatch.setattr(cluster, "is_dgx_spark", lambda: True)
    monkeypatch.setattr(cluster, "peer_ip_for", lambda: "192.168.200.13")
    monkeypatch.setattr(
        cluster, "cabled_rails", lambda: [{"ipv4": ["192.168.200.12"], "netdev": "x"}]
    )
    monkeypatch.setattr(cluster, "nccl_env", lambda: {})
    plan = cluster.train_launch_plan("/work/my train.py")
    assert "'/work/my train.py'" in plan["node0"]
    assert "'/work/my train.py'" in plan["node1"]


def test_the_peer_replica_executable_is_quoted() -> None:
    source = (REPO / "studio" / "spark_cluster.py").read_text()
    assert 'peer_server_cmd = f\'"{peer_bin_dir}/llama-server"\'' in source


def test_a_shared_parameter_is_scaled_once() -> None:
    """A V layout puts the first and last stage on one rank, so a tied embedding is the same
    Parameter in two stage modules: it was divided once per module, 1/M**2 against 1/M."""
    source = (REPO / "studio" / "spark_pipeline.py").read_text()
    body = source.split("def scale_grads_after_step", 1)[1].split("\n    where =", 1)[0]
    assert "seen = set()" in body and "id(p) not in seen" in body


def test_the_refusal_no_longer_points_at_the_legacy_backend() -> None:
    source = (REPO / "studio" / "spark_pipeline.py").read_text()
    refusal = source.split("carries {sorted(skipped)}", 1)[1].split(")", 1)[0]
    assert "single node" in refusal
    assert "NOT " in refusal and "legacy" in refusal


def test_blank_jsonl_lines_are_skipped() -> None:
    source = (REPO / "studio" / "spark_pipeline.py").read_text()
    assert "json.loads(line)" in source and "if line.strip()" in source


def test_the_peer_stage_must_report_zero_before_collection(cluster, monkeypatch) -> None:
    codes = {"rc": 0}

    def _run(argv, **kw):
        class R:
            returncode = codes["rc"]
            stdout = b""
            stderr = b""

        return R()

    monkeypatch.setattr(cluster.subprocess, "run", _run)
    assert cluster.wait_for_peer_stage("h", "u", "/tmp/p")["ok"] is True

    # Gone, but it recorded a nonzero status: not a finished run.
    codes["rc"] = 2
    bad = cluster.wait_for_peer_stage("h", "u", "/tmp/p")
    assert bad["ok"] is False and "NONZERO" in bad["why"]

    # Gone with no status at all, which is what an abrupt death looks like.
    codes["rc"] = 3
    gone = cluster.wait_for_peer_stage("h", "u", "/tmp/p")
    assert gone["ok"] is False and "no exit status" in gone["why"]

    # And the remote command really does consult the status file, not just the pid.
    source = (REPO / "studio" / "spark_cluster.py").read_text()
    assert "cat {rc_file}" in source or "$(cat {rc_file}" in source


def test_a_failed_peer_probe_launch_does_not_start_rank_zero() -> None:
    source = (REPO / "studio" / "spark_cluster.py").read_text()
    probe = source.split("def nccl_bandwidth", 1)[1].split("\ndef ", 1)[0]
    assert "if started.returncode != 0:" in probe
    # And it stops whatever it may have started before giving up.
    after = probe.split("if started.returncode != 0:", 1)[1]
    assert "stop_peer_nccl_probe" in after.split("return None", 1)[0]


def test_the_load_balancer_bounds_each_connection_attempt(lb) -> None:
    assert lb.CONNECT_TIMEOUT > 0
    source = (REPO / "studio" / "spark_lb.py").read_text()
    assert "asyncio.wait_for(" in source and "timeout = CONNECT_TIMEOUT" in source
    assert "except (OSError, asyncio.TimeoutError):" in source
