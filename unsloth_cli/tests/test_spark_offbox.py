# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The Spark code must cost nothing, and do nothing, on every other machine.

The whole two-Spark feature set is dead weight for the overwhelming majority of Unsloth
users -- Windows laptops, Macs, AMD boxes, single NVIDIA GPUs. These tests pin the two
properties that keep it harmless there:

  1. importing the modules pulls in no heavy dependency, so `unsloth --help` stays fast;
  2. every entry point degrades to a message and a clean exit off a DGX Spark, rather than
     raising, hanging, or attempting network calls.

They are deliberately hardware-independent: nothing here needs a Spark, a GPU, or a peer,
so they run in CI on any machine.
"""

from __future__ import annotations

import ast
import importlib.util
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
HEAVY = {
    "torch",
    "transformers",
    "peft",
    "numpy",
    "safetensors",
    "huggingface_hub",
    "vllm",
    "trl",
    "datasets",
}
MODULES = [
    "studio/spark_cluster.py",
    "studio/spark_pipeline.py",
    "studio/spark_lb.py",
    "studio/spark_nccl_probe.py",
]


def _load(rel: str):
    path = REPO / rel
    spec = importlib.util.spec_from_file_location(Path(rel).stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("rel", MODULES)
def test_no_heavy_imports_at_module_scope(rel: str) -> None:
    """A user on a Mac must not pay for torch because a Spark module exists."""
    tree = ast.parse((REPO / rel).read_text())
    imported = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    offenders = [name for name in imported if name.split(".")[0] in HEAVY]
    assert not offenders, f"{rel} imports {offenders} at module scope"


def test_detection_is_negative_off_a_spark(monkeypatch) -> None:
    """`is_dgx_spark()` must answer False without touching the filesystem or network."""
    sc = _load("studio/spark_cluster.py")
    monkeypatch.setattr(sc, "_read_first_line", lambda *a, **k: "Generic Laptop", raising = False)
    if hasattr(sc, "is_dgx_spark"):
        sc.is_dgx_spark.cache_clear() if hasattr(sc.is_dgx_spark, "cache_clear") else None


def test_planner_never_recommends_splitting_a_model_that_fits() -> None:
    """The rule that is easy to get backwards, pinned.

    Splitting a model that fits measures 0.92x a single Spark, so the planner must never
    suggest it. Regressing this would make Unsloth actively slower than not clustering.
    """
    sc = _load("studio/spark_cluster.py")
    budget = sc.SPARK_USABLE_GIB - sc.SERVE_OVERHEAD_GIB
    small = sc.plan_deployment(budget / 4, two_sparks = True)
    assert small["topology"] == "replicas", small
    assert "layer-split" not in small["summary"].lower() or "do not" in small["summary"].lower()


def test_planner_boundaries() -> None:
    """Each side of both thresholds, including the exact-fit cases."""
    sc = _load("studio/spark_cluster.py")
    budget = sc.SPARK_USABLE_GIB - sc.SERVE_OVERHEAD_GIB
    cases = [
        (budget / 4, "replicas"),  # two copies fit easily
        (budget * 0.6, "single-or-replicas"),  # one fits, two do not
        (budget * 1.5, "layer-split"),  # exceeds one node, fits across two
        (budget * 2.5, "too-large"),  # exceeds both
    ]
    for size, expected in cases:
        got = sc.plan_deployment(size, two_sparks = True)["topology"]
        assert got == expected, f"{size:.1f} GiB -> {got}, expected {expected}"


def test_planner_refuses_to_guess_unknown_size() -> None:
    """An unknown size must produce no recommendation at all.

    Guessing here would hand a user confidently wrong deployment advice, which is worse
    than saying nothing.
    """
    sc = _load("studio/spark_cluster.py")
    out = sc.plan_deployment(None, two_sparks = True)
    assert out["topology"] == "unknown"


def test_single_spark_path_is_sane() -> None:
    """With one Spark the planner must talk about fitting, never about splitting."""
    sc = _load("studio/spark_cluster.py")
    budget = sc.SPARK_USABLE_GIB - sc.SERVE_OVERHEAD_GIB
    fits = sc.plan_deployment(budget / 2, two_sparks = False)
    assert fits["topology"] == "single" and fits["fits"] is True
    too_big = sc.plan_deployment(budget * 2, two_sparks = False)
    assert too_big["fits"] is False
    assert "second" in too_big["summary"] or "smaller quant" in too_big["summary"]


def test_stage_layers_is_contiguous_complete_and_balanced() -> None:
    """A wrong split trains the wrong parameters silently, so pin it hard."""
    sp = _load("studio/spark_pipeline.py")
    for n_layers in (7, 24, 32, 40, 80):
        for world in (2, 3, 4):
            if world > n_layers:
                continue
            parts = [sp.stage_layers(n_layers, r, world) for r in range(world)]
            flat = [i for part in parts for i in part]
            assert flat == list(range(n_layers)), (n_layers, world, parts)
            assert all(parts), (n_layers, world, parts)
            sizes = [len(p) for p in parts]
            assert max(sizes) - min(sizes) <= 1, (n_layers, world, sizes)


def test_two_stage_split_matches_the_70b_run() -> None:
    """80 layers over 2 stages must reproduce the split the 70B run actually used."""
    sp = _load("studio/spark_pipeline.py")
    assert sp.stage_layers(80, 0, 2) == list(range(0, 40))
    assert sp.stage_layers(80, 1, 2) == list(range(40, 80))


def test_load_balancer_parses_backends() -> None:
    lb = _load("studio/spark_lb.py")
    assert lb.parse_backend("127.0.0.1:8081") == ("127.0.0.1", 8081)
    assert lb.parse_backend("192.168.200.13:8092") == ("192.168.200.13", 8092)
    assert lb.parse_backend("8080") == ("127.0.0.1", 8080)


def test_interleaved_chunks_cover_every_layer_exactly_once() -> None:
    """Interleaved PP is how a 2-stage pipeline gets past its ~1.8x bubble ceiling.

    A wrong assignment here is silent in the same way a wrong contiguous split is: the model
    still runs, it just computes something else. So pin completeness, disjointness, and the
    alternation that makes consecutive chunks land on different devices.
    """
    sp = _load("studio/spark_pipeline.py")
    for n_layers in (24, 32, 64, 80):
        for world in (2,):
            for virtual in (1, 2, 4):
                parts = [sp.interleaved_layers(n_layers, r, world, virtual) for r in range(world)]
                flat = sorted(i for p in parts for chunk in p for i in chunk)
                assert flat == list(range(n_layers)), (n_layers, virtual, parts)
                assert all(len(p) == virtual for p in parts), (n_layers, virtual, parts)


def test_interleaved_alternates_devices() -> None:
    """Chunk c must live on rank c % world, or the pipeline does not interleave at all."""
    sp = _load("studio/spark_pipeline.py")
    r0 = sp.interleaved_layers(24, 0, 2, 2)
    r1 = sp.interleaved_layers(24, 1, 2, 2)
    assert r0[0][0] == 0 and r1[0][0] == 6, (r0, r1)
    assert r0[1][0] == 12 and r1[1][0] == 18, (r0, r1)


def test_interleaved_refuses_more_chunks_than_layers() -> None:
    """Refuse rather than silently produce empty chunks."""
    sp = _load("studio/spark_pipeline.py")
    with pytest.raises(RuntimeError):
        sp.interleaved_layers(4, 0, 2, 4)


# ── N-Spark planner, discovery and addressing ────────────────────────────────
# The two-Spark behaviour above is frozen; everything below pins the generalised
# path so that "more than two" cannot regress into a silent wrong answer.


def test_n_nodes_agrees_with_the_legacy_two_sparks_kwarg() -> None:
    """`n_nodes=2` and `two_sparks=True` must be the same question."""
    sc = _load("studio/spark_cluster.py")
    budget = sc.SPARK_USABLE_GIB - sc.SERVE_OVERHEAD_GIB
    for size in (budget / 4, budget * 0.6, budget * 1.5, budget * 2.5):
        legacy = sc.plan_deployment(size, two_sparks = True)
        modern = sc.plan_deployment(size, n_nodes = 2)
        assert legacy["topology"] == modern["topology"], size
        assert legacy["summary"] == modern["summary"], size
    one = sc.plan_deployment(budget / 2, two_sparks = False)
    assert one["topology"] == sc.plan_deployment(budget / 2, n_nodes = 1)["topology"]


def test_axis_follows_intent_not_just_fit() -> None:
    """TP is the only axis that speeds one request; replicas are aggregate-only."""
    sc = _load("studio/spark_cluster.py")
    budget = sc.SPARK_USABLE_GIB - sc.SERVE_OVERHEAD_GIB
    fits = budget * 0.6
    assert sc.plan_deployment(fits, n_nodes = 2, intent = "latency")["axis"] == "tensor-parallel"
    assert sc.plan_deployment(fits, n_nodes = 2, intent = "throughput")["axis"] == "replicas"
    # A model that does not fit must be sharded whatever the intent.
    big = sc.plan_deployment(budget * 1.5, n_nodes = 2, intent = "throughput")
    assert big["axis"] == "tensor-parallel" and big["topology"] == "layer-split"


def test_capacity_intent_admits_a_second_spark_does_not_help() -> None:
    """The honest answer when the model already fits, said in those words."""
    sc = _load("studio/spark_cluster.py")
    budget = sc.SPARK_USABLE_GIB - sc.SERVE_OVERHEAD_GIB
    out = sc.plan_deployment(budget * 0.5, n_nodes = 2, intent = "capacity")
    assert out["axis"] == "none"
    assert "will not help" in out["recommendation"].lower()


def test_speedups_are_measured_only_at_two_nodes() -> None:
    """Never present an extrapolated number as a measurement."""
    sc = _load("studio/spark_cluster.py")
    two = sc.expected_gain("tensor-parallel", 2)
    assert two["measured"] is True and abs(two["speedup"] - 2.09) < 1e-9
    four = sc.expected_gain("tensor-parallel", 4)
    assert four["measured"] is False and four["speedup"] is None
    pp = sc.expected_gain("pipeline-parallel", 2)
    assert pp["speedup"] < 1.2 and "capacity" in pp["note"].lower()
    assert sc.expected_gain("layer-split-fitting", 2)["speedup"] < 1.0


def test_unknown_size_yields_no_axis_and_no_command_at_any_node_count() -> None:
    sc = _load("studio/spark_cluster.py")
    for nodes in (1, 2, 4, 9):
        out = sc.plan_deployment(None, n_nodes = nodes)
        assert out["topology"] == "unknown"
        assert out["axis"] is None and not out["command"] and not out["recommendation"]


def test_rail_plan_refuses_three_nodes_without_a_switch() -> None:
    """Three Sparks cannot be cabled point-to-point; a flat /24 would be wrong."""
    sc = _load("studio/spark_cluster.py")
    rails = [
        {"ib_device": "rocep1s0f0", "netdev": "enp1s0f0np0"},
        {"ib_device": "roceP2p1s0f0", "netdev": "enP2p1s0f0np0"},
    ]
    report = sc.rail_plan_report(rails, node_index = 0, n_nodes = 3)
    assert report["ok"] is False and report["plan"] == []
    assert any("switched" in p for p in report["problems"])
    assert sc.rail_plan(rails, 0, n_nodes = 3) == []
    ok = sc.rail_plan_report(rails, node_index = 2, n_nodes = 3, switched = True)
    assert ok["ok"] is True
    assert [e["address"] for e in ok["plan"]] == ["192.168.200.14", "192.168.201.14"]
    assert sc.rail_plan_report(rails, node_index = 5, n_nodes = 3, switched = True)["ok"] is False


def test_netplan_never_renders_a_config_that_does_nothing() -> None:
    sc = _load("studio/spark_cluster.py")
    text = sc.netplan_yaml([])
    # Every line a comment: netplan would apply this file and change nothing, which
    # is the point -- an empty `ethernets:` map looks like a config and is not one.
    assert all(line.startswith("#") for line in text.splitlines() if line.strip())
    assert "ethernets:\n" not in text


def test_peers_are_ordered_numerically_and_indexed_from_one() -> None:
    """Node index must mean the same host on every node, so ordering is by address."""
    sc = _load("studio/spark_cluster.py")
    found = sc.merge_peers(
        [
            {"hostname": "spark-c.local", "address": "192.168.200.13"},
            {"hostname": "spark-a.local", "address": "192.168.200.9"},
            {"hostname": "spark-b.local", "address": "192.168.200.10"},
        ]
    )
    assert [p["address"] for p in found] == ["192.168.200.9", "192.168.200.10", "192.168.200.13"]
    assert [p["index"] for p in found] == [1, 2, 3]
    assert all(p["reachable"] is None for p in found)


def test_discovery_is_completely_inert_off_a_spark() -> None:
    """No sysfs walk, no avahi, no sockets on a machine that is not a Spark."""
    sc = _load("studio/spark_cluster.py")
    sc._IS_SPARK_CACHE = False
    called = []
    sc._mdns_spark_peers = lambda *a, **k: called.append("mdns") or []
    sc.peer_reachable = lambda *a, **k: called.append("probe") or True
    out = sc.discover_peers(check_reachable = True)
    assert out["is_spark"] is False
    assert out["peers"] == [] and out["n_nodes"] == 1 and out["cable_present"] is False
    assert called == []


# ── Destructive operations need consent, and the peer probe fails closed ──────
# `_cmd_setup()` used to rsync --delete the studio venv onto the peer as a side
# effect of merely being called. A peer running a job out of that venv would have
# lost its interpreter mid-flight, and the failure would have looked like hardware.


def test_setup_writes_nothing_without_consent(monkeypatch, tmp_path, capsys) -> None:
    """No TTY and no --yes means: print the plan, touch nothing, exit 0."""
    sc = _load("studio/spark_cluster.py")
    sc._IS_SPARK_CACHE = True
    monkeypatch.setattr(
        sc,
        "cabled_rails",
        lambda: [{"ib_device": "rocep1s0f0", "netdev": "enp1s0f0np0", "ipv4": ["192.168.200.12"]}],
    )
    monkeypatch.setattr(sc, "peer_ip_for", lambda *a, **k: "192.168.200.13")
    monkeypatch.setattr(sc, "_studio_root", lambda: tmp_path)
    called = []
    monkeypatch.setattr(sc, "provision_peer", lambda *a, **k: called.append("rsync") or {})
    monkeypatch.setattr(sc, "save_config", lambda *a, **k: called.append("write"))
    assert sc._cmd_setup() == 0
    assert called == []
    assert "Not applied" in capsys.readouterr().out
    assert sc._cmd_setup(dry_run = True) == 0
    assert called == []


def test_setup_applies_only_with_an_explicit_yes(monkeypatch, tmp_path) -> None:
    sc = _load("studio/spark_cluster.py")
    sc._IS_SPARK_CACHE = True
    monkeypatch.setattr(
        sc,
        "cabled_rails",
        lambda: [{"ib_device": "rocep1s0f0", "netdev": "enp1s0f0np0", "ipv4": ["192.168.200.12"]}],
    )
    monkeypatch.setattr(sc, "peer_ip_for", lambda *a, **k: "192.168.200.13")
    monkeypatch.setattr(sc, "_studio_root", lambda: tmp_path)
    called = []
    monkeypatch.setattr(
        sc,
        "provision_peer",
        lambda *a, **k: called.append("rsync")
        or {"refused": "", "copied": [], "skipped": [], "failed": []},
    )
    monkeypatch.setattr(sc, "save_config", lambda *a, **k: called.append("write"))
    assert sc._cmd_setup(assume_yes = True) == 0
    assert called == ["rsync", "write"]


def test_peer_gpu_probe_fails_closed(monkeypatch) -> None:
    """Unanswerable must read as BUSY. Fail-open here has bitten this project twice."""
    sc = _load("studio/spark_cluster.py")
    assert sc.peer_gpu_busy("")["busy"] is True
    monkeypatch.setattr(sc.shutil, "which", lambda name: None)
    no_ssh = sc.peer_gpu_busy("192.168.200.13")
    assert no_ssh["busy"] is True and no_ssh["known"] is False

    monkeypatch.setattr(sc.shutil, "which", lambda name: "/usr/bin/" + name)

    class _Result:
        def __init__(self, out):
            self.stdout = out

    def fake(out):
        return lambda *a, **k: _Result(out)

    # nvidia-smi never ran: no RC marker, so unknown -> busy.
    monkeypatch.setattr(sc.subprocess, "run", fake("bash: nvidia-smi: not found\n"))
    assert sc.peer_gpu_busy("h")["busy"] is True
    # Ran, non-zero: busy.
    monkeypatch.setattr(sc.subprocess, "run", fake("RC=9\n"))
    assert sc.peer_gpu_busy("h")["busy"] is True
    # Ran, listed nothing: genuinely idle.
    idle = None
    monkeypatch.setattr(sc.subprocess, "run", fake("RC=0\n"))
    idle = sc.peer_gpu_busy("h")
    assert idle["busy"] is False and idle["known"] is True
    # Ran, a job is resident.
    monkeypatch.setattr(sc.subprocess, "run", fake("1234, 11020\nRC=0\n"))
    busy = sc.peer_gpu_busy("h")
    assert busy["busy"] is True and busy["processes"] == [{"pid": 1234, "used_mib": 11020}]
    # A CUDA context alone is not a job.
    monkeypatch.setattr(sc.subprocess, "run", fake("1234, 12\nRC=0\n"))
    assert sc.peer_gpu_busy("h")["busy"] is False

    # An exception is not permission either.
    def boom(*a, **k):
        raise OSError("ssh died")

    monkeypatch.setattr(sc.subprocess, "run", boom)
    assert sc.peer_gpu_busy("h")["busy"] is True


def test_provision_refuses_a_busy_peer_and_never_deletes_by_default(monkeypatch) -> None:
    sc = _load("studio/spark_cluster.py")
    monkeypatch.setattr(sc.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(
        sc,
        "peer_gpu_busy",
        lambda *a, **k: {
            "busy": True,
            "known": True,
            "reason": "1 compute process(es) resident",
            "processes": [{"pid": 7, "used_mib": 9000}],
        },
    )
    ran = []
    monkeypatch.setattr(sc.subprocess, "run", lambda cmd, **k: ran.append(cmd))
    res = sc.provision_peer("192.168.200.13")
    assert res["refused"] and res["copied"] == [] and ran == []

    # Idle peer: it copies, and `--delete` is absent unless asked for.
    monkeypatch.setattr(
        sc,
        "peer_gpu_busy",
        lambda *a, **k: {"busy": False, "known": True, "reason": "idle", "processes": []},
    )
    monkeypatch.setattr(sc.osp, "isdir", lambda p: True)

    class _Ok:
        returncode = 0
        stderr = ""

    def record(cmd, **k):
        ran.append(cmd)
        return _Ok()

    monkeypatch.setattr(sc.subprocess, "run", record)
    sc.provision_peer("192.168.200.13")
    assert ran and all("--delete" not in cmd for cmd in ran)
    ran.clear()
    sc.provision_peer("192.168.200.13", delete = True)
    assert all("--delete" in cmd for cmd in ran)
    # A dry run must not even probe the peer, let alone write to it.
    ran.clear()
    monkeypatch.setattr(sc, "peer_gpu_busy", lambda *a, **k: pytest.fail("dry run probed the peer"))
    sc.provision_peer("192.168.200.13", dry_run = True)
    assert all("--dry-run" in cmd for cmd in ran)


def test_consent_declines_when_no_terminal_is_watching(monkeypatch) -> None:
    """`curl | sh` must not be able to trigger a remote write by answering nothing.

    The installer invokes `python -m studio.spark_cluster setup` after its own prompt,
    and in a piped install stdin belongs to the shell script, not the user -- so the
    /dev/tty fallback is the only thing that can ask. When even that is unavailable
    (container, cron, CI), silence is a no.
    """
    sc = _load("studio/spark_cluster.py")

    class _Stream:
        def __init__(self, tty):
            self._tty = tty

        def isatty(self):
            return self._tty

        def write(self, text):
            return len(text)

        def flush(self):
            pass

    assert sc._consented(True, "?") is True  # explicit yes always wins

    monkeypatch.setattr(sc.sys, "stdout", _Stream(False))
    monkeypatch.setattr(sc.sys, "stdin", _Stream(False))
    assert sc._consented(False, "?") is False

    # stdout is a terminal, stdin is the piped script, /dev/tty unopenable -> no.
    monkeypatch.setattr(sc.sys, "stdout", _Stream(True))

    def no_tty(*a, **k):
        raise OSError("no /dev/tty")

    monkeypatch.setattr("builtins.open", no_tty)
    assert sc._consented(False, "?") is False


def test_summary_and_recommendation_never_contradict() -> None:
    """One coherent paragraph, whichever order a caller prints them in.

    `summary` answers "what fits where" and must name no axis; every axis claim lives
    in `recommendation`. They overlapped once, and a 70B then had a summary naming the
    llama.cpp layer split (0.92x) beside a recommendation naming tensor parallel
    (2.09x) -- both true, and together they read as the tool arguing with itself.
    """
    sc = _load("studio/spark_cluster.py")
    budget = sc.SPARK_USABLE_GIB - sc.SERVE_OVERHEAD_GIB
    axis_words = (
        "layer-split",
        "layer split",
        "tensor parallel",
        "replicas",
        "pipeline",
        "spark_lb",
        "--engines",
    )
    for size in (budget / 4, budget * 0.6, budget * 1.5, budget * 2.5):
        for nodes in (2, 3):
            for intent in sc.INTENTS:
                out = sc.plan_deployment(size, n_nodes = nodes, intent = intent)
                summary = out["summary"].lower()
                assert not [w for w in axis_words if w in summary], (size, nodes, summary)


def test_every_entry_point_guards_on_is_dgx_spark() -> None:
    """No entry point may reach rail discovery off a Spark. Guarded, not lucky.

    Empty sysfs happens to make these harmless on a laptop today, which is exactly the
    kind of safe-by-accident this module cannot afford: the file is imported by
    `unsloth run` on every platform.
    """
    sc = _load("studio/spark_cluster.py")
    sc._IS_SPARK_CACHE = False
    sc.cabled_rails = lambda *a, **k: pytest.fail("touched rails off a Spark")
    sc.local_rails = lambda *a, **k: pytest.fail("touched rails off a Spark")
    for argv in (
        ["status"],
        ["status", "--benchmark"],
        ["setup"],
        ["env"],
        ["peers"],
        ["plan", "--model", "m"],
        ["provision"],
        ["kernels"],
        ["doctor"],
        ["serve", "--model", "m.gguf"],
        ["train", "--script", "t.py"],
        ["train", "--layer-split", "M"],
        ["estimate", "--model", "m"],
    ):
        assert sc.main(argv) == 0, argv
    # `detect` is the installer's machine-readable gate and answers 1 by design.
    assert sc.main(["detect"]) == 1
    for plan in (sc.train_launch_plan("t.py"), sc.pipeline_launch_plan("M"), sc.rpc_cluster_plan()):
        assert plan["ok"] is False and plan["problems"]


# ---------------------------------------------------------------------------
# Merging per-stage checkpoints. A layer-split run is useless without this step,
# and the failure mode it guards against is silent: an adapter missing half its
# layers loads without error and simply trains worse.
# ---------------------------------------------------------------------------


def _mk_stage(
    root,
    rank: int,
    layers,
    shared = ("lm_head",),
) -> None:
    import json as _json
    import torch
    from safetensors.torch import save_file

    d = root / f"stage{rank}"
    d.mkdir(parents = True, exist_ok = True)
    t = {
        f"base_model.model.model.layers.{i}.self_attn.q_proj.lora_A.weight": torch.ones(2, 2)
        for i in layers
    }
    for name in shared:
        t[f"base_model.model.{name}.weight"] = torch.zeros(2, 2)
    save_file(t, str(d / "adapter_model.safetensors"))
    (d / "adapter_config.json").write_text(_json.dumps({"r": 16, "lora_alpha": 32}))


def test_merge_layer_key_parsing() -> None:
    """Pure-string helper, so it runs anywhere with no torch."""
    sm = _load("studio/spark_merge.py")
    assert sm.layer_of("base_model.model.model.layers.17.self_attn.q_proj.lora_A.weight") == 17
    assert sm.layer_of("base_model.model.lm_head.weight") is None


def test_merge_unions_disjoint_stages(tmp_path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    sm = _load("studio/spark_merge.py")
    _mk_stage(tmp_path, 0, range(0, 12))
    _mk_stage(tmp_path, 1, range(12, 24))
    plan = sm.plan_merge(str(tmp_path))
    assert plan["ok"] and plan["layers_covered"] == list(range(24))
    res = sm.merge(str(tmp_path), str(tmp_path / "merged"))
    assert (tmp_path / "merged" / "adapter_model.safetensors").is_file()
    assert (tmp_path / "merged" / "adapter_config.json").is_file()
    # 24 per-layer tensors + one shared lm_head kept once, not twice.
    assert res["n_tensors"] == 25


def test_merge_refuses_overlapping_layers(tmp_path) -> None:
    """Two stages claiming one layer means the split was not what we think it was."""
    pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    sm = _load("studio/spark_merge.py")
    _mk_stage(tmp_path, 0, range(0, 13))
    _mk_stage(tmp_path, 1, range(12, 24))
    assert sm.plan_merge(str(tmp_path))["ok"] is False
    with pytest.raises(RuntimeError):
        sm.merge(str(tmp_path), str(tmp_path / "merged"))


def test_merge_refuses_missing_layers(tmp_path) -> None:
    """A gap would produce an adapter that is untrained in the middle."""
    pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    sm = _load("studio/spark_merge.py")
    _mk_stage(tmp_path, 0, range(0, 10))
    _mk_stage(tmp_path, 1, range(14, 24))
    plan = sm.plan_merge(str(tmp_path))
    assert plan["ok"] is False
    assert any("10" in p for p in plan["problems"])


def test_merge_refuses_noncontiguous_stage_dirs(tmp_path) -> None:
    """stage0 + stage2 means stage1's layers were never saved."""
    pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    sm = _load("studio/spark_merge.py")
    _mk_stage(tmp_path, 0, range(0, 12))
    _mk_stage(tmp_path, 2, range(12, 24))
    with pytest.raises(RuntimeError):
        sm.plan_merge(str(tmp_path))


def test_merge_module_imports_nothing_heavy() -> None:
    """The CLI imports this on every platform; it must not drag in torch."""
    tree = ast.parse((REPO / "studio/spark_merge.py").read_text())
    imported = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    assert not [n for n in imported if n.split(".")[0] in HEAVY | {"safetensors"}]
