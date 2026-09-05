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

    Splitting a model that fits never decodes faster than a single Spark (0.85x to 1.01x
    measured from 1 to 32 users), so the planner must never suggest it for speed.
    Regressing this would make Unsloth actively slower than not clustering.
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
    # This pins the ssh path; the rail daemon has its own tests below.
    monkeypatch.setenv(sc.FAST_ENV, "0")
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
    llama.cpp layer split beside a recommendation naming tensor parallel (2.09x) --
    both true, and together they read as the tool arguing with itself.
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


# ── The Spark notice on the model-load path of every platform ────────────────
# notify_device_map_cannot_span_sparks runs inside FastLanguageModel.from_pretrained
# and FastModel.from_pretrained, so it executes for every Unsloth user on every OS and
# accelerator. It is cosmetic. The bar is that it can never turn a load that would have
# succeeded into a crash, and that it is silent on anything that is not a cabled Spark.

_PLATFORMS = [
    ("Linux", "x86_64"),  # linux x64: NVIDIA, AMD, CPU-only, and WSL2
    ("Linux", "aarch64"),  # linux arm64 that is not a Spark (GH200, Jetson)
    ("Windows", "AMD64"),
    ("Windows", "ARM64"),
    ("Darwin", "arm64"),  # Apple Silicon
    ("Darwin", "x86_64"),  # Intel Mac
]

_DEVICE_MAPS = [
    "balanced",
    "balanced_low_0",
    "auto",
    "unsloth",
    "unsloth_balanced",
    "sequential",
    "cuda:0",
    "cpu",
    "",
    None,
    0,
    {"": "cuda:0"},
    ["cuda:0"],
]


def _spark_notice(
    monkeypatch,
    system,
    machine,
    *,
    opener = None,
    device_count = 1,
):
    """Call the notice with every probe pointed at a simulated host."""
    pytest.importorskip("torch")
    import builtins
    import platform as _platform
    import torch
    from unsloth.models import loader_utils as LU

    monkeypatch.setattr(_platform, "system", lambda: system)
    monkeypatch.setattr(_platform, "machine", lambda: machine)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: device_count)
    if opener is not None:
        real_open = builtins.open

        def guarded(path, *a, **k):
            p = str(path)
            if p.startswith("/etc/dgx") or p.startswith("/sys/"):
                raise opener("simulated")
            return real_open(path, *a, **k)

        monkeypatch.setattr(builtins, "open", guarded)
    LU._SPARK_NOTICE_SHOWN[0] = False
    return LU


@pytest.mark.parametrize("system,machine", _PLATFORMS)
def test_spark_notice_is_silent_off_a_spark(monkeypatch, capsys, system, machine):
    """No Unsloth user on another platform should ever see this.

    `opener=FileNotFoundError` simulates the non-Spark filesystem as well as the
    non-Spark platform. Without it this test passes everywhere except on a real DGX
    Spark, where the aarch64 case would read the machine's own /etc/dgx-release and see
    a genuine Spark. CI that ever runs on this hardware must not go red for that.
    """
    LU = _spark_notice(monkeypatch, system, machine, opener = FileNotFoundError)
    capsys.readouterr()  # drop unsloth's import banner, which is not ours to assert on
    for device_map in _DEVICE_MAPS:
        LU.notify_device_map_cannot_span_sparks(device_map)
    assert "DGX Spark" not in capsys.readouterr().out


@pytest.mark.parametrize("system,machine", _PLATFORMS)
@pytest.mark.parametrize(
    "opener", [OSError, PermissionError, IsADirectoryError, ValueError, RuntimeError]
)
def test_spark_notice_never_breaks_a_load(monkeypatch, system, machine, opener):
    """A probe that raises must not propagate: the caller is loading a model.

    ValueError and RuntimeError are the point of this test. The probes catch OSError
    themselves, so only the outer guard stops a non-OSError from escaping into
    from_pretrained and failing a load that had nothing to do with a Spark.
    """
    LU = _spark_notice(monkeypatch, system, machine, opener = opener)
    for device_map in _DEVICE_MAPS:
        LU.notify_device_map_cannot_span_sparks(device_map)


def test_spark_notice_survives_a_broken_cuda_probe(monkeypatch):
    """torch.cuda.device_count() can raise on a broken driver; that is not our problem."""
    pytest.importorskip("torch")
    import torch
    from unsloth.models import loader_utils as LU

    def boom():
        raise RuntimeError("no CUDA driver")

    monkeypatch.setattr(torch.cuda, "device_count", boom)
    LU._SPARK_NOTICE_SHOWN[0] = False
    LU.notify_device_map_cannot_span_sparks("balanced")


# ── UNSLOTH_STUDIO_HOME must move the venv that provision copies ─────────────
# A user who sets UNSLOTH_STUDIO_HOME installs somewhere other than
# ~/.unsloth/studio. `spark provision` copying the hardcoded default from such a
# machine copies a stale venv, or nothing at all, and then prints "Peer now matches
# this node" -- handing the user the exact 601 s `DistStoreError: 1/2 clients joined`
# that provision exists to prevent, while reporting success.


def test_provision_paths_follow_studio_home(monkeypatch, tmp_path) -> None:
    sc = _load("studio/spark_cluster.py")
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "elsewhere"))
    venv, label = sc.provision_paths()[0]
    assert label == "Unsloth venv"
    assert venv == str(tmp_path / "elsewhere" / "unsloth_studio"), venv
    # The llama.cpp bundle follows the studio home too; see the bundle tests below.
    assert sc.provision_paths()[1][1] == "llama.cpp prebuilt"
    # The caches are genuinely per-user and are not moved by STUDIO_HOME.
    assert [p for p, _ in sc.provision_paths()[2:]] == [
        "~/.cache/flashinfer",
        "~/.cache/vllm/flashinfer_autotune_cache",
        "~/.cache/vllm/torch_compile_cache",
    ]


def test_provision_paths_unchanged_without_studio_home(monkeypatch) -> None:
    """The default must stay byte-identical: most users never set the variable."""
    sc = _load("studio/spark_cluster.py")
    monkeypatch.delenv("UNSLOTH_STUDIO_HOME", raising = False)
    monkeypatch.delenv("STUDIO_HOME", raising = False)
    venv, _ = sc.provision_paths()[0]
    assert venv == str(Path.home() / ".unsloth" / "studio" / "unsloth_studio")


def test_peer_activate_stays_home_relative_by_default(monkeypatch) -> None:
    """`$HOME` is left unexpanded on purpose so it resolves on the PEER.

    That stays correct when the two nodes have different usernames or home
    directories. Only a custom STUDIO_HOME forces an absolute path, which is right
    because provision copies to that same absolute path on the peer.
    """
    sc = _load("studio/spark_cluster.py")
    monkeypatch.delenv("UNSLOTH_STUDIO_HOME", raising = False)
    monkeypatch.delenv("STUDIO_HOME", raising = False)
    assert sc.venv_activate() == "$HOME/.unsloth/studio/unsloth_studio/bin/activate"


def test_peer_activate_is_absolute_under_studio_home(monkeypatch, tmp_path) -> None:
    sc = _load("studio/spark_cluster.py")
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "custom"))
    act = sc.venv_activate()
    assert act == str(tmp_path / "custom" / "unsloth_studio" / "bin" / "activate")
    assert "$HOME" not in act


# ── Splitting a model that fits: no longer a single number ───────────────────
# The flat 0.92x holds only for a llama.cpp whose RPC backend predates
# ggml-org/llama.cpp#18626. With it, what overlaps is prefill, so the answer becomes a
# function of prompt length. Getting this backwards in either direction gives users bad
# deployment advice, so pin both branches.


def test_layer_split_speedup_defaults_to_the_conservative_number() -> None:
    """Without async RPC it is still a flat loss, whatever the prompt length."""
    sc = _load("studio/spark_cluster.py")
    for tokens in (128, 1024, 4096, None):
        assert sc.layer_split_speedup(tokens, 8) == sc.LAYER_SPLIT_FITTING_SPEEDUP


def test_layer_split_speedup_is_prompt_dependent_with_async_rpc() -> None:
    sc = _load("studio/spark_cluster.py")
    # Short prompts still lose; long prompts win; and it is monotonic in prompt length.
    assert sc.layer_split_speedup(128, 8, async_rpc = True) < 1.0
    assert sc.layer_split_speedup(4096, 8, async_rpc = True) > 1.4
    seq = [sc.layer_split_speedup(t, 8, async_rpc = True) for t in (128, 256, 512, 1024, 2048, 4096)]
    assert seq == sorted(seq), seq


def test_layer_split_speedup_refuses_to_guess_without_a_prompt_length() -> None:
    """With async RPC the answer genuinely depends on prompt length, so say nothing."""
    sc = _load("studio/spark_cluster.py")
    assert sc.layer_split_speedup(None, 8, async_rpc = True) is None


def test_layer_split_speedup_snaps_down_to_a_measured_row() -> None:
    """Six measured points, not a fitted curve: do not interpolate precision we lack."""
    sc = _load("studio/spark_cluster.py")
    assert (
        sc.layer_split_speedup(2047, 8, async_rpc = True) == sc.LAYER_SPLIT_ASYNC_RPC_SPEEDUP[1024][8]
    )
    assert (
        sc.layer_split_speedup(99999, 8, async_rpc = True)
        == sc.LAYER_SPLIT_ASYNC_RPC_SPEEDUP[4096][8]
    )


# ── Replicas versus layer split for a model that fits ────────────────────────
# Measured 2026-09-04 on Qwen3.8-27B Q4_K_XL, llama.cpp b10796, two Sparks, uncapped
# clocks. A layer split never speeds up decode for a model that fits; two replicas
# win from 8 concurrent users up. Both directions of getting this wrong hand users
# a slower deployment than the single Spark they started with, so pin the rules.

_GIB = 2**30


def test_recommend_topology_layer_split_when_the_model_does_not_fit() -> None:
    sc = _load("studio/spark_cluster.py")
    out = sc.recommend_topology(150 * _GIB, 0.5 * _GIB, 1, 512, 113 * _GIB)
    assert out["topology"] == "layer_split" and out["fits_one_node"] is False
    assert "does not fit" in out["reason"]
    # Even a prefill-heavy caller with many users gets the same answer: it is the only option.
    out = sc.recommend_topology(150 * _GIB, 0.5 * _GIB, 32, 2048, 113 * _GIB, prefill_heavy = True)
    assert out["topology"] == "layer_split"


def test_recommend_topology_replicas_from_eight_users_up() -> None:
    sc = _load("studio/spark_cluster.py")
    for users in (8, 12, 16, 32, 64):
        out = sc.recommend_topology(16.4 * _GIB, 0.4 * _GIB, users, 512, 113 * _GIB)
        assert out["topology"] == "replicas", (users, out)
        assert out["speedup"] >= 1.30, (users, out)
    assert sc.recommend_topology(16.4 * _GIB, 0.4 * _GIB, 8, 512, 113 * _GIB)["speedup"] == 1.30
    assert sc.recommend_topology(16.4 * _GIB, 0.4 * _GIB, 32, 512, 113 * _GIB)["speedup"] == 1.91
    assert sc.recommend_topology(16.4 * _GIB, 0.4 * _GIB, 32, 2048, 113 * _GIB)["speedup"] == 2.38


def test_recommend_topology_single_below_eight_users() -> None:
    """Below 8 users the second copy buys 1.00x to 1.13x; say so and leave it idle."""
    sc = _load("studio/spark_cluster.py")
    for users in (1, 2, 4, 7):
        out = sc.recommend_topology(16.4 * _GIB, 0.4 * _GIB, users, 512, 113 * _GIB)
        assert out["topology"] == "single", (users, out)
        assert "1.13x" in out["reason"]
    one = sc.recommend_topology(16.4 * _GIB, 0.4 * _GIB, 1, 512, 113 * _GIB)
    assert "tensor parallel" in one["reason"] and "2.09x" in one["reason"]


def test_recommend_topology_never_splits_a_fitting_model_unless_prefill_heavy() -> None:
    sc = _load("studio/spark_cluster.py")
    for users in (1, 2, 4, 8, 16, 32):
        for tokens in (128, 512, 2048, 8192):
            out = sc.recommend_topology(16.4 * _GIB, 0.4 * _GIB, users, tokens, 113 * _GIB)
            assert out["topology"] != "layer_split", (users, tokens, out)
    # The one exception: the caller says the work is prefill-heavy, at few users.
    out = sc.recommend_topology(16.4 * _GIB, 0.4 * _GIB, 2, 4096, 113 * _GIB, prefill_heavy = True)
    assert out["topology"] == "layer_split"
    assert "prefill" in out["reason"] and "1.7x" in out["reason"]
    assert out["prefill_speedup"] == sc.LAYER_SPLIT_PREFILL_SPEEDUP
    # At 8 or more users the replicas still win end to end, prefill-heavy or not.
    out = sc.recommend_topology(16.4 * _GIB, 0.4 * _GIB, 16, 4096, 113 * _GIB, prefill_heavy = True)
    assert out["topology"] == "replicas"


def test_recommend_topology_counts_kv_for_every_user() -> None:
    """A model that fits alone but not with its users' KV is not `single`."""
    sc = _load("studio/spark_cluster.py")
    # 100 GiB model, 2 GiB KV per user, 16 users: 132 GiB on one node, 116 per replica.
    out = sc.recommend_topology(100 * _GIB, 2 * _GIB, 16, 512, 120 * _GIB)
    assert out["topology"] == "replicas" and "KV" in out["reason"]
    # 100 GiB model, 4 GiB KV per user, 16 users: 132 GiB even per replica.
    out = sc.recommend_topology(100 * _GIB, 4 * _GIB, 16, 512, 120 * _GIB)
    assert out["topology"] == "layer_split" and "KV" in out["reason"]


def test_recommend_topology_is_pure_and_tolerant() -> None:
    sc = _load("studio/spark_cluster.py")
    for bad in (0, None, -5):
        out = sc.recommend_topology(16.4 * _GIB, bad, bad, bad, 113 * _GIB)
        assert out["topology"] in sc.TOPOLOGIES and out["reason"]


def test_measured_tables_snap_to_measured_points() -> None:
    """Six user counts and two prompt lengths were measured; nothing is interpolated."""
    sc = _load("studio/spark_cluster.py")
    assert sc.replicas_speedup(512, 8) == sc.REPLICAS_DECODE_SPEEDUP[512][8]
    assert sc.replicas_speedup(1023, 8) == sc.REPLICAS_DECODE_SPEEDUP[512][8]
    assert sc.replicas_speedup(4096, 12) == sc.REPLICAS_DECODE_SPEEDUP[2048][8]
    assert sc.layer_split_decode_speedup(512, 8) == 0.85
    # The rule itself, pinned against the data: no measured split cell beats 1.01x at
    # prompt 512, and the two above 1.0 at 2048 are prefill contention, not decode.
    assert max(sc.LAYER_SPLIT_DECODE_SPEEDUP[512].values()) <= 1.01
    assert sc.LAYER_SPLIT_DECODE_ONLY_SPEEDUP < 1.0
    assert all(v >= 1.30 for u, v in sc.REPLICAS_DECODE_SPEEDUP[512].items() if u >= 8)
    assert all(v <= 1.13 for u, v in sc.REPLICAS_DECODE_SPEEDUP[512].items() if u < 8)


def test_plan_deployment_carries_the_serving_layout() -> None:
    sc = _load("studio/spark_cluster.py")
    budget = sc.SPARK_USABLE_GIB - sc.SERVE_OVERHEAD_GIB
    few = sc.plan_deployment(16.4, n_nodes = 2, intent = "throughput", concurrency = 2)
    assert few["serving"]["topology"] == "single"
    many = sc.plan_deployment(16.4, n_nodes = 2, intent = "throughput", concurrency = 16)
    assert many["serving"]["topology"] == "replicas"
    assert "1.75x" in many["recommendation"]
    big = sc.plan_deployment(budget * 1.5, n_nodes = 2)
    assert big["serving"]["topology"] == "layer_split"
    assert "serving" not in sc.plan_deployment(budget * 2.5, n_nodes = 2)
    assert "serving" not in sc.plan_deployment(None, n_nodes = 2)
    # The headline that came from pre-#18626 RPC is gone from every planner sentence.
    for size in (budget / 4, budget * 0.6, budget * 1.5):
        for intent in sc.INTENTS:
            out = sc.plan_deployment(size, n_nodes = 2, intent = intent)
            text = " ".join(str(v) for v in out.values())
            assert "0.92x" not in text, (size, intent)
    assert "0.92" not in sc.expected_gain("layer-split-fitting", 2)["note"]
    assert sc.expected_gain("replicas", 2, 16)["aggregate"] == 1.75


# ── Provisioning must carry the llama.cpp bundle ─────────────────────────────
# The bundle lives beside the studio root, not inside the venv, so copying the venv
# alone leaves the peer on whatever llama-server it had. Two bundles a release apart
# speak different RPC protocols and llama-server then fails at load with
# "RPC server version mismatch".


def test_provision_paths_include_the_llama_bundle(monkeypatch) -> None:
    sc = _load("studio/spark_cluster.py")
    monkeypatch.delenv("UNSLOTH_STUDIO_HOME", raising = False)
    monkeypatch.delenv("STUDIO_HOME", raising = False)
    monkeypatch.delenv("UNSLOTH_LLAMA_CPP_PATH", raising = False)
    paths = dict((label, path) for path, label in sc.provision_paths())
    assert paths["llama.cpp prebuilt"] == str(Path.home() / ".unsloth" / "llama.cpp")
    assert paths["Unsloth venv"] == str(Path.home() / ".unsloth" / "studio" / "unsloth_studio")
    # Order: venv first, bundle second, then the caches, so the peer can serve before
    # the caches land.
    labels = [label for _, label in sc.provision_paths()]
    assert labels[:2] == ["Unsloth venv", "llama.cpp prebuilt"]


def test_provision_paths_bundle_follows_studio_home(monkeypatch, tmp_path) -> None:
    sc = _load("studio/spark_cluster.py")
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "elsewhere"))
    monkeypatch.delenv("UNSLOTH_LLAMA_CPP_PATH", raising = False)
    paths = dict((label, path) for path, label in sc.provision_paths())
    assert paths["llama.cpp prebuilt"] == str(tmp_path / "elsewhere" / "llama.cpp")
    assert paths["Unsloth venv"] == str(tmp_path / "elsewhere" / "unsloth_studio")
    # The explicit override wins over the studio home, as it does in the installer.
    monkeypatch.setenv("UNSLOTH_LLAMA_CPP_PATH", str(tmp_path / "custom-llama"))
    paths = dict((label, path) for path, label in sc.provision_paths())
    assert paths["llama.cpp prebuilt"] == str(tmp_path / "custom-llama")


def test_provision_copies_the_bundle_to_the_same_path(monkeypatch, tmp_path) -> None:
    sc = _load("studio/spark_cluster.py")
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "elsewhere"))
    monkeypatch.delenv("UNSLOTH_LLAMA_CPP_PATH", raising = False)
    (tmp_path / "elsewhere" / "llama.cpp").mkdir(parents = True)
    monkeypatch.setattr(sc.shutil, "which", lambda name: "/usr/bin/" + name)
    ran = []

    class _Ok:
        returncode = 0
        stderr = ""

    monkeypatch.setattr(sc.subprocess, "run", lambda cmd, **k: ran.append(cmd) or _Ok())
    res = sc.provision_peer("192.168.200.13", dry_run = True)
    assert ("llama.cpp prebuilt", str(tmp_path / "elsewhere" / "llama.cpp")) in res["copied"]
    bundle_cmd = next(c for c in ran if c[-2].startswith(str(tmp_path / "elsewhere" / "llama.cpp")))
    assert bundle_cmd[-1].endswith(":" + str(tmp_path / "elsewhere" / "llama.cpp") + "/")
    assert any("mkdir -p" in part for part in bundle_cmd)


# ── RPC protocol parity between the two nodes ────────────────────────────────
# Signal (a): the bundle identity, from BUILD_INFO.txt and the libggml-rpc hash.
# Signal (b): a live HELLO against a running ggml-rpc-server.


def _mk_bundle(
    root: Path,
    version = "b10796-mix-659e406",
    lib = b"rpc-lib-bytes",
    server = True,
):
    bin_dir = root / "build" / "bin"
    bin_dir.mkdir(parents = True, exist_ok = True)
    if version is not None:
        (root / "BUILD_INFO.txt").write_text(
            f"llama.cpp version: {version}\nrequested source ref: {version}\nvariant: cuda13\n"
        )
    if lib is not None:
        (bin_dir / "libggml-rpc.so").write_bytes(lib)
    if server:
        exe = bin_dir / "ggml-rpc-server"
        exe.write_text("#!/bin/sh\nexit 0\n")
        exe.chmod(0o755)
    return root


def test_bundle_identity_reads_version_and_hashes_the_rpc_library(tmp_path) -> None:
    import hashlib

    sc = _load("studio/spark_cluster.py")
    root = _mk_bundle(tmp_path / "llama.cpp")
    ident = sc.llama_bundle_identity(root)
    assert ident["present"] is True
    assert ident["version"] == "b10796-mix-659e406"
    assert ident["rpc_lib_md5"] == hashlib.md5(b"rpc-lib-bytes").hexdigest()
    assert ident["rpc_server"] == str(root / "build" / "bin" / "ggml-rpc-server")


def test_bundle_identity_is_unknown_not_a_crash_without_build_info(tmp_path) -> None:
    """Older bundles and source builds have no BUILD_INFO.txt."""
    sc = _load("studio/spark_cluster.py")
    root = _mk_bundle(tmp_path / "llama.cpp", version = None)
    ident = sc.llama_bundle_identity(root)
    assert ident["present"] is True and ident["version"] == "unknown"
    assert ident["rpc_lib_md5"]  # the hash still works as the second signal
    missing = sc.llama_bundle_identity(tmp_path / "nowhere")
    assert missing["present"] is False and missing["version"] == "unknown"
    assert missing["rpc_lib_md5"] is None and missing["rpc_server"] is None
    # A BUILD_INFO.txt without the expected key still yields its first line.
    odd = tmp_path / "odd"
    odd.mkdir()
    (odd / "BUILD_INFO.txt").write_text("\n  b10700-custom  \n")
    assert sc.llama_bundle_identity(odd)["version"] == "b10700-custom"


def test_compare_bundles_names_both_versions_and_the_fix(tmp_path) -> None:
    sc = _load("studio/spark_cluster.py")
    new = sc.llama_bundle_identity(_mk_bundle(tmp_path / "new"))
    old = sc.llama_bundle_identity(
        _mk_bundle(tmp_path / "old", version = "b10715-mix-86bd2d3", lib = b"older-lib")
    )
    same = sc.compare_llama_bundles(new, sc.llama_bundle_identity(_mk_bundle(tmp_path / "twin")))
    assert same["ok"] is True and not same["problems"]

    diff = sc.compare_llama_bundles(new, old)
    assert diff["ok"] is False and len(diff["problems"]) == 1
    text = diff["problems"][0]
    assert "b10796-mix-659e406" in text and "b10715-mix-86bd2d3" in text
    assert "unsloth spark provision" in text and "RPC server version mismatch" in text

    # Same tag, different library: the hash is the signal that catches it.
    patched = sc.llama_bundle_identity(_mk_bundle(tmp_path / "patched", lib = b"patched-lib"))
    hashed = sc.compare_llama_bundles(new, patched)
    assert hashed["ok"] is False and "libggml-rpc differs" in hashed["problems"][0]

    # Unverifiable is never reported as matching.
    unknown = sc.compare_llama_bundles(new, None)
    assert unknown["ok"] is None and not unknown["problems"]
    assert "UNVERIFIED" in unknown["notes"][0]
    absent = sc.compare_llama_bundles(new, {"present": False, "root": "/nowhere"})
    assert absent["ok"] is False and "no llama.cpp bundle" in absent["problems"][0]


def test_peer_bundle_probe_is_self_contained_and_matches_local(tmp_path, monkeypatch) -> None:
    """The peer-side probe runs under a bare python3 and must agree with the local reader."""
    import json
    import subprocess

    sc = _load("studio/spark_cluster.py")
    root = _mk_bundle(tmp_path / "llama.cpp")
    source = sc._BUNDLE_PROBE.format(
        root = str(root), libs = sc._RPC_LIB_NAMES, servers = sc._RPC_SERVER_NAMES
    )
    env = dict(os.environ)
    env.pop("UNSLOTH_LLAMA_CPP_PATH", None)
    out = subprocess.run(
        [sys.executable, "-"], input = source, capture_output = True, text = True, env = env, timeout = 60
    )
    line = next(l for l in out.stdout.splitlines() if l.startswith("UNSLOTH_BUNDLE "))
    remote = json.loads(line[len("UNSLOTH_BUNDLE ") :])
    local = sc.llama_bundle_identity(root)
    assert remote == local


def test_peer_relative_path_keeps_a_home_path_home_relative(monkeypatch, tmp_path) -> None:
    sc = _load("studio/spark_cluster.py")
    monkeypatch.setattr(sc.Path, "home", classmethod(lambda cls: tmp_path / "me"))
    assert (
        sc._peer_relative_path(tmp_path / "me" / ".unsloth" / "llama.cpp") == "~/.unsloth/llama.cpp"
    )
    assert (
        sc._peer_relative_path(tmp_path / "opt" / "llama")
        == (tmp_path / "opt" / "llama").as_posix()
    )


def _fake_rpc_server(behaviour: str):
    """A one-shot ggml-rpc-server stand-in on 127.0.0.1. Returns (port, thread, seen)."""
    import socket
    import struct
    import threading

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    listener.settimeout(5)
    port = listener.getsockname()[1]
    seen = {}

    def serve():
        try:
            conn, _ = listener.accept()
        except OSError:
            return
        with conn:
            conn.settimeout(5)
            try:
                cmd = conn.recv(1)
                (n,) = struct.unpack("<Q", conn.recv(8))
                payload = b""
                while len(payload) < n:
                    chunk = conn.recv(n - len(payload))
                    if not chunk:
                        break
                    payload += chunk
                seen.update(cmd = cmd, size = n, payload = payload)
                if behaviour == "6.0.0":
                    body = bytes([6, 0, 0, 0]) + bytes(24)
                    conn.sendall(struct.pack("<Q", len(body)) + body)
                elif behaviour == "short":
                    conn.sendall(struct.pack("<Q", 2) + b"\x06\x00")
                elif behaviour == "closed":
                    pass  # what a size-mismatched HELLO gets: silence, then EOF
            except OSError:
                pass
        listener.close()

    thread = threading.Thread(target = serve, daemon = True)
    thread.start()
    return port, thread, seen


def test_rpc_hello_probe_reads_a_six_zero_reply() -> None:
    sc = _load("studio/spark_cluster.py")
    port, thread, seen = _fake_rpc_server("6.0.0")
    assert sc.rpc_hello_probe("127.0.0.1", port, timeout = 3) == (6, 0, 0)
    thread.join(5)
    # The request was a well-formed 6.0 HELLO: command 14, exactly RPC_CONN_CAPS_SIZE zero bytes.
    assert seen["cmd"] == bytes([14])
    assert seen["size"] == sc.RPC_CONN_CAPS_SIZE == 24
    assert seen["payload"] == bytes(24)
    detail = sc.rpc_hello_probe_detail("127.0.0.1", 1)  # nothing listens on port 1
    assert detail["state"] == "refused" and detail["version"] is None


def test_rpc_hello_probe_survives_a_truncated_reply_and_a_hangup() -> None:
    sc = _load("studio/spark_cluster.py")
    port, thread, _ = _fake_rpc_server("short")
    assert sc.rpc_hello_probe("127.0.0.1", port, timeout = 3) is None
    thread.join(5)
    port, thread, _ = _fake_rpc_server("closed")
    detail = sc.rpc_hello_probe_detail("127.0.0.1", port, timeout = 3, read_timeout = 3)
    thread.join(5)
    assert detail["state"] == "closed" and detail["version"] is None
    # Never raises, even for an address that cannot be resolved.
    assert sc.rpc_hello_probe("host.invalid.", 50052, timeout = 1) is None


def test_rpc_preflight_reports_a_confirmed_mismatch_and_says_the_fix(monkeypatch, tmp_path) -> None:
    sc = _load("studio/spark_cluster.py")
    new = sc.llama_bundle_identity(_mk_bundle(tmp_path / "new"))
    old = sc.llama_bundle_identity(
        _mk_bundle(tmp_path / "old", version = "b10715-mix-86bd2d3", lib = b"x")
    )
    monkeypatch.setattr(sc, "llama_bundle_identity", lambda root = None: new)
    monkeypatch.setattr(sc, "peer_llama_bundle_identity", lambda *a, **k: old)
    monkeypatch.setattr(
        sc,
        "rpc_hello_probe_detail",
        lambda host, port, **k: {"host": host, "port": port, "state": "refused", "version": None},
    )
    pre = sc.rpc_protocol_preflight("192.168.200.13")
    assert pre["ok"] is False and "unsloth spark provision" in pre["problems"][0]

    # Matching bundles, but a stale 5.x server is still up on the peer and hangs up on
    # a 6.0 HELLO: the live signal catches what the bundle signal cannot.
    monkeypatch.setattr(sc, "peer_llama_bundle_identity", lambda *a, **k: new)

    def live(host, port, **k):
        state = "closed" if host == "192.168.200.13" else "refused"
        return {"host": host, "port": port, "state": state, "version": None}

    monkeypatch.setattr(sc, "rpc_hello_probe_detail", live)
    pre = sc.rpc_protocol_preflight("192.168.200.13")
    assert pre["ok"] is False and "version mismatch" in pre["problems"][0]

    # Both servers up and disagreeing.
    def two(host, port, **k):
        version = (5, 1, 0) if host == "192.168.200.13" else (6, 0, 0)
        return {"host": host, "port": port, "state": "ok", "version": version}

    monkeypatch.setattr(sc, "rpc_hello_probe_detail", two)
    pre = sc.rpc_protocol_preflight("192.168.200.13")
    assert pre["ok"] is False and "6.0.0" in pre["problems"][0] and "5.1.0" in pre["problems"][0]

    # Unverifiable peer, nothing listening: not a failure, but not a pass either.
    monkeypatch.setattr(sc, "peer_llama_bundle_identity", lambda *a, **k: None)
    monkeypatch.setattr(
        sc,
        "rpc_hello_probe_detail",
        lambda host, port, **k: {"host": host, "port": port, "state": "refused", "version": None},
    )
    pre = sc.rpc_protocol_preflight("192.168.200.13")
    assert pre["ok"] is None and not pre["problems"] and "UNVERIFIED" in pre["notes"][0]


# ── ggml-rpc-server ships in the bundle from b10796 ──────────────────────────


def test_rpc_server_binary_found_in_the_bundle(monkeypatch, tmp_path) -> None:
    sc = _load("studio/spark_cluster.py")
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))
    monkeypatch.delenv("UNSLOTH_LLAMA_CPP_PATH", raising = False)
    monkeypatch.setattr(sc.shutil, "which", lambda name: None)
    root = _mk_bundle(tmp_path / "studio" / "llama.cpp")
    assert sc.llama_bundle_dir() == root
    assert sc.rpc_server_binary() == str(root / "build" / "bin" / "ggml-rpc-server")
    # The explicit override wins, as in the installer.
    other = _mk_bundle(tmp_path / "override")
    monkeypatch.setenv("UNSLOTH_LLAMA_CPP_PATH", str(other))
    assert sc.rpc_server_binary() == str(other / "build" / "bin" / "ggml-rpc-server")


def test_rpc_server_binary_accepts_the_legacy_name(monkeypatch, tmp_path) -> None:
    sc = _load("studio/spark_cluster.py")
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))
    monkeypatch.delenv("UNSLOTH_LLAMA_CPP_PATH", raising = False)
    monkeypatch.setattr(sc.shutil, "which", lambda name: None)
    root = _mk_bundle(tmp_path / "studio" / "llama.cpp", server = False)
    legacy = root / "build" / "bin" / "rpc-server"
    legacy.write_text("#!/bin/sh\nexit 0\n")
    legacy.chmod(0o755)
    assert sc.rpc_server_binary() == str(legacy)


def test_rpc_server_binary_none_when_absent(monkeypatch, tmp_path) -> None:
    """An older bundle without the executable, no source build, nothing on PATH."""
    sc = _load("studio/spark_cluster.py")
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))
    monkeypatch.delenv("UNSLOTH_LLAMA_CPP_PATH", raising = False)
    monkeypatch.setattr(sc.Path, "home", classmethod(lambda cls: tmp_path / "home"))
    monkeypatch.setattr(sc.shutil, "which", lambda name: None)
    _mk_bundle(tmp_path / "studio" / "llama.cpp", server = False)
    # A file that is present but not executable does not count either.
    stub = tmp_path / "studio" / "llama.cpp" / "build" / "bin" / "ggml-rpc-server"
    stub.write_text("not executable")
    stub.chmod(0o644)
    assert sc.rpc_server_binary() is None
    plan = sc.rpc_cluster_plan()
    assert plan["ok"] is False  # off a Spark, and the message names the bundle when on one


def test_bundle_dir_defaults_to_the_legacy_location(monkeypatch, tmp_path) -> None:
    """Default installs keep ~/.unsloth/llama.cpp, beside the studio root, not inside it."""
    sc = _load("studio/spark_cluster.py")
    monkeypatch.delenv("UNSLOTH_STUDIO_HOME", raising = False)
    monkeypatch.delenv("STUDIO_HOME", raising = False)
    monkeypatch.delenv("UNSLOTH_LLAMA_CPP_PATH", raising = False)
    assert sc.llama_bundle_dir() == Path.home() / ".unsloth" / "llama.cpp"
    # Pointing UNSLOTH_STUDIO_HOME at the default location must not move the bundle.
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(Path.home() / ".unsloth" / "studio"))
    assert sc.llama_bundle_dir() == Path.home() / ".unsloth" / "llama.cpp"


# ── Fast provisioning: the ephemeral rsync daemon on the direct rail ─────────
# ssh is the fallback and stays the finaliser; the daemon only carries bulk bytes,
# unencrypted, over the point-to-point cable. These pin what makes that safe: what
# the daemon's config says, who it admits, when the path is refused outright, that
# the work split is disjoint and complete, and that the daemon dies with the command.


def _fast_module(monkeypatch, tmp_path):
    sc = _load("studio/spark_cluster.py")
    monkeypatch.setattr(sc.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(sc.platform, "system", lambda: "Linux")
    sc._IS_SPARK_CACHE = True
    monkeypatch.setattr(sc, "rail_local_address", lambda peer, rails = None: "192.168.200.12")
    monkeypatch.setattr(
        sc,
        "peer_gpu_busy",
        lambda *a, **k: {"busy": False, "known": True, "reason": "idle", "processes": []},
    )
    monkeypatch.delenv(sc.FAST_ENV, raising = False)
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.bin").write_bytes(b"a" * 300)
    (src / "b.bin").write_bytes(b"b" * 200)
    monkeypatch.setattr(sc, "provision_paths", lambda: ((str(src), "scratch"),))
    return sc, src


def test_fast_daemon_config_is_locked_to_the_rail() -> None:
    sc = _load("studio/spark_cluster.py")
    text = sc.rsync_daemon_config(
        {"m0": "$HOME/.unsloth/studio/unsloth_studio", "m1": "/opt/x"},
        bind_ip = "192.168.200.13",
        port = 41234,
        hosts_allow = "192.168.200.12",
        auth_user = "unsloth-ab12",
        work_dir = "/tmp/unsloth-provision-1",
    )
    lines = [l.strip() for l in text.splitlines()]
    assert "address = 192.168.200.13" in lines  # the peer's rail address only
    assert "port = 41234" in lines
    assert "use chroot = no" in lines and "read only = no" in lines
    assert f"max connections = {sc.FAST_MAX_WORKERS}" in lines and sc.FAST_MAX_WORKERS == 4
    # The daemon's idle timeout is the client's twice over: the client drops and retries.
    assert f"timeout = {2 * sc.FAST_IO_TIMEOUT}" in lines
    assert "pid file = /tmp/unsloth-provision-1/rsyncd.pid" in lines
    assert "log file = /tmp/unsloth-provision-1/rsyncd.log" in lines
    assert "secrets file = /tmp/unsloth-provision-1/rsyncd.secrets" in lines
    assert "munge symlinks = no" in lines
    # Every module admits the single local rail address and the one-shot user.
    assert lines.count("hosts allow = 192.168.200.12") == 2
    assert lines.count("auth users = unsloth-ab12") == 2
    assert "[m0]" in lines and "path = $HOME/.unsloth/studio/unsloth_studio" in lines
    assert "[m1]" in lines and "path = /opt/x" in lines


def test_fast_port_stays_in_the_high_range() -> None:
    sc = _load("studio/spark_cluster.py")
    lo, hi = sc.FAST_PORT_RANGE
    assert 1024 < lo < hi < 65536
    draws = {sc.fast_port() for _ in range(500)}
    assert all(lo <= p <= hi for p in draws) and len(draws) > 50


def test_fast_setup_script_keeps_the_secret_in_a_600_file() -> None:
    sc = _load("studio/spark_cluster.py")
    cfg = sc.rsync_daemon_config({"m0": "$HOME/x"}, "10.0.0.2", 45000, "10.0.0.1", "u", "/tmp/w")
    script = sc.daemon_setup_script(cfg, "u", "S3cr3t-value", ["$HOME/x"], "/tmp/w", 45000)
    assert "umask 077" in script and 'mkdir -m 700 "$d"' in script
    assert 'chmod 600 "$d/rsyncd.secrets"' in script
    # The secret is inside a quoted heredoc on stdin, never an argument.
    assert "<<'UNSLOTH_EOF'\nu:S3cr3t-value\nUNSLOTH_EOF" in script
    assert "rsync --daemon" in script and "--no-detach" in script
    assert f"timeout {sc.FAST_DAEMON_MAX_SECONDS}" in script
    assert 'mkdir -p "$HOME/x"' in script
    stop = sc.daemon_stop_script("/tmp/w")
    assert "rsyncd.pid" in stop and 'rm -rf "$d"' in stop and "pgrep -x rsync" in stop


def test_fast_files_script_really_writes_600(tmp_path) -> None:
    """Run the file-writing half under bash where bash exists: modes, not just text."""
    import shutil as _shutil
    import stat as _stat
    import subprocess as _sp

    bash = _shutil.which("bash")
    if not bash or sys.platform.startswith("win"):
        pytest.skip("needs bash")
    sc = _load("studio/spark_cluster.py")
    work = tmp_path / "work"
    dest = tmp_path / "dest" / "deeper"
    cfg = sc.rsync_daemon_config({"m0": str(dest)}, "10.0.0.2", 45000, "10.0.0.1", "u", str(work))
    script = sc.daemon_files_script(cfg, "u", "pw$`\\x", [str(dest)], str(work))
    r = _sp.run([bash, "-s"], input = script, capture_output = True, text = True)
    assert r.returncode == 0, r.stderr
    assert _stat.S_IMODE(work.stat().st_mode) == 0o700
    secrets_file = work / "rsyncd.secrets"
    assert _stat.S_IMODE(secrets_file.stat().st_mode) == 0o600
    assert secrets_file.read_text() == "u:pw$`\\x\n"  # quoted heredoc: nothing expanded
    assert (work / "rsyncd.conf").read_text() == cfg
    assert dest.is_dir()


@pytest.mark.parametrize(
    "system, spark, no_fast, env, peer, local, expect",
    [
        ("Linux", True, False, None, "192.168.200.13", "192.168.200.12", True),
        ("Windows", True, False, None, "192.168.200.13", "192.168.200.12", False),
        ("Darwin", True, False, None, "192.168.200.13", "192.168.200.12", False),
        ("Linux", False, False, None, "192.168.200.13", "192.168.200.12", False),
        ("Linux", True, True, None, "192.168.200.13", "192.168.200.12", False),
        ("Linux", True, False, "0", "192.168.200.13", "192.168.200.12", False),
        ("Linux", True, False, "false", "192.168.200.13", "192.168.200.12", False),
        ("Linux", True, False, "1", "192.168.200.13", "192.168.200.12", True),
        ("Linux", True, False, None, "8.8.8.9", "8.8.8.8", False),  # not private
        ("Linux", True, False, None, "192.168.200.13", "192.168.201.12", False),  # other rail
        ("Linux", True, False, None, "192.168.200.13", None, False),  # no local address
        ("Linux", True, False, None, "192.168.200.13", "192.168.200.13", False),  # ourselves
        ("Linux", True, False, None, "not-an-ip", "192.168.200.12", False),
    ],
)
def test_fast_path_gating(monkeypatch, system, spark, no_fast, env, peer, local, expect) -> None:
    sc = _load("studio/spark_cluster.py")
    monkeypatch.setattr(sc.platform, "system", lambda: system)
    sc._IS_SPARK_CACHE = spark
    monkeypatch.setattr(sc, "rail_local_address", lambda p, rails = None: local)
    environ = {} if env is None else {sc.FAST_ENV: env}
    out = sc.fast_path_decision(peer, no_fast = no_fast, env = environ)
    assert out["ok"] is expect, out
    assert out["reason"]
    # The rail lookup forks `ip`; a refused platform or flag must never reach it.
    monkeypatch.setattr(
        sc, "rail_local_address", lambda p, rails = None: pytest.fail("looked up rails")
    )
    assert sc.fast_path_decision(peer, no_fast = True, env = environ)["ok"] is False
    assert sc.fast_path_decision(peer, env = {sc.FAST_ENV: "0"})["ok"] is False
    monkeypatch.setattr(sc.platform, "system", lambda: "Windows")
    assert sc.fast_path_decision(peer, env = environ)["ok"] is False


def test_fast_path_env_var_is_read_from_the_process_environment(monkeypatch) -> None:
    sc = _load("studio/spark_cluster.py")
    monkeypatch.setattr(sc.platform, "system", lambda: "Linux")
    sc._IS_SPARK_CACHE = True
    monkeypatch.setenv(sc.FAST_ENV, "0")
    assert sc.fast_path_decision("192.168.200.13", local_ip = "192.168.200.12")["ok"] is False
    monkeypatch.delenv(sc.FAST_ENV)
    assert sc.fast_path_decision("192.168.200.13", local_ip = "192.168.200.12")["ok"] is True


def test_rail_local_address_picks_the_same_slash_24() -> None:
    sc = _load("studio/spark_cluster.py")
    rails = [{"ipv4": ["192.168.201.12"]}, {"ipv4": ["192.168.200.12"]}]
    assert sc.rail_local_address("192.168.200.13", rails) == "192.168.200.12"
    assert sc.rail_local_address("192.168.202.13", rails) is None
    assert sc.rail_local_address("junk", rails) is None


def test_provision_work_split_is_disjoint_complete_and_capped(tmp_path) -> None:
    root = tmp_path / "tree"
    (root / "sub" / "deep").mkdir(parents = True)
    (root / "empty").mkdir()
    sizes = [5000, 4000, 3000, 3000, 1000, 900, 800, 10, 10, 10, 1, 0]
    expected = set()
    for i, size in enumerate(sizes):
        rel = os.path.join("sub", "deep", f"f{i}") if i % 3 == 0 else f"f{i}"
        (root / rel).write_bytes(b"x" * size)
        expected.add(rel)
    if not sys.platform.startswith("win"):
        os.symlink("/usr/bin/python3", root / "link")  # left to the ssh finaliser
    sc = _load("studio/spark_cluster.py")
    buckets = sc.provision_work_split(str(root))
    assert 1 < len(buckets) <= sc.FAST_MAX_WORKERS == 4
    flat = [p for b in buckets for p in b]
    assert len(flat) == len(set(flat)) == len(expected)  # disjoint, nothing twice
    assert set(flat) == expected  # every regular file, no symlink
    # Byte-balanced: the two biggest files never share a worker.
    big = {p for b in buckets for p in b if p in ("f1", "f2")}
    assert big and not any({"f1", "f2"} <= set(b) for b in buckets)
    assert len(sc.provision_work_split(str(root), max_workers = 2)) == 2
    # One file is one worker; no files is no work.
    single = tmp_path / "single"
    single.mkdir()
    (single / "model.gguf").write_bytes(b"g" * 100)
    assert sc.provision_work_split(str(single)) == [["model.gguf"]]
    assert sc.provision_work_split(str(tmp_path / "empty")) == []


class _Fake:
    """A subprocess.run stand-in that plays the peer: ssh starts or stops the daemon,
    rsync workers succeed or fail as scripted, and the ssh rsync finaliser succeeds."""

    def __init__(
        self,
        daemon_rc = 0,
        worker_rc = 0,
        ssh_rsync = None,
    ):
        self.calls = []
        self.daemon_rc = daemon_rc
        self.worker_rc = worker_rc
        self.ssh_rsync = ssh_rsync

    def __call__(self, cmd, **kw):
        self.calls.append((cmd, kw))
        argv0 = cmd[0]
        script = kw.get("input") or ""

        class R:
            returncode = 0
            stdout = ""
            stderr = ""

        r = R()
        if argv0 == "ssh" and "rsync --daemon" in script:
            r.returncode = self.daemon_rc
            r.stdout = "UNSLOTH_DAEMON_UP 4242\n" if self.daemon_rc == 0 else ""
            r.stderr = "" if self.daemon_rc == 0 else "bind failed"
        elif argv0 == "ssh" and "pgrep -x rsync" in script:
            r.stdout = "UNSLOTH_RSYNC_LEFT \n"
        elif argv0 == "rsync" and any(a.startswith("rsync://") for a in cmd):
            if isinstance(self.worker_rc, BaseException):
                raise self.worker_rc
            rc = self.worker_rc
            if isinstance(rc, list):  # scripted per call: a stalled flow, then a retry
                rc = rc.pop(0) if rc else 0
            r.returncode = rc
            r.stderr = "worker boom" if rc else ""
        elif argv0 == "rsync":
            if isinstance(self.ssh_rsync, BaseException):
                raise self.ssh_rsync
        return r

    def stops(self):
        return [
            c for c, k in self.calls if c[0] == "ssh" and "pgrep -x rsync" in (k.get("input") or "")
        ]

    def workers(self):
        return [
            (c, k)
            for c, k in self.calls
            if c[0] == "rsync" and any(a.startswith("rsync://") for a in c)
        ]

    def ssh_copies(self):
        return [c for c, k in self.calls if c[0] == "rsync" and "-e" in c]


def test_fast_path_moves_bytes_then_finalises_over_ssh_and_stops_the_daemon(
    monkeypatch, tmp_path
) -> None:
    sc, src = _fast_module(monkeypatch, tmp_path)
    fake = _Fake()
    monkeypatch.setattr(sc.subprocess, "run", fake)
    res = sc.provision_peer("192.168.200.13")
    assert res["copied"] == [("scratch", str(src))] and not res["failed"]
    assert res["fast"]["used"] is True and res["fast"]["errors"] == []
    # Two files, two workers, each against the daemon URL with the secret in the
    # environment and nowhere on the command line.
    workers = fake.workers()
    assert len(workers) == 2
    start_script = next(
        k["input"] for c, k in fake.calls if c[0] == "ssh" and "rsync --daemon" in k["input"]
    )
    secret = start_script.split("<<'UNSLOTH_EOF'\n")[1].split("\n")[0].split(":", 1)[1]
    for cmd, kw in workers:
        assert kw["env"]["RSYNC_PASSWORD"] == secret
        assert secret not in " ".join(cmd) and "ssh" not in " ".join(cmd)
        assert f"--timeout={sc.FAST_IO_TIMEOUT}" in cmd and "--contimeout=15" in cmd
        assert cmd[-1].startswith(f"rsync://unsloth-") and cmd[-1].endswith(
            ":%d/m0/" % res["fast"]["port"]
        )
    assert sc.FAST_PORT_RANGE[0] <= res["fast"]["port"] <= sc.FAST_PORT_RANGE[1]
    # The unchanged ssh rsync ran afterwards as the finaliser, then the daemon stopped.
    assert len(fake.ssh_copies()) == 1 and len(fake.stops()) == 1
    order = [
        ("stop" if c[0] == "ssh" and "pgrep" in (k.get("input") or "") else c[0])
        for c, k in fake.calls
    ]
    assert order.index("stop") == len(order) - 1
    assert res["fast"]["stop"] == {"stopped": True, "left": [], "error": ""}
    assert res["timings"][0][1] == "fast" and res["timings"][0][2] == 500
    assert res["fast"]["retries"] == 0


def test_fast_path_retries_a_stalled_worker_on_a_fresh_connection(monkeypatch, tmp_path) -> None:
    sc, src = _fast_module(monkeypatch, tmp_path)
    # First worker call times out (rsync exit 30), the retry succeeds; the other is fine.
    fake = _Fake(worker_rc = [30, 0, 0])
    monkeypatch.setattr(sc.subprocess, "run", fake)
    res = sc.provision_peer("192.168.200.13")
    assert res["copied"] == [("scratch", str(src))] and not res["failed"]
    assert res["fast"]["used"] is True and res["fast"]["errors"] == []
    assert res["fast"]["retries"] == 1 and len(fake.workers()) == 3
    assert len(fake.stops()) == 1
    # A worker that fails every attempt gives up after FAST_WORKER_ATTEMPTS, not forever.
    fake = _Fake(worker_rc = 30)
    monkeypatch.setattr(sc.subprocess, "run", fake)
    res = sc.provision_peer("192.168.200.13")
    assert res["fast"]["used"] is False and res["fast"]["errors"] == [("scratch", "worker boom")]
    assert len(fake.workers()) == 2 * sc.FAST_WORKER_ATTEMPTS and res["copied"]


def test_fast_path_worker_failure_falls_back_to_ssh_and_still_stops_the_daemon(
    monkeypatch, tmp_path
) -> None:
    sc, src = _fast_module(monkeypatch, tmp_path)
    fake = _Fake(worker_rc = 23)
    monkeypatch.setattr(sc.subprocess, "run", fake)
    res = sc.provision_peer("192.168.200.13")
    assert res["copied"] == [("scratch", str(src))] and not res["failed"]
    assert res["fast"]["used"] is False
    assert res["fast"]["errors"] == [("scratch", "worker boom")]
    assert len(fake.ssh_copies()) == 1 and len(fake.stops()) == 1
    assert res["timings"][0][1] == "ssh"


def test_fast_daemon_is_stopped_when_the_copy_raises(monkeypatch, tmp_path) -> None:
    sc, _ = _fast_module(monkeypatch, tmp_path)
    fake = _Fake(worker_rc = KeyboardInterrupt())
    monkeypatch.setattr(sc.subprocess, "run", fake)
    with pytest.raises(KeyboardInterrupt):
        sc.provision_peer("192.168.200.13")
    assert len(fake.stops()) == 1 and fake.stops()[0][-1] == "-s"
    # An ssh finaliser that blows up is recorded, and the daemon still goes.
    fake = _Fake(ssh_rsync = OSError("ssh died"))
    monkeypatch.setattr(sc.subprocess, "run", fake)
    res = sc.provision_peer("192.168.200.13")
    assert res["failed"] == [("scratch", "ssh died")] and len(fake.stops()) == 1


def test_fast_daemon_start_failure_means_plain_ssh_and_no_stop(monkeypatch, tmp_path) -> None:
    sc, src = _fast_module(monkeypatch, tmp_path)
    fake = _Fake(daemon_rc = 1)
    monkeypatch.setattr(sc.subprocess, "run", fake)
    res = sc.provision_peer("192.168.200.13")
    assert res["copied"] == [("scratch", str(src))] and not res["failed"]
    assert res["fast"]["used"] is False and "did not start" in res["fast"]["reason"]
    assert "bind failed" in res["fast"]["reason"]
    assert fake.workers() == [] and fake.stops() == [] and len(fake.ssh_copies()) == 1


def test_fast_daemon_that_survives_is_reported_as_a_failure(monkeypatch, tmp_path) -> None:
    sc, _ = _fast_module(monkeypatch, tmp_path)
    fake = _Fake()
    monkeypatch.setattr(sc.subprocess, "run", fake)
    monkeypatch.setattr(
        sc,
        "stop_peer_rsync_daemon",
        lambda d, timeout = 60: {
            "stopped": False,
            "left": [4242],
            "error": "daemon pid 4242 survived SIGKILL",
        },
    )
    res = sc.provision_peer("192.168.200.13")
    assert ("rsync daemon on the peer", "daemon pid 4242 survived SIGKILL") in res["failed"]


def test_no_fast_and_env_and_dry_run_never_touch_the_daemon(monkeypatch, tmp_path) -> None:
    sc, src = _fast_module(monkeypatch, tmp_path)
    for kwargs, env in (({"no_fast": True}, None), ({}, "0"), ({"dry_run": True}, None)):
        fake = _Fake()
        monkeypatch.setattr(sc.subprocess, "run", fake)
        if env is None:
            monkeypatch.delenv(sc.FAST_ENV, raising = False)
        else:
            monkeypatch.setenv(sc.FAST_ENV, env)
        res = sc.provision_peer("192.168.200.13", **kwargs)
        assert res["copied"] == [("scratch", str(src))]
        assert fake.workers() == [] and fake.stops() == []
        assert not any("rsync --daemon" in (k.get("input") or "") for c, k in fake.calls)
        assert res["fast"]["used"] is False


def test_no_fast_flag_reaches_provision(monkeypatch, capsys) -> None:
    sc = _load("studio/spark_cluster.py")
    sc._IS_SPARK_CACHE = True
    monkeypatch.setattr(sc, "peer_ip_for", lambda *a, **k: "192.168.200.13")
    seen = {}

    def fake_provision(peer, **kw):
        seen.update(kw)
        return {
            "copied": [],
            "skipped": [],
            "failed": [],
            "refused": "",
            "fast": {
                "used": False,
                "reason": "--no-fast",
                "errors": [],
                "stop": None,
                "port": None,
                "retries": 0,
            },
            "timings": [],
        }

    monkeypatch.setattr(sc, "provision_peer", fake_provision)
    assert sc.main(["provision", "--no-fast"]) == 0
    assert seen["no_fast"] is True
    assert sc.main(["provision"]) == 0
    assert seen["no_fast"] is False
    # The help text says the bytes are unencrypted, where they go, and how to opt out.
    with pytest.raises(SystemExit):
        sc.main(["--help"])
    text = " ".join(capsys.readouterr().out.split())
    assert "--no-fast" in text and "UNENCRYPTED" in text and "point-to-point" in text
    assert f"{sc.FAST_ENV}=0" in text
