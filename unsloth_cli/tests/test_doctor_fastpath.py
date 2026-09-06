# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The fast-path parity check, driven with fake node responses.

The bug this guards against is silent by construction: a node missing `causal_conv1d`
and `flash-linear-attention` trained Qwen3.5 at 1183 tok/s against the other node's 2593
and raised nothing at all, so the only thing that can catch it is a comparison somebody
actually runs. These tests pin the three properties that make it worth running:

  1. it reports the node, the package, both versions and a command that fixes it;
  2. it never fails open -- an unreachable peer, or a runtime probe that did not report,
     answers UNKNOWN, because a check that says OK when it did not look is worse than
     no check;
  3. it is inert without a peer, and costs a non-Spark machine nothing.

Nothing here needs a Spark, a GPU, a peer, or a network.
"""

from __future__ import annotations

import pytest

from unsloth_cli.commands import doctor as D


def _node(host: str, **overrides) -> dict:
    """A probe result for a healthy node, before overrides are applied."""
    node = {
        "host": host,
        "executable": "/home/u/.unsloth/studio/unsloth_studio/bin/python",
        "pkg_torch": "2.11.0+cu130",
        "pkg_transformers": "5.5.0",
        "pkg_trl": "0.23.1",
        "pkg_peft": "0.18.1",
        "pkg_accelerate": "1.14.0",
        "pkg_bitsandbytes": "0.50.2",
        "pkg_triton": "3.6.0",
        "pkg_flash-attn": "2.8.1",
        "pkg_causal_conv1d": "1.7.0",
        "pkg_flash-linear-attention": "0.5.2",
        "pkg_fla-core": "0.5.2",
        "pkg_xformers": None,
        "gate_causal_conv1d_fn": True,
        "gate_fla_chunk_gated_delta_rule": True,
        "gate_flash_attn_func": True,
        "gate_triton": True,
        "gate_xformers_memory_efficient_attention": "no (ModuleNotFoundError)",
        "tf_is_causal_conv1d_available": True,
        "tf_is_flash_linear_attention_available": True,
        "tf_is_flash_attn_2_available": True,
        "tf_is_flash_attn_3_available": False,
    }
    node.update(overrides)
    return node


@pytest.fixture
def fake_probes(monkeypatch):
    """Serve fixed probe results in place of running anything, on either node."""

    def install(
        local,
        peer,
        local_err = None,
        peer_err = None,
    ):
        monkeypatch.setattr(D, "_run_probe_local", lambda *a, **k: (local, local_err), raising = True)
        monkeypatch.setattr(D, "_run_probe_peer", lambda *a, **k: (peer, peer_err), raising = True)

    return install


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def test_matching_nodes_report_nothing() -> None:
    """Two nodes with the same stack must produce no findings at all.

    A check that reports something on a healthy pair is a check people stop reading.
    """
    assert D.compare_fastpath(_node("a"), _node("b")) == []


def test_absent_on_both_nodes_is_not_a_finding() -> None:
    """xformers on neither node is nobody's problem: both ranks take the same path."""
    findings = D.compare_fastpath(_node("a"), _node("b"))
    assert not [f for f in findings if f["name"] == "xformers"]


def test_missing_package_names_the_lagging_node() -> None:
    """The measured fault: causal_conv1d and fla on the peer only."""
    local = _node(
        "slow",
        **{
            "pkg_causal_conv1d": None,
            "pkg_flash-linear-attention": None,
            "pkg_fla-core": None,
            "gate_causal_conv1d_fn": "no (ModuleNotFoundError)",
            "gate_fla_chunk_gated_delta_rule": "no (ModuleNotFoundError)",
            "tf_is_causal_conv1d_available": False,
            "tf_is_flash_linear_attention_available": False,
        },
    )
    findings = D.compare_fastpath(local, _node("fast"))
    by_name = {(f["kind"], f["name"]): f for f in findings}
    assert by_name[("package", "causal_conv1d")]["lagging"] == "local"
    assert by_name[("package", "causal_conv1d")]["peer"] == "1.7.0"
    assert by_name[("package", "causal_conv1d")]["local"] is None
    assert by_name[("package", "fla-core")]["lagging"] == "local"
    # The runtime gate must diverge too, not just the package list.
    assert by_name[("gate", "causal_conv1d_fn")]["lagging"] == "local"
    assert by_name[("gate", "is_causal_conv1d_available")]["lagging"] == "local"


def test_version_mismatch_is_a_finding_with_no_lagging_side() -> None:
    """Both nodes have it, at different versions: neither is 'behind', both are wrong."""
    findings = D.compare_fastpath(_node("a", pkg_transformers = "4.57.6"), _node("b"))
    f = next(f for f in findings if f["name"] == "transformers")
    assert (f["local"], f["peer"], f["lagging"]) == ("4.57.6", "5.5.0", None)


def test_transformers_4x_missing_gates_is_not_a_finding() -> None:
    """transformers 4.x does not define the newer gates. Absent on both is not a split."""
    old = {k: v for k, v in _node("a").items() if not k.startswith("tf_")}
    other = {k: v for k, v in _node("b").items() if not k.startswith("tf_")}
    assert D.compare_fastpath(old, other) == []


def test_both_slow_note_only_for_gates_a_pair_can_fix() -> None:
    """FA3 off on both is normal here; causal_conv1d off on both is worth saying."""
    a = _node("a", tf_is_causal_conv1d_available = False)
    b = _node("b", tf_is_causal_conv1d_available = False)
    assert D.fastpath_both_slow(a, b) == ["is_causal_conv1d_available"]
    assert D.fastpath_both_slow(_node("a"), _node("b")) == []


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def test_report_prints_a_pasteable_fix_for_the_lagging_node(fake_probes, capsys) -> None:
    local = _node("slow", pkg_causal_conv1d = None, gate_causal_conv1d_fn = "no (ImportError)")
    fake_probes(local, _node("fast"))
    rc = D.check_fastpath("10.0.0.2")
    out = capsys.readouterr().out
    assert rc == 1
    assert "causal_conv1d" in out
    assert "1.7.0" in out and "<absent>" in out
    assert "fix, on this Spark" in out
    assert 'python3 -m pip install "causal_conv1d==1.7.0"' in out
    # The peer is fine, so nothing may tell the operator to touch it.
    assert "fix, on the peer" not in out


def test_report_fixes_the_peer_when_the_peer_is_the_one_behind(fake_probes, capsys) -> None:
    fake_probes(_node("fast"), _node("slow", pkg_causal_conv1d = None))
    rc = D.check_fastpath("10.0.0.2")
    out = capsys.readouterr().out
    assert rc == 1
    assert "fix, on the peer" in out
    assert "ssh " in out and "10.0.0.2" in out
    assert "causal_conv1d==1.7.0" in out


def test_matching_nodes_report_ok(fake_probes, capsys) -> None:
    fake_probes(_node("a"), _node("b"))
    rc = D.check_fastpath("10.0.0.2")
    out = capsys.readouterr().out
    assert rc == 0
    assert "OK" in out
    assert "DIFFERENCE" not in out


def test_unreachable_peer_is_unknown_not_ok(fake_probes, capsys) -> None:
    """The failure mode this whole module exists to avoid: a check that fails open."""
    fake_probes(_node("a"), None, peer_err = "ssh: connect to host 10.0.0.2 port 22: timed out")
    rc = D.check_fastpath("10.0.0.2")
    out = capsys.readouterr().out
    assert rc == 2
    assert "UNKNOWN" in out
    assert "OK --" not in out
    assert "timed out" in out


def test_local_probe_failure_is_unknown(fake_probes, capsys) -> None:
    fake_probes(None, _node("b"), local_err = "TimeoutExpired: 300s")
    rc = D.check_fastpath("10.0.0.2")
    out = capsys.readouterr().out
    assert rc == 2
    assert "UNKNOWN" in out and "this Spark" in out


def test_runtime_probe_that_did_not_report_is_unknown(fake_probes, capsys) -> None:
    """Package versions still compare, but an unrun gate probe must not read as OK."""
    local = {k: v for k, v in _node("a").items() if not k.startswith(("gate_", "tf_"))}
    local["gates_error"] = "Segmentation fault (core dumped)"
    fake_probes(local, _node("b"))
    rc = D.check_fastpath("10.0.0.2")
    out = capsys.readouterr().out
    assert rc == 2
    assert "UNKNOWN" in out
    assert "Segmentation fault" in out
    # The missing gates must not be dressed up as divergences: nothing was measured.
    assert "runtime fast path" not in out


def test_a_real_divergence_outranks_an_unknown_probe(fake_probes, capsys) -> None:
    """A version split is still a version split when the gate probe crashed."""
    local = {k: v for k, v in _node("a").items() if not k.startswith(("gate_", "tf_"))}
    local["gates_error"] = "Segmentation fault (core dumped)"
    local["pkg_causal_conv1d"] = None
    fake_probes(local, _node("b"))
    rc = D.check_fastpath("10.0.0.2")
    out = capsys.readouterr().out
    assert rc == 1
    assert "fix, on this Spark" in out
    assert "UNKNOWN" in out


def test_node_whose_fast_path_is_installed_but_does_not_import(fake_probes, capsys) -> None:
    """An installed package whose extension will not load is exactly as slow as none.

    The versions match, so a package-list comparison alone reports a healthy pair here.
    """
    local = _node("a", gate_causal_conv1d_fn = "no (ImportError)")
    fake_probes(local, _node("b"))
    rc = D.check_fastpath("10.0.0.2")
    out = capsys.readouterr().out
    assert rc == 1
    assert "causal_conv1d_fn (runtime fast path)" in out
    assert "no (ImportError)" in out


# ---------------------------------------------------------------------------
# The probe itself
# ---------------------------------------------------------------------------


def test_probe_source_compiles_and_prints_its_marker() -> None:
    for runtime in (True, False):
        src = D.fastpath_probe_source(runtime = runtime)
        compile(src, "<fastpath probe>", "exec")
        assert "UNSLOTH_FASTPATH " in src
    compile(D.FASTPATH_RUNTIME_SOURCE, "<fastpath runtime probe>", "exec")


def test_probe_runs_here_and_reports_this_interpreter() -> None:
    """It must survive a machine that has none of these packages, and still answer."""
    import json
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, "-c", D.fastpath_probe_source(runtime = False)],
        capture_output = True,
        text = True,
        timeout = 120,
    )
    assert proc.returncode == 0, proc.stderr
    line = next(l for l in proc.stdout.splitlines() if l.startswith("UNSLOTH_FASTPATH "))
    payload = json.loads(line[len("UNSLOTH_FASTPATH ") :])
    for name, _aliases in D.FASTPATH_PACKAGES:
        assert "pkg_" + name in payload


def test_extract_still_defaults_to_the_parity_marker() -> None:
    """The parity check shares these helpers; adding a marker must not have moved it."""
    assert D._extract('UNSLOTH_PARITY {"a": 1}', "") == ({"a": 1}, None)
    assert D._extract('UNSLOTH_FASTPATH {"a": 1}', "")[0] is None
    assert D._extract('UNSLOTH_FASTPATH {"a": 1}', "", "UNSLOTH_FASTPATH")[0] == {"a": 1}
