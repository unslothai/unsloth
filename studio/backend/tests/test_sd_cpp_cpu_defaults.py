# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""sd.cpp ``--threads``: ``UNSLOTH_CPU_THREADS`` reaches it, and a cgroup CPU quota caps the default."""

from __future__ import annotations

import pytest

from core.inference import sd_cpp_backend as bk


def _cgroups(
    tmp_path,
    quotas,
    own = "/user.slice/app.scope",
):
    """A cgroup v2 tree where ``quotas`` maps a cgroup path to its cpu.max line, plus /proc/self/cgroup."""
    root = tmp_path / "cg"
    for rel, line in quotas.items():
        d = root / rel.lstrip("/")
        d.mkdir(parents = True, exist_ok = True)
        (d / "cpu.max").write_text(line + "\n")
    (root / own.lstrip("/")).mkdir(parents = True, exist_ok = True)
    proc = tmp_path / "cgroup"
    proc.write_text(f"0::{own}\n")
    return str(root), str(proc)


@pytest.mark.parametrize(
    "quotas, expected",
    [
        ({}, None),
        ({"/user.slice/app.scope": "max 100000"}, None),
        ({"/user.slice/app.scope": "400000 100000"}, 4),
        ({"/user.slice/app.scope": "250000 100000"}, 3),
        ({"/user.slice": "200000 100000", "/user.slice/app.scope": "800000 100000"}, 2),
        ({"/": "600000 100000"}, 6),
        ({"/user.slice/app.scope": "garbage"}, None),
    ],
)
def test_the_tightest_quota_over_the_own_cgroup_and_its_parents(tmp_path, quotas, expected):
    root, proc = _cgroups(tmp_path, quotas)
    assert bk._cgroup_cpu_limit(root, proc) == expected


def test_a_container_with_its_own_cgroup_namespace_reads_the_root(tmp_path):
    root, proc = _cgroups(tmp_path, {"/": "400000 100000"}, own = "/")
    assert bk._cgroup_cpu_limit(root, proc) == 4


def test_no_cgroup_v2_entry_reads_the_root_only(tmp_path):
    root, _ = _cgroups(tmp_path, {"/": "300000 100000"})
    missing = str(tmp_path / "missing")
    assert bk._cgroup_cpu_limit(root, missing) == 3


def _host(
    monkeypatch,
    *,
    cpus = 192,
    limit = None,
    platform = "linux",
):
    monkeypatch.delenv("UNSLOTH_CPU_THREADS", raising = False)
    monkeypatch.setattr(bk.sys, "platform", platform)
    monkeypatch.setattr(bk.os, "cpu_count", lambda: cpus)
    monkeypatch.setattr(bk, "_cgroup_cpu_limit", lambda: limit)


def test_without_a_quota_the_default_is_unchanged(monkeypatch):
    _host(monkeypatch)
    assert bk._default_threads() == 96
    _host(monkeypatch, cpus = None)
    assert bk._default_threads() == 8


def test_a_quota_caps_the_default(monkeypatch):
    _host(monkeypatch, limit = 4)
    assert bk._default_threads() == 4
    _host(monkeypatch, cpus = 8, limit = 16)
    assert bk._default_threads() == 4


def test_other_platforms_never_read_cgroups(monkeypatch):
    _host(monkeypatch, cpus = 10, platform = "darwin")
    monkeypatch.setattr(bk, "_apple_performance_cores", lambda: None)
    monkeypatch.setattr(bk, "_cgroup_cpu_limit", lambda: pytest.fail("no cgroups off Linux"))
    assert bk._default_threads() == 5


@pytest.mark.parametrize("cpus, p_cores", [(3, 3), (10, 8), (16, 12), (10, 4)])
def test_macos_uses_the_performance_core_count_not_half_the_cores(monkeypatch, cpus, p_cores):
    _host(monkeypatch, cpus = cpus, platform = "darwin")
    monkeypatch.setattr(bk, "_apple_performance_cores", lambda: p_cores)
    assert bk._default_threads() == p_cores
    monkeypatch.setenv("UNSLOTH_CPU_THREADS", "2")
    assert bk._default_threads() == 2


def test_apple_performance_cores_is_none_off_macos(monkeypatch):
    monkeypatch.setattr(bk.sys, "platform", "linux")
    assert bk._apple_performance_cores() is None


@pytest.mark.parametrize("value, expected", [("12", 12), ("0", 4), ("x", 4), ("", 4), ("-3", 4)])
def test_unsloth_cpu_threads_wins_over_the_default_and_the_quota(monkeypatch, value, expected):
    _host(monkeypatch, limit = 4)
    monkeypatch.setenv("UNSLOTH_CPU_THREADS", value)
    assert bk._default_threads() == expected


_TOPOLOGIES = {
    # M1 Pro: 8 Performance + 2 Efficiency.
    "m1pro": {"hw.nperflevels": 2, "hw.perflevel0.name": "Performance", "hw.perflevel0.physicalcpu": 8,
              "hw.perflevel1.name": "Efficiency", "hw.perflevel1.physicalcpu": 2, "hw.physicalcpu": 10},
    # M5 Max (ggml-org/llama.cpp#29782): 6 Super + 12 Performance, no efficiency tier.
    "m5max": {"hw.nperflevels": 2, "hw.perflevel0.name": "Super", "hw.perflevel0.physicalcpu": 6,
              "hw.perflevel1.name": "Performance", "hw.perflevel1.physicalcpu": 12, "hw.physicalcpu": 18},
    # Virtual M1 runner: one tier.
    "vm": {"hw.nperflevels": 1, "hw.perflevel0.name": "Performance", "hw.perflevel0.physicalcpu": 3,
           "hw.physicalcpu": 3},
    # No perflevel sysctls (older macOS / Intel).
    "intel": {"hw.physicalcpu": 6},
    # Levels without names: only level 0 is known to be fast.
    "unnamed": {"hw.nperflevels": 2, "hw.perflevel0.physicalcpu": 4, "hw.perflevel1.physicalcpu": 6,
                "hw.physicalcpu": 10},
}


@pytest.mark.parametrize("topology, expected", [("m1pro", 8), ("m5max", 18), ("vm", 3), ("intel", 6), ("unnamed", 4)])
def test_apple_cores_count_every_tier_but_efficiency(monkeypatch, topology, expected):
    table = _TOPOLOGIES[topology]
    monkeypatch.setattr(bk.sys, "platform", "darwin")
    monkeypatch.setattr(bk, "_sysctl", lambda name, as_str = False: table.get(name))
    assert bk._apple_performance_cores() == expected
