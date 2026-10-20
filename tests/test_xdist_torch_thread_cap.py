# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""tests/conftest.py gives each xdist worker one share of the CPUs for torch.

Without it every worker's torch pool is as wide as the machine, and CPU training tests
thrash each other on CI's 4 vCPU runners (test_decision_model's Clef training went past
the 330 s timeout). The end-to-end check only means something under xdist; the rules it
relies on (explicit settings win, cgroup quotas count) are checked directly below.
"""

import importlib.util
import os
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "xdist_cpu_share_under_test", Path(__file__).resolve().parent / "_shared" / "xdist_cpu_share.py"
)
share_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(share_mod)


def test_each_xdist_worker_gets_one_cpu_share():
    workers = int(os.environ.get("PYTEST_XDIST_WORKER_COUNT") or 0)
    if workers <= 1:
        pytest.skip("not running under pytest-xdist")
    recorded = os.environ.get(share_mod.MARKER)
    assert recorded, "conftest did not apply a per-worker CPU share"
    if recorded == "explicit":
        pytest.skip("the caller set a thread count; conftest left it alone")
    import torch

    assert recorded == str(max(1, share_mod.usable_cpus() // workers))
    for var in share_mod.THREAD_VARS:
        assert os.environ[var] == recorded, var
    assert torch.get_num_threads() == int(recorded)


def test_share_divides_the_cpus_between_workers():
    env = {"PYTEST_XDIST_WORKER_COUNT": "4"}
    assert share_mod.apply(env, cpus = 16) == "4"
    assert all(env[var] == "4" for var in share_mod.THREAD_VARS)
    env = {"PYTEST_XDIST_WORKER_COUNT": "8"}
    assert share_mod.apply(env, cpus = 4) == "1"


def test_serial_runs_are_left_alone():
    for env in ({}, {"PYTEST_XDIST_WORKER_COUNT": "1"}):
        assert share_mod.apply(env, cpus = 16) is None
        assert not any(var in env for var in share_mod.THREAD_VARS)


@pytest.mark.parametrize("var", share_mod.THREAD_VARS)
def test_one_explicit_thread_setting_keeps_all_of_them_unset(var):
    # MKL_NUM_THREADS outranks OMP_NUM_THREADS in torch, so filling the others in beside an
    # explicit one would silently override it.
    env = {"PYTEST_XDIST_WORKER_COUNT": "4", var: "3"}
    assert share_mod.apply(env, cpus = 16) == "explicit"
    assert env[var] == "3"
    assert [v for v in share_mod.THREAD_VARS if v in env] == [var]


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_text(text, encoding = "utf-8")


def test_cgroup_v2_quota_takes_the_tightest_ancestor(tmp_path):
    root = tmp_path / "cg"
    _write(root / "cgroup.controllers", "cpu memory\n")
    _write(root / "cpu.max", "max 100000\n")
    _write(root / "job" / "cpu.max", "400000 100000\n")
    _write(root / "job" / "leaf" / "cpu.max", "max 100000\n")
    proc = tmp_path / "proc_cgroup"
    _write(proc, "0::/job/leaf\n")
    assert share_mod.cgroup_cpu_quota(str(root), str(proc)) == 4.0


def test_cgroup_v2_without_a_quota_is_unlimited(tmp_path):
    root = tmp_path / "cg"
    _write(root / "cgroup.controllers", "cpu\n")
    _write(root / "cpu.max", "max 100000\n")
    proc = tmp_path / "proc_cgroup"
    _write(proc, "0::/\n")
    assert share_mod.cgroup_cpu_quota(str(root), str(proc)) is None


def test_cgroup_v1_quota_is_read(tmp_path):
    root = tmp_path / "cg"
    _write(root / "cpu,cpuacct" / "cpu.cfs_quota_us", "-1\n")
    _write(root / "cpu,cpuacct" / "cpu.cfs_period_us", "100000\n")
    _write(root / "cpu,cpuacct" / "docker" / "abc" / "cpu.cfs_quota_us", "200000\n")
    _write(root / "cpu,cpuacct" / "docker" / "abc" / "cpu.cfs_period_us", "100000\n")
    proc = tmp_path / "proc_cgroup"
    _write(proc, "4:cpu,cpuacct:/docker/abc\n3:memory:/docker/abc\n")
    assert share_mod.cgroup_cpu_quota(str(root), str(proc)) == 2.0


def test_usable_cpus_is_capped_by_the_quota(monkeypatch):
    monkeypatch.setattr(share_mod, "cgroup_cpu_quota", lambda: 2.5)
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(64)), raising = False)
    assert share_mod.usable_cpus() == 2
    monkeypatch.setattr(share_mod, "cgroup_cpu_quota", lambda: None)
    assert share_mod.usable_cpus() == 64
