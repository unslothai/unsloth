# SPDX-License-Identifier: AGPL-3.0-only
"""Quota parsing fixtures for the CPU-share helper, without importing torch."""

import importlib.util
import os
from pathlib import Path

import pytest


ROOT = Path(os.environ.get("UNSLOTH_REPO_ROOT", Path(__file__).resolve().parents[1]))
SPEC = importlib.util.spec_from_file_location(
    "xdist_cpu_share_fixture_cases", ROOT / "tests/_shared/xdist_cpu_share.py"
)
assert SPEC is not None and SPEC.loader is not None
SHARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SHARE)


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.mark.parametrize(
    ("leaf", "expected"),
    [
        pytest.param("0 100000\n", 4.0, id="zero-quota"),
        pytest.param("400000 0\n", 4.0, id="zero-period"),
        pytest.param("invalid 100000\n", 4.0, id="invalid-quota"),
        pytest.param(None, 4.0, id="missing-leaf-quota"),
        pytest.param("150000 100000\n", 1.5, id="tighter-fractional-leaf"),
    ],
)
def test_v2_bad_leaf_does_not_hide_parent_limit(tmp_path, leaf, expected):
    root = tmp_path / "cgroup"
    write(root / "cgroup.controllers", "cpu memory\n")
    write(root / "cpu.max", "max 100000\n")
    write(root / "job/cpu.max", "400000 100000\n")
    if leaf is not None:
        write(root / "job/leaf/cpu.max", leaf)
    proc = tmp_path / "proc-cgroup"
    write(proc, "0::/job/leaf\n")

    result = SHARE.cgroup_cpu_quota(str(root), str(proc))
    print(f"v2 leaf={leaf!r}: {result!r}")
    assert result == expected


@pytest.mark.parametrize(
    ("quota", "period"),
    [
        pytest.param("-1\n", "100000\n", id="unlimited-leaf"),
        pytest.param("200000\n", "0\n", id="zero-period"),
        pytest.param("200000\n", "invalid\n", id="invalid-period"),
    ],
)
def test_v1_reverse_controller_mount_keeps_parent_limit(tmp_path, quota, period):
    root = tmp_path / "cgroup"
    mount = root / "cpuacct,cpu"
    write(mount / "cpu.cfs_quota_us", "200000\n")
    write(mount / "cpu.cfs_period_us", "100000\n")
    write(mount / "job/leaf/cpu.cfs_quota_us", quota)
    write(mount / "job/leaf/cpu.cfs_period_us", period)
    proc = tmp_path / "proc-cgroup"
    write(proc, "4:cpu,cpuacct:/job/leaf\n3:memory:/job/leaf\n")

    result = SHARE.cgroup_cpu_quota(str(root), str(proc))
    print(f"v1 cpuacct,cpu quota={quota!r}, period={period!r}: {result!r}")
    assert result == 2.0
